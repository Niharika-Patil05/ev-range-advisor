"""Train + evaluate everything, write reports and save the artefacts the app needs.

    python scripts/run_pipeline.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from .config import DEFAULT_SEED, MODELS_DIR, REPORTS_DIR, VehicleSpec
from .data.synthetic import generate_battery_cycles, generate_trip_dataset
from .features.trip_features import ALL_FEATURES, FEATURE_GROUPS, TARGET
from .models.coupling import usable_energy_wh
from .models.range_model import HybridResidual, SklearnRegressor, _hgb, model_factories
from .models.soh_model import fit_final, lobo_evaluate
from .models.uncertainty import ConformalInterval, QuantileHybrid, coverage


# ----------------------------------------------------------------------------- metrics
def regression_metrics(y, yhat) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return dict(MAE=float(np.mean(np.abs(y - yhat))),
                RMSE=float(np.sqrt(np.mean((y - yhat) ** 2))),
                MAPE=float(np.mean(np.abs(y - yhat) / y) * 100))


def range_km_error(df: pd.DataFrame, yhat, vehicle: VehicleSpec) -> float:
    """Mean absolute error of the predicted RANGE, in kilometres.

    Replaces the former `range_mape`, which was redundant: with the same usable
    energy U on both sides, |U/yhat - U/y| / (U/y) simplifies exactly to
    |y - yhat| / yhat, i.e. it was MAPE with a different denominator and told us
    nothing about the battery side. Kilometres do not cancel, and km is what a
    rider actually experiences.
    """
    usable = usable_energy_wh(df["soc_start"], df["soh"], df["temp_c"], vehicle)
    r_true, r_pred = usable / df[TARGET].to_numpy(), usable / np.asarray(yhat)
    return float(np.mean(np.abs(r_pred - r_true)))


def cross_val_mape(factory, df: pd.DataFrame, n_splits: int = 5) -> tuple[float, float]:
    """Group K-fold by route: a route never appears in both train and validation."""
    scores = []
    for tr, va in GroupKFold(n_splits).split(df, groups=df["route_id"]):
        m = factory().fit(df.iloc[tr], df.iloc[tr][TARGET])
        scores.append(regression_metrics(df.iloc[va][TARGET], m.predict(df.iloc[va]))["MAPE"])
    return float(np.mean(scores)), float(np.std(scores))


def ablation_table(dev: pd.DataFrame, vehicle: VehicleSpec, seed: int, folds: int) -> pd.DataFrame:
    """Drop one ML feature group at a time. NOTE: for the hybrid, the physics baseline still uses the
    route/weather/state inputs internally; only the ML layer loses them."""
    variants = [("full", []), ("without route features", ["route"]),
                ("without weather features", ["weather"]),
                ("without vehicle state (SoH, load, aux, style)", ["state"])]
    rows = []
    for label, drop in variants:
        feats = [f for g in ("route", "weather", "state") if g not in drop for f in FEATURE_GROUPS[g]]
        gbm_m, _ = cross_val_mape(lambda: SklearnRegressor(_hgb(seed), feats), dev, folds)
        hyb_m, _ = cross_val_mape(lambda: HybridResidual(feats, seed=seed), dev, folds)
        rows.append(dict(variant=label, gbm_CV_MAPE=gbm_m, hybrid_CV_MAPE=hyb_m))
    rows.append(dict(variant="physics baseline only (no ML)",
                     gbm_CV_MAPE=np.nan,
                     hybrid_CV_MAPE=cross_val_mape(model_factories(vehicle, seed)["physics_only"], dev, folds)[0]))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main entry
def train_all(vehicle: VehicleSpec, n_routes: int = 60, trips_per_route: int = 20, seed: int = DEFAULT_SEED,
              save: bool = True, models_dir: Path = MODELS_DIR, reports_dir: Path = REPORTS_DIR,
              make_plots: bool = True, cv_folds: int = 5, alpha: float = 0.1, verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)

    # ---- data + route-wise splits (train / calibration / test never share a route)
    df = generate_trip_dataset(vehicle, n_routes, trips_per_route, seed)
    dev_i, test_i = next(GroupShuffleSplit(1, test_size=0.25, random_state=seed).split(df, groups=df["route_id"]))
    dev, test = df.iloc[dev_i].reset_index(drop=True), df.iloc[test_i].reset_index(drop=True)
    tr_i, cal_i = next(GroupShuffleSplit(1, test_size=0.25, random_state=seed).split(dev, groups=dev["route_id"]))
    train, cal = dev.iloc[tr_i], dev.iloc[cal_i]
    assert not set(train.route_id) & set(test.route_id) and not set(cal.route_id) & set(test.route_id)
    log(f"Trips: train={len(train)} calibration={len(cal)} test={len(test)} (split by route)")

    # ---- range models
    fitted, rows, preds = {}, [], {}
    for name, factory in model_factories(vehicle, seed).items():
        m = factory().fit(train, train[TARGET])
        p = m.predict(test)
        fitted[name], preds[name] = m, p
        met = regression_metrics(test[TARGET], p)
        cv_mean, cv_std = cross_val_mape(factory, dev, cv_folds)
        rows.append(dict(model=name, **met, range_km_MAE=range_km_error(test, p, vehicle),
                         CV_MAPE=cv_mean, CV_MAPE_std=cv_std))
    range_results = pd.DataFrame(rows).set_index("model")
    log("\nRange models (holdout = unseen routes; MAE/RMSE in Wh/km, MAPE in %):")
    log(range_results.round(2).to_string())

    # ---- uncertainty
    hybrid = fitted["hybrid_physics_ml"]
    conf = ConformalInterval(alpha).fit(cal[TARGET], hybrid.predict(cal))
    c_lo, c_hi = conf.interval(preds["hybrid_physics_ml"])
    qh = QuantileHybrid(ALL_FEATURES, alpha, seed).fit(train, train[TARGET])
    q_lo, q_hi = qh.interval(test)
    mid = preds["hybrid_physics_ml"]
    uncertainty = pd.DataFrame([
        dict(method="split conformal", target=1 - alpha, coverage=coverage(test[TARGET], c_lo, c_hi),
             mean_rel_width=float(np.mean((c_hi - c_lo) / mid))),
        dict(method="quantile regression", target=1 - alpha, coverage=coverage(test[TARGET], q_lo, q_hi),
             mean_rel_width=float(np.mean((q_hi - q_lo) / mid))),
    ]).set_index("method")
    log(f"\nUncertainty ({int((1-alpha)*100)}% intervals on unseen routes):")
    log(uncertainty.round(3).to_string())

    # ---- ablation
    ablation = ablation_table(dev, vehicle, seed, cv_folds)
    log("\nAblation (group-CV MAPE %, lower is better):")
    log(ablation.round(2).to_string(index=False))

    # ---- SoH
    cycles = generate_battery_cycles(seed=seed)
    soh_summary, soh_oof = lobo_evaluate(cycles, seed)
    log("\nSoH, leave-one-battery-out (MAE in SoH percentage points):")
    log(soh_summary.round(2).to_string())
    soh_model = fit_final(cycles, seed)

    system = dict(hybrid=hybrid, conformal=conf, soh_model=soh_model,
                  meta=dict(seed=seed, data="SYNTHETIC", vehicle=vehicle.name),
                  vehicle=vehicle)
    results = dict(range_results=range_results, uncertainty=uncertainty, ablation=ablation,
                   soh=soh_summary, soh_oof=soh_oof, test=test, preds=preds, system=system,
                   conformal_q=conf.q_)

    if save:
        models_dir, reports_dir = Path(models_dir), Path(reports_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(system, models_dir / "system.joblib")
        range_results.to_csv(reports_dir / "range_results.csv")
        uncertainty.to_csv(reports_dir / "uncertainty.csv")
        ablation.to_csv(reports_dir / "ablation.csv", index=False)
        soh_summary.to_csv(reports_dir / "soh_results.csv")
        (reports_dir / "summary.json").write_text(json.dumps(dict(
            data="SYNTHETIC - proves the pipeline works, not real-world accuracy",
            vehicle=vehicle.name,
            best_range_model=str(range_results["MAPE"].idxmin()),
            hybrid_MAPE=float(range_results.loc["hybrid_physics_ml", "MAPE"]),
            conformal_coverage=float(uncertainty.loc["split conformal", "coverage"]),
            soh_best=str(soh_summary.index[0]), soh_best_MAE_pp=float(soh_summary["MAE_mean"].iloc[0]),
        ), indent=2))
        if make_plots:
            from .plots import make_all_plots
            make_all_plots(results, reports_dir)
        log(f"\nSaved models -> {models_dir}, reports -> {reports_dir}")
    return results


def load_or_train_system(vehicle: VehicleSpec, models_dir: Path = MODELS_DIR) -> dict:
    """Load the saved artefacts, or train a small system if none exist.

    A cached system is only reused if it was trained for THIS vehicle -- otherwise
    the app would silently serve a model fitted to a different set of road-load
    parameters, which is precisely the failure the C1 refactor exists to prevent.
    """
    path = Path(models_dir) / "system.joblib"
    if path.exists():
        system = joblib.load(path)
        if system.get("meta", {}).get("vehicle") == vehicle.name:
            return system
    return train_all(vehicle, n_routes=45, trips_per_route=15, make_plots=False, verbose=False)["system"]
