"""Shared machinery for the real-data experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import VehicleSpec
from ..features.segment_features import ALL_FEATURES, ML_ONLY_FEATURES, TARGET
from ..models.coupling import usable_energy_wh
from ..models.range_model import ConstantRated, HybridResidual, PhysicsOnly, SklearnRegressor


def _hgb(seed: int, **kw):
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=20, random_state=seed, **kw)


def model_factories(vehicle: VehicleSpec, seed: int = 0,
                    features: list[str] | None = None) -> dict:
    """The model set for E0/E1/E2.

    Pure-ML models deliberately do NOT see `phys_wh_km`: otherwise "does the physics
    prior help?" would be answered by giving the answer to both sides.
    Preprocessing lives inside a Pipeline so scalers are fitted per fold, never on
    the whole dataset (a preprocessing leak).
    """
    ml = features if features is not None else ML_ONLY_FEATURES
    hyb = (ml + ["phys_wh_km"]) if features is not None else ALL_FEATURES
    return {
        "rated_range": lambda: ConstantRated(vehicle.rated_wh_per_km),
        "physics_only": lambda: PhysicsOnly(),
        "ridge": lambda: SklearnRegressor(make_pipeline(StandardScaler(), Ridge(alpha=1.0)), ml),
        "random_forest": lambda: SklearnRegressor(
            RandomForestRegressor(300, min_samples_leaf=3, n_jobs=-1, random_state=seed), ml),
        "gbm": lambda: SklearnRegressor(_hgb(seed), ml),
        "hybrid_physics_ml": lambda: HybridResidual(hyb, seed=seed),
    }


def regression_metrics(y, yhat) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return dict(MAE=float(np.mean(np.abs(y - yhat))),
                RMSE=float(np.sqrt(np.mean((y - yhat) ** 2))),
                MAPE=float(np.mean(np.abs(y - yhat) / y) * 100),
                medAPE=float(np.median(np.abs(y - yhat) / y) * 100))


def range_km_mae(df: pd.DataFrame, yhat, vehicle: VehicleSpec) -> float:
    """Error in predicted RANGE, in km -- what a rider actually experiences."""
    usable = usable_energy_wh(df["soc_start"], 1.0, df["temp_c"], vehicle)
    return float(np.mean(np.abs(usable / np.asarray(yhat) - usable / df[TARGET].to_numpy())))


def evaluate(protocol, df: pd.DataFrame, vehicle: VehicleSpec, seed: int,
             features: list[str] | None = None,
             leakage_check: bool = True) -> pd.DataFrame:
    """Fit and score every model under one split protocol. One row per model."""
    from ..evaluation.leakage_checks import assert_disjoint_groups

    rows = []
    for name, factory in model_factories(vehicle, seed, features).items():
        errs, preds, truths, frames = [], [], [], []
        for tr, te in protocol.split(df):
            train, test = df.iloc[tr], df.iloc[te]
            if leakage_check and protocol.level != "L0":
                assert_disjoint_groups(train, test, "route_id", label=protocol.level)
            m = factory().fit(train, train[TARGET])
            p = m.predict(test)
            preds.append(np.asarray(p, float))
            truths.append(test[TARGET].to_numpy(float))
            frames.append(test)
        y = np.concatenate(truths)
        yhat = np.concatenate(preds)
        allte = pd.concat(frames)
        rows.append(dict(model=name, protocol=protocol.level,
                         **regression_metrics(y, yhat),
                         range_km_MAE=range_km_mae(allte, yhat, vehicle),
                         n_test=len(y)))
    return pd.DataFrame(rows)


def load_segments(eps_m: float = 500.0) -> pd.DataFrame:
    from ..config import NISSAN_LEAF_2013
    from ..data.ved import build_segment_table, load_raw

    seg, _ = build_segment_table(load_raw(), vehicle=NISSAN_LEAF_2013, eps_m=eps_m)
    return seg
