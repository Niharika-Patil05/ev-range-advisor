"""Train the model the app serves, on REAL data.

    python scripts/train_real_system.py

Until now the app served a model fitted to the synthetic generator while the reported
results came from real data. The two did not correspond, and a viewer watching the
demo was not seeing any number this project reports.

This trains the hybrid on real VED segments using only the features a route planner
can supply before a trip happens, calibrates the conformal band on held-out routes,
fits the SoH model on real NASA cells, and records the MEASURED accuracy in the
artefact so the app can display what it is actually serving.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import MODELS_DIR, NISSAN_LEAF_2013  # noqa: E402
from src.data.nasa import SOH_FEATURES, load_cycles  # noqa: E402
from src.evaluation.harness import git_revision  # noqa: E402
from src.evaluation.leakage_checks import assert_three_way_disjoint  # noqa: E402
from src.evaluation.splits import RouteWise  # noqa: E402
from src.experiments._common import load_segments, regression_metrics  # noqa: E402
from src.features.segment_features import DEPLOYABLE_WITH_PHYSICS, TARGET  # noqa: E402
from src.models.range_model import HybridResidual  # noqa: E402
from src.models.uncertainty import ConformalInterval  # noqa: E402

ALPHA = 0.10
SEED = 0


def main() -> None:
    vehicle = NISSAN_LEAF_2013
    seg = load_segments()
    print(f"{len(seg)} real segments, {seg.route_id.nunique()} routes")

    # Honest accuracy for the served configuration, measured on unseen routes before
    # anything is fitted on everything.
    y, yhat = [], []
    for tr, te in RouteWise(n_splits=5, seed=SEED, mode="shuffle").split(seg):
        m = HybridResidual(DEPLOYABLE_WITH_PHYSICS, seed=SEED).fit(
            seg.iloc[tr], seg.iloc[tr][TARGET])
        y.append(seg.iloc[te][TARGET].to_numpy())
        yhat.append(np.asarray(m.predict(seg.iloc[te])))
    cv = regression_metrics(np.concatenate(y), np.concatenate(yhat))
    print(f"held-out (unseen routes): MAPE {cv['MAPE']:.2f} %, MAE {cv['MAE']:.1f} Wh/km")

    # Route-disjoint train / calibration split for the deployed artefact.
    rng = np.random.default_rng(SEED)
    routes = np.asarray(pd.unique(seg.route_id), dtype=object)
    rng.shuffle(routes)
    cut = int(len(routes) * 0.75)
    train = seg[seg.route_id.isin(routes[:cut])].reset_index(drop=True)
    cal = seg[seg.route_id.isin(routes[cut:])].reset_index(drop=True)
    assert_three_way_disjoint(train, cal, cal.iloc[0:0], "route_id")
    print(f"train {len(train)} / calibration {len(cal)} segments (route-disjoint)")

    hybrid = HybridResidual(DEPLOYABLE_WITH_PHYSICS, seed=SEED).fit(train, train[TARGET])
    conformal = ConformalInterval(ALPHA).fit(cal[TARGET], hybrid.predict(cal))
    print(f"conformal q = {conformal.q_:.3f} (relative half-width)")

    cycles, _ = load_cycles()
    soh_model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, random_state=SEED).fit(
        cycles[SOH_FEATURES], cycles.soh)
    print(f"SoH model fitted on {len(cycles)} cycles from "
          f"{cycles.battery_id.nunique()} real cells")

    system = dict(
        hybrid=hybrid, conformal=conformal, soh_model=soh_model,
        features=DEPLOYABLE_WITH_PHYSICS, soh_features=SOH_FEATURES,
        vehicle=vehicle,
        meta=dict(
            data="REAL", vehicle=vehicle.name, seed=SEED, git_revision=git_revision(),
            n_segments=len(seg), n_routes=int(seg.route_id.nunique()),
            n_vehicles=int(seg.vehicle_id.nunique()),
            holdout_MAPE=cv["MAPE"], holdout_MAE=cv["MAE"],
            conformal_alpha=ALPHA, conformal_q=conformal.q_,
            soh_cells=int(cycles.battery_id.nunique()),
            soh_lobo_MAE_pp=4.68,      # measured in E5
            source="VED + eVED (consumption), NASA PCoE (SoH)",
        ))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / "system_real.joblib"
    joblib.dump(system, out)
    print(f"\nwrote {out}")
    print("The app will now prefer this over the synthetic model.")


if __name__ == "__main__":
    main()
