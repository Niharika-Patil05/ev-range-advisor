"""E5 - State-of-Health estimation on REAL cells, leave-one-battery-out.

The archived synthetic result was 1.11 percentage points MAE, which meant nothing:
every synthetic health indicator was a monotone function of SoH plus small noise, so
the model inverted an identity we had written. Real cells have knee points, path
dependence and cell-to-cell scatter in the indicator-to-SoH mapping.

Protocol: leave-one-battery-out. Each cell is predicted by a model that never saw
it. This is far harder than the random-cycle splits common in the literature, and it
is the only protocol that answers the question a vehicle actually poses -- estimating
the health of a pack the model was not trained on.

CLAUDE.md rule 2 is asserted at runtime: no feature may derive from the cycle's own
capacity. Every indicator comes from the PRECEDING CHARGE curve.

Run:  python -m src.experiments.e5_soh_lobo
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import REPORTS_DIR  # noqa: E402
from src.data.nasa import SOH_FEATURES, load_cycles  # noqa: E402
from src.evaluation.harness import run_experiment  # noqa: E402
from src.evaluation.leakage_checks import (assert_disjoint_groups,  # noqa: E402
                                           assert_no_capacity_derived_features)
from src.evaluation.provenance import ConclusionType, Evidence, Provenance  # noqa: E402
from src.evaluation.splits import LeaveOneBatteryOut  # noqa: E402


def models(seed: int) -> dict:
    return {
        "mean_baseline": (None, []),
        "cycle_only_linear": (LinearRegression(), ["cycle"]),
        "ridge": (make_pipeline(StandardScaler(), Ridge(alpha=1.0)), SOH_FEATURES),
        "random_forest": (RandomForestRegressor(300, min_samples_leaf=5, n_jobs=-1,
                                                random_state=seed), SOH_FEATURES),
        "gbm": (HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                              random_state=seed), SOH_FEATURES),
    }


def main() -> None:
    cycles, attr = load_cycles()
    print("=== ATTRITION ===")
    print(attr.to_string(index=False))
    print(f"\nE5: {len(cycles)} cycles, {cycles.battery_id.nunique()} cells, "
          f"SoH {cycles.soh.min():.2f}-{cycles.soh.max():.2f}\n")

    # CLAUDE.md rule 2, enforced rather than asserted in prose.
    assert_no_capacity_derived_features(SOH_FEATURES)

    def one_seed(seed: int) -> pd.DataFrame:
        rows = []
        for name, (est, feats) in models(seed).items():
            errs, per_cell = [], []
            for tr, te in LeaveOneBatteryOut().split(cycles):
                train, test = cycles.iloc[tr], cycles.iloc[te]
                assert_disjoint_groups(train, test, "battery_id", label="LOBO")
                if est is None:
                    pred = np.full(len(test), train.soh.mean())
                else:
                    from sklearn.base import clone
                    m = clone(est).fit(train[feats], train.soh)
                    pred = m.predict(test[feats])
                e = np.abs(pred - test.soh.to_numpy())
                errs.append(e)
                per_cell.append(dict(battery_id=test.battery_id.iloc[0],
                                     MAE_pp=float(e.mean() * 100)))
            allerr = np.concatenate(errs)
            pc = pd.DataFrame(per_cell)
            rows.append(dict(model=name,
                             MAE_pp=float(allerr.mean() * 100),
                             RMSE_pp=float(np.sqrt((allerr ** 2).mean()) * 100),
                             worst_cell_MAE_pp=float(pc.MAE_pp.max()),
                             median_cell_MAE_pp=float(pc.MAE_pp.median()),
                             n_cells=len(pc)))
        return pd.DataFrame(rows)

    res = run_experiment(
        "E5_soh_lobo",
        "SoH estimation on real NASA cells, leave-one-battery-out",
        one_seed,
        Evidence(
            experiment="E5 - SoH estimation, leave-one-battery-out",
            measured=["charge voltage, current and temperature curves",
                      "discharge capacity (LABEL ONLY)"],
            derived=["constant-current and constant-voltage durations",
                     "time in a fixed voltage window", "incremental-capacity peak",
                     "temperature rise", "charge energy accepted",
                     "SoH = capacity / robust reference capacity"],
            external=[],
            assumptions=["reference capacity = median of the 5 largest plausible "
                         "capacities, not the first (9 cells begin with a partial cycle)",
                         "capacities outside 0.2-2.2 Ah treated as measurement failures",
                         "health indicators taken from the PRECEDING charge curve"],
            split_protocol="leave-one-battery-out; each cell predicted by a model that "
                           "never saw it",
            n=f"{len(cycles)} cycles, {cycles.battery_id.nunique()} cells",
            provenance=Provenance.REAL,
            conclusion_type=ConclusionType.DEMONSTRATED,
            claim_strength="lab 18650 cells under controlled cycling; NOT a vehicle pack",
            known_confounds=["cells were aged at different ambient temperatures (4-44 C)",
                             "protocols differ across cell groups",
                             "lab cells lack pack effects: imbalance, BMS logic, thermal gradients"],
        ),
        REPORTS_DIR, index_cols=["model"], verbose=True,
        config=dict(features=SOH_FEATURES),
    )

    s = res["summary"].sort_values("MAE_pp_mean")
    print("\n=== SoH MAE in percentage points, leave-one-battery-out (5 seeds) ===")
    print(f"{'model':20s}{'MAE':>8}{'+/-':>7}{'RMSE':>8}{'median cell':>13}{'worst cell':>12}")
    for _, r in s.iterrows():
        print(f"{r['model']:20s}{r['MAE_pp_mean']:8.2f}{r['MAE_pp_std']:7.2f}"
              f"{r['RMSE_pp_mean']:8.2f}{r['median_cell_MAE_pp_mean']:13.2f}"
              f"{r['worst_cell_MAE_pp_mean']:12.2f}")
    print(f"\n  archived SYNTHETIC result for comparison: 1.11 pp "
          f"(circular, see reports/archive_synthetic/)")
    best = s.iloc[0]
    print(f"  best real model: {best['model']} at {best['MAE_pp_mean']:.2f} pp, "
          f"{best['MAE_pp_mean']/1.11:.1f}x worse than the synthetic figure")


if __name__ == "__main__":
    main()
