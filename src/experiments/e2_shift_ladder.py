"""E2 - does the physics prior buy ROBUSTNESS, or only accuracy?  (RQ1)

The headline experiment. Every model is evaluated under progressively harder
distribution shift, and the metric that matters is not accuracy but the
DEGRADATION RATIO

    D = MAPE(L_k) / MAPE(L1)

A model can win on accuracy and lose on D. Published hybrid papers report
in-distribution accuracy; the prior's real selling point -- that road-load physics
extrapolates where gradient-boosted trees structurally cannot -- is asserted far
more often than it is measured.

    L0  random rows          leaky control
    L1  unseen routes        the reference level
    L2  unseen temperature   train warm, test cold
    L3  unseen vehicle       leave-one-vehicle-out (n=2 usable, indicative only)

Run:  python -m src.experiments.e2_shift_ladder
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import NISSAN_LEAF_2013, REPORTS_DIR  # noqa: E402
from src.evaluation.harness import run_experiment  # noqa: E402
from src.evaluation.provenance import ConclusionType, Evidence, Provenance  # noqa: E402
from src.evaluation.splits import (LeaveOneVehicleOut,  # noqa: E402
                                   LeaveOneVehicleOutRouteDisjoint, RandomRows,
                                   RouteWise, TemperatureRegime)
from src.experiments._common import evaluate, load_segments  # noqa: E402


def main() -> None:
    vehicle = NISSAN_LEAF_2013
    seg = load_segments()
    print(f"E2: {len(seg)} segments, {seg.route_id.nunique()} routes, "
          f"{seg.vehicle_id.nunique()} vehicles\n")

    def one_seed(seed: int) -> pd.DataFrame:
        out = []
        for proto in (RandomRows(test_size=0.25, seed=seed, n_splits=5),
                      RouteWise(n_splits=5, seed=seed, mode="shuffle"),
                      TemperatureRegime(cold_quantile=1 / 3, seed=seed),
                      LeaveOneVehicleOut(),
                      LeaveOneVehicleOutRouteDisjoint()):
            out.append(evaluate(proto, seg, vehicle, seed))
        return pd.concat(out, ignore_index=True)

    res = run_experiment(
        "E2_shift_ladder",
        "Degradation of every model across four levels of distribution shift",
        one_seed,
        Evidence(
            experiment="E2 - distribution-shift ladder (RQ1)",
            measured=["HV battery current and voltage", "vehicle speed",
                      "onboard outside-air temperature", "HVAC power", "SoC"],
            derived=["wh_per_km", "route_id by clustering", "physics baseline",
                     "congestion proxy", "aggressiveness"],
            external=["eVED elevation, gradient, speed limits"],
            assumptions=["mass = kerb + 80 kg occupant",
                         "route clustering tolerance 500 m",
                         "cold = lowest tertile of segment-mean onboard temperature, "
                         "fixed before the experiment (HYPOTHESES.md)"],
            split_protocol="L0 random / L1 route-wise / L2 temperature regime / "
                           "L3 leave-one-vehicle-out / L3b vehicle AND route disjoint; "
                           "5 seeds",
            n=f"{len(seg)} segments, {seg.route_id.nunique()} routes, "
              f"2 usable vehicles of 3",
            provenance=Provenance.REAL,
            conclusion_type=ConclusionType.DEMONSTRATED,
            claim_strength="association-only; shift robustness for ONE vehicle model "
                           "in ONE city. L3 rests on n=2 and is indicative only.",
            known_confounds=["cold segments co-occur with high HVAC load and winter driving",
                             "L2 discards ~42% of data to keep routes disjoint",
                             "vehicle 541 contributes only 10 segments",
                             "L3 leaves 16-30% of test segments on routes seen in "
                             "training; L3b removes that and is the stricter claim"],
        ),
        REPORTS_DIR, index_cols=["model", "protocol"], verbose=True,
        config=dict(eps_m=500.0, cold_quantile=1 / 3),
    )

    s = res["summary"]
    mape = s.pivot(index="model", columns="protocol", values="MAPE_mean")
    print("\n=== MAPE by shift level (mean over 5 seeds) ===")
    print(mape.round(2).to_string())

    print("\n=== DEGRADATION RATIO  D = MAPE(Lk) / MAPE(L1)   [the RQ1 metric] ===")
    d = mape.div(mape["L1"], axis=0)[["L0", "L2", "L3", "L3b"]]
    print(d.round(3).to_string())
    print("\n  D < 1 at L0 is expected: the leaky control flatters every model.")
    print("  D at L2 and L3 is what RQ1 asks about. LOWER = degrades more gracefully.")

    # D alone is not interpretable: each level scores a DIFFERENT test set, so a low
    # D can mean "easier test set" rather than "robust model". physics_only fits
    # nothing, so its D measures test-set difficulty alone. Dividing by it isolates
    # the degradation a model incurs BECAUSE it learned.
    ctrl = d.loc["physics_only"]
    rel = d.div(ctrl, axis=1)
    print("\n=== D normalised by the non-learning control (physics_only) ===")
    print("    isolates degradation caused by LEARNING, not by test-set difficulty")
    print(rel.round(3).to_string())

    ranked = rel[["L2", "L3b"]].mean(axis=1).sort_values()
    print("\n  mean normalised D over the genuine shifts (L2, L3b), best first:")
    for m, v in ranked.items():
        print(f"    {m:20s} {v:.3f}")
    d.to_csv(res["out_dir"] / "degradation_ratio.csv")
    rel.to_csv(res["out_dir"] / "degradation_ratio_normalised.csv")

    # Per-fold detail: `evaluate` pools folds before scoring, so re-run L3b fold by
    # fold. Vehicle 541 contributes 10 segments and its fold is not meaningful.
    print("\n=== L3b fold by fold (the vehicle-shift claim rests on n=2) ===")
    from src.experiments._common import model_factories, regression_metrics
    from src.features.segment_features import TARGET
    proto = LeaveOneVehicleOutRouteDisjoint()
    rows = []
    for tr, te in proto.split(seg):
        train, test = seg.iloc[tr], seg.iloc[te]
        held = int(test.vehicle_id.iloc[0])
        for name, factory in model_factories(vehicle, 0).items():
            pr = factory().fit(train, train[TARGET]).predict(test)
            rows.append(dict(held_out=held, n_train=len(train), n_test=len(test),
                             model=name, **regression_metrics(test[TARGET], pr)))
    folds = pd.DataFrame(rows)
    print(folds.pivot(index="model", columns="held_out", values="MAPE").round(2).to_string())
    print("\n  fold sizes:")
    for h, g in folds.groupby("held_out"):
        print(f"    vehicle {h}: train {g.n_train.iloc[0]}, test {g.n_test.iloc[0]}"
              + ("   <-- too small to support a claim" if g.n_test.iloc[0] < 30 else ""))
    folds.to_csv(res["out_dir"] / "l3b_per_fold.csv", index=False)


if __name__ == "__main__":
    main()
