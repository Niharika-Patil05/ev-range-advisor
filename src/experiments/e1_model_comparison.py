"""E1 - the project's first honest model comparison.

Six models on real VED segments under route-wise splitting, five seeds, mean +/- std.
Replaces every number in reports/archive_synthetic/, which was circular by
construction (docs/00_INVESTIGATION_REPORT.md section 3).

Also reports sensitivity to the route-clustering tolerance, since `route_id` is
DERIVED here and no single tolerance is self-evidently correct.

Run:  python -m src.experiments.e1_model_comparison
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import NISSAN_LEAF_2013, REPORTS_DIR  # noqa: E402
from src.evaluation.harness import run_experiment  # noqa: E402
from src.evaluation.provenance import ConclusionType, Evidence, Provenance  # noqa: E402
from src.evaluation.splits import RouteWise  # noqa: E402
from src.experiments._common import evaluate, load_segments  # noqa: E402


def main() -> None:
    vehicle = NISSAN_LEAF_2013
    seg = load_segments()
    print(f"E1: {len(seg)} segments, {seg.route_id.nunique()} routes, "
          f"{seg.vehicle_id.nunique()} vehicles\n")

    res = run_experiment(
        "E1_model_comparison",
        "Six models under route-wise splitting on real VED segments",
        lambda seed: evaluate(RouteWise(n_splits=5, seed=seed, mode="shuffle"),
                              seg, vehicle, seed),
        Evidence(
            experiment="E1 - model comparison on real data",
            measured=["HV battery current and voltage", "vehicle speed",
                      "onboard outside-air temperature", "HVAC power", "SoC"],
            derived=["wh_per_km = -integral(V*I dt) / speed-integrated distance",
                     "route_id by DBSCAN on trip endpoints (500 m)",
                     "congestion proxy = posted limit - observed speed",
                     "aggressiveness = mean positive acceleration",
                     "phys_wh_km from road load over the measured trace"],
            external=["eVED map-matched elevation, gradient, speed limits"],
            assumptions=["mass = kerb 1493 kg + 80 kg occupant; payload UNOBSERVED",
                         "Cd*A and crr at datasheet values (E7: crr not identifiable)",
                         "regen_eff = 0.55 constant",
                         "rated_range baseline uses the EPA 121 km figure"],
            split_protocol="L1 route-wise, 5 folds x 5 seeds, groups disjoint (asserted)",
            n=f"{len(seg)} segments, {seg.route_id.nunique()} routes, "
              f"{seg.vehicle_id.nunique()} vehicles (only 2 contribute meaningfully)",
            provenance=Provenance.REAL,
            conclusion_type=ConclusionType.DEMONSTRATED,
            claim_strength="association-only; accuracy on unseen routes for ONE vehicle "
                           "model in ONE city over ONE year",
            known_confounds=["payload unobserved", "temperature co-occurs with HVAC use",
                             "derived routes inherit clustering error",
                             "2 usable vehicles, so this is not a vehicle-level claim"],
        ),
        REPORTS_DIR, index_cols=["model"], verbose=True,
        config=dict(eps_m=500.0, folds=5, protocol="L1"),
    )

    s = res["summary"].sort_values("MAPE_mean")
    print("\n=== E1: consumption prediction on UNSEEN ROUTES (mean +/- std, 5 seeds) ===")
    for _, r in s.iterrows():
        print(f"  {r['model']:20s} MAPE {r['MAPE_mean']:6.2f} +/- {r['MAPE_std']:4.2f} % "
              f"| medAPE {r['medAPE_mean']:5.2f} % | MAE {r['MAE_mean']:6.2f} Wh/km "
              f"| range err {r['range_km_MAE_mean']:5.2f} km")

    best = s.iloc[0]
    phys = s[s.model == "physics_only"].iloc[0]
    rated = s[s.model == "rated_range"].iloc[0]
    print(f"\n  best model: {best['model']} at {best['MAPE_mean']:.2f}% MAPE")
    print(f"  vs physics-only  {phys['MAPE_mean']:.2f}%  -> "
          f"{phys['MAPE_mean'] - best['MAPE_mean']:+.2f} pp")
    print(f"  vs rated range   {rated['MAPE_mean']:.2f}%  -> "
          f"{rated['MAPE_mean'] - best['MAPE_mean']:+.2f} pp")

    print("\n=== sensitivity to the derived route-clustering tolerance ===")
    rows = []
    for eps in (300.0, 500.0, 800.0):
        sg = load_segments(eps_m=eps)
        r = evaluate(RouteWise(n_splits=5, seed=0, mode="shuffle"), sg, vehicle, 0)
        r["eps_m"] = eps
        r["n_routes"] = sg.route_id.nunique()
        rows.append(r)
    sens = pd.concat(rows, ignore_index=True)
    print(sens.pivot(index="model", columns="eps_m", values="MAPE").round(2).to_string())
    sens.to_csv(res["out_dir"] / "route_tolerance_sensitivity.csv", index=False)
    print("\n  If the ranking held across tolerances, the conclusion does not depend")
    print("  on an arbitrary clustering choice.")


if __name__ == "__main__":
    main()
