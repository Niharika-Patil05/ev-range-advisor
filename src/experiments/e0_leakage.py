"""E0 - how much accuracy does a naive split fabricate?

The same models, the same data, two split protocols: shuffled rows (L0) and
unseen routes (L1). The gap is the accuracy a careless evaluation invents.

This is the answer to "how did you prevent data leakage?" -- given as a measured
number rather than a claim.

Run:  python -m src.experiments.e0_leakage
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import NISSAN_LEAF_2013, REPORTS_DIR  # noqa: E402
from src.evaluation.harness import run_experiment  # noqa: E402
from src.evaluation.provenance import ConclusionType, Evidence, Provenance  # noqa: E402
from src.evaluation.splits import RandomRows, RouteWise  # noqa: E402
from src.experiments._common import evaluate, load_segments  # noqa: E402


def main() -> None:
    vehicle = NISSAN_LEAF_2013
    seg = load_segments()
    print(f"E0: {len(seg)} segments, {seg.route_id.nunique()} routes, "
          f"{seg.vehicle_id.nunique()} vehicles\n")

    def one_seed(seed: int) -> pd.DataFrame:
        leaky = evaluate(RandomRows(test_size=0.25, seed=seed, n_splits=5),
                         seg, vehicle, seed)
        honest = evaluate(RouteWise(n_splits=5, seed=seed, mode="shuffle"),
                          seg, vehicle, seed)
        return pd.concat([leaky, honest], ignore_index=True)

    res = run_experiment(
        "E0_leakage",
        "Random-row splitting vs route-wise splitting on identical real data",
        one_seed,
        Evidence(
            experiment="E0 - leakage demonstration",
            measured=["HV battery current and voltage", "vehicle speed",
                      "onboard air temperature", "HVAC power", "SoC"],
            derived=["wh_per_km", "route_id by DBSCAN on trip endpoints",
                     "congestion proxy", "physics baseline"],
            external=["eVED elevation, gradient and speed limits"],
            assumptions=["vehicle mass = kerb + 80 kg occupant",
                         "route clustering tolerance 500 m"],
            split_protocol="L0 random rows vs L1 unseen routes, 5 folds, 5 seeds",
            n=f"{len(seg)} segments, {seg.route_id.nunique()} routes",
            provenance=Provenance.REAL,
            conclusion_type=ConclusionType.DEMONSTRATED,
            claim_strength="direct comparison of evaluation protocols on identical data",
            known_confounds=["derived routes inherit clustering error",
                             "only 2 vehicles contribute meaningfully"],
        ),
        REPORTS_DIR, index_cols=["model", "protocol"], verbose=True,
        config=dict(eps_m=500.0, folds=5),
    )

    s = res["summary"]
    piv = s.pivot(index="model", columns="protocol", values="MAPE_mean")
    piv["inflation_pp"] = piv["L1"] - piv["L0"]
    piv["inflation_pct"] = 100 * (piv["L1"] - piv["L0"]) / piv["L1"]

    # Control for test-set composition. `physics_only` and `rated_range` fit nothing,
    # so they CANNOT memorise a route; any gap they show is just the two protocols
    # scoring different test sets. The excess over that control is the part
    # attributable to memorisation.
    control = float(piv.loc["physics_only", "inflation_pp"])
    piv["excess_over_control_pp"] = piv["inflation_pp"] - control

    print("\n=== MAPE by protocol (mean over 5 seeds) ===")
    print(piv.round(2).to_string())
    print(f"\ninflation_pp           error a random split HIDES, percentage points")
    print(f"excess_over_control_pp inflation beyond the non-learning control "
          f"({control:.2f} pp), i.e. the part a model can only have gained by "
          f"memorising routes")
    piv.to_csv(res["out_dir"] / "leakage_inflation.csv")


if __name__ == "__main__":
    main()
