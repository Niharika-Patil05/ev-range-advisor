"""E3 - which feature groups actually carry signal?

On synthetic data this question was unanswerable: the generator built consumption
FROM rain and temperature, so "weather matters" was guaranteed by construction
(docs/00_INVESTIGATION_REPORT.md section 3). On real data the answer is unknown in
advance, and it is the evidence for the synopsis's third objective -- the impact of
weather, traffic and driving behaviour.

Two models are ablated side by side, because they are affected differently:

  gbm     sees ONLY the listed features, so dropping a group removes that
          information entirely.
  hybrid  keeps `phys_wh_km`, and the physics layer consumes speed, grade and
          measured HVAC internally. Dropping a group from the ML layer therefore
          does NOT hide it from the physics layer, and the hybrid should be less
          sensitive. That asymmetry is reported, not hidden.

Run:  python -m src.experiments.e3_ablation
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
from src.evaluation.splits import RouteWise  # noqa: E402
from src.experiments._common import evaluate, load_segments  # noqa: E402
from src.features.segment_features import FEATURE_GROUPS, ML_ONLY_FEATURES  # noqa: E402

ABLATABLE = ("route", "weather", "state", "traffic")


def main() -> None:
    vehicle = NISSAN_LEAF_2013
    seg = load_segments()
    print(f"E3: {len(seg)} BEV segments, {seg.route_id.nunique()} routes\n")
    for g in ABLATABLE:
        print(f"  {g:9s} {FEATURE_GROUPS[g]}")

    def one_seed(seed: int) -> pd.DataFrame:
        rows = []
        for label in ("full",) + tuple(f"minus_{g}" for g in ABLATABLE):
            drop = () if label == "full" else (label.removeprefix("minus_"),)
            feats = [f for g in ABLATABLE if g not in drop for f in FEATURE_GROUPS[g]]
            r = evaluate(RouteWise(n_splits=5, seed=seed, mode="shuffle"),
                         seg, vehicle, seed, features=feats)
            r["variant"] = label
            r["n_features"] = len(feats)
            rows.append(r)
        return pd.concat(rows, ignore_index=True)

    res = run_experiment(
        "E3_ablation",
        "Feature-group ablation on real BEV segments, route-wise",
        one_seed,
        Evidence(
            experiment="E3 - feature-group ablation",
            measured=["HV battery current and voltage", "vehicle speed",
                      "onboard outside-air temperature", "HVAC power", "SoC"],
            derived=["wh_per_km", "stops, speed statistics, aggressiveness",
                     "congestion proxy = posted limit - observed speed",
                     "elevation gain/loss", "physics baseline"],
            external=["eVED elevation, gradient, speed limits (OSM)"],
            assumptions=["mass = kerb + 80 kg occupant", "route tolerance 500 m"],
            split_protocol="L1 route-wise, 5 folds x 5 seeds",
            n=f"{len(seg)} segments, {seg.route_id.nunique()} routes, 2 usable vehicles",
            provenance=Provenance.REAL,
            conclusion_type=ConclusionType.DEMONSTRATED,
            claim_strength="ASSOCIATION ONLY. Dropping a feature group changes the "
                           "MODEL's error; it does not measure a causal effect on "
                           "consumption. Groups are mutually confounded.",
            known_confounds=[
                "temperature co-occurs with HVAC load, season and winter driving",
                "the traffic proxy is derived from speed, so it overlaps the route group",
                "the hybrid's physics layer still sees dropped variables internally",
            ],
        ),
        REPORTS_DIR, index_cols=["model", "variant"], verbose=True,
        config=dict(groups=list(ABLATABLE), eps_m=500.0),
    )

    s = res["summary"]
    for model in ("gbm", "hybrid_physics_ml"):
        m = s[s.model == model].set_index("variant")
        full = m.loc["full", "MAPE_mean"]
        print(f"\n=== {model}: MAPE when a group is REMOVED (full = {full:.2f} %) ===")
        print(f"{'removed group':16s}{'MAPE':>8}{'+/- std':>9}{'delta pp':>10}{'delta %':>9}")
        for g in ABLATABLE:
            row = m.loc[f"minus_{g}"]
            print(f"{g:16s}{row['MAPE_mean']:8.2f}{row['MAPE_std']:9.2f}"
                  f"{row['MAPE_mean']-full:+10.2f}{100*(row['MAPE_mean']-full)/full:+8.1f}%")

    piv = s.pivot(index="variant", columns="model", values="MAPE_mean")
    piv.to_csv(res["out_dir"] / "ablation_matrix.csv")
    print("\n  A LARGER delta means the group carried more information the model used.")
    print("  This is a statement about the model, not about the world.")

    # ---- single-feature drops, to test the pre-registered H4 hypotheses ----------
    # Group-level ablation cannot separate aux_power_w from soc_start, so H4a needs
    # a finer pass. Single-feature drops UNDERSTATE the importance of correlated
    # features, because a dropped feature's information can survive in its neighbours.
    print("\n=== single-feature drops (gbm, 3 seeds) -- tests H4a-H4d ===")
    singles = ["aux_power_w", "temp_c", "speed_deficit_kmh", "speed_limit_kmh",
               "elev_gain_m", "abs_grade_pct", "mean_speed_kmh", "accel_pos_mean",
               "stops_per_km", "soc_start"]
    rows = []
    for seed in (0, 1, 2):
        base = evaluate(RouteWise(n_splits=5, seed=seed, mode="shuffle"),
                        seg, vehicle, seed, features=ML_ONLY_FEATURES)
        b = float(base[base.model == "gbm"].MAPE.iloc[0])
        for f in singles:
            feats = [x for x in ML_ONLY_FEATURES if x != f]
            r = evaluate(RouteWise(n_splits=5, seed=seed, mode="shuffle"),
                         seg, vehicle, seed, features=feats)
            rows.append(dict(dropped=f, seed=seed,
                             delta_pp=float(r[r.model == "gbm"].MAPE.iloc[0]) - b))
    sing = pd.DataFrame(rows).groupby("dropped").delta_pp.agg(["mean", "std"])
    sing = sing.sort_values("mean", ascending=False)
    print(f"{'dropped feature':20s}{'delta pp':>10}{'+/- std':>9}")
    for f, r in sing.iterrows():
        print(f"{f:20s}{r['mean']:+10.2f}{r['std']:9.2f}")
    sing.to_csv(res["out_dir"] / "single_feature_drops.csv")


if __name__ == "__main__":
    main()
