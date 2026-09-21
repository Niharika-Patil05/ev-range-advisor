"""E4 - cross-vehicle generalization using PHEVs in electric mode.

E2 could not separate two explanations for its central finding. Holding out a
vehicle there left only 139 training segments, so "learning fails under vehicle
shift" was confounded with "learning fails with little data".

PHEVs running engine-off are physically battery-electric vehicles, and VED has
1,897 such segments across 23 vehicles -- four times the BEV data. That is enough
to run leave-one-vehicle-out with PLENTY of training data, which separates the two
explanations.

    L3c  leave-one-vehicle-out WITHIN the PHEV fleet (23 vehicles, ample training data)
    L4   BEV -> PHEV-electric transfer (unseen powertrain entirely)

One confound has to be handled explicitly. HVAC instrumentation differs between the
fleets: both channels are 100% available for the BEVs, while for PHEVs A/C is 72%
available and the heater only 12%. Measured auxiliary power therefore averages 700 W
on BEVs and 275 W on PHEVs, and much of that gap is INSTRUMENTATION, not physics. So
every comparison is run twice, with and without the HVAC features.

Run:  python -m src.experiments.e4_cross_vehicle
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import NISSAN_LEAF_2013, REPORTS_DIR  # noqa: E402
from src.data.ved import (NEEDED_COLUMNS, PHEV_COLUMNS,  # noqa: E402
                          build_segment_table, load_raw, load_static, phev_vehicle_ids)
from src.evaluation.harness import run_experiment  # noqa: E402
from src.evaluation.provenance import ConclusionType, Evidence, Provenance  # noqa: E402
from src.evaluation.splits import LeaveOneVehicleOutRouteDisjoint  # noqa: E402
from src.experiments._common import evaluate, load_segments, model_factories, regression_metrics  # noqa: E402
from src.features.segment_features import ALL_FEATURES, TARGET  # noqa: E402

HVAC_FEATURES = ["aux_power_w", "hvac_on"]
MIN_SEGMENTS_PER_VEHICLE = 30


def load_phev_segments(eps_m: float = 500.0) -> pd.DataFrame:
    st = load_static().set_index("vehicle_id")
    raw = load_raw(vehicle_ids=phev_vehicle_ids(),
                   columns=NEEDED_COLUMNS + PHEV_COLUMNS, prefer_enriched=True)
    raw = raw.assign(VehId=raw.VehId.astype(int), Trip=raw.Trip.astype(int))
    seg, attr = build_segment_table(raw, vehicle=NISSAN_LEAF_2013, eps_m=eps_m,
                                    vehicle_ids=phev_vehicle_ids(),
                                    mass_by_vehicle=st.mass_kg.to_dict(),
                                    electric_mode_only=True)
    seg["vehicle_model"] = seg.vehicle_id.map(st.vehicle_model)
    print(attr.to_string(index=False))
    return seg


def clean(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    keep = df[feats + [TARGET]].notna().all(axis=1)
    if (~keep).any():
        print(f"  dropped {int((~keep).sum())} segments with missing features")
    return df[keep].reset_index(drop=True)


def main() -> None:
    vehicle = NISSAN_LEAF_2013
    bev = load_segments()
    print(f"\nBEV: {len(bev)} segments, {bev.vehicle_id.nunique()} vehicles")
    phev = load_phev_segments()
    print(f"PHEV-electric: {len(phev)} segments, {phev.vehicle_id.nunique()} vehicles, "
          f"{phev.route_id.nunique()} routes")

    variants = {
        "with_hvac": ALL_FEATURES,
        "without_hvac": [f for f in ALL_FEATURES if f not in HVAC_FEATURES],
    }

    # ---------------- L3c: leave-one-vehicle-out inside the PHEV fleet -------------
    print("\n=== L3c: leave-one-vehicle-out WITHIN the PHEV fleet ===")
    big = phev.vehicle_id.value_counts()
    keep_v = set(big[big >= MIN_SEGMENTS_PER_VEHICLE].index)
    sub = phev[phev.vehicle_id.isin(keep_v)].reset_index(drop=True)
    print(f"  {len(keep_v)} vehicles with >= {MIN_SEGMENTS_PER_VEHICLE} segments, "
          f"{len(sub)} segments total")

    def l3c(seed: int) -> pd.DataFrame:
        out = []
        for label, feats in variants.items():
            d = clean(sub, feats)
            r = evaluate(LeaveOneVehicleOutRouteDisjoint(), d, vehicle, seed,
                         features=[f for f in feats if f != "phys_wh_km"])
            r["variant"] = label
            out.append(r)
        return pd.concat(out, ignore_index=True)

    res = run_experiment(
        "E4_cross_vehicle", "L3c leave-one-vehicle-out within the PHEV fleet",
        l3c,
        Evidence(
            experiment="E4 - cross-vehicle generalization (PHEV electric mode)",
            measured=["HV battery current and voltage", "vehicle speed", "engine RPM",
                      "onboard air temperature", "HVAC power where instrumented", "SoC"],
            derived=["wh_per_km", "electric-mode filter from engine RPM",
                     "route_id by clustering", "physics baseline with per-vehicle mass"],
            external=["eVED elevation, gradient, speed limits"],
            assumptions=["PHEV mass from VED's BINNED Generalized_Weight (+/-5%)",
                         "Leaf Cd*A, crr, eta applied to all PHEVs -- not re-identified",
                         "missing HVAC channel treated as zero draw",
                         "engine-off guard band 30 s"],
            split_protocol="L3c leave-one-vehicle-out, vehicle AND route disjoint, "
                           f">= {MIN_SEGMENTS_PER_VEHICLE} segments per vehicle, 5 seeds",
            n=f"{len(sub)} PHEV-electric segments, {len(keep_v)} vehicles",
            provenance=Provenance.REAL,
            conclusion_type=ConclusionType.DEMONSTRATED,
            claim_strength="association-only; PHEVs in charge-depleting mode, not BEVs",
            known_confounds=[
                "HVAC instrumentation differs between fleets (BEV 100%, PHEV A/C 72% heater 12%)",
                "vehicle parameters are the Leaf's, not re-identified per PHEV model",
                "engine-off segments may be systematically shorter or gentler than engine-on ones",
            ],
        ),
        REPORTS_DIR, index_cols=["model", "variant"], verbose=True,
        config=dict(min_segments_per_vehicle=MIN_SEGMENTS_PER_VEHICLE, eps_m=500.0),
    )
    piv = res["summary"].pivot(index="model", columns="variant", values="MAPE_mean")
    print("\n  MAPE under L3c (mean over 5 seeds):")
    print(piv.round(2).to_string())

    # ---------------- L4: BEV -> PHEV transfer -------------------------------------
    print("\n=== L4: train on BEV, test on PHEV-electric (unseen powertrain) ===")
    rows = []
    for label, feats in variants.items():
        tr = clean(bev, feats)
        te = clean(phev, feats)
        ml = [f for f in feats if f != "phys_wh_km"]
        for name, factory in model_factories(vehicle, 0, features=ml).items():
            m = factory().fit(tr, tr[TARGET])
            rows.append(dict(model=name, variant=label, direction="BEV->PHEV",
                             **regression_metrics(te[TARGET], m.predict(te))))
        # and the reverse, which has far more training data
        for name, factory in model_factories(vehicle, 0, features=ml).items():
            m = factory().fit(te, te[TARGET])
            rows.append(dict(model=name, variant=label, direction="PHEV->BEV",
                             **regression_metrics(tr[TARGET], m.predict(tr))))
    l4 = pd.DataFrame(rows)
    print(l4.pivot_table(index="model", columns=["direction", "variant"],
                         values="MAPE").round(2).to_string())
    l4.to_csv(res["out_dir"] / "l4_transfer.csv", index=False)
    print(f"\nwrote {res['out_dir']}")


if __name__ == "__main__":
    main()
