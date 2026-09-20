"""E7 - fit the Leaf's road-load parameters to measured VED segments.

Answers two questions, and the second matters more than the first:
  1. How much of the 22.5% physics-only error do fitted parameters absorb?
  2. WHICH parameters can this data identify at all?

Run:  python -m src.experiments.e7_parameter_identification
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.calibration.identify import (bootstrap_fit, fit, predict_energy_wh,  # noqa: E402
                                      segment_basis)
from src.config import NISSAN_LEAF_2013, REPORTS_DIR  # noqa: E402
from src.data.ved import NOMINAL_OCCUPANT_KG, build_segment_table, load_raw  # noqa: E402
from src.evaluation.provenance import ConclusionType, Evidence, Provenance  # noqa: E402


def build_basis(raw: pd.DataFrame, seg: pd.DataFrame, vehicle) -> pd.DataFrame:
    """One row of basis integrals per surviving segment, aligned to `seg`."""
    keep = set(zip(seg.vehicle_id, seg.trip))
    rows, index = [], []
    for (v, t), d in raw.groupby(["VehId", "Trip"], sort=True):
        if (int(v), int(t)) not in keep:
            continue
        d = d.sort_values("Timestamp(ms)")
        ts = d["Timestamp(ms)"].to_numpy(float) / 1000.0
        spd = d["Vehicle Speed[km/h]"].to_numpy(float) / 3.6
        grade = (np.nan_to_num(d["Gradient"].to_numpy(float))
                 if "Gradient" in d.columns else 0.0)
        hvac = np.nan_to_num((d["Heater Power[Watts]"]
                              + d["Air Conditioning Power[Watts]"]).to_numpy(float))
        rows.append(segment_basis(
            ts, spd, grade, mass_kg=vehicle.curb_mass_kg + NOMINAL_OCCUPANT_KG,
            vehicle=vehicle, temp_c=float(np.nanmedian(d["OAT[DegC]"])),
            aux_power_w=hvac + vehicle.aux_power_w))
        index.append((int(v), int(t)))
    b = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(index, names=["vehicle_id", "trip"]))
    return b.loc[list(zip(seg.vehicle_id, seg.trip))].reset_index(drop=True)


def main() -> None:
    vehicle = NISSAN_LEAF_2013
    raw = load_raw()
    seg, _ = build_segment_table(raw, vehicle=vehicle)
    basis = build_basis(raw, seg, vehicle)
    y = seg["energy_wh"].to_numpy(float)
    cd_a = vehicle.cd * vehicle.frontal_area_m2

    print(f"E7: {len(seg)} segments, {seg.route_id.nunique()} routes\n")

    nominal = predict_energy_wh(basis, vehicle.crr, cd_a,
                                vehicle.drivetrain_eff, vehicle.regen_eff)
    print("=== nominal (datasheet) parameters ===")
    print(f"  crr={vehicle.crr:.4f}  eta={vehicle.drivetrain_eff:.3f}  "
          f"MAPE={np.mean(np.abs(nominal-y)/y)*100:5.2f}%  "
          f"bias={np.mean((nominal-y)/y)*100:+5.2f}%")

    print("\n=== fits (all relative losses) ===")
    fits = {}
    for label, kw in [("eta only, crr fixed", dict(bounds={"crr": (vehicle.crr, vehicle.crr)})),
                      ("crr+eta, abs_rel", dict(loss_name="abs_rel")),
                      ("crr+eta, sq_rel", dict(loss_name="sq_rel")),
                      ("crr+eta, log_sq", dict(loss_name="log_sq"))]:
        r = fit(basis, y, vehicle=vehicle, **kw)
        fits[label] = r
        print(f"  {label:22s} crr={r['crr']:.4f}  eta={r['drivetrain_eff']:.3f}  "
              f"MAPE={r['mape']:5.2f}%  bias={r['bias']:+6.2f}%")
    best = min(fits.values(), key=lambda r: r["mape"])
    nominal_mape = float(np.mean(np.abs(nominal - y) / y) * 100)
    print(f"\n  best fitted MAPE {best['mape']:.2f}% vs nominal {nominal_mape:.2f}%"
          f"  -> improvement {nominal_mape - best['mape']:+.2f} percentage points")
    a, b_ = fits["eta only, crr fixed"], fits["crr+eta, abs_rel"]

    print("\n=== route-level bootstrap (200 resamples of whole routes) ===")
    boot = bootstrap_fit(basis, y, seg["route_id"].to_numpy(), vehicle=vehicle, n_boot=200)
    for col in ("crr", "drivetrain_eff"):
        q = boot[col].quantile([0.025, 0.5, 0.975])
        width = (q.iloc[2] - q.iloc[0]) / max(abs(q.iloc[1]), 1e-9) * 100
        print(f"  {col:16s} median {q.iloc[1]:.4f}   95% CI [{q.iloc[0]:.4f}, "
              f"{q.iloc[2]:.4f}]   width {width:5.1f}% of median")
    r = float(np.corrcoef(boot["crr"], boot["drivetrain_eff"])[0, 1])
    print(f"\n  corr(crr, eta) across bootstrap = {r:+.3f}  (the degeneracy direction)")

    print("\n=== is the residual structured? (log(measured/physics) vs conditions) ===")
    lr = np.log(y / nominal)
    for c in ["temp_c", "soc_start", "mean_speed_kmh", "stops_per_km",
              "aux_power_w", "distance_km"]:
        print(f"  {c:18s} r = {np.corrcoef(lr, seg[c])[0, 1]:+.3f}")
    print("  A constant crr and eta cannot absorb structure like this; that is the")
    print("  empirical case for the hybrid's learned residual layer.")

    out = REPORTS_DIR / "E7_parameter_identification"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        dict(variant="nominal", crr=vehicle.crr, drivetrain_eff=vehicle.drivetrain_eff,
             mape=float(np.mean(np.abs(nominal - y) / y) * 100),
             bias=float(np.mean((nominal - y) / y) * 100)),
        dict(variant="fit_eta_only", **{k: a[k] for k in ("crr", "drivetrain_eff", "mape", "bias")}),
        dict(variant="fit_crr_and_eta", **{k: b_[k] for k in ("crr", "drivetrain_eff", "mape", "bias")}),
    ]).to_csv(out / "metrics_summary.csv", index=False)
    boot.to_csv(out / "bootstrap.csv", index=False)

    evidence = Evidence(
        experiment="E7 - road-load parameter identification",
        measured=["HV battery current and voltage (segment energy)", "vehicle speed",
                  "onboard outside-air temperature", "HVAC power"],
        derived=["segment energy = -integral(V*I dt)", "road-load basis integrals",
                 "road grade from eVED elevation"],
        external=["eVED map-matched elevation and gradient"],
        assumptions=[
            f"mass = kerb {vehicle.curb_mass_kg} kg + {NOMINAL_OCCUPANT_KG} kg occupant; "
            f"payload is UNOBSERVED in VED and maps almost directly onto crr",
            f"Cd*A fixed at datasheet {cd_a:.3f} m^2; urban speeds do not excite the aero term",
            f"regen efficiency fixed at {vehicle.regen_eff}",
            "tractive/regenerative mask held fixed during fitting",
        ],
        split_protocol="parameter fit, not a predictive evaluation; uncertainty from a "
                       "route-level bootstrap (whole routes resampled, never rows)",
        n=f"{len(seg)} segments, {seg.route_id.nunique()} routes, "
          f"{seg.vehicle_id.nunique()} vehicles",
        provenance=Provenance.REAL,
        conclusion_type=ConclusionType.DEMONSTRATED,
        claim_strength="association-only; parameter estimates are conditional on the "
                       "model form and the fixed assumptions above",
        known_confounds=[
            "payload unobserved (mass error ~+/-5%)",
            "crr and eta degenerate: corr +0.867 across bootstrap",
            "temperature co-occurs with HVAC use, low speed and winter driving",
        ],
    )
    (out / "EVIDENCE.md").write_text(evidence.to_markdown())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
