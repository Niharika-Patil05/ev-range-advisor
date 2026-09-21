"""E6 - how much of the range error is the battery's fault?  (RQ3)

⚠️  COUPLED-SIM.  THIS IS A PROPAGATION STUDY, NOT END-TO-END VALIDATION.  ⚠️

No public dataset contains both a vehicle's trips and its own pack's SoH trajectory.
So the two halves are joined ONLY through the physical identity the project is built
on, never by merging rows:

    range = usable_energy(SoH, SoC, T) / consumption(route, weather, state)

The consumption error is REAL, measured on VED segments (E1). The SoH error is REAL,
measured leave-one-battery-out on NASA cells (E5). The COUPLING between them is
simulated, because a lab 18650 is not the pack in that Leaf. Every number here is
labelled COUPLED-SIM and none of it is evidence that the system works end to end.

Method. In logs the identity separates cleanly:

    log range = log(nominal) + log(SoH) + log(derate(T)) + log(SoC) - log(consumption)

so independent relative errors add in variance, and the share each source
contributes is directly readable.

Run:  python -m src.experiments.e6_soh_range_coupling
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import NISSAN_LEAF_2013, REPORTS_DIR  # noqa: E402
from src.data.nasa import SOH_FEATURES, load_cycles  # noqa: E402
from src.evaluation.provenance import (ConclusionType, Evidence,  # noqa: E402
                                       Provenance)
from src.evaluation.splits import LeaveOneBatteryOut, RouteWise  # noqa: E402
from src.experiments._common import load_segments  # noqa: E402
from src.features.segment_features import ALL_FEATURES, TARGET  # noqa: E402
from src.models.coupling import temp_derate, usable_energy_wh  # noqa: E402
from src.models.range_model import HybridResidual  # noqa: E402

N_MC = 20_000
DERATE_COEF_RANGE = (0.003, 0.009)   # the placeholder is 0.006 per degC below 20


def soh_relative_errors(seed: int = 0) -> np.ndarray:
    """REAL out-of-fold SoH errors from E5, as relative errors est/true - 1."""
    cycles, _ = load_cycles()
    est = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                        random_state=seed)
    out = []
    for tr, te in LeaveOneBatteryOut().split(cycles):
        train, test = cycles.iloc[tr], cycles.iloc[te]
        pred = clone(est).fit(train[SOH_FEATURES], train.soh).predict(test[SOH_FEATURES])
        out.append(pred / test.soh.to_numpy() - 1.0)
    return np.concatenate(out)


def consumption_relative_errors(seg: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """REAL out-of-fold consumption errors from the hybrid on unseen routes."""
    out = []
    for tr, te in RouteWise(n_splits=5, seed=seed, mode="shuffle").split(seg):
        train, test = seg.iloc[tr], seg.iloc[te]
        p = HybridResidual(ALL_FEATURES, seed=seed).fit(train, train[TARGET]).predict(test)
        out.append(np.asarray(p) / test[TARGET].to_numpy() - 1.0)
    return np.concatenate(out)


def main() -> None:
    vehicle = NISSAN_LEAF_2013
    rng = np.random.default_rng(0)
    seg = load_segments()

    soh_err = soh_relative_errors()
    cons_err = consumption_relative_errors(seg)
    print("=== REAL error distributions being propagated ===")
    for name, e in (("SoH (E5, leave-one-battery-out)", soh_err),
                    ("consumption (E1, unseen routes)", cons_err)):
        print(f"  {name:36s} n={len(e):5d}  sd(log) {np.std(np.log1p(e)):.4f}  "
              f"MAE {np.mean(np.abs(e))*100:5.2f} %")

    # ---- variance decomposition in log space ------------------------------------
    s_soh = float(np.std(np.log1p(soh_err)))
    s_cons = float(np.std(np.log1p(cons_err)))
    lo, hi = DERATE_COEF_RANGE
    temps = seg.temp_c.to_numpy()
    d_lo = np.array([max(0.6, 1 - lo * max(0.0, 20 - t)) for t in temps])
    d_hi = np.array([max(0.6, 1 - hi * max(0.0, 20 - t)) for t in temps])
    s_derate = float(np.std(np.log(d_hi / d_lo)) / 2)

    parts = {"consumption model": s_cons ** 2, "SoH estimation": s_soh ** 2,
             "temperature derate": s_derate ** 2}
    total = sum(parts.values())
    print("\n=== variance decomposition of log(range)   [COUPLED-SIM] ===")
    for k, v in sorted(parts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:22s} sd {np.sqrt(v):.4f}   {100*v/total:5.1f} % of variance")
    print(f"  {'TOTAL':22s} sd {np.sqrt(total):.4f}  -> ~{100*np.sqrt(total):.1f} % "
          f"relative range uncertainty (1 sigma)")

    # ---- Monte Carlo: what does an SoH error cost in kilometres? -----------------
    print("\n=== km of range error per percentage point of SoH error  [COUPLED-SIM] ===")
    rows = []
    for soh_true in (1.00, 0.90, 0.80):
        for soc in (100.0, 50.0):
            base_u = usable_energy_wh(soc, soh_true, 20.0, vehicle)
            cons = float(np.median(seg[TARGET]))
            base_range = base_u / cons
            e = rng.choice(soh_err, N_MC)
            r = usable_energy_wh(soc, np.clip(soh_true * (1 + e), 0.05, 1.2), 20.0,
                                 vehicle) / cons
            km_err = np.abs(r - base_range)
            pp_err = np.abs(e) * soh_true * 100
            ok = pp_err > 1e-9
            rows.append(dict(soh=soh_true, soc_pct=soc, base_range_km=base_range,
                             mean_km_err=float(km_err.mean()),
                             km_per_pp=float(np.mean(km_err[ok] / pp_err[ok]))))
    t = pd.DataFrame(rows)
    print(f"{'SoH':>6}{'SoC %':>7}{'range km':>10}{'mean km err':>13}{'km per pp':>11}")
    for _, r in t.iterrows():
        print(f"{r['soh']:6.2f}{r['soc_pct']:7.0f}{r['base_range_km']:10.1f}"
              f"{r['mean_km_err']:13.2f}{r['km_per_pp']:11.3f}")

    out = REPORTS_DIR / "E6_soh_range_coupling"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "km_per_pp_soh.csv", index=False)
    pd.DataFrame([dict(source=k, sd=float(np.sqrt(v)), share_pct=100 * v / total)
                  for k, v in parts.items()]).to_csv(out / "variance_decomposition.csv",
                                                     index=False)
    (out / "EVIDENCE.md").write_text(Evidence(
        experiment="E6 - SoH to range propagation (RQ3)",
        measured=["VED pack current and voltage (consumption error)",
                  "NASA charge curves and discharge capacity (SoH error)"],
        derived=["out-of-fold relative errors from E1 and E5",
                 "log-space variance decomposition"],
        external=[],
        simulated=["the COUPLING between a lab 18650 cell and a vehicle pack",
                   f"{N_MC} Monte-Carlo draws from the measured SoH error distribution",
                   "temperature-derate coefficient swept over 0.003-0.009 per degC"],
        assumptions=["error sources independent (they are not perfectly so)",
                     "the lab-cell SoH error distribution applies to a vehicle pack",
                     "nominal pack energy 24 kWh, unchanged by ageing"],
        split_protocol="errors are out-of-fold: route-wise for consumption, "
                       "leave-one-battery-out for SoH",
        n=f"{len(cons_err)} consumption errors, {len(soh_err)} SoH errors",
        provenance=Provenance.COUPLED_SIM,
        conclusion_type=ConclusionType.SIMULATED,
        claim_strength="propagation study; NOT real-world end-to-end validation",
        known_confounds=["lab cells are not vehicle packs (no imbalance, BMS, gradients)",
                         "NASA cells are 2 Ah 18650s, the Leaf pack is 24 kWh",
                         "SoH error has a heavy tail: worst cell 13.9 pp in E5"],
    ).to_markdown())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
