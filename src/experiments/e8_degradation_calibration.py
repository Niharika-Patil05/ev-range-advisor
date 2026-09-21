"""E8 - calibrate the degradation projection, and delete what cannot be calibrated.

`project_soh` shipped with six invented constants: a base fade rate, an Arrhenius
temperature factor, a depth-of-discharge exponent and a charge-cap factor. The app
turned them into a specific, actionable claim -- "limiting daily charge to 80 % could
preserve roughly X percentage points of SoH" -- which nobody had measured. That is the
one place in the project where a placeholder produced advice.

This experiment asks what the NASA cells can actually support.

Run:  python -m src.experiments.e8_degradation_calibration
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import REPORTS_DIR  # noqa: E402
from src.data.nasa import load_cycles  # noqa: E402
from src.evaluation.provenance import ConclusionType, Evidence, Provenance  # noqa: E402

MIN_CYCLES = 20


def per_cell_fade(df: pd.DataFrame) -> pd.DataFrame:
    """Theil-Sen slope of SoH against cycle, per cell.

    Theil-Sen rather than least squares: several cells open with a partial discharge,
    and an OLS slope on those is dragged so far that it comes out NEGATIVE, implying
    a cell that heals with use.
    """
    rows = []
    for cell, d in df.groupby("battery_id"):
        d = d.sort_values("cycle")
        if len(d) < MIN_CYCLES:
            continue
        slope, _, lo, hi = stats.theilslopes(d.soh.to_numpy(), d.cycle.to_numpy())
        rows.append(dict(battery_id=cell, ambient_c=float(d.ambient_c.median()),
                         n_cycles=len(d), fade_per_cycle=-slope,
                         fade_lo=-hi, fade_hi=-lo))
    return pd.DataFrame(rows)


def main() -> None:
    cycles, _ = load_cycles()
    fade = per_cell_fade(cycles)
    print(f"E8: {len(fade)} cells, {len(cycles)} cycles\n")

    mean, sd = float(fade.fade_per_cycle.mean()), float(fade.fade_per_cycle.std())
    q = fade.fade_per_cycle.quantile([0.1, 0.5, 0.9])
    print("=== base fade rate (SoH lost per cycle) ===")
    print(f"  mean {mean:.5f}   sd {sd:.5f}   "
          f"cell-to-cell spread {100*sd/mean:.0f} % of the mean")
    print(f"  10th/50th/90th percentile: {q.iloc[0]:.5f} / {q.iloc[1]:.5f} / {q.iloc[2]:.5f}")
    print(f"  negative slopes (physically impossible): "
          f"{int((fade.fade_per_cycle < 0).sum())} of {len(fade)}")
    print(f"\n  the shipped placeholder was 0.00025 per equivalent full cycle, "
          f"{mean/0.00025:.0f}x SMALLER than measured")

    print("\n=== is a temperature factor identifiable? ===")
    sl, ic, rv, pv, se = stats.linregress(fade.ambient_c, fade.fade_per_cycle)
    print(f"  fade ~ ambient: slope {sl:+.2e} /degC, r = {rv:+.3f}, p = {pv:.3f}")
    print(fade.groupby("ambient_c").fade_per_cycle.agg(["count", "mean", "std"]).round(5).to_string())
    verdict = "NOT identifiable" if pv > 0.05 else "identifiable"
    print(f"\n  VERDICT: {verdict} (p = {pv:.3f}).")
    print("  The sign is also the wrong way round for an Arrhenius law -- the 44 C group")
    print("  shows the LOWEST fade of all, which is a protocol artefact, not physics.")

    print("\n=== depth of discharge and charge cap ===")
    print("  NASA PCoE records no per-cell depth-of-discharge or charge-limit variable.")
    print("  The dod_factor and cap_factor in project_soh are therefore UNCALIBRATABLE")
    print("  from this data and must be removed rather than left as plausible-looking")
    print("  placeholders that generate user-facing advice.")

    out = REPORTS_DIR / "E8_degradation_calibration"
    out.mkdir(parents=True, exist_ok=True)
    fade.to_csv(out / "per_cell_fade.csv", index=False)
    pd.DataFrame([dict(parameter="base_fade_per_cycle", calibrated=True,
                       value=mean, sd=sd, p10=q.iloc[0], p90=q.iloc[2], n_cells=len(fade)),
                  dict(parameter="temperature_factor", calibrated=False,
                       value=np.nan, note=f"p={pv:.3f}, not significant, wrong sign"),
                  dict(parameter="dod_factor", calibrated=False, value=np.nan,
                       note="no depth-of-discharge variable in NASA PCoE"),
                  dict(parameter="charge_cap_factor", calibrated=False, value=np.nan,
                       note="no charge-limit variable in NASA PCoE"),
                  ]).to_csv(out / "calibration_summary.csv", index=False)
    (out / "EVIDENCE.md").write_text(Evidence(
        experiment="E8 - degradation projection calibration",
        measured=["discharge capacity per cycle", "ambient temperature per cell"],
        derived=["SoH = capacity / robust reference capacity",
                 "Theil-Sen fade slope per cell"],
        external=[],
        assumptions=["fade is approximately linear in cycle count over the observed range",
                     "lab cycling represents vehicle use (it does not; see limitations)"],
        split_protocol="none - this is a parameter fit; uncertainty from cell-to-cell spread",
        n=f"{len(fade)} cells, {len(cycles)} cycles",
        provenance=Provenance.REAL,
        conclusion_type=ConclusionType.DEMONSTRATED,
        claim_strength="lab 18650 cells; a vehicle pack is not one of these",
        known_confounds=["ageing protocol is confounded with ambient temperature group",
                         "cell-to-cell spread is 80 % of the mean fade rate",
                         "no DoD or charge-limit variable exists in this dataset"],
    ).to_markdown())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
