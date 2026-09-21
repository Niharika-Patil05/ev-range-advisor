"""Loader for the NASA PCoE Li-ion battery ageing dataset (experiment E5).

Each .mat file holds one cell's full history as a list of cycles, each typed
'charge', 'discharge' or 'impedance'. A discharge cycle carries a measured
Capacity; SoH is that capacity divided by the cell's first measured capacity.

THE LEAKAGE RULE THAT GOVERNS THIS MODULE (CLAUDE.md rule 2)
-----------------------------------------------------------
No feature may be derived from the cycle's own capacity. SoH IS capacity divided by
initial capacity, so any capacity-derived feature hands the model the answer and the
resulting accuracy is meaningless.

Every health indicator here is therefore computed from the PRECEDING CHARGE curve,
or from an impedance sweep -- never from the discharge that produced the label. This
also matches how such a model would work on a vehicle: you observe a charge, then
predict health, without first running a controlled capacity test.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DATA_DIR
from ..evaluation.leakage_checks import attrition_row
from ..evaluation.provenance import Provenance, tag

NASA_DIR = DATA_DIR / "raw" / "nasa_pcoe"

# Features the SoH model may use. All come from charge curves or impedance sweeps.
SOH_FEATURES = [
    "cc_time_min",      # constant-current charge duration: shortens as capacity fades
    "cv_time_min",      # constant-voltage tail: lengthens as impedance grows
    "v_window_min",     # time spent in a fixed voltage window during charge
    "ic_peak",          # incremental-capacity dQ/dV peak: tracks active-material loss
    "ic_peak_v",        # voltage at that peak: shifts with degradation mode
    "temp_rise_c",      # heating during charge: grows with internal resistance
    "temp_mean_c",
    "charge_energy_wh",  # energy accepted during charge (NOT the discharge capacity)
]

CC_CURRENT_FRAC = 0.9       # constant-current phase = current above this fraction of max
V_WINDOW = (3.9, 4.1)       # volts

# These are 2 Ah 18650s. Capacities outside this band are measurement failures, not
# cells: the raw data contains discharges recording exactly 0.000 Ah (aborted tests)
# and a few above nominal (2.4-2.6 Ah), which are physically impossible for a used
# cell. Both would corrupt the SoH label.
PLAUSIBLE_CAPACITY_AH = (0.2, 2.2)
NOMINAL_CAPACITY_AH = 2.0
MAX_PLAUSIBLE_SOH = 1.05    # slight headroom for measurement noise


def _charge_features(d: dict) -> dict | None:
    t = np.asarray(d["Time"], float)
    v = np.asarray(d["Voltage_measured"], float)
    i = np.asarray(d["Current_measured"], float)
    temp = np.asarray(d["Temperature_measured"], float)
    if len(t) < 20 or not np.isfinite(v).any():
        return None
    order = np.argsort(t)
    t, v, i, temp = t[order], v[order], i[order], temp[order]
    dt = np.diff(t)

    imax = np.nanmax(i)
    cc = i >= CC_CURRENT_FRAC * imax if imax > 0 else np.zeros_like(i, bool)
    cc_time = float(np.nansum(dt[cc[:-1]]) / 60.0)
    cv_time = float(np.nansum(dt[~cc[:-1]]) / 60.0)

    in_win = (v >= V_WINDOW[0]) & (v <= V_WINDOW[1])
    v_window = float(np.nansum(dt[in_win[:-1]]) / 60.0)

    # Incremental capacity dQ/dV on the charge curve. Q is the integrated current, so
    # this is charge ACCEPTED, not the discharge capacity that defines the label.
    q = np.concatenate([[0.0], np.cumsum(i[:-1] * dt)]) / 3600.0      # Ah
    ic_peak, ic_peak_v = np.nan, np.nan
    rising = np.diff(v) > 1e-4
    if rising.sum() > 10:
        vv, qq = v[:-1][rising], q[:-1][rising]
        s = np.argsort(vv)
        vv, qq = vv[s], qq[s]
        dv = np.diff(vv)
        good = dv > 1e-4
        if good.sum() > 5:
            ic = np.diff(qq)[good] / dv[good]
            ic = pd.Series(ic).rolling(5, center=True, min_periods=1).mean().to_numpy()
            k = int(np.nanargmax(ic))
            ic_peak, ic_peak_v = float(ic[k]), float(vv[:-1][good][k])

    return dict(
        cc_time_min=cc_time,
        cv_time_min=cv_time,
        v_window_min=v_window,
        ic_peak=ic_peak,
        ic_peak_v=ic_peak_v,
        temp_rise_c=float(np.nanmax(temp) - np.nanmin(temp)),
        temp_mean_c=float(np.nanmean(temp)),
        charge_energy_wh=float(np.nansum(v[:-1] * i[:-1] * dt) / 3600.0),
    )


def load_cell(path: Path) -> pd.DataFrame:
    """One row per discharge cycle: SoH label plus features from the preceding charge."""
    import scipy.io

    m = scipy.io.loadmat(path, simplify_cells=True)
    key = next(k for k in m if not k.startswith("__"))
    cycles = m[key]["cycle"]
    if isinstance(cycles, dict):
        cycles = [cycles]

    rows, last_charge, cycle_no = [], None, 0
    for c in cycles:
        ctype = c.get("type")
        if ctype == "charge":
            last_charge = _charge_features(c.get("data", {}))
        elif ctype == "discharge":
            cycle_no += 1
            # Some cells record a discharge with an EMPTY Capacity array rather than
            # a missing field, so length must be checked before indexing.
            cap_raw = np.atleast_1d(c.get("data", {}).get("Capacity", np.nan)).ravel()
            if cap_raw.size == 0 or last_charge is None:
                continue
            cap = float(cap_raw[0])
            if not np.isfinite(cap):
                continue
            rows.append(dict(battery_id=path.stem, cycle=cycle_no,
                             capacity_ah=float(cap),
                             ambient_c=float(c.get("ambient_temperature", np.nan)),
                             **last_charge))
    if not rows:
        return pd.DataFrame(columns=["battery_id", "cycle", "soh"] + SOH_FEATURES)
    df = pd.DataFrame(rows)
    # SoH = capacity / the cell's own first measured capacity. The LABEL may use
    # capacity; the FEATURES may not.
    lo, hi = PLAUSIBLE_CAPACITY_AH
    df["capacity_plausible"] = df["capacity_ah"].between(lo, hi)
    good = df.loc[df.capacity_plausible, "capacity_ah"]
    if len(good) < 5:
        return df.iloc[0:0]

    # Reference capacity: the median of the five largest PLAUSIBLE capacities, not the
    # first. Nine of these cells begin with a partial discharge (B0033 starts at 0.69
    # Ah against a later 1.89), so dividing by the first value produces SoH above 20.
    # A high quantile is robust to both a partial first cycle and to the occasional
    # over-nominal reading.
    df["reference_capacity_ah"] = float(good.nlargest(5).median())
    df["soh"] = df["capacity_ah"] / df["reference_capacity_ah"]
    return df


def load_cycles(nasa_dir: Path = NASA_DIR,
                min_cycles: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All cells, one row per discharge cycle. Returns (cycles, attrition log)."""
    files = sorted(Path(nasa_dir).rglob("*.mat"))
    if not files:
        raise FileNotFoundError(
            f"No .mat files under {nasa_dir}. Run: "
            f"python scripts/fetch_data.py --dataset nasa_pcoe --extract"
        )
    frames = [d for f in files if len(d := load_cell(f)) >= min_cycles]
    out = pd.concat(frames, ignore_index=True)
    log = [attrition_row("cells_loaded", len(files), out.battery_id.nunique(),
                         f"cells with >= {min_cycles} usable discharge cycles")]

    before = len(out)
    out = out[out.capacity_plausible].reset_index(drop=True)
    log.append(attrition_row("plausible_capacity", before, len(out),
                             f"capacity outside {PLAUSIBLE_CAPACITY_AH} Ah "
                             f"(0.000 Ah aborted tests, above-nominal readings)"))

    before = len(out)
    out = out[out.soh.between(0.0, MAX_PLAUSIBLE_SOH)].reset_index(drop=True)
    log.append(attrition_row("plausible_soh", before, len(out),
                             f"SoH above {MAX_PLAUSIBLE_SOH}"))

    before = len(out)
    out = out.dropna(subset=["soh"] + SOH_FEATURES).reset_index(drop=True)
    log.append(attrition_row("complete_features", before, len(out),
                             "missing health indicator"))
    return tag(out, Provenance.REAL, "NASA_PCoE"), pd.DataFrame(log)
