"""Hooks for REAL data. Fill these in during Phase 2 of the plan.

Trip data expected schema (one row per trip): the columns produced by
`src.features.trip_features.summarize_trip` plus `route_id`, `trip_id` and the measured
`wh_per_km` target. Build it from your logged/public trips by (1) making a segment table per trip
and (2) calling summarize_trip(segments, conditions).

Battery data expected schema: see BATTERY_COLUMNS. For NASA / CALCE / Oxford, compute the health
indicators per cycle (internal-resistance proxy, constant-current charge time, time in a voltage
window, incremental-capacity peak, temperature stats) and the label soh = capacity / initial capacity.
Never use the cycle's own capacity as an input feature - that leaks the label.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

BATTERY_COLUMNS = ["battery_id", "cycle", "soh", "ir_mohm", "cc_time_min",
                   "v_window_min", "ic_peak", "temp_mean_c", "temp_rise_c"]
TRIP_REQUIRED = ["route_id", "trip_id", "wh_per_km", "soc_start"]


def load_battery_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in BATTERY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Battery CSV is missing columns: {missing}")
    return df


def load_trip_csv(path: str | Path) -> pd.DataFrame:
    from ..features.trip_features import ALL_FEATURES

    df = pd.read_csv(path)
    missing = [c for c in ALL_FEATURES + TRIP_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Trip CSV is missing columns: {missing}")
    return df
