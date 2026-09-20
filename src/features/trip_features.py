"""Trip-level feature engineering shared by training and the app."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import VehicleSpec
from ..physics.road_load import segment_energy_wh

FEATURE_GROUPS = {
    "route": [
        "distance_km", "mean_speed_kmh", "speed_std_kmh", "mean_grade_pct",
        "max_grade_pct", "elev_gain_m", "elev_loss_m", "stops_per_km",
    ],
    "weather": ["temp_c", "headwind_ms", "rain"],
    "state": ["load_kg", "soh", "aux_on", "style"],  # style: 0 eco, 1 normal, 2 sporty
    "physics": ["phys_wh_km"],
}
ML_ONLY_FEATURES = [f for g in ("route", "weather", "state") for f in FEATURE_GROUPS[g]]
ALL_FEATURES = ML_ONLY_FEATURES + FEATURE_GROUPS["physics"]
TARGET = "wh_per_km"

DEFAULT_CONDITIONS = dict(
    temp_c=30.0, headwind_ms=0.0, rain=0, load_kg=75.0,
    soc_start=100.0, soh=1.0, aux_on=0, style=1,
)


def summarize_trip(segments: pd.DataFrame, cond: dict, vehicle: VehicleSpec) -> dict:
    """Collapse a segment table + conditions into one feature row (incl. the physics baseline)."""
    c = {**DEFAULT_CONDITIONS, **cond}
    L = segments["length_m"].to_numpy(float)
    g = segments["grade"].to_numpy(float)
    v = np.maximum(segments["speed_ms"].to_numpy(float), 0.5)
    s = segments["stops"].to_numpy(float)

    dist_m = L.sum()
    dist_km = dist_m / 1000.0
    mean_speed = dist_m / (L / v).sum()
    speed_std = float(np.sqrt((L * (v - mean_speed) ** 2).sum() / dist_m))

    phys_wh = segment_energy_wh(
        L, g, v, s,
        mass_kg=vehicle.curb_mass_kg + c["load_kg"],
        vehicle=vehicle,
        temp_c=c["temp_c"],
        headwind_ms=c["headwind_ms"],
        aux_extra_w=c["aux_on"] * vehicle.aux_extra_w,
    ).sum()

    return dict(
        distance_km=dist_km,
        mean_speed_kmh=mean_speed * 3.6,
        speed_std_kmh=speed_std * 3.6,
        mean_grade_pct=float((L * g).sum() / dist_m * 100),
        max_grade_pct=float(np.abs(g).max() * 100),
        elev_gain_m=float((L * np.clip(g, 0, None)).sum()),
        elev_loss_m=float((L * np.clip(-g, 0, None)).sum()),
        stops_per_km=float(s.sum() / dist_km),
        temp_c=float(c["temp_c"]),
        headwind_ms=float(c["headwind_ms"]),
        rain=float(c["rain"]),
        load_kg=float(c["load_kg"]),
        soh=float(c["soh"]),
        aux_on=float(c["aux_on"]),
        style=float(c["style"]),
        soc_start=float(c["soc_start"]),
        phys_wh_km=float(phys_wh / dist_km),
    )
