"""SYNTHETIC data generators so the whole pipeline runs before real data is plugged in.

!! Everything here is generated from assumptions written by us. Results on this data prove that the
!! code works, NOT that the model works on real vehicles. Never report them as real-world accuracy.
Replace with real data (see README, "Plugging in real data").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DEFAULT_SEED, DEFAULT_VEHICLE, VehicleSpec
from ..features.trip_features import summarize_trip
from ..physics.road_load import segment_energy_wh

ROUTE_TYPES = {
    "city":    dict(speed=(6, 11),  stops=(1.0, 3.0), grade_sigma=0.004, amp=0.005, seg=(150, 400), dist=(5, 25)),
    "hilly":   dict(speed=(4, 9),   stops=(0.2, 1.0), grade_sigma=0.010, amp=0.050, seg=(200, 500), dist=(8, 40)),
    "highway": dict(speed=(12, 15), stops=(0.0, 0.2), grade_sigma=0.006, amp=0.005, seg=(500, 1000), dist=(15, 60)),
}
STYLE_SPEED_SCALE = {0: 0.93, 1: 1.00, 2: 1.10}  # eco / normal / sporty


def make_route(rng: np.random.Generator, route_type: str = "city", distance_km: float | None = None,
               vehicle: VehicleSpec = DEFAULT_VEHICLE) -> pd.DataFrame:
    """Random segment table (length_m, grade, speed_ms, stops) for one route."""
    cfg = ROUTE_TYPES[route_type]
    distance_km = distance_km if distance_km is not None else rng.uniform(*cfg["dist"])
    lengths, total = [], 0.0
    while total < distance_km * 1000:
        seg = rng.uniform(*cfg["seg"])
        lengths.append(seg)
        total += seg
    L = np.array(lengths)
    n = len(L)

    speed = np.clip(rng.uniform(*cfg["speed"]) + rng.normal(0, 1.0, n), 2.5, vehicle.max_speed_ms)
    phase, period = rng.uniform(0, 2 * np.pi), rng.uniform(15, 40)
    grade, g = np.zeros(n), 0.0
    for i in range(n):
        g = 0.85 * g + rng.normal(0, cfg["grade_sigma"])
        grade[i] = g + cfg["amp"] * np.sin(2 * np.pi * i / period + phase)
    grade = np.clip(grade, -0.12, 0.12)
    stops = rng.poisson(rng.uniform(*cfg["stops"]) * L / 1000.0)
    return pd.DataFrame(dict(length_m=L, grade=grade, speed_ms=speed, stops=stops.astype(float)))


def sample_conditions(rng: np.random.Generator) -> dict:
    return dict(
        temp_c=float(np.clip(rng.normal(28, 8), 5, 44)),
        headwind_ms=float(np.clip(rng.normal(0, 2.5), -8, 8)),
        rain=int(rng.random() < 0.2),
        load_kg=float(rng.uniform(50, 160)),
        soc_start=float(rng.uniform(30, 100)),
        soh=float(rng.uniform(0.70, 1.0)),
        aux_on=int(rng.random() < 0.3),
        style=int(rng.choice([0, 1, 2], p=[0.25, 0.55, 0.20])),
    )


def _true_wh_per_km(seg: pd.DataFrame, cond: dict, rng: np.random.Generator,
                    vehicle: VehicleSpec) -> float:
    """Hidden 'ground truth' consumption. It deliberately contains effects the physics baseline does
    not know (Crr calibration error, wet road, cold tyres, riding style, battery losses that grow as
    SoH drops), so that there is a residual for the ML layer to learn."""
    L = seg["length_m"].to_numpy()
    v = seg["speed_ms"].to_numpy() * STYLE_SPEED_SCALE[int(cond["style"])]
    crr_scale = 1.15 * (1.25 if cond["rain"] else 1.0) * (1 + 0.004 * max(0.0, 15 - cond["temp_c"]))
    wh = segment_energy_wh(
        L, seg["grade"].to_numpy(), v, seg["stops"].to_numpy(),
        mass_kg=vehicle.curb_mass_kg + cond["load_kg"], vehicle=vehicle,
        temp_c=cond["temp_c"], headwind_ms=cond["headwind_ms"],
        crr_scale=crr_scale, aux_extra_w=cond["aux_on"] * vehicle.aux_extra_w,
    ).sum()
    battery_losses = 1 + 0.25 * (1 - cond["soh"]) + 0.006 * max(0.0, 10 - cond["temp_c"])
    return float(wh / (L.sum() / 1000.0) * battery_losses * rng.lognormal(0, 0.03))


def generate_trip_dataset(n_routes: int = 60, trips_per_route: int = 20, seed: int = DEFAULT_SEED,
                          vehicle: VehicleSpec = DEFAULT_VEHICLE) -> pd.DataFrame:
    """One row per trip: features + `wh_per_km` (target). `route_id` is the grouping key for splits."""
    rng = np.random.default_rng(seed)
    types = list(ROUTE_TYPES)
    rows = []
    for r in range(n_routes):
        rtype = types[r % len(types)]
        seg = make_route(rng, rtype, vehicle=vehicle)
        for t in range(trips_per_route):
            cond = sample_conditions(rng)
            row = summarize_trip(seg, cond, vehicle)
            row.update(route_id=r, route_type=rtype, trip_id=f"R{r:03d}_T{t:02d}",
                       wh_per_km=_true_wh_per_km(seg, cond, rng, vehicle))
            rows.append(row)
    return pd.DataFrame(rows)


def generate_battery_cycles(n_cells: int = 10, max_cycles: int = 600, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Synthetic per-cycle battery health data (stand-in for NASA / CALCE / Oxford features).

    Columns: battery_id, cycle, soh (label) and observable health indicators.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for b in range(n_cells):
        temp, dod = rng.uniform(20, 40), rng.uniform(0.5, 1.0)
        rate = rng.uniform(0.00012, 0.00030) * (1 + 0.03 * (temp - 25)) * (0.6 + 0.8 * dod)
        n = np.arange(1, max_cycles + 1)
        soh = 1 - rate * n * (1 + n / 1000) + rng.normal(0, 0.002, len(n))
        ir_base, cc_scale, ic_scale = rng.uniform(25, 35), rng.uniform(0.95, 1.05), rng.uniform(0.9, 1.1)
        frames.append(pd.DataFrame(dict(
            battery_id=f"B{b:02d}", cycle=n, soh=soh,
            ir_mohm=ir_base * (1 + 1.5 * (1 - soh)) + rng.normal(0, 1.0, len(n)),
            cc_time_min=100 * soh * cc_scale + rng.normal(0, 1.0, len(n)),
            v_window_min=25 * soh**1.5 + rng.normal(0, 0.5, len(n)),
            ic_peak=1.8 * soh**2 * ic_scale + rng.normal(0, 0.03, len(n)),
            temp_mean_c=temp + rng.normal(0, 1.0, len(n)),
            temp_rise_c=3 + 8 * (1 - soh) + rng.normal(0, 0.5, len(n)),
        )))
    return pd.concat(frames, ignore_index=True)
