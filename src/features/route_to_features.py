"""Turn a planned route into the feature row the deployed model expects.

The models are trained on ANALYSIS SEGMENTS derived from 1 Hz logged traces. A
planned route is not that: it is a table of (length, grade, speed, stops) with no
time axis. This module reconstructs a plausible per-second trace from the plan so
that `phys_wh_km` is computed by the SAME function used at training time
(`trace_energy_wh`), rather than by a second, subtly different estimator.

The reconstruction is an approximation and the app says so: real driving has
acceleration structure a plan cannot know. What it does capture is segment-to-segment
speed changes and the stop-start events that dominate urban energy use.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import VehicleSpec
from ..physics.road_load import trace_energy_wh

STOP_DECEL_S = 6.0      # seconds to slow to rest and pull away again
DT = 1.0


def segments_to_trace(segments: pd.DataFrame, *, stops_total: float | None = None
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct (time, speed, grade) at 1 Hz from a planned segment table.

    Each segment is travelled at its planned speed; stops are inserted as a dip to
    rest and back, which is what puts the inertia term into the physics baseline.
    """
    L = segments["length_m"].to_numpy(float)
    v = np.maximum(segments["speed_ms"].to_numpy(float), 0.5)
    g = segments["grade"].to_numpy(float)
    stops = (segments["stops"].to_numpy(float) if "stops" in segments
             else np.zeros(len(L)))
    if stops_total is not None and stops.sum() > 0:
        stops = stops * (stops_total / stops.sum())

    speeds, grades = [], []
    for length, speed, grade, n_stop in zip(L, v, g, stops):
        k = int(round(n_stop))
        # A stop ADDS time; it does not remove road. An earlier version overwrote
        # cruise samples with the deceleration dip, which silently shortened the
        # route and made stops look FREE. Here the dip is built first, and the
        # cruise portion is sized so the segment still covers its stated length.
        dip = np.abs(np.linspace(-1.0, 1.0, max(int(STOP_DECEL_S / DT), 2))) * speed
        dip_distance = float(np.trapezoid(dip, dx=DT)) * k
        cruise_distance = max(length - dip_distance, 0.0)
        n_cruise = max(int(round(cruise_distance / max(speed, 0.5) / DT)), 1)

        if k == 0:
            seg_v = np.full(n_cruise, speed)
        else:
            chunks, per = [], n_cruise // (k + 1)
            for j in range(k):
                chunks.append(np.full(max(per, 1), speed))
                chunks.append(dip)
            chunks.append(np.full(max(n_cruise - per * k, 1), speed))
            seg_v = np.concatenate(chunks)
        speeds.append(seg_v)
        grades.append(np.full(len(seg_v), grade))
    speed_trace = np.concatenate(speeds) if speeds else np.array([0.5, 0.5])
    grade_trace = np.concatenate(grades) if grades else np.array([0.0, 0.0])
    t = np.arange(len(speed_trace), dtype=float) * DT
    return t, speed_trace, grade_trace


def route_to_features(segments: pd.DataFrame, cond: dict, vehicle: VehicleSpec,
                      occupant_kg: float = 80.0) -> dict:
    """One feature row matching `DEPLOYABLE_WITH_PHYSICS`."""
    L = segments["length_m"].to_numpy(float)
    g = segments["grade"].to_numpy(float)
    v = np.maximum(segments["speed_ms"].to_numpy(float), 0.5)
    stops = segments["stops"].to_numpy(float) if "stops" in segments else np.zeros(len(L))

    dist_m = float(L.sum())
    dist_km = dist_m / 1000.0
    mean_speed = dist_m / float((L / v).sum())
    speed_std = float(np.sqrt((L * (v - mean_speed) ** 2).sum() / dist_m))

    aux_w = float(cond.get("aux_power_w", 0.0))
    t, sp, gr = segments_to_trace(segments)
    phys = trace_energy_wh(t, sp, gr,
                           mass_kg=vehicle.curb_mass_kg + occupant_kg,
                           vehicle=vehicle, temp_c=float(cond.get("temp_c", 20.0)),
                           headwind_ms=float(cond.get("headwind_ms", 0.0)),
                           aux_power_w=np.full(len(sp), aux_w + vehicle.aux_power_w))

    return dict(
        distance_km=dist_km,
        mean_speed_kmh=mean_speed * 3.6,
        speed_std_kmh=speed_std * 3.6,
        stops_per_km=float(stops.sum() / dist_km),
        elev_gain_m=float((L * np.clip(g, 0, None)).sum()),
        elev_loss_m=float((L * np.clip(-g, 0, None)).sum()),
        net_elev_m=float((L * g).sum()),
        abs_grade_pct=float((L * np.abs(g)).sum() / dist_m * 100),
        temp_c=float(cond.get("temp_c", 20.0)),
        aux_power_w=aux_w,
        hvac_on=float(aux_w > 10.0),
        soc_start=float(cond.get("soc_start", 100.0)),
        phys_wh_km=phys["total_wh"] / dist_km,
    )
