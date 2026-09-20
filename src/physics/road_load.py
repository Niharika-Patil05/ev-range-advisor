"""Physics baseline: road-load energy model (per route segment, battery side).

For every segment of length L, speed v, grade g (rise/run):

    F_roll  = m * g0 * Crr * cos(theta)
    F_aero  = 0.5 * rho * Cd * A * v_rel * |v_rel|      (v_rel = v + headwind)
    F_grade = m * g0 * sin(theta)
    E_wheel = (F_roll + F_aero + F_grade) * L

Battery energy = E_wheel / drivetrain_eff when driving,
                 E_wheel * regen_eff      when the net wheel energy is negative (downhill braking).
Every stop costs the kinetic energy needed to re-accelerate (net of regen on braking).
Auxiliary load is integrated over the segment travel time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import VehicleSpec

G0 = 9.81  # m/s^2


def air_density(temp_c: float) -> float:
    """Dry-air density at sea level (kg/m^3)."""
    return 101325.0 / (287.05 * (temp_c + 273.15))


def segment_energy_wh(
    length_m,
    grade,
    speed_ms,
    stops,
    *,
    mass_kg: float,
    vehicle: VehicleSpec,
    temp_c: float = 25.0,
    headwind_ms: float = 0.0,
    crr_scale: float = 1.0,
    aux_extra_w: float = 0.0,
) -> np.ndarray:
    """Battery energy (Wh) for each segment. All array inputs must have equal length."""
    L = np.asarray(length_m, dtype=float)
    g = np.asarray(grade, dtype=float)
    v = np.maximum(np.asarray(speed_ms, dtype=float), 0.5)
    n_stops = np.asarray(stops, dtype=float)

    theta = np.arctan(g)
    rho = air_density(temp_c)

    f_roll = mass_kg * G0 * vehicle.crr * crr_scale * np.cos(theta)
    v_rel = v + headwind_ms
    f_aero = 0.5 * rho * vehicle.cd * vehicle.frontal_area_m2 * v_rel * np.abs(v_rel)
    f_grade = mass_kg * G0 * np.sin(theta)

    e_wheel = (f_roll + f_aero + f_grade) * L  # J
    e_batt = np.where(e_wheel >= 0, e_wheel / vehicle.drivetrain_eff, e_wheel * vehicle.regen_eff)

    ke = 0.5 * mass_kg * v**2
    e_stops = n_stops * ke * (1.0 / vehicle.drivetrain_eff - vehicle.regen_eff)

    e_aux = (vehicle.aux_power_w + aux_extra_w) * (L / v)

    return (e_batt + e_stops + e_aux) / 3600.0


def trip_energy_wh(segments: pd.DataFrame, **kwargs) -> float:
    """Total battery energy (Wh) for a segments DataFrame
    (columns: length_m, grade, speed_ms, stops)."""
    return float(
        segment_energy_wh(
            segments["length_m"], segments["grade"], segments["speed_ms"], segments["stops"], **kwargs
        ).sum()
    )


def trace_energy_wh(
    time_s,
    speed_ms,
    grade=0.0,
    *,
    mass_kg: float,
    vehicle: VehicleSpec,
    temp_c: float = 25.0,
    headwind_ms: float = 0.0,
    aux_power_w=None,
    crr_scale: float = 1.0,
) -> dict:
    """Road-load energy integrated over a measured speed trace.

    `segment_energy_wh` above works on an idealised segment table with one average
    speed per segment and a stop count. Real logged data is better than that: we have
    the actual speed every second, so acceleration is observed rather than inferred
    from a stop count, and the kinetic-energy term becomes exact instead of a
    per-stop approximation.

    Per interval, with v taken at the left endpoint and the real dt:

        F_roll  = m g Crr cos(theta)
        F_aero  = 0.5 rho Cd A v_rel |v_rel|        (v_rel = v + headwind)
        F_grade = m g sin(theta)
        F_inert = m a                               (a = dv/dt, signed)
        P_wheel = (F_roll + F_aero + F_grade + F_inert) v

    Battery energy is P_wheel/eta while tractive and P_wheel*regen_eff while the
    wheel power is negative, which is what makes terrain cost energy even on a route
    that returns to its starting altitude. Auxiliary load is added as measured.

    Returns the total plus its components, because the split is what makes the
    physics baseline interpretable in the app and in the report.
    """
    t = np.asarray(time_s, dtype=float)
    v = np.asarray(speed_ms, dtype=float)
    g = np.broadcast_to(np.asarray(grade, dtype=float), v.shape)

    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("time_s must be strictly increasing")

    v0 = v[:-1]
    accel = np.diff(v) / dt
    theta = np.arctan(g[:-1])
    rho = air_density(temp_c)

    f_roll = mass_kg * G0 * vehicle.crr * crr_scale * np.cos(theta)
    v_rel = v0 + headwind_ms
    f_aero = 0.5 * rho * vehicle.cd * vehicle.frontal_area_m2 * v_rel * np.abs(v_rel)
    f_grade = mass_kg * G0 * np.sin(theta)
    f_inert = mass_kg * accel

    # The drivetrain sees the NET force, so the split below is an attribution of the
    # total rather than four independent energies; each component is converted with
    # the efficiency branch that the net power was actually in.
    p_wheel = (f_roll + f_aero + f_grade + f_inert) * v0
    e_wheel = p_wheel * dt
    tractive = e_wheel >= 0
    conv = np.where(tractive, 1.0 / vehicle.drivetrain_eff, vehicle.regen_eff)

    e_total = float(np.nansum(e_wheel * conv))
    aux = (np.full_like(v0, vehicle.aux_power_w) if aux_power_w is None
           else np.asarray(aux_power_w, dtype=float)[:len(v0)])
    e_aux = float(np.nansum(aux * dt))

    return dict(
        total_wh=(e_total + e_aux) / 3600.0,
        roll_wh=float(np.nansum(f_roll * v0 * dt * conv)) / 3600.0,
        aero_wh=float(np.nansum(f_aero * v0 * dt * conv)) / 3600.0,
        grade_wh=float(np.nansum(f_grade * v0 * dt * conv)) / 3600.0,
        inertia_wh=float(np.nansum(f_inert * v0 * dt * conv)) / 3600.0,
        aux_wh=e_aux / 3600.0,
    )
