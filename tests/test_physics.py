from dataclasses import replace

import numpy as np
import pytest

from src.config import E_SCOOTER_PLACEHOLDER
from src.physics.road_load import G0, air_density, segment_energy_wh

V = E_SCOOTER_PLACEHOLDER


def one(grade=0.0, speed=10.0, stops=0.0, mass=170.0, vehicle=V, **kw):
    return float(segment_energy_wh([1000.0], [grade], [speed], [stops], mass_kg=mass, vehicle=vehicle, **kw)[0])


def test_flat_matches_hand_calculation():
    rho = air_density(25.0)
    f = mass = 170.0
    f_roll = mass * G0 * V.crr
    f_aero = 0.5 * rho * V.cd * V.frontal_area_m2 * 10.0**2
    expected = ((f_roll + f_aero) * 1000 / V.drivetrain_eff + V.aux_power_w * 100.0) / 3600.0
    assert one() == pytest.approx(expected, rel=1e-9)


def test_uphill_costs_more_downhill_less():
    assert one(grade=0.05) > one() > one(grade=-0.03)


def test_no_regen_downhill_never_negative():
    v = replace(V, regen_eff=0.0)
    assert one(grade=-0.10, vehicle=v) >= 0.0


def test_headwind_load_and_stops_increase_energy():
    assert one(headwind_ms=5.0) > one()
    assert one(mass=230.0) > one()
    assert one(stops=2.0) > one()


def test_aux_load_increases_energy():
    assert one(aux_extra_w=60.0) > one()


# --------------------------------------------------------------- trace integration
from src.config import NISSAN_LEAF_2013  # noqa: E402
from src.physics.road_load import trace_energy_wh  # noqa: E402

LEAF = NISSAN_LEAF_2013


def steady(v_ms=15.0, n=601, grade=0.0, mass=1588.0, **kw):
    t = np.arange(float(n))
    return trace_energy_wh(t, np.full(n, v_ms), grade, mass_kg=mass,
                           vehicle=LEAF, aux_power_w=np.zeros(n), **kw)


def test_trace_steady_cruise_matches_hand_calculation():
    """No acceleration, no grade, no aux: energy is roll + aero only."""
    from src.physics.road_load import air_density
    v, n, mass = 15.0, 601, 1588.0
    rho = air_density(25.0)
    f_roll = mass * G0 * LEAF.crr
    f_aero = 0.5 * rho * LEAF.cd * LEAF.frontal_area_m2 * v**2
    expected = (f_roll + f_aero) * v * (n - 1) / LEAF.drivetrain_eff / 3600.0
    assert steady()["total_wh"] == pytest.approx(expected, rel=1e-9)


def test_trace_components_sum_to_total():
    r = steady()
    parts = r["roll_wh"] + r["aero_wh"] + r["grade_wh"] + r["inertia_wh"] + r["aux_wh"]
    assert parts == pytest.approx(r["total_wh"], rel=1e-9)


def test_trace_steady_speed_has_no_inertia_term():
    assert steady()["inertia_wh"] == pytest.approx(0.0, abs=1e-9)


def test_trace_uphill_costs_more_than_flat_costs_more_than_downhill():
    assert steady(grade=0.03)["total_wh"] > steady()["total_wh"] > steady(grade=-0.03)["total_wh"]


def test_trace_aero_scales_with_square_of_speed():
    """Doubling speed over the same DISTANCE should roughly quadruple aero energy."""
    slow = steady(v_ms=10.0, n=1001)     # 10 m/s for 1000 s = 10 km
    fast = steady(v_ms=20.0, n=501)      # 20 m/s for  500 s = 10 km
    assert fast["aero_wh"] / slow["aero_wh"] == pytest.approx(4.0, rel=1e-6)


def test_trace_round_trip_over_a_hill_still_costs_energy():
    """Climb then descend to the same altitude. Because going up costs 1/eta and
    coming down returns only regen_eff, the terrain is not free -- this is the
    asymmetry that makes elevation gain a real feature in flat Ann Arbor."""
    n = 600
    t = np.arange(float(n))
    v = np.full(n, 15.0)
    grade = np.concatenate([np.full(n // 2, 0.02), np.full(n - n // 2, -0.02)])
    r = trace_energy_wh(t, v, grade, mass_kg=1588.0, vehicle=LEAF,
                        aux_power_w=np.zeros(n))
    flat = trace_energy_wh(t, v, 0.0, mass_kg=1588.0, vehicle=LEAF,
                           aux_power_w=np.zeros(n))
    assert r["total_wh"] > flat["total_wh"]


def test_trace_accelerating_costs_more_than_steady():
    t = np.arange(101.0)
    accel = trace_energy_wh(t, np.linspace(0, 20, 101), 0.0, mass_kg=1588.0,
                            vehicle=LEAF, aux_power_w=np.zeros(101))
    assert accel["inertia_wh"] > 0


def test_trace_measured_aux_load_is_added():
    base = steady()
    with_aux = trace_energy_wh(np.arange(601.0), np.full(601, 15.0), 0.0,
                               mass_kg=1588.0, vehicle=LEAF,
                               aux_power_w=np.full(601, 1000.0))
    # 1000 W for 600 s = 166.7 Wh
    assert with_aux["total_wh"] - base["total_wh"] == pytest.approx(1000 * 600 / 3600, rel=1e-6)


def test_trace_rejects_non_increasing_time():
    with pytest.raises(ValueError, match="strictly increasing"):
        trace_energy_wh([0.0, 1.0, 1.0], [10.0, 10.0, 10.0], 0.0,
                        mass_kg=1588.0, vehicle=LEAF)


def test_trace_headwind_increases_energy():
    assert steady(headwind_ms=5.0)["total_wh"] > steady()["total_wh"]
