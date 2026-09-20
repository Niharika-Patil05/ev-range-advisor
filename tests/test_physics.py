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
