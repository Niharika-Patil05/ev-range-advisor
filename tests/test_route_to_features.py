import numpy as np
import pandas as pd
import pytest

from src.config import NISSAN_LEAF_2013 as LEAF
from src.features.route_to_features import route_to_features, segments_to_trace
from src.features.segment_features import DEPLOYABLE_WITH_PHYSICS


def route(n=10, length=500.0, speed=14.0, grade=0.0, stops=0.0):
    return pd.DataFrame({"length_m": np.full(n, length), "speed_ms": np.full(n, speed),
                         "grade": np.full(n, grade), "stops": np.full(n, stops)})


def test_produces_exactly_the_deployable_feature_row():
    f = route_to_features(route(), {"temp_c": 20.0, "soc_start": 90.0}, LEAF)
    assert set(f) == set(DEPLOYABLE_WITH_PHYSICS)


def test_distance_and_grade_features_match_hand_calculation():
    f = route_to_features(route(n=10, length=500.0, grade=0.02),
                          {"temp_c": 20.0}, LEAF)
    assert f["distance_km"] == pytest.approx(5.0)
    assert f["elev_gain_m"] == pytest.approx(100.0)      # 5000 m x 0.02
    assert f["elev_loss_m"] == pytest.approx(0.0)
    assert f["net_elev_m"] == pytest.approx(100.0)
    assert f["abs_grade_pct"] == pytest.approx(2.0)


def test_reconstructed_trace_preserves_distance():
    seg = route(n=8, length=400.0, speed=12.0)
    t, v, g = segments_to_trace(seg)
    assert np.trapezoid(v, t) == pytest.approx(seg.length_m.sum(), rel=0.02)


def test_stops_put_energy_into_the_physics_baseline():
    """Stop-start is what makes urban driving expensive; a plan with stops must cost
    more than the same route driven without them."""
    cond = {"temp_c": 20.0}
    without = route_to_features(route(stops=0.0), cond, LEAF)["phys_wh_km"]
    with_stops = route_to_features(route(stops=2.0), cond, LEAF)["phys_wh_km"]
    assert with_stops > without


def test_uphill_route_costs_more_than_flat():
    cond = {"temp_c": 20.0}
    assert (route_to_features(route(grade=0.03), cond, LEAF)["phys_wh_km"]
            > route_to_features(route(grade=0.0), cond, LEAF)["phys_wh_km"])


def test_auxiliary_load_raises_the_baseline_and_sets_the_flag():
    cold = route_to_features(route(), {"temp_c": -5.0, "aux_power_w": 2000.0}, LEAF)
    mild = route_to_features(route(), {"temp_c": -5.0, "aux_power_w": 0.0}, LEAF)
    assert cold["phys_wh_km"] > mild["phys_wh_km"]
    assert cold["hvac_on"] == 1.0 and mild["hvac_on"] == 0.0


def test_physics_baseline_is_physically_plausible_for_a_leaf():
    f = route_to_features(route(speed=14.0), {"temp_c": 20.0}, LEAF)
    assert 60.0 < f["phys_wh_km"] < 300.0


def test_reconstructed_trace_preserves_distance_even_with_stops():
    """A stop adds TIME, it does not remove road. An earlier version overwrote
    cruise samples with the deceleration dip, which shortened the route and made
    stops appear to cost nothing."""
    seg = route(n=6, length=600.0, speed=13.0, stops=2.0)
    t, v, g = segments_to_trace(seg)
    assert np.trapezoid(v, t) == pytest.approx(seg.length_m.sum(), rel=0.05)


def test_more_stops_cost_monotonically_more():
    cond = {"temp_c": 20.0}
    e = [route_to_features(route(stops=k), cond, LEAF)["phys_wh_km"] for k in (0.0, 1.0, 3.0)]
    assert e[0] < e[1] < e[2]
