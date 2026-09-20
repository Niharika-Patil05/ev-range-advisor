import numpy as np
import pandas as pd
import pytest

from src.evaluation.splits import (CrossVehicleModel, LeaveOneBatteryOut,
                                   LeaveOneVehicleOut, RandomRows, RouteWise,
                                   TemperatureRegime)


def toy(n_routes=9, per_route=8, seed=0):
    """Routes span a range of temperatures, as real trips do."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_routes):
        for t in range(per_route):
            rows.append(dict(route_id=r,
                             vehicle_id=f"V{r % 3}",
                             vehicle_model="leaf" if r % 3 else "phev",
                             temp_c=float(rng.uniform(-5, 35)),
                             y=rng.normal()))
    return pd.DataFrame(rows)


def test_routewise_never_shares_a_route():
    df = toy()
    for tr, te in RouteWise(n_splits=3).split(df):
        assert not set(df.iloc[tr].route_id) & set(df.iloc[te].route_id)


def test_routewise_covers_every_row_across_folds():
    df = toy()
    seen = np.concatenate([te for _, te in RouteWise(n_splits=3).split(df)])
    assert sorted(seen) == list(range(len(df)))


def test_random_rows_does_leak_routes_which_is_the_point():
    df = toy()
    tr, te = next(RandomRows(seed=0).split(df))
    # L0 exists to demonstrate leakage; if this ever stopped overlapping,
    # experiment E0 would have nothing to measure.
    assert set(df.iloc[tr].route_id) & set(df.iloc[te].route_id)


def test_temperature_regime_tests_on_cold_and_trains_on_warm():
    df = toy()
    proto = TemperatureRegime(cold_quantile=1 / 3)
    tr, te = next(proto.split(df))
    assert df.iloc[te].temp_c.max() <= proto.last_attrition_["cold_threshold_c"]
    assert df.iloc[tr].temp_c.min() > proto.last_attrition_["cold_threshold_c"]


def test_temperature_regime_keeps_routes_disjoint():
    df = toy()
    tr, te = next(TemperatureRegime().split(df))
    assert not set(df.iloc[tr].route_id) & set(df.iloc[te].route_id)


def test_temperature_regime_records_what_it_dropped():
    df = toy()
    proto = TemperatureRegime()
    next(proto.split(df))
    a = proto.last_attrition_
    # the two-stage partition necessarily discards the warm segments of test-routes
    # and the cold segments of train-routes; we report how many rather than lose them
    assert a["n_discarded"] > 0
    assert a["n_train_kept"] + a["n_test_kept"] + a["n_discarded"] == len(df)


def test_temperature_regime_survives_routes_that_span_all_seasons():
    """Every route is driven in all weather -- the real VED situation. A naive
    filter-on-temperature-first protocol empties the training set here."""
    df = toy(n_routes=12, per_route=10)
    assert df.groupby("route_id").temp_c.min().max() < df.temp_c.quantile(1 / 3)
    tr, te = next(TemperatureRegime().split(df))
    assert len(tr) > 0 and len(te) > 0


def test_temperature_regime_without_route_guard_leaks_routes():
    df = toy()
    tr, te = next(TemperatureRegime(enforce_route_disjoint=False).split(df))
    # documents precisely the confound the guard removes
    assert set(df.iloc[tr].route_id) & set(df.iloc[te].route_id)


def test_leave_one_vehicle_out_holds_out_exactly_one():
    df = toy()
    folds = list(LeaveOneVehicleOut().split(df))
    assert len(folds) == df.vehicle_id.nunique()
    for tr, te in folds:
        assert df.iloc[te].vehicle_id.nunique() == 1
        assert not set(df.iloc[tr].vehicle_id) & set(df.iloc[te].vehicle_id)


def test_cross_vehicle_model_transfer():
    df = toy()
    tr, te = next(CrossVehicleModel(train_models=["leaf"], test_models=["phev"]).split(df))
    assert set(df.iloc[tr].vehicle_model) == {"leaf"}
    assert set(df.iloc[te].vehicle_model) == {"phev"}


def test_cross_vehicle_model_rejects_overlapping_sets():
    df = toy()
    with pytest.raises(ValueError, match="overlap"):
        next(CrossVehicleModel(train_models=["leaf"], test_models=["leaf"]).split(df))


def test_leave_one_battery_out():
    cells = pd.DataFrame({"battery_id": ["B0"] * 3 + ["B1"] * 3, "soh": [1.0] * 6})
    folds = list(LeaveOneBatteryOut().split(cells))
    assert len(folds) == 2
    for tr, te in folds:
        assert not set(cells.iloc[tr].battery_id) & set(cells.iloc[te].battery_id)


def test_leave_one_vehicle_out_does_not_promise_route_disjointness():
    """L3 separates vehicles, not roads. Two vehicles can drive the same route, so
    asserting route disjointness there would assert something never promised."""
    df = toy()
    assert LeaveOneVehicleOut().disjoint_keys == ("vehicle_id",)
    assert "route_id" not in LeaveOneVehicleOut().disjoint_keys


def test_l3b_separates_vehicles_and_routes():
    from src.evaluation.splits import LeaveOneVehicleOutRouteDisjoint
    df = toy()
    proto = LeaveOneVehicleOutRouteDisjoint()
    assert proto.disjoint_keys == ("vehicle_id", "route_id")
    for tr, te in proto.split(df):
        assert not set(df.iloc[tr].vehicle_id) & set(df.iloc[te].vehicle_id)
        assert not set(df.iloc[tr].route_id) & set(df.iloc[te].route_id)


def test_l3b_is_stricter_than_l3():
    """L3b must train on no more data than L3, because it drops shared routes."""
    from src.evaluation.splits import LeaveOneVehicleOutRouteDisjoint
    df = toy()
    l3 = {int(df.iloc[te].vehicle_id.iloc[0] != df.iloc[te].vehicle_id.iloc[0]) or
          len(tr) for tr, te in LeaveOneVehicleOut().split(df)}
    l3b = {len(tr) for tr, te in LeaveOneVehicleOutRouteDisjoint().split(df)}
    assert max(l3b) <= max(l3)


def test_random_rows_promises_nothing():
    assert RandomRows().disjoint_keys == ()
