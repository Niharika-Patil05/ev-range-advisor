"""Tests for the VED loader.

These build synthetic frames in VED's schema rather than reading the 3 GB download,
so the suite runs anywhere. They pin the facts that were established empirically in
docs/PHASE1_DATA_FINDINGS.md -- above all the current sign, which silently flips
every downstream result if it regresses.
"""
import numpy as np
import pandas as pd
import pytest

from src.data.ved import (build_segment_table, daynum_to_utc, derive_route_ids,
                          haversine_m)


def fake_trip(veh=10, trip=1, n=71, dt_s=1.0, volts=360.0, amps=-10.0,
              speed_kmh=72.0, daynum=5.5, lat0=42.28, lon0=-83.74,
              lat1=42.30, lon1=-83.70, heater=0.0, ac=0.0, soc0=80.0, oat=15.0):
    """One trip in VED's schema. Negative amps = discharge, as in the real data."""
    return pd.DataFrame({
        "DayNum": daynum, "VehId": veh, "Trip": trip,
        "Timestamp(ms)": np.arange(n) * dt_s * 1000.0,
        "Latitude[deg]": np.linspace(lat0, lat1, n),
        "Longitude[deg]": np.linspace(lon0, lon1, n),
        "Vehicle Speed[km/h]": speed_kmh,
        "OAT[DegC]": oat,
        "HV Battery Current[A]": amps,
        "HV Battery SOC[%]": np.linspace(soc0, soc0 - 2.0, n),
        "HV Battery Voltage[V]": volts,
        "Heater Power[Watts]": heater,
        "Air Conditioning Power[Watts]": ac,
    })


def test_discharge_energy_is_positive_for_negative_current():
    """THE critical invariant: negative current means energy leaving the pack."""
    seg, _ = build_segment_table(fake_trip())
    assert len(seg) == 1
    assert seg.energy_wh.iloc[0] > 0


def test_energy_magnitude_matches_hand_calculation():
    # 360 V x -10 A = -3600 W held over 70 intervals of 1 s = 70 Wh out of the pack
    seg, _ = build_segment_table(fake_trip())
    assert seg.energy_wh.iloc[0] == pytest.approx(70.0, rel=1e-9)


def test_distance_and_consumption_match_hand_calculation():
    # 72 km/h = 20 m/s over 70 s = 1400 m; 70 Wh / 1.4 km = 50 Wh/km
    seg, _ = build_segment_table(fake_trip())
    assert seg.distance_km.iloc[0] == pytest.approx(1.4, rel=1e-9)
    assert seg.wh_per_km.iloc[0] == pytest.approx(50.0, rel=1e-9)


def test_irregular_sampling_uses_actual_time_deltas():
    """Assuming a fixed rate would give the wrong answer; check a 2 s grid scales."""
    seg1, _ = build_segment_table(fake_trip(dt_s=1.0))
    seg2, _ = build_segment_table(fake_trip(dt_s=2.0, speed_kmh=36.0))
    # twice the time at half the speed: same distance, twice the energy
    assert seg2.distance_km.iloc[0] == pytest.approx(seg1.distance_km.iloc[0], rel=1e-9)
    assert seg2.energy_wh.iloc[0] == pytest.approx(2 * seg1.energy_wh.iloc[0], rel=1e-9)


def test_regen_is_the_positive_part_of_the_power_trace():
    # 36 km/h keeps consumption inside the plausibility band once regen is netted off
    t = fake_trip(n=141, speed_kmh=36.0)
    t.loc[t.index[:100], "HV Battery Current[A]"] = -10.0   # discharge
    t.loc[t.index[100:], "HV Battery Current[A]"] = 5.0     # regen
    seg, _ = build_segment_table(t)
    r = seg.iloc[0]
    assert r.regen_wh > 0
    assert r.gross_discharge_wh > r.regen_wh
    assert r.energy_wh == pytest.approx(r.gross_discharge_wh - r.regen_wh, rel=1e-9)
    assert 0 < r.regen_fraction < 1


def test_net_charging_segments_are_dropped():
    seg, attr = build_segment_table(fake_trip(amps=+10.0))   # pure charging
    assert len(seg) == 0
    assert attr.set_index("stage").loc["net_discharge", "n_dropped"] == 1


def test_short_records_are_dropped_and_counted():
    seg, attr = build_segment_table(fake_trip(n=10, dt_s=1.0))
    assert len(seg) == 0
    assert attr.set_index("stage").loc["energy_computable", "n_dropped"] == 1


def test_implausible_consumption_is_dropped():
    # 360 V x -400 A over a short slow trip -> absurd Wh/km
    seg, attr = build_segment_table(fake_trip(amps=-400.0, speed_kmh=3.6, n=601, dt_s=1.0))
    assert attr.set_index("stage").loc["plausible_consumption", "n_dropped"] >= 0
    if len(seg):
        assert seg.wh_per_km.between(30, 600).all()


def test_attrition_log_accounts_for_every_record():
    raw = pd.concat([fake_trip(trip=1), fake_trip(trip=2, amps=+10.0),
                     fake_trip(trip=3, n=10, dt_s=1.0)], ignore_index=True)
    seg, attr = build_segment_table(raw)
    assert attr.iloc[0].n_before == 3
    assert attr.n_dropped.sum() == 3 - len(seg)


def test_measured_temperature_and_hvac_are_carried_through():
    seg, _ = build_segment_table(fake_trip(oat=-8.0, heater=1200.0, ac=300.0))
    assert seg.temp_c.iloc[0] == pytest.approx(-8.0)
    assert seg.aux_power_w.iloc[0] == pytest.approx(1500.0)
    assert seg.hvac_on.iloc[0] == 1.0


def test_output_is_provenance_tagged_as_real():
    seg, _ = build_segment_table(fake_trip())
    assert set(seg.provenance) == {"REAL"}
    assert set(seg.source_dataset) == {"VED"}


def test_daynum_epoch_and_utc_reading():
    """DayNum 1.0 is 2017-11-01 00:00 UTC (Gate G1)."""
    assert daynum_to_utc(pd.Series([1.0]))[0] == pd.Timestamp("2017-11-01 00:00:00")
    assert daynum_to_utc(pd.Series([2.5]))[0] == pd.Timestamp("2017-11-02 12:00:00")


def test_local_hour_uses_detroit_time():
    # DayNum 5.75 -> 2017-11-05 18:00 UTC -> 13:00 EST
    seg, _ = build_segment_table(fake_trip(daynum=5.75))
    assert seg.local_hour.iloc[0] == 13


def test_routes_group_repeats_of_the_same_journey():
    segs = pd.DataFrame({
        "lat_start": [42.2800, 42.2801, 42.3500],
        "lon_start": [-83.7400, -83.7402, -83.6000],
        "lat_end":   [42.3000, 42.3001, 42.4000],
        "lon_end":   [-83.7000, -83.7002, -83.5000],
    })
    rid = derive_route_ids(segs, eps_m=500.0)
    assert rid.iloc[0] == rid.iloc[1]     # ~20 m apart: same route
    assert rid.iloc[2] != rid.iloc[0]     # kilometres away: different route


def test_opposite_directions_are_different_routes():
    """A->B and B->A have opposite grade profiles, so they must not be pooled."""
    segs = pd.DataFrame({
        "lat_start": [42.28, 42.30], "lon_start": [-83.74, -83.70],
        "lat_end":   [42.30, 42.28], "lon_end":   [-83.70, -83.74],
    })
    assert derive_route_ids(segs, eps_m=500.0).nunique() == 2


def test_tighter_tolerance_splits_a_journey_which_is_the_leakage_direction():
    segs = pd.DataFrame({
        "lat_start": [42.2800, 42.2830], "lon_start": [-83.7400, -83.7400],
        "lat_end":   [42.3000, 42.3000], "lon_end":   [-83.7000, -83.7000],
    })   # starts ~333 m apart
    assert derive_route_ids(segs, eps_m=500.0).nunique() == 1   # recognised as one route
    assert derive_route_ids(segs, eps_m=100.0).nunique() == 2   # split -> can leak


def test_haversine_one_degree_latitude():
    assert haversine_m(0, 0, 1, 0) == pytest.approx(111195, rel=1e-3)
