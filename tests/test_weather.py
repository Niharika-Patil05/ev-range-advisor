"""Tests for the Open-Meteo archive join.

The network is never touched: `fetch_hourly` caches to disk, so the tests seed the
cache and assert on the geometry and the join.
"""
import json

import numpy as np
import pandas as pd
import pytest

from src.data.weather_archive import (ANN_ARBOR, attach_weather, bearing_deg,
                                      fetch_hourly, headwind_component)


def test_bearing_north_east_south_west():
    assert bearing_deg(0, 0, 1, 0) == pytest.approx(0, abs=1e-6)     # north
    assert bearing_deg(0, 0, 0, 1) == pytest.approx(90, abs=1e-6)    # east
    assert bearing_deg(1, 0, 0, 0) == pytest.approx(180, abs=1e-6)   # south
    assert bearing_deg(0, 1, 0, 0) == pytest.approx(270, abs=1e-6)   # west


def test_headwind_sign_convention():
    """`wind_from_deg` is meteorological: the direction the wind blows FROM."""
    assert headwind_component(5, 0, 0) == pytest.approx(5)      # heading north, wind from north
    assert headwind_component(5, 180, 0) == pytest.approx(-5)   # tailwind
    assert abs(headwind_component(5, 90, 0)) < 1e-9             # crosswind


def _seed_cache(tmp_path, hours=48, start="2017-12-30", end="2018-01-05"):
    """Write the cache file under the EXACT name attach_weather will look for.

    Getting this wrong does not fail loudly: a cache miss silently issues a live
    request, so the test would pass or fail against real Ann Arbor weather. The
    date range is therefore derived the same way the code derives it.
    """
    times = pd.date_range("2018-01-01", periods=hours, freq="h")
    payload = {"hourly": {
        "time": [t.isoformat() for t in times],
        "temperature_2m": list(np.linspace(-5, 5, hours)),
        "wind_speed_10m": [4.0] * hours,
        "wind_direction_10m": [0.0] * hours,       # wind from the north
        "precipitation": [0.0] * (hours // 2) + [1.0] * (hours - hours // 2),
    }}
    lat, lon = ANN_ARBOR
    (tmp_path / f"openmeteo_{lat:.2f}_{lon:.2f}_{start}_{end}.json").write_text(
        json.dumps(payload))
    return times


def _range_for(ts) -> tuple[str, str]:
    """Mirror of the window attach_weather requests: min-2d to max+2d."""
    ts = pd.to_datetime(pd.Series(ts))
    return ((ts.min() - pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
            (ts.max() + pd.Timedelta(days=2)).strftime("%Y-%m-%d"))


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Any cache miss must fail the test rather than quietly going to the network."""
    import requests

    def _boom(*a, **k):
        raise AssertionError("test attempted a live HTTP request; the cache name "
                             "must match what attach_weather derives")
    monkeypatch.setattr(requests, "get", _boom)


def test_fetch_hourly_uses_the_cache_and_never_calls_the_network(tmp_path):
    _seed_cache(tmp_path)
    df = fetch_hourly(*ANN_ARBOR, "2017-12-30", "2018-01-05", cache_dir=tmp_path)
    assert len(df) == 48 and "wind_speed_10m" in df.columns


def test_attach_weather_joins_on_place_and_utc_hour(tmp_path):
    times = pd.date_range("2018-01-01", periods=48, freq="h")
    start, end = _range_for([times[3], times[40]])
    _seed_cache(tmp_path, start=start, end=end)
    seg = pd.DataFrame({
        "timestamp_utc": [times[3], times[40]],
        # heading due north, into a wind blowing from the north
        "lat_start": [42.20, 42.20], "lon_start": [-83.74, -83.74],
        "lat_end": [42.30, 42.30], "lon_end": [-83.74, -83.74],
    })
    out = attach_weather(seg, *ANN_ARBOR, cache_dir=tmp_path)
    assert out.headwind_ms.iloc[0] == pytest.approx(4.0, abs=1e-6)   # full headwind
    assert out.rain.iloc[0] == 0.0 and out.rain.iloc[1] == 1.0


def test_attach_weather_requires_a_timestamp():
    with pytest.raises(KeyError, match="timestamp_utc"):
        attach_weather(pd.DataFrame({"lat_start": [0.0]}))


def test_heading_reverses_headwind_into_tailwind(tmp_path):
    times = pd.date_range("2018-01-01", periods=48, freq="h")
    start, end = _range_for([times[3]])
    _seed_cache(tmp_path, start=start, end=end)
    south = pd.DataFrame({
        "timestamp_utc": [times[3]],
        "lat_start": [42.30], "lon_start": [-83.74],
        "lat_end": [42.20], "lon_end": [-83.74],     # heading south, wind from north
    })
    out = attach_weather(south, *ANN_ARBOR, cache_dir=tmp_path)
    assert out.headwind_ms.iloc[0] == pytest.approx(-4.0, abs=1e-6)
