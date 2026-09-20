import numpy as np
import pytest

from src.features.route_pipeline import (bearing_deg, build_segments, grades_from_elevation,
                                         haversine_m, headwind_component, resample_polyline)


def test_haversine_one_degree_latitude():
    assert haversine_m(0, 0, 1, 0) == pytest.approx(111195, rel=1e-3)


def test_resample_preserves_length_and_spacing():
    lats = np.linspace(16.0, 16.1, 20)
    lons = np.full(20, 74.0)
    la, lo, d = resample_polyline(lats, lons, spacing_m=500)
    assert d[0] == 0 and d[-1] == pytest.approx(haversine_m(16.0, 74.0, 16.1, 74.0), rel=1e-3)
    assert np.allclose(np.diff(d), np.diff(d)[0])


def test_grade_from_constant_slope():
    dist = np.arange(0, 3001, 300.0)
    g = grades_from_elevation(dist, dist * 0.05)
    assert np.allclose(g, 0.05, atol=1e-6)


def test_headwind_geometry():
    assert headwind_component(5, 0, 0) == pytest.approx(5)      # heading north, wind from north
    assert headwind_component(5, 180, 0) == pytest.approx(-5)   # tailwind
    assert abs(headwind_component(5, 90, 0)) < 1e-9            # crosswind
    assert bearing_deg(0, 0, 1, 0) == pytest.approx(0, abs=1e-6)


def test_build_segments_speed_capped():
    dist = np.arange(0, 3001, 300.0)
    seg = build_segments(dist, np.zeros_like(dist), avg_speed_ms=40.0, stops_per_km=1.0)
    assert seg["speed_ms"].max() <= 15.0 and len(seg) == 10
