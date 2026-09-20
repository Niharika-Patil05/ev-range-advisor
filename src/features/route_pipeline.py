"""Route -> segment table (length, grade, speed, stops) + weather.

Pure-geometry helpers (tested offline) + thin network wrappers around free, key-less services:
  * OSRM public demo server  : route geometry  (router.project-osrm.org)
  * Open-Meteo elevation API : elevation       (SRTM-based, ~90 m resolution)
  * Open-Meteo forecast API  : temperature / wind / rain
All responses are cached on disk. If any call fails, RoutePlanningError is raised and the app falls
back to the demo presets. NOTE: the network wrappers could not be tested in the build sandbox;
test them on a machine with internet (see README). For road-type / speed-limit features, upgrade
to osmnx (Phase 2, Track C).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CACHE_DIR, VehicleSpec

OSRM_URL = "https://router.project-osrm.org/route/v1/driving"
ELEV_URL = "https://api.open-meteo.com/v1/elevation"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


class RoutePlanningError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- geometry helpers
def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlmb = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(lon2 - lon1)
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return float((np.degrees(np.arctan2(x, y)) + 360) % 360)


def headwind_component(wind_speed_ms: float, wind_from_deg: float, heading_deg: float) -> float:
    """Positive = headwind. `wind_from_deg` is the direction the wind blows FROM (meteorological)."""
    return float(wind_speed_ms * np.cos(np.radians(heading_deg - wind_from_deg)))


def resample_polyline(lats, lons, spacing_m: float = 300.0, max_points: int = 300):
    """Resample a polyline to ~evenly spaced points. Returns (lat, lon, cumulative_distance_m)."""
    lats, lons = np.asarray(lats, float), np.asarray(lons, float)
    step = haversine_m(lats[:-1], lons[:-1], lats[1:], lons[1:])
    cum = np.concatenate([[0.0], np.cumsum(step)])
    total = cum[-1]
    if total <= 0:
        raise RoutePlanningError("Route has zero length.")
    n = int(min(max_points, max(2, np.ceil(total / spacing_m) + 1)))
    targets = np.linspace(0, total, n)
    return np.interp(targets, cum, lats), np.interp(targets, cum, lons), targets


def grades_from_elevation(dist_m, elev_m, smooth: int = 3, max_grade: float = 0.15) -> np.ndarray:
    """Per-segment grade (rise/run) from elevation samples. SRTM is noisy, so the grade series is
    smoothed with a centred moving average (smoothing grade, not elevation, keeps the ends unbiased)."""
    dz = np.diff(np.asarray(elev_m, float))
    dd = np.maximum(np.diff(np.asarray(dist_m, float)), 1.0)
    grade = pd.Series(dz / dd).rolling(smooth, center=True, min_periods=1).mean().to_numpy()
    return np.clip(grade, -max_grade, max_grade)


def build_segments(dist_m, elev_m, avg_speed_ms: float, stops_per_km: float,
                   vehicle: VehicleSpec) -> pd.DataFrame:
    dist_m = np.asarray(dist_m, float)
    length = np.diff(dist_m)
    return pd.DataFrame(dict(
        length_m=length,
        grade=grades_from_elevation(dist_m, elev_m),
        speed_ms=np.full(len(length), float(np.clip(avg_speed_ms, 3.0, vehicle.max_speed_ms))),
        stops=stops_per_km * length / 1000.0,
    ))


# ----------------------------------------------------------------------------- network wrappers
def _cached_get(url: str, params: dict, timeout: float = 15.0) -> dict:
    import requests

    key = hashlib.sha1((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()
    path = Path(CACHE_DIR) / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # network down, rate limit, bad JSON ...
        raise RoutePlanningError(f"Request to {url} failed: {exc}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def fetch_osrm_route(origin, dest) -> dict:
    """origin/dest = (lat, lon). Returns dict(lats, lons, duration_s)."""
    coords = f"{origin[1]},{origin[0]};{dest[1]},{dest[0]}"
    data = _cached_get(f"{OSRM_URL}/{coords}", dict(overview="full", geometries="geojson"))
    if data.get("code") != "Ok" or not data.get("routes"):
        raise RoutePlanningError(f"OSRM found no route ({data.get('code')}).")
    route = data["routes"][0]
    pts = np.array(route["geometry"]["coordinates"])  # [lon, lat]
    return dict(lats=pts[:, 1], lons=pts[:, 0], duration_s=float(route["duration"]),
                distance_m=float(route["distance"]))


def fetch_elevations(lats, lons) -> np.ndarray:
    out = []
    for i in range(0, len(lats), 100):  # API limit: 100 coordinates per request
        la, lo = lats[i:i + 100], lons[i:i + 100]
        data = _cached_get(ELEV_URL, dict(latitude=",".join(f"{x:.5f}" for x in la),
                                          longitude=",".join(f"{x:.5f}" for x in lo)))
        out.extend(data["elevation"])
    return np.asarray(out, float)


def fetch_weather(lat: float, lon: float) -> dict:
    data = _cached_get(WEATHER_URL, dict(
        latitude=round(lat, 3), longitude=round(lon, 3), wind_speed_unit="ms",
        current="temperature_2m,precipitation,wind_speed_10m,wind_direction_10m"))
    cur = data.get("current")
    if not cur:
        raise RoutePlanningError("Weather API returned no current data.")
    return dict(temp_c=cur["temperature_2m"], rain=int(cur["precipitation"] > 0.1),
                wind_speed_ms=cur["wind_speed_10m"], wind_from_deg=cur["wind_direction_10m"])


def plan_route(origin, dest, vehicle: VehicleSpec, stops_per_km: float = 1.0):
    """Full online pipeline. Returns (segments, weather_dict_with_headwind, meta)."""
    route = fetch_osrm_route(origin, dest)
    lat, lon, dist = resample_polyline(route["lats"], route["lons"])
    elev = fetch_elevations(lat, lon)
    avg_speed = route["distance_m"] / max(route["duration_s"], 1.0)
    segments = build_segments(dist, elev, avg_speed, stops_per_km, vehicle)
    w = fetch_weather(*origin)
    heading = bearing_deg(origin[0], origin[1], dest[0], dest[1])
    weather = dict(temp_c=w["temp_c"], rain=w["rain"],
                   headwind_ms=headwind_component(w["wind_speed_ms"], w["wind_from_deg"], heading))
    meta = dict(distance_km=route["distance_m"] / 1000, elevation=elev, dist_m=dist)
    return segments, weather, meta


# ----------------------------------------------------------------------------- offline demo presets
PRESET_ROUTES = {
    "City commute (flat, ~12 km)": ("city", 12.0, 101),
    "Ghat climb (hilly, ~25 km)": ("hilly", 25.0, 202),
    "Highway run (~45 km)": ("highway", 45.0, 303),
}


def load_preset(name: str, vehicle: VehicleSpec) -> pd.DataFrame:
    from ..data.synthetic import make_route

    rtype, dist, seed = PRESET_ROUTES[name]
    return make_route(np.random.default_rng(seed), vehicle, rtype, dist)
