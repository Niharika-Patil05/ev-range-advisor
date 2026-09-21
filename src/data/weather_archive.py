"""Historical weather from the Open-Meteo archive, cached on disk.

Gate G1 (docs/PHASE1_DATA_FINDINGS.md) established that VED's `DayNum` clock is UTC,
by showing that reading it as UTC yields 1 night-time trip against 277 at commute
hours while reading it as local time implies 30 night-time trips and a 21:00 rush
hour. That result is what licenses this join.

WHY ONE REQUEST COVERS EVERYTHING
---------------------------------
VED is geo-fenced to Ann Arbor, which sits inside a single reanalysis grid cell
(ECMWF IFS is ~9 km). Fetching per-segment coordinates would issue hundreds of
requests for values that are identical by construction. One hourly series for the
whole period is both faster and more honest about the resolution actually available.

WHAT THIS IS NOT
----------------
A grid-cell area average, not a roadside measurement, and "weather as it turned out"
rather than the forecast a driver had at departure. Wind in particular is a
LOW-CONFIDENCE feature; whether it carries any signal is a question for the ablation,
not an assumption.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CACHE_DIR

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
ANN_ARBOR = (42.28, -83.74)
HOURLY_VARS = ("temperature_2m", "wind_speed_10m", "wind_direction_10m", "precipitation")
RAIN_MM_THRESHOLD = 0.1


def fetch_hourly(lat: float, lon: float, start: str, end: str,
                 cache_dir: Path = CACHE_DIR, timeout: float = 120.0) -> pd.DataFrame:
    """Hourly weather for one grid cell, cached so re-runs need no network.

    Requested with `timezone=GMT` and converted explicitly downstream: the timezone
    assumption stays one visible line in our own code rather than a hidden
    server-side behaviour.
    """
    cache = Path(cache_dir) / f"openmeteo_{lat:.2f}_{lon:.2f}_{start}_{end}.json"
    if cache.exists():
        payload = json.loads(cache.read_text())
    else:
        import requests

        resp = requests.get(ARCHIVE_URL, timeout=timeout, params=dict(
            latitude=lat, longitude=lon, start_date=start, end_date=end,
            hourly=",".join(HOURLY_VARS), wind_speed_unit="ms", timezone="GMT"))
        resp.raise_for_status()
        payload = resp.json()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(payload))

    h = payload["hourly"]
    df = pd.DataFrame({k: h[k] for k in HOURLY_VARS})
    df["time_utc"] = pd.to_datetime(h["time"])
    return df


def bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2, float) - np.asarray(lon1, float))
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def headwind_component(wind_speed_ms, wind_from_deg, heading_deg):
    """Positive = headwind. `wind_from_deg` is meteorological (the direction it blows FROM)."""
    return np.asarray(wind_speed_ms, float) * np.cos(
        np.radians(np.asarray(wind_from_deg, float) - np.asarray(heading_deg, float)))


def attach_weather(segments: pd.DataFrame, lat: float = ANN_ARBOR[0],
                   lon: float = ANN_ARBOR[1], cache_dir: Path = CACHE_DIR
                   ) -> pd.DataFrame:
    """Add `headwind_ms`, `wind_speed_ms` and `rain` by joining on (place, UTC hour).

    The join key is physical: the vehicle genuinely was in that cell in that hour.
    """
    if "timestamp_utc" not in segments.columns:
        raise KeyError("segments need `timestamp_utc`; build them with src.data.ved")

    ts = pd.to_datetime(segments["timestamp_utc"])
    start = (ts.min() - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    end = (ts.max() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    wx = fetch_hourly(lat, lon, start, end, cache_dir).set_index("time_utc")

    hours = ts.dt.round("h")
    out = segments.copy()
    out["wind_speed_ms"] = hours.map(wx["wind_speed_10m"]).astype(float)
    wind_from = hours.map(wx["wind_direction_10m"]).astype(float)
    precip = hours.map(wx["precipitation"]).astype(float)

    heading = bearing_deg(out["lat_start"], out["lon_start"],
                          out["lat_end"], out["lon_end"])
    out["headwind_ms"] = headwind_component(out["wind_speed_ms"], wind_from, heading)
    out["rain"] = (precip > RAIN_MM_THRESHOLD).astype(float)
    return out
