"""Loader for the Vehicle Energy Dataset (VED) battery-electric vehicles.

Turns VED's weekly 1-Hz-ish CSVs into one row per ANALYSIS SEGMENT, carrying a
measured consumption target and the features the rest of the pipeline expects.

Everything here is driven by what the data actually does, which in several places is
not what the dataset paper says. See docs/PHASE1_DATA_FINDINGS.md for the evidence.

Three facts that govern this module
-----------------------------------
1. NEGATIVE CURRENT IS DISCHARGE. The signed integral of V*I is negative in 100.0%
   of the 406 trips with a SoC drop over 1 pp, so discharge energy is -int(V*I dt).
   Getting this backwards silently flips every downstream result.
2. SAMPLING IS IRREGULAR. Inter-row spacing runs from 0.1 s to 13.4 s (median 0.9 s)
   and values are held, not interpolated. Energy must therefore be integrated with
   the ACTUAL time deltas; assuming 1 Hz would be wrong by a variable factor.
3. "TRIP" IS A FRAGMENT. De-identification removed rows near each journey's ends, so
   a record is a recorded segment (median 3.6 km), not a door-to-door journey. We
   call it an analysis segment and never claim otherwise.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DATA_DIR, VehicleSpec
from ..evaluation.leakage_checks import attrition_row
from ..evaluation.provenance import Provenance, tag
from ..physics.road_load import trace_energy_wh

# The three battery-electric vehicles, confirmed from VED_Static_Data_PHEV&EV.xlsx.
# 541 is retained but contributes only ~10 trips, so it cannot support a
# leave-one-vehicle-out fold on its own (docs/PHASE1_DATA_FINDINGS.md section 3.4).
BEV_VEHICLE_IDS: tuple[int, ...] = (10, 455, 541)

RAW_DIR = DATA_DIR / "raw" / "ved"
DYNAMIC_DIR = RAW_DIR / "dynamic"
EVED_DIR = DATA_DIR / "raw" / "eved" / "eVED"

# eVED is a strict SUPERSET of VED, not a companion table needing a join. Verified on
# the BEV rows of one week: identical row and trip counts, a 100% exact match on
# (VehId, Trip, Timestamp), and bit-identical HV current, voltage, speed and OAT.
# Reading eVED directly therefore removes an entire class of join risk, and is
# preferred whenever the files are present.
ENRICHED_COLUMNS = [
    "Elevation Smoothed[m]", "Elevation Raw[m]", "Gradient",
    "Matchted Latitude[deg]", "Matched Longitude[deg]",   # sic: typo is theirs
    "Speed Limit[km/h]", "Match Type",
]

NEEDED_COLUMNS = [
    "DayNum", "VehId", "Trip", "Timestamp(ms)",
    "Latitude[deg]", "Longitude[deg]", "Vehicle Speed[km/h]",
    "OAT[DegC]", "HV Battery Current[A]", "HV Battery SOC[%]",
    "HV Battery Voltage[V]", "Heater Power[Watts]", "Air Conditioning Power[Watts]",
]

# DayNum 1.0 == 2017-11-01 00:00:00. Gate G1 established that this clock is UTC:
# read as UTC it yields 1 trip between 00:00 and 05:00 local against 277 at commute
# hours; read as local time it would imply 30 night-time trips and a 21:00 rush hour.
DAYNUM_EPOCH_UTC = pd.Timestamp("2017-11-01 00:00:00")
LOCAL_TZ = "America/Detroit"

EARTH_RADIUS_M = 6_371_000.0

# Quality gates for a usable segment. Deliberately loose: the point is to exclude
# records that cannot support an energy calculation, not to exclude inconvenient ones.
MIN_ROWS = 30
MIN_DURATION_S = 60.0
MIN_DISTANCE_KM = 0.5
PLAUSIBLE_WH_PER_KM = (30.0, 600.0)   # generous; the literature band is ~120-250

# VED does not observe payload or occupancy (limitation L5), so the mass used by the
# physics baseline is an ASSUMPTION: kerb mass plus one nominal occupant. Vehicle mass
# therefore carries an unquantified error of roughly +/-5%, and it is a known confound
# rather than a measurement.
NOMINAL_OCCUPANT_KG = 80.0


def daynum_to_utc(daynum: pd.Series) -> pd.Series:
    """DayNum -> UTC timestamp. See Gate G1 in docs/PHASE1_DATA_FINDINGS.md."""
    return DAYNUM_EPOCH_UTC + pd.to_timedelta((daynum - 1.0) * 86400.0, unit="s")


def load_raw(vehicle_ids: tuple[int, ...] = BEV_VEHICLE_IDS,
             dynamic_dir: Path | None = None,
             columns: list[str] | None = None,
             prefer_enriched: bool = True) -> pd.DataFrame:
    """Read the weekly CSVs, keeping only the requested vehicles.

    Prefers eVED (which carries VED's columns plus elevation, gradient and speed
    limits) and falls back to plain VED, so the pipeline runs either way.
    """
    enriched = prefer_enriched and any(Path(EVED_DIR).glob("eVED_*_week.csv"))
    if dynamic_dir is None:
        dynamic_dir = EVED_DIR if enriched else DYNAMIC_DIR
    pattern = "eVED_*_week.csv" if enriched else "VED_*_week.csv"
    files = sorted(Path(dynamic_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No weekly CSVs in {dynamic_dir}. Run: "
            f"python scripts/fetch_data.py --dataset ved --extract   (and --dataset eved)"
        )
    cols = (columns or NEEDED_COLUMNS) + (ENRICHED_COLUMNS if enriched else [])
    frames = []
    for f in files:
        d = pd.read_csv(f, usecols=lambda c: c in cols, low_memory=False)
        d = d[d["VehId"].isin(vehicle_ids)]
        if len(d):
            frames.append(d)
    return pd.concat(frames, ignore_index=True)


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def parse_speed_limit(values) -> np.ndarray:
    """eVED stores the limit as text, and ~1% of rows carry a range like '48-40'
    where the posted limit changes along the matched way. Take the mean of the
    endpoints rather than dropping the row or silently picking the first number."""
    out = []
    for v in values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append(np.nan)
            continue
        parts = [p for p in str(v).split("-") if p.strip()]
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            out.append(np.nan)
        else:
            out.append(float(np.mean(nums)) if nums else np.nan)
    return np.asarray(out, dtype=float)


def _segment_row(d: pd.DataFrame, vehicle: VehicleSpec) -> dict | None:
    """Collapse one (VehId, Trip) group into a single feature row.

    Returns None when the record cannot support an energy calculation.
    """
    d = d.sort_values("Timestamp(ms)")
    t_s = d["Timestamp(ms)"].to_numpy(float) / 1000.0
    dt = np.diff(t_s)
    if len(dt) < MIN_ROWS - 1 or dt.sum() < MIN_DURATION_S:
        return None

    v = d["HV Battery Voltage[V]"].to_numpy(float)
    i = d["HV Battery Current[A]"].to_numpy(float)
    speed_ms = d["Vehicle Speed[km/h]"].to_numpy(float) / 3.6

    # Zero-order hold, which is how VED stores these channels: each sample is held
    # until the next one arrives, so the left-hand value applies across each interval.
    power_w = (v * i)[:-1]
    # Sign discipline, since it is easy to get backwards: negative power = discharge,
    # so the NEGATIVE part of the power trace is energy leaving the pack and the
    # POSITIVE part is regenerative braking putting energy back in.
    energy_wh = -float(np.nansum(power_w * dt) / 3600.0)              # net discharge
    gross_discharge_wh = -float(np.nansum(np.minimum(power_w, 0.0) * dt) / 3600.0)
    regen_wh = float(np.nansum(np.maximum(power_w, 0.0) * dt) / 3600.0)

    distance_km = float(np.nansum(speed_ms[:-1] * dt) / 1000.0)
    if distance_km < MIN_DISTANCE_KM:
        return None

    moving = speed_ms[:-1] > 0.5
    stops = int(np.sum(np.diff(moving.astype(int)) == -1))
    hvac = (d["Heater Power[Watts]"].to_numpy(float)
            + d["Air Conditioning Power[Watts]"].to_numpy(float))[:-1]

    lat, lon = d["Latitude[deg]"].to_numpy(float), d["Longitude[deg]"].to_numpy(float)
    gps_km = float(np.nansum(haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])) / 1000.0)
    soc = d["HV Battery SOC[%]"].to_numpy(float)

    duration_s = float(dt.sum())

    # ---- eVED enrichment, when present -------------------------------------
    # Terrain note (docs/PHASE1_DATA_FINDINGS.md): Ann Arbor is gently rolling, and
    # 96.7% of rows have |gradient| < 0.5%, so an average-grade feature is close to
    # dead. Cumulative climb is NOT: a median trip gains ~28 m, and because climbing
    # costs 1/eta while descending returns only eta_regen, terrain still accounts for
    # roughly 12% of trip energy. Hence elev_gain_m and elev_loss_m rather than a
    # mean gradient.
    extra: dict = {}
    if "Elevation Smoothed[m]" in d.columns:
        el = d["Elevation Smoothed[m]"].to_numpy(float)
        dz = np.diff(el)
        extra.update(
            elev_gain_m=float(np.nansum(np.clip(dz, 0, None))),
            elev_loss_m=float(np.nansum(np.clip(dz, None, 0)) * -1.0),
            elev_span_m=float(np.nanmax(el) - np.nanmin(el)),
            net_elev_m=float(el[-1] - el[0]),
        )
    if "Gradient" in d.columns:
        gr = d["Gradient"].to_numpy(float)[:-1]
        w = dt / max(dt.sum(), 1e-9)
        extra.update(
            mean_grade_pct=float(np.nansum(np.nan_to_num(gr) * w) * 100.0),
            abs_grade_pct=float(np.nansum(np.abs(np.nan_to_num(gr)) * w) * 100.0),
        )
    if "Speed Limit[km/h]" in d.columns:
        lim = parse_speed_limit(d["Speed Limit[km/h]"].to_numpy(object))[:-1]
        spd_kmh = speed_ms[:-1] * 3.6
        ok = np.isfinite(lim)
        if ok.any():
            # Congestion proxy: how far below the posted limit the vehicle actually
            # travelled. DERIVED, and a proxy -- VED has no traffic feed. This is how
            # the project addresses the synopsis's traffic objective reproducibly.
            extra.update(
                speed_limit_kmh=float(np.nansum(lim[ok] * dt[ok]) / np.nansum(dt[ok])),
                speed_deficit_kmh=float(
                    np.nansum((lim[ok] - spd_kmh[ok]) * dt[ok]) / np.nansum(dt[ok])),
            )
    # ---- physics baseline over the measured trace --------------------------
    # This is the `phys_wh_km` the hybrid corrects. It integrates road load over the
    # real speed trace, so acceleration is observed rather than inferred from a stop
    # count, and it uses the MEASURED auxiliary load rather than a nominal constant.
    grade_trace = (np.nan_to_num(d["Gradient"].to_numpy(float))
                   if "Gradient" in d.columns else 0.0)
    try:
        phys = trace_energy_wh(
            t_s, speed_ms, grade_trace,
            mass_kg=vehicle.curb_mass_kg + NOMINAL_OCCUPANT_KG,
            vehicle=vehicle,
            temp_c=float(np.nanmedian(d["OAT[DegC]"])),
            aux_power_w=np.nan_to_num(hvac) + vehicle.aux_power_w,
        )
        extra["phys_wh_km"] = phys["total_wh"] / distance_km
        extra.update({f"phys_{k}": v for k, v in phys.items() if k != "total_wh"})
    except ValueError:
        extra["phys_wh_km"] = np.nan

    return dict(**extra, 
        vehicle_id=int(d["VehId"].iloc[0]),
        trip=int(d["Trip"].iloc[0]),
        daynum=float(d["DayNum"].iloc[0]),
        n_rows=int(len(d)),
        duration_s=duration_s,
        distance_km=distance_km,
        gps_distance_km=gps_km,
        energy_wh=energy_wh,
        gross_discharge_wh=gross_discharge_wh,
        regen_wh=regen_wh,
        regen_fraction=regen_wh / gross_discharge_wh if gross_discharge_wh > 0 else np.nan,
        wh_per_km=energy_wh / distance_km,
        mean_speed_kmh=distance_km / (duration_s / 3600.0),
        speed_std_kmh=float(np.nanstd(speed_ms[:-1] * 3.6)),
        stops_per_km=stops / distance_km,
        # Aggressiveness proxy: mean positive acceleration. DERIVED, not a labelled
        # driving style -- it stands in for driver intent, which VED does not record.
        accel_pos_mean=float(np.nanmean(np.clip(np.diff(speed_ms) / np.maximum(dt, 0.1), 0, None))),
        temp_c=float(np.nanmedian(d["OAT[DegC]"])),           # MEASURED onboard
        aux_power_w=float(np.nanmean(hvac)),                   # MEASURED HVAC load
        hvac_on=float(np.nanmean(hvac > 10.0)),
        soc_start=float(soc[0]),
        soc_end=float(soc[-1]),
        soc_drop_pp=float(soc[0] - soc[-1]),
        lat_start=float(lat[0]), lon_start=float(lon[0]),
        lat_end=float(lat[-1]), lon_end=float(lon[-1]),
    )


# Column order of a segment row, also used to give an empty result a valid schema.
_SEGMENT_COLUMNS = (
    "vehicle_id", "trip", "daynum", "n_rows", "duration_s", "distance_km",
    "gps_distance_km", "energy_wh", "gross_discharge_wh", "regen_wh",
    "regen_fraction", "wh_per_km", "mean_speed_kmh", "speed_std_kmh",
    "stops_per_km", "accel_pos_mean", "temp_c", "aux_power_w", "hvac_on",
    "soc_start", "soc_end", "soc_drop_pp",
    "lat_start", "lon_start", "lat_end", "lon_end",
    # present only when eVED enrichment is available
    "elev_gain_m", "elev_loss_m", "elev_span_m", "net_elev_m",
    "mean_grade_pct", "abs_grade_pct", "speed_limit_kmh", "speed_deficit_kmh",
    "phys_wh_km", "phys_roll_wh", "phys_aero_wh", "phys_grade_wh",
    "phys_inertia_wh", "phys_aux_wh",
)


def derive_route_ids(segments: pd.DataFrame, eps_m: float = 500.0) -> pd.Series:
    """Derive a route grouping key, because VED has none.

    A route is taken to be a recurring (origin, destination) pair: start points and
    end points are clustered separately with DBSCAN on the haversine metric, and the
    route is the pair of cluster labels. A->B and B->A stay distinct, which is right
    because they have opposite grade profiles.

    Choosing `eps_m` is a real experimental decision, and the two failure modes are
    NOT symmetric:

      * TOO TIGHT -- repeats of one physical journey land in different clusters, so
        the same route can appear in both train and test. This is the LEAKAGE
        direction and the one to avoid.
      * TOO LOOSE -- distinct journeys merge, groups grow and splits become
        conservative rather than leaky. At 1500 m the whole of Ann Arbor collapses
        into 3 routes, which is safe but useless for grouped CV.

    Measured on the 504 BEV trips:

        eps (m)   routes   singletons   trips in routes seen >=3x
             50      373          308                         118
            300      220          153                         301
            500      136           76                         376
           1500        3            0                         502

    The default is 500 m. Because the asymmetry above puts the risk on the tight
    side, we prefer the larger value that still leaves enough groups for 5-fold
    grouped CV. It is also the physically honest choice here: de-identification
    truncates each record at a major intersection, so repeats of one journey do not
    begin at the same point and a tolerance of a few hundred metres is required to
    recognise them as the same route.

    Experiments must report a sensitivity check across this parameter rather than
    treat any one value as correct.
    """
    from sklearn.cluster import DBSCAN

    if len(segments) == 0:
        return pd.Series([], dtype=object, index=segments.index, name="route_id")

    def labels(lat, lon):
        x = np.radians(np.c_[np.asarray(lat, float), np.asarray(lon, float)])
        return DBSCAN(eps=eps_m / EARTH_RADIUS_M, min_samples=1,
                      metric="haversine").fit_predict(x)

    start = labels(segments["lat_start"], segments["lon_start"])
    end = labels(segments["lat_end"], segments["lon_end"])
    return pd.Series([f"R{a:04d}_{b:04d}" for a, b in zip(start, end)],
                     index=segments.index, name="route_id")


def build_segment_table(raw: pd.DataFrame | None = None, *,
                        vehicle: VehicleSpec | None = None,
                        eps_m: float = 500.0,
                        vehicle_ids: tuple[int, ...] = BEV_VEHICLE_IDS,
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the analysis-segment table and its attrition log.

    Returns (segments, attrition). Every exclusion is counted with a reason, because
    silently dropping awkward records is selection bias.
    """
    from ..config import NISSAN_LEAF_2013

    # VED's three BEVs are all 2013 Leafs, so that is the default -- but it is passed
    # explicitly, never taken from a module-level global (Phase 0, item C1).
    vehicle = NISSAN_LEAF_2013 if vehicle is None else vehicle
    raw = load_raw(vehicle_ids) if raw is None else raw
    groups = list(raw.groupby(["VehId", "Trip"], sort=True))
    log = [attrition_row("raw_trips", len(groups), len(groups), "all recorded trips")]

    rows = [r for _, d in groups if (r := _segment_row(d, vehicle)) is not None]
    n_after_quality = len(rows)
    log.append(attrition_row("energy_computable", len(groups), n_after_quality,
                             f">={MIN_ROWS} rows, >={MIN_DURATION_S:.0f}s, "
                             f">={MIN_DISTANCE_KM}km"))
    # Keep the schema even when nothing survives, so callers (and the remaining
    # filters below) can rely on the columns existing. An empty table is a valid
    # answer; a crash here would hide a total-rejection case behind a KeyError.
    seg = pd.DataFrame(rows, columns=None if rows else list(_SEGMENT_COLUMNS))

    before = len(seg)
    seg = seg[seg["energy_wh"] > 0].reset_index(drop=True)
    log.append(attrition_row("net_discharge", before, len(seg),
                             "net charging segments (regen-dominated fragments)"))

    before = len(seg)
    lo, hi = PLAUSIBLE_WH_PER_KM
    seg = seg[seg["wh_per_km"].between(lo, hi)].reset_index(drop=True)
    log.append(attrition_row("plausible_consumption", before, len(seg),
                             f"wh_per_km outside {lo}-{hi} (sensor or integration failure)"))

    seg["route_id"] = derive_route_ids(seg, eps_m=eps_m)
    seg["timestamp_utc"] = daynum_to_utc(seg["daynum"])
    seg["local_hour"] = (seg["timestamp_utc"].dt.tz_localize("UTC")
                         .dt.tz_convert(LOCAL_TZ).dt.hour)
    seg["trip_id"] = (seg["vehicle_id"].astype(str) + "_" + seg["trip"].astype(str))

    seg = tag(seg, Provenance.REAL, "VED")
    return seg, pd.DataFrame(log)
