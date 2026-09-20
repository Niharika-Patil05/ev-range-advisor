"""Feature groups for the real VED analysis segments.

The synthetic scaffold's groups (src/features/trip_features.py) describe an
idealised segment table. Real logged data has different, and mostly better,
columns: temperature and auxiliary load are measured rather than assumed, and
there is a traffic proxy the synthetic generator never had.

Groups exist so the ablation (E3) can drop one family at a time, and so every
feature's provenance tag is visible at the point of use:

    M = measured by a sensor      D = derived from measurements
    X = external dataset          A = assumption
"""
from __future__ import annotations

TARGET = "wh_per_km"

FEATURE_GROUPS: dict[str, list[str]] = {
    # D, from the 1 Hz speed trace and eVED elevation
    "route": [
        "distance_km", "mean_speed_kmh", "speed_std_kmh", "stops_per_km",
        "elev_gain_m", "elev_loss_m", "net_elev_m", "abs_grade_pct",
    ],
    # M, onboard outside-air temperature
    "weather": ["temp_c"],
    # M/D, vehicle state: measured HVAC load, measured SoC, derived aggressiveness
    "state": ["aux_power_w", "hvac_on", "soc_start", "accel_pos_mean"],
    # D from X, the congestion proxy -- eVED speed limits vs observed speed.
    # This is how the project addresses the synopsis's traffic objective without a
    # paid feed. It is a PROXY, and E3 measures its contribution rather than
    # assuming it.
    "traffic": ["speed_limit_kmh", "speed_deficit_kmh"],
    # D from M+X+A, the road-load baseline the hybrid corrects
    "physics": ["phys_wh_km"],
}

ML_ONLY_FEATURES = [f for g in ("route", "weather", "state", "traffic")
                    for f in FEATURE_GROUPS[g]]
ALL_FEATURES = ML_ONLY_FEATURES + FEATURE_GROUPS["physics"]


def features_without(*groups: str) -> list[str]:
    """Feature list with whole groups removed, for the ablation."""
    drop = set(groups)
    return [f for g, feats in FEATURE_GROUPS.items()
            if g not in drop and g != "physics" for f in feats]
