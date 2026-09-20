"""Central configuration.

Vehicle parameters live ONLY here (CLAUDE.md rule 7).

Design note (Phase 0, item C1): there is deliberately **no module-level
`DEFAULT_VEHICLE`**. Every function that needs a vehicle takes it as a required
argument. The reason is safety, not style: while a default existed, a call site
that forgot to pass `vehicle` would silently model a generic e-scooter. Once the
Nissan Leaf became the primary research vehicle that failure mode would have
produced wrong physics with no error and no warning.

`VehicleSpec` has no field defaults for the same reason -- `VehicleSpec()` must
not quietly construct some particular vehicle.

PARAMETER PROVENANCE: every number below is currently an ASSUMPTION (tag A),
taken from datasheets and published literature, NOT fitted to our data.
Experiment E7 replaces `crr` and `drivetrain_eff` (and `cd`/`frontal_area_m2`
if they prove separately identifiable) with values fitted from real segments,
each carrying a confidence interval. Until then, no result may be presented as
resting on measured vehicle parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

DEFAULT_SEED = 42


@dataclass(frozen=True)
class VehicleSpec:
    """Road-load and pack parameters for one vehicle.

    No field has a default: constructing a vehicle is always an explicit act.
    """

    name: str
    curb_mass_kg: float          # vehicle + battery, without occupants/payload
    crr: float                   # rolling resistance coefficient
    cd: float                    # aerodynamic drag coefficient
    frontal_area_m2: float
    drivetrain_eff: float        # battery -> wheel efficiency
    regen_eff: float             # fraction of braking/downhill energy recovered
    aux_power_w: float           # always-on auxiliary load
    aux_extra_w: float           # additional load when "high auxiliary load" is on
    max_speed_ms: float
    pack_voltage_v: float
    pack_capacity_ah: float
    rated_range_km: float        # manufacturer's / regulator's claimed range
    param_source: str            # WHERE these numbers came from -- keep honest

    @property
    def nominal_energy_wh(self) -> float:
        return self.pack_voltage_v * self.pack_capacity_ah

    @property
    def rated_wh_per_km(self) -> float:
        return self.nominal_energy_wh / self.rated_range_km


# --------------------------------------------------------------------------- vehicles
# PRIMARY RESEARCH VEHICLE. The three battery-electric vehicles in VED are all
# 2013 Nissan Leafs (24 kWh), so this is the vehicle every real-data experiment
# (E0-E4, E7-E10) is run on.
NISSAN_LEAF_2013 = VehicleSpec(
    name="2013 Nissan Leaf (24 kWh) - VED primary vehicle",
    curb_mass_kg=1493.0,     # datasheet kerb mass; occupants/payload NOT observed in VED
    crr=0.009,               # low-rolling-resistance tyres, literature range 0.008-0.010
    cd=0.29,                 # published drag coefficient
    frontal_area_m2=2.28,    # published frontal area
    drivetrain_eff=0.89,     # battery -> wheel, literature range 0.88-0.92
    regen_eff=0.55,          # ASSUMPTION: constant. Real regen depends on decel rate,
                             # SoC and temperature -- a known simplification the ML
                             # residual layer is expected to absorb (see RQ1).
    aux_power_w=250.0,       # DC-DC + 12 V loads. NOTE: VED logs A/C and heater power
                             # directly, so HVAC is MEASURED, not taken from here.
    aux_extra_w=1500.0,      # indicative HVAC draw, used only for the app's what-if
    max_speed_ms=41.7,       # ~150 km/h
    pack_voltage_v=360.0,    # nominal pack voltage
    pack_capacity_ah=66.7,   # 360 V x 66.7 Ah ~= 24.0 kWh nominal
    rated_range_km=121.0,    # EPA 2013 rating (75 mi). NEDC claimed 199 km -- we use
                             # the stricter figure so the `rated_range` baseline is fair.
    param_source="Datasheet / published literature. ASSUMPTION (tag A) pending E7 fit.",
)

# Demonstration vehicle only. These numbers were never measured or fitted; they
# are the original scaffold's placeholders. The app may demo this vehicle, but
# NO experimental result is reported for it.
E_SCOOTER_PLACEHOLDER = VehicleSpec(
    name="Generic e-scooter (PLACEHOLDER - demo only, never used for results)",
    curb_mass_kg=100.0,
    crr=0.012,
    cd=0.90,
    frontal_area_m2=0.65,
    drivetrain_eff=0.80,
    regen_eff=0.40,
    aux_power_w=30.0,
    aux_extra_w=60.0,
    max_speed_ms=15.0,
    pack_voltage_v=60.0,
    pack_capacity_ah=30.0,
    rated_range_km=90.0,
    param_source="PLACEHOLDER - invented for the synthetic scaffold. Not measured.",
)
