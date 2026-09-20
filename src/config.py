"""Central configuration.

IMPORTANT: every number in VehicleSpec is a PLACEHOLDER for a generic e-scooter.
Replace them with your real vehicle's datasheet / measured values (Phase 0 of the plan).
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
    name: str = "Generic e-scooter (PLACEHOLDER - replace with your vehicle)"
    curb_mass_kg: float = 100.0      # vehicle + battery, without rider/load
    crr: float = 0.012               # rolling resistance coefficient (calibrate on real data)
    cd: float = 0.90                 # drag coefficient (rider included)
    frontal_area_m2: float = 0.65
    drivetrain_eff: float = 0.80     # battery -> wheel efficiency
    regen_eff: float = 0.40          # fraction of braking/downhill energy recovered (0 = no regen)
    aux_power_w: float = 30.0        # always-on auxiliary load (controller, lights, display)
    aux_extra_w: float = 60.0        # extra load when "high auxiliary load" is switched on
    max_speed_ms: float = 15.0       # ~54 km/h
    pack_voltage_v: float = 60.0
    pack_capacity_ah: float = 30.0
    rated_range_km: float = 90.0     # manufacturer's claimed range

    @property
    def nominal_energy_wh(self) -> float:
        return self.pack_voltage_v * self.pack_capacity_ah

    @property
    def rated_wh_per_km(self) -> float:
        return self.nominal_energy_wh / self.rated_range_km


DEFAULT_VEHICLE = VehicleSpec()
