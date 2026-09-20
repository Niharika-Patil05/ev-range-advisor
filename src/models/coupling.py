"""SoH <-> range coupling and the battery what-if simulator."""
from __future__ import annotations

import numpy as np

from ..config import VehicleSpec


def temp_derate(temp_c: float) -> float:
    """Usable-capacity factor vs temperature (cold packs deliver less). Rule-of-thumb placeholder:
    -0.6 % per degC below 20 degC. Calibrate on your own cell data."""
    return float(max(0.6, 1.0 - 0.006 * max(0.0, 20.0 - temp_c)))


def usable_energy_wh(soc_pct, soh, temp_c, vehicle: VehicleSpec):
    """Usable energy = nominal * SoH * temperature derating * SoC."""
    derate = np.vectorize(temp_derate)(temp_c)
    return vehicle.nominal_energy_wh * np.asarray(soh) * derate * np.asarray(soc_pct) / 100.0


def project_soh(
    soh0: float,
    months: int = 24,
    cycles_per_week: float = 5.0,
    charge_cap_pct: float = 100.0,
    avg_dod: float = 0.7,
    avg_temp_c: float = 30.0,
) -> np.ndarray:
    """Semi-empirical SoH projection (array of length months+1, month 0 = today).

    INDICATIVE ONLY. The constants below are placeholders chosen to give plausible trends
    (Arrhenius-like temperature doubling per 10 degC, higher fade at high state-of-charge
    and deep discharge). Calibrate them against your lab-cell datasets before quoting numbers.
    """
    base_fade_per_efc = 0.00025                     # SoH lost per equivalent full cycle at 25 degC
    temp_factor = 2.0 ** ((avg_temp_c - 25.0) / 10.0)
    dod_factor = (max(avg_dod, 0.05) / 0.7) ** 0.8
    cap_factor = max(0.7, 1.0 + 0.5 * (charge_cap_pct - 80.0) / 20.0)
    efc_per_month = cycles_per_week * 4.345 * avg_dod
    fade_per_month = base_fade_per_efc * temp_factor * dod_factor * cap_factor * efc_per_month
    out = soh0 - fade_per_month * np.arange(months + 1)
    return np.clip(out, 0.0, 1.0)
