"""SoH <-> range coupling and the battery what-if simulator."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import VehicleSpec


def temp_derate(temp_c: float) -> float:
    """Usable-capacity factor vs temperature (cold packs deliver less). Rule-of-thumb placeholder:
    -0.6 % per degC below 20 degC. Calibrate on your own cell data."""
    return float(max(0.6, 1.0 - 0.006 * max(0.0, 20.0 - temp_c)))


def usable_energy_wh(soc_pct, soh, temp_c, vehicle: VehicleSpec):
    """Usable energy = nominal * SoH * temperature derating * SoC."""
    derate = np.vectorize(temp_derate)(temp_c)
    return vehicle.nominal_energy_wh * np.asarray(soh) * derate * np.asarray(soc_pct) / 100.0


# Measured on 26 NASA PCoE cells (experiment E8), Theil-Sen slope of SoH against
# cycle count. The shipped placeholder was 0.00025 per cycle -- NINE TIMES smaller
# than reality, so the old projection understated degradation badly.
#
# The cell-to-cell spread is 80 % of the mean, which is why this is reported as a
# BAND rather than a line. A single curve would imply a precision the data does not
# have.
FADE_PER_CYCLE = 0.00222          # mean across cells
FADE_PER_CYCLE_P10 = 0.00057      # optimistic cell
FADE_PER_CYCLE_P90 = 0.00346      # pessimistic cell
WEEKS_PER_MONTH = 4.345


def project_soh(soh0: float, n_cycles: int = 200, step: int = 5) -> pd.DataFrame:
    """Project SoH against CYCLE COUNT as a band, from measured lab-cell fade rates.

    Returns `cycle`, `central`, `optimistic`, `pessimistic`.

    WHY THE X AXIS IS CYCLES AND NOT MONTHS
    ---------------------------------------
    The earlier version took months and cycles-per-week and produced a calendar
    forecast. Once the fade rate was actually measured (E8: 0.00222 per cycle) that
    became untenable. At five cycles a week, two years is 521 cycles, and
    521 x 0.00222 is more than a whole battery: the projection predicts a DEAD PACK
    in two years, which every real EV contradicts.

    The rate is not wrong; the extrapolation was. These NASA cells are cycled at full
    depth of discharge and high rate, reaching 80 % SoH in roughly 90 cycles. A
    vehicle pack doing shallow partial cycles typically takes on the order of a
    thousand. Converting lab cycles into vehicle months would need an equivalence
    factor of order ten that NOTHING in this project measures.

    So the function reports what the data supports -- fade per lab cycle, with the
    cell-to-cell band -- and refuses to convert it into calendar time.

    WHAT IT DELIBERATELY NO LONGER DOES
    -----------------------------------
    Earlier versions multiplied the fade rate by an Arrhenius temperature factor, a
    depth-of-discharge exponent and a charge-cap factor, and the app turned the last
    into user-facing advice ("charging to 80 % preserves X percentage points"). All
    three were invented, and E8 found the NASA cells cannot support any of them:
    temperature is not identifiable (p = 0.087, and the fitted sign is the wrong way
    round for an Arrhenius law), and no DoD or charge-limit variable is recorded at
    all. A projection that cannot distinguish a good charging habit from a bad one
    should not pretend to.
    """
    cycles = np.arange(0, int(n_cycles) + 1, max(1, int(step)))
    out = pd.DataFrame({"cycle": cycles})
    for name, rate in (("central", FADE_PER_CYCLE),
                       ("optimistic", FADE_PER_CYCLE_P10),
                       ("pessimistic", FADE_PER_CYCLE_P90)):
        out[name] = np.clip(soh0 - rate * cycles, 0.0, 1.0)
    return out


def cycles_to_threshold(soh0: float, threshold: float = 0.80) -> dict:
    """Lab cycles until SoH reaches `threshold`, per cell-to-cell band."""
    def n(rate):
        return float("inf") if soh0 <= threshold else (soh0 - threshold) / rate
    return {"optimistic": n(FADE_PER_CYCLE_P10), "central": n(FADE_PER_CYCLE),
            "pessimistic": n(FADE_PER_CYCLE_P90)}
