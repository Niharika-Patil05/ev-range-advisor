# H4c — Are wind and rain useful features?

**Status:** the last untested pre-registered hypothesis. Now tested. **SUPPORTED.**
**Data:** 480 real BEV segments joined to the Open-Meteo archive on (grid cell, UTC hour).
Provenance `REAL` (reanalysis model output).

## Why this was outstanding

Gate G1 validated the weather join months of work ago, but wind and precipitation were
never wired into the feature table — the `weather` group was onboard temperature alone.
H4c predicted headwind would be weak, and the scorecard carried it as **NOT TESTED**
rather than quietly assuming the answer.

## Result — adding wind and rain makes the models slightly *worse*

| model | without wind/rain | with wind/rain | Δ |
|---|---|---|---|
| hybrid_physics_ml | 13.49 ± 0.79 % | 13.61 ± 0.83 % | **+0.120 ± 0.137** |
| gbm | 18.06 ± 1.42 % | 18.17 ± 1.49 % | **+0.108 ± 0.164** |

Both deltas are within noise, but neither is an improvement — three extra features
bought nothing.

Single-feature drops from the extended set (gbm, 3 seeds):

| dropped | ΔMAPE |
|---|---|
| **temp_c** (onboard) | **+1.507 ± 0.542** |
| headwind_ms | +0.050 ± 0.175 |
| rain | −0.054 ± 0.033 |
| wind_speed_ms | −0.084 ± 0.128 |

**H4c — "headwind will be a weak feature." SUPPORTED.** It contributes
+0.05 ± 0.18 pp — indistinguishable from zero, and thirty times smaller than the
onboard temperature it sits beside.

## Why, and the wider point

Limitation L6 predicted this: reanalysis is a **grid-cell area average**, ~9 km for
ECMWF IFS. Ann Arbor fits inside roughly one cell, so every segment in a given hour
receives *identical* wind. The quantity that actually matters — air movement relative
to a vehicle on a specific street, in traffic, among buildings — is not resolved at
that scale, and no amount of joining will recover it.

**The sanity check that makes the point sharper.** Reanalysis temperature agrees with
VED's onboard sensor at only **r = 0.882, RMSE 5.60 °C**. Had temperature been taken
from reanalysis instead of the vehicle, the project's **second-strongest feature**
would have carried a 5.6 °C error.

> So the one weather variable that matters is available onboard at far higher
> fidelity, and the ones available only from reanalysis carry no measurable signal.
> **Instrumenting the vehicle beat joining an external dataset** — the same conclusion
> the auxiliary-power finding reached from the opposite direction.

## Consequence

The features are implemented, tested and available (`src/data/weather_archive.py`),
but are **not** part of the deployed feature set. They are kept because the negative
result is worth having and because a fleet in more varied terrain, or a vehicle
without an onboard temperature sensor, might find them useful.

## Limitations

- One city, one grid cell — wind may well matter where terrain or exposure varies.
- "Weather as it turned out", not the forecast available at departure.
- Rain is a binary from a precipitation threshold (0.1 mm/h); road *surface* wetness,
  which is what actually changes rolling resistance, is not observed at all.
