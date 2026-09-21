# E4 — Cross-vehicle generalization, using PHEVs in electric mode

**Data:** 1,897 PHEV electric-mode segments, 23 vehicles, 415 routes + 480 BEV segments. Provenance `REAL`.
**Purpose:** separate the two explanations E2 could not.

## The confound E4 exists to resolve

E2 found that under vehicle shift, `physics_only` beat every learned model. But
holding out a BEV left only **139 training segments**, so *"learning fails under
vehicle shift"* was inseparable from *"learning fails with little data"*.

PHEVs running engine-off are physically battery-electric: same road load, same pack
energy from `V·I`. VED has **1,897 such segments across 23 vehicles** — four times
the BEV data — which allows leave-one-vehicle-out with ample training data.

## Result 1 — the confound is resolved: it was data scarcity, not vehicle shift

**L3c**, leave-one-vehicle-out within the PHEV fleet (12 vehicles with ≥30 segments,
1,764 segments, vehicle **and** route disjoint):

| model | with HVAC | without HVAC |
|---|---|---|
| **hybrid_physics_ml** | **15.99** | **16.70** |
| gbm | 20.19 | 21.71 |
| random_forest | 21.56 | 22.40 |
| ridge | 23.63 | 25.10 |
| physics_only | 26.73 | 26.73 |
| rated_range | 41.67 | 41.67 |

> **With adequate training data, the hybrid wins decisively under genuine vehicle
> shift — 15.99 % against 26.73 % for physics-only.** E2's contrary finding was an
> artefact of training on 139 segments, not evidence that learning fails across
> vehicles.

This changes E2's headline conclusion, and the E2 findings file is annotated
accordingly rather than rewritten. **The learned residual's value is conditional on
training-data volume, not on vehicle identity.**

## Result 2 — cross-powertrain transfer works, but only in one direction

**L4**, training on one fleet and testing on the other:

| model | BEV→PHEV (with / without HVAC) | PHEV→BEV (with / without HVAC) |
|---|---|---|
| hybrid_physics_ml | **20.83 / 19.62** | 26.94 / 29.10 |
| gbm | 22.78 / 23.08 | 34.90 / 31.19 |
| random_forest | 23.13 / 22.60 | 33.63 / 31.83 |
| ridge | 28.29 / 25.73 | 36.36 / 32.47 |
| physics_only | 27.23 | **22.54** |

- **BEV → PHEV transfer succeeds.** The hybrid reaches 19.6–20.8 %, beating
  physics-only at 27.2 %, on an entirely unseen powertrain.
- **PHEV → BEV transfer fails.** `physics_only` (22.54 %) beats every learned model,
  the best of which is the hybrid at 26.94 %. More training data did not help.

### Why the asymmetry — one hypothesis tested and refuted, one surviving

**Refuted: "engine-off segments are gentler driving."** It was a natural guess — a
PHEV starts its engine under load — but the data says the opposite. PHEV-electric
segments are **faster** (median 39.2 vs 32.4 km/h), **harder accelerating**
(0.30 vs 0.26 m/s²), longer (4.4 vs 3.8 km) and have **fewer stops**
(0.63 vs 1.00 per km). They are more highway-like, not gentler.

**Surviving explanation: the HVAC instrumentation gap.** Both HVAC channels are 100 %
available for BEVs; for PHEVs A/C is 72 % and the heater only **12 %**. Measured
auxiliary power therefore has a **median of 390 W on BEVs and 0 W on PHEVs**. A model
trained on PHEVs learns that auxiliary load is irrelevant, then meets BEV data where
it varies 0–3,800 W and correlates with consumption at r = +0.66 — a first-order
effect it was taught to ignore.

**The evidence is mixed and we do not overstate it.** Dropping the HVAC features
improves PHEV→BEV for `gbm` (34.90 → 31.19), `random_forest` (33.63 → 31.83) and
`ridge` (36.36 → 32.47), consistent with the story — but makes the **hybrid worse**
(26.94 → 29.10), which it does not explain. The asymmetry is therefore **partly
explained**. PHEV segments also skew warmer (median 15.5 vs 11.5 °C), so the PHEV
training set under-represents cold operation.

## Limitations

- **Vehicle parameters are the Leaf's**, applied to all PHEVs. Only mass is
  per-vehicle, and that comes from VED's **binned** `Generalized_Weight` (±5 %).
  Cd·A, crr and η were *not* re-identified per model, contrary to the original plan —
  E7 showed crr is not identifiable from this data, so per-model fitting would have
  produced unstable numbers.
- **`vehicle_model` is engine displacement**, a proxy: two different models could
  share a displacement.
- **The electric-mode filter is a selection**, rejecting 46.5 % of PHEV segments.
  Engine-off segments are not a random sample of PHEV driving.
- **PHEVs are not BEVs.** They carry an engine and its mass, and a smaller pack. L3c
  is a genuine vehicle-shift result *for PHEVs in charge-depleting mode*, not a BEV
  result.
- Association only. One city, one year.

## Two bugs this experiment exposed

1. **HVAC silently voided by NaN propagation.** `heater + ac` yields NaN whenever
   either channel is missing, so `aux_power_w` was entirely NaN for PHEVs. Invisible
   on BEVs, where both channels are 100 % available. Now each channel is treated as
   zero when absent, with `hvac_coverage` recording how much was actually measured —
   so "measured zero" and "not measured" stay distinguishable.
2. **The electric-mode filter silently rejected the entire fleet.** `load_raw` does
   not request `Engine RPM` by default, and the mask returned `False` for a missing
   column, producing an attrition line reading 100 % as though the data were simply
   unusable. It now raises. Both failures are covered by tests.
