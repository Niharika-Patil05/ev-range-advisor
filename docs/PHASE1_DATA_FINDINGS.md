# Phase 1 — Findings from the real VED data

**Date:** 20 September 2026 · **Status:** data acquired and characterised; Gate G1 **PASSED**.
**Supersedes, where they differ:** [PHASE0_VERIFICATION.md](PHASE0_VERIFICATION.md) (which was based on the papers).
**Data:** VED downloaded (176 MB compressed → 3.0 GB, 54 weekly CSVs), Apache-2.0.

Phase 0 verified the datasets *from their publications*. This document records what the
**actual bytes** say. Several published claims did not survive contact with the data.

---

## 1. Headline numbers

| Quantity | Value |
|---|---|
| Battery-electric vehicles | **3** — `VehId` **10, 455, 541** (confirmed from the static file) |
| **Usable** BEVs | **2** — vehicle 541 has only **10 trips** (5 with a usable SoC drop) |
| BEV trips | **504** total, **500** usable (≥30 rows, ≥60 s) |
| BEV rows | 476,308 |
| Period | 2017-11-01 → 2018-11-09 (a full year) |
| Median trip distance | **3.6 km** (10th pct 1.1, 90th pct 10.5, max 32.2) |
| Median consumption | **146 Wh/km** (mean 161, IQR 116–186) |
| Onboard temperature range | **−14.5 °C to +35.5 °C**, median 11.5 |
| Trips with HVAC on | **72 %**, median **650 W** when on |

---

## 2. ✅ Confirmed from data

### 2.1 Consumption is physically plausible — the pipeline is sound
Median **146 Wh/km**; **83.7 %** of trips fall in 100–350 Wh/km, the credible band for a
2013 Leaf (EPA ≈ 186 Wh/km). Obtained independently of any model, by integrating
measured pack power. **This is the project's first real evidence of anything.**

### 2.2 Temperature and HVAC are MEASURED (Phase 0 findings F3 and F4 confirmed)
For the three BEVs, `OAT[DegC]`, `Air Conditioning Power[Watts]` and
`Heater Power[Watts]` are all **100 % available**. So:
- the temperature feature is a **measurement**, not a 9 km reanalysis estimate;
- the previously invented `aux_on` binary becomes a **measured auxiliary load**.

HVAC is on in **72 %** of trips at a median **650 W**. At the observed average speeds
that is of order **20 % of total consumption** — a first-order effect, now observed
rather than assumed. (`Air Conditioning Power[kW]` is 0 % available; use the Watts channel.)

### 2.3 Temperature coverage supports the L2 experiment
−14.5 °C to +35.5 °C, 5th percentile −5.5 °C. A genuine Michigan winter is present, so
the train-warm/test-cold shift experiment (RQ1, level L2) is viable on real data.

### 2.4 PHEV fleet supports the L4 substitute
24 PHEVs across **four engine displacements** (1.4 / 1.5 / 1.8 / 2.0 L), consistent with
the "4 models" in the paper. The BEV → PHEV-electric-mode transfer (Phase 0 §A6) is
viable. `Engine RPM` is **0 %** for BEVs and available for PHEVs, as expected, so it can
serve as the electric-mode filter.

---

## 3. 🔴 Corrections to Phase 0

### 3.1 CURRENT SIGN: negative = discharge — **resolved, and counter-intuitive**
Phase 0 said "unverified — do not assume". Now determined, and it is the opposite of the
naive reading.

Across **406 trips** with a SoC drop > 1 pp, the signed integral `Σ V·I·Δt` is
**negative in 100.0 % of cases**. Energy leaving the battery therefore appears as
negative power.

> **Discharge energy = −∫ V·I dt**

Cross-validated on a single trip: SoC fell 2.7 pp (≈ 648 Wh of 24 kWh nominal) while
the integral gave 705 Wh — **agreement within 9 %**, which validates the sign convention
and the integration method together.

### 3.2 GPS is NOT null-padded — the fogging concern was overstated
Phase 0 (F2) predicted GPS missing at trip ends, forcing an "analysis segment" redesign.
In the data, GPS is **100 % non-null**, and GPS path length agrees with
speed-integrated distance (**ratio 0.91–1.05** across sampled trips).

**Interpretation:** de-identification removed whole *rows* rather than blanking columns,
so energy, distance and route features computed over the retained rows all refer to the
same interval and are mutually consistent. The redesign is therefore **not needed for
internal consistency**.

⚠️ **But the caveat survives in a different form:** the retained record may be a
*fragment* of the physical journey. Median trip distance is only **3.6 km**. We are
modelling *recorded segments*, not complete door-to-door journeys, and must say so.
The term "analysis segment" is kept for this reason.

### 3.3 Sampling is IRREGULAR, not the published fixed rates
The paper lists current at 1 s, voltage 5 s, SoC 60 s. Observed instead:
- inter-row Δt: **median 0.9 s, min 0.1 s, max 13.4 s** — irregular;
- values are **held (zero-order hold), not interpolated or NaN**;
- on a sample trip, current changed every ~5.9 rows, voltage ~7.4, SoC ~78, GPS ~7.4.

> **Consequence:** energy must be integrated with **actual Δt**, never by assuming 1 Hz.
> Because voltage arrives already held, zero-order hold is what the data provides; the
> planned ZOH-vs-linear-interpolation error budget still applies but starts from ZOH.

### 3.4 Only TWO usable vehicles, not three
`VehId` 541 contributes **10 trips**. Leave-one-vehicle-out (L3) therefore rests on
**n = 2**, not 3.

> **This is a downgrade and must be stated plainly.** L3 becomes indicative only. The
> weight of the vehicle-shift claim shifts to **L4** (BEV → PHEV-electric, up to 27
> vehicles), which is now the load-bearing generalization experiment.

### 3.5 VED has no `route_id`
The project's entire split discipline groups on `route_id`, which **does not exist** in
VED. Routes must be **derived**, by clustering trip start/end coordinates (a trip from
home to work recurs and should form one group).

> **New Phase 1 task.** Derived routes are a **D (derived)** quantity with their own
> error, and the clustering tolerance becomes an experimental parameter. If clustering
> is too loose, distinct routes merge and L1 leaks optimistically.

---

## 4. 🚦 Gate G1 — the weather join

The plan required the timezone ambiguity (`DayNum` never states local vs UTC) to be
resolved **empirically**, with acceptance at r ≥ 0.90 and a single dominant peak.

### 4.1 The planned test FAILED
Correlating onboard `OAT` against reanalysis temperature:

| Hypothesis | raw r | deseasonalised r |
|---|---|---|
| DayNum is UTC | 0.886 | **0.157** |
| DayNum is EST (UTC−5) | 0.808 | −0.310 |
| DayNum is EDT (UTC−4) | 0.825 | −0.240 |
| UTC+5 (physically impossible control) | 0.869 | 0.111 |

Raw correlation is high for *every* hypothesis because the seasonal cycle dominates and a
5-hour shift barely disturbs it. Removing the seasonal baseline — which isolates the
diurnal signal that actually discriminates — collapses the best correlation to **0.157**,
only **+0.046** above a physically impossible control.

**Diagnosis:** vehicle outside-air-temperature sensors are heavily damped, quantised
(0.5 °C steps, ~4 changes per trip) and heat-soaked by pavement and powertrain. They
track the season well and the hour badly. **The onboard sensor cannot arbitrate the
timezone.** Reported as a failure, not worked around by lowering the threshold.

### 4.2 A better clock: human driving behaviour — **PASSED decisively**
Driving has a far stronger daily rhythm than a damped thermistor. Trip start times on the
raw `DayNum` clock show a **dead zone from 05:00 to 10:00** and peaks at 12:00 and 21:00 —
impossible for local time.

| Hypothesis | trips 00:00–05:00 local | trips at commute hours | ratio |
|---|---|---|---|
| **DayNum is UTC** (local = DayNum − 5) | **1** | **277** | **277 : 1** ✓ |
| DayNum is local | 30 | 84 | 2.8 : 1 ✗ |
| DayNum = local + 4 (EDT) | 1 | 272 | 272 : 1 ✓ |

> **Conclusion: `DayNum` is UTC.** Converting to America/Detroit yields a textbook
> commuter distribution — almost nobody driving 00:00–05:00, clear morning and evening
> peaks. Treating it as local time implies 30 night-time trips and a 21:00 rush hour.
>
> The test cannot separate EST from EDT (a 1-hour difference), which is immaterial:
> the correct handling is DST-aware conversion from UTC, which we now apply.

**GATE G1: PASSED**, by a decisive margin, on the second test.

**Consequences:** the reanalysis join is sound, so wind and precipitation remain usable —
subject to the standing 9 km grid-resolution caveat (limitation L6), which makes
`headwind_ms` a low-confidence feature whose contribution E3 will measure rather than assume.
The pre-declared fallback (drop wind and rain) is **not needed**.

---

## 5. 🔬 An unplanned finding: effective pack energy is measurable

Regressing integrated discharge energy against SoC drop gives Wh per SoC-percent, i.e.
an **effective usable pack energy**:

| Vehicle | n trips | effective usable pack | R² | vs 24 kWh nominal |
|---|---|---|---|---|
| 10 | 128 | **14.5 kWh** | 0.72 | 60 % |
| 455 | 190 | **15.7 kWh** | 0.72 | 65 % |
| 541 | 5 | — | — | insufficient |

**Why this matters:** it is a *real, per-vehicle, measured* quantity bearing on pack
state — precisely what limitation L8 said no public dataset provides. It offers a partial,
honest anchor for the usable-energy term in RQ3.

**Why it must not be over-claimed.** The 60–65 % figure is **not** an SoH measurement.
At least four effects are confounded and this data cannot separate them:
1. usable capacity is below nominal by design (~21–22 kWh when new);
2. real degradation (these are 4–5-year-old Leafs);
3. VED's SoC may not be the usable-window SoC;
4. SoC is sampled at 60 s and quantised, biasing short trips.

> **Permitted claim:** *"the measured energy-per-SoC-percent implies an effective usable
> pack energy of 14.5–15.7 kWh for these two vehicles (R² = 0.72), against 24 kWh nominal;
> the split between design margin, degradation and SoC definition is not identifiable from
> this data."*
> **Forbidden claim:** *"these packs are at 60 % state of health."*

---

## 6. Impact on the plan

| Item | Change |
|---|---|
| Dataset size | **500 trips, 2 usable vehicles.** Modest but workable for trip-level modelling |
| **L3** (leave-one-vehicle-out) | **Downgraded to n = 2** — indicative only |
| **L4** (BEV → PHEV-electric) | **Promoted** to the load-bearing vehicle-shift experiment |
| **L2** (temperature regime) | **Confirmed viable** — a full year, −14.5 to +35.5 °C |
| `route_id` | **Must be derived** by clustering; new task, new error source |
| Energy integration | Use **actual Δt** and `E = −∫V·I dt` |
| `temp_c` | **Measured onboard** (Gate G1 outcome does not affect it) |
| `aux_power_w` | **Measured** — replaces the invented binary |
| Wind / rain | **Retained**, low-confidence, contribution measured in E3 |
| Trip semantics | "Analysis segment": a recorded fragment, median 3.6 km, not a full journey |

## 6b. eVED — the join that turned out not to be a join

**Verified on the BEV rows:** eVED is a **strict superset** of VED, not a companion
table. For one week: identical row counts (3,633 = 3,633), identical trip counts
(11 = 11), a **100.0 % exact match** on `(VehId, Trip, Timestamp)`, and
**bit-identical** `HV Battery Current`, `Voltage`, `Vehicle Speed` and `OAT`.

> **Consequence: there is no join to get wrong.** The loader reads eVED directly and
> falls back to plain VED only if eVED is absent. This removes an entire risk class
> (K3) rather than mitigating it. Also note the published clone URL carries a
> `Datarepo@` username; the repository is public and needs no credential, and the
> data is one 656 MB zip that is better fetched directly than cloned.

Enrichment availability on BEV rows: elevation **100 %**, gradient **98.5 %**,
speed limit **99.1 %** (stored as text; ~1 % are ranges such as `48-40` where the
posted limit changes along the way, parsed as the mean of the endpoints).

### 6b.1 🔴 Terrain: the instantaneous grade is dead, the cumulative climb is not

| Statistic | Value |
|---|---|
| rows with \|gradient\| < 0.5 % | **96.7 %** |
| rows with \|gradient\| > 2 % | **0.22 %** |
| gradient std | 0.0023 (0.23 %) |
| median per-trip elevation **gain** | **41 m** (mean 49, max 212) |
| mean per-trip **net** elevation change | **−1.2 m** |

Two conclusions, and they point in opposite directions:

- **`mean_grade_pct` is a dead feature here** (std 0.03 percentage points across
  segments). Limitation L3 is confirmed quantitatively: this data cannot excite the
  instantaneous-slope term, so nothing about steep-terrain performance is testable.
- **But terrain still costs real energy.** Trips return to roughly their starting
  altitude (net −1.2 m), yet a median trip climbs 41 m, and climbing costs `1/η`
  while descending returns only `η_regen`. For the median trip that asymmetry is
  ≈ 137 Wh spent against ≈ 70 Wh recovered — a **net ≈ 67 Wh, about 12 % of the
  568 Wh median trip energy**.

> So the loader exposes `elev_gain_m` and `elev_loss_m` rather than an average
> gradient. A project that had only computed mean grade would have concluded,
> wrongly, that terrain is irrelevant in this dataset.

### 6b.2 The traffic proxy is real
`speed_deficit_kmh` = posted limit − observed speed: mean **20.3 km/h**, std 9.6,
range −8 to 66. It has genuine spread, so the synopsis's traffic objective is
addressed by a reproducible derived feature rather than a paid API. It remains a
**proxy** (tag D), and E3 will measure its contribution rather than assume it.

## 7. Still to verify
- Per-vehicle availability of HVAC channels for 541 (only 10 trips).
- eVED download and the exact record-level join (Phase 0 verified the keys; the join itself is untested).
- Whether derived-route clustering yields enough distinct groups for L1.
- The ZOH-vs-interpolation energy error budget, quantified.
