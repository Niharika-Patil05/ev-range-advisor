# Pre-Registered Hypotheses

**Written: 20 September 2026 — BEFORE any real data was downloaded, any loader was written, or any experiment was run.**

**Purpose.** This file exists to prevent result-shopping. A study that evaluates models across five split levels, four interval methods and multiple feature groups has many degrees of freedom; without a record of what we predicted *beforehand*, a favourable subset could be selected after the fact and presented as the finding. That is the single largest integrity risk in this project (risk K9).

**Binding rules for the team:**
1. This file is committed **before** E2, E4 and E6 are run. It is never edited to match results.
2. **Every level, every model and every method is reported** — including unflattering ones.
3. If a prediction is wrong, **the report says it was wrong.** A refuted pre-registered hypothesis is a *result*, not a failure.
4. Analyses not listed here are labelled **exploratory** in the report, and are not presented as confirmatory.

---

## H1 — Physics prior and distribution shift (RQ1)

**H1.a** Under in-distribution route-wise splitting (L1), the hybrid will beat pure-ML models, **but by a modest margin** — we predict the gap to be far smaller than the synthetic 2.93 % vs 10.4 % and possibly within noise.

**H1.b (primary prediction)** The **degradation ratio** `D = MAPE(L_k)/MAPE(L1)` will be **smaller for the hybrid than for pure-ML models** at the temperature-regime shift (L2) and the cross-vehicle-model shift (L4).
*Reasoning:* gradient-boosted trees cannot extrapolate beyond the training envelope; the road-load term remains physically valid outside it.

**H1.c** `random_forest` and `gbm` will degrade **most sharply at L2** (unseen cold), because temperature enters consumption through several channels that trees can only interpolate.

**H1.d** The **L0 random split will materially overstate accuracy** relative to L1 — we predict at least a 30 % relative reduction in reported MAPE.

**H1.e** `physics_only` with **identified** parameters (E7) will beat `physics_only` with nominal parameters, and may beat `linear`.

**What would refute H1:** the hybrid showing an equal or larger degradation ratio than pure ML at L2/L4. **This is a publishable negative result and will be reported as such.**

---

## H2 — Conditional calibration (RQ2)

**H2.a** Split conformal will achieve **marginal** coverage close to the 90 % target on L1.

**H2.b (primary prediction)** **Conditional coverage will fail in the coldest temperature subgroup**, falling below 85 % while marginal coverage stays near 90 %.
*Reasoning:* cold segments are a minority of the calibration set, and their residuals are larger, so a single global quantile under-covers them.

**H2.c** Naive quantile regression will continue to **under-cover** (the current synthetic result is 0.73 against a 0.90 target); **CQR will restore approximately correct marginal coverage** at the cost of wider intervals.

**H2.d** **Mondrian conformal** (groups = temperature bin × route type) will improve **worst-subgroup** coverage relative to split conformal, at the cost of a wider mean interval.

**H2.e** Under the L2 shift, **all** methods will lose coverage, because conformal guarantees assume exchangeability, which shift violates by construction.

**What would refute H2:** uniform conditional coverage across all subgroups under split conformal — which would make Mondrian unnecessary and would itself be a clean, reportable finding.

---

## H3 — SoH→range propagation (RQ3) · `COUPLED-SIM`

**H3.a** Real leave-one-battery-out SoH error on NASA cells will be **substantially worse than the synthetic 1.11 pp — we predict 2–5 percentage points.**

**H3.b** NASA→Oxford transfer (E5b) will be **worse still**, because chemistry, format and duty cycle all differ.

**H3.c (primary prediction)** At long segment lengths and low starting SoC, **SoH-estimation error will contribute a larger share of total range-error variance than consumption-model error.**
*Reasoning:* SoH scales the entire usable-energy term, so its relative error passes through to range roughly one-for-one, whereas consumption error partially averages out over a longer segment.

**H3.d** The relationship between SoH error and range error will be **approximately linear** in the 80–100 % SoH band, giving a stable "km per percentage point" figure.

**What would refute H3:** consumption error dominating at all segment lengths — which would be an equally useful engineering conclusion (it would say the battery model matters less than expected for range prediction).

---

## H4 — Features and data quality (exploratory, stated for completeness)

**H4.a** Measured onboard **auxiliary power (A/C + heater)** will be a stronger predictor than ambient temperature alone, because it captures HVAC demand directly rather than by proxy.

**H4.b** The **traffic proxy** (`speed_deficit`) will carry information beyond mean speed and stop rate.

**H4.c** `headwind_ms`, derived from a 9 km reanalysis grid, will be a **weak** feature — possibly indistinguishable from noise at segment level.

**H4.d** **Grade features will be weakly informative in this dataset**, because Ann Arbor geo-fencing gives low grade variance (limitation L3). This is a property of the *data*, not of the physics.

---

## Declared analysis decisions (fixed in advance)

| Decision | Value |
|---|---|
| Primary metric | MAPE on `wh_per_km`; secondary `range_km_error` |
| Seeds | 5 per experiment; mean ± std always reported |
| Primary split for headline numbers | **L1 (route-wise)** |
| Test set | Touched **once**, at the end. Hyper-parameters tuned by nested CV on the dev set only |
| Significance | Paired per-route errors + Wilcoxon signed-rank; **n reported beside every claim** |
| Interval target | 90 % (α = 0.10) |
| Cold subgroup definition | Lowest tertile of segment-mean onboard temperature, fixed before E4 |
| Exclusions | Every exclusion logged with reason and count in the attrition log |

---

## Signatures

Pre-registered by the project team before data collection.

| Name | Role | Date |
|---|---|---|
| Sharyu Sachin Jadhav | SoH, coupling | 20-09-2026 |
| Megha Satish Kamble | Range models, experiments | 20-09-2026 |
| Niharika Mahadev Patil | Uncertainty, advisor, app | 20-09-2026 |
| Viraj Sarjerao Mohite | Route, elevation, weather | 20-09-2026 |
| Atharva Yogesh Varude | Vehicle parameters, physics, validation | 20-09-2026 |

*Reviewed by: Dr. N. Vengadachalam (guide) — __________*
