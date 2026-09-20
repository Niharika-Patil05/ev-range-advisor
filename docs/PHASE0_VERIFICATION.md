# Phase 0 — Verification Audit
## Datasets, joins, and citations, checked before any code is written

**Date:** 20 September 2026 · **Status:** Phase 0 audit complete. **No source code modified.**
**Supersedes assumptions in:** [01_RESEARCH_PLAN.md](01_RESEARCH_PLAN.md) (v1, exploratory)
**Authoritative plan:** [RESEARCH_PLAN_V2.md](RESEARCH_PLAN_V2.md)

Everything below was checked against primary sources in this session. Where a claim could not be verified, it says so.

---

# PART A — DATASET AND JOIN VERIFICATION

## A1. The headline result of this audit

The proposed **VED + eVED + Open-Meteo** join **is scientifically defensible — but not in the form assumed in plan v1.** Five findings materially change the design. Three of them make the project *better*; two impose real constraints.

| # | Finding | Effect |
|---|---|---|
| **F1** | VED signals are **multi-rate**, not 1 Hz: Current 1 s, **Voltage 5 s**, **SOC 60 s**, GPS 3 s, Speed 1 s, HVAC 60 s | Energy integration needs explicit resampling + an error budget. **Constraint.** |
| **F2** | VED GPS is **privacy-fogged at trip start and end** | Cannot divide whole-trip energy by GPS distance. Forces a redefinition of the unit of analysis. **Major constraint.** |
| **F3** | VED carries **onboard Outside Air Temperature at 100 % availability for EVs** | Temperature becomes a **measured** variable, not a reanalysis estimate. **Major improvement** — and it makes the weather join *falsifiable*. |
| **F4** | VED carries **A/C Power and Heater Power** channels | The invented `aux_on` binary becomes a **measured auxiliary load**. **Major improvement** — HVAC is the dominant cold-weather range factor in a real EV. |
| **F5** | eVED **preserves `VehId, Trip, DayNum, Timestamp`** | The VED↔eVED join is **exact and record-level**, not fuzzy. **Join verified.** |

---

## A2. VED — verified specification

**Citation (verified):** G. Oh, D. J. LeBlanc, H. Peng, "Vehicle Energy Dataset (VED), A Large-Scale Dataset for Vehicle Energy Consumption Research," *IEEE Transactions on Intelligent Transportation Systems*, vol. 23, no. 4, pp. 3302–3312, 2022. DOI [10.1109/TITS.2020.3035596](https://doi.org/10.1109/TITS.2020.3035596). Preprint [arXiv:1905.02081](https://arxiv.org/abs/1905.02081). Data: [github.com/gsoh/VED](https://github.com/gsoh/VED), **Apache-2.0**.

### A2.1 Signal sampling rates (verified from the paper) — **this was wrong in plan v1**

| Signal | Rate | EV availability |
|---|---|---|
| HV Battery **Current** [A] | **1 s** | 100 % |
| HV Battery **Voltage** [V] | **5 s** | 100 % |
| HV Battery **SOC** [%] | **60 s** | 100 % |
| Vehicle Speed [km/h] | 1 s | ✅ |
| GPS lat/lon | **3 s** | ✅ (fogged — see A2.2) |
| Outside Air Temperature [°C] | — | **100 %** |
| A/C Power [kW]/[W], Heater Power [W] | 60 s | verify per-vehicle at load |
| Engine RPM | 2 s | **0 % for EVs**, 100 % for PHEVs |
| Fuel Rate | — | **0 % for EVs**, 50 % for PHEVs |

**Consequence for energy integration.** `P = V·I` cannot be formed directly: current arrives every second, voltage every five.

> **Adopted method (E1-prep):** resample voltage onto the 1 s current grid by **linear interpolation** (pack voltage varies slowly and smoothly under load relative to current), then integrate `E = Σ V(t)·I(t)·Δt`.
> **Mandatory error budget:** recompute with **zero-order hold** and report the difference as a systematic-uncertainty band on every Wh/km figure. If the two differ by more than ~2 %, the interpolation choice becomes a reported limitation rather than a footnote.
> **Independent cross-check:** compare integrated energy against the **SOC drop × nominal pack capacity** (24 kWh). SOC is coarse (60 s, ~0.5 % resolution) so this is a *sanity* check, not a calibration — agreement within ~10 % on long segments validates the sign convention and the integration.

**Sign convention is unverified.** Whether discharge is positive or negative current is not stated in the sources checked. **Phase 1 task:** determine empirically — on a segment with a known large SOC drop, the sign that yields positive net discharge energy is the discharge sign. **Do not assume.**

### A2.2 GPS privacy fogging — the most consequential finding

The VED authors applied three de-identification steps:
1. **Random fogging** — GPS at the **start and end of every trip is masked**.
2. **Geo-fencing** — data restricted to Ann Arbor city limits.
3. **Major-intersection bounding** — only the trace between the first and last major intersections is retained.

**Why this breaks the naive approach.** Plan v1 assumed `Wh/km = (∫V·I over trip) / (GPS distance)`. But energy is logged for the *whole* trip while GPS covers only the *middle*. Dividing full-trip energy by truncated distance **systematically inflates Wh/km**, and the inflation is worst on short trips — a bias correlated with trip length, i.e. exactly the kind of confound that produces a spurious "distance affects efficiency" result.

> **Adopted resolution — redefine the unit of analysis.**
> The unit is not a *trip*. It is an **analysis segment**: the contiguous window in which GPS is available.
> - Energy: integrated **over exactly that window only**.
> - Distance: integrated from **1 s vehicle speed** over that same window (more reliable than 3 s GPS differencing, and immune to GPS noise).
> - Route features (grade, elevation, speed limit) describe **that same window**.
>
> Energy, distance and route features then all refer to one identical, well-defined interval. **Terminology discipline:** the report says "analysis segment" throughout, never "trip". Say this unprompted in the viva — it demonstrates you read the dataset paper rather than the column headers.

**Additional consequences:**
- Segments exclude cold-start/park-up behaviour and the first/last minutes, which is when HVAC transients are largest. **Stated limitation.**
- Geo-fencing to Ann Arbor means **low grade variance**. Ann Arbor is gently rolling. The `F_grade = m·g·sinθ` term is therefore **weakly excited** by this data, so the project cannot claim validated performance on steep terrain. The hilly/ghat presets in the app remain a **demonstration**, explicitly not validated.

### A2.3 Vehicles

- **3 pure BEVs**, all **2013 Nissan Leaf, 24 kWh** → the approved primary vehicle.
- **24 PHEVs across 4 models** — see A6, which uses them to repair the statistical weakness of a 3-vehicle study.
- Static files give `VehId, EngineType, Vehicle Class, Engine Configuration & Displacement, Transmission, Drive Wheels, Generalized_Weight[lb]`. **Note `Generalized_Weight` is binned/generalised, not an exact kerb mass** — so vehicle mass is an **assumption**, not a measurement, and payload is entirely unobserved.

### A2.4 Time encoding

`DayNum = 1` ⇒ 1 Nov 2017 00:00:00, fractional days give time of day; `Timestamp(ms)` is within-trip. **Neither source states whether this is local time or UTC.** Ann Arbor is UTC−5 (EST) / UTC−4 (EDT), and the dataset spans a full year, so it crosses two DST transitions. **An unresolved 4–5 hour offset would corrupt every weather-derived feature** by sampling the wrong part of the diurnal cycle.

**This is resolved empirically in A4.3 — not assumed.**

---

## A3. eVED — verified, and the join is exact

**Citation:** S. Zhang et al., "Extended Vehicle Energy Dataset (eVED)," [arXiv:2203.08630](https://arxiv.org/abs/2203.08630), 2022; conference version at IEEE VTC2025-Spring, Oslo.

| Item | Verified value |
|---|---|
| **Join keys preserved** | **`VehId`, `Trip`, `DayNum`, `Timestamp`** — identical to VED |
| Join type | **Exact, record-level.** Not a spatial or temporal approximation |
| Added | Map-matched lat/lon, **road elevation** (12,609,170 records), **speed limit** (12,203,044), speed limit with direction, intersections, bus stops, crossings, traffic signals, stop signs, railway crossings, motorway junctions |
| Coverage | All VED GPS trace records |
| Method | HMM map-matching via the Valhalla routing engine; attributes from QGIS / Overpass / OSM / Google Maps |
| Licence | **Apache-2.0** |
| **Download** | `git clone https://Datarepo@bitbucket.org/datarepo/eved-dataset.git` |
| Doc mirror | [github.com/zhangsl2013/eVED](https://github.com/zhangsl2013/eVED) |

> ⚠️ **Correction to plan v1 and to several published citations:** the widely-quoted URL `github.com/zhangs12013/eVED` **returns HTTP 404**. The correct path is **`zhangsl2013`** (letters "s-l", not "s-one"). Data lives on Bitbucket, not GitHub.

**Unverified:** map-matching accuracy metrics and elevation vertical accuracy are **not stated** in the sources checked. **Phase 1 must measure a proxy**: compare eVED elevation against Open-Meteo/SRTM elevation at the same matched coordinates and report the RMS difference as the elevation-uncertainty band. Do not quote an accuracy figure the authors never published.

**Verdict on VED + eVED: ✅ JOIN VERIFIED.** Same records, same keys, same provenance, compatible licence. This is an *enrichment*, not a merge of independent sources.

---

## A4. Weather — Open-Meteo archive, and how the join is made falsifiable

### A4.1 Verified API facts

| Item | Verified |
|---|---|
| Endpoint | **`https://archive-api.open-meteo.com/v1/archive`** (**not** the `/v1/forecast` endpoint the repo currently calls) |
| API key | **Not required** for non-commercial use |
| Licence | **CC-BY 4.0** — attribution required in the report |
| Required params | `latitude`, `longitude`, `start_date`, `end_date` (ISO-8601), `hourly=` |
| Variables needed | `temperature_2m`, `wind_speed_10m`, `wind_direction_10m`, `precipitation` |
| Resolution over 2017–18 | **ECMWF IFS at 9 km** (2017→present) — better than ERA5's 0.25 °/~25 km. ERA5-Land 0.1 ° (~11 km) also available |
| Timezone | `timezone` parameter; default GMT |
| Rate limits | **Not explicitly published.** Treat as a real risk → cache aggressively (K-new-1) |

**Adopted convention:** request with **`timezone=GMT`** and perform the offset conversion **explicitly in our own code**, so the timezone assumption is a single visible, testable line rather than a hidden server-side behaviour.

### A4.2 Is the join scientifically defensible?

**Yes, with three stated caveats**, because it is a **physical join on (place, time)** — the vehicle genuinely was at that coordinate in that hour, and the reanalysis genuinely describes conditions there. It is *not* a schema merge of two unrelated tables.

**Caveats that go in the report:**
1. **Scale mismatch.** A 9 km grid cell is an area average. Local gusts, street-canyon wind, and road-surface wetness are unresolved. Wind in particular is the weakest variable — `headwind_ms` derived from a 9 km gridded wind is a **coarse estimate**, and should be treated as a low-confidence feature.
2. **Reanalysis ≠ observation.** ERA5/IFS is a physical model constrained by observations, not a measurement at the roadside.
3. **"Weather as it turned out", not "the forecast at departure."** A deployed advisor would face forecast error on top of model error. We do not model that, and we say so.

### A4.3 🔬 The timezone problem, and how we make the join falsifiable

The unresolved local/UTC ambiguity (A2.4) is the single largest silent-failure risk in the data pipeline. Rather than assume an offset, **we determine it empirically and, in doing so, validate the entire weather join.**

> **Procedure — Phase 1 Gate G1 (must pass before any modelling):**
> 1. VED provides **onboard Outside Air Temperature** at **100 % availability for EVs** (F3). This is a *measured* temperature at the vehicle.
> 2. Fetch Open-Meteo `temperature_2m` for the matched coordinates and date range.
> 3. For each candidate offset ∈ {UTC−0, −4, −5, and DST-aware Detroit local}, compute the correlation and RMS error between onboard OAT and the reanalysis temperature across all segments.
> 4. **The correct offset is the one that maximises correlation.** A well-posed join should show a sharply peaked, unambiguous optimum, with the diurnal and seasonal cycles aligning.
>
> **Gate criterion:** the join is accepted only if the best offset gives **Pearson r ≥ 0.9** on segment-mean temperature, with a clearly dominant peak.
> **If the gate fails:** the join is broken or the coordinates are unusable. **Stop and diagnose — do not proceed to modelling.**

**Why this is more than a bug-check.** It converts an untestable assumption into a measured, reported quantity, and it yields an independent estimate of reanalysis-vs-onboard agreement that goes straight into the limitations section. *This is a strong thing to present in the viva: "we did not assume our weather join was correct — we tested it against an onboard sensor, and here is the correlation."*

**Known bias to handle:** onboard air-temperature sensors read high when a vehicle is stationary or just after soak (heat from pavement and the powertrain). Mitigation: use the **median OAT over the moving portion** of a segment, and exclude the first 2 minutes.

### A4.4 What weather is actually used for

Because of F3, the roles change from plan v1:

| Variable | Source | Provenance |
|---|---|---|
| `temp_c` | **Onboard VED OAT sensor** (primary) | **Measured** |
| `headwind_ms` | Open-Meteo wind speed + direction × segment bearing | **External + Derived** (low confidence) |
| `rain` | Open-Meteo `precipitation > threshold` | **External** |
| *(reanalysis temperature)* | Open-Meteo | **Validation only** (Gate G1), not a model input |

**This is strictly better than plan v1**, which would have used a 9–25 km grid average as the temperature feature while an onboard sensor sat unused in the same file.

---

## A5. Battery datasets — status

| Dataset | Status | Action |
|---|---|---|
| **NASA PCoE** (Saha & Goebel) | ✅ Public, free, no registration. ~34 18650 cells, 2 Ah, CC-CV 1.5 A→4.2 V, multiple temperatures | **Primary for E5.** Confirm exact file structure at download |
| **Oxford Battery Degradation Dataset 1** (Birkl, 2017) | ✅ Public. 8 Kokam 740 mAh pouch cells, 40 °C, Artemis urban drive-cycle discharge | **Cross-dataset transfer (E5b)** |
| **MIT/Stanford–Toyota** (Severson et al., *Nature Energy* 2019) | ✅ Public. 124 A123 LFP cells | **Optional tier only** |
| **EVBattery** (Tsinghua, Figshare) | ✅ Open access. 464 EVs, >1.2 M charging snippets | **Optional / future work.** Adds real pack-level SoH |
| **IEEE DataPort Indian e-cycle** | ⛔ **Dropped per decision 3** (access-restricted) | Not used. Flag only if it becomes essential |
| **ZTBus** | ⛔ **Out of primary scope per decision 2** | Documented as future work |
| **DualEMobility e-scooter** | 🟡 Demoted to Optional | 30 real trips is validation-only; also contains 10,000 SDV-synthetic rows that must be excluded |

**The rule that does not change:** battery cell data is **never merged** with trip data. The two connect only through `range = usable_energy(SoH, SoC, T) / consumption(...)`, and any such result is tagged **`COUPLED-SIM`**.

---

## A6. 🔧 Repairing the 3-vehicle problem without ZTBus

Decision 2 removes the cross-vehicle-class experiment. With only 3 BEVs, a leave-one-vehicle-out result rests on n = 3 — too weak to carry RQ1 alone.

**Verified substitute, entirely inside VED:** VED contains **24 PHEVs across 4 models**. When a PHEV runs in charge-depleting (electric) mode it is, physically, a battery-electric vehicle — the same road-load physics, the same `P = V·I` pack energy.

> **Adopted: E2 level L4 becomes cross-vehicle-model transfer** — fit on the 3 BEV Leafs, test on **PHEV electric-mode segments** (4 models, up to 24 vehicles).
>
> **Electric-mode filter — verified feasible:** `Fuel Rate` is only 50 % available for PHEVs and **0 % for EVs**, so it cannot be the filter. **`Engine RPM` is 100 % available for PHEVs.** Filter on **`Engine RPM == 0` for the entire segment**, plus a guard band excluding segments within N seconds of any non-zero RPM.

**Why this is a better fit than ZTBus for this project:** it needs no new data pipeline, no new licence, no new vehicle class; it reuses the identical loader and feature code; and it raises the vehicle count for the shift claim from **3 to as many as 27**.

**Caveats to state:** PHEV packs are smaller and may be operated closer to their power limits; kerb masses differ and are only *generalised* in the static file; brief engine starts must be excluded conservatively. Vehicle parameters must be **re-identified per model**, not shared.

---

# PART B — CITATION AUDIT

Every reference in the submitted synopsis was checked against primary sources.

## B1. ⛔ Reference [1] is fabricated — conclusive

> **Synopsis [1]:** A. Kumar and B. Singh, "Electric Vehicle Range Prediction using Machine Learning Techniques," *IEEE Transactions on Intelligent Transportation Systems*, vol. 23, no. 5, pp. 1234–1242, 2022.

**Three independent lines of evidence:**

1. **No such paper is findable.** Repeated targeted searches on title, authors, venue and volume return nothing.
2. **🔴 The pagination is structurally impossible.** IEEE T-ITS volume 23 (2022) is paginated continuously across issues. The **verified** VED paper sits at **vol. 23, no. 4, pp. 3302–3312**. Issue 4 therefore already spans pages in the 3000s, so **issue 5 begins above ~3900**. A paper at **pages 1234–1242 cannot be in issue 5** — that page range falls in an early issue of the volume. **The citation contradicts itself.**
3. **Placeholder signature.** "pp. 1234–1242" is a canonical filler page range, paired with maximally generic author names ("A. Kumar and B. Singh").

**Verdict: fabricated. Remove it.** This is the classic signature of an AI-generated or padded reference list, and it is the first citation an examiner would spot-check.

### Proposed legitimate replacements (verified, same topic)
| Use it for | Replacement |
|---|---|
| General ML range prediction (best single substitute) | **"Electric Vehicle Range Prediction Models: A Systematic Review of Machine Learning, Mathematical, and Simulation Approaches," *World Electric Vehicle Journal*, vol. 16, no. 11, 607, 2025.** DOI [10.3390/wevj16110607](https://doi.org/10.3390/wevj16110607) |
| The dataset + an actual T-ITS citation | **G. Oh, D. J. LeBlanc, H. Peng, IEEE T-ITS, vol. 23, no. 4, pp. 3302–3312, 2022.** DOI [10.1109/TITS.2020.3035596](https://doi.org/10.1109/TITS.2020.3035596) |
| The physics baseline | **C. Fiori, K. Ahn, H. A. Rakha, "Power-based electric vehicle energy consumption model," *Applied Energy*, vol. 168, 2016** ⚙️ verify DOI |

## B2. References [2]–[5] — verified, with corrections required

| Ref | Verdict | Correction needed |
|---|---|---|
| **[2]** L. Mbagaya, K. Reddy, A. Botes, "Machine Learning Techniques for Battery State of Health Prediction: A Comparative Review," *World Electric Vehicle Journal*, **16(11), 594, 2025**. DOI [10.3390/wevj16110594](https://doi.org/10.3390/wevj16110594) | ✅ **Real** | 🔴 **Literature Review §4(2) describes this same paper as "Elsevier, 2023".** It is **MDPI, 2025**. Fix the venue and year. |
| **[3]** S. Jo, S. Jung, T. Roh, "Battery State-of-Health Estimation Using Machine Learning and Preprocessing with Relative State-of-Charge," ***Energies*, 14(21), 7206, 2021**. DOI [10.3390/en14217206](https://doi.org/10.3390/en14217206) | ✅ **Real** | Add article number **7206**. Authors: Sungwoo Jo, Sunkyu Jung, Taemoon Roh (ETRI, Korea) |
| **[4]** L. Zhang et al., "Improved LSTM based state of health estimation using random segments of the charging curves **for lithium-ion batteries**," *J. Energy Storage*, vol. 74, 109370, 2023 | ✅ **Real** | Title in the synopsis is **truncated** — restore "for lithium-ion batteries" |
| **[5]** S. V. Kulkarni et al., "Advanced battery diagnostics for electric vehicles using CAN based BMS data with EKF and data driven predictive models," ***Scientific Reports*, 2025**. DOI [10.1038/s41598-025-18042-6](https://doi.org/10.1038/s41598-025-18042-6) | ✅ **Real** | Year is **2025**, not "2025/2026". Add the DOI |

## B3. Literature Review §4 entries (1), (3) and (4)

These are described **by topic only — no author, venue, or DOI**:
- (1) "State of Health Estimation in EV Batteries Using AI-enhanced BMS" (IEEE, 2024)
- (3) "Advanced Battery Diagnostics for Electric Vehicles Using Data-Driven Models" (Springer, 2024)
- (4) "Battery State-of-Health Estimation Using Machine Learning and Preprocessing" (IJER, 2023)

**Assessment:** (3) is probably a garbled restatement of **[5]** (Kulkarni, *Sci. Rep.* 2025 — Nature Portfolio, not Springer 2024), and (4) is probably a garbled restatement of **[3]** (Jo et al., *Energies* 2021 — not IJER 2023). **Entry (1) could not be matched to any real paper.**

**Action:** rewrite Literature Review §4 so that **every entry carries an author, venue, year and DOI that a reader can open**, and remove duplicates that restate the same work under a different venue. Verified replacements are in [02_RELATED_WORK.md](02_RELATED_WORK.md).

## B4. Standing rule for the rest of the project

> **No citation enters the report unless a team member has personally opened it and recorded its DOI.**
> Maintain `docs/REFERENCES.bib` with a `verified_by` and `verified_date` field on every entry. An unopened citation is a liability, not a reference.

---

# PART C — WHAT CHANGED, AND WHAT IT MEANS

## C1. Corrections to plan v1

| Plan v1 assumption | Verified reality | Impact |
|---|---|---|
| "1 Hz HV current and voltage" | Current 1 s, **Voltage 5 s, SOC 60 s** | Resampling + error budget required |
| "Wh/km = trip energy ÷ GPS distance" | **GPS is fogged at both ends** | Unit of analysis becomes the **analysis segment**; distance from 1 s speed |
| "ERA5 supplies temperature" | **Onboard OAT at 100 % for EVs** | Temperature becomes **measured**; reanalysis is demoted to join *validation* |
| "`aux_on` stays a synthetic binary" | **A/C and Heater power channels exist** | Auxiliary load becomes **measured** |
| "Filter PHEV electric mode on Fuel Rate == 0" | Fuel Rate **0 % for EVs**, 50 % for PHEVs; **RPM 100 % for PHEVs** | Filter on **Engine RPM == 0** |
| "eVED at github.com/zhangs12013/eVED" | **404.** Correct: `zhangsl2013`; data on Bitbucket | Corrected in the manifest |
| "Use ERA5 0.25°" | **ECMWF IFS 9 km covers 2017–18** | Better resolution available |
| "Cross-class transfer via ZTBus" | Out of scope (decision 2) | Replaced by **PHEV electric-mode transfer**, 3 → up to 27 vehicles |
| "Synopsis ref [1] is probably fabricated" | **Conclusively fabricated** (pagination proof) | Remove; replacements proposed |

## C2. Net assessment

**Three findings strengthen the project:**
- Temperature and auxiliary load become **measured quantities** rather than assumptions — a genuine gain in EV-domain credibility, since HVAC is the dominant cold-weather range factor in a real EV.
- The weather join becomes **empirically falsifiable** via the onboard-sensor cross-check (Gate G1), instead of an untested assumption.
- The PHEV electric-mode substitute gives the vehicle-shift experiment **up to 27 vehicles instead of 3**, without adding a dataset.

**Two findings constrain it:**
- The **analysis segment** replaces the trip as the unit of analysis, and the terminology must be used consistently everywhere.
- **Low grade variance** in geo-fenced Ann Arbor means steep-terrain performance **cannot be validated** — only demonstrated.

**Verdict: the data plan is sound and the joins are defensible.** Proceed to implementation, subject to Gate G1.
