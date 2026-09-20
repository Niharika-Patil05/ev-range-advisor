# Investigation Report — what I inspected, what I found, and why it matters

**Audience:** Niharika + the 4 teammates (Sharyu, Megha, Viraj, Atharva) and Dr. N. Vengadachalam.
**Status:** Investigation only. **No source code was modified, no data generated, no models trained.**
**Date:** 20 September 2026.
**Companion documents:** [01_RESEARCH_PLAN.md](01_RESEARCH_PLAN.md) (the plan), [02_RELATED_WORK.md](02_RELATED_WORK.md) (papers).

---

## 0. How I worked (so you can defend the process)

| Step | What I did | Why |
|---|---|---|
| 1 | Read every file in `src/`, `app/`, `tests/`, `scripts/`, `reports/`, `README.md`, `CLAUDE.md` | The plan has to describe *your* code, not a generic template |
| 2 | Read the numbers in `reports/*.csv` and `summary.json` and traced each one back to the code that produced it | To find out whether the results mean what they appear to mean |
| 3 | Read `Minor_Project_synopsis (1).pdf` (the submitted synopsis) | Source of truth for original objectives, scope, terminology |
| 4 | Compared synopsis → code → reports three ways | To separate *proposed*, *implemented*, and *demonstrated* |
| 5 | Searched the actual public-data landscape (EV trips, battery ageing, weather, elevation, traffic) and checked size / licence / access for each | Never design a plan around data you cannot get |
| 6 | Searched the literature for the methods we intend to use | To know what is already published, so we do not claim novelty we do not have |
| 7 | Verified the synopsis's own citations | An examiner will check them |

Everything below is traceable to a file, a line, a dataset page, or a paper.

---

## 1. What the synopsis proposed

From `Minor_Project_synopsis (1).pdf` (submitted 20-03-2026, Dept. of Electrical Engineering, ADCET Ashta, guide Dr. N. Vengadachalam):

**Objectives (Section 5):**
1. Accurate EV range prediction model using ML.
2. Battery State-of-Health (SoH) estimation.
3. Analyse impact of real-time factors — **weather, traffic, driving behaviour**.
4. Intelligent suggestions to improve efficiency.

**Methodology (Section 5):** data collection (SOC, voltage, current, weather, traffic, road elevation, driving behaviour) → preprocessing → model development (Regression / RF / NN) → prediction system → performance evaluation → UI → system validation.

**Block diagram:** Input Data → Preprocessing → ML Model (Random Forest) → {Range Prediction, Battery Health (SoH)} → Recommendations.

**Two constraints the synopsis itself imposes, which I have respected throughout the plan:**
- *"The system is designed as a fully software-based solution"* and *"does not require additional hardware components."* → **No hardware/logger work is in the plan.** This also agrees with rule 8 in `CLAUDE.md`.
- *"cost-effective... uses open-source tools and simulation platforms."* → **Every dataset and API in the plan is free and key-less or free-with-registration.** No paid API is required.

---

## 2. What actually exists in the repository

The repo is **not** a stub. It is a competently engineered, fully wired scaffold — roughly 1,100 lines of clean, typed, documented Python with 14 passing tests. Credit where due: the architecture is better than most final-year projects at this stage.

```
src/physics/road_load.py      road-load energy model (roll + aero + grade + stops + aux), unit-tested
src/features/trip_features.py segment table + conditions -> 16 trip-level features + physics baseline
src/features/route_pipeline.py geometry helpers (tested) + OSRM/Open-Meteo wrappers (UNTESTED)
src/data/synthetic.py         synthetic trip generator + synthetic battery-cycle generator
src/data/loaders.py           real-data loaders: schema validation ONLY, never exercised
src/models/range_model.py     ConstantRated, PhysicsOnly, Linear, RF, GBM, HybridResidual
src/models/uncertainty.py     split conformal + quantile-regression intervals
src/models/soh_model.py       SoH regression with leave-one-battery-out (LOBO)
src/models/coupling.py        usable_energy_wh(), temp_derate(), project_soh()
src/advisor/advisor.py        range band + counterfactual "why" + rule-based tips
src/pipeline.py               trains everything, writes reports/ and models/system.joblib
app/streamlit_app.py          2 tabs: Trip advisor, Battery health
```

**The hybrid model** (`range_model.py:58-77`) is the technical centre of the project and it is a genuinely sound idea:

```
prediction = phys_wh_km * exp( f(features) ),   f learned on log(y_measured / y_physics)
```

Learning a **log-ratio residual** is better than learning the raw residual: it keeps the prediction strictly positive, makes the correction multiplicative (scale-free across trip lengths), and means the ML layer only has to learn *what the physics got wrong*, not the energy itself. This is the right architecture and it should stay.

**Split discipline** (`pipeline.py:73-77`) is already correct: `GroupShuffleSplit` on `route_id` for train/cal/test, `GroupKFold` for CV, and an explicit `assert` that no route leaks across splits. Battery models already use leave-one-battery-out. This is better hygiene than most published student work.

---

## 3. The central finding: every reported number is circular

This is the most important thing in this report, so it gets its own section.

`reports/summary.json` says the hybrid achieves **2.93 % MAPE**. Here is where that number comes from.

The "ground truth" consumption is generated by `synthetic.py:61-76`:

```python
def _true_wh_per_km(seg, cond, rng, vehicle):
    crr_scale = 1.15 * (1.25 if cond["rain"] else 1.0) * (1 + 0.004*max(0, 15-cond["temp_c"]))
    wh = segment_energy_wh(...)                      # <-- the SAME function
    battery_losses = 1 + 0.25*(1-cond["soh"]) + 0.006*max(0, 10-cond["temp_c"])
    return wh/(dist_km) * battery_losses * rng.lognormal(0, 0.03)
```

The physics feature `phys_wh_km` is produced by `trip_features.py:42-49` — **the same `segment_energy_wh` call**, with `crr_scale = 1.0` and the nominal speed.

Therefore, by construction:

```
log(y_true / phys_wh_km) = log(crr_scale) + log(battery_losses) + log(style speed effect) + N(0, 0.03)
                         = a smooth, low-dimensional, deterministic function of
                           {rain, temp_c, soh, style}  +  3 % noise
```

The hybrid's ML layer is handed `rain`, `temp_c`, `soh`, and `style` as input features and asked to learn exactly that function. **A gradient-boosted tree recovers it almost perfectly, and 2.93 % is simply the 3 % lognormal noise floor.** The model did not discover anything; it inverted an algebraic identity we wrote ourselves.

The same circularity invalidates three more results:

- **The ablation** (`reports/ablation.csv`) concludes "weather features matter" (GBM 10.4 → 21.4 MAPE without them). Guaranteed: `crr_scale` is *built from* `rain` and `temp_c`. The experiment can only confirm its own premise.
- **The SoH result** (`reports/soh_results.csv`, RF LOBO MAE **1.11 pp**). In `synthetic.py:97-119` every health indicator is a monotone function of `soh` plus small noise (`ir_mohm = f(soh)`, `cc_time_min = 100*soh*scale`, `ic_peak = 1.8*soh²*scale`). Real cells have knee points, path dependence, and cell-to-cell scatter in the HI→SoH mapping. Expect real LOBO error **2–5×** worse.
- **`range_MAPE`** (`pipeline.py:32-36`) looks like independent validation of the SoH↔range coupling. It is not. Algebraically:
  ```
  |r_pred - r_true| / r_true  =  |U/ŷ - U/y| / (U/y)  =  |y - ŷ| / ŷ
  ```
  The usable energy `U` cancels exactly. `range_MAPE` is just MAPE with `ŷ` in the denominator instead of `y` — which is why the CSV shows 2.927 vs 2.957 for the hybrid. **It measures nothing about the battery side of the system.**

**None of this is a criticism of the code.** `README.md` lines 40-45 say this explicitly and honestly ("The hybrid model looks excellent (~3 % MAPE) partly *by construction*"). The scaffold did its job. But it means: **the project currently has zero evidence about the real world, and every headline number must be discarded and re-earned.** That, not new model architectures, is the work.

---

## 4. Proposed vs. implemented vs. missing

| Synopsis promised | In the code? | Reality |
|---|---|---|
| ML range prediction | ✅ 6 models | Trained only on self-generated data |
| SoH estimation | ✅ with LOBO | Synthetic cells with near-trivial HI→SoH mapping |
| **Weather** impact | ✅ feature + live API | Live API **never executed** (`route_pipeline.py` header admits this) |
| **Road elevation** | ✅ grade in physics + Open-Meteo elevation | Elevation API never executed |
| **Traffic** | ❌ | Only a UI slider (`streamlit_app.py:46`) that maps to `stops_per_km`. Never a model feature. **Synopsis objective 3 is currently unmet.** |
| **Driving behaviour** | ⚠️ partial | A single 3-level `style` integer, injected by us, not measured |
| Recommendations | ✅ | Rules + counterfactuals; works, but thresholds are placeholders |
| UI | ✅ Streamlit | Functional; shows no evidence/experiments to an examiner |
| Validation on real data | ❌ | Loaders exist but have never been run on a real file |

**Beyond the synopsis, these are also missing and matter for a research-grade project:** experiment tracking, multi-seed runs, any distribution-shift evaluation, conditional (not just marginal) interval coverage, SHAP or any feature attribution on the ML layer, calibrated vehicle parameters, and calibrated degradation constants.

---

## 5. Engineering defects and design blockers found by reading the code

Ordered by how much they block the research work.

| # | Finding | Location | Why it matters |
|---|---|---|---|
| **B1** | `summarize_trip()` and every model hard-code `DEFAULT_VEHICLE` | `trip_features.py:29`, `range_model.py:80` | **Hard blocker.** Real data means a Nissan Leaf, a bus, an e-scooter — different mass/Cd/A/η. Cross-vehicle generalization (the strongest experiment available) is *impossible* until `vehicle` is threaded through as a parameter. |
| **B2** | No SoH *estimate* ever reaches the range model | `pipeline.py` | `soh` is an oracle input feature and an oracle multiplier in `usable_energy_wh`. The "coupling" in the project title is **asserted, never measured.** This is also the single best research opening (see plan, RQ3). |
| B3 | `temp_derate` (−0.6 %/°C) and all six constants in `project_soh` are invented | `coupling.py:9-42` | The whole Battery tab is currently uncalibrated fiction. Code comments say so honestly. |
| B4 | Quantile intervals badly under-cover: **0.73 vs 0.90 target** | `reports/uncertainty.csv` | Real defect, not a synthetic artefact: quantiles are fitted on the *log-residual* then exponentiated, and are never conformalized. Fix = CQR. |
| B5 | Conformal coverage is **marginal only** | `uncertainty.py:9-27` | 91 % overall can hide 60 % coverage on cold or hilly trips. Untested. |
| B6 | Train/serve skew in `stops` | `synthetic.py:44` (Poisson ints) vs `route_pipeline.py:83` (fractional) | Models are trained on integer stop counts and served fractional ones. |
| B7 | `soc_start` is computed and returned but is in no feature group | `trip_features.py:18,68` | Dead feature. Also a missed physical effect (low SoC → lower pack voltage → higher current → higher I²R loss). |
| B8 | `rain` reaches the ML layer but not the physics layer | `trip_features.py:47` passes no `crr_scale` | Asymmetry between the two halves of the hybrid. Defensible, but must be stated. |
| B9 | Network wrappers never executed | `route_pipeline.py:88-151` | Unknown whether OSRM/Open-Meteo calls work at all. Untested code in a demo path. |
| B10 | Single seed, single run, no experiment IDs | `config.py:18`, `pipeline.py` | Cannot report mean ± std; cannot reproduce a specific figure later. |
| B11 | `models/system.joblib` is a build artefact sitting in the tree; repo is **not a git repository** | — | No version history at all. For a 5-person team over 4 months this is a real risk. |
| B12 | 752 MB `venv/` inside the project directory | — | Will be committed by accident the moment git is initialised. |

---

## 6. Dataset landscape — what actually exists (verified, not assumed)

I checked availability, size, variables, and licence for each. **Verdict column is my recommendation.**

### 6.1 Real EV driving / energy data

| Dataset | What it really is | Fit | Verdict |
|---|---|---|---|
| **VED** (Vehicle Energy Dataset, Univ. of Michigan) — [github.com/gsoh/VED](https://github.com/gsoh/VED) | 383 vehicles, Ann Arbor, Nov 2017–Nov 2018, ~374,000 miles, 1 Hz OBD-II. Includes **HV Battery Current [A], Voltage [V], SOC [%]** + GPS + speed. Apache-2.0. | Lets you compute **measured Wh/km** by integrating V·I — exactly the target this project needs. | ✅ **Primary spine.** |
| ⚠️ VED caveat | Of 383 vehicles only **3 are pure BEVs**, all 2013 Nissan Leaf (24 kWh); 24 more are PHEVs. | You get thousands of *trips* but only **3 vehicles**. | Vehicle-wise generalization within VED is weak — must be supplemented. State this limitation loudly. |
| **eVED** — [github.com/zhangs12013/eVED](https://github.com/zhangs12013/eVED), data at `bitbucket.org/datarepo/eved-dataset` | VED with GPS map-matched via HMM/Valhalla, plus **12.6 M elevation** and **12.2 M speed-limit** records. | Gives trustworthy **road grade** (VED's raw GPS altitude is too noisy) and enables a **congestion proxy** = speed limit − observed speed. | ✅ **Use instead of raw VED GPS.** |
| **ZTBus** (ETH Zürich, *Scientific Data* 2023) — DOI [10.3929/ethz-b-000626723](https://doi.org/10.3929/ethz-b-000626723) | **1,409 full-day missions**, 2 electric trolley buses, Zurich, several years. Power demand, odometry, GPS, **ambient temperature**, door openings, **passenger count**. | Passenger count → **time-varying mass**, which is a real physical variable this project currently only simulates. A completely different vehicle class. | ✅ **The cross-vehicle-class test.** Strongest available generalization experiment. |
| **DualEMobility** (Dublin City Univ.) — [github.com/SFIEssential/DualEMobilityData-datasets](https://github.com/SFIEssential/DualEMobilityData-datasets) | **30 e-scooter + 36 e-bike** real trips. GPS, altitude, speed, SoC, weather. MIT licence. | Same vehicle class as your `VehicleSpec` (e-scooter). | ⚠️ **Validation only.** 30 trips is far too few to train on. **Contains 10,000 synthetic records generated with Synthetic Data Vault — these must be excluded and never counted as real.** |
| **Indian e-cycle SOC/SOE dataset** — IEEE DataPort | 30 trips on Indian national highways, Sep–Dec 2025, riders 55–105 kg, calm vs aggressive riding. | Closest thing to your geography, and it actually varies **load** and **riding style** — your two weakest features. | ⚠️ **Enhancement.** Check institutional IEEE DataPort access first; do not build a deliverable on it. |
| **Argonne D3** (Downloadable Dynamometer Database) — [anl.gov/taps/downloadable-dynamometer-database](https://www.anl.gov/taps/downloadable-dynamometer-database) | Chassis-dyno tests incl. BEVs, 10 Hz, with **battery current, voltage, temperature** and wheel force. Free. | **Grade = 0, wind = 0, no traffic** — a controlled environment. | ✅ **Use for physics-parameter identification** (fit Crr and drivetrain η where confounders are absent), *not* for range claims. |

### 6.2 Real battery-ageing data

| Dataset | What it is | Verdict |
|---|---|---|
| **NASA PCoE** (B0005–B0018 etc.) | ~34 18650 cells, 2 Ah, CC-CV 1.5 A to 4.2 V, several temperatures, charge/discharge/EIS. | ✅ **Primary.** Small cell count, short cycle life, noisy — that is *realistic*. |
| **Oxford Battery Degradation** | 8 Kokam 740 mAh pouch cells, 40 °C, **Artemis urban drive-cycle discharge**, periodic characterisation. | ✅ **Second cell chemistry/format → cross-dataset transfer test.** The drive-cycle discharge is the closest lab analogue to vehicle use. |
| **MIT/Stanford–Toyota (Severson et al.)** | **124** A123 LFP 18650 cells, 72 fast-charge protocols, 30 °C, fixed 4C discharge. | ⚙️ **Enhancement.** Big enough for a real LOBO study, but it is a *cycle-life prediction* dataset (constant conditions); less suited to condition-dependent SoH. |
| **CALCE** (Univ. of Maryland) | LCO/LFP/NMC, cylindrical/pouch/prismatic, incl. dynamic driving profiles. | ⚙️ Optional third source. |
| **EVBattery** (Tsinghua, on Figshare) | **464 real EVs**, 3 manufacturers, **>1.2 M charging snippets**, with health and capacity labels. | ✅ **Important.** This is the only way to address the project's stated limitation *"lab-cell SoH does not transfer to a vehicle pack."* It gives **pack-level, real-vehicle SoH**. |

### 6.3 Weather, elevation, routing, traffic

| Source | Status |
|---|---|
| **Open-Meteo Historical Weather API** (ERA5, 1940→present, hourly, ~9–25 km) | ✅ **Free, no API key, CC-BY 4.0.** This is the critical enabler: VED ran Nov 2017–Nov 2018, so we need *historical* weather, not forecasts. The repo currently calls the *forecast* endpoint — must be changed. |
| **Open-Meteo Elevation** (SRTM ~90 m) | ✅ Free, no key. Adequate, but eVED's elevation is better for VED. |
| **OSRM public demo server** | ⚠️ Free, no key, but explicitly for light use. Fine for the live demo, **not** for bulk processing. Cache everything (the repo already does). |
| **Traffic (historical)** | ❌ **Not freely or reproducibly available.** HERE/TomTom require keys and offer no free historical tier; Uber Movement is discontinued. **Recommendation: do not promise live traffic.** Instead derive a reproducible congestion proxy from the data itself — `speed_deficit = speed_limit − observed_mean_speed` (eVED gives speed limits) plus stop rate and speed variability. This *honestly* fulfils synopsis objective 3 and is fully reproducible. |

### 6.4 Can these datasets be combined?

This deserves care, because casually merging them is the fastest way to fail a viva.

- ✅ **VED + eVED** — legitimate. Same records, same vehicles; eVED is an enrichment of VED, joined on identical keys.
- ✅ **VED/eVED + ERA5 weather** — legitimate. The join is on **(latitude, longitude, timestamp)**, i.e. a *physical* join: the trip genuinely happened at that place and hour, and ERA5 genuinely describes the weather there. Caveat to state: ERA5 is a ~25 km reanalysis grid, so it is an area average, not the microclimate at the vehicle.
- ❌ **VED + NASA/Oxford cells — CANNOT be merged into one training table.** They share variable *names* (temperature, current, voltage) but nothing else: different objects (a 2 Ah 18650 cell vs a 24 kWh pack), different conditions, different sampling, no shared entity. Joining them because the columns look alike is exactly the error the brief warned about.
- ⚠️ **But they can be legitimately *coupled*** — not by merging rows, but through the **physical equation** the project is already built on:
  ```
  range = usable_energy(SoH, SoC, T) / consumption(route, weather, state)
  ```
  Each side is fitted on its own real dataset; the coupling is a **propagation / sensitivity study**, not a joint fit. It must be labelled as such in every table and figure. The plan formalises this with a mandatory provenance tag: `REAL`, `SYNTHETIC`, or `COUPLED-SIM`.
- ⚠️ **VED + ZTBus + e-scooter** — never pooled into one training set. They are used as **separate domains** in a transfer experiment: fit on one, re-identify vehicle parameters, test on another. That is the experiment, not a data-cleaning step.

---

## 7. Literature check — what is already published

I checked whether our intended methods are novel. Blunt summary:

- **Physics-informed / hybrid road-load + ML residual for EV consumption — already published** (multiple 2023–2026 papers). Our hybrid is sound but **not novel as an architecture**.
- **Conformal prediction / CQR for EV energy and range — already published** (e.g. a 2026 *Energy Informatics* study comparing SVR-CP, adaptive CP, and LightGBM-CQR). Not novel.
- **SHAP for EV energy-consumption feature importance — already published** (WEVJ 2021 onward). Not novel.
- **SoH-coupled range estimation with uncertainty — exists**, mostly in the control/filtering literature (particle filters, EKF, Bayesian filtering), usually on one vehicle or in simulation.

**Therefore we must not claim to have invented any of these.** What is genuinely under-served, and what the plan targets, is the *combination under a rigorous protocol*:

> There is no public, reproducible benchmark that evaluates a physics-informed EV consumption model **across route-, season-, and vehicle-class shifts on real open data**, reports **conditional (not just marginal) interval coverage**, and quantifies **how much of the range-prediction error budget comes from SoH estimation error rather than consumption error**.

That is a modest, honest, achievable contribution. Details in [01_RESEARCH_PLAN.md](01_RESEARCH_PLAN.md) §13.

---

## 8. ⚠️ Citation integrity warning (act on this before the report)

I attempted to verify the synopsis's own references.

- **Reference [1]** — *A. Kumar and B. Singh, "Electric Vehicle Range Prediction using Machine Learning Techniques," IEEE Transactions on Intelligent Transportation Systems, vol. 23, no. 5, pp. 1234–1242, 2022.*
  **Could not be located.** The page range "1234–1242" is a well-known placeholder pattern, and the author names are generic. **Treat this as probably non-existent** (a common symptom of an AI-generated or padded reference list). An external examiner who searches one citation will very likely search this one. **Remove or replace it.**
- **Reference [2]** — *L. Mbagaya, K. Reddy, A. Botes, "Machine Learning Techniques for Battery State of Health Prediction: A Comparative Review," World Electric Vehicle Journal, vol. 16, no. 11, p. 594, 2025.* ✅ **Verified real** — DOI [10.3390/wevj16110594](https://doi.org/10.3390/wevj16110594). Note the synopsis's Literature Review section §4(2) describes the *same* paper but attributes it to "Elsevier, 2023" — **an internal inconsistency to fix.**
- Literature Review entries §4(1), (3), (4) are described by topic only, with no author/DOI. **Each needs a real, checkable citation.**

[02_RELATED_WORK.md](02_RELATED_WORK.md) supplies a verified replacement bibliography.

---

## 9. Viva vulnerabilities as the project stands today

If the project were examined tomorrow, these questions would end it:

1. *"Your ground truth is generated by the same physics equation your model uses as an input. What have you actually learned?"* — **No answer exists today.**
2. *"Your title says SoH-coupled. Where in your results is SoH estimated rather than assumed?"* — **Nowhere.** `soh` is a slider.
3. *"Your synopsis promises traffic analysis. Show me."* — **Not implemented.**
4. *"Your conformal interval covers 91 %. What is the coverage on cold-weather trips specifically?"* — **Unmeasured.**
5. *"Why is your quantile method at 73 % when it targets 90 %?"* — Currently no diagnosis.
6. *"Your battery advice says charging to 80 % preserves X percentage points. Where does that constant come from?"* — **Invented** (`coupling.py:35-38`).
7. *"Random Forest scores 18.6 % MAPE, worse than plain physics at 11.3 %. Explain."* — Actually answerable (trees cannot extrapolate beyond the training envelope and the pure-ML models are denied `phys_wh_km`), and it is a *good* result to discuss — but nobody has written it down.

Each of these maps to a specific work item in the plan.

---

## 10. Bottom line

**What you have:** a clean, well-tested, honestly documented software scaffold with a sound hybrid architecture and correct split hygiene. As *engineering*, it is already above the bar.

**What you do not have:** a single piece of evidence about the real world, a measured SoH↔range coupling, traffic, or any experiment whose outcome was not determined in advance by its own data generator.

**What the plan does:** keeps the identity, the pipeline, the hybrid, the conformal band and the advisor exactly as they are — and replaces the synthetic evidence base with real public data, adds the three experiments whose results are genuinely unknown in advance, and calibrates the constants that are currently invented.

Nothing in the plan discards your work. It converts a working demo into a defensible study.
