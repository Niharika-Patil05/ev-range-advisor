# Related Work — papers behind this project, and exactly what we take from each

**Purpose:** so that for every design decision in the repository you can name a paper, say what it did, say what *we* used from it, and say where you differ. That is what "backing" means in a viva.

**How to read each entry:**
- **What it does** — the paper's own contribution
- **→ Used in this project for** — the specific module, decision, or experiment it justifies
- **⚠️ Where we differ / what we do NOT take** — so you never over-claim

**Verification status:** ✅ = I verified the citation in this session. ⚙️ = well-established work cited from knowledge — **verify the DOI on IEEE Xplore / the publisher site before it goes in your report.** Never paste a citation you have not personally opened.

---

## ⛔ FIRST — a correction to the synopsis's own reference list

| Ref in synopsis | Status | Action |
|---|---|---|
| **[1]** A. Kumar and B. Singh, *"Electric Vehicle Range Prediction using Machine Learning Techniques,"* IEEE T-ITS, vol. 23, no. 5, pp. 1234–1242, 2022 | ❌ **Could not be located.** Generic author names + the placeholder page range "1234–1242" are classic markers of a fabricated reference | **Remove it.** Replace with §1.1 or §5.1 below |
| **[2]** L. Mbagaya, K. Reddy, A. Botes, WEVJ vol. 16, no. 11, p. 594, 2025 | ✅ **Verified real** | Keep. **But fix the internal inconsistency:** Literature Review §4(2) describes this same paper as "Elsevier, 2023" |
| **[3]** S. Jo, S. Jung, T. Roh, *Energies*, vol. 14, no. 21, 2021 | ⚙️ Plausible — verify | Verify DOI |
| **[4]** L. Zhang et al., *J. Energy Storage*, vol. 74, p. 109370, 2023 | ⚙️ Plausible — verify | Verify DOI |
| **[5]** S. V. Kulkarni et al., *Scientific Reports*, "2025/2026" | ⚠️ Ambiguous year, unverified | Verify or remove |
| Literature Review §4 entries (1), (3), (4) | ⚠️ Described by topic only — **no author, venue or DOI** | Each needs a real checkable citation |

**An external examiner who checks one citation will check the most suspicious one.** Fix this before submission.

---

## 1. Datasets — the papers you MUST cite (using data obliges you to cite it)

### 1.1 ✅ VED — the project's data spine
**G. Oh, D. J. Leblanc, H. Peng, "Vehicle Energy Dataset (VED), A Large-Scale Dataset for Vehicle Energy Consumption Research," *IEEE Transactions on Intelligent Transportation Systems*, vol. 23, no. 4, 2022.** DOI [10.1109/TITS.2020.3035596](https://doi.org/10.1109/TITS.2020.3035596) · preprint [arXiv:1905.02081](https://arxiv.org/abs/1905.02081) · data [github.com/gsoh/VED](https://github.com/gsoh/VED) · Apache-2.0

**What it does:** releases 1 Hz OBD-II logs from 383 personal vehicles in Ann Arbor over a full year (~374,000 miles), including **HV battery current, voltage and SOC** for the electrified vehicles, with GPS.

**→ Used in this project for:** *everything real on the consumption side.* Measured Wh/km is obtained by integrating instantaneous pack power `P = V·I` over each trip and dividing by GPS distance — this becomes the target `wh_per_km` that currently comes from `src/data/synthetic.py`. Source for E0, E1, E2, E3, E4, E10.

**⚠️ Where we differ / limits:** the paper is a *dataset* paper, not a modelling result — it sets no accuracy bar for us. Critically, **only 3 of the 383 vehicles are pure BEVs** (2013 Nissan Leaf, 24 kWh), with ~24 PHEVs. This single fact drives limitation #1 in the plan and forces the ZTBus/e-scooter domains. Quote the "3 BEVs" number yourself before an examiner finds it.

### 1.2 ✅ eVED — the enrichment that makes grade trustworthy
**S. Zhang et al., "Extended Vehicle Energy Dataset (eVED): an enhanced large-scale dataset for deep learning on vehicle trip energy consumption," [arXiv:2203.08630](https://arxiv.org/abs/2203.08630), 2022.** Data: `bitbucket.org/datarepo/eved-dataset`

**What it does:** map-matches VED's noisy OBD GPS onto the road network with a Hidden Markov Model (via the Valhalla routing engine), then attaches **12.6 M road-elevation** and **12.2 M speed-limit** records from OSM/Overpass/QGIS.

**→ Used in this project for:** (a) reliable **road grade**, which feeds `F_grade = m·g·sinθ` in `src/physics/road_load.py` — raw GPS altitude is far too noisy for a grade derivative; (b) **speed limits**, which make the **congestion proxy** `speed_deficit = speed_limit − observed_speed` possible. That proxy is how the project finally delivers **synopsis objective 3 (traffic)** without a paid API.

**⚠️ Where we differ:** the title says "for deep learning"; we deliberately do **not** use deep learning on this tabular, trip-level target (plan §7.1). We use eVED purely as a feature source.

### 1.3 ✅ ZTBus — the cross-vehicle-class test
**F. Wegmann, ... , C. Onder et al., "ZTBus: A Large Dataset of Time-Resolved City Bus Driving Missions," *Scientific Data* 10, 2023.** DOI [10.1038/s41597-023-02600-6](https://doi.org/10.1038/s41597-023-02600-6) · data DOI [10.3929/ethz-b-000626723](https://doi.org/10.3929/ethz-b-000626723) · preprint [arXiv:2303.08667](https://arxiv.org/abs/2303.08667)

**What it does:** 1,409 full-day driving missions of electric city buses in Zurich across all seasons, with power demand, propulsion-system signals, odometry, GPS, **ambient temperature**, door openings and **passenger counts**.

**→ Used in this project for:** **experiment E2 level L4 — the strongest generalization test available.** We fit the hybrid on cars, re-identify only the `VehicleSpec` parameters (mass, Cd·A, Crr, η), and test on buses. If the *model form* transfers while only parameters change, that is real evidence the physics prior is doing work. **Passenger count also gives genuinely time-varying mass** — a physical variable the project currently only simulates via a `load_kg` slider.

**⚠️ Where we differ / risk K4:** these are **trolley** buses — check the power-source semantics in the paper before assuming all traction energy comes from an onboard pack. **Read this paper before writing the loader.** Buses also never pool with cars into one training set; they are a separate domain.

### 1.4 ✅ DualEMobility — your own vehicle class
**Data-driven Energy Consumption Modelling for Electric Micromobility using an Open Dataset, [arXiv:2403.17632](https://arxiv.org/abs/2403.17632), 2024.** Data: [github.com/SFIEssential/DualEMobilityData-datasets](https://github.com/SFIEssential/DualEMobilityData-datasets) (MIT)

**What it does:** releases 30 e-scooter and 36 e-bike real trips from Dublin City University (GPS, altitude, speed, SoC, weather) and compares ML models against physical models, reporting large ML gains.

**→ Used in this project for:** external validation in the **same vehicle class as `src/config.py`'s e-scooter `VehicleSpec`**, and as a second domain at E2-L4.

**⚠️ Two hard cautions.** (1) **30 trips is a validation set, never a training set** — with route-wise grouping there are barely any independent groups. (2) The repository **also contains 10,000 rows generated by the Synthetic Data Vault.** These must be filtered out and never counted as real — exactly the mistake `CLAUDE.md` rule 3 exists to prevent. Their headline "ML beats physics by 83 %" is measured on a tiny dataset with SoC read by OCR from a phone screen; **do not cite it as a benchmark**, cite it as a data source.

### 1.5 ⚙️ NASA PCoE battery ageing data
**B. Saha and K. Goebel, "Battery Data Set," NASA Ames Prognostics Data Repository, NASA Ames Research Center.**

**What it does:** ~34 Li-ion 18650 cells (2 Ah) cycled to failure under charge/discharge/EIS profiles at several temperatures — the most-used SoH benchmark in the field.

**→ Used in this project for:** replacing `generate_battery_cycles()` in `src/data/synthetic.py` with real cells for **E5**. Health indicators (internal-resistance proxy, CC-charge time, time in a voltage window, IC-curve peak, temperature rise) are computed per cycle, and SoH = capacity/initial capacity is the label — with the `CLAUDE.md` rule 2 guarantee that **no feature derives from the cycle's own capacity**.

**⚠️ Where we differ:** most NASA papers report random-split or single-cell results. **We keep leave-one-battery-out**, which is far harder and will look "worse" — and that honesty is the point (risk K5).

### 1.6 ⚙️ Oxford Battery Degradation Dataset
**C. R. Birkl, "Oxford Battery Degradation Dataset 1," University of Oxford Research Archive, 2017.**

**What it does:** 8 Kokam 740 mAh pouch cells aged at 40 °C under the **Artemis urban drive cycle**, with periodic characterisation.

**→ Used in this project for:** **E5b, the cross-dataset transfer test** — train SoH on NASA, test on Oxford without retraining. Different chemistry, different format, different duty cycle. This is the battery-side analogue of E2's distribution-shift ladder, and it is the honest way to ask "does our SoH model generalize?"

**⚠️ Note:** the drive-cycle discharge makes Oxford the closest lab analogue to vehicle use — worth saying in the viva when asked why not only NASA.

### 1.7 ⚙️ Severson et al. — large-scale cycle-life prediction
**K. A. Severson et al., "Data-driven prediction of battery cycle life before capacity degradation," *Nature Energy* 4, 383–391, 2019.** (Related: P. M. Attia et al., *Nature* 578, 2020.)

**What it does:** 124 A123 LFP/graphite cells under 72 fast-charging protocols; predicts cycle life from the **first ~100 cycles** using features from the voltage-capacity curve difference ΔQ(V).

**→ Used in this project for:** (a) the **ΔQ(V) / incremental-capacity feature idea** behind the `ic_peak` and `v_window_min` health indicators already in `src/models/soh_model.py` — this paper is *why* those features exist; (b) **Optional** tier: the only dataset large enough (124 cells) to make a neural SoH model defensible.

**⚠️ Where we differ:** Severson predicts **cycle life** (one number per cell) under **constant** laboratory conditions. We estimate **SoH per cycle under varying conditions**. Different task — do not present their accuracy as a target for ours.

### 1.8 ✅ EVBattery — real vehicle packs, not lab cells
**"EVBattery: A Large-Scale Electric Vehicle Dataset for Battery Health and Capacity Estimation," [arXiv:2201.12358](https://arxiv.org/abs/2201.12358);** data on Figshare.

**What it does:** the first large public **real-world** EV battery dataset — 464 vehicles from 3 manufacturers, >1.2 million charging snippets, with health and capacity labels.

**→ Used in this project for:** **Enhancement tier / RQ4** — the only available way to attack the project's own stated limitation, *"lab-cell SoH does not transfer exactly to a vehicle pack"* (README line 80). It supplies **pack-level SoH from real vehicles.**

**⚠️ Where we differ:** it has charging snippets but **no trips**, so it still cannot give end-to-end trip+SoH validation. Limitation #2 stands regardless.

### 1.9 ✅ Weather — ERA5 via Open-Meteo
**Open-Meteo Historical Weather API** (ERA5 / ERA5-Land reanalysis, hourly from 1940, ~9–25 km) — [open-meteo.com/en/docs/historical-weather-api](https://open-meteo.com/en/docs/historical-weather-api). **Free, no API key, CC-BY 4.0.** Underlying: **Hersbach et al., "The ERA5 global reanalysis," *QJRMS* 146, 2020** ⚙️.

**→ Used in this project for:** joining real temperature, wind speed/direction and precipitation onto every VED trip by **(latitude, longitude, hour)**. Feeds `temp_c`, `headwind_ms`, `rain` in `src/features/trip_features.py` and the air-density term `ρ = P/(R·T)` in the physics model. **This is the single API that makes the real-data plan possible**, because VED ran Nov 2017 – Nov 2018 and we need *history*, not a forecast.

**⚠️ Repository change required:** `src/features/route_pipeline.py` currently calls the **forecast** endpoint (`/v1/forecast`). Historical joins need the **archive** endpoint. Also state the honest caveat: ERA5 is a grid-cell average, and it is *"weather as it turned out"*, not *"the forecast a rider had at departure"*.

### 1.10 ✅ Argonne D3 — controlled physics calibration
**Downloadable Dynamometer Database (D3), Advanced Mobility Technology Laboratory, Argonne National Laboratory** — [anl.gov/taps/downloadable-dynamometer-database](https://www.anl.gov/taps/downloadable-dynamometer-database)

**What it does:** public chassis-dynamometer test data on advanced vehicles including BEVs — elapsed time, speed, wheel force, **battery pack current and voltage**, temperature.

**→ Used in this project for:** **E7, physics-parameter identification.** On a dynamometer, grade = 0 and wind = 0, so Crr and drivetrain efficiency are identifiable without the confounders that plague road data. This is what turns `src/config.py`'s honest `"PLACEHOLDER"` into fitted values with confidence intervals — and satisfies `CLAUDE.md` rule 7.

**⚠️ Where we differ:** dyno data must **never** be used to claim real-world range accuracy. It is a laboratory instrument for parameter fitting only.

---

## 2. The physics baseline

### 2.1 ⚙️ The road-load EV energy model
**C. Fiori, K. Ahn, H. A. Rakha, "Power-based electric vehicle energy consumption model: Model development and validation," *Applied Energy* 168, 2016.**

**What it does:** derives and validates the instantaneous power-based EV consumption model — rolling resistance, aerodynamic drag, grade, inertia, drivetrain efficiency, and a **regenerative-braking efficiency term that depends on deceleration**.

**→ Used in this project for:** this is **the paper behind `src/physics/road_load.py`**. Your `F_roll + F_aero + F_grade`, the `e_batt = e_wheel/η` vs `e_wheel·η_regen` branch (`road_load.py:58`), and the kinetic-energy cost per stop all follow this family of models. **When an examiner asks "where does your physics come from?", this is the answer.**

**⚠️ Where we differ — and this matters for RQ1:** Fiori's model is **instantaneous (per second)**; ours is aggregated to **trip level over segments**, and our `regen_eff` is a **constant 0.40** rather than deceleration-dependent. That deliberate simplification is precisely the gap the ML residual layer is asked to absorb — and it is a *good* viva answer: "we know exactly what our physics layer omits, and we measure what the ML layer recovers."

### 2.2 ⚙️ Drive cycles and EV consumption standards
**Artemis / WLTP / IDC drive-cycle literature** (e.g. André, "The ARTEMIS European driving cycles," *Sci. Total Environ.*, 2004) ⚙️; for the Indian context, **the Indian Drive Cycle (IDC/MIDC)** and recent Indian two-wheeler drive-cycle work (e.g. Pandey et al., "Drive Cycle-Based Estimation of Energy Consumption for Electric Two-Wheeler," *Energy Storage*, 2024 ✅).

**→ Used in this project for:** justifying why manufacturer "rated range" (your `ConstantRated` baseline, `range_model.py:18`) is optimistic — it is measured on a standard cycle, not on a hilly Maharashtra ghat in the rain. **This is the domain argument for why your whole project exists**, and it directly supports the synopsis's problem statement.

---

## 3. Range & consumption prediction — the direct competitors

### 3.1 ✅ Systematic review (read this first — it maps the whole field)
**"Electric Vehicle Range Prediction Models: A Systematic Review of Machine Learning, Mathematical, and Simulation Approaches," *World Electric Vehicle Journal*, vol. 16, no. 11, 607, 2025.** DOI [10.3390/wevj16110607](https://doi.org/10.3390/wevj16110607)

**→ Used in this project for:** the **literature-review chapter** and for positioning. It gives you the three-way taxonomy — mathematical/physics, ML, simulation — into which your hybrid sits *across* two categories. Cite it when you say "hybrid approaches combine the extrapolation of physics models with the flexibility of learned models."

### 3.2 ✅ Uncertainty-aware EV consumption and range (the closest work to RQ2)
**"Machine learning-based uncertainty quantification for energy consumption and driving range estimation in electric cargo vehicles," *Energy Informatics*, 2026.** DOI [10.1186/s42162-026-00682-1](https://doi.org/10.1186/s42162-026-00682-1)

**What it does:** compares uncertainty-aware models for EV energy and range — **SVR conformal prediction, adaptive conformal prediction, and LightGBM conformalized quantile regression (CQR)**.

**→ Used in this project for:** (a) **proof that conformal prediction for EV range is already published** — so §13.1 of the plan can honestly say we do not claim it as novel; (b) the **method menu for E4**: CQR is exactly the principled fix for the 0.73-vs-0.90 quantile defect (B4) in `src/models/uncertainty.py`.

**⚠️ Where we differ — this is RQ2's opening:** work of this kind reports **marginal** coverage. We report **per-subgroup (conditional) coverage under distribution shift** — cold vs. warm, flat vs. hilly — and add **Mondrian conformal** when marginal coverage hides conditional failure. That gap is our contribution 2.

### 3.3 ✅ Remaining range for **two-wheelers** (closest to your vehicle)
**"Real-Time Prediction of Remaining Driving Range for Electric Motorcycle Applications," *IEEE Transactions on Intelligent Transportation Systems*, 2024.** DOI [10.1109/TITS.2024.3445161](https://doi.org/10.1109/TITS.2024.3445161)

**→ Used in this project for:** the **vehicle-class justification**. Nearly all EV range literature is about cars; your `VehicleSpec` is an e-scooter. This paper establishes that light electrified two-wheelers are a legitimate, separately-studied case (different mass, aero, speed envelope and regen behaviour) — important both for the introduction and for answering *"why not just use car results?"*

### 3.4 ⚙️/✅ Data-driven consumption baselines to compare against
- **"Optimizing electric vehicle energy consumption prediction through machine learning and ensemble approaches," *Scientific Reports*, 2025** ✅ — ensemble/stacking on real data; useful as a *strong pure-ML* reference point, and its own stated limitation ("reliance on a single region restricts generalizability") is **literally the motivation for our RQ1**.
- **"Evaluating machine learning algorithms for energy consumption prediction in electric vehicles: A comparative study," *Scientific Reports*, 2025** ✅ — model-comparison protocol to mirror in E1.
- **Modi, Bhattacharya, Basak, "Estimation of energy consumption of electric vehicles using deep convolutional neural network...," [arXiv:2008.11112](https://arxiv.org/abs/2008.11112)** ✅ — cite as the deep-learning alternative **you considered and rejected**, with the reason (trip-level scalar target, 3 vehicles).
- **"Predicting Electric Vehicle Energy Consumption from Field Data," Chalmers University** ✅ — field-data methodology.

**→ Used in this project for:** E1's baseline set, and for the sentence *"our hybrid is compared against a properly-tuned gradient-boosting model, not a straw man."*

---

## 4. Uncertainty quantification — the methods you will implement

### 4.1 ⚙️ Conformal prediction (the foundation)
**V. Vovk, A. Gammerman, G. Shafer, *Algorithmic Learning in a Random World*, Springer, 2005.**
**J. Lei, M. G'Sell, A. Rinaldo, R. J. Tibshirani, L. Wasserman, "Distribution-Free Predictive Inference for Regression," *JASA* 113(523), 2018.**

**What they do:** establish that with a calibration set held out from training, one can build prediction intervals with a **finite-sample, distribution-free marginal coverage guarantee**, assuming only **exchangeability**.

**→ Used in this project for:** this is **the theory behind `src/models/uncertainty.py:9-27`** — the split-conformal class you already have, including the `ceil((n+1)(1−α))/n` quantile level on line 21, which is the finite-sample correction straight out of this literature. **When asked "why is your band 90 % and not just ±2σ?", this is the answer:** no Gaussian assumption is needed.

**⚠️ The assumption to state yourself:** exchangeability is **violated by distribution shift** — which is exactly why RQ2 measures coverage under shift rather than assuming the guarantee holds.

### 4.2 ⚙️ Conformalized Quantile Regression — the fix for your broken intervals
**Y. Romano, E. Patterson, E. J. Candès, "Conformalized Quantile Regression," *NeurIPS* 2019.**

**What it does:** fits quantile regressors, then **conformalizes** their outputs on a calibration set — combining the *adaptive* width of quantile regression with the *guaranteed* coverage of conformal prediction.

**→ Used in this project for:** **E4, and the direct repair of defect B4.** Your `QuantileHybrid` (`uncertainty.py:30-43`) currently under-covers badly (**0.73 vs 0.90**, `reports/uncertainty.csv`) precisely because it is quantile regression *without* conformalization — and because quantiles are fitted on the log-residual then exponentiated. CQR is the textbook fix, and being able to explain *why* your current method fails is stronger than having never tried it.

### 4.3 ⚙️ Mondrian / group-conditional conformal
**V. Vovk, "Conditional validity of inductive conformal predictors," *ACML* 2012.** (Related: Barber, Candès, Ramdas, Tibshirani, "The limits of distribution-free conditional predictive inference," 2020 ⚙️.)

**What it does:** partitions the calibration set into groups and calibrates **within each group**, yielding coverage per group rather than only on average. The companion result proves that *exact* conditional coverage is impossible distribution-free — group-conditional is the achievable compromise.

**→ Used in this project for:** **the core method of RQ2.** Groups = temperature bin × route type. This is what turns *"our interval covers 90 % on average"* into *"our interval covers ≥85 % even on cold hilly trips"* — which is the claim a rider actually needs. Citing the impossibility result too shows you know **why** you chose group-conditional instead of promising full conditional coverage.

---

## 5. Battery State of Health

### 5.1 ✅ Comparative review (the synopsis's own verified reference)
**L. Mbagaya, K. Reddy, A. Botes, "Machine Learning Techniques for Battery State of Health Prediction: A Comparative Review," *World Electric Vehicle Journal*, vol. 16, no. 11, 594, 2025.** DOI [10.3390/wevj16110594](https://doi.org/10.3390/wevj16110594)

**What it does:** reviews SVR, Random Forest, CNN and LSTM for SoH estimation.

**→ Used in this project for:** justifying the model set in `src/models/soh_model.py` (Linear / RF / GBM) and the decision to **start with tree ensembles on hand-crafted health indicators** rather than deep sequence models.

**⚠️ Important honesty point — read this carefully.** Your synopsis §4(2) states *"LSTM consistently outperforms other models."* Reviews of this kind typically compare results reported **across different papers, datasets and splits** — that is not a controlled comparison. **Do not repeat "LSTM is best" as established fact.** The defensible position: *"reviews report strong LSTM results on sequential charging data; with 8–34 cells and leave-one-battery-out validation, a tree ensemble on physically-motivated health indicators is the better-supported choice, and we show our LOBO numbers rather than citing others' random-split numbers."* **That single sentence will impress an examiner more than adding an LSTM would.**

### 5.2 ⚙️ Incremental capacity analysis — where your features come from
**Literature on IC/DV analysis for SoH** (e.g. Weng, Cui, Sun, Peng, *J. Power Sources* 235, 2013; Dubarry & Liaw, ICA methodology) ⚙️; plus **Severson et al. 2019** (§1.7) for ΔQ(V).

**→ Used in this project for:** the physical justification of `SOH_FEATURES` in `src/models/soh_model.py:9` — `ic_peak` (incremental-capacity peak height/position tracks active-material loss), `v_window_min` (time in a voltage window tracks capacity fade), `cc_time_min` (constant-current charge duration shortens as capacity fades), `ir_mohm` (internal-resistance growth tracks power fade). **These are not arbitrary columns — each has an electrochemical rationale, and you should be able to give it.**

**⚠️ The leakage rule this enforces:** none of these may be computed from the cycle's own capacity (`CLAUDE.md` rule 2). The plan adds a unit test that fails if a capacity-derived column enters `SOH_FEATURES`.

### 5.3 ✅ SoH from real vehicles, not lab cells
- **"Driving behavior-guided battery health monitoring for electric vehicles using machine learning," [arXiv:2309.14125](https://arxiv.org/abs/2309.14125)** ✅
- **"Machine learning-based state of health prediction for battery systems in real-world electric vehicles," *Journal of Energy Storage*, 2023** ✅
- **"Multi-modal framework for battery state of health evaluation using open-source electric vehicle data," 2025** ✅

**→ Used in this project for:** evidence for the **Enhancement-tier RQ4** and for the limitation you must state: lab cells behave differently from packs in vehicles. The first paper is especially apt — it links **driving behaviour** to battery health, which is conceptually the bridge your project's title promises.

### 5.4 ⚙️ SoH-coupled range / remaining discharge energy
**Literature decomposing remaining driving range into remaining discharge energy × energy-consumption rate, with particle/Kalman filtering for uncertainty** (several *J. Energy Storage* / *Applied Energy* works, 2017–2026) ⚙️ — verify specific DOIs before citing.

**What it does:** treats range as `RDR = RDE / ECR`, estimates battery state with a particle or extended Kalman filter, and propagates the particle distribution into a **probabilistic range**.

**→ Used in this project for:** (a) confirming that SoH-coupled range estimation **is already a studied problem** — plan §13.1 says so explicitly; (b) it is the conceptual parent of `usable_energy_wh()` in `src/models/coupling.py`.

**⚠️ Where we differ — this is RQ3's opening:** that literature is largely **filtering-based, single-vehicle, and often simulation-only**. We instead take a **measured leave-one-battery-out SoH error distribution from real cells** and propagate it by Monte Carlo, then **decompose the range-error variance** into consumption error vs. SoH error vs. temperature-derate uncertainty — producing the engineering quantity *"km of range error per percentage point of SoH error."* That decomposition, and its honest `COUPLED-SIM` labelling, is contribution 3.

---

## 6. Explainability

### 6.1 ⚙️ SHAP
**S. M. Lundberg, S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," *NIPS* 2017.** (Tree SHAP: Lundberg et al., *Nature Machine Intelligence* 2, 2020 ⚙️.)

**→ Used in this project for:** **E9**, computing per-feature attributions on the hybrid's ML **residual** layer (`HybridResidual.est`).

**⚠️ The subtlety that makes E9 interesting:** SHAP here explains only the *residual* model — it is structurally blind to everything the physics layer already accounts for, and its units are log-ratio, not kilometres. Your existing `advisor.explain()` counterfactual re-runs the **whole** pipeline and returns **kilometres**. E9 measures where the two rankings agree and disagree; the disagreements localise what physics explains versus what ML absorbs. **Saying this out loud demonstrates you understand what SHAP is, which is rarer than using it.**

### 6.2 ✅ SHAP applied to EV energy consumption
**"Improved Prediction of Total Energy Consumption and Feature Analysis in Electric Vehicles Using Machine Learning and Shapley Additive Explanations Method," *World Electric Vehicle Journal*, vol. 12, no. 3, 94, 2021.** DOI [10.3390/wevj12030094](https://doi.org/10.3390/wevj12030094)

**→ Used in this project for:** precedent for SHAP-based feature analysis on EV consumption — and evidence (plan §13.1) that **we do not claim SHAP-for-EV as novel.**

### 6.3 ⚙️ Counterfactual explanation
**S. Wachter, B. Mittelstadt, C. Russell, "Counterfactual Explanations without Opening the Black Box," *Harvard J. Law & Technology* 31(2), 2018.**

**→ Used in this project for:** the **theoretical name and justification for `src/advisor/advisor.py:39-66`**, which you built without knowing it has a literature. Counterfactuals answer *"what would have to be different for the outcome to change?"* — actionable by construction, which is exactly what a range advisor needs. **Cite this and your advisor stops looking ad-hoc and starts looking principled.**

---

## 7. Leakage, validation and reproducibility

### 7.1 ⚙️ Leakage in applied ML
**S. Kapoor, A. Narayanan, "Leakage and the Reproducibility Crisis in ML-based Science," *Patterns* 4(9), 2023.**

**What it does:** documents leakage across hundreds of papers in many fields and proposes model-info sheets for reproducible ML-based science.

**→ Used in this project for:** the **framing of experiment E0** and of plan §10.1's leakage register. This is the citation that turns *"we split by route"* into *"we adopted a documented protocol against a known, field-wide failure mode, and quantified what it would have cost us."* **E0 — showing your own random-split number next to your grouped-split number — is the single most persuasive slide you can put in front of an examiner, and this paper is why.**

### 7.2 ⚙️ Grouped and shifted evaluation
**Standard grouped cross-validation practice** (Kohavi 1995 ⚙️; scikit-learn's `GroupKFold` documentation), plus the **distribution-shift / domain-generalization** literature (e.g. Koh et al., "WILDS: A Benchmark of in-the-Wild Distribution Shifts," *ICML* 2021 ⚙️).

**→ Used in this project for:** WILDS is the **conceptual template for E2's shift ladder** — the idea that a benchmark should be organised *by shift type* (route → season → vehicle → vehicle class), with in-distribution and out-of-distribution numbers reported side by side, and a **degradation ratio** as the headline. We are, in effect, building a small WILDS-style benchmark for EV energy consumption. **That framing is your strongest answer to "what is new?"**

---

## 8. How to use this file

**In the report's Literature Review:** §3.1 (review) → §2.1 (physics) → §3.2–3.4 (ML competitors) → §5.1–5.3 (SoH) → §4 (uncertainty) → §6 (explainability) → §7 (validity). Then a paragraph stating exactly what is *not* novel and what is (plan §13).

**In the viva:** for any *"why did you...?"* question, the answer form is
> *"Because [paper] showed [finding]; we used [specific thing] in [specific file/experiment]; we differ in [specific way], because [our data/constraint]."*

That third clause — **how you differ** — is what separates a student who read the papers from one who cited them.

**Before submission:** open every ⚙️ entry on the publisher's site and confirm authors, venue, volume, pages, year and DOI. **Delete synopsis reference [1].** Anything you cannot open, you do not cite.
