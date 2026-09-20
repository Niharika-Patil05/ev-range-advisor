# Route-Aware, Uncertainty-Aware EV Range Prediction with an SoH-Coupled Battery Advisor

Final-year project scaffold (Electrical Engineering, ADCET Ashta). It runs **end to end today on
synthetic data**, so every module can be tested, and each synthetic piece has a clearly marked slot
where your real data goes.

```
route (A->B) -> segments (length, grade, speed, stops) --+
weather (temp, wind, rain) ------------------------------+--> physics road-load baseline (Wh/km)
                                                          +--> ML corrects log(actual/physics)
battery SoH -> usable energy = nominal x SoH x temp derate x SoC
range = usable energy / predicted Wh/km  --> 90 % conformal band --> advisor (why + what to do) --> Streamlit app
```

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_pipeline.py        # trains, evaluates, writes reports/ + models/  (~1 min)
pytest -q                             # 14 tests
streamlit run app/streamlit_app.py    # the app (trains quickly by itself if models/ is empty)
```

## What is implemented (and how far it is tested)
| Module | File | Status |
|---|---|---|
| Road-load physics baseline | `src/physics/road_load.py` | Implemented, unit-tested vs hand calculation |
| Trip features | `src/features/trip_features.py` | Implemented, tested |
| Baselines: rated range, physics-only, linear, RF, GBM | `src/models/range_model.py` | Implemented, evaluated with route-wise splits |
| Hybrid physics + ML | `src/models/range_model.py` | Implemented, evaluated |
| Uncertainty: split conformal + quantile regression | `src/models/uncertainty.py` | Implemented; coverage measured on unseen routes |
| SoH estimation (leave-one-battery-out) | `src/models/soh_model.py` | Implemented on synthetic cells |
| SoH <-> range coupling, what-if simulator | `src/models/coupling.py` | Implemented; **what-if constants are placeholders** |
| Advisor: counterfactual explanations + rules | `src/advisor/advisor.py` | Implemented, tested |
| Geometry (resample, grade, headwind) | `src/features/route_pipeline.py` | Implemented, tested offline |
| Online route/elevation/weather (OSRM, Open-Meteo) | `src/features/route_pipeline.py` | **Written but NOT tested** (no internet in build sandbox). Test it first. |
| Streamlit app | `app/streamlit_app.py` | Runs; smoke-tested with Streamlit's AppTest |
| LSTM/GRU, SHAP, osmnx road types, real-data logger | - | **Not included** (roadmap below) |

## Read this before quoting any result
The numbers in `reports/` come from **synthetic** data that we generated from our own assumptions
(`src/data/synthetic.py`). The hybrid model looks excellent (~3 % MAPE) partly *by construction*: the
hidden "truth" is built from the same physics plus a few effects the ML layer can learn, with 3 % noise.
It proves the code and evaluation work. It says nothing about real accuracy.
Your project's real contribution starts when you replace the synthetic data.

## Plugging in real data (in this order)
1. **Vehicle** - edit `src/config.py` with your vehicle's mass, battery, Crr/Cd/A (measure or datasheet).
2. **Battery / SoH** - build a CSV with the columns in `src/data/loaders.py` (`BATTERY_COLUMNS`) from
   NASA / CALCE / Oxford cycles and use `load_battery_csv`. Replace `generate_battery_cycles` in
   `src/pipeline.py`. Verify dataset links and licences first.
3. **Trips** - for each logged or public trip create a segment table and call `summarize_trip`;
   add the measured `wh_per_km`, `route_id`, `trip_id`. Replace `generate_trip_dataset` with `load_trip_csv`.
   Keep `route_id` meaningful (route, vehicle, or day) because all splits group on it.
4. **Calibrate physics** - fit `crr` and `drivetrain_eff` on measured flat trips *before* training the ML layer.
5. **Calibrate the what-if** - fit the constants in `project_soh` to your degradation data.
6. **Test the online route pipeline** on a machine with internet, then compare the resulting gradients
   with a known route.
7. Update the `meta["data"]` label in `src/pipeline.py` from `SYNTHETIC` to your data source.

## Suggested validation to report
Baselines vs hybrid (already produced), the ablation (`reports/ablation.csv`), interval coverage
(`reports/uncertainty.csv`; conformal should land near 90 %, note quantile regression under-covers here),
SoH leave-one-battery-out MAE, then the same tables on your **real logged trips**.

## Team map
Sharyu: `soh_model.py`, `coupling.py` | Megha: `range_model.py`, experiments | Niharika: `uncertainty.py`,
`advisor.py`, `app/` | Viraj: `route_pipeline.py` (elevation, road data) | Atharva: `config.py` + physics
parameters, field logging, validation. Everyone: tests, report, and understanding the whole pipeline.

## Roadmap / stretch
- Add LSTM/GRU for SoH and for sequence-level range (`torch`); compare against RF/GBM.
- SHAP for the ML residual layer (counterfactual explanations already work through the physics layer).
- osmnx for road type and speed limits; per-segment speed profiles instead of one average speed.
- Real logger: ESP32 + hall-effect current sensor + GPS or the BMS UART (approved by the lab; never
  the INA219 on a scooter pack). Use logged trips for validation only.
- Deploy the app (Streamlit Community Cloud) for the demo, and record a backup demo video.

## Honest limitations (put these in your report)
Lab-cell SoH does not transfer exactly to a vehicle pack; SRTM elevation is ~90 m resolution;
the app uses one average speed per route unless you add per-segment speeds; OSRM's demo server
is for light use only; temperature derating and degradation constants are rules of thumb until calibrated.
