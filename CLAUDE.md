# Project: Route-aware, uncertainty-aware EV range prediction + SoH-coupled battery advisor

Final-year project, Dept. of Electrical Engineering, ADCET Ashta. Team: 3 AI&DS + 2 Civil students.
Guide: Dr. N. Vengadachalam. Read README.md first.

## Architecture (one line)
route segments -> physics road-load baseline -> ML residual (log-ratio) -> consumption Wh/km
-> range = usable energy (nominal x SoH x temp derate x SoC) / consumption -> conformal band
-> advisor (counterfactual explanations + rules) -> Streamlit app.

## Non-negotiable rules
1. NEVER split data by random rows. Split by `route_id` (or vehicle/battery). Battery models: leave-one-battery-out.
2. NEVER use a cycle's own capacity (or anything derived from the SoH label) as an input feature.
3. Real vs synthetic data must always be labelled. Synthetic results are never reported as real accuracy.
4. Real logged trips are for VALIDATION; do not mix them into training without saying so in the report.
5. Physics module (`src/physics/road_load.py`) stays pure and unit-tested; change it only with tests.
6. Every experiment: fixed seed, results written to `reports/`. Don't hand-edit numbers.
7. Vehicle constants live only in `src/config.py` (currently PLACEHOLDERS).
8. Do not wire or modify a live battery/vehicle; hardware work needs guide/lab approval.

## Commands
- `python scripts/run_pipeline.py [--quick]`  train + evaluate + write reports/ and models/
- `pytest -q`                                  run all tests (~40 s)
- `streamlit run app/streamlit_app.py`         run the app

## Where things are
src/physics (road-load) | src/features (trip features, route pipeline) | src/data (synthetic + real loaders)
src/models (range, SoH, uncertainty, coupling) | src/advisor | src/pipeline.py | app/ | tests/

## Style
Type hints, short docstrings that explain WHY, no magic numbers outside config/constants,
small functions, tests for anything numeric. Explain non-obvious code so students can defend it in the viva.
