"""Streamlit front-end.  Run:  streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

from app.panels import (PROVENANCE_BADGE, list_experiments, load_findings,
                        load_summary, soh_sensitivity_curve,
                        temperature_sensitivity_curve)
from src.advisor.advisor import Advisor
from src.config import E_SCOOTER_PLACEHOLDER, NISSAN_LEAF_2013
from src.features.route_pipeline import PRESET_ROUTES, RoutePlanningError, load_preset, plan_route
from src.models.coupling import cycles_to_threshold, project_soh
from src.models.soh_model import SOH_FEATURES
from src.pipeline import load_or_train_system

st.set_page_config(page_title="EV Range & Battery Advisor", page_icon="🔋", layout="wide")


# The vehicle is chosen first and passed explicitly everywhere (Phase 0, item C1).
# Nothing falls back to a module-level default, so the app can never silently
# serve a model fitted to a different vehicle's road-load parameters.
VEHICLES = {
    "2013 Nissan Leaf (primary research vehicle)": NISSAN_LEAF_2013,
    "Generic e-scooter (demo only)": E_SCOOTER_PLACEHOLDER,
}


@st.cache_resource(show_spinner="Loading models (the first run trains them on synthetic data)...")
def get_system(_vehicle):
    return load_or_train_system(_vehicle)


with st.sidebar:
    st.header("Vehicle")
    vehicle = VEHICLES[st.selectbox("Vehicle", list(VEHICLES))]

system = get_system(vehicle)
advisor = Advisor(system, vehicle)

st.title("🔋 Route-aware EV Range & Battery Health Advisor")
st.caption(f"Vehicle: {vehicle.name}  |  Model data: {system['meta']['data']}")
if system["meta"]["data"] == "SYNTHETIC":
    st.warning("Models are currently trained on SYNTHETIC data. Numbers demonstrate the workflow, "
               "not real-world accuracy. Replace with real data (see README).")

# ------------------------------------------------------------------ sidebar inputs
with st.sidebar:
    st.header("Trip")
    source = st.radio("Route source", ["Demo route (offline)", "Custom coordinates (needs internet)"])
    segments, weather, elevation_profile = None, {}, None
    if source.startswith("Demo"):
        name = st.selectbox("Route", list(PRESET_ROUTES))
        segments = load_preset(name, vehicle)
    else:
        o = st.text_input("Origin lat, lon", "16.8524, 74.5815")
        d = st.text_input("Destination lat, lon", "16.7050, 74.2433")
        traffic = st.select_slider("Traffic", ["light", "moderate", "heavy"], "moderate")
        use_weather = st.checkbox("Use live weather for the origin", True)
        if st.button("Plan route"):
            try:
                origin = tuple(float(x) for x in o.split(","))
                dest = tuple(float(x) for x in d.split(","))
                seg, wx, meta = plan_route(origin, dest, vehicle,
                                           {"light": 0.3, "moderate": 1.0, "heavy": 2.5}[traffic])
                st.session_state["planned"] = (seg, wx if use_weather else {})
            except (RoutePlanningError, ValueError) as exc:
                st.error(f"Could not plan route: {exc}. Use a demo route instead.")
        if "planned" in st.session_state:
            segments, weather = st.session_state["planned"]

    st.header("Vehicle & conditions")
    soc = st.slider("Battery now (%)", 5, 100, 90)
    soh = st.slider("Battery health, SoH (%)", 60, 100, 90) / 100
    load = st.slider("Rider + load (kg)", 40, 180, 80)
    style = st.select_slider("Riding style", ["eco", "normal", "sporty"], "normal")
    aux = st.checkbox("High auxiliary load (lights, accessories)")
    temp = st.slider("Temperature (°C)", 0, 45, int(round(weather.get("temp_c", 30))))
    wind = st.slider("Headwind (m/s, negative = tailwind)", -8.0, 8.0, float(round(weather.get("headwind_ms", 0.0), 1)))
    rain = st.checkbox("Rain / wet road", bool(weather.get("rain", 0)))

cond = dict(soc_start=soc, soh=soh, load_kg=load, style=["eco", "normal", "sporty"].index(style),
            aux_on=int(aux), temp_c=temp, headwind_ms=wind, rain=int(rain))

tab_trip, tab_battery, tab_evidence, tab_sens = st.tabs(
    ["Trip advisor", "Battery health", "Evidence", "Sensitivity"])

# ------------------------------------------------------------------ trip tab
with tab_trip:
    if segments is None:
        st.info("Choose a demo route or plan a custom route in the sidebar.")
    else:
        p = advisor.predict(segments, cond)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Expected range", f"{p['range_km']:.0f} km")
        c2.metric("90% range band", f"{p['range_lo_km']:.0f} – {p['range_hi_km']:.0f} km")
        c3.metric("Trip distance", f"{p['distance_km']:.1f} km")
        c4.metric("Battery at arrival", f"{max(p['soc_end_pct'], 0):.0f} %")

        for m in advisor.advise(segments, cond):
            getattr(st, m["level"])(m["text"])

        left, right = st.columns(2)
        with left:
            st.subheader("Elevation profile")
            dist_km = np.cumsum(segments["length_m"]) / 1000
            elev = np.cumsum(segments["length_m"] * segments["grade"])
            st.line_chart(pd.DataFrame({"distance (km)": dist_km, "elevation change (m)": elev}).set_index("distance (km)"))
        with right:
            st.subheader("Why is my range what it is?")
            why = advisor.explain(segments, cond)
            if why:
                st.bar_chart(pd.DataFrame(why).set_index("factor")["km_lost"])
                st.caption("Kilometres of range you would regain if the factor were neutral.")
            else:
                st.write("No major range-reducing factors detected.")
        with st.expander("Model details"):
            st.write(f"Predicted consumption: **{p['wh_per_km']:.1f} Wh/km** "
                     f"(physics-only baseline: {p['phys_wh_km']:.1f} Wh/km, "
                     f"rated: {vehicle.rated_wh_per_km:.1f} Wh/km).")
            st.write(f"Usable energy now: {p['usable_wh']:.0f} Wh of {vehicle.nominal_energy_wh:.0f} Wh nominal.")

# ------------------------------------------------------------------ battery tab
with tab_battery:
    st.subheader("Projected battery health (lab cells)")
    st.caption("Fade measured on 26 NASA lab cells (experiment E8), shown as a band "
               "because the cell-to-cell spread is 80 % of the mean.")
    n_cycles = st.slider("Lab cycles to project", 20, 400, 200, step=10)
    proj = project_soh(soh, n_cycles).set_index("cycle") * 100
    st.line_chart(proj[["pessimistic", "central", "optimistic"]])
    eol = cycles_to_threshold(soh, 0.80)
    if np.isfinite(eol["central"]):
        st.metric("Lab cycles to 80 % SoH (central)", f"{eol['central']:.0f}",
                  help=f"Band across cells: {eol['pessimistic']:.0f} to "
                       f"{eol['optimistic']:.0f} cycles.")
    st.error(
        "**Read the x axis: these are LAB CYCLES, not months, and that is deliberate.**\n\n"
        "These NASA cells are cycled at full depth of discharge and high rate, reaching "
        "80 % SoH in roughly 90 cycles. A vehicle pack doing shallow partial cycles "
        "typically takes on the order of a thousand. Extrapolating this rate over "
        "calendar time — five charges a week for two years is 521 cycles — would "
        "predict a dead pack in two years, which every real EV contradicts. The "
        "equivalence factor between a lab cycle and a vehicle cycle is of order ten "
        "and **nothing in this project measures it**, so no calendar forecast is offered.\n\n"
        "**Charging-habit advice has been removed.** Earlier versions claimed that "
        "limiting daily charge to 80 % preserves a specific number of SoH points. That "
        "came from an invented constant. E8 found NASA PCoE records no charge-limit or "
        "depth-of-discharge variable, and no temperature factor is identifiable either "
        "(p = 0.087, with the sign the wrong way round for an Arrhenius law).")

    with st.expander("Estimate SoH from cycle measurements"):
        defaults = dict(ir_mohm=32.0, cc_time_min=90.0, v_window_min=21.0, ic_peak=1.5, temp_mean_c=30.0, temp_rise_c=4.0)
        cols = st.columns(3)
        vals = {f: cols[i % 3].number_input(f, value=defaults[f]) for i, f in enumerate(SOH_FEATURES)}
        if st.button("Estimate SoH"):
            est = float(system["soh_model"].predict(pd.DataFrame([vals])[SOH_FEATURES])[0])
            st.success(f"Estimated SoH: {est*100:.1f} %  (set the sidebar slider to use it)")


# ------------------------------------------------------------------ evidence tab
with tab_evidence:
    st.subheader("What has actually been measured")
    st.caption("Rendered live from reports/. Every experiment writes a MANIFEST.json "
               "recording its seeds, split protocol and code revision, and an "
               "EVIDENCE.md stating what was measured, derived, external, simulated "
               "and assumed.")

    exps = list_experiments()
    if exps.empty:
        st.info("No experiment reports found. Run e.g. "
                "`python -m src.experiments.e1_model_comparison`.")
    else:
        for tag, (icon, meaning) in PROVENANCE_BADGE.items():
            if (exps.provenance == tag).any():
                st.markdown(f"{icon} **{tag}** — {meaning}")
        st.dataframe(exps[["experiment", "provenance", "conclusion", "protocol",
                           "n", "seeds"]], width='stretch', hide_index=True)

        chosen = st.selectbox("Experiment", sorted(Path(e).name for e in exps.path))
        summary = load_summary(chosen)
        if summary is not None:
            cols = [c for c in summary.columns if c.endswith(("_mean", "_std"))
                    or not c.endswith("_count")]
            st.dataframe(summary[cols].round(3), width='stretch', hide_index=True)
        findings = load_findings(chosen)
        if findings:
            with st.expander("Findings, including what was refuted", expanded=False):
                st.markdown(findings)

    scorecard = Path("reports/HYPOTHESIS_SCORECARD.md")
    if scorecard.exists():
        with st.expander("Pre-registration scorecard (15 hypotheses, 6 refuted)"):
            st.markdown(scorecard.read_text())

# ------------------------------------------------------------------ sensitivity tab
with tab_sens:
    st.subheader("What-if: how much does each factor move the range?")
    p_now = advisor.predict(segments, cond) if segments is not None else None
    cons = (p_now["wh_per_km"] if p_now else 146.3)
    st.caption(f"Using {cons:.0f} Wh/km"
               + (" from the current route." if p_now else
                  " (median measured on real VED segments) — pick a route for your own."))

    st.markdown("#### Range against battery health")
    st.markdown("🔶 **COUPLED-SIM** — " + PROVENANCE_BADGE["COUPLED-SIM"][1])
    curve = soh_sensitivity_curve(vehicle, cons, soc, temp)
    st.line_chart(curve.set_index("soh_pct")["range_km"])
    slope = float(curve.km_per_pp.mean())
    st.metric("Range lost per percentage point of SoH", f"{slope:.2f} km")
    st.caption(
        f"Experiment E6 measured {slope:.2f} km per point at this state of charge, and "
        f"found it stable across the 80–100 % band. It halves when the battery is half "
        f"full, because range is proportional to usable energy. E6 also found that the "
        f"**consumption model contributes 71.9 % of range-prediction variance against "
        f"26.2 % for SoH estimation** — improving consumption prediction matters about "
        f"three times more.")

    st.markdown("#### Range against temperature (capacity derate only)")
    tcurve = temperature_sensitivity_curve(vehicle, cons, soc, soh)
    st.line_chart(tcurve.set_index("temp_c")["range_km"])
    st.warning(
        "This shows the **capacity derate term alone**, whose coefficient is still a "
        "placeholder. It does NOT include temperature's effect on consumption itself, "
        "which the real data says is the larger channel: E6 attributes only 1.9 % of "
        "range variance to the derate, while E3 measured temperature and the measured "
        "HVAC load it drives as the two strongest single features after route geometry.")
