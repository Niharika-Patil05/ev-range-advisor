"""Streamlit front-end.  Run:  streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

from src.advisor.advisor import Advisor
from src.config import E_SCOOTER_PLACEHOLDER, NISSAN_LEAF_2013
from src.features.route_pipeline import PRESET_ROUTES, RoutePlanningError, load_preset, plan_route
from src.models.coupling import project_soh
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

tab_trip, tab_battery = st.tabs(["Trip advisor", "Battery health"])

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
    st.subheader("What-if: charging habits (indicative trends)")
    a, b, c = st.columns(3)
    cpw = a.slider("Charge cycles per week", 1, 14, 5)
    dod = b.slider("Average depth of discharge", 0.2, 1.0, 0.7)
    months = c.slider("Months to project", 6, 60, 24)
    proj = pd.DataFrame({
        "charge to 100 %": project_soh(soh, months, cpw, 100, dod, temp) * 100,
        "charge to 80 %": project_soh(soh, months, cpw, 80, dod, temp) * 100,
    })
    proj.index.name = "month"
    st.line_chart(proj)
    st.caption("Semi-empirical model with placeholder constants; calibrate on lab-cell data before quoting numbers. "
               "Lab cells are not vehicle packs.")

    with st.expander("Estimate SoH from cycle measurements"):
        defaults = dict(ir_mohm=32.0, cc_time_min=90.0, v_window_min=21.0, ic_peak=1.5, temp_mean_c=30.0, temp_rise_c=4.0)
        cols = st.columns(3)
        vals = {f: cols[i % 3].number_input(f, value=defaults[f]) for i, f in enumerate(SOH_FEATURES)}
        if st.button("Estimate SoH"):
            est = float(system["soh_model"].predict(pd.DataFrame([vals])[SOH_FEATURES])[0])
            st.success(f"Estimated SoH: {est*100:.1f} %  (set the sidebar slider to use it)")
