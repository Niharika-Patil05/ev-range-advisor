"""Advisor: range band, 'why' explanations (counterfactuals) and actionable tips."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import VehicleSpec
from ..features.route_to_features import route_to_features
from ..features.trip_features import DEFAULT_CONDITIONS, summarize_trip
from ..models.coupling import temp_derate, usable_energy_wh


class Advisor:
    """`system` is the dict produced by src.pipeline.train_all()['system']."""

    def __init__(self, system: dict, vehicle: VehicleSpec):
        self.hybrid = system["hybrid"]
        self.conformal = system["conformal"]
        self.vehicle = vehicle
        self.meta = system.get("meta", {})
        # A system trained on real segments carries its feature list; the legacy
        # synthetic system does not. The feature schemas differ, so the row builder
        # is chosen from the artefact rather than assumed.
        self.features = system.get("features")
        self.is_real = self.meta.get("data") == "REAL"

    def _feature_row(self, segments: pd.DataFrame, c: dict) -> dict:
        if self.features is not None:
            return route_to_features(segments, c, self.vehicle)
        return summarize_trip(segments, c, self.vehicle)

    # ------------------------------------------------------------------ prediction
    def predict(self, segments: pd.DataFrame, cond: dict) -> dict:
        c = {**DEFAULT_CONDITIONS, **cond}
        feats = self._feature_row(segments, c)
        wh_km = float(self.hybrid.predict(pd.DataFrame([feats]))[0])
        lo, hi = (float(x[0]) for x in self.conformal.interval(np.array([wh_km])))
        usable = float(usable_energy_wh(c["soc_start"], c["soh"], c["temp_c"], self.vehicle))
        full = float(usable_energy_wh(100.0, c["soh"], c["temp_c"], self.vehicle))
        dist = feats["distance_km"]
        energy = wh_km * dist
        return dict(
            distance_km=dist, wh_per_km=wh_km, wh_per_km_lo=lo, wh_per_km_hi=hi,
            usable_wh=usable, trip_energy_wh=energy,
            range_km=usable / wh_km, range_lo_km=usable / hi, range_hi_km=usable / lo,
            soc_end_pct=c["soc_start"] - energy / full * 100.0,
            phys_wh_km=feats["phys_wh_km"], features=feats,
        )

    # ------------------------------------------------------------------ explanations
    def explain(self, segments: pd.DataFrame, cond: dict, min_km: float = 0.5) -> list[dict]:
        """Counterfactual attribution: 'how many km would I gain if this factor were neutral?'
        Dependency-free stand-in for SHAP that also works through the physics layer."""
        c = {**DEFAULT_CONDITIONS, **cond}
        base = self.predict(segments, c)["range_km"]
        cases = []
        if self.features is not None:
            # Real-data schema: the counterfactuals are over the quantities this
            # model actually consumes.
            if c.get("aux_power_w", 0.0) > 10:
                cases.append(("Heating / cooling", segments, {**c, "aux_power_w": 0.0}))
            if c["temp_c"] < 15:
                cases.append(("Cold weather", segments, {**c, "temp_c": 25.0}))
            if c["soh"] < 0.98:
                cases.append(("Battery ageing (SoH)", segments, {**c, "soh": 1.0}))
            if (segments["length_m"] * segments["grade"].abs()).sum() > 50:
                cases.append(("Hills / gradient", segments.assign(grade=0.0), c))
            if "stops" in segments and segments["stops"].sum() > 0:
                cases.append(("Stop-start traffic", segments.assign(stops=0.0), c))
        else:
            if c["headwind_ms"] > 0.5:
                cases.append(("Headwind", segments, {**c, "headwind_ms": 0.0}))
            if c["temp_c"] < 15:
                cases.append(("Cold weather", segments, {**c, "temp_c": 25.0}))
            if c["rain"]:
                cases.append(("Rain / wet road", segments, {**c, "rain": 0}))
            if c["aux_on"]:
                cases.append(("High auxiliary load", segments, {**c, "aux_on": 0}))
            if c["style"] > 0:
                cases.append(("Riding style (vs. eco)", segments, {**c, "style": 0}))
            if c["soh"] < 0.98:
                cases.append(("Battery ageing (SoH)", segments, {**c, "soh": 1.0}))
            if c["load_kg"] > 80:
                cases.append(("Extra load (above 80 kg)", segments, {**c, "load_kg": 80.0}))
            if (segments["length_m"] * segments["grade"].abs()).sum() > 50:
                cases.append(("Hills / gradient", segments.assign(grade=0.0), c))
        out = []
        for label, seg, cc in cases:
            gain = self.predict(seg, cc)["range_km"] - base
            if gain >= min_km:
                out.append(dict(factor=label, km_lost=gain))
        return sorted(out, key=lambda d: -d["km_lost"])

    # ------------------------------------------------------------------ advice
    def advise(self, segments: pd.DataFrame, cond: dict) -> list[dict]:
        c = {**DEFAULT_CONDITIONS, **cond}
        p = self.predict(segments, c)
        d, msgs = p["distance_km"], []

        if p["range_km"] < d:
            msgs.append(dict(level="error", text=(
                f"Expected range ({p['range_km']:.0f} km) is shorter than the trip ({d:.0f} km). "
                f"Charge first or plan a charging stop.")))
        elif p["range_lo_km"] < d:
            msgs.append(dict(level="warning", text=(
                f"Tight: the trip is {d:.0f} km but the pessimistic range is {p['range_lo_km']:.0f} km. "
                f"You would likely make it, but with no safety margin.")))
        else:
            msgs.append(dict(level="success", text=(
                f"Comfortable: even the pessimistic range ({p['range_lo_km']:.0f} km) covers the "
                f"{d:.0f} km trip. Expected battery at arrival: {p['soc_end_pct']:.0f} %.")))

        if p["range_lo_km"] < d * 1.15 and c.get("aux_power_w", 0.0) > 10:
            alt = self.predict(segments, {**c, "aux_power_w": 0.0})
            gain = alt["range_km"] - p["range_km"]
            if gain >= 0.5:
                msgs.append(dict(level="info", text=(
                    f"Turning off heating or cooling would add about {gain:.0f} km.")))
        if p["range_lo_km"] < d * 1.15 and c.get("style", 0) > 0:
            alt = self.predict(segments, {**c, "style": 0})
            gain = alt["range_km"] - p["range_km"]
            if gain >= 0.5:
                msgs.append(dict(level="info", text=f"Riding in eco mode would add about {gain:.0f} km of range."))
        if p["range_lo_km"] < d * 1.15 and c["aux_on"]:
            alt = self.predict(segments, {**c, "aux_on": 0})
            gain = alt["range_km"] - p["range_km"]
            if gain >= 0.5:
                msgs.append(dict(level="info", text=f"Switching off auxiliary loads would add about {gain:.1f} km."))

        if c["soh"] < 0.80:
            msgs.append(dict(level="warning", text=(
                f"Battery health is {c['soh']*100:.0f} %, typically below the ~80 % end-of-life "
                f"threshold. Plan a battery check or replacement.")))
        # The charging-habit advice that used to live here ("limiting daily charge
        # to 80 % could preserve roughly X percentage points of SoH") has been
        # REMOVED. It was computed from an invented charge-cap factor, and
        # experiment E8 established that NASA PCoE records no charge-limit variable,
        # so the claim could not be calibrated from any data this project holds. It
        # was the only place where a placeholder produced user-facing advice.
        return msgs
