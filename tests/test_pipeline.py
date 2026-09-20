import numpy as np
import pytest

from src.advisor.advisor import Advisor
from src.features.route_pipeline import PRESET_ROUTES, load_preset
from src.pipeline import train_all


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("out")
    return train_all(n_routes=30, trips_per_route=10, save=True, models_dir=tmp / "m",
                     reports_dir=tmp / "r", make_plots=True, cv_folds=3, verbose=False)


def test_hybrid_beats_baselines(results):
    rr = results["range_results"]["MAPE"]
    assert rr["hybrid_physics_ml"] < rr["physics_only"] < rr["rated_range"]


def test_conformal_coverage_reasonable(results):
    cov = results["uncertainty"].loc["split conformal", "coverage"]
    assert 0.75 <= cov <= 1.0


def test_soh_lobo_mae_small(results):
    assert results["soh"]["MAE_mean"].min() < 3.0  # percentage points


def test_advisor_physical_sanity(results):
    adv = Advisor(results["system"])
    seg = load_preset(list(PRESET_ROUTES)[1])
    base = dict(soc_start=90, soh=0.9, load_kg=80, style=1, aux_on=0, temp_c=30, headwind_ms=0, rain=0)
    r0 = adv.predict(seg, base)["range_km"]
    assert adv.predict(seg, {**base, "headwind_ms": 6})["range_km"] < r0       # headwind hurts
    assert adv.predict(seg, {**base, "soc_start": 45})["range_km"] < r0        # less charge, less range
    assert adv.predict(seg, {**base, "soh": 0.7})["range_km"] < r0             # aged battery, less range
    lo, p50, hi = (adv.predict(seg, base)[k] for k in ("range_lo_km", "range_km", "range_hi_km"))
    assert lo < p50 < hi
    assert any(e["factor"] == "Battery ageing (SoH)" for e in adv.explain(seg, base))
    climb = seg.assign(grade=0.03)  # sustained net climb must show up as a range-reducing factor
    assert any(e["factor"] == "Hills / gradient" for e in adv.explain(climb, base))
    assert adv.advise(seg, base)
