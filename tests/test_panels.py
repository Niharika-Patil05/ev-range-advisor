import json

import numpy as np
import pandas as pd
import pytest

from app.panels import (PROVENANCE_BADGE, list_experiments, load_summary,
                        soh_sensitivity_curve, temperature_sensitivity_curve)
from src.config import NISSAN_LEAF_2013 as LEAF


def test_every_provenance_tag_has_a_badge():
    from src.evaluation.provenance import Provenance
    for p in Provenance:
        assert p.value in PROVENANCE_BADGE


def test_coupled_sim_badge_says_it_is_not_validation():
    _, text = PROVENANCE_BADGE["COUPLED-SIM"]
    assert "NOT end-to-end validation" in text


def test_list_experiments_reads_manifests(tmp_path):
    d = tmp_path / "E_demo"
    d.mkdir()
    (d / "MANIFEST.json").write_text(json.dumps(
        {"experiment": "E_demo", "provenance": "REAL", "conclusion_type": "DEMONSTRATED",
         "seeds": [0, 1, 2], "n": "10 segments", "split_protocol": "L1"}))
    out = list_experiments(tmp_path)
    assert len(out) == 1 and out.provenance.iloc[0] == "REAL" and out.seeds.iloc[0] == 3


def test_list_experiments_skips_unreadable_manifests(tmp_path):
    d = tmp_path / "E_bad"
    d.mkdir()
    (d / "MANIFEST.json").write_text("{not json")
    assert list_experiments(tmp_path).empty


def test_missing_summary_returns_none(tmp_path):
    assert load_summary("nope", tmp_path) is None


def test_soh_curve_slope_matches_the_e6_identity():
    """Range is proportional to SoH, so km-per-pp is range/100 at that SoH."""
    c = soh_sensitivity_curve(LEAF, consumption_wh_km=146.3, soc_pct=100.0, temp_c=20.0)
    row = c.iloc[len(c) // 2]
    assert row.km_per_pp == pytest.approx(row.range_km / row.soh_pct, rel=1e-6)
    assert c.km_per_pp.std() < 1e-6          # constant slope: the curve is a line


def test_soh_curve_halves_with_state_of_charge():
    full = soh_sensitivity_curve(LEAF, 146.3, 100.0, 20.0)
    half = soh_sensitivity_curve(LEAF, 146.3, 50.0, 20.0)
    assert half.km_per_pp.mean() == pytest.approx(full.km_per_pp.mean() / 2, rel=1e-6)


def test_temperature_curve_is_flat_above_the_derate_threshold():
    t = temperature_sensitivity_curve(LEAF, 146.3, 100.0, 0.9)
    warm = t[t.temp_c >= 20]
    assert warm.range_km.std() < 1e-9
    assert t[t.temp_c == -10].range_km.iloc[0] < warm.range_km.iloc[0]


# --------------------------------------------------------- degradation projection
def test_projection_returns_a_band_not_a_line():
    from src.models.coupling import project_soh
    p = project_soh(1.0, n_cycles=100)
    assert {"cycle", "central", "optimistic", "pessimistic"} <= set(p.columns)
    last = p.iloc[-1]
    assert last.pessimistic < last.central < last.optimistic


def test_projection_is_in_lab_cycles_not_calendar_months():
    """Extrapolating the measured lab fade rate over calendar time predicts a dead
    pack in two years, which every real EV contradicts. The x axis is cycles, and
    the conversion to months is deliberately not offered."""
    import inspect

    from src.models.coupling import project_soh
    params = set(inspect.signature(project_soh).parameters)
    assert "n_cycles" in params
    assert "months" not in params and "cycles_per_week" not in params


def test_lab_cells_reach_end_of_life_far_faster_than_a_vehicle_pack():
    """~90 lab cycles to 80 % SoH, against ~1000 for a vehicle pack. Documenting the
    gap is the point: it is why no calendar forecast is offered."""
    from src.models.coupling import cycles_to_threshold
    n = cycles_to_threshold(1.0, 0.80)
    assert 70 < n["central"] < 120
    assert n["pessimistic"] < n["central"] < n["optimistic"]


def test_projection_uses_the_measured_fade_rate_not_the_old_placeholder():
    """The shipped placeholder was 0.00025 per cycle; E8 measured 0.00222, nine times
    larger, so the old projection understated degradation badly."""
    from src.models.coupling import FADE_PER_CYCLE, project_soh
    assert FADE_PER_CYCLE == pytest.approx(0.00222)
    p = project_soh(1.0, n_cycles=50, step=1)
    assert p.central.iloc[-1] == pytest.approx(1.0 - FADE_PER_CYCLE * 50, rel=1e-9)


def test_projection_no_longer_accepts_uncalibratable_factors():
    """temperature, depth-of-discharge and charge-cap arguments are gone: E8 showed
    none of them can be calibrated from the data this project holds."""
    import inspect

    from src.models.coupling import project_soh
    params = set(inspect.signature(project_soh).parameters)
    assert params == {"soh0", "n_cycles", "step"}
    for gone in ("charge_cap_pct", "avg_dod", "avg_temp_c"):
        assert gone not in params


def test_advisor_no_longer_gives_uncalibrated_charging_advice():
    import inspect

    from src.advisor import advisor as adv
    src = inspect.getsource(adv)
    assert "charge to 80" not in src.lower()
    assert "project_soh" not in src.split('"""')[0] + src  # not imported or called


def test_every_experiment_directory_carries_a_manifest():
    """The README promises every experiment writes a MANIFEST.json, and the evidence
    panel discovers experiments by scanning for them. Three non-seed experiments
    (a parameter fit, a propagation study, a calibration) previously wrote results
    without one and were invisible in the app."""
    from pathlib import Path

    from src.config import REPORTS_DIR
    missing = [d.name for d in Path(REPORTS_DIR).glob("E*/")
               if d.is_dir() and not (d / "MANIFEST.json").exists()]
    assert not missing, f"experiments without a manifest: {missing}"


def test_write_manifest_records_provenance_and_refuses_bad_claims(tmp_path):
    from src.evaluation.harness import write_manifest
    from src.evaluation.provenance import (ConclusionType, Evidence, Provenance,
                                           ProvenanceError)
    ev = Evidence(experiment="E_x", provenance=Provenance.COUPLED_SIM,
                  conclusion_type=ConclusionType.SIMULATED, n="10")
    m = write_manifest("E_x", "d", tmp_path, ev)
    assert m["provenance"] == "COUPLED-SIM"
    assert (tmp_path / "MANIFEST.json").exists() and (tmp_path / "EVIDENCE.md").exists()

    with pytest.raises(ProvenanceError):
        write_manifest("E_y", "d", tmp_path,
                       Evidence(experiment="E_y", provenance=Provenance.SYNTHETIC,
                                conclusion_type=ConclusionType.SIMULATED, n="10")
                       .__class__(experiment="E_y", provenance=Provenance.SYNTHETIC,
                                  conclusion_type=ConclusionType.DEMONSTRATED, n="10"))
