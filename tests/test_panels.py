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
