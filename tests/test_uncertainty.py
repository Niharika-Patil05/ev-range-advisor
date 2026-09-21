"""Tests for interval methods (RQ2).

The point of conformal prediction is a coverage GUARANTEE, so these tests check
coverage on synthetic data where the true distribution is known -- a legitimate use
of simulation, since it tests the method rather than making a claim about EVs.
"""
import numpy as np
import pandas as pd
import pytest

from src.models.uncertainty import (ConformalInterval, MondrianConformal,
                                    _conformal_quantile, coverage, coverage_by_group)


def test_conformal_quantile_uses_the_finite_sample_correction():
    s = np.arange(100.0) / 100
    # ceil(101*0.9)/100 = 0.91, not 0.90
    assert _conformal_quantile(s, 0.1) == pytest.approx(0.91)


def test_conformal_quantile_degrades_gracefully_when_too_few_points():
    """With n so small that the required level exceeds 1, no finite quantile can
    guarantee coverage; return the max rather than silently under-covering."""
    s = np.array([0.1, 0.2, 0.3])
    assert _conformal_quantile(s, 0.01) == 0.3


def test_split_conformal_achieves_marginal_coverage():
    rng = np.random.default_rng(0)
    y_cal, yhat_cal = rng.lognormal(5, 0.2, 500), np.full(500, np.exp(5))
    c = ConformalInterval(alpha=0.1).fit(y_cal, yhat_cal)
    y_te, yhat_te = rng.lognormal(5, 0.2, 2000), np.full(2000, np.exp(5))
    lo, hi = c.interval(yhat_te)
    assert 0.85 <= coverage(y_te, lo, hi) <= 0.95


def test_mondrian_fixes_a_group_that_split_conformal_under_covers():
    """The RQ2 motivation, as a test.

    One group is much noisier than the other. A single pooled quantile splits the
    difference and under-covers the noisy group; per-group quantiles do not.
    """
    rng = np.random.default_rng(1)
    n = 1500
    g = np.where(rng.random(n) < 0.25, "cold", "warm")
    scale = np.where(g == "cold", 0.45, 0.06)
    yhat = np.full(n, 100.0)
    y = yhat * (1 + rng.normal(0, scale))

    half = n // 2
    cal, te = slice(0, half), slice(half, n)

    pooled = ConformalInterval(alpha=0.1).fit(y[cal], yhat[cal])
    lo, hi = pooled.interval(yhat[te])
    pooled_cov = coverage_by_group(y[te], lo, hi, g[te]).set_index("group")

    mond = MondrianConformal(alpha=0.1).fit(y[cal], yhat[cal], g[cal])
    mlo, mhi = mond.interval(yhat[te], g[te])
    mond_cov = coverage_by_group(y[te], mlo, mhi, g[te]).set_index("group")

    # pooled under-covers the noisy group ...
    assert pooled_cov.loc["cold", "coverage"] < 0.85
    # ... and Mondrian repairs it
    assert mond_cov.loc["cold", "coverage"] > pooled_cov.loc["cold", "coverage"]
    assert mond_cov.loc["cold", "coverage"] >= 0.82
    # at the cost of a wider band for that group
    assert mond_cov.loc["cold", "mean_rel_width"] > pooled_cov.loc["cold", "mean_rel_width"]
    # and a TIGHTER band for the quiet group, which pooled calibration over-covered
    assert mond_cov.loc["warm", "mean_rel_width"] < pooled_cov.loc["warm", "mean_rel_width"]


def test_mondrian_falls_back_and_records_small_groups():
    rng = np.random.default_rng(2)
    g = np.array(["big"] * 200 + ["tiny"] * 5)
    yhat = np.full(len(g), 100.0)
    y = yhat * (1 + rng.normal(0, 0.1, len(g)))
    m = MondrianConformal(alpha=0.1, min_per_group=20).fit(y, yhat, g)
    assert "big" in m.q_by_group_
    assert "tiny" not in m.q_by_group_
    assert m.small_groups_ == [("tiny", 5)]
    lo, hi = m.interval(yhat, g)          # tiny falls back to the pooled quantile
    assert np.isfinite(lo).all() and np.isfinite(hi).all()


def test_coverage_by_group_reports_width_alongside_coverage():
    y = np.array([100.0, 100.0, 100.0, 100.0])
    lo, hi = np.array([90.0] * 4), np.array([110.0] * 4)
    out = coverage_by_group(y, lo, hi, np.array(["a", "a", "b", "b"]))
    assert set(out.group) == {"a", "b"}
    assert (out.coverage == 1.0).all()
    assert out.mean_rel_width.iloc[0] == pytest.approx(0.2)
