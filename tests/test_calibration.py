"""Tests for E7 parameter identification.

The basis reduction is an optimisation: it must reproduce the full trace
integration exactly for the parameters its mask was built with, or every fitted
number downstream is meaningless.
"""
import numpy as np
import pandas as pd
import pytest

from src.calibration.identify import fit, predict_energy_wh, segment_basis
from src.config import NISSAN_LEAF_2013 as LEAF
from src.physics.road_load import trace_energy_wh

MASS = 1573.0


def trace(n=601, v_ms=15.0, grade=0.0, aux=250.0, seed=None):
    t = np.arange(float(n))
    if seed is not None:
        rng = np.random.default_rng(seed)
        v = np.clip(v_ms + rng.normal(0, 4.0, n), 0.5, 35.0)
        g = rng.normal(0, 0.01, n)
    else:
        v, g = np.full(n, v_ms), np.full(n, grade)
    return t, v, g, np.full(n, aux)


def test_basis_reproduces_full_trace_integration():
    t, v, g, aux = trace(seed=1)
    b = segment_basis(t, v, g, mass_kg=MASS, vehicle=LEAF, aux_power_w=aux)
    got = predict_energy_wh(pd.DataFrame([b]), LEAF.crr,
                            LEAF.cd * LEAF.frontal_area_m2,
                            LEAF.drivetrain_eff, LEAF.regen_eff)[0]
    want = trace_energy_wh(t, v, g, mass_kg=MASS, vehicle=LEAF, aux_power_w=aux)["total_wh"]
    assert got == pytest.approx(want, rel=1e-9)


def test_basis_reproduces_trace_on_a_flat_steady_run():
    t, v, g, aux = trace()
    b = segment_basis(t, v, g, mass_kg=MASS, vehicle=LEAF, aux_power_w=aux)
    got = predict_energy_wh(pd.DataFrame([b]), LEAF.crr,
                            LEAF.cd * LEAF.frontal_area_m2,
                            LEAF.drivetrain_eff, LEAF.regen_eff)[0]
    want = trace_energy_wh(t, v, g, mass_kg=MASS, vehicle=LEAF, aux_power_w=aux)["total_wh"]
    assert got == pytest.approx(want, rel=1e-9)


def test_energy_increases_with_rolling_resistance():
    t, v, g, aux = trace(seed=2)
    b = pd.DataFrame([segment_basis(t, v, g, mass_kg=MASS, vehicle=LEAF, aux_power_w=aux)])
    lo = predict_energy_wh(b, 0.008, 0.66, 0.89, LEAF.regen_eff)[0]
    hi = predict_energy_wh(b, 0.014, 0.66, 0.89, LEAF.regen_eff)[0]
    assert hi > lo


def test_energy_decreases_with_drivetrain_efficiency():
    t, v, g, aux = trace(seed=3)
    b = pd.DataFrame([segment_basis(t, v, g, mass_kg=MASS, vehicle=LEAF, aux_power_w=aux)])
    assert (predict_energy_wh(b, 0.009, 0.66, 0.75, LEAF.regen_eff)[0]
            > predict_energy_wh(b, 0.009, 0.66, 0.95, LEAF.regen_eff)[0])


def test_fit_recovers_known_parameters_from_clean_synthetic_energy():
    """Identifiability check on REAL-shaped traces with KNOWN parameters.

    Legitimate use of simulation: the truth is a parameter value we chose, which no
    real dataset can provide. It tests the ESTIMATOR, not the physics.
    """
    rows, truth_crr, truth_eta = [], 0.0115, 0.845
    for s in range(60):
        t, v, g, aux = trace(n=401, seed=s)
        rows.append(segment_basis(t, v, g, mass_kg=MASS, vehicle=LEAF, aux_power_w=aux))
    b = pd.DataFrame(rows)
    y = predict_energy_wh(b, truth_crr, LEAF.cd * LEAF.frontal_area_m2,
                          truth_eta, LEAF.regen_eff)
    got = fit(b, y, vehicle=LEAF)
    assert got["crr"] == pytest.approx(truth_crr, rel=0.05)
    assert got["drivetrain_eff"] == pytest.approx(truth_eta, rel=0.05)
    assert got["mape"] < 0.5


def _fit_over_seeds(sigma=0.08, n_seed=8, loss_name="abs_rel"):
    """Fit the same known truth under several independent noise draws."""
    rows = [segment_basis(*trace(n=401, seed=s)[:3], mass_kg=MASS, vehicle=LEAF,
                          aux_power_w=trace(n=401, seed=s)[3]) for s in range(60)]
    b = pd.DataFrame(rows)
    clean = predict_energy_wh(b, 0.0115, 0.66, 0.845, LEAF.regen_eff)
    out = []
    for k in range(n_seed):
        noisy = clean * np.random.default_rng(100 + k).lognormal(0, sigma, len(clean))
        out.append(fit(b, noisy, vehicle=LEAF, loss_name=loss_name))
    return pd.DataFrame(out)


NOISE_LEVELS = (0.05, 0.10, 0.15)


def test_drivetrain_efficiency_is_identifiable_under_noise():
    """eta scales every tractive term, including the inertia term that dominates
    urban segments, so the loss surface is sharp in eta and the estimate is stable
    at EVERY noise level tested."""
    for sigma in NOISE_LEVELS:
        r = _fit_over_seeds(sigma=sigma)
        assert (r.drivetrain_eff - 0.845).abs().max() < 0.10, f"sigma={sigma}"
        assert r.drivetrain_eff.std() < 0.05, f"sigma={sigma}"


def test_rolling_resistance_is_NOT_reliably_identifiable():
    """The central E7 finding.

    crr scales only the rolling term (~144 Wh of a ~480 Wh segment), so the surface
    is shallow in crr. Across independent noise draws the estimate is unstable and
    often runs to its bound, and the failure is not even monotonic in noise level --
    the signature of a shallow, multi-modal surface. Reporting a confident crr from
    this data would be wrong, which is why E7 recommends holding it at its datasheet
    value and fitting eta alone.
    """
    spreads = {}
    for sigma in NOISE_LEVELS:
        r = _fit_over_seeds(sigma=sigma)
        spreads[sigma] = (r.crr.max() - r.crr.min()) / 0.0115
    # Unstable at most noise levels, and NOT monotonic in noise -- at some sigma the
    # optimiser happens to land well, which is exactly why a single run must not be
    # taken as evidence that crr was identified.
    assert max(spreads.values()) > 0.5, f"crr looked stable everywhere: {spreads}"
    assert sum(v > 0.5 for v in spreads.values()) >= 2, f"spreads={spreads}"


def test_the_degeneracy_is_a_diagonal_valley():
    """Raising crr and eta together leaves tractive energy nearly unchanged."""
    rows = [segment_basis(*trace(n=401, seed=s)[:3], mass_kg=MASS, vehicle=LEAF,
                          aux_power_w=trace(n=401, seed=s)[3]) for s in range(20)]
    b = pd.DataFrame(rows)
    ref = predict_energy_wh(b, 0.0115, 0.66, 0.845, LEAF.regen_eff)

    def rel(crr, eta):
        p = predict_energy_wh(b, crr, 0.66, eta, LEAF.regen_eff)
        return float(np.mean(np.abs(p - ref) / ref))

    # moving along the valley (both up) hurts far less than moving across it
    along = rel(0.0160, 0.880)
    across = rel(0.0160, 0.810)
    assert along < across
