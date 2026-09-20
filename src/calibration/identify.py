"""Fit vehicle road-load parameters to measured data (experiment E7).

Why this exists
---------------
`src/config.py` ships datasheet values for the Leaf, all tagged as assumptions. The
physics-only baseline built from them is 22.5% MAPE with a -4.3% bias, so some of
that error is simply wrong constants rather than missing physics. E7 asks how much
of it fitted parameters can absorb, and -- just as importantly -- which parameters
the data can identify at all.

How the fit is made tractable
-----------------------------
Re-integrating every trace for every parameter guess would be slow, so each segment
is reduced once to a set of BASIS INTEGRALS. Holding the tractive/regenerative mask
fixed (it is set by inertia and grade, which the fitted parameters barely move),
battery energy is linear in the parameters:

    E(crr, cdA, eta) = (1/eta) * (crr*R_t + cdA*A_t + G_t + I_t)
                     + eta_regen * (crr*R_r + cdA*A_r + G_r + I_r)
                     + E_aux

with R, A, G, I the rolling, aero, grade and inertia basis integrals accumulated
separately over tractive (_t) and regenerative (_r) intervals. This is exact given
the mask, and it turns a slow simulation into arithmetic.

The identifiability problem, stated up front
--------------------------------------------
Scaling crr and cdA up by k while scaling eta up by k leaves the tractive term
unchanged. Drivetrain efficiency is therefore NOT separately identifiable from the
resistance coefficients using total energy alone. Two weak levers break the
degeneracy -- the regenerative branch multiplies by eta_regen instead of 1/eta, and
auxiliary load bypasses the drivetrain entirely -- but they are weak. The fit
reports the degeneracy rather than hiding it behind a confident-looking number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import VehicleSpec
from ..physics.road_load import G0, air_density

BASIS_COLUMNS = ("roll_t", "aero_t", "grade_t", "inert_t",
                 "roll_r", "aero_r", "grade_r", "inert_r", "aux_wh")


def segment_basis(time_s, speed_ms, grade, *, mass_kg: float, vehicle: VehicleSpec,
                  temp_c: float = 25.0, aux_power_w=None) -> dict:
    """Reduce one trace to basis integrals, in Wh at the wheel (aux already at the pack).

    The tractive mask is evaluated once with the vehicle's current parameters.
    """
    t = np.asarray(time_s, float)
    v = np.asarray(speed_ms, float)
    g = np.broadcast_to(np.asarray(grade, float), v.shape)
    dt = np.diff(t)
    v0, theta = v[:-1], np.arctan(g[:-1])
    accel = np.diff(v) / dt
    rho = air_density(temp_c)

    roll = mass_kg * G0 * np.cos(theta) * v0 * dt / 3600.0          # x crr
    aero = 0.5 * rho * v0 * np.abs(v0) * v0 * dt / 3600.0           # x cd*A
    grad = mass_kg * G0 * np.sin(theta) * v0 * dt / 3600.0
    inert = mass_kg * accel * v0 * dt / 3600.0

    # Mask from the NET wheel power under current parameters.
    p = (mass_kg * G0 * vehicle.crr * np.cos(theta)
         + 0.5 * rho * vehicle.cd * vehicle.frontal_area_m2 * v0 * np.abs(v0)
         + mass_kg * G0 * np.sin(theta) + mass_kg * accel) * v0
    tract = p >= 0

    aux = (np.full_like(v0, vehicle.aux_power_w) if aux_power_w is None
           else np.asarray(aux_power_w, float)[:len(v0)])
    out = {}
    for name, arr in (("roll", roll), ("aero", aero), ("grade", grad), ("inert", inert)):
        out[f"{name}_t"] = float(np.nansum(arr[tract]))
        out[f"{name}_r"] = float(np.nansum(arr[~tract]))
    out["aux_wh"] = float(np.nansum(aux * dt) / 3600.0)
    return out


def predict_energy_wh(basis: pd.DataFrame, crr: float, cd_a: float,
                      eta: float, regen_eff: float) -> np.ndarray:
    """Battery energy from basis integrals and parameters."""
    tract = crr * basis["roll_t"] + cd_a * basis["aero_t"] + basis["grade_t"] + basis["inert_t"]
    regen = crr * basis["roll_r"] + cd_a * basis["aero_r"] + basis["grade_r"] + basis["inert_r"]
    return (tract / eta + regen * regen_eff + basis["aux_wh"]).to_numpy(float)


def fit(basis: pd.DataFrame, measured_wh, *, vehicle: VehicleSpec,
        fit_cd_a: bool = False, bounds: dict | None = None,
        loss_name: str = "abs_rel") -> dict:
    """Fit crr (and optionally cd*A) and eta against measured energy.

    All losses are RELATIVE, so long segments do not dominate short ones.

      abs_rel  mean |pred-y|/y   -- the default: robust, and it is the quantity we
                                   actually report, so the fit optimises what is scored
      sq_rel   mean ((pred-y)/y)^2 -- outlier-dominated; on this data it trades a worse
                                   mean error for a better fit to a few large segments
      log_sq   mean (log pred - log y)^2 -- scale-symmetric, gives the least biased fit

    The choice matters less than it looks: on the real VED segments all three land
    within 0.5 percentage points of MAPE of each other and of the nominal parameters.
    """
    from scipy.optimize import minimize

    y = np.asarray(measured_wh, float)
    b = bounds or {}
    crr_lo, crr_hi = b.get("crr", (0.004, 0.025))
    eta_lo, eta_hi = b.get("eta", (0.55, 0.98))
    cda_lo, cda_hi = b.get("cd_a", (0.3, 1.5))
    cd_a_nominal = vehicle.cd * vehicle.frontal_area_m2

    def unpack(x):
        return (x[0], x[2] if fit_cd_a else cd_a_nominal, x[1])

    def loss(x):
        crr, cd_a, eta = unpack(x)
        pred = predict_energy_wh(basis, crr, cd_a, eta, vehicle.regen_eff)
        if np.any(pred <= 0):
            return 1e6
        if loss_name == "sq_rel":
            return float(np.mean(((pred - y) / y) ** 2))
        if loss_name == "log_sq":
            return float(np.mean((np.log(pred) - np.log(y)) ** 2))
        if loss_name == "abs_rel":
            return float(np.mean(np.abs(pred - y) / y))
        raise ValueError(f"unknown loss_name {loss_name!r}")

    x0 = [vehicle.crr, vehicle.drivetrain_eff] + ([cd_a_nominal] if fit_cd_a else [])
    bnds = [(crr_lo, crr_hi), (eta_lo, eta_hi)] + ([(cda_lo, cda_hi)] if fit_cd_a else [])
    res = minimize(loss, x0, bounds=bnds, method="L-BFGS-B")
    crr, cd_a, eta = unpack(res.x)
    pred = predict_energy_wh(basis, crr, cd_a, eta, vehicle.regen_eff)
    return dict(crr=float(crr), cd_a=float(cd_a), drivetrain_eff=float(eta),
                loss_name=loss_name, loss=float(res.fun),
                mape=float(np.mean(np.abs(pred - y) / y) * 100),
                bias=float(np.mean((pred - y) / y) * 100),
                converged=bool(res.success))


def bootstrap_fit(basis: pd.DataFrame, measured_wh, groups, *, vehicle: VehicleSpec,
                  n_boot: int = 200, seed: int = 0, fit_cd_a: bool = False,
                  loss_name: str = "abs_rel") -> pd.DataFrame:
    """Resample whole ROUTES, not rows, so the interval respects the grouping.

    Resampling rows would treat repeats of one route as independent evidence and
    produce confidence intervals that are far too narrow.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(measured_wh, float)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rows = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in pick])
        try:
            rows.append(fit(basis.iloc[idx], y[idx], vehicle=vehicle, fit_cd_a=fit_cd_a,
                            loss_name=loss_name))
        except Exception:
            continue
    return pd.DataFrame(rows)
