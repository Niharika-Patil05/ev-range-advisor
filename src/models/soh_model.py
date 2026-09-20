"""Battery State-of-Health estimation with leave-one-battery-out validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

SOH_FEATURES = ["ir_mohm", "cc_time_min", "v_window_min", "ic_peak", "temp_mean_c", "temp_rise_c"]


def _models(seed=0):
    return {
        "cycle_only_linear": (LinearRegression(), ["cycle"]),
        "random_forest": (RandomForestRegressor(200, min_samples_leaf=5, n_jobs=-1, random_state=seed), SOH_FEATURES),
        "gbm": (HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=seed), SOH_FEATURES),
    }


def lobo_evaluate(df: pd.DataFrame, seed: int = 0):
    """Leave-one-battery-out: each cell is predicted by a model that never saw it.
    Returns (per-model summary DataFrame, out-of-fold predictions DataFrame)."""
    cells = df["battery_id"].unique()
    recs, oof = [], []
    for name, (_, feats) in _models(seed).items():
        for cell in cells:
            train, test = df[df.battery_id != cell], df[df.battery_id == cell]
            est = _models(seed)[name][0]
            est.fit(train[feats], train["soh"])
            pred = est.predict(test[feats])
            recs.append(dict(model=name, battery_id=cell, MAE=float(np.mean(np.abs(pred - test["soh"])))))
            oof.append(test[["battery_id", "cycle", "soh"]].assign(model=name, pred=pred))
    per_cell = pd.DataFrame(recs)
    summary = per_cell.groupby("model")["MAE"].agg(["mean", "std", "max"]).rename(
        columns={"mean": "MAE_mean", "std": "MAE_std", "max": "MAE_worst_cell"})
    summary[["MAE_mean", "MAE_std", "MAE_worst_cell"]] *= 100  # in SoH percentage points
    return summary.sort_values("MAE_mean"), pd.concat(oof, ignore_index=True)


def fit_final(df: pd.DataFrame, seed: int = 0):
    est, feats = _models(seed)["random_forest"]
    est.fit(df[feats], df["soh"])
    return est
