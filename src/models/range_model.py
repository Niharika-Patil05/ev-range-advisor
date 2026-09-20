"""Range (energy-consumption) models: baselines + the physics-informed hybrid.

All models predict consumption in Wh/km. Range = usable energy / consumption.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import VehicleSpec
from ..features.trip_features import ALL_FEATURES, ML_ONLY_FEATURES


class ConstantRated:
    """Industry baseline: the manufacturer's rated Wh/km, regardless of conditions."""
    def __init__(self, wh_per_km: float):
        self.wh_per_km = wh_per_km

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.full(len(X), self.wh_per_km)


class PhysicsOnly:
    """Road-load model with nominal parameters, no learning."""
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return X["phys_wh_km"].to_numpy()


class SklearnRegressor:
    """Wraps an sklearn estimator so it selects its own feature columns from a DataFrame."""
    def __init__(self, estimator, features):
        self.estimator, self.features = estimator, list(features)

    def fit(self, X, y):
        self.estimator.fit(X[self.features], np.asarray(y))
        return self

    def predict(self, X):
        return self.estimator.predict(X[self.features])


def _hgb(seed=0, **kw):
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
        random_state=seed, **kw)


class HybridResidual:
    """Physics + ML: predict = phys_wh_km * exp(f(features)), where f is learned on log(y / phys).

    The physics layer supplies the bulk of the energy estimate; ML only corrects what the physics
    misses (tyre/wet-road effects, riding style, battery losses, calibration error).
    `quantile` (0-1) trains a quantile version instead of the mean.
    """
    def __init__(self, features=ALL_FEATURES, seed: int = 0, quantile: float | None = None):
        self.features = list(features)
        if "phys_wh_km" not in self.features:
            self.features.append("phys_wh_km")
        kw = dict(loss="quantile", quantile=quantile) if quantile is not None else {}
        self.est = _hgb(seed, **kw)

    def fit(self, X, y):
        self.est.fit(X[self.features], np.log(np.asarray(y) / X["phys_wh_km"].to_numpy()))
        return self

    def predict(self, X):
        return X["phys_wh_km"].to_numpy() * np.exp(self.est.predict(X[self.features]))


def model_factories(vehicle: VehicleSpec, seed: int = 0) -> dict:
    """name -> zero-argument constructor. Pure-ML baselines deliberately do NOT see phys_wh_km."""
    return {
        "rated_range": lambda: ConstantRated(vehicle.rated_wh_per_km),
        "physics_only": lambda: PhysicsOnly(),
        "linear": lambda: SklearnRegressor(make_pipeline(StandardScaler(), LinearRegression()), ML_ONLY_FEATURES),
        "random_forest": lambda: SklearnRegressor(
            RandomForestRegressor(300, min_samples_leaf=3, n_jobs=-1, random_state=seed), ML_ONLY_FEATURES),
        "gbm": lambda: SklearnRegressor(_hgb(seed), ML_ONLY_FEATURES),
        "hybrid_physics_ml": lambda: HybridResidual(ALL_FEATURES, seed=seed),
    }
