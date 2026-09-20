"""Prediction intervals for consumption (and therefore range)."""
from __future__ import annotations

import numpy as np

from .range_model import HybridResidual


class ConformalInterval:
    """Split-conformal interval on *relative* error: y in [yhat*(1-q), yhat*(1+q)].

    Fit on a calibration set of routes the model was NOT trained on.
    """
    def __init__(self, alpha: float = 0.1):
        self.alpha, self.q_ = alpha, None

    def fit(self, y_true, y_pred):
        y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
        scores = np.abs(y_true - y_pred) / y_pred
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q_ = float(min(np.quantile(scores, level, method="higher"), 0.95))
        return self

    def interval(self, y_pred):
        y = np.asarray(y_pred, float)
        return y * (1 - self.q_), y * (1 + self.q_)


class QuantileHybrid:
    """Alternative: two quantile-regression hybrids (alpha/2 and 1-alpha/2)."""
    def __init__(self, features, alpha: float = 0.1, seed: int = 0):
        self.lo = HybridResidual(features, seed=seed, quantile=alpha / 2)
        self.hi = HybridResidual(features, seed=seed, quantile=1 - alpha / 2)

    def fit(self, X, y):
        self.lo.fit(X, y)
        self.hi.fit(X, y)
        return self

    def interval(self, X):
        a, b = self.lo.predict(X), self.hi.predict(X)
        return np.minimum(a, b), np.maximum(a, b)


def coverage(y_true, lo, hi) -> float:
    y_true = np.asarray(y_true)
    return float(np.mean((y_true >= lo) & (y_true <= hi)))
