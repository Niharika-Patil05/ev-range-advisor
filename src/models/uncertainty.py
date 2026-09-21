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


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile: ceil((n+1)(1-alpha))/n of the scores.

    The (n+1) correction is what makes the coverage guarantee hold at finite n
    rather than only asymptotically. With too few points the level exceeds 1 and no
    finite quantile can guarantee coverage; we return the maximum score and the
    caller is expected to report that the group was too small.
    """
    scores = np.asarray(scores, float)
    n = len(scores)
    if n == 0:
        return float("inf")
    level = np.ceil((n + 1) * (1 - alpha)) / n
    if level > 1.0:
        return float(np.max(scores))
    return float(np.quantile(scores, level, method="higher"))


class ConformalizedQuantile:
    """CQR (Romano, Patterson & Candes, NeurIPS 2019).

    Plain quantile regression gives intervals that ADAPT to the input but carry no
    coverage guarantee, and in this project's own measurements it under-covered
    badly (0.73 against a 0.90 target). CQR keeps the adaptive width and restores
    the guarantee by conformalising the quantile outputs on a held-out calibration
    set:

        score_i = max(lo(x_i) - y_i,  y_i - hi(x_i))
        interval = [lo(x) - q,  hi(x) + q]

    A positive score means the point fell outside the predicted band, a negative one
    means it fell comfortably inside, so `q` can widen OR tighten the band.
    """

    def __init__(self, features, alpha: float = 0.1, seed: int = 0):
        self.alpha = alpha
        self.lo = HybridResidual(features, seed=seed, quantile=alpha / 2)
        self.hi = HybridResidual(features, seed=seed, quantile=1 - alpha / 2)
        self.q_ = None

    def fit(self, X, y):
        self.lo.fit(X, y)
        self.hi.fit(X, y)
        return self

    def calibrate(self, X_cal, y_cal):
        lo, hi = self.lo.predict(X_cal), self.hi.predict(X_cal)
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        scores = np.maximum(lo - np.asarray(y_cal, float), np.asarray(y_cal, float) - hi)
        self.q_ = _conformal_quantile(scores, self.alpha)
        return self

    def interval(self, X):
        if self.q_ is None:
            raise RuntimeError("CQR must be calibrated on a held-out set before use.")
        lo, hi = self.lo.predict(X), self.hi.predict(X)
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        return lo - self.q_, hi + self.q_


class MondrianConformal:
    """Group-conditional (Mondrian) conformal prediction (Vovk, ACML 2012).

    Split conformal guarantees MARGINAL coverage: 90% of predictions are covered on
    average. That is the wrong guarantee for range anxiety, because a rider is
    stranded by a band that fails on a cold evening, not by one that fails on
    average. Mondrian calibrates a separate quantile WITHIN each group, so coverage
    holds per group instead.

    Exact conditional coverage is impossible distribution-free (Barber et al.), so
    group-conditional is the achievable compromise -- which is why the groups are
    chosen in advance rather than discovered.

    Groups with too few calibration points fall back to the pooled quantile, and
    `small_groups_` records which, because silently using a meaningless group
    quantile would manufacture a guarantee that is not there.
    """

    def __init__(self, alpha: float = 0.1, min_per_group: int = 20):
        self.alpha, self.min_per_group = alpha, min_per_group
        self.q_by_group_: dict = {}
        self.q_pooled_ = None
        self.small_groups_: list = []

    def fit(self, y_true, y_pred, groups):
        y_true = np.asarray(y_true, float)
        y_pred = np.asarray(y_pred, float)
        groups = np.asarray(groups)
        scores = np.abs(y_true - y_pred) / y_pred          # relative, as in ConformalInterval
        self.q_pooled_ = _conformal_quantile(scores, self.alpha)
        self.q_by_group_, self.small_groups_ = {}, []
        for g in np.unique(groups):
            m = groups == g
            if m.sum() < self.min_per_group:
                self.small_groups_.append((g, int(m.sum())))
                continue
            self.q_by_group_[g] = _conformal_quantile(scores[m], self.alpha)
        return self

    def interval(self, y_pred, groups):
        y = np.asarray(y_pred, float)
        q = np.array([self.q_by_group_.get(g, self.q_pooled_) for g in np.asarray(groups)])
        return y * (1 - q), y * (1 + q)


def coverage_by_group(y_true, lo, hi, groups) -> "pd.DataFrame":
    """Per-group coverage and mean relative width.

    Coverage without width is meaningless -- an interval of +/-infinity covers
    everything -- so both are always reported together.
    """
    import pandas as pd

    y_true = np.asarray(y_true, float)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    mid = 0.5 * (lo + hi)
    df = pd.DataFrame({"group": np.asarray(groups),
                       "covered": (y_true >= lo) & (y_true <= hi),
                       "rel_width": (hi - lo) / np.where(mid == 0, np.nan, mid)})
    out = df.groupby("group").agg(n=("covered", "size"),
                                  coverage=("covered", "mean"),
                                  mean_rel_width=("rel_width", "mean"))
    return out.reset_index()
