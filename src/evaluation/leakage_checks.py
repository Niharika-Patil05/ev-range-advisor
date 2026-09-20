"""Executable leakage checks.

CLAUDE.md rules 1 and 2 are stated in prose; here they are enforced. Experiments call
these before reporting, so a leak fails the run rather than quietly inflating a score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Substrings that betray a feature computed from the label. Using a cycle's own
# capacity (or anything derived from it) to predict that cycle's SoH is circular:
# SoH IS capacity / initial capacity, so such a feature leaks the answer.
CAPACITY_DERIVED_PATTERNS = (
    "capacity", "soh", "ah_discharged", "ah_charged", "energy_discharged",
    "q_discharge", "q_charge", "remaining_useful", "rul", "fade",
)


class LeakageError(AssertionError):
    """Raised when a split or a feature set would leak."""


def assert_disjoint_groups(train: pd.DataFrame, test: pd.DataFrame, key: str,
                           *, label: str = "") -> None:
    """Fail if any group appears on both sides of a split."""
    shared = set(train[key].unique()) & set(test[key].unique())
    if shared:
        example = sorted(map(str, shared))[:5]
        raise LeakageError(
            f"{label or 'split'}: {len(shared)} value(s) of '{key}' appear in BOTH "
            f"train and test, e.g. {example}. Grouped splitting has failed."
        )


def assert_three_way_disjoint(train: pd.DataFrame, calibration: pd.DataFrame,
                              test: pd.DataFrame, key: str = "route_id") -> None:
    """Conformal calibration must share no group with train or test.

    A calibration set overlapping train gives optimistically small residuals and so
    an interval that is too narrow; overlapping test invalidates the guarantee.
    """
    assert_disjoint_groups(train, test, key, label="train/test")
    assert_disjoint_groups(calibration, test, key, label="calibration/test")
    assert_disjoint_groups(train, calibration, key, label="train/calibration")


def assert_no_capacity_derived_features(features: list[str], *,
                                        allow: tuple[str, ...] = ()) -> None:
    """Enforce CLAUDE.md rule 2 for the SoH model.

    `allow` exists for the deliberate exception: `soh` is a legitimate INPUT to the
    range model (usable energy scales with it), while being the forbidden label for
    the battery model. The caller states which context it is in.
    """
    offenders = [
        f for f in features
        if f not in allow and any(p in f.lower() for p in CAPACITY_DERIVED_PATTERNS)
    ]
    if offenders:
        raise LeakageError(
            f"Feature(s) {offenders} look derived from the SoH label (capacity). "
            f"Using a cycle's own capacity to predict its SoH is circular "
            f"(CLAUDE.md rule 2). Add to `allow` only with a written justification."
        )


def assert_no_temporal_leak(train: pd.DataFrame, test: pd.DataFrame,
                            time_col: str, *, strict: bool = True) -> None:
    """Fail if the test period starts before the training period ends.

    Only for protocols that claim a temporal generalization result. The temperature
    split is NOT temporal, so it does not call this.
    """
    if train.empty or test.empty:
        return
    train_end, test_start = train[time_col].max(), test[time_col].min()
    if strict and test_start <= train_end:
        raise LeakageError(
            f"Temporal leak: test begins at {test_start} but training runs to "
            f"{train_end}. A model claiming temporal generalization cannot be "
            f"trained on the future."
        )


def assert_preprocessing_is_fold_local(estimator) -> None:
    """Warn-by-failing if a scaler was fitted outside the fold.

    Fitting a scaler on the full dataset before splitting leaks test-set statistics
    (mean, variance) into training. sklearn Pipelines fit per fold; bare estimators
    handed pre-scaled arrays do not.
    """
    from sklearn.pipeline import Pipeline

    needs_pipeline = hasattr(estimator, "mean_") or hasattr(estimator, "scale_")
    if needs_pipeline and not isinstance(estimator, Pipeline):
        raise LeakageError(
            f"{type(estimator).__name__} appears to be a fitted scaler used outside a "
            f"Pipeline. Wrap preprocessing in a Pipeline so it is fitted per fold."
        )


def attrition_row(stage: str, n_before: int, n_after: int, reason: str) -> dict:
    """One line of the attrition log.

    Silently dropping inconvenient records is selection bias. Every exclusion is
    recorded with a reason and a count, and the table is published.
    """
    return {
        "stage": stage,
        "n_before": int(n_before),
        "n_after": int(n_after),
        "n_dropped": int(n_before - n_after),
        "pct_dropped": round(100.0 * (n_before - n_after) / max(n_before, 1), 2),
        "reason": reason,
    }


def summarise_groups(df: pd.DataFrame, keys: tuple[str, ...]) -> dict[str, int]:
    """Counts printed beside every claim, so sample size is never implicit."""
    out = {"n_rows": len(df)}
    for k in keys:
        if k in df.columns:
            out[f"n_{k}"] = int(pd.Series(df[k]).nunique())
    return out
