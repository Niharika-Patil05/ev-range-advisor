import pandas as pd
import pytest

from src.evaluation.leakage_checks import (LeakageError, assert_disjoint_groups,
                                           assert_no_capacity_derived_features,
                                           assert_no_temporal_leak,
                                           assert_three_way_disjoint, attrition_row,
                                           summarise_groups)
from src.models.soh_model import SOH_FEATURES


def frame(routes):
    return pd.DataFrame({"route_id": routes, "t": range(len(routes))})


def test_disjoint_groups_accepts_clean_split():
    assert_disjoint_groups(frame([1, 2]), frame([3, 4]), "route_id") is None


def test_disjoint_groups_rejects_overlap():
    with pytest.raises(LeakageError, match="appear in BOTH"):
        assert_disjoint_groups(frame([1, 2]), frame([2, 3]), "route_id")


def test_three_way_split_catches_calibration_overlap():
    with pytest.raises(LeakageError, match="calibration/test"):
        assert_three_way_disjoint(frame([1]), frame([3]), frame([3]))


def test_capacity_derived_features_are_rejected():
    with pytest.raises(LeakageError, match="derived from the SoH label"):
        assert_no_capacity_derived_features(["ir_mohm", "discharge_capacity_ah"])


def test_soh_itself_is_rejected_as_an_soh_feature():
    with pytest.raises(LeakageError):
        assert_no_capacity_derived_features(["soh", "ir_mohm"])


def test_soh_is_allowed_where_it_is_legitimately_an_input():
    # the range model consumes SoH as an input (usable energy scales with it);
    # only the battery model may not use it as a feature for its own label
    assert_no_capacity_derived_features(["soh", "temp_c"], allow=("soh",))


def test_the_shipped_soh_feature_list_is_clean():
    """Regression guard for CLAUDE.md rule 2 against the real feature list."""
    assert_no_capacity_derived_features(SOH_FEATURES)


def test_temporal_leak_detected():
    train = pd.DataFrame({"t": [5, 6, 7]})
    test = pd.DataFrame({"t": [3, 4]})
    with pytest.raises(LeakageError, match="Temporal leak"):
        assert_no_temporal_leak(train, test, "t")


def test_temporal_order_respected():
    assert_no_temporal_leak(pd.DataFrame({"t": [1, 2]}), pd.DataFrame({"t": [3]}), "t") is None


def test_attrition_row_counts_exclusions():
    row = attrition_row("gps_fog", 1000, 820, "no contiguous GPS window")
    assert row["n_dropped"] == 180 and row["pct_dropped"] == 18.0


def test_summarise_groups_reports_sample_sizes():
    df = pd.DataFrame({"route_id": [1, 1, 2], "vehicle_id": ["A", "A", "B"]})
    out = summarise_groups(df, ("route_id", "vehicle_id"))
    assert out == {"n_rows": 3, "n_route_id": 2, "n_vehicle_id": 2}
