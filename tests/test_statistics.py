"""Check descriptive distributions, actual denominators and paired agreement."""

import numpy as np
import pandas as pd
import pytest

from best_cars.analysis import cohort_scores, source_agreement
from best_cars.schema import empty_evidence
from best_cars.statistics import (
    counted_failure_rates,
    pairwise_source_agreement,
    source_rate_summary,
)


def test_source_summary_keeps_units_and_actual_counts_separate(evidence):
    summary = source_rate_summary(evidence=evidence).set_index(keys="source_id")
    assert summary.loc["tuv", "observations"] == 12
    assert summary.loc["tuv", "model_series"] == 6
    assert summary.loc["tuv", "mean_rate"] == 2.5
    assert summary.loc["tuv", "median_rate"] == 2.5
    assert summary.loc["tuv", "p25_rate"] == 1
    assert summary.loc["tuv", "p75_rate"] == 4
    assert summary.loc["tuv", "minimum_rate"] == 0
    assert summary.loc["tuv", "maximum_rate"] == 5
    assert summary.loc["tuv", "counted_tests"] == 1200
    assert summary.loc["tuv", "counted_failures"] == 30
    assert summary.loc["tuv", "counted_test_failure_pct"] == 2.5
    assert summary.loc["tuv", "count_coverage_pct"] == 100
    assert summary.loc["adac", "unit"] == "per_1000_vehicles"
    assert pd.isna(summary.loc["adac", "counted_tests"])
    assert pd.isna(summary.loc["adac", "counted_test_failure_pct"])
    assert source_rate_summary(evidence=empty_evidence()).empty


def test_summary_does_not_treat_unweighted_mean_as_fleet_rate(evidence):
    frame = evidence.loc[evidence.source_id.eq("tuv")].copy()
    frame.loc[frame.model.eq("Test5"), "sample_size"] = 1000
    frame.loc[frame.model.eq("Test5"), "event_count"] = 50
    # One unknown numerator removes that row only from the pooled count subset.
    frame.loc[frame.index[0], "event_count"] = np.nan
    summary = source_rate_summary(evidence=frame).iloc[0]
    assert summary.mean_rate == 2.5
    assert summary.counted_observations == 11
    assert summary.counted_tests == 2900
    assert summary.counted_failures == 120
    assert summary.counted_test_failure_pct == pytest.approx(100 * 120 / 2900)


def test_counted_intervals_preserve_published_rate_and_unknown_denominators(evidence):
    frame = evidence.copy()
    frame.loc[0, "value"] = 0.4  # Published rounding need not match actual k/n.
    counted = counted_failure_rates(evidence=frame)
    assert len(counted) == 12
    assert counted.loc[0, "value"] == 0.4
    assert counted.loc[0, "counted_rate_pct"] == 0
    assert counted.loc[0, "wilson_lower_pct"] == 0
    assert counted.loc[0, "wilson_upper_pct"] > 0
    assert counted.wilson_lower_pct.le(counted.counted_rate_pct).all()
    assert counted.wilson_upper_pct.ge(counted.counted_rate_pct).all()
    assert counted_failure_rates(evidence=empty_evidence()).empty


def test_paired_agreement_uses_only_observed_pairs_and_average_ranks(evidence):
    wide = source_agreement(scored=cohort_scores(evidence=evidence))
    result = pairwise_source_agreement(wide=wide).iloc[0]
    assert result.paired_models == 6
    assert result.spearman_rho == pytest.approx(-1)
    assert result.median_absolute_score_gap == pytest.approx(60)
    wide.iloc[0, 0] = np.nan
    assert pairwise_source_agreement(wide=wide).iloc[0].paired_models == 5
    assert "Too few" in pairwise_source_agreement(wide=wide, min_pairs=6).iloc[0].status
    wide.iloc[:, 0] = 50
    result = pairwise_source_agreement(wide=wide).iloc[0]
    assert np.isnan(result.spearman_rho)
    assert "Constant" in result.status
    assert pairwise_source_agreement(wide=wide.iloc[:, :1]).empty
    assert pairwise_source_agreement(wide=pd.DataFrame()).empty


@pytest.mark.parametrize("value", [-1, 101, np.inf, "invalid"])
def test_agreement_rejects_invalid_scores(value):
    wide = pd.DataFrame(data={"a": [value], "b": [50]})
    with pytest.raises(ValueError):
        pairwise_source_agreement(wide=wide)


def test_agreement_rejects_duplicates_and_too_small_threshold():
    wide = pd.DataFrame(
        data=[[1, 2], [3, 4]], index=["same", "same"], columns=["a", "b"]
    )
    with pytest.raises(ValueError, match="unique"):
        pairwise_source_agreement(wide=wide)
    with pytest.raises(ValueError):
        pairwise_source_agreement(wide=pd.DataFrame(), min_pairs=2)
    duplicate_columns = pd.DataFrame(data=[[1, 2]], columns=["a", "a"])
    with pytest.raises(ValueError):
        pairwise_source_agreement(wide=duplicate_columns)
