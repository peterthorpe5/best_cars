"""Intervals require actual counts; rate-only sources remain visible as points."""

import numpy as np
import pandas as pd
import pytest

from best_cars.comparison_charts import source_rate_chart
from best_cars.interval_evidence import source_interval_availability, source_rate_points


def test_counted_point_uses_actual_pair_and_preserves_rounded_rate(evidence):
    row = evidence.loc[evidence.source_id.eq("tuv")].iloc[:1].copy()
    row["sample_size"] = 746
    row["event_count"] = 11
    row["value"] = 1.5
    original = row.copy()
    result = source_rate_points(evidence=row).iloc[0]
    assert result.value == 1.5
    assert result.point_rate == pytest.approx(100 * 11 / 746)
    assert result.interval_lower == pytest.approx(0.8253148, abs=0.000001)
    assert result.interval_upper == pytest.approx(2.6209406, abs=0.000001)
    assert "actual counts" in result.interval_status
    pd.testing.assert_frame_equal(row, original)
    zero = source_rate_points(evidence=row.assign(event_count=0, value=0)).iloc[0]
    assert zero.point_rate == 0 and zero.interval_lower == 0 and zero.interval_upper > 0
    all_failed = source_rate_points(
        evidence=row.assign(event_count=746, value=100)
    ).iloc[0]
    assert all_failed.point_rate == 100 and all_failed.interval_upper == 100


def test_missing_pairs_and_rate_scaling_do_not_create_intervals(evidence):
    rate_only = evidence.assign(sample_size=np.nan, event_count=np.nan)
    points = source_rate_points(evidence=rate_only)
    assert points.point_rate.equals(points.value)
    assert points.interval_lower.isna().all() and points.interval_upper.isna().all()
    partial = evidence.loc[evidence.source_id.eq("tuv")].assign(event_count=np.nan)
    assert source_rate_points(evidence=partial).interval_lower.isna().all()
    # A rate expressed per 1,000 is not a proportion with n=1,000.
    adac = evidence.loc[evidence.source_id.eq("adac")].assign(
        sample_size=1000, event_count=15
    )
    assert source_rate_points(evidence=adac).interval_lower.isna().all()
    assert source_rate_points(evidence=adac).point_rate.equals(
        adac.value.reset_index(drop=True)
    )
    assert source_rate_points(evidence=evidence.iloc[:0]).empty
    with pytest.raises(ValueError):
        source_rate_points(evidence=pd.DataFrame())
    with pytest.raises(ValueError):
        source_rate_points(evidence=evidence.assign(event_count=200))


def test_availability_counts_observation_rows_not_vehicles(evidence):
    summary = source_interval_availability(evidence=evidence).set_index("source_id")
    assert summary.loc["tuv", "observation_rows"] == 12
    assert summary.loc["tuv", "rows_with_wilson"] == 12
    assert summary.loc["adac", "known_denominator_rows"] == 0
    assert summary.loc["adac", "rate_only_rows"] == 12
    assert source_interval_availability(evidence=evidence.iloc[:0]).empty


def test_rate_only_charts_have_open_points_and_no_error_arrays(evidence):
    tuv = evidence.loc[evidence.source_id.eq("tuv")].assign(
        sample_size=np.nan, event_count=np.nan
    )
    for colour in ["make", "uniform"]:
        figure = source_rate_chart(evidence=tuv, colour_by=colour)
        assert "interval unavailable" in figure.layout.title.text
        assert all(trace.marker.symbol == "circle-open" for trace in figure.data)
        assert all(trace.error_x.array is None for trace in figure.data)
        assert sorted(value for trace in figure.data for value in trace.x) == sorted(
            tuv.value
        )
    adac = source_rate_chart(evidence=evidence.loc[evidence.source_id.eq("adac")])
    assert "per 1,000" in adac.layout.xaxis.title.text
    assert not source_rate_chart(evidence=evidence.iloc[:0]).data


def test_mixed_count_availability_keeps_bars_only_for_known_pairs(evidence):
    tuv = evidence.loc[evidence.source_id.eq("tuv")].copy()
    tuv.loc[tuv.model_id.eq("toyota:test0"), ["sample_size", "event_count"]] = np.nan
    figure = source_rate_chart(evidence=tuv)
    assert {trace.marker.symbol for trace in figure.data} == {"circle", "circle-open"}
    for trace in figure.data:
        if trace.marker.symbol == "circle-open":
            assert trace.error_x.array is None
        else:
            assert trace.error_x.array is not None and len(trace.error_x.array) == len(
                trace.x
            )
    figure = source_rate_chart(
        evidence=tuv, top_n=2, sort_by="point_rate", ascending=False
    )
    assert sorted(value for trace in figure.data for value in trace.x) == [5, 5]
    assert source_rate_chart(evidence=tuv, sort_by="original").data
    for kwargs in [{"top_n": 0}, {"sort_by": "unknown"}, {"colour_by": "model"}]:
        with pytest.raises(ValueError):
            source_rate_chart(evidence=tuv, **kwargs)
    with pytest.raises(ValueError, match="separate"):
        source_rate_chart(evidence=evidence)
