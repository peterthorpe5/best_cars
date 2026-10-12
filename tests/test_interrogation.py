"""Shared-cohort and source-omission checks must preserve missing evidence."""

import numpy as np
import pytest

from best_cars.analysis import cohort_scores
from best_cars.interrogation import (
    cohort_peer_benchmarks,
    shared_cohort_contrasts,
    shared_cohort_evidence,
    source_omission_sensitivity,
    weighting_scenarios,
)


def test_peer_benchmark_excludes_target_preserves_units_and_requires_peers(evidence):
    result = cohort_peer_benchmarks(scored=cohort_scores(evidence=evidence))
    row = result.loc[
        result.model_id.eq("toyota:test0") & result.source_id.eq("tuv")
    ].iloc[0]
    assert row.peer_median_rate == 3 and row.rate_gap == -3
    assert result.peer_models.eq(5).all()
    assert set(result.unit) == {"percent", "per_1000_vehicles"}
    assert cohort_peer_benchmarks(
        scored=evidence.loc[~evidence.model_id.isin(["toyota:test4", "toyota:test5"])]
    ).empty
    assert cohort_peer_benchmarks(scored=evidence.iloc[:0]).empty
    with pytest.raises(ValueError):
        cohort_peer_benchmarks(scored=evidence.drop(columns="model_id"))


def test_weight_scenario_ranks_share_same_universe_and_keep_original_scores(evidence):
    scored = cohort_scores(evidence=evidence)
    result = weighting_scenarios(scored=scored, min_cohorts=1)
    assert result.scenario.nunique() == 3
    assert result.groupby(by="scenario").model_id.nunique().eq(6).all()
    balanced = result.loc[result.scenario.eq("Balanced")]
    assert balanced.score.eq(50).all() and balanced.scenario_rank.eq(1).all()
    assert weighting_scenarios(scored=scored.iloc[:0]).empty
    with pytest.raises(ValueError):
        weighting_scenarios(scored=scored, min_sources=0)


def test_shared_cohorts_use_every_selected_model_and_keep_units(evidence):
    scored = cohort_scores(evidence=evidence)
    ids = ["toyota:test0", "toyota:test1", "toyota:test2"]
    matched = shared_cohort_evidence(scored=scored, model_ids=ids)
    assert len(matched) == 12
    contrasts = shared_cohort_contrasts(scored=scored, model_ids=ids)
    assert len(contrasts) == 6
    assert contrasts.shared_cohorts.eq(2).all()
    assert set(contrasts.unit) == {"percent", "per_1000_vehicles"}
    first = contrasts.loc[
        contrasts.first_model.eq("Toyota Test0")
        & contrasts.second_model.eq("Toyota Test1")
    ]
    assert first.loc[first.unit.eq("percent"), "mean_rate_gap"].iloc[0] == -1
    assert first.loc[first.unit.eq("per_1000_vehicles"), "mean_rate_gap"].iloc[0] == 1


def test_shared_cohorts_do_not_fill_missing_age_or_mix_report_years(evidence):
    frame = evidence.drop(index=0)
    ids = ["toyota:test0", "toyota:test1"]
    matched = shared_cohort_evidence(scored=frame, model_ids=ids)
    assert len(matched) == 6  # One TÜV age excluded for both models.
    frame = frame.loc[~(frame.model_id.eq("toyota:test0") & frame.source_id.eq("tuv"))]
    frame.loc[frame.model_id.eq("toyota:test0"), "report_year"] = 2026
    assert shared_cohort_contrasts(scored=frame, model_ids=ids).empty


@pytest.mark.parametrize(
    "ids",
    [["toyota:test0"], ["toyota:test0", "toyota:test0"], ["toyota:test0", "absent"]],
)
def test_shared_cohort_rejects_invalid_selections(evidence, ids):
    with pytest.raises(ValueError):
        shared_cohort_evidence(scored=evidence, model_ids=ids)


def test_shared_cohorts_reject_grouped_series(evidence):
    evidence.loc[0, "model_grouped"] = True
    with pytest.raises(ValueError, match="ungrouped"):
        shared_cohort_evidence(
            scored=evidence, model_ids=["toyota:test0", "toyota:test1"]
        )


def test_source_sensitivity_reports_missing_remaining_cohorts(evidence):
    scored = cohort_scores(evidence=evidence)
    result = source_omission_sensitivity(scored=scored)
    assert len(result) == 12
    assert result.score_without_source.isna().all()
    assert result.paired_models.eq(0).all()
    assert result.status.eq("Insufficient remaining evidence").all()


def test_source_sensitivity_matches_rank_universe_and_keeps_original_scores(evidence):
    scored = cohort_scores(evidence=evidence)
    original = scored.relative_score.copy()
    result = source_omission_sensitivity(scored=scored, min_cohorts=1)
    assert result.paired_models.eq(6).all()
    assert result.baseline_score.eq(50).all()
    assert result.baseline_paired_rank.eq(1).all()
    assert result.rank_without_source.between(left=1, right=6).all()
    assert scored.relative_score.equals(original)
    assert np.isfinite(result.score_change).all()
    assert source_omission_sensitivity(
        scored=scored, weights={"adac": 0}, min_sources=1, min_cohorts=1
    ).empty
    assert source_omission_sensitivity(scored=scored.iloc[:0]).empty
    with pytest.raises(ValueError):
        source_omission_sensitivity(scored=scored, weights={"tuv": -1})
