"""Tests for coverage, weights, ties and sampling intervals."""

import numpy as np
import pandas as pd
import pytest

from best_cars.analysis import (
    cohort_scores,
    reliability_ranking,
    select_evidence,
    source_agreement,
    wilson_interval,
)
from best_cars.schema import empty_evidence


def test_relative_ordering_and_disagreement(evidence):
    scored = cohort_scores(evidence=evidence)
    first = scored.loc[scored.model_id.eq("toyota:test0")]
    assert first.loc[first.source_id.eq("tuv"), "relative_score"].eq(100).all()
    assert first.loc[first.source_id.eq("adac"), "relative_score"].eq(0).all()
    ranked = reliability_ranking(scored=scored, min_sources=2)
    assert ranked.score.eq(50).all()
    assert ranked.sources.eq(2).all()
    assert ranked.score_spread.max() == 100


def test_weights_and_missing_are_not_zero(evidence):
    scored = cohort_scores(evidence=evidence)
    missing = scored.loc[
        ~(scored.model_id.eq("toyota:test0") & scored.source_id.eq("adac"))
    ]
    ranked = reliability_ranking(scored=missing)
    row = ranked.loc[ranked.model_id.eq("toyota:test0")].iloc[0]
    assert row.score == 100
    assert row.weight_coverage == 50
    assert row.sources == 1
    ranked = reliability_ranking(scored=scored, weights={"tuv": 0, "adac": 2})
    assert ranked.iloc[0].model_id == "toyota:test5"
    assert ranked.sources.eq(1).all()
    assert ranked.weight_coverage.eq(100).all()


def test_many_cohorts_do_not_give_one_source_more_votes(evidence):
    extra = evidence.loc[evidence.source_id.eq("tuv")].copy()
    extra["report_year"] = 2024
    extra["observation_year"] = 2023
    scored = cohort_scores(
        evidence=pd.concat(objs=[evidence, extra], ignore_index=True)
    )
    assert reliability_ranking(scored=scored).score.eq(50).all()


def test_display_filters_keep_original_percentiles(evidence):
    full = cohort_scores(evidence=evidence)
    selected = select_evidence(evidence=evidence, sources=["tuv"], age_range=(4, 4))
    assert list(selected.relative_score) == list(
        full.loc[full.source_id.eq("tuv") & full.age_min.eq(4), "relative_score"]
    )
    assert select_evidence(evidence=evidence, sources=[]).empty
    assert select_evidence(evidence=evidence, min_sample=200).source_id.eq("adac").all()
    assert (
        select_evidence(evidence=evidence, cohort_range=(2018, 2018))
        .age_min.eq(6)
        .all()
    )


def test_latest_report_is_selected_separately(evidence):
    older = evidence.copy()
    older["report_year"] = 2024
    older["observation_year"] = 2023
    full = pd.concat(objs=[evidence, older], ignore_index=True)
    assert select_evidence(evidence=full).report_year.eq(2025).all()
    assert len(select_evidence(evidence=full, latest_only=False)) == len(full)


def test_ties_small_cohorts_and_grouped_models(evidence):
    equal = evidence.copy()
    equal["value"] = 2.0
    assert cohort_scores(evidence=equal).relative_score.eq(50).all()
    small = evidence.loc[evidence.model_id.isin(["toyota:test0", "toyota:test1"])]
    assert cohort_scores(evidence=small).relative_score.isna().all()
    grouped = cohort_scores(evidence=evidence)
    grouped.loc[grouped.model_id.eq("toyota:test0"), "model_grouped"] = True
    assert "toyota:test0" not in set(reliability_ranking(scored=grouped).model_id)
    assert "toyota:test0" not in source_agreement(scored=grouped).index


@pytest.mark.parametrize("weight", [-1, np.inf, np.nan])
def test_bad_weights(evidence, weight):
    with pytest.raises(ValueError):
        reliability_ranking(
            scored=cohort_scores(evidence=evidence), weights={"tuv": weight}
        )


def test_zero_weights_and_invalid_thresholds(evidence):
    scored = cohort_scores(evidence=evidence)
    with pytest.raises(ValueError):
        reliability_ranking(scored=scored, weights={"tuv": 0, "adac": 0})
    with pytest.raises(ValueError):
        reliability_ranking(scored=scored, min_sources=0)
    assert reliability_ranking(scored=scored, min_sources=3).empty
    with pytest.raises(ValueError):
        cohort_scores(evidence=evidence, min_peers=1)
    with pytest.raises(ValueError):
        select_evidence(evidence=evidence, min_sample=-1)
    with pytest.raises(ValueError):
        select_evidence(evidence=evidence, age_range=(9, 2))


def test_empty_analysis():
    empty = empty_evidence()
    assert cohort_scores(evidence=empty).empty
    assert select_evidence(evidence=empty).empty
    assert reliability_ranking(scored=empty).empty
    assert source_agreement(scored=empty).empty


def test_wilson_intervals():
    low, high = wilson_interval(failures=0, tests=100)
    assert low == 0
    assert 3.6 < high < 3.8
    low, high = wilson_interval(failures=50, tests=100)
    assert 40 < low < 41
    assert 59 < high < 60
    assert wilson_interval(failures=100, tests=100)[1] == 100


@pytest.mark.parametrize(
    "failures,tests", [(-1, 3), (4, 3), (0, 0), (1.5, 3), (True, 3), (1, False)]
)
def test_invalid_wilson_counts(failures, tests):
    with pytest.raises(ValueError):
        wilson_interval(failures=failures, tests=tests)
