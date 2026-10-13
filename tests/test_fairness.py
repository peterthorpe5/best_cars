"""Preference comparisons must not reward missing outcomes or mismatched ages."""

import numpy as np
import pandas as pd
import pytest

from best_cars.analysis import cohort_scores
from best_cars.fairness import (
    cohort_contexts,
    coverage_matched_ranking,
    evidence_profiles,
    source_cohort_ranking,
)


def test_separate_profiles_and_exact_cohort_ordering(evidence):
    scored = cohort_scores(evidence=evidence)
    profiles = evidence_profiles(scored=scored)
    assert len(profiles) == 12
    assert set(profiles.outcome) == {"Inspection condition", "Roadside breakdowns"}
    assert profiles.observations.eq(2).all()
    contexts = cohort_contexts(scored=scored, source_id="tuv")
    assert len(contexts) == 2
    ranking = source_cohort_ranking(
        scored=scored, source_id="tuv", context_id=contexts.context_id.iloc[0]
    )
    assert ranking.cohorts.eq(1).all()
    assert ranking.iloc[0].model_id == "toyota:test0"
    assert ranking.iloc[0].score == 100
    assert evidence_profiles(scored=scored.iloc[:0]).empty
    assert cohort_contexts(scored=scored, source_id="absent").empty
    with pytest.raises(ValueError):
        source_cohort_ranking(scored=scored, source_id="tuv", context_id="absent")
    with pytest.raises(ValueError):
        evidence_profiles(scored=pd.DataFrame())
    with pytest.raises(ValueError):
        cohort_contexts(scored=pd.DataFrame(), source_id="tuv")


def test_missing_source_is_excluded_and_singleton_has_no_rank(evidence):
    scored = cohort_scores(evidence=evidence)
    missing = scored.loc[
        ~(scored.model_id.eq("toyota:test0") & scored.source_id.eq("adac"))
    ]
    ranked = coverage_matched_ranking(scored=missing)
    assert "toyota:test0" not in set(ranked.model_id)
    assert ranked.weight_coverage.eq(100).all()
    unequal = scored.drop(index=0)
    ranked = coverage_matched_ranking(scored=unequal)
    target = ranked.loc[ranked.model_id.eq("toyota:test0")].iloc[0]
    assert target.comparison_models == 1
    assert pd.isna(target["rank"])
    assert pd.isna(target.score)
    assert "no composite score or rank" in target.comparison_status
    peers = ranked.loc[~ranked.model_id.eq("toyota:test0")]
    assert peers.comparison_models.eq(5).all()
    assert peers.comparison_group_id.nunique() == 1
    assert coverage_matched_ranking(scored=scored.iloc[:0]).empty


def test_outcome_first_weighting_does_not_double_count_inspections(evidence):
    extra = evidence.loc[evidence.source_id.eq("adac")].copy()
    extra["source_id"] = "traficom"
    extra["source_name"] = "Traficom"
    extra["metric"] = "defects"
    extra["unit"] = "percent"
    scored = cohort_scores(
        evidence=pd.concat(objs=[evidence, extra], ignore_index=True)
    )
    ranked = coverage_matched_ranking(scored=scored)
    target = ranked.loc[ranked.model_id.eq("toyota:test0")].iloc[0]
    assert target.inspection_score == 50 and target.breakdown_score == 0
    assert target.score == 25  # Not the old three-source average of 33.3.
    emphasised = coverage_matched_ranking(scored=scored, breakdown_weight=2)
    assert emphasised.loc[emphasised.model_id.eq("toyota:test0"), "score"].iloc[
        0
    ] == pytest.approx(expected=50 / 3)
    only_inspections = coverage_matched_ranking(scored=scored, breakdown_weight=0)
    assert only_inspections.sources.eq(2).all()
    assert only_inspections.breakdown_score.isna().all()
    assert only_inspections.score.eq(50).all()
    weighted = coverage_matched_ranking(scored=scored, weights={"tuv": 3})
    assert (
        weighted.loc[weighted.model_id.eq("toyota:test0"), "inspection_score"].iloc[0]
        == 75
    )
    assert coverage_matched_ranking(scored=scored, min_sources=4).empty
    assert coverage_matched_ranking(scored=scored, min_cohorts=100).empty
    with pytest.raises(ValueError):
        coverage_matched_ranking(scored=scored, inspection_weight=-1)
    with pytest.raises(ValueError):
        coverage_matched_ranking(scored=scored, inspection_weight=0, breakdown_weight=0)
    with pytest.raises(ValueError):
        coverage_matched_ranking(scored=scored, breakdown_weight=np.inf)
    with pytest.raises(ValueError):
        coverage_matched_ranking(scored=scored.assign(source_id="unknown"))


def test_coverage_signatures_ignore_row_order_but_keep_years_and_units(evidence):
    scored = cohort_scores(evidence=evidence)
    baseline = coverage_matched_ranking(scored=scored)
    shuffled = coverage_matched_ranking(scored=scored.sample(frac=1, random_state=3))
    pd.testing.assert_frame_equal(left=baseline, right=shuffled)
    changed = scored.copy()
    changed.loc[changed.model_id.eq("toyota:test0"), "cohort_year_start"] -= 1
    changed.loc[changed.model_id.eq("toyota:test0"), "cohort_year_end"] -= 1
    result = coverage_matched_ranking(scored=changed)
    assert pd.isna(result.loc[result.model_id.eq("toyota:test0"), "rank"].iloc[0])


def test_exploratory_missing_evidence_is_explicit_and_bounded(evidence):
    from best_cars.fairness import exploratory_ranking

    scored = cohort_scores(evidence=evidence)
    incomplete = scored.loc[
        ~(scored.model_id.eq("toyota:test0") & scored.source_id.eq("adac"))
    ]
    result = exploratory_ranking(scored=incomplete)
    target = result.loc[result.model_id.eq("toyota:test0")].iloc[0]
    assert target.missing_sources == "adac"
    assert target.weight_coverage == 50
    assert target.score == 75
    assert target.score_lower == 50 and target.score_upper == 100
    assert (
        target.best_missing_evidence_rank
        <= target["rank"]
        <= target.worst_missing_evidence_rank
    )
    assert target.youngest_age <= target.oldest_age
    assert target.known_sample_rows > 0
    complete = result.loc[result.missing_sources.eq("None")]
    assert complete.score_lower.equals(complete.score_upper)
    assert (
        exploratory_ranking(scored=incomplete, min_sources=2)
        .model_id.ne("toyota:test0")
        .all()
    )
    assert exploratory_ranking(scored=scored.iloc[:0]).empty
    only_inspections = exploratory_ranking(scored=incomplete, breakdown_weight=0)
    assert only_inspections.weight_coverage.eq(100).all()
