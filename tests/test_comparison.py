"""Verify exact matching, historical populations and honest comparison graphics."""

import numpy as np
import pandas as pd
import pytest

from best_cars.analysis import cohort_scores
from best_cars.comparison import (
    matched_model_rates,
    temporal_profiles,
    wilson_age_groups,
)
from best_cars.comparison_charts import (
    evidence_coverage_chart,
    matched_rate_chart,
    peer_gap_chart,
    temporal_chart,
)
from best_cars.interrogation import cohort_peer_benchmarks
from best_cars.statistics import counted_failure_rates

MODELS = ["toyota:test0", "toyota:test1"]
NAMES = {"toyota:test0": "Toyota Test0", "toyota:test1": "Toyota Test1"}


def test_pairs_preserve_exact_contexts_and_original_units(evidence):
    paired = matched_model_rates(
        evidence=evidence, model_ids=MODELS, reference_id=MODELS[0]
    )
    assert len(paired) == 4 and paired.model_id.eq(MODELS[1]).all()
    assert paired.loc[paired.source_id.eq("tuv"), "rate_gap"].eq(1).all()
    assert paired.loc[paired.source_id.eq("adac"), "rate_gap"].eq(-1).all()
    assert set(paired.unit) == {"percent", "per_1000_vehicles"}
    changed = evidence.copy()
    changed.loc[changed.model_id.eq(MODELS[1]), "cohort_year_end"] += 1
    assert matched_model_rates(
        evidence=changed, model_ids=MODELS, reference_id=MODELS[0]
    ).empty
    assert matched_model_rates(
        evidence=evidence, model_ids=[MODELS[0]], reference_id=MODELS[0]
    ).empty
    for models, reference in [(MODELS + MODELS[:1], MODELS[0]), (MODELS, "absent")]:
        with pytest.raises(ValueError):
            matched_model_rates(
                evidence=evidence, model_ids=models, reference_id=reference
            )
    grouped = evidence.assign(model_grouped=True)
    with pytest.raises(ValueError):
        matched_model_rates(evidence=grouped, model_ids=MODELS, reference_id=MODELS[0])
    duplicate = evidence.iloc[:1].copy()
    duplicate["comparison_group"] = "alternative_label"
    # Canonical rows can differ by label, but matching remains context-specific.
    result = matched_model_rates(
        evidence=pd.concat(objs=[evidence, duplicate]),
        model_ids=MODELS,
        reference_id=MODELS[0],
    )
    assert len(result) == 4


def test_temporal_age_holds_age_fixed_and_registration_holds_cohort(evidence):
    old = evidence.copy()
    old["report_year"] = 2024
    old["observation_year"] = 2023
    old["cohort_year_start"] -= 1
    old["cohort_year_end"] -= 1
    history = pd.concat(objs=[old, evidence], ignore_index=True)
    age = temporal_profiles(evidence=history, model_ids=MODELS, mode="age")
    assert len(age) == 16
    assert age.groupby("series_id").age_min.nunique().eq(1).all()
    assert age.groupby("series_id").cohort_year_start.nunique().eq(2).all()
    # Some different age bands can share a registration cohort across years;
    # only identical recorded bounds should be connected.
    registration = temporal_profiles(
        evidence=history, model_ids=MODELS, mode="registration"
    )
    assert registration.empty
    old["cohort_year_start"] += 1
    old["cohort_year_end"] += 1
    old["age_min"] -= 1
    old["age_max"] -= 1
    registration = temporal_profiles(
        evidence=pd.concat(objs=[old, evidence]), model_ids=MODELS, mode="registration"
    )
    assert len(registration) == 8 and registration.source_id.eq("adac").all()
    assert registration.groupby("series_id").cohort_year_start.nunique().eq(1).all()
    assert registration.groupby("series_id").age_min.nunique().eq(2).all()
    assert temporal_profiles(evidence=evidence, model_ids=MODELS).empty
    assert temporal_profiles(evidence=evidence.iloc[:0], model_ids=MODELS).empty
    with pytest.raises(ValueError):
        temporal_profiles(evidence=evidence, model_ids=MODELS, mode="unknown")
    with pytest.raises(ValueError):
        temporal_profiles(evidence=evidence, model_ids=MODELS + MODELS[:1])
    extra = evidence.copy()
    extra["report_year"] = 2026
    with pytest.raises(ValueError, match="Multiple estimates"):
        temporal_profiles(evidence=pd.concat(objs=[evidence, extra]), model_ids=MODELS)


def test_coverage_colours_indicate_data_not_scores(evidence):
    mask = (
        evidence.model_id.eq(MODELS[1])
        & evidence.source_id.eq("adac")
        & evidence.age_min.eq(4)
    )
    incomplete = evidence.loc[~mask]
    figure = evidence_coverage_chart(evidence=incomplete, model_ids=MODELS, names=NAMES)
    trace = figure.data[0]
    assert trace.z.shape == (2, 4)
    assert set(np.asarray(trace.z).flatten()) == {0, 1, 2}
    assert np.asarray(trace.text)[np.asarray(trace.z) == 0].tolist() == ["No data"]
    assert set(np.asarray(trace.text)[np.asarray(trace.z) == 1]) == {"Rate only"}
    assert set(np.asarray(trace.text)[np.asarray(trace.z) == 2]) == {"n=100"}
    assert "coverage, not reliability" in figure.layout.xaxis.title.text
    assert not evidence_coverage_chart(
        evidence=evidence.iloc[:0], model_ids=MODELS, names=NAMES
    ).data
    with pytest.raises(ValueError):
        evidence_coverage_chart(
            evidence=evidence, model_ids=MODELS + MODELS[:1], names=NAMES
        )
    with pytest.raises(ValueError):
        evidence_coverage_chart(
            evidence=pd.concat(objs=[evidence, evidence.iloc[:1]]),
            model_ids=MODELS,
            names=NAMES,
        )


def test_peer_gaps_preserve_sign_zero_reference_and_source_axes(evidence):
    benchmarks = cohort_peer_benchmarks(scored=cohort_scores(evidence=evidence))
    selected = benchmarks.loc[benchmarks.model_id.eq(MODELS[0])]
    for colour in ["model", "make"]:
        figure = peer_gap_chart(benchmarks=selected, colour_by=colour)
        assert len(figure.data) == 2
        values = [float(value) for trace in figure.data for value in trace.y]
        assert sorted(values) == [-3, -3, 3, 3]
        assert len(figure.layout.shapes) == 2
        assert all(axis.matches is None for axis in figure.select_yaxes())
    assert not peer_gap_chart(benchmarks=selected.iloc[:0]).data
    with pytest.raises(ValueError):
        peer_gap_chart(benchmarks=selected, colour_by="unknown")


def test_matched_scatter_separates_units_and_uses_parity_line(evidence):
    paired = matched_model_rates(
        evidence=evidence, model_ids=MODELS, reference_id=MODELS[0]
    )
    for colour in ["model", "make"]:
        figure = matched_rate_chart(paired=paired, colour_by=colour)
        assert len(figure.data) == 4
        for index in [1, 3]:
            assert list(figure.data[index].x) == list(figure.data[index].y)
        for xaxis, yaxis in zip(
            figure.select_xaxes(), figure.select_yaxes(), strict=True
        ):
            assert xaxis.range == yaxis.range
    assert not matched_rate_chart(paired=paired.iloc[:0]).data
    with pytest.raises(ValueError):
        matched_rate_chart(paired=paired, colour_by="unknown")


def test_time_lines_do_not_join_different_models_or_cohorts(evidence):
    old = evidence.assign(report_year=2024, observation_year=2023)
    old["cohort_year_start"] -= 1
    old["cohort_year_end"] -= 1
    profiles = temporal_profiles(
        evidence=pd.concat(objs=[old, evidence]), model_ids=MODELS
    )
    for colour in ["model", "make"]:
        figure = temporal_chart(profiles=profiles, colour_by=colour)
        assert len(figure.data) == 8
        assert all(list(trace.x) == [2023, 2024] for trace in figure.data)
        assert all(len(trace.y) == 2 for trace in figure.data)
    assert not temporal_chart(profiles=profiles.iloc[:0]).data
    with pytest.raises(ValueError):
        temporal_chart(profiles=profiles, colour_by="unknown")


def test_wilson_panels_preserve_counts_and_separate_ages_sources_reports(evidence):
    counted = counted_failure_rates(evidence=evidence)
    original = counted.copy()
    groups = wilson_age_groups(counted=counted)
    assert len(groups) == 2
    assert sum(len(group) for group in groups.values()) == len(counted)
    for group in groups.values():
        assert group.age_min.nunique() == 1
        assert group.source_id.nunique() == 1 and group.report_year.nunique() == 1
    pd.testing.assert_frame_equal(left=counted, right=original)
    newer = counted.assign(report_year=2026, observation_year=2025)
    assert len(wilson_age_groups(counted=pd.concat(objs=[counted, newer]))) == 4
    other_source = counted.assign(source_id="mot", source_name="UK MOT", country="UK")
    assert len(wilson_age_groups(counted=pd.concat(objs=[counted, other_source]))) == 4
    five_year = counted.loc[counted.age_min.eq(4)].assign(
        age_min=5,
        age_max=5,
        comparison_group="age_5",
        cohort_year_start=2019,
        cohort_year_end=2019,
    )
    neighbours = pd.concat(objs=[counted, five_year])
    exact = wilson_age_groups(counted=neighbours)
    nearby = wilson_age_groups(counted=neighbours, grouping="two_year")
    assert len(exact) == 3 and len(nearby) == 2
    assert any(set(group.age_min) == {4, 5} for group in nearby.values())
    assert (
        sum(group.event_count.sum() for group in nearby.values())
        == neighbours.event_count.sum()
    )
    assert len(wilson_age_groups(counted=neighbours, grouping="mixed")) == 1
    wide = counted.iloc[:1].assign(age_min=3, age_max=6)
    assert "age 3–6" in next(iter(wilson_age_groups(counted=wide, grouping="two_year")))
    assert wilson_age_groups(counted=counted.iloc[:0]) == {}
    with pytest.raises(ValueError):
        wilson_age_groups(counted=counted, grouping="unknown")
    with pytest.raises(ValueError):
        wilson_age_groups(counted=pd.DataFrame())
