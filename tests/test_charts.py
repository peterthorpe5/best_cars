"""Charts must preserve units, gaps and bounded score scales."""

import pandas as pd
import plotly.graph_objects as go
import pytest

from best_cars.analysis import cohort_scores, reliability_ranking, source_agreement
from best_cars.charts import (
    age_chart,
    age_score_heatmap,
    agreement_heatmap,
    agreement_scatter,
    count_uncertainty_chart,
    mileage_chart,
    ranking_chart,
    rate_distribution_chart,
    score_spread_chart,
    style_figure,
)
from best_cars.statistics import counted_failure_rates


def test_ranking_and_style(evidence):
    ranking = reliability_ranking(scored=cohort_scores(evidence=evidence))
    figure = ranking_chart(ranking=ranking)
    assert figure.layout.xaxis.range == (0, 105)
    assert sum(len(trace.x) for trace in figure.data) == 6
    assert "plotly" in figure.to_html(include_plotlyjs=True).lower()
    assert style_figure(figure=go.Figure(), title="Test").layout.title.text == "Test"
    with pytest.raises(ValueError):
        ranking_chart(ranking=ranking, top_n=0)


def test_age_panels_and_empty_cases(evidence):
    scored = cohort_scores(evidence=evidence)
    figure = age_chart(scored=scored, model_ids=["toyota:test0"], relative=True)
    assert len(figure.data) == 2
    assert not age_chart(scored=scored, model_ids=[]).data
    assert not ranking_chart(ranking=pd.DataFrame()).data
    assert not agreement_heatmap(wide=pd.DataFrame(), names={}).data


def test_agreement_and_missing_cells(evidence):
    wide = source_agreement(scored=cohort_scores(evidence=evidence))
    heatmap = agreement_heatmap(wide=wide, names={})
    assert heatmap.data[0].zmin == 0 and heatmap.data[0].zmax == 100
    figure = agreement_scatter(wide=wide, first="tuv", second="adac", names={})
    assert len(figure.data[0].x) == 6
    assert len(figure.layout.shapes) == 1
    with pytest.raises(ValueError):
        agreement_scatter(wide=wide, first="tuv", second="tuv", names={})


def test_mileage_requires_comparable_odometer_measurements(evidence):
    with pytest.raises(ValueError):
        mileage_chart(evidence=evidence)
    one = evidence.loc[evidence.source_id.eq("tuv")].copy()
    assert not mileage_chart(evidence=one).data
    one["mileage_km"] = 40000
    one["mileage_basis"] = "mean_odometer"
    assert len(mileage_chart(evidence=one).data[0].x) == len(one)


def test_distribution_panels_preserve_units_and_show_mean(evidence):
    figure = rate_distribution_chart(evidence=evidence)
    assert len(figure.data) == 4  # Two published ages in each source.
    assert all(trace.boxmean is True for trace in figure.data)
    assert figure.layout.yaxis.matches is None
    assert figure.layout.yaxis2.matches is None
    assert any(
        "per 1,000" in annotation.text for annotation in figure.layout.annotations
    )
    assert not rate_distribution_chart(evidence=evidence.iloc[:0]).data


def test_count_uncertainty_uses_actual_counts_and_bounds(evidence):
    counted = counted_failure_rates(evidence=evidence)
    counted.loc[0, "sample_size"] = 1000
    figure = count_uncertainty_chart(counted=counted, top_n=3)
    assert len(figure.data[0].x) == 3
    assert figure.data[0].customdata[0][0] == 1000
    assert figure.data[0].error_x.symmetric is False
    assert min(figure.data[0].error_x.array) >= 0
    assert min(figure.data[0].error_x.arrayminus) >= 0
    assert not count_uncertainty_chart(counted=counted.iloc[:0]).data
    with pytest.raises(ValueError):
        count_uncertainty_chart(counted=counted, top_n=0)


def test_score_spread_keeps_all_models_and_source_metadata(evidence):
    ranking = reliability_ranking(scored=cohort_scores(evidence=evidence))
    figure = score_spread_chart(ranking=ranking)
    assert sum(len(trace.x) for trace in figure.data) == len(ranking)
    assert figure.layout.xaxis.range == (-2, 102)
    assert figure.layout.yaxis.range[0] < 0
    assert figure.layout.yaxis.range[1] >= ranking.score_spread.max()
    assert not score_spread_chart(ranking=ranking.iloc[:0]).data


def test_age_heatmap_uses_full_peer_scores_and_missing_cells(evidence):
    scored = cohort_scores(evidence=evidence)
    one = scored.loc[scored.source_id.eq("tuv")].copy()
    one = one.drop(index=0)
    figure = age_score_heatmap(scored=one, max_models=6)
    assert figure.data[0].zmin == 0 and figure.data[0].zmax == 100
    assert figure.layout.yaxis.autorange == "reversed"
    assert len(figure.data[0].y) == 6
    assert figure.layout.xaxis.type == "category"
    assert figure.layout.yaxis.type == "category"
    assert len(figure.data) == 2
    assert any("No score" in row for row in figure.data[0].text)
    assert len(age_score_heatmap(scored=one, max_models=3, order="both").data[0].y) == 3
    assert age_score_heatmap(scored=one, order="weakest").data[0].y[0] == "Toyota Test5"
    assert (
        age_score_heatmap(scored=one, order="strongest").data[0].y[0] == "Toyota Test0"
    )
    assert not age_score_heatmap(scored=one.iloc[:0]).data
    with pytest.raises(ValueError):
        age_score_heatmap(scored=scored)
    with pytest.raises(ValueError):
        age_score_heatmap(scored=one, max_models=0)
    with pytest.raises(ValueError):
        age_score_heatmap(scored=one, order="unknown")
    with pytest.raises(ValueError):
        age_score_heatmap(scored=one, measure="unknown")
    raw = age_score_heatmap(scored=one, measure="rate")
    assert raw.data[0].zmax == one.value.max()
    for row, texts in zip(raw.data[0].z, raw.data[0].text, strict=True):
        for value, label in zip(row, texts, strict=True):
            if pd.notna(value):
                assert label == f"{value:.1f}"
    assert raw.data[0].colorscale[0][1] == "#007f87"
    assert raw.data[0].colorscale[-1][1] == "#c95b55"
    assert (
        age_score_heatmap(scored=one, measure="rate", rate_ceiling=50).data[0].zmax
        == 50
    )
    with pytest.raises(ValueError):
        age_score_heatmap(scored=one, measure="rate", rate_ceiling=1)
    duplicate_age = one.iloc[:1].copy()
    duplicate_age["comparison_group"] = "another"
    with pytest.raises(ValueError, match="same age"):
        age_score_heatmap(
            scored=pd.concat(objs=[one, duplicate_age], ignore_index=True)
        )


def test_age_colours_are_stable_across_source_capitalisation(evidence):
    scored = cohort_scores(evidence=evidence)
    scored.loc[scored.source_id.eq("adac"), "model"] = scored.loc[
        scored.source_id.eq("adac"), "model"
    ].str.upper()
    figure = age_chart(scored=scored, model_ids=["toyota:test0", "toyota:test1"])
    colours = {}
    for trace in figure.data:
        model = trace.name.split(",")[0]
        colours.setdefault(model, set()).add(trace.line.color)
    assert len(colours) == 2
    assert all(len(values) == 1 for values in colours.values())
    ranking = reliability_ranking(scored=scored)
    assert all("source" in trace.name for trace in ranking_chart(ranking=ranking).data)
