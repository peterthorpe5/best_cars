"""Screenshot regressions: eligibility, ties, denominators and readable marks."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bs4 import BeautifulSoup
from streamlit.testing.v1 import AppTest

from best_cars.analysis import cohort_scores, reliability_ranking
from best_cars.charts import (
    _make_colours,
    _rate_axis_title,
    count_uncertainty_chart,
    ranking_chart,
    rate_distribution_chart,
    scenario_rank_chart,
    score_spread_chart,
)
from best_cars.comparison_charts import source_rate_chart
from best_cars.interrogation import weighting_scenarios
from best_cars.report import report_html
from best_cars.schema import empty_events
from best_cars.statistics import counted_failure_rates
from best_cars.storage import save_snapshot

APP = Path(__file__).parents[1] / "app.py"


def test_curated_make_colours_separate_previous_collisions_and_allow_highlights():
    makes = ["Volkswagen", "Audi", "Ford", "Hyundai", "Porsche", "Mazda"]
    colours = _make_colours(makes=makes)
    rgb = np.array(
        [
            [int(value[start : start + 2], 16) for start in [1, 3, 5]]
            for value in colours.values()
        ]
    )
    distances = np.sqrt(((rgb[:, None] - rgb[None, :]) ** 2).sum(axis=2))
    assert distances[np.triu_indices(n=len(makes), k=1)].min() > 60
    assert _make_colours(makes=["Audi"]) == {"Audi": colours["Audi"]}
    assert _make_colours(makes=["AUDI"])["AUDI"] == colours["Audi"]
    assert (
        _make_colours(makes=["Skoda", "Škoda"])["Skoda"]
        == _make_colours(makes=["Škoda"])["Škoda"]
    )
    assert _make_colours(makes=["Unlisted make"]) == _make_colours(
        makes=["Unlisted make"]
    )
    highlighted = _make_colours(makes=makes, highlight_makes=["Volkswagen"])
    assert highlighted["Volkswagen"] == colours["Volkswagen"]
    assert {highlighted[make] for make in makes if make != "Volkswagen"} == {"#CCD2D8"}
    with pytest.raises(ValueError, match="non-empty"):
        _make_colours(makes=[""])
    assert "Significant-defect" in _rate_axis_title(source_id="tuv", unit="percent")
    assert "Failed inspections" in _rate_axis_title(source_id="mot", unit="percent")
    assert "per 1,000" in _rate_axis_title(source_id="adac", unit="per_1000_vehicles")


def test_bar_labels_stay_horizontal_outside_and_highlighting_preserves_scores(evidence):
    ranking = reliability_ranking(scored=cohort_scores(evidence=evidence))
    ranking.loc[ranking.index[0], "make"] = "Volkswagen"
    original = ranking.copy(deep=True)
    figure = ranking_chart(
        ranking=ranking, colour_by="make", highlight_makes=["Volkswagen"]
    )
    assert sum(len(trace.x) for trace in figure.data) == len(ranking)
    assert all(
        trace.textangle == 0 and trace.textposition == "outside"
        for trace in figure.data
    )
    assert {trace.marker.color for trace in figure.data} == {"#CCD2D8", "#0072B2"}
    pd.testing.assert_frame_equal(left=ranking, right=original)


def test_spread_omits_single_source_models_instead_of_showing_zero_agreement(evidence):
    ranking = reliability_ranking(scored=cohort_scores(evidence=evidence))
    one_source = ranking.assign(sources=1, score_spread=0)
    empty = score_spread_chart(ranking=one_source)
    assert not empty.data
    assert "cannot establish agreement" in empty.layout.annotations[0].text
    mixed = ranking.copy()
    mixed.loc[mixed.index[:2], "sources"] = 1
    mixed.loc[:, "score_spread"] = 0
    figure = score_spread_chart(ranking=mixed)
    assert sum(len(trace.x) for trace in figure.data) == len(ranking) - 2
    assert figure.layout.yaxis.range[1] == 5  # Genuine equal source means stay visible.


def test_scenario_matrix_preserves_ties_ranks_and_full_comparison_population(evidence):
    scenarios = weighting_scenarios(
        scored=cohort_scores(evidence=evidence), min_cohorts=1
    )
    selected = ["toyota:test0", "toyota:test1"]
    figure = scenario_rank_chart(scenarios=scenarios, model_ids=selected, view="matrix")
    matrix = figure.data[0]
    assert matrix.type == "heatmap" and matrix.z.shape == (2, 3)
    assert list(matrix.z[:, 0]) == [1, 1]  # Ties are retained, not jittered.
    assert matrix.zmax == 6  # Selecting two rows must not rerank them as 1 and 2.
    for row, model in enumerate(selected):
        expected = scenarios.loc[scenarios.model_id.eq(model)].set_index(
            keys="scenario"
        )
        assert (
            list(matrix.z[row])
            == expected.reindex(index=list(matrix.x)).scenario_rank.tolist()
        )
    assert "6 comparable models" in figure.layout.title.text
    with pytest.raises(ValueError):
        scenario_rank_chart(scenarios=scenarios, model_ids=selected, view="unknown")
    mixed = scenarios.copy()
    mixed.loc[mixed.model_id.eq(selected[0]), "comparison_group_id"] = "other"
    with pytest.raises(ValueError, match="one complete"):
        scenario_rank_chart(scenarios=mixed, model_ids=selected)
    with pytest.raises(ValueError, match="one complete"):
        scenario_rank_chart(scenarios=scenarios.assign(sources=1), model_ids=selected)
    incomplete = scenarios.drop(index=scenarios.index[0])
    with pytest.raises(ValueError, match="complete"):
        scenario_rank_chart(scenarios=incomplete, model_ids=selected)


def test_distribution_boxes_align_and_keep_age_report_year_and_units_separate(evidence):
    original = evidence.copy(deep=True)
    figure = rate_distribution_chart(evidence=evidence)
    assert len(figure.data) == 4
    assert all(trace.orientation == "h" and trace.boxmean for trace in figure.data)
    assert all(trace.meta["observations"] == 6 for trace in figure.data)
    assert all(
        list(trace.q1) == [1.25]
        and list(trace.median) == [2.5]
        and list(trace.q3) == [3.75]
        for trace in figure.data
    )
    assert figure.layout.boxmode == "overlay"
    assert all("6 rows" in trace.name for trace in figure.data)
    assert figure.layout.yaxis.title.text == figure.layout.yaxis2.title.text == ""
    assert "per 1,000" in figure.layout.xaxis.title.text
    assert "Significant-defect" in figure.layout.xaxis2.title.text
    pooled = rate_distribution_chart(evidence=evidence, group_by_age=False)
    assert len(pooled.data) == 2 and all(
        trace.meta["observations"] == 12 for trace in pooled.data
    )
    extra = evidence.loc[evidence.source_id.eq("tuv")].assign(report_year=2026)
    separate = rate_distribution_chart(
        evidence=pd.concat(objs=[evidence, extra], ignore_index=True)
    )
    assert len(separate.layout.annotations) == 3
    pd.testing.assert_frame_equal(left=evidence, right=original)
    with pytest.raises(ValueError, match="boolean"):
        rate_distribution_chart(evidence=evidence, group_by_age="no")
    with pytest.raises(ValueError):
        rate_distribution_chart(evidence=evidence.assign(value=-1))


def test_outlier_tooltips_use_readable_metadata_and_unknown_sample_labels(evidence):
    frame = evidence.copy()
    frame.loc[
        frame.source_id.eq("adac") & frame.model_id.eq("toyota:test0"), "value"
    ] = 90
    figure = rate_distribution_chart(evidence=frame)
    outliers = [trace for trace in figure.data if trace.type == "scatter"]
    assert len(outliers) == 2 and all(list(trace.x) == [90] for trace in outliers)
    adac_boxes = [
        trace for trace in figure.data if trace.type == "box" and trace.xaxis == "x"
    ]
    assert len(adac_boxes) == 2 and all(
        list(trace.upperfence) == [4] for trace in adac_boxes
    )
    for trace in outliers:
        assert "Sample size unavailable" in trace.customdata[0]
        assert (
            "source_panel" not in trace.hovertemplate
            and "sample_size" not in trace.hovertemplate
        )
        assert "NaN" not in trace.customdata[0]


def test_rate_marks_and_wilson_labels_remain_distinct_and_keep_exact_values(evidence):
    tuv = evidence.loc[evidence.source_id.eq("tuv") & evidence.age_min.eq(4)].copy()
    tuv.loc[tuv.index[0], "make"] = "Volkswagen"
    rate_only = tuv.assign(sample_size=np.nan, event_count=np.nan)
    figure = source_rate_chart(evidence=rate_only, highlight_makes=["Volkswagen"])
    assert all(
        trace.marker.symbol == "circle-open" and trace.marker.line.width == 2
        for trace in figure.data
    )
    assert all(trace.error_x.array is None for trace in figure.data)
    assert all(" · age " not in label for trace in figure.data for label in trace.y)
    assert all(label.endswith("%") for trace in figure.data for label in trace.text)
    assert {trace.marker.color for trace in figure.data} == {"#0072B2", "#CCD2D8"}
    uniform = source_rate_chart(evidence=rate_only, colour_by="uniform")
    assert len(uniform.data) == 1 and not uniform.data[0].showlegend
    counted = counted_failure_rates(evidence=tuv)
    wilson = count_uncertainty_chart(
        counted=counted, colour_by="make", highlight_makes=["Volkswagen"]
    )
    assert all(" · age " not in label for trace in wilson.data for label in trace.y)
    assert all(trace.mode == "markers+text" for trace in wilson.data)
    assert sorted(value for trace in wilson.data for value in trace.x) == sorted(
        counted.counted_rate_pct
    )
    assert all("95% Wilson" in trace.hovertemplate for trace in wilson.data)


def test_single_source_app_explains_unavailable_diagnostics(
    evidence, tmp_path, monkeypatch
):
    save_snapshot(directory=tmp_path, evidence=evidence, events=empty_events())
    monkeypatch.setenv("BEST_CARS_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(script_path=APP, default_timeout=45).run()
    app.multiselect(key="filter_sources").set_value(v=["tuv"])
    app.number_input(key="filter_min_sources").set_value(v=1)
    app.number_input(key="filter_min_cohorts").set_value(v=1).run()
    app.radio(key="page").set_value(v="Deeper analysis").run()
    assert not app.exception
    assert any(
        "Weighting sensitivity requires at least two" in item.value for item in app.info
    )
    assert any(
        "disagreement is unavailable, not zero" in item.value for item in app.info
    )
    keys = {chart.key for chart in app.get("plotly_chart")}
    assert "deep_spread_plot" not in keys and "deep_scenario_chart_plot" not in keys
    app.multiselect(key="deep_published_rates_highlight").set_value(v=["Toyota"]).run()
    app.text_input(key="deep_published_rates_query").set_value(v="absent").run()
    assert not app.exception
    assert app.multiselect(key="deep_published_rates_highlight").value == []
    app.checkbox(key="deep_distribution_ages").set_value(v=False).run()
    assert not app.exception and any(
        "pool different ages" in item.value for item in app.warning
    )


def test_app_selects_one_scenario_group_and_resets_model_filters(
    evidence, tmp_path, monkeypatch
):
    second = evidence.copy()
    second["model_id"] = second.model_id.str.replace(
        pat="test", repl="extra", regex=False
    )
    second["model"] = second.model.str.replace(pat="Test", repl="Extra", regex=False)
    second["source_model"] = second.source_model.str.replace(
        pat="Test", repl="Extra", regex=False
    )
    extra = second.loc[second.age_min.eq(6)].copy()
    extra["age_min"] = extra["age_max"] = 8
    extra["cohort_year_start"] = extra["cohort_year_end"] = 2016
    extra["comparison_group"] = "age_8"
    combined = pd.concat(objs=[evidence, second, extra], ignore_index=True)
    save_snapshot(directory=tmp_path, evidence=combined, events=empty_events())
    monkeypatch.setenv("BEST_CARS_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(script_path=APP, default_timeout=45).run()
    app.radio(key="page").set_value(v="Deeper analysis").run()
    assert not app.exception
    assert len(app.selectbox(key="deep_scenario_group").options) == 2
    group_values = list(app.selectbox(key="deep_scenario_group").proto.options)
    assert all("6 comparable models" in label for label in group_values)
    chart = next(
        chart
        for chart in app.get("plotly_chart")
        if chart.key == "deep_scenario_chart_plot"
    )
    assert json.loads(chart.proto.spec)["data"][0]["type"] == "heatmap"
    first = app.multiselect(key="deep_scenario_models").options[0]
    app.multiselect(key="deep_scenario_models").select(v=first).run()
    app.selectbox(key="deep_scenario_group").select_index(index=1).run()
    assert not app.exception
    assert app.multiselect(key="deep_scenario_models").value == []
    app.selectbox(key="deep_scenario_layout").set_value(v="lines").run()
    assert not app.exception


def test_single_source_report_explains_unavailable_sensitivity(evidence):
    content = report_html(
        evidence=evidence.loc[evidence.source_id.eq("tuv")], events=empty_events()
    )
    soup = BeautifulSoup(markup=content, features="html.parser")
    assert "Weighting sensitivity requires at least two" in soup.get_text()
    assert "disagreement is unavailable, not zero" in soup.get_text()
    assert "Horizontal boxes separate published ages" in soup.get_text()
    assert "<script src=" not in content
