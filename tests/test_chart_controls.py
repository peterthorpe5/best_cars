"""Visual controls must preserve evidence, model trajectories and missingness."""

import numpy as np
import pandas as pd
import pytest

from best_cars.analysis import cohort_scores, reliability_ranking
from best_cars.charts import (
    _make_colours,
    age_chart,
    age_score_heatmap,
    count_uncertainty_chart,
    mileage_chart,
    ranking_chart,
    scenario_rank_chart,
    score_spread_chart,
)
from best_cars.interrogation import weighting_scenarios
from best_cars.statistics import counted_failure_rates


def test_heatmap_missing_layer_never_draws_a_dash_on_observed_values(evidence):
    scored = cohort_scores(evidence=evidence)
    one = scored.loc[scored.source_id.eq("tuv")].drop(index=0)
    for measure in ["relative", "rate"]:
        figure = age_score_heatmap(scored=one, measure=measure)
        missing = pd.isna(figure.data[0].z)
        text = np.asarray(figure.data[1].text)
        assert np.all(text[~missing] == "")
        assert np.all(text[missing] == "—")
        assert np.all(pd.isna(figure.data[1].z)[~missing])


def test_make_colours_stay_stable_after_filtering_and_sorting(evidence):
    colours = _make_colours(makes=["Toyota", "BMW", "Audi"])
    assert _make_colours(makes=["BMW"]) == {"BMW": colours["BMW"]}
    assert _make_colours(makes=["Audi", "BMW", "Toyota"]) == colours
    assert _make_colours(makes=[]) == {}
    scored = cohort_scores(evidence=evidence)
    ranking = reliability_ranking(scored=scored)
    chart = ranking_chart(ranking=ranking, top_n=4, colour_by="make")
    assert sum(len(trace.x) for trace in chart.data) == 4
    assert all(trace.name == "Toyota" for trace in chart.data)
    assert list(chart.layout.yaxis.categoryarray) == [
        f"Toyota {row.model}" for row in ranking.head(n=4).iloc[::-1].itertuples()
    ]
    with pytest.raises(ValueError):
        ranking_chart(ranking=ranking, colour_by="unknown")


def test_score_sized_mileage_keeps_unscored_models_and_real_zeroes(evidence):
    scored = cohort_scores(evidence=evidence)
    one = scored.loc[scored.source_id.eq("tuv")].copy()
    one["mileage_km"] = 50000
    one["mileage_basis"] = "mean_odometer"
    scores = pd.DataFrame(
        data={"model_id": ["toyota:test0", "toyota:test1"], "score": [100.0, 0.0]}
    )
    chart = mileage_chart(
        evidence=one, colour_by="make", model_scores=scores, size_by_score=True
    )
    assert sum(len(trace.x) for trace in chart.data) == len(one)
    sizes = np.concatenate([np.asarray(trace.marker.size) for trace in chart.data])
    assert sizes.max() == 25 and sizes.min() == 7
    assert all(
        trace.marker.sizemode == "diameter" and trace.marker.sizeref == 1
        for trace in chart.data
    )
    assert any(
        "No optional preference score" in str(trace.customdata) for trace in chart.data
    )
    assert any(trace.marker.symbol == "circle-open" for trace in chart.data)
    assert scores.score.tolist() == [100, 0]
    with pytest.raises(ValueError):
        mileage_chart(evidence=one, colour_by="unknown")
    with pytest.raises(ValueError):
        mileage_chart(evidence=one, size_by_score=True)
    with pytest.raises(ValueError):
        mileage_chart(evidence=one, model_scores=pd.concat(objs=[scores, scores]))
    with pytest.raises(ValueError):
        mileage_chart(evidence=one, model_scores=scores.assign(score=101))
    mixed = one.copy()
    mixed.loc[mixed.index[0], "report_year"] = 2026
    with pytest.raises(ValueError):
        mileage_chart(evidence=mixed)


def test_wilson_sort_colour_and_limit_preserve_interval_values(evidence):
    counted = counted_failure_rates(evidence=evidence)
    counted.loc[counted.model_id.eq("toyota:test0"), "make"] = "BMW"
    original = counted.copy(deep=True)
    chart = count_uncertainty_chart(
        counted=counted,
        top_n=8,
        sort_by="counted_rate_pct",
        ascending=True,
        colour_by="make",
    )
    assert sum(len(trace.x) for trace in chart.data) == 8
    assert set(trace.name for trace in chart.data) == {"Toyota", "BMW"}
    assert all(min(trace.error_x.array) >= 0 for trace in chart.data)
    assert chart.layout.yaxis.autorange == "reversed"
    pd.testing.assert_frame_equal(left=counted, right=original)
    original_order = count_uncertainty_chart(
        counted=counted, top_n=2, sort_by="original"
    )
    assert list(original_order.data[0].x) == counted.head(n=2).counted_rate_pct.tolist()
    for keyword in [{"sort_by": "unknown"}, {"colour_by": "unknown"}]:
        with pytest.raises(ValueError):
            count_uncertainty_chart(counted=counted, **keyword)


def test_same_make_models_keep_separate_age_and_scenario_trajectories(evidence):
    scored = cohort_scores(evidence=evidence)
    ids = ["toyota:test0", "toyota:test1"]
    age = age_chart(scored=scored, model_ids=ids, colour_by="make")
    assert len(age.data) == 4  # Two models in each of two source panels.
    assert all(len(trace.x) == 2 for trace in age.data)
    scenarios = weighting_scenarios(scored=scored, min_cohorts=1)
    figure = scenario_rank_chart(
        scenarios=scenarios, model_ids=ids, colour_by="make", size_by_score=True
    )
    assert len(figure.data) == 2
    assert all(len(trace.x) == 3 for trace in figure.data)
    assert figure.data[0].line.color == figure.data[1].line.color
    assert all(7 <= size <= 25 for trace in figure.data for size in trace.marker.size)
    assert not scenario_rank_chart(scenarios=scenarios, model_ids=[]).data
    for model_ids in [["absent"], [ids[0], ids[0]]]:
        with pytest.raises(ValueError):
            scenario_rank_chart(scenarios=scenarios, model_ids=model_ids)
    with pytest.raises(ValueError):
        scenario_rank_chart(
            scenarios=pd.concat(objs=[scenarios, scenarios]), model_ids=ids
        )
    with pytest.raises(ValueError):
        scenario_rank_chart(scenarios=scenarios.assign(score=101), model_ids=ids)
    with pytest.raises(ValueError):
        age_chart(scored=scored, model_ids=ids, colour_by="unknown")
    ranking = reliability_ranking(scored=scored)
    spread = score_spread_chart(ranking=ranking, colour_by="make", size_by_score=True)
    assert sum(len(trace.x) for trace in spread.data) == len(ranking)
    with pytest.raises(ValueError):
        score_spread_chart(ranking=ranking, colour_by="unknown")
