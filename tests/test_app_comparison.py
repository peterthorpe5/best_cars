"""Exercise expanded comparison views, historical context and missing evidence."""

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from best_cars.schema import empty_events
from best_cars.storage import save_snapshot

APP = Path(__file__).parents[1] / "app.py"


def test_comparison_tabs_single_model_history_and_matched_reference(
    evidence, tmp_path, monkeypatch
):
    old = evidence.assign(report_year=2024, observation_year=2023).copy()
    old["cohort_year_start"] -= 1
    old["cohort_year_end"] -= 1
    old_registration = evidence.assign(report_year=2024, observation_year=2023).copy()
    old_registration["age_min"] -= 1
    old_registration["age_max"] -= 1
    old_registration["comparison_group"] = (
        "registration_" + old_registration.cohort_year_start.astype(str)
    )
    save_snapshot(
        directory=tmp_path,
        evidence=pd.concat(objs=[old, old_registration, evidence]),
        events=empty_events(),
    )
    monkeypatch.setenv("BEST_CARS_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(script_path=APP, default_timeout=45).run()
    app.radio(key="page").set_value(v="Compare cars").run()
    assert len(app.tabs) == 6 and not app.exception
    app.multiselect(key="compare_models").set_value(v=["toyota:test0"]).run()
    assert not app.exception and app.multiselect(key="comparison_time_series").options
    assert any("two ungrouped" in item.value for item in app.info)
    app.checkbox(key="comparison_relative").set_value(v=True).run()
    app.selectbox(key="comparison_time_mode").set_value(v="registration").run()
    app.multiselect(key="comparison_time_series").set_value(v=[]).run()
    assert not app.exception
    app.checkbox(key="comparison_history").set_value(v=False).run()
    assert any("two loaded observation years" in item.value for item in app.info)
    app.checkbox(key="comparison_history").set_value(v=True).run()
    app.multiselect(key="compare_models").set_value(
        v=["toyota:test0", "toyota:test1"]
    ).run()
    app.selectbox(key="comparison_reference").set_value(v="toyota:test1").run()
    app.selectbox(key="comparison_colour").set_value(v="make").run()
    app.selectbox(key="comparison_count_order").set_value(v="counted_rate_pct").run()
    app.checkbox(key="comparison_count_ascending").set_value(v=False).run()
    assert app.selectbox(key="comparison_count_chart_grouping").value == "exact"
    assert len(app.multiselect(key="comparison_count_chart_panels").value) == 2
    app.selectbox(key="comparison_count_chart_grouping").set_value(v="two_year").run()
    app.selectbox(key="comparison_count_chart_grouping").set_value(v="mixed").run()
    app.selectbox(key="comparison_count_chart_grouping").set_value(v="exact").run()
    app.checkbox(key="comparison_shared").set_value(v=True).run()
    assert not app.exception and len(app.get("plotly_chart")) >= 6
    app.multiselect(key="comparison_sources").set_value(v=["adac"]).run()
    assert any("count pairs" in item.value for item in app.info)
    app.multiselect(key="comparison_sources").set_value(v=[]).run()
    assert not app.exception and any(
        "comparison chart filters" in item.value for item in app.info
    )


def test_deeper_wilson_panels_default_to_separate_ages(evidence, tmp_path, monkeypatch):
    save_snapshot(directory=tmp_path, evidence=evidence, events=empty_events())
    monkeypatch.setenv("BEST_CARS_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(script_path=APP, default_timeout=45).run()
    app.radio(key="page").set_value(v="Deeper analysis").run()
    assert app.selectbox(key="deep_counts_grouping").value == "exact"
    assert len(app.multiselect(key="deep_counts_panels").value) == 2
    app.selectbox(key="deep_counts_grouping").set_value(v="two_year").run()
    app.multiselect(key="deep_counts_panels").set_value(v=[]).run()
    assert not app.exception
    app.selectbox(key="deep_counts_grouping").set_value(v="mixed").run()
    assert len(app.multiselect(key="deep_counts_panels").value) == 1
    app.slider(key="deep_count_age").set_value(v=(4, 4)).run()
    app.selectbox(key="deep_counts_grouping").set_value(v="exact").run()
    assert len(app.multiselect(key="deep_counts_panels").value) == 1
    assert not app.exception


def test_empty_shared_support_retains_coverage_and_historical_explanation(
    evidence, tmp_path, monkeypatch
):
    changed = evidence.copy()
    changed.loc[changed.model_id.eq("toyota:test1"), "cohort_year_end"] += 1
    save_snapshot(directory=tmp_path, evidence=changed, events=empty_events())
    monkeypatch.setenv("BEST_CARS_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(script_path=APP, default_timeout=45).run()
    app.radio(key="page").set_value(v="Compare cars").run()
    app.multiselect(key="compare_models").set_value(
        v=["toyota:test0", "toyota:test1"]
    ).run()
    app.checkbox(key="comparison_shared").set_value(v=True).run()
    assert not app.exception
    assert any("no shared published cohorts" in item.value for item in app.info)
    app.checkbox(key="comparison_shared").set_value(v=False).run()
    app.slider(key="comparison_ages").set_value(v=(30, 40)).run()
    assert not app.exception and any(
        "comparison chart filters" in item.value for item in app.info
    )
