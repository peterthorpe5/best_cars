"""Exercise deeper analysis and screenshot-driven UI interactions."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from best_cars.parsers import parse_nhtsa_json
from best_cars.schema import empty_events
from best_cars.storage import save_snapshot

APP = Path(__file__).parents[1] / "app.py"


@pytest.fixture
def running_app(evidence, tmp_path, monkeypatch):
    """Render a deterministic app with two sources and known counts."""
    save_snapshot(directory=tmp_path, evidence=evidence, events=empty_events())
    monkeypatch.setenv("BEST_CARS_DATA_DIR", str(tmp_path))
    return AppTest.from_file(script_path=APP, default_timeout=45).run()


def test_heatmap_views_search_and_age_ordering(running_app):
    app = running_app
    app.radio(key="page").set_value(v="Evidence agreement").run()
    for view in [
        "Weakest relative evidence",
        "Largest source disagreement",
        "Strongest relative evidence",
        "All qualifying models (up to 80)",
    ]:
        app.selectbox(key="heatmap_view").set_value(v=view).run()
        assert not app.exception
    app.text_input(key="heatmap_query").set_value(v="absent").run()
    assert not app.exception and app.info
    app.radio(key="page").set_value(v="Age and mileage").run()
    app.selectbox(key="age_heatmap_order").set_value(v="weakest").run()
    app.text_input(key="age_heatmap_query").set_value(v="absent").run()
    assert not app.exception


def test_shared_cohort_switch_and_deeper_analysis_missing_and_zero_weights(running_app):
    app = running_app
    app.radio(key="page").set_value(v="Compare cars").run()
    app.checkbox(key="comparison_shared").set_value(v=True).run()
    assert not app.exception
    app.multiselect(key="compare_models").set_value(v=["toyota:test0"]).run()
    assert not app.exception
    app.radio(key="page").set_value(v="Deeper analysis").run()
    app.text_input(key="sensitivity_query").set_value(v="Test0").run()
    assert not app.exception
    app.slider(key="weight_adac").set_value(v=0.0)
    app.slider(key="weight_tuv").set_value(v=0.0).run()
    assert not app.exception
    app.multiselect(key="filter_sources").set_value(v=[]).run()
    assert not app.exception


def test_saved_safety_type_filter_and_full_event(evidence, tmp_path, monkeypatch):
    events = parse_nhtsa_json(
        payload={
            "Count": 1,
            "results": [
                {
                    "NHTSACampaignNumber": "20V123000",
                    "Component": "Fuel system",
                    "Summary": "Long readable recall summary",
                    "Remedy": "Repair",
                }
            ],
        },
        event_type="recall",
        make="Toyota",
        model="Corolla",
        model_year=2020,
        source_url="https://api.nhtsa.gov/data",
        retrieved_at="2026-10-12T10:00:00+00:00",
    )
    save_snapshot(directory=tmp_path, evidence=evidence, events=events)
    monkeypatch.setenv("BEST_CARS_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(script_path=APP, default_timeout=45).run()
    app.radio(key="page").set_value(v="Recalls and complaints").run()
    assert any("Long readable" in item.value for item in app.text)
    app.multiselect(key="events_types").set_value(v=["complaint"]).run()
    assert not app.exception
    app.multiselect(key="events_types").set_value(v=[]).run()
    assert not app.exception


def test_single_source_rankings_and_model_diagnostic(running_app):
    app = running_app
    app.radio(key="page").set_value(v="Find reliable cars").run()
    assert app.selectbox(key="inspect_model").value
    app.selectbox(key="ranking_view").set_value(v="tuv").run()
    app.number_input(key="single_source_cohorts").set_value(v=1).run()
    assert not app.exception
    app.text_input(key="find_query").set_value(v="Test0").run()
    assert app.selectbox(key="inspect_model").value == "toyota:test0"
    assert not app.exception


def test_weighting_presets_and_manual_override(running_app):
    app = running_app
    app.selectbox(key="weight_preset").set_value(
        v="Emphasise avoiding breakdowns"
    ).run()
    assert app.slider(key="weight_adac").value == 2.0
    app.selectbox(key="weight_preset").set_value(
        v="Emphasise inspection condition"
    ).run()
    assert app.slider(key="weight_adac").value == 0.5
    app.slider(key="weight_adac").set_value(v=0.75).run()
    assert app.slider(key="weight_adac").value == 0.75
    app.selectbox(key="weight_preset").set_value(v="Balanced").run()
    assert app.slider(key="weight_adac").value == 1.0
    assert not app.exception
