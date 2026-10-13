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
        "Largest source disagreement",
        "Alphabetical source profiles (up to 80)",
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
    assert app.selectbox(key="find_cohort").value
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


def test_electric_categories_empty_and_deeper_scenario_controls(running_app):
    app = running_app
    app.radio(key="page").set_value(v="Electric and hybrid vehicles").run()
    for category in ["Battery electric", "Plug-in hybrid", "Hybrid"]:
        app.selectbox(key="electric_category").set_value(v=category).run()
        assert not app.exception and app.info
    app.checkbox(key="electric_mixed").set_value(v=True).run()
    assert not app.exception
    app.radio(key="page").set_value(v="Deeper analysis").run()
    app.selectbox(key="deep_scenario_order").set_value(v="Largest rank movement").run()
    app.selectbox(key="deep_peer_model").set_value(v="toyota:test0").run()
    assert not app.exception


def test_electric_populated_page_and_mixed_family_opt_in(
    evidence, tmp_path, monkeypatch
):
    evidence["model_id"] = evidence.model_id.replace(
        to_replace={
            "toyota:test0": "tesla:model 3",
            "toyota:test1": "toyota:prius plug in hybrid",
            "toyota:test2": "toyota:prius plus",
            "toyota:test3": "toyota:corolla",
        }
    )
    save_snapshot(directory=tmp_path, evidence=evidence, events=empty_events())
    monkeypatch.setenv("BEST_CARS_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(script_path=APP, default_timeout=45).run()
    app.radio(key="page").set_value(v="Electric and hybrid vehicles").run()
    assert not app.exception
    assert app.multiselect(key="electric_age_models").value == ["tesla:model 3"]
    for category in ["Plug-in hybrid", "Hybrid"]:
        app.selectbox(key="electric_category").set_value(v=category).run()
        assert not app.exception
    app.checkbox(key="electric_mixed").set_value(v=True).run()
    assert set(app.multiselect(key="electric_age_models").options) == {
        "Toyota Test2",
        "Toyota Test3",
    }
    assert not app.exception
    app.radio(key="page").set_value(v="Deeper analysis").run()
    app.selectbox(key="deep_scenario_order").set_value(v="Largest rank movement").run()
    app.selectbox(key="deep_peer_model").set_value(v="tesla:model 3").run()
    assert not app.exception


def test_ranking_wilson_and_scenario_visual_controls(running_app):
    app = running_app
    app.radio(key="page").set_value(v="Find reliable cars").run()
    app.selectbox(key="ranking_view").set_value(v="tuv").run()
    app.number_input(key="find_top_n").set_value(v=3).run()
    app.selectbox(key="find_colour").set_value(v="make").run()
    assert not app.exception
    app.radio(key="page").set_value(v="Deeper analysis").run()
    app.number_input(key="deep_scenario_count").set_value(v=6)
    app.selectbox(key="deep_scenario_colour").set_value(v="make")
    app.checkbox(key="deep_scenario_size").set_value(v=True).run()
    app.multiselect(key="deep_scenario_models").set_value(
        v=["toyota:test0", "toyota:test1"]
    ).run()
    assert not app.exception
    app.selectbox(key="deep_count_sort").set_value(v="counted_rate_pct")
    app.checkbox(key="deep_count_ascending").set_value(v=True)
    app.selectbox(key="deep_count_colour").set_value(v="make")
    app.number_input(key="deep_count_limit").set_value(v=12).run()
    app.multiselect(key="deep_count_models").set_value(v=["toyota:test0"]).run()
    app.slider(key="deep_count_age").set_value(v=(30, 40)).run()
    assert not app.exception and app.info
    app.selectbox(key="deep_spread_colour").set_value(v="make")
    app.checkbox(key="deep_spread_size").set_value(v=True).run()
    assert not app.exception


def test_mileage_explorer_discovery_sizing_and_empty_filters(
    evidence, tmp_path, monkeypatch
):
    mask = evidence.source_id.eq("tuv")
    evidence.loc[mask, "mileage_km"] = 50000
    evidence.loc[mask, "mileage_basis"] = "mean_odometer"
    save_snapshot(directory=tmp_path, evidence=evidence, events=empty_events())
    monkeypatch.setenv("BEST_CARS_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(script_path=APP, default_timeout=45).run()
    app.radio(key="page").set_value(v="Mileage explorer").run()
    assert app.selectbox(key="mileage_source").value == "tuv"
    app.selectbox(key="mileage_colour").set_value(v="make")
    app.checkbox(key="mileage_size_score").set_value(v=True).run()
    app.multiselect(key="mileage_makes").set_value(v=["Toyota"]).run()
    assert not app.exception
    app.text_input(key="mileage_query").set_value(v="absent").run()
    assert not app.exception and app.info
    app.text_input(key="mileage_query").set_value(v="").run()
    app.slider(key="mileage_age_range").set_value(v=(30, 40)).run()
    assert not app.exception and app.info


def test_source_first_defaults_profiles_and_repair_context(running_app):
    app = running_app
    app.text_input(key="overview_profile_query").set_value(v="Test0").run()
    profiles = app.dataframe[-1].value
    assert len(profiles) == 2 and set(profiles.source_id) == {"tuv", "adac"}
    app.radio(key="page").set_value(v="Find reliable cars").run()
    assert app.selectbox(key="ranking_view").value == "choose"
    assert not app.get("plotly_chart")
    app.radio(key="page").set_value(v="Data sources and fault definitions").run()
    for source in [
        "TÜV inspection defects",
        "ADAC roadside breakdowns",
        "Traficom Finnish inspections",
        "UK MOT import",
        "NHTSA safety events",
    ]:
        app.selectbox(key="source_definition").set_value(v=source).run()
        assert not app.exception
    app.radio(key="page").set_value(v="Repair costs").run()
    assert not app.exception
    assert any(item.value == "£0.00" for item in app.metric)
    assert any("annual" in item.value.lower() for item in app.caption)
