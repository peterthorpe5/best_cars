"""Exercise every Streamlit page and key filter edge cases without network I/O."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from best_cars.app import PAGES
from best_cars.schema import empty_events
from best_cars.storage import save_snapshot

APP = Path(__file__).parents[1] / "app.py"


@pytest.mark.parametrize("page", PAGES)
def test_all_app_pages(page, evidence, tmp_path, monkeypatch):
    save_snapshot(directory=tmp_path, evidence=evidence, events=empty_events())
    monkeypatch.setenv("BEST_CARS_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(script_path=APP, default_timeout=30).run()
    app.radio(key="page").set_value(v=page).run()
    assert not app.exception, [exception.message for exception in app.exception]
    assert len(app.title) >= 1


def test_no_sources_and_zero_weights_do_not_crash(evidence, tmp_path, monkeypatch):
    save_snapshot(directory=tmp_path, evidence=evidence, events=empty_events())
    monkeypatch.setenv("BEST_CARS_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(script_path=APP, default_timeout=30).run()
    app.multiselect(key="filter_sources").set_value(v=[]).run()
    assert not app.exception
    app.multiselect(key="filter_sources").set_value(v=["tuv", "adac"]).run()
    app.slider(key="weight_tuv").set_value(v=0.0)
    app.slider(key="weight_adac").set_value(v=0.0).run()
    assert not app.exception and len(app.warning) >= 1


def test_search_no_match_and_empty_comparison(evidence, tmp_path, monkeypatch):
    save_snapshot(directory=tmp_path, evidence=evidence, events=empty_events())
    monkeypatch.setenv("BEST_CARS_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(script_path=APP, default_timeout=30).run()
    app.radio(key="page").set_value(v="Find reliable cars").run()
    app.text_input(key="find_query").set_value(v="nothing_matches_this").run()
    assert not app.exception and len(app.info) >= 1
    app.radio(key="page").set_value(v="Compare cars").run()
    app.multiselect(key="compare_models").set_value(v=[]).run()
    assert not app.exception and len(app.info) >= 1


def test_corrupt_snapshot_is_reported(tmp_path, monkeypatch):
    (tmp_path / "current.json").write_text(data='{"snapshot":"../outside"}')
    monkeypatch.setenv("BEST_CARS_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(script_path=APP, default_timeout=30).run()
    assert not app.exception and len(app.error) == 1


def test_registration_filter_and_shortlist_survive_display_search(
    evidence, tmp_path, monkeypatch
):
    save_snapshot(directory=tmp_path, evidence=evidence, events=empty_events())
    monkeypatch.setenv("BEST_CARS_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(script_path=APP, default_timeout=30).run()
    app.checkbox(key="filter_registration").set_value(v=True).run()
    app.slider(key="filter_cohort_years").set_value(v=(2018, 2020)).run()
    app.radio(key="page").set_value(v="Find reliable cars").run()
    app.multiselect(key="saved_shortlist").set_value(v=["toyota:test0"]).run()
    app.text_input(key="find_query").set_value(v="no_model_matches").run()
    assert app.multiselect(key="saved_shortlist").value == ["toyota:test0"]
    assert not app.exception
