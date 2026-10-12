"""Exercise explicit app refresh/lookup actions with deterministic transports."""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import best_cars.app as app_module
from best_cars.parsers import parse_nhtsa_json
from best_cars.storage import save_snapshot

APP = Path(__file__).parents[1] / "app.py"


@pytest.mark.parametrize("failure", [False, True])
def test_safety_lookup_success_failure_and_saved_event_filter(
    evidence, tmp_path, monkeypatch, failure
):
    events = parse_nhtsa_json(
        payload={
            "Count": 1,
            "results": [
                {
                    "NHTSACampaignNumber": "20V123000",
                    "Component": "Fuel system",
                    "Summary": "Published defect",
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

    def lookup(**kwargs):
        if failure:
            raise OSError("Source unavailable")
        return events, tmp_path

    monkeypatch.setattr(app_module, "nhtsa_lookup", lookup)
    app = AppTest.from_file(script_path=APP, default_timeout=30).run()
    app.radio(key="page").set_value(v="Recalls and complaints").run()
    [
        button
        for button in app.button
        if button.label == "Retrieve official NHTSA records"
    ][0].click().run()
    assert not app.exception
    assert len(app.error if failure else app.success) == 1
    app.text_input(key="events_query").set_value(v="no_such_component").run()
    assert not app.exception


@pytest.mark.parametrize("status", ["updated", "failed", "exception"])
def test_refresh_action_statuses(evidence, tmp_path, monkeypatch, status):
    from best_cars.schema import empty_events

    save_snapshot(directory=tmp_path, evidence=evidence, events=empty_events())
    monkeypatch.setenv("BEST_CARS_DATA_DIR", str(tmp_path))

    def refresh(**kwargs):
        if status == "exception":
            raise ValueError("No sources selected")
        return pd.DataFrame(
            data=[{"source_id": "adac", "status": status, "rows": 0}]
        ), tmp_path

    monkeypatch.setattr(app_module, "refresh_sources", refresh)
    app = AppTest.from_file(script_path=APP, default_timeout=30).run()
    app.radio(key="page").set_value(v="Data management").run()
    app.button(key="refresh_run").click().run()
    assert not app.exception
    if status == "exception":
        assert len(app.error) == 1
    elif status == "failed":
        assert len(app.warning) == 1
    else:
        assert len(app.success) == 1
