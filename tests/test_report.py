"""Offline report exports must contain actual source dates and escaping."""

import pandas as pd
import pytest
from bs4 import BeautifulSoup

from best_cars.report import main, report_html, report_table
from best_cars.schema import empty_events, empty_evidence
from best_cars.storage import save_snapshot


def test_report_html_includes_real_coverage_and_explanations(evidence):
    content = report_html(evidence=evidence, events=empty_events())
    assert "24" in content and "Rate observations" in content
    assert "observation year" in content and "2025" in content
    assert "not failure probabilities" in content
    assert "plotly" in content.lower()
    assert "<script src=" not in content
    soup = BeautifulSoup(markup=content, features="html.parser")
    assert len(soup.select(".plotly-graph-div")) == 9
    assert soup.select_one("#rate-summary table") is not None
    assert soup.select_one("#counted-rates table") is not None
    assert soup.select_one("#agreement-summary table") is not None
    assert soup.select_one("#peer-benchmarks table") is not None
    assert soup.select_one("#weight-scenarios table") is not None
    assert soup.select_one("#electrified") is not None
    assert soup.select_one("#model-ranking input[type=search]") is not None
    assert "No lifespan estimate has been calculated" in soup.get_text()
    assert "95% Wilson" in soup.get_text()


def test_searchable_table_escapes_labels_and_keeps_unknowns():
    frame = pd.DataFrame(
        data={"model": ['<script>alert("x")</script>'], "sample_size": [float("nan")]}
    )
    content = report_table(frame=frame, table_id='rates" onclick="bad')
    soup = BeautifulSoup(markup=content, features="html.parser")
    assert not soup.find_all(name="script")
    assert "Unavailable" in soup.get_text()
    assert soup.find(name="div").get("onclick") is None
    assert soup.find(name="input").get("type") == "search"
    assert "No observations" in report_table(frame=pd.DataFrame(), table_id="empty")


def test_report_handles_empty_selection_and_odometer_panel(evidence):
    content = report_html(evidence=empty_evidence(), events=empty_events())
    assert "No observations meet this selection" in content
    frame = evidence.copy()
    mask = frame.source_id.eq("tuv")
    frame.loc[mask, "mileage_km"] = 40000
    frame.loc[mask, "mileage_basis"] = "mean_odometer"
    content = report_html(evidence=frame, events=empty_events())
    soup = BeautifulSoup(markup=content, features="html.parser")
    assert len(soup.select("#mileage .plotly-graph-div")) == 1


def test_report_export_and_bad_extension(evidence, tmp_path):
    save_snapshot(directory=tmp_path / "data", evidence=evidence, events=empty_events())
    output = tmp_path / "report.html"
    assert (
        main(argv=["--data-dir", str(tmp_path / "data"), "--output", str(output)]) == 0
    )
    assert output.read_text().startswith("<!doctype html>")
    with pytest.raises(SystemExit):
        main(argv=["--output", str(tmp_path / "bad.txt")])
