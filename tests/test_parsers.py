"""Regression tests for source-specific units, gaps and numeric structures."""

import pandas as pd
import pytest

from best_cars.parsers import (
    jsonstat_records,
    parse_adac_pdf,
    parse_adac_words,
    parse_nhtsa_json,
    parse_traficom_json,
    parse_traficom_xlsx,
    parse_tuv_html,
    parse_tuv_tsv,
    utc_now,
)

NOW = "2026-10-12T10:00:00+00:00"


def test_legacy_tuv_keeps_unknown_denominator_and_approximate_cohort():
    frame = pd.DataFrame(
        data=[
            {
                "year": 2025,
                "age_group": "2-3",
                "rank": 1,
                "make_model": "Honda Jazz",
                "faults_pct": 2.4,
                "odometer_thousand_km": 28,
            }
        ]
    )
    result = parse_tuv_tsv(frame=frame, retrieved_at=NOW)
    row = result.iloc[0]
    assert row.value == 2.4 and row.unit == "percent"
    assert row.mileage_km == 28_000
    assert pd.isna(row.sample_size)
    assert row.cohort_year_start == 2021 and row.cohort_year_end == 2022
    assert "approximate" in row.notes
    frame.loc[0, "age_group"] = "young"
    with pytest.raises(ValueError):
        parse_tuv_tsv(frame=frame, retrieved_at=NOW)
    with pytest.raises(ValueError):
        parse_tuv_tsv(frame=pd.DataFrame(), retrieved_at=NOW)


def test_tuv_html_ties_and_decimal_commas():
    html = "<body>Order<br>Make/Model<br>Faults (%)<br>Odometer (thous. km)<br>"
    for index in range(6):
        rank = f"{index + 1}<br>" if index != 1 else ""
        html += f"{rank}Toyota Test{index}<br>2,{index}<br>30<br>"
    html += "Source: TÜV</body>"
    result = parse_tuv_html(
        html=html, report_year=2025, age_group="2-3", retrieved_at=NOW
    )
    assert len(result) == 6
    assert result.source_rank.iloc[1] == 1
    assert result.value.iloc[1] == 2.1
    for bad in ["<html>Access denied</html>", html.replace("2,3", "broken")]:
        with pytest.raises(ValueError):
            parse_tuv_html(
                html=bad, report_year=2025, age_group="2-3", retrieved_at=NOW
            )


def test_jsonstat_sparse_suppressed_and_dimension_checks():
    payload = {
        "id": ["year", "info"],
        "size": [1, 2],
        "dimension": {
            "year": {"category": {"index": {"2025": 0}, "label": {"2025": "2025"}}},
            "info": {
                "category": {"index": ["a", "b"], "label": {"a": "Count", "b": "Rate"}}
            },
        },
        "value": {"0": 100},
    }
    records = jsonstat_records(payload=payload)
    assert records[0]["info"] == "Count" and records[0]["value"] == 100
    assert pd.isna(records[1]["value"])
    payload["value"] = [100, ".."]
    assert pd.isna(jsonstat_records(payload=payload)[1]["value"])
    payload["value"] = [100]
    with pytest.raises(ValueError):
        jsonstat_records(payload=payload)
    with pytest.raises(ValueError):
        jsonstat_records(payload={})


def traficom_payload():
    """Return a tiny schema-accurate PxWeb response with a suppressed cohort."""
    axes = {
        "Katsastusvuosi": ["2025"],
        "Merkki ja mallisarja": [
            "Toyota Yaris",
            "Makes in total - Models in total",
            "Honda Jazz",
        ],
        "Käyttöönottovuosi": ["2021"],
        "Tiedot": [
            "Number of periodic inspections",
            "Average mileage",
            "Fail-rate",
            "Failed",
        ],
    }
    return {
        "id": list(axes),
        "size": [len(values) for values in axes.values()],
        "dimension": {
            code: {
                "category": {
                    "index": {v: i for i, v in enumerate(values)},
                    "label": {v: v for v in values},
                }
            }
            for code, values in axes.items()
        },
        "value": [100, 42000, 5, 5, 1000, 30000, 3, 30, None, None, None, None],
    }


def test_traficom_exact_test_counts_and_totals():
    result = parse_traficom_json(payload=traficom_payload(), retrieved_at=NOW)
    assert len(result) == 1
    row = result.iloc[0]
    assert row.report_year == 2026 and row.observation_year == 2025
    assert row.sample_size == 100 and row.event_count == 5 and row.value == 5
    assert row.age_min == 4
    with pytest.raises(ValueError):
        parse_traficom_json(
            payload={
                "id": ["unexpected"],
                "size": [1],
                "dimension": {"unexpected": {"category": {"index": ["a"]}}},
                "value": [1],
            },
            retrieved_at=NOW,
        )


def test_traficom_workbook(tmp_path):
    rows = [
        [None] * 10,
        [
            "Käyttöönottovuosi",
            "Merkki",
            "Malli",
            "count",
            "fail",
            "mean",
            "median",
            "fault1",
            "fault2",
            "fault3",
        ],
        [2021, "Toyota", "YARIS", 100, 5, 42000, 40000, "Jarrut", None, None],
        [2021, "Toyota", "Mallit yhteensä", 1000, 5, 42000, 40000, None, None, None],
    ]
    path = tmp_path / "source.xlsx"
    pd.DataFrame(data=rows).to_excel(excel_writer=path, header=False, index=False)
    result = parse_traficom_xlsx(path=path, observation_year=2025, retrieved_at=NOW)
    assert len(result) == 1 and result.iloc[0].mileage_km == 42000
    assert pd.isna(result.iloc[0].event_count)
    assert "Jarrut" in result.iloc[0].notes
    pd.DataFrame(data=[["Unexpected"]]).to_excel(
        excel_writer=path, header=False, index=False
    )
    with pytest.raises(ValueError):
        parse_traficom_xlsx(path=path, observation_year=2025, retrieved_at=NOW)


def adac_words(*, label="Toyota RAV4", top=110):
    """Return fixed-coordinate words with one missing rate and a mileage cell."""
    words = [{"text": "Laufleistung", "x0": 180, "x1": 215, "top": top - 10}]
    for index, year in enumerate(range(2022, 2014, -1)):
        centre = 250 + index * 25
        words.append(
            {"text": str(year), "x0": centre - 9, "x1": centre + 9, "top": top - 10}
        )
        if index != 1:
            words.append(
                {
                    "text": str(index + 1).replace(".", ","),
                    "x0": centre - 5,
                    "x1": centre + 5,
                    "top": top,
                }
            )
    for index, word in enumerate(label.split()):
        words.append(
            {"text": word, "x0": 60 + index * 40, "x1": 95 + index * 40, "top": top}
        )
    words.append({"text": "12", "x0": 195, "x1": 205, "top": top})
    return words


def test_adac_empty_cells_do_not_shift_years_or_mileage():
    result = parse_adac_words(
        pages=[adac_words()],
        report_year=2025,
        retrieved_at=NOW,
        source_url="https://example.org/report.pdf",
    )
    assert len(result) == 7
    assert 2021 not in set(result.cohort_year_start)
    assert result.loc[result.cohort_year_start.eq(2020), "value"].iloc[0] == 3
    assert result.value.max() == 8
    assert result.sample_size.isna().all() and result.mileage_km.isna().all()
    with pytest.raises(ValueError):
        parse_adac_words(
            pages=[[]],
            report_year=2025,
            retrieved_at=NOW,
            source_url="https://example.org/report.pdf",
        )


def test_adac_pdf_wrapper(tmp_path, monkeypatch):
    class Page:
        def extract_words(self, *, x_tolerance):
            return adac_words()

    class Document:
        pages = [Page()]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(
        sys.modules, "pdfplumber", SimpleNamespace(open=lambda **kwargs: Document())
    )
    assert (
        len(
            parse_adac_pdf(
                path=tmp_path / "input.pdf", report_year=2025, retrieved_at=NOW
            )
        )
        == 7
    )


def test_nhtsa_events_are_context_with_duplicates_removed():
    recall = {
        "NHTSACampaignNumber": "20V123000",
        "Summary": "A fuel pump fault",
        "Component": "Fuel system",
        "Remedy": "Replace the pump",
    }
    rows = parse_nhtsa_json(
        payload={"Count": 2, "results": [recall, recall]},
        event_type="recall",
        make="Toyota",
        model="Corolla",
        model_year=2020,
        source_url="https://api.nhtsa.gov/example",
        retrieved_at=NOW,
    )
    assert len(rows) == 1 and rows.iloc[0].country == "United States"
    complaint = {
        "odiNumber": 1234567,
        "summary": "A reported fault",
        "components": "BRAKES",
    }
    result = parse_nhtsa_json(
        payload={"Count": 1, "results": [complaint]},
        event_type="complaint",
        make="Toyota",
        model="Corolla",
        model_year=2020,
        source_url="https://api.nhtsa.gov/example",
        retrieved_at=NOW,
    )
    assert result.iloc[0].event_id == "1234567"
    assert result.iloc[0].summary == "A reported fault"
    lower_case = parse_nhtsa_json(
        payload={"count": 1, "results": [complaint]},
        event_type="complaint",
        make="Toyota",
        model="Corolla",
        model_year=2020,
        source_url="https://api.nhtsa.gov/example",
        retrieved_at=NOW,
    )
    assert len(lower_case) == 1
    assert parse_nhtsa_json(
        payload={"Count": 0, "results": []},
        event_type="recall",
        make="a",
        model="b",
        model_year=2020,
        source_url="https://example.org",
        retrieved_at=NOW,
    ).empty


@pytest.mark.parametrize(
    "payload,event_type",
    [
        ({}, "recall"),
        ({"Count": 1, "results": []}, "recall"),
        ({"Count": 1, "results": [{}]}, "recall"),
        ({"Count": 0, "results": []}, "unknown"),
    ],
)
def test_nhtsa_invalid_envelopes(payload, event_type):
    with pytest.raises(ValueError):
        parse_nhtsa_json(
            payload=payload,
            event_type=event_type,
            make="Toyota",
            model="Corolla",
            model_year=2020,
            source_url="https://example.org",
            retrieved_at=NOW,
        )


def test_timestamp_is_parseable_utc():
    value = pd.to_datetime(arg=utc_now(), utc=True)
    assert value.year >= 2026
