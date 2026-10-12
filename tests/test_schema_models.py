"""Tests for conservative model matching and fail-closed data validation."""

import numpy as np
import pytest

from best_cars.models import identify_model, normalise_text
from best_cars.schema import as_number, empty_events, validate_evidence


@pytest.mark.parametrize(
    "label,expected",
    [
        ("VW Golf", "volkswagen:golf"),
        ("Škoda Karoq", "skoda:karoq"),
        ("Mercedes-Benz B-Klasse", "mercedes benz:b class"),
        ("Mercedes-Benz B-Class", "mercedes benz:b class"),
        ("BMW 3er-Reihe", "bmw:3 series"),
        ("BMW 3-sarja", "bmw:3 series"),
        ("Toyota RAV 4", "toyota:rav4"),
        ("Hyundai IONIQ5", "hyundai:ioniq 5"),
        ("VW Nutzfahrzeuge Caddy", "volkswagen:caddy"),
    ],
)
def test_explicit_model_aliases(label, expected):
    assert identify_model(label=label).model_id == expected


def test_grouped_and_unspecified_models_stay_distinct():
    assert identify_model(label="KIA Ceed/XCeed").grouped
    assert identify_model(label="Mini").grouped
    assert identify_model(
        label="BMW 2er-Reihe (inkl. Grand- und Active-Tourer)"
    ).grouped
    assert (
        identify_model(label="Opel Corsa").model_id
        != identify_model(label="Vauxhall Corsa").model_id
    )
    assert (
        identify_model(label="Mazda 2", make_hint=None).model_id
        == identify_model(label="2", make_hint="Mazda").model_id
    )


@pytest.mark.parametrize("label", [None, "", "UnknownModel"])
def test_invalid_labels(label):
    with pytest.raises(ValueError):
        identify_model(label=label)


def test_text_normalisation():
    assert normalise_text(text=" Škoda   Karoq ") == "skoda karoq"
    with pytest.raises(ValueError):
        normalise_text(text=" ")


@pytest.mark.parametrize(
    "raw,expected", [("2,4", 2.4), (" 24.5 ", 24.5), (3, 3), ("1\u00a0000", 1000)]
)
def test_number_formats(raw, expected):
    assert as_number(value=raw) == expected


@pytest.mark.parametrize("raw", [None, "..", "", np.nan, "—"])
def test_suppression_stays_missing(raw):
    assert np.isnan(as_number(value=raw, allow_missing=True))
    with pytest.raises(ValueError):
        as_number(value=raw)


@pytest.mark.parametrize("raw", ["broken", "inf", "1,2,3", "NaN!"])
def test_malformed_numbers(raw):
    with pytest.raises(ValueError):
        as_number(value=raw)


def test_schema_valid_copy(evidence):
    result = validate_evidence(frame=evidence)
    assert len(result) == len(evidence)
    assert result is not evidence
    assert not empty_events().shape[0]


@pytest.mark.parametrize(
    "column,value",
    [
        ("value", -1),
        ("value", np.inf),
        ("value", np.nan),
        ("sample_size", 0),
        ("event_count", 101),
        ("age_min", 99),
        ("cohort_year_start", 3000),
        ("unit", "guess"),
        ("evidence_type", "recall"),
        ("source_id", ""),
        ("source_url", "file:///private"),
        ("model_grouped", "perhaps"),
        ("report_year", 2024.5),
        ("sample_size", 10.5),
        ("retrieved_at", "yesterday"),
    ],
)
def test_invalid_evidence(evidence, column, value):
    evidence[column] = evidence[column].astype(object)
    evidence.loc[0, column] = value
    with pytest.raises((ValueError, TypeError)):
        validate_evidence(frame=evidence)


def test_percentage_bound_duplicate_and_missing(evidence):
    evidence.loc[0, "value"] = 101
    with pytest.raises(ValueError):
        validate_evidence(frame=evidence)
    with pytest.raises(ValueError):
        validate_evidence(frame=evidence.drop(columns="value"))
    evidence.loc[0, "value"] = 0
    duplicate = evidence.iloc[[0, 0]]
    with pytest.raises(ValueError):
        validate_evidence(frame=duplicate)
