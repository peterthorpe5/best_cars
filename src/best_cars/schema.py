"""Strict evidence validation; absent denominators never become zero."""

import logging
import re

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
EVIDENCE_COLUMNS = [
    "source_id",
    "source_name",
    "evidence_type",
    "country",
    "report_year",
    "observation_year",
    "cohort_year_start",
    "cohort_year_end",
    "age_min",
    "age_max",
    "source_model",
    "model_id",
    "make",
    "model",
    "model_grouped",
    "metric",
    "value",
    "unit",
    "sample_size",
    "event_count",
    "mileage_km",
    "mileage_basis",
    "source_rank",
    "comparison_group",
    "source_url",
    "retrieved_at",
    "notes",
]
EVENT_COLUMNS = [
    "source_id",
    "country",
    "event_type",
    "event_id",
    "make",
    "model",
    "model_year",
    "date",
    "component",
    "summary",
    "consequence",
    "remedy",
    "source_url",
    "retrieved_at",
]
NUMERIC_COLUMNS = [
    "report_year",
    "observation_year",
    "cohort_year_start",
    "cohort_year_end",
    "age_min",
    "age_max",
    "value",
    "sample_size",
    "event_count",
    "mileage_km",
    "source_rank",
]


def empty_evidence() -> pd.DataFrame:
    """Return a typed empty evidence table."""
    return pd.DataFrame(columns=EVIDENCE_COLUMNS)


def empty_events() -> pd.DataFrame:
    """Return an empty recalls/complaints table."""
    return pd.DataFrame(columns=EVENT_COLUMNS)


def as_number(*, value: object, allow_missing: bool = False) -> float:
    """Parse a finite EU/UK number without interpreting suppression as zero.

    Args:
        value: Source numeric value or decimal-comma text.
        allow_missing: Whether missing/suppressed values may return NaN.

    Returns:
        A finite float, or NaN for an explicitly missing value.

    Raises:
        ValueError: If a non-missing value is malformed or infinite.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        if allow_missing:
            return float("nan")
        raise ValueError("A required numeric value is missing.")
    text = str(value).strip().replace("\u00a0", "").replace(" ", "")
    if text in {"", "..", "...", ".", "-", "—", "NaN", "nan"}:
        if allow_missing:
            return float("nan")
        raise ValueError("A required numeric value is suppressed or missing.")
    if not re.fullmatch(r"-?\d+(?:[.,]\d+)?", text):
        raise ValueError(f"Invalid numeric value: {value!r}")
    result = float(text.replace(",", "."))
    if not np.isfinite(result):
        raise ValueError("Numeric values must be finite.")
    return result


def validate_evidence(*, frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise complete, unique model-cohort evidence records.

    Args:
        frame: Canonical long-form evidence table.

    Returns:
        A validated copy in canonical column order.

    Raises:
        ValueError: For absent fields, malformed values or duplicate cohorts.
    """
    missing = set(EVIDENCE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing evidence columns: {sorted(missing)}")
    result = frame.loc[:, EVIDENCE_COLUMNS].copy()
    if result.empty:
        return result
    for column in NUMERIC_COLUMNS:
        result[column] = pd.to_numeric(arg=result[column], errors="raise")
        finite = result[column].dropna()
        if not np.isfinite(finite).all() or (finite < 0).any():
            raise ValueError(f"{column} must contain non-negative finite values.")
    required = [
        "source_id",
        "source_name",
        "country",
        "source_model",
        "model_id",
        "make",
        "model",
        "metric",
        "unit",
        "comparison_group",
        "source_url",
        "retrieved_at",
    ]
    for column in required:
        if (
            result[column].isna().any()
            or result[column].astype(str).str.strip().eq("").any()
        ):
            raise ValueError(f"{column} contains missing labels.")
    for column in [
        "report_year",
        "observation_year",
        "cohort_year_start",
        "cohort_year_end",
        "age_min",
        "age_max",
        "value",
    ]:
        if result[column].isna().any():
            raise ValueError(f"{column} is required for every evidence row.")
    for column in [
        "report_year",
        "observation_year",
        "cohort_year_start",
        "cohort_year_end",
        "age_min",
        "age_max",
        "sample_size",
        "event_count",
    ]:
        if (result[column].dropna() % 1 != 0).any():
            raise ValueError(f"{column} must be an integer when available.")
    if not result["evidence_type"].isin(["inspection", "breakdown", "mot"]).all():
        raise ValueError("Only inspection, breakdown and MOT rates belong in evidence.")
    if not result["unit"].isin(["percent", "per_1000_vehicles"]).all():
        raise ValueError("Unrecognised evidence unit.")
    if (result.loc[result["unit"].eq("percent"), "value"] > 100).any():
        raise ValueError("Percentage values must lie between 0 and 100.")
    if (result["sample_size"].dropna() <= 0).any():
        raise ValueError("Known sample sizes must be positive.")
    if (result["event_count"] > result["sample_size"]).any():
        raise ValueError("Failure counts must not exceed the number of tests.")
    if (result["age_min"] > result["age_max"]).any() or (
        result["cohort_year_start"] > result["cohort_year_end"]
    ).any():
        raise ValueError("Cohort and age ranges must be ordered.")
    raw_boolean = result["model_grouped"].astype(str).str.lower()
    if not raw_boolean.isin(["true", "false"]).all():
        raise ValueError("model_grouped must be True or False.")
    result["model_grouped"] = raw_boolean.eq("true")
    key = [
        "source_id",
        "report_year",
        "observation_year",
        "comparison_group",
        "model_id",
        "metric",
    ]
    if result.duplicated(subset=key).any():
        raise ValueError("Duplicate source/model/cohort evidence records.")
    if (
        not result["source_url"]
        .astype(str)
        .str.startswith(("https://", "http://"))
        .all()
    ):
        raise ValueError("Evidence source URLs must be HTTP(S).")
    pd.to_datetime(
        arg=result["retrieved_at"], utc=True, format="ISO8601", errors="raise"
    )
    LOGGER.info(
        "Validated %s evidence rows from %s sources",
        len(result),
        result.source_id.nunique(),
    )
    return result.reset_index(drop=True)
