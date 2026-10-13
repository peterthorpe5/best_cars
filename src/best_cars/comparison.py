"""Prepare matched model comparisons and explicitly qualified time profiles."""

import logging

import pandas as pd

from best_cars.interrogation import MATCH_KEYS
from best_cars.schema import validate_evidence

LOGGER = logging.getLogger(__name__)


def matched_model_rates(
    *, evidence: pd.DataFrame, model_ids: list[str], reference_id: str
) -> pd.DataFrame:
    """Pair each selected model with a reference on exact published contexts.

    Args:
        evidence: Canonical evidence retaining all cohort identifiers.
        model_ids: Distinct ungrouped models selected for the comparison.
        reference_id: Selected model against which other models are compared.

    Returns:
        Paired rates and gaps in each source's original units. Negative gaps
        favour the compared model. Unmatched rows remain absent, never zero.

    Raises:
        ValueError: For invalid evidence, repeated selections or reference IDs.
    """
    frame = validate_evidence(frame=evidence)
    if len(model_ids) != len(set(model_ids)) or reference_id not in model_ids:
        raise ValueError("Select distinct models and a reference from that selection.")
    selected = frame.loc[frame.model_id.isin(model_ids)].copy()
    if selected.model_grouped.any():
        raise ValueError("Matched comparisons require ungrouped model series.")
    if selected.duplicated(subset=MATCH_KEYS + ["model_id"]).any():
        raise ValueError("Matched comparisons require unique model-context rows.")
    reference = selected.loc[
        selected.model_id.eq(reference_id), MATCH_KEYS + ["value", "make", "model"]
    ].rename(
        columns={
            "value": "reference_rate",
            "make": "reference_make",
            "model": "reference_model",
        }
    )
    compared = selected.loc[~selected.model_id.eq(reference_id)]
    paired = compared.merge(
        right=reference, on=MATCH_KEYS, how="inner", validate="many_to_one"
    )
    paired["rate_gap"] = paired.value - paired.reference_rate
    LOGGER.info("Matched %s model-cohort rates against %s", len(paired), reference_id)
    return paired


def temporal_profiles(
    *, evidence: pd.DataFrame, model_ids: list[str], mode: str = "age"
) -> pd.DataFrame:
    """Retain time series with at least two distinct observation years.

    Args:
        evidence: Canonical historical evidence, including older reports.
        model_ids: Models to display, with ungrouped series only.
        mode: Age holds the published age band fixed; registration holds exact
            recorded registration bounds fixed. Approximate TÜV registration
            bounds are excluded from the registration view.

    Returns:
        Observation rows with a readable series label and unique series ID.
        These are repeated population summaries, not linked individual cars.

    Raises:
        ValueError: For invalid evidence, selections, mode or ambiguous years.
    """
    if mode not in {"age", "registration"} or len(model_ids) != len(set(model_ids)):
        raise ValueError("Require distinct models and age or registration mode.")
    frame = validate_evidence(frame=evidence)
    frame = frame.loc[frame.model_id.isin(model_ids) & ~frame.model_grouped].copy()
    if mode == "registration":
        frame = frame.loc[~frame.source_id.eq("tuv")].copy()
    bounds = (
        ["age_min", "age_max"]
        if mode == "age"
        else ["cohort_year_start", "cohort_year_end"]
    )
    keys = ["model_id", "source_id", "metric", "unit", *bounds]
    if frame.duplicated(subset=keys + ["observation_year"]).any():
        raise ValueError(
            "Multiple estimates occupy one time-series year; narrow the report selection."
        )
    years = frame.groupby(by=keys, dropna=False).observation_year.transform(
        func="nunique"
    )
    frame = frame.loc[years.ge(2)].copy()
    frame["series_id"] = (
        frame.groupby(by=keys, dropna=False, sort=True).ngroup().astype(str)
    )
    prefix = "Age" if mode == "age" else "Registered"
    frame["series_label"] = [
        f"{prefix} {int(low)}" + (f"–{int(high)}" if low != high else "")
        for low, high in zip(frame[bounds[0]], frame[bounds[1]], strict=True)
    ]
    LOGGER.info("Prepared %s historical rows in %s mode", len(frame), mode)
    return frame.sort_values(by=["series_id", "observation_year"]).reset_index(
        drop=True
    )


def wilson_age_groups(
    *, counted: pd.DataFrame, grouping: str = "exact"
) -> dict[str, pd.DataFrame]:
    """Split counted observations into separate age/source/report panels.

    Args:
        counted: Output of counted_failure_rates with canonical context fields.
        grouping: Exact recorded age/cohort contexts, two-year age groups, or
            mixed ages within each source/report for explicit exploratory use.

    Returns:
        Readable panel labels mapped to original rows. Counts and intervals are
        never pooled or recalculated. Wide recorded bands are kept intact.

    Raises:
        ValueError: For invalid grouping or missing canonical evidence fields.
    """
    if grouping not in {"exact", "two_year", "mixed"}:
        raise ValueError("Choose exact, two_year or mixed Wilson age grouping.")
    validate_evidence(frame=counted)
    if counted.empty:
        return {}
    frame = counted.copy()
    keys = ["source_id", "report_year", "observation_year", "country", "metric", "unit"]
    if grouping == "exact":
        keys += [
            "age_min",
            "age_max",
            "cohort_year_start",
            "cohort_year_end",
            "comparison_group",
        ]
    elif grouping == "two_year":
        low = (frame.age_min // 2) * 2
        wide = frame.age_max.gt(low + 1)
        frame["panel_age_min"] = low.where(~wide, frame.age_min)
        frame["panel_age_max"] = (low + 1).where(~wide, frame.age_max)
        keys += ["panel_age_min", "panel_age_max"]
    panels = {}
    for _, rows in frame.groupby(by=keys, dropna=False, sort=True):
        row = rows.iloc[0]
        low = row.panel_age_min if grouping == "two_year" else rows.age_min.min()
        high = row.panel_age_max if grouping == "two_year" else rows.age_max.max()
        age = f"age {int(low)}" + (f"–{int(high)}" if low != high else "")
        label = f"{row.source_name} · {age} · report {int(row.report_year)} · observations {int(row.observation_year)}"
        if grouping == "exact":
            label += f" · registered {int(row.cohort_year_start)}–{int(row.cohort_year_end)} · {row.comparison_group}"
        if label in panels:
            raise ValueError(
                "Wilson panel labels are ambiguous across source contexts."
            )
        panels[label] = rows.drop(
            columns=["panel_age_min", "panel_age_max"], errors="ignore"
        ).copy()
    LOGGER.info(
        "Separated %s counted observations into %s age panels", len(frame), len(panels)
    )
    return panels
