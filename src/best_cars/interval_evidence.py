"""Expose interval eligibility without reconstructing unpublished denominators."""

import logging

import numpy as np
import pandas as pd

from best_cars.schema import validate_evidence
from best_cars.statistics import counted_failure_rates

LOGGER = logging.getLogger(__name__)


def source_rate_points(*, evidence: pd.DataFrame) -> pd.DataFrame:
    """Prepare rates for every source and Wilson bounds only for actual counts.

    Args:
        evidence: Canonical observations with original published rates and units.

    Returns:
        Original rows plus point_rate, optional interval bounds, a point basis,
        and interval status. Counted percentage points use failures/tests rather
        than the rounded published percentage. Other points retain the source
        value exactly. Missing intervals remain NaN, never zero-width bars.

    Raises:
        ValueError: For malformed evidence or inconsistent recorded counts.
    """
    frame = validate_evidence(frame=evidence).reset_index(drop=True)
    frame["point_rate"] = frame.value
    frame["interval_lower"] = np.nan
    frame["interval_upper"] = np.nan
    frame["point_basis"] = "Published rate"
    frame["interval_status"] = "Unavailable: actual failed/test count pair missing"
    frame.loc[frame.unit.eq("per_1000_vehicles"), "interval_status"] = (
        "Unavailable: Wilson models a proportion, not a breakdown-event rate"
    )
    known = (
        frame.unit.eq("percent") & frame.sample_size.notna() & frame.event_count.notna()
    )
    counted = counted_failure_rates(evidence=frame)
    frame.loc[known, "point_rate"] = counted.counted_rate_pct.to_numpy()
    frame.loc[known, "interval_lower"] = counted.wilson_lower_pct.to_numpy()
    frame.loc[known, "interval_upper"] = counted.wilson_upper_pct.to_numpy()
    frame.loc[known, "point_basis"] = "Actual failed inspections / tests × 100"
    frame.loc[known, "interval_status"] = (
        "95% Wilson binomial interval from actual counts"
    )
    LOGGER.info(
        "Prepared %s rate points, %s with count-based Wilson bounds",
        len(frame),
        int(known.sum()),
    )
    return frame


def source_interval_availability(*, evidence: pd.DataFrame) -> pd.DataFrame:
    """Summarise which imported sources/reports support count-based intervals.

    Args:
        evidence: Canonical selected evidence, without inferred exposure counts.

    Returns:
        A table separated by source, report and unit, with actual availability
        counts and representative provenance URLs. Observation counts are rows,
        not inspections, vehicles or a denominator to use in an interval.

    Raises:
        ValueError: For malformed canonical evidence or inconsistent counts.
    """
    frame = source_rate_points(evidence=evidence)
    frame["has_wilson"] = frame.interval_lower.notna()
    result = (
        frame.groupby(
            by=["source_id", "source_name", "report_year", "unit"],
            dropna=False,
            sort=True,
        )
        .agg(
            observation_rows=("model_id", "size"),
            rows_with_wilson=("has_wilson", "sum"),
            known_denominator_rows=("sample_size", "count"),
            source_url=("source_url", "first"),
        )
        .reset_index()
    )
    result["rate_only_rows"] = result.observation_rows - result.rows_with_wilson
    return result
