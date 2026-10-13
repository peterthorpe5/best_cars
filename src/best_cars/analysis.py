"""Transparent comparisons within comparable source cohorts."""

import logging
from collections.abc import Mapping, Sequence
from math import sqrt

import numpy as np
import pandas as pd

from best_cars.schema import validate_evidence

LOGGER = logging.getLogger(__name__)


def wilson_interval(*, failures: int, tests: int) -> tuple[float, float]:
    """Return a 95% Wilson interval for an observed test-failure percentage.

    Args:
        failures: Number of failed tests.
        tests: Positive number of completed tests.

    Returns:
        Lower and upper percentage bounds; not a causal reliability interval.

    Raises:
        ValueError: If counts are non-integer, negative or inconsistent.
    """
    if isinstance(tests, bool) or isinstance(failures, bool):
        raise ValueError("Counts must be integers.")
    if (
        int(tests) != tests
        or int(failures) != failures
        or tests <= 0
        or not 0 <= failures <= tests
    ):
        raise ValueError(
            "Require integer counts with 0 <= failures <= tests and tests > 0."
        )
    z = 1.959963984540054
    p = failures / tests
    denominator = 1 + z * z / tests
    centre = (p + z * z / (2 * tests)) / denominator
    radius = z * sqrt(p * (1 - p) / tests + z * z / (4 * tests * tests)) / denominator
    lower = 0.0 if failures == 0 else max(0.0, centre - radius) * 100
    upper = 100.0 if failures == tests else min(1.0, centre + radius) * 100
    return lower, upper


def cohort_scores(*, evidence: pd.DataFrame, min_peers: int = 5) -> pd.DataFrame:
    """Calculate favourable percentiles before applying display filters.

    Args:
        evidence: Full canonical dataset, including all comparison peers.
        min_peers: Minimum distinct models in a source/year/cohort/segment.

    Returns:
        Evidence with peer counts and relative scores (100 = lowest rate).
        Grouped models remain visible but cannot enter cross-source rankings.

    Raises:
        ValueError: If validation fails or min_peers is less than two.
    """
    if min_peers < 2:
        raise ValueError("At least two peers are required.")
    frame = validate_evidence(frame=evidence)
    frame["peer_count"] = pd.Series(dtype="int64")
    frame["relative_score"] = pd.Series(dtype="float64")
    if frame.empty:
        return frame
    groups = [
        "source_id",
        "report_year",
        "observation_year",
        "comparison_group",
        "metric",
    ]
    grouped = frame.groupby(by=groups, dropna=False)["value"]
    frame["peer_count"] = grouped.transform(func="size")
    ranks = grouped.rank(method="average", ascending=True)
    frame["relative_score"] = (
        100 * (frame["peer_count"] - ranks) / (frame["peer_count"] - 1)
    )
    # Equal-rate cohorts carry no information about relative ordering.
    equal = grouped.transform(func="nunique").eq(1)
    frame.loc[equal, "relative_score"] = 50.0
    frame.loc[frame.peer_count < min_peers, "relative_score"] = np.nan
    return frame


def select_evidence(
    *,
    evidence: pd.DataFrame,
    sources: Sequence[str] | None = None,
    latest_only: bool = True,
    age_range: tuple[int, int] | None = None,
    cohort_range: tuple[int, int] | None = None,
    min_sample: int = 0,
) -> pd.DataFrame:
    """Select evidence after percentile calculation against the full peer set.

    Args:
        evidence: Full canonical dataset.
        sources: Optional source IDs; an empty selection returns no evidence.
        latest_only: Use the latest available report separately for each source.
        age_range: Retain age intervals overlapping this inclusive range.
        cohort_range: Retain registration-year intervals overlapping this range.
        min_sample: Exclude known samples below this size; keep unknown samples.

    Returns:
        Selected scored evidence, with unknown samples explicitly retained.

    Raises:
        ValueError: For invalid numeric filters.
    """
    if min_sample < 0:
        raise ValueError("Minimum sample size cannot be negative.")
    for interval in [age_range, cohort_range]:
        if interval is not None and (len(interval) != 2 or interval[0] > interval[1]):
            raise ValueError("Filter ranges must contain ordered endpoints.")
    frame = cohort_scores(evidence=evidence)
    if sources is not None:
        frame = frame.loc[frame.source_id.isin(sources)].copy()
    if frame.empty:
        return frame
    if latest_only:
        latest = frame.groupby(by="source_id")["report_year"].transform(func="max")
        frame = frame.loc[frame.report_year.eq(latest)].copy()
    if age_range is not None:
        frame = frame.loc[
            (frame.age_min <= age_range[1]) & (frame.age_max >= age_range[0])
        ]
    if cohort_range is not None:
        frame = frame.loc[
            (frame.cohort_year_start <= cohort_range[1])
            & (frame.cohort_year_end >= cohort_range[0])
        ]
    frame = frame.loc[frame.sample_size.isna() | frame.sample_size.ge(min_sample)]
    return frame.reset_index(drop=True)


def reliability_ranking(
    *,
    scored: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
    min_sources: int = 1,
    min_cohorts: int = 1,
) -> pd.DataFrame:
    """Reproduce the legacy unbalanced source-average ordering.

    Historical API retained for reproducibility and low-level summaries. This
    is not the current app/CLI ranking: use coverage_matched_ranking for
    outcome-balanced, identical-coverage preference comparisons.

    Args:
        scored: Output of select_evidence, with scores calculated before filtering.
        weights: Non-negative source weights; missing sources default to one.
        min_sources: Required number of scored sources per model.
        min_cohorts: Required number of source-cohort observations per model.

    Returns:
        Ranked model-series table with source breadth, variability and coverage.
        The composite is an exploratory relative ordering, never failure risk.

    Raises:
        ValueError: For invalid weights or coverage thresholds.
    """
    if min_sources < 1 or min_cohorts < 1:
        raise ValueError("Coverage thresholds must be positive.")
    columns = [
        "rank",
        "model_id",
        "make",
        "model",
        "score",
        "sources",
        "cohorts",
        "score_spread",
        "worst_cohort_score",
        "weight_coverage",
        "source_ids",
    ]
    if scored.empty:
        return pd.DataFrame(columns=columns)
    weights = dict(weights or {})
    if any(not np.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError("Source weights must be finite and non-negative.")
    effective = {s: float(weights.get(s, 1.0)) for s in scored.source_id.unique()}
    total_weight = sum(effective.values())
    if total_weight <= 0:
        raise ValueError("At least one source weight must be positive.")
    frame = scored.loc[~scored.model_grouped & scored.relative_score.notna()].copy()
    frame = frame.loc[frame.source_id.map(arg=effective).gt(0)]
    records = []
    for model_id, group in frame.groupby(by="model_id", sort=True):
        source_means = group.groupby(by="source_id")["relative_score"].mean()
        if len(source_means) < min_sources or len(group) < min_cohorts:
            continue
        available_weight = sum(effective[s] for s in source_means.index)
        score = (
            sum(effective[s] * source_means.loc[s] for s in source_means.index)
            / available_weight
        )
        records.append(
            {
                "model_id": model_id,
                "make": group.iloc[0]["make"],
                "model": group.iloc[0]["model"],
                "score": score,
                "sources": len(source_means),
                "cohorts": len(group),
                "score_spread": float(source_means.max() - source_means.min()),
                "worst_cohort_score": float(group.relative_score.min()),
                "weight_coverage": 100 * available_weight / total_weight,
                "source_ids": "; ".join(source_means.index),
            }
        )
    if not records:
        return pd.DataFrame(columns=columns)
    result = (
        pd.DataFrame.from_records(data=records)
        .sort_values(
            by=["score", "sources", "cohorts", "model_id"],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
    )
    result.insert(
        loc=0,
        column="rank",
        value=result.score.rank(method="min", ascending=False).astype(int),
    )
    LOGGER.info("Ranked %s model series", len(result))
    return result.loc[:, columns]


def source_agreement(*, scored: pd.DataFrame) -> pd.DataFrame:
    """Return mean source percentiles by model for inspecting disagreement.

    Args:
        scored: Scored evidence selected by the user's filters.

    Returns:
        Wide table indexed by model ID, keeping unavailable sources as NaN.
    """
    if scored.empty:
        return pd.DataFrame()
    eligible = scored.loc[~scored.model_grouped & scored.relative_score.notna()]
    return eligible.pivot_table(
        index="model_id", columns="source_id", values="relative_score", aggfunc="mean"
    )
