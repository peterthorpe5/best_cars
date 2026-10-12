"""Descriptive statistics with explicit populations, units and denominators."""

import logging
from itertools import combinations

import numpy as np
import pandas as pd

from best_cars.analysis import wilson_interval
from best_cars.schema import validate_evidence

LOGGER = logging.getLogger(__name__)
SUMMARY_COLUMNS = [
    "source_id",
    "source_name",
    "report_year",
    "observation_year",
    "metric",
    "unit",
    "observations",
    "model_series",
    "mean_rate",
    "median_rate",
    "p10_rate",
    "p25_rate",
    "p75_rate",
    "p90_rate",
    "minimum_rate",
    "maximum_rate",
    "counted_observations",
    "count_coverage_pct",
    "counted_tests",
    "counted_failures",
    "counted_test_failure_pct",
]
AGREEMENT_COLUMNS = [
    "first_source",
    "second_source",
    "paired_models",
    "spearman_rho",
    "median_absolute_score_gap",
    "status",
]


def source_rate_summary(*, evidence: pd.DataFrame) -> pd.DataFrame:
    """Describe published model-cohort rates separately by source, year and unit.

    Args:
        evidence: Complete canonical rows for the explicit reporting selection.

    Returns:
        Unweighted model-cohort rate distributions and count coverage. A pooled
        failure percentage uses only rows with both actual test/failure counts;
        it describes those counted tests, not an inferred national fleet.

    Raises:
        ValueError: If canonical evidence validation fails.
    """
    frame = validate_evidence(frame=evidence)
    groups = SUMMARY_COLUMNS[:6]
    records = []
    for key, group in frame.groupby(by=groups, sort=True, dropna=False):
        counted = group.loc[
            group.unit.eq("percent")
            & group.sample_size.notna()
            & group.event_count.notna()
        ]
        total_tests = int(counted.sample_size.sum()) if len(counted) else None
        total_failures = int(counted.event_count.sum()) if len(counted) else None
        row = dict(zip(groups, key, strict=True))
        row.update(
            observations=len(group),
            model_series=group.model_id.nunique(),
            mean_rate=float(group.value.mean()),
            median_rate=float(group.value.median()),
            p10_rate=float(group.value.quantile(q=0.10)),
            p25_rate=float(group.value.quantile(q=0.25)),
            p75_rate=float(group.value.quantile(q=0.75)),
            p90_rate=float(group.value.quantile(q=0.90)),
            minimum_rate=float(group.value.min()),
            maximum_rate=float(group.value.max()),
            counted_observations=len(counted),
            count_coverage_pct=100 * len(counted) / len(group),
            counted_tests=total_tests,
            counted_failures=total_failures,
            counted_test_failure_pct=(
                100 * total_failures / total_tests
                if total_tests is not None
                else np.nan
            ),
        )
        records.append(row)
    LOGGER.info("Summarised %s source/year/metric distributions", len(records))
    return pd.DataFrame.from_records(data=records, columns=SUMMARY_COLUMNS)


def counted_failure_rates(*, evidence: pd.DataFrame) -> pd.DataFrame:
    """Attach count-based percentages and 95% Wilson intervals to counted rows.

    Args:
        evidence: Canonical observations; unknown denominators remain absent.

    Returns:
        Only percentage rows with actual numerator/denominator pairs, retaining
        original published rates alongside calculated count-based rates.

    Raises:
        ValueError: If canonical counts or evidence are invalid.
    """
    frame = validate_evidence(frame=evidence)
    frame = frame.loc[
        frame.unit.eq("percent") & frame.sample_size.notna() & frame.event_count.notna()
    ].copy()
    intervals = [
        wilson_interval(failures=int(row.event_count), tests=int(row.sample_size))
        for row in frame.itertuples()
    ]
    frame["counted_rate_pct"] = 100 * frame.event_count / frame.sample_size
    frame["wilson_lower_pct"] = [interval[0] for interval in intervals]
    frame["wilson_upper_pct"] = [interval[1] for interval in intervals]
    return frame.reset_index(drop=True)


def pairwise_source_agreement(
    *, wide: pd.DataFrame, min_pairs: int = 5
) -> pd.DataFrame:
    """Describe paired source-score association without filling missing models.

    Args:
        wide: Unique model-by-source matrix of scores between zero and 100.
        min_pairs: Minimum paired models required to calculate rank correlation.

    Returns:
        Paired-model counts, Spearman rank correlations and median absolute
        percentile gaps. Constant scores or small samples retain missing rho.

    Raises:
        ValueError: For duplicate identities, invalid scores or min_pairs < 3.
    """
    if min_pairs < 3 or not wide.index.is_unique or not wide.columns.is_unique:
        raise ValueError("Require unique models/sources and at least three pairs.")
    numeric = wide.apply(func=pd.to_numeric, errors="raise")
    observed = numeric.to_numpy(dtype=float)
    present = observed[~np.isnan(observed)]
    if not np.isfinite(present).all() or (present < 0).any() or (present > 100).any():
        raise ValueError("Source scores must be finite values between zero and 100.")
    records = []
    for first, second in combinations(iterable=numeric.columns, r=2):
        paired = numeric.loc[:, [first, second]].dropna()
        rho = np.nan
        status = "Too few paired models"
        if len(paired) >= min_pairs:
            status = "Constant scores; correlation undefined"
            if paired[first].nunique() > 1 and paired[second].nunique() > 1:
                # Pearson on average ranks is Spearman, including tied scores.
                ranks = paired.rank(method="average")
                rho = float(ranks[first].corr(other=ranks[second], method="pearson"))
                status = "Descriptive association; cohorts may differ"
        records.append(
            {
                "first_source": first,
                "second_source": second,
                "paired_models": len(paired),
                "spearman_rho": rho,
                "median_absolute_score_gap": (
                    float((paired[first] - paired[second]).abs().median())
                    if len(paired)
                    else np.nan
                ),
                "status": status,
            }
        )
    return pd.DataFrame.from_records(data=records, columns=AGREEMENT_COLUMNS)
