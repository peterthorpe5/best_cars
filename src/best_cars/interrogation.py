"""Explore cohort overlap and dependence on individual evidence sources."""

from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np
import pandas as pd

from best_cars.analysis import reliability_ranking
from best_cars.schema import validate_evidence

MATCH_KEYS = [
    "source_id",
    "report_year",
    "observation_year",
    "metric",
    "unit",
    "comparison_group",
    "age_min",
    "age_max",
    "cohort_year_start",
    "cohort_year_end",
]
CONTRAST_COLUMNS = [
    "source_name",
    "report_year",
    "observation_year",
    "unit",
    "first_model",
    "second_model",
    "shared_cohorts",
    "first_mean_rate",
    "second_mean_rate",
    "mean_rate_gap",
    "median_rate_gap",
]
SENSITIVITY_COLUMNS = [
    "model_id",
    "make",
    "model",
    "omitted_source",
    "baseline_score",
    "score_without_source",
    "score_change",
    "paired_models",
    "baseline_paired_rank",
    "rank_without_source",
    "rank_movement",
    "status",
]


def shared_cohort_evidence(
    *, scored: pd.DataFrame, model_ids: Sequence[str]
) -> pd.DataFrame:
    """Restrict evidence to cohorts containing every explicitly selected model.

    Args:
        scored: Selected canonical evidence; original peer scores are untouched.
        model_ids: At least two distinct, ungrouped model identities.

    Returns:
        Canonical rows sharing source, report, outcome, unit and cohort bounds.
        This matches published cohorts, not individual cars or exact engines.

    Raises:
        ValueError: For duplicate/missing/grouped models or fewer than two IDs.
    """
    wanted = list(model_ids)
    if len(wanted) < 2 or len(wanted) != len(set(wanted)):
        raise ValueError("Select at least two different model series.")
    frame = validate_evidence(frame=scored)
    if not set(wanted).issubset(set(frame.model_id)):
        raise ValueError("Selected model evidence is unavailable under these filters.")
    frame = frame.loc[frame.model_id.isin(wanted)].copy()
    if frame.model_grouped.any():
        raise ValueError("Shared-cohort comparisons require ungrouped model series.")
    overlap = frame.groupby(by=MATCH_KEYS, dropna=False).model_id.transform(
        func="nunique"
    )
    return frame.loc[overlap.eq(len(wanted))].reset_index(drop=True)


def shared_cohort_contrasts(
    *, scored: pd.DataFrame, model_ids: Sequence[str]
) -> pd.DataFrame:
    """Calculate descriptive rate gaps using the same cohorts for all models.

    Args:
        scored: Selected evidence, keeping sources and outcomes separate.
        model_ids: Distinct model series to compare on their common support.

    Returns:
        Source/year-specific unweighted mean and median paired cohort gaps.
        A positive gap means the first model has a higher, less favourable rate.
        No significance test, causal adjustment or lifetime estimate is implied.
    """
    matched = shared_cohort_evidence(scored=scored, model_ids=model_ids)
    records = []
    for _key, group in matched.groupby(
        by=["source_name", "report_year", "observation_year", "metric", "unit"],
        sort=True,
        dropna=False,
    ):
        wide = group.pivot(index=MATCH_KEYS, columns="model_id", values="value")
        names = {row.model_id: f"{row.make} {row.model}" for row in group.itertuples()}
        for first, second in combinations(iterable=model_ids, r=2):
            gap = wide[first] - wide[second]
            records.append(
                {
                    "source_name": group.iloc[0].source_name,
                    "report_year": group.iloc[0].report_year,
                    "observation_year": group.iloc[0].observation_year,
                    "unit": group.iloc[0].unit,
                    "first_model": names[first],
                    "second_model": names[second],
                    "shared_cohorts": len(wide),
                    "first_mean_rate": wide[first].mean(),
                    "second_mean_rate": wide[second].mean(),
                    "mean_rate_gap": gap.mean(),
                    "median_rate_gap": gap.median(),
                }
            )
    return pd.DataFrame.from_records(data=records, columns=CONTRAST_COLUMNS)


def source_omission_sensitivity(
    *,
    scored: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
    min_sources: int = 2,
    min_cohorts: int = 3,
) -> pd.DataFrame:
    """Measure descriptive score/rank changes when each active source is omitted.

    Args:
        scored: Selected evidence with original full-peer cohort percentiles.
        weights: Current source weights; omitted sources receive zero.
        min_sources: Baseline source threshold.
        min_cohorts: Retained minimum scored source-cohort observation count.

    Returns:
        Results for baseline-eligible models. Both ranks are recomputed within
        the same surviving model set for each omission, preventing spurious
        movement from vanished competitors. Missing alternative evidence stays
        missing. Alternative scores require at least one remaining source.

    Raises:
        ValueError: For invalid baseline weights or thresholds.
    """
    baseline = reliability_ranking(
        scored=scored,
        weights=weights,
        min_sources=min_sources,
        min_cohorts=min_cohorts,
    ).set_index(keys="model_id")
    if baseline.empty:
        return pd.DataFrame(columns=SENSITIVITY_COLUMNS)
    effective = {
        source: float((weights or {}).get(source, 1.0))
        for source in scored.source_id.unique()
    }
    active = [source for source, weight in effective.items() if weight > 0]
    if len(active) < 2:
        return pd.DataFrame(columns=SENSITIVITY_COLUMNS)
    records = []
    for omitted in sorted(active):
        adjusted = dict(effective)
        adjusted[omitted] = 0
        alternative = reliability_ranking(
            scored=scored,
            weights=adjusted,
            min_sources=1,
            min_cohorts=min_cohorts,
        ).set_index(keys="model_id")
        common = baseline.index.intersection(other=alternative.index)
        before_ranks = baseline.loc[common, "score"].rank(method="min", ascending=False)
        after_ranks = alternative.loc[common, "score"].rank(
            method="min", ascending=False
        )
        for model_id, row in baseline.iterrows():
            observed = model_id in common
            after = float(alternative.loc[model_id, "score"]) if observed else np.nan
            records.append(
                {
                    "model_id": model_id,
                    "make": row.make,
                    "model": row.model,
                    "omitted_source": omitted,
                    "baseline_score": row.score,
                    "score_without_source": after,
                    "score_change": after - row.score,
                    "paired_models": len(common),
                    "baseline_paired_rank": before_ranks.loc[model_id]
                    if observed
                    else np.nan,
                    "rank_without_source": after_ranks.loc[model_id]
                    if observed
                    else np.nan,
                    "rank_movement": before_ranks.loc[model_id]
                    - after_ranks.loc[model_id]
                    if observed
                    else np.nan,
                    "status": "Descriptive sensitivity"
                    if observed
                    else "Insufficient remaining evidence",
                }
            )
    return pd.DataFrame.from_records(data=records, columns=SENSITIVITY_COLUMNS)
