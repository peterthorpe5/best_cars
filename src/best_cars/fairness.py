"""Outcome-first evidence profiles and coverage-matched preference comparisons."""

import hashlib
import json
import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd

from best_cars.analysis import reliability_ranking as legacy_ranking

LOGGER = logging.getLogger(__name__)
OUTCOMES = {
    "adac": "Roadside breakdowns",
    "tuv": "Inspection condition",
    "traficom": "Inspection condition",
    "mot": "Inspection condition",
}
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


def evidence_profiles(*, scored: pd.DataFrame) -> pd.DataFrame:
    """Summarise separate source outcomes without assigning a combined rank.

    Args:
        scored: Canonical selected evidence with original peer percentiles.

    Returns:
        One model/source profile with coverage, raw units and mean percentile.
        Different age ranges remain explicit; no fleet or lifetime prediction.

    Raises:
        ValueError: If required evidence/scoring fields are absent.
    """
    required = {
        "model_id",
        "make",
        "model",
        "source_id",
        "source_name",
        "model_grouped",
        "relative_score",
        "value",
        "unit",
        "age_min",
        "age_max",
        "sample_size",
        "report_year",
        "observation_year",
    }
    if not required.issubset(scored.columns):
        raise ValueError("Evidence profiles require canonical scored observations.")
    frame = scored.loc[~scored.model_grouped].copy()
    profile = (
        frame.groupby(
            by=[
                "model_id",
                "make",
                "model",
                "source_id",
                "source_name",
                "unit",
                "report_year",
                "observation_year",
            ],
            dropna=False,
        )
        .agg(
            mean_relative_score=("relative_score", "mean"),
            mean_published_rate=("value", "mean"),
            observations=("value", "size"),
            scored_cohorts=("relative_score", "count"),
            youngest_age=("age_min", "min"),
            oldest_age=("age_max", "max"),
            known_count_rows=("sample_size", "count"),
        )
        .reset_index()
    )
    profile["outcome"] = profile.source_id.map(arg=OUTCOMES).fillna(
        value="Unclassified outcome"
    )
    LOGGER.debug("Produced %s separate source profiles", len(profile))
    return profile


def cohort_contexts(*, scored: pd.DataFrame, source_id: str) -> pd.DataFrame:
    """List exact published cohort contexts for a source-specific comparison.

    Args:
        scored: Canonical selected evidence retaining original peer scores.
        source_id: Explicit source to inspect.

    Returns:
        Context identifiers and readable labels; no ages or units are pooled.

    Raises:
        ValueError: If context metadata are absent.
    """
    if not set(MATCH_KEYS).issubset(scored.columns):
        raise ValueError("Source cohorts require complete context metadata.")
    contexts = (
        scored.loc[scored.source_id.eq(source_id), MATCH_KEYS]
        .drop_duplicates()
        .sort_values(by=["report_year", "age_min", "age_max"])
    )
    contexts = contexts.copy()
    contexts["context_id"] = [
        hashlib.sha256(json.dumps(obj=row, default=str).encode()).hexdigest()[:12]
        for row in contexts.to_numpy().tolist()
    ]
    contexts["context_label"] = [
        f"Report {row.report_year} · observed {row.observation_year} · age {row.age_min:g}–{row.age_max:g} · registrations {row.cohort_year_start:g}–{row.cohort_year_end:g} · {row.metric}"
        for row in contexts.itertuples()
    ]
    return contexts.reset_index(drop=True)


def source_cohort_ranking(
    *, scored: pd.DataFrame, source_id: str, context_id: str
) -> pd.DataFrame:
    """Rank only models in one explicit source/outcome/age/registration cohort.

    Args:
        scored: Selected evidence with original full-peer percentiles.
        source_id: Explicit source identifier.
        context_id: Identifier returned by cohort_contexts.

    Returns:
        A source-specific cohort ordering; its score is not failure probability.

    Raises:
        ValueError: For an unavailable context identifier.
    """
    contexts = cohort_contexts(scored=scored, source_id=source_id)
    selected = contexts.loc[contexts.context_id.eq(context_id)]
    if len(selected) != 1:
        raise ValueError("Select an available exact source cohort.")
    mask = scored.source_id.eq(source_id)
    for key in MATCH_KEYS:
        value = selected.iloc[0][key]
        mask &= scored[key].isna() if pd.isna(value) else scored[key].eq(value)
    result = legacy_ranking(scored=scored.loc[mask], min_sources=1, min_cohorts=1)
    result.attrs["ranking_label"] = "Relative ordering within one published cohort"
    return result


def coverage_matched_ranking(
    *,
    scored: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
    min_sources: int = 1,
    min_cohorts: int = 1,
    inspection_weight: float = 1.0,
    breakdown_weight: float = 1.0,
) -> pd.DataFrame:
    """Rank preferences only within identical source/cohort coverage blocks.

    Args:
        scored: Canonical selected evidence with unchanged peer percentiles.
        weights: Within-outcome source preferences; zero omits a source.
        min_sources: Additional minimum source requirement.
        min_cohorts: Minimum scored observations per model.
        inspection_weight: Preference weight for the inspection outcome.
        breakdown_weight: Preference weight for roadside breakdowns.

    Returns:
        Eligible models with block-specific ranks, comparison identifiers,
        competitor counts and separate outcome scores. Every model must cover
        every active source. Cohort coverage must match exactly within a block;
        singleton blocks have neither a composite score nor a rank. Blocks cannot be
        ordered against each other. No missing-score imputation is performed.

    Raises:
        ValueError: For unknown source outcomes or invalid preference weights.
    """
    values = [inspection_weight, breakdown_weight]
    if any(not np.isfinite(value) or value < 0 for value in values) or sum(values) <= 0:
        raise ValueError("Outcome weights must be finite, non-negative and non-zero.")
    # Reuse validated source-weight, eligibility and canonical ranking machinery.
    ranked = legacy_ranking(
        scored=scored, weights=weights, min_sources=min_sources, min_cohorts=min_cohorts
    )
    extra = [
        "comparison_group_id",
        "comparison_models",
        "inspection_score",
        "breakdown_score",
        "comparison_status",
    ]
    if ranked.empty:
        return ranked.reindex(columns=list(ranked.columns) + extra)
    preferences = dict(weights or {})
    outcome_weights = {
        "Inspection condition": inspection_weight,
        "Roadside breakdowns": breakdown_weight,
    }
    unknown = set(scored.source_id) - set(OUTCOMES)
    if unknown:
        raise ValueError(f"Unclassified reliability sources: {sorted(unknown)}")
    active = {
        source
        for source in scored.source_id.unique()
        if preferences.get(source, 1.0) > 0 and outcome_weights[OUTCOMES[source]] > 0
    }
    if not active:
        return ranked.iloc[:0].reindex(columns=list(ranked.columns) + extra)
    rows = scored.loc[
        ~scored.model_grouped
        & scored.relative_score.notna()
        & scored.source_id.isin(active)
    ]
    ranked_index = ranked.set_index(keys="model_id")
    records = []
    for _model_id, group in rows.groupby(by="model_id", sort=True):
        if (
            set(group.source_id) != active
            or len(group) < min_cohorts
            or len(active) < min_sources
        ):
            continue
        means = group.groupby(by="source_id").relative_score.mean()
        domain_scores = {}
        for outcome in set(OUTCOMES[source] for source in active):
            sources = [source for source in active if OUTCOMES[source] == outcome]
            denominator = sum(preferences.get(source, 1.0) for source in sources)
            domain_scores[outcome] = (
                sum(means[source] * preferences.get(source, 1.0) for source in sources)
                / denominator
            )
        denominator = sum(outcome_weights[outcome] for outcome in domain_scores)
        score = (
            sum(
                domain_scores[outcome] * outcome_weights[outcome]
                for outcome in domain_scores
            )
            / denominator
        )
        signature = (
            group.loc[:, MATCH_KEYS]
            .astype(str)
            .sort_values(by=MATCH_KEYS)
            .to_numpy()
            .tolist()
        )
        identifier = hashlib.sha256(
            json.dumps(obj=signature, ensure_ascii=True).encode()
        ).hexdigest()[:12]
        row = ranked_index.loc[_model_id].to_dict()
        row.update(
            model_id=_model_id,
            sources=len(active),
            cohorts=len(group),
            source_ids="; ".join(sorted(active)),
            score_spread=float(means.max() - means.min()),
            worst_cohort_score=float(group.relative_score.min()),
        )
        row.update(
            score=score,
            weight_coverage=100.0,
            comparison_group_id=identifier,
            inspection_score=domain_scores.get("Inspection condition", np.nan),
            breakdown_score=domain_scores.get("Roadside breakdowns", np.nan),
        )
        records.append(row)
    if not records:
        return ranked.iloc[:0].reindex(columns=list(ranked.columns) + extra)
    result = pd.DataFrame.from_records(data=records)
    result["comparison_models"] = result.groupby(
        by="comparison_group_id"
    ).model_id.transform(func="size")
    result["rank"] = (
        result.groupby(by="comparison_group_id")
        .score.rank(method="min", ascending=False)
        .astype("Int64")
    )
    result.loc[result.comparison_models.lt(2), "rank"] = pd.NA
    result.loc[result.comparison_models.lt(2), "score"] = np.nan
    result["comparison_status"] = np.where(
        result.comparison_models.ge(2),
        "Coverage-matched preference comparison",
        "No comparable competitor; no composite score or rank",
    )
    LOGGER.info(
        "Produced %s preference profiles in %s coverage blocks",
        len(result),
        result.comparison_group_id.nunique(),
    )
    return result.sort_values(
        by=["comparison_group_id", "rank", "model_id"], na_position="last"
    ).reset_index(drop=True)


def exploratory_ranking(
    *,
    scored: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
    min_sources: int = 1,
    min_cohorts: int = 1,
    inspection_weight: float = 1.0,
    breakdown_weight: float = 1.0,
) -> pd.DataFrame:
    """Order incomplete profiles using explicit neutral missing-evidence assumptions.

    Missing source percentiles use 50 solely for exploratory ordering. Bounds
    substitute 0 and 100; they are sensitivity bounds, not confidence intervals.
    Different age coverage remains a limitation and is exported alongside ranks.

    Args:
        scored: Selected canonical observations with original peer percentiles.
        weights: Non-negative source preferences within each outcome.
        min_sources: Minimum observed active sources.
        min_cohorts: Minimum observed active source/cohort rows.
        inspection_weight: Weight for the inspection-condition outcome.
        breakdown_weight: Weight for the attended-breakdown outcome.

    Returns:
        Exploratory ranks with evidence gaps, sample context and sensitivity bounds.

    Raises:
        ValueError: For invalid weights, thresholds or unclassified outcomes.
    """
    coverage_matched_ranking(
        scored=scored,
        weights=weights,
        min_sources=min_sources,
        min_cohorts=min_cohorts,
        inspection_weight=inspection_weight,
        breakdown_weight=breakdown_weight,
    )
    result = legacy_ranking(
        scored=scored, weights=weights, min_sources=min_sources, min_cohorts=min_cohorts
    )
    if result.empty:
        return result
    preferences = dict(weights or {})
    domain_weights = {
        "Inspection condition": inspection_weight,
        "Roadside breakdowns": breakdown_weight,
    }
    active = {
        s
        for s in scored.source_id.unique()
        if preferences.get(s, 1.0) > 0 and domain_weights[OUTCOMES[s]] > 0
    }
    if not active:
        return result.iloc[:0]
    domains = {OUTCOMES[s] for s in active}
    denominator = sum(domain_weights[d] for d in domains)
    shares = {
        s: domain_weights[OUTCOMES[s]]
        / denominator
        * preferences.get(s, 1.0)
        / sum(preferences.get(t, 1.0) for t in active if OUTCOMES[t] == OUTCOMES[s])
        for s in active
    }
    evidence = scored.loc[
        ~scored.model_grouped
        & scored.relative_score.notna()
        & scored.source_id.isin(active)
    ]
    records = []
    for row in result.to_dict(orient="records"):
        group = evidence.loc[evidence.model_id.eq(row["model_id"])]
        means = group.groupby(by="source_id").relative_score.mean()
        if len(means) < min_sources or len(group) < min_cohorts:
            continue
        known = sum(shares[s] * means[s] for s in means.index)
        missing_share = sum(shares[s] for s in active - set(means.index))
        counts = (
            pd.to_numeric(group["sample_size"], errors="coerce")
            if "sample_size" in group
            else pd.Series(dtype=float)
        )
        row.update(
            score=known + 50 * missing_share,
            score_lower=known,
            score_upper=known + 100 * missing_share,
            weight_coverage=100 * (1 - missing_share),
            missing_sources="; ".join(sorted(active - set(means.index))) or "None",
            sources=len(means),
            cohorts=len(group),
            source_ids="; ".join(sorted(means.index)),
            youngest_age=group.age_min.min(),
            oldest_age=group.age_max.max(),
            known_sample_rows=int(counts.notna().sum()),
            minimum_published_sample=counts.min(),
            comparison_group_id="exploratory",
            comparison_models=len(result),
            comparison_status="Exploratory; missing sources assume percentile 50; ages may differ",
        )
        records.append(row)
    frame = pd.DataFrame.from_records(data=records)
    if frame.empty:
        return result.iloc[:0]
    frame["rank"] = frame.score.rank(method="min", ascending=False).astype("Int64")
    frame["comparison_models"] = len(frame)
    frame["best_missing_evidence_rank"] = [
        1 + int(frame.score_lower.gt(row.score_upper).sum())
        for row in frame.itertuples()
    ]
    frame["worst_missing_evidence_rank"] = [
        int(frame.score_upper.ge(row.score_lower).sum()) for row in frame.itertuples()
    ]
    LOGGER.info(
        "Produced %s exploratory profiles with explicit evidence gaps", len(frame)
    )
    frame = frame.sort_values(by=["rank", "model_id"]).reset_index(drop=True)
    frame.attrs["ranking_label"] = (
        "Exploratory preference ordering · evidence coverage can differ"
    )
    return frame
