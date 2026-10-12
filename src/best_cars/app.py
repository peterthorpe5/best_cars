"""Streamlit reliability explorer: evidence, comparison and interpretable graphics."""

import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from best_cars.analysis import (
    reliability_ranking,
    select_evidence,
    source_agreement,
    wilson_interval,
)
from best_cars.charts import (
    age_chart,
    age_score_heatmap,
    agreement_heatmap,
    agreement_scatter,
    count_uncertainty_chart,
    mileage_chart,
    ranking_chart,
    rate_distribution_chart,
    score_spread_chart,
)
from best_cars.interrogation import (
    shared_cohort_contrasts,
    shared_cohort_evidence,
    source_omission_sensitivity,
)
from best_cars.pipeline import nhtsa_lookup, refresh_sources
from best_cars.statistics import (
    counted_failure_rates,
    pairwise_source_agreement,
    source_rate_summary,
)
from best_cars.storage import (
    data_directory,
    excel_bytes,
    load_snapshot,
    source_catalogue,
    tsv_bytes,
)
from best_cars.weighting import PRESETS, preference_weights

LOGGER = logging.getLogger(__name__)
SOURCE_NAMES = {
    "adac": "ADAC breakdowns",
    "traficom": "Traficom inspections",
    "tuv": "TÜV defects",
    "mot": "UK MOT",
}
PAGES = [
    "Overview",
    "Find reliable cars",
    "Compare cars",
    "Age and mileage",
    "Evidence agreement",
    "Deeper analysis",
    "Recalls and complaints",
    "Sources and methods",
    "Data management",
]
METHODS = """Scores compare published rates within each source, report and age/registration-year cohort. Lower rates receive higher relative percentiles. At least five peers are needed. Cohort scores are averaged per model per source; selected source weights then combine those averages. Missing sources are omitted and available weight is reported. The score is an exploratory ordering, not a probability, prediction, causal estimate or recommendation for an individual car. Grouped/unspecified models are shown in raw evidence but excluded from cross-source rankings. Recalls and voluntary complaints never enter the score."""


def show_table(*, frame: pd.DataFrame, name: str, key: str) -> None:
    """Display a browsable table with TSV and formatted Excel downloads.

    Args:
        frame: Exact displayed/exported rows.
        name: File-name stem.
        key: Unique Streamlit widget prefix.
    """
    labels = {
        "score": "Relative score",
        "sources": "Sources",
        "cohorts": "Source-cohort observations",
        "score_spread": "Source disagreement",
        "worst_cohort_score": "Lowest cohort score",
        "weight_coverage": "Available source weight (%)",
        "source_ids": "Evidence sources",
        "source_name": "Source",
        "sample_size": "Tests",
        "event_count": "Failed tests",
        "value": "Published rate",
        "unit": "Rate units",
        "model_series": "Model series",
        "adac": "ADAC score",
        "traficom": "Traficom score",
        "tuv": "TÜV score",
        "mot": "UK MOT score",
    }
    config = {}
    for column in frame.columns:
        title = labels.get(column, column.replace("_", " ").capitalize())
        if column in {"model_id", "source_id"}:
            config[column] = None
        elif pd.api.types.is_numeric_dtype(frame[column]):
            integer = column in {
                "rank",
                "sources",
                "cohorts",
                "report_year",
                "observation_year",
                "model_year",
                "sample_size",
                "event_count",
                "age_min",
                "age_max",
                "paired_models",
                "shared_cohorts",
            }
            config[column] = st.column_config.NumberColumn(
                label=title, format="%.0f" if integer else "%.1f"
            )
        else:
            config[column] = st.column_config.TextColumn(label=title)
    st.dataframe(data=frame, hide_index=True, width="stretch", column_config=config)
    left, right = st.columns(spec=2)
    with left:
        st.download_button(
            label="Download TSV",
            data=tsv_bytes(frame=frame),
            file_name=f"{name}.tsv",
            mime="text/tab-separated-values",
            key=f"{key}_tsv",
        )
    with right:
        st.download_button(
            label="Download Excel",
            data=excel_bytes(sheets={"Results": frame}),
            file_name=f"{name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"{key}_xlsx",
        )


def show_chart(*, figure, explanation: str, key: str) -> None:
    """Render a chart with interpretation help and an interactive HTML download.

    Args:
        figure: Plotly figure.
        explanation: Plain-language explanation of axes and limitations.
        key: Unique widget prefix.
    """
    if not figure.data:
        st.info(body=figure.layout.title.text or "No chart data meet these filters.")
        return
    st.plotly_chart(
        figure_or_data=figure,
        width="stretch",
        key=f"{key}_plot",
        config={
            "displaylogo": False,
            "toImageButtonOptions": {"format": "png", "scale": 2},
        },
    )
    with st.expander(label="? What does this chart show?"):
        st.write(explanation)
    st.download_button(
        label="Download interactive chart",
        data=figure.to_html(full_html=True, include_plotlyjs=True),
        file_name=f"{key}.html",
        mime="text/html",
        key=f"{key}_html",
    )


def sidebar_filters(
    *, evidence: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float], int, int]:
    """Collect evidence filters without changing the original comparison peers.

    Args:
        evidence: Complete loaded canonical observations.

    Returns:
        Selected scored observations, source weights and coverage thresholds.
    """
    with st.sidebar:
        st.subheader(body="Evidence filters")
        source_ids = sorted(evidence.source_id.unique())
        selected = st.multiselect(
            label="Sources",
            options=source_ids,
            format_func=lambda source: {
                "adac": "ADAC · breakdowns",
                "traficom": "Traficom · inspections",
                "tuv": "TÜV · defects",
                "mot": "UK MOT · inspections",
            }.get(source, source),
            default=source_ids,
            help="Only sources with loaded reliability rates appear here. Safety events are separate.",
            key="filter_sources",
        )
        latest = st.checkbox(
            label="Latest loaded report per source",
            value=True,
            key="filter_latest",
            help="A source's latest loaded report may be older than another source's. See source dates.",
        )
        max_age = int(evidence.age_max.max()) if not evidence.empty else 20
        age = st.slider(
            label="Observed vehicle age",
            min_value=0,
            max_value=max(20, max_age),
            value=(2, min(15, max(20, max_age))),
            key="filter_age",
        )
        cohort_range = None
        if st.checkbox(
            label="Filter by registration years", value=False, key="filter_registration"
        ):
            first_year = (
                int(evidence.cohort_year_start.min()) if not evidence.empty else 2000
            )
            last_year = (
                int(evidence.cohort_year_end.max()) if not evidence.empty else 2030
            )
            last_year = max(first_year + 1, last_year)
            cohort_range = st.slider(
                label="Registration year range",
                min_value=first_year,
                max_value=last_year,
                value=(first_year, last_year),
                key="filter_cohort_years",
                help="ADAC/Traficom cohorts use registration years. TÜV bounds are approximate age-bin estimates, not verified production years.",
            )
        min_sample = st.number_input(
            label="Minimum known test count",
            min_value=0,
            value=100,
            step=100,
            key="filter_sample",
            help="Unknown denominators remain visible and labelled; this filter excludes known smaller samples.",
        )
        min_sources = st.number_input(
            label="Minimum sources per model",
            min_value=1,
            max_value=max(1, len(source_ids)),
            value=max(1, min(2, len(source_ids))),
            step=1,
            key="filter_min_sources",
        )
        min_cohorts = st.number_input(
            label="Minimum source-cohort observations",
            min_value=1,
            value=3,
            step=1,
            key="filter_min_cohorts",
        )
        weights = {}
        with st.expander(label="Adjust source weights"):
            preset = st.selectbox(
                label="Starting-weight scenario",
                options=list(PRESETS),
                key="weight_preset",
            )
            starting = preference_weights(sources=selected, preset=preset)
            changed_preset = st.session_state.get("previous_weight_preset") != preset
            for source, value in starting.items():
                if changed_preset or f"weight_{source}" not in st.session_state:
                    st.session_state[f"weight_{source}"] = value
            st.session_state["previous_weight_preset"] = preset
            st.caption(
                body="Scenarios express priorities, not measured trust. Balanced uses 1:1:1; breakdown emphasis sets ADAC to 2; inspection emphasis sets official inspection counts to 2 and ADAC to 0.5. Other sources remain 1. Sliders can override; zero omits a source. Weights cannot fix different ages, generations or markets."
            )
            for source in selected:
                weights[source] = st.slider(
                    label=f"{source} weight",
                    min_value=0.0,
                    max_value=3.0,
                    value=None,
                    step=0.25,
                    key=f"weight_{source}",
                )
            if weights != starting:
                st.caption(
                    body="Your sliders now differ from the starting scenario. The ranking uses the current slider values."
                )
    scored = select_evidence(
        evidence=evidence,
        sources=selected,
        latest_only=latest,
        age_range=age,
        cohort_range=cohort_range,
        min_sample=int(min_sample),
    )
    return scored, weights, int(min_sources), int(min_cohorts)


def overview_page(
    *,
    evidence: pd.DataFrame,
    scored: pd.DataFrame,
    ranking: pd.DataFrame,
    manifest: dict,
) -> None:
    """Render loaded evidence coverage and a cautious initial shortlist.

    Args:
        evidence: All loaded observations.
        scored: User-selected scored observations.
        ranking: Model ranking meeting coverage filters.
        manifest: Verified snapshot metadata.
    """
    st.title(body="Find a car you can trust")
    st.write(
        "Explore inspection failures and roadside breakdowns. Compare models, follow the age patterns, and see how strong the evidence is."
    )
    st.caption(
        body=f"Current filters: {len(scored):,} observations · {len(ranking):,} qualifying models. Model series can span engines and generations; relative scores are not failure probabilities."
    )
    columns = st.columns(spec=4)
    for column, label, value in zip(
        columns,
        [
            "Model series",
            "Rate observations",
            "Reliability sources",
            "Observed markets",
        ],
        [
            evidence.loc[~evidence.model_grouped].model_id.nunique(),
            len(evidence),
            evidence.source_id.nunique(),
            evidence.country.nunique(),
        ],
        strict=True,
    ):
        column.metric(label=label, value=f"{value:,}")
    if manifest.get("baseline_only"):
        st.info(
            body="Historical TÜV data for 2024–2025 are loaded. Open Data management to add verified ADAC and Traficom data, or select the supplied verified data folder when launching."
        )
    st.caption(
        body="Loaded source dates are shown below. A later retrieval date does not make an older report newer."
    )
    coverage = (
        evidence.groupby(
            by=["source_name", "country", "report_year", "observation_year"]
        )
        .agg(model_series=("model_id", "nunique"), observations=("model_id", "size"))
        .reset_index()
    )
    st.dataframe(
        data=coverage.rename(
            columns={
                "report_year": "Report year",
                "observation_year": "Observation year",
                "source_name": "Source",
                "country": "Market",
                "model_series": "Model series",
                "observations": "Rate observations",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    if not ranking.empty:
        show_chart(
            figure=ranking_chart(ranking=ranking, top_n=12),
            key="overview_rank",
            explanation="Bars show relative cohort performance across the selected sources. Higher scores mean lower observed rates relative to peers, not lower predicted failure probability. Colour shows source breadth. Use Find reliable cars to inspect coverage and change weights.",
        )
    else:
        st.info(
            body="No models meet the current coverage filters. Reduce the minimum sources or cohort count, or load another dataset."
        )
    with st.expander(label="? How should I use this evidence?"):
        st.write(
            "Start with cars of the age you intend to buy. Prefer model series with several sources and cohorts, then check the raw observations for disagreement. These tables rarely separate engines, trim or generation; inspect service history and the actual vehicle before buying. German and Finnish observations are useful evidence, with different market and climate conditions from the UK."
        )


def find_page(*, scored: pd.DataFrame, ranking: pd.DataFrame) -> None:
    """Render model search, shortlist creation and source-by-source evidence.

    Args:
        scored: Filtered canonical evidence.
        ranking: Eligible model ranking.
    """
    st.title(body="Find reliable cars")
    st.caption(
        body="Search model series, inspect evidence breadth, and build a shortlist."
    )
    ranking_view = st.selectbox(
        label="Rank evidence by",
        options=["combined", *sorted(scored.source_id.unique())],
        format_func=lambda source: (
            "Combined sources"
            if source == "combined"
            else {
                "adac": "ADAC breakdowns only",
                "traficom": "Traficom inspections only",
                "tuv": "TÜV defects only",
                "mot": "UK MOT only",
            }.get(source, source)
        ),
        key="ranking_view",
    )
    if ranking_view != "combined":
        source_cohorts = st.number_input(
            label="Minimum scored cohorts in this single-source view",
            min_value=1,
            value=3,
            step=1,
            key="single_source_cohorts",
        )
        ranking = reliability_ranking(
            scored=scored.loc[scored.source_id.eq(ranking_view)],
            min_sources=1,
            min_cohorts=int(source_cohorts),
        )
        st.caption(
            body="Single-source ranking uses only this source and the cohort minimum above. The sidebar's cross-source count and weights apply to the combined view. Outcomes and peer sets differ between sources."
        )
    query = st.text_input(
        label="Search make or model",
        placeholder="For example: Honda Jazz, Toyota, Golf",
        key="find_query",
    )
    makes = st.multiselect(
        label="Makes", options=sorted(scored.make.unique()), key="find_makes"
    )
    displayed = ranking.copy()
    if query:
        displayed = displayed.loc[
            (displayed.make + " " + displayed.model).str.contains(
                pat=query, case=False, regex=False
            )
        ]
    if makes:
        displayed = displayed.loc[displayed.make.isin(makes)]
    if displayed.empty:
        st.info(
            body="No ranked models match. Adjust the search or evidence coverage filters."
        )
    else:
        show_chart(
            figure=ranking_chart(ranking=displayed, top_n=20),
            key="find_rank",
            explanation="Scores retain the full original peer cohort even when you search one make. Source count, cohort count, worst cohort and available weight help you judge breadth. Two scores based on different source sets may have different meanings.",
        )
        table = displayed.copy()
        for name in ["score", "score_spread", "worst_cohort_score", "weight_coverage"]:
            table[name] = table[name].round(decimals=2)
        show_table(frame=table, name="reliability_shortlist", key="find_results")
    if not ranking.empty:
        options = list(ranking.model_id)
        names = dict(
            zip(ranking.model_id, ranking.make + " " + ranking.model, strict=True)
        )
        shortlist = st.multiselect(
            label="Save models to your shortlist",
            options=options,
            default=[
                key
                for key in st.session_state.get("shortlist_ids", [])
                if key in options
            ],
            format_func=lambda key: names[key],
            key="saved_shortlist",
        )
        st.session_state["shortlist_ids"] = shortlist
    with st.expander(label="? Methods and table columns"):
        st.write(METHODS)
        st.write(
            "score_spread is the difference between the best and worst source average. weight_coverage is the percentage of selected source weight supported by this model. Neither is a calibrated confidence interval. cohorts counts source-cohort observations, not independent vehicles."
        )
    raw = scored.copy()
    if query:
        raw = raw.loc[raw.source_model.str.contains(pat=query, case=False, regex=False)]
    if makes:
        raw = raw.loc[raw.make.isin(makes)]
    with st.expander(label="Browse original observations, including grouped models"):
        show_table(frame=raw, name="source_observations", key="find_raw")
    ungrouped = scored.loc[~scored.model_grouped].drop_duplicates(subset="model_id")
    if not ungrouped.empty:
        names = dict(
            zip(ungrouped.model_id, ungrouped.make + " " + ungrouped.model, strict=True)
        )
        inspector_ids = list(names)
        if query:
            inspector_ids = [
                key
                for key in inspector_ids
                if query.casefold() in names[key].casefold()
            ]
        if inspector_ids:
            st.subheader(body="Why does this model rank here?")
            chosen = st.selectbox(
                label="Inspect a model's evidence",
                options=inspector_ids,
                format_func=lambda key: names[key],
                key="inspect_model",
            )
            rows = scored.loc[scored.model_id.eq(chosen)].copy()
            profile = (
                rows.groupby(by=["source_name", "unit"], dropna=False)
                .agg(
                    mean_relative_score=("relative_score", "mean"),
                    scored_cohorts=("relative_score", "count"),
                    observations=("value", "size"),
                    youngest_age=("age_min", "min"),
                    oldest_age=("age_max", "max"),
                    mean_published_rate=("value", "mean"),
                    known_denominator_rows=("sample_size", "count"),
                )
                .reset_index()
            )
            st.caption(
                body="Each source contributes one mean cohort percentile in the combined score, with your chosen weights. Means can cover different ages, registration cohorts and generations. Published rates retain different units; they are not averaged across sources."
            )
            show_table(
                frame=profile, name="model_source_diagnostic", key="model_diagnostic"
            )
            coverage = profile.loc[:, ["youngest_age", "oldest_age"]].drop_duplicates()
            if len(coverage) > 1:
                st.warning(
                    body="The sources cover different age ranges for this model. Treat its combined score as exploratory; use source-specific or narrow-age views and shared-cohort comparisons."
                )
            if (
                profile.mean_relative_score.max() - profile.mean_relative_score.min()
                >= 30
            ):
                st.warning(
                    body="The source scores strongly disagree. Inspect the raw rates and outcome definitions before interpreting the combined position."
                )
            with st.expander(
                label="Exact observations, counts and original source links"
            ):
                show_table(
                    frame=rows,
                    name="model_diagnostic_observations",
                    key="model_diagnostic_raw",
                )


def comparison_page(*, scored: pd.DataFrame, ranking: pd.DataFrame) -> None:
    """Compare up to six explicitly selected model series.

    Args:
        scored: Selected scored evidence.
        ranking: Current eligible model ranking.
    """
    st.title(body="Compare cars")
    identities = scored.drop_duplicates(subset="model_id").sort_values(
        by=["make", "model"]
    )
    names = dict(
        zip(identities.model_id, identities.make + " " + identities.model, strict=True)
    )
    available = list(names)
    preferred = [
        key for key in st.session_state.get("shortlist_ids", []) if key in available
    ]
    default = preferred[:6] or [
        key for key in ranking.head(n=3).model_id if key in available
    ]
    selected = st.multiselect(
        label="Choose up to six model series",
        options=available,
        default=default,
        format_func=lambda key: names[key],
        max_selections=6,
        key="compare_models",
    )
    if not selected:
        st.info(body="Select models to display their evidence.")
        return
    selected_rows = scored.loc[scored.model_id.isin(selected)].copy()
    matched_only = st.checkbox(
        label="Compare only cohorts shared by every selected model",
        key="comparison_shared",
        help="Match source, report, outcome, age bounds and registration cohort. Exact engines, generation and usage are not controlled.",
    )
    chart_rows = selected_rows
    if len(selected) >= 2 and not selected_rows.model_grouped.any():
        matched = shared_cohort_evidence(scored=scored, model_ids=selected)
        st.caption(
            body=f"Shared-cohort support: {len(matched):,} observations of {len(selected_rows):,} selected. No missing cohorts are filled."
        )
        if matched_only:
            chart_rows = matched
        with st.expander(label="Compare rates on identical published cohorts"):
            contrasts = shared_cohort_contrasts(scored=scored, model_ids=selected)
            if contrasts.empty:
                st.info(
                    body="These models have no shared published cohorts under the current filters. Choose fewer models or broaden the age range."
                )
            else:
                st.caption(
                    body="Positive gaps mean the first model has a higher, less favourable rate. Means give each shared cohort equal weight; sources and units remain separate. These are descriptive gaps, not significance tests."
                )
                show_table(
                    frame=contrasts,
                    name="shared_cohort_comparison",
                    key="comparison_shared_rates",
                )
    elif matched_only:
        st.info(
            body="Shared-cohort comparisons need at least two ungrouped model series."
        )
    table = ranking.loc[ranking.model_id.isin(selected)]
    if not table.empty:
        show_table(frame=table, name="car_comparison", key="comparison_rank")
    show_chart(
        figure=age_chart(scored=chart_rows, model_ids=selected),
        key="comparison_age",
        explanation="Each source has its own panel and units. A marker is a published age cohort; lines connect different cohorts rather than following the same cars. TÜV age bins use their midpoints. Missing cohorts are absent, never zero.",
    )
    known = counted_failure_rates(evidence=chart_rows)
    if not known.empty:
        intervals = [
            wilson_interval(failures=int(row.event_count), tests=int(row.sample_size))
            for row in known.itertuples()
        ]
        known["failure_pct_lower_95"] = [pair[0] for pair in intervals]
        known["failure_pct_upper_95"] = [pair[1] for pair in intervals]
        with st.expander(label="Known test counts and 95% intervals"):
            st.caption(
                body="Wilson intervals describe binomial sampling uncertainty in observed tests. Repeated tests and systematic differences are not accounted for; no intervals are invented for unpublished denominators."
            )
            show_table(frame=known, name="comparison_test_counts", key="comparison_ci")
    with st.expander(label="All selected evidence and source links"):
        show_table(
            frame=selected_rows, name="comparison_evidence", key="comparison_raw"
        )


def age_mileage_page(*, scored: pd.DataFrame) -> None:
    """Show source-specific age trends and genuine odometer measurements.

    Args:
        scored: Selected evidence.
    """
    st.title(body="Age and mileage")
    if scored.empty:
        st.info(body="No evidence matches the selected filters.")
        return
    source = st.selectbox(
        label="Source to explore",
        options=sorted(scored.source_id.unique()),
        format_func=lambda source: SOURCE_NAMES.get(source, source),
        key="age_source",
    )
    rows = scored.loc[scored.source_id.eq(source)].copy()
    year = st.selectbox(
        label="Report year",
        options=sorted(rows.report_year.astype(int).unique(), reverse=True),
        key="age_year",
    )
    rows = rows.loc[rows.report_year.eq(year)]
    identities = rows.drop_duplicates(subset="model_id").sort_values(
        by=["make", "model"]
    )
    names = dict(
        zip(identities.model_id, identities.make + " " + identities.model, strict=True)
    )
    selected = st.multiselect(
        label="Models for the age profile",
        options=list(names),
        default=list(names)[:3],
        format_func=lambda key: names[key],
        max_selections=8,
        key="age_models",
    )
    relative = st.checkbox(
        label="Show relative cohort scores", value=False, key="age_relative"
    )
    if selected:
        show_chart(
            figure=age_chart(scored=rows, model_ids=selected, relative=relative),
            key="age_profile",
            explanation="Age profiles use different registration cohorts and may cross model generations. A rising failure rate with age does not estimate how your own car will age. Relative scores compare each point with its source-cohort peers.",
        )
    if rows.mileage_basis.eq("mean_odometer").any() and rows.mileage_km.notna().any():
        show_chart(
            figure=mileage_chart(evidence=rows),
            key="mileage_scatter",
            explanation="Each dot is a published model cohort. The x-axis is mean odometer mileage; the y-axis is an inspection rate. Colour represents observed age. This plot shows associations without adjusting for maintenance, generation or usage.",
        )
    else:
        st.info(
            body="Mean odometer data are unavailable for this source. Select Traficom or TÜV to explore mileage; ADAC reports a different annual-mileage measure."
        )
    st.subheader(body="Better and poorer evidence across ages")
    map_order = st.selectbox(
        label="Age heatmap ordering",
        options=["both", "weakest", "strongest"],
        format_func=lambda value: {
            "both": "Both ranking ends",
            "weakest": "Weakest relative evidence first",
            "strongest": "Strongest relative evidence first",
        }[value],
        key="age_heatmap_order",
    )
    map_limit = st.slider(
        label="Models in age heatmap",
        min_value=10,
        max_value=80,
        value=30,
        step=5,
        key="age_heatmap_limit",
    )
    heatmap_query = st.text_input(
        label="Search age heatmap models", key="age_heatmap_query"
    )
    map_rows = rows
    if heatmap_query:
        map_rows = rows.loc[
            (rows.make + " " + rows.model).str.contains(
                pat=heatmap_query, case=False, regex=False
            )
        ]
    show_chart(
        figure=age_score_heatmap(
            scored=map_rows, max_models=map_limit, order=map_order
        ),
        key="age_heatmap",
        explanation="Each cell compares a model with its original source-cohort peers. Teal means lower published rates relative to peers; red means higher rates. Grey cells have no score and are never zero. Hover for the raw rate in this source's units. Row ordering averages available age scores, so coverage can differ. These are different cars/cohorts, not longitudinal survival or controlled engine comparisons.",
    )
    with st.expander(label="? Why is the old faults-per-kilometre ratio absent?"):
        st.write(
            "Dividing an inspection failure percentage by average mileage is not an exposure-adjusted failure rate. A model can look better simply because its owners drive farther. The original scripts retain the legacy field for compatibility; the new app uses raw rates, same-age comparisons and explicit mileage context."
        )


def agreement_page(*, scored: pd.DataFrame, ranking: pd.DataFrame) -> None:
    """Inspect contradictory evidence without forcing it into a single conclusion.

    Args:
        scored: Selected evidence.
        ranking: Current model ranking.
    """
    st.title(body="Where the evidence agrees")
    st.write(
        "Inspection failures and breakdowns answer different questions. A disagreement can be useful when choosing what to investigate."
    )
    wide = source_agreement(scored=scored)
    if wide.empty:
        st.info(body="No scored model evidence is available.")
        return
    identities = scored.drop_duplicates(subset="model_id")
    names = dict(
        zip(identities.model_id, identities.make + " " + identities.model, strict=True)
    )
    heatmap_view = st.selectbox(
        label="Heatmap view",
        options=[
            "Both ends of the ranking",
            "Strongest relative evidence",
            "Weakest relative evidence",
            "Largest source disagreement",
            "All qualifying models (up to 80)",
        ],
        key="heatmap_view",
    )
    query = st.text_input(label="Search heatmap models", key="heatmap_query")
    candidates = ranking.copy()
    if query:
        candidates = candidates.loc[
            (candidates.make + " " + candidates.model).str.contains(
                pat=query, case=False, regex=False
            )
        ]
    if heatmap_view.startswith("Both"):
        selection = pd.concat(
            objs=[candidates.head(n=12), candidates.tail(n=12)]
        ).drop_duplicates(subset="model_id")
    elif heatmap_view.startswith("Weakest"):
        selection = candidates.sort_values(by="score", ascending=True).head(n=24)
    elif heatmap_view.startswith("Largest"):
        selection = candidates.sort_values(by="score_spread", ascending=False).head(
            n=24
        )
    else:
        selection = candidates.head(n=80 if heatmap_view.startswith("All") else 24)
    top = [key for key in selection.model_id if key in wide.index]
    if not top:
        top = [
            key
            for key in wide.index
            if not query or query.casefold() in names.get(key, key).casefold()
        ][:24]
    st.caption(
        body=f"Showing {len(top)} model series. Red = poorer relative evidence; teal = better; grey ‘No data’ = unavailable. The ‘both ends’ view omits middle-ranked models. Scores are relative percentiles, not failure probabilities."
    )
    show_chart(
        figure=agreement_heatmap(wide=wide.loc[top], names=names),
        key="agreement_heatmap",
        explanation="Cells show one mean relative score per source for the current age filters. Teal is favourable; red is unfavourable. Blank cells mean no usable source evidence. These averages may cover different cohorts; this is a descriptive comparison, not a meta-analysis.",
    )
    if len(wide.columns) >= 2:
        first = st.selectbox(
            label="Horizontal-axis source",
            options=list(wide.columns),
            format_func=lambda source: SOURCE_NAMES.get(source, source),
            key="agreement_first",
        )
        alternatives = [column for column in wide.columns if column != first]
        second = st.selectbox(
            label="Vertical-axis source",
            options=alternatives,
            key="agreement_second",
            format_func=lambda source: SOURCE_NAMES.get(source, source),
        )
        show_chart(
            figure=agreement_scatter(
                wide=wide, first=first, second=second, names=names
            ),
            key="agreement_scatter",
            explanation="Points on the diagonal have similar average percentiles in the two sources. Points far away highlight disagreement. Hover for model names. A high correlation would not make inspection defects and roadside breakdowns the same outcome.",
        )
    export = wide.reset_index()
    export.insert(loc=1, column="Model series", value=export.model_id.map(arg=names))
    show_table(frame=export, name="source_agreement", key="agreement_table")
    with st.expander(label="How strongly do sources agree across overlapping models?"):
        st.caption(
            body="Spearman correlation describes ordering agreement on paired model averages; cohorts can differ. At least five pairs are needed. This is not a causal or independent meta-analysis."
        )
        show_table(
            frame=pairwise_source_agreement(wide=wide),
            name="source_agreement_statistics",
            key="paired_agreement",
        )


def deeper_analysis_page(
    *,
    scored: pd.DataFrame,
    ranking: pd.DataFrame,
    weights: dict[str, float],
    min_sources: int,
    min_cohorts: int,
) -> None:
    """Explore descriptive distributions, actual counts and source sensitivity.

    Args:
        scored: Evidence selected by the sidebar's explicit filters.
        ranking: Baseline ranking for the selected weights and thresholds.
        weights: Current non-negative source weights.
        min_sources: Baseline minimum source count.
        min_cohorts: Required scored source-cohort observations.
    """
    st.title(body="Deeper data interrogation")
    st.caption(
        body="All results use your current evidence filters. These analyses describe loaded model-series/cohort observations, not exact engines, individual cars or lifespans."
    )
    if scored.empty:
        st.info(body="No evidence matches the current filters.")
        return
    st.subheader(body="Rate distributions and data coverage")
    show_chart(
        figure=rate_distribution_chart(evidence=scored),
        key="deep_distributions",
        explanation="Each source keeps its outcome and units. Box centres are medians, boxes span quartiles, and dashed markers show means. Each published cohort gets equal weight; this is not a fleet-weighted rate and can mix ages.",
    )
    show_table(
        frame=source_rate_summary(evidence=scored),
        name="source_rate_statistics",
        key="deep_statistics",
    )
    st.subheader(body="Actual failure counts and sampling uncertainty")
    counted = counted_failure_rates(evidence=scored)
    show_chart(
        figure=count_uncertainty_chart(counted=counted),
        key="deep_counts",
        explanation="These are the 24 largest counted cohorts, ordered by sample size. Points use actual failed tests divided by tests, with 95% Wilson intervals. Different ages must not be ranked directly. Repeated cars, maintenance and selection effects are outside these intervals.",
    )
    with st.expander(label="Inspect all counted rates and intervals"):
        show_table(
            frame=counted, name="counted_failure_intervals", key="deep_count_table"
        )
    st.subheader(body="How dependent is the ranking on one source?")
    show_chart(
        figure=score_spread_chart(ranking=ranking),
        key="deep_spread",
        explanation="Source-score spread measures disagreement, not statistical confidence. Hover for cohort count and evidence coverage. One-source models have no between-source comparison.",
    )
    if ranking.empty:
        st.info(
            body="No ranked models meet the current weights or coverage thresholds. Broaden the filters to explore source sensitivity."
        )
        return
    sensitivity = source_omission_sensitivity(
        scored=scored, weights=weights, min_sources=min_sources, min_cohorts=min_cohorts
    )
    if sensitivity.empty:
        st.info(
            body="Source-omission sensitivity needs ranked models and at least two sources with positive weights."
        )
    else:
        query = st.text_input(
            label="Search sensitivity results by model", key="sensitivity_query"
        )
        if query:
            sensitivity = sensitivity.loc[
                (sensitivity.make + " " + sensitivity.model).str.contains(
                    pat=query, case=False, regex=False
                )
            ]
        st.caption(
            body="Each row removes one source. Both ranks use the same remaining model set for that omission; positive rank movement means improvement. The alternative needs one remaining source and retains your cohort-count threshold. Unavailable alternatives are missing, not zero. Changes indicate sensitivity to evidence choices, not confidence intervals."
        )
        show_table(
            frame=sensitivity,
            name="source_omission_sensitivity",
            key="deep_sensitivity",
        )


def safety_page(*, events: pd.DataFrame, directory: Path) -> None:
    """Browse saved safety events and execute explicit exact-US-model lookups.

    Args:
        events: Saved recall and complaint records.
        directory: Active local data folder.
    """
    st.title(body="Recalls and safety complaints")
    st.info(
        body="These are safety signals. Counts have no fleet denominator and never enter reliability rankings. NHTSA records concern US vehicles; UK engines, specifications and recall scope may differ."
    )
    st.link_button(
        label="Check UK recalls with DVSA",
        url="https://www.check-vehicle-recalls.service.gov.uk/",
    )
    st.link_button(
        label="Check a vehicle with the manufacturer's recall service",
        url="https://www.gov.uk/check-vehicle-recall",
    )
    with st.form(key="us_lookup"):
        st.subheader(body="Look up an exact US vehicle")
        left, middle, right = st.columns(spec=[2, 2, 1])
        make = left.text_input(label="US make", value="Toyota", key="us_make")
        model = middle.text_input(label="US model", value="Corolla", key="us_model")
        year = right.number_input(
            label="US model year",
            min_value=1950,
            max_value=2100,
            value=2020,
            step=1,
            key="us_year",
        )
        types = st.multiselect(
            label="Safety evidence",
            options=["recall", "complaint"],
            default=["recall"],
            key="us_types",
        )
        fresh = st.checkbox(label="Refresh cached lookup", value=False, key="us_force")
        submitted = st.form_submit_button(label="Retrieve official NHTSA records")
    if submitted:
        try:
            with st.spinner(text="Retrieving official records…"):
                found, _ = nhtsa_lookup(
                    directory=directory,
                    make=make,
                    model=model,
                    model_year=int(year),
                    event_types=types,
                    force=fresh,
                )
            st.success(
                body=f"Lookup completed: {len(found)} records. No returned records does not prove that the vehicle has no defects."
            )
            _, events, _ = load_snapshot(directory=directory)
        except (OSError, ValueError, KeyError, TypeError) as error:
            LOGGER.exception("Safety lookup failed")
            st.error(body=f"Lookup failed; previous results retained. {error}")
    if events.empty:
        st.caption(
            body="No safety records have been loaded. Use an explicit lookup above."
        )
        return
    browse_types = st.multiselect(
        label="Saved event types to display",
        options=["recall", "complaint"],
        default=["recall", "complaint"],
        key="events_types",
        help="This controls the saved table. The lookup form above controls new records retrieved.",
    )
    query = st.text_input(
        label="Filter saved make, model or component", key="events_query"
    )
    filtered = events.loc[events.event_type.isin(browse_types)].copy()
    if query:
        filtered = filtered.loc[
            (
                filtered.make + " " + filtered.model + " " + filtered.component
            ).str.contains(pat=query, case=False, regex=False)
        ]
    st.caption(
        body=f"{len(filtered):,} saved events match. Counts reflect reporting and vehicle popularity, not comparative failure risks."
    )
    show_table(frame=filtered, name="safety_context", key="events_table")
    if not filtered.empty:
        selected_event = st.selectbox(
            label="Read an event in full",
            options=range(len(filtered)),
            format_func=lambda index: (
                f"{filtered.iloc[index].event_type} · {filtered.iloc[index].event_id} · {filtered.iloc[index].make} {filtered.iloc[index].model}"
            ),
            key="event_detail",
        )
        item = filtered.iloc[selected_event]
        with st.expander(label="Full event details", expanded=True):
            for field in ["component", "summary", "consequence", "remedy"]:
                if pd.notna(item[field]) and str(item[field]).strip():
                    st.markdown(body=f"**{field.capitalize()}**")
                    st.text(body=str(item[field]))
            st.link_button(label="Official source", url=str(item.source_url))


def methods_page(*, evidence: pd.DataFrame) -> None:
    """Explain methodology, source availability and glossary terms.

    Args:
        evidence: All currently loaded reliability observations.
    """
    st.title(body="Sources, methods and meaning")
    st.write(METHODS)
    catalogue = source_catalogue()
    catalogue["loaded_rate_rows"] = (
        catalogue.source_id.map(arg=evidence.source_id.value_counts())
        .fillna(value=0)
        .astype(int)
    )
    show_table(frame=catalogue, name="data_sources", key="catalogue")
    for title, body in {
        "Inspection failure": "A roadworthiness test outcome. It can reflect servicing, tyres, lights, brakes and structural faults; it does not capture every mechanical repair or roadside failure.",
        "Roadside breakdown": "A failure attended by the reporting breakdown service. The ADAC rate uses registered vehicles as exposure; it does not cover all non-breakdown repairs or every assistance provider.",
        "Relative score": "A source-cohort percentile: 100 for the lowest rate, 0 for the highest, with average ranks for ties. Scores are averaged within each source before weighting sources. The scale is relative to available models, not a failure percentage.",
        "Sample size and intervals": "Where actual failure/test counts exist, Wilson intervals can describe sampling uncertainty. TÜV's secondary tables and ADAC PDFs do not publish model denominators here; these remain unknown. Repeated tests can violate independent-binomial assumptions.",
        "Registration cohort and report year": "A 2025 ADAC report records 2024 breakdowns. A Traficom 2026 publication records 2025 inspections. TÜV age-bin registration-year estimates are approximate and are not confirmed production years.",
        "Model matching": "Matching is conservative and explicit: punctuation, accents and documented labels such as VW/Volkswagen or B-Klasse/B-Class are normalised. Opel and Vauxhall, distinct engines and combined model groups are not silently merged. Source labels remain visible.",
        "Breadth and disagreement": "More sources and cohorts give broader evidence, but are not automatically independent or unbiased. Between-source spread flags different relative outcomes. Missing sources are never assigned zero or treated as proof of reliability.",
        "Newest reports and restricted sources": "Official 2026 ADAC and TÜV summaries exist, but a complete accessible numeric model table is not automatically available. The app uses actually loaded tables and labels their years. Which?, Consumer Reports and selective What Car? articles are documented candidates, not claimed imported feeds.",
    }.items():
        with st.expander(label=f"? {title}"):
            st.write(body)


def management_page(
    *, directory: Path, evidence: pd.DataFrame, events: pd.DataFrame, manifest: dict
) -> None:
    """Manage explicit source refreshes and export complete data with provenance.

    Args:
        directory: Current external data directory.
        evidence: Complete loaded rate observations.
        events: Complete loaded safety events.
        manifest: Snapshot metadata.
    """
    st.title(body="Data management")
    st.caption(body=f"Data folder: {directory}")
    st.write(
        "Refreshes download public source tables and preserve previous observations when a source fails. Loading the app itself does not contact external services."
    )
    selected = st.multiselect(
        label="Sources to refresh",
        options=["adac", "traficom", "tuv"],
        default=["adac", "traficom"],
        key="refresh_selection",
    )
    force = st.checkbox(label="Re-download cached source files", key="refresh_force")
    offline = st.checkbox(label="Use cached files only", key="refresh_offline")
    if st.button(label="Refresh selected evidence", key="refresh_run"):
        try:
            with st.spinner(text="Downloading and validating source tables…"):
                status, _ = refresh_sources(
                    directory=directory, sources=selected, force=force, offline=offline
                )
            st.dataframe(data=status, hide_index=True)
            if status.status.eq("failed").any():
                st.warning(
                    body="Some sources failed. Their previous observations remain available; inspect the status details above."
                )
            else:
                st.success(
                    body="All selected sources updated. Use Reload saved data to display the new snapshot."
                )
        except (OSError, ValueError, KeyError, TypeError) as error:
            st.error(body=f"Refresh could not complete: {error}")
    if st.button(label="Reload saved data", key="reload_data"):
        st.rerun()
    st.subheader(body="Complete dataset exports")
    st.download_button(
        label="Download all evidence as TSV",
        data=tsv_bytes(frame=evidence),
        file_name="all_reliability_evidence.tsv",
        mime="text/tab-separated-values",
        key="all_evidence",
    )
    st.download_button(
        label="Download complete Excel workbook",
        data=excel_bytes(
            sheets={
                "Reliability evidence": evidence,
                "Safety events": events,
                "Sources": source_catalogue(),
            }
        ),
        file_name="best_cars_evidence.xlsx",
        key="all_excel",
    )
    with st.expander(label="Snapshot provenance"):
        st.json(body=manifest)
    with st.expander(label="Import UK MOT or another verified local source"):
        st.write(
            "Use the CLI import command for sizeable MOT downloads; it streams chunks and deduplicates test IDs across monthly files on disk. The README documents 2024 headered comma-delimited results, historical headerless pipe-delimited files, TSV aggregates, optional ADAC PDF support and Traficom workbooks. Imports validate before updating a saved snapshot."
        )
        st.code(
            body='best-cars import --kind mot-raw --input "/path/to/MOT testing data results (2024).zip" --delimiter comma --mot-conflicts exclude --data-dir "$HOME/best_cars_data"',
            language="bash",
        )


def run() -> None:
    """Run the local explorer, failing visibly if a saved dataset is corrupt."""
    st.set_page_config(
        page_title="Best Cars · Reliability explorer", page_icon="🚗", layout="wide"
    )
    st.markdown(
        body="""<style>
    .block-container {padding-top:2rem; max-width:1650px;}
    h1 {letter-spacing:-0.03em; font-weight:750;}
    [data-testid="stMetric"] {background:#eef6f7; border-radius:12px; padding:16px; color:#15364a;}
    [data-testid="stSidebar"] h2 {font-size:1.2rem;}
    </style>""",
        unsafe_allow_html=True,
    )
    directory = data_directory()
    with st.sidebar:
        st.title(body="Best Cars")
        st.caption(body="RELIABILITY EXPLORER")
        page = st.radio(label="Explore", options=PAGES, key="page")
    try:
        evidence, events, manifest = load_snapshot(directory=directory)
    except (OSError, ValueError, KeyError) as error:
        st.error(
            body=f"Saved dataset could not be verified: {error}. Select a valid data folder or restore a previous snapshot."
        )
        return
    scored, weights, min_sources, min_cohorts = sidebar_filters(evidence=evidence)
    if page in {"Recalls and complaints", "Sources and methods", "Data management"}:
        st.sidebar.caption(
            body="Reliability filters apply to the analysis pages. This page uses complete saved sources or its own safety filters."
        )
    try:
        ranking = reliability_ranking(
            scored=scored,
            weights=weights,
            min_sources=min_sources,
            min_cohorts=min_cohorts,
        )
    except ValueError as error:
        st.warning(body=str(error))
        ranking = reliability_ranking(scored=scored.iloc[:0])
    if page == "Overview":
        overview_page(
            evidence=evidence, scored=scored, ranking=ranking, manifest=manifest
        )
    elif page == "Find reliable cars":
        find_page(scored=scored, ranking=ranking)
    elif page == "Compare cars":
        comparison_page(scored=scored, ranking=ranking)
    elif page == "Age and mileage":
        age_mileage_page(scored=scored)
    elif page == "Evidence agreement":
        agreement_page(scored=scored, ranking=ranking)
    elif page == "Deeper analysis":
        deeper_analysis_page(
            scored=scored,
            ranking=ranking,
            weights=weights,
            min_sources=min_sources,
            min_cohorts=min_cohorts,
        )
    elif page == "Recalls and complaints":
        safety_page(events=events, directory=directory)
    elif page == "Sources and methods":
        methods_page(evidence=evidence)
    else:
        management_page(
            directory=directory, evidence=evidence, events=events, manifest=manifest
        )


if __name__ == "__main__":
    run()
