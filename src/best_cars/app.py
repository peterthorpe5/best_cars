"""Streamlit reliability explorer: evidence, comparison and interpretable graphics."""

import io
import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from best_cars.analysis import (
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
    scenario_rank_chart,
    score_spread_chart,
)
from best_cars.comparison import (
    matched_model_rates,
    temporal_profiles,
    wilson_age_groups,
)
from best_cars.comparison_charts import (
    evidence_coverage_chart,
    matched_rate_chart,
    peer_gap_chart,
    source_rate_chart,
    temporal_chart,
)
from best_cars.fairness import (
    cohort_contexts,
    evidence_profiles,
    exploratory_ranking,
    source_cohort_ranking,
)
from best_cars.fairness import coverage_matched_ranking as reliability_ranking
from best_cars.interrogation import (
    MATCH_KEYS,
    cohort_peer_benchmarks,
    shared_cohort_contrasts,
    shared_cohort_evidence,
    source_omission_sensitivity,
    weighting_scenarios,
)
from best_cars.interval_evidence import source_interval_availability, source_rate_points
from best_cars.pipeline import nhtsa_lookup, refresh_sources
from best_cars.propulsion import propulsion_catalogue, select_propulsion
from best_cars.repair_costs import repair_budget, repair_references
from best_cars.source_guide import PAGE_HELP, SOURCE_GUIDE
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
    "Mileage explorer",
    "Evidence agreement",
    "Deeper analysis",
    "Electric and hybrid vehicles",
    "Repair costs",
    "Data sources and fault definitions",
    "Recalls and complaints",
    "Sources and methods",
    "Data management",
]
METHODS = """Start with separate inspection-condition and roadside-breakdown evidence. Source rankings use explicit published cohorts. The optional combined view weights sources within outcomes, then inspection and breakdown outcomes. Exploratory mode retains differing coverage: missing source percentiles explicitly assume 50; bounds substitute 0 and 100 and are sensitivity ranges, not confidence intervals. Ages, sample sizes and missing sources remain visible. Strict mode requires every active source and identical recorded source/cohort coverage; ranks apply within one block only. Neither mode estimates engine durability, lifetime, repair cost or failure probability. Recalls and voluntary complaints never enter scores."""


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
    labels["score"] = (
        "Source-cohort percentile"
        if frame.attrs.get("ranking_label")
        else "Optional preference score"
    )
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
    displayed = frame.copy()
    for column in ["source_ids", "omitted_source"]:
        if column in displayed:
            displayed[column] = displayed[column].map(
                arg=lambda value: "; ".join(
                    SOURCE_NAMES.get(item.strip(), item.strip())
                    for item in str(value).split(";")
                )
            )
    st.dataframe(data=displayed, hide_index=True, width="stretch", column_config=config)
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


def make_highlights(
    *, makes: list[str], label: str, key: str, enabled: bool
) -> list[str]:
    """Select display highlights while removing stale make selections.

    Args:
        makes: Makes available under the current chart filters.
        label: Human-readable control label.
        key: Unique widget identifier.
        enabled: Whether the chart currently uses make colours.

    Returns:
        Selected available makes; this does not filter evidence or ranks.
    """
    available = sorted(set(makes))
    current = st.session_state.get(key, [])
    valid = [make for make in current if make in available]
    if valid != current:
        st.session_state[key] = valid
    return st.multiselect(
        label=label,
        options=available,
        key=key,
        disabled=not enabled,
        help="Choose Make colours to highlight manufacturers. Other competitors remain visible in grey; values and ranks do not change.",
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
            if changed_preset or "inspection_weight" not in st.session_state:
                st.session_state["inspection_weight"] = (
                    2.0 if preset == "Emphasise inspection condition" else 1.0
                )
                st.session_state["breakdown_weight"] = (
                    2.0 if preset == "Emphasise avoiding breakdowns" else 1.0
                )
            for source, value in starting.items():
                if changed_preset or f"weight_{source}" not in st.session_state:
                    st.session_state[f"weight_{source}"] = value
            st.session_state["previous_weight_preset"] = preset
            st.caption(
                body="These controls affect only optional preference composites. Inspection and breakdown outcomes receive separate weights first. Source sliders distribute influence within an outcome; with one source in an outcome, its non-zero source weight cancels. Zero omits a source. No weights establish trust, lifetime or repair cost. Exploratory comparisons display gaps; strict comparisons require identical coverage."
            )
            st.slider(
                label="Inspection outcome preference",
                min_value=0.0,
                max_value=3.0,
                value=None,
                step=0.25,
                key="inspection_weight",
            )
            st.slider(
                label="Roadside breakdown outcome preference",
                min_value=0.0,
                max_value=3.0,
                value=None,
                step=0.25,
                key="breakdown_weight",
            )
            for source in selected:
                weights[source] = st.slider(
                    label=f"{SOURCE_NAMES.get(source, source)} weight",
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
    st.title(body="Understand a car's reliability evidence")
    st.write(
        "Explore inspection failures and roadside breakdowns. Compare models, follow the age patterns, and see how strong the evidence is."
    )
    st.caption(
        body=f"Current filters: {len(scored):,} observations. Inspection condition and roadside breakdowns are different outcomes. Model series can span engines and generations; scores do not predict lifetime or repair bills."
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
    st.subheader(body="Separate source evidence comes first")
    profile_query = st.text_input(
        label="Search evidence profiles by make or model", key="overview_profile_query"
    )
    profiles = evidence_profiles(scored=scored)
    if profile_query:
        profiles = profiles.loc[
            (profiles.make + " " + profiles.model).str.contains(
                pat=profile_query, case=False, regex=False
            )
        ]
    st.caption(
        body="These are source profiles, not an overall league table. Mean percentiles can cover different ages. Use Find reliable cars to choose a source and exact published cohort, or search a model here to inspect conflicting outcomes."
    )
    show_table(
        frame=profiles,
        name="separate_source_evidence_profiles",
        key="overview_profiles",
    )
    st.info(
        body="An excellent inspection result can coexist with roadside breakdowns. Neither alone establishes expensive mechanical faults or short service life. Optional preference comparisons are secondary. Choose exploratory coverage with explicit gaps or strict matching."
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
        options=[
            "choose",
            *sorted(
                scored.source_id.unique(),
            ),
            "combined",
        ],
        format_func=lambda source: (
            "Select the outcome and source to compare"
            if source == "choose"
            else "Optional combined preference comparison"
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
    if ranking_view == "choose":
        st.info(
            body="Choose an inspection source to compare roadworthiness, or ADAC to compare attended breakdowns. There is no default universal reliability league table."
        )
        ranking = ranking.iloc[:0]
    elif ranking_view != "combined":
        contexts = cohort_contexts(scored=scored, source_id=ranking_view)
        labels = dict(zip(contexts.context_id, contexts.context_label, strict=True))
        chosen_context = st.selectbox(
            label="Exact published cohort to compare",
            options=list(labels),
            format_func=lambda key: labels[key],
            key="find_cohort",
        )
        ranking = source_cohort_ranking(
            scored=scored, source_id=ranking_view, context_id=chosen_context
        )
        st.caption(
            body="This ranking compares one source, outcome, report, observed age band and registration cohort. Sidebar source-count and preference weights affect only optional composites. Scores retain original full-peer comparisons and do not measure lifetime, repair costs or faults per distance."
        )
    else:
        st.warning(
            body="This is an optional preference ordering. Exploratory mode shows differing coverage and neutral missing-source assumptions; strict mode compares only matching coverage. Neither establishes a universal reliability verdict."
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
        top_n = st.number_input(
            label="Models in ranking chart",
            min_value=1,
            max_value=max(1, len(displayed)),
            value=min(20, len(displayed)),
            key="find_top_n",
        )
        rank_colour = st.selectbox(
            label="Ranking colours",
            options=["sources", "make"],
            format_func=lambda value: (
                "Evidence breadth" if value == "sources" else "Make"
            ),
            key="find_colour",
        )
        highlights = make_highlights(
            label="Highlight makes without removing competitors",
            makes=displayed.make.tolist(),
            key="find_highlight_makes",
            enabled=rank_colour == "make",
        )
        st.caption(
            body="These are relative index scores, not percentages of reliable cars. A score of 99.6 does not mean a 99.6% chance of reliability. Make colours identify manufacturers, not better or worse outcomes."
        )
        show_chart(
            figure=ranking_chart(
                ranking=displayed,
                top_n=int(top_n),
                colour_by=rank_colour,
                highlight_makes=highlights,
            ),
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
        inspector_ids = list(
            dict.fromkeys(list(ranking.model_id) + sorted(names, key=names.get))
        )
        if query:
            inspector_ids = [
                key
                for key in inspector_ids
                if query.casefold() in names[key].casefold()
            ]
        if inspector_ids:
            st.subheader(body="Inspect a model's evidence and disagreements")
            chosen = st.selectbox(
                label="Inspect a model's evidence",
                options=inspector_ids,
                format_func=lambda key: names[key],
                key="inspect_model",
            )
            rows = scored.loc[scored.model_id.eq(chosen)].copy()
            if chosen not in set(ranking.model_id):
                st.info(
                    body="This model is outside the selected source-cohort or optional comparison block. Its evidence remains available below; no rank against the displayed competitors is implied."
                )
            profile = evidence_profiles(scored=rows)
            st.caption(
                body="These separate source averages are descriptive profiles, not an overall verdict. They can cover different ages and generations. Optional composites balance outcomes first. Exploratory mode shows coverage gaps; strict mode matches coverage. Raw rates retain different units."
            )
            show_table(
                frame=profile, name="model_source_diagnostic", key="model_diagnostic"
            )
            coverage = profile.loc[:, ["youngest_age", "oldest_age"]].drop_duplicates()
            if len(coverage) > 1:
                st.warning(
                    body="The sources cover different age ranges for this model. These averages do not make a matched comparison. Use exact-cohort source rankings and inspect original rates."
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


def comparison_page(
    *,
    scored: pd.DataFrame,
    ranking: pd.DataFrame,
    historical: pd.DataFrame | None = None,
) -> None:
    """Explore one model or compare up to twenty explicitly selected series.

    Args:
        scored: Selected scored evidence.
        ranking: Current eligible model ranking.
        historical: Evidence using the same sidebar filters with older reports
            retained, used only by the explicitly labelled historical tab.
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
        label="Choose up to twenty model series",
        options=available,
        default=default,
        format_func=lambda key: names[key],
        max_selections=20,
        key="compare_models",
    )
    if not selected:
        st.info(body="Select models to display their evidence.")
        return
    selected_rows = scored.loc[scored.model_id.isin(selected)].copy()
    st.caption(
        body="Select one model for a detailed profile, or several to compare. Inspection percentages and breakdown events per 1,000 vehicles remain separate. Age profiles compare different cars, not a tracked vehicle's ageing."
    )
    with st.expander(label="Comparison chart filters"):
        source_options = sorted(selected_rows.source_id.unique())
        compared_sources = st.multiselect(
            label="Sources in comparison charts",
            options=source_options,
            default=source_options,
            format_func=lambda value: SOURCE_NAMES.get(value, value),
            key="comparison_sources",
        )
        compared_ages = st.slider(
            label="Comparison chart age range",
            min_value=0,
            max_value=40,
            value=(0, 40),
            key="comparison_ages",
        )
    selected_rows = selected_rows.loc[
        selected_rows.source_id.isin(compared_sources)
        & selected_rows.age_max.ge(compared_ages[0])
        & selected_rows.age_min.le(compared_ages[1])
    ].copy()
    if selected_rows.empty:
        st.info(body="No selected observations match these comparison chart filters.")
        return
    st.subheader(body="Selected models: separate source profiles")
    show_table(
        frame=evidence_profiles(scored=selected_rows),
        name="comparison_source_profiles",
        key="comparison_profiles",
    )
    st.caption(
        body="Compare source definitions and age coverage before interpreting differences. The optional table follows the selected exploratory or strict method. In strict mode, models outside the block have no overall rank."
    )
    matched_only = st.checkbox(
        label="Compare only cohorts shared by every selected model",
        key="comparison_shared",
        help="Match source, report, outcome, age bounds and registration cohort. Exact engines, generation and usage are not controlled.",
    )
    chart_rows = selected_rows
    if len(selected) >= 2 and not selected_rows.model_grouped.any():
        matched = shared_cohort_evidence(scored=selected_rows, model_ids=selected)
        st.caption(
            body=f"Shared-cohort support: {len(matched):,} observations of {len(selected_rows):,} selected. No missing cohorts are filled."
        )
        if matched_only:
            chart_rows = matched.merge(
                right=scored.loc[:, MATCH_KEYS + ["model_id", "relative_score"]],
                on=MATCH_KEYS + ["model_id"],
                how="left",
                validate="one_to_one",
            )
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
        with st.expander(label="Optional combined preference scores and coverage"):
            show_table(frame=table, name="car_comparison", key="comparison_rank")
    comparison_colour = st.selectbox(
        label="Comparison profile colours",
        options=["model", "make"],
        format_func=lambda value: value.capitalize(),
        key="comparison_colour",
    )
    comparison_visualisations(
        scored=scored,
        rows=chart_rows,
        coverage_rows=selected_rows,
        historical=historical,
        selected=selected,
        names=names,
        colour_by=comparison_colour,
        sources=compared_sources,
        age_range=compared_ages,
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


def comparison_visualisations(
    *,
    scored: pd.DataFrame,
    rows: pd.DataFrame,
    coverage_rows: pd.DataFrame,
    historical: pd.DataFrame | None,
    selected: list[str],
    names: dict[str, str],
    colour_by: str,
    sources: list[str],
    age_range: tuple[int, int],
) -> None:
    """Render focused comparison views without mixing outcomes or peer sets.

    Args:
        scored: Complete sidebar-filtered peer population, before model selection.
        rows: Selected models and optional exact shared-cohort restriction.
        coverage_rows: Recorded selected evidence before shared-cohort restriction.
        historical: Older loaded reports under the same sidebar filters.
        selected: Ordered model selections, including unavailable shared rows.
        names: Display names for selected model IDs.
        colour_by: Model or manufacturer colour preference.
        sources: Page-level selected sources.
        age_range: Page-level overlapping age bounds.
    """
    tabs = st.tabs(
        tabs=[
            "Age profiles",
            "Same-age peers",
            "Matched models",
            "Rates and intervals",
            "Evidence coverage",
            "Across years",
        ]
    )
    with tabs[0]:
        relative = st.checkbox(
            label="Show relative peer percentiles",
            key="comparison_relative",
            help="Raw rates describe the published outcome. Percentiles compare each age point with its original full peer group, and are not failure probabilities.",
        )
        show_chart(
            figure=age_chart(
                scored=rows, model_ids=selected, relative=relative, colour_by=colour_by
            ),
            key="comparison_age",
            explanation="Each source has its own panel and units. A marker is a published age cohort; lines connect different cohorts rather than following the same cars. TÜV age bins use midpoints. Reports use different line styles. Missing cohorts are absent, never zero.",
        )
    with tabs[1]:
        benchmarks = cohort_peer_benchmarks(scored=scored)
        contexts = rows.loc[:, MATCH_KEYS + ["model_id"]]
        benchmarks = benchmarks.merge(
            right=contexts,
            on=MATCH_KEYS + ["model_id"],
            how="inner",
            validate="one_to_one",
        )
        st.caption(
            body="Negative bars favour the selected model: its rate is below the median of other models in that exact published cohort. Inspection gaps are percentage points; ADAC gaps are events per 1,000 vehicles."
        )
        show_chart(
            figure=peer_gap_chart(benchmarks=benchmarks, colour_by=colour_by),
            key="comparison_peers",
            explanation="Every reference excludes the target model and needs at least four other ungrouped models. Source, report, observation year, outcome, unit, age and registration bounds match. Peer models retain the complete sidebar-filtered population. These are descriptive gaps, not significance tests or adjustments for engine, mileage and maintenance.",
        )
        with st.expander(label="Download exact-cohort peer comparisons"):
            show_table(
                frame=benchmarks,
                name="comparison_peer_benchmarks",
                key="comparison_peer_table",
            )
    with tabs[2]:
        if len(selected) < 2 or rows.model_grouped.any():
            st.info(
                body="Select at least two ungrouped model series to compare exact published cohorts against a reference."
            )
        else:
            reference = st.selectbox(
                label="Reference model for matched rates",
                options=selected,
                format_func=lambda value: names[value],
                key="comparison_reference",
            )
            paired = matched_model_rates(
                evidence=rows, model_ids=selected, reference_id=reference
            )
            st.caption(
                body=f"{len(paired):,} exact-cohort pairs. Below the diagonal favours the compared model; above favours the reference. Models can appear at several ages. Unmatched observations are omitted."
            )
            show_chart(
                figure=matched_rate_chart(paired=paired, colour_by=colour_by),
                key="comparison_matched_rates",
                explanation="The horizontal rate belongs to the reference model and the vertical rate to the compared model. Both use the same source, report, outcome, age and registration cohort. Axes have equal numeric ranges within each panel. Different panels have different units. This controls recorded cohorts, not engines, countries across sources or individual vehicle use.",
            )
            show_table(
                frame=paired,
                name="comparison_matched_rates",
                key="comparison_matched_table",
            )
    with tabs[3]:
        show_interval_method_help(evidence=rows, key="comparison_interval_help")
        show_source_rate_panels(evidence=rows, key="comparison_published_rates")
        counted = counted_failure_rates(evidence=rows)
        if counted.empty:
            st.info(
                body="No actual failure/test count pairs are available for these models. Published ADAC and bundled TÜV denominators are unknown; confidence intervals are not invented."
            )
        else:
            st.caption(
                body="These points are failed inspections divided by recorded tests. Published rounded rates remain on hover. Unknown denominators are excluded, not treated as zero."
            )
            limit = st.number_input(
                label="Counted cohorts per age chart",
                min_value=1,
                max_value=max(24, len(counted)),
                value=min(24, len(counted)),
                key="comparison_count_limit",
            )
            ordering = st.selectbox(
                label="Counted cohort ordering",
                options=["age_min", "sample_size", "counted_rate_pct", "original"],
                format_func=lambda value: {
                    "age_min": "Observed age",
                    "sample_size": "Test count",
                    "counted_rate_pct": "Failure rate",
                    "original": "Original order",
                }[value],
                key="comparison_count_order",
            )
            ascending = st.checkbox(
                label="Ascending counted cohort order",
                value=True,
                key="comparison_count_ascending",
            )
            show_wilson_panels(
                counted=counted,
                top_n=int(limit),
                sort_by=ordering,
                ascending=ascending,
                colour_by="make" if colour_by == "make" else "uniform",
                key="comparison_count_chart",
            )
    with tabs[4]:
        st.caption(
            body="This grid shows recorded evidence before the shared-cohort restriction, so unmatched observations and gaps remain visible."
        )
        show_chart(
            figure=evidence_coverage_chart(
                evidence=coverage_rows, model_ids=selected, names=names
            ),
            key="comparison_coverage",
            explanation="Colours indicate evidence availability, not reliability. Grey: no recorded row under current filters. Gold: a published rate without a failure/test count pair. Teal: a rate plus actual counts, with n equal to tests for that particular cell. Counts are not pooled across years and sources; inspections need not represent unique cars.",
        )
    with tabs[5]:
        earlier = st.checkbox(
            label="Include earlier loaded reports in this time view",
            value=True,
            key="comparison_history",
            help="Only this tab includes older reports. Source, age, registration-year and minimum known sample filters still apply.",
        )
        history = historical if earlier and historical is not None else scored
        history = history.loc[
            history.model_id.isin(selected)
            & history.source_id.isin(sources)
            & history.age_max.ge(age_range[0])
            & history.age_min.le(age_range[1])
        ]
        mode = st.selectbox(
            label="Hold constant across years",
            options=["age", "registration"],
            format_func=lambda value: (
                "Published age band"
                if value == "age"
                else "Recorded registration cohort"
            ),
            key="comparison_time_mode",
        )
        st.caption(
            body="This tab uses historical population summaries independently of the shared-cohort checkbox. Observation year is the calendar year for ADAC/Traficom and the inspection-window ending year for TÜV."
        )
        try:
            profiles = temporal_profiles(
                evidence=history, model_ids=selected, mode=mode
            )
        except ValueError as error:
            st.warning(body=str(error))
            profiles = pd.DataFrame()
        if profiles.empty:
            st.info(
                body="No selected series has at least two loaded observation years under these filters. The saved ADAC snapshot has one observation year. Earlier reports cannot be inferred or fabricated."
            )
        else:
            labels = {
                key: f"{names.get(group.model_id.iloc[0], group.model_id.iloc[0])} · {SOURCE_NAMES.get(group.source_id.iloc[0], group.source_id.iloc[0])} · {group.series_label.iloc[0]}"
                for key, group in profiles.groupby(by="series_id", sort=False)
            }
            chosen = st.multiselect(
                label="Historical series to plot",
                options=list(labels),
                default=list(labels)[:6],
                format_func=lambda value: labels[value],
                max_selections=30,
                key="comparison_time_series",
            )
            show_chart(
                figure=temporal_chart(
                    profiles=profiles.loc[profiles.series_id.isin(chosen)],
                    colour_by=colour_by,
                ),
                key="comparison_time_chart",
                explanation="Age mode holds age bounds fixed but compares different registration cohorts in different years. Registration mode follows summaries of the same recorded registration cohort as its age changes, not the same individual cars; survival, inspection eligibility and fleet composition can change. Approximate TÜV registration estimates are excluded from registration mode. Lines do not estimate individual ageing, service life or a causal trend.",
            )
            with st.expander(label="Historical observations and provenance"):
                show_table(
                    frame=profiles,
                    name="comparison_historical_profiles",
                    key="comparison_history_table",
                )


def show_interval_method_help(*, evidence: pd.DataFrame, key: str) -> None:
    """Explain source counts, Wilson calculations and unsupported intervals.

    Args:
        evidence: Selected original observations with provenance and counts.
        key: Unique widget/download prefix for the containing page.
    """
    with st.expander(label="? Where do the points and interval bars come from?"):
        st.markdown(
            body="**Traficom** is Finland's official transport authority. Its Transport Register supplies periodic-inspection outcomes. Our API import reads **Number of periodic inspections** as n and **Failed** as x, for each model, registration year and inspection year. An inspection can fail on roadworthiness faults; this is not a count of every repair or a count of individual defect items. A car can appear in more than one inspection."
        )
        st.link_button(
            label="Traficom's official data and statistical definitions",
            url="https://tieto.traficom.fi/en/statistics/statistics-inspections",
        )
        st.markdown(
            body="**The point is 100 × x/n.** We retain the source's rounded published percentage for comparison. **The bars are calculated by this app**, using a two-sided 95% Wilson score interval with z = 1.959963984540054, no continuity correction. We calculate a separate interval for each recorded count pair; grouping does not pool counts."
        )
        st.latex(body=r"\hat p = x/n,\quad d=1+z^2/n,\quad c=(\hat p+z^2/(2n))/d")
        st.latex(
            body=r"h=\frac{z\sqrt{\hat p(1-\hat p)/n+z^2/(4n^2)}}{d},\quad [L,U]=100[c-h,c+h]"
        )
        st.link_button(
            label="NIST reference: Wilson intervals for proportions",
            url="https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm",
        )
        st.markdown(
            body="**Interpretation:** the interval describes binomial repeat-sampling uncertainty under independent, comparable test outcomes. It is not a 95% probability that a particular car will fail, a prediction interval for next year's rate, or an engine-lifetime estimate. The observed percentage itself is known from the recorded tests. For near-complete administrative coverage, these model-based bars are not uncertainty about the observed census percentage. Repeat inspections, maintenance, model variants, selection and systematic differences remain outside them. Overlapping bars alone are not a formal test of differences."
        )
        st.markdown(
            body="**TÜV:** our imported Car-Recalls.eu summaries report significant-defect percentages and odometer summaries but no model/age numerator–denominator pairs. The report's total inspection count cannot be assigned to each model. **ADAC:** published rates are attended breakdown events per 1,000 registered vehicles; the imported model exposure and event counts are unavailable. The value 1,000 is a scaling unit, not the actual sample size. Both sources can have rate plots, but these imports cannot support Wilson bars. **MOT:** genuine imported initial failure/test counts can support Wilson intervals automatically."
        )
        show_table(
            frame=source_interval_availability(evidence=evidence),
            name="interval_availability",
            key=f"{key}_availability",
        )
        st.caption(
            body="Availability counts above are observation rows, not vehicle or test denominators. Open points mean a published rate without a calculable interval, not a zero-width interval or evidence of perfect reliability."
        )
        counted = counted_failure_rates(evidence=evidence).reset_index(drop=True)
        if not counted.empty:
            examples = {
                repr(
                    (
                        row.source_id,
                        int(row.report_year),
                        int(row.observation_year),
                        row.comparison_group,
                        row.model_id,
                        row.metric,
                    )
                ): row
                for row in counted.itertuples()
            }
            labels = {
                identifier: f"{row.make} {row.model} · age {row.age_min:g}–{row.age_max:g} · {row.source_name} · observations {int(row.observation_year)}"
                for identifier, row in examples.items()
            }
            preferred = [
                identifier
                for identifier, row in examples.items()
                if row.model_id == "volkswagen:t roc" and row.age_min == 4
            ]
            example_key = f"{key}_example"
            if st.session_state.get(example_key) not in examples:
                st.session_state[example_key] = (
                    preferred[0] if preferred else next(iter(examples))
                )
            chosen = st.selectbox(
                label="Worked example from the actual loaded counts",
                options=list(examples),
                index=None,
                format_func=lambda identifier: labels.get(identifier, identifier),
                key=example_key,
            )
            row = examples[chosen]
            st.write(
                f"x = {int(row.event_count):,} failed tests; n = {int(row.sample_size):,} tests. The plotted rate is {row.counted_rate_pct:.4f}% (published: {row.value:g}%). The calculated 95% Wilson interval is {row.wilson_lower_pct:.4f}–{row.wilson_upper_pct:.4f}%. The left whisker spans {row.counted_rate_pct - row.wilson_lower_pct:.4f} percentage points and the right spans {row.wilson_upper_pct - row.counted_rate_pct:.4f}; Wilson bars can be asymmetric."
            )
            st.link_button(
                label="Original source for this worked example", url=str(row.source_url)
            )


def show_source_rate_panels(*, evidence: pd.DataFrame, key: str) -> None:
    """Plot original source rates by age, retaining honest interval availability.

    Args:
        evidence: Current page-filtered canonical observations.
        key: Unique source, make, age-panel and chart widget prefix.
    """
    if evidence.empty:
        st.info(body="No published rate observations match these filters.")
        return
    options = sorted(evidence.source_id.unique())
    source = st.selectbox(
        label="Source for age-grouped rate plots",
        options=options,
        index=options.index("tuv") if "tuv" in options else 0,
        format_func=lambda value: SOURCE_NAMES.get(value, value),
        key=f"{key}_source",
    )
    rows = evidence.loc[evidence.source_id.eq(source)].copy()
    makes = st.multiselect(
        label="Makes in these source-rate plots",
        options=sorted(rows.make.unique()),
        key=f"{key}_makes",
    )
    query = st.text_input(label="Search these source-rate plots", key=f"{key}_query")
    if makes:
        rows = rows.loc[rows.make.isin(makes)]
    if query:
        rows = rows.loc[
            (rows.make + " " + rows.model).str.contains(
                pat=query, case=False, regex=False
            )
        ]
    limit = st.number_input(
        label="Models per published-rate age chart",
        min_value=1,
        max_value=max(24, len(rows)),
        value=24,
        key=f"{key}_limit",
    )
    order = st.selectbox(
        label="Published-rate chart ordering",
        options=["point_rate", "make", "sample_size", "original"],
        format_func=lambda value: {
            "point_rate": "Observed rate",
            "make": "Make",
            "sample_size": "Known test count",
            "original": "Original row order",
        }[value],
        key=f"{key}_order",
    )
    ascending = st.checkbox(
        label="Sort published-rate charts ascending", value=True, key=f"{key}_ascending"
    )
    colour = st.selectbox(
        label="Published-rate chart colours",
        options=["make", "uniform"],
        format_func=lambda value: "Make" if value == "make" else "One colour",
        key=f"{key}_colour",
    )
    highlights = make_highlights(
        label="Highlight published-rate makes",
        makes=rows.make.tolist(),
        key=f"{key}_highlight",
        enabled=colour == "make",
    )
    if order == "sample_size" and rows.sample_size.isna().all():
        st.caption(
            body="This selection has no published model-level test counts; test-count sorting retains original row order."
        )
    panels = wilson_age_groups(counted=rows, grouping="exact")
    signature = tuple(panels)
    if st.session_state.get(f"{key}_signature") != signature:
        st.session_state[f"{key}_panels"] = list(panels)
        st.session_state[f"{key}_signature"] = signature
    selected = st.multiselect(
        label="Published-rate age panels to display",
        options=list(panels),
        key=f"{key}_panels",
    )
    st.caption(
        body="Open points: published rate, interval unavailable. Solid points with whiskers: actual failed/test counts and calculated 95% Wilson bounds. Missing bars indicate unavailable intervals. Each panel keeps its source, report, outcome and age/cohort separate; make colours do not indicate good or bad reliability."
    )
    if not panels:
        st.info(body="No source-rate observations match this make/model search.")
    for index, label in enumerate(selected):
        st.markdown(body=f"**{label}**")
        show_chart(
            figure=source_rate_chart(
                evidence=panels[label],
                top_n=int(limit),
                sort_by=order,
                ascending=ascending,
                colour_by=colour,
                highlight_makes=highlights,
            ),
            key=f"{key}_{index}",
            explanation="The horizontal coordinate is the source's rate in its own units, not a combined score. TÜV points describe significant inspection-defect percentages; ADAC points describe breakdown events per 1,000 registered vehicles. Counted inspection points use actual failed tests / tests. Error bars appear only when the imported numerator and denominator support a Wilson interval. No sample sizes are inferred from rates, overall report totals, mileage, fleet thresholds or scaling units. Lower rates are favourable within the specified outcome and cohort.",
        )
    with st.expander(label="Rates, interval status and source links for these panels"):
        show_table(
            frame=source_rate_points(evidence=rows),
            name="published_rate_points",
            key=f"{key}_table",
        )


def show_wilson_panels(
    *,
    counted: pd.DataFrame,
    top_n: int,
    sort_by: str,
    ascending: bool,
    colour_by: str,
    key: str,
    highlight_makes: list[str] | None = None,
) -> None:
    """Display independently sorted Wilson charts for selected age groups.

    Args:
        counted: Canonical rows with actual counts and precomputed intervals.
        top_n: Maximum model-cohort rows per chart, never a pooled sample count.
        sort_by: Existing count-chart ordering preference.
        ascending: Ascending or descending ordering within every panel.
        colour_by: Uniform or make colours.
        key: Unique widget/chart key prefix for the containing app view.
        highlight_makes: Emphasise selected makes while retaining competitors.
    """
    grouping = st.selectbox(
        label="Group Wilson charts by age",
        options=["exact", "two_year", "mixed"],
        format_func=lambda value: {
            "exact": "Exact published age / cohort",
            "two_year": "Two-year age groups",
            "mixed": "Mixed ages within a source/report",
        }[value],
        key=f"{key}_grouping",
        help="Exact matching is the default. Two-year groups deliberately allow nearby ages; rates and intervals are not pooled. Every chart keeps its source and report separate.",
    )
    panels = wilson_age_groups(counted=counted, grouping=grouping)
    if not panels:
        st.info(body="No actual counted cohorts match these Wilson filters.")
        return
    signature = (grouping, tuple(panels))
    if st.session_state.get(f"{key}_panel_signature") != signature:
        st.session_state[f"{key}_panels"] = list(panels)
        st.session_state[f"{key}_panel_signature"] = signature
    selected = st.multiselect(
        label="Age panels to display",
        options=list(panels),
        key=f"{key}_panels",
    )
    if grouping != "exact":
        st.caption(
            body="These panels contain different recorded ages. Two-year groups are approximate comparisons; mixed-age panels do not establish an age-adjusted ranking."
        )
    st.caption(
        body=f"Showing {len(selected)} age panels. The cohort limit ({top_n}) and sorting apply separately within each chart; no test counts or failure rates are combined."
    )
    for index, label in enumerate(selected):
        rows = panels[label]
        st.markdown(body=f"**{label}**")
        st.caption(
            body=f"Showing up to {min(top_n, len(rows)):,} of {len(rows):,} model-cohort observations in this panel."
        )
        figure = count_uncertainty_chart(
            counted=rows,
            top_n=top_n,
            sort_by=sort_by,
            ascending=ascending,
            colour_by=colour_by,
            highlight_makes=highlight_makes,
        )
        first = rows.iloc[0]
        low, high = int(rows.age_min.min()), int(rows.age_max.max())
        age_label = str(low) if low == high else f"{low}–{high}"
        figure.update_layout(
            title=f"{SOURCE_NAMES.get(first.source_id, first.source_id)} · age {age_label} · report {int(first.report_year)} · 95% Wilson intervals"
        )
        show_chart(
            figure=figure,
            key=key if index == 0 else f"{key}_{index}",
            explanation="Each point uses actual failed inspections divided by tests and its own 95% Wilson interval. Source/report contexts stay separate. Wide and overlapping intervals qualify apparent differences. A model can appear in different age panels; this is not a brand ranking or a tracked ageing curve. Repeated inspections, maintenance and selection effects are not covered by these sampling intervals.",
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
        max_selections=30,
        key="age_models",
    )
    relative = st.checkbox(
        label="Show relative cohort scores", value=False, key="age_relative"
    )
    age_colour = st.selectbox(
        label="Age profile colours",
        options=["model", "make"],
        format_func=lambda value: value.capitalize(),
        key="age_colour",
    )
    if selected:
        show_chart(
            figure=age_chart(
                scored=rows, model_ids=selected, relative=relative, colour_by=age_colour
            ),
            key="age_profile",
            explanation="Age profiles use different registration cohorts and may cross model generations. A rising failure rate with age does not estimate how your own car will age. Relative scores compare each point with its source-cohort peers.",
        )
    if rows.mileage_basis.eq("mean_odometer").any() and rows.mileage_km.notna().any():
        mileage_colour = st.selectbox(
            label="Mileage scatter colours",
            options=["age", "make"],
            format_func=lambda value: "Age" if value == "age" else "Make",
            key="age_mileage_colour",
        )
        show_chart(
            figure=mileage_chart(evidence=rows, colour_by=mileage_colour),
            key="mileage_scatter",
            explanation="Each dot is a published model cohort. The x-axis is mean odometer mileage; the y-axis is an inspection rate. Colour represents observed age. This plot shows associations without adjusting for maintenance, generation or usage.",
        )
    else:
        st.info(
            body="Mean odometer data are unavailable for this source. Open Mileage explorer in the sidebar, or select Traficom/TÜV above. ADAC reports a different annual-mileage measure."
        )
    st.subheader(body="Better and poorer evidence across ages")
    map_measure = st.selectbox(
        label="Heatmap cell values",
        options=["relative", "rate"],
        format_func=lambda value: (
            "Relative score · higher is better"
            if value == "relative"
            else "Published fault/breakdown rate · lower is better"
        ),
        key="age_heatmap_measure",
    )
    st.caption(
        body="Cell numbers and colours use the same measure. Teal = better; red = poorer; grey dash = unavailable. Relative scores use a fixed 0–100 scale. Published rates use this source/report's maximum rate; rates at different ages are not directly comparable."
    )
    map_order = st.selectbox(
        label="Age heatmap ordering",
        options=["alphabetical", "both", "weakest", "strongest"],
        format_func=lambda value: {
            "alphabetical": "Model names alphabetically",
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
            scored=map_rows,
            max_models=map_limit,
            order=map_order,
            measure=map_measure,
            rate_ceiling=float(rows.value.max()),
        ),
        key="age_heatmap",
        explanation="Each cell compares a model with its original source-cohort peers. Teal means lower published rates relative to peers; red means higher rates. Grey cells have no score and are never zero. Hover for the raw rate in this source's units. Row ordering averages available age scores, so coverage can differ. These are different cars/cohorts, not longitudinal survival or controlled engine comparisons.",
    )
    with st.expander(label="? Why is the old faults-per-kilometre ratio absent?"):
        st.write(
            "Dividing an inspection failure percentage by average mileage is not an exposure-adjusted failure rate. A model can look better simply because its owners drive farther. The original scripts retain the legacy field for compatibility; the new app uses raw rates, same-age comparisons and explicit mileage context."
        )


def mileage_page(*, scored: pd.DataFrame, ranking: pd.DataFrame) -> None:
    """Expose mileage plots with filters and explicit combined-score sizing.

    Args:
        scored: Full sidebar-selected source-cohort evidence.
        ranking: Current combined ranking, used only for optional point sizing.
    """
    st.title(body="Mileage explorer")
    st.caption(
        body="The mileage-versus-inspection chart from the HTML report. Sidebar filters apply first; controls below can narrow the plotted observations further."
    )
    eligible = scored.loc[
        scored.mileage_basis.eq("mean_odometer") & scored.mileage_km.notna()
    ]
    if eligible.empty:
        st.info(
            body="No mean-odometer observations match. Include Traficom or TÜV and broaden the sidebar filters. ADAC's mileage measure is different."
        )
        return
    sources = sorted(eligible.source_id.unique())
    source = st.selectbox(
        label="Mileage evidence source",
        options=sources,
        index=sources.index("traficom") if "traficom" in sources else 0,
        format_func=lambda value: SOURCE_NAMES.get(value, value),
        key="mileage_source",
    )
    rows = eligible.loc[eligible.source_id.eq(source)].copy()
    year = st.selectbox(
        label="Mileage report year",
        options=sorted(rows.report_year.unique(), reverse=True),
        key="mileage_year",
    )
    rows = rows.loc[rows.report_year.eq(year)]
    makes = st.multiselect(
        label="Mileage makes", options=sorted(rows.make.unique()), key="mileage_makes"
    )
    query = st.text_input(label="Search mileage models", key="mileage_query")
    age_range = st.slider(
        label="Mileage chart age range",
        min_value=0,
        max_value=40,
        value=(2, 20),
        key="mileage_age_range",
    )
    rows = rows.loc[rows.age_max.ge(age_range[0]) & rows.age_min.le(age_range[1])]
    if makes:
        rows = rows.loc[rows.make.isin(makes)]
    if query:
        rows = rows.loc[
            (rows.make + " " + rows.model).str.contains(
                pat=query, case=False, regex=False
            )
        ]
    colour = st.selectbox(
        label="Mileage point colours",
        options=["age", "make"],
        format_func=lambda value: "Observed age" if value == "age" else "Make",
        key="mileage_colour",
    )
    size = st.checkbox(
        label="Size mileage points by optional within-block preference score",
        key="mileage_size_score",
    )
    st.caption(
        body=f"{len(rows):,} plotted cohort observations. Optional point sizes follow the selected comparison method. Small open markers mean no comparable preference score, not poor reliability. Age bands overlapping the range are retained; points are cohort averages."
    )
    show_chart(
        figure=mileage_chart(
            evidence=rows, colour_by=colour, model_scores=ranking, size_by_score=size
        ),
        key="mileage_explorer",
        explanation="Mean odometer against inspection failure/defect rate within one source and report. Colour may show age or make; score-sized marker diameters run from 7 to 25 pixels. The optional preference score is repeated for its model cohorts only within the selected coverage block. Missing scores are shown as small open markers, not poor reliability. Association is not controlled for maintenance, generation or usage.",
    )
    with st.expander(label="Plotted observations and original source links"):
        show_table(frame=rows, name="mileage_observations", key="mileage_export")


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
            "Largest source disagreement",
            "Alphabetical source profiles (up to 80)",
        ],
        key="heatmap_view",
    )
    query = st.text_input(label="Search heatmap models", key="heatmap_query")
    candidates = identities.loc[
        identities.model_id.isin(wide.index), ["model_id", "make", "model"]
    ].copy()
    candidates["score_spread"] = candidates.model_id.map(
        arg=wide.max(axis=1) - wide.min(axis=1)
    )
    if query:
        candidates = candidates.loc[
            (candidates.make + " " + candidates.model).str.contains(
                pat=query, case=False, regex=False
            )
        ]
    if heatmap_view.startswith("Largest"):
        selection = candidates.sort_values(by="score_spread", ascending=False).head(
            n=24
        )
    else:
        selection = candidates.sort_values(by=["make", "model"]).head(n=80)
    top = [key for key in selection.model_id if key in wide.index]
    if not top:
        top = [
            key
            for key in wide.index
            if not query or query.casefold() in names.get(key, key).casefold()
        ][:24]
    st.caption(
        body=f"Showing {len(top)} descriptive source profiles, independent of composite eligibility. Means can cover different ages and outcomes. There is no combined best-to-worst ordering on this page; grey cells are unavailable evidence."
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


def show_weighting_sensitivity(
    *, scored: pd.DataFrame, ranking: pd.DataFrame, min_sources: int, min_cohorts: int
) -> None:
    """Explore preset ranks within one explicitly selected coverage group.

    Args:
        scored: Current evidence with original peer percentiles retained.
        ranking: Baseline eligible model profiles.
        min_sources: Minimum number of observed sources per scenario.
        min_cohorts: Minimum number of source-cohort observations per model.
    """
    if scored.source_id.nunique() < 2:
        st.info(
            body="Weighting sensitivity requires at least two selected sources. With one source, changing the balance between sources or outcomes cannot test a different evidence priority."
        )
        return
    scenarios = weighting_scenarios(
        scored=scored, min_sources=min_sources, min_cohorts=min_cohorts
    )
    scenarios = scenarios.loc[scenarios.model_id.isin(ranking.model_id)]
    if scenarios.empty:
        st.info(
            body="No models meet the current coverage thresholds in all three preference scenarios. Broaden the age/source filters or adjust coverage thresholds."
        )
        return
    baseline = scenarios.loc[scenarios.scenario.eq("Balanced")]
    selected = comparison_block_selector(ranking=baseline, key="deep_scenario_group")
    if selected.empty:
        return
    scenarios = scenarios.loc[scenarios.model_id.isin(selected.model_id)].copy()
    bounds = scenarios.groupby(by=["model_id", "make", "model"], as_index=False).agg(
        best_rank=("scenario_rank", "min"),
        worst_rank=("scenario_rank", "max"),
        lowest_score=("score", "min"),
        highest_score=("score", "max"),
    )
    bounds["rank_range"] = bounds.worst_rank - bounds.best_rank
    competitors = int(scenarios.comparison_models.iloc[0])
    bounds["comparison_models"] = competitors
    order = st.selectbox(
        label="Weighting sensitivity view",
        options=["Strongest worst-case rank", "Largest rank movement"],
        key="deep_scenario_order",
    )
    bounds = bounds.sort_values(
        by=["worst_rank", "model"]
        if order.startswith("Strongest")
        else ["rank_range", "model"],
        ascending=[order.startswith("Strongest"), True],
        kind="stable",
    )
    signature = tuple(sorted(bounds.model_id))
    if st.session_state.get("deep_scenario_signature") != signature:
        st.session_state["deep_scenario_makes"] = []
        st.session_state["deep_scenario_models"] = []
        st.session_state["deep_scenario_count"] = min(20, len(bounds))
        st.session_state["deep_scenario_signature"] = signature
    scenario_count = st.number_input(
        label="Models in weighting chart",
        min_value=1,
        max_value=len(bounds),
        key="deep_scenario_count",
    )
    scenario_makes = st.multiselect(
        label="Weighting chart makes",
        options=sorted(bounds.make.unique()),
        key="deep_scenario_makes",
    )
    names = dict(zip(bounds.model_id, bounds.make + " " + bounds.model, strict=True))
    scenario_models = st.multiselect(
        label="Specific weighting chart models (optional)",
        options=list(names),
        format_func=lambda value: names.get(value, value),
        key="deep_scenario_models",
    )
    view = st.selectbox(
        label="Weighting chart layout",
        options=["matrix", "lines"],
        format_func=lambda value: (
            "Rank matrix · shows ties separately"
            if value == "matrix"
            else "Rank trajectories"
        ),
        key="deep_scenario_layout",
    )
    colour = st.selectbox(
        label="Weighting chart colours",
        options=["model", "make"],
        format_func=lambda value: value.capitalize(),
        key="deep_scenario_colour",
        disabled=view == "matrix",
    )
    if view == "matrix":
        st.caption(
            body="Cell colours show rank within this comparison group. Make colours and point sizes apply to the trajectory layout."
        )
    size = st.checkbox(
        label="Size weighting points by scenario score",
        key="deep_scenario_size",
        disabled=view == "matrix",
    )
    if bounds.rank_range.eq(0).all():
        st.info(
            body="Ranks did not change under these three presets. This describes only the tested preferences; it is not a confidence bound or proof of general robustness."
        )
    chosen = bounds.loc[bounds.make.isin(scenario_makes)] if scenario_makes else bounds
    if scenario_models:
        chosen = chosen.loc[chosen.model_id.isin(scenario_models)]
    ids = chosen.head(n=int(scenario_count)).model_id.tolist()
    if ids:
        show_chart(
            figure=scenario_rank_chart(
                scenarios=scenarios,
                model_ids=ids,
                colour_by=colour,
                size_by_score=size,
                view=view,
            ),
            key="deep_scenario_chart",
            explanation=f"All ranks belong to one identical-coverage group of {competitors} competitors. Display filters do not recompute ranks. Rank 1 is best; ties retain the same rank. The matrix uses rank colours, not make colours. Trajectories offer model/make colours and optional point sizes. These three presets are preference examples, not confidence intervals or calibrated source trust. Custom sidebar weights are examined in source omission below.",
        )
    else:
        st.info(body="No models match these weighting-chart make/model filters.")
    show_table(
        frame=bounds, name="weighting_scenario_rank_ranges", key="deep_scenario_bounds"
    )
    with st.expander(label="Selected comparison group's scenarios and exact coverage"):
        show_table(frame=scenarios, name="weighting_scenarios", key="deep_scenarios")
        reference_id = selected.model_id.iloc[0]
        show_table(
            frame=scored.loc[
                scored.model_id.eq(reference_id),
                [
                    "source_name",
                    "report_year",
                    "observation_year",
                    "age_min",
                    "age_max",
                    "cohort_year_start",
                    "cohort_year_end",
                ],
            ],
            name="weighting_group_contexts",
            key="deep_scenario_contexts",
        )
    LOGGER.debug(
        "Rendered weighting sensitivity for one group of %d competitors", competitors
    )


def deeper_analysis_page(
    *,
    scored: pd.DataFrame,
    ranking: pd.DataFrame,
    weights: dict[str, float],
    min_sources: int,
    min_cohorts: int,
    peer_evidence: pd.DataFrame | None = None,
) -> None:
    """Explore descriptive distributions, actual counts and source sensitivity.

    Args:
        scored: Evidence selected by the sidebar's explicit filters.
        ranking: Baseline ranking for the selected weights and thresholds.
        weights: Current non-negative source weights.
        min_sources: Baseline minimum source count.
        min_cohorts: Required scored source-cohort observations.
        peer_evidence: Optional complete filtered reference population, retained
            when examining a propulsion subset.
    """
    st.title(body="Deeper data interrogation")
    st.caption(
        body="All results use your current evidence filters. These analyses describe loaded model-series/cohort observations, not exact engines, individual cars or lifespans."
    )
    st.info(
        body="Combined-score and weighting diagnostics below are secondary preference analyses. Ranks are meaningful only within an identical-coverage comparison block. Source profiles and exact-cohort raw rates remain the primary evidence."
    )
    if scored.empty:
        st.info(body="No evidence matches the current filters.")
        return
    st.subheader(body="How does a model compare with peers at the same age?")
    reference = scored if peer_evidence is None else peer_evidence
    benchmarks = cohort_peer_benchmarks(scored=reference)
    benchmarks = benchmarks.loc[benchmarks.model_id.isin(scored.model_id)]
    if benchmarks.empty:
        st.info(
            body="Same-cohort benchmarking needs at least five ungrouped model series in a published cohort."
        )
    else:
        identities = benchmarks.drop_duplicates(subset="model_id")
        names = dict(
            zip(
                identities.model_id,
                identities.make + " " + identities.model,
                strict=True,
            )
        )
        chosen = st.selectbox(
            label="Model for same-cohort interrogation",
            options=sorted(names, key=names.get),
            format_func=lambda key: names[key],
            key="deep_peer_model",
        )
        profile = benchmarks.loc[benchmarks.model_id.eq(chosen)].copy()
        profile["age_band"] = (
            profile.age_min.astype(str) + "–" + profile.age_max.astype(str)
        )
        profile["panel"] = (
            profile.source_name
            + " · "
            + profile.unit
            + " · report "
            + profile.report_year.astype(str)
        )
        figure = px.bar(
            data_frame=profile,
            x="age_band",
            y="rate_gap",
            facet_row="panel",
            color="panel",
            hover_data=[
                "value",
                "peer_median_rate",
                "peer_models",
                "report_year",
                "sample_size",
            ],
            labels={
                "rate_gap": "Rate minus peer median",
                "age_band": "Published age band",
            },
        )
        figure.update_yaxes(matches=None)
        figure.update_layout(
            height=max(420, profile.panel.nunique() * 240), showlegend=False
        )
        for annotation in figure.layout.annotations:
            annotation.update(text=annotation.text.replace("panel=", ""), textangle=0)
        show_chart(
            figure=figure,
            key="deep_peer_gaps",
            explanation="Each bar compares this model with the median of OTHER ungrouped model series in the exact same source, report, outcome, age and registration cohort. Negative is a lower, favourable rate. Sources keep their own units. These descriptive gaps do not adjust for mileage, maintenance, engine or selection effects. Propulsion filtering retains the full filtered reference population.",
        )
        show_table(
            frame=profile.loc[
                :,
                [
                    "source_name",
                    "report_year",
                    "age_min",
                    "age_max",
                    "unit",
                    "value",
                    "peer_median_rate",
                    "rate_gap",
                    "peer_models",
                    "sample_size",
                ],
            ],
            name="same_cohort_peer_benchmarks",
            key="deep_peer_table",
        )
    st.subheader(body="Which models remain strong under different priorities?")
    show_weighting_sensitivity(
        scored=scored, ranking=ranking, min_sources=min_sources, min_cohorts=min_cohorts
    )
    st.subheader(body="Rate distributions and data coverage")
    distribution_sources = st.multiselect(
        label="Distribution sources (empty shows all)",
        options=sorted(scored.source_id.unique()),
        format_func=lambda value: SOURCE_NAMES.get(value, value),
        key="deep_distribution_sources",
    )
    separate_ages = st.checkbox(
        label="Separate distributions by published age",
        value=True,
        key="deep_distribution_ages",
    )
    distributions = (
        scored.loc[scored.source_id.isin(distribution_sources)]
        if distribution_sources
        else scored
    )
    if not separate_ages:
        st.warning(
            body="These boxes pool different ages. Their positions depend on the selected age mix; they do not rank manufacturers or individual-car reliability."
        )
    show_chart(
        figure=rate_distribution_chart(
            evidence=distributions, group_by_age=separate_ages
        ),
        key="deep_distributions",
        explanation="Sources, reports and observation years retain separate axes and units. Published ages are separated by default. Box centres are medians, boxes span quartiles, whiskers extend to the most extreme observations within 1.5 times the interquartile range, and dotted markers show means. Each published model-cohort row gets equal weight; row counts are not test counts or unique cars. Outliers are observations outside the whisker rule, not necessarily errors. These are descriptive distributions, not fleet-weighted failure rates.",
    )
    show_table(
        frame=source_rate_summary(evidence=distributions),
        name="source_rate_statistics",
        key="deep_statistics",
    )
    st.subheader(body="Actual failure counts and sampling uncertainty")
    show_interval_method_help(evidence=scored, key="deep_interval_help")
    counted = counted_failure_rates(evidence=scored)
    count_sources = st.multiselect(
        label="Wilson chart sources",
        options=sorted(counted.source_id.unique()),
        format_func=lambda value: SOURCE_NAMES.get(value, value),
        key="deep_count_sources",
    )
    count_makes = st.multiselect(
        label="Wilson chart makes",
        options=sorted(counted.make.unique()),
        key="deep_count_makes",
    )
    count_query = st.text_input(
        label="Search Wilson chart models", key="deep_count_query"
    )
    count_identities = counted.drop_duplicates(subset="model_id")
    count_names = dict(
        zip(
            count_identities.model_id,
            count_identities.make + " " + count_identities.model,
            strict=True,
        )
    )
    count_models = st.multiselect(
        label="Specific Wilson chart models (optional)",
        options=sorted(count_names, key=count_names.get),
        format_func=lambda key: count_names[key],
        key="deep_count_models",
    )
    count_ages = st.slider(
        label="Wilson chart age range",
        min_value=0,
        max_value=40,
        value=(2, 20),
        key="deep_count_age",
    )
    count_limit = st.number_input(
        label="Cohorts per Wilson chart",
        min_value=1,
        max_value=max(24, len(counted)),
        value=24,
        key="deep_count_limit",
    )
    count_sort = st.selectbox(
        label="Wilson chart ordering",
        options=["sample_size", "counted_rate_pct", "age_min", "make", "original"],
        format_func=lambda value: {
            "sample_size": "Test count",
            "counted_rate_pct": "Failure rate",
            "age_min": "Observed age",
            "make": "Make",
            "original": "Original row order",
        }[value],
        key="deep_count_sort",
    )
    count_ascending = st.checkbox(
        label="Sort Wilson chart ascending", key="deep_count_ascending"
    )
    count_colour = st.selectbox(
        label="Wilson chart colours",
        options=["uniform", "make"],
        format_func=lambda value: "One colour" if value == "uniform" else "Make",
        key="deep_count_colour",
    )
    count_highlights = make_highlights(
        label="Highlight Wilson-chart makes",
        makes=counted.make.tolist(),
        key="deep_count_highlight",
        enabled=count_colour == "make",
    )
    plotted_counts = counted.loc[
        counted.age_max.ge(count_ages[0]) & counted.age_min.le(count_ages[1])
    ]
    if count_sources:
        plotted_counts = plotted_counts.loc[
            plotted_counts.source_id.isin(count_sources)
        ]
    if count_makes:
        plotted_counts = plotted_counts.loc[plotted_counts.make.isin(count_makes)]
    if count_query:
        plotted_counts = plotted_counts.loc[
            (plotted_counts.make + " " + plotted_counts.model).str.contains(
                pat=count_query, case=False, regex=False
            )
        ]
    if count_models:
        plotted_counts = plotted_counts.loc[plotted_counts.model_id.isin(count_models)]
    st.caption(
        body=f"{len(plotted_counts):,} counted cohorts match these chart filters. Points remain source/report/age-specific; a model can appear more than once. Sorting by rate does not make unmatched ages comparable."
    )
    show_wilson_panels(
        counted=plotted_counts,
        top_n=int(count_limit),
        sort_by=count_sort,
        ascending=count_ascending,
        colour_by=count_colour,
        key="deep_counts",
        highlight_makes=count_highlights,
    )
    with st.expander(label="Inspect all counted rates and intervals"):
        show_table(
            frame=counted, name="counted_failure_intervals", key="deep_count_table"
        )
    st.subheader(body="Published rates by age — TÜV, ADAC and other sources")
    show_source_rate_panels(evidence=scored, key="deep_published_rates")
    st.subheader(body="How dependent is the ranking on one source?")
    comparable = ranking.loc[
        ranking.sources.ge(2) & ranking.score.notna() & ranking.score_spread.notna()
    ]
    excluded = int(ranking.sources.lt(2).sum())
    unscored = len(ranking) - len(comparable) - excluded
    if comparable.empty:
        st.info(
            body="Between-source disagreement requires at least two observed sources per model. With one source, disagreement is unavailable, not zero."
        )
    else:
        st.caption(
            body=f"Showing {len(comparable):,} models with at least two observed sources; excluded {excluded:,} single-source models and {unscored:,} unscored profiles. Source means can still cover different ages and generations."
        )
        spread_colour = st.selectbox(
            label="Score disagreement colours",
            options=["sources", "make"],
            format_func=lambda value: (
                "Source breadth" if value == "sources" else "Make"
            ),
            key="deep_spread_colour",
        )
        spread_size = st.checkbox(
            label="Size disagreement points by optional preference score",
            key="deep_spread_size",
        )
        show_chart(
            figure=score_spread_chart(
                ranking=comparable, colour_by=spread_colour, size_by_score=spread_size
            ),
            key="deep_spread",
            explanation="Only models with at least two observed sources are plotted. Source-score spread measures descriptive disagreement, not statistical confidence, precision or matched-age agreement. Single-source models are excluded, not plotted at zero. Hover for cohort count and coverage; make colours identify manufacturers. The optional composite remains a preference index, not a failure probability.",
        )
    if ranking.empty:
        st.info(
            body="No ranked models meet the current weights or coverage thresholds. Broaden the filters to explore source sensitivity."
        )
        return
    sensitivity = source_omission_sensitivity(
        scored=scored,
        weights=weights,
        min_sources=min_sources,
        min_cohorts=min_cohorts,
        inspection_weight=float(st.session_state.get("inspection_weight", 1.0)),
        breakdown_weight=float(st.session_state.get("breakdown_weight", 1.0)),
    )
    sensitivity = sensitivity.loc[sensitivity.model_id.isin(ranking.model_id)]
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


def electrified_page(*, scored: pd.DataFrame, weights: dict[str, float]) -> None:
    """Explore reviewed electric/hybrid series without guessing variant rates.

    Args:
        scored: Full filtered evidence retaining original peer scores.
        weights: Current sidebar source preference weights.
    """
    st.title(body="Electric and hybrid vehicles")
    st.caption(
        body="Separate battery-electric, plug-in hybrid and conventional hybrid views. This is a conservative reviewed catalogue, not complete market coverage. Unreviewed models remain unclassified."
    )
    category = st.selectbox(
        label="Propulsion to explore",
        options=["Battery electric", "Plug-in hybrid", "Hybrid"],
        key="electric_category",
    )
    mixed = st.checkbox(
        label="Include mixed model families for context",
        value=False,
        key="electric_mixed",
        help="Mixed-family reliability observations are not specific to the electric or hybrid variant. They remain explicitly labelled in the catalogue.",
    )
    catalogue = propulsion_catalogue(evidence=scored)
    with st.expander(
        label="Reviewed eligibility, exclusions and manufacturer references"
    ):
        show_table(
            frame=catalogue, name="propulsion_eligibility", key="electric_catalogue"
        )
    selected = select_propulsion(
        evidence=scored, category=category, include_mixed=mixed
    )
    if mixed:
        st.warning(
            body="Mixed families are included only for context. Their rates cannot be attributed to a hybrid or plug-in variant."
        )
    if selected.empty:
        st.info(
            body="No reviewed series in this category match the current age/source filters. Broaden those filters; unclassified models are not silently reassigned."
        )
        return
    st.info(
        body="Published inspection and breakdown results do not measure traction-battery capacity, degradation, charging reliability or battery replacement costs. Those outcomes need additional data."
    )
    st.subheader(body="Separate electrified source evidence")
    show_table(
        frame=evidence_profiles(scored=selected),
        name="electrified_source_profiles",
        key="electric_profiles",
    )
    minimum = st.number_input(
        label="Minimum sources for this propulsion view",
        min_value=1,
        max_value=max(1, selected.source_id.nunique()),
        value=1,
        key="electric_min_sources",
    )
    cohorts = st.number_input(
        label="Minimum observations for this propulsion view",
        min_value=1,
        max_value=100,
        value=1,
        key="electric_min_cohorts",
    )
    ranking_function = (
        exploratory_ranking
        if str(st.session_state.get("comparison_method", "Exploratory")).startswith(
            "Exploratory"
        )
        else reliability_ranking
    )
    ranking = ranking_function(
        scored=selected,
        weights=weights,
        min_sources=int(minimum),
        min_cohorts=int(cohorts),
        inspection_weight=float(st.session_state.get("inspection_weight", 1.0)),
        breakdown_weight=float(st.session_state.get("breakdown_weight", 1.0)),
    )
    ranking = comparison_block_selector(ranking=ranking, key="electric_block")
    st.caption(
        body="Charts follow the chosen exploratory or strict method. Missing categories and uncertain propulsion variants are not silently reassigned."
    )
    st.caption(
        body=f"{selected.model_id.nunique()} eligible model series · {len(selected):,} observations · page-specific coverage thresholds above. Original scores still compare with the full source-cohort peer population."
    )
    electric_limit = 20
    electric_colour = "sources"
    if not ranking.empty:
        electric_limit = int(
            st.number_input(
                label="Models in electrified ranking chart",
                min_value=1,
                max_value=len(ranking),
                value=min(20, len(ranking)),
                key="electric_top_n",
            )
        )
        electric_colour = st.selectbox(
            label="Electrified ranking colours",
            options=["sources", "make"],
            format_func=lambda value: (
                "Evidence breadth" if value == "sources" else "Make"
            ),
            key="electric_rank_colour",
        )
    show_chart(
        figure=ranking_chart(
            ranking=ranking, top_n=electric_limit, colour_by=electric_colour
        ),
        key="electric_ranking",
        explanation="This ordering is restricted to the selected reviewed propulsion series. Scores retain their original full-peer comparisons; this is not a controlled electric-versus-petrol comparison.",
    )
    show_table(frame=ranking, name="electrified_ranking", key="electric_rank_table")
    available = selected.drop_duplicates(subset="model_id")
    names = dict(
        zip(available.model_id, available.make + " " + available.model, strict=True)
    )
    model_ids = st.multiselect(
        label="Compare electrified model age profiles",
        options=sorted(names, key=names.get),
        default=list(names)[:3],
        format_func=lambda key: names[key],
        key="electric_age_models",
    )
    electric_age_colour = st.selectbox(
        label="Electrified age profile colours",
        options=["model", "make"],
        format_func=lambda value: value.capitalize(),
        key="electric_age_colour",
    )
    show_chart(
        figure=age_chart(
            scored=selected, model_ids=model_ids, colour_by=electric_age_colour
        ),
        key="electric_age_chart",
        explanation="Rates are separated by source and units. Colours identify the same model across panels. Age cohorts contain different cars; lines do not measure battery degradation or track individual vehicles.",
    )
    with st.expander(label="Exact selected observations and source links"):
        show_table(frame=selected, name="electrified_observations", key="electric_raw")
    deeper_analysis_page(
        scored=selected,
        ranking=ranking,
        weights=weights,
        min_sources=int(minimum),
        min_cohorts=int(cohorts),
        peer_evidence=scored,
    )


def comparison_block_selector(*, ranking: pd.DataFrame, key: str) -> pd.DataFrame:
    """Select one coverage block and prevent cross-block rankings on charts.

    Args:
        ranking: Coverage-matched preference profiles, including singleton rows.
        key: Unique Streamlit comparison-block widget identifier.

    Returns:
        At least two comparable model profiles in one block, or an empty table.
    """
    if not ranking.empty and ranking.comparison_group_id.eq("exploratory").all():
        st.warning(
            body="Exploratory ordering: evidence and ages differ between cars. Missing source scores assume percentile 50. Score bounds show the effect of assuming 0 or 100 instead; they are not confidence intervals. Check source results before deciding."
        )
        show_table(
            frame=ranking,
            name="Exploratory ranking and evidence gaps",
            key=f"{key}_coverage",
        )
        return ranking
    if ranking.empty:
        st.info(
            body="No complete preference profile meets these settings. Models with missing evidence remain available in separate source profiles."
        )
        return ranking
    groups = (
        ranking.loc[ranking.comparison_models.ge(2)]
        .groupby(by="comparison_group_id")
        .agg(
            models=("model_id", "size"),
            competitors=("comparison_models", "first"),
            cohorts=("cohorts", "first"),
            sources=("sources", "first"),
        )
        .sort_values(by=["models", "cohorts"], ascending=False)
    )
    if groups.empty:
        st.info(
            body="No models share identical source and cohort coverage under these filters. No overall ranks are assigned. Use source profiles, an exact published cohort or narrower evidence filters."
        )
        return ranking.iloc[:0]
    labels = {
        group: f"{row.models} shown / {int(row.competitors)} comparable models · {row.sources} sources · {row.cohorts} source-cohorts · block {group}"
        for group, row in groups.iterrows()
    }
    if st.session_state.get(key) not in labels:
        st.session_state[key] = next(iter(labels))
    chosen = st.selectbox(
        label="Identical-coverage comparison block",
        options=list(labels),
        format_func=lambda value: labels.get(value, value),
        index=None,
        key=key,
    )
    selected = (
        ranking.loc[ranking.comparison_group_id.eq(chosen)]
        .sort_values(by=["rank", "model_id"])
        .copy()
    )
    st.caption(
        body=f"Showing {len(selected)} models from a group of {int(selected.comparison_models.iloc[0])} competitors. Ranks retain that comparison population; display filters do not recompute them. Scores and ranks from different groups must not be compared. Singleton evidence profiles have no rank."
    )
    return selected


def repair_cost_page() -> None:
    """Show dated generic prices and user-edited quote scenarios separately."""
    st.title(body="Repair costs and fault severity")
    st.write(
        "Inspection defects, roadside inconvenience and expensive mechanical repairs are different outcomes. Our rate tables do not contain a model-specific breakdown of repair costs or fault frequencies."
    )
    st.info(
        body="Corolla's ADAC 2025 fault list identifies 12 V starter-battery issues for registration years 2019–2022. This does not establish engine failure or traction-battery degradation. The source does not provide the counts needed to price those faults per Corolla."
    )
    st.link_button(
        label="Read the original ADAC fault list",
        url="https://assets.adac.de/image/upload/v1744788916/ADAC-eV/KOR/Text/PDF/Pannenstatistik_2025_final_nrmvls.pdf",
    )
    references = repair_references()
    st.subheader(body="Published UK price context")
    st.caption(
        body="These dated marketplace averages are general job context, not quotations for your model or a repair-cost ranking. The provider's last-12-months table is the stated basis; other sections on its page quote different averages. Confirm parts, labour, VAT, engine, postcode and repair scope with a garage."
    )
    show_table(
        frame=references, name="repair_price_references", key="repair_references"
    )
    st.link_button(
        label="Get vehicle-specific price estimates with ClickMechanic",
        url="https://www.clickmechanic.com/price-estimates",
    )
    st.subheader(body="Your explicit repair-budget scenario")
    st.caption(
        body="Enter actual quotes or your assumptions and choose job quantities. All quantities start at zero. This is a what-if budget, not predicted annual ownership cost. No job frequency is inferred from a percentile, inspection rate or ADAC total."
    )
    uploaded = st.file_uploader(
        label="Optional quote/budget TSV", type=["tsv"], key="repair_quotes"
    )
    estimates = references.copy()
    estimates["quantity"] = 0.0
    if uploaded is not None:
        try:
            estimates = pd.read_csv(
                filepath_or_buffer=io.BytesIO(uploaded.getvalue()), sep="\t"
            )
            repair_budget(estimates=estimates)
        except (ValueError, TypeError) as error:
            st.error(body=f"Quote TSV could not be validated: {error}")
            return
    edited = st.data_editor(
        data=estimates,
        num_rows="dynamic",
        hide_index=True,
        key="repair_editor",
        column_config={
            "unit_cost_gbp": st.column_config.NumberColumn(
                label="Cost per job (£)", min_value=0.0
            ),
            "quantity": st.column_config.NumberColumn(
                label="Jobs in your scenario", min_value=0.0
            ),
        },
    )
    try:
        budget = repair_budget(estimates=edited)
    except (ValueError, TypeError) as error:
        st.error(body=str(error))
        return
    budget["entry_origin"] = (
        "User-provided budget assumptions; not verified quotes"
        if uploaded is not None
        else "Editable scenario with reference prices; not a vehicle quote"
    )
    if budget.scenario_cost_gbp.isna().any():
        st.warning(
            body="A selected job has no known price. A complete total is unavailable; unknown costs are not counted as free repairs."
        )
    else:
        st.metric(
            label="Explicit scenario total",
            value=f"£{budget.scenario_cost_gbp.sum():,.2f}",
        )
    show_table(frame=budget, name="repair_budget_scenario", key="repair_budget")


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


def source_information_page() -> None:
    """Explain denominators, defect definitions and source provenance."""
    st.title(body="Data sources and fault definitions")
    st.write(
        "A fault means something different in each source. Check the measurement, denominator, period and limits before comparing numbers."
    )
    source = st.selectbox(
        label="Source definition to read",
        options=list(SOURCE_GUIDE),
        key="source_definition",
    )
    guide = SOURCE_GUIDE[source]
    if st.checkbox(
        label="Explain interval bars and denominator availability",
        key="source_interval_explanation",
    ):
        st.info(
            body="Open Deeper analysis or Compare cars → Rates and intervals, then expand ‘Where do the points and interval bars come from?’ for the exact source fields, calculation, assumptions and a worked example from loaded counts."
        )
    st.subheader(body="What the number measures")
    st.write(guide["meaning"])
    with st.expander(label="? What counts as a fault?", expanded=True):
        st.write(guide["faults"])
    with st.expander(label="? What conclusions are unsupported?", expanded=True):
        st.write(guide["limits"])
    st.link_button(label="Where the imported data come from", url=guide["data_url"])
    st.link_button(
        label="Read the original source definitions", url=guide["definition_url"]
    )
    with st.expander(
        label="? Why isn't TÜV a faults-per-mile measurement?", expanded=True
    ):
        st.write(
            "The defect percentage counts inspected vehicles with significant defects. Average odometer mileage is a separate summary. Dividing them does not give fault events per mile: that requires event counts and corresponding total distance exposure. High-mileage models can look artificially favourable under this ratio. Historical scripts retain the legacy field, but it is not used in the app's comparisons."
        )
    with st.expander(label="? Report year, inspection period and approximate cohorts"):
        st.write(
            "TÜV Report 2025 covers July 2023 to June 2024 inspections; our observation-year label 2024 denotes the window's ending year. TÜV registration bands are approximate. ADAC Report 2025 concerns calendar-year 2024 breakdowns. Traficom observations are calendar-year inspections, published later. Selecting the same age does not make these periods identical."
        )
        st.link_button(
            label="TÜV's original 2025 methodology",
            url="https://www.tuev-verband.de/pressemitteilungen/tuev-report-2025-elektroautos",
        )
    with st.expander(label="? What supports repair-cost or lifespan estimates?"):
        st.write(
            "Repair-cost predictions need variant-specific fault counts and invoice prices. Lifetimes need linked vehicle histories and censoring assumptions. These fields are absent from our summaries. Frequency, severity, repair cost and durability are separate quantities."
        )


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
        "Relative score": "A source-cohort percentile: 100 for the lowest rate, 0 for the highest, with average ranks for ties. It is not a failure percentage. Optional preferences balance inspection and breakdown outcomes first; exploratory ranks show coverage gaps and strict ranks apply within one matching block.",
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
    with st.expander(label="? How should I interpret this page?"):
        st.markdown(body=PAGE_HELP[page])
        st.caption(
            body="For measurement definitions and provenance, open Data sources and fault definitions. Each chart also has interpretation help."
        )
    if page in {
        "Recalls and complaints",
        "Sources and methods",
        "Data management",
        "Repair costs",
        "Data sources and fault definitions",
    }:
        st.sidebar.caption(
            body="Reliability filters apply to the analysis pages. This page uses complete saved sources or its own safety filters."
        )
    comparison_mode = st.sidebar.selectbox(
        label="Combined comparison method",
        options=[
            "Exploratory · show incomplete evidence",
            "Strict · identical coverage",
        ],
        key="comparison_method",
        help="Exploratory rankings show missing evidence and age coverage. Missing source percentiles assume 50; bounds assume 0–100, not statistical confidence. Strict comparisons require matching evidence.",
    )
    ranking_function = (
        exploratory_ranking
        if comparison_mode.startswith("Exploratory")
        else reliability_ranking
    )
    try:
        ranking = ranking_function(
            scored=scored,
            weights=weights,
            min_sources=min_sources,
            min_cohorts=min_cohorts,
            inspection_weight=float(st.session_state.get("inspection_weight", 1.0)),
            breakdown_weight=float(st.session_state.get("breakdown_weight", 1.0)),
        )
    except ValueError as error:
        st.warning(body=str(error))
        ranking = reliability_ranking(scored=scored.iloc[:0])
    if page not in {
        "Recalls and complaints",
        "Sources and methods",
        "Data management",
        "Repair costs",
        "Data sources and fault definitions",
    }:
        with st.expander(label="Optional combined comparison · evidence coverage"):
            ranking = comparison_block_selector(
                ranking=ranking, key="main_comparison_block"
            )
    if page == "Overview":
        overview_page(
            evidence=evidence, scored=scored, ranking=ranking, manifest=manifest
        )
    elif page == "Find reliable cars":
        find_page(scored=scored, ranking=ranking)
    elif page == "Compare cars":
        historical = select_evidence(
            evidence=evidence,
            sources=st.session_state.get(
                "filter_sources", list(evidence.source_id.unique())
            ),
            latest_only=False,
            age_range=st.session_state.get("filter_age", (2, 15)),
            cohort_range=st.session_state.get("filter_cohort_years")
            if st.session_state.get("filter_registration", False)
            else None,
            min_sample=int(st.session_state.get("filter_sample", 100)),
        )
        comparison_page(scored=scored, ranking=ranking, historical=historical)
    elif page == "Age and mileage":
        age_mileage_page(scored=scored)
    elif page == "Mileage explorer":
        mileage_page(scored=scored, ranking=ranking)
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
    elif page == "Electric and hybrid vehicles":
        electrified_page(scored=scored, weights=weights)
    elif page == "Recalls and complaints":
        safety_page(events=events, directory=directory)
    elif page == "Repair costs":
        repair_cost_page()
    elif page == "Sources and methods":
        methods_page(evidence=evidence)
    elif page == "Data sources and fault definitions":
        source_information_page()
    else:
        management_page(
            directory=directory, evidence=evidence, events=events, manifest=manifest
        )


if __name__ == "__main__":
    run()
