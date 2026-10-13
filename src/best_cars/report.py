"""Create a self-contained interactive overview from verified evidence."""

import argparse
import html
import logging
from pathlib import Path

import pandas as pd

from best_cars.analysis import select_evidence, source_agreement
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
from best_cars.comparison import (
    matched_model_rates,
    temporal_profiles,
    wilson_age_groups,
)
from best_cars.comparison_charts import (
    evidence_coverage_chart,
    matched_rate_chart,
    peer_gap_chart,
    temporal_chart,
)
from best_cars.fairness import (
    cohort_contexts,
    coverage_matched_ranking,
    evidence_profiles,
    exploratory_ranking,
    source_cohort_ranking,
)
from best_cars.interrogation import cohort_peer_benchmarks, weighting_scenarios
from best_cars.propulsion import propulsion_catalogue
from best_cars.repair_costs import repair_references
from best_cars.source_guide import SOURCE_GUIDE
from best_cars.statistics import (
    counted_failure_rates,
    pairwise_source_agreement,
    source_rate_summary,
)
from best_cars.storage import data_directory, load_snapshot, source_catalogue

LOGGER = logging.getLogger(__name__)


def report_table(*, frame: pd.DataFrame, table_id: str) -> str:
    """Build an escaped, searchable and sortable table for the offline report.

    Args:
        frame: Descriptive values with missing observations retained.
        table_id: Internal HTML identifier, escaped before insertion.

    Returns:
        Accessible search input and a table with explicit missing-value labels.
    """
    if frame.empty:
        return '<p class="note">No observations meet this selection.</p>'
    safe_id = html.escape(s=table_id, quote=True)
    display = frame.rename(
        columns={name: str(name).replace("_", " ") for name in frame.columns}
    )
    table = display.round(decimals=2).to_html(
        index=False, border=0, escape=True, na_rep="Unavailable"
    )
    return (
        f'<div class="filter-table" id="{safe_id}">'
        f'<label for="{safe_id}-search">Search this table</label>'
        f'<input id="{safe_id}-search" type="search" placeholder="Type a model, source or year…">'
        '<p class="table-hint">Select a column heading to sort. Missing values are unavailable, not zero.</p>'
        f'<div class="scroll">{table}</div></div>'
    )


def report_html(*, evidence: pd.DataFrame, events: pd.DataFrame) -> str:
    """Build an offline HTML report with real charts and explicit source dates.

    Args:
        evidence: Complete validated canonical reliability observations.
        events: Saved safety-context events.

    Returns:
        Self-contained UTF-8 HTML, including Plotly once and no external assets.
    """
    scored = select_evidence(evidence=evidence, age_range=(2, 15), min_sample=100)
    preference_profiles = coverage_matched_ranking(
        scored=scored,
        min_sources=min(2, max(1, evidence.source_id.nunique())),
        min_cohorts=3,
    )
    eligible_blocks = preference_profiles.loc[
        preference_profiles.comparison_models.ge(2)
    ]
    if eligible_blocks.empty:
        ranking = preference_profiles.iloc[:0]
    else:
        block = (
            eligible_blocks.groupby(by="comparison_group_id")
            .size()
            .sort_values(ascending=False)
            .index[0]
        )
        ranking = eligible_blocks.loc[
            eligible_blocks.comparison_group_id.eq(block)
        ].sort_values(by="rank")
    coverage = (
        evidence.groupby(
            by=["source_name", "country", "report_year", "observation_year"]
        )
        .agg(models=("model_id", "nunique"), observations=("model_id", "size"))
        .reset_index()
    )
    names_frame = evidence.drop_duplicates(subset="model_id")
    names = dict(
        zip(
            names_frame.model_id,
            names_frame.make + " " + names_frame.model,
            strict=True,
        )
    )
    wide = source_agreement(scored=scored)
    keys = [key for key in ranking.head(n=25).model_id if key in wide.index]
    first_chart = ranking_chart(ranking=ranking, top_n=20).to_html(
        full_html=False, include_plotlyjs=True
    )
    primary_charts = []
    for source_id in sorted(scored.source_id.unique()):
        contexts = cohort_contexts(scored=scored, source_id=source_id)
        if contexts.empty:
            continue
        context = contexts.iloc[0]
        ordering = source_cohort_ranking(
            scored=scored, source_id=source_id, context_id=context.context_id
        )
        primary_charts.append(
            f"<h3>{html.escape(s=str(source_id))}: {html.escape(s=context.context_label)}</h3>"
            + ranking_chart(ranking=ordering, top_n=12).to_html(
                full_html=False, include_plotlyjs=False
            )
        )
    profile_table = report_table(
        frame=evidence_profiles(scored=scored), table_id="source-profiles"
    )
    all_preference_table = report_table(
        frame=preference_profiles, table_id="coverage-preferences"
    )
    exploratory_table = report_table(
        frame=exploratory_ranking(scored=scored, min_sources=2, min_cohorts=3),
        table_id="exploratory-ranking",
    )
    repair_table = report_table(frame=repair_references(), table_id="repair-prices")
    second_chart = agreement_heatmap(wide=wide.loc[keys], names=names).to_html(
        full_html=False, include_plotlyjs=False
    )
    summary = source_rate_summary(evidence=scored)
    benchmarks = cohort_peer_benchmarks(scored=scored)
    example_id = (
        "toyota:corolla"
        if "toyota:corolla" in set(benchmarks.model_id)
        else (benchmarks.model_id.iloc[0] if not benchmarks.empty else "")
    )
    peer_table = report_table(
        frame=benchmarks.loc[
            benchmarks.model_id.eq(example_id),
            [
                "make",
                "model",
                "source_name",
                "report_year",
                "age_min",
                "age_max",
                "unit",
                "value",
                "peer_median_rate",
                "rate_gap",
                "peer_models",
            ],
        ],
        table_id="peer-benchmarks",
    )
    scenarios = weighting_scenarios(scored=scored)
    scenarios = scenarios.loc[scenarios.model_id.isin(ranking.model_id)]
    scenario_table = report_table(frame=scenarios, table_id="weight-scenarios")
    propulsion = propulsion_catalogue(evidence=scored)
    propulsion_table = report_table(
        frame=propulsion.loc[~propulsion.category.eq("Unclassified")],
        table_id="propulsion-review",
    )
    counted = counted_failure_rates(evidence=scored)
    paired = pairwise_source_agreement(wide=wide)
    distribution_chart = rate_distribution_chart(evidence=scored).to_html(
        full_html=False, include_plotlyjs=False
    )
    uncertainty_chart = "".join(
        f"<h3>{html.escape(s=label)}</h3>"
        + count_uncertainty_chart(counted=group, top_n=24, colour_by="make").to_html(
            full_html=False, include_plotlyjs=False
        )
        for label, group in wilson_age_groups(counted=counted).items()
    )
    spread_chart = score_spread_chart(ranking=ranking).to_html(
        full_html=False, include_plotlyjs=False
    )
    age_profiles = age_chart(
        scored=scored, model_ids=ranking.head(n=6).model_id.tolist()
    ).to_html(full_html=False, include_plotlyjs=False)
    mileage_sections = []
    age_heatmaps = []
    for _source_id, group in scored.groupby(by="source_id", sort=True):
        age_heatmaps.append(
            age_score_heatmap(
                scored=group, max_models=30, order="alphabetical"
            ).to_html(full_html=False, include_plotlyjs=False)
        )
        eligible = group.loc[
            group.mileage_basis.eq("mean_odometer") & group.mileage_km.notna()
        ]
        if eligible.empty:
            continue
        mileage_sections.append(
            f"<h3>{html.escape(s=str(group.iloc[0].source_name))}</h3>"
            + mileage_chart(evidence=group).to_html(
                full_html=False, include_plotlyjs=False
            )
        )
    pair_charts = []
    for row in paired.itertuples():
        if row.paired_models >= 5:
            pair_charts.append(
                agreement_scatter(
                    wide=wide,
                    first=row.first_source,
                    second=row.second_source,
                    names=names,
                ).to_html(full_html=False, include_plotlyjs=False)
            )
    counted_columns = [
        "source_name",
        "make",
        "model",
        "observation_year",
        "age_min",
        "age_max",
        "sample_size",
        "event_count",
        "value",
        "counted_rate_pct",
        "wilson_lower_pct",
        "wilson_upper_pct",
    ]
    safety_summary = (
        events.groupby(
            by=["country", "make", "model", "model_year", "event_type"], dropna=False
        )
        .size()
        .reset_index(name="records")
    )
    coverage_table = report_table(frame=coverage, table_id="coverage")
    summary_table = report_table(frame=summary, table_id="rate-summary")
    counts_table = report_table(
        frame=counted.loc[:, counted_columns], table_id="counted-rates"
    )
    agreement_table = report_table(frame=paired, table_id="agreement-summary")
    ranking_table = report_table(frame=ranking, table_id="model-ranking")
    safety_table = report_table(frame=safety_summary, table_id="safety-records")
    source_links = "".join(
        f'<li><a href="{html.escape(s=str(row.url), quote=True)}">{html.escape(s=row.name)}</a>: {html.escape(s=row.limitations)}</li>'
        for row in source_catalogue().itertuples()
        if row.source_id
        in {"tuv", "adac", "traficom", "mot", "nhtsa_recalls", "nhtsa_complaints"}
    )
    definitions = "".join(
        f"<details><summary>{html.escape(s=name)}</summary>"
        + "".join(
            f"<p>{html.escape(s=guide[field])}</p>"
            for field in ["meaning", "faults", "limits"]
        )
        + f'<p><a href="{html.escape(s=guide["definition_url"], quote=True)}">Official definitions</a></p></details>'
        for name, guide in SOURCE_GUIDE.items()
    )
    comparison_ids = [
        key
        for key in ["volkswagen:t roc", "toyota:corolla", "toyota:c hr"]
        if key in set(scored.model_id)
    ]
    if not comparison_ids:
        comparison_ids = (
            scored.loc[~scored.model_grouped]
            .model_id.drop_duplicates()
            .head(n=3)
            .tolist()
        )
    example_rows = scored.loc[scored.model_id.isin(comparison_ids)]
    comparison_figures = []
    if comparison_ids:
        comparison_figures = [
            age_chart(scored=example_rows, model_ids=comparison_ids),
            peer_gap_chart(
                benchmarks=benchmarks.loc[benchmarks.model_id.isin(comparison_ids)]
            ),
            evidence_coverage_chart(
                evidence=example_rows, model_ids=comparison_ids, names=names
            ),
        ]
        if len(comparison_ids) > 1:
            comparison_figures.append(
                matched_rate_chart(
                    paired=matched_model_rates(
                        evidence=example_rows,
                        model_ids=comparison_ids,
                        reference_id=comparison_ids[0],
                    )
                )
            )
        history = select_evidence(
            evidence=evidence, latest_only=False, age_range=(2, 15), min_sample=100
        )
        for mode in ["age", "registration"]:
            profiles = temporal_profiles(
                evidence=history, model_ids=comparison_ids, mode=mode
            )
            if not profiles.empty:
                figure = temporal_chart(profiles=profiles)
                figure.update_layout(
                    title="Across years · "
                    + (
                        "published age held fixed"
                        if mode == "age"
                        else "recorded registration cohort held fixed"
                    )
                )
                comparison_figures.append(figure)
    comparison_html = "".join(
        figure.to_html(full_html=False, include_plotlyjs=False)
        for figure in comparison_figures
    )
    comparison_labels = html.escape(s=", ".join(names[key] for key in comparison_ids))
    return f"""<!doctype html><html lang="en-GB"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Best Cars | Reliability evidence</title>
    <style>body{{font-family:Arial,sans-serif;background:#edf3f5;color:#163548;margin:0}}
    main{{max-width:1180px;margin:35px auto;padding:36px;background:white;border-radius:20px}}
    h1{{font-size:40px;letter-spacing:-1px;margin-bottom:10px}}h2{{margin-top:32px}}
    .tag{{color:#007f87;font-size:13px;letter-spacing:2px;font-weight:bold}}
    .intro{{font-size:18px;max-width:900px;line-height:1.6}}
    .cards{{display:flex;gap:12px;flex-wrap:wrap;margin:25px 0}}.card{{background:#edf6f6;padding:22px;border-radius:12px;flex:1}}
    .card b{{font-size:28px;display:block}}table{{border-collapse:collapse;width:100%;font-size:13px}}
    td,th{{padding:10px;border-bottom:1px solid #dbe3e8;text-align:left}}th{{background:#edf6f6}}
    .scroll{{overflow:auto;max-height:580px}}p,li{{line-height:1.65}}.note{{border-left:4px solid #007f87;padding:12px 20px;background:#f4f8f9}}
    nav{{display:flex;flex-wrap:wrap;gap:8px;margin:25px 0}}nav a{{padding:9px 14px;background:#edf6f6;border-radius:8px;text-decoration:none}}
    section{{scroll-margin-top:20px}}label{{display:block;font-weight:bold;margin:15px 0 8px}}input{{padding:12px;border:1px solid #b5c6ce;border-radius:8px;width:min(90%,420px);font:inherit}}
    .table-hint{{font-size:12px;color:#476674}}th{{cursor:pointer;position:sticky;top:0;white-space:nowrap}}th:focus{{outline:2px solid #007f87}}details{{margin:18px 0}}summary{{cursor:pointer;font-weight:bold}}
    a{{color:#235e9c}}@media(max-width:700px){{main{{padding:16px;margin:8px}}h1{{font-size:30px}}}}</style></head>
    <body><main><div class="tag">BEST CARS · RELIABILITY EVIDENCE</div>
    <h1>Look beyond a brand's reputation</h1>
    <p class="intro">Start with separate inspection and roadside-breakdown evidence. These outcomes do not establish engine durability, service life or repair cost. Optional exploratory rankings show missing evidence; strict comparisons match source/cohort coverage. Charts support hover, zoom and legend controls; tables support search and sorting. This report works offline.</p>
    <div class="cards"><div class="card"><b>{len(evidence):,}</b>Rate observations</div>
    <div class="card"><b>{evidence.loc[~evidence.model_grouped].model_id.nunique():,}</b>Model series</div>
    <div class="card"><b>{evidence.source_id.nunique()}</b>Reliability sources</div>
    <div class="card"><b>{len(events):,}</b>Saved safety events</div></div>
    <nav aria-label="Report sections"><a href="#ranking">Models</a><a href="#comparison">Model comparisons</a><a href="#rates">Distributions</a><a href="#uncertainty">Counts &amp; uncertainty</a><a href="#age">Age</a><a href="#mileage">Mileage</a><a href="#agreement">Source agreement</a><a href="#longevity">Life expectancy</a><a href="#methods">Methods</a></nav>
    <p class="note">This overview uses the latest loaded report per source, ages 2–15, at least 100 tests where a denominator is known, and at least two sources where available. Relative scores are exploratory percentiles, not failure probabilities. They cover model series, often spanning engines and generations.</p>
    <h2>What is actually loaded?</h2>{coverage_table}
    <p>Report year and observation year differ. TÜV's 2024–2025 secondary tables and ADAC's 2025 report are historical evidence; the Finnish data include inspections in 2025, published in 2026. Newer public summaries are not passed off as complete model tables.</p>
    <section id="ranking"><h2>Separate outcomes and exact-cohort comparisons</h2>
    <p>Each chart below compares one explicit source, report and age/registration cohort. It shows relative positions in that cohort, not a general reliability league table. The Streamlit app lets you select other cohorts. Source profiles retain conflicting outcomes and different age coverage.</p>{"".join(primary_charts)}{profile_table}
    <details><summary>Exploratory combined ranking with evidence gaps</summary><p>Missing source percentiles explicitly assume 50. Lower/upper score and possible-rank bounds substitute 0 and 100; they are missing-evidence sensitivity, not confidence intervals. Age mixes can still differ. Sample rows are not unique vehicles. No source is universally trusted.</p>{exploratory_table}</details><details><summary>Strict preference comparison: one identical-coverage block</summary>{first_chart}<p>This secondary comparison covers {len(ranking)} models in one coverage block. Every model has all active sources and identical published cohort coverage. Inspection sources are averaged within an inspection outcome, then inspection and breakdown receive equal outcome weights by default. Scores from different coverage blocks cannot be compared. Singleton profiles have no rank.</p>{ranking_table}{spread_chart}{all_preference_table}</details></section>
    <section id="deeper"><h2>Same-cohort peers and sensitivity to priorities</h2>
    <p>The example below compares one model with the median of other ungrouped models in its exact source, report, age and registration cohort. Negative gaps indicate a lower published rate. At least four other models are required; source units remain separate. The Streamlit app lets you choose any supported model.</p>{peer_table}
    <details><summary>Browse all three preference scenarios</summary>{scenario_table}</details><p>Scenario ranks share the same common eligible competitors. They express preferences, not confidence intervals or calibrated trust.</p></section>
    <section id="electrified"><h2>Electric, plug-in hybrid and hybrid eligibility</h2>{propulsion_table}
    <p>This conservative catalogue is not exhaustive. Unreviewed identities are unclassified. Mixed-family rows are context only: their pooled reliability rates are not specific to a hybrid or electric variant. The dedicated Streamlit page excludes mixed families by default and provides age profiles, peer benchmarks and sensitivity analysis. No battery-health or battery-lifetime estimate is available.</p></section>
    <section id="comparison"><h2>Explore selected models in more detail</h2><p>Illustrative selection: {comparison_labels}. Choose any one model or up to twenty in the Streamlit Compare cars page. These examples are not a list of recommended purchases.</p>{comparison_html}<p>Source units stay separate. Peer-gap bars are negative when the selected model has a lower rate than the median of other models in its exact published cohort. Matched-model scatter points below the diagonal favour the compared model. Coverage colours indicate data availability, not reliability. Wilson intervals use actual inspection counts only.</p><p>The historical charts include earlier loaded reports where at least two observation years exist. Age mode compares different registration cohorts at the same age; registration mode follows repeated summaries of recorded registration cohorts, excludes approximate TÜV cohort estimates, and does not track individual cars. Observation years for TÜV label inspection-window endings. No trajectory estimates lifetime or a causal effect of ageing.</p></section>
    <section id="repair"><h2>Repair cost context, not a model ownership-cost ranking</h2>{repair_table}<p>These dated UK marketplace job averages are general context, not quotations for Corolla or any other model. The current inspection and breakdown data lack model-specific fault frequencies, severity and repair invoices. The Streamlit repair page supports explicit job/quote budget scenarios; it does not predict annual costs or assign repair-cost penalties to reliability scores.</p></section>
    <section id="rates"><h2>Means, medians and rate distributions</h2>{distribution_chart}
    <p>The box centre is the median, the box spans the 25th–75th percentiles, and the dotted marker shows the mean. These distributions give every published model-cohort observation equal weight and mix the selected ages. They are not national fleet failure rates or lifespan estimates. Each source keeps its own units and definition.</p>
    {summary_table}<p>The counted-test failure percentage pools actual failures and tests only within the displayed source/year/metric group. Count coverage shows which observations contribute. Missing denominators are not inferred; the counted subset need not represent the whole fleet.</p></section>
    <section id="uncertainty"><h2>Actual test counts and uncertainty</h2>{uncertainty_chart}
    <p>Separate charts match exact recorded age/cohort, source, report and observation year. Each chart shows up to 24 model-cohort rows, chosen independently by largest test sample. Points use failures divided by tests; error bars are 95% Wilson intervals. Counts are never pooled between charts. Narrow intervals describe sampling precision under a binomial model; repeated vehicles, maintenance, usage and selection bias remain outside those intervals. The app also offers two-year groups and age-panel selection.</p>
    <details><summary>Search all counted cohorts and their intervals</summary>{counts_table}</details></section>
    <section id="age"><h2>Example age profiles from one coverage block</h2>{age_profiles}
    <h3>Source-specific cohort evidence across ages</h3>{"".join(age_heatmaps)}<p>Heatmaps show up to 30 model series alphabetically, not a general best-to-worst list. Each cell retains its source-cohort peer comparison. Red means a less favourable percentile, teal more favourable, and grey no score. Hover for the raw rate. The app supports search and other descriptive row orderings.</p>
    <p>Each point is a published age or registration cohort. Lines connect different cars and can span model generations; they do not track individual vehicles ageing. Source-specific vertical scales preserve percentages and breakdowns per 1,000 vehicles.</p></section>
    <section id="mileage"><h2>Mileage and inspection outcomes</h2>{"".join(mileage_sections) or "<p>No mean-odometer observations are available in this selection.</p>"}
    <p>Points describe model-cohort mean odometers, coloured by age. An association between mileage and failures can reflect age, driver behaviour, servicing or fleet composition. No failure rate is divided by mileage. ADAC annual mileage for breakdown vehicles is a different measure and is excluded here.</p></section>
    <section id="agreement"><h2>Agreement and gaps between sources</h2>{second_chart}
    <p>Blank cells mean no comparable evidence. Different outcomes, countries and registration cohorts can explain disagreements. A green inspection result does not guarantee a low roadside breakdown rate.</p>
    {agreement_table}<p>Spearman rho measures ordering agreement among model series observed in both sources: +1 is the same ordering, 0 indicates no monotonic association, and −1 is reversed ordering. The paired-model count is essential context. These are descriptive associations between source averages; cohorts are not matched and no causal or significance claim is made.</p>
    <details><summary>Inspect all source-pair scatter plots</summary>{"".join(pair_charts) or "<p>No source pair has at least five observed models.</p>"}</details></section>
    <section id="longevity"><h2>Life expectancy: linked vehicle histories required</h2>
    <p>MOT histories can support estimates of service life on British roads. The <a href="https://www.nature.com/articles/s41560-024-01698-1">2025 Nature Energy study</a> demonstrates this approach using linked test histories and survival analysis. Disappearance is an inferred fleet exit: export, storage, accidents and economic decisions can also stop tests.</p>
    <p class="note">No lifespan estimate has been calculated in this report. The current importer produces failure-rate aggregates; longitudinal vehicle IDs and dates must be retained in a separate history layer before estimating longevity.</p>
    <p>The analysis should account for cars still observed at the actual data cut-off, entry around the first MOT, delayed tests, and an uncertain exit interval after the last test. Compare gap rules such as 15, 18 and 21 months. Report median road-service lifetime only when estimable; use a restricted mean to a stated horizon when the tail is unobserved. New models cannot yet supply observed full lifetimes.</p>
    <p>Useful graphics would include survival curves with uncertainty, survival to ages 10/15/20, and remaining service life conditional on current age. Results should be split by model series, registration cohort and fuel type where coverage is adequate.</p></section>
    <section id="definitions"><h2>What each source considers a fault</h2>{definitions}</section><section id="methods"><h2>Methods and source links</h2><p>Lower raw rates are favourable. Source rankings compare exact published cohorts with original full-peer percentiles. Optional composites require all active sources and identical source/cohort coverage within each comparison block; inspection and breakdown outcome weights are separate. A singleton has no rank. Source-profile means spanning ages are descriptive, not an overall league table. The legacy unbalanced average is no longer the default. There is no mileage division, inferred fleet denominator, fault-severity imputation or recall-count reliability score.</p>
    <ul>{source_links}</ul><p>NHTSA safety lookups concern exact US make/model/year combinations. Saved complaints are voluntary reports with no exposure denominator; they are kept out of the rankings. UK MOT import support is supplied, but no national UK bulk dataset is bundled in this overview.</p>
    <h3>Exact safety queries saved in this snapshot</h3>{safety_table}</section>
    </main><script>
    for (const panel of document.querySelectorAll('.filter-table')) {{
      const input = panel.querySelector('input');
      const table = panel.querySelector('table');
      input.addEventListener('input', () => {{
        const term = input.value.trim().toLocaleLowerCase('en-GB');
        for (const row of table.tBodies[0].rows) {{ row.hidden = !row.textContent.toLocaleLowerCase('en-GB').includes(term); }}
      }});
      for (const [column, heading] of Array.from(table.tHead.rows[0].cells).entries()) {{
        heading.tabIndex = 0;
        heading.title = 'Sort this column';
        const sortRows = () => {{
          const ascending = heading.dataset.order !== 'asc';
          const rows = Array.from(table.tBodies[0].rows);
          rows.sort((a, b) => {{
            const x = a.cells[column].textContent.trim(); const y = b.cells[column].textContent.trim();
            if (x === 'Unavailable' || y === 'Unavailable') return x === y ? 0 : x === 'Unavailable' ? 1 : -1;
            const numeric = x !== '' && y !== '' && Number.isFinite(Number(x)) && Number.isFinite(Number(y));
            const delta = numeric ? Number(x) - Number(y) : x.localeCompare(y, 'en-GB', {{numeric:true}});
            return ascending ? delta : -delta;
          }});
          for (const row of rows) table.tBodies[0].appendChild(row);
          heading.dataset.order = ascending ? 'asc' : 'desc';
        }};
        heading.addEventListener('click', sortRows);
        heading.addEventListener('keydown', (event) => {{ if (event.key === 'Enter' || event.key === ' ') {{ event.preventDefault(); sortRows(); }} }});
      }}
    }}
    </script></body></html>"""


def main(*, argv: list[str] | None = None) -> int:
    """Export the self-contained interactive report with named CLI settings.

    Args:
        argv: Optional argument list.

    Returns:
        Zero on successful export.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(args=argv)
    if args.output.suffix.lower() != ".html":
        parser.error("--output must have an .html extension")
    evidence, events, _ = load_snapshot(
        directory=data_directory(supplied=args.data_dir)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        data=report_html(evidence=evidence, events=events), encoding="utf-8"
    )
    LOGGER.info("Exported interactive overview to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
