"""Interactive graphics with explicit scales and source-aware labels."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PALETTE = ["#007f87", "#2856a8", "#d8832f", "#8545a8", "#c24a62", "#568337"]
SOURCE_NAMES = {
    "adac": "ADAC breakdowns",
    "traficom": "Traficom inspections",
    "tuv": "TÜV defects",
    "mot": "UK MOT",
}


def style_figure(*, figure: go.Figure, title: str, height: int = 480) -> go.Figure:
    """Apply readable common chart styling.

    Args:
        figure: Plotly chart to style.
        title: Human-readable title.
        height: Pixel height.

    Returns:
        The styled chart.
    """
    figure.update_layout(
        title=title,
        template="plotly_white",
        height=height,
        font={"family": "Arial, sans-serif", "size": 15},
        margin={"l": 20, "r": 20, "t": 60, "b": 35},
        legend={"title": {"text": ""}},
        hoverlabel={"namelength": -1},
    )
    return figure


def ranking_chart(*, ranking: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Show the highest relative evidence scores alongside source breadth.

    Args:
        ranking: Validated ranking table.
        top_n: Maximum models shown.

    Returns:
        Interactive bar chart; a score is not a failure probability.

    Raises:
        ValueError: If top_n is non-positive.
    """
    if top_n < 1:
        raise ValueError("Chart model count must be positive.")
    frame = ranking.head(n=top_n).copy()
    if frame.empty:
        return style_figure(figure=go.Figure(), title="No models meet these filters")
    frame["car"] = frame.make + " " + frame.model
    frame["source_count"] = frame.sources.astype(int).astype(str) + " sources"
    figure = px.bar(
        data_frame=frame.iloc[::-1],
        x="score",
        y="car",
        orientation="h",
        color="source_count",
        color_discrete_map={
            "1 sources": "#8596a6",
            "2 sources": "#70aca8",
            "3 sources": "#007f87",
            "4 sources": "#2856a8",
        },
        text=frame.iloc[::-1].score.round(decimals=1),
        hover_data=["cohorts", "score_spread", "weight_coverage", "worst_cohort_score"],
        labels={
            "score": "Relative evidence score (0–100; higher is better)",
            "car": "",
            "sources": "Sources",
            "source_count": "Evidence breadth",
            "cohorts": "Observed cohorts",
            "score_spread": "Between-source spread",
            "weight_coverage": "Available weight (%)",
        },
    )
    figure.update_xaxes(range=[0, 105])
    figure.update_yaxes(
        categoryorder="array", categoryarray=frame.car.iloc[::-1].tolist()
    )
    return style_figure(
        figure=figure,
        title="Models with the strongest relative evidence",
        height=max(400, len(frame) * 27 + 100),
    )


def age_chart(
    *, scored: pd.DataFrame, model_ids: list[str], relative: bool = False
) -> go.Figure:
    """Plot observed age profiles separately for each source and report.

    Args:
        scored: Selected canonical evidence.
        model_ids: Explicit model IDs to compare.
        relative: Show cohort percentiles instead of different raw rates.

    Returns:
        Faceted line/marker chart. Points are cross-sectional, not tracked cars.
    """
    frame = scored.loc[scored.model_id.isin(model_ids)].copy()
    if frame.empty:
        return style_figure(
            figure=go.Figure(), title="No age evidence for this selection"
        )
    # Source labels can differ in case; one model ID must keep one name/colour.
    identities = frame.drop_duplicates(subset="model_id")
    names = dict(
        zip(identities.model_id, identities.make + " " + identities.model, strict=True)
    )
    frame["car"] = frame.model_id.map(arg=names)
    colour_map = {
        names[key]: PALETTE[index % len(PALETTE)]
        for index, key in enumerate(model_ids)
        if key in names
    }
    frame["age"] = (frame.age_min + frame.age_max) / 2
    frame["report"] = frame.report_year.astype(int).astype(str)
    frame["source_panel"] = frame.source_name + frame.unit.map(
        arg={"percent": " · %", "per_1000_vehicles": " · per 1,000 vehicles"}
    )
    frame = frame.sort_values(by=["source_panel", "car", "report", "age"])
    value = "relative_score" if relative else "value"
    figure = px.line(
        data_frame=frame,
        x="age",
        y=value,
        color="car",
        line_dash="report",
        facet_row="source_panel",
        markers=True,
        color_discrete_map=colour_map,
        hover_data=[
            "age_min",
            "age_max",
            "cohort_year_start",
            "sample_size",
            "source_model",
        ],
        labels={
            "age": "Observed vehicle age (years)",
            "car": "Model series",
            "value": "Published rate",
            "relative_score": "Relative score",
            "report": "Report year",
        },
    )
    figure.update_yaxes(matches=None)
    if relative:
        figure.update_yaxes(range=[0, 105])
    for annotation in figure.layout.annotations:
        annotation.text = annotation.text.replace("source_panel=", "")
        domain = min(
            (axis.domain for axis in figure.select_yaxes()),
            key=lambda bounds: abs((bounds[0] + bounds[1]) / 2 - annotation.y),
        )
        annotation.update(
            textangle=0,
            x=0,
            xanchor="left",
            yanchor="bottom",
            y=domain[1],
            yshift=6,
            font={"size": 13},
        )
    figure.update_layout(legend={"orientation": "h", "y": -0.15})
    return style_figure(
        figure=figure,
        title="How the observed results vary with age",
        height=max(420, frame.source_panel.nunique() * 260),
    )


def agreement_heatmap(*, wide: pd.DataFrame, names: dict[str, str]) -> go.Figure:
    """Show agreement and gaps between source-specific relative scores.

    Args:
        wide: Model-by-source matrix; unavailable cells must remain NaN.
        names: Human-readable model labels.

    Returns:
        Interactive heatmap with an explicit 0–100 colour scale.
    """
    if wide.empty:
        return style_figure(figure=go.Figure(), title="No overlapping source evidence")
    figure = go.Figure(
        data=go.Heatmap(
            z=wide.to_numpy(),
            x=[
                {
                    "adac": "ADAC breakdowns",
                    "traficom": "Traficom inspections",
                    "tuv": "TÜV defects",
                    "mot": "UK MOT",
                }.get(key, key)
                for key in wide.columns
            ],
            y=[names.get(key, key) for key in wide.index],
            zmin=0,
            zmax=100,
            colorscale=[[0, "#c95b55"], [0.5, "#f1e9c3"], [1, "#007f87"]],
            hoverongaps=False,
            text=[
                [f"{value:.0f}" if pd.notna(value) else "No data" for value in row]
                for row in wide.to_numpy()
            ],
            texttemplate="%{text}",
            textfont={"size": 12},
            colorbar={"title": {"text": "Relative<br>score"}},
            hovertemplate="%{y}<br>%{x}: %{z:.1f}<extra></extra>",
        )
    )
    figure.update_yaxes(autorange="reversed", type="category")
    figure.update_xaxes(type="category")
    figure.update_layout(plot_bgcolor="#edf1f4")
    for row_index, key in enumerate(wide.index):
        for column_index, _source in enumerate(wide.columns):
            if pd.isna(wide.iloc[row_index, column_index]):
                figure.add_annotation(
                    x=figure.data[0].x[column_index],
                    y=names.get(key, key),
                    text="No data",
                    showarrow=False,
                    font={"size": 11, "color": "#536570"},
                )
    return style_figure(
        figure=figure,
        title="Where the sources agree — and where they differ",
        height=max(400, len(wide) * 25 + 120),
    )


def age_score_heatmap(
    *,
    scored: pd.DataFrame,
    max_models: int = 30,
    order: str = "both",
    measure: str = "relative",
    rate_ceiling: float | None = None,
) -> go.Figure:
    """Show favourable/unfavourable cohort scores by model and observed age.

    Args:
        scored: One source/report/observation-year/metric scored selection.
        max_models: Maximum rows to show legibly.
        order: Both ranking ends, weakest first or strongest first.
        measure: Relative percentile or published rate; labels and colours use
            exactly the same numeric matrix in either view.
        rate_ceiling: Optional full-source rate maximum, retained during search.

    Returns:
        Red-to-teal cohort percentile cells with raw rates on hover. Missing or
        unscored cells are grey and labelled; zero scores remain red.

    Raises:
        ValueError: For mixed contexts, invalid ordering or non-positive limits.
    """
    if (
        max_models < 1
        or order not in {"both", "weakest", "strongest"}
        or measure not in {"relative", "rate"}
    ):
        raise ValueError("Require a positive row limit and a recognised ordering.")
    if any(
        scored[column].nunique() > 1
        for column in ["source_id", "report_year", "observation_year", "metric", "unit"]
    ):
        raise ValueError("Age heatmaps require one source, report, year and outcome.")
    frame = scored.loc[~scored.model_grouped].copy()
    if frame.empty:
        return style_figure(
            figure=go.Figure(), title="No ungrouped age evidence available"
        )
    means = (
        frame.groupby(by="model_id")
        .relative_score.mean()
        .sort_values(ascending=False, kind="stable")
    )
    if order == "weakest":
        keys = (
            means.sort_values(ascending=True, kind="stable")
            .head(n=max_models)
            .index.tolist()
        )
    elif order == "strongest":
        keys = means.head(n=max_models).index.tolist()
    elif len(means) <= max_models:
        keys = means.index.tolist()
    else:
        upper = (max_models + 1) // 2
        keys = list(means.head(n=upper).index) + list(
            means.tail(n=max_models - upper).index
        )
    frame["age_band"] = [
        f"{row.age_min:g}–{row.age_max:g}"
        if row.age_min != row.age_max
        else f"{row.age_min:g}"
        for row in frame.itertuples()
    ]
    if frame.duplicated(subset=["model_id", "age_band"]).any():
        raise ValueError(
            "Multiple cohorts map to the same age cell; select a narrower context."
        )
    bands = (
        frame.sort_values(by=["age_min", "age_max"]).age_band.drop_duplicates().tolist()
    )
    scores = frame.pivot(
        index="model_id", columns="age_band", values="relative_score"
    ).reindex(index=keys, columns=bands)
    rates = frame.pivot(index="model_id", columns="age_band", values="value").reindex(
        index=keys, columns=bands
    )
    names_frame = frame.drop_duplicates(subset="model_id").set_index(keys="model_id")
    names = [
        f"{names_frame.loc[key, 'make']} {names_frame.loc[key, 'model']}"
        for key in keys
    ]
    relative = measure == "relative"
    values = scores if relative else rates
    unit = "%" if frame.iloc[0].unit == "percent" else "per 1,000 vehicles"
    if rate_ceiling is not None and (
        not np.isfinite(rate_ceiling) or rate_ceiling < frame.value.max()
    ):
        raise ValueError("Rate ceiling must be finite and cover every observed rate.")
    maximum = (
        100
        if relative
        else float(frame.value.max() if rate_ceiling is None else rate_ceiling)
    )
    maximum = max(maximum, 1e-9)
    figure = go.Figure(
        data=go.Heatmap(
            z=values.to_numpy(),
            x=bands,
            y=names,
            zmin=0,
            zmax=maximum,
            colorscale=(
                [[0, "#c95b55"], [0.5, "#f1e9c3"], [1, "#007f87"]]
                if relative
                else [[0, "#007f87"], [0.5, "#f1e9c3"], [1, "#c95b55"]]
            ),
            text=[
                [
                    (f"{value:.0f}" if relative else f"{value:.1f}")
                    if pd.notna(value)
                    else "No score"
                    for value in row
                ]
                for row in values.to_numpy()
            ],
            texttemplate="%{text}",
            customdata=(rates if relative else scores).to_numpy(),
            hovertemplate=(
                "%{y}<br>Age: %{x}<br>Relative score: %{z:.1f}<br>Published rate: %{customdata:.2f} "
                + unit
                + "<extra></extra>"
                if relative
                else "%{y}<br>Age: %{x}<br>Published rate: %{z:.2f} "
                + unit
                + "<br>Relative score: %{customdata:.1f}<extra></extra>"
            ),
            hoverongaps=False,
            colorbar={
                "title": {
                    "text": "Score<br>higher = better"
                    if relative
                    else f"Rate ({unit})<br>lower = better"
                }
            },
        )
    )
    figure.add_trace(
        go.Heatmap(
            z=np.where(values.isna(), 0.0, np.nan),
            x=bands,
            y=names,
            colorscale=[[0, "#edf1f4"], [1, "#edf1f4"]],
            showscale=False,
            zmin=0,
            zmax=1,
            text=np.full(shape=values.shape, fill_value="—"),
            texttemplate="%{text}",
            hoverongaps=False,
            hovertemplate="%{y}<br>Age: %{x}<br>No available value<extra></extra>",
        )
    )
    figure.update_yaxes(
        autorange="reversed",
        type="category",
        categoryorder="array",
        categoryarray=names,
    )
    figure.update_xaxes(
        title_text="Observed age (years)",
        type="category",
        categoryorder="array",
        categoryarray=bands,
    )
    figure.update_layout(plot_bgcolor="#edf1f4")
    return style_figure(
        figure=figure,
        title=f"{frame.iloc[0].source_name} · {'relative scores (higher = better)' if relative else 'published rates (lower = better)'}",
        height=max(450, len(keys) * 30 + 140),
    )


def mileage_chart(*, evidence: pd.DataFrame) -> go.Figure:
    """Compare odometer mileage and inspection failure rates within one source.

    Args:
        evidence: Selected rows from one source/report, with odometer metadata.

    Returns:
        Scatter plot showing published cohort rates, not individual vehicles.

    Raises:
        ValueError: If incompatible evidence sources/units are mixed.
    """
    if evidence.source_id.nunique() > 1 or evidence.unit.nunique() > 1:
        raise ValueError("Mileage charts require one source and one metric unit.")
    frame = evidence.loc[
        evidence.mileage_basis.eq("mean_odometer") & evidence.mileage_km.notna()
    ].copy()
    if frame.empty:
        return style_figure(
            figure=go.Figure(), title="Odometer data unavailable for this source"
        )
    frame["car"] = frame.make + " " + frame.model
    frame["age_mid"] = (frame.age_min + frame.age_max) / 2
    figure = px.scatter(
        data_frame=frame,
        x="mileage_km",
        y="value",
        color="age_mid",
        hover_name="car",
        hover_data=["source_model", "sample_size", "report_year", "age_min", "age_max"],
        color_continuous_scale="Teal",
        labels={
            "mileage_km": "Mean odometer reading (km)",
            "value": "Inspection failure/defect rate (%)",
            "age_mid": "Age (years)",
        },
    )
    return style_figure(figure=figure, title="Mileage, age and inspection outcomes")


def agreement_scatter(
    *, wide: pd.DataFrame, first: str, second: str, names: dict[str, str]
) -> go.Figure:
    """Compare two source percentile averages for models with both observations.

    Args:
        wide: Model-by-source score matrix.
        first: X-axis source ID.
        second: Y-axis source ID.
        names: Model ID to display label.

    Returns:
        Scatter plot with a diagonal agreement guide; unavailable models excluded.

    Raises:
        ValueError: If two different existing sources are not selected.
    """
    if first == second or first not in wide.columns or second not in wide.columns:
        raise ValueError("Select two different evidence sources.")
    frame = wide.loc[:, [first, second]].dropna().copy()
    frame["car"] = [names.get(key, key) for key in frame.index]
    figure = px.scatter(
        data_frame=frame,
        x=first,
        y=second,
        hover_name="car",
        labels={
            first: f"{SOURCE_NAMES.get(first, first)}: relative score",
            second: f"{SOURCE_NAMES.get(second, second)}: relative score",
        },
    )
    figure.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=100,
        y1=100,
        line={"color": "#999999", "dash": "dot"},
    )
    figure.update_xaxes(range=[-2, 102])
    figure.update_yaxes(range=[-2, 102])
    return style_figure(
        figure=figure, title="Different evidence can tell different stories"
    )


def rate_distribution_chart(*, evidence: pd.DataFrame) -> go.Figure:
    """Show model-cohort rate distributions on separate source-specific axes.

    Args:
        evidence: Canonical observations for the explicit report selection.

    Returns:
        Box plots with separate units, median, quartiles, outliers and means.
    """
    if evidence.empty:
        return style_figure(figure=go.Figure(), title="No rate distributions available")
    frame = evidence.copy()
    frame["car"] = frame.make + " " + frame.model
    frame["source_panel"] = frame.source_name + frame.unit.map(
        arg={"percent": " · %", "per_1000_vehicles": " · per 1,000 vehicles"}
    )
    frame["distribution"] = "Published cohort rates"
    figure = px.box(
        data_frame=frame,
        x="distribution",
        y="value",
        facet_row="source_panel",
        points="outliers",
        hover_name="car",
        hover_data=["age_min", "age_max", "report_year", "sample_size"],
        labels={"value": "Published model-cohort rate"},
        color_discrete_sequence=PALETTE,
    )
    figure.update_traces(boxmean=True)
    figure.update_yaxes(matches=None)
    figure.update_xaxes(showticklabels=False, title_text="")
    for annotation in figure.layout.annotations:
        annotation.text = annotation.text.replace("source_panel=", "")
        domain = min(
            (axis.domain for axis in figure.select_yaxes()),
            key=lambda bounds: abs((bounds[0] + bounds[1]) / 2 - annotation.y),
        )
        annotation.update(
            textangle=0,
            x=0,
            xanchor="left",
            yanchor="bottom",
            y=domain[1],
            yshift=6,
            font={"size": 13},
        )
    return style_figure(
        figure=figure,
        title="Rate distributions within each source",
        height=max(420, frame.source_panel.nunique() * 250),
    )


def count_uncertainty_chart(*, counted: pd.DataFrame, top_n: int = 24) -> go.Figure:
    """Show actual test-failure percentages and their sampling uncertainty.

    Args:
        counted: Output of counted_failure_rates, with known denominators.
        top_n: Maximum cohorts displayed, chosen by largest test sample.

    Returns:
        Count-based point estimates and 95% Wilson error bars. Ordering by
        sample size is not a reliability ranking or an age-adjusted comparison.

    Raises:
        ValueError: If top_n is non-positive.
    """
    if top_n < 1:
        raise ValueError("Chart cohort count must be positive.")
    if counted.empty:
        return style_figure(figure=go.Figure(), title="No actual test counts available")
    frame = counted.sort_values(by="sample_size", ascending=False).head(n=top_n)
    labels = [
        f"{row.make} {row.model} · {SOURCE_NAMES.get(row.source_id, row.source_id)} · age {row.age_min:g}–{row.age_max:g} · report {row.report_year}"
        for row in frame.itertuples()
    ]
    figure = go.Figure(
        data=go.Scatter(
            x=frame.counted_rate_pct,
            y=labels,
            mode="markers",
            marker={"color": PALETTE[0], "size": 8},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": frame.wilson_upper_pct - frame.counted_rate_pct,
                "arrayminus": frame.counted_rate_pct - frame.wilson_lower_pct,
            },
            customdata=frame.loc[:, ["sample_size", "event_count", "value"]].to_numpy(),
            hovertemplate=(
                "%{y}<br>Counted failure rate: %{x:.2f}%"
                "<br>Tests: %{customdata[0]:,.0f}<br>Failed tests: %{customdata[1]:,.0f}"
                "<br>Published rate: %{customdata[2]:.2f}%<extra></extra>"
            ),
        )
    )
    figure.update_xaxes(title_text="Counted inspection failures (%)")
    figure.update_yaxes(autorange="reversed")
    return style_figure(
        figure=figure,
        title="Largest counted cohorts: rates and 95% Wilson intervals",
        height=max(420, len(frame) * 27 + 110),
    )


def score_spread_chart(*, ranking: pd.DataFrame) -> go.Figure:
    """Compare composite evidence scores with the spread between source means.

    Args:
        ranking: Ranked ungrouped models with source breadth and disagreement.

    Returns:
        An interactive model-series scatter plot; spread is not uncertainty.
    """
    if ranking.empty:
        return style_figure(figure=go.Figure(), title="No ranked models available")
    frame = ranking.copy()
    frame["car"] = frame.make + " " + frame.model
    frame["source_count"] = frame.sources.astype(str)
    figure = px.scatter(
        data_frame=frame,
        x="score",
        y="score_spread",
        color="source_count",
        hover_name="car",
        hover_data=["cohorts", "worst_cohort_score", "weight_coverage", "source_ids"],
        color_discrete_sequence=PALETTE,
        labels={
            "score": "Relative evidence score (0–100)",
            "score_spread": "Range of source means (percentile points)",
            "source_count": "Scored sources",
        },
    )
    figure.update_xaxes(range=[-2, 102])
    figure.update_yaxes(range=[-2, 102])
    return style_figure(
        figure=figure, title="Strong scores and disagreement between sources"
    )
