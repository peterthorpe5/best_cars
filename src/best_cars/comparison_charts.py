"""Model-focused graphics with source units and comparison context intact."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from best_cars.charts import PALETTE, SOURCE_NAMES, _make_colours, style_figure
from best_cars.interrogation import MATCH_KEYS
from best_cars.interval_evidence import source_rate_points


def _chart_frame(*, evidence: pd.DataFrame) -> pd.DataFrame:
    """Add consistent model, age and source-panel display labels."""
    frame = evidence.copy()
    if frame.empty:
        return frame
    names = frame.drop_duplicates(subset="model_id").set_index(keys="model_id")
    frame["car"] = frame.model_id.map(arg=(names.make + " " + names.model).to_dict())
    frame["age_band"] = [
        str(int(low)) + (f"–{int(high)}" if low != high else "")
        for low, high in zip(frame.age_min, frame.age_max, strict=True)
    ]
    frame["panel"] = (
        frame.source_id.map(arg=SOURCE_NAMES).fillna(frame.source_name)
        + frame.unit.map(
            arg={"percent": " · %", "per_1000_vehicles": " · per 1,000 vehicles"}
        )
        + " · report "
        + frame.report_year.astype(int).astype(str)
    )
    return frame


def _finish_panels(*, figure: go.Figure, title: str, panels: int) -> go.Figure:
    """Place panel labels above their independent source axes."""
    figure.update_yaxes(matches=None)
    for annotation in figure.layout.annotations:
        annotation.text = annotation.text.replace("panel=", "")
        domain = min(
            (axis.domain for axis in figure.select_yaxes()),
            key=lambda bounds: abs(sum(bounds) / 2 - annotation.y),
        )
        annotation.update(
            x=0,
            xanchor="left",
            y=domain[1],
            yanchor="bottom",
            yshift=8,
            textangle=0,
            font={"size": 13},
        )
    return style_figure(figure=figure, title=title, height=max(430, panels * 250))


def peer_gap_chart(*, benchmarks: pd.DataFrame, colour_by: str = "model") -> go.Figure:
    """Show each model's rate minus its exact-cohort peer median.

    Args:
        benchmarks: Output of cohort_peer_benchmarks restricted to selected rows.
        colour_by: Distinct model colours or stable make colours.

    Returns:
        Source/report panels with a zero reference; negative gaps are favourable.

    Raises:
        ValueError: If the colour setting is unrecognised.
    """
    if colour_by not in {"model", "make"}:
        raise ValueError("Choose model or make colours.")
    frame = _chart_frame(evidence=benchmarks)
    if frame.empty:
        return style_figure(
            figure=go.Figure(), title="No same-cohort peer reference available"
        )
    frame = frame.sort_values(by=["panel", "age_min", "age_max", "car"])
    figure = px.bar(
        data_frame=frame,
        x="age_band",
        y="rate_gap",
        color="car" if colour_by == "model" else "make",
        barmode="group",
        facet_row="panel",
        color_discrete_sequence=PALETTE,
        color_discrete_map=_make_colours(makes=frame.make.tolist())
        if colour_by == "make"
        else None,
        hover_name="car",
        hover_data=[
            "value",
            "peer_median_rate",
            "peer_models",
            "cohort_year_start",
            "cohort_year_end",
        ],
        labels={
            "rate_gap": "Rate minus other-model median",
            "age_band": "Published age band",
            "car": "Model",
            "value": "Model rate",
            "peer_median_rate": "Other-model median",
            "peer_models": "Other models",
        },
    )
    figure.add_hline(y=0, line_color="#536570", line_width=1)
    figure.update_xaxes(type="category")
    return _finish_panels(
        figure=figure,
        title="Better or worse than peers at the same age?",
        panels=frame.panel.nunique(),
    )


def evidence_coverage_chart(
    *, evidence: pd.DataFrame, model_ids: list[str], names: dict[str, str]
) -> go.Figure:
    """Display recorded coverage and count availability without scoring gaps.

    Args:
        evidence: Selected canonical observations from any number of sources.
        model_ids: Ordered models, including those lacking a shared cohort.
        names: Display names keyed by selected model ID.

    Returns:
        Categorical coverage grid. Grey is absent; gold is rate-only evidence;
        teal means a rate with both actual numerator and denominator available.

    Raises:
        ValueError: For duplicate model selections or model-context records.
    """
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("Coverage needs distinct models.")
    frame = evidence.loc[evidence.model_id.isin(model_ids)].copy()
    if frame.empty:
        return style_figure(figure=go.Figure(), title="No selected evidence coverage")
    if frame.duplicated(subset=MATCH_KEYS + ["model_id"]).any():
        raise ValueError("Coverage requires unique model-context records.")
    frame["context"] = frame.groupby(by=MATCH_KEYS, dropna=False, sort=True).ngroup()
    contexts = frame.drop_duplicates(subset="context").sort_values(
        by=["source_id", "report_year", "age_min", "age_max", "cohort_year_start"]
    )
    positions = {value: index for index, value in enumerate(contexts.context)}
    model_positions = {value: index for index, value in enumerate(model_ids)}
    z = np.zeros((len(model_ids), len(contexts)), dtype=int)
    text = np.full(z.shape, "No data", dtype=object)
    hover = np.full(z.shape, "No recorded row under these filters", dtype=object)
    for row in frame.itertuples():
        cell = (model_positions[row.model_id], positions[row.context])
        counted = pd.notna(row.sample_size) and pd.notna(row.event_count)
        z[cell] = 2 if counted else 1
        text[cell] = f"n={int(row.sample_size):,}" if counted else "Rate only"
        hover[cell] = (
            f"Published rate: {row.value:g} ({row.unit})"
            + (
                f"<br>Failed tests: {int(row.event_count):,} / {int(row.sample_size):,}"
                if counted
                else "<br>Actual failure/test count pair unavailable"
            )
            + f"<br>Observation year: {row.observation_year}<br>Registration: {row.cohort_year_start}–{row.cohort_year_end}"
        )
    labels = [
        f"{SOURCE_NAMES.get(row.source_id, row.source_id)} · {row.report_year}<br>age {row.age_min:g}–{row.age_max:g}"
        for row in contexts.itertuples()
    ]
    figure = go.Figure(
        data=go.Heatmap(
            z=z,
            x=list(range(len(labels))),
            y=[names.get(key, key) for key in model_ids],
            text=text,
            texttemplate="%{text}",
            textfont={"size": 11},
            customdata=hover,
            zmin=0,
            zmax=2,
            colorscale=[
                [0, "#e8edf0"],
                [0.249, "#e8edf0"],
                [0.25, "#f2db9f"],
                [0.749, "#f2db9f"],
                [0.75, "#78b9b4"],
                [1, "#78b9b4"],
            ],
            colorbar={
                "tickvals": [0, 1, 2],
                "ticktext": ["No data", "Rate only", "Rate + counts"],
                "title": "Evidence type",
            },
            xgap=2,
            ygap=2,
            hovertemplate="%{y}<br>%{customdata}<extra></extra>",
        )
    )
    figure.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(labels))),
        ticktext=labels,
        tickangle=-45,
        title_text="Published source/report/age contexts; colours show coverage, not reliability",
    )
    figure.update_yaxes(autorange="reversed", type="category")
    return style_figure(
        figure=figure,
        title="Where evidence exists — and where counts are missing",
        height=max(470, len(model_ids) * 45 + 230),
    )


def temporal_chart(*, profiles: pd.DataFrame, colour_by: str = "model") -> go.Figure:
    """Plot qualified same-age or registration-cohort summaries through time.

    Args:
        profiles: Output of temporal_profiles with at least two years per series.
        colour_by: Separate models or stable manufacturer colours.

    Returns:
        Distinct lines per cohort and model, with original source units.

    Raises:
        ValueError: If colour settings are invalid.
    """
    if colour_by not in {"model", "make"}:
        raise ValueError("Choose model or make colours.")
    frame = _chart_frame(evidence=profiles)
    if frame.empty:
        return style_figure(
            figure=go.Figure(),
            title="At least two observation years are needed per series",
        )
    frame["panel"] = frame.source_id.map(arg=SOURCE_NAMES).fillna(
        frame.source_name
    ) + frame.unit.map(
        arg={"percent": " · %", "per_1000_vehicles": " · per 1,000 vehicles"}
    )
    figure = px.line(
        data_frame=frame.sort_values(by=["series_id", "observation_year"]),
        x="observation_year",
        y="value",
        color="car" if colour_by == "model" else "make",
        line_group="series_id",
        line_dash="series_label",
        facet_row="panel",
        markers=True,
        hover_name="car",
        color_discrete_sequence=PALETTE,
        color_discrete_map=_make_colours(makes=frame.make.tolist())
        if colour_by == "make"
        else None,
        hover_data=[
            "report_year",
            "age_min",
            "age_max",
            "cohort_year_start",
            "cohort_year_end",
            "sample_size",
        ],
        labels={
            "observation_year": "Observation year / inspection-window ending year",
            "value": "Published rate",
            "series_label": "Comparison",
            "car": "Model",
        },
    )
    figure.update_xaxes(dtick=1)
    return _finish_panels(
        figure=figure,
        title="Changes across observation years",
        panels=frame.panel.nunique(),
    )


def matched_rate_chart(*, paired: pd.DataFrame, colour_by: str = "model") -> go.Figure:
    """Compare exact-cohort model rates with a selected reference model.

    Args:
        paired: Output of matched_model_rates.
        colour_by: Model or manufacturer colour preference.

    Returns:
        Source/report scatter panels with equal axes and a parity diagonal.
        Points below the diagonal favour the compared model, not the reference.

    Raises:
        ValueError: For an unrecognised colour preference.
    """
    if colour_by not in {"model", "make"}:
        raise ValueError("Choose model or make colours.")
    frame = _chart_frame(evidence=paired)
    if frame.empty:
        return style_figure(
            figure=go.Figure(),
            title="No exact published cohorts shared with the reference",
        )
    panels = list(frame.panel.unique())
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=len(panels),
        cols=1,
        subplot_titles=panels,
        vertical_spacing=min(0.08, 0.3 / len(panels)),
    )
    colours = (
        _make_colours(makes=frame.make.tolist())
        if colour_by == "make"
        else {
            car: PALETTE[index % len(PALETTE)]
            for index, car in enumerate(frame.car.unique())
        }
    )
    shown = set()
    reference = f"{frame.reference_make.iloc[0]} {frame.reference_model.iloc[0]}"
    for index, panel in enumerate(panels, start=1):
        group = frame.loc[frame.panel.eq(panel)]
        limit = max(
            1.0, float(max(group.value.max(), group.reference_rate.max())) * 1.1
        )
        for label, rows in group.groupby(
            by="make" if colour_by == "make" else "car", sort=False
        ):
            figure.add_trace(
                go.Scatter(
                    x=rows.reference_rate,
                    y=rows.value,
                    mode="markers",
                    name=label,
                    legendgroup=label,
                    showlegend=label not in shown,
                    marker={"color": colours[label], "size": 10},
                    customdata=rows.loc[:, ["car", "age_band", "rate_gap"]].to_numpy(),
                    hovertemplate="%{customdata[0]} · age %{customdata[1]}<br>Reference rate: %{x:g}<br>Compared rate: %{y:g}<br>Gap: %{customdata[2]:.2f}<extra></extra>",
                ),
                row=index,
                col=1,
            )
            shown.add(label)
        figure.add_trace(
            go.Scatter(
                x=[0, limit],
                y=[0, limit],
                mode="lines",
                line={"color": "#8596a6", "dash": "dash"},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=index,
            col=1,
        )
        figure.update_xaxes(
            range=[0, limit],
            title_text=f"{reference}: published rate",
            row=index,
            col=1,
        )
        figure.update_yaxes(
            range=[0, limit],
            title_text="Compared model: published rate",
            row=index,
            col=1,
        )
    return style_figure(
        figure=figure,
        title="Matched cohorts: below the diagonal favours the compared model",
        height=max(460, len(panels) * 300),
    )


def source_rate_chart(
    *,
    evidence: pd.DataFrame,
    top_n: int = 24,
    sort_by: str = "point_rate",
    ascending: bool = True,
    colour_by: str = "make",
) -> go.Figure:
    """Plot one source/report's age-panel rates with available count intervals.

    Args:
        evidence: Canonical rows from one source/report/observation-year/unit.
        top_n: Maximum rows, chosen after the requested within-panel sort.
        sort_by: Point rate, sample size, age, make or original input order.
        ascending: Direction of the within-panel ordering.
        colour_by: Make colours or one uniform colour.

    Returns:
        Horizontal rate points. Solid points have actual-count Wilson intervals;
        open points have published rates only, with no inferred error bars.

    Raises:
        ValueError: For invalid options or mixed source/outcome contexts.
    """
    if (
        top_n < 1
        or sort_by not in {"point_rate", "sample_size", "age_min", "make", "original"}
        or colour_by not in {"make", "uniform"}
    ):
        raise ValueError(
            "Choose a positive row limit and recognised rate-chart settings."
        )
    frame = source_rate_points(evidence=evidence)
    if any(
        frame[column].nunique() > 1
        for column in ["source_id", "report_year", "observation_year", "unit", "metric"]
    ):
        raise ValueError(
            "Rate panels must keep source, report, observation year and outcome separate."
        )
    if frame.empty:
        return style_figure(
            figure=go.Figure(), title="No published rates match these filters"
        )
    if sort_by != "original":
        frame = frame.sort_values(
            by=sort_by, ascending=ascending, kind="stable", na_position="last"
        )
    frame = frame.head(n=top_n)
    frame["label"] = [
        f"{row.make} {row.model} · age {row.age_min:g}–{row.age_max:g}"
        for row in frame.itertuples()
    ]
    frame["has_interval"] = frame.interval_lower.notna()
    colours = _make_colours(makes=frame.make.tolist())
    figure = go.Figure()
    for (make, known), rows in frame.groupby(by=["make", "has_interval"], sort=False):
        details = [
            f"Published rate: {row.value:g}<br>Point basis: {row.point_basis}"
            + (
                f"<br>Failed tests: {int(row.event_count):,} / {int(row.sample_size):,}<br>95% Wilson: {row.interval_lower:.2f}–{row.interval_upper:.2f}%"
                if known
                else f"<br>{row.interval_status}"
            )
            + f"<br>Observation year: {row.observation_year}<br>Registration: {row.cohort_year_start}–{row.cohort_year_end}"
            for row in rows.itertuples()
        ]
        interval = (
            {
                "type": "data",
                "symmetric": False,
                "array": rows.interval_upper - rows.point_rate,
                "arrayminus": rows.point_rate - rows.interval_lower,
            }
            if known
            else None
        )
        figure.add_trace(
            go.Scatter(
                x=rows.point_rate,
                y=rows.label,
                mode="markers",
                name=f"{make} · {'95% Wilson' if known else 'rate only'}",
                marker={
                    "color": colours[make] if colour_by == "make" else PALETTE[0],
                    "size": 9,
                    "symbol": "circle" if known else "circle-open",
                },
                error_x=interval,
                customdata=details,
                hovertemplate="%{y}<br>Plotted rate: %{x:.2f}<br>%{customdata}<extra></extra>",
            )
        )
    row = frame.iloc[0]
    unit = (
        "Inspection outcome (%)"
        if row.unit == "percent"
        else "Breakdown events per 1,000 registered vehicles"
    )
    figure.update_xaxes(title_text=unit, rangemode="tozero")
    figure.update_yaxes(
        type="category",
        autorange="reversed",
        categoryorder="array",
        categoryarray=frame.label.tolist(),
    )
    suffix = (
        "available 95% Wilson intervals"
        if frame.has_interval.any()
        else "published rates · interval unavailable"
    )
    return style_figure(
        figure=figure,
        title=f"{SOURCE_NAMES.get(row.source_id, row.source_name)} · report {int(row.report_year)} · {suffix}",
        height=max(420, len(frame) * 27 + 110),
    )
