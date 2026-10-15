"""Interactive graphics with explicit scales and source-aware labels."""

import hashlib
import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from best_cars.schema import validate_evidence
from best_cars.weighting import PRESETS

LOGGER = logging.getLogger(__name__)
PALETTE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#B38D00",
    "#6A3D9A",
    "#56B4E9",
    "#8F5C38",
    "#C73542",
    "#4A7F2B",
    "#008A9B",
    "#B24785",
    "#59666F",
    "#E17C05",
    "#665191",
    "#1B7837",
    "#A55B00",
    "#8C3C67",
    "#3F4C8A",
    "#827717",
]
# Explicit assignments keep common makes distinct and stable between views.
# Direct labels and highlighting remain necessary for large categorical sets.
MAKE_COLOURS = {
    "volkswagen": "#0072B2",
    "toyota": "#D55E00",
    "honda": "#009E73",
    "suzuki": "#CC79A7",
    "renault": "#B38D00",
    "audi": "#6A3D9A",
    "mazda": "#56B4E9",
    "porsche": "#8F5C38",
    "seat": "#C73542",
    "skoda": "#4A7F2B",
    "volvo": "#006166",
    "hyundai": "#B24785",
    "mercedes-benz": "#59666F",
    "kia": "#E17C05",
    "opel": "#665191",
    "subaru": "#1B7837",
    "dacia": "#A55B00",
    "peugeot": "#8C3C67",
    "ford": "#526C2A",
    "mitsubishi": "#827717",
    "lexus": "#303030",
    "bmw": "#2B5C4A",
    "mini": "#B34B00",
    "tesla": "#AD1457",
    "nissan": "#657800",
    "fiat": "#9A6324",
    "citroën": "#9B51A6",
    "chevrolet": "#305F91",
    "jeep": "#506835",
    "land rover": "#464646",
    "smart": "#008C95",
    "jaguar": "#7E435E",
    "saab": "#796A35",
}
SOURCE_NAMES = {
    "adac": "ADAC breakdowns",
    "traficom": "Traficom inspections",
    "tuv": "TÜV defects",
    "mot": "UK MOT",
}


def _rate_axis_title(*, source_id: str, unit: str) -> str:
    """Describe the source outcome, its denominator and favourable direction.

    Args:
        source_id: Canonical source identifier.
        unit: Canonical rate unit.

    Returns:
        Readable axis label without equating different source outcomes.
    """
    if unit == "per_1000_vehicles":
        return "Breakdowns per 1,000 registered vehicles · lower is better"
    if source_id == "tuv":
        return "Significant-defect rate (%) · lower is better"
    return "Failed inspections (%) · lower is better"


def _make_colours(
    *, makes: list[str], highlight_makes: list[str] | None = None
) -> dict[str, str]:
    """Return deterministic make colours that survive filtering and sorting.

    Args:
        makes: Manufacturer display names.
        highlight_makes: Optional makes to emphasise; other makes become grey.

    Returns:
        Curated stable colours, with deterministic fallbacks for unlisted makes.

    Raises:
        ValueError: If a make is not a non-empty string.
    """
    labels = makes + list(highlight_makes or [])
    if any(not isinstance(make, str) or not make.strip() for make in labels):
        raise ValueError("Make labels must be non-empty strings.")
    highlighted = {make.strip().casefold() for make in highlight_makes or []}
    colours = {}
    for make in sorted(set(makes)):
        key = make.strip().casefold().replace("škoda", "skoda")
        fallback = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
        colours[make] = MAKE_COLOURS.get(key, PALETTE[fallback % len(PALETTE)])
        if highlighted and make.strip().casefold() not in highlighted:
            colours[make] = "#CCD2D8"
    LOGGER.debug("Assigned categorical colours to %d makes", len(colours))
    return colours


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
        title={
            "text": title,
            "x": 0.02,
            "xanchor": "left",
            "y": 1,
            "yanchor": "top",
            "pad": {"t": 14},
            "font": {"size": 18},
        },
        template="plotly_white",
        height=height,
        font={"family": "Arial, sans-serif", "size": 15},
        margin={"l": 20, "r": 20, "t": 75, "b": 45},
        legend={"title": {"text": ""}},
        hoverlabel={"namelength": -1, "font": {"size": 13}},
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    figure.update_xaxes(automargin=True, gridcolor="#E8EDF1", zerolinecolor="#BECAD3")
    figure.update_yaxes(automargin=True, tickfont={"size": 12}, gridcolor="#F0F3F5")
    return figure


def ranking_chart(
    *,
    ranking: pd.DataFrame,
    top_n: int = 15,
    colour_by: str = "sources",
    highlight_makes: list[str] | None = None,
) -> go.Figure:
    """Show the highest relative evidence scores alongside source breadth.

    Args:
        ranking: Validated ranking table.
        top_n: Maximum models shown.
        colour_by: Evidence source count or manufacturer.
        highlight_makes: Keep these makes coloured and mute other bars.

    Returns:
        Interactive bar chart; a score is not a failure probability.

    Raises:
        ValueError: If top_n is non-positive.
    """
    if top_n < 1 or colour_by not in {"sources", "make"}:
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
        color="source_count" if colour_by == "sources" else "make",
        color_discrete_map=(
            {
                "1 sources": "#8596a6",
                "2 sources": "#70aca8",
                "3 sources": "#007f87",
                "4 sources": "#2856a8",
            }
            if colour_by == "sources"
            else _make_colours(
                makes=frame.make.tolist(), highlight_makes=highlight_makes
            )
        ),
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
    figure.update_traces(
        texttemplate="%{x:.1f}",
        textposition="outside",
        textangle=0,
        cliponaxis=False,
        textfont={"size": 12, "color": "#243746"},
    )
    figure.update_layout(
        uniformtext={"minsize": 11, "mode": "show"},
        legend={"traceorder": "normal", "font": {"size": 12}},
    )
    figure.update_yaxes(
        categoryorder="array", categoryarray=frame.car.iloc[::-1].tolist()
    )
    return style_figure(
        figure=figure,
        title=ranking.attrs.get(
            "ranking_label",
            "Optional preference comparison · same evidence coverage required",
        ),
        height=max(400, len(frame) * 27 + 100),
    )


def age_chart(
    *,
    scored: pd.DataFrame,
    model_ids: list[str],
    relative: bool = False,
    colour_by: str = "model",
) -> go.Figure:
    """Plot observed age profiles separately for each source and report.

    Args:
        scored: Selected canonical evidence.
        model_ids: Explicit model IDs to compare.
        relative: Show cohort percentiles instead of different raw rates.
        colour_by: Separate model colours or stable manufacturer colours.

    Returns:
        Faceted line/marker chart. Points are cross-sectional, not tracked cars.
    """
    if colour_by not in {"model", "make"}:
        raise ValueError("Age-profile colours must use model or make.")
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
        color="car" if colour_by == "model" else "make",
        line_group="car",
        line_dash="report",
        facet_row="source_panel",
        markers=True,
        color_discrete_map=colour_map
        if colour_by == "model"
        else _make_colours(makes=frame.make.tolist()),
        hover_name="car",
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


def scenario_rank_chart(
    *,
    scenarios: pd.DataFrame,
    model_ids: list[str],
    colour_by: str = "model",
    size_by_score: bool = False,
    view: str = "lines",
) -> go.Figure:
    """Plot scenario rank movement without joining different models of a make.

    Args:
        scenarios: Long-form scenario ranks and current scenario scores.
        model_ids: Explicit model identities to display; ranks remain untouched.
        colour_by: Model or stable manufacturer colours.
        size_by_score: Marker diameters increase with each scenario's score.
        view: Separate trajectories or a labelled rank matrix that preserves ties.

    Returns:
        Ranks within one identical-coverage comparison group only.

    Raises:
        ValueError: For invalid columns, scores, duplicate cells or selection.
    """
    required = {
        "model_id",
        "make",
        "model",
        "scenario",
        "scenario_rank",
        "score",
        "cohorts",
        "sources",
        "comparison_group_id",
        "comparison_models",
    }
    if (
        colour_by not in {"model", "make"}
        or view not in {"lines", "matrix"}
        or not required.issubset(scenarios.columns)
        or len(model_ids) != len(set(model_ids))
    ):
        raise ValueError(
            "Require valid scenario data, colour mode and distinct models."
        )
    if (
        scenarios.duplicated(subset=["model_id", "scenario"]).any()
        or not scenarios.score.between(left=0, right=100).all()
        or not scenarios.scenario_rank.ge(1).all()
        or not scenarios.scenario.isin(PRESETS).all()
    ):
        raise ValueError("Scenario cells must be unique with valid scores and ranks.")
    if not set(model_ids).issubset(set(scenarios.model_id)):
        raise ValueError("Selected scenario model is unavailable.")
    frame = scenarios.loc[scenarios.model_id.isin(model_ids)].copy()
    if frame.empty:
        return style_figure(figure=go.Figure(), title="No scenario models selected")
    if (
        frame.comparison_group_id.isna().any()
        or frame.comparison_group_id.nunique() != 1
        or frame.sources.min() < 2
        or frame.comparison_models.nunique() != 1
        or frame.comparison_models.min() < 2
        or (frame.scenario_rank > frame.comparison_models).any()
        or not frame.groupby(by="model_id").scenario.nunique().eq(len(PRESETS)).all()
    ):
        raise ValueError(
            "Scenario ranks require one complete, comparable coverage group."
        )
    competitors = int(frame.comparison_models.iloc[0])
    frame["car"] = frame.make + " " + frame.model
    frame["scenario"] = pd.Categorical(
        values=frame.scenario, categories=list(PRESETS), ordered=True
    )
    frame = frame.sort_values(by=["model_id", "scenario"])
    if view == "matrix":
        ranks = frame.pivot(
            index="model_id", columns="scenario", values="scenario_rank"
        )
        scores = frame.pivot(index="model_id", columns="scenario", values="score")
        ranks = ranks.reindex(index=model_ids, columns=list(PRESETS))
        scores = scores.reindex(index=model_ids, columns=list(PRESETS))
        names = frame.drop_duplicates(subset="model_id").set_index(keys="model_id").car
        figure = go.Figure(
            data=go.Heatmap(
                z=ranks.to_numpy(dtype=float),
                x=list(PRESETS),
                y=names.reindex(index=model_ids).tolist(),
                customdata=scores.to_numpy(dtype=float),
                text=ranks.to_numpy(dtype=int),
                texttemplate="%{text}",
                colorscale=[[0, "#B8DCD9"], [0.5, "#F3F4F5"], [1, "#E3B7B2"]],
                zmin=1,
                zmax=competitors,
                colorbar={
                    "title": {"text": "Rank in<br>this group"},
                    "dtick": max(1, competitors // 8),
                },
                xgap=3,
                ygap=3,
                hovertemplate=(
                    "%{y}<br>%{x}<br>Rank: %{z:.0f} of "
                    + str(competitors)
                    + "<br>Preference score: %{customdata:.1f} / 100<extra></extra>"
                ),
            )
        )
        figure.update_yaxes(autorange="reversed", type="category")
        figure.update_xaxes(side="top", tickfont={"size": 12})
        return style_figure(
            figure=figure,
            title=f"Ranks under different priorities · {competitors} comparable models",
            height=max(380, len(model_ids) * 31 + 160),
        )
    figure = px.line(
        data_frame=frame,
        x="scenario",
        y="scenario_rank",
        color="car" if colour_by == "model" else "make",
        line_group="model_id",
        markers=True,
        hover_name="car",
        hover_data={
            "score": ":.1f",
            "cohorts": True,
            "sources": True,
            "model_id": False,
        },
        color_discrete_map=_make_colours(makes=frame.make.tolist())
        if colour_by == "make"
        else None,
        color_discrete_sequence=PALETTE,
        category_orders={"scenario": list(PRESETS)},
        labels={
            "scenario_rank": f"Rank within this group ({competitors} competitors)",
            "scenario": "Preference scenario",
            "car": "Model",
            "make": "Make",
            "cohorts": "Source-cohort rows",
            "sources": "Observed sources",
            "score": "Preference score",
        },
    )
    symbols = ["circle", "square", "diamond", "triangle-up", "cross", "triangle-down"]
    model_names = (
        frame.drop_duplicates(subset="model_id").set_index(keys="model_id").car
    )
    for index, trace in enumerate(figure.data):
        trace.name = model_names[trace.customdata[0][3]]
        trace.showlegend = True
        trace.marker.symbol = symbols[index % len(symbols)]
        trace.marker.size = [
            7 + float(row[0]) * 0.18 if size_by_score else 7 for row in trace.customdata
        ]
    figure.update_yaxes(autorange="reversed", dtick=1 if competitors <= 20 else None)
    return style_figure(
        figure=figure, title="Rank sensitivity to evidence priorities", height=560
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
        or order not in {"alphabetical", "both", "weakest", "strongest"}
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
    if order == "alphabetical":
        keys = (
            frame.sort_values(by=["make", "model"])
            .model_id.drop_duplicates()
            .head(n=max_models)
            .tolist()
        )
    elif order == "weakest":
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
            text=np.where(values.isna(), "—", ""),
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


def mileage_chart(
    *,
    evidence: pd.DataFrame,
    colour_by: str = "age",
    model_scores: pd.DataFrame | None = None,
    size_by_score: bool = False,
) -> go.Figure:
    """Compare odometer mileage and inspection failure rates within one source.

    Args:
        evidence: Selected rows from one source/report, with odometer metadata.
        colour_by: Observed age or manufacturer.
        model_scores: Current combined model ranking, with model_id and score.
        size_by_score: Use marker diameters of 7–25 pixels for known scores.
            Unscored models retain seven-pixel markers and an explicit status.

    Returns:
        Scatter plot showing published cohort rates, not individual vehicles.

    Raises:
        ValueError: If incompatible evidence sources/units are mixed.
    """
    if colour_by not in {"age", "make"}:
        raise ValueError("Mileage colours must use age or make.")
    if (
        evidence.source_id.nunique() > 1
        or evidence.unit.nunique() > 1
        or evidence.report_year.nunique() > 1
        or evidence.metric.nunique() > 1
    ):
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
    scores = pd.Series(dtype=float)
    if model_scores is not None:
        if (
            not {"model_id", "score"}.issubset(model_scores.columns)
            or model_scores.model_id.duplicated().any()
            or not model_scores.score.between(left=0, right=100).all()
        ):
            raise ValueError(
                "Marker scores require unique models and finite 0–100 scores."
            )
        scores = model_scores.set_index(keys="model_id").score
    if size_by_score and model_scores is None:
        raise ValueError("Score-sized markers require a current model ranking.")
    frame["model_score"] = frame.model_id.map(arg=scores)
    frame["score_status"] = np.where(
        frame.model_score.notna(),
        "Optional within-block preference score",
        "No optional preference score in this coverage block",
    )
    frame["marker_diameter"] = 7.0 + (
        frame.model_score.fillna(value=0) * 0.18 if size_by_score else 0
    )
    figure = px.scatter(
        data_frame=frame,
        x="mileage_km",
        y="value",
        color="age_mid" if colour_by == "age" else "make",
        size="marker_diameter",
        symbol="score_status" if size_by_score else None,
        symbol_map={
            "Optional within-block preference score": "circle",
            "No optional preference score in this coverage block": "circle-open",
        },
        hover_name="car",
        hover_data={
            "source_model": True,
            "sample_size": True,
            "report_year": True,
            "age_min": True,
            "age_max": True,
            "model_score": ":.1f",
            "score_status": True,
            "marker_diameter": False,
        },
        color_continuous_scale="Teal",
        color_discrete_map=_make_colours(makes=frame.make.tolist()),
        labels={
            "mileage_km": "Mean odometer reading (km)",
            "value": "Inspection failure/defect rate (%)",
            "age_mid": "Age (years)",
            "model_score": "Optional within-block preference score",
            "score_status": "Score coverage",
            "make": "Make",
        },
    )
    figure.update_traces(
        marker={
            "sizemode": "diameter",
            "sizeref": 1,
            "opacity": 0.75,
            "line": {"width": 0.4, "color": "#ffffff"},
        }
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


def rate_distribution_chart(
    *, evidence: pd.DataFrame, group_by_age: bool = True
) -> go.Figure:
    """Show horizontal distributions within source/report and published age.

    Args:
        evidence: Canonical observations for the explicit report selection.
        group_by_age: Keep published age bands separate; False deliberately pools ages.

    Returns:
        Box plots with separate units, median, quartiles, outliers and means.

    Raises:
        ValueError: If evidence is invalid or the grouping flag is not boolean.
    """
    if not isinstance(group_by_age, bool):
        raise ValueError("Age grouping must be a boolean choice.")
    if evidence.empty:
        return style_figure(figure=go.Figure(), title="No rate distributions available")
    frame = validate_evidence(frame=evidence)
    frame["car"] = frame.make + " " + frame.model
    keys = ["source_id", "report_year", "observation_year", "metric", "unit"]
    panels = list(frame.groupby(by=keys, sort=True, dropna=False))
    titles = [
        f"{SOURCE_NAMES.get(key[0], key[0])} · report {key[1]} · observations {key[2]}"
        + (" · per 1,000 vehicles" if key[4] == "per_1000_vehicles" else " · %")
        for key, _group in panels
    ]
    row_counts = [
        max(1, len(group.loc[:, ["age_min", "age_max"]].drop_duplicates()))
        if group_by_age
        else 1
        for _key, group in panels
    ]
    heights = [max(250, rows * 36 + 130) for rows in row_counts]
    figure = make_subplots(
        rows=len(panels),
        cols=1,
        shared_xaxes=False,
        row_heights=heights,
        subplot_titles=titles,
        vertical_spacing=min(0.12, 0.42 / len(panels)) if len(panels) > 1 else 0,
    )
    for panel_index, (key, panel) in enumerate(panels, start=1):
        groups = (
            list(panel.groupby(by=["age_min", "age_max"], sort=True))
            if group_by_age
            else [(None, panel)]
        )
        labels = []
        for ages, group in groups:
            age = (
                (
                    str(int(ages[0]))
                    if ages[0] == ages[1]
                    else f"{ages[0]:g}–{ages[1]:g}"
                )
                if ages
                else "All selected ages"
            )
            label = (
                f"Age {age} · {len(group):,} rows"
                if ages
                else f"{age} · {len(group):,} rows"
            )
            labels.append(label)
            q1, q3 = group.value.quantile(q=[0.25, 0.75])
            iqr = q3 - q1
            inside = group.value.between(left=q1 - 1.5 * iqr, right=q3 + 1.5 * iqr)
            inliers = group.loc[inside, "value"]
            figure.add_trace(
                go.Box(
                    q1=[q1],
                    median=[group.value.median()],
                    q3=[q3],
                    mean=[group.value.mean()],
                    lowerfence=[inliers.min()],
                    upperfence=[inliers.max()],
                    y=[label],
                    meta={"observations": len(group)},
                    orientation="h",
                    name=label,
                    boxmean=True,
                    boxpoints=False,
                    quartilemethod="linear",
                    line={"color": "#007F87", "width": 2},
                    fillcolor="rgba(0,127,135,0.18)",
                    showlegend=False,
                    alignmentgroup="rates",
                    offsetgroup="rates",
                ),
                row=panel_index,
                col=1,
            )
            # Separate outlier markers keep per-record metadata off box statistics.
            outliers = group.loc[~inside]
            if not outliers.empty:
                details = [
                    f"{row.car}<br>Age {row.age_min:g}–{row.age_max:g}"
                    + (
                        f"<br>Recorded tests: {int(row.sample_size):,}"
                        if pd.notna(row.sample_size)
                        else "<br>Sample size unavailable"
                    )
                    for row in outliers.itertuples()
                ]
                figure.add_trace(
                    go.Scatter(
                        x=outliers.value,
                        y=[label] * len(outliers),
                        mode="markers",
                        marker={"size": 7, "color": "#007F87"},
                        showlegend=False,
                        customdata=details,
                        hovertemplate="%{customdata}<br>Published rate: %{x:.2f}<extra></extra>",
                    ),
                    row=panel_index,
                    col=1,
                )
        figure.update_xaxes(
            row=panel_index,
            col=1,
            rangemode="tozero",
            title_text=_rate_axis_title(source_id=key[0], unit=key[4]),
            matches=None,
        )
        figure.update_yaxes(
            row=panel_index,
            col=1,
            title_text="",
            matches=None,
            type="category",
            categoryorder="array",
            categoryarray=labels,
            autorange="reversed",
        )
    figure.update_layout(boxmode="overlay", showlegend=False)
    for annotation in figure.layout.annotations:
        annotation.update(x=0, xanchor="left", font={"size": 13})
    return style_figure(
        figure=figure,
        title="Published rate distributions · "
        + ("separate age bands" if group_by_age else "selected ages pooled"),
        height=max(420, sum(heights) + 140),
    )


def count_uncertainty_chart(
    *,
    counted: pd.DataFrame,
    top_n: int = 24,
    sort_by: str = "sample_size",
    ascending: bool = False,
    colour_by: str = "uniform",
    highlight_makes: list[str] | None = None,
) -> go.Figure:
    """Show actual test-failure percentages and their sampling uncertainty.

    Args:
        counted: Output of counted_failure_rates, with known denominators.
        top_n: Maximum cohorts displayed, chosen by largest test sample.
        sort_by: Sample size, failure rate, age, make or original input order.
        ascending: Ascending sort; ignored for original input order.
        colour_by: Uniform markers or stable manufacturer colours.
        highlight_makes: Keep selected makes coloured without removing competitors.

    Returns:
        Count-based point estimates and 95% Wilson error bars. Ordering by
        sample size is not a reliability ranking or an age-adjusted comparison.

    Raises:
        ValueError: If top_n is non-positive.
    """
    if (
        top_n < 1
        or sort_by
        not in {"sample_size", "counted_rate_pct", "age_min", "make", "original"}
        or colour_by not in {"uniform", "make"}
    ):
        raise ValueError("Chart cohort count must be positive.")
    if counted.empty:
        return style_figure(figure=go.Figure(), title="No actual test counts available")
    frame = counted.copy()
    if sort_by != "original":
        frame = frame.sort_values(by=sort_by, ascending=ascending, kind="stable")
    frame = frame.head(n=top_n)
    same_context = all(
        frame[column].nunique(dropna=False) == 1
        for column in [
            "source_id",
            "report_year",
            "observation_year",
            "age_min",
            "age_max",
            "comparison_group",
        ]
    )
    frame["label"] = [
        f"{row.make} {row.model}"
        + (
            ""
            if same_context
            else f" · {SOURCE_NAMES.get(row.source_id, row.source_id)} · age {row.age_min:g}–{row.age_max:g} · report {row.report_year}"
        )
        for row in frame.itertuples()
    ]
    figure = go.Figure()
    colours = _make_colours(makes=frame.make.tolist(), highlight_makes=highlight_makes)
    groups = (
        list(frame.groupby(by="make", sort=False))
        if colour_by == "make"
        else [("Counted cohorts", frame)]
    )
    for make, group in groups:
        figure.add_trace(
            go.Scatter(
                x=group.counted_rate_pct,
                y=group.label,
                name=make,
                showlegend=colour_by == "make",
                mode="markers+text",
                text=[f"{value:.1f}%" for value in group.counted_rate_pct],
                textposition="top center",
                textfont={"size": 11, "color": "#243746"},
                cliponaxis=False,
                marker={
                    "color": colours[make] if colour_by == "make" else PALETTE[0],
                    "size": 10,
                },
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": group.wilson_upper_pct - group.counted_rate_pct,
                    "arrayminus": group.counted_rate_pct - group.wilson_lower_pct,
                    "thickness": 1.5,
                    "width": 3,
                },
                customdata=group.loc[
                    :,
                    [
                        "sample_size",
                        "event_count",
                        "value",
                        "wilson_lower_pct",
                        "wilson_upper_pct",
                        "report_year",
                        "observation_year",
                        "age_min",
                        "age_max",
                    ],
                ].to_numpy(),
                hovertemplate=(
                    "%{y}<br>Counted failure rate: %{x:.2f}%"
                    "<br>Tests: %{customdata[0]:,.0f}<br>Failed tests: %{customdata[1]:,.0f}"
                    "<br>Published rate: %{customdata[2]:.2f}%"
                    "<br>95% Wilson: %{customdata[3]:.2f}–%{customdata[4]:.2f}%"
                    "<br>Report: %{customdata[5]:.0f} · observations: %{customdata[6]:.0f}"
                    "<br>Age: %{customdata[7]:.0f}–%{customdata[8]:.0f}<extra></extra>"
                ),
            )
        )
    figure.update_xaxes(
        title_text="Counted inspection failures (%) · lower is better",
        range=[0, max(0.2, float(frame.wilson_upper_pct.max()) * 1.08)],
    )
    figure.update_yaxes(
        autorange="reversed",
        type="category",
        categoryorder="array",
        categoryarray=frame.label.tolist(),
    )
    context = "Counted cohorts"
    if same_context:
        row = frame.iloc[0]
        age = (
            str(int(row.age_min))
            if row.age_min == row.age_max
            else f"{row.age_min:g}–{row.age_max:g}"
        )
        context = f"{SOURCE_NAMES.get(row.source_id, row.source_id)} · age {age} · report {int(row.report_year)}"
    return style_figure(
        figure=figure,
        title=f"{context} · rates and 95% Wilson intervals",
        height=max(420, len(frame) * 32 + 130),
    )


def score_spread_chart(
    *, ranking: pd.DataFrame, colour_by: str = "sources", size_by_score: bool = False
) -> go.Figure:
    """Compare composite evidence scores with the spread between source means.

    Args:
        ranking: Ranked ungrouped models with source breadth and disagreement.
        colour_by: Source breadth or manufacturer.
        size_by_score: Larger markers indicate higher combined scores.

    Returns:
        An interactive model-series scatter plot; spread is not uncertainty.
    """
    if colour_by not in {"sources", "make"}:
        raise ValueError("Scatter colours must use source breadth or make.")
    if ranking.empty:
        return style_figure(figure=go.Figure(), title="No ranked models available")
    frame = ranking.loc[
        ranking.sources.ge(2) & ranking.score.notna() & ranking.score_spread.notna()
    ].copy()
    if frame.empty:
        LOGGER.info("Between-source spread unavailable for single-source models")
        figure = go.Figure()
        figure.add_annotation(
            text="At least two observed sources per model are required.<br>One source cannot establish agreement.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        return style_figure(
            figure=figure, title="Between-source disagreement unavailable", height=250
        )
    frame["car"] = frame.make + " " + frame.model
    frame["source_count"] = frame.sources.astype(str)
    frame["marker_diameter"] = 7.0 + (frame.score * 0.18 if size_by_score else 0)
    figure = px.scatter(
        data_frame=frame,
        x="score",
        y="score_spread",
        color="source_count" if colour_by == "sources" else "make",
        size="marker_diameter",
        hover_name="car",
        hover_data=["cohorts", "worst_cohort_score", "weight_coverage", "source_ids"],
        color_discrete_sequence=PALETTE,
        color_discrete_map=_make_colours(makes=frame.make.tolist())
        if colour_by == "make"
        else None,
        labels={
            "score": "Relative evidence score (0–100)",
            "score_spread": "Range of source means (percentile points)",
            "source_count": "Scored sources",
        },
    )
    figure.update_traces(marker={"sizemode": "diameter", "sizeref": 1, "opacity": 0.8})
    figure.update_xaxes(range=[-2, 102])
    ceiling = float(frame.score_spread.max())
    figure.update_yaxes(
        range=[-max(1, ceiling * 0.04), min(102, max(5, ceiling * 1.10))]
    )
    return style_figure(
        figure=figure, title="Strong scores and disagreement between sources"
    )
