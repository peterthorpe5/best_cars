Historical version 2.x documentation. Current comparison methods are in FAIR_COMPARISONS.md; old global ranks and source-weight defaults are not current app results.

# Chart controls and screenshot corrections — version 2.4

## Where to find the mileage figure

The HTML screenshot at 15.51.00 is the mileage-versus-inspection scatter.
The three 16.02 screenshots show ADAC selected on Age and mileage. ADAC has
no comparable mean-odometer field, so that chart is unavailable for ADAC.
Choosing Traficom or TÜV reveals it. Version 2.4 adds a dedicated **Mileage
explorer** page, defaulting to Traficom where it is available.

The new page supports source/report, make, model search and observed-age
filters; age or make colours; and optional point sizing by the **current
combined model score**. The sidebar filters apply before page filters. Model
scores use current sidebar weights and eligibility thresholds; page display
filters do not recalculate ranks or percentiles. Larger diameters mean higher
scores, not a measured failure probability. Diameters run from 7 to 25 pixels.
Unscored models are retained with seven-pixel markers and an explicit hover
status; zero scores also have small markers and remain actual zeroes.

## Heatmap bug

The 15.52.17 screenshot shows dashes crossing numeric scores. This was a real
rendering bug: the grey missing-cell trace had dash text at every matrix cell,
even where its colour value was missing. Some Plotly renderers still draw such
text. Dash text is now empty at every observed cell, with a regression test
that checks both text and colour masks. Actual score/rate values are unchanged.

## Other controls

| Visual | Controls |
| --- | --- |
| Find reliable cars | Any top-N up to the number of matching ranked models; colour by evidence breadth or make |
| Wilson intervals | Source, make, explicit models, model search, age range, cohort limit, original order or sorting by tests/rate/age/make, ascending direction, make colours |
| Weighting sensitivity | Any model count up to eligible models; make/model selections; model or make colours; point diameters by scenario-specific score |
| Source disagreement scatter | Source breadth or make colours; optional diameters by combined score |
| Compare cars | Up to twenty models; model or make colours |
| Age profiles | Up to thirty models; model or make colours |
| Electrified ranking/profiles | Adjustable top-N ranking and manufacturer colouring |

Wilson points represent counted cohorts, not unique cars: a model may have
several ages/reports. Filtering, colouring and ordering preserve failure counts
and Wilson endpoints. Sorting rates across unmatched ages does not create an
age-adjusted ranking. Age filters retain bands overlapping the requested range.

Make colours are deterministic across chart filters. Similar hues can occur in
large manufacturer sets; the legend and hover labels remain authoritative.
When models share a make colour, their lines remain separate model trajectories.
The weighting ranks retain the full common eligible competitor universe before
display filtering. Marker size in this chart uses each scenario's own score.

Very large selections can make legends and plots dense; use make/model filters,
legend toggles and downloadable charts/tables to focus the inspection.
Heatmap colours continue to represent scores or rates; manufacturer colouring
is deliberately reserved for charts where it preserves that meaning.
# Version 3.3 additions

Make colours now use curated assignments shared across views. Ranking bars,
published-rate panels and Deeper analysis Wilson plots support highlighting
without removing competitors. Labels remain horizontal and rates are printed
beside points. See [VISUAL_REVIEW_V33.md](VISUAL_REVIEW_V33.md) for interpretation.

Weighting sensitivity selects one comparison group, defaults to a rank matrix,
and is unavailable for single-source selections. Between-source disagreement
omits single-source models. Horizontal rate distributions separate published
ages by default; pooling ages requires an explicit checkbox change.
