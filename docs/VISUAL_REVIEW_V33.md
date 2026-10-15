# Chart interpretation and readability — version 3.3

This revision addresses the six screenshots reviewed on 14 October 2026.
It updates graphics and diagnostic eligibility; no source data were refreshed
and no manufacturer receives a special reliability adjustment.

## Between-source disagreement

A model must have at least two observed sources and a usable preference score
to enter the disagreement scatter. One-source profiles are excluded, not drawn
at zero. Their disagreement is unavailable. The page reports exclusions and
explains why a single-source selection cannot support this analysis.

The vertical axis follows the observed spread rather than leaving most of a
fixed 0–100 panel empty. Spread remains a descriptive range of source mean
percentiles, not a confidence interval or a matched-age agreement measure.

## Weighting sensitivity

With only one selected source, the page explains why the preset comparison is
unavailable. With multiple sources, choose one identical-coverage group before
plotting. The selector displays its competitor population; the expandable table
shows exact sources, report/observation years, ages and registration cohorts.
Groups cannot be combined on one rank axis. The chart API rejects mixed groups.

The default rank matrix gives each model a separate row and each preset a
column. Tied ranks remain tied, without jitter or arbitrary tie-breaking. Cell
colours encode rank within this group, not make, confidence or failure risk.
The alternative trajectory view offers model/make colours, marker shapes and
optional score-sized points. Model/make filters change displayed rows without
recomputing their ranks against a smaller population. Switching comparison
groups resets incompatible model filters and the row limit.

If none of the ranks change, the page states that this applies only to the
three tested preference presets. It does not establish general robustness.

## Make colours and highlighting

Common makes have explicit categorical colour assignments shared across the
app and report. Assignments survive filtering and sorting. Other makes receive
deterministic palette fallbacks. Large categorical sets cannot all be made
equally distinguishable; direct model labels and search remain essential.

Find reliable cars, the published-rate panels and the Deeper analysis Wilson
plots support make highlighting. Selected makes stay coloured; others become
grey. Highlighting retains competitors and leaves all values and ranks intact.
Colours identify makes and do not encode better or worse reliability.

Ranking values use fixed horizontal labels outside bars, including integer-like
values previously rotated vertically. Relative scores are index values, not
probabilities: 99.6 is not a 99.6% probability of a reliable car.

## Published rates and Wilson panels

Published rates have larger outlined points, readable numeric labels, shorter
model labels within a single age context and source-specific axes stating that
lower rates are favourable. Hollow points retain unavailable intervals; thicker
outlines do not imply more statistical certainty. Model denominators remain
unknown unless actual counts were imported.

Wilson charts retain separate exact age/source/report panels by default. Point
labels show counted percentages; hover shows failures, tests, published rates,
interval endpoints and observation/report years. Counts and intervals are not
pooled. Short labels are used only when the panel context is homogeneous.

## Rate distributions

Horizontal boxes align ages within separate source, report, observation-year,
metric and unit contexts. Published age bands are separate by default. The
explicit pooling option warns that an age mix affects the distribution.

Every published model-cohort row receives equal weight. The row count is not a
test count or a number of unique cars. The box spans the linear 25th and 75th
percentiles, the central line is the median, and the dotted marker is the mean.
Whiskers reach the most extreme recorded observations within 1.5 interquartile
ranges. Outlier markers have readable model/age metadata and show “Sample size
unavailable” rather than NaN. Box-summary hover does not attach one model's
metadata to the distribution as a whole.

These plots describe the selected evidence. They do not provide a manufacturer
league table, national fleet failure rate, lifetime or repair-cost estimate.
