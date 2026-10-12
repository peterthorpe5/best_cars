# Reliability comparison methods

Each row represents a source/model-series/cohort observation, not an individual car. Inspection percentages and breakdowns per 1,000 vehicles retain their original units. Unknown denominators stay missing; safety events stay separate. Canonical keys, finite numerical data, count fields and ordered age/cohort ranges are validated before saving.

## Cohort percentiles and source weights

For N source-cohort peers and a lower-is-better average rank r:

`relative_score = 100 × (N − r) / (N − 1)`.

At least five peers are required. Ties use average ranks; an all-equal cohort receives 50. Grouped source series can occur among raw peers, but are excluded from cross-source model rankings. Comparisons span available model classes; segment, engine, generation, mileage, servicing and driver behaviour are not controlled.

First calculate one mean cohort score per model per source, then combine available source means using explicit weights. Missing source evidence is not zero. `weight_coverage` is completeness, not confidence. Zero-weight sources do not enter ranks/source counts. Original peer scores are computed before display filters; latest reports are selected independently by source.

Source cohorts can differ. `cohorts` counts source-cohort observations, `score_spread` is the range of source means, and `worst_cohort_score` is the lowest observed percentile. These are descriptive breadth/disagreement measures. Inspection outcomes can be correlated; equal weights do not establish independent meta-analytic evidence. The composite is exploratory, not a calibrated failure probability.

## Counts and uncertainty

Where actual failed-test counts k and test counts n exist, the comparison view gives a two-sided 95% Wilson interval for k/n. Repeated vehicles and systematic usage/maintenance/selection differences are not included in this binomial interval. Rounded published rates can differ slightly from 100 × k/n. No intervals are invented for unpublished TÜV/ADAC model denominators.

## Time, age and mileage

ADAC records an observation year and first-registration year; Traficom records an inspection and registration year. Age is their calendar-year difference. TÜV gives age bins whose registration-year bounds are approximate; they are not confirmed production dates.

An age profile connects different registration cohorts and can cross model changes. It does not follow the same cars through time or predict ageing causally. Missing cohorts are absent, never zero.

Mileage plots use mean inspection odometer readings. ADAC's annual mileage for breakdown vehicles is a different measurement and is excluded. Dividing a failure percentage by mean odometer mileage is not a statistically justified exposure adjustment; the old ratio is absent from new rankings.

## Model identity and safety

Aliases are explicit label translations: VW/Volkswagen, B-Klasse/B-Class, BMW 3er-Reihe/3-sarja, accents/case and harmless punctuation. Slash/comma/multiple-series/unspecified groups stay grouped. Opel/Vauxhall, engines, fuel types and generations are not silently merged. Original labels remain visible; no fuzzy merge is performed.

NHTSA exact US make/model/year results are safety context with no comparable fleet denominator. Recall/complaint counts never enter the score. Empty lookups mean no records returned, not proof of no defects. UK manufacturer/DVSA scope must be checked for the actual vehicle.

## Expanded HTML descriptive statistics

Source/report/observation-year/metric/unit groups retain separate scales.
Means, medians and percentiles give each selected model-cohort observation one
equal weight; they are not population-weighted national rates. Pooled failure
percentages sum actual failures and tests only over rows with both counts.
Count coverage and that subset's size are shown; no unknown denominator is
filled. Intervals use count-derived rates alongside the original published
rate, avoiding rounded-rate centres outside their actual-count intervals.

Pairwise source agreement uses only model series observed in both sources.
Spearman rho is Pearson correlation of average ranks, including ties; at least
five pairs and variation in both sources are required. Missing cells are not
imputed. Paired cohorts are not necessarily matched, so the results are
descriptive association, with sample sizes and median absolute percentile gap;
no p-values or causal agreement claim are reported. Source-score spread is
disagreement between means, not a sampling confidence interval.

The counts view displays the largest observed test cohorts, ordered by sample
size rather than reliability. See [LONGEVITY.md](LONGEVITY.md) for the separate
history-retention and survival-analysis requirements. No life expectancy is
derived from the existing cross-sectional evidence.
