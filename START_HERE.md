# Best Cars 3.3 — start here

Follow MAC_INSTALL.md to preview and apply the cumulative overlay on your Mac.
Set CAR_UPDATE_DIR to Downloads/best_cars_upgrade_20261014_v33. Reinstall from
your GitHub checkout; check version 3.3.0 and its import path before launch.

Version 3.3 fixes the chart issues reviewed in your screenshots. Single-source
disagreement is unavailable rather than zero, and weighting sensitivity requires
multiple sources. Choose one identical-coverage comparison group before plotting;
the rank matrix keeps tied models visible and the alternative trajectories retain
model labels even with make colours. Display filters do not recompute ranks.

Common make colours are curated and consistent across views. Ranking bars use
horizontal values outside the bars. Published rates use larger outlined points,
numeric values and explicit outcome/unit axes. Wilson plots retain same-age
panels and show readable values and interval endpoints. Make highlighting mutes
other makes without removing competitors or changing values.

Horizontal distributions align separate published ages within each source and
report context. Quartiles, means and actual 1.5-IQR whisker endpoints are computed
explicitly; outlier tooltips show human labels and unavailable sample sizes.
Pooling ages is an explicit option with an interpretation warning. Chart titles
and subplot spacing have been visually checked on rendered real-data charts.
See overlay/docs/VISUAL_REVIEW_V33.md for the audit and interpretation guide.

Earlier version 3.2 adds age-separated horizontal point plots for TÜV, ADAC and other
sources, with original units and explicit interval availability. Open points
retain published rates without inventing model sample sizes; solid inspection
points use actual counts and 95% Wilson intervals. Expand the interval-method
help in Deeper analysis or Compare cars → Rates and intervals for original
source fields, the formula, assumptions and a worked example from loaded counts.
See overlay/docs/INTERVALS_AND_RATE_PLOTS.md.

Keep using the existing Data folder shown in Data management after updating code,
especially if you have imported MOT records or refreshed sources. The bundled
snapshot is unchanged and contains no UK MOT data. MAC_INSTALL.md now makes this
separation explicit. Code updates do not replace your external dataset.

Earlier version 3.1 expanded Compare cars with six tabs: age profiles, same-age peers,
matched model rates against a selected reference, rates and available intervals,
evidence coverage and historical population summaries. Select one model for a
focused profile, or up to twenty. Model/make colours, source/age filters and
interactive chart downloads remain available. Earlier reports can be included
in the historical tab while other views use the latest loaded report.

Wilson plots in Compare cars and Deeper analysis now use separate exact-age,
source/report panels by default. Optional two-year groups and age-panel selection
allow nearby ages to be explored without pooling counts or intervals. Limits
and sorting apply independently within every age panel. The offline HTML also
uses separate age panels and includes illustrative T-Roc/Corolla/C-HR comparisons.
See overlay/docs/COMPARISON_VIEWS.md for instructions and interpretation.

Separate source evidence and exact published cohorts are primary. The optional
combined view offers an exploratory broad ranking, with missing sources, age
coverage, available sample sizes and missing-evidence score/rank sensitivity.
Missing source percentiles explicitly assume 50; bounds substitute 0 and 100.
These are assumptions, not estimates or confidence intervals. Different age and
model-generation mixes still limit comparisons. Strict mode requires identical
recorded coverage and ranks only within a matching comparison block.

Inspection sources combine within one outcome before inspection and breakdown
outcomes receive equal default weights. Weights express user preferences; no
source is universally trusted and Toyota receives no special treatment.

The new Repair costs page provides dated generic UK job prices and an editable
quote/job budget, not a prediction of annual car repair bills. The dedicated
source-definition page and page help explain faults, denominators and limitations.
See overlay/docs/FAIR_COMPARISONS.md for the methodological audit.

All earlier chart controls, heatmap fixes, EV/hybrid pages, offline imports and
native PDF isolation fixes are included. The real snapshot is unchanged: 8,924
observations plus saved exact US 2020 Corolla safety records. No lifespan or
traction-battery-health estimates are fabricated. PDF dependencies are optional.

The HTML report is offline and searchable. TSV exports use ages 2–15, minimum
known sample 100; optional rankings require 2 sources and 3 observations. Missing
sample sizes remain unknown. Cohort rows are not unique vehicles. The app allows
filters and eligibility thresholds to change. Scenario exports use strict common
competitors and are not a sensitivity distribution for the broad ranking. New
comparison exports illustrate T-Roc/Corolla/C-HR under those report filters;
registration time profiles exclude approximate TÜV bounds.

Your extra Fiesta files and other unlisted files are preserved. Listed replaced
files receive backups. No commit or push is performed. changes.patch is only for
a clean original checkout; use the overlay updater for your current checkout.
