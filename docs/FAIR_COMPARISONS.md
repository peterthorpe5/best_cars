# Fair comparisons — version 3.0

The old composite was an average of whichever source scores each car happened to
have. A two-source inspection-only model and a three-source model including roadside
breakdowns therefore answered different questions. Age/generation coverage also
differed. This could create a misleading universal reliability league table.

## Two useful comparisons

Separate source evidence is the primary view. Percentiles are calculated within
published source/report/age or registration cohorts, using the original peer group.
A high percentile means a low rate relative to those peers, not low failure risk.

The exploratory combined comparison retains a broad list while exposing gaps.
Inspection sources combine within an inspection outcome; inspection and roadside
breakdowns receive equal default outcome weights. Source and outcome weights are
editable preferences, not scientific estimates of source trustworthiness.

Missing sources assume percentile 50 for the displayed exploratory ordering.
This is an explicit neutral reference assumption, not an estimate of the car's
unobserved performance. Bounds substitute 0 and 100 for every missing source.
Possible rank bounds compare these score bounds across the same eligible models.
They describe only missing-source sensitivity, not statistical confidence, age-mix
uncertainty or a probability that one car is better. Narrow bounds do not certify a
rank: completely observed cars can still have mismatched ages and generations.

The strict option requires all active sources and identical recorded cohort
coverage. It compares only within that block and suppresses singleton scores/ranks.
It is a useful check but cannot rank every model against every other model.

## Coverage and sample sizes

Tables retain source IDs, missing source IDs, source/cohort row counts, youngest and
oldest recorded ages, rows with known sample sizes and minimum published sample.
A source/cohort row is not a vehicle count. Samples may repeat vehicles across
years and sources. Unknown samples are missing, not zero. Minimum sample size is
context, not a pooled exposure denominator. Matching recorded age bounds does not
prove that engine variants, generations, countries or driving patterns match.

## Corolla

Corolla remains visible in all source profiles and in exploratory comparisons.
Traficom inspection evidence, TÜV significant-defect percentages and ADAC attended
breakdowns disagree. The app does not tune weights to reputation or owner experience.
The ADAC 2025 fault list identifies starter-battery problems for Corolla registration
cohorts 2019–2022; this does not establish engine durability or a model repair budget.
Older and newer Corolla cohorts should be investigated separately.

## What counts as a fault

- TÜV: significant roadworthiness inspection defects; percent of inspected vehicles.
  Mileage is descriptive odometer context, not the denominator of the defect rate.
- Traficom: failed inspections / tests in the published Finnish inspection cohort.
- ADAC: attended roadside breakdowns per 1,000 registered vehicles, not per kilometre.
- UK MOT: supported initial-test imports use failure and pass-after-rectification
  semantics; no national vehicle-history database is bundled.
- NHTSA: US recalls and voluntary complaints are safety context, excluded from ranks.

Repair costs remain separate. Dated generic marketplace job averages can seed an
editable quote/job scenario; quantities default to zero. No fault frequency is
inferred from inspection percentages, and missing selected prices prevent a total.

## Primary references

- [ADAC 2025 report](https://assets.adac.de/image/upload/v1744788916/ADAC-eV/KOR/Text/PDF/Pannenstatistik_2025_final_nrmvls.pdf)
- [Traficom inspection definitions](https://tieto.traficom.fi/en/statistics/statistics-inspections)
- [TÜV 2025 inspection window](https://www.tuev-verband.de/pressemitteilungen/tuev-report-2025-elektroautos)
- [TÜV defect categories](https://www.tuv.com/germany/de/lp/lp-haupt-und-abgasuntersuchung/distribution-pages/m%C3%A4ngelklassen/)
- [FixMyCar price context](https://www.whocanfixmycar.com/car-repair)

Earlier weighting and screenshot documents are historical records, superseded by
this methodology. The data snapshot remains 8,924 real observations; no life
expectancy, battery-health or model-specific annual repair frequencies are invented.
