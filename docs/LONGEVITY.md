# Estimating car service life from MOT histories

MOT histories are a credible foundation for estimating **service life on British
roads**. Missing later tests do not establish mechanical death: scrappage,
export, accidents, SORN/storage, exemptions and economic decisions can all affect
observed records. The outcome must be named accurately.

The primary methodological precedent is Nguyen-Tien et al.,
[The closing longevity gap between battery electric vehicles and internal
combustion vehicles in Great Britain](https://www.nature.com/articles/s41560-024-01698-1),
Nature Energy (2025), DOI 10.1038/s41560-024-01698-1. It analysed linked MOT
histories, modelled interval-censored retirement, and compared 15-, 18- and
21-month test-gap definitions. Its estimates are not imported as results here.

## What is implemented now

The application imports initial-test failure counts. A bounded 2024 archive
preview has verified actual source fields and format. No national longitudinal
history has been collected or analysed, and no lifespan numbers are claimed.
The importer currently discards vehicle ID and detailed test dates when creating
aggregates; a separate retained-history pipeline is required.

## Required history pipeline

1. Collect successive official annual files with resumable downloads, checksums
   and an inventory of the actual date ranges. Earlier years are published by
   DfT; DVSA publishes from 2023. Keep acquisition, release and cut-off dates.
2. Store all completed test histories by anonymised vehicle ID in an indexed or
   columnar database. Retests can demonstrate that a car remains observed, even
   though the failure-rate denominator excludes them. Deduplicate tests and
   inspect repeated-day records and conflicting vehicle metadata.
3. Audit identifiers across releases. The guide describes IDs based on
   registration and VIN, so changes/corrections could interrupt a history.
   Inspect known long gaps followed by reappearance before inferring exits.
4. Clean first-use dates, placeholder dates, mileage reversals/units and vehicle
   classes. Fuel labels changed historically and may misclassify hybrids.
   Class 4 includes light goods and other vehicles beyond passenger cars.
5. Mark recent histories as right-censored at the **actual last complete data
   date**, not today's date. Treat inferred exit as an interval after the last
   observed test. Calibrate missed-test rules to observed return gaps and show
   sensitivity to several thresholds. Account for exceptional MOT extensions.
6. Model delayed entry: most ordinary cars first appear around their first MOT,
   rather than when new. Deaths before entry are unobserved; results are
   conditional on this observed population unless external registration data
   support a broader inference. Early computerisation years are incomplete.
7. Fit survival methods appropriate for delayed entry and interval censoring.
   Compare empirical and parametric assumptions where feasible; an ordinary
   average of last-test ages or Kaplan–Meier applied to invented exact death
   dates is not sufficient. Use vehicle-level uncertainty, not test-level
   independence. Separate model series, cohorts and fuels only with adequate
   observed follow-up and sufficient inferred exits.

## Statistics and graphics to report

| Output | Required interpretation |
| --- | --- |
| Survival curve | Probability of still being in observed road service at each age; show uncertainty and numbers at risk. |
| Survival to 10/15/20 years | Publish only ages supported by the observed follow-up, or label extrapolations clearly. |
| Median service lifetime | Age when the estimated survival curve crosses 50%; otherwise show “not reached/estimable”. |
| Restricted mean service life | Area under the survival curve up to a stated, supported age horizon. |
| Full mean lifetime | Requires observing the tail or an explicitly identified extrapolation model. |
| Remaining life at a current age | Conditional on already surviving to that age; identify whether mileage/usage are actually modelled. |
| Mileage before inferred exit | Last recorded odometer is a lower bound, not exact scrappage mileage. |
| Sensitivity | Compare missed-test thresholds, cleaning choices and modelling assumptions. |

NHTSA recall/complaint records provide fault and safety context for exact US
vehicles. They do not track the complete population through fleet entry and
exit, and cannot independently establish mean or median vehicle lifespan.
Do not infer UK engine/model equivalence or rank models by unadjusted complaint
counts. Inspection rates, roadside breakdown rates and longevity answer
different questions and should remain separately visible.
