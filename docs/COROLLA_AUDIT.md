# Corolla source and ranking audit — 12 October 2026

The supplied snapshot correctly maps Toyota Corolla, COROLLA and Toyota Corolla
source labels to `toyota:corolla`. Corolla Verso remains a separate model and
Auris is not silently merged. Corolla is present in all three reliability
sources and is not excluded as a grouped or unmatched model.

Using the screenshot's age range 4–20, latest loaded reports, minimum known test
count 100, at least two sources, at least three source-cohort observations and
equal weights, the existing combined ranking places Corolla **83rd**, with a
score **43.2532**. These figures reproduce the actual local dataset; they are
not an external assertion about the best car or exact engine/generation.

| Source | Mean favourable percentile | Selected rows | Selected ages |
| --- | ---: | ---: | --- |
| Traficom inspections (2025 observations, 2026 publication) | 94.9913 | 17 | 4–20 |
| TÜV 2025 secondary inspection tables | 23.2143 | 1 | 4–5 |
| ADAC 2025 report (2024 breakdowns) | 11.5539 | 2 | 4–5 |

The equal-source average of those three percentiles is the displayed combined
score. They do not mean 95%, 23% or 12% of Corollas are reliable, nor measure
repair cost or lifetime.

## Original-value verification

- The [actual TÜV 2025 secondary table for 4–5-year-old cars](https://car-recalls.eu/reliability/reliability-tuv-report-2025-4-5-years/)
  explicitly lists Toyota Corolla at source rank 87, 11.3% faults and 63,000 km.
  This agrees with the loaded row. It is a secondary transcription, not a newly
  acquired primary TÜV microdataset.
- The [original official ADAC 2025 PDF](https://assets.adac.de/image/upload/v1744788916/ADAC-eV/KOR/Text/PDF/Pannenstatistik_2025_final_nrmvls.pdf)
  lists Corolla rates 10.4, 10.9, 15 and 20 per 1,000 for registration years
  2022, 2021, 2020 and 2019. The screenshot's minimum age of four retains 15
  and 20. The PDF's Corolla fault list identifies starter-battery issues for
  registration years 2019–2022. These rates and column alignments were rechecked.
- Traficom's stored rows are highly favourable relative to same-cohort peers,
  including 1.9% failed inspections for the four-year cohort and 4.5% for five
  years. Finland's inspections, cars and reporting period differ from Germany.

## Interpretation and UI correction

The Corolla's absence from a top-20 combined chart is not evidence of a missing
Toyota import. The audit also exposes a substantive limitation: the combined
source averages cover different age/generation mixes and measure different
outcomes. A high roadside breakdown rate can concern a starter battery rather
than engine or transmission durability. Survey rankings, inspection rankings
and attended-breakdown rankings need not agree.

Version 2.2 adds source-specific rankings and an individual model-source
diagnostic showing actual scores, coverage and original rows. It warns when
source age ranges differ or scores strongly disagree. Shared-cohort comparison
and source-omission sensitivity make the effect of choices visible. Corolla's
score has not been altered to fit a reputation-based expectation.

The underlying three-source snapshot remains unchanged. Age-2-to-15 defaults
produce a different Corolla position (81st, score 44.8245); any comparison must
state its filter settings. Narrowing the age range improves age comparability
but does not eliminate market, test-regime or engine/generation differences.
