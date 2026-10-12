# Choosing sources and weights

The scores are favourable **within-source cohort percentiles**, not the same
underlying rate measured three times. A value of 95 versus 12 is not evidence
that one source is accurate and another wrong. Inspections, attended breakdowns,
repair severity and vehicle lifespan answer different questions.

| Source | Strongest use in this app | Evidence strengths | Limits |
| --- | --- | --- | --- |
| Traficom | Finnish inspection condition by model/registration cohort | Official API; actual tests and failed-test counts; counts permit explicit intervals | Finnish vehicles/test regime; repeated inspections and servicing; model series span generations; no all-repair or lifetime outcome |
| ADAC | Avoiding attended roadside breakdowns | Official original PDF; rates per registered vehicles; cohort-specific values and model fault context | Not all repairs; model denominators not provided in the imported PDF; battery/no-start events can dominate without indicating engine failure |
| TÜV | Additional German inspection evidence | Published source-cohort rates and odometers | Imported secondary tables; unknown model denominators; inspections measure roadworthiness, servicing and wear, not all mechanical repairs |
| UK MOT | British inspection condition, when a national dataset is actually imported | Official raw records and test counts; possible separate longitudinal histories | Not bundled nationally; substantial cleaning/deduplication; Class 4 includes more than passenger cars; missing tests do not establish mechanical death |

For inspection questions, Traficom currently has the most explicit numerator/
denominator audit trail among the loaded sources. For roadside-breakdown
avoidance, ADAC's outcome is more directly relevant. This does not establish a
single universally most trustworthy source or calibrated weights.

## Implemented preference scenarios

| Starting scenario | ADAC | Traficom | TÜV | MOT if loaded |
| --- | ---: | ---: | ---: | ---: |
| Balanced | 1 | 1 | 1 | 1 |
| Emphasise avoiding breakdowns | 2 | 1 | 1 | 1 |
| Emphasise inspection condition | 0.5 | 2 | 1 | 2 |

These are transparent illustrative preferences. They are **not** weights fitted
to an external validation dataset, statistical precision, source quality or
failure probability. Other loaded sources default to one. Sources absent from
the snapshot are not fabricated. Sliders can override the starting scenario,
and the UI says when they differ. Switching scenarios resets the starting
weights; the ranking uses the current values. Zero excludes a source entirely.

Equal weights remain the default comparison scenario. Weighting is useful for
stating priorities; it cannot correct unequal ages, generation, market or
observational bias. Avoid choosing weights simply because they restore an
expected brand ranking. See COROLLA_AUDIT.md for a concrete counterexample.

## Recommended interrogation sequence

1. Choose the relevant age or registration cohort. Inspect each source's
   coverage and outcome, retaining raw rates and units.
2. Use a single-source view when the question is specific to that outcome.
3. Compare shortlisted models on shared published cohorts, while recognising
   remaining engine, generation, usage and maintenance differences.
4. Inspect source-specific scores and raw values for surprising models.
5. Compare explicit weighting scenarios and source-omission sensitivity. If the
   result moves greatly, report that dependence rather than claiming certainty.

Calibrating a broad reliability prediction would need an independently observed,
relevant endpoint: repair costs, major failures, breakdowns or longitudinal road
service. That validation is not available in the current snapshot. No inverse-
variance meta-analysis is attempted across different outcomes or missing model
denominators.
