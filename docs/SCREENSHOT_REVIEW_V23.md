# Screenshot review and version 2.3 — 12 October 2026

Ten usable screenshots show all nine version 2.2 pages, including two deeper
analysis views. The 15.17.50 attachment is a narrow image strip and cannot be
meaningfully assessed. Review uses the supplied images, not recreated screenshots.

| Page | Finding | Change |
| --- | --- | --- |
| Overview | Loaded totals and filtered counts differ; weights use raw IDs | Existing scope caption retained; readable source weight labels |
| Find reliable cars | Inspector starts at a model outside the displayed ranking; source IDs remain raw | Ranked models offered first; explicit eligibility message for other models; readable source labels |
| Compare cars | Source labels overlap plotted lines | Facet headings placed above panel domains; consistent model colours retained |
| Age and mileage | Score heatmap can be mistaken for fault percentages; numeric-looking age labels plus missing annotations clutter the plot | Explicit score/rate selector; labels and colours share one matrix; categorical axes; separate grey missing-value layer |
| Evidence agreement | Score colour mapping is consistent, but direction needs clear interpretation | Fixed 0–100 scale retained; numeric cells retain relative score interpretation |
| Deeper analysis | Box headings are vertical, boxes appear at inconsistent horizontal locations, distributions mix ages; sensitivity is table-heavy | Aligned boxes and horizontal headings; exact-cohort peer median gaps; scenario rank movement chart and rank-range table |
| Recalls and complaints | Long narratives cannot fit compact tables | Full-event reader retained; event filters remain independent of reliability filters |
| Sources and methods | Access restrictions and outcome differences matter | Existing source catalogue retained; new propulsion methodology documented |
| Data management | Saved data are being loaded from the downloaded package | Explicit folder display retained; reinstall and overlay instructions updated |

## Heatmap semantics

Relative scores are cohort percentiles: high score/teal means a lower rate
relative to peers. They are not failure percentages. The published-rate mode
shows the actual percentage or breakdowns per 1,000 vehicles: low rate/teal is
favourable, high rate/red is poorer. Both use exactly the same values for text
and colour. Scores use 0–100; raw rates use zero to the full selected source/report
maximum, so changing the model search does not rescale raw colours. A grey dash
is unavailable evidence, never a favourable zero. Raw rates across ages are not
controlled comparisons. Row ordering continues to use mean available scores.

## Deeper interrogation

Same-cohort benchmarks exclude the target model from the reference median and
require at least four other ungrouped models. Source, report, observation year,
outcome, units, comparison group, ages and registration cohort must match.
Negative rate gaps are favourable; units are never pooled across sources.
Benchmarks use the current filtered reference population, not a national fleet.

Three preference scenarios use the same common eligible competitors for ranks.
Rank ranges describe choices of source weights, not statistical uncertainty.
The existing custom-weight/source-omission analysis remains separate. None of
these analyses measures causation, repairs, engine reliability or service life.

## Electric and hybrid page

See PROPULSION.md. All three categories have age profiles, original observations,
same-cohort benchmarks, scenario sensitivity, actual-count intervals and source
omission analysis wherever the available data support them. Single-model views
cannot establish a best-in-market model. Battery-health data are not fabricated.
