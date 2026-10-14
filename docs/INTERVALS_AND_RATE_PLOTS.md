# Where interval bars come from — version 3.2

Deeper analysis and Compare cars → Rates and intervals now include an expandable
explanation, original source links, an interval-availability table, the exact Wilson
formula and a worked example selected from actual loaded counts.

## From Traficom records to a point and its bars

Traficom is Finland's transport authority. Its annual inspection statistics use
the Transport Register. Our model/registration-year API import reads:

| Original API field | App meaning |
| --- | --- |
| Number of periodic inspections | n: denominator of recorded inspections |
| Failed | x: number of failed inspections |
| Fail-rate | Rounded published percentage, retained alongside x/n |
| Inspection year and year of entry into service | Observation year and registration cohort; age is their difference |

The unit is an inspection, not necessarily a distinct car or individual defect.
An inspection can find several faults but enters the failed-test numerator once.
Repeated vehicles and fleet composition can affect comparisons.

The point is **100 × x/n**. The app calculates a two-sided **95% Wilson score
interval**, with z = 1.959963984540054 and no continuity correction:

```text
p = x/n
d = 1 + z²/n
c = (p + z²/(2n)) / d
h = z × sqrt[p(1 − p)/n + z²/(4n²)] / d
[lower, upper] = 100 × [c − h, c + h]
```

Endpoints are bounded by 0 and 100. The point remains the observed x/n, not the
Wilson centre c. Left/right whiskers are point − lower and upper − point, so
their lengths can differ. Zero observed failures still have a positive upper
bound. These intervals are computed by the app, not supplied by Traficom.

For the loaded four-year-old T-Roc cohort in Finnish 2025 inspections:

- x = 11 failures, n = 746 tests.
- Counted percentage = 1.4745%; published rounded percentage = 1.5%.
- 95% Wilson interval = 0.8253–2.6209%.

The UI example is dynamic and uses the current loaded evidence; it does not
silently substitute these numbers for a different source or cohort.

## What the interval does and does not mean

Wilson is a binomial model for independent, comparable test outcomes. Under its
assumptions, repeated interval constructions have approximately 95% coverage of
the underlying process proportion. It is not a 95% probability that a particular
car will fail, a prediction of next year's rate, or an engine-life estimate.

The observed rate of the recorded tests is already known. For census-like
administrative coverage, a model-based repeat-sampling interval is not uncertainty
about that observed census percentage. It does not correct maintenance, inspection
rules, variant/generation differences, survival, selection or repeated vehicles.
An interval overlap check alone is not a formal hypothesis test.

Age/source/report grouping changes which rows are shown; it never pools counts
or recomputes the point/interval from an average of different cohorts.

## Equivalent point plots for other sources

| Source/import | What can be plotted now | Interval availability |
| --- | --- | --- |
| Traficom | Counted inspection-failure percentages, with published rates retained | Wilson from each actual x,n pair |
| TÜV summaries via Car-Recalls.eu | Significant-defect percentages in age bands | Loaded tables omit model/cohort x,n; open points without inferred bars |
| ADAC official PDF | Attended breakdown events per 1,000 registered vehicles | Model exposure/event counts are not imported; Wilson is not an event-rate interval |
| UK MOT imports | Initial failed tests / eligible initial tests | Wilson when genuine aggregate counts have been imported; no national MOT snapshot bundled |

The TÜV report's overall inspection total cannot be assigned to individual models.
The ADAC scaling unit **1,000 is not an observed sample size**. Percentages, total
report counts, mean mileage, minimum fleet thresholds and numbers of published
rows are never used to manufacture a missing numerator or denominator.

The new horizontal source-rate plots separate exact age/cohort, source, report,
observation year and units. Open points mean a published rate with no calculable
interval. Solid points have actual inspection count pairs and Wilson whiskers.
An absent interval is not zero uncertainty or perfect reliability. If genuine
inspection count pairs are imported later, the corresponding intervals appear
automatically; no source identity is hardcoded as incapable of an interval.

## Sources and statistical reference

- [Traficom definitions and official files](https://tieto.traficom.fi/en/statistics/statistics-inspections)
- [Imported Traficom model/registration-year API table](https://trafi2.stat.fi/PXWeb/api/v1/en/TraFi/Katsastuksen_vikatilastot/040_kats_tau_104.px)
- [NIST: Wilson intervals for proportions](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm)
- [TÜV 2025 report methodology](https://www.tuev-verband.de/pressemitteilungen/tuev-report-2025-elektroautos)
- [Secondary TÜV summary source](https://car-recalls.eu/reliability/)
- [ADAC 2025 official report](https://assets.adac.de/image/upload/v1744788916/ADAC-eV/KOR/Text/PDF/Pannenstatistik_2025_final_nrmvls.pdf)
