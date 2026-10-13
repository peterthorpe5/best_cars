# Compare one model or several — version 3.1

Open **Compare cars**, clear the initial selection and choose **Volkswagen T-Roc**
for a focused profile. Add other models for comparisons, up to twenty at a time.
Use model or make colours and the local source/age filters. Every chart has an
interpretation box and an interactive HTML download; tables export TSV and Excel.

## Six complementary views

| Tab | What it answers | Interpretation |
| --- | --- | --- |
| Age profiles | How do published rates vary across vehicle ages? | Each source has its own units and panel. Raw rates are the default; original peer percentiles are optional. Report years have separate line styles. |
| Same-age peers | Does this model perform better than its contemporaries? | Negative bars mean a lower rate than the median of other models in the exact published cohort. This is descriptive, not a significance test. |
| Matched models | How do two models compare on genuinely shared recorded contexts? | Choose a reference. Each point pairs source, report, observation year, outcome, age and registration bounds. Below the diagonal favours the compared model. |
| Count uncertainty | How precise are the inspection estimates? | Separate panels match exact published age/cohort, source and report by default. Optional two-year groups retain nearby ages without pooling counts. Points use actual failures divided by tests, with 95% Wilson intervals. Published rounded rates are retained on hover. Unknown denominators have no invented intervals. |
| Evidence coverage | Which observations and counts are available? | Grey: no row under current filters. Gold: rate only. Teal: rate with numerator and denominator; n means tests for that cell, not unique vehicles. |
| Across years | Does the published picture change between observation years? | Hold the age band or the recorded registration cohort constant. Only series with at least two observation years are plotted. |

## Age and time are different

An age profile compares different registration cohorts in a published report.
A line from ages four to seven does not follow the same individual cars.

The **Across years** tab can include earlier loaded reports even while the sidebar
uses the latest report for other views. It keeps the same source, sidebar age,
registration-year and minimum known sample filters, plus local source/age filters.
It is independent of the shared-cohort checkbox, whose restriction concerns the
current-report charts. The history checkbox states this exception explicitly.

- **Published age band:** compare, for example, four-year-old T-Rocs in 2023,
  2024 and 2025. These are different registration cohorts. This can show changes
  in reported performance at a fixed age, not a causal improvement in production.
- **Recorded registration cohort:** compare repeated summaries of vehicles first
  registered in a particular year as the cohort gets older. Fleet composition,
  survival, inspection eligibility and servicing can change. Individual cars are
  not linked. Approximate TÜV registration-band estimates are excluded here.

ADAC and Traficom observation years are calendar years. TÜV observation labels
denote the inspection window's ending year. The bundled ADAC report covers one
observation year, so no ADAC time trend can be fabricated from it.

## Matching, peers and gaps

The shared-cohort checkbox restricts current-report age, peer, matched-rate and
count charts to contexts available for **every** selected model. Pairwise matched
rates without that restriction match each competitor with the chosen reference;
different pairs can have different coverage. Pair counts remain explicit.

The coverage grid deliberately shows recorded evidence **before** that shared
restriction so that unavailable comparisons and unmatched observations stay visible.
Missing coverage is not a reliability score of zero. Source/cohort rows are not
model counts and are not pooled across sources into a vehicle denominator.

Peer benchmarks always use the complete sidebar-filtered reference population,
before selecting individual models. Each target is excluded from its median;
at least four other ungrouped models are required. A narrowed display selection
does not silently redefine the peer comparison.

TÜV and Traficom gaps are percentage points. ADAC gaps are breakdown events per
1,000 registered vehicles. Source panels never pool these into one raw rate.
Exact recorded matching does not control engine, gearbox, mileage, maintenance,
country differences across sources or model generation.

## What these views cannot establish

They do not predict an individual car's future failures, engine lifetime, battery
health, annual repair costs or the effect of a particular repair. A low inspection
rate can coexist with frequent starting problems. A known model-specific fault
list is context, not a count that can be subtracted from aggregate breakdown rates.

The offline report includes illustrative T-Roc/Corolla/C-HR comparisons where these
models are loaded, with fallback examples for other datasets. These are examples,
not a curated list of recommended purchases. Full model and filter selections are
available in Streamlit. The bundled real snapshot is unchanged.

## Wilson panels across the app

Both Compare cars and Deeper analysis use separate exact-age Wilson panels by
default. Select the age panels to display; limits and sorting apply independently
within each chart. Optional two-year groups allow nearby ages without treating
them as exactly matched. Wide source age bands stay intact. Mixed-age charts
remain an explicit exploratory option. Sources, reports and observation years
are separated in every mode. No counts are combined and no existing interval
is recalculated by grouping.
