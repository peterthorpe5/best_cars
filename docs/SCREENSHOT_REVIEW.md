# Critical screenshot review and version 2.2 changes

Eight attached screenshots showed the car app. The 14:06:27 attachment showed
an unrelated C-terminal protein-motif conservation chart and was excluded.
Uploaded screenshots were read directly; none were altered or copied into this
delivery. The review is based on those screenshots and the actual app/data.

| Observed problem | Consequence | Implemented change |
| --- | --- | --- |
| Rankings displayed only the top models | Poorer evidence and contradictory cases were hard to discover | Best/worst/both-ends/disagreement heatmap views and model search. |
| Source counts used a continuous colour bar with values such as 2.2 | Suggested fractional sources or unjustified precision | Discrete source-count legend. |
| The same model had differently capitalised names across source panels | Different colours implied different models; comparisons were confusing | One name and colour per model ID across panels. |
| Tables displayed internal IDs and excessive decimal places | Difficult to read and explain | Friendly headings, hidden internal IDs and sensible display precision; exports retain canonical fields. |
| Long facet labels ran vertically | Source/units were hard to read | Horizontal source labels and larger chart text. |
| Missing ADAC odometers produced a large empty chart | Looked broken and occupied space | Explanatory empty state with alternative-source guidance. |
| Heatmaps concentrated on the highest-ranked cars | A user could not easily inspect unreliable models | Red-to-teal model/age heatmaps; original full-peer scores retained before search. |
| Blank cells lacked an explicit interpretation | Missing observations could be mistaken for good performance | Grey backgrounds with explicit No data/No score annotations; genuine zero scores remain red. |
| Comparison scores covered unequal cohorts | Composite positions could obscure age/generation differences | Shared-cohort switch and source-separated rate-gap tables with overlap counts. |
| A single ranking concealed source dependence | Surprise cases, such as Corolla, were difficult to diagnose | Single-source rankings, model-source diagnostics, age coverage and source-omission sensitivity. |
| Safety lookup selection was separate from saved-table contents | Selecting recall still left saved complaints visible | Independent saved-event-type filtering and full event reader. |
| Existing HTML statistics were not prominent in the app | Richer analysis required leaving the app | New Deeper analysis page with distributions, actual-count intervals, source association and sensitivity. |

## Statistical safeguards

Heatmaps show favourable relative cohort scores, not absolute probabilities of
mechanical failure. Each source retains its own outcome and rate unit. Both-end
views explicitly omit middle-ranked models; row limits are stated. Grouped model
series do not enter the model-by-age heatmap. Missing scores are not imputed.

Shared cohorts require the same source, report/observation year, metric, units,
age bounds and registration-cohort bounds for every selected model. They still
do not establish identical engines, model generation, use or maintenance. Rate
gaps are descriptive, with equal cohort weights, and no unsupported significance
or causal claim. Original percentile peer sets remain intact.

Source-omission sensitivity reranks both versions within the same available
model universe for each omission, so a car does not appear to improve merely
because rivals disappear. Alternatives require at least one remaining scored
source and retain the specified cohort minimum. Missing alternatives are
explicitly unavailable. This is sensitivity analysis, not a confidence interval.

## Priorities requiring further real data

- A longitudinal MOT history database and survival analysis remain separate work;
  this version does not manufacture life-expectancy estimates.
- Engine/generation/fuel-specific rankings require trustworthy source mappings
  and adequate denominators. These filters cannot be inferred from model names.
- Repair costs, severity, servicing and usage would help distinguish an expensive
  mechanical failure from a roadside battery incident. Those data are not loaded.
- Broader UK and owner-survey evidence would improve market coverage; restricted
  or selectively published tables must not be treated as complete ranking peers.

The improvements are tested through pure-function checks and Streamlit AppTest.
Reviewing the user's screenshots is not a claim of new browser screenshot
validation on the updated application.
