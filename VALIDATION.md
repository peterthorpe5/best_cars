# Version 3.3 validation

- 221 automated tests passed; coverage 97.15% (minimum 95%).
- All 13 pages passed Streamlit AppTest with 8,924 actual observations.
- Comparison tests cover exact paired contexts and source units, original peer
  reference populations, coverage colours, absent denominators, shared-cohort
  percentile retention, local filters, separate model/cohort time lines, both
  historical modes, and exact/two-year/mixed Wilson grouping without count pooling.
- Added tests verify known-count Wilson endpoints, rounded-rate retention,
  zero-failure positive upper bounds, unavailable denominators, open points with
  no error arrays, actual-count solid points, source-unit separation, and new
  source-rate/help UI controls. Earlier regression tests remain included.
- Previous parser, MOT, PDF isolation, chart, heatmap, source, refresh and overlay
  regression tests remain included.
- New regression tests cover single-source diagnostic eligibility, comparison-group
  separation, tied-rank matrices, stable make colours/highlighting, horizontal
  labels, exact age/report distributions, quartiles/whiskers and readable outliers.
- Nine real-data chart variants rendered and visually inspected using headless
  Plotly/Kaleido: ranking, highlighting, TÜV rates, Wilson intervals, the rank
  matrix and trajectories, separate-age/pooled distributions and disagreement.
- Ruff lint/format checks passed. Wheel and sdist 3.3.0 built successfully.
- Offline HTML embeds Plotly; searchable tables include source profiles, broad
  exploratory gaps and strict comparison profiles.
- Cumulative patch application and overlay preview/apply/idempotency checks
  preserve an extra Fiesta file and Git HEAD.

Validation uses Linux/Python 3.12, AppTest and rendered Plotly charts; it does not
include physical macOS execution. No national MOT collection, verified model
repair frequencies, battery-health measurements or lifespan estimates are bundled.
