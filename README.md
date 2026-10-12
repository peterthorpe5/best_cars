# Best Cars — reliability evidence explorer

An interactive **Streamlit** app and reproducible Python pipeline for comparing car reliability evidence. Version 2.2 adds a richer offline HTML report, current public MOT file support and a safer Mac update workflow. It includes model search, shortlists, comparisons, age/mileage graphics, source disagreement, safety context, provenance and expandable methods help.

Mac installation and overlay instructions are in [docs/MAC_INSTALL.md](docs/MAC_INSTALL.md). Saved-data browsing does not import PDF/native cryptography dependencies. ADAC PDF acquisition is an optional `pdf` extra.

All tabular outputs are **TSV**, compressed TSV or formatted Excel. No comma-separated output files are written. The existing scripts, cached HTML, results and figures remain available.

## Start with the supplied upgrade

The downloadable upgrade contains `overlay/` (complete runnable code), `verified_data/` (actual processed observations), `changes.patch`, an offline interactive HTML overview and `START_HERE.md`.

```bash
cd "$HOME/Downloads/best_cars_upgrade_20261012/overlay"
conda create --name best_cars python=3.11 pip -y
conda activate best_cars
python -m pip install -e '.[dev]'
best-cars-app \
  --data-dir "$HOME/Downloads/best_cars_upgrade_20261012/verified_data" \
  --server-port 8501
```

Open `http://127.0.0.1:8501`. Stop with **Ctrl+C**. If another app uses that port, use `--server-port 8502`.

## Install from a checkout

Python **3.11+** is required. macOS/Linux and Conda are supported. The validated build used Python 3.12 and Streamlit 1.65.

```bash
git clone https://github.com/peterthorpe5/best_cars.git
cd best_cars
conda create --name best_cars python=3.11 pip -y
conda activate best_cars
python -m pip install -e '.[dev]'
best-cars refresh --data-dir "$HOME/best_cars_data" --sources adac traficom
best-cars-app --data-dir "$HOME/best_cars_data" --server-port 8501
```

Alternatively create and activate a virtual environment with `python3 -m venv .venv`, then use the same pip command. `environment.yml` provides a Conda definition.

The app works immediately with the **real historical TÜV baseline**, even offline. New source downloads are explicit. Results default to `$HOME/best_cars_data`; override with `--data-dir` or `BEST_CARS_DATA_DIR`. Keep source responses, snapshots and outputs outside the code checkout. Use one importing/refreshing process per data directory at a time.

The root Streamlit entry point is `app.py`:

```bash
export BEST_CARS_DATA_DIR="$HOME/best_cars_data"
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

## Actual data and source availability

The verified 12 October 2026 dataset contains **8,924 reliability observations** and **314 US safety-context records from one explicit example lookup**. No fabricated demo rows are supplied.

| Source | Actual content in the supplied snapshot | Role |
| --- | --- | --- |
| TÜV via Car-Recalls.eu | 1,224 observations; reports 2024/2025; six age groups | Significant inspection-defect percentages; existing real repository data |
| ADAC | 969 observations; 158 parsed source model groups; report 2025 / breakdown year 2024 | Breakdowns per 1,000 registered vehicles from the official PDF |
| Traficom, Finland | 6,731 observations; inspection years 2023–2025 / publications 2024–2026 | Model/cohort inspection failures, test counts, failed-test counts and odometer data |
| NHTSA recalls and complaints | US 2020 Toyota Corolla: 3 recalls and 311 complaints at acquisition | Official safety context; exact make/model/year queries can add other vehicles |
| UK DVSA MOT | Official national files identified; full bulk data **not bundled** | Tested local importer for original results or TSV aggregates |

The **2026 ADAC and TÜV summaries are published**, but complete accessible model-level numeric feeds were not verified for those editions. The default uses the actual accessible 2025 ADAC PDF and existing TÜV 2024/2025 tables, alongside newer Traficom data. It does not pass off older tables or selective winners as complete 2026 results.

The UK recall service was inaccessible from the build environment; the app provides its direct check link without claiming a downloaded UK recall database. Which?, What Car? and Consumer Reports remain documented restricted/selective candidates, without invented rows or claimed bulk integrations. See [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

## App pages

1. **Overview:** coverage, report/observation dates and an initial shortlist.
2. **Find reliable cars:** search, filter age/sample/breadth, adjust source weights, save a shortlist.
3. **Compare cars:** up to six model series, raw source panels and intervals for known test counts.
4. **Age and mileage:** cross-sectional age profiles and inspection odometer plots.
5. **Evidence agreement:** heatmaps and pairwise source comparisons with missing cells left blank.
6. **Recalls and complaints:** explicit US vehicle queries plus official UK check links.
7. **Sources and methods:** availability, limitations and plain-language glossary help.
8. **Data management:** explicit refreshes, statuses, reloads and complete exports.
9. **Deeper analysis:** source-specific rate distributions, actual-count intervals,
   coverage statistics and rankings recalculated after omitting each source.

Version 2.2 adds best-to-worst heatmap views, source-specific model-by-age
heatmaps, fixed model colours across source labels, single-source rankings,
model-level explanations and shared-cohort comparisons. Missing cells are grey
and labelled rather than treated as favourable zeroes. See
[docs/SCREENSHOT_REVIEW.md](docs/SCREENSHOT_REVIEW.md) for the critical review
and [docs/COROLLA_AUDIT.md](docs/COROLLA_AUDIT.md) for the source/ranking audit.

Charts support hover, zoom, PNG export and offline interactive HTML downloads. Tables provide TSV and Excel downloads with frozen headers and filters. Every chart includes interpretation help.

## Ranking methods

- Lower observed rates receive higher relative percentiles **within each source/report/age or registration cohort**. Ties use average ranks; at least five peers are required.
- Cohort scores are averaged once per model per source, then combined using selected source weights. More rows from one source do not give it more votes.
- Missing sources are omitted, never zero. `weight_coverage` reports the fraction of selected weight actually available.
- `sources`, `cohorts`, `worst_cohort_score` and `score_spread` expose breadth and disagreement; they are not calibrated confidence measures.
- Display searches happen after peer-percentile calculation, so searching one make does not change the original peer comparison.
- Grouped/unspecified model labels remain in raw evidence but are excluded from cross-source rankings.
- Recall/complaint counts do not enter scores because those endpoints lack a comparable fleet denominator.

The score is an **exploratory relative ordering**, not a failure probability, forecast or a substitute for checking an individual car. Sources differ in outcome, country, climate, cohorts, test rules and usage. Restrict the age range, inspect raw evidence and check the actual vehicle's service history.

The old `faults_per_thousand_km` field remains in the original scripts. The new app does not use it: dividing a failure percentage by mean odometer mileage is not a valid mileage-adjusted rate. The original `year_of_production` label is replaced with explicitly qualified approximate registration-cohort bounds. Engines and exact generations are not inferred. See [docs/METHODS.md](docs/METHODS.md).

## Refreshes and offline browsing

```bash
best-cars refresh --data-dir "$HOME/best_cars_data" --sources adac traficom
best-cars refresh --data-dir "$HOME/best_cars_data" --sources adac traficom --force
best-cars refresh --data-dir "$HOME/best_cars_data" --sources adac traficom --offline
best-cars refresh --data-dir "$HOME/best_cars_data" --sources tuv --tuv-years 2024 2025
best-cars sources
```

App startup does not contact external sources. Cached responses and saved compressed TSV tables carry SHA-256 digests. A complete validated snapshot is written before `current.json` advances; old snapshots remain intact. A failed source refresh retains its previous observations and records the failure. CLI exit statuses: success 0, data error 1, partial refresh failure 2.

The supplied `verified_data/` contains processed data, without bulk response caches. It supports offline browsing. Offline **source re-parsing** requires the original source responses to have been downloaded/cached first. See the source audit for exact provenance URLs.

## UK MOT import

The replacement [DVSA anonymised MOT data page](https://www.data.gov.uk/dataset/c63fca52-ae4c-4b75-bab5-8b4735e1a4c9/anonymised-mot-tests-and-results) includes **2024** results. Earlier years remain on the [Department for Transport page](https://www.data.gov.uk/dataset/e3939ef8-30c7-4ca8-9c7c-ad9475cc9b2f/anonymised-mot-tests-and-results). The current authorised MOT-history API is separate; no credentials are assumed.

Historical results use **pipe-delimited**, headerless records with 14 documented columns. A bounded preview of the actual 2024 archive instead contains **comma-delimited records with a header and 15 columns**, adding `completed_date`. Use the appropriate explicit delimiter/header setting. The importer accepts text, gzip or a ZIP containing multiple results files, deduplicating across all members. macOS archive metadata are skipped. Keep failure-item files separate.

For the actual 2024 results archive:

```bash
best-cars import \
  --kind mot-raw \
  --input "$HOME/Downloads/MOT testing data results (2024).zip" \
  --delimiter comma \
  --mot-conflicts exclude \
  --data-dir "$HOME/best_cars_data"
```

For a historical headerless pipe-delimited release:

```bash
best-cars import \
  --kind mot-raw \
  --input "$HOME/Downloads/test_result_2023.zip" \
  --delimiter pipe \
  --no-header \
  --data-dir "$HOME/best_cars_data"
```

The importer streams chunks and deduplicates **test IDs** using temporary on-disk SQLite; conflicting duplicate IDs are rejected. Class 4 `NT` tests with `P`, `F`, `PRS` outcomes form the denominator; `F` plus `PRS` form the initial-failure numerator. Retests, abandoned and aborted tests are excluded. Class 4 includes some light goods and other vehicles, so inspect model labels carefully.

The 2024 preview contains conflicting duplicate records. The default conflict
policy stops visibly. The example explicitly chooses `--mot-conflicts exclude`,
which removes every detected conflicting eligible test ID entirely, including
later appearances, and writes the excluded count into the saved evidence notes.
Such exclusions can introduce selection bias; audit them before using national
rates. Counts for distinct raw labels sharing an explicit canonical alias are
pooled, preserving original labels and weighting odometers by known readings.

The official guide flags ambiguous mileage units before 2022. Only corrected 2022+ readings are interpreted as miles and converted to kilometres; zero/missing readings remain unknown. Test counts are not necessarily independent cars. Large national files need sufficient disk space and may take considerable time.

For an existing carefully computed aggregate, supply TSV columns `make`, `model`, `observation_year`, `registration_year`, `tests`, `failures`; optional `mean_mileage_km`:

```bash
best-cars import --kind mot-aggregate --input uk_mot_counts.tsv --data-dir "$HOME/best_cars_data"
```

`failures` means initial F plus PRS, not arbitrary repairs. Conservative model matching can retain trim-specific MOT labels separately until a justified explicit alias is added.

## Other imports and exact US safety lookups

```bash
best-cars import --kind traficom-xlsx --input inspections_2025.xlsx --year 2025 --data-dir "$HOME/best_cars_data"
best-cars import --kind legacy-tuv --input tuv_multiyear.tsv --data-dir "$HOME/best_cars_data"
best-cars import --kind evidence --input verified_evidence.tsv --data-dir "$HOME/best_cars_data"
best-cars nhtsa --make Toyota --model Corolla --model-year 2020 --data-dir "$HOME/best_cars_data"
```

ADAC local import uses `--kind adac-pdf --input FILE --year REPORT_YEAR --source-url EXACT_DOWNLOAD_URL`. Supply an actual official PDF and its correct report year. The parser rejects changed/malformed table layouts rather than guessing columns. Traficom XLSX `--year` means inspection year.

To enable PDF imports, install `python -m pip install -e '.[pdf]'` from this checkout. Native PDF dependency errors are reported as import failures while existing snapshots remain available; the Mac guide describes the `cryptography`/OpenSSL repair.

Local reliability imports replace the complete source/report pairs they contain; supply complete source/report tables. NHTSA lookup replaces only the requested exact US make/model/year/event-type records. A successful empty lookup is not proof of no defects. UK-to-US model mappings are never guessed.

## Exports, verification and compatibility

```bash
best-cars rank --data-dir "$HOME/best_cars_data" --output reliability_ranking.tsv --min-sources 2 --min-cohorts 3 --age-min 2 --age-max 15
python -m best_cars.report --data-dir "$HOME/best_cars_data" --output reliability_overview.html
python -m pytest --cov=best_cars --cov-report=term-missing
python -m ruff check src tests scripts app.py
python -m ruff format --check src tests scripts app.py
python -m build
python scripts/smoke_app.py --data-dir "$HOME/best_cars_data"
```

The self-contained HTML now includes source-specific means, medians, percentiles,
rate distributions, counted-test coverage, 95% Wilson intervals, age/mileage
profiles, paired Spearman agreement, source-score spread and searchable/sortable
tables for every qualifying model. All statistics describe the selected evidence.
Unweighted model-cohort means and count-pooled percentages remain distinct.

MOT records can support longevity estimates after linking per-vehicle histories.
The current importer aggregates failure rates and does **not** retain those
histories or calculate lifespans. See [docs/LONGEVITY.md](docs/LONGEVITY.md) for
censoring, inferred fleet exit, median lifetime and restricted-mean requirements.

The HTML overview is self-contained and works offline. CLI ranking uses equal source weights; the app supports personalised weights.

Tests cover scoring, weights, missing evidence, counts, Wilson intervals, source layouts, PDF year/blank-cell alignment, initial MOT semantics, duplicate IDs, bounded caching/offline behaviour, snapshot integrity, failed refreshes, CLI settings, downloads and all nine app pages. Test fixtures are explicit synthetic examples; user-facing result tables are real evidence. New public functions use keyword-only settings, defensive validation, logging and Google-style docstrings. Calls use named arguments wherever the library API permits them.

| Folder/file | Purpose |
| --- | --- |
| `src/best_cars/` | Installable package, CLI, source adapters, app and report exporter |
| `src/best_cars/data/` | Real historical TÜV baseline and source catalogue |
| `tests/` | Deterministic automated tests |
| `scripts/` | Actual-data source and app smoke checks |
| `docs/` | Source audit, methods, migration and original instructions |
| `app.py` | Root Streamlit entry point |
| Original scripts, tables and figures | Original workflow retained |

Original instructions are retained in [docs/LEGACY_README.md](docs/LEGACY_README.md); see [docs/MIGRATION.md](docs/MIGRATION.md). Source attribution remains attached to every observation. This code does not confer rights to third-party report prose/images or restricted datasets; no report prose/image archive is reissued.
