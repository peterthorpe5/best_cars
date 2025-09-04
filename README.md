# TÜV 2025 Reliability Scraper & Analyst

Scrape the **Car-Recalls.eu** TÜV Report 2025 reliability tables, cache the raw HTML locally, parse them into a tidy table, and generate simple analyses and plots.

All outputs are **TSV** (tab-separated). No comma-separated files are written.

---

## Features

- **Caches** the TÜV 2025 reliability pages for six age groups:
  - 2–3, 4–5, 6–7, 8–9, 10–11, 12–13 years.
- **Parses** the “Order / Make/Model / Faults (%) / Odometer (thous. km)” tables, handling:
  - EU decimal commas (e.g. `2,6` → `2.6`),
  - tied ranks (missing rank lines carry the previous rank),
  - the site’s **vertical** four-line table header.
- **Normalises** with a rough comparison metric:
  - `faults_per_thousand_km = faults_pct / odometer_thousand_km`
- **Filters** by make/model (substrings or regex).
- **Reports** models that are **consistently top performers** across age groups.
- **Plots**:
  - Top-K by fewest faults per age group (horizontal bars).
  - Rank across age groups for selected models.

> Note: The normalisation above is a simple heuristic; it’s useful for rough comparisons rather than rigorous statistics.

**Data source:** TÜV 2025 reliability summary pages on [car-recalls.eu](https://car-recalls.eu/).  
Example pages:
- 2–3 years: <https://car-recalls.eu/reliability/reliability-tuv-report-2025-2-3-years/>
- 4–5 years: <https://car-recalls.eu/reliability/reliability-tuv-report-2025-4-5-years/>
- 6–7 years: <https://car-recalls.eu/reliability/reliability-tuv-report-2025-6-7-years/>
- 8–9 years: <https://car-recalls.eu/reliability/reliability-tuv-report-2025-8-9-years/>
- 10–11 years: <https://car-recalls.eu/reliability/reliability-tuv-report-2025-10-11-years/>
- 12–13 years: <https://car-recalls.eu/reliability/reliability-tuv-report-2025-12-13-years/>

---

## Installation

You only need **Python 3.10+** and a few packages.

### Option A — Conda (recommended)

1. Install **Miniforge** (Conda-forge based).  
2. Create and activate an environment:
   ```bash
   conda create --name tuv python=3.11 -y
   conda activate tuv
   ```

### Option B — System Python + venv

- **macOS (Homebrew):**
  ```bash
  brew install python@3.11
  python3.11 -m venv .venv
  source .venv/bin/activate
  ```

- **Ubuntu/Debian:**
  ```bash
  sudo apt-get update
  sudo apt-get install -y python3 python3-venv python3-pip
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### Python packages

Install via `requirements.txt` (recommended):
```bash
pip install -r requirements.txt
```

Or install explicitly:
```bash
pip install requests beautifulsoup4 pandas matplotlib
```

Example `requirements.txt`:
```text
requests>=2.31
beautifulsoup4>=4.12
pandas>=2.0
matplotlib>=3.7
```

---

## Quick start

### 1) Cache HTML, then parse & analyse
```bash
# Step 1 — cache all six pages
python scrape_tuv_2025.py --cache_only --html_cache_dir tuv_html

# Step 2 — parse from cache, write TSV, make plots, print consistency report
python scrape_tuv_2025.py   --out_tsv tuv2025_all.tsv   --html_cache_dir tuv_html   --use_cache   --plot_top_per_age 10   --plot_cars "Honda Jazz,Mazda CX-3"   --report_top_consistent 10
```

### 2) Fully offline (e.g. cluster blocks Python HTTP)
```bash
# Fetch pages yourself into .tuv_html/
mkdir -p .tuv_html
curl -L -o .tuv_html/2-3.html  https://car-recalls.eu/reliability/reliability-tuv-report-2025-2-3-years/
curl -L -o .tuv_html/4-5.html  https://car-recalls.eu/reliability/reliability-tuv-report-2025-4-5-years/
curl -L -o .tuv_html/6-7.html  https://car-recalls.eu/reliability/reliability-tuv-report-2025-6-7-years/
curl -L -o .tuv_html/8-9.html  https://car-recalls.eu/reliability/reliability-tuv-report-2025-8-9-years/
curl -L -o .tuv_html/10-11.html https://car-recalls.eu/reliability/reliability-tuv-report-2025-10-11-years/
curl -L -o .tuv_html/12-13.html https://car-recalls.eu/reliability/reliability-tuv-report-2025-12-13-years/

# Then run the parser offline:
python scrape_tuv_2025.py   --out_tsv tuv2025_all.tsv   --html_cache_dir .tuv_html   --use_cache
```

---

## Usage examples

- Basic scrape (online), TSV only:
  ```bash
  python scrape_tuv_2025.py --out_tsv tuv2025_all.tsv
  ```

- Filter models of interest and save a filtered TSV:
  ```bash
  python scrape_tuv_2025.py     --out_tsv tuv2025_all.tsv     --filter_contains "Honda,Toyota"     --filter_out_tsv tuv2025_honda_toyota.tsv
  ```

- Regex filter + plots + consistency report:
  ```bash
  python scrape_tuv_2025.py     --out_tsv tuv2025_all.tsv     --filter_regex "(Honda|Toyota)\s"     --plot_top_per_age 15     --report_top_consistent 10
  ```

---

## Command-line reference

```
--out_tsv PATH                 Path to write the combined TSV (required unless --cache_only).
--out_dir PATH                 Directory for plots (default: figures).
--ages CSV                     Age groups (default: 2-3,4-5,6-7,8-9,10-11,12-13).
--html_cache_dir PATH          Folder for cached HTML (default: .tuv_html).
--use_cache                    Parse only from cached HTML (no network).
--cache_only                   Download HTML and exit (no parsing).
--force_refresh                Re-download HTML even if cache exists.
--filter_contains CSV          Substrings to keep (case-insensitive).
--filter_regex REGEX           Case-insensitive regex to filter make/model.
--filter_out_tsv PATH          Save filtered results to this TSV (optional).
--plot_top_per_age INT         If >0, plot top-K by fewest faults per age group.
--plot_cars CSV                Model names to plot rank across age groups.
--report_top_consistent INT    If >0, print a top-N consistency table.
--verbosity 0|1|2              Log level: 0=WARNING, 1=INFO, 2=DEBUG (default: 1).
```

---

## Outputs

- **Combined table** (TSV):
  - Columns: `age_group`, `year_of_production`, `rank`, `make_model`, `faults_pct`, `odometer_thousand_km`, `faults_per_thousand_km`.
- **Filtered table** (optional TSV) with the same columns.
- **Plots** (optional PNGs under `figures/`):
  - `top_{K}_age_{group}.png`
  - `rank_over_age_{model}.png`
- **Consistency report** (optional, printed as TSV to stdout):
  - `make_model`, `appearances_top_n`, `groups_covered`, `median_rank`, `mean_faults_pct`.

---

## Troubleshooting

- **“No data parsed; nothing to write.”**  
  Ensure the cache files exist and contain real HTML (`.tuv_html/<age>.html`).  
  The parser detects the site’s **vertical** table header; if the publisher changes layout again, the header logic may need updating.

- **Cluster blocks Python HTTP**  
  Use the offline flow: fetch HTML with `curl`/`wget`, then run with `--use_cache`.

- **Matplotlib**  
  Each plot is a separate figure; no specific colours or styles are forced (defaults only).

---

## Acknowledgements

- Data: TÜV 2025 reliability summaries from <https://car-recalls.eu/>.  
- UK English spelling and **tab-separated** outputs throughout.
