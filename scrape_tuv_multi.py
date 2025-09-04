#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scrape TÜV reliability tables (Car-Recalls.eu) across multiple years and analyse trends.

This script downloads and caches the TÜV reliability pages for chosen YEARS
(e.g., 2025 and 2024) and AGE GROUPS (2–3, 4–5, 6–7, 8–9, 10–11, 12–13 years).
It parses the vertical four-line table header used on the site and the streamed
row format: [optional rank], model, faults (% with EU comma), odometer (thous. km).

Key features
------------
- Multi-year scraping (default: 2025,2024), with per-year cache folders.
- Robust header detection (vertical 4-line header found within a short window).
- Tied ranks supported (missing rank line inherits last rank).
- Normalisation column: faults_per_thousand_km = faults_pct / odometer_thousand_km.
- Filtering by make/model (substring or regex).
- Analyses:
  1) Consistency across all (year, age) combinations: who appears top-N most often.
  2) Year-over-year change for a given age group (delta in rank between years).
  3) Best overall: roll-up across all available (year, age) with mean faults (%) and coverage.
- Optional plots:
  - Top-K by fewest faults per (year, age).
  - Rank-over-age for selected models per year (multi-year overlay).

Outputs
-------
- Combined TSV with columns:
  ['year','age_group','year_of_production','rank','make_model',
   'faults_pct','odometer_thousand_km','faults_per_thousand_km']
- Optional filtered TSV.
- PNG plots in an output directory.

Notes
-----
- URL pattern assumed:
  https://car-recalls.eu/reliability/reliability-tuv-report-{YEAR}-{AGE}-years/
  e.g. reliability-tuv-report-2025-2-3-years/  (same pattern usually works for 2024).
  If the publisher changes slugs, you can pre-download pages with curl into the cache.
- Normalisation is a rough heuristic; treat with caution for cross-segment comparisons.

Author
------
UK English spelling, PEP 8 docstrings, tab-separated outputs, named arguments only.
"""

from __future__ import annotations

import argparse
import html as ihtml
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import requests
from bs4 import BeautifulSoup


AGE_GROUPS_DEFAULT: List[str] = ["2-3", "4-5", "6-7", "8-9", "10-11", "12-13"]

AGE_TO_YOP_2025: Dict[str, str] = {
    "2-3": "2021–2022",
    "4-5": "2019–2020",
    "6-7": "2017–2018",
    "8-9": "2015–2016",
    "10-11": "2013–2014",
    "12-13": "2011–2012",
}

# 2024 mapping (one year older)
AGE_TO_YOP_2024: Dict[str, str] = {
    "2-3": "2020–2021",
    "4-5": "2018–2019",
    "6-7": "2016–2017",
    "8-9": "2014–2015",
    "10-11": "2012–2013",
    "12-13": "2010–2011",
}


def setup_logger(verbosity: int) -> logging.Logger:
    """
    Configure a simple logger.

    Parameters
    ----------
    verbosity : int
        0=WARNING, 1=INFO, 2=DEBUG.

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    return logging.getLogger("tuv_multi")


def url_for(year: int, age_label: str) -> str:
    """
    Build the expected page URL for a given year and age group.

    Parameters
    ----------
    year : int
        TÜV summary year, e.g., 2025 or 2024.
    age_label : str
        Age group label such as '2-3'.

    Returns
    -------
    str
        Fully-qualified URL.
    """
    return f"https://car-recalls.eu/reliability/reliability-tuv-report-{year}-{age_label}-years/"


def year_to_yop(year: int) -> Dict[str, str]:
    """
    Map an age group to a Year-Of-Production string for a given TÜV year.

    Parameters
    ----------
    year : int
        TÜV summary year.

    Returns
    -------
    Dict[str, str]
        Mapping of age label to YOP string.
    """
    if year == 2025:
        return AGE_TO_YOP_2025
    if year == 2024:
        return AGE_TO_YOP_2024
    # Fallback: approximate by shifting 2025 mapping
    shift = 2025 - year
    approx: Dict[str, str] = {}
    for ag, yop in AGE_TO_YOP_2025.items():
        # crude shift on the YYYY–YYYY string
        m = re.findall(r"\d{4}", yop)
        if len(m) == 2:
            a, b = int(m[0]) - shift, int(m[1]) - shift
            approx[ag] = f"{a}–{b}"
        else:
            approx[ag] = yop
    return approx


def robust_fetch(url: str, retries: int = 4, backoff: float = 1.7, timeout: int = 40, logger: Optional[logging.Logger] = None) -> str:
    """
    Fetch a URL with retries and return HTML content.

    Parameters
    ----------
    url : str
        Target URL.
    retries : int, optional
        Number of attempts, by default 4.
    backoff : float, optional
        Base for exponential backoff, by default 1.7.
    timeout : int, optional
        Per-request timeout seconds, by default 40.
    logger : logging.Logger, optional
        Logger for diagnostics.

    Returns
    -------
    str
        HTML content as text.

    Raises
    ------
    RuntimeError
        If all attempts fail or response is suspiciously small.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Connection": "keep-alive",
        "Referer": "https://car-recalls.eu/",
    }
    last_exc: Optional[BaseException] = None
    with requests.Session() as sess:
        for attempt in range(1, retries + 1):
            try:
                resp = sess.get(url=url, headers=headers, timeout=timeout, allow_redirects=True)
                if resp.status_code >= 400:
                    raise requests.HTTPError(f"HTTP {resp.status_code} for {url}")
                text = resp.text or ""
                if len(text) < 500:
                    raise RuntimeError(f"Suspiciously small response ({len(text)} bytes)")
                return text
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if logger:
                    logger.warning("Fetch attempt %d/%d failed: %s", attempt, retries, exc)
                if attempt < retries:
                    time.sleep(backoff ** attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_exc}")


def cache_file(cache_root: Path, year: int, age_label: str) -> Path:
    """
    Compute cache path for a given (year, age group).

    Parameters
    ----------
    cache_root : Path
        Root cache directory.
    year : int
        TÜV year.
    age_label : str
        Age group label.

    Returns
    -------
    Path
        Path to cached HTML file.
    """
    return cache_root / str(year) / f"{age_label}.html"


def ensure_cached_html(
    year: int,
    age_label: str,
    cache_root: Path,
    force_refresh: bool,
    logger: logging.Logger,
) -> Path:
    """
    Ensure the page HTML is cached on disk; download if missing or forced.

    Parameters
    ----------
    year : int
        TÜV year.
    age_label : str
        Age group label.
    cache_root : Path
        Cache root folder.
    force_refresh : bool
        Re-download even if present.
    logger : logging.Logger
        Logger.

    Returns
    -------
    Path
        Path to cached HTML.
    """
    fp = cache_file(cache_root=cache_root, year=year, age_label=age_label)
    if fp.exists() and not force_refresh:
        logger.info("Using cache: %s", fp)
        return fp
    fp.parent.mkdir(parents=True, exist_ok=True)
    u = url_for(year=year, age_label=age_label)
    logger.info("Downloading %s → %s", u, fp)
    html_text = robust_fetch(url=u, logger=logger)
    fp.write_text(html_text, encoding="utf-8")
    return fp


def read_cached_html(cache_path: Path) -> str:
    """
    Read HTML text from a cached file.

    Parameters
    ----------
    cache_path : Path
        Cached HTML path.

    Returns
    -------
    str
        HTML content.
    """
    return cache_path.read_text(encoding="utf-8")


def find_table_start(lines: List[str]) -> Optional[int]:
    """
    Find the index of the start of data after detecting a vertical 4-line header.

    The header tokens 'Order', 'Make/Model', 'Faults', 'Odometer' should appear
    in this order within a sliding window (not necessarily strictly consecutive).
    The function returns the index of the first line AFTER the header block.

    Parameters
    ----------
    lines : list of str
        Cleaned text lines.

    Returns
    -------
    int or None
        Index in `lines` where data begins, or None if no header found.
    """
    tokens = ["order", "make/model", "faults", "odometer"]
    L = len(lines)
    for i in range(0, L - 1):
        # look ahead up to 10 lines for the four tokens in order
        window = [ln.lower() for ln in lines[i : min(L, i + 10)]]
        pos = []
        idx = 0
        for tok in tokens:
            try:
                j = next(k for k in range(idx, len(window)) if tok in window[k])
            except StopIteration:
                pos = []
                break
            pos.append(j)
            idx = j + 1
        if len(pos) == 4:
            # start after the last token line in the window
            return i + pos[-1] + 1
    return None


def extract_table_lines(page_html: str, logger: Optional[logging.Logger] = None) -> List[str]:
    """
    Extract table lines by sweeping the page text.

    Parameters
    ----------
    page_html : str
        Raw HTML content.
    logger : logging.Logger, optional
        Logger for diagnostics.

    Returns
    -------
    List[str]
        Lines between header and footer markers, cleaned.
    """
    soup = BeautifulSoup(page_html, "html.parser")
    text = soup.get_text(separator="\n")
    text = ihtml.unescape(text).replace("\xa0", " ")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    start = find_table_start(lines=lines)
    if start is None:
        if logger:
            logger.error("Could not detect vertical header; site structure may have changed.")
        return []

    out: List[str] = []
    terminating = re.compile(r"^(Source:|Subscribe to our newsletter|//\s*Featured|Car Recalls Family)", re.I)
    for ln in lines[start:]:
        if terminating.search(ln):
            break
        ln = re.sub(pattern=r"†\S+", repl="", string=ln)
        ln = re.sub(pattern=r"\s{2,}", repl=" ", string=ln)
        out.append(ln.strip())
    return out


def parse_rows(lines: Iterable[str], logger: Optional[logging.Logger] = None) -> List[Dict[str, object]]:
    """
    Parse streamed table lines into structured rows, handling tied ranks.

    The stream format per row is:
        [optional rank], model, faults (EU decimal comma), odometer (int).
    For ties, the rank line is omitted and we carry the last seen rank.

    Parameters
    ----------
    lines : Iterable[str]
        Lines beneath the header.
    logger : logging.Logger, optional
        Logger for debug info.

    Returns
    -------
    list of dict
        Rows with keys: rank, make_model, faults_pct, odometer_thousand_km.
    """
    as_list = list(lines)
    i = 0
    re_int = re.compile(r"^\d{1,3}$")
    re_faults = re.compile(r"^\d{1,2},\d$")

    rows: List[Dict[str, object]] = []
    last_rank: Optional[int] = None

    while i <= len(as_list) - 3:
        if re_int.match(as_list[i]):
            last_rank = int(as_list[i])
            i += 1
        rank = last_rank

        if i >= len(as_list):
            break
        model = as_list[i].strip()
        i += 1

        if i >= len(as_list) or not re_faults.match(as_list[i]):
            if logger and logger.isEnabledFor(logging.DEBUG):
                logger.debug("Unexpected faults token near model=%r; skipping.", model)
            i += 1
            continue
        faults_pct = float(as_list[i].replace(",", "."))
        i += 1

        if i >= len(as_list) or not re_int.match(as_list[i]):
            if logger and logger.isEnabledFor(logging.DEBUG):
                logger.debug("Unexpected odometer token near model=%r; skipping.", model)
            i += 1
            continue
        odometer_thousand_km = int(as_list[i])
        i += 1

        if rank is None:
            continue

        rows.append(
            {
                "rank": rank,
                "make_model": model,
                "faults_pct": faults_pct,
                "odometer_thousand_km": odometer_thousand_km,
            }
        )

    return rows


def df_from_html(year: int, age_label: str, html_text: str) -> pd.DataFrame:
    """
    Convert one page HTML to a tidy DataFrame for a given (year, age group).

    Parameters
    ----------
    year : int
        TÜV year.
    age_label : str
        Age group label.
    html_text : str
        HTML content of the page.

    Returns
    -------
    pd.DataFrame
        Tidy table with an added normalisation column.
    """
    lines = extract_table_lines(page_html=html_text)
    parsed = parse_rows(lines=lines)
    df = pd.DataFrame(parsed)
    if df.empty:
        return df

    yop_map = year_to_yop(year=year)

    df.insert(0, "year", year)
    df.insert(1, "age_group", age_label)
    df.insert(2, "year_of_production", yop_map.get(age_label, ""))
    df["faults_per_thousand_km"] = df["faults_pct"] / df["odometer_thousand_km"]
    return df


def filter_df(df: pd.DataFrame, contains: Optional[List[str]] = None, regex: Optional[str] = None) -> pd.DataFrame:
    """
    Filter the table by make/model substrings and/or a regex.

    Parameters
    ----------
    df : pd.DataFrame
        Combined table.
    contains : list of str, optional
        Case-insensitive substrings to include (any match).
    regex : str, optional
        Case-insensitive regex pattern.

    Returns
    -------
    pd.DataFrame
        Filtered copy.
    """
    mask = pd.Series(True, index=df.index)
    if contains:
        patt = "|".join([re.escape(s.strip()) for s in contains if s.strip()])
        if patt:
            mask &= df["make_model"].str.contains(patt, case=False, regex=True)
    if regex:
        mask &= df["make_model"].str.contains(regex, case=False, regex=True)
    return df.loc[mask].copy()


def consistency_report_across_years(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Count how often each model appears in the top-N across all (year, age) pairs.

    Parameters
    ----------
    df : pd.DataFrame
        Combined table with 'year', 'age_group', and 'rank'.
    top_n : int, optional
        Threshold for 'top', inclusive.

    Returns
    -------
    pd.DataFrame
        Summary columns: make_model, appearances_top_n, distinct_years, groups_covered, median_rank, mean_faults_pct.
    """
    in_top = df.assign(in_top=(df["rank"] <= top_n))
    rep = (
        in_top.groupby("make_model", as_index=False)
        .agg(
            appearances_top_n=("in_top", "sum"),
            groups_covered=("age_group", "nunique"),
            distinct_years=("year", "nunique"),
            median_rank=("rank", "median"),
            mean_faults_pct=("faults_pct", "mean"),
        )
        .sort_values(["appearances_top_n", "distinct_years", "groups_covered", "median_rank"], ascending=[False, False, False, True])
    )
    return rep


def year_change_report(
    df: pd.DataFrame,
    age_group: str,
    baseline_year: int,
    compare_year: int,
    top_k: int = 25,
) -> pd.DataFrame:
    """
    Compare ranks between two years for a specific age group.

    Parameters
    ----------
    df : pd.DataFrame
        Combined table.
    age_group : str
        Age group label (e.g., '4-5').
    baseline_year : int
        Year to subtract from (e.g., 2024).
    compare_year : int
        Year to compare to (e.g., 2025).
    top_k : int, optional
        Limit to top-K by baseline rank for compactness.

    Returns
    -------
    pd.DataFrame
        Columns: make_model, rank_baseline, rank_compare, delta (compare - baseline), faults_baseline, faults_compare.
    """
    a = df[(df["age_group"] == age_group) & (df["year"] == baseline_year)][["make_model", "rank", "faults_pct"]]
    b = df[(df["age_group"] == age_group) & (df["year"] == compare_year)][["make_model", "rank", "faults_pct"]]
    a = a.rename(columns={"rank": "rank_baseline", "faults_pct": "faults_baseline"})
    b = b.rename(columns={"rank": "rank_compare", "faults_pct": "faults_compare"})
    merged = pd.merge(a, b, on="make_model", how="inner")
    merged["delta"] = merged["rank_compare"] - merged["rank_baseline"]
    merged = merged.sort_values(["rank_baseline", "make_model"]).head(top_k)
    return merged


def best_overall_report(df: pd.DataFrame, min_groups: int = 6, top_k: int = 25) -> pd.DataFrame:
    """
    Simple roll-up: which models have low mean faults and appear widely.

    Parameters
    ----------
    df : pd.DataFrame
        Combined table.
    min_groups : int, optional
        Minimum number of (year, age) occurrences required to be considered.
    top_k : int, optional
        Number of rows to return.

    Returns
    -------
    pd.DataFrame
        Columns: make_model, n_occurrences, groups_covered, years_covered, mean_faults_pct, median_rank.
    """
    grp = (
        df.groupby("make_model", as_index=False)
        .agg(
            n_occurrences=("rank", "count"),
            groups_covered=("age_group", "nunique"),
            years_covered=("year", "nunique"),
            mean_faults_pct=("faults_pct", "mean"),
            median_rank=("rank", "median"),
        )
        .query("n_occurrences >= @min_groups")
        .sort_values(["mean_faults_pct", "median_rank", "groups_covered", "years_covered"], ascending=[True, True, False, False])
        .head(top_k)
    )
    return grp


def plot_top_per_year_age(df: pd.DataFrame, out_dir: Path, top_k: int = 10) -> None:
    """
    Plot horizontal bars of top-K by fewest faults per (year, age_group).

    Parameters
    ----------
    df : pd.DataFrame
        Combined table.
    out_dir : Path
        Output directory for PNGs.
    top_k : int, optional
        Number of models to include.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for (year, ag), sub in df.groupby(["year", "age_group"], sort=False):
        top = sub.sort_values(["faults_pct", "rank"], ascending=[True, True]).head(top_k)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top["make_model"], top["faults_pct"])
        ax.invert_yaxis()
        ax.set_xlabel("Faults (%)")
        ax.set_title(f"TÜV {year} – top {top_k} fewest faults – Age {ag}")
        fig.tight_layout()
        fig.savefig(out_dir / f"top_{top_k}_year_{year}_age_{ag.replace('-', '_')}.png", dpi=150)
        plt.close(fig)


def plot_rank_over_age_multi_year(df: pd.DataFrame, models: List[str], out_dir: Path) -> None:
    """
    Plot rank across age groups for selected models, one line per year.

    Parameters
    ----------
    df : pd.DataFrame
        Combined table.
    models : list of str
        Model substrings to include (case-insensitive).
    out_dir : Path
        Output directory for PNGs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    age_order = AGE_GROUPS_DEFAULT
    for model_pat in models:
        sub = df[df["make_model"].str.contains(model_pat, case=False, regex=True)].copy()
        if sub.empty:
            continue
        sub["age_group"] = pd.Categorical(sub["age_group"], categories=age_order, ordered=True)
        fig, ax = plt.subplots(figsize=(9, 5))
        for year, g in sub.groupby("year"):
            g = g.sort_values("age_group")
            ax.plot(g["age_group"].astype(str), g["rank"], marker="o", label=str(year))
        ax.invert_yaxis()
        ax.set_ylabel("Rank (1 = best)")
        ax.set_title(f"TÜV rank across age groups – {model_pat}")
        ax.legend(title="Year")
        fig.tight_layout()
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model_pat)
        fig.savefig(out_dir / f"rank_over_age_multi_{safe}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    """
    CLI entry point for multi-year TÜV scraping and analysis.
    """
    parser = argparse.ArgumentParser(description="Scrape multi-year TÜV reliability tables and analyse trends.")
    parser.add_argument("--out_tsv", type=str, default="", help="Path to write the combined TSV (required unless --cache_only).")
    parser.add_argument("--out_dir", type=str, default="figures", help="Directory for plots.")
    parser.add_argument("--years", type=str, default="2025,2024", help="Comma-separated TÜV years to process (e.g., 2025,2024).")
    parser.add_argument("--ages", type=str, default="2-3,4-5,6-7,8-9,10-11,12-13", help="Comma-separated age groups to process.")
    parser.add_argument("--html_cache_dir", type=str, default=".tuv_html", help="Root folder for cached HTML.")
    parser.add_argument("--use_cache", action="store_true", help="Parse only from cached HTML (no network).")
    parser.add_argument("--cache_only", action="store_true", help="Download HTML and exit (no parsing).")
    parser.add_argument("--force_refresh", action="store_true", help="Re-download HTML even if cache exists.")
    parser.add_argument("--filter_contains", type=str, default="", help="Comma-separated substrings to keep (case-insensitive).")
    parser.add_argument("--filter_regex", type=str, default="", help="Case-insensitive regex to filter make/model.")
    parser.add_argument("--filter_out_tsv", type=str, default="", help="Optional TSV path to save filtered rows.")
    parser.add_argument("--plot_top_per_year_age", type=int, default=0, help="If >0, plot top-K by fewest faults per (year, age).")
    parser.add_argument("--plot_cars", type=str, default="", help="Comma-separated model patterns to plot rank across ages (multi-year).")
    parser.add_argument("--report_top_consistent", type=int, default=0, help="If >0, cross-year top-N consistency report.")
    parser.add_argument("--report_year_change", type=str, default="", help="Age group for year-over-year delta (e.g., '4-5').")
    parser.add_argument("--baseline_year", type=int, default=2024, help="Baseline year for delta report (default: 2024).")
    parser.add_argument("--compare_year", type=int, default=2025, help="Comparison year for delta report (default: 2025).")
    parser.add_argument("--report_best_overall", type=int, default=0, help="If >0, print simple 'best overall' top-K by mean faults.")
    parser.add_argument("--min_groups_overall", type=int, default=6, help="Min (year,age) occurrences required for 'best overall'.")
    parser.add_argument("--verbosity", type=int, default=1, help="0=WARNING, 1=INFO, 2=DEBUG.")
    args = parser.parse_args()

    logger = setup_logger(verbosity=args.verbosity)
    years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
    ages = [a.strip() for a in args.ages.split(",") if a.strip()]
    cache_root = Path(args.html_cache_dir)

    if args.cache_only:
        for year in years:
            for ag in ages:
                try:
                    ensure_cached_html(year=year, age_label=ag, cache_root=cache_root, force_refresh=args.force_refresh, logger=logger)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Cache failed for year=%s age=%s: %s", year, ag, exc)
        logger.info("Cache-only complete.")
        sys.exit(0)

    if not args.out_tsv:
        parser.error("--out_tsv is required unless --cache_only is used.")

    frames: List[pd.DataFrame] = []
    for year in years:
        for ag in ages:
            fp = cache_file(cache_root=cache_root, year=year, age_label=ag)
            if not args.use_cache:
                try:
                    ensure_cached_html(year=year, age_label=ag, cache_root=cache_root, force_refresh=args.force_refresh, logger=logger)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Download failed for year=%s age=%s: %s", year, ag, exc)
                    continue
            try:
                html_text = read_cached_html(cache_path=fp)
                df_ag = df_from_html(year=year, age_label=ag, html_text=html_text)
            except Exception as exc:  # noqa: BLE001
                logger.error("Parsing failed for year=%s age=%s: %s", year, ag, exc)
                continue
            if df_ag.empty:
                logger.warning("No rows parsed for year=%s age=%s; skipping.", year, ag)
                continue
            frames.append(df_ag)

    if not frames:
        logger.error("No data parsed; nothing to write.")
        sys.exit(2)

    df_all = pd.concat(frames, ignore_index=True)

    # Nice ordering
    df_all["age_group"] = pd.Categorical(df_all["age_group"], categories=AGE_GROUPS_DEFAULT, ordered=True)
    df_all = df_all.sort_values(["year", "age_group", "rank", "make_model"]).reset_index(drop=True)

    # Save combined TSV
    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out_path, sep="\t", index=False)
    logger.info("Wrote combined TSV: %s (rows: %d)", out_path, len(df_all))

    # Optional filtering
    contains_list = [s.strip() for s in args.filter_contains.split(",") if s.strip()]
    regex = args.filter_regex.strip() or None
    if contains_list or regex:
        df_filt = filter_df(df=df_all, contains=contains_list or None, regex=regex)
        if args.filter_out_tsv:
            filt_path = Path(args.filter_out_tsv)
            filt_path.parent.mkdir(parents=True, exist_ok=True)
            df_filt.to_csv(filt_path, sep="\t", index=False)
            logger.info("Wrote filtered TSV: %s (rows: %d)", filt_path, len(df_filt))
        else:
            with pd.option_context("display.max_rows", 50, "display.max_columns", None):
                print(df_filt.head(50).to_string(index=False))

    # Optional plots
    out_dir = Path(args.out_dir)
    if args.plot_top_per_year_age and args.plot_top_per_year_age > 0:
        plot_top_per_year_age(df=df_all, out_dir=out_dir, top_k=int(args.plot_top_per_year_age))
        logger.info("Saved top-per-(year,age) plots to: %s", out_dir)

    if args.plot_cars:
        cars = [s.strip() for s in args.plot_cars.split(",") if s.strip()]
        if cars:
            plot_rank_over_age_multi_year(df=df_all, models=cars, out_dir=out_dir)
            logger.info("Saved rank-over-age multi-year plots to: %s", out_dir)

    # Reports
    if args.report_top_consistent and args.report_top_consistent > 0:
        rep = consistency_report_across_years(df=df_all, top_n=int(args.report_top_consistent))
        cols = ["make_model", "appearances_top_n", "distinct_years", "groups_covered", "median_rank", "mean_faults_pct"]
        print(rep[cols].head(50).to_csv(sep="\t", index=False))

    if args.report_year_change:
        delta = year_change_report(
            df=df_all,
            age_group=args.report_year_change,
            baseline_year=int(args.baseline_year),
            compare_year=int(args.compare_year),
            top_k=50,
        )
        print(delta.to_csv(sep="\t", index=False))

    if args.report_best_overall and args.report_best_overall > 0:
        best = best_overall_report(df=df_all, min_groups=int(args.min_groups_overall), top_k=int(args.report_best_overall))
        print(best.to_csv(sep="\t", index=False))


if __name__ == "__main__":
    main()
