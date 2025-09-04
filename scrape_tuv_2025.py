#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scrape & Cache TÜV 2025 Reliability Tables (Car-Recalls.eu)

This script downloads the TÜV Report 2025 reliability pages for six age groups,
saves the raw HTML locally (one file per age group), and parses the saved files
to produce a combined TSV. It also offers filtering, plotting, and a “top-N
consistency” report across age groups.

Key features
------------
- Caches HTML in a chosen folder and parses from those files (reproducible).
- Offline mode: `--use_cache` parses only local HTML (no network).
- Cache-only mode: `--cache_only` just downloads HTML and exits.
- Robust fetching with retries and friendly headers.
- Parses EU decimal commas and tied ranks.
- Adds a normalised metric: faults_per_thousand_km = faults_pct / odometer_thousand_km.
- Outputs TSV only (no comma-separated files), UK English spellings, PEP 8 docstrings.
- Plots use Matplotlib defaults, one chart per figure, no colours specified.

Typical usage
-------------
# 1) Fetch & cache HTML, then parse and save TSV + figures
python scrape_tuv_2025.py \
    --out_tsv tuv2025_all.tsv \
    --html_cache_dir .tuv_html \
    --plot_top_per_age 10 \
    --plot_cars "Honda Jazz,Mazda CX-3" \
    --report_top_consistent 10

# 2) Cache HTML only (e.g. prefetch, then parse later/offline)
python scrape_tuv_2025.py \
    --cache_only \
    --html_cache_dir .tuv_html

# 3) Fully offline parse (e.g. you used curl to save pages into .tuv_html)
python scrape_tuv_2025.py \
    --out_tsv tuv2025_all.tsv \
    --html_cache_dir .tuv_html \
    --use_cache

Dependencies
------------
pip install requests beautifulsoup4 pandas matplotlib
"""

from __future__ import annotations

import argparse
import html as ihtml
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import requests
from bs4 import BeautifulSoup


TUV_2025_URLS: Dict[str, str] = {
    "2-3": "https://car-recalls.eu/reliability/reliability-tuv-report-2025-2-3-years/",
    "4-5": "https://car-recalls.eu/reliability/reliability-tuv-report-2025-4-5-years/",
    "6-7": "https://car-recalls.eu/reliability/reliability-tuv-report-2025-6-7-years/",
    "8-9": "https://car-recalls.eu/reliability/reliability-tuv-report-2025-8-9-years/",
    "10-11": "https://car-recalls.eu/reliability/reliability-tuv-report-2025-10-11-years/",
    "12-13": "https://car-recalls.eu/reliability/reliability-tuv-report-2025-12-13-years/",
}

AGE_TO_YOP: Dict[str, str] = {
    "2-3": "2021–2022",
    "4-5": "2019–2020",
    "6-7": "2017–2018",
    "8-9": "2015–2016",
    "10-11": "2013–2014",
    "12-13": "2011–2012",
}


def setup_logger(verbosity: int) -> logging.Logger:
    """
    Create and configure a basic logger.

    Parameters
    ----------
    verbosity : int
        0 = WARNING, 1 = INFO, 2 = DEBUG.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    return logging.getLogger("tuv2025")


def robust_fetch(url: str, retries: int = 4, backoff: float = 1.7, timeout: int = 40, logger: Optional[logging.Logger] = None) -> str:
    """
    Fetch a URL with retries and return HTML text.

    Parameters
    ----------
    url : str
        Target URL to fetch.
    retries : int, optional
        Number of retry attempts, by default 4.
    backoff : float, optional
        Exponential backoff base, by default 1.7.
    timeout : int, optional
        Per-request timeout in seconds, by default 40.
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
    raise RuntimeError(f"Failed to fetch {url}: {last_exc}")  # noqa: TRY003


def cache_path_for_age(cache_dir: Path, age_label: str) -> Path:
    """
    Compute the cache filepath for a given age group.

    Parameters
    ----------
    cache_dir : Path
        Directory holding cached HTML files.
    age_label : str
        Age group label (e.g., '2-3').

    Returns
    -------
    Path
        Expected HTML cache file path.
    """
    return cache_dir / f"{age_label}.html"


def ensure_cached_html(age_label: str, url: str, cache_dir: Path, force_refresh: bool, logger: logging.Logger) -> Path:
    """
    Ensure the HTML for an age group is cached; download if missing or refresh requested.

    Parameters
    ----------
    age_label : str
        Age group label.
    url : str
        Source URL.
    cache_dir : Path
        Folder to store cached HTML.
    force_refresh : bool
        If True, re-download even if cached file exists.
    logger : logging.Logger
        Logger for progress messages.

    Returns
    -------
    Path
        Path to the cached HTML file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = cache_path_for_age(cache_dir=cache_dir, age_label=age_label)
    if fp.exists() and not force_refresh:
        logger.info("Using cached HTML for %s: %s", age_label, fp)
        return fp

    logger.info("Downloading %s → %s", url, fp)
    html_text = robust_fetch(url=url, logger=logger)
    fp.write_text(html_text, encoding="utf-8")
    return fp


def read_cached_html(cache_file: Path) -> str:
    """
    Read an HTML file from cache.

    Parameters
    ----------
    cache_file : Path
        Path to a cached HTML file.

    Returns
    -------
    str
        HTML content as text.
    """
    return cache_file.read_text(encoding="utf-8")


def extract_table_lines(page_html: str, logger: Optional[logging.Logger] = None) -> List[str]:
    """
    Extract raw lines of the TÜV table from the HTML.

    This version detects the vertical four-line header:
    'Order' / 'Make/Model' / 'Faults (%)' / 'Odometer (thous. km)' on
    consecutive lines, then collects rows until a 'Source:' marker.

    Parameters
    ----------
    page_html : str
        The HTML content of the page.
    logger : logging.Logger, optional
        Logger for messages.

    Returns
    -------
    List[str]
        Lines immediately after the header, up to but not including the footer.
    """
    soup = BeautifulSoup(page_html, "html.parser")
    text = soup.get_text(separator="\n")
    text = ihtml.unescape(text).replace("\xa0", " ")

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Find the vertical header (four consecutive lines)
    header_idx: Optional[int] = None
    for i in range(0, max(0, len(lines) - 3)):
        if (
            lines[i].lower() == "order"
            and "make/model" in lines[i + 1].lower()
            and "faults" in lines[i + 2].lower()
            and "odometer" in lines[i + 3].lower()
        ):
            header_idx = i
            break

    if header_idx is None:
        if logger:
            logger.error("Could not locate vertical table header. Page structure may have changed.")
        return []

    out: List[str] = []
    terminating = re.compile(r"^(Source:|Subscribe to our newsletter|//\s*Featured|Car Recalls Family)", re.I)
    for ln in lines[header_idx + 4 :]:
        if terminating.search(ln):
            break
        ln = re.sub(r"†\S+", "", ln)   # tidy dagger-like artefacts
        ln = re.sub(r"\s{2,}", " ", ln)
        out.append(ln.strip())

    return out



_ROW_RE = re.compile(
    r"""
    ^\s*
    (?:(?P<rank>\d{1,3})\s+)?         # optional leading rank (carry over for ties)
    (?P<model>.+?)\s+
    (?P<faults>\d{1,2},\d)\s+         # EU decimal comma
    (?P<odo>\d{1,3})\s*               # integer thousands
    $
    """,
    re.VERBOSE,
)

def parse_rows(lines: Iterable[str], logger: Optional[logging.Logger] = None) -> List[Dict[str, object]]:
    """
    Parse table lines into structured rows, handling tied ranks.

    The table is not a single-line CSV-like layout. It streams as:
    [optional rank], model, faults (EU decimal comma), odometer (int).
    When a tie occurs, the rank line is omitted; we carry forward the
    last seen rank.

    Parameters
    ----------
    lines : Iterable[str]
        Lines after the header (as returned by extract_table_lines).
    logger : logging.Logger, optional
        Logger for debug/trace messages.

    Returns
    -------
    List[Dict[str, object]]
        Parsed rows with keys: rank, make_model, faults_pct, odometer_thousand_km.
    """
    rows: List[Dict[str, object]] = []
    last_rank: Optional[int] = None

    it = list(lines)
    i = 0
    re_int = re.compile(r"^\d{1,3}$")
    re_faults = re.compile(r"^\d{1,2},\d$")  # e.g. 2,6  or 10,3

    while i <= len(it) - 3:
        # Optional rank line
        if re_int.match(it[i]):
            rank = int(it[i])
            last_rank = rank
            i += 1
        else:
            rank = last_rank

        # Model
        if i >= len(it):
            break
        model = it[i].strip()
        i += 1

        # Faults (%)
        if i >= len(it) or not re_faults.match(it[i]):
            if logger and logger.isEnabledFor(logging.DEBUG):
                logger.debug("Skipping candidate row due to unexpected faults token near: %r", model)
            i += 1
            continue
        faults_pct = float(it[i].replace(",", "."))
        i += 1

        # Odometer (thous. km)
        if i >= len(it) or not re_int.match(it[i]):
            if logger and logger.isEnabledFor(logging.DEBUG):
                logger.debug("Skipping candidate row due to unexpected odometer token near: %r", model)
            i += 1
            continue
        odometer_thousand_km = int(it[i])
        i += 1

        if rank is None:
            # If the very first row lacked a rank, we cannot assign one
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




def df_from_html(age_label: str, html_text: str) -> pd.DataFrame:
    """
    Convert HTML text for one age group into a tidy DataFrame.

    Parameters
    ----------
    age_label : str
        Age group label (e.g., '2-3').
    html_text : str
        HTML content for that page.

    Returns
    -------
    pd.DataFrame
        Columns: ['age_group','year_of_production','rank','make_model',
                  'faults_pct','odometer_thousand_km','faults_per_thousand_km'].
    """
    lines = extract_table_lines(page_html=html_text)
    parsed = parse_rows(lines=lines)
    df = pd.DataFrame(parsed)
    if df.empty:
        return df
    df.insert(0, "age_group", age_label)
    df.insert(1, "year_of_production", AGE_TO_YOP.get(age_label, ""))
    df["faults_per_thousand_km"] = df["faults_pct"] / df["odometer_thousand_km"]
    return df


def filter_df(df: pd.DataFrame, contains: Optional[List[str]] = None, regex: Optional[str] = None) -> pd.DataFrame:
    """
    Filter rows by make/model substrings and/or regex.

    Parameters
    ----------
    df : pd.DataFrame
        Combined table to filter.
    contains : list of str, optional
        Case-insensitive substrings to keep (any match).
    regex : str, optional
        Case-insensitive regex to apply.

    Returns
    -------
    pd.DataFrame
        Filtered copy (or original if no filters active).
    """
    mask = pd.Series(True, index=df.index)
    if contains:
        patt = "|".join([re.escape(s.strip()) for s in contains if s.strip()])
        if patt:
            mask &= df["make_model"].str.contains(patt, case=False, regex=True)
    if regex:
        mask &= df["make_model"].str.contains(regex, case=False, regex=True)
    return df.loc[mask].copy()


def consistency_report(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Summarise consistent top performers across age groups.

    Parameters
    ----------
    df : pd.DataFrame
        Combined table containing ranks.
    top_n : int, optional
        Threshold for counting a 'top' appearance (rank <= top_n).

    Returns
    -------
    pd.DataFrame
        Summary with appearances_top_n, groups_covered, median_rank, mean_faults_pct.
    """
    in_top = df.assign(in_top=(df["rank"] <= top_n))
    rep = (
        in_top.groupby("make_model", as_index=False)
        .agg(
            appearances_top_n=("in_top", "sum"),
            groups_covered=("age_group", "nunique"),
            median_rank=("rank", "median"),
            mean_faults_pct=("faults_pct", "mean"),
        )
        .sort_values(["appearances_top_n", "groups_covered", "median_rank"], ascending=[False, False, True])
    )
    return rep


def plot_top_per_age(df: pd.DataFrame, out_dir: Path, top_k: int = 10) -> None:
    """
    Plot bar charts of the top-K (fewest faults) per age group.

    Parameters
    ----------
    df : pd.DataFrame
        Combined table.
    out_dir : Path
        Directory to save figures.
    top_k : int, optional
        Number of models to show per age group.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for ag, sub in df.groupby("age_group", sort=False):
        top = sub.sort_values(["faults_pct", "rank"], ascending=[True, True]).head(top_k)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top["make_model"], top["faults_pct"])
        ax.invert_yaxis()
        ax.set_xlabel("Faults (%)")
        ax.set_title(f"TÜV 2025 top {top_k} by fewest faults – Age {ag} ({AGE_TO_YOP.get(ag,'')})")
        fig.tight_layout()
        fig.savefig(out_dir / f"top_{top_k}_age_{ag.replace('-', '_')}.png", dpi=150)
        plt.close(fig)


def plot_rank_over_age(df: pd.DataFrame, models: List[str], out_dir: Path) -> None:
    """
    Plot rank across age groups for selected car models.

    Parameters
    ----------
    df : pd.DataFrame
        Combined table.
    models : list of str
        Model name substrings to include (case-insensitive).
    out_dir : Path
        Directory for saved figures.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = "|".join([re.escape(m) for m in models if m.strip()])
    sub = df[df["make_model"].str.contains(wanted, case=False, regex=True)].copy()
    if sub.empty:
        return
    age_order = list(TUV_2025_URLS.keys())
    sub["age_group"] = pd.Categorical(sub["age_group"], categories=age_order, ordered=True)
    for name, g in sub.groupby("make_model"):
        g = g.sort_values("age_group")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(g["age_group"].astype(str), g["rank"], marker="o")
        ax.invert_yaxis()
        ax.set_ylabel("Rank (1 = best)")
        ax.set_title(f"TÜV 2025 rank across age groups – {name}")
        fig.tight_layout()
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
        fig.savefig(out_dir / f"rank_over_age_{safe}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    """
    CLI entry point. Manages caching, parsing, filtering, plotting, and reports.
    """
    parser = argparse.ArgumentParser(description="Scrape, cache, and analyse TÜV 2025 reliability tables.")
    parser.add_argument("--out_tsv", type=str, default="", help="Path to write combined TSV (required unless --cache_only).")
    parser.add_argument("--out_dir", type=str, default="figures", help="Directory for plots.")
    parser.add_argument("--ages", type=str, default="2-3,4-5,6-7,8-9,10-11,12-13", help="Comma-separated age groups to process.")
    parser.add_argument("--html_cache_dir", type=str, default=".tuv_html", help="Folder to store cached HTML.")
    parser.add_argument("--use_cache", action="store_true", help="Parse only from cached HTML (no network).")
    parser.add_argument("--cache_only", action="store_true", help="Download HTML and exit without parsing.")
    parser.add_argument("--force_refresh", action="store_true", help="Re-download HTML even if cache exists.")
    parser.add_argument("--filter_contains", type=str, default="", help="Comma-separated substrings to keep (make/model).")
    parser.add_argument("--filter_regex", type=str, default="", help="Regex (case-insensitive) to filter make/model.")
    parser.add_argument("--filter_out_tsv", type=str, default="", help="Optional TSV to save filtered rows.")
    parser.add_argument("--plot_top_per_age", type=int, default=0, help="If >0, plot top-K by fewest faults per age group.")
    parser.add_argument("--plot_cars", type=str, default="", help="Comma-separated models to plot rank over age groups.")
    parser.add_argument("--report_top_consistent", type=int, default=0, help="If >0, print top-N consistency table.")
    parser.add_argument("--verbosity", type=int, default=1, help="0=WARNING, 1=INFO, 2=DEBUG.")
    args = parser.parse_args()

    logger = setup_logger(verbosity=args.verbosity)

    ages = [a.strip() for a in args.ages.split(",") if a.strip()]
    cache_dir = Path(args.html_cache_dir)

    if args.cache_only:
        for ag in ages:
            url = TUV_2025_URLS[ag]
            try:
                ensure_cached_html(age_label=ag, url=url, cache_dir=cache_dir, force_refresh=args.force_refresh, logger=logger)
            except Exception as exc:  # noqa: BLE001
                logger.error("Caching failed for %s: %s", ag, exc)
        logger.info("Cache-only complete.")
        sys.exit(0)

    if not args.out_tsv:
        parser.error("--out_tsv is required unless --cache_only is used.")

    frames: List[pd.DataFrame] = []
    for ag in ages:
        cache_file = cache_path_for_age(cache_dir=cache_dir, age_label=ag)
        if args.use_cache:
            if not cache_file.exists():
                logger.error("Missing cached HTML for %s: %s", ag, cache_file)
                continue
        else:
            try:
                ensure_cached_html(age_label=ag, url=TUV_2025_URLS[ag], cache_dir=cache_dir, force_refresh=args.force_refresh, logger=logger)
            except Exception as exc:  # noqa: BLE001
                logger.error("Download failed for %s: %s", ag, exc)
                continue

        try:
            html_text = read_cached_html(cache_file=cache_file)
            df_ag = df_from_html(age_label=ag, html_text=html_text)
        except Exception as exc:  # noqa: BLE001
            logger.error("Parsing failed for %s: %s", ag, exc)
            continue

        if not df_ag.empty:
            frames.append(df_ag)
        else:
            logger.warning("No rows parsed for %s; skipping.", ag)

    if not frames:
        logger.error("No data parsed; nothing to write.")
        sys.exit(2)

    df_all = pd.concat(frames, ignore_index=True)

    # Nice ordering
    age_order = list(TUV_2025_URLS.keys())
    df_all["age_group"] = pd.Categorical(df_all["age_group"], categories=age_order, ordered=True)
    df_all = df_all.sort_values(["age_group", "rank", "make_model"]).reset_index(drop=True)

    # Save combined TSV (tab-separated)
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
            # Preview to stdout
            with pd.option_context("display.max_rows", 50, "display.max_columns", None):
                print(df_filt.head(50).to_string(index=False))

    # Optional plots
    out_dir = Path(args.out_dir)
    if args.plot_top_per_age and args.plot_top_per_age > 0:
        plot_top_per_age(df=df_all, out_dir=out_dir, top_k=int(args.plot_top_per_age))
        logger.info("Saved top-per-age plots to: %s", out_dir)

    if args.plot_cars:
        cars = [s.strip() for s in args.plot_cars.split(",") if s.strip()]
        if cars:
            plot_rank_over_age(df=df_all, models=cars, out_dir=out_dir)
            logger.info("Saved rank-over-age plots to: %s", out_dir)

    # Optional consistency report
    if args.report_top_consistent and args.report_top_consistent > 0:
        rep = consistency_report(df=df_all, top_n=int(args.report_top_consistent))
        cols = ["make_model", "appearances_top_n", "groups_covered", "median_rank", "mean_faults_pct"]
        print(rep[cols].head(25).to_csv(sep="\t", index=False))


if __name__ == "__main__":
    main()
