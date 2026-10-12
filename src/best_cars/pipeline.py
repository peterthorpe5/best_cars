"""Refresh independent data sources while retaining last-known-good results."""

import logging
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

from best_cars.http import CachedClient
from best_cars.parsers import (
    ADAC_2025_PDF,
    NHTSA_BASE,
    TRAFICOM_API,
    parse_adac_pdf,
    parse_nhtsa_json,
    parse_traficom_json,
    parse_tuv_html,
    utc_now,
)
from best_cars.schema import empty_events, validate_evidence
from best_cars.storage import load_snapshot, save_snapshot

LOGGER = logging.getLogger(__name__)
TUV_AGES = ("2-3", "4-5", "6-7", "8-9", "10-11", "12-13")


def fetch_traficom(
    *, client: CachedClient, years: Sequence[int] | None = None
) -> pd.DataFrame:
    """Download all model cohorts via the official PxWeb API in bounded yearly queries.

    Args:
        client: Cached HTTP client.
        years: Inspection years; defaults to all years currently published.

    Returns:
        Combined, validated inspection rates/counts, excluding totals.

    Raises:
        ValueError: If a requested year or required API variable is unavailable.
        OSError: If download/cache access fails.
    """
    metadata = client.request_json(url=TRAFICOM_API)
    variables = {item["code"]: item for item in metadata["variables"]}
    if "Katsastusvuosi" not in variables:
        raise ValueError("Traficom inspection-year variable was not found.")
    available = variables["Katsastusvuosi"]["values"]
    selected = [str(year) for year in years] if years is not None else available
    if not selected or set(selected) - set(available):
        raise ValueError(
            f"Requested Traficom years unavailable; available: {available}"
        )
    tables = []
    for year in selected:
        query = []
        for code, variable in variables.items():
            values = [year] if code == "Katsastusvuosi" else variable["values"]
            query.append(
                {"code": code, "selection": {"filter": "item", "values": values}}
            )
        payload = client.request_json(
            url=TRAFICOM_API,
            query={"query": query, "response": {"format": "json-stat2"}},
        )
        tables.append(
            parse_traficom_json(
                payload=payload, retrieved_at=client.last_acquired_at or utc_now()
            )
        )
    return validate_evidence(frame=pd.concat(objs=tables, ignore_index=True))


def fetch_adac(
    *, client: CachedClient, report_year: int = 2025, pdf_url: str = ADAC_2025_PDF
) -> pd.DataFrame:
    """Download and parse an official ADAC PDF, with explicit report year.

    Args:
        client: Cached HTTP client.
        report_year: Publication/report year.
        pdf_url: Verified official PDF URL.

    Returns:
        Model-by-registration-year breakdown rates.

    Raises:
        ValueError: If parsing produces implausibly little model evidence.
    """
    data = client.request_bytes(url=pdf_url)
    if not data.startswith(b"%PDF"):
        raise ValueError("ADAC endpoint did not return a PDF.")
    path = client.cache_dir / f"adac_report_{report_year}.pdf"
    path.write_bytes(data)
    frame = parse_adac_pdf(
        path=path,
        report_year=report_year,
        retrieved_at=client.last_acquired_at or utc_now(),
        source_url=pdf_url,
    )
    if frame.model_id.nunique() < 100:
        raise ValueError(
            "ADAC parser found fewer than 100 model series; refusing a partial refresh."
        )
    return frame


def fetch_tuv(*, client: CachedClient, years: Sequence[int]) -> pd.DataFrame:
    """Download complete six-age TÜV tables for the requested report years.

    Args:
        client: Cached HTTP client.
        years: Explicit report years; no speculative URL years are auto-added.

    Returns:
        Combined validated inspection evidence.

    Raises:
        ValueError: If requested years are empty or a source table has changed.
    """
    if not years:
        raise ValueError("At least one TÜV report year is required.")
    tables = []
    for year in years:
        for age in TUV_AGES:
            url = f"https://car-recalls.eu/reliability/reliability-tuv-report-{year}-{age}-years/"
            html = client.request_bytes(url=url).decode(encoding="utf-8")
            tables.append(
                parse_tuv_html(
                    html=html,
                    report_year=year,
                    age_group=age,
                    retrieved_at=client.last_acquired_at or utc_now(),
                )
            )
    return validate_evidence(frame=pd.concat(objs=tables, ignore_index=True))


def replace_evidence(
    *, original: pd.DataFrame, replacement: pd.DataFrame
) -> pd.DataFrame:
    """Replace only the source/report pairs present in a validated import.

    Args:
        original: Existing complete evidence.
        replacement: Newly validated full source/report tables.

    Returns:
        Combined evidence preserving unrelated sources and historical reports.
    """
    replacement = validate_evidence(frame=replacement)
    if replacement.empty:
        return validate_evidence(frame=original)
    pairs = set(zip(replacement.source_id, replacement.report_year, strict=True))
    keep = [
        (source, year) not in pairs
        for source, year in zip(original.source_id, original.report_year, strict=True)
    ]
    return validate_evidence(
        frame=pd.concat(objs=[original.loc[keep], replacement], ignore_index=True)
    )


def refresh_sources(
    *,
    directory: Path,
    sources: Sequence[str] = ("adac", "traficom"),
    offline: bool = False,
    force: bool = False,
    tuv_years: Sequence[int] = (2024, 2025),
) -> tuple[pd.DataFrame, Path]:
    """Commit successful refreshes and record failures without losing previous data.

    Args:
        directory: Local dataset folder.
        sources: Explicit refreshable source IDs.
        offline: Parse only verified cached responses.
        force: Re-download previously cached responses.
        tuv_years: Explicit TÜV report years when TÜV refresh is selected.

    Returns:
        Source status table and saved complete snapshot directory.

    Raises:
        ValueError: For unknown/empty source selections.
    """
    if not sources or set(sources) - {"tuv", "adac", "traficom"}:
        raise ValueError(
            "Refresh sources must be a non-empty selection of tuv, adac, traficom."
        )
    evidence, events, _ = load_snapshot(directory=directory)
    client = CachedClient(cache_dir=directory / "cache", offline=offline, refresh=force)
    statuses = []
    for source in dict.fromkeys(sources):
        try:
            if source == "adac":
                imported = fetch_adac(client=client)
            elif source == "traficom":
                imported = fetch_traficom(client=client)
            else:
                imported = fetch_tuv(client=client, years=tuv_years)
            evidence = replace_evidence(original=evidence, replacement=imported)
            statuses.append(
                {
                    "source_id": source,
                    "status": "updated",
                    "rows": len(imported),
                    "checked_at": utc_now(),
                    "detail": "Complete parsed source tables saved.",
                }
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            LOGGER.exception(
                "Refresh failed for %s; preserving existing source evidence", source
            )
            statuses.append(
                {
                    "source_id": source,
                    "status": "failed",
                    "rows": 0,
                    "checked_at": utc_now(),
                    "detail": f"{type(error).__name__}: {error}. Previous observations retained.",
                }
            )
    status = pd.DataFrame.from_records(data=statuses)
    snapshot = save_snapshot(
        directory=directory, evidence=evidence, events=events, status=status
    )
    return status, snapshot


def nhtsa_lookup(
    *,
    directory: Path,
    make: str,
    model: str,
    model_year: int,
    event_types: Sequence[str] = ("recall", "complaint"),
    offline: bool = False,
    force: bool = False,
) -> tuple[pd.DataFrame, Path]:
    """Query exact US vehicle names and persist a complete successful lookup.

    Args:
        directory: Local dataset folder.
        make: US-market make; explicitly supplied by the user.
        model: Exact US-market model.
        model_year: US model year, not a production-year estimate.
        event_types: recall and/or complaint.
        offline: Use verified cached API responses only.
        force: Refresh cached responses.

    Returns:
        Safety events for this query and the saved dataset snapshot.

    Raises:
        ValueError: For invalid names, year, event selection or API response.
        OSError: If a required API call fails; previous results stay intact.
    """
    if (
        not make.strip()
        or not model.strip()
        or not 1950 <= model_year <= datetime_year_limit()
    ):
        raise ValueError("Supply non-empty US make/model names and a valid model year.")
    if not event_types or set(event_types) - {"recall", "complaint"}:
        raise ValueError("Select recall and/or complaint events.")
    client = CachedClient(cache_dir=directory / "cache", offline=offline, refresh=force)
    tables = []
    for kind in dict.fromkeys(event_types):
        suffix = (
            "recalls/recallsByVehicle"
            if kind == "recall"
            else "complaints/complaintsByVehicle"
        )
        url = (
            NHTSA_BASE
            + "/"
            + suffix
            + "?"
            + urlencode(
                query={
                    "make": make.strip(),
                    "model": model.strip(),
                    "modelYear": model_year,
                }
            )
        )
        payload = client.request_json(url=url)
        tables.append(
            parse_nhtsa_json(
                payload=payload,
                event_type=kind,
                make=make.strip(),
                model=model.strip(),
                model_year=model_year,
                source_url=url,
                retrieved_at=client.last_acquired_at or utc_now(),
            )
        )
    fresh = pd.concat(objs=tables, ignore_index=True) if tables else empty_events()
    evidence, events, _ = load_snapshot(directory=directory)
    replace = (
        events.make.astype(str).str.casefold().eq(make.strip().casefold())
        & events.model.astype(str).str.casefold().eq(model.strip().casefold())
        & pd.to_numeric(arg=events.model_year, errors="coerce").eq(model_year)
        & events.event_type.isin(event_types)
    )
    merged = pd.concat(objs=[events.loc[~replace], fresh], ignore_index=True)
    status = pd.DataFrame(
        data=[
            {
                "source_id": "nhtsa",
                "status": "queried",
                "rows": len(fresh),
                "checked_at": utc_now(),
                "detail": f"US {make} {model} {model_year}: {', '.join(event_types)}; no records also counts as a successful query, not proof of no safety defects.",
            }
        ]
    )
    snapshot = save_snapshot(
        directory=directory, evidence=evidence, events=merged, status=status
    )
    return fresh, snapshot


def datetime_year_limit() -> int:
    """Return the latest reasonable announced model year."""
    return int(utc_now()[:4]) + 1
