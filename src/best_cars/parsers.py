"""Source adapters for actual published tables and documented APIs."""

import itertools
import logging
import re
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from best_cars.models import MAKES, identify_model
from best_cars.schema import (
    EVENT_COLUMNS,
    as_number,
    empty_events,
    validate_evidence,
)

LOGGER = logging.getLogger(__name__)
TRAFICOM_API = "https://trafi2.stat.fi/PXWeb/api/v1/en/TraFi/Katsastuksen_vikatilastot/040_kats_tau_104.px"
TRAFICOM_PAGE = "https://tieto.traficom.fi/en/statistics/statistics-inspections"
ADAC_2025_PDF = "https://assets.adac.de/image/upload/v1744788916/ADAC-eV/KOR/Text/PDF/Pannenstatistik_2025_final_nrmvls.pdf"
NHTSA_BASE = "https://api.nhtsa.gov"
MOT_PAGE = "https://www.data.gov.uk/dataset/c63fca52-ae4c-4b75-bab5-8b4735e1a4c9/anonymised-mot-tests-and-results"


def evidence_row(
    *,
    source_id: str,
    source_name: str,
    evidence_type: str,
    country: str,
    report_year: int,
    observation_year: int,
    cohort_start: int,
    cohort_end: int,
    age_min: int,
    age_max: int,
    label: str,
    value: float,
    metric: str,
    unit: str,
    source_url: str,
    retrieved_at: str,
    comparison_group: str,
    make_hint: str | None = None,
    sample_size: float = np.nan,
    event_count: float = np.nan,
    mileage_km: float = np.nan,
    mileage_basis: str = "unavailable",
    source_rank: float = np.nan,
    notes: str = "",
) -> dict:
    """Build a canonical record without inventing unpublished sample sizes.

    Args:
        source_id: Stable adapter/source identifier.
        source_name: Human-readable evidence source.
        evidence_type: inspection, breakdown or mot.
        country: Observed market, not the user's country.
        report_year: Publication/report year.
        observation_year: Year during which tests/breakdowns occurred.
        cohort_start: Earliest registration year in the cohort.
        cohort_end: Latest registration year in the cohort.
        age_min: Minimum observed age.
        age_max: Maximum observed age.
        label: Original model label.
        value: Published metric value.
        metric: Metric identifier.
        unit: percent or per_1000_vehicles.
        source_url: Exact source location.
        retrieved_at: ISO timestamp for acquisition.
        comparison_group: Comparable source stratum.
        make_hint: Explicit make for structured inputs.
        sample_size: Known test denominator, otherwise NaN.
        event_count: Known numerator, otherwise NaN.
        mileage_km: Published mileage, otherwise NaN.
        mileage_basis: Description of the mileage measurement.
        source_rank: Published position, otherwise NaN.
        notes: Source qualifications.

    Returns:
        A canonical evidence dictionary.
    """
    identity = identify_model(label=label, make_hint=make_hint)
    return {
        "source_id": source_id,
        "source_name": source_name,
        "evidence_type": evidence_type,
        "country": country,
        "report_year": report_year,
        "observation_year": observation_year,
        "cohort_year_start": cohort_start,
        "cohort_year_end": cohort_end,
        "age_min": age_min,
        "age_max": age_max,
        "source_model": f"{make_hint} {label}" if make_hint else label,
        "model_id": identity.model_id,
        "make": identity.make,
        "model": identity.model,
        "model_grouped": identity.grouped,
        "metric": metric,
        "value": value,
        "unit": unit,
        "sample_size": sample_size,
        "event_count": event_count,
        "mileage_km": mileage_km,
        "mileage_basis": mileage_basis,
        "source_rank": source_rank,
        "comparison_group": comparison_group,
        "source_url": source_url,
        "retrieved_at": retrieved_at,
        "notes": notes,
    }


def parse_tuv_tsv(*, frame: pd.DataFrame, retrieved_at: str) -> pd.DataFrame:
    """Convert the repository's historical TÜV TSV to canonical evidence.

    Args:
        frame: Existing 2024/2025 table, with year, age_group and faults_pct.
        retrieved_at: Acquisition timestamp, distinct from the report year.

    Returns:
        Validated inspection evidence, retaining raw rates and source ranks.

    Raises:
        ValueError: For malformed ages, columns or numerical data.
    """
    required = {
        "year",
        "age_group",
        "rank",
        "make_model",
        "faults_pct",
        "odometer_thousand_km",
    }
    if not required.issubset(frame.columns):
        raise ValueError(
            f"TÜV input lacks columns: {sorted(required - set(frame.columns))}"
        )
    records = []
    for record in frame.to_dict(orient="records"):
        match = re.fullmatch(r"(\d+)-(\d+)", str(record["age_group"]))
        if not match:
            raise ValueError(f"Invalid TÜV age group: {record['age_group']}")
        lower, upper = int(match.group(1)), int(match.group(2))
        year = int(record["year"])
        records.append(
            evidence_row(
                source_id="tuv",
                source_name="TÜV (Car-Recalls.eu tables)",
                evidence_type="inspection",
                country="Germany",
                report_year=year,
                observation_year=year - 1,
                cohort_start=year - 1 - upper,
                cohort_end=year - 1 - lower,
                age_min=lower,
                age_max=upper,
                label=str(record["make_model"]),
                value=as_number(value=record["faults_pct"]),
                metric="significant_defects_pct",
                unit="percent",
                source_url=f"https://car-recalls.eu/reliability/reliability-tuv-report-{year}-{lower}-{upper}-years/",
                retrieved_at=retrieved_at,
                comparison_group=f"age_{lower}_{upper}",
                mileage_km=as_number(
                    value=record["odometer_thousand_km"], allow_missing=True
                )
                * 1000,
                mileage_basis="mean_odometer",
                source_rank=as_number(value=record["rank"]),
                notes="Secondary transcription of TÜV inspection data. Registration-year range is approximate; not verified production years. Model sample size not published in these tables.",
            )
        )
    return validate_evidence(frame=pd.DataFrame.from_records(data=records))


def parse_tuv_html(
    *, html: str, report_year: int, age_group: str, retrieved_at: str
) -> pd.DataFrame:
    """Read the publisher's vertical TÜV table, preserving blank tied ranks.

    Args:
        html: Cached or downloaded HTML with a real reliability table.
        report_year: Published TÜV report year.
        age_group: Published age range, e.g. 2-3.
        retrieved_at: Acquisition timestamp.

    Returns:
        Validated canonical evidence.

    Raises:
        ValueError: If the table header or row structure has changed.
    """
    soup = BeautifulSoup(markup=html, features="html.parser")
    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in soup.get_text(separator="\n").replace("\xa0", " ").splitlines()
        if line.strip()
    ]
    start = None
    for index, line in enumerate(lines):
        if line.lower() != "order":
            continue
        window = lines[index : index + 10]
        headers = [
            next((i for i, s in enumerate(window) if token in s.lower()), -1)
            for token in ["order", "make/model", "faults", "odometer"]
        ]
        if -1 not in headers and headers == sorted(headers):
            start = index + headers[-1] + 1
            break
    if start is None:
        raise ValueError("TÜV table header not found; HTML was not imported.")
    rows = []
    last_rank = None
    index = start
    while index < len(lines):
        if re.match(
            r"^(Source:|Subscribe|Car Recalls Family|// Featured)",
            lines[index],
            flags=re.I,
        ):
            break
        if re.fullmatch(r"\d{1,3}", lines[index]):
            last_rank = int(lines[index])
            index += 1
        if index + 2 >= len(lines):
            break
        label, faults, mileage = lines[index : index + 3]
        if (
            last_rank is None
            or not re.fullmatch(r"\d+(?:[.,]\d+)?", faults)
            or not re.fullmatch(r"\d+", mileage)
        ):
            raise ValueError(
                f"Malformed TÜV table row near {label!r}; no partial table saved."
            )
        rows.append(
            {
                "year": report_year,
                "age_group": age_group,
                "rank": last_rank,
                "make_model": label,
                "faults_pct": faults,
                "odometer_thousand_km": mileage,
            }
        )
        index += 3
    if len(rows) < 5:
        raise ValueError("TÜV table contained too few model rows.")
    return parse_tuv_tsv(
        frame=pd.DataFrame.from_records(data=rows), retrieved_at=retrieved_at
    )


def jsonstat_records(*, payload: Mapping) -> list[dict]:
    """Decode labelled JSON-stat2 dimensions, including sparse and suppressed cells.

    Args:
        payload: JSON-stat2 dataset returned by the official PxWeb API.

    Returns:
        Records with one labelled dimension per column plus value.

    Raises:
        ValueError: If dimensions or values have inconsistent sizes.
    """
    identifiers = payload.get("id", [])
    sizes = payload.get("size", [])
    if not identifiers or len(identifiers) != len(sizes):
        raise ValueError("Invalid JSON-stat dimension metadata.")
    axes = []
    for code, size in zip(identifiers, sizes, strict=True):
        category = payload["dimension"][code]["category"]
        index = category["index"]
        ordered = (
            index
            if isinstance(index, list)
            else [key for key, _ in sorted(index.items(), key=lambda item: item[1])]
        )
        if len(ordered) != size:
            raise ValueError("JSON-stat dimension size differs from its index.")
        labels = category.get("label", {})
        axes.append([labels.get(value, value) for value in ordered])
    values = payload.get("value", [])
    expected = int(np.prod(sizes))
    if isinstance(values, list) and len(values) != expected:
        raise ValueError("JSON-stat cell count differs from its dimensions.")
    records = []
    for index, combination in enumerate(itertools.product(*axes)):
        value = values.get(str(index)) if isinstance(values, dict) else values[index]
        record = dict(zip(identifiers, combination, strict=True))
        record["value"] = as_number(value=value, allow_missing=True)
        records.append(record)
    return records


def parse_traficom_json(*, payload: Mapping, retrieved_at: str) -> pd.DataFrame:
    """Parse official model/registration-year inspection rates and denominators.

    Args:
        payload: PxWeb 040_kats_tau_104.px JSON-stat2 response.
        retrieved_at: Acquisition timestamp.

    Returns:
        Model-specific Finnish inspection evidence; totals and suppression excluded.

    Raises:
        ValueError: If API fields or model labels change.
    """
    raw = pd.DataFrame.from_records(data=jsonstat_records(payload=payload))
    dimensions = [
        "Katsastusvuosi",
        "Merkki ja mallisarja",
        "Käyttöönottovuosi",
        "Tiedot",
    ]
    if not set(dimensions + ["value"]).issubset(raw.columns):
        raise ValueError("Traficom API dimensions have changed.")
    wide = raw.pivot(
        index=dimensions[:3], columns="Tiedot", values="value"
    ).reset_index()
    required = {"Number of periodic inspections", "Fail-rate", "Failed"}
    if not required.issubset(wide.columns):
        raise ValueError("Traficom response is missing rates or test denominators.")
    records = []
    for item in wide.to_dict(orient="records"):
        label = item["Merkki ja mallisarja"]
        cohort = str(item["Käyttöönottovuosi"])
        if "total" in label.lower() or not cohort.isdigit():
            continue
        count = item["Number of periodic inspections"]
        if pd.isna(count) or count <= 0 or pd.isna(item["Fail-rate"]):
            continue
        year = int(item["Katsastusvuosi"])
        cohort_year = int(cohort)
        records.append(
            evidence_row(
                source_id="traficom",
                source_name="Traficom inspections",
                evidence_type="inspection",
                country="Finland",
                report_year=year + 1,
                observation_year=year,
                cohort_start=cohort_year,
                cohort_end=cohort_year,
                age_min=year - cohort_year,
                age_max=year - cohort_year,
                label=label.replace(" - ", " "),
                value=item["Fail-rate"],
                metric="inspection_failure_pct",
                unit="percent",
                sample_size=count,
                event_count=item["Failed"],
                mileage_km=item.get("Average mileage", np.nan),
                mileage_basis="mean_odometer",
                source_url=TRAFICOM_API,
                retrieved_at=retrieved_at,
                comparison_group=f"registration_{cohort_year}",
                notes="Finnish periodic inspections; model-series labels may cover several generations. Denominators are tests, not necessarily distinct cars.",
            )
        )
    if not records:
        raise ValueError("Traficom returned no usable model-level inspection evidence.")
    return validate_evidence(frame=pd.DataFrame.from_records(data=records))


def parse_traficom_xlsx(
    *, path: Path, observation_year: int, retrieved_at: str
) -> pd.DataFrame:
    """Import Traficom's published workbook, retaining common fault descriptions.

    Args:
        path: Original Finnish XLSX downloaded from Traficom.
        observation_year: Year of inspection.
        retrieved_at: Acquisition timestamp.

    Returns:
        Model-specific inspection evidence with published sample sizes.

    Raises:
        ValueError: If the documented workbook columns cannot be located.
    """
    raw = pd.read_excel(io=path, header=None)
    header_rows = raw.index[
        raw.iloc[:, 0].astype(str).str.strip().eq("Käyttöönottovuosi")
    ]
    if len(header_rows) != 1:
        raise ValueError("Traficom workbook header could not be located uniquely.")
    records = []
    for values in raw.iloc[int(header_rows[0]) + 1 :].itertuples(
        index=False, name=None
    ):
        cohort, make, model, count, rate, mean_mileage, _median, *faults = values
        if pd.isna(cohort) or not str(cohort).replace(".0", "").isdigit():
            continue
        if pd.isna(make) or pd.isna(model) or "yhteensä" in str(model):
            continue
        cohort_year = int(cohort)
        records.append(
            evidence_row(
                source_id="traficom",
                source_name="Traficom inspections",
                evidence_type="inspection",
                country="Finland",
                report_year=observation_year + 1,
                observation_year=observation_year,
                cohort_start=cohort_year,
                cohort_end=cohort_year,
                age_min=observation_year - cohort_year,
                age_max=observation_year - cohort_year,
                make_hint=str(make),
                label=str(model),
                value=as_number(value=rate),
                metric="inspection_failure_pct",
                unit="percent",
                sample_size=as_number(value=count),
                mileage_km=as_number(value=mean_mileage, allow_missing=True),
                mileage_basis="mean_odometer",
                source_url=TRAFICOM_PAGE,
                retrieved_at=retrieved_at,
                comparison_group=f"registration_{cohort_year}",
                notes="Common failure reasons (original Finnish): "
                + "; ".join(str(f).replace("\n", " ") for f in faults if pd.notna(f)),
            )
        )
    if not records:
        raise ValueError("Traficom workbook contained no usable model records.")
    return validate_evidence(frame=pd.DataFrame.from_records(data=records))


def parse_adac_words(
    *,
    pages: list[list[dict]],
    report_year: int,
    retrieved_at: str,
    source_url: str,
) -> pd.DataFrame:
    """Extract ADAC rates by PDF cell coordinates, keeping empty years empty.

    Args:
        pages: Words returned by pdfplumber, with x0, x1 and top coordinates.
        report_year: Published report year; its observation year is one earlier.
        retrieved_at: Acquisition timestamp.
        source_url: Exact downloaded PDF URL.

    Returns:
        Breakdown rates by first registration year. Annual mileage is labelled.

    Raises:
        ValueError: If tables are absent, model rows ambiguous or cells malformed.
    """
    records = []
    previous_layout = None
    for words in pages:
        year_lines = defaultdict(list)
        for word in words:
            if re.fullmatch(r"20\d{2}", word["text"]):
                year_lines[round(word["top"] / 3) * 3].append(word)
        headers = []
        for top, year_words in sorted(year_lines.items()):
            mileage = [
                w
                for w in words
                if w["text"] == "Laufleistung" and abs(w["top"] - top) < 15
            ]
            if len(year_words) == 8 and mileage:
                headers.append(
                    (top, sorted(year_words, key=lambda w: w["x0"]), mileage[0]["x0"])
                )
        if previous_layout is not None:
            _, old_years, old_right = previous_layout
            # A table continued on the next page retains the same PDF columns.
            headers.insert(0, (0, [{**w, "top": 0} for w in old_years], old_right))
        last_model_y = 0
        last_layout = None
        for header_index, (top, years, label_right) in enumerate(headers):
            if len(years) != 8 or [int(w["text"]) for w in years] != list(
                range(report_year - 3, report_year - 11, -1)
            ):
                continue
            centres = [(w["x0"] + w["x1"]) / 2 for w in years]
            width = float(np.median(np.diff(centres)))
            left_edge = centres[0] - width / 2
            right_edge = centres[-1] + width / 2
            upper = (
                headers[header_index + 1][0] - 35
                if header_index + 1 < len(headers)
                else 770
            )
            body = [
                w
                for w in words
                if w["top"] > max(w["top"] for w in years) + 4 and w["top"] < upper
            ]
            numeric_lines = defaultdict(list)
            for word in body:
                if left_edge <= (
                    word["x0"] + word["x1"]
                ) / 2 <= right_edge and re.fullmatch(r"\d+(?:[.,]\d+)?", word["text"]):
                    numeric_lines[round(word["top"] / 3) * 3].append(word)
            for line_y, cells in sorted(numeric_lines.items()):
                neighbours = sorted(numeric_lines)
                position = neighbours.index(line_y)
                prev_y = neighbours[position - 1] if position else top
                next_y = (
                    neighbours[position + 1]
                    if position + 1 < len(neighbours)
                    else line_y + 20
                )
                label_words = sorted(
                    [
                        w
                        for w in body
                        if w["x0"] < label_right
                        and max((prev_y + line_y) / 2, line_y - 12)
                        < w["top"]
                        <= min((line_y + next_y) / 2, line_y + 12)
                    ],
                    key=lambda w: (round(w["top"] / 3), w["x0"]),
                )
                # The annual mileage column lies between model label and rates.
                mileage_words = [
                    w
                    for w in label_words
                    if re.fullmatch(r"\d{1,2}", w["text"])
                    and w["x0"] > left_edge - 1.5 * width
                ]
                label_words = [w for w in label_words if w not in mileage_words]
                label = " ".join(w["text"] for w in label_words).strip()
                if not any(
                    label.casefold().startswith(make.casefold() + " ") for make in MAKES
                ):
                    continue
                last_model_y = line_y
                last_layout = (top, years, label_right)
                if len(label) > 140:
                    raise ValueError(f"Ambiguous ADAC PDF model label: {label!r}")
                used = set()
                for cell in cells:
                    centre = (cell["x0"] + cell["x1"]) / 2
                    column = min(range(8), key=lambda i: abs(centre - centres[i]))
                    if column in used:
                        raise ValueError(f"Ambiguous ADAC PDF rate cell for {label}.")
                    used.add(column)
                    cohort = int(years[column]["text"])
                    age = report_year - 1 - cohort
                    records.append(
                        evidence_row(
                            source_id="adac",
                            source_name="ADAC breakdown statistics",
                            evidence_type="breakdown",
                            country="Germany",
                            report_year=report_year,
                            observation_year=report_year - 1,
                            cohort_start=cohort,
                            cohort_end=cohort,
                            age_min=age,
                            age_max=age,
                            label=label,
                            value=as_number(value=cell["text"]),
                            metric="breakdowns_per_1000",
                            unit="per_1000_vehicles",
                            source_url=source_url,
                            retrieved_at=retrieved_at,
                            comparison_group=f"registration_{cohort}",
                            # Omit annual mileage from the common odometer field.
                            notes="ADAC-attended breakdowns per 1,000 registered vehicles; model denominators not published. Annual mileage of breakdown vehicles is not comparable to inspection odometer readings.",
                        )
                    )
        previous_layout = last_layout if last_model_y > 650 else None
    if not records:
        raise ValueError("ADAC PDF tables not found; no data were imported.")
    return validate_evidence(frame=pd.DataFrame.from_records(data=records))


def parse_adac_pdf(
    *, path: Path, report_year: int, retrieved_at: str, source_url: str = ADAC_2025_PDF
) -> pd.DataFrame:
    """Read a locally cached ADAC PDF into coordinate-based table extraction.

    Args:
        path: Official ADAC report PDF.
        report_year: Report year, not observation year.
        retrieved_at: Acquisition timestamp.
        source_url: Download URL used for provenance.

    Returns:
        Validated model/year roadside breakdown evidence.

    Raises:
        ValueError: If optional PDF support cannot load or report tables fail.
    """
    try:
        import pdfplumber
    except (ImportError, OSError) as error:
        LOGGER.exception("Optional ADAC PDF dependency could not load")
        raise ValueError(
            "ADAC PDF support could not load. Browsing saved data remains available. "
            "Install the project's [pdf] extra; for native cryptography/OpenSSL "
            "errors, follow docs/MAC_INSTALL.md."
        ) from error
    with pdfplumber.open(path_or_fp=path) as document:
        pages = [page.extract_words(x_tolerance=1) for page in document.pages]
    return parse_adac_words(
        pages=pages,
        report_year=report_year,
        retrieved_at=retrieved_at,
        source_url=source_url,
    )


def parse_nhtsa_json(
    *,
    payload: Mapping,
    event_type: str,
    make: str,
    model: str,
    model_year: int,
    source_url: str,
    retrieved_at: str,
) -> pd.DataFrame:
    """Parse NHTSA recalls/complaints as safety context, never failure rates.

    Args:
        payload: Official API response.
        event_type: recall or complaint.
        make: Explicit US-market make queried.
        model: Exact US model queried; no UK model alias inference.
        model_year: US model year queried.
        source_url: Exact query URL.
        retrieved_at: Acquisition timestamp.

    Returns:
        De-duplicated safety events, with no invented exposure denominator.

    Raises:
        ValueError: If the result envelope or identifiers are invalid.
    """
    if event_type not in {"recall", "complaint"}:
        raise ValueError("NHTSA event_type must be recall or complaint.")
    items = payload.get("results")
    count = payload.get("Count", payload.get("count"))
    if not isinstance(items, list) or count is None:
        raise ValueError("NHTSA returned an invalid result envelope.")
    if int(count) != len(items):
        raise ValueError("NHTSA result count does not match its response records.")
    if not items:
        return empty_events()
    rows = []
    for item in items:
        event_id = (
            item.get("NHTSACampaignNumber")
            if event_type == "recall"
            else item.get("odiNumber")
        )
        if not event_id:
            raise ValueError("NHTSA event identifier missing.")
        rows.append(
            {
                "source_id": f"nhtsa_{event_type}s",
                "country": "United States",
                "event_type": event_type,
                "event_id": str(event_id),
                "make": make,
                "model": model,
                "model_year": model_year,
                "date": str(
                    item.get("ReportReceivedDate", item.get("dateComplaintFiled", ""))
                ),
                "component": str(item.get("Component", item.get("components", ""))),
                "summary": str(item.get("Summary", item.get("summary", ""))),
                "consequence": str(item.get("Consequence", "")),
                "remedy": str(item.get("Remedy", "")),
                "source_url": source_url,
                "retrieved_at": retrieved_at,
            }
        )
    return (
        pd.DataFrame.from_records(data=rows, columns=EVENT_COLUMNS)
        .drop_duplicates(
            subset=["event_type", "event_id", "make", "model", "model_year"]
        )
        .reset_index(drop=True)
    )


def utc_now() -> str:
    """Return an ISO UTC acquisition timestamp."""
    return datetime.now(tz=UTC).isoformat(timespec="seconds")
