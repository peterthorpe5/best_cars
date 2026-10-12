"""Bounded-memory imports for genuine UK MOT tests and aggregate counts."""

import logging
import re
import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from best_cars.parsers import MOT_PAGE, evidence_row
from best_cars.schema import as_number, validate_evidence

LOGGER = logging.getLogger(__name__)
RAW_COLUMNS = [
    "test_id",
    "vehicle_id",
    "test_date",
    "test_class_id",
    "test_type",
    "test_result",
    "test_mileage",
    "postcode_area",
    "make",
    "model",
    "colour",
    "fuel_type",
    "cylinder_capacity",
    "first_use_date",
]
FIELD_ALIASES = {re.sub(r"[^a-z0-9]", "", name): name for name in RAW_COLUMNS}
FIELD_ALIASES.update({"testclassid": "test_class_id", "firstusedate": "first_use_date"})


def mot_raw_chunks(
    *, path: Path, delimiter: str, header: bool, chunksize: int
) -> Iterator[pd.DataFrame]:
    """Read bounded chunks from text, gzip or all results members of a ZIP.

    Args:
        path: Local official results file or results-only archive.
        delimiter: Explicit delimiter for this source release.
        header: Whether each source file includes column names.
        chunksize: Maximum records held in each chunk.

    Yields:
        Source records without converting identifiers or odometer text.

    Raises:
        ValueError: For invalid settings or a ZIP without results files.
    """
    if chunksize < 1 or delimiter not in {"\t", ",", "|", ";"}:
        raise ValueError("Require a positive chunk size and a supported delimiter.")
    with ExitStack() as archive_stack:
        archive = None
        members = [path]
        if path.suffix.lower() == ".zip":
            archive = archive_stack.enter_context(ZipFile(file=path))
            members = [
                name
                for name in archive.namelist()
                if not name.endswith("/")
                and Path(name).suffix.lower() in {".txt", ".tsv", ".csv"}
                and not name.startswith("__MACOSX/")
                and not Path(name).name.startswith("._")
            ]
            if not members:
                raise ValueError("MOT ZIP contains no delimited results files.")
        for member in members:
            with ExitStack() as member_stack:
                stream = (
                    member_stack.enter_context(archive.open(name=member))
                    if archive is not None
                    else member
                )
                reader = member_stack.enter_context(
                    pd.read_csv(
                        filepath_or_buffer=stream,
                        sep=delimiter,
                        dtype=str,
                        header=0 if header else None,
                        names=None if header else RAW_COLUMNS,
                        chunksize=chunksize,
                        keep_default_na=False,
                        encoding="utf-8-sig",
                    )
                )
                LOGGER.info("Reading MOT results member %s", member)
                yield from reader


def mot_test_dates(*, values: pd.Series) -> pd.Series:
    """Parse ISO and historical day-first MOT dates without swapping months.

    Args:
        values: Required source test or first-use dates.

    Returns:
        UTC timestamps suitable for date comparisons and registration cohorts.

    Raises:
        ValueError: For missing or malformed dates.
    """
    iso = values.astype(str).str.match(pat=r"^\d{4}-\d{2}-\d{2}(?:$|[T ])")
    result = pd.Series(data=pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    for mask, date_format in [(iso, "ISO8601"), (~iso, "mixed")]:
        if mask.any():
            result.loc[mask] = pd.to_datetime(
                arg=values.loc[mask],
                format=date_format,
                dayfirst=date_format == "mixed",
                errors="raise",
                utc=True,
            )
    if result.isna().any():
        raise ValueError("MOT test and first-use dates must not be missing.")
    return result


def parse_mot_aggregate(*, frame: pd.DataFrame, retrieved_at: str) -> pd.DataFrame:
    """Convert model/registration-year initial-test counts into failure evidence.

    Args:
        frame: TSV-style aggregates with make, model, observation_year,
            registration_year, tests and failures (F plus PRS).
        retrieved_at: Acquisition timestamp.

    Returns:
        Validated MOT failure rates with actual test denominators.

    Raises:
        ValueError: For malformed counts or missing required columns.
    """
    required = {
        "make",
        "model",
        "observation_year",
        "registration_year",
        "tests",
        "failures",
    }
    if not required.issubset(frame.columns):
        raise ValueError(
            f"MOT aggregates lack: {sorted(required - set(frame.columns))}"
        )
    rows = []
    for item in frame.to_dict(orient="records"):
        count = as_number(value=item["tests"])
        failures = as_number(value=item["failures"])
        if count <= 0 or failures > count or failures < 0 or count % 1 or failures % 1:
            raise ValueError("MOT counts require 0 <= failures <= tests and tests > 0.")
        raw_year = as_number(value=item["observation_year"])
        raw_cohort = as_number(value=item["registration_year"])
        if raw_year % 1 or raw_cohort % 1:
            raise ValueError("MOT observation/registration years must be integers.")
        year = int(raw_year)
        cohort = int(raw_cohort)
        age = year - cohort
        rows.append(
            evidence_row(
                source_id="mot",
                source_name="DVSA initial MOT tests",
                evidence_type="mot",
                country="Great Britain",
                report_year=year + 1,
                observation_year=year,
                cohort_start=cohort,
                cohort_end=cohort,
                age_min=age,
                age_max=age,
                make_hint=str(item["make"]),
                label=str(item["model"]),
                value=100 * failures / count,
                metric="initial_mot_failure_pct",
                unit="percent",
                sample_size=count,
                event_count=failures,
                source_url=MOT_PAGE,
                retrieved_at=retrieved_at,
                mileage_km=as_number(
                    value=item.get("mean_mileage_km"), allow_missing=True
                ),
                mileage_basis="mean_odometer",
                comparison_group=f"registration_{cohort}",
                notes="Class 4 normal initial tests only; F and PRS count as initial failures. Retests and aborted/abandoned tests excluded. Class 4 also includes some light goods and other vehicles. Source aggregates supplied locally.",
            )
        )
        rows[-1]["_raw_label"] = f"{item['make']} {item['model']}"
        mileage_tests = as_number(value=item.get("mileage_tests", count))
        if mileage_tests % 1 or not 0 <= mileage_tests <= count:
            raise ValueError(
                "MOT mileage counts must be integers between zero and tests."
            )
        rows[-1]["_mileage_tests"] = (
            mileage_tests if pd.notna(rows[-1]["mileage_km"]) else 0
        )
    if not rows:
        raise ValueError("MOT input contains no aggregate records.")
    return pool_mot_aliases(frame=pd.DataFrame.from_records(data=rows))


def pool_mot_aliases(*, frame: pd.DataFrame) -> pd.DataFrame:
    """Pool distinct raw labels sharing an explicitly normalised cohort identity.

    Args:
        frame: Parsed MOT rows with private raw-label and known-mileage counts.

    Returns:
        Canonical unique observations, summing counts and weighting odometers
        by the number of non-missing readings. Original labels are retained.

    Raises:
        ValueError: For repeated raw aggregates or invalid pooled evidence.
    """
    keys = [
        "source_id",
        "report_year",
        "observation_year",
        "comparison_group",
        "model_id",
        "metric",
    ]
    if not frame.duplicated(subset=keys).any():
        return validate_evidence(frame=frame)
    pooled = []
    for _key, group in frame.groupby(by=keys, sort=True, dropna=False):
        if group._raw_label.duplicated().any():
            raise ValueError("Duplicate raw MOT aggregates would double-count tests.")
        row = group.iloc[0].to_dict()
        row["sample_size"] = group.sample_size.sum()
        row["event_count"] = group.event_count.sum()
        row["value"] = 100 * row["event_count"] / row["sample_size"]
        readings = group.loc[group.mileage_km.notna() & group._mileage_tests.gt(0)]
        row["mileage_km"] = (
            (readings.mileage_km * readings._mileage_tests).sum()
            / readings._mileage_tests.sum()
            if len(readings)
            else float("nan")
        )
        if len(group) > 1:
            row["source_model"] = " | ".join(group._raw_label)
            row["notes"] += (
                " Counts pooled only across explicit canonical label aliases; original raw labels retained."
            )
        pooled.append(row)
    return validate_evidence(frame=pd.DataFrame.from_records(data=pooled))


def aggregate_mot_raw(
    *,
    path: Path,
    retrieved_at: str,
    delimiter: str = "\t",
    header: bool = True,
    chunksize: int = 100_000,
    conflict_policy: str = "error",
) -> pd.DataFrame:
    """Stream official MOT records, deduplicating test IDs in temporary SQLite.

    Args:
        path: Original delimited text, .gz or ZIP containing results files.
        retrieved_at: Acquisition timestamp.
        delimiter: Explicit input delimiter; output is always TSV.
        header: Whether the source file has column names.
        chunksize: Maximum rows held in memory per source chunk.
        conflict_policy: Stop on conflicting eligible test IDs by default;
            "exclude" quarantines every detected conflicting ID entirely.

    Returns:
        Canonical age-cohort failure rates from all imported initial tests.

    Raises:
        ValueError: For unsupported file structure, unknown normal-test outcomes,
            conflicting duplicate test IDs, invalid dates or no eligible rows.
    """
    if (
        chunksize < 1
        or delimiter not in {"\t", ",", "|", ";"}
        or conflict_policy not in {"error", "exclude"}
    ):
        raise ValueError(
            "Require a positive chunk size and a supported input delimiter."
        )
    with tempfile.TemporaryDirectory(prefix="best_cars_mot_") as temporary:
        with sqlite3.connect(
            database=str(Path(temporary) / "tests.sqlite")
        ) as database:
            database.execute(
                "CREATE TABLE tests (test_id TEXT PRIMARY KEY, make TEXT, model TEXT, year INTEGER, cohort INTEGER, failed INTEGER, mileage REAL)"
            )
            database.execute("CREATE TABLE quarantined (test_id TEXT PRIMARY KEY)")
            for chunk_number, chunk in enumerate(
                iterable=mot_raw_chunks(
                    path=path, delimiter=delimiter, header=header, chunksize=chunksize
                ),
                start=1,
            ):
                chunk = chunk.rename(
                    columns={
                        name: FIELD_ALIASES.get(
                            re.sub(r"[^a-z0-9]", "", str(name).lower()),
                            str(name).lower(),
                        )
                        for name in chunk.columns
                    }
                )
                required = {
                    "test_id",
                    "test_class_id",
                    "test_type",
                    "test_result",
                    "test_date",
                    "first_use_date",
                    "make",
                    "model",
                }
                if not required.issubset(chunk.columns):
                    raise ValueError(
                        f"MOT raw file lacks: {sorted(required - set(chunk.columns))}"
                    )
                eligible = chunk.loc[
                    chunk.test_class_id.str.strip().eq("4")
                    & chunk.test_type.str.strip().eq("NT")
                ].copy()
                allowed = {"P", "F", "PRS", "ABA", "ABR"}
                if set(eligible.test_result.str.strip()) - allowed:
                    raise ValueError(
                        "Unexpected MOT outcome code; refusing to guess its meaning."
                    )
                eligible = eligible.loc[
                    eligible.test_result.str.strip().isin(["P", "F", "PRS"])
                ]
                eligible["test_result"] = eligible.test_result.str.strip()
                if eligible.empty:
                    continue
                dates = mot_test_dates(values=eligible.test_date)
                first = mot_test_dates(values=eligible.first_use_date)
                if (first > dates).any():
                    raise ValueError("MOT first-use date occurs after its test date.")
                eligible["year"] = dates.dt.year
                eligible["cohort"] = first.dt.year
                for item in eligible.to_dict(orient="records"):
                    if (
                        not item["test_id"].strip()
                        or not item["make"].strip()
                        or not item["model"].strip()
                    ):
                        raise ValueError(
                            "Eligible MOT records need test IDs, make and model."
                        )
                    mileage = as_number(
                        value=item.get("test_mileage"), allow_missing=True
                    )
                    # Zero readings are missing; earlier releases may mix units.
                    mileage = (
                        mileage * 1.609344
                        if item["year"] >= 2022 and mileage > 0
                        else None
                    )
                    record = (
                        item["test_id"],
                        item["make"],
                        item["model"],
                        int(item["year"]),
                        int(item["cohort"]),
                        int(item["test_result"] in {"F", "PRS"}),
                        mileage,
                    )
                    if (
                        database.execute(
                            "SELECT 1 FROM quarantined WHERE test_id = ?", (record[0],)
                        ).fetchone()
                        is not None
                    ):
                        continue
                    previous = database.execute(
                        "SELECT * FROM tests WHERE test_id = ?", (record[0],)
                    ).fetchone()
                    if previous is not None:
                        if previous != record:
                            if conflict_policy == "error":
                                raise ValueError(
                                    f"Conflicting duplicate MOT test ID: {record[0]}. "
                                    "Audit the source or explicitly select the exclusion policy."
                                )
                            database.execute(
                                "DELETE FROM tests WHERE test_id = ?", (record[0],)
                            )
                            database.execute(
                                "INSERT INTO quarantined VALUES (?)", (record[0],)
                            )
                    else:
                        database.execute(
                            "INSERT INTO tests VALUES (?, ?, ?, ?, ?, ?, ?)", record
                        )
                database.commit()
                LOGGER.info(
                    "Processed MOT chunk %s (%s eligible normal tests)",
                    chunk_number,
                    len(eligible),
                )
            aggregates = pd.read_sql_query(
                sql="SELECT make, model, year AS observation_year, cohort AS registration_year, COUNT(*) AS tests, SUM(failed) AS failures, AVG(mileage) AS mean_mileage_km, COUNT(mileage) AS mileage_tests FROM tests GROUP BY make, model, year, cohort",
                con=database,
            )
            quarantined = database.execute(
                "SELECT COUNT(*) FROM quarantined"
            ).fetchone()[0]
    result = parse_mot_aggregate(frame=aggregates, retrieved_at=retrieved_at)
    result.attrs["mot_import_audit"] = {
        "conflicting_eligible_test_ids_excluded": quarantined,
        "conflict_policy": conflict_policy,
    }
    if quarantined:
        LOGGER.warning(
            "Excluded %s conflicting eligible MOT test IDs; selection bias may remain",
            quarantined,
        )
        result["notes"] += (
            f" Import audit: {quarantined} conflicting eligible test IDs excluded entirely; exclusion may introduce selection bias."
        )
    return result
