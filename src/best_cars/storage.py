"""Versioned, checksummed TSV snapshots outside the software checkout."""

import hashlib
import io
import json
import logging
import os
import uuid
from importlib.resources import files
from pathlib import Path

import pandas as pd

from best_cars.parsers import parse_tuv_tsv, utc_now
from best_cars.schema import EVENT_COLUMNS, empty_events, validate_evidence

LOGGER = logging.getLogger(__name__)


def data_directory(*, supplied: str | Path | None = None) -> Path:
    """Resolve an explicit data folder or BEST_CARS_DATA_DIR/default home folder.

    Args:
        supplied: Explicit local data directory, when set.

    Returns:
        Absolute expanded folder path; no folder is created here.
    """
    value = (
        supplied
        or os.environ.get("BEST_CARS_DATA_DIR")
        or Path.home() / "best_cars_data"
    )
    return Path(value).expanduser().resolve()


def baseline_evidence() -> pd.DataFrame:
    """Load the real historical TÜV snapshot shipped with the package."""
    source = files("best_cars").joinpath("data/tuv_baseline.tsv")
    with source.open(mode="rb") as stream:
        legacy = pd.read_csv(filepath_or_buffer=stream, sep="\t")
    return parse_tuv_tsv(frame=legacy, retrieved_at="2025-09-04T16:21:41+00:00")


def source_catalogue() -> pd.DataFrame:
    """Load the audited source catalogue, including availability limitations."""
    with files("best_cars").joinpath("data/sources.tsv").open(mode="rb") as stream:
        return pd.read_csv(filepath_or_buffer=stream, sep="\t").fillna(value="")


def tsv_bytes(*, frame: pd.DataFrame) -> bytes:
    """Return a UTF-8 TSV download with a header and no index.

    Args:
        frame: Table to export.

    Returns:
        Tab-separated bytes, never comma-separated output.
    """
    return frame.to_csv(sep="\t", index=False, lineterminator="\n").encode(
        encoding="utf-8"
    )


def excel_bytes(*, sheets: dict[str, pd.DataFrame]) -> bytes:
    """Export formatted worksheets with frozen headers and filters.

    Args:
        sheets: Worksheet titles mapped to data tables.

    Returns:
        A complete Excel workbook suitable for download.

    Raises:
        ValueError: If no worksheets or invalid worksheet names are supplied.
    """
    if not sheets:
        raise ValueError("At least one worksheet is required.")
    for name in sheets:
        if not name or len(name) > 31 or any(c in name for c in "[]:*?/\\"):
            raise ValueError(f"Invalid worksheet name: {name!r}")
    buffer = io.BytesIO()
    with pd.ExcelWriter(path=buffer, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(
                excel_writer=writer, sheet_name=name, index=False, freeze_panes=(1, 0)
            )
            sheet = writer.sheets[name]
            sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                letter = column[0].column_letter
                sheet.column_dimensions[letter].width = min(
                    55, max(12, max(len(str(c.value or "")) for c in column[:100]) + 2)
                )
    return buffer.getvalue()


def save_snapshot(
    *,
    directory: Path,
    evidence: pd.DataFrame,
    events: pd.DataFrame,
    status: pd.DataFrame | None = None,
) -> Path:
    """Validate a complete snapshot and atomically advance its current pointer.

    Args:
        directory: Data directory outside the software repository.
        evidence: Canonical reliability observations.
        events: Canonical safety events, including empty tables.
        status: Optional refresh/import status table.

    Returns:
        Newly committed snapshot directory; previous snapshots remain intact.

    Raises:
        ValueError: If evidence or safety-event columns are invalid.
        OSError: If writing the snapshot fails; previous pointer stays unchanged.
    """
    evidence = validate_evidence(frame=evidence)
    if not set(EVENT_COLUMNS).issubset(events.columns):
        raise ValueError("Safety events do not have the canonical columns.")
    snapshot = (
        directory
        / "snapshots"
        / f"{utc_now().replace(':', '').replace('+', '_')}_{uuid.uuid4().hex[:8]}"
    )
    snapshot.mkdir(parents=True, exist_ok=False)
    paths = {
        "evidence": snapshot / "evidence.tsv.gz",
        "events": snapshot / "events.tsv.gz",
    }
    evidence.to_csv(
        path_or_buf=paths["evidence"], sep="\t", index=False, compression="gzip"
    )
    events.loc[:, EVENT_COLUMNS].to_csv(
        path_or_buf=paths["events"], sep="\t", index=False, compression="gzip"
    )
    if status is not None:
        status.to_csv(path_or_buf=snapshot / "source_status.tsv", sep="\t", index=False)
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "evidence_rows": len(evidence),
        "event_rows": len(events),
        "source_counts": evidence.source_id.value_counts().to_dict(),
        "files": {
            key: {
                "name": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for key, path in paths.items()
        },
    }
    (snapshot / "manifest.json").write_text(
        data=json.dumps(obj=manifest, indent=2) + "\n"
    )
    pointer = {"snapshot": os.path.relpath(path=snapshot, start=directory)}
    temp = directory / f"current_{uuid.uuid4().hex}.tmp"
    temp.write_text(data=json.dumps(obj=pointer) + "\n")
    temp.replace(target=directory / "current.json")
    LOGGER.info(
        "Saved %s evidence rows and %s safety events to %s",
        len(evidence),
        len(events),
        snapshot,
    )
    return snapshot


def load_snapshot(*, directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load a verified current snapshot or the real bundled TÜV baseline.

    Args:
        directory: Local dataset directory.

    Returns:
        Evidence, safety events and a snapshot manifest.

    Raises:
        OSError: If a saved snapshot is absent/corrupt or points outside the data folder.
        ValueError: If stored evidence fails validation.
    """
    pointer_path = directory / "current.json"
    if not pointer_path.exists():
        baseline = baseline_evidence()
        return (
            baseline,
            empty_events(),
            {
                "baseline_only": True,
                "evidence_rows": len(baseline),
                "created_at": "2025-09-04",
            },
        )
    pointer = json.loads(s=pointer_path.read_text())
    snapshot = (directory / pointer["snapshot"]).resolve()
    if os.path.commonpath(paths=[str(snapshot), str(directory.resolve())]) != str(
        directory.resolve()
    ):
        raise OSError("Snapshot pointer refers outside the selected data directory.")
    manifest = json.loads(s=(snapshot / "manifest.json").read_text())
    tables = {}
    for name in ["evidence", "events"]:
        path = snapshot / manifest["files"][name]["name"]
        if os.path.commonpath(paths=[str(path.resolve()), str(snapshot)]) != str(
            snapshot
        ):
            raise OSError("Manifest file refers outside its snapshot.")
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != manifest["files"][name]["sha256"]
        ):
            raise OSError(f"Saved {name} table checksum does not match its manifest.")
        tables[name] = pd.read_csv(
            filepath_or_buffer=path,
            sep="\t",
            keep_default_na=False,
            na_values=[""],
            dtype={"event_id": str, "model_id": str},
        )
    return (
        validate_evidence(frame=tables["evidence"]),
        tables["events"].fillna(value=""),
        manifest,
    )
