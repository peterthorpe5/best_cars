"""Named-option command-line interface for reproducible car evidence."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from best_cars.analysis import reliability_ranking, select_evidence
from best_cars.mot import aggregate_mot_raw, parse_mot_aggregate
from best_cars.parsers import (
    parse_adac_pdf,
    parse_traficom_xlsx,
    parse_tuv_tsv,
    utc_now,
)
from best_cars.pipeline import nhtsa_lookup, refresh_sources, replace_evidence
from best_cars.schema import validate_evidence
from best_cars.storage import (
    data_directory,
    load_snapshot,
    save_snapshot,
    source_catalogue,
    tsv_bytes,
)

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the documented CLI with named input/output options."""
    parser = argparse.ArgumentParser(
        description="Best Cars: transparent, multi-source reliability evidence"
    )
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("sources", help="List verified and candidate evidence sources")
    refresh = commands.add_parser(
        "refresh", help="Refresh source tables; keep previous data on failures"
    )
    refresh.add_argument("--data-dir", type=Path)
    refresh.add_argument(
        "--sources",
        nargs="+",
        choices=["tuv", "adac", "traficom"],
        default=["adac", "traficom"],
    )
    refresh.add_argument("--tuv-years", nargs="+", type=int, default=[2024, 2025])
    refresh.add_argument("--offline", action="store_true")
    refresh.add_argument("--force", action="store_true")
    rank = commands.add_parser("rank", help="Export a relative evidence ranking as TSV")
    rank.add_argument("--data-dir", type=Path)
    rank.add_argument("--output", type=Path, required=True)
    rank.add_argument("--min-sources", type=int, default=1)
    rank.add_argument("--min-cohorts", type=int, default=3)
    rank.add_argument("--min-sample", type=int, default=100)
    rank.add_argument("--age-min", type=int, default=2)
    rank.add_argument("--age-max", type=int, default=15)
    rank.add_argument("--all-reports", action="store_true")
    importer = commands.add_parser("import", help="Import local real source data")
    importer.add_argument("--data-dir", type=Path)
    importer.add_argument(
        "--kind",
        required=True,
        choices=[
            "legacy-tuv",
            "traficom-xlsx",
            "adac-pdf",
            "mot-aggregate",
            "mot-raw",
            "evidence",
        ],
    )
    importer.add_argument("--input", type=Path, required=True)
    importer.add_argument("--year", type=int)
    importer.add_argument("--source-url")
    importer.add_argument(
        "--delimiter", choices=["tab", "comma", "pipe", "semicolon"], default="tab"
    )
    importer.add_argument("--no-header", action="store_true")
    importer.add_argument(
        "--mot-conflicts",
        choices=["error", "exclude"],
        default="error",
        help="Stop on conflicting eligible MOT test IDs, or explicitly exclude them with an audit note.",
    )
    lookup = commands.add_parser("nhtsa", help="Exact US make/model/year safety lookup")
    lookup.add_argument("--data-dir", type=Path)
    lookup.add_argument("--make", required=True)
    lookup.add_argument("--model", required=True)
    lookup.add_argument("--model-year", type=int, required=True)
    lookup.add_argument(
        "--event-types",
        nargs="+",
        choices=["recall", "complaint"],
        default=["recall", "complaint"],
    )
    lookup.add_argument("--offline", action="store_true")
    lookup.add_argument("--force", action="store_true")
    return parser


def import_source(*, args: argparse.Namespace) -> pd.DataFrame:
    """Parse an explicitly supplied local source without changing other datasets.

    Args:
        args: Validated import command options.

    Returns:
        Validated canonical reliability observations.

    Raises:
        ValueError: For missing source/year options or malformed data.
    """
    now = utc_now()
    if args.kind == "adac-pdf":
        if args.year is None or not args.source_url:
            raise ValueError("ADAC import requires --year and the exact --source-url.")
        return parse_adac_pdf(
            path=args.input,
            report_year=args.year,
            source_url=args.source_url,
            retrieved_at=now,
        )
    if args.kind == "traficom-xlsx":
        if args.year is None:
            raise ValueError("Traficom XLSX import requires --year (inspection year).")
        return parse_traficom_xlsx(
            path=args.input, observation_year=args.year, retrieved_at=now
        )
    if args.kind == "mot-raw":
        delimiters = {"tab": "\t", "comma": ",", "pipe": "|", "semicolon": ";"}
        return aggregate_mot_raw(
            path=args.input,
            retrieved_at=now,
            delimiter=delimiters[args.delimiter],
            header=not args.no_header,
            conflict_policy=args.mot_conflicts,
        )
    frame = pd.read_csv(filepath_or_buffer=args.input, sep="\t")
    if args.kind == "legacy-tuv":
        return parse_tuv_tsv(frame=frame, retrieved_at=now)
    if args.kind == "mot-aggregate":
        return parse_mot_aggregate(frame=frame, retrieved_at=now)
    return validate_evidence(frame=frame)


def main(*, argv: list[str] | None = None) -> int:
    """Execute a named-option command and return a useful process exit status.

    Args:
        argv: Optional explicit CLI argument list for testing.

    Returns:
        Zero on success; one for source/data errors; two for partial refresh failure.
    """
    args = build_parser().parse_args(args=argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.command == "sources":
            sys.stdout.write(tsv_bytes(frame=source_catalogue()).decode())
            return 0
        directory = data_directory(supplied=args.data_dir)
        if args.command == "refresh":
            status, snapshot = refresh_sources(
                directory=directory,
                sources=args.sources,
                offline=args.offline,
                force=args.force,
                tuv_years=args.tuv_years,
            )
            sys.stdout.write(tsv_bytes(frame=status).decode())
            LOGGER.info("Snapshot: %s", snapshot)
            return 2 if status.status.eq("failed").any() else 0
        if args.command == "rank":
            if args.output.suffix.lower() != ".tsv":
                raise ValueError("Rank output must have a .tsv extension.")
            evidence, _, _ = load_snapshot(directory=directory)
            scored = select_evidence(
                evidence=evidence,
                latest_only=not args.all_reports,
                age_range=(args.age_min, args.age_max),
                min_sample=args.min_sample,
            )
            ranking = reliability_ranking(
                scored=scored,
                min_sources=args.min_sources,
                min_cohorts=args.min_cohorts,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(tsv_bytes(frame=ranking))
            LOGGER.info("Wrote %s model rows to %s", len(ranking), args.output)
            return 0
        if args.command == "import":
            imported = import_source(args=args)
            evidence, events, _ = load_snapshot(directory=directory)
            evidence = replace_evidence(original=evidence, replacement=imported)
            snapshot = save_snapshot(
                directory=directory, evidence=evidence, events=events
            )
            LOGGER.info("Imported %s rows; snapshot %s", len(imported), snapshot)
            return 0
        events, snapshot = nhtsa_lookup(
            directory=directory,
            make=args.make,
            model=args.model,
            model_year=args.model_year,
            event_types=args.event_types,
            offline=args.offline,
            force=args.force,
        )
        LOGGER.info("Saved %s US safety events to %s", len(events), snapshot)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        LOGGER.error("Command failed: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
