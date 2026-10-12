"""Read-only integration check against actual downloaded source files."""

import argparse
import logging
from pathlib import Path

from best_cars.parsers import parse_adac_pdf, parse_traficom_xlsx, utc_now
from best_cars.storage import baseline_evidence


def main(*, argv: list[str] | None = None) -> int:
    """Print source coverage for local official source files.

    Args:
        argv: Optional named CLI arguments.

    Returns:
        Zero when both adapters validate successfully.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adac-pdf", type=Path, required=True)
    parser.add_argument("--traficom-xlsx", type=Path, required=True)
    args = parser.parse_args(args=argv)
    logging.basicConfig(level=logging.INFO)
    baseline = baseline_evidence()
    print("TÜV", len(baseline), baseline.model_id.nunique())
    adac = parse_adac_pdf(path=args.adac_pdf, report_year=2025, retrieved_at=utc_now())
    print("ADAC", len(adac), adac.model_id.nunique())
    print(
        adac.loc[
            adac.source_model.isin(["Toyota RAV4", "Toyota C-HR", "VW Golf", "BMW i3"]),
            ["source_model", "cohort_year_start", "value"],
        ].to_string(index=False)
    )
    traficom = parse_traficom_xlsx(
        path=args.traficom_xlsx, observation_year=2025, retrieved_at=utc_now()
    )
    print("Traficom", len(traficom), traficom.model_id.nunique())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
