"""Published repair-price context and explicit user-selected budget scenarios."""

import logging
from importlib.resources import files

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def repair_references() -> pd.DataFrame:
    """Load dated UK marketplace price references, never vehicle-specific quotes.

    Returns:
        Published job averages with source URLs, price basis and retrieval date.

    Raises:
        ValueError: If bundled references contain invalid prices or provenance.
    """
    with (
        files("best_cars")
        .joinpath("data/repair_prices.tsv")
        .open(mode="r", encoding="utf-8") as stream
    ):
        frame = pd.read_csv(filepath_or_buffer=stream, sep="\t")
    if (
        not np.isfinite(frame.unit_cost_gbp).all()
        or frame.unit_cost_gbp.lt(0).any()
        or frame.repair_job.duplicated().any()
        or not frame.source_url.str.startswith("https://").all()
    ):
        raise ValueError("Repair references require valid prices and source URLs.")
    return frame


def repair_budget(*, estimates: pd.DataFrame) -> pd.DataFrame:
    """Calculate an explicit job-count budget without inferring fault frequency.

    Args:
        estimates: repair_job, unit_cost_gbp and quantity columns. Prices may be
            missing; quantities must be finite and non-negative. Other quote
            provenance fields are retained in the returned table.

    Returns:
        Validated rows with scenario costs and missing-price status. Unknown
        prices for selected jobs remain unknown, never free repairs.

    Raises:
        ValueError: For invalid quantities, prices, jobs or required fields.
    """
    required = {"repair_job", "unit_cost_gbp", "quantity"}
    if not required.issubset(estimates.columns):
        raise ValueError("Budget TSV requires repair_job, unit_cost_gbp and quantity.")
    result = estimates.copy()
    if (
        result.repair_job.isna().any()
        or result.repair_job.astype(str).str.strip().eq("").any()
    ):
        raise ValueError("Every budget row needs a repair-job description.")
    for column in ["unit_cost_gbp", "quantity"]:
        result[column] = pd.to_numeric(arg=result[column], errors="raise")
    if not np.isfinite(result.quantity).all() or result.quantity.lt(0).any():
        raise ValueError("Job quantities must be finite and non-negative.")
    known = result.unit_cost_gbp.dropna()
    if not np.isfinite(known).all() or known.lt(0).any():
        raise ValueError("Known job prices must be finite and non-negative.")
    result["scenario_cost_gbp"] = result.quantity * result.unit_cost_gbp
    result.loc[result.quantity.eq(0), "scenario_cost_gbp"] = 0.0
    result["price_status"] = np.where(
        result.quantity.gt(0) & result.unit_cost_gbp.isna(),
        "Missing selected-job price; total unavailable",
        "Explicit scenario assumption",
    )
    LOGGER.debug("Calculated %s explicit repair-budget rows", len(result))
    return result
