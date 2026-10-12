"""Conservative, auditable propulsion eligibility for model-series evidence."""

import logging
from importlib.resources import files

import pandas as pd

LOGGER = logging.getLogger(__name__)
CATEGORIES = ["Battery electric", "Plug-in hybrid", "Hybrid"]


def propulsion_catalogue(*, evidence: pd.DataFrame) -> pd.DataFrame:
    """Attach reviewed eligibility without guessing propulsion from a model name.

    Args:
        evidence: Canonical evidence containing model identities and grouping.

    Returns:
        One row per model with category, attribution status and review source.
        Unknown models stay unclassified. Mixed model families are candidates,
        never propulsion-specific rate estimates. Grouped series are excluded.

    Raises:
        ValueError: For absent identity fields or invalid catalogue duplicates.
    """
    required = {"model_id", "make", "model", "model_grouped"}
    if not required.issubset(evidence.columns):
        raise ValueError("Propulsion review requires canonical model identities.")
    with (
        files("best_cars")
        .joinpath("data/propulsion.tsv")
        .open(mode="r", encoding="utf-8") as stream
    ):
        reviewed = pd.read_csv(
            filepath_or_buffer=stream, sep="\t", keep_default_na=False
        )
    if (
        reviewed.model_id.duplicated().any()
        or not reviewed.category.isin(CATEGORIES).all()
    ):
        raise ValueError("Propulsion catalogue has duplicate or invalid entries.")
    models = evidence.loc[
        ~evidence.model_grouped, ["model_id", "make", "model"]
    ].drop_duplicates(subset="model_id")
    result = models.merge(
        right=reviewed, on="model_id", how="left", validate="one_to_one"
    )
    result["category"] = result.category.fillna(value="Unclassified")
    result["attribution"] = result.attribution.fillna(value="Not reviewed")
    result["review_url"] = result.review_url.fillna(value="")
    result["note"] = result.note.fillna(
        value="No propulsion-specific classification is asserted."
    )
    LOGGER.debug("Reviewed propulsion eligibility for %s model series", len(result))
    return result


def select_propulsion(
    *, evidence: pd.DataFrame, category: str, include_mixed: bool = False
) -> pd.DataFrame:
    """Select reviewed series while preserving original cohort percentile scores.

    Args:
        evidence: Selected canonical evidence, optionally with scored columns.
        category: Battery electric, Plug-in hybrid or Hybrid.
        include_mixed: Explicitly include mixed-family candidates for context.

    Returns:
        Unmodified evidence rows for eligible series. No scores are recomputed.

    Raises:
        ValueError: If the category is unrecognised.
    """
    if category not in CATEGORIES:
        raise ValueError("Select a recognised propulsion category.")
    catalogue = propulsion_catalogue(evidence=evidence)
    eligible = catalogue.category.eq(category)
    if not include_mixed:
        eligible &= catalogue.attribution.eq("Dedicated series")
    return evidence.loc[
        evidence.model_id.isin(catalogue.loc[eligible, "model_id"])
    ].copy()
