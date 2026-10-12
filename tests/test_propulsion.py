"""Propulsion eligibility must never silently attribute mixed-family rates."""

import pandas as pd
import pytest

from best_cars.analysis import cohort_scores
from best_cars.propulsion import propulsion_catalogue, select_propulsion


def test_dedicated_mixed_unknown_and_grouped_models(evidence):
    frame = cohort_scores(evidence=evidence)
    mapping = {
        "toyota:test0": "tesla:model 3",
        "toyota:test1": "toyota:corolla",
        "toyota:test2": "toyota:prius plug in hybrid",
        "toyota:test3": "toyota:prius plus",
    }
    frame["model_id"] = frame.model_id.replace(to_replace=mapping)
    catalogue = propulsion_catalogue(evidence=frame).set_index(keys="model_id")
    assert catalogue.loc["tesla:model 3", "attribution"] == "Dedicated series"
    assert catalogue.loc["toyota:corolla", "attribution"] == "Mixed family"
    assert catalogue.loc["toyota:test5", "category"] == "Unclassified"
    electric = select_propulsion(evidence=frame, category="Battery electric")
    assert set(electric.model_id) == {"tesla:model 3"}
    pd.testing.assert_series_equal(
        left=electric.relative_score, right=frame.loc[electric.index, "relative_score"]
    )
    assert set(select_propulsion(evidence=frame, category="Hybrid").model_id) == {
        "toyota:prius plus"
    }
    assert set(
        select_propulsion(
            evidence=frame, category="Hybrid", include_mixed=True
        ).model_id
    ) == {"toyota:corolla", "toyota:prius plus"}
    assert set(
        select_propulsion(evidence=frame, category="Plug-in hybrid").model_id
    ) == {"toyota:prius plug in hybrid"}
    frame.loc[frame.model_id.eq("tesla:model 3"), "model_grouped"] = True
    assert select_propulsion(evidence=frame, category="Battery electric").empty
    assert propulsion_catalogue(evidence=frame.iloc[:0]).empty
    with pytest.raises(ValueError):
        select_propulsion(evidence=frame, category="guess")
    with pytest.raises(ValueError):
        propulsion_catalogue(evidence=pd.DataFrame())
