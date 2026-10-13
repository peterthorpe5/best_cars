"""Repair scenarios must retain unknown prices and never infer fault frequency."""

import numpy as np
import pandas as pd
import pytest

from best_cars.repair_costs import repair_budget, repair_references


def test_reference_provenance_and_explicit_quantities():
    references = repair_references()
    assert len(references) == 4
    assert references.source_url.str.startswith("https://").all()
    assert references.retrieved_date.eq("2026-10-12").all()
    assert references.currency.eq("GBP").all()
    budget = repair_budget(estimates=references.assign(quantity=0))
    assert budget.scenario_cost_gbp.sum() == 0
    selected = references.assign(quantity=[1, 0, 0, 0])
    assert repair_budget(estimates=selected).scenario_cost_gbp.sum() == 137
    assert "quantity" not in references.columns


def test_unknown_selected_prices_do_not_become_free_repairs():
    frame = pd.DataFrame(
        data={
            "repair_job": ["Actual garage quote", "Unknown job", "Not selected"],
            "unit_cost_gbp": [200, np.nan, np.nan],
            "quantity": [2, 1, 0],
        }
    )
    result = repair_budget(estimates=frame)
    assert result.scenario_cost_gbp.iloc[0] == 400
    assert pd.isna(result.scenario_cost_gbp.iloc[1])
    assert result.scenario_cost_gbp.iloc[2] == 0
    assert "unavailable" in result.price_status.iloc[1]
    assert repair_budget(estimates=frame.iloc[:0]).empty
    for changed in [
        frame.drop(columns="quantity"),
        frame.assign(quantity=-1),
        frame.assign(quantity=np.inf),
        frame.assign(unit_cost_gbp=-1),
        frame.assign(unit_cost_gbp=np.inf),
        frame.assign(repair_job=""),
        frame.assign(repair_job=None),
    ]:
        with pytest.raises(ValueError):
            repair_budget(estimates=changed)
