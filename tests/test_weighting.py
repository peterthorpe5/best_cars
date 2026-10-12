"""Preference presets must be explicit, reversible and distinct from trust."""

import pytest

from best_cars.weighting import preference_weights


def test_preference_scenarios_keep_absent_sources_absent():
    sources = ["tuv", "adac", "traficom"]
    assert preference_weights(sources=sources, preset="Balanced") == {
        source: 1.0 for source in sources
    }
    assert (
        preference_weights(sources=sources, preset="Emphasise avoiding breakdowns")[
            "adac"
        ]
        == 2
    )
    inspection = preference_weights(
        sources=sources, preset="Emphasise inspection condition"
    )
    assert inspection == {"tuv": 1.0, "adac": 0.5, "traficom": 2.0}
    assert "mot" not in inspection
    assert preference_weights(sources=[], preset="Balanced") == {}
    assert preference_weights(
        sources=["mot"], preset="Emphasise inspection condition"
    ) == {"mot": 2.0}


@pytest.mark.parametrize(
    "sources,preset",
    [(["adac"], "unknown"), (["adac", "adac"], "Balanced"), ([""], "Balanced")],
)
def test_preference_scenarios_reject_invalid_settings(sources, preset):
    with pytest.raises(ValueError):
        preference_weights(sources=sources, preset=preset)
