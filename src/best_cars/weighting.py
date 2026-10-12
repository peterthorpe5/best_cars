"""Explicit preference scenarios, not estimated source-quality weights."""

from collections.abc import Sequence

PRESETS = {
    "Balanced": {},
    "Emphasise avoiding breakdowns": {"adac": 2.0},
    "Emphasise inspection condition": {"traficom": 2.0, "mot": 2.0, "adac": 0.5},
}


def preference_weights(*, sources: Sequence[str], preset: str) -> dict[str, float]:
    """Return transparent starting weights for an explicitly selected question.

    Args:
        sources: Loaded source identifiers; absent sources are not fabricated.
        preset: One of the documented illustrative preference scenarios.

    Returns:
        Positive weights for loaded sources, defaulting to one. These are
        user-preference examples, not calibrated trust, precision or probability.

    Raises:
        ValueError: For unknown presets or invalid/duplicate source identifiers.
    """
    if (
        preset not in PRESETS
        or len(sources) != len(set(sources))
        or any(not isinstance(source, str) or not source.strip() for source in sources)
    ):
        raise ValueError("Require a known preset and distinct non-empty source IDs.")
    return {source: PRESETS[preset].get(source, 1.0) for source in sources}
