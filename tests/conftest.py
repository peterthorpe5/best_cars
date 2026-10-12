"""Small explicit fixtures for evidence and real-world source edge cases."""

import pandas as pd
import pytest

from best_cars.parsers import evidence_row

NOW = "2026-10-12T10:00:00+00:00"


@pytest.fixture
def evidence():
    """Return six peers across two sources and two age cohorts."""
    records = []
    for source in ["tuv", "adac"]:
        for age in [4, 6]:
            for index in range(6):
                records.append(
                    evidence_row(
                        source_id=source,
                        source_name=source,
                        evidence_type="inspection" if source == "tuv" else "breakdown",
                        country="Germany",
                        report_year=2025,
                        observation_year=2024,
                        cohort_start=2024 - age,
                        cohort_end=2024 - age,
                        age_min=age,
                        age_max=age,
                        label=f"Toyota Test{index}",
                        value=float(index if source == "tuv" else 5 - index),
                        metric="defects" if source == "tuv" else "breakdowns",
                        unit="percent" if source == "tuv" else "per_1000_vehicles",
                        source_url="https://example.org/data",
                        retrieved_at=NOW,
                        comparison_group=f"age_{age}",
                        sample_size=100 if source == "tuv" else float("nan"),
                        event_count=index if source == "tuv" else float("nan"),
                    )
                )
    return pd.DataFrame.from_records(data=records)
