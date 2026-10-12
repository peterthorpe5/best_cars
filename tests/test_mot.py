"""UK initial-test semantics, PRS outcomes and duplicate handling."""

from zipfile import ZipFile

import pandas as pd
import pytest

from best_cars.mot import (
    RAW_COLUMNS,
    aggregate_mot_raw,
    mot_raw_chunks,
    mot_test_dates,
    parse_mot_aggregate,
)

NOW = "2026-10-12T10:00:00+00:00"


def test_mot_counts_and_validation():
    frame = pd.DataFrame(
        data=[
            {
                "make": "Toyota",
                "model": "Yaris",
                "observation_year": 2023,
                "registration_year": 2018,
                "tests": 100,
                "failures": 9,
            }
        ]
    )
    result = parse_mot_aggregate(frame=frame, retrieved_at=NOW)
    assert result.iloc[0].value == 9 and result.iloc[0].sample_size == 100
    assert result.iloc[0].event_count == 9 and result.iloc[0].age_min == 5
    frame.loc[0, "failures"] = 101
    with pytest.raises(ValueError):
        parse_mot_aggregate(frame=frame, retrieved_at=NOW)
    with pytest.raises(ValueError):
        parse_mot_aggregate(frame=pd.DataFrame(), retrieved_at=NOW)
    frame.loc[0, "failures"] = 9
    frame["registration_year"] = frame.registration_year.astype(float)
    frame.loc[0, "registration_year"] = 2018.5
    with pytest.raises(ValueError, match="integers"):
        parse_mot_aggregate(frame=frame, retrieved_at=NOW)


def raw_mot_table():
    """Return documented source fields with retests, PRS and aborted results."""
    rows = []
    for index, kind, outcome in [
        (1, "NT", "P"),
        (2, "NT", "F"),
        (3, "NT", "PRS"),
        (4, "RT", "P"),
        (5, "NT", "ABA"),
    ]:
        rows.append(
            [
                str(index),
                str(index),
                "10-01-2023",
                "4",
                kind,
                outcome,
                "10000",
                "DD",
                "Toyota",
                "Yaris",
                "Blue",
                "PE",
                "1000",
                "01-01-2018",
            ]
        )
    rows.append(rows[0].copy())
    return pd.DataFrame(data=rows, columns=RAW_COLUMNS)


def test_streaming_initial_tests_exclude_retests_and_deduplicate(tmp_path):
    frame = raw_mot_table()
    path = tmp_path / "mot.tsv"
    frame.to_csv(path_or_buf=path, sep="\t", index=False)
    result = aggregate_mot_raw(path=path, retrieved_at=NOW, chunksize=2)
    row = result.iloc[0]
    assert row.sample_size == 3 and row.event_count == 2
    assert row.value == pytest.approx(100 * 2 / 3)
    assert row.mileage_km == pytest.approx(16093.44)


def test_official_pipe_headerless_zip(tmp_path):
    path = tmp_path / "mot.zip"
    with ZipFile(file=path, mode="w") as archive:
        archive.writestr(
            zinfo_or_arcname="results.txt",
            data=raw_mot_table().to_csv(sep="|", header=False, index=False),
        )
    assert (
        aggregate_mot_raw(
            path=path, retrieved_at=NOW, delimiter="|", header=False
        ).sample_size.iloc[0]
        == 3
    )


def test_conflicting_duplicate_and_unknown_outcome(tmp_path):
    frame = raw_mot_table()
    frame.loc[len(frame) - 1, "test_result"] = "F"
    path = tmp_path / "mot.tsv"
    frame.to_csv(path_or_buf=path, sep="\t", index=False)
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        aggregate_mot_raw(path=path, retrieved_at=NOW)
    frame.loc[0, "test_result"] = "UNKNOWN"
    frame.to_csv(path_or_buf=path, sep="\t", index=False)
    with pytest.raises(ValueError, match="outcome"):
        aggregate_mot_raw(path=path, retrieved_at=NOW)


def test_invalid_source_dates_and_structure(tmp_path):
    frame = raw_mot_table().iloc[:1].copy()
    frame.loc[0, "first_use_date"] = "01-01-2024"
    path = tmp_path / "mot.tsv"
    frame.to_csv(path_or_buf=path, sep="\t", index=False)
    with pytest.raises(ValueError, match="first-use"):
        aggregate_mot_raw(path=path, retrieved_at=NOW)
    frame.drop(columns="make").to_csv(path_or_buf=path, sep="\t", index=False)
    with pytest.raises(ValueError, match="lacks"):
        aggregate_mot_raw(path=path, retrieved_at=NOW)
    with pytest.raises(ValueError):
        aggregate_mot_raw(path=path, retrieved_at=NOW, chunksize=0)


def test_2024_results_archive_reads_all_months_and_ignores_metadata(tmp_path):
    frame = raw_mot_table().iloc[:3].copy()
    frame["test_date"] = "2024-08-19"
    frame["first_use_date"] = "2018-03-01"
    frame["completed_date"] = "2024-08-19T10:24:00.000Z"
    frame["test_mileage"] = "10000.0"
    path = tmp_path / "official2024.zip"
    with ZipFile(file=path, mode="w") as archive:
        for month, selection in [("08", frame.iloc[:2]), ("09", frame.iloc[2:])]:
            archive.writestr(
                zinfo_or_arcname=f"MOT testing data results (2024)/test_result_2024{month}.csv",
                data=selection.to_csv(index=False),
            )
        # Duplicates across members must still count only once.
        archive.writestr(
            zinfo_or_arcname="duplicate.tsv", data=frame.iloc[:1].to_csv(index=False)
        )
        archive.writestr(
            zinfo_or_arcname="__MACOSX/._test_result_202408.csv", data="metadata"
        )
        archive.writestr(zinfo_or_arcname="._ignored.csv", data="metadata")
    chunks = list(mot_raw_chunks(path=path, delimiter=",", header=True, chunksize=1))
    assert len(chunks) == 4 and all(len(chunk) == 1 for chunk in chunks)
    result = aggregate_mot_raw(path=path, retrieved_at=NOW, delimiter=",", chunksize=1)
    assert result.iloc[0].sample_size == 3 and result.iloc[0].event_count == 2
    assert result.iloc[0].observation_year == 2024
    assert result.iloc[0].mileage_km == pytest.approx(16093.44)


def test_mot_reader_settings_and_empty_archive(tmp_path):
    with pytest.raises(ValueError, match="delimiter"):
        list(
            mot_raw_chunks(
                path=tmp_path / "unused", delimiter=":", header=True, chunksize=1
            )
        )
    with pytest.raises(ValueError, match="chunk"):
        list(
            mot_raw_chunks(
                path=tmp_path / "unused", delimiter="|", header=False, chunksize=0
            )
        )
    path = tmp_path / "empty.zip"
    with ZipFile(file=path, mode="w") as archive:
        archive.writestr(zinfo_or_arcname="README.md", data="No results")
    with pytest.raises(ValueError, match="no delimited"):
        aggregate_mot_raw(path=path, retrieved_at=NOW)


def test_mot_dates_respect_iso_month_and_historical_day_first():
    dates = mot_test_dates(
        values=pd.Series(data=["2024-03-01", "01-03-2024", "2024-08-19"])
    )
    assert dates.dt.month.tolist() == [3, 3, 8]
    assert dates.dt.day.tolist() == [1, 1, 19]
    assert str(dates.dtype) == "datetime64[ns, UTC]"
    for invalid in ["", "not a date", "2024-02-30"]:
        with pytest.raises(ValueError):
            mot_test_dates(values=pd.Series(data=[invalid]))


def test_quarantine_excludes_conflicting_id_and_later_reappearances(tmp_path):
    frame = raw_mot_table().iloc[:3].copy()
    conflicting = frame.iloc[:1].copy()
    conflicting["test_mileage"] = "20000"
    frame = pd.concat(objs=[frame, conflicting, frame.iloc[:1]], ignore_index=True)
    path = tmp_path / "source.tsv"
    frame.to_csv(path_or_buf=path, sep="\t", index=False)
    result = aggregate_mot_raw(
        path=path, retrieved_at=NOW, chunksize=1, conflict_policy="exclude"
    )
    assert result.iloc[0].sample_size == 2
    assert result.iloc[0].event_count == 2
    assert (
        result.attrs["mot_import_audit"]["conflicting_eligible_test_ids_excluded"] == 1
    )
    assert "1 conflicting eligible test IDs excluded" in result.iloc[0].notes
    with pytest.raises(ValueError):
        aggregate_mot_raw(path=path, retrieved_at=NOW, conflict_policy="guess")


def test_raw_alias_counts_pool_with_actual_mileage_reading_weights():
    frame = pd.DataFrame(
        data=[
            {
                "make": "VW",
                "model": "Golf",
                "observation_year": 2024,
                "registration_year": 2018,
                "tests": 100,
                "failures": 5,
                "mean_mileage_km": 10000,
                "mileage_tests": 50,
            },
            {
                "make": "VOLKSWAGEN",
                "model": "GOLF",
                "observation_year": 2024,
                "registration_year": 2018,
                "tests": 200,
                "failures": 25,
                "mean_mileage_km": 30000,
                "mileage_tests": 150,
            },
        ]
    )
    result = parse_mot_aggregate(frame=frame, retrieved_at=NOW)
    assert len(result) == 1
    assert result.iloc[0].sample_size == 300
    assert result.iloc[0].event_count == 30
    assert result.iloc[0].value == 10
    assert result.iloc[0].mileage_km == 25000
    assert "VW Golf" in result.iloc[0].source_model
    assert "VOLKSWAGEN GOLF" in result.iloc[0].source_model
    unknown = frame.copy()
    unknown["mean_mileage_km"] = float("nan")
    assert pd.isna(
        parse_mot_aggregate(frame=unknown, retrieved_at=NOW).iloc[0].mileage_km
    )
    with pytest.raises(ValueError, match="double-count"):
        parse_mot_aggregate(
            frame=pd.concat(objs=[frame, frame.iloc[:1]], ignore_index=True),
            retrieved_at=NOW,
        )
    frame.loc[0, "mileage_tests"] = 101
    with pytest.raises(ValueError, match="mileage counts"):
        parse_mot_aggregate(frame=frame, retrieved_at=NOW)
