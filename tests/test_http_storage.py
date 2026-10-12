"""Cache integrity, failure handling and atomic snapshot tests."""

import io
import json
from unittest.mock import MagicMock

import pandas as pd
import pytest
import requests
from openpyxl import load_workbook

from best_cars.http import CachedClient
from best_cars.schema import empty_events
from best_cars.storage import (
    baseline_evidence,
    data_directory,
    excel_bytes,
    load_snapshot,
    save_snapshot,
    source_catalogue,
    tsv_bytes,
)


def fake_session(*, content=b'{"ok": true}'):
    """Return an injectable context-managed HTTP response."""
    session = MagicMock()
    response = MagicMock()
    response.iter_content.return_value = [content]
    session.request.return_value.__enter__.return_value = response
    return session


def test_download_cache_offline_and_json(tmp_path):
    session = fake_session()
    client = CachedClient(cache_dir=tmp_path, session=session)
    assert client.request_json(url="https://example.org/data") == {"ok": True}
    acquired = client.last_acquired_at
    assert session.request.call_count == 1
    assert client.request_json(url="https://example.org/data") == {"ok": True}
    assert client.last_acquired_at == acquired
    assert session.request.call_count == 1
    offline = CachedClient(cache_dir=tmp_path, offline=True, session=MagicMock())
    assert offline.request_json(url="https://example.org/data") == {"ok": True}
    assert not offline.session.request.called
    next(tmp_path.glob("*.bin")).write_bytes(b"corrupt")
    with pytest.raises(OSError, match="checksum"):
        offline.request_json(url="https://example.org/data")


def test_missing_offline_cache_and_post_identity(tmp_path):
    client = CachedClient(cache_dir=tmp_path, offline=True)
    with pytest.raises(OSError, match="No verified"):
        client.request_json(url="https://example.org/data")
    session = fake_session()
    client = CachedClient(cache_dir=tmp_path, session=session)
    client.request_json(url="https://example.org/data", query={"year": 2025})
    client.request_json(url="https://example.org/data", query={"year": 2024})
    assert len(list(tmp_path.glob("*.bin"))) == 2
    assert session.request.call_args.kwargs["method"] == "POST"


def test_bounded_retries_and_bad_payloads(tmp_path, monkeypatch):
    session = MagicMock()
    session.request.side_effect = requests.ConnectionError("network failed")
    monkeypatch.setattr("best_cars.http.time.sleep", lambda duration: None)
    client = CachedClient(cache_dir=tmp_path, attempts=2, session=session)
    with pytest.raises(OSError):
        client.request_json(url="https://example.org/data")
    assert session.request.call_count == 2
    assert not list(tmp_path.glob("*.bin"))
    for content, limit in [(b"", 100), (b"123456", 3), (b"[]", 100)]:
        client = CachedClient(
            cache_dir=tmp_path,
            refresh=True,
            max_bytes=limit,
            session=fake_session(content=content),
        )
        with pytest.raises(ValueError):
            client.request_json(url="https://example.org/data")
    with pytest.raises(ValueError):
        client.request_bytes(url="http://example.org")


def test_snapshot_roundtrip_and_corruption(evidence, tmp_path):
    events = empty_events()
    snapshot = save_snapshot(directory=tmp_path, evidence=evidence, events=events)
    loaded, loaded_events, manifest = load_snapshot(directory=tmp_path)
    assert len(loaded) == len(evidence) and loaded_events.empty
    assert manifest["schema_version"] == 1
    assert loaded.model_grouped.dtype == bool
    (snapshot / "evidence.tsv.gz").write_bytes(b"corrupt")
    with pytest.raises(OSError, match="checksum"):
        load_snapshot(directory=tmp_path)


def test_failed_save_preserves_current_pointer(evidence, tmp_path):
    save_snapshot(directory=tmp_path, evidence=evidence, events=empty_events())
    previous = (tmp_path / "current.json").read_bytes()
    evidence.loc[0, "value"] = -1
    with pytest.raises(ValueError):
        save_snapshot(directory=tmp_path, evidence=evidence, events=empty_events())
    assert (tmp_path / "current.json").read_bytes() == previous
    with pytest.raises(ValueError):
        save_snapshot(
            directory=tmp_path, evidence=evidence.iloc[:0], events=pd.DataFrame()
        )


def test_pointer_cannot_leave_dataset_folder(tmp_path):
    (tmp_path / "current.json").write_text(
        data=json.dumps(obj={"snapshot": "../outside"})
    )
    with pytest.raises(OSError, match="outside"):
        load_snapshot(directory=tmp_path)


def test_baseline_and_catalogue_are_real_not_demo(tmp_path):
    baseline = baseline_evidence()
    assert len(baseline) == 1224
    assert set(baseline.report_year) == {2024, 2025}
    assert "honda:jazz" in set(baseline.model_id)
    loaded, events, manifest = load_snapshot(directory=tmp_path)
    assert len(loaded) == 1224 and events.empty and manifest["baseline_only"]
    catalogue = source_catalogue()
    assert {"tuv", "adac", "traficom", "mot", "nhtsa_recalls"}.issubset(
        catalogue.source_id
    )


def test_data_path_and_tsv_exports(evidence, tmp_path, monkeypatch):
    monkeypatch.setenv("BEST_CARS_DATA_DIR", str(tmp_path))
    assert data_directory() == tmp_path
    assert data_directory(supplied=tmp_path / "other") == tmp_path / "other"
    result = tsv_bytes(frame=evidence)
    assert b"\t" in result.splitlines()[0]
    assert (
        pd.read_csv(filepath_or_buffer=io.BytesIO(result), sep="\t").shape
        == evidence.shape
    )


def test_excel_headers_filters_and_download(evidence):
    raw = excel_bytes(sheets={"Evidence": evidence})
    book = load_workbook(filename=io.BytesIO(raw))
    assert book["Evidence"].freeze_panes == "A2"
    assert book["Evidence"].auto_filter.ref.startswith("A1:")
    assert book["Evidence"].max_row == len(evidence) + 1
    for sheets in [{}, {"invalid/name": evidence}]:
        with pytest.raises(ValueError):
            excel_bytes(sheets=sheets)
