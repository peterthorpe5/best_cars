"""Refresh preservation, exact US queries, CLI exits and launcher settings."""

import argparse
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from best_cars import cli, launcher, pipeline
from best_cars.schema import empty_events
from best_cars.storage import load_snapshot, save_snapshot

NOW = "2026-10-12T10:00:00+00:00"


def test_replacement_preserves_other_sources_and_reports(evidence):
    fresh = evidence.loc[evidence.source_id.eq("tuv")].copy()
    fresh["value"] = 2
    result = pipeline.replace_evidence(original=evidence, replacement=fresh)
    assert len(result) == len(evidence)
    assert result.loc[result.source_id.eq("tuv"), "value"].eq(2).all()
    assert (
        result.loc[result.source_id.eq("adac"), "value"].tolist()
        == evidence.loc[evidence.source_id.eq("adac"), "value"].tolist()
    )
    assert len(
        pipeline.replace_evidence(original=evidence, replacement=fresh.iloc[:0])
    ) == len(evidence)


def test_failed_refresh_keeps_previous_source(evidence, tmp_path, monkeypatch):
    save_snapshot(directory=tmp_path, evidence=evidence, events=empty_events())

    def fail(**kwargs):
        raise OSError("Source unavailable")

    monkeypatch.setattr(pipeline, "fetch_adac", fail)
    monkeypatch.setattr(
        pipeline,
        "fetch_traficom",
        lambda **kwargs: evidence.loc[evidence.source_id.eq("tuv")],
    )
    status, _ = pipeline.refresh_sources(
        directory=tmp_path, sources=["adac", "traficom"]
    )
    assert list(status.status) == ["failed", "updated"]
    loaded, _, _ = load_snapshot(directory=tmp_path)
    assert len(loaded) == len(evidence)
    with pytest.raises(ValueError):
        pipeline.refresh_sources(directory=tmp_path, sources=["unverified"])


def test_tuv_refresh_calls_all_age_groups(evidence, monkeypatch):
    client = MagicMock()
    client.last_acquired_at = NOW
    client.request_bytes.return_value = b"<html>data</html>"
    monkeypatch.setattr(
        pipeline,
        "parse_tuv_html",
        lambda **kwargs: evidence.loc[
            evidence.source_id.eq("tuv") & evidence.age_min.eq(4)
        ].assign(comparison_group=kwargs["age_group"]),
    )
    result = pipeline.fetch_tuv(client=client, years=[2025])
    assert client.request_bytes.call_count == 6 and len(result) == 36
    with pytest.raises(ValueError):
        pipeline.fetch_tuv(client=client, years=[])


def test_adac_download_and_partial_rejection(evidence, tmp_path, monkeypatch):
    client = MagicMock()
    client.last_acquired_at = NOW
    client.cache_dir = tmp_path
    client.request_bytes.return_value = b"%PDF fake source"
    monkeypatch.setattr(pipeline, "parse_adac_pdf", lambda **kwargs: evidence)
    with pytest.raises(ValueError, match="partial"):
        pipeline.fetch_adac(client=client)
    many = pd.concat(
        objs=[
            evidence.assign(model_id=evidence.model_id + str(index))
            for index in range(20)
        ],
        ignore_index=True,
    )
    monkeypatch.setattr(pipeline, "parse_adac_pdf", lambda **kwargs: many)
    assert len(pipeline.fetch_adac(client=client)) == len(many)
    client.request_bytes.return_value = b"Access denied"
    with pytest.raises(ValueError, match="PDF"):
        pipeline.fetch_adac(client=client)


def test_traficom_year_queries_and_errors(evidence, monkeypatch):
    client = MagicMock()
    client.last_acquired_at = NOW
    metadata = {
        "variables": [
            {"code": "Katsastusvuosi", "values": ["2025"]},
            {"code": "Tiedot", "values": ["Lkm"]},
        ]
    }
    client.request_json.side_effect = [metadata, {"example": "data"}]
    monkeypatch.setattr(pipeline, "parse_traficom_json", lambda **kwargs: evidence)
    assert len(pipeline.fetch_traficom(client=client, years=[2025])) == len(evidence)
    assert (
        client.request_json.call_args.kwargs["query"]["response"]["format"]
        == "json-stat2"
    )
    client.request_json.side_effect = [metadata]
    with pytest.raises(ValueError, match="unavailable"):
        pipeline.fetch_traficom(client=client, years=[2026])
    client.request_json.side_effect = [{"variables": []}]
    with pytest.raises(ValueError):
        pipeline.fetch_traficom(client=client)


def test_nhtsa_query_saved_without_entering_rate_scores(
    evidence, tmp_path, monkeypatch
):
    save_snapshot(directory=tmp_path, evidence=evidence, events=empty_events())
    client = MagicMock()
    client.last_acquired_at = NOW
    client.request_json.return_value = {
        "Count": 1,
        "results": [{"NHTSACampaignNumber": "20V123000", "Summary": "Published fault"}],
    }
    monkeypatch.setattr(pipeline, "CachedClient", lambda **kwargs: client)
    fresh, _ = pipeline.nhtsa_lookup(
        directory=tmp_path,
        make="Land Rover",
        model="Range Rover",
        model_year=2020,
        event_types=["recall"],
    )
    assert len(fresh) == 1
    assert "Land+Rover" in client.request_json.call_args.kwargs["url"]
    rates, events, _ = load_snapshot(directory=tmp_path)
    assert len(rates) == len(evidence) and len(events) == 1
    assert not rates.source_id.str.startswith("nhtsa").any()
    client.request_json.return_value = {"Count": 0, "results": []}
    fresh, _ = pipeline.nhtsa_lookup(
        directory=tmp_path,
        make="Land Rover",
        model="Range Rover",
        model_year=2020,
        event_types=["recall"],
    )
    assert fresh.empty and load_snapshot(directory=tmp_path)[1].empty
    for options in [
        {"make": "", "model": "x", "model_year": 2020},
        {"make": "x", "model": "y", "model_year": 1900},
        {"make": "x", "model": "y", "model_year": 2020, "event_types": []},
    ]:
        with pytest.raises(ValueError):
            pipeline.nhtsa_lookup(directory=tmp_path, **options)
    assert pipeline.datetime_year_limit() >= 2027


def test_cli_sources_rank_and_bad_output(evidence, tmp_path, capsys):
    assert cli.build_parser().parse_args(args=["sources"]).command == "sources"
    assert cli.main(argv=["sources"]) == 0
    assert "traficom\t" in capsys.readouterr().out
    save_snapshot(directory=tmp_path, evidence=evidence, events=empty_events())
    output = tmp_path / "ranking.tsv"
    assert (
        cli.main(argv=["rank", "--data-dir", str(tmp_path), "--output", str(output)])
        == 0
    )
    assert len(pd.read_csv(filepath_or_buffer=output, sep="\t")) == 6
    assert cli.main(argv=["rank", "--output", str(tmp_path / "bad.csv")]) == 1


def test_cli_refresh_status_and_import(tmp_path, monkeypatch):
    status = pd.DataFrame(data=[{"status": "failed", "source_id": "adac"}])
    monkeypatch.setattr(cli, "refresh_sources", lambda **kwargs: (status, tmp_path))
    assert cli.main(argv=["refresh", "--data-dir", str(tmp_path)]) == 2
    path = tmp_path / "counts.tsv"
    pd.DataFrame(
        data=[
            {
                "make": "Toyota",
                "model": "Yaris",
                "observation_year": 2023,
                "registration_year": 2018,
                "tests": 100,
                "failures": 5,
            }
        ]
    ).to_csv(path_or_buf=path, sep="\t", index=False)
    assert (
        cli.main(
            argv=[
                "import",
                "--kind",
                "mot-aggregate",
                "--input",
                str(path),
                "--data-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert "mot" in set(load_snapshot(directory=tmp_path)[0].source_id)
    args = argparse.Namespace(kind="adac-pdf", input=path, year=None, source_url=None)
    with pytest.raises(ValueError):
        cli.import_source(args=args)


def test_cli_mot_raw_passes_explicit_conflict_policy(tmp_path, monkeypatch, evidence):
    captured = {}

    def raw_import(**kwargs):
        captured.update(kwargs)
        return evidence

    monkeypatch.setattr(cli, "aggregate_mot_raw", raw_import)
    args = cli.build_parser().parse_args(
        args=[
            "import",
            "--kind",
            "mot-raw",
            "--input",
            str(tmp_path / "source.zip"),
            "--delimiter",
            "comma",
            "--mot-conflicts",
            "exclude",
        ]
    )
    assert cli.import_source(args=args).equals(evidence)
    assert captured["delimiter"] == "," and captured["header"] is True
    assert captured["conflict_policy"] == "exclude"
    args.kind = "traficom-xlsx"
    with pytest.raises(ValueError):
        cli.import_source(args=args)


def test_cli_nhtsa_and_launcher_named_options(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli, "nhtsa_lookup", lambda **kwargs: (empty_events(), tmp_path)
    )
    assert (
        cli.main(
            argv=[
                "nhtsa",
                "--make",
                "Toyota",
                "--model",
                "Corolla",
                "--model-year",
                "2020",
            ]
        )
        == 0
    )
    call = MagicMock(return_value=0)
    monkeypatch.setattr(launcher.subprocess, "call", call)
    assert (
        launcher.main(argv=["--data-dir", str(tmp_path), "--server-port", "8502"]) == 0
    )
    assert call.call_args.kwargs["env"]["BEST_CARS_DATA_DIR"] == str(tmp_path)
    assert "127.0.0.1" in call.call_args.kwargs["args"]
    with pytest.raises(SystemExit):
        launcher.main(argv=["--server-port", "99999"])


def test_local_source_checker_entrypoint(evidence, tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "scripts" / "check_source_parsers.py"
    spec = importlib.util.spec_from_file_location(name="source_checker", location=path)
    module = importlib.util.module_from_spec(spec=spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "parse_adac_pdf", lambda **kwargs: evidence)
    monkeypatch.setattr(module, "parse_traficom_xlsx", lambda **kwargs: evidence)
    assert (
        module.main(
            argv=[
                "--adac-pdf",
                str(tmp_path / "a.pdf"),
                "--traficom-xlsx",
                str(tmp_path / "t.xlsx"),
            ]
        )
        == 0
    )


def test_full_data_app_checker_success_and_failure(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "scripts" / "smoke_app.py"
    spec = importlib.util.spec_from_file_location(name="app_checker", location=path)
    module = importlib.util.module_from_spec(spec=spec)
    spec.loader.exec_module(module)
    simulated = MagicMock()
    simulated.exception = []
    simulated.radio.return_value.set_value.return_value.run.return_value = simulated
    fake_test = MagicMock()
    fake_test.from_file.return_value.run.return_value = simulated
    monkeypatch.setattr(module, "AppTest", fake_test)
    assert module.main(argv=["--data-dir", str(tmp_path)]) == 0
    simulated.exception = [MagicMock(message="A page failed")]
    with pytest.raises(RuntimeError, match="page failed"):
        module.main(argv=["--data-dir", str(tmp_path)])
