"""Saved-data startup must not depend on native PDF cryptography imports."""

import builtins
import subprocess
import sys
from pathlib import Path

import pytest

from best_cars.parsers import parse_adac_pdf


def test_pdf_import_failure_has_actionable_message(tmp_path, monkeypatch):
    original = builtins.__import__

    def block_pdf(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pdfplumber":
            raise ImportError("symbol not found: _OSSL_get_max_threads")
        return original(
            name=name, globals=globals, locals=locals, fromlist=fromlist, level=level
        )

    monkeypatch.setattr(builtins, "__import__", block_pdf)
    with pytest.raises(ValueError, match="MAC_INSTALL") as error:
        parse_adac_pdf(
            path=tmp_path / "report.pdf",
            report_year=2025,
            retrieved_at="2026-10-12T10:00:00+00:00",
        )
    assert "Browsing saved data remains available" in str(error.value)


def test_startup_and_baseline_loading_with_pdf_and_cryptography_blocked(tmp_path):
    program = """
import builtins
import sys
original = builtins.__import__
def block(name, globals=None, locals=None, fromlist=(), level=0):
    if name.startswith(("pdfplumber", "pdfminer", "cryptography")):
        raise ImportError("Simulated native PDF binding failure")
    return original(name=name, globals=globals, locals=locals, fromlist=fromlist, level=level)
builtins.__import__ = block
from pathlib import Path
from best_cars import launcher, app, report
from best_cars.storage import load_snapshot
evidence, events, manifest = load_snapshot(directory=Path(sys.argv[1]))
assert len(evidence) == 1224
assert report.report_html(evidence=evidence, events=events).startswith("<!doctype html>")
print("Saved-data startup and report work without PDF/native cryptography imports")
"""
    result = subprocess.run(
        args=[sys.executable, "-c", program, str(tmp_path / "data")],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "work without PDF" in result.stdout
