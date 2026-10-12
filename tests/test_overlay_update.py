"""Overlay updates preserve user additions and restore replacements on failure."""

import csv
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    name="overlay_update",
    location=Path(__file__).parents[1] / "scripts" / "overlay_update.py",
)
updater = importlib.util.module_from_spec(spec=SPEC)
SPEC.loader.exec_module(updater)


@pytest.fixture
def overlay_case(tmp_path):
    """Create a checkout with user files and two explicitly listed updates."""
    repo = tmp_path / "repo"
    overlay = tmp_path / "overlay"
    repo.mkdir()
    overlay.mkdir()
    (repo / ".git").mkdir()
    (repo / ".git" / "HEAD").write_text(data="User backup commit")
    (repo / "old.txt").write_text(data="Local changes")
    (repo / "tuv2025_all_fieste.tsv").write_text(data="User Fiesta analysis")
    (overlay / "old.txt").write_text(data="Updated app")
    (overlay / "new.txt").write_text(data="New statistics")
    # Bundled legacy files are not applied unless explicitly listed.
    (overlay / "tuv2025_all_fieste.tsv").write_text(data="Must not replace user work")
    manifest = {
        name: updater.file_sha256(path=overlay / name)
        for name in ["old.txt", "new.txt"]
    }
    manifest_path = tmp_path / "UPDATE_FILES.tsv"
    with manifest_path.open(mode="w", newline="") as stream:
        writer = csv.DictWriter(f=stream, fieldnames=["path", "sha256"], delimiter="\t")
        writer.writeheader()
        writer.writerows(
            rowdicts=[
                {"path": name, "sha256": digest} for name, digest in manifest.items()
            ]
        )
    return repo, overlay, manifest, manifest_path


def test_preview_apply_and_idempotency_preserve_git_and_user_files(
    overlay_case, tmp_path
):
    repo, overlay, manifest, path = overlay_case
    assert updater.read_manifest(path=path) == manifest
    plan = updater.overlay_plan(repo=repo, overlay=overlay, manifest=manifest)
    assert {item["action"] for item in plan} == {"add", "replace"}
    assert (repo / "old.txt").read_text() == "Local changes"
    backup = updater.apply_overlay(
        repo=repo,
        overlay=overlay,
        manifest=manifest,
        backup_parent=tmp_path / "backups",
    )
    assert (repo / "old.txt").read_text() == "Updated app"
    assert (repo / "new.txt").read_text() == "New statistics"
    assert (backup / "files" / "old.txt").read_text() == "Local changes"
    assert (backup / "RESTORE_FILES.tsv").is_file()
    assert (repo / ".git" / "HEAD").read_text() == "User backup commit"
    assert (repo / "tuv2025_all_fieste.tsv").read_text() == "User Fiesta analysis"
    assert updater.overlay_plan(repo=repo, overlay=overlay, manifest=manifest) == []
    assert (
        updater.apply_overlay(
            repo=repo,
            overlay=overlay,
            manifest=manifest,
            backup_parent=tmp_path / "backups",
        )
        is None
    )


def test_manifest_rejects_invalid_and_duplicate_paths(tmp_path):
    path = tmp_path / "manifest.tsv"
    for relative in ["../escape", "/absolute", ".git/HEAD", "bad\\path", ""]:
        path.write_text(data=f"path\tsha256\n{relative}\t{'0' * 64}\n")
        with pytest.raises(ValueError):
            updater.read_manifest(path=path)
    path.write_text(data="bad\theader\n")
    with pytest.raises(ValueError):
        updater.read_manifest(path=path)
    path.write_text(data="path\tsha256\n")
    with pytest.raises(ValueError):
        updater.read_manifest(path=path)
    path.write_text(data=f"path\tsha256\na\t{'0' * 64}\na\t{'0' * 64}\n")
    with pytest.raises(ValueError):
        updater.read_manifest(path=path)


def test_preflight_rejects_checksum_symlink_and_directory_conflicts(overlay_case):
    repo, overlay, manifest, _ = overlay_case
    with pytest.raises(ValueError, match="checksum"):
        updater.overlay_plan(repo=repo, overlay=overlay, manifest={"old.txt": "0" * 64})
    (repo / "new.txt").symlink_to(target=repo / "old.txt")
    with pytest.raises(ValueError, match="Symlinks"):
        updater.overlay_plan(repo=repo, overlay=overlay, manifest=manifest)
    (repo / "new.txt").unlink()
    (repo / "new.txt").mkdir()
    with pytest.raises(ValueError, match="regular"):
        updater.overlay_plan(repo=repo, overlay=overlay, manifest=manifest)
    with pytest.raises(ValueError, match="inside"):
        updater.overlay_plan(
            repo=repo, overlay=overlay, manifest={"../escape": "0" * 64}
        )
    with pytest.raises(ValueError, match="overlap"):
        updater.overlay_plan(repo=repo, overlay=repo, manifest=manifest)
    (repo / ".git").rename(target=repo / "Git-backup")
    with pytest.raises(ValueError, match="Git checkout"):
        updater.overlay_plan(repo=repo, overlay=overlay, manifest=manifest)


@pytest.mark.parametrize("failure_call", [2, 3])
def test_atomic_copy_and_failure_roll_back_replacements(
    overlay_case, tmp_path, monkeypatch, failure_call
):
    repo, overlay, manifest, _ = overlay_case
    (overlay / "zzz.txt").write_text(data="Third update")
    manifest["zzz.txt"] = updater.file_sha256(path=overlay / "zzz.txt")
    original = updater.atomic_copy
    calls = 0

    def fail_second_copy(*, source, target):
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise OSError("Simulated disk failure")
        return original(source=source, target=target)

    monkeypatch.setattr(updater, "atomic_copy", fail_second_copy)
    with pytest.raises(RuntimeError, match="restored"):
        updater.apply_overlay(
            repo=repo,
            overlay=overlay,
            manifest=manifest,
            backup_parent=tmp_path / "backups",
        )
    assert (repo / "old.txt").read_text() == "Local changes"
    assert not (repo / "new.txt").exists()
    assert not (repo / "zzz.txt").exists()
    with pytest.raises(ValueError, match="outside"):
        updater.apply_overlay(
            repo=repo,
            overlay=overlay,
            manifest=manifest,
            backup_parent=repo / "backups",
        )


def test_cli_preview_apply_and_errors(overlay_case, capsys):
    repo, overlay, _, path = overlay_case
    argv = ["--repo", str(repo), "--overlay", str(overlay), "--manifest", str(path)]
    assert updater.main(argv=argv) == 0
    assert "Preview only" in capsys.readouterr().out
    assert updater.main(argv=[*argv, "--apply"]) == 0
    assert "Applied successfully" in capsys.readouterr().out
    assert updater.main(argv=[*argv, "--apply"]) == 0
    assert "Already up to date" in capsys.readouterr().out
    path.write_text(data="invalid manifest")
    assert updater.main(argv=argv) == 1
    assert "Update stopped" in capsys.readouterr().out
