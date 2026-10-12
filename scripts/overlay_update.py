"""Preview/apply manifest-listed updates with backups, without changing Git."""

import argparse
import csv
import hashlib
import logging
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def file_sha256(*, path: Path) -> str:
    """Hash a file in bounded memory.

    Args:
        path: File to inspect.

    Returns:
        Lowercase SHA-256 hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open(mode="rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(*, path: Path) -> dict[str, str]:
    """Read an explicit TSV list of update paths and their expected hashes.

    Args:
        path: Package UPDATE_FILES.tsv, with path and sha256 columns.

    Returns:
        Relative project paths mapped to verified package digests.

    Raises:
        ValueError: For malformed, duplicate, unsafe or reserved paths.
    """
    result = {}
    with path.open(mode="r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(f=stream, delimiter="\t")
        if reader.fieldnames != ["path", "sha256"]:
            raise ValueError("Update manifest must contain path and sha256 columns.")
        for row in reader:
            relative = row["path"]
            parts = Path(relative).parts
            if (
                not parts
                or Path(relative).is_absolute()
                or ".." in parts
                or ".git" in parts
                or "\\" in relative
                or relative in result
                or not re.fullmatch(pattern=r"[0-9a-f]{64}", string=row["sha256"] or "")
            ):
                raise ValueError(f"Invalid update manifest entry: {relative!r}")
            result[relative] = row["sha256"]
    if not result:
        raise ValueError("Update manifest contains no files.")
    return result


def overlay_plan(
    *, repo: Path, overlay: Path, manifest: Mapping[str, str]
) -> list[dict[str, str]]:
    """Verify an update and identify only manifest-listed changed files.

    Args:
        repo: Existing local Git checkout.
        overlay: Unpacked package overlay directory.
        manifest: Validated relative paths and package digests.

    Returns:
        Add/replace operations, with existing and replacement file digests.
        Unlisted files, identical files and Git metadata receive no operation.

    Raises:
        ValueError: For overlapping roots, symlinks, unsafe paths or bad hashes.
    """
    repo = repo.resolve(strict=True)
    overlay = overlay.resolve(strict=True)
    if not repo.is_dir() or not (repo / ".git").exists() or not overlay.is_dir():
        raise ValueError(
            "Require an existing Git checkout and unpacked overlay directory."
        )
    if repo == overlay or repo in overlay.parents or overlay in repo.parents:
        raise ValueError("Repository and overlay directories must not overlap.")
    changes = []
    for relative, expected in sorted(manifest.items()):
        parts = Path(relative).parts
        if (
            not parts
            or Path(relative).is_absolute()
            or ".." in parts
            or ".git" in parts
            or "\\" in relative
        ):
            raise ValueError("Update paths must remain inside the checkout.")
        source = overlay / relative
        target = repo / relative
        for root, item in [(overlay, source), (repo, target)]:
            if not item.resolve().is_relative_to(root):
                raise ValueError(f"Update path escapes its root: {relative}")
            if any(
                part.is_symlink()
                for part in [item, *item.parents]
                if part != root and root in part.parents
            ):
                raise ValueError(
                    f"Symlinks are not overwritten or followed: {relative}"
                )
        if not source.is_file() or (target.exists() and not target.is_file()):
            raise ValueError(f"Update path is not a regular file: {relative}")
        if file_sha256(path=source) != expected:
            raise ValueError(f"Package checksum mismatch: {relative}")
        previous = file_sha256(path=target) if target.exists() else ""
        if previous != expected:
            changes.append(
                {
                    "path": relative,
                    "action": "replace" if previous else "add",
                    "before": previous,
                    "after": expected,
                }
            )
    return changes


def atomic_copy(*, source: Path, target: Path) -> None:
    """Replace one file using a temporary neighbour and an atomic rename.

    Args:
        source: Verified regular file to copy.
        target: Regular destination file within the checked project root.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="best_cars_update_", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(src=source, dst=temporary)
        os.replace(src=temporary, dst=target)
    finally:
        temporary.unlink(missing_ok=True)


def apply_overlay(
    *, repo: Path, overlay: Path, manifest: Mapping[str, str], backup_parent: Path
) -> Path | None:
    """Back up replaced files before applying the verified update plan.

    Args:
        repo: Existing Git checkout.
        overlay: Unpacked package overlay directory.
        manifest: Paths and expected hashes from the package manifest.
        backup_parent: Backup directory outside both repository and overlay.

    Returns:
        Backup directory with restore files and a TSV record, or None if current.

    Raises:
        ValueError: For invalid paths, package checksums or backup location.
        RuntimeError: If application fails; applied files are rolled back.
    """
    repo = repo.resolve(strict=True)
    overlay = overlay.resolve(strict=True)
    backup_parent = backup_parent.resolve()
    if (
        backup_parent == repo
        or repo in backup_parent.parents
        or backup_parent == overlay
        or overlay in backup_parent.parents
    ):
        raise ValueError("Backups must be outside the repository and overlay.")
    changes = overlay_plan(repo=repo, overlay=overlay, manifest=manifest)
    if not changes:
        return None
    backup_parent.mkdir(parents=True, exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix="best_cars_", dir=backup_parent))
    for item in changes:
        if item["action"] == "replace":
            target = backup / "files" / item["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src=repo / item["path"], dst=target)
    with (backup / "RESTORE_FILES.tsv").open(
        mode="w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            f=stream, fieldnames=["path", "action", "before", "after"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(rowdicts=changes)
    applied = []
    try:
        for item in changes:
            if file_sha256(path=overlay / item["path"]) != item["after"]:
                raise ValueError(f"Package changed during update: {item['path']}")
            atomic_copy(source=overlay / item["path"], target=repo / item["path"])
            applied.append(item)
    except Exception as error:
        for item in reversed(applied):
            if item["action"] == "replace":
                atomic_copy(
                    source=backup / "files" / item["path"], target=repo / item["path"]
                )
            else:
                (repo / item["path"]).unlink(missing_ok=True)
        raise RuntimeError(
            f"Update failed; applied files restored. Backups: {backup}"
        ) from error
    LOGGER.info("Applied %s file updates; backups: %s", len(changes), backup)
    return backup


def main(*, argv: list[str] | None = None) -> int:
    """Preview updates by default; apply only with an explicit CLI switch.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Zero on preview/success; one for a visible update error.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(args=argv)
    try:
        manifest = read_manifest(path=args.manifest)
        changes = overlay_plan(repo=args.repo, overlay=args.overlay, manifest=manifest)
        for item in changes:
            print(f"{item['action']:7} {item['path']}")
        print(
            f"{len(changes)} changed files; unlisted files and Git metadata are preserved."
        )
        if args.apply:
            backup = apply_overlay(
                repo=args.repo,
                overlay=args.overlay,
                manifest=manifest,
                backup_parent=args.backup_dir
                or args.repo.resolve().parent / "best_cars_backups",
            )
            print(
                f"Applied successfully. Backup: {backup}"
                if backup
                else "Already up to date."
            )
        else:
            print("Preview only. Add --apply to copy these updates with backups.")
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Update stopped: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
