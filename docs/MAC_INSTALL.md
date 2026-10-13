# Mac update and launch — version 3.1

These commands use your existing Miniforge environment and this checkout:
`/Users/PThorpe001/github_repos/best_cars`.

Your backup commit changed HEAD, so the previous exact-base guard stopped as
designed. The subsequent `git apply` output contained whitespace warnings,
which are not themselves an application failure. Run `git status --short` to
see the resulting changes. The new overlay route works with a different HEAD
and an already partly/applied previous upgrade; it does not run `git apply`.

## 1. Extract the new package

Extract `best_cars_upgrade_20261013_v31.zip` with Finder. Check that Downloads
contains `best_cars_upgrade_20261013_v31/UPDATE_FILES.tsv` and `overlay/`.
Set these variables in the Terminal where you will run the following commands:

```bash
conda activate best_cars
CAR_UPDATE_DIR="$HOME/Downloads/best_cars_upgrade_20261013_v31"
CAR_REPO_DIR="/Users/PThorpe001/github_repos/best_cars"
```

## 2. Preview the overlay

```bash
python "$CAR_UPDATE_DIR/overlay/scripts/overlay_update.py" \
  --repo "$CAR_REPO_DIR" \
  --overlay "$CAR_UPDATE_DIR/overlay" \
  --manifest "$CAR_UPDATE_DIR/UPDATE_FILES.tsv"
```

This only lists additions/replacements and validates package hashes. It does
not require HEAD to match an old commit. The manifest lists the upgrade files
only: other scripts/data/figures remain untouched, including your additional
`tuv2025_all_fieste.tsv` and `figures/rank_over_age_Ford_Fiesta.png`.

## 3. Apply the previewed updates

```bash
python "$CAR_UPDATE_DIR/overlay/scripts/overlay_update.py" \
  --repo "$CAR_REPO_DIR" \
  --overlay "$CAR_UPDATE_DIR/overlay" \
  --manifest "$CAR_UPDATE_DIR/UPDATE_FILES.tsv" \
  --apply
```

Each replaced file is backed up **before** changes begin, outside the checkout
under `/Users/PThorpe001/github_repos/best_cars_backups/`. The updater prints
the exact backup directory. `files/` holds originals and `RESTORE_FILES.tsv`
records operations. It does not delete unlisted files, change Git metadata,
commit or push. If copying fails, already applied file operations are rolled
back. Rerunning an unchanged package reports that it is up to date.

## 4. Reinstall from the GitHub checkout and check the import path

Your earlier editable install pointed to the Downloads overlay. Reinstall from
the updated checkout so future edits and launches use the same project:

```bash
cd "$CAR_REPO_DIR"
python -m pip install -e '.[dev]'
python -m pip check
python -c "import best_cars; print(best_cars.__version__); print(best_cars.__file__)"
```

Expect `3.1.0` and a module path under
`/Users/PThorpe001/github_repos/best_cars/src/best_cars/`.

## 5. Launch using the supplied actual data

```bash
best-cars-app \
  --data-dir "$CAR_UPDATE_DIR/verified_data" \
  --server-port 8501
```

Open http://127.0.0.1:8501. If the port is occupied, stop your earlier app with
Ctrl+C or choose `--server-port 8502`. Browsing saved data does not import
`pdfplumber`, `pdfminer` or their native cryptography dependency. The HTML
report also opens directly without Python.

Keep the supplied data folder available while using this path. To store it
outside Downloads, copy `verified_data/` to a durable data directory and pass
that directory to `--data-dir`; avoid overwriting a different existing snapshot
directory. App startup does not download new source files.

## Optional: repair PDF support for fresh ADAC PDF imports

The reported `_OSSL_get_max_threads` error comes from a native cryptography
binding failing to load. Its precise environment cause has not been reproduced
on your Mac. Version 3.0 loads this dependency only when a PDF import is asked
for and shows a clear import error while preserving previously saved evidence.

From the activated `best_cars` environment and checkout, request prebuilt
cryptography/CFFI wheels rather than a local native build:

```bash
python -m pip install --upgrade pip
python -m pip install --force-reinstall --no-cache-dir \
  --only-binary=:all: cryptography cffi
python -m pip install -e '.[pdf]'
python -c "import pdfplumber; from cryptography.hazmat.backends.openssl.backend import backend; print(pdfplumber.__version__); print(backend.openssl_version_text())"
```

The [official cryptography installation guide](https://cryptography.io/en/latest/installation/)
describes statically linked macOS wheels and upgrading pip. These repair
commands require a compatible wheel for the Mac/Python architecture; no native
compilation is silently attempted. If they still fail, saved-data browsing
remains available. Inspect `python -m pip show cryptography`,
`python -m pip --version` and `python -c "import platform; print(platform.machine())"`
before changing architecture or system OpenSSL installations.

## 6. Verify, commit and push after reviewing

Stop the app in this Terminal or use another activated Terminal:

```bash
cd "$CAR_REPO_DIR"
python -m pytest --cov=best_cars --cov-report=term-missing
python -m ruff check src tests scripts app.py
python -m ruff format --check src tests scripts app.py
git status --short
git diff --stat
git add README.md .gitignore .github .streamlit MANIFEST.in app.py docs \
  environment.yml pyproject.toml requirements.txt scripts src tests
git commit -m "Upgrade reliability explorer, statistics report and Mac setup"
git push origin main
```

Review the status/diff before the add/commit/push commands. They assume your
current branch is `main`. Your backup commit is preserved; no reset, history
rewrite, force push or whole-directory overwrite is used.
