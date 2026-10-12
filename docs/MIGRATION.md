# Migration from the original TÜV scripts

Original scripts, HTML, TSVs and figures are retained. Their previous README is in `docs/LEGACY_README.md`. The installable package copies the real multiyear baseline from commit `4ba08eda47799918fcd6ca64d7011c9a341282f3`, so wheels work without the original checkout layout.

| Original concept | New treatment |
| --- | --- |
| Raw TÜV percentage | `value`, `unit=percent`, explicit significant-defect metric |
| `year` | `report_year` plus separate `observation_year` |
| `year_of_production` | Qualified approximate registration-cohort bounds |
| Faults divided by odometer | Retained only in legacy outputs; excluded from new scores |
| Raw average rank | Cohort percentiles averaged per source before weighting |
| Missing source | Explicit gap, never zero |
| Model string | Original label plus conservative model identity |
| No app | Eight-page Streamlit explorer |

Keep data outside the software repository. Follow the download's `START_HERE.md` for a guarded base-specific patch or overlay. Do not rsync `--delete` or overwrite `.git`. The complete `overlay/` also runs independently with `--data-dir` pointing to `verified_data/`.
