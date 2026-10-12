# Best Cars 2.3 — start here

Use MAC_INSTALL.md for the preview/apply overlay route on your existing Mac
checkout. Set CAR_UPDATE_DIR to Downloads/best_cars_upgrade_20261012_v23.
Reinstall from the checkout and check version 2.3.0 before launching.

Version 2.3 reviews the latest screenshots and adds:
- Explicit score-versus-published-rate heatmaps with matching text/colour values,
  categorical age axes and a separate grey missing-value layer.
- Same-cohort peer median comparisons excluding the target model.
- Weighting scenario rank movement and rank-range summaries on common models.
- A dedicated Electric and hybrid vehicles page with battery-electric, plug-in
  hybrid and hybrid selections, age profiles and deeper interrogation.
- Reviewed propulsion eligibility, manufacturer references and explicit mixed
  family/unknown handling. Coverage is conservative, not exhaustive.
- Readable source labels and panel headings positioned above charts.

See overlay/docs/SCREENSHOT_REVIEW_V23.md and overlay/docs/PROPULSION.md.
The HTML report works offline; TSV exports use default report filters, ages
2–15 and minimum known sample 100. Corolla weighting examples retained from
2.2 use ages 4–20, as stated in the accompanying source audit.

The data snapshot is unchanged: 8,924 real reliability observations and the
saved exact US 2020 Corolla safety example. No battery-health estimates or
per-vehicle longevity database are fabricated. PDF dependencies remain optional
for saved-data browsing. The cumulative manifest preserves your extra Fiesta
files and all unlisted files; changed listed files receive backups.

The legacy-base changes.patch is for a clean original checkout only. Use the
overlay updater for your existing newer or previously updated checkout.
No remote GitHub mutation is performed.
