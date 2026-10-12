# Verified car reliability evidence

This directory contains the current processed snapshot: 8,924 actual reliability
observations from TÜV, ADAC and Traficom, plus 314 NHTSA safety records for the
explicit US 2020 Toyota Corolla example. The package includes no national UK MOT
bulk snapshot. Launch with --data-dir pointing here. No source download caches
are bundled; offline browsing works immediately. Source re-parsing offline needs
cached source responses, which can be downloaded explicitly through the CLI.

The source audit is in ../overlay/docs/DATA_SOURCES.md. Compressed TSVs and their
manifest checksums are preserved. Model counts are model-series identities and
may span engines/generations. Safety records never enter reliability rankings.
