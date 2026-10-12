# Version 2.3 validation

- 177 automated tests passed; package line coverage 97.48% (minimum 95%).
- All ten Streamlit pages passed AppTest with 8,924 actual observations.
- Unit tests cover heatmap text/colour matrices, colour direction, fixed scales,
  categorical axes, missing layers, exact-cohort peer reference exclusion,
  scenario common competitors, propulsion mixed-family exclusion and opt-in.
- Populated and empty electric, plug-in and hybrid UI paths are exercised.
- Ruff lint/format and Git whitespace checks passed.
- Wheel and source distribution version 2.3.0 built successfully.
- Cumulative patch clean-base application check and overlay preview/apply/
  idempotency checks passed, preserving extra user files and Git HEAD.
- Standalone wheel includes the reviewed propulsion catalogue.
- Offline HTML has embedded Plotly and no remote asset dependency.

Review used ten usable uploaded screenshots; the 15.17.50 strip was unusable.
New native UI browser screenshot validation and physical macOS execution are
not claimed. Test runtime is Linux/Python 3.12.14. No national MOT collection,
battery state-of-health measurements or car lifespan estimates are bundled.
Previous native-PDF isolation and MOT parser tests remain in the passing suite.
