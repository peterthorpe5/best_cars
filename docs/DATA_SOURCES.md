# Data-source audit — 12 October 2026

## Incorporated reliability tables

| Source | Verification and actual content | Interpretation |
| --- | --- | --- |
| TÜV / Car-Recalls.eu | 1,224 real repository rows for reports 2024/2025; raw rates, odometer readings and source ranks retained. | Secondary transcription of German inspection defects; no model denominators supplied. |
| ADAC | Official 2025 PDF downloaded; 969 numeric observations, 158 parsed model groups; selected Toyota RAV4/C-HR, VW Golf and BMW i3 values checked against table cells. | Attended roadside breakdowns in 2024 per 1,000 registered vehicles; no model denominators supplied. |
| Traficom | Official metadata and full annual PxWeb queries: 2,204 + 2,256 + 2,271 usable rows for inspections 2023/2024/2025. Published 2025 XLSX also inspected. | Finnish model/cohort inspection rates, actual test/failed-test counts and odometer data; totals/suppressed cells excluded. |

Primary references:

- [ADAC 2026 public article](https://www.adac.de/rund-ums-fahrzeug/unfall-schaden-panne/adac-pannenstatistik-2026/).
- [Actual official ADAC 2025 PDF used](https://assets.adac.de/image/upload/v1744788916/ADAC-eV/KOR/Text/PDF/Pannenstatistik_2025_final_nrmvls.pdf).
- [Traficom statistics and workbook downloads](https://tieto.traficom.fi/en/statistics/statistics-inspections).
- [Traficom model-level table](https://trafi2.stat.fi/PXWeb/pxweb/en/TraFi/TraFi__Katsastuksen_vikatilastot/040_kats_tau_104.px/).
- [Exact Traficom API](https://trafi2.stat.fi/PXWeb/api/v1/en/TraFi/Katsastuksen_vikatilastot/040_kats_tau_104.px).
- [Traficom open-data terms](https://tieto.traficom.fi/en/open-data).
- [TÜV-Verband 2026 summary](https://www.tuev-verband.de/pressemitteilungen/tuev-report-2026).

An official download link surfaced in the 2026 ADAC article led to a PDF still labelled 2025 with numeric tables for 2024; it is not treated as 2026 numeric evidence. The default adapter uses the original unambiguous 2025 PDF. Report totals of vehicle series can differ from distinct parsed combined source groups.

A secondary TÜV 2026 URL following the historical pattern returned 404. The primary summary gives selected winners/comparisons, not a verified complete bulk model table here. Selective winners are not ranked as though their peer set were complete.

## UK MOT — genuine data, local import supported

- [Official DVSA results for 2023/2024](https://www.data.gov.uk/dataset/c63fca52-ae4c-4b75-bab5-8b4735e1a4c9/anonymised-mot-tests-and-results).
- [Historical Department for Transport results](https://www.data.gov.uk/dataset/e3939ef8-30c7-4ca8-9c7c-ad9475cc9b2f/anonymised-mot-tests-and-results).
- [Official field/outcome guide](https://data.dft.gov.uk/anonymised-mot-test/test_data/mot-testing-data-user-guide-v5.1.odt).
- [Separate authorised bulk history service](https://documentation.history.mot.api.gov.uk/mot-history-api/download-vehicle-mot-history-data).

The replacement DVSA listing, updated 23 May 2025, includes 2024. Both 2023 and 2024 result archive access was verified. The 2024 ZIP is 1,531,265,132 bytes compressed and contains twelve monthly result files (plus macOS metadata). A bounded compressed prefix was decompressed and its actual header/schema inspected. This verifies acquisition and parsing feasibility, not collection of a complete national dataset. No preview is passed off as a representative UK ranking; no national UK snapshot is bundled.

The published 2024 link requires spaces encoded as `%20` rather than the listing's encoded plus characters. The verified public URL is [2024 results ZIP](https://edh-dvsa-data-gov-uk-files-prod.s3.eu-west-1.amazonaws.com/MOT%20testing%20data%20results%20%282024%29.zip). This is a large download; inspect available disk space first.

The guide specifies historical pipe-delimited results with 14 columns. The actual 2024 archive uses headered comma-delimited files with 15 fields, adding `completed_date`; decimal odometer strings and ISO dates are preserved/parsed correctly. The importer streams all result members, skips macOS metadata and deduplicates test IDs across the archive on disk; conflicting records fail. Its denominator selects Class 4 normal `NT` tests with `P`, `F`, `PRS` outcomes; initial failures include `F` and `PRS`. Retests/aborted/abandoned tests are excluded. Class 4 includes vehicles beyond passenger cars. Earlier mileage-unit ambiguity remains missing.

The guide defines Vehicle ID as based on registration and VIN, permitting longitudinal tracking. The current failure-rate importer does not retain this field. Cross-release identifier continuity, test gaps and registration changes need auditing before estimating service life; see LONGEVITY.md.

The bounded preview comprises 11,076 source records, including duplicates and
conflicting eligible odometer records. With explicit quarantine, it produced
5,512 deduplicated eligible initial tests over 1,199 model/registration cohorts,
excluding four detected conflicting eligible IDs. These numbers describe that
archive prefix only; it is not a representative sample and is not included in
the verified ranking snapshot. Strict conflict handling remains the default.
Duplicate canonical aliases are pooled from actual counts; repeated identical
raw aggregates are rejected to prevent double counting.

The data carry the [Open Government Licence](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/). Historical open files do not imply access to a current authenticated vehicle-history feed.

## Incorporated official safety context

- [NHTSA documentation and datasets](https://www.nhtsa.gov/nhtsa-datasets-and-apis).
- [Exact recall example](https://api.nhtsa.gov/recalls/recallsByVehicle?make=toyota&model=corolla&modelYear=2020).
- [Exact complaints example](https://api.nhtsa.gov/complaints/complaintsByVehicle?make=toyota&model=corolla&modelYear=2020).

Both endpoints were exercised. Saved example: **US 2020 Toyota Corolla only**, with 3 recalls and 311 complaints at acquisition. This is not a complete US vehicle database. Exact names are supplied for every lookup; UK equivalents are not inferred. VIN fields are not retained in normalised event outputs.

Voluntary complaints have reporting/selection bias; campaigns differ in scope and practice. No comparable fleet exposure appears in these endpoints, so neither count enters the reliability ranking.

## Identified without imported tables

| Source | Finding | Treatment |
| --- | --- | --- |
| DVSA UK recalls | Official service identified; access/redirect errors in the build environment. | Direct app check link; no downloaded UK recall data claimed. |
| Which? | Credible UK survey; full model tables restricted. | Candidate entry only. |
| What Car? | Published owner survey but selective articles; no reproducible complete bulk feed verified. | Candidate entry only. |
| Consumer Reports | US survey/predicted reliability; full access restricted. | Candidate entry only. |

Every incorporated rate retains source, original model label, country, report/observation years, registration cohort, metric, units, known counts, mileage basis, exact URL and acquisition time. Snapshots include checksums and counts. Explicit refreshes fail visibly and preserve previous observations on source errors. Availability and schemas can change; this audit records what was actually verified on its date.
