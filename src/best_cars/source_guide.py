"""Plain-language source definitions and page-specific interpretation help."""

SOURCE_GUIDE = {
    "TÜV inspection defects": {
        "meaning": "The imported tables describe the percentage of inspected vehicles with significant roadworthiness defects, by model and age band. A vehicle can have more than one defect: this percentage is not a count of individual fault events. Average odometer mileage is separate context, not the rate denominator.",
        "faults": "German HU inspections assess roadworthiness. TÜV describes minor, significant and dangerous defect categories, plus vehicles deemed unsafe. Typical problems include lights, brakes, tyres, springs/suspension, corrosion and oil leaks. A minor defect is not automatically equivalent to a significant inspection failure. The imported summary does not retain defect-level categories for individual models.",
        "limits": "This measures condition at inspection, including wear and maintenance. It does not measure every repair, engine durability, lifetime, breakdowns per mile or repair invoices. The app imports 2024/2025 secondary Car-Recalls.eu tables, not a primary TÜV microdatabase. Model denominators are unavailable, so no TÜV model-specific Wilson interval is calculated. Registration cohorts are approximate; report year is not the inspection date.",
        "data_url": "https://car-recalls.eu/reliability/",
        "definition_url": "https://www.tuv.com/germany/de/lp/lp-haupt-und-abgasuntersuchung/distribution-pages/m%C3%A4ngelklassen/",
    },
    "ADAC roadside breakdowns": {
        "meaning": "The loaded 2025 official report describes attended breakdown events during 2024 per 1,000 registered vehicles, separated by registration year. This denominator is registered vehicles, not distance travelled or repair invoices.",
        "faults": "A roadside breakdown requiring assistance can concern starting or electrical problems as well as mechanical faults. The report's model fault list identifies Corolla starter-battery issues for 2019–2022 registrations. A 12 V battery problem is not evidence of traction-battery degradation or engine failure.",
        "limits": "Assistance events do not cover every garage repair or every assistance provider. Model denominators and fault-specific counts are unavailable in our import, so no model-specific confidence bounds or expected repair bills are inferred. A low percentile means poor relative performance among reported peers, not a high probability of a catastrophic failure.",
        "data_url": "https://assets.adac.de/image/upload/v1744788916/ADAC-eV/KOR/Text/PDF/Pannenstatistik_2025_final_nrmvls.pdf",
        "definition_url": "https://www.adac.de/rund-ums-fahrzeug/unfall-schaden-panne/adac-pannenstatistik-2026/",
    },
    "Traficom Finnish inspections": {
        "meaning": "Official Finnish periodic-inspection statistics report failed inspections, inspection counts and odometer summaries by model and year of entry into service. The inspected-event count, not kilometres, is the failure-rate denominator. Actual counts support the app's Wilson intervals.",
        "faults": "A recorded inspection decision can fail because of roadworthiness faults. The source publishes reasons for rejection and vehicle fault information; our loaded reliability rows retain aggregate rates and counts, not a complete fault-by-fault repair history.",
        "limits": "The unit is an inspection, not necessarily a unique car. Finnish testing, climate, servicing and fleet composition differ from Germany and the UK. Miles driven and survival cannot be reconstructed from mean odometers. Loaded observation years are 2023–2025; publications follow the inspection year.",
        "data_url": "https://tieto.traficom.fi/en/statistics/statistics-inspections",
        "definition_url": "https://tieto.traficom.fi/en/statistics/statistics-inspections",
    },
    "UK MOT import": {
        "meaning": "Local DVSA raw imports aggregate eligible initial Class 4 tests. Failed tests and passes after rectification are counted as initial failures; repeated and aborted tests are excluded by the importer. The denominator is eligible tests.",
        "faults": "An MOT checks specified roadworthiness and environmental requirements. Inspection faults are not a complete catalogue of mechanical breakdowns or all repairs made between tests. This app's aggregate importer does not retain per-fault severity or repair invoices.",
        "limits": "No national MOT snapshot is bundled. Imported aggregates do not retain linked per-vehicle histories for lifespan estimation. Genuine faults per distance would require fault-event counts and corresponding vehicle-distance exposure, not average odometers.",
        "data_url": "https://www.data.gov.uk/dataset/c63fca52-ae4c-4b75-bab5-8b4735e1a4c9/anonymised-mot-tests-and-results",
        "definition_url": "https://www.gov.uk/check-mot-history",
    },
    "NHTSA safety events": {
        "meaning": "Official recalls and voluntary complaints refer to exact US make/model/year combinations. A recall campaign is a safety action; a complaint is an owner report. Event counts are not fleet failure rates.",
        "faults": "These are safety signals and narratives, rather than equivalent inspection failures. Complaints can be unverified, repeated or influenced by reporting behaviour; UK specifications and recall applicability can differ.",
        "limits": "No comparable exposure denominator exists in these lookups. Records are excluded from reliability and repair-cost scores. The saved example concerns the US 2020 Toyota Corolla, not every Corolla generation.",
        "data_url": "https://www.nhtsa.gov/nhtsa-datasets-and-apis",
        "definition_url": "https://www.nhtsa.gov/recalls",
    },
}

PAGE_HELP = {
    "Overview": "Start with separate source profiles. A favourable inspection result can coexist with roadside breakdowns. Profiles are visible even when there is no fair overall rank. Mean percentiles spanning ages are descriptive, not predictions.",
    "Find reliable cars": "Choose the outcome and source you wish to compare, then an exact published age/registration cohort. No source is preselected as universally trustworthy. Optional exploratory composites disclose missing sources and assume percentile 50; strict mode matches coverage. A top position concerns this comparison only.",
    "Compare cars": "Select one model or several. Tabs show age profiles, exact-cohort peer gaps, matched model rates, counted Wilson intervals, evidence coverage and historical summaries. Coverage colours indicate data availability, not reliability. Across-years mode holds age or registration bounds fixed; it never tracks individual cars. Shared-cohort mode matches all selected models in the current-report charts.",
    "Age and mileage": "Heatmap cells use either source-cohort percentiles (higher better) or published rates (lower better). Mileage is context, not a fault-rate denominator. Alphabetical row ordering avoids suggesting a general league table across unmatched ages.",
    "Mileage explorer": "Odometers are cohort averages. Score-sized points use an optional within-block preference score; small open markers indicate unavailable scores, not unreliable cars. Association with mileage does not establish causation or faults per mile.",
    "Evidence agreement": "Mean source percentiles can cover different cohorts. This page highlights conflicting observations; it does not rank general reliability. Missing cells are not zeroes or proof of good performance.",
    "Deeper analysis": "Peer gaps match exact source contexts. Preference sensitivity compares the same coverage block. Wilson plots use separate age/source/report panels by default, with optional two-year groups and panel selection. Intervals describe sampling precision in counted tests, not repair cost, lifetime or unmeasured systematic bias.",
    "Electric and hybrid vehicles": "Propulsion eligibility is conservatively reviewed. Mixed model families are optional context only. The data do not measure battery health, battery replacement risk or charging reliability.",
    "Repair costs": "Published job prices are dated marketplace context. Enter actual quotes and explicit job counts for a budget scenario. No frequency or annual repair cost is inferred from inspection rates, breakdown totals or percentiles.",
    "Data sources and fault definitions": "Read the outcome, denominator, fault definition and limitations separately. A source's definition of a fault determines what conclusions its numbers can support.",
    "Recalls and complaints": "Safety events concern exact US vehicles, not a comparative failure probability. Check UK recall applicability for the specific car. These counts never enter reliability scores.",
    "Sources and methods": "Source-cohort percentiles retain their original peers. Optional composites balance outcomes first. Exploratory mode exposes gaps and sensitivity bounds; strict mode matches coverage. No score estimates faults per mile, mechanical lifespan or expected repair bills.",
    "Data management": "Refreshes preserve prior evidence when a source fails. Loaded report dates matter more than retrieval dates. Importing aggregate MOT data does not create a survival database or defect-level repair history.",
}
