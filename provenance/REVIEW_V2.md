# Second release review: methods, evidence, and remaining limits

This review concerns the local release package, not a new set of paper results. It fixes default code paths and adds explicit evaluation records. It does not erase historical selection, certify the experiments as risk-free, or turn previously inspected data into an unseen test set.

## Changes and interpretation

| Issue found in historical workflow | v2 action | Remaining limit |
|---|---|---|
| Wind sources selected/renamed by test results; special PV seeds selected by test R-squared | Default training includes all raw sources and requested seeds. Raw IDs are stable; test-driven selector scripts are guarded before execution. | Archived display IDs, weights, and results still reflect history. New test intervals were not unknown to earlier researchers. |
| Month-wise split after window construction | New chronological interval-local windows with disjoint raw-observation intervals | One final season is not a complete seasonal/generalization study. |
| Daytime-only PV rows concatenated across nights | Full regular grids; missing periods remain visible | Real source clock, timezone, and measurement latency need confirmation. |
| Target-time weather available to the physics branch but not all baselines | Physics inputs now come from the last past row, with the same masking/imputation as temporal inputs | Forecast horizon and synthetic outages are limited; real weather latency and outages need separate study. |
| Physical inputs remained clean during historical missing-window experiments | All models receive identical sensor masks; physical inputs use those same masked values | Masks operate on window cells. The same timestamp in different windows may get different masks. This is not a continuous sensor-outage model. |
| Extreme thresholds/fallback depended on full data or test subset | Training-only thresholds, no test-based fallback; report all test targets and subset counts | Extreme-subset membership still uses observed target weather to describe conditions, never as a model input. Missing target weather is not classified as extreme. |
| Unequal historical postprocessing/information paths | Save both raw outputs and a common train-derived clipped output | A train-derived bound is empirical, not verified equipment nameplate capacity. Different architectures and physics priors are not equal parameter budgets. |
| Mixed or overwritten experiment outputs | New/empty output directories, immutable input directories, protocol saved before fitting, per-run failures and all-seed summaries | Users can still manually compare many experiments; reporting discipline and independent validation remain necessary. |
| Scheduling changed renewable supply and vehicle load together | Fixed-load default; both cases use the same vehicle work-power input profiles | Realized power and action times can differ. The historical paired mode remains explicitly labelled. |
| Station scale used later observations | Fixed-load scaling uses the source prefix before the first replay day, with a shared scale and peak clipping | Archived predictors still use target-time weather and non-independent training periods. This is offline replay, not a new held-out benchmark. |
| Derived flight profiles could be mistaken for raw measurements | Raw/redacted, idealized, and simulated records remain separate and documented | Vehicle-power table generation was not recovered. Alignment and battery assumptions require author review. |
| Chinese/mixed code comments | All Python comments and docstrings are English; automated scan added | Original filenames, data headers, and runtime strings are retained where needed. |

## Explicit preprocessing rules

The reviewed scope is PV year 2023 and wind year 2020, selected from the existing source definitions, not by new test scores. All raw files remain unchanged. This is a new documented protocol, not a claim of prospective preregistration.

Power duplicates are averaged at the same timestamp. Wind direction is converted to sine/cosine before averaging duplicate directions. PV weather is resampled into right-closed, right-labelled bins, then joined backward within one cadence. Records are reindexed to a fixed full-year grid. Invalid timestamps, records outside the stated year, duplicate timestamps, off-grid observations, missing labels, and invalid physical readings are counted in audit JSON. Off-grid observations are excluded rather than silently shifted.

Nonfinite readings become missing. Negative generation is excluded from the nonnegative-generation target; this choice must be checked against the source definition, since negative net power can represent real consumption. Negative irradiance/wind speed and temperatures at or below absolute zero are treated as missing. No test-error-based cleaning, daytime filter, or performance-based source removal is used. No claim is made that these checks detect all sensor faults.

Training medians fill missing features, with missingness indicators given to every model. Labels are not imputed for scoring. Each split loses its own initial window/horizon warmup rows, recorded in split counts. Time membership and training statistics are exported. Input hashes and code hashes accompany reviewed checkpoints; the reviewed prediction exporter verifies them and refuses changed snapshots.

Default hyperparameters and seed lists are fixed in code and recorded before a run. Validation selects the best epoch, while testing is evaluated after fitting each requested model/seed. Test results must not be used to modify settings and then be presented as a first-time evaluation. Per-source all-seed summaries are not a substitute for considering source variability. No full new benchmark has been completed during this packaging review.

## Verification actually performed

Machine-readable details are in `reviewed_verification.json` and `reviewed_data_audits.json`. These are software/data-audit records, not performance tables for a paper.

- Preprocessing audit completed for every supplied raw PV and wind source.
- Regression checks passed for causal aggregation, split isolation, train-only statistics, model forward/backward interfaces, consistent sensor masking, empty-subset reporting, fixed vehicle input loads, English comments/docstrings, and immediate legacy guards.
- One-epoch development training on one raw PV source and one raw wind source, using DNN and PINN, completed with validation-selected checkpoints and all missing-input evaluation outputs.
- Reviewed checkpoint prediction export was tested with frozen statistics and code/input hash verification. Valid-target predictions were compared against the training evaluation output.
- The low-initial-charge G3 fixed-load scheduling pair ran to completion. Its metadata and finite histories were checked. It is a functional replay test, not a forecast skill claim.
- Every supplied historical checkpoint was rechecked for forward/backward compatibility. CPU predictions have small differences from archived values, recorded in `reviewed_archived_model_check.json`; exact archived numeric equality is not asserted.
- Distributed data, archived weights, and reference results were checked against the preceding release checksums. The original research directory was not edited.

The full reviewed all-source/all-model/all-seed training matrix, all scheduling groups under the new default, GPU determinism, physical hardware operation, and independent external validation were **not** completed in this review. Development-run outputs are kept outside the publication package rather than mixed into `results_reference/`.

## Publication checklist requiring author decisions

1. Confirm dataset origin, redistribution permission, and code license. No license or legal clearance can be inferred from local possession.
2. Confirm timestamps/timezones, measurement units, negative power semantics, missing-data definitions, and whether each feature is actually available at prediction time.
3. Supply the missing generation steps for the vehicle-power tables, and verify idealization/battery assumptions against laboratory records. Confirm remaining local geometry and flight fields are appropriate for release; removal of coordinates/free text is not a full privacy certification.
4. Complete the new training matrix without selecting sources/seeds by test outcomes. Use genuinely independent data or a separately locked future evaluation before strong generalization claims.
5. For an end-to-end forecast/scheduling claim, integrate reviewed out-of-sample forecasts into a corresponding independently timed scheduling study. The current scheduler deliberately continues to label its archived-model use as replay.
6. Align the manuscript, supplement, and captions with the experiment actually reported. Do not attribute the new protocol to old checkpoints or replace old tables with development-run metrics.

## Provenance

`source_manifest.json` records original source paths/hashes and initial copy transformations. `packaging_changes.json` records current differences from those source files. `reviewed_changes.json` records v1-to-v2 differences, including newly authored files. `SHA256SUMS.csv` covers this release; the preceding manifest is kept under `history/`. The old version and verification notes are retained for historical interpretation. Inspecting history is supported; running guarded legacy selection scripts is not.
