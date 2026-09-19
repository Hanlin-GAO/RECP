# Source history and release changes

This file records the first packaging pass and the historical experiments. The v2 default behavior is documented in [REVIEW_V2.md](REVIEW_V2.md) and the root README. Statements below about preserving the original training procedure refer to v1, not the v2 training entry. Original data, archived weights, and reference results remain unchanged.

## Selected working version

The input repository contained one Git commit (`d161440`, dated 2026-06-21, `Initial upload`) and later working-tree changes to manuscript files. Multiple experiments and manuscript versions were stored side by side in the directory; Git history alone does not identify a unique final experiment. The working-tree files, rather than a clean checkout of the commit, were the packaging source.

This package uses the `VStry/main.py`, `model.py`, `wind_main.py`, and `wind_model.py` implementations for prediction, and `VSgo/mix/run_manual_validation_matrix.py` plus its imported modules for scheduling. The complete `validation_matrix_manual_first_frame` CSV/JSON collection was retained as the scheduling reference. It has twelve groups and two cases per group, and is the collection cited in `supply/Manuscript_v6.tex`. The other `validation_matrix` and incomplete `validation_matrix_manual_first_frame_experiment_power` directories were not merged into it.

## How the implementation fits together

- `main.py`: PV data loading, feature windows, month-wise splits, train-only scaling, training/evaluation, extreme-weather subset and missing-feature tests.
- `model.py`: DNN, LSTM with attention, GRU, TCN, Transformer, and the PV physics-informed model. The PV model combines a physical irradiance/temperature branch with a temporal correction.
- `wind_main.py` and `wind_model.py`: wind preprocessing and training; a temperature-adjusted turbine power curve is combined with a GRU branch and an input-dependent gate.
- `io_utils.py`: reconstructs the appropriate model from a checkpoint, evaluates it on the archived feature tables, resamples power and supplies it to scheduling. Output clipping/low-source rules follow the original implementation.
- `run_manual_validation_matrix.py`: chooses renewable time slices and initial-charge templates, assigns the experimental/proxy load columns, runs each pair, and writes histories, figures and metrics.
- `simulator.py`, `planning.py`, `kinematics.py`, `station_manager.py`: mission states, return/resume decisions, travel-energy estimates, charging reservations, FIFO queues, power limits and energy updates.
- `scenarios.py` and `drone_profiles.py`: read field geometry, manually annotated routes and derived transport timing/energy profiles.

This release preserves these operational algorithms. It does not add a new global optimizer, a hardware controller, or an implementation of every theoretical phase-diagram/threshold construction described in the manuscript.

## Wind source identities: verified from data

| Archived directory used for inference | Original workbook matched to its processed records |
|---|---|
| `Turbine_1` | `Wind 6.xlsx` |
| `Turbine_2` | `Wind 4.xlsx` |
| `Turbine_3` | `Wind 3.xlsx` |

The mapping was checked against all six original workbooks using the original loader. Every timestamp and each wind-speed, direction, temperature and power value matched the corresponding processed table within the recorded tolerance; the observed maximum difference was zero. The machine-readable evidence is `wind_source_mapping.json`.

The historical `wind_train_v6.log` selects original sources 3, 1, 4; `wind_train_v7.log` selects 6, 5, 3. Neither list completely matches the current processed collection. Therefore logs are preserved as history, not treated as an authoritative mapping for the present checkpoints. The exact operation that changed the second retained source was not established from the available files.

The legacy `select_best_3()` ranks sources using normal-test and extreme-test PINN performance, renames selected directories, and deletes unselected output directories. Its original source is preserved in `code/legacy/forecasting/wind_main.py`. The active release entry trains all original sources and retains their identities, without performing that selection or deletion. This changes output bookkeeping/selection, not model definitions or their loss functions. Archived retained-source results should not be presented as an unselected six-source benchmark.

## PV retraining and saved results

`retrain_pinn_only.py`, `retrain_pinn_inv3.py`, `retrain_pinn_multiseed.py`, `gen_sweep_inv3.py`, `gen_extreme_0miss.py` and `eval_metrics.py` are preserved under `code/legacy/forecasting/`. They document iterations that can overwrite results produced by the standard entry. In particular, the multi-seed script explicitly chooses a seed by test-set R². It is not part of the default release training pipeline.

The current Inverter_3 metrics JSON lacks the `config` section present for the other retained sources. The original scheduling loader already falls back to the maximum processed power multiplied by 1.05 for rated-power postprocessing; that behavior was kept. Checkpoint architecture dimensions are inferred from saved parameter shapes. The supplied weights, feature tables, and saved predictions pass the release loading checks, but the file history does not prove that one standard training command produced all current artifacts.

## What the paired scheduling experiment measures

The two branches share scene geometry, routes, task settings and initial-charge configuration. They differ in station generation **and** vehicle WORK-phase load. Period/charge-specific experimental power columns are used in the reference branch; the `normal` proxy is used in the forecast branch. Flight-profile timing remains derived from the processed flight records. This is a paired replay/simulation comparison, not an isolated forecast-error ablation and not proof of live hardware deployment.

PV and wind profiles are drawn from different years and scaled to station capacity. Full-record inference includes training-period observations. Neither those full-record predictions nor the scheduling slices should be labelled as an independent held-out forecasting test. The physics models also consume target-time weather variables; an operational future-time forecast needs an explicit source for those variables.

## Packaging edits

1. Input data, processed data, model weights, code, archived results and newly generated outputs were separated. Path modules remove dependence on the original folder layout and drive letters.
2. Full-record inference and actual-power reads are cached in memory during a run, so the same source/model is not loaded again for every case. The cached values and interpolation calculations are unchanged.
3. A CLI wrapper was added for inference, training, plotting and scheduling. It rejects nonempty output directories and protects distributed source/input/reference folders. The low-level scientific scripts retain their original overwrite behavior within their chosen working output directories.
4. The inactive vehicle-output-mode option now accepts only its implemented value, instead of silently accepting other strings. The paired measured/proxy logic is unchanged.
5. Flight preprocessing gained explicit input/output paths. Its alignment, idealized-motion construction, power estimation and repeated-cycle simulation are preserved.
6. Only the unused `latitude`, `longitude` and free-text `message` columns were removed from the released raw/idealized flight CSV copies. Existing numeric values in retained columns were not changed. The original files were left intact.
7. Old scripts and training logs relevant to lineage were retained separately. Paper drafts, virtual environments, editor configuration, caches, duplicate/old figures, standalone diagnostics and abandoned validation outputs were omitted.
8. The flight-cycle script now requests a writable NumPy copy before assigning the first timestep; pandas 3 returns a read-only view in the tested environment. This compatibility fix preserves the timestep values and simulation equations.

See `source_manifest.json` for each included original file, its relative source path, original hash, copied hash, and data-column removals. `packaging_changes.json` lists later changes to copied files. `SHA256SUMS.csv` covers the delivered package. Repacking does not rewrite the original `local` directory.

## Release metadata still owned by the authors

The inspected source contains no explicit code/data license. Dataset download URLs and redistribution permissions were not established. No license has been fabricated, and no public upload has been performed. The authors should set the intended licenses and dataset attribution before publishing. The citation title/authors/version should follow the manuscript version the authors choose to release.
