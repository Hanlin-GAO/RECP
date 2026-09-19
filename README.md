# RECP: renewable forecasting and mission scheduling

This local release-preparation package contains forecasting code, a rover/drone scheduling simulator, separate input data, archived weights, and reference results. See [README_zh.md](README_zh.md) for Chinese instructions.

The reviewed **v2** training entry uses a new evaluation protocol. It does not reproduce or replace the paper's historical training procedure. The supplied `models/`, processed data, and `results_reference/` still belong to the historical experiments. New results must be reported separately. No hardware-control driver is included.

## Layout

| Directory | Contents |
|---|---|
| `code/forecasting/` | Models, reviewed training, and archived inference support |
| `code/scheduling/` | Mission and charging simulation; fixed-load and historical replay modes |
| `code/preprocessing/` | Flight alignment, idealization, and cycle simulation |
| `code/legacy/` | Inspection-only historical scripts; execution is blocked |
| `data/forecasting/raw/` | Original PV/weather records and all six wind workbooks |
| `data/forecasting/processed/` | Historical feature tables matched to archived weights |
| `data/experiments/` | Vehicle-power records, redacted flight logs, derived flight profiles |
| `data/scenes/` | Local geometry and manual routes |
| `models/`, `results_reference/` | Historical weights and numerical outputs, unchanged in v2 |
| `provenance/` | Source mapping, changes, checksums, review, and verification |
| `scripts/` | Run commands and regression checks |
| `runs/` | New local outputs, not distributed as paper results |

Code and experimental data are physically separate. Keep sibling directories together. Paths resolve from the package location. All Python comments and docstrings are English. Some filenames and runtime field names retain their original language for compatibility.

## Environment and checks

Use Python 3.11 or 3.12. Tested and original environments are recorded separately under `provenance/`.

```sh
python -m venv .venv
```

Activate `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on Linux/macOS, then:

```sh
python -m pip install -r requirements.txt
python scripts/verify_manifest.py
python scripts/check_protocol.py
python scripts/check_release.py
```

`check_protocol.py` tests split isolation, causal weather bins, train-only statistics, missing-input parity, model interfaces, fixed vehicle loads, English comments, and blocked legacy execution. `check_release.py` checks archived models and a short simulation; passing it does not establish independent forecast accuracy. Only load trusted, verified checkpoints. CPU is the default and was tested. GPU runs may need a deterministic CUDA environment and are not certified here.

## Reviewed training: audit first

```sh
python scripts/reproduce.py train --domain pv --audit-only --output-dir runs/audit_pv
python scripts/reproduce.py train --domain wind --audit-only --output-dir runs/audit_wind
```

The default includes every raw source: three PV inverters and six wind sources. Here, `Turbine_5` means `Wind 5.xlsx`, not a renamed historical display ID. Optional `--station` selects an explicit subset and records that scope; do not present it as an all-source benchmark.

The protocol in `code/forecasting/reviewed_protocol.py` uses:

- Fixed source years and regular time grids, without daytime compression or performance-based source ranking. Raw files remain unchanged.
- Causal PV weather bins, duplicate-time averaging, retained missing periods, and counts for excluded/invalid rows. Negative generation is outside this protocol's nonnegative-generation target definition; it is retained in the raw files.
- January-August training, September-October validation, and November-December testing. Windows are built within each interval, without shared raw-observation intervals across boundaries.
- Training-only imputation, scaling, empirical power bounds, and extreme-weather thresholds.
- The same available past window for every model. Physics inputs use its last row after the same masking/imputation, not target-time weather.
- All requested models, sources, and seeds. Validation MSE selects checkpoints; test scores do not select seeds, sources, or epochs. Failed runs are recorded and prevent a successful-run-only aggregate.
- Full test results as primary outputs. Extreme subsets use training thresholds without a test-driven fallback. Both raw and uniformly clipped predictions are saved. Fixed synthetic window-level sensor masks are shared across models and seeds; they do not represent every real outage pattern.

Train with fixed defaults:

```sh
python scripts/reproduce.py train --domain pv --output-dir runs/reviewed_pv
python scripts/reproduce.py train --domain wind --output-dir runs/reviewed_wind
```

Defaults use six model types, three seeds, the same maximum epoch budget, and the same validation patience. Architecture parameter counts and training costs are not necessarily equal. Before fitting, `protocol.json` records settings, source scope, code hashes, and environment details. Each source gets preprocessing counts, frozen statistics, split-membership tables, per-seed checkpoints, histories, predictions, and metrics. The summary includes all requested seeds, not just the best one.

For a development check only:

```sh
python scripts/reproduce.py train --domain pv --station Inverter_1 --epochs 1 --seeds 42 --models dnn pinn --output-dir runs/dev_pv
```

Changed budgets, models, or seeds are marked as development runs. A one-epoch result is not a paper result. Existing nonempty output directories are rejected.

Export test-interval predictions from a newly trained checkpoint:

```sh
python scripts/predict_reviewed.py --checkpoint runs/reviewed_pv/Inverter_1/seed_42/pinn/checkpoint.pt --output runs/reviewed_pv_seed42.csv
```

This checks recorded code/input hashes and uses frozen statistics without refitting. It also predicts grid timestamps with missing target labels; training evaluation scores only valid labels. New checkpoints use a different input schema from historical weights and are not automatically used by the scheduler.

## Scheduling replay

The default holds input vehicle work-power traces fixed and changes station renewable supply:

```sh
python scripts/reproduce.py schedule --groups G3 --output-dir runs/fixed_G3
```

Each group writes actual-generation and predicted-generation cases, state/power histories, events, and comparison metadata. Station scaling uses observations before the first replay day, with the same scale and peak limit in both cases. Vehicle load *input profiles* are identical; realized powers may differ when schedule states differ.

This remains an **offline replay using archived predictors**, not an independent end-to-end forecast test. Archived predictors used target-time weather and training periods overlapping or later than replay dates. PV and wind sources also come from different years and are scaled to the station model. One example does not establish performance across all scenarios. To run all supplied groups, use `--groups G1 G2 G3 G4 G5 G6 G7 G8 G9 G10 G11 G12`.

For explicit historical behavior:

```sh
python scripts/reproduce.py schedule --groups G1 --comparison-protocol historical_paired --output-dir runs/historical_G1
```

That mode changes both renewable supply and measured-versus-proxy vehicle loads, and uses full-record station scaling. It is not a forecast-only ablation. The mode is recorded in outputs. Plot new histories with `code/scheduling/build_validation_group_panels.py` using `--results-dir` and `--output-dir`.

## Historical forecasting and plots

```sh
python scripts/reproduce.py replay-forecast --domain pv --station Inverter_1 --model pinn
python scripts/reproduce.py replay-forecast --domain wind --station Turbine_2 --model pinn
python scripts/reproduce.py plot --domain pv --output-dir runs/plots/pv
python scripts/reproduce.py plot --domain wind --output-dir runs/plots/wind
```

`replay-forecast` infers the full processed record, including training periods, and labels it historical/non-held-out in a sidecar. Its wind display IDs follow `provenance/wind_source_mapping.json`, not raw-source IDs. Plot commands use archived tables, not reviewed training outputs. Historical trainers and selection scripts remain for inspection, but their direct script entry points are disabled.

## Derived flight data

```sh
python code/preprocessing/idealize_flight_logs.py
python code/preprocessing/simulate_three_drone_same_soc.py
```

These write under `runs/flight/processed/`; the simulator still uses distributed processed profiles. Idealized and simulated profiles are **derived data**, not untouched measurements. See [data/README.md](data/README.md).

## Interpretation and publication status

Read [provenance/REVIEW_V2.md](provenance/REVIEW_V2.md). Prior [version notes](provenance/VERSION_NOTES.md) and [verification notes](provenance/VERIFICATION.md) are retained as historical records.

Code changes cannot make an inspected test set unseen again. Full retraining and independent data are still needed before making new generalization claims or updating paper conclusions. Dataset redistribution rights, code licensing, source time conventions, and some experimental-data generation steps need author confirmation. No license is inferred, no risk-free certification is made, and nothing has been published online.
