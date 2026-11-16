# Research-pipeline inputs

Inputs are organized by task and remain separate from generated outputs. Protocol JSON files specify match partitions and model settings. Where supplied, manifests record SHA-256 hashes. Numerical observation tables and fold artifacts are not included in the source archive; obtain source data and prepare the required inputs before executing a task.

## Shared tracking and partitions

`folds.json` lists the two test matches in each of five outer folds. `shared/tracking/matches/<match_id>/<match_id>_tracking.parquet` contains complete preprocessed tracking. `frame_states.py` uses the public state getter to reconstruct carry starts and pass releases. Reference state arrays support numerical checks; pass probability additionally reads the observed target `r_target`.

## Task inputs

| Directory | Required observations and supporting inputs |
|---|---|
| `ball_drive/` | `rows.parquet` and aligned `arrays.npz`: 9,114 attempts, 456 losses. `reference_parameters/physical_splits.json` supplies split-specific dispersion values if physical fits are not regenerated first. |
| `pass_probability/` | `rows.parquet` and `arrays.npz`: 6,706 passes, 5,742 completed and 964 failed. Reference boundary diagnostics and trajectories are split by match. See [details](pass_probability/README.md). |
| `pass_selection/` | `rows.parquet`: 6,810 decisions, including 1,009 failed passes; `protocol.json`; `matches/<match>/players.parquet` linking event identities to tracking aliases. |
| `expected_threat/` | `events/{carries,passes,shots,losses}.parquet`, continuous coordinates and `protocol.json`. These are the xT cohort, not canonical carry observations. |
| `binary_classification/` | Action observations, residual labels, protocols and upstream artifacts; see [details](binary_classification/README.md). |
| `composed_risk/` | `labels.parquet`: 15,085 verified states, with `band_eligible` identifying the 14,919 terminal-covered states. Possession boundaries, carrier/team and next control are observed metadata. |
| `reports/` | `cases.json` selects the observed report possessions; reference tables support numerical comparisons. |

Physical, xT and choice-model dependencies are explicit under `upstream/fold_0/` through `fold_4/`; deployment dependencies are under `upstream/production/`. Fold artifacts must correspond to the declared fitting/calibration split. Newly generated fits are not substituted automatically.

For pass selection, physical and xT artifacts use the matching six fitting matches. Reference screening results are under `reference/screening.csv`. The `smoke/` fixture contains one real pass per match with complete grids and memberships; it is a verification input, not another benchmark cohort.

For xT, `protocol.json` distinguishes grid-comparison 8+2 splits, six-match downstream maps and the ten-match production map. Reference maps and validation summaries are kept under `reference/`.

## Regeneration and verification

The [preprocessing stage](../preprocessing/README.md) reads original public files and writes regenerated task tables. `preprocessing/matches/<match>/` contains reference possession, action, identity and event tables for explicit comparison. Check keys and labels before replacing inputs.

Pass-probability feature preparation first runs `generate_trajectories`, then `prepare_features`. It selects trajectories per pass; the reported benchmark used fitting-split selection. See [the task guide](../pass_probability/README.md) before interpreting a new run as a reproduction.

`composed_risk/reference/components.parquet` supports composition from stored component predictions with `--replay-accepted`. Ordinary execution uses newly prepared features and fitted fold models. `reference/fold_stack/` supports inference checks without refitting. Report outputs never overwrite input tables.

Generated data are saved under `pipeline/outputs/<task>/`, with separate features, training, scores and production directories. xT additionally separates validation and feature maps; pass selection separates flat and CNL outputs; binary classification separates action and residual tasks. Choose a fresh output root for each execution.
