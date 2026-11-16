# PRESS research pipeline

The procedural pipeline prepares observations, estimates model parameters, evaluates held-out predictions and generates descriptive reports. The inference package is in `PRESS/`; running the research pipeline does not replace its pretrained models.

## Reading and execution order

`00_pipeline.py` prints commands in dependency order. Add `--execute` to run them, or select a stage with `--step 05`, for example.

| Stage | Directory | Reading order |
|---|---|---|
| 01 | `preprocessing/` | `features.py` → `pipeline.py`; `tracking.py` for kinematics and segmentation, `passes.py` for pass links |
| 02 | `expected_threat/` | `model.py` → `training.py` → `pipeline.py` |
| 03 | `ball_drive/physical/` | `model.py` → `training.py` → `pipeline.py` |
| 04 | `ball_drive/catboost/` | `features.py` → `training.py` → `pipeline.py` |
| 05 | `pass_probability/` | `model.py` → `training.py` → `pipeline.py` |
| 06 | `pass_selection/` | `features.py` → `training.py` → `pipeline.py`; shared features for flat and CNL models |
| 07 | `binary_classification/` | `features.py` → `training.py` → `pipeline.py`; action selection before residual risk |
| 08 | `composed_risk/` | `features.py` → `scores.py` → `pipeline.py` |
| 09 | `reports/` | `features.py` → `scores.py` → `cases.py` → `pipeline.py` |
| 10 | `benchmarks.py` | Physical-parameter and performance summaries |

`frame_states.py` reads observed frames through `PRESS.utils.state.get_pv_players`. Carries use `frame_start`; passes use the release at `frame_end`. `io.py` provides table and summary serialization.

## Inputs, dependencies and outputs

[Input specifications](inputs/README.md) describe the observation tables and protocols in `inputs/<task>/`. Upstream physical, xT and pass-selection artifacts are explicit inputs under `upstream/fold_<f>/` or `upstream/production/`. The execution plan does not automatically install new fits as downstream inputs: the caller must supply compatible artifacts from the required split.

Preprocessing reconstructs cohorts from public SkillCorner files under `outputs/preprocessing/task_inputs/<task>/`. Compare row keys and labels with the reference cohorts before substituting these inputs. Task-specific outcome filters are defined in each task's `filters.py`.

Generated objects are separated into feature, training, score and production directories under `outputs/<task>/`. The two binary tasks share code but have distinct outputs. Select a new `--output-root` for a new execution; completed outputs are protected against overwriting. Exported production fits are not automatically installed in PRESS.

The published benchmarks and frozen classifiers use **a preliminary BCOS formulation with effective hazards**. Standalone BCOS in the package defaults to the final transport-with-killing formulation. Downstream retraining with that formulation remains planned.

The downloadable pass-probability pipeline selects one trajectory estimator on each fitting split, as in the published benchmark. Its [task guide](released/pass_probability/README.md) describes the selection and the two candidate trajectory inputs.

## Task guides

- [Ball-drive risk](ball_drive/TRAINING.md): physical estimation and CatBoost comparison.
- [Pass probability](released/pass_probability/README.md): trajectories, MLE and calibration.
- [Pass selection](pass_selection/README.md): features, flat/CNL learning, screening and temperature scaling.
- [Expected Threat](expected_threat/README.md): transitions, smoothing, grid comparison and fitted maps.
- [Action selection and residual risk](binary_classification/README.md): shared features, cross-fitted carry probability and binary learning.
- [Composed risk](composed_risk/README.md): held-out h/xLoss, metrics and band bootstrap.
- [Reports](reports/README.md): possession patterns, player/team summaries and observed cases.

## Evaluation and release

Overall h/xLoss metrics use 15,085 verified held-out states. Bands and reports use 14,919 states from 7,265 possessions with a terminal canonical action; micropossessions are retained. Band intervals use 2,000 match-bootstrap replicates, seed 20261030 and fixed predictions and thresholds. Player/team E−O intervals resample possessions; player intervals require at least 20 exposed possessions. Pattern shares use Wilson intervals.

[Verification](checks/README.md) exercises each stage with observed data and short fits. These checks do not replace full hyperparameter searches or establish their convergence.

`release.py` exports an explicit list of source files and aggregate statistics. Full match inputs, posterior objects, fold boosters and bootstrap replicates are excluded. Pretrained inference models are distributed with the package.
