# Action selection and residual risk

The tasks share three scripts: `features.py` defines and calculates covariates, `training.py` fits and evaluates native binary XGBoost models, and `pipeline.py` executes the stages.

## Workflow

1. `prepare_action_features` reads observed states, evaluates raw pitch control, destination choice, calibrated pass probability, physical ball-drive risk and preliminary effective-hazard BCOS, then builds the scalar features.
2. `train_validate_test` fits encoding and trees on six matches, selects configuration and prefix on two fixed validation matches, and evaluates on two test matches.
3. `compute_pooled_metrics` combines the five test sets.
4. `train_production_model` fits the deployment model.
5. `prepare_residual_features` reuses action features, adds cross-fitted P(carry), and joins verified terminal outcomes and observed elapsed time. Training, evaluation and refitting then follow the same binary procedure with the residual target.

`train_model` calls `xgb.train` with `binary:logistic`. For score f and p=sigmoid(f), derivatives are g=p−y and h=p(1−p). There are no custom derivatives, class weights or resampling. Initialization follows native XGBoost behavior.

| Property | Action selection | Residual risk |
|---|---|---|
| State | Canonical pass/carry decision | Start of a successful canonical one-second carry |
| Positive label | Pass chosen | Terminal individual-possession loss |
| Cohort | 16,918 states: 7,770 passes and 9,148 carries | 7,820 states: 1,423 terminal losses |
| Inputs | 28 numerical features, start type and previous action | Same feature family; elapsed time replaces carry count, with additional P(carry) |
| Deployment calibration | Native probabilities | Native probabilities |

Outcome verification does not filter action selection. Residual risk requires verified outcomes and retains fouls won and clearances according to the segmentation conventions.

## Features and dependencies

`build_features` returns a dictionary of named scalars, one per output column. Destination-choice and completion vectors contain one entry per candidate. Success/failure weighted summaries retain their branch probability; they are not renormalized within the branch.

Pass selection receives raw pitch control. Downstream success/failure summaries receive isotonic-calibrated completion probability. The physical carry model returns loss probability; these classifiers receive its complement, carry success. P(carry) is the distinct probability of choosing to carry.

`pipeline/frame_states.state_at_frame` calls `PRESS.utils.state.get_pv_players` with the observed carrier. The roster layout is ten teammates, carrier, eleven opponents. Preliminary BCOS uses the fold's ball-drive dispersion, carrier speed cap 6.6 m/s, defender cap 7 m/s, acceleration scales 7 m/s², radius 40 m, area grid 1 m, path step 0.03125 m, time step 0.04 s and horizon 15 s.

## Cross-fitted P(carry)

- **Fold training:** three action fits each use four of the six fitting matches and predict the other two. They use the main fold's selected configuration and tree count; categorical vocabularies are fitted on their four matches.
- **Validation/test:** the main action model fitted on six matches predicts the held-out rows. There is no eight-match refit.
- **Production training:** five action models each fit eight matches and predict two, using the production configuration and tree count. At inference, PRESS uses the action model fitted on all ten matches.

Only the action classifier is cross-fitted internally. It is a flexible model trained on pass/carry choices at many of the same carry checkpoints used for residual risk, so this step limits in-sample recognition of the observed choice. The physical ball-drive model is more constrained; pass probability and pass selection are fitted on pass observations, with less direct checkpoint overlap. Their inputs, together with xT and BCOS, remain those of the outer fold and can still be in-sample on fitting rows. The three rotations do not select hyperparameters or tree counts.

## Search, metrics and production

The grid combines depth (3, 4, 6, 8), child weight (1, 5, 20), L2 (0.1, 1, 3, 10) and eta (0.03, 0.05, 0.1): 144 configurations per fold. Settings are CPU histogram boosting, max_bin=256, row/column subsampling 0.8, at most 2,000 trees, patience 100 and seed 20261026+fold.

Native validation CE selects the configuration and prefix. Encoding learned on six matches is applied unchanged to validation/test. Pooled metrics concatenate all five test sets before computing CE, gain, Brier, AUC and ten-bin ECE.

`select_production_configuration` minimizes mean validation CE across the five grids. Its tree count is the median of the five prefixes for that same configuration, not the median of the five fold winners. The distributed models have 277 action trees and 118 residual trees. Production refitting uses all ten matches without early stopping or scaling.

`save_feature_importance` normalizes total split gain across trees and saves both encoded-column and regrouped categorical summaries.

## Commands and outputs

Inputs are in `pipeline/inputs/binary_classification/`, with shared tracking in `pipeline/inputs/shared/tracking/`. Outputs are separated by task under `pipeline/outputs/binary_classification/`.

```bash
python pipeline/binary_classification/pipeline.py --task action_selection --stage features --fold 0
python pipeline/binary_classification/pipeline.py --task action_selection --stage train --fold 0 --threads 4
# After action features and all five action fits:
python pipeline/binary_classification/pipeline.py --task residual_risk --stage features --fold 0 --threads 4
python pipeline/binary_classification/pipeline.py --task residual_risk --stage train --fold 0 --threads 4
```

`--stage scores` pools test results; `--stage features --production-features` prepares ten-match inputs; `--stage refit` exports the model. Complete action selection before preparing residual risk, including production. `--stage all --task both` follows this order. Completed outputs are not overwritten.

## Checks

`checks/verify_pipeline.py` covers formulas on real states, encoding, short native fits, 4+2 cross-fitting and reference metrics. `checks/verify_production.py` checks refit/export compatibility with public readers. Small verification refits are not expected to reproduce the distributed predictions exactly and do not replace benchmark weights or a full search.
