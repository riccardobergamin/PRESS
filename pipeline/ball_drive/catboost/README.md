# CatBoost ball-drive comparison

Read `features.py` → `training.py` → `pipeline.py`. `run_pipeline` coordinates feature preparation, training and pooled scoring; `train_model` calls native `CatBoostClassifier.fit` with `Logloss`.

## Features and cohort

The cohort contains 9,114 attempts and 456 losses. A loss before the end of the one-second window still has target 1. CatBoost is an exploratory benchmark comparison; the physical model is deployed.

The 50 features comprise carrier speed; nearest-boundary distances at 0, 0.5 and 1 s; the three smallest TTIs at each of those times; eleven encounter probabilities at 0.1 s intervals; initial/projected carrier and goal geometry; relative distance, speed, closing, transverse motion and position for the three initially nearest defenders; two local counts and nearest-pair geometry.

States are reconstructed through the shared tracking getter. TTI uses speed cap 7 m/s and linear/turning acceleration scales 7 m/s², preserving initial overspeed. Encounter curves use the physical ball-drive dispersion fitted on the corresponding six inner-fitting or eight development matches. Shared geometry is calculated once; training loads the prepared features.

## Training and calibration

The 36 configurations combine depth (4, 6, 8), learning rate (0.02, 0.05, 0.1) and L2 (1, 3, 10, 30). Trees are symmetric, with Plain boosting and Bayesian bootstrap. Within each outer fold, four 6+2 rotations evaluate 3,000-tree fits at 25-tree checkpoints. Pooled validation CE selects configuration and length.

- `natural`: all fitting observations.
- `loss10`: all losses and nine successes per loss, without replacement.
- Validation/test: unchanged natural prevalence.

The loss10 selection criterion applies analytical prior correction. With the selected configuration, seeds 19, 41 and 73 generate predictions covering the eight development matches out of fold, then three new eight-match fits predict the outer test pair. Probabilities are averaged across seeds. Platt and isotonic calibration use inner out-of-fold predictions and natural labels. All calibration variants are reported separately; test labels do not fit the calibrators. There is no ten-match CatBoost deployment refit.

## Execution

```bash
python pipeline/ball_drive/catboost/pipeline.py --stage features \
  --physical-splits pipeline/outputs/ball_drive/physical/physical_splits.json
python pipeline/ball_drive/catboost/pipeline.py --stage train --fold 0 --sampling natural --threads 4
# After both sampling arms and all five folds:
python pipeline/ball_drive/catboost/pipeline.py --stage scores
```

For a complete new estimation, use the `physical_splits.json` produced by the physical pipeline. Completed output directories are protected against overwriting. The benchmark uses CatBoost 1.2.10.

`pipeline/checks/catboost_steps.py` checks TTI, features, sampling and metric replay. This does not replace a full search or verify native fitting when CatBoost is unavailable.
