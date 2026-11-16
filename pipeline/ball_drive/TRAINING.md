# Ball-drive estimation and validation

Both procedures reconstruct the observed state at `frame_start` through `pipeline/frame_states.py` and `PRESS.utils.state.get_pv_players`. Reference arrays support numerical checks. The common cohort contains 9,114 carry attempts, including 456 losses, under the segmentation and outcome rules in the [benchmark](../../benchmarks/01_ball_drive_risk.qmd).

| Stage | Physical model | CatBoost comparison |
|---|---|---|
| Definitions | `physical/model.py`: covariates, spline bases, survival, likelihood and priors | `catboost/features.py`: 50 features and encounter curves |
| Estimation | `physical/training.py`: MAP, Laplace, regularization selection, metrics and refit | `catboost/training.py`: tuning, sampling, calibration and metrics |
| Execution | `physical/pipeline.py`: fold fits, aggregation and production refit | `catboost/pipeline.py`: features and natural/loss10 evaluation |

## Physical model

`build_spline_basis` estimates one covariate's basis from fitting observations. `evaluate_spline_basis` applies that basis to new values. `prepare_model_data` assembles covariates and spline matrices for fitting, validation or testing; `build_prior_precision` separately builds Gaussian precisions from the selected regularization strengths and fitted curvature penalties. Formulas are documented beside these operations.

`fit_split` prepares fitting and evaluation arrays, builds the priors, fits the model and predicts with fixed coefficients. `fit_map` evaluates 33 starting points, runs the selected L-BFGS-B optimizations, retains the converged solution with the lowest objective and applies Laplace approximation. The Bernoulli likelihood and priors include the Gamma prior on arrival dispersion and the log-coordinate Jacobian.

The physical calculation uses carrier speed, signed nearest-boundary distance and defender approach angle; Spearman-style TTI includes turning and preserves initial overspeed. The one-second horizon uses 0.04 s steps and exponential survival updates. Spline bases are fitted independently on each fitting subset, with five knots for speed/boundary and four for angle.

Each outer fold contains four 6+2 rotations within eight development matches. Sixteen configurations are compared by pooled validation CE. `select_regularization` computes paired match differences and a match-jackknife standard error, then applies the prespecified simplicity order among eligible candidates. The selected model is refitted on eight matches and evaluated on two. Production refitting uses all ten matches and does not contribute to held-out metrics.

`physical_splits.json` records the arrival dispersions from the six-match inner fits and eight-match outer fits for CatBoost feature construction.

## CatBoost comparison

The [CatBoost guide](catboost/README.md) describes the natural-prevalence and loss10 arms. The grid is depth 4/6/8, learning rate 0.02/0.05/0.1 and L2 1/3/10/30, with symmetric trees and Plain boosting. Four inner rotations select the configuration and a prefix scored every 25 trees up to 3,000. The loss10 criterion uses analytically prior-corrected probabilities.

Undersampling retains all losses and up to nine successes per loss, without replacement, only within fitting subsets. Three seeds (19, 41, 73) generate inner out-of-fold predictions for Platt/isotonic calibration and eight-match fits for outer testing. Native, prior-corrected, Platt and isotonic results are reported separately. Each encounter curve uses the physical dispersion fitted on the same six or eight matches. CatBoost uses native `Logloss`, without custom derivatives or explicit Bayesian coefficient priors. The benchmark version is 1.2.10.

## Commands

From the repository root, with PRESS installed:

```bash
export JAX_ENABLE_X64=1
python pipeline/ball_drive/physical/pipeline.py \
  --cache pipeline/inputs/ball_drive \
  --output pipeline/outputs/ball_drive/physical
```

Use `--fold 0` for a single fold, `--collect-only` after all folds, and `--refit` for production estimation. Outputs are not automatically installed in PRESS.

```bash
python pipeline/ball_drive/catboost/pipeline.py --stage features \
  --physical-splits pipeline/outputs/ball_drive/physical/physical_splits.json
python pipeline/ball_drive/catboost/pipeline.py --stage train --threads 4
python pipeline/ball_drive/catboost/pipeline.py --stage scores
```

An explicit reference input is `pipeline/inputs/ball_drive/reference_parameters/physical_splits.json`. Use `--stage train --fold 0 --sampling natural` to select one CatBoost fit.

## Verification scope

`check_training_versions.py` compares formulas, gradients, regularization selection, features and reference metrics, including a short physical MAP fit and exported-model loading. These checks are distinct from the complete nested search. Native CatBoost fitting requires its optional dependency; replaying features and metrics alone does not test tree training.
