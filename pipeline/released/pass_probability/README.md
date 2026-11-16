# Pass probability: released training pipeline

This is the pass-probability source distributed with PRESS. Its trajectory-estimator selection follows the published [benchmark](../../../benchmarks/02_pass_probability.qmd): one method is selected on the fitting matches for each six-match inner fit, eight-match outer refit and ten-match production refit.

Read `model.py` for flight reconstruction, control integration and the likelihood; `training.py` for split-specific method selection, MLE, calibration and evaluation; and `pipeline.py` for feature construction and the ordered stages.

## Trajectory preparation and split selection

`generate_trajectories` reconstructs both the five-position fit and the mean-velocity projection for each accepted pass. It stores both paths and their forecast errors. `prepare_features` computes player arrival times and ball-control masks along both paths. Each `fit_on_matches` call then averages the forecast errors within every fitting match, averages those match means equally, and selects one method for the entire fitting split. The other two calibration or test matches do not enter that choice. An unusable position fit uses the reconstruction function's mean-velocity fallback for that pass. Ties favor the position fit.

This selection applies independently inside the four 6+2 rotations, the eight-match outer physical fit and the ten-match production fit. Endpoint pitch-control inputs are independent of the observed-flight method. The two physical parameters are estimated by three complete BFGS starts on the selected paths, and the lowest admissible training negative log-likelihood is retained. Cross-fitted endpoint probabilities from the four inner rotations train isotonic and Platt calibration; the selected isotonic map and eight-match physical fit predict the outer test matches. The production physical model is fitted on ten matches, and its isotonic map uses raw outer-held-out scores.

The cohort and outcome rules, movement settings, calibration results and reported metrics are specified in the [benchmark](../../../benchmarks/02_pass_probability.qmd). The released code does not contain raw match data or fitted fold objects. It requires accepted pass rows, tracking and the task inputs described in `inputs/pass_probability/README.md`.

## Commands

From the repository root, with PRESS installed and input paths populated:

```bash
export JAX_ENABLE_X64=1
python pipeline/pass_probability/pipeline.py --stage features \
  --data-root pipeline/inputs/pass_probability \
  --output pipeline/outputs/pass_probability
python pipeline/pass_probability/pipeline.py --stage train
python pipeline/pass_probability/pipeline.py --stage scores
python pipeline/pass_probability/pipeline.py --stage refit
```

The website's download places this released source at `pipeline/pass_probability/`. The linked website source is under `pipeline/released/pass_probability/` to keep it distinct from local development code. The output directory must be new; generated arrays, fit objects, predictions and the production JSON are saved there.
