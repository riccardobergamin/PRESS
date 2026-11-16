# Pass probability

The physical model uses vmax=5.5 m/s, linear and turning acceleration scales of 7 m/s², preserved initial overspeed, trajectory-based MLE, exponential competing-risk integration and isotonic endpoint calibration.

## Reading order

1. `model.py`: reconstruct launch speed and angles from five positions at 0, 0.1, 0.2, 0.3 and 0.4 s; compare with mean-velocity projection; calculate trajectory and destination TTI; select hypothetical endpoint flights; integrate control probabilities and the Bernoulli likelihood. There are no priors.
2. `training.py`: optimize physical parameters, generate cross-fitted endpoint scores, calibrate, evaluate and refit production parameters.
3. `pipeline.py`: `generate_trajectories` reconstructs and selects trajectories; `prepare_features` loads them and computes TTI, control masks and endpoint inputs. Training reuses these arrays.

Release states come from `pipeline/frame_states.py` and `get_pv_players`. Reference `arrays.npz` retains initial states for checks; the pipeline reads its observed target `r_target`. Shared numerical routines are `PRESS.physics.times.time_to_intercept`, `PRESS.physics.flight_ball.ball_trajectory` and `generate_ball_trajectories`.

`select_flight` interpolates candidate flight times and heights, rejects unavailable candidates and selects the arrival closest to the receiver TTI. `prepare_pitch_control_inputs` constructs the arrays; `integrate_pitch_control_logs` integrates competing probabilities; `predict_pitch_control` returns raw attacking-team PPCF. Calibration follows this calculation.

## Estimation and calibration

Three complete BFGS optimizations start from (lambda, s)=(4.3, 0.43), (3, 0.3), (5, 0.5), without a fixed learning rate. MLE uses completion along the reconstructed trajectory. The same fitted parameters then enter endpoint PPCF based on the initial state and a hypothetical flight. This endpoint score is calibrated; calibration does not re-estimate lambda or s.

Four inner 6+2 rotations generate out-of-fold endpoint scores over the eight development matches for isotonic and Platt calibration. Physical parameters are refitted on eight matches before outer testing. Production physical parameters use all ten matches; production isotonic calibration uses raw outer-test scores, not in-sample scores from that refit.

Failure includes opponent control and the mass remaining uncontrolled at the horizon. The passer is excluded; other players, including goalkeepers, compete for control. Trajectory integration uses 251 samples from 0 to 10 s at 0.04 s intervals, each with a left-step update (effective summed horizon 10.04 s). Endpoint PPCF uses 1,500 updates of 0.01 s to 15 s.

## Trajectory selection and benchmark scope

The pipeline compares reconstruction methods **per pass**, using mean 3D error from 0.7 s to the next control, within 10 s. It saves only the lower-error trajectory. Ties or missing comparison frames favor a usable position fit; otherwise mean velocity is used.

Training of the published model selected one estimator by aggregated error on each fitting split. The released procedural pipeline uses the simpler per-pass choice above. This can change fitted parameters and calibration. Existing benchmark metrics refer to the split-level procedure; evaluating the released pipeline's outputs requires a new validation. The pipeline does not modify the distributed model.

## Input contract

The cohort contains 6,706 passes: 5,742 completed and 964 failed, with durations from 0.5 to 10 s inclusive. Pass verification occurs upstream.

```text
pipeline/inputs/pass_probability/
  rows.parquet
  arrays.npz
  boundaries.parquet
  matches/<match_id>/boundaries.parquet
  matches/<match_id>/trajectories.npz
```

`rows.parquet` contains identifiers, labels and release/next-control frames. In `arrays.npz`, `p_pos` and `p_vel` have shape (N,22,2); `x_bc` and `r_target` have shape (N,2). The roster is ten teammates, passer at index 10, eleven opponents. Positions are metres and velocities m/s; distant padding (at least 500 m) is masked. Boundary/path files are numerical references and do not replace trajectory estimation.

`pass_table` stores observations; `model_inputs` stores TTI, control-height and valid-player arrays plus endpoint inputs; `parameter_estimates` stores lambda, s and optimization diagnostics. `fit_on_matches` selects fitting rows and estimates parameters.

## Commands and outputs

```bash
export JAX_ENABLE_X64=1
python pipeline/pass_probability/pipeline.py --stage features \
  --data-root pipeline/inputs/pass_probability \
  --output pipeline/outputs/pass_probability
python pipeline/pass_probability/pipeline.py --stage train --fold 0
# After all five folds:
python pipeline/pass_probability/pipeline.py --stage scores
python pipeline/pass_probability/pipeline.py --stage refit
```

Default tracking is `pipeline/inputs/shared/tracking/`; override it with `--tracking-root`. Omit `--fold` to fit all folds. `--stage all` executes the workflow. Outputs are separated into `trajectories/<match_id>/`, `features/`, `training/fold_<f>/`, `scores/` and `production/`. Completed directories are protected; the exported `PassRiskModel` JSON is not installed automatically.

## Verification scope

Checks cover observed-trajectory reconstruction, TTI, flight times, probabilities, likelihood derivatives, three-start MLE, calibrator application and reconstruction, exported-model compatibility and metric replay. They also exercise feature-cache loading and overwrite protection. Replaying 6,706 stored predictions gives CE 0.13112077, gain 68.1917% and ECE 0.18331%. These checks do not constitute a new complete five-fold training.
