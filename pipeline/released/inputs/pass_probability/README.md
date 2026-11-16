# Pass-probability inputs

The accepted cohort contains 6,706 passes from ten matches: 5,742 completions and 964 failures. These numerical inputs are stored locally and are not included in the source archive.

| File | Content |
|---|---|
| `rows.parquet` | Pass identifiers, release and next-control frames, observed outcomes. |
| `arrays.npz` | Aligned release states and observed target positions; the pipeline reconstructs player states from tracking. |
| `boundaries.parquet`, `matches/<match_id>/boundaries.parquet` | Reference reconstruction diagnostics for checks. |
| `matches/<match_id>/trajectories.npz` | Reference trajectories for numerical checks. |

The released feature stage reads the accepted rows and original tracking under `pipeline/inputs/shared/tracking/`. It reconstructs both candidate trajectories per pass and stores `fit_positions.npy`, `mean_velocity.npy` and `diagnostics.parquet` in each output `trajectories/<match_id>/` folder. The feature stage then computes separate path TTI and low-ball arrays for both candidates; `trajectory_errors.parquet` preserves aligned comparison errors. The fitting stage chooses one method separately on each fitting split using those errors. Calibration and test matches do not influence that choice.

Run `python pipeline/pass_probability/pipeline.py --stage features` from the project root, or pass explicit `--data-root`, `--tracking-root` and `--output` paths. The [task guide](../../pass_probability/README.md) describes training, evaluation and production refitting. Reference paths are used for checks and do not replace reconstruction from tracking.
