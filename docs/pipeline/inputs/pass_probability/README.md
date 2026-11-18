# Pass-probability inputs

The benchmark cohort contains 6,706 passes from ten matches: 5,742 completions and 964 failures. Stage 01 generates the numerical inputs from the public match files; the source archive supplies the code and protocols.

| File | Content |
|---|---|
| `rows.parquet` | Pass identifiers, release and next-control frames, observed outcomes. |
| `arrays.npz` | Observed target positions aligned with the rows; player states are reconstructed from tracking. |

The feature stage reads the generated rows and prepared tracking selected by `--tracking-root`. It reconstructs both candidate trajectories per pass and stores `fit_positions.npy`, `mean_velocity.npy` and `diagnostics.parquet` in each output `trajectories/<match_id>/` folder. The feature stage then computes separate path TTI and low-ball arrays for both candidates; `trajectory_errors.parquet` preserves aligned comparison errors. The fitting stage chooses one method separately on each fitting split using those errors. Calibration and test matches do not influence that choice.

The top-level execution plan supplies the generated `--data-root`, `--tracking-root` and `--output` paths. The [task guide](../../pass_probability/README.md) describes training, evaluation and production refitting.
