# Final pass-probability inputs

Accepted inputs organized by match.
The review pipeline uses this folder as its default `--data-root`.

| File | Content |
|---|---|
| `rows.parquet` | Pass table: identifiers, release/reception frames and completion labels. |
| `arrays.npz` | Release-state player positions/velocities, carrier position and target position, aligned with the pass table. |
| `boundaries.parquet` | Reference forecast errors/diagnostics for numerical verification. |
| `matches/<match_id>/boundaries.parquet` | Reference diagnostics split by match. |
| `matches/<match_id>/trajectories.npz` | Reference paths for numerical verification, shape `(N_match,251,3)`. |

Cohort: **6,706 passes, 5,742 completed and 964 failed**, from ten matches.
Input copies are checked against the SHA-256 hashes in `../manifest.json`.

From the project root, with PRESS installed, build the model-input arrays once:

```bash
python pipeline/pass_probability/pipeline.py --stage features
```

This first runs `generate_trajectories`: estimate all passes' launch conditions,
compare position-fit and mean-velocity paths against observed positions and
save ONLY the chosen path and comparison diagnostics. Results are saved
in `pipeline/outputs/pass_probability/trajectories/<match_id>/`.

Selection takes place once per pass, using mean 3D forecast error before the
next control. With no valid comparison, prefer a usable position fit; exact
ties also prefer that fit. The reported benchmark uses split-level selection on training matches.
The per-pass criterion requires a new training and validation run before
its performance can be compared with the reported results.

Then `prepare_features` loads those paths and the release states, computes TTI,
hypothetical flight times and control masks, and saves the model-input arrays
in `pipeline/outputs/pass_probability/features/`. Training reuses these arrays.

Tracking is always read from `pipeline/inputs/shared/tracking/` by default.
`--tracking-root` overrides that location. The reference paths listed above
are not used in place of trajectory estimation by the pipeline.

Training, scores and production outputs stay separate under
`pipeline/outputs/pass_probability/`. See
[the pipeline guide](../../pass_probability/README.md) for the remaining stages.
