# Data setup

Raw tracking/event files and match-level feature tables are not stored here. Obtain the source data from [SkillCorner opendata](https://github.com/SkillCorner/opendata) and follow its attribution terms [@skillcorner-opendata]. Its MIT license file covers software; it does not specify separate data redistribution terms. This repository includes learned XGBoost and ball-drive parameter files and aggregate benchmark summaries. Pass selection also includes per-choice OOF loss tables for independent score verification; these contain identifiers and losses, without raw tracking rows or candidate feature matrices.

Set `SKILLCORNER_OPENDATA` and `SKILLCORNER_ANALYSIS` to your local repository roots for the match guide. The original scripts do not consume `PRESS_DATA_DIR`; update their machine-specific paths before using them. The original preprocessing and possession-segmentation scripts live in the sibling research repository at `skillcorner-analysis/scripts/analysis/preprocess_matches.py` and `skillcorner-analysis/scripts/analysis/ball_drive_segmentation.py`; they currently use machine-specific paths and are documented as source references rather than moved or modified here.

The committed vignette inputs are synthetic and clearly labeled. They are sufficient to rebuild all figures and notebook outputs. The final production XGBoost model parameters used by the package are included separately under `PRESS/utils/data/`: the transition booster, action-selection booster, and action-selection feature-column schema. Their source is documented in the repository README; they contain learned parameters rather than tracking rows.

See [the match guide](../usage.qmd) for `PRESS.from_skillcorner`, possession/frame getters and a complete checkpoint calculation, and [data setup](../data.qmd) for the expected processed-file layout.

The final ball-drive artifact is `PRESS/utils/data/ball_drive_model.json`, loaded by `BallDriveModel.pretrained()`. Its source and verification are in [refit provenance](../docs/ball-drive-refit-provenance.json). `data/benchmarks/ball_drive/` contains pooled OOF metric and calibration summaries plus hashes of the research tables; these are separate from the full-data refit diagnostics.

The [arrival-dispersion summary](benchmarks/ball_drive/arrival_dispersion_laplace.json) contains the ball-drive marginal Laplace intervals for `s` and the separate pass-risk MLE point estimate. Regenerate it with `python scripts/build_arrival_dispersion_summary.py` from the repository root. It reads the frozen fitted artifacts without retraining; these conditional parameter intervals are distinct from OOF scores and between-fold variability.

The final pass artifact is `PRESS/utils/data/pass_risk_model.json`, loaded by `PassRiskModel.pretrained()` and used by `compute_ppcf` by default. It contains the full-data trajectory-MLE parameters and saved isotonic knots; [pass provenance](../docs/pass-risk-refit-provenance.json) verifies its hash and runtime agreement. The accepted cohort is 6,706 passes (964 failures), with continuous matched endpoints, outfield passers and inclusive duration [0.5, 10] s. `data/benchmarks/pass_risk/` contains aggregate OOF, curvature and off-ball speed summaries. Full per-pass arrays remain in the research repository; the pass cache is `2026-10-18_pass_boundaries_duration_v4`, derived from the shared v5 tracking base without changing retained positions or velocities.

The operational pass-selection artifact is `PRESS/utils/data/pass_selection_model.json` plus its checksum-verified `.ubj`, loaded by `PassSelectionModel.pretrained()`. The full-ten-match flat refit has 1,444 trees and the corrected `ahead_of_carrier_and_target_v1` feature contract. [Pass-selection provenance](../docs/pass-selection-final-provenance.json) identifies screening54, benchmark56 and deployment refit57. `data/benchmarks/pass_selection/` contains screening, pooled/fold/match scores and per-choice OOF losses for both flat and CNL models. Historical tables and package weights are preserved separately; CNL deployment weights remain in the research archive.

`data/benchmarks/action_selection/` contains the final AS70 aggregate pooled,
fold and match scores, calibration bins, exact feature schemas and selected
hyperparameters, plus published comparison values and the separate context and
temperature diagnostics. No raw tracking or individual prediction arrays are
exported. Regenerate the aggregate export with
`python scripts/export_action_selection_benchmark.py` when the sibling research
repository is available; render its figures independently with
`python scripts/build_action_selection_benchmark_figures.py`. The
[provenance](../docs/action-selection-benchmark-provenance.json) separates this
completed development benchmark from the still-historical installed action
booster. Full-data AS refitting and runtime alignment have not yet been exported.
