# Pass-risk benchmark

The cohort contains 6,706 passes, including 964 failures. Evaluation uses five
outer folds, nested calibration and reconstructed ball trajectories. Initial
player speed is preserved when it exceeds vmax; the same TTI is used for
trajectory fitting, endpoint PPCF and receiver timing for hypothetical flights.
Settings: vmax=5.5, a_linear=7, a_turn=7.
OOF CE=0.131121, gain=68.192%, ECEq10=0.1833%.

The production artifact is `skillcorner_outfield_pass_risk_uniform_tti_v4`,
with lambda_control=5.437829753 and s=0.294559749. Its isotonic map is fitted
to cross-fitted PPCF scores. The production refit is distinct from held-out
evaluation. See `docs/pass-risk-refit-provenance.json` and the
pass-probability vignette.

Quantile bins use stable score sorting after canonical
match_id/pass_id/frame_end ordering. Curvature gains use subgroup-training
nulls; gain_vs_global_null is separate.
