# Frozen pass-risk benchmark — final uniform TTI

Accepted run41: 6,706 passes,964 failures; unchanged cohort, five outer folds,
nested calibration, reconstructed paths and likelihood. The only physical
change from run22 is preserving v0 when above vmax, applied to the same TTI in
real-path fitting, endpoint PPCF and receiver timing for hypothetical flights.
Settings: vmax5.5,a_linear7,a_turn7. Refitted learned quantities: lambda,s and
isotonic knots. OOF CE0.131121,gain68.192%,ECEq10 0.1833%.

The installed full-data artifact is skillcorner_outfield_pass_risk_uniform_tti_v4,
with lambda_control=5.437829753 and s=0.294559749. Its final isotonic uses run41 outer
cross-fitted PPCF scores. This deployment refit does not replace OOF evaluation.
See docs/pass-risk-refit-provenance.json and the pass-probability vignette.

Historical run22 aggregates remain under old/run22_pre_uniform_tti_20261022/.
The former artifact is archived under docs/old/pass_model_pre_uniform_tti_20261022/.
The cap5.0 sensitivity is historical; the final run fixes cap5.5. Quantile bins
use stable score sorting after canonical match_id/pass_id/frame_end ordering.
Curvature gains use subgroup-training nulls; gain_vs_global_null is separate.
