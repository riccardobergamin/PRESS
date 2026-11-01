# Frozen pass-risk benchmark

The user accepted the run22 training cohort:6706 passes,964 failures, outfield
passers, continuous matched endpoints and inclusive duration[0.5,10]seconds.
All valid curvature levels remain. The selected model is trajectory MLE with
vmax5.5,a7 and isotonic calibration. Pooled nested OOF:CE0.141912,gain65.57%,
ECEq10 0.3529%. Platt remains a comparator, not the production calibrator.

The verified full-data artifact is installed as PRESS/utils/data/pass_risk_model.json.
Its lambda_control is5.206773,s0.330402,boundary fit_positions. The operational
fit is separate from the OOF performance. Provenance is in
[pass refit provenance](../../../docs/pass-risk-refit-provenance.json). Old aggregates remain under old/.
Shorter passes with an adapted boundary estimator remain a future extension.
