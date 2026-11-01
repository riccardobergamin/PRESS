# Frozen model feature importance

`conceptual_split_gain.csv`: final production model and five matching outer
ensembles for pass destination choice, action choice and augmented residual risk.
`importance_summary.csv`: production ranking/share, mean and min/max normalized
gain across the five selected outer models. No new fitting or selection.

Total gain is attributed training split-loss reduction, normalized separately
within each ensemble. One-hot levels are summed into their conceptual category;
unused inputs retain zero. This is not accuracy contribution, causal attribution
or a cross-task comparable effect. Correlation and split opportunities affect
the allocation. Five-fold ranges are not confidence intervals. Calibration
does not change the underlying tree split gains.

The separate BCOS-only OOF permutation diagnostic is under `examples/bcos/`.
It is available for action selection and residual risk, not presented as a
permutation test of the structured pass-destination model.

Rebuild plots with `python scripts/build_feature_importance_figures.py`.
