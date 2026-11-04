# Production xT display

`xt_grid.npy` is bit-identical to the 10x6 `xt_grid` embedded in both deployed
choice and action-selection models. It is the all-ten-match smoothed fit,
not an outer-fold prediction surface. Axis order is (longitudinal x, lateral y),
with attack towards increasing x. The display does not interpolate the cells.

`manifest.json` comes from the frozen production fit. `coverage.csv`,
`metrics.csv` and `map_stability.csv` are the archived exploratory resolution
comparison on the same event tables. Component scores are not validation of
the complete future-goal xT target. Models and weights have not been changed.

Rebuild with `python -m scripts.build_xt_vignette_figures`.
