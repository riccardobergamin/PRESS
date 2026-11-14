# BCOS descriptive example

Brisbane FC–Perth Glory, match 1925299, 21 December 2024.

- `example_summary.json` and `example_field.npz`: real frame 3800, production
  accessibility probabilities at the native one-metre destinations. The player
  plot is generated with `PRESS.visualization.plot_bcos`.
- `spatial_cells.csv`: checkpoint-weighted mean area on a 10x6 carrier-origin
  display grid, with distinct possession counts. Every occupied cell is shown,
  including cells with fewer than five possessions; `low_support` flags them.
- `checkpoint_series.csv`: all 1,928 production BCOS checkpoint values, with
  half, elapsed half-minute, team and carrier-origin third. No tracking positions
  or full feature table are exported. Values have not been refitted.
- `period_third_summary.csv`: half/third summaries of that same series.
- `temporal_thirds.csv`: five-minute-bin aggregates for inspection. The plotted
  trend uses centred rolling five-minute means directly from the checkpoints.
- `grid_support.csv`, `grid_bootstrap_summary.csv`, `grid_mean_stability.csv`
  and numerical convergence tables: retained exploratory audits. The earlier
  8x5 support rule is not the selection rule for the current 10x6 display.

The sample consists of canonical outfield/no-shot action-selection states, not a uniform
time sample. All values use production carry dispersion. Spatial bins locate
the carrier, not accessible destinations; temporal summaries condition on the
carrier's third. These are descriptions, not match or scoring predictions.

From the repository root:
`python -m scripts.build_bcos_vignette_figures`.

Display-only gap handling: `spatial_display_cells.csv` records the three empty
origin cells filled from the checkpoint-weighted means of adjacent observed
cells; they are hatched and labelled 0. `temporal_display.csv` distinguishes
actual five-minute window means from internal linear interpolation, plotted
with dashed segments. Original checkpoint values, spatial aggregates and
reported statistics remain unchanged. No model is refitted.
