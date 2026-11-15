# Verification

The checks are outside the installable package. Install pytest in the PRESS
environment, then run from the project root:

```sh
PYTHONPATH=.:verification MPLBACKEND=Agg PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest verification -q
```

No check retrains a fitted classifier. Artificial inputs are confined to unit
tests; website examples use real tracking data.

- `test_transport.py`: initial conditioning, no-killing boundary inflow,
  finite-time mass balance, constant killing against independent quadrature,
  space/time-varying killing against the appendix's density/current solution,
  competing causes, deterministic limit and refinement.
- `test_bcos_transport.py`: shared movement law, grid/area consistency,
  defender padding, no-defender limit and retention of the frozen classifier
  feature alongside the final spatial map.
- `test_external_schema.py`, `test_external_workflow.py`: explicit mappings,
  units, synchronization, periods, missing endpoints, canonical segmentation,
  unknown outcomes and restart-specific reception/interception context.
- `test_xt_input.py`: custom-map propagation through all contextual models,
  arbitrary resolution, native defaults and result-export consistency.
- Remaining checks cover existing fitted models, inference, feature contracts,
  portable artifacts, plots and numerical-result round trips.

For observed-data refinement, provide the selected frame table and downloaded
processed match root explicitly:

```sh
PYTHONPATH=. python verification/check_bcos_convergence.py \
  selected_frames.csv MATCH_ROOT convergence.csv
```

`selected_frames.csv` contains `match_id,frame`; each match directory contains
`<match_id>_tracking.parquet`. Numerical comparison of two BCOS formulations is
an internal diagnostic, not an evaluation of predictive performance.
