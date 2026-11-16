# Expected Threat

Read `model.py` for cells, transitions, goal probability and Bellman iteration; `training.py` for smoothing selection and evaluation; `pipeline.py` for execution.

## Inputs and model

`pipeline/inputs/expected_threat/events/` contains `carries.parquet`, `passes.parquet`, `shots.parquet` and `losses.parquet`. The reference cohort has 9,071 individual-possession displacements, 6,355 completed passes, 212 shots (23 goals) and 2,214 terminal losses. These displacements are not canonical one-second carries. The xT cohort differs from the pass-probability cohort.

Cell assignments are computed from continuous coordinates on a 105 × 68 m pitch. For origin i and destination j:

```text
q_i  = p_shot,i * p_goal,i
M_ij = p_pass,i*T_pass,ij + p_carry,i*T_carry,ij
xT   = q + M*xT
```

The four event frequencies share a cell-level denominator; loss has zero continuation value. Transition distributions are conditional on the corresponding event table. These quantities are distinct from the learned action policy and physical carry probabilities.

Transition smoothing acts on destinations separately for each origin. Bandwidth in metres is converted into x/y Gaussian scales. Unobserved origins receive uniform destination distributions. Shot and goal counts are smoothed separately before taking their ratio. The four event frequencies are not smoothed.

Bellman iteration starts at zero and stops when the maximum change is below 1e-12. The surface and its components are saved together.

## Three estimation roles

**Grid comparison.** Resolutions 8×5, 10×6, 12×8 and 16×12 use the same events and five outer 8+2 partitions. Leave-one-match-out validation within the eight development matches selects each component's bandwidth from 0, 3, 5, 8, 12 and 18 m. Transition scores are event-weighted destination-density NLL; shot scores are Bernoulli CE. The largest bandwidth within one paired match-jackknife standard error of the best is selected.

Pooled outer-test shot CE and Brier score compare resolutions. The 10×6 grid is the coarsest within one paired standard error of minimum CE. This is an exploratory parsimony rule, not proof of statistical equivalence. Coverage and descriptive map variability supplement the comparison. Transition scores account for cell area; the evaluation does not directly validate the full xT surface against future goals.

**Feature maps.** Each downstream fold uses xT fitted only on its six fitting matches, with 5+1 smoothing selection and fixed grid. Its validation and test observations use this frozen map.

**Production.** The ten-match 10×6 map uses 9+1 smoothing selection. Selected bandwidths are 8 m for passes and 5 m for carries and shots; convergence takes 139 iterations. The six-match shot bandwidth is 8 m in folds 0 and 4 and 5 m elsewhere. Production maps never replace fold maps in held-out evaluation.

## Execution

```bash
python pipeline/expected_threat/pipeline.py --stage validation
python pipeline/expected_threat/pipeline.py --stage scores
python pipeline/expected_threat/pipeline.py --stage maps
python pipeline/expected_threat/pipeline.py --stage production
```

Use `--grid 10 6` with validation for one resolution, or `--fold 0` with maps for one fitting map. `--stage all` executes every stage. Choose a fresh `--output`; existing result directories are protected.

Outputs under `pipeline/outputs/expected_threat/` separate `validation/<grid>/fold_0..4/`, `scores/`, `feature_maps/fold_0..4/` and `production/`. `protocol.json` records partitions; manifests record input hashes. `checks/verify_pipeline.py` compares reference components and procedures without installing output in PRESS.

Conceptual reference: [Singh, Introducing Expected Threat](https://karun.in/blog/expected-threat.html). The separate pass/carry transitions and loss branch are the PRESS specification. See the [xT benchmark appendix](../../appendix/xt_benchmark.qmd).
