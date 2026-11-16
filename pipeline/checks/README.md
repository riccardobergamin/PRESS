# Pipeline verification

The default checks parse the source, run each command-line help page, prepare one observed match from the original SkillCorner files, and re-export the source archive. Full task checks use completed numerical inputs from a research run and short fits; they do not repeat complete XGBoost or CatBoost searches.

```bash
python pipeline/checks/run.py --output /path/to/new_checks
# Add --raw-root /path/to/opendata/data/matches --execute to run the default checks.
# Use --step pass_selection only after preparing the full task inputs.
```

`--package-root` selects a PRESS checkout without modifying the Python installation. `--raw-root` selects the public source data. Each command has a log; `checks.json` summarizes its outcome.

| Stage | Checks |
|---|---|
| Syntax and CLI | Parse scripts and run pipeline `--help` commands |
| Preprocessing | Read a real match; estimate kinematics; match boundaries and segment actions |
| xT | Events, four resolutions, smoothing selection, fold/production maps and reference comparisons |
| Physical ball-drive | Covariates, bases, priors, short MAP fit, Laplace, predictions, metrics and exported-model loading |
| CatBoost | TTI, features, training-only sampling, prior correction and calibration/metric replay; native fitting requires CatBoost |
| Pass probability | Observed trajectories, TTI, PPCF, small multistart MLE, Platt/isotonic, metrics and serialization |
| Pass selection | Candidate alignment, probabilities, gradients, curvature, reduced screening, tree/nesting updates, early stopping, temperature, scores and export |
| Action/residual | Features, encoding, short native fits, selection, P(carry) cross-fitting, refit and export |
| h/xLoss | Component composition, targets, fold nulls, pooled CE/gain/ECE and band bootstrap |
| Reports | Micropossessions, patterns, entity scores, intervals and case tables |
| Release | Explicit export of code and statistical summaries |

## Pass-selection checks

`pass_selection_steps.py` retains the full destination grid and ten nests for one observed pass per match. It compares features with the package builder, checks `CNL(ρ=1)=flat`, normalization and gradients against automatic differentiation. Flat curvature is checked against its formula; CNL curvature follows the RUMBoost convention and is not the exact Hessian.

A 22-tree fit exercises the nesting update at tree 20. A separate fit tests selection of the joint tree/nesting checkpoint. Tests also cover validation temperature, test predictions, serialization, importance and a small production export. Five short folds exercise pooled coverage, median tree count and out-of-fold production temperature. Frozen physical features are interface fixtures, not a new fold-specific scientific evaluation.

A skipped native fit is not a passed test. Short-fit results do not establish full-search convergence or replace published benchmark results.
