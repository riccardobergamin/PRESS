# Pass selection

`features.py` constructs covariates; `training.py` defines likelihoods and learning; `pipeline.py` coordinates the stages. Start with `run_pipeline`, then `train_validate_test` for one fold and `train_model` for the tree-building loop.

```text
prepare_features → compare_configurations → select_best_configuration
→ train_validate_test → compute_pooled_metrics → train_production_model
```

## Model inputs and learning

`load_pass_tables` returns three DataFrames:

| Object | Index | Columns | Rows |
|---|---|---|---:|
| `candidate_features` | `(pass_index, destination)` | F features | C_tot |
| `observed_choices` | `pass_index` | `match_id`, `pass_id`, `frame_end`, `outcome`, `chosen_cell`, `n_candidates` | P |
| `log_memberships` | Same as candidate features | `log_alpha_0` through `log_alpha_9` | C_tot |

P is the number of passes, C the cells per pass, F the features and N the teammates/nests; T denotes temperature. `C_tot = sum_i C_i`, or P×C for equal grids. `pass_index` is a table index; original identifiers remain in `observed_choices`. Memberships are present only for CNL and refer to roster slots, not fixed player identities across passes. They are not utility features.

```python
candidate_features.loc[i]   # C × F candidate features
observed_choices.loc[i]     # observed destination and original identifiers
log_memberships.loc[i]      # C × N log memberships
```

`L(u, y, rho, log_alpha, condition=True)` returns observed-choice losses. With `condition=False`, it returns negative log probabilities for all destinations. `loss_gradient` returns the gradient and scalar curvature for the next tree. Inputs have shapes u=(P,C), y=(P,), log_alpha=(P,C,N), with no memberships for flat softmax.

`build_xgboost_input`, `xgboost_losses` and `xgboost_gradients` convert these arrays for XGBoost. Groups of 32 limit working memory; every pass contributes to learning. Utilities are the sum of tree outputs. Each `model.boost(..., grad=g, hess=h)` adds one tree, then validation CE is recomputed. `best_tree_count` and `best_rho` retain the jointly selected checkpoint; `rounds_without_improvement` controls early stopping.

For CNL, rho=1/mu is updated every 20 trees with bounded SLSQP and damping 0.1. Screening starts from the current rho; evaluation/production fits additionally scan [1,1.5,2,3,5,10]. Flat curvature uses the Friedman factor C/(C−1); CNL uses the RUMBoost scalar-curvature convention. Formula annotations are adjacent to the implementation.

`compute_pass_losses` returns per-pass NLL; `compute_metrics` summarizes a score table; `compute_pooled_metrics` combines the five outer test sets. Temperature scaling uses fixed selected trees and nesting parameters.

## Features

The cohort contains 6,810 passes. Release states use `frame_states.py` and `get_pv_players`. Fold inputs contain physical and xT models associated with six fitting matches; production inputs use ten-match models. Pass and carry arrival dispersions remain distinct. Categorical vocabularies are learned on fitting rows.

Features cover pass geometry, carrier speed, raw target pitch control, carrier/target xT and pressure-line indices, one-second carry loss, match clock and score, start type, previous action and player counts beyond transverse lines. Counts use the carrier team's attacking direction for both rosters. CNL memberships are normalized individual raw pitch-control contributions of the ten non-carrier teammates.

## Screening and evaluation

Five configurations per family are screened on fold 0's six fitting and two validation matches. Native validation CE selects the configuration; screening uses no test scores or temperature scaling. Utilities start at zero and rho at one.

| Configuration | Depth | Child weight | L2 | Eta | Tree ceiling |
|---|---:|---:|---:|---:|---:|
| `01_depth6_child5` | 6 | 5 | 3 | 0.05 | 2,000 |
| `02_depth3_child5` | 3 | 5 | 1 | 0.05 | 3,000 |
| `03_depth4_child20` | 4 | 20 | 1 | 0.05 | 2,000 |
| `04_depth4_child1` | 4 | 1 | 1 | 0.05 | 2,000 |
| `05_depth4_child5_baseline` | 4 | 5 | 1 | 0.05 | 2,000 |

Selected settings are depth 4/child 20/L2 1 for flat and depth 6/child 5/L2 3 for CNL. Both use CPU histogram boosting, row/column subsampling 0.8, max_bin=256, patience 100 and seed 20261019+fold.

Five 6+2+2 fits follow. The fixed validation pair selects the joint ensemble/rho checkpoint and then temperature; there is no eight-match choice-model refit before testing. Screening is shared across folds, not an independently nested search. Pooled metrics concatenate the test predictions.

Production uses ten matches and the median selected tree count: 1,444 flat or 359 CNL. Production temperature is fitted to native outer out-of-fold distributions, not in-sample refit predictions.

## Commands

```bash
python pipeline/pass_selection/pipeline.py --stage features --fold 0
python pipeline/pass_selection/pipeline.py --stage screen --threads 4
# Prepare the other folds, then:
python pipeline/pass_selection/pipeline.py --stage train --threads 4
python pipeline/pass_selection/pipeline.py --stage scores
python pipeline/pass_selection/pipeline.py --stage features --production-features
python pipeline/pass_selection/pipeline.py --stage refit --threads 4
```

Select one screening job with `--family cnl --configuration 01_depth6_child5`; aggregate completed configurations with `--stage screen-report --family cnl`. `--stage all` executes the full sequence in a fresh output directory. Skipping screening uses the declared configurations; partial screening must be completed. Outputs are protected against overwriting.

## Outputs and checks

Shared feature caches are under `features/`; screening results under `screening/<family>/<configuration>/`; each family has `training/fold_<f>/`, `scores/` and `production/`. Training outputs include native test log probabilities for production calibration.

Feature importance is normalized total split gain, summing one-hot levels first. Production values and fold mean/minimum/maximum remain separate; fold ranges are not confidence intervals or causal effects.

`checks/verify_pipeline.py` exercises ten real releases, table alignment, loss derivatives, 22-tree fits, stopping, calibration and serialization. Short fits do not select final hyperparameters. See the [benchmark](../../benchmarks/03_pass_selection.qmd) for the RUMBoost article, cross-nested method and source-code attribution.
