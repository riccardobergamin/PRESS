# Action-selection and residual-risk inputs

| Input | Content |
|---|---|
| `action_rows.parquet` | 16,918 canonical decisions, including 7,770 passes, with match/action/frame keys, matched carriers, labels and observed context |
| `residual_labels.parquet` | 7,820 successful carries with verified terminal outcomes, including 1,423 terminal losses, observed possession age and parent identifiers |
| `protocol.json` | Match partitions, the 144-configuration grid, seeds and preliminary effective-hazard BCOS settings |
| `upstream/fold_0..4/` | Physical models, flat pass selection and xT associated with the fold's six fitting matches |
| `upstream/production/` | Full-ten-match upstream models |
| `reference/` | Numerical tables, grids and selections for verification; `action_features_subset` is only a held-out subset, not a complete training cache |
| `manifest.json` | File hashes and cohort counts |

Tracking is shared under `../shared/tracking/`. Carrier identities are joined by match/action/frame keys; unknown context categories are retained. Both classifiers use native probabilities. Residual training receives cross-fitted P(carry); validation and test receive the corresponding main action-selection model's predictions.

Outputs are written to separate task and feature/training/score/production directories under `pipeline/outputs/binary_classification/`. Reference tables are not automatically used as new training features or fitted models.
