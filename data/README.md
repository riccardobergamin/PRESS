# Distributed objects and local research data

The public resources support the final accepted models, their benchmarks and the real-match examples. Raw SkillCorner tracking and event files are obtained from [SkillCorner Open Data](https://github.com/SkillCorner/opendata), following its attribution and usage terms. The [dataset guide](../data.qmd) describes preprocessing, and the [research pipeline](../pipeline.qmd) provides the source code and input/output layout.

## Included with PRESS

All five pretrained inference models are retained under `PRESS/utils/data/`:

| Component | Production artifacts |
|---|---|
| Ball-drive risk | `ball_drive_model.json`, containing the MAP coefficients, spline bases and physics settings |
| Pass probability | `pass_risk_model.json`, containing the physical parameters and isotonic map |
| Pass selection | `pass_selection_model.json` and `pass_selection_model.ubj` |
| Action selection | `action_selection_model.json` and `action_selection_model.ubj` |
| Residual risk | `residual_risk_model.json` and `residual_risk_model.ubj` |

The package also retains the supporting feature schema used by action selection. These are the accepted full-data production models. Their separate held-out results are reported in the benchmarks; deployment refits are not held-out evaluations.

## Included with the site and source release

- **Final benchmark summaries:** pooled, fold-level and match-level metrics; calibration-bin summaries; selected hyperparameters; physical-parameter estimates and uncertainty summaries; documented comparisons and numerical checks.
- **Feature importance:** aggregate production and fold summaries, without full fitted fold ensembles.
- **Reports and case studies:** band summaries, player/team descriptive scores, pattern shares and the selected examples' identifiers and predicted quantities.
- **Derived spatial examples:** the small production xT grid and the selected real-frame BCOS probability field. These contain model-derived surfaces, not raw tracking. BCOS checkpoint values and spatial/time aggregates support the descriptive single-match figures; they are not complete training-feature tables.
- **Provenance:** model identifiers, source descriptions, numerical checks and file hashes.
- **Research code:** the procedural `pipeline/` source, split/grid protocols and documentation, also provided as a [downloadable archive](../pipeline/press-training-pipeline.zip).

The [object inventory](../docs/release-objects.json) records the included data and production artifacts, their roles and SHA-256 hashes. Figure-generation scripts use these summaries or explicit local research inputs; rendering the site does not require raw tracking.

## Private research objects

Raw and processed match tables, complete feature matrices, row-level held-out predictions, full posterior/covariance objects, bootstrap replicates, auxiliary cross-fitting ensembles and full training-search outputs are not included in the release.

The public site includes the aggregate results needed to inspect the reported findings. Reproducing the complete training requires obtaining the source data and running the pipeline. The [pipeline guide](../pipeline.qmd) distinguishes inputs, generated outputs, benchmark fits and production refits.
