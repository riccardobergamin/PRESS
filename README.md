# PRESS

[![Documentation](https://img.shields.io/badge/Documentation-PRESS%20Website-blue?style=flat)](https://riccardobergamin.github.io/PRESS/)

<img src="branding/press-logo.svg" align="right" width="155" alt="PRESS logo" />

**Possession Risk Estimate from Spatio-temporal State** (PRESS) is a work-in-progress, physics-informed framework for estimating football possession risk from tracking data. It combines physical models of passing and carrying with learned action and destination choices.

**[Website & documentation](https://riccardobergamin.github.io/PRESS/)** · [Getting started](https://riccardobergamin.github.io/PRESS/usage.html) · [Benchmarks](https://riccardobergamin.github.io/PRESS/benchmark.html)

## Motivation and scope

PRESS asks how difficult it is for a team to retain the ball given the current game state and ball carrier. Pressure can restrict time, space and passing options, increase interception risk and alter decisions. The framework adapts concepts from pitch control and expected possession value (EPV) research to model possession vulnerability (Spearman et al., 2017; Fernández et al., 2019, 2021; Bekkers, 2025; SkillCorner, 2026).

This is a portfolio research project exploring EPV, tracking data and alternative modelling choices through reusable software and reproducible analyses. The relatively simple models reflect the available ten-match sample and project scope; see [Modelling choices](pipeline.qmd#modelling-choices).

PRESS distinguishes two outcomes:

- **Hazard (h):** probability of losing *team possession* during the ball carrier's next on-ball action. A completed pass continues team possession with the receiver.
- **xLoss:** probability that the *current carrier's individual possession* eventually ends in a loss, accounting for the continuation after a successful carry.

The hazard is evaluated at usable tracking frames and is interpreted as a discrete-time survival process across successive carriers. Canonical action starts define benchmark checkpoints; additional snapshots can describe within-possession trajectories. See the [hazard and bands vignette](vignettes/06_composed_risk/06_composed_risk.ipynb) and [segmentation guide](vignettes/setup/possession_and_checkpoint_setup.ipynb).

## Model overview

![PRESS model overview](figures/model_overview.svg)

At each evaluated frame, the two action branches are combined as:

`h = P(drive) × drive_loss + P(pass) × pass_failure_expected`

where pass failure is averaged over candidate destinations. Individual-possession loss additionally accounts for continuation after a successful carry:

`xLoss = h + P(carry) × (1 − drive_loss) × residual_loss`

The framework comprises five fitted components:

- **Ball-drive risk:** defender-specific competing loss processes, with additive penalized spline effects for carrier speed, signed boundary distance and defender approach angle.
- **Pass probability:** trajectory-based arrival and control probabilities, fitted by maximum likelihood and calibrated with isotonic regression.
- **Pass selection:** a flat XGBoost categorical choice model over candidate destinations.
- **Action selection:** an XGBoost binary pass-versus-carry policy.
- **Residual risk:** an XGBoost model of individual-possession loss conditional on a successful carry.

Operational hazard bands use possession half-life under a constant hazard, with cutoffs of 8, 4 and 2 actions. Full mathematical details, estimation and model assumptions are covered in the [model documentation](https://riccardobergamin.github.io/PRESS/).

### Fitted models

| Task | Model | Deployment fit |
|---|---|---|
| Ball-drive risk | Competing-risk physical model with additive penalized splines | Maximum a posteriori refit on the ten matches |
| Pass probability | Trajectory-likelihood physical model with isotonic calibration | Maximum likelihood refit on the ten matches |
| Pass selection | Flat XGBoost categorical choice model | 1,444 trees with temperature scaling |
| Action selection | XGBoost binary policy | Native probabilities, no scaling |
| Residual risk | XGBoost conditional loss model | 118 trees with native probabilities |

The package checks the integrity and feature contract of each tree model when it is loaded. The distributed models are refitted on all ten matches, and the benchmark metrics come from separate held-out predictions.
## Quick start

PRESS represents one match. `PRESS.from_skillcorner(...)` loads SkillCorner files and prepares velocities, possessions and actions; prepared data from other providers can be passed directly to `PRESS(tracking_data, ...)`. Initialization validates and copies inputs without running inference.

The workflow is **load → navigate → compute → retrieve → plot or save**. For example:

```pythonimport os
from pathlib import Path
from PRESS import PRESS
from PRESS.visualization import plot_ball_drive_risk

match_dir = Path(os.environ["SKILLCORNER_OPENDATA"]) / "data/matches/2006229"
press = PRESS.from_skillcorner(match_dir)
actions = press.get_actions(valid_only=True)
frame_id = int(actions.frame_start.iloc[0])
press.compute_ball_drive_risk(frame_id)
result = press.get_ball_drive_risk_result(frame_id)
fig, ax = plot_ball_drive_risk(press, frame_id, show=False)
fig.savefig("drive_risk.png", dpi=160)
press.save_results("outputs/match_results")```

`compute_risk_frame(frame_id)` estimates h; setting `include_residual=True` also computes conditional residual risk and xLoss. Plot functions return figure and axes. Results are saved as JSON metadata and compressed NPZ arrays, with fingerprints for inputs and models. See [Using the package](usage.qmd) and the [match workflow](vignettes/00_match_workflow/00_match_workflow.ipynb).

## Performance

Held-out results pool the five outer match folds. The deployment refits are separate from these predictions, and the [scope of the development benchmark](benchmark.qmd#scope-of-the-development-benchmark) is described in the benchmark overview.

| Task | Cohort | Cross-entropy | Gain over null | ECE |
|---|---|---:|---:|---:|
| Ball-drive risk | 9,114 carries, 456 losses | 0.143483 | 27.80% | 0.475% |
| Pass probability | 6,706 passes, 964 losses | 0.131121 | 68.19% | 0.183% |
| Pass selection | 6,810 decisions | 5.827787 | McFadden R² 34.322% | not reported |
| Action selection | 16,918 decisions | 0.568592 | 17.602% | 1.026% |
| Residual risk | 7,820 checkpoints | 0.419506 | 11.793% | 1.053% |
| Hazard h | 15,085 checkpoints, 1,465 losses | 0.272014 | 14.765% | 0.749% |
| xLoss | 15,085 checkpoints, 2,888 losses | 0.421925 | 13.769% | 0.930% |

Pass probability uses isotonic calibration and pass selection uses the flat model with temperature scaling. The task benchmarks give the cohorts, null models, comparisons with published results and calibration plots.
## Installation and reproducibility

Python 3.11 (the verification interpreter) or 3.12 is supported for the documented reproduction workflow. To install the package with data-loading and vignette dependencies and run the checks:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -e '.[io,vignettes]'
python scripts/check_environment.py --extras io,vignettes
python -m examples.checkpoint --help
python scripts/rebuild_vignettes.py
python scripts/check_docs.py
```

Re-executing SkillCorner examples requires the original match JSON, tracking JSONL and dynamic-event CSV under `SKILLCORNER_OPENDATA`; precomputed Parquet files are not required. Saved figures and notebook outputs can be rendered without the original input data. The command-line example requires `--match-dir`. See the [data and preprocessing guide](data.qmd) and [data setup notes](data/README.md).

## Documentation

The **[PRESS website](https://riccardobergamin.github.io/PRESS/)** contains the complete user guide, methodological vignettes, benchmark evaluations and API reference.

- **Getting started:** [datasets and preprocessing](data.qmd), [using the package](usage.qmd), [possession segmentation](vignettes/setup/possession_and_checkpoint_setup.ipynb), [real-match workflow](vignettes/00_match_workflow/00_match_workflow.ipynb) and [other data providers](external-data.qmd).
- **Models:** [ball-drive risk](vignettes/01_ball_drive_risk/01_ball_drive_risk.ipynb), [pass probability](vignettes/02_pass_probability/02_pass_probability.ipynb), [pass selection](vignettes/03_pass_selection/03_pass_selection.ipynb), [action selection](vignettes/04_action_selection/04_action_selection.ipynb) and [residual risk](vignettes/05_residual_risk/05_residual_risk.ipynb).
- **Risk and applications:** [composed hazard and bands](vignettes/06_composed_risk/06_composed_risk.ipynb), [case studies and reports](vignettes/07_case_studies/07_case_studies.ipynb), [Ball Carrier Open Space](vignettes/spatial_features/bcos/10_bcos.ipynb) and [Expected Threat](vignettes/spatial_features/expected_threat/11_expected_threat.ipynb).
- **Evaluation and reference:** [benchmark overview](benchmark.qmd), [API](api.qmd), [solvers](solvers.qmd), [getters](reference/getters.qmd), [plots](reference/plots.qmd), [research pipeline](pipeline.qmd) and [roadmap](roadmap.qmd).

## Data and repository organization

PRESS uses ten Australian A-League 2024/25 matches from [SkillCorner open data](https://github.com/SkillCorner/opendata), subject to SkillCorner's attribution request. Raw tracking and event files and match-level derived tables are not distributed here. The MIT license applies to the software and does not establish data redistribution rights. The repository includes fitted model artifacts, aggregate benchmark results and saved real-match illustrations.

- `PRESS/`: importable package (`core`, `io`, `model`, `physics`, `utils`, `spatial_features`, `visualization`).
- `vignettes/`, `benchmarks/`: notebooks and task-level evaluations.
- `pipeline/`: preprocessing, feature generation, training, evaluation and deployment refits.
- `scripts/`, `examples/`: checks, notebook rebuilding and command-line examples.
- `docs/`, `figures/`, `data/`: provenance, figures and data setup notes.
- `*.qmd`, `reference/`, `_quarto.yml`: website content and navigation; `.github/workflows/publish-site.yml` publishes the site.
- `references/refs.bib`, `REFERENCES.md`: bibliographic records.

## AI assistance

The author designed the modeling framework and validation protocols, and developed the core software architecture. Codex (OpenAI) assisted with pipeline implementation, code refinement, and debugging under the author's supervision.

## Citation

Cite a release using [CITATION.cff](CITATION.cff), including the repository version or commit used.

## References

- Bekkers, J. (2025). Pressing intensity: an intuitive measure for pressing in soccer. [arXiv:2501.04712](https://arxiv.org/abs/2501.04712).
- Fernández, J., Bornn, L. and Cervone, D. (2019). Decomposing the immeasurable sport: a deep learning expected possession value framework for soccer. MIT Sloan Sports Analytics Conference.
- Fernández, J., Bornn, L. and Cervone, D. (2021). A framework for the fine-grained evaluation of the instantaneous expected value of soccer possessions. Machine Learning 110, 1389–1427. [doi:10.1007/s10994-021-05989-6](https://doi.org/10.1007/s10994-021-05989-6).
- Singer, J. D. and Willett, J. B. (2003). Applied Longitudinal Data Analysis: Modeling Change and Event Occurrence. Oxford University Press.
- Singh, K. (2018). [Introducing Expected Threat (xT)](https://karun.in/blog/expected-threat.html).
- SkillCorner (2026). [Introducing SkillCorner's new pressure metrics](https://skillcorner.com/us/articles/introducing-skillcorners-new-pressure-metrics).
- Spearman, W., Basye, A., Dick, G., Hotovy, R. and Pop, P. (2017). [Physics-based modeling of pass probabilities in soccer](https://static.hudl.com/craft/downloads/SSAC17-Physics-Based-Modeling-of-Pass-Probabilities-in-Soccer.pdf). MIT Sloan Sports Analytics Conference.
