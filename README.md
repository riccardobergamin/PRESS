# PRESS

<img src="branding/press-logo.svg" align="right" width="155" alt="PRESS logo" />

**Possession Risk Estimate from Spatio-temporal State** (PRESS) is a work-in-progress, physics-informed framework for estimating the risk that a team loses possession. It combines physics-based pass and carry components with learned action and destination choices, and it evaluates possession vulnerability at two horizons: the next on-ball action of the ball carrier (the hazard h) and the eventual outcome of the carrier's individual possession (xLoss).

## Objectives and motivation

This is a portfolio research project to learn the main ideas behind expected possession value (EPV) (Fernández et al., 2019, 2021), work with player tracking data and explore some alternative modelling choices. The goal is to package the research code into reusable components with reproducible examples and results.

The motivating question is: given the current game state and the player in possession, how difficult is it for the team to retain the ball? Pressure can constrain a carrier's options, raise interception risk and affect action choice. PRESS adapts building blocks from pitch control and EPV research to express that question as a team-possession hazard over canonical action opportunities (Spearman et al., 2017; Fernández et al., 2021; Bekkers, 2025; SkillCorner, 2026).

The model style is hybrid: pass and carry outcomes use physics-inspired motion and arrival probabilities, while action selection and pass destination use tree models. The risk unit is team possession, represented as a discrete-time survival process over a sequence of canonical on-ball actions across successive carriers. A successful pass is nonterminal: it continues team possession with the receiver. It ends only the passer's individual control, which is the separate endpoint used by xLoss. The current-action hazard can be evaluated at any usable tracking frame attributable to the current carrier. Canonical action starts define the benchmark, and separately marked snapshots can enrich a descriptive trajectory.

Pressure means constraining the carrier's time, space and usable passing options in this project. The local next-step estimate is forward-looking, whereas the observed-possession summary is retrospective. See the [hazard and bands vignette](vignettes/06_composed_hazard_and_bands/06_composed_risk_and_bands.ipynb) for the interpretation of the hazard and the [possession segmentation page](vignettes/setup/possession_and_checkpoint_setup.ipynb) for the definition of canonical actions and checkpoints.

## Model overview

![PRESS model overview](figures/model_overview.svg)

At each evaluated frame, PRESS composes the drive and pass branches as

`h = P(drive) × drive_loss + P(pass) × pass_failure_expected`

where the pass failure is averaged over candidate destinations. The expected loss of the individual possession adds the continuation after a successful carry:

`xLoss = h + P(carry) × (1 − drive_loss) × residual_loss`

Operational difficulty bands follow from the half-life of possession under a constant hazard, with cutoffs at half-lives of 8, 4 and 2 actions.

**Ball-drive risk.** The drive solver accumulates defender-specific loss probabilities under a shared remaining survival mass. The tackle rate has additive penalized spline effects for carrier speed, signed distance to the pitch boundary and defender approach angle. `BallDriveModel.pretrained()` loads the ten-match refit, which the frame API uses by default. The drive model names its baseline tackle rate `lambda_0`, and `lambda_control` belongs to pass control.

**Pass probability.** Two physical parameters, the control rate and the arrival dispersion, are fitted by maximum likelihood along reconstructed pass trajectories and transferred to target pitch control, with exact-distance flight and exponential integration. An isotonic map calibrates the completion probability. `PassRiskModel.pretrained()` loads parameters and calibration, `PRESS(..., pass_risk_model=model)` accepts it explicitly, and `compute_ppcf` uses it by default. The [vignette](vignettes/02_pass_probability/02_pass_probability.ipynb) explains flight, control and model assumptions, and the [provenance file](docs/pass-risk-refit-provenance.json) records the refit.

**Choice and residual models.** Pass selection is a flat XGBoost categorical model over candidate destinations. Action selection is a binary XGBoost pass-versus-carry policy. Residual risk estimates the probability that the individual possession eventually ends in a loss after a successful carry. The tree models use features derived from the physical components together with tactical and contextual information.

### Fitted models shipped with the package

| Task | Model | Deployment fit |
|---|---|---|
| Ball-drive risk | Competing-risk physical model with additive penalized splines | Maximum a posteriori refit on the ten matches |
| Pass probability | Trajectory-likelihood physical model with isotonic calibration | Maximum likelihood refit on the ten matches |
| Pass selection | Flat XGBoost categorical choice model | 1,444 trees with temperature scaling |
| Action selection | XGBoost binary policy | Native probabilities, no scaling |
| Residual risk | XGBoost conditional loss model | 118 trees with native probabilities |

The package checks the integrity and feature contract of each tree model when it is loaded. The distributed models are refitted on all ten matches, and the benchmark metrics come from separate held-out predictions.

## Package API

`PRESS` represents one match. Use `PRESS.from_skillcorner(...)` to read original SkillCorner match files and prepare velocities, possessions and actions in memory. For other data already prepared by the caller, initialize `PRESS(tracking_data, ...)` directly; `PRESS.io.segment_possessions(...)` constructs individual controls/actions when they are missing. Input frequency, pitch dimensions and source-ID mapping are explicit. Initialization validates and copies the inputs and performs no inference.

The workflow is load → navigate → compute → retrieve → plot or save. Team and individual possessions have separate getters. Any usable frame can be evaluated independently of canonical checkpoint sampling. `compute_risk_frame(frame_id)` computes h, and `include_residual=True` also computes the conditional residual risk and the terminal individual xLoss, deriving the elapsed time from annotations and never imputing zero. The fitted xT surface and the frozen model parameters are included in the pretrained artifacts.

```python
from PRESS import PRESS
from PRESS.visualization import plot_ball_drive_risk

press = PRESS(tracking, player_possessions_data=individuals,
              actions_data=actions, fps=input_frequency)
frame_id = int(press.get_checkpoints().frame_id.iloc[0])
press.compute_ball_drive_risk(frame_id)
result = press.get_ball_drive_risk_result(frame_id)
fig, ax = plot_ball_drive_risk(press, frame_id, show=False)
fig.savefig("drive_risk.png", dpi=160)
press.save_results("outputs/match_results")
```

All plot functions return figure and axes. Getters copy results by default. Saved results use JSON metadata and compressed NPZ arrays, with input and model fingerprints and no pickle. An upstream recomputation invalidates the dependent outputs.

## Documentation

**User guide.** [Using the package](usage.qmd) follows the path from raw observations to a possession-level risk profile:

1. [Dataset and preprocessing](data.qmd): preprocess one observed match and inspect the input tables.
2. [Possession segmentation and context](vignettes/setup/possession_and_checkpoint_setup.ipynb): individual possessions, canonical actions and checkpoints, additional snapshots, start type and previous action.
3. [Installation and use](vignettes/00_match_workflow/00_match_workflow.ipynb): a complete real-match workflow with h and xLoss, plots, getters and result persistence.

**Models.** Each vignette explains the concept, the mathematics and an observed match example using the package visualization.

- [01 · Ball-drive risk](vignettes/01_ball_drive_risk/01_ball_drive_risk.ipynb): geometry, carrier speed, boundary distance and competing defender causes.
- [02 · Pass probability](vignettes/02_pass_probability/02_pass_probability.ipynb): simplified flight path and arrival and control accumulation.
- [03 · Pass selection](vignettes/03_pass_selection/03_pass_selection.ipynb): destination choice across candidate cells.
- [04 · Action selection](vignettes/04_action_selection/04_action_selection.ipynb): pass-versus-carry choice and its calibration.
- [05 · Residual risk](vignettes/05_residual_risk/05_residual_risk.ipynb): conditional terminal outcome after a successful carry.

**Risk profiles and reports.**

- [06 · Composed hazard and bands](vignettes/06_composed_hazard_and_bands/06_composed_risk_and_bands.ipynb): the hazard, half-life bands and held-out band validation.
- [07 · Case studies](vignettes/07_case_studies/07_case_studies.ipynb): pressure patterns, observed cases and the team and player report.

**Benchmarks.** The [benchmark overview](benchmark.qmd) describes the dataset, the shared evaluation protocol and the metrics. Each task has its own page: [ball-drive risk](benchmarks/01_ball_drive_risk.qmd), [pass probability](benchmarks/02_pass_probability.qmd), [pass selection](benchmarks/03_pass_selection.qmd), [action selection](benchmarks/04_action_selection.qmd), [residual risk](benchmarks/05_residual_risk.qmd) and [composed risk](benchmarks/06_composed_risk.qmd).

**Spatial features: accessibility and value.** [Ball Carrier Open Space](vignettes/spatial_features/bcos/10_bcos.ipynb), whose scalar area is a feature of action selection, and [Expected Threat](vignettes/spatial_features/expected_threat/11_expected_threat.ipynb), which builds the xT grid consumed by PRESS (Singh, 2018).

**Reference.** [Initialization and classes](api.qmd), [solvers](solvers.qmd), [getters and saved results](reference/getters.qmd), [plot functions](reference/plots.qmd), the [research pipeline](pipeline.qmd) and the [project plan](roadmap.qmd).

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

## Reproduce

Use Python 3.11 (the verification interpreter) or 3.12. Create the environment pinned to the versions used for the executable checks, then run the examples and notebooks, saving outputs and PNG/SVG figures:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -e '.[io,vignettes]'
python scripts/check_environment.py --extras io,vignettes
python -m examples.checkpoint --help
python scripts/rebuild_vignettes.py
python scripts/check_docs.py
```

The checked-in model examples use observed matches and saved figures. Re-executing SkillCorner examples requires the original match JSON, tracking JSONL and dynamic-event CSV under the root selected by `SKILLCORNER_OPENDATA`; no precomputed Parquet files are required. Rendering saved outputs does not require the input data. The command-line example requires `--match-dir` and reads an observed match. See the [dataset and preprocessing guide](data.qmd) and [data/README.md](data/README.md) for local data preprocessing.

## Build the website

Install Quarto 1.10.19 and the pinned Python environment above, activate that environment, then run:

```bash
quarto render --no-execute
```

Quarto renders the saved notebook outputs as pages in `_site/`. Re-executing the notebook code requires the local processed matches. The publishing workflow is started manually through GitHub Actions. In the repository settings, select **Settings → Pages → Build and deployment → GitHub Actions** for the first deployment. The intended site URL is <https://riccardobergamin.github.io/PRESS/>.

## Data

PRESS uses ten matches of the Australian A-League 2024/25 season from the [SkillCorner open-data repository](https://github.com/SkillCorner/opendata), which requests attribution. Raw SkillCorner tracking and event files and match-level derived tables are not included in this repository. The MIT license covers the software and sets no data redistribution terms. The repository includes the learned XGBoost and ball-drive parameter files, aggregate benchmark summaries and saved real-match illustrations.

## Repository map

- `PRESS/`: the importable package, split into `core`, `io`, `model`, `physics`, `utils`, `spatial_features` and `visualization`.
- `vignettes/`: runnable notebooks for the models, the case studies, the match workflow and the setup page. BCOS and expected threat are under `vignettes/spatial_features/`.
- `benchmarks/`: one benchmark page per task, and `benchmark.qmd` for the shared protocol.
- `index.qmd`, `data.qmd`, `usage.qmd`, `api.qmd`, `solvers.qmd`, `reference/`, `pipeline.qmd`, `roadmap.qmd`, `_quarto.yml`: site pages and navigation.
- `docs/`, `figures/`, `data/`: provenance records, figures and data setup notes.
- `pipeline/`: procedural research code for preprocessing, features, training, evaluation and production refits, linked from `pipeline.qmd`.
- `scripts/`, `examples/`: environment checks, notebook rebuild and command-line examples.
- `.github/workflows/publish-site.yml`: GitHub Pages build and deployment.
- `references/refs.bib`, `REFERENCES.md`: citation records and unresolved references.
- `archive/legacy_package/`: unused implementations, outside the installed package.

## AI assistance

The author designed the modeling framework and validation protocols, and developed the core software architecture. Codex (OpenAI) assisted with pipeline implementation, code refinement, and debugging under the author's supervision.

## Citation

Cite a release using the metadata in [CITATION.cff](CITATION.cff), and include the repository version or commit used.

## References

- Bekkers, J. (2025). Pressing intensity: an intuitive measure for pressing in soccer. [arXiv:2501.04712](https://arxiv.org/abs/2501.04712).
- Fernández, J., Bornn, L. and Cervone, D. (2019). Decomposing the immeasurable sport: a deep learning expected possession value framework for soccer. MIT Sloan Sports Analytics Conference.
- Fernández, J., Bornn, L. and Cervone, D. (2021). A framework for the fine-grained evaluation of the instantaneous expected value of soccer possessions. Machine Learning 110, 1389–1427. [doi:10.1007/s10994-021-05989-6](https://doi.org/10.1007/s10994-021-05989-6).
- Singer, J. D. and Willett, J. B. (2003). Applied Longitudinal Data Analysis: Modeling Change and Event Occurrence. Oxford University Press.
- Singh, K. (2018). [Introducing Expected Threat (xT)](https://karun.in/blog/expected-threat.html).
- SkillCorner (2026). [Introducing SkillCorner's new pressure metrics](https://skillcorner.com/us/articles/introducing-skillcorners-new-pressure-metrics).
- Spearman, W., Basye, A., Dick, G., Hotovy, R. and Pop, P. (2017). [Physics-based modeling of pass probabilities in soccer](https://static.hudl.com/craft/downloads/SSAC17-Physics-Based-Modeling-of-Pass-Probabilities-in-Soccer.pdf). MIT Sloan Sports Analytics Conference.
