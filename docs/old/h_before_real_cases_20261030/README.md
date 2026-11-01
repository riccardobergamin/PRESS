# PRESS

**Possession Risk Estimate from Spatio-temporal State** is a work-in-progress, physics-informed framework for estimating the discrete-time risk that a football team loses possession at its next action. The code combines physics-based pass and carry components with learned action and destination choices. [@spearman2017; @fernandez2019] **own derivation**

## Objectives and motivation

This project is a portfolio research project to learn the main ideas behind expected possession value (EPV), work with player tracking data, and explore a physics-based formulation of ball-drive likelihood. The goal is to package the existing research code into reusable components with reproducible examples and results. [@fernandez2019] **own derivation**

The motivating question is: given the current game state and the player in possession, how difficult is it for the team to retain the ball? Pressure can constrain a carrier's options, raise interception risk, and affect action choice. PRESS adapts building blocks from pitch control and EPV research to express that question as a team-possession hazard. [@spearman2017; @fernandez2019; @bekkers2025; @skillcorner-pressure] **own derivation**

The model style is hybrid: pass and carry outcomes use physics-inspired motion and arrival probabilities; action selection and pass destination use tree models. The possession unit is a team sequence of player possessions. A completed pass is a nonterminal transition; loss is terminal. The next-action hazard can be evaluated at any usable tracking frame attributable to the current carrier. For a possession trajectory, an action-based discretization supplies the current default sampling; this choice will be reviewed during benchmarking. [@fernandez2019] **design choice** [@singer2003] **own derivation**

Pressure means constraining the carrier's time, space and usable passing options in this project. The local next-step estimate is forward-looking; the observed-possession summary is retrospective. See [interpretation and action units](interpretation.qmd) for the verified Fernandez convention and the separate individual-possession endpoint. **design choice**

## Model overview

![PRESS model overview](figures/model_overview.svg)

At each evaluated frame, PRESS composes the drive and pass branches as

`h_t = P(drive | s_t) × drive_risk(s_t) + P(pass | s_t) × pass_failure_expected(s_t)`.

The drive solver accumulates defender-specific loss probabilities under a shared remaining survival mass. The final fitted model uses additive penalized spline effects for carrier speed, signed boundary distance and individual interception angle in the log tackle rate. `BallDriveModel.pretrained()` loads its verified ten-match refit, also used by the frame API by default. The historical log-linear solver remains available by supplying its legacy coefficients explicitly. `lambda_control` belongs to pass control; the drive solver names its baseline `lambda_0` and its cause-specific intensity `lambda_def`. **own derivation** [@spearman2017]

The pass component uses a verified trajectory-MLE fit transferred to target pitch control, with exact-distance flight and mass-preserving exponential integration. `PassRiskModel.pretrained()` loads its parameters and isotonic calibration; `PRESS(..., pass_risk_model=model)` accepts it explicitly, and `compute_ppcf` uses it by default. The accepted cohort has 6,706 passes (964 failures), outfield passers and duration [0.5, 10] s. The final run41 artifact uses the same TTI for trajectory fitting, target control and receiver/flight timing: v_max=5.5 m/s, a=a_turn=7 m/s², preserving initial speed above v_max; the [vignette](vignettes/02_pass_probability/02_pass_probability.ipynb) explains flight, control, speed quantiles and model assumptions. [Pass refit provenance](docs/pass-risk-refit-provenance.json).

## Package API

`PRESS` accepts wide tracking data and optional event data. `get_state(frame_id)` extracts carrier, teammate, and defender positions and velocities. The core class computes pitch control/pass probability, ball-drive loss, transition choice, action selection, and the composed hazard. BCOS is an optional extra in `PRESS.extras.BCOS`; the current action-selection booster still consumes a scalar BCOS feature, supplied explicitly to `compute_action_selection_probability`. Result access for core outputs is available through `get_result(component, frame_id)` and component getters such as `get_ppcf_result` and `get_hazard_result`. **design choice**

Post-processing helpers in `PRESS.utils.possession` expose `hazard_to_band`, `possession_survival`, and `classify_possession_typology`. The core visualization package exposes `plot_frame`, `plot_ppcf`, and `plot_pi`, plus pitch views `plot_transition` and `plot_action_selection`; optional BCOS visualization is available through `plot_bcos` with a `PRESS.extras.BCOS` instance.

Use `get_possessions`, `get_possession`, `get_player_possessions`, `get_frames`, `get_frame` and `get_checkpoints` to select inputs. `PRESS.from_skillcorner` loads processed match files; `compute_frame` evaluates any usable frame, resolving annotated event context or accepting it explicitly. `compute_possession` returns a sampled hazard trajectory: its default includes canonical action starts and additional evaluation snapshots, labeled separately; `include_snapshots=False` selects canonical rows and `frame_ids` accepts custom frames. The frame evaluator is independent of that sampling. Initialization performs no model calculation.

See the [end-to-end match usage guide](usage.qmd) for loading processed tracking, selecting any usable frame, and computing its components. The [Python API reference](api.qmd) documents public methods, getters, plots, and possession utilities. The [action plan](roadmap.qmd) explains the package and portfolio structure.

The installed destination-choice ensemble is `PRESS/utils/data/pass_selection_model.json` + `pass_selection_model.ubj`, from the verified full-data flat refit in research experiment57. Its [provenance](docs/pass-selection-final-provenance.json) identifies weights, physics and calibration. The previous `transition_model.json` is retained for historical comparisons. The revised [action-selection benchmark](benchmark.qmd#action-selection-benchmark) is complete: 30 conceptual inputs, 16,918 decisions, pooled CE **0.568592**, null gain **17.602%** and quantile ECE **1.026%**. Action selection loads the verified full-data `action_selection_model.json` manifest and checksum-verified `.ubj`: 277 trees, 30 conceptual inputs, no scaling. [Refit provenance](docs/action-selection-refit-provenance.json) links the installed runtime and upstream model hashes; the held-out benchmark remains separate. The [example vignette](vignettes/04_action_selection/04_action_selection.ipynb) defines every feature and evaluates the installed model on a synthetic frame. The verified ball-drive refit is `PRESS/utils/data/ball_drive_model.json`, with [export provenance](docs/ball-drive-refit-provenance.json), and is the default frame-level carry model. These are learned parameters, not raw tracking records.

## Vignettes

Every vignette starts with the idea and a code-generated visual. All examples use deterministic toy inputs unless the vignette explicitly reads local data. Mathematical descriptions tagged **own derivation** or **design choice** are project formulations.

- [00 · Match, possession and frame workflow](vignettes/00_match_workflow/00_match_workflow.ipynb) — initialize one match, select a frame, compute and retrieve its components.
- [01 · Ball-drive risk](vignettes/01_ball_drive_risk/01_ball_drive_risk.ipynb) — geometry, carrier speed, boundary distance, and competing defender causes.
- [02 · Pass probability](vignettes/02_pass_probability/02_pass_probability.ipynb) — simplified flight path and Spearman-style arrival/control accumulation.
- [03 · Transition model](vignettes/03_transition_model/03_transition_model.ipynb) — destination choice across candidate receivers.
- [04 · Action selection](vignettes/04_action_selection/04_action_selection.ipynb) — pass-versus-carry likelihood, all 30 benchmark inputs, 4+4 option summaries, package pitch example and held-out calibration. Fractional possession age is excluded; completed whole carries are included.
- [05 · Composed hazard and bands](vignettes/05_composed_hazard_and_bands/05_composed_hazard_and_bands.ipynb) — branch mixture and half-life bands.
- [05a · Residual risk](vignettes/05a_residual_risk/05a_residual_risk.ipynb) — conditional terminal outcome, all 31 inputs, model use and feature importance.
- [05b · Composed h and xLoss benchmark](vignettes/05b_composed_risk_benchmark/05b_composed_risk_benchmark.ipynb) — distinct targets, held-out protocol, scores and pooled calibration.
- [06 · Possession and checkpoints](vignettes/06_possession_and_checkpoint_setup/06_possession_and_checkpoint_setup.ipynb) — team possession construction and the configurable default sampling of a trajectory.
- [07 · Survival curve](vignettes/07_survival_curve/07_survival_curve.ipynb) — retrospective cumulative possession difficulty and band trajectory.
- [08 · Possession typologies](vignettes/08_possession_typologies/08_possession_typologies.ipynb) — eight mutually exclusive sequence categories.
- [09 · League-level report](vignettes/09_league_level_report/09_league_level_report.ipynb) — league-level summaries and category case-study workflow.
- [Extra · Ball Carrier Open Space](vignettes/extras/bcos/10_bcos.ipynb) — concept, conditional-hazard mathematics, real-frame calculation and one-match spatial/temporal summaries; its scalar area is used by action selection.
- [Extra · Expected Threat](vignettes/extras/expected_threat/11_expected_threat.ipynb) — xT value iteration on a toy grid and the match-derived grid consumed by PRESS [@singh2018xt].



## Reproduce

Use Python 3.11 (the verification interpreter) or 3.12. Create the environment pinned to the versions used for the executable checks, then run the examples and notebooks, saving outputs and PNG/SVG figures:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -e '.[io,vignettes,dev]'
python scripts/check_environment.py --extras io,vignettes,dev
python -m examples.checkpoint
python -m pytest -q tests
python scripts/rebuild_vignettes.py
python scripts/check_docs.py
```

The notebooks use fixed random seeds and synthetic examples so that the build does not require raw tracking files. See [data setup](data/README.md) for local data preprocessing.

## Build the website

Install Quarto 1.10.19 and the pinned Python environment above, activate that environment, then run:

```bash
quarto render
```

Quarto renders the notebooks as pages in `_site/`, executing their code and including outputs. The GitHub Actions workflow deploys the rendered site to GitHub Pages when changes reach `main`. In repository settings, select **Settings → Pages → Build and deployment → GitHub Actions** for the first deployment. The intended site URL is <https://riccardobergamin.github.io/PRESS/>.

## Data

Raw SkillCorner tracking/event files and match-level derived tables are not included. The project uses the SkillCorner open-data repository, which describes an Australian A-League sample and requests attribution. Its MIT license file covers software and does not specify separate data redistribution terms; this repository documents that distinction and includes learned XGBoost and ball-drive parameter files, aggregate benchmark summaries and synthetic vignette examples. [@skillcorner-opendata]

## Performance

The final ball-drive development benchmark pools five held-out match pairs: 9,114 attempts, 456 losses (OBE-verified loss-ending possessions); cross-entropy **0.143483**, gain over the training-fold null **27.798%**, and ten-quantile-bin ECE **0.475%**. These ten matches also informed model-family selection, so they are not an untouched external evaluation. The bundled full-data refit is distinct from these OOF predictions. See the [benchmark](benchmark.qmd) for cohort, comparisons, calibration, limitations and a runnable fitted-model pitch example. The pass development benchmark gives **CE 0.131121, gain 68.19%, ECE 0.183%** with isotonic on 6,706 held-out passes. Both are pooled development OOF results, separate from the bundled full-data fits. Action selection and residual-risk evaluation are complete. The verified residual cohort gives **CE 0.419506, gain 11.793%, ECE 1.053%**; composed individual xLoss gives **CE 0.421925, gain 13.769%, ECE 0.930%** on its separate terminal target. The verified 118-tree native residual refit is installed. The [residual-risk vignette](vignettes/05a_residual_risk/05a_residual_risk.ipynb) explains the conditional model; a [separate composed-risk benchmark](vignettes/05b_composed_risk_benchmark/05b_composed_risk_benchmark.ipynb) evaluates h and xLoss. [Refit provenance](docs/residual-risk-refit-provenance.json).

## Repository map

- `PRESS/`: existing importable model package; retained in its current location.
- `vignettes/`: runnable component notebooks; BCOS is under `vignettes/extras/`.
- `usage.qmd`, `api.qmd`, `solvers.qmd`, `roadmap.qmd`: match workflow, public function reference, solver reference and action plan.
- `_quarto.yml`, `index.qmd`, `benchmark.qmd`, `data.qmd`: site navigation and pages.
- `.github/workflows/publish-site.yml`: GitHub Pages build and deployment on `main`.
- `scripts/rebuild_vignettes.py`: one-command notebook execution and figure export.
- `references/refs.bib`, `REFERENCES.md`: citation records and unresolved references.
- `data/README.md`: data setup notes.

## Citation

Cite a release using the metadata in [CITATION.cff](CITATION.cff), and include the repository version or commit used. **design choice**

The raw-to-base pipeline lives in the sibling `skillcorner-analysis/scripts/pipeline`; inputs and versioned outputs live in `opendata`. The outfield-carrier v5 base regeneration and training-state validation are complete; trajectories remain validated in the historical v4 root and have not been regenerated for the new filtered indices. [pipeline.qmd](pipeline.qmd) records the actual layout, native-geometry conventions, exported training bases and the accepted segmentation convention and remaining model-feature/benchmark review. `PRESS.from_skillcorner` accepts either CSV or parquet action tables.

The carry solver uses exponential Euler to preserve competing-risk probability mass; explicit `integration_method="euler"` reproduces the previous update. Numerical method and step are separate from the tackle-rate functional form. The current versioned base root is `2026-10-15_training_v5_outfield`, with the accepted v2 segmentation, outfield carriers and no shot-ending possessions in training. All goalkeeper tracking, intervals and inference checkpoints remain available. The historical reproduction is frozen separately in v4.

Pass destination choice is finalized as a flat XGBoost categorical model. The [benchmark](benchmark.qmd#pass-selection-benchmark) records the five-configuration capacity screening and both final flat/CNL runs (pooled calibrated McFadden 34.322% / 34.516%). The verified full-ten-match flat refit is installed: 1,444 trees and temperature 1.013295. `PassSelectionModel.pretrained()` loads it, and `PRESS` uses it for destination choice by default. Its saved xT, granular context and corrected carrier/destination ahead-count schema travel with the weights. The CNL refit is retained only in research history. Historical transition weights remain archived; the benchmark continues to use held-out predictions, not full-refit scores.


`ResidualRiskModel.pretrained()` loads terminal individual-loss continuation after a
successful1s carry. `compose_individual_xloss` adds it to the local hazard as
`h + P(carry)*(1-drive_loss)*residual_loss`; completed passes terminate the
individual sequence. The [main example](vignettes/05a_residual_risk/05a_residual_risk.ipynb)
uses installed weights and explains all31inputs. This optional terminal forecast
does not change the pressure hazard or claim to reproduce proprietary xLoss.
