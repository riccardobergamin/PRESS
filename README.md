# PRESS

**Possession Risk Estimate from Spatio-temporal State** is a work-in-progress, physics-informed framework for estimating the discrete-time risk that a football team loses possession at its next action. The code combines physics-based pass and carry components with learned action and destination choices. [@spearman2017; @fernandez2019] **own derivation**

## Objectives and motivation

This project is a portfolio research project to learn the main ideas behind expected possession value (EPV), work with player tracking data, and explore a physics-based formulation of ball-drive likelihood. The goal is to package the existing research code into reusable components with reproducible examples and results. [@fernandez2019] **own derivation**

The motivating question is: given the current game state and the player in possession, how difficult is it for the team to retain the ball? Pressure can constrain a carrier's options, raise interception risk, and affect action choice. PRESS adapts building blocks from pitch control and EPV research to express that question as a team-possession hazard. [@spearman2017; @fernandez2019; @bekkers2025; @skillcorner-pressure] **own derivation**

The model style is hybrid: pass and carry outcomes use physics-inspired motion and arrival probabilities; action selection and pass destination use tree models. The possession unit is a team sequence of player possessions. A completed pass is a nonterminal transition; loss is terminal. The discrete-time hazard is evaluated at action checkpoints attributable to the current carrier. [@singer2003] **own derivation**

## Model overview

![PRESS model overview](figures/model_overview.svg)

At each checkpoint, PRESS composes the drive and pass branches as

`h_t = P(drive | s_t) × drive_risk(s_t) + P(pass | s_t) × pass_failure_expected(s_t)`.

The drive solver accumulates defender-specific loss probabilities under a shared remaining survival mass. In the implementation, the defensive intensity uses `lambda_0 * exp(beta_v * carrier_speed + beta_B * minimum_distance_to_any_pitch_boundary)`. `lambda_control` belongs to pass control; the drive solver names its baseline `lambda_0` and its cause-specific intensity `lambda_def`. **own derivation** [@spearman2017]

## Package API

`PRESS` accepts wide tracking data and optional event data. `get_state(frame_id)` extracts carrier, teammate, and defender positions and velocities. The component methods compute BCOS, pitch control/pass probability, ball-drive loss, transition choice, action selection, and the composed hazard. Result access is available through `get_result(component, frame_id)` and the component getters such as `get_bcos_result`, `get_ppcf_result`, and `get_hazard_result`. **design choice**

The visualization package exposes `plot_frame`, `plot_bcos`, `plot_ppcf`, and `plot_pi`. The plotting functions consume a `PRESS` instance and a frame identifier.

Learned transition and action-selection inference requires the JSON model files under `PRESS/utils/data/`; those local model artifacts are not included in the reproducible toy build.

## Vignettes

Every vignette starts with the idea and a code-generated visual. All examples use deterministic toy inputs unless the vignette explicitly reads local data. Mathematical descriptions tagged **own derivation** or **design choice** are project formulations.

- [01 · Ball-drive risk](vignettes/01_ball_drive_risk/01_ball_drive_risk.ipynb) — geometry, carrier speed, boundary distance, and competing defender causes.
- [02 · Pass probability](vignettes/02_pass_probability/02_pass_probability.ipynb) — simplified flight path and Spearman-style arrival/control accumulation.
- [03 · Transition model](vignettes/03_transition_model/03_transition_model.ipynb) — destination choice across candidate receivers.
- [04 · Action selection](vignettes/04_action_selection/04_action_selection.ipynb) — drive-versus-pass probability; elapsed time is excluded from the feature list.
- [05 · Composed hazard and bands](vignettes/05_composed_hazard_and_bands/05_composed_hazard_and_bands.ipynb) — branch mixture and half-life bands.
- [06 · Possession and checkpoints](vignettes/06_possession_and_checkpoint_setup/06_possession_and_checkpoint_setup.ipynb) — team possession construction and one-second carry checkpoints.
- [07 · Survival curve](vignettes/07_survival_curve/07_survival_curve.ipynb) — retrospective cumulative possession difficulty and band trajectory.
- [08 · Possession typologies](vignettes/08_possession_typologies/08_possession_typologies.ipynb) — eight mutually exclusive sequence categories.
- [09 · League-level report](vignettes/09_league_level_report/09_league_level_report.ipynb) — league-level summaries and category case-study workflow.
- [10 · Ball Carrier Open Space](vignettes/10_bcos/10_bcos.ipynb) — open-space probability field.

## Reproduce

Create the pinned environment and execute all notebooks, saving executed notebook outputs and PNG/SVG figures:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[vignettes]'
python scripts/rebuild_vignettes.py
```

The notebooks use fixed random seeds and synthetic examples so that the build does not require raw tracking files. See [data setup](data/README.md) for local data preprocessing.

## Data

Raw SkillCorner files are not included. The project uses the SkillCorner open-data repository, which describes an Australian A-League sample and requests attribution. The repository's MIT license file is for software; it does not state a separate data redistribution license, so this repository includes neither raw data nor real-match derived artifacts. [@skillcorner-opendata]

## Performance

to be added after the final benchmark

## Repository map

- `PRESS/`: existing importable model package; retained in its current location.
- `vignettes/`: runnable component notebooks.
- `scripts/rebuild_vignettes.py`: one-command notebook execution and figure export.
- `references/refs.bib`, `REFERENCES.md`: citation records and unresolved references.
- `data/README.md`: data setup notes.

## Citation

Cite a release using the metadata in [CITATION.cff](CITATION.cff), and include the repository version or commit used. **design choice**
