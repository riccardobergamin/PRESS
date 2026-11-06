# Archived package implementations

These sources are retained for research history and are not imported or installed with PRESS. They are not the canonical segmentation/training pipeline.

- `core/old`, `utils/old`: superseded event segmentation and physics wrappers.
- `extras/bcos_legacy.py`, `model/solver_bcos.py`, `physics/hazard_rates_bcos.py`: BCOS implementation superseded by the verified fine-path solver (`PRESS.model.bcos_fine`).
- `utils/segmenters.py`: unused event-based segmentation, superseded by explicit canonical actions/intervals supplied to PRESS.
- `utils/estimators.py`: unused exploratory speed summaries; pretrained physics settings remain in model manifests.

`model/solver_intensity.py` preserves the superseded log-linear proportional-hazard ball-drive solver. It is not exported or called by PRESS; the active solver is `PRESS.model.BallDriveModel`. Its historical tests are retained under `tests/`. The fitted spline model and its weights are unchanged.

`utils/possession_typologies.py` preserves the unused eight-label trajectory classifier; the current three report patterns are defined independently in the report. Four-band and survival utilities remain active.

The historical choice-model paths are also archived:

- `utils/pass_selection.py` and `utils/data/pass_selection_legacy.json`: the initial 13-feature destination builder and booster.
- `utils/action_selection.py`: the initial grouped-category action builder, loaders and native-reader bridge.
- `model/pass_selection_with_forward_bypass.py`: the reader before removal of the superseded `forward_bypass_v1` feature contract. Active pass selection requires `ahead_of_carrier_and_target_v1`.
- `core/computation_with_legacy_choices.py` and `core/navigation_with_legacy_choice_filter.py`: the orchestration before removal of historical fallback paths and model-dependent context filtering.

The active package uses the final fitted classes directly. Missing or unsupported model artifacts raise an error rather than selecting a historical implementation. The production weights and final feature calculations are unchanged.
