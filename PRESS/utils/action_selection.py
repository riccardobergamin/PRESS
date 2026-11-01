"""
PRESS.utils.action_selection

Feature extraction and model loading for the trained XGBoost action
selection (pass vs ball_drive) model.

Mirrors PRESS.utils.transition in structure. Kept separate from the
PRESS class itself.

Expected layout (to deploy, copy the two artifacts produced by
refit_action_selection_production.py):
    PRESS/utils/data/action_selection_model.json
    PRESS/utils/data/action_selection_feature_columns.json
"""

from pathlib import Path
import json

import numpy as np
import xgboost as xgb


# ============================================================
# MODEL / FEATURE-COLUMNS PATHS
# ============================================================

_ACTION_SELECTION_MODEL_PATH = (
    Path(__file__).resolve().parent / "data" / "action_selection_model.json"
)

_ACTION_SELECTION_FEATURE_COLUMNS_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "action_selection_feature_columns.json"
)

ACTION_SELECTION_PHYSICAL_FEATURES = [
    "bcos",
    "rppcf",
    "ball_drive_prob",
    "xT_team_pass",
    "xT_opp_pass",
    "opp_pressure_line_index_pass",
    "team_pressure_line_index_pass",
    "team_pressure_line_index_exposure",
    "xT_team_drive",
    "xT_opp_drive",
    "opp_pressure_line_index_drive",
    "team_pressure_line_index_drive",
]

ACTION_SELECTION_CATEGORICAL_FEATURES = [
    "start_type_std",
    "previous_action_std",
]

# action selection target: pass=1, ball_drive=0 (same convention as
# train_action_selection.py) -- booster.predict() therefore returns
# P(pass).
ACTION_SELECTION_TARGET_MAP = {"pass": 1, "ball_drive": 0}

_action_selection_booster = None
_action_selection_feature_cols = None


def get_action_selection_booster():
    """
    Load the trained action selection booster + its exact one-hot
    feature-column layout once, cache at module level (shared across
    all PRESS instances) -- same pattern as get_transition_booster().
    """

    global _action_selection_booster, _action_selection_feature_cols

    modern=load_current_action_selection_model()
    if modern is not None:return modern.booster,modern.feature_names

    if _action_selection_booster is None:
        _action_selection_booster = xgb.Booster()
        _action_selection_booster.load_model(
            str(_ACTION_SELECTION_MODEL_PATH)
        )

        with open(_ACTION_SELECTION_FEATURE_COLUMNS_PATH) as f:
            meta = json.load(f)
        _action_selection_feature_cols = meta["feature_cols_full_order"]

    return _action_selection_booster, _action_selection_feature_cols


# ============================================================
# FEATURE VECTOR
#
# Unlike the transition model (one row per candidate destination,
# scored with a grouped softmax), action selection scores a SINGLE
# instantaneous state (frame_id) with one sigmoid -- returns a
# (1, n_features) row, not a per-destination matrix.
# ============================================================

def build_action_selection_features(
    x_bc,
    v_bc,
    r_defs,
    r_atts,
    attacking_side,
    targets,
    transition_probabilities,
    pass_success_probability,
    bcos,
    ball_drive_loss,
    xT_grid,
    start_type,
    previous_action,
    *, pitch_length=105.0, pitch_width=68.0, match_clock_min=np.nan,
    score_difference_prefix=np.nan,completed_canonical_drives=np.nan,
):
    """
    x_bc, v_bc, r_defs, r_atts, attacking_side  : from PRESS.get_state()
    targets                                      : PPCF grid (N, 2),
                                                     self.results["PPCF"][frame_id]["grid"]
    transition_probabilities                     : (N,) softmax choice
                                                     probabilities,
                                                     self.results["PPCF"][frame_id]["Transition probabilities"]
    pass_success_probability                     : (N,) calibrated pass success,
                                                     self.results["PPCF"][frame_id]["Pass probability"]
    bcos                                         : scalar supplied from
                                                     PRESS.extras.BCOS;
                                                     feature retained from
                                                     the trained model
    ball_drive_loss                              : scalar,
                                                     self.results["PI"][frame_id]["Ball drive loss probability"]
                                                     (NOTE: this is P(loss). The model feature is
                                                     P(success) = 1 - ball_drive_loss -- inverted here,
                                                     matching the offline feature-engineering script.
                                                     The destination-choice feature uses loss directly;
                                                     the two saved model schemas have different aliases.)
    xT_grid                                      : xT grid array
    start_type, previous_action                  : raw category strings for this action
                                                     (e.g. "reception_open_play", "ball_drive"),
                                                     matched against the "start_type_std_*" /
                                                     "previous_action_std_*" one-hot columns saved
                                                     alongside the model.

    Returns X, shape (1, n_features), columns in the exact order saved
    in action_selection_feature_columns.json.
    """

    modern=load_current_action_selection_model()
    if modern is not None:
        return modern.build_features(x_bc=x_bc,v_bc=v_bc,r_defs=r_defs,r_atts=r_atts,
            attacking_side=attacking_side,targets=targets,transition_probabilities=transition_probabilities,
            pass_success_probability=pass_success_probability,bcos=bcos,ball_drive_loss=ball_drive_loss,
            start_type_raw=start_type,previous_action_raw=previous_action,match_clock_min=match_clock_min,
            score_difference_prefix=score_difference_prefix,completed_canonical_drives=completed_canonical_drives,
            pitch_length=pitch_length,pitch_width=pitch_width)

    from PRESS.utils.state import (
        get_xt_values,
        compute_dynamic_pressure_lines,
        get_target_pressure_line_index,
    )

    tr = np.asarray(transition_probabilities)
    pass_prob = np.asarray(pass_success_probability)
    targets = np.asarray(targets, dtype=float)
    x_bc = np.asarray(x_bc, dtype=float)
    v_bc = np.asarray(v_bc, dtype=float)

    def_side = (
        "right_to_left" if attacking_side == "left_to_right"
        else "left_to_right"
    )

    # --------------------------------------------------------
    # ball_drive_prob -- P(success), inverted from the stored
    # "Ball drive loss probability" (P(loss)). See docstring note.
    # --------------------------------------------------------

    ball_drive_prob = 1.0 - float(np.asarray(ball_drive_loss))

    # --------------------------------------------------------
    # rppcf (per-cell, then summed once) = tr * pass_prob
    # --------------------------------------------------------

    rppcf_per_cell = tr * pass_prob
    rppcf = float(rppcf_per_cell.sum())

    # --------------------------------------------------------
    # xT / pressure lines at destinations
    # --------------------------------------------------------

    xT_team, xT_opp = get_xt_values(targets, attacking_side, xT_grid,
                                 pitch_length=pitch_length, pitch_width=pitch_width)

    pressure_lines = compute_dynamic_pressure_lines(
        r_defs=r_defs, attacking_side=attacking_side, n_lines=3,
    )
    team_pressure_lines = compute_dynamic_pressure_lines(
        r_defs=r_atts, attacking_side=def_side, n_lines=3,
    )

    opp_pressure_line_index = get_target_pressure_line_index(
        targets=targets, pressure_lines=pressure_lines,
        attacking_side=attacking_side,
    )
    team_pressure_line_index = get_target_pressure_line_index(
        targets=targets, pressure_lines=team_pressure_lines,
        attacking_side=def_side,
    )

    # --------------------------------------------------------
    # PASS metrics (success-weighted: rppcf = tr*pass_prob;
    # exposure/opp metrics: fail-weighted, tr*(1-pass_prob))
    # --------------------------------------------------------

    xT_team_pass = float(np.sum(rppcf_per_cell * xT_team))

    xT_opp_pass = float(
        np.sum(tr * (1.0 - pass_prob) * xT_opp)
    )

    opp_pressure_line_index_pass = float(
        np.sum(rppcf_per_cell * opp_pressure_line_index)
    )

    team_pressure_line_index_pass = float(
        np.sum(rppcf_per_cell * team_pressure_line_index)
    )

    team_pressure_line_index_exposure = float(
        np.sum(tr * (1.0 - pass_prob) * team_pressure_line_index)
    )

    # --------------------------------------------------------
    # DRIVE metrics -- evaluated at the 1s-extrapolated ball
    # carrier position (x_bc + v_bc * 1.0), NOT at x_bc itself
    # (that's what the transition model's "_bc" features use)
    # --------------------------------------------------------

    x_bc_exp = x_bc + v_bc * 1.0

    xT_team_drive, xT_opp_drive = get_xt_values(
        x_bc_exp[None, :], attacking_side, xT_grid, pitch_length=pitch_length, pitch_width=pitch_width,
    )
    xT_team_drive = float(np.asarray(xT_team_drive).squeeze())
    xT_opp_drive = float(np.asarray(xT_opp_drive).squeeze())

    opp_pressure_line_index_drive = float(
        np.asarray(
            get_target_pressure_line_index(
                targets=x_bc_exp[None, :], pressure_lines=pressure_lines,
                attacking_side=attacking_side,
            )
        ).squeeze()
    )

    team_pressure_line_index_drive = float(
        np.asarray(
            get_target_pressure_line_index(
                targets=x_bc_exp[None, :],
                pressure_lines=team_pressure_lines,
                attacking_side=def_side,
            )
        ).squeeze()
    )

    # --------------------------------------------------------
    # Assemble physical features
    # --------------------------------------------------------

    physical_values = {
        "bcos": float(np.asarray(bcos)),
        "rppcf": rppcf,
        "ball_drive_prob": ball_drive_prob,
        "xT_team_pass": xT_team_pass,
        "xT_opp_pass": xT_opp_pass,
        "opp_pressure_line_index_pass": opp_pressure_line_index_pass,
        "team_pressure_line_index_pass": team_pressure_line_index_pass,
        "team_pressure_line_index_exposure": team_pressure_line_index_exposure,
        "xT_team_drive": xT_team_drive,
        "xT_opp_drive": xT_opp_drive,
        "opp_pressure_line_index_drive": opp_pressure_line_index_drive,
        "team_pressure_line_index_drive": team_pressure_line_index_drive,
    }

    # --------------------------------------------------------
    # Assemble one-hot categorical features against the exact
    # column layout saved with the production model (never
    # inferred/hardcoded here, to avoid a silent train/inference
    # mismatch if the category set ever changes).
    # --------------------------------------------------------

    _, feature_cols = get_action_selection_booster()

    start_type_col = f"start_type_std_{start_type}"
    previous_action_col = f"previous_action_std_{previous_action}"

    if start_type_col not in feature_cols:
        raise ValueError(
            f"Unknown start_type '{start_type}' -- expected one of the "
            f"start_type_std_* columns saved in "
            f"action_selection_feature_columns.json, got column name "
            f"'{start_type_col}' which is not in the trained feature set."
        )

    if previous_action_col not in feature_cols:
        raise ValueError(
            f"Unknown previous_action '{previous_action}' -- expected one "
            f"of the previous_action_std_* columns saved in "
            f"action_selection_feature_columns.json, got column name "
            f"'{previous_action_col}' which is not in the trained feature "
            f"set."
        )

    row = []
    for col in feature_cols:
        if col in physical_values:
            row.append(physical_values[col])
        elif col == start_type_col or col == previous_action_col:
            row.append(1.0)
        else:
            # any other one-hot column (a category not observed in
            # this particular action) is 0
            row.append(0.0)

    return np.asarray(row, dtype=float).reshape(1, -1)


# ============================================================
# PREDICTION
# ============================================================

def predict_action_selection_probability(X):
    """
    Score the single-row feature vector with the trained booster.

    Returns a scalar float: P(pass) (target_map: pass=1, ball_drive=0,
    same convention as train_action_selection.py).
    """

    booster, feature_cols = get_action_selection_booster()

    scores = booster.predict(xgb.DMatrix(X, feature_names=feature_cols))

    return float(scores[0])


from functools import lru_cache
@lru_cache(maxsize=1)
def load_current_action_selection_model():
    """Return the versioned refit reader; None preserves historical artifacts."""
    artifact=json.loads(_ACTION_SELECTION_MODEL_PATH.read_text())
    if artifact.get('model_type')!='binary_action_choice':return None
    from PRESS.model.action_selection import ActionSelectionModel
    return ActionSelectionModel.from_json(_ACTION_SELECTION_MODEL_PATH)
