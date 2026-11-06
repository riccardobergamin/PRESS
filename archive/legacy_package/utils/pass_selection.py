"""
PRESS.utils.pass_selection

Feature extraction and model loading for the trained XGBoost
pass selection model.

The legacy feature contract remains available for older artifacts.
Current destination choice uses PRESS.model.PassSelectionModel and
PRESS.compute_pass_selection().
"""

from pathlib import Path

import numpy as np
import xgboost as xgb

# ============================================================
# MODEL PATH / LOADING
#
# The trained booster is shipped as a package data file, loaded via
# a path relative to this module's location -- works regardless of
# the current working directory.
#
# Expected layout: PRESS/utils/data/pass_selection_legacy.json
# Adjust if the actual data directory differs.
# ============================================================

_LEGACY_PASS_SELECTION_MODEL_PATH = (
    Path(__file__).resolve().parent / "data" / "pass_selection_legacy.json"
)

PASS_SELECTION_FEATURE_COLS = [
    "pass_distance",
    "pass_angle",
    "passer_speed",
    "target_pitch_control",
    "target_xt_team",
    "target_xt_opp",
    "opp_pressure_line_index",
    "team_pressure_line_index",
    "ball_drive_prob",
    "xt_team_bc",
    "xt_opp_bc",
    "opp_pressure_index_bc",
    "team_pressure_index_bc",
]

_pass_selection_booster = None


def get_pass_selection_booster():
    """
    Load the trained pass selection model booster once and cache it at
    module level (shared across all PRESS instances).
    """

    global _pass_selection_booster

    if _pass_selection_booster is None:
        _pass_selection_booster = xgb.Booster()
        _pass_selection_booster.load_model(str(_LEGACY_PASS_SELECTION_MODEL_PATH))

    return _pass_selection_booster

# ============================================================
# PASS GEOMETRY
#
# Not available in PRESS.utils.state -- distance and signed angle
# between v_bc and the vector towards each candidate destination,
# same formula used at training time.
# ============================================================


def compute_pass_geometry(x_bc, v_bc, targets):
    """Return distance and signed angle from carrier velocity to each target."""
    x_bc = np.asarray(x_bc, dtype=float)
    v_bc = np.asarray(v_bc, dtype=float)
    targets = np.asarray(targets, dtype=float)

    dx = targets[:, 0] - x_bc[0]
    dy = targets[:, 1] - x_bc[1]

    distance = np.hypot(dx, dy)

    speed = np.linalg.norm(v_bc)

    if speed < 1e-8:
        angle = np.full(len(targets), np.nan)
    else:
        angle = np.arctan2(
            v_bc[0] * dy - v_bc[1] * dx,
            v_bc[0] * dx + v_bc[1] * dy,
        )

    return distance, angle

# ============================================================
# FEATURE MATRIX
#
# Builds the 13-column feature matrix (in PASS_SELECTION_FEATURE_COLS
# order) for a single frame's full candidate grid, given the state
# and already-computed PPCF/ball-drive-loss results.
# ============================================================


def build_pass_selection_features(
    x_bc,
    v_bc,
    r_defs,
    r_atts,
    attacking_side,
    targets,
    target_pitch_control,
    ball_drive_prob,
    xT_grid,
    *, pitch_length=105.0, pitch_width=68.0,
):
    """
    x_bc, v_bc, r_defs, r_atts, attacking_side : from PRESS.get_state()
    targets                                     : PPCF grid (N, 2)
    target_pitch_control                        : (N,) from compute_ppcf()
    ball_drive_prob                             : scalar, from
                                                    compute_ball_drive_risk()
    xT_grid                                     : xT grid array

    Returns X, shape (N, 13), columns in PASS_SELECTION_FEATURE_COLS order.
    """

    from PRESS.utils.state import (
        get_xt_values,
        compute_dynamic_pressure_lines,
        get_target_pressure_line_index,
    )

    def_side = (
        "right_to_left" if attacking_side == "left_to_right"
        else "left_to_right"
    )

    # --------------------------------------------------------
    # xT (destination + passer position)
    # --------------------------------------------------------

    xt_team_target, xt_opp_target = get_xt_values(
        targets, attacking_side, xT_grid, pitch_length=pitch_length, pitch_width=pitch_width,
    )

    xt_team_bc, xt_opp_bc = get_xt_values(
        x_bc[None, :], attacking_side, xT_grid, pitch_length=pitch_length, pitch_width=pitch_width,
    )

    # --------------------------------------------------------
    # Dynamic pressure lines (both sides)
    # --------------------------------------------------------

    opp_pressure_lines = compute_dynamic_pressure_lines(
        r_defs=r_defs,
        attacking_side=attacking_side,
        n_lines=3,
    )

    team_pressure_lines = compute_dynamic_pressure_lines(
        r_defs=r_atts,
        attacking_side=def_side,
        n_lines=3,
    )

    opp_pressure_line_index = get_target_pressure_line_index(
        targets=targets,
        pressure_lines=opp_pressure_lines,
        attacking_side=attacking_side,
    )

    team_pressure_line_index = get_target_pressure_line_index(
        targets=targets,
        pressure_lines=team_pressure_lines,
        attacking_side=def_side,
    )

    opp_pressure_index_bc = get_target_pressure_line_index(
        targets=x_bc[None, :],
        pressure_lines=opp_pressure_lines,
        attacking_side=attacking_side,
    )

    team_pressure_index_bc = get_target_pressure_line_index(
        targets=x_bc[None, :],
        pressure_lines=team_pressure_lines,
        attacking_side=def_side,
    )

    # --------------------------------------------------------
    # Pass geometry
    # --------------------------------------------------------

    pass_distance, pass_angle = compute_pass_geometry(x_bc, v_bc, targets)

    passer_speed = float(np.linalg.norm(v_bc))

    # --------------------------------------------------------
    # Assemble
    # --------------------------------------------------------

    n_destinations = len(targets)

    feature_lookup = {
        "pass_distance": pass_distance,
        "pass_angle": pass_angle,
        "passer_speed": np.full(n_destinations, passer_speed),
        "target_pitch_control": np.asarray(target_pitch_control),
        "target_xt_team": xt_team_target,
        "target_xt_opp": xt_opp_target,
        "opp_pressure_line_index": opp_pressure_line_index,
        "team_pressure_line_index": team_pressure_line_index,
        "ball_drive_prob": np.full(n_destinations, float(ball_drive_prob)),
        "xt_team_bc": np.full(n_destinations, float(xt_team_bc[0])),
        "xt_opp_bc": np.full(n_destinations, float(xt_opp_bc[0])),
        "opp_pressure_index_bc": np.full(
            n_destinations, float(opp_pressure_index_bc[0])
        ),
        "team_pressure_index_bc": np.full(
            n_destinations, float(team_pressure_index_bc[0])
        ),
    }

    return np.column_stack(
        [feature_lookup[col] for col in PASS_SELECTION_FEATURE_COLS]
    )

# ============================================================
# PREDICTION
# ============================================================


def predict_pass_selection_probabilities(X):
    """
    Score the feature matrix with the trained booster and return
    the softmax choice probabilities over this single choice set
    (the full candidate grid for one frame).
    """

    booster = get_pass_selection_booster()

    scores = booster.predict(xgb.DMatrix(X))

    scores = scores - scores.max()
    exp_scores = np.exp(scores)

    return exp_scores / exp_scores.sum()
