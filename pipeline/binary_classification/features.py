"""Features shared by action selection and residual risk.

Read build_features for the mathematical summaries of a single state.
prepare_action_features reads actual frames and calculates these summaries.
prepare_residual_features reuses them and adds cross-fitted P(carry).
The accepted state/label tables are inputs: no new cohort filter is applied.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xgboost as xgb

from PRESS.model import BallDriveModel, PassRiskModel, PassSelectionModel
from PRESS.model.bcos_fine import compute_bcos_state
from PRESS.model.action_features import ahead, line_values, xt
from PRESS.utils.state import compute_dynamic_pressure_lines

from pipeline.frame_states import state_at_frame
from pipeline.pass_selection.features import (
    candidate_grid, make_arrival_calculator, pitch_control_by_player,
)
from pipeline.binary_classification.training import (
    KEYS, NUMERIC_FEATURES, fit_category_vocabulary, build_xgboost_input,
    train_model, write_json,
)


# ============================================================
# 1. SINGLE-STATE MATHEMATICAL FEATURE DEFINITIONS
# ============================================================

def build_features(state, checkpoint, targets, pass_choice, pass_completion,
                   drive_loss, bcos_area, xt_grid, own_lines, opponent_lines):
    """One current state -> the 28 numerical action-selection predictors.

    targets: (C,2), pass_choice q: (C,), calibrated completion p: (C,).
    Players: (22,2), ordered ten teammates / carrier / eleven opponents.
    Pass outcome weights are q*p and q*(1-p). They keep their branch mass:
    SUM_j q_j*p_j*a_j is not divided by total success probability.
    checkpoint contains observed start-time context, never a future outcome.
    """
    bc, velocity = state["x_bc"], state["v_bc"]
    teammates, opponents = state["r_atts"], state["r_defs"]
    direction = 1 if state["attacking_side"] == "left_to_right" else -1
    length, width = state["pitch_length"], state["pitch_width"]
    q, p = np.asarray(pass_choice, float), np.asarray(pass_completion, float)
    success_weight = q * p
    failure_weight = q * (1 - p)

    # Carrier's projected position after 1 s, with current velocity held fixed.
    projected_carrier = (bc + velocity)[None]
    speed = float(np.linalg.norm(velocity))
    motion_angle = np.nan
    if speed >= 1e-8:
        motion_angle = float(np.arctan2(direction * velocity[1], direction * velocity[0]))

    # Three ordered line centres: own-team index has opposite orientation,
    # while both signed distances use the carrier team's attack direction.
    own_index, _ = line_values(projected_carrier, own_lines, -direction)
    opponent_index, opponent_distance = line_values(projected_carrier, opponent_lines, direction)
    own_ordered = np.sort(-direction * np.asarray(own_lines, float))
    own_distance = np.nan
    if np.isfinite(own_index[0]):
        nearest_own_line = -direction * own_ordered[int(own_index[0]) - 1]
        own_distance = direction * (nearest_own_line - projected_carrier[0, 0])

    # Each target's spatial descriptors, before any choice/outcome averaging.
    own_target_line, _ = line_values(targets, own_lines, -direction)
    opponent_target_line, _ = line_values(targets, opponent_lines, direction)
    own_target_xt = xt(targets, direction, length, width, xt_grid)
    opponent_target_xt = xt(targets, -direction, length, width, xt_grid)
    teammates_ahead = ahead(teammates, targets, direction)
    opponents_ahead = ahead(opponents, targets, direction)

    # Keep named columns next to their definitions, rather than an unnamed
    # list zipped to column names. The final matrix order is fixed separately.
    return dict(
        # Observed match context and the number of ALREADY completed carries.
        match_clock_min=checkpoint.match_clock_min,
        score_difference_prefix=checkpoint.score_difference_prefix,
        completed_canonical_drives=checkpoint.completed_canonical_drives,

        # Current carrier motion, one-second retention, and open space.
        carrier_x_attack=direction * bc[0],
        carrier_y_attack=direction * bc[1],
        carrier_speed=speed,
        carrier_velocity_attack_angle=motion_angle,
        drive_success_1s=1 - drive_loss,
        bcos=bcos_area,

        # Spatial/tactical context at the ONE-SECOND carrier projection.
        xt_team_drive_1s=xt(projected_carrier, direction, length, width, xt_grid)[0],
        xt_opp_drive_1s=xt(projected_carrier, -direction, length, width, xt_grid)[0],
        team_line_index_drive_1s=own_index[0],
        opp_line_index_drive_1s=opponent_index[0],
        team_line_distance_drive_1s=own_distance,
        opp_line_distance_drive_1s=opponent_distance[0],
        teammates_ahead_bc=ahead(teammates, bc[None], direction)[0],
        opponents_ahead_bc=ahead(opponents, bc[None], direction)[0],

        # Hypothetical pass outcomes, integrated over the choice distribution.
        pass_success_mean=success_weight.sum(),
        xt_team_pass_success=success_weight @ own_target_xt,
        xt_opp_pass_failure=failure_weight @ opponent_target_xt,
        team_line_pass_success=success_weight @ own_target_line,
        team_line_pass_failure=failure_weight @ own_target_line,
        opp_line_pass_success=success_weight @ opponent_target_line,
        opp_line_pass_failure=failure_weight @ opponent_target_line,
        teammates_ahead_pass_success=success_weight @ teammates_ahead,
        teammates_ahead_pass_failure=failure_weight @ teammates_ahead,
        opponents_ahead_pass_success=success_weight @ opponents_ahead,
        opponents_ahead_pass_failure=failure_weight @ opponents_ahead,
    )


def calculate_bcos(state, drive_model, settings):
    """BCOS with this split's physical ball-drive s and the final TTI routine.

    Uniform carrier cap=6.6 m/s; defender cap=7 m/s; accelerations=7 m/s².
    Metre area grid, 0.03125 m path step, dt=.04 s, radius=40 m, horizon=15 s.
    This calculates the actual feature; it does not substitute a coarse map.
    """
    physics = drive_model.to_dict()["physics"]
    configuration = [np.exp(drive_model.theta[3]), 6.6, physics["v_max"],
                     7., physics["a_linear"], 7., physics["a_turn"]]
    field = compute_bcos_state(
        state["p_pos"], state["p_vel"], configuration=configuration,
        pitch_length=state["pitch_length"], pitch_width=state["pitch_width"],
        **settings,
    )
    return float(field["BCOS area"])


# ============================================================
# 2. REAL FRAMES -> SPLIT-SPECIFIC PHYSICS -> ACTION FEATURES
# ============================================================

def prepare_action_features(data_root, tracking_root, output, split_name, matches=None):
    """Calculate one feature table per match, once for each outer split.

    Fold upstream weights/xT come from the same six fitting matches.
    Production upstream weights/xT come from all ten. No branch uses the
    observed destination or outcome as a predictor. Targets span the pitch.
    """
    jax.config.update("jax_enable_x64", True)
    data_root, output = Path(data_root), Path(output)
    protocol = json.loads((data_root / "protocol.json").read_text())
    rows = pd.read_parquet(data_root / "action_rows.parquet")
    rows["match_id"] = rows.match_id.astype(str)
    if matches is not None:
        rows = rows.loc[rows.match_id.isin(matches)]
    upstream = data_root / "upstream" / split_name
    drive_model = BallDriveModel.from_json(upstream / "ball_drive_model.json")
    pass_model = PassRiskModel.from_json(upstream / "pass_risk_model.json")
    choice_model = PassSelectionModel.from_json(upstream / "pass_selection_model.json")
    choice_model.booster.set_param({"nthread": 1})
    calculate_arrivals = make_arrival_calculator(pass_model.physics)
    log_parameters = jnp.log(jnp.array([pass_model.lambda_control, pass_model.s]))
    folder = output / "action_selection" / "features" / split_name
    folder.mkdir(parents=True, exist_ok=True)

    for match_id, checkpoints in rows.groupby("match_id", sort=True):
        target_file = folder / f"{match_id}.parquet"
        if target_file.exists():
            raise FileExistsError(target_file)
        tracking = pd.read_parquet(Path(tracking_root) / "matches" / match_id / f"{match_id}_tracking.parquet")
        records = []
        for checkpoint in checkpoints.itertuples(index=False):
            # A. Same getter and authoritative matched carrier as other tasks.
            state = state_at_frame(
                tracking, checkpoint.frame_start,
                checkpoint.player_id, checkpoint.attacking_side,
            )
            targets = candidate_grid(checkpoint.pitch_length, checkpoint.pitch_width)
            own_direction = (
                "right_to_left" if state["attacking_side"] == "left_to_right"
                else "left_to_right"
            )
            own_lines = compute_dynamic_pressure_lines(state["r_atts"], own_direction)
            opponent_lines = compute_dynamic_pressure_lines(state["r_defs"], state["attacking_side"])

            # B. RAW control enters pass selection; ISOTONIC completion enters
            # the success/failure averages below. These are distinct inputs.
            player_tti, ball_time = calculate_arrivals(
                state["p_pos"], state["p_vel"], state["x_bc"], targets,
            )
            valid_players = np.linalg.norm(state["p_pos"], axis=-1) < 500
            valid_players[10] = False
            raw_control, _ = pitch_control_by_player(log_parameters, player_tti, ball_time, valid_players)
            raw_control = np.clip(np.asarray(raw_control), 0, 1)
            pass_completion = np.asarray(pass_model.calibrate(raw_control))
            drive_loss = float(drive_model.predict(
                state["r_defs"], state["v_defs"], state["x_bc"], state["v_bc"],
                physics_kwargs=dict(
                    pitch_length=checkpoint.pitch_length,
                    pitch_width=checkpoint.pitch_width,
                ),
            )["loss_probability"])

            # C. Fitted FLAT pass-choice distribution, including its temperature.
            choice_features = choice_model.build_features(
                x_bc=state["x_bc"], v_bc=state["v_bc"], r_atts=state["r_atts"], r_defs=state["r_defs"],
                targets=targets, attacking_side=state["attacking_side"], target_pitch_control=raw_control,
                ball_drive_prob=drive_loss, start_type_raw=checkpoint.start_type_raw,
                previous_action_raw=checkpoint.previous_action_raw,
                match_clock_min=checkpoint.match_clock_min, score_difference_prefix=checkpoint.score_difference_prefix,
                pitch_length=checkpoint.pitch_length, pitch_width=checkpoint.pitch_width,
                pressure_lines=(opponent_lines, own_lines),
            )
            pass_choice = choice_model.predict_features(choice_features, calibrated=True)

            # D. Longer-range space, then explicit scalar feature definitions.
            bcos_area = calculate_bcos(state, drive_model, protocol["bcos"])
            record = build_features(state, checkpoint, targets, pass_choice, pass_completion,
                                    drive_loss, bcos_area, choice_model.xt_grid, own_lines, opponent_lines)
            identity_columns = KEYS + ["y", "start_type_raw", "previous_action_raw"]
            record.update({key: getattr(checkpoint, key) for key in identity_columns})
            records.append(record)
            if len(records) % 25 == 0:
                print("ACTION_FEATURES", split_name, match_id, len(records), "/", len(checkpoints), flush=True)
        table = pd.DataFrame(records)
        table.to_parquet(target_file, index=False)


# ============================================================
# 3. ACTION P(carry) WITHOUT IN-SAMPLE ACTION-CLASSIFIER SCORES
# ============================================================

def cross_fit_carry_probability(action_states, split, configuration, tree_count,
                                fixed_parameters, seed, threads, folder, fold=None):
    """One native P(carry) per state, preserving the accepted stacking scheme.

    Outer fold: three 4-fit/2-held action fits score the six fitting matches;
    the existing six-fit action model scores outer validation and test.
    Production: five 8-fit/2-held action fits score all ten training matches.
    Physical/pass-choice/xT/BCOS inputs remain frozen from the parent split.
    This is cross-fitting the action CLASSIFIER, not the whole upstream chain.
    """
    carry_probability = np.full(len(action_states), np.nan)
    if fold is None:
        groups = [(sorted(set(action_states.match_id) - set(held)), held)
                  for held in split]
    else:
        groups = []
        for pair in range(3):
            held = split["fit_matches"][2 * pair:2 * pair + 2]
            fitting = [match for match in split["fit_matches"] if match not in held]
            groups.append((fitting, held))

    for pair, (fitting_matches, held_matches) in enumerate(groups):
        fitting_states = action_states.loc[action_states.match_id.isin(fitting_matches)]
        held_states = action_states.loc[action_states.match_id.isin(held_matches)]
        vocabulary = fit_category_vocabulary(fitting_states)
        D_fit = build_xgboost_input(fitting_states, "action_selection", vocabulary, threads)
        D_held = build_xgboost_input(held_states, "action_selection", vocabulary, threads, with_labels=False)
        fit_seed = seed + pair if fold is None else seed + 1000 + 100 * fold + pair
        parameters = fixed_parameters | configuration | dict(nthread=threads, seed=fit_seed)
        model, _ = train_model(D_fit, parameters, max_rounds=tree_count)
        carry_probability[held_states.index] = 1 - model.predict(D_held)
        pair_folder = Path(folder) / f"pair_{pair}"
        pair_folder.mkdir(parents=True, exist_ok=False)
        model.save_model(pair_folder / "model.ubj")
        write_json(pair_folder / "provenance.json", dict(
            fitting_matches=fitting_matches, held_matches=held_matches,
            vocabulary=vocabulary, rounds=tree_count, parameters=parameters,
            upstream_scope="frozen parent-split physics/choice/xT/BCOS",
        ))
    return carry_probability


def prepare_residual_features(data_root, output, split_name, threads=1):
    """Reuse action features, attach cross-fitted P(carry), select verified labels.

    residual_labels.parquet is the final accepted 7,820-state cohort. Joining
    on KEYS changes the target to terminal loss. Elapsed time comes from its
    observed possession age; completed_canonical_drives is not an RR predictor.
    """
    data_root, output = Path(data_root), Path(output)
    protocol = json.loads((data_root / "protocol.json").read_text())
    folder = output / "residual_risk" / "features" / split_name
    folder.mkdir(parents=True, exist_ok=False)
    source = output / "action_selection" / "features" / split_name
    action_states = pd.concat(
        [pd.read_parquet(source / f"{m}.parquet") for m in protocol["matches"]],
        ignore_index=True,
    )
    action_root = output / "action_selection"

    if split_name == "production":
        manifest = json.loads((action_root / "production/action_selection_model.json").read_text())
        configuration = {key: manifest["training"]["parameters"][key] for key in ["max_depth", "min_child_weight", "reg_lambda", "eta"]}
        tree_count = manifest["training"]["rounds"]
        held_pairs = [protocol["folds"][str(f)]["test_matches"] for f in range(5)]
        carry_probability = cross_fit_carry_probability(
            action_states, held_pairs, configuration, tree_count, protocol["fixed_parameters"],
            protocol["seed"], threads, folder / "action_crossfit",
        )
    else:
        fold = int(split_name.split("_")[-1])
        split = protocol["folds"][str(fold)]
        fitted_folder = action_root / "training" / split_name
        selection = json.loads((fitted_folder / "selection.json").read_text())
        specification = json.loads((fitted_folder / "feature_spec.json").read_text())
        configuration = {key: selection[key] for key in ["max_depth", "min_child_weight", "reg_lambda", "eta"]}
        carry_probability = cross_fit_carry_probability(
            action_states, split, configuration, selection["best_round"], protocol["fixed_parameters"],
            protocol["seed"], threads, folder / "action_crossfit", fold=fold,
        )
        # The matching six-fit action model supplies native validation/test scores.
        held_states = action_states.loc[action_states.match_id.isin(split["validation_matches"] + split["test_matches"])]
        model = xgb.Booster()
        model.load_model(fitted_folder / "model.ubj")
        model.set_param({"nthread": threads})
        D_held = build_xgboost_input(held_states, "action_selection", specification["vocabulary"], threads, with_labels=False)
        carry_probability[held_states.index] = 1 - model.predict(D_held)

    if not np.isfinite(carry_probability).all():
        raise ValueError("Missing cross-fitted P(carry) for an action state.")
    action_states["action_carry_probability"] = carry_probability
    labels = pd.read_parquet(data_root / "residual_labels.parquet")
    # The residual label table supplies actual elapsed time. Only the action
    # target is discarded; it must be replaced by the terminal-loss target.
    action_states = action_states.drop(columns=["y"])
    residual_states = action_states.merge(labels, on=KEYS, how="inner", validate="one_to_one")
    residual_states["y"] = residual_states.residual_loss.astype(int)
    if len(residual_states) != len(labels):
        raise ValueError("A verified residual state is missing from the action feature table.")
    for match_id, states in residual_states.groupby("match_id", sort=True):
        states.to_parquet(folder / f"{match_id}.parquet", index=False)
