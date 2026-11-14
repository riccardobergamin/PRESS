"""Release-frame covariates shared by flat and cross-nested pass selection.

Input: the accepted pass table, preprocessed tracking, and frozen upstream
models. Output: one feature matrix per match/split, plus fixed memberships.
No pass selection is trained here. No new cohort filter is introduced.
"""

import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from PRESS.model import BallDriveModel, PassRiskModel
from PRESS.physics.flight_ball import generate_ball_trajectories
from PRESS.physics.times import time_to_intercept
from PRESS.utils.state import compute_dynamic_pressure_lines

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.frame_states import state_at_frame


CONTEXT_COLUMNS = ["start_type_raw", "previous_action_raw"]
COUNT_COLUMNS = ["opponents_ahead_bc", "teammates_ahead_bc",
                 "opponents_ahead_target", "teammates_ahead_target"]


def write_json(path, content):
    Path(path).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def fit_category_vocabulary(pass_table, fitting_matches):
    """Learn one-hot categories on the six fitting matches (all ten in refit)."""
    fitting_rows = pass_table.loc[pass_table.match_id.isin(fitting_matches)]
    return {
        column: sorted(fitting_rows[column].dropna().unique()) + ["__missing__", "__unseen__"]
        for column in CONTEXT_COLUMNS
    }


def feature_names(vocabulary):
    """Column order is the same for both model families and public inference."""
    names = [
        "pass_distance", "pass_angle", "passer_speed", "target_pitch_control",
        "target_xt_team", "target_xt_opp", "opp_pressure_line_index", "team_pressure_line_index",
        "ball_drive_prob", "xt_team_bc", "xt_opp_bc", "opp_pressure_index_bc", "team_pressure_index_bc",
    ]
    for column, categories in vocabulary.items():
        names.extend(f"{column}={category}" for category in categories)
    return names + ["match_clock_min", "score_difference_prefix"] + COUNT_COLUMNS


def candidate_grid(length, width):
    """All native one-metre cell centres; the categorical denominator keeps all cells."""
    nx, ny = int(round(length)), int(round(width))
    if not np.allclose([nx, ny], [length, width], atol=1e-8, rtol=0):
        raise ValueError("The accepted grid requires integer pitch dimensions.")
    x, y = np.meshgrid(np.arange(nx) + .5 - length / 2,
                       np.arange(ny) + .5 - width / 2, indexing="ij")
    return np.column_stack([x.ravel(), y.ravel()]).astype(np.float32)


def observed_destination_cell(x, y, length, width):
    """One-metre observed destination bin, with the accepted boundary convention.

    Usually this is the nearest centre. At an EXACT cell boundary, floor assigns
    the point to the next cell; argmin(distance) would instead select the previous
    equal-distance centre. Preserve the original half-open bins, including ties.
    """
    ix = np.clip(np.floor(x + length / 2).astype(int), 0, int(length) - 1)
    iy = np.clip(np.floor(y + width / 2).astype(int), 0, int(width) - 1)
    return ix * int(width) + iy


def release_state(tracking, pass_record, players):
    """Read one matched release; the matched event identifies the actual passer.

    Player order: ten teammates, carrier at index 10, eleven opponents.
    PRESS pads missing slots at (1000,1000), with velocity zero.
    """
    alias = players.loc[players.player_id.eq(pass_record.player_id), "player_possession"]
    if len(alias) != 1:
        raise ValueError("The matched passer must map to exactly one tracking alias.")
    return state_at_frame(tracking, pass_record.frame_end, alias.iloc[0], pass_record.attacking_side)


def make_arrival_calculator(settings):
    """Precompute the flight lookup once; return player TTI and ball time per cell.

    Use the SAME TTI for flight-boundary selection and pitch control, including
    rotation and preservation of initial speeds above v_max. Geometry is float32;
    competing-risk integration below accumulates in float64, as in the benchmark.
    """
    precision = getattr(jax, "enable_x64", None)
    if precision is None:
        from jax.experimental import enable_x64 as precision
    with precision(False):
        times = jnp.arange(0., settings["t_max_flight"] + settings["dt_ball"],
                           settings["dt_ball"], dtype=jnp.float32)
        flights = generate_ball_trajectories(
            jnp.linspace(*settings["speed_range"], settings["speed_candidates"], dtype=jnp.float32),
            jnp.linspace(*np.deg2rad(settings["elevation_degrees"]),
                         settings["elevation_candidates"], dtype=jnp.float32),
            times, **{k: settings[k] for k in ["rho", "Cd", "A", "m", "g"]},
        )

    @jax.jit
    def calculate(positions, velocities, carrier_position, targets):
        with precision(False):
            positions, velocities, carrier_position, targets = [
                jnp.asarray(a, jnp.float32) for a in [positions, velocities, carrier_position, targets]
            ]
            player_tti = time_to_intercept(
                positions, velocities, targets,
                v_max=settings["v_max"], a_max=settings["a_max"],
                a_turn=settings["a_turn"],
                preserve_initial_overspeed=settings["preserve_initial_overspeed"],
            )
            # At each destination the closest teammate defines the desired ball time.
            nearest = jnp.argmin(jnp.linalg.norm(positions[:10, None] - targets[None], axis=-1), axis=0)
            desired_time = player_tti[nearest, jnp.arange(len(targets))]
            distances = jnp.linalg.norm(targets - carrier_position, axis=-1)

            def select_flight(distance, receiver_time):
                # 1. Interpolate each candidate's time at the target distance.
                ball_times = jax.vmap(
                    lambda path: jnp.interp(distance, path[:, 0], times),
                )(flights)
                # 2. Evaluate its height at that time.
                heights = jax.vmap(
                    lambda path, hit_time: jnp.interp(hit_time, times, path[:, 1]),
                )(flights, ball_times)
                # 3. Keep candidates that reach the destination below z_max.
                reachable = ((heights <= settings["z_max"])
                             & (distance <= flights[:, :, 0].max(axis=1)))
                ball_times = jnp.where(reachable, ball_times, jnp.inf)
                # 4. Choose the arrival closest to the nearest teammate's TTI.
                return ball_times[jnp.argmin(abs(ball_times - receiver_time))]

            ball_time = jax.vmap(select_flight)(distances, desired_time)
        return player_tti.T, ball_time  # (cells,22), (cells,)

    return calculate


def pitch_control_step(control, t, player_tti, ball_arrival_time,
                       lambda_control, s, valid_players, dt=.01):
    """One exponential-Euler update at every fixed candidate destination.

    control=(U, P_attack, P_defense, P_teammates).
    U, P_attack, P_defense: (cells,); P_teammates: (cells,10).
    player_tti: (cells,22), ordered teammates / passer / opponents.

    U is the probability the ball has not yet been controlled. P_attack and
    P_defense are the accumulated control probabilities of the two teams.
    P_teammates retains the attacking contribution of EACH possible receiver.
    """
    U, P_attack, P_defense, P_teammates = control

    # 1. Arrival probabilities F_i(t)=sigmoid(pi*(t-TTI_i)/(sqrt(3)*s)).
    F = jax.nn.sigmoid(jnp.pi / (jnp.sqrt(3.) * s) * (t - player_tti))
    F = jnp.where(valid_players, F, 0.)  # padding and passer cannot control

    # 2. Sum over PLAYERS (axis=1) for every candidate destination.
    # Individual rates are r_i=lambda_control*F_i.
    sum_F_attack = F[:, :10].sum(axis=1)
    sum_F_defense = F[:, 11:].sum(axis=1)
    sum_F_all = sum_F_attack + sum_F_defense

    # 3. Integrated total rate H=dt*lambda_control*sum_i F_i.
    # No control is allowed before the hypothetical ball arrival.
    ball_can_be_controlled = t >= ball_arrival_time
    H = jnp.where(ball_can_be_controlled, dt * lambda_control * sum_F_all, 0.)

    # 4. Probability mass captured in this interval: dP_total=U*(1-exp(-H)).
    # expm1 is accurate also when H is small.
    dP_total = U * (-jnp.expm1(-H))

    # 5. Allocate that mass in proportion to the competing rates.
    # The common lambda_control cancels: r_i/sum_j r_j = F_i/sum_j F_j.
    # With no available player, dP_total=0; denominator=1 avoids division by zero.
    denominator = jnp.where(sum_F_all > 0, sum_F_all, 1.)
    dP_attack = dP_total * sum_F_attack / denominator
    dP_defense = dP_total * sum_F_defense / denominator
    dP_teammates = dP_total[:, None] * F[:, :10] / denominator[:, None]

    # 6. Update the uncontrolled mass and accumulated control probabilities.
    # U_next=U*exp(-H); P_next=P+dP. U+P_attack+P_defense stays one.
    next_U = U * jnp.exp(-H)
    next_P_attack = P_attack + dP_attack
    next_P_defense = P_defense + dP_defense
    next_P_teammates = P_teammates + dP_teammates
    return next_U, next_P_attack, next_P_defense, next_P_teammates


@jax.jit
def pitch_control_by_player(log_parameters, player_tti, ball_arrival_time, valid_players):
    """Integrate fixed-target control for 15 s, using 1,500 steps of .01 s.

    Returns P_attack (cells,) and P_teammates (cells,10). P_attack is the RAW
    pitch-control feature; individual contributions define nest memberships.
    No isotonic or temperature scaling enters this calculation.
    """
    lambda_control, s = jnp.exp(log_parameters)
    player_tti = jnp.asarray(player_tti, jnp.float64)
    times = jnp.arange(1500, dtype=jnp.float32) * jnp.float32(.01)
    n_cells = len(player_tti)

    # Initial state: U=1, both team probabilities and all individual controls=0.
    initial_control = (
        jnp.ones(n_cells, jnp.float64),
        jnp.zeros(n_cells, jnp.float64),
        jnp.zeros(n_cells, jnp.float64),
        jnp.zeros((n_cells, 10), jnp.float64),
    )

    def step_at_time(control, t):
        next_control = pitch_control_step(
            control, t, player_tti, ball_arrival_time, lambda_control, s, valid_players,
        )
        return next_control, None

    # Chronological loop: every time step receives the previous control state.
    (U, P_attack, P_defense, P_teammates), _ = jax.lax.scan(
        step_at_time, initial_control, times,
    )
    return P_attack, P_teammates


def normalized_log_memberships(individual_control, valid_teammates):
    """alpha_jm=PC_jm/sum_m PC_jm. Memberships stay fixed during training.

    Exact-zero total control: keep the cell, use uniform membership over observed
    teammates. Missing teammates have alpha=0 (log alpha=-infinity).
    """
    valid = np.asarray(valid_teammates, bool)
    if not valid.any():
        raise ValueError("No observed teammate for nest membership.")
    contributions = np.where(valid, np.asarray(individual_control, float), 0.)
    total = contributions.sum(axis=-1, keepdims=True)
    if not np.isfinite(contributions).all() or (contributions < 0).any():
        raise ValueError("Invalid individual pitch control.")
    with np.errstate(divide="ignore", invalid="ignore"):
        log_alpha = np.log(contributions) - np.log(total)
    fallback = total[..., 0] == 0
    log_alpha = np.where(fallback[..., None],
                         np.where(valid, -np.log(valid.sum()), -np.inf), log_alpha)
    return log_alpha.astype(np.float32)


def build_features(state, targets, raw_control, drive_loss, xt_surface, vocabulary, pass_record):
    """Construct the candidate covariates for ONE observed pass.

    Inputs:
      state: player positions/velocities at the matched release frame.
      targets: all candidate cell centres, shape (cells,2).
      raw_control: raw attacking pitch control at each cell, shape (cells,).
      drive_loss: physical probability of carry LOSS within one second, scalar.
      xt_surface: frozen (10,6) xT map for the requested fitting split.
      vocabulary: one-hot categories learned only on the fitting matches.
      pass_record: known-at-release context and pitch dimensions.

    Output: (cells,features), in feature_names(vocabulary) order.
    One row is one POSSIBLE destination. The observed destination and outcome
    are not predictors. Read: helper functions -> calculations -> feature set.
    """
    carrier_position = state["x_bc"]
    carrier_velocity = state["v_bc"]
    attack_sign = 1 if pass_record.attacking_side == "left_to_right" else -1
    n_cells = len(targets)

    # 1. Support functions. Define them all BEFORE calculating/assembling features.

    def repeat_at_all_targets(value):
        """A carrier/context scalar is the same for every candidate destination."""
        return np.full(n_cells, value)

    def xt_at_points(points, direction):
        """Orient XY toward this team's attack, then read the frozen xT cell."""
        points = points * direction
        ix = np.clip(((points[:, 0] + pass_record.pitch_length / 2)
                      / pass_record.pitch_length * 10).astype(int), 0, 9)
        iy = np.clip(((points[:, 1] + pass_record.pitch_width / 2)
                      / pass_record.pitch_width * 6).astype(int), 0, 5)
        return xt_surface[ix, iy]

    def nearest_line_index(x, centres, direction):
        """Index 1,2,3 of the nearest longitudinal pressure-line centre.

        Sort in the relevant team's attack direction. This is a nearest-line
        index, not a count of lines crossed by the pass.
        """
        centres = np.sort(direction * np.asarray(centres, float))
        if not np.isfinite(centres).all():
            return np.full(len(x), np.nan)
        distance_to_lines = abs(direction * x[:, None] - centres[None])
        return (1 + np.argmin(distance_to_lines, axis=1)).astype(np.float32)

    def players_ahead_of_line(positions, locations):
        """N_G(z)=sum_i 1[attack_sign*x_i > attack_sign*z_x].

        Same carrier-team attack sign for BOTH rosters. Exclude padding;
        the own roster already excludes the carrier. All positions come from
        the release frame, including when the query is a candidate destination.
        A backward candidate can have MORE players ahead; no subtraction/clipping.
        """
        valid = np.isfinite(positions).all(axis=1) & (np.linalg.norm(positions, axis=1) < 500)
        positions = positions[valid]
        return (attack_sign * positions[:, None, 0] > attack_sign * locations[None, :, 0]).sum(axis=0)

    # 2. State at the carrier and the two teams' current pressure lines.
    carrier_speed = np.linalg.norm(carrier_velocity)
    opposite_direction = "right_to_left" if attack_sign == 1 else "left_to_right"
    opponent_lines = compute_dynamic_pressure_lines(state["r_defs"], pass_record.attacking_side)
    own_lines = compute_dynamic_pressure_lines(state["r_atts"], opposite_direction)

    # 3. Candidate pass geometry relative to current carrier motion.
    # delta_j=target_j-x_bc; angle_j=atan2(v_bc cross delta_j, v_bc dot delta_j).
    # The direction is undefined for a stationary carrier: retain NaN.
    displacement = targets - carrier_position
    pass_distance = np.linalg.norm(displacement, axis=1)
    pass_angle = np.full(n_cells, np.nan) if carrier_speed < 1e-8 else np.arctan2(
        carrier_velocity[0] * displacement[:, 1] - carrier_velocity[1] * displacement[:, 0],
        displacement @ carrier_velocity,
    )

    # 4. Final feature set. Scalars are repeated; destination-dependent values vary.
    # Semantic grouping is independent of the model's fixed input-column order.
    feature_columns = dict(
        # Geometry and tactical state at the CANDIDATE DESTINATION.
        pass_distance=pass_distance,
        pass_angle=pass_angle,
        target_pitch_control=raw_control,  # raw PC, without isotonic scaling
        target_xt_team=xt_at_points(targets, attack_sign),
        target_xt_opp=xt_at_points(targets, -attack_sign),
        opp_pressure_line_index=nearest_line_index(targets[:, 0], opponent_lines, attack_sign),
        team_pressure_line_index=nearest_line_index(targets[:, 0], own_lines, -attack_sign),
        opponents_ahead_target=players_ahead_of_line(state["r_defs"], targets),
        teammates_ahead_target=players_ahead_of_line(state["r_atts"], targets),

        # State AT THE CARRIER, shared by every destination.
        # carrier_position[None] changes (2,) -> (1,2) for the point evaluators.
        passer_speed=repeat_at_all_targets(carrier_speed),
        ball_drive_prob=repeat_at_all_targets(drive_loss),
        xt_team_bc=repeat_at_all_targets(xt_at_points(carrier_position[None], attack_sign)[0]),
        xt_opp_bc=repeat_at_all_targets(xt_at_points(carrier_position[None], -attack_sign)[0]),
        opp_pressure_index_bc=repeat_at_all_targets(nearest_line_index(carrier_position[:1], opponent_lines, attack_sign)[0]),
        team_pressure_index_bc=repeat_at_all_targets(nearest_line_index(carrier_position[:1], own_lines, -attack_sign)[0]),
        opponents_ahead_bc=repeat_at_all_targets(players_ahead_of_line(state["r_defs"], carrier_position[None])[0]),
        teammates_ahead_bc=repeat_at_all_targets(players_ahead_of_line(state["r_atts"], carrier_position[None])[0]),

        # Context known at the release frame.
        match_clock_min=repeat_at_all_targets(pass_record.match_clock_min),
        score_difference_prefix=repeat_at_all_targets(pass_record.score_difference_prefix),
        
    )
    
    # One-hot context: fitting categories stay frozen on validation/test.
    # Absent values -> missing bucket; new categories -> unseen bucket.
    for column, categories in vocabulary.items():
        value = getattr(pass_record, column)
        key = "__missing__" if pd.isna(value) else value if value in categories else "__unseen__"
        for category in categories:
            feature_columns[f"{column}={category}"] = repeat_at_all_targets(key == category)

    # 5. Assemble by accepted COLUMN NAME, not by dictionary insertion order.
    # Reordering the semantic blocks above therefore never permutes model inputs.
    matrix = np.column_stack([feature_columns[name] for name in feature_names(vocabulary)]).astype(np.float32)
    if matrix.shape != (n_cells, len(feature_names(vocabulary))) or np.isinf(matrix).any():
        raise ValueError("Invalid choice features; NaN is allowed, infinity is not.")
    return matrix


def prepare_features(data_root, tracking_root, output, split_name):
    """Calculate choice-model inputs with the physical estimates of ONE split.

    Example split_name='fold_2':
      fitting matches <- protocol['folds']['2']['fit_matches'] (six matches)
      pass parameters <- upstream/fold_2/pass_fit.json
      pass physics    <- upstream/fold_2/pass_risk_model.json
      carry model     <- upstream/fold_2/ball_drive_model.json (MAP + spline bases)
      xT surface      <- upstream/fold_2/xt.npy

    Calculate features for all ten matches using THESE SAME frozen estimates.
    Validation/test matches do not change the physical parameters or vocabulary.
    For split_name='production', load upstream/production/ instead; the physical
    estimates, xT and vocabulary then come from all ten matches.

    Per release frame:
      current tracking -> player TTI and hypothetical ball flight
                       -> RAW team pitch control and receiver contributions
      same tracking    -> physical one-second ball-drive LOSS probability
      these quantities -> build_features -> save the complete candidate matrix

    No physical or choice model is fitted here. Output arrays per match:
      features.npy         (passes,cells,features), shared by flat and CNL
      log_memberships.npy  (passes,cells,10), individual-control memberships
      targets.npy          (cells,2), all native one-metre candidate centres
      rows.parquet         keys/context plus outcome and chosen-cell label
    """

    # 1. Which estimates must this split use?
    # The directory name explicitly selects the corresponding six-match fits.
    protocol = json.loads((data_root / "protocol.json").read_text())
    if split_name == "production":
        fitting_matches = protocol["matches"]
        estimates_folder = data_root / "upstream" / "production"
    else:
        fold_number = split_name.removeprefix("fold_")
        split = protocol["folds"][fold_number]
        fitting_matches = split["fit_matches"]
        estimates_folder = data_root / "upstream" / f"fold_{fold_number}"

    # 2. PASS RISK: read lambda_control, s and the saved TTI/ball-flight settings.
    # These are the trajectory-MLE estimates of this split. The saved isotonic
    # calibrator is NOT used: pass selection needs raw pitch control as a feature.
    pass_model_path = estimates_folder / "pass_risk_model.json"
    pass_risk_model = PassRiskModel.from_json(pass_model_path)
    lambda_control = pass_risk_model.lambda_control
    arrival_dispersion_s = pass_risk_model.s
    pass_physics = pass_risk_model.physics
    pass_model_export = pass_risk_model.to_dict()

    # Keep the original [log(lambda_control),log(s)] optimizer coordinates for
    # numerical reproduction. The two accepted export formats store them here:
    if split_name == "production":
        pass_parameter_estimates = pass_model_export["fit"]
        pass_fitting_matches = pass_model_export["provenance"]["matches"]
    else:
        pass_parameter_path = estimates_folder / "pass_fit.json"
        pass_parameter_estimates = json.loads(pass_parameter_path.read_text())
        pass_fitting_matches = pass_parameter_estimates["fit_matches"]
    pass_control_log_parameters = jnp.asarray(pass_parameter_estimates["log_parameters"], dtype=jnp.float64)

    if set(pass_fitting_matches) != set(fitting_matches):
        raise ValueError("Pass-risk estimates do not match this split's fitting matches.")
    if not np.allclose(np.exp(np.asarray(pass_control_log_parameters)),
                       [lambda_control, arrival_dispersion_s], rtol=1e-12, atol=0.):
        raise ValueError("Pass-risk parameters differ between the saved fit and model export.")

    # 3. BALL DRIVE: load this split's MAP coefficients AND its fitted spline bases.
    # predict() will use those fixed coefficients/bases to integrate carry risk.
    # In particular, its own s is the ball-drive estimate, not pass risk's s above.
    drive_model_path = estimates_folder / "ball_drive_model.json"
    ball_drive_model = BallDriveModel.from_json(drive_model_path)
    # Its parameter vector starts [log(lambda_0), beta_v, beta_B, log(s_drive), ...].
    # This value is recorded below for clarity; predict() already reads it itself.
    ball_drive_dispersion_s = float(np.exp(ball_drive_model.theta[3]))
    drive_fitting_matches = ball_drive_model.to_dict()["training"]["match_ids"]
    if set(drive_fitting_matches) != set(fitting_matches):
        raise ValueError("Ball-drive estimates do not match this split's fitting matches.")

    # 4. Other frozen inputs: this split's xT surface and fitting-only categories.
    pass_table = pd.read_parquet(data_root / "rows.parquet")
    pass_table["match_id"] = pass_table.match_id.astype(str)
    xt_surface = np.load(estimates_folder / "xt.npy")
    vocabulary = fit_category_vocabulary(pass_table, fitting_matches)
    column_names = feature_names(vocabulary)

    # 5. Build hypothetical flight candidates ONCE with this pass model's physics.
    # For each release, calculate_arrivals will evaluate the shared TTI routine
    # and select a ball arrival time for every candidate destination.
    calculate_arrivals = make_arrival_calculator(pass_physics)

    folder = output / "features" / split_name
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "feature_spec.json", dict(
        names=column_names, vocabulary=vocabulary, fitting_matches=fitting_matches,
        physical_inputs=dict(
            pass_risk_model=str(pass_model_path), ball_drive_model=str(drive_model_path),
            lambda_control=float(lambda_control), pass_s=float(arrival_dispersion_s),
            ball_drive_s=ball_drive_dispersion_s,
            pitch_control_calibration="raw, no isotonic",
        ),
    ))

    # 6. Load each match once. Process exactly its accepted pass rows, in order.
    for match_id, match_passes in pass_table.groupby("match_id", sort=True):
        match_folder = folder / match_id
        match_folder.mkdir()
        tracking = pd.read_parquet(tracking_root / "matches" / match_id / f"{match_id}_tracking.parquet")
        players = pd.read_parquet(data_root / "matches" / match_id / "players.parquet")
        length, width = match_passes.iloc[0][["pitch_length", "pitch_width"]]
        targets = candidate_grid(length, width)
        np.save(match_folder / "targets.npy", targets)

        # Large arrays are allocated on disk and filled one release at a time.
        feature_matrices = np.lib.format.open_memmap(
            match_folder / "features.npy", mode="w+", dtype=np.float32,
            shape=(len(match_passes), len(targets), len(column_names)),
        )
        log_memberships = np.lib.format.open_memmap(
            match_folder / "log_memberships.npy", mode="w+", dtype=np.float32,
            shape=(len(match_passes), len(targets), 10),
        )
        chosen_cells = []

        for i, record in enumerate(match_passes.itertuples(index=False)):
            # A. Initial state from the matched RELEASE frame, via get_pv_players.
            state = release_state(tracking, record, players)

            # B. RAW PITCH CONTROL with this fold's lambda_control and pass s.
            # First calculate TTI_i(target_j) and the hypothetical ball arrival.
            # Shapes: player_tti=(cells,22); ball_arrival_time=(cells,).
            player_tti, ball_arrival_time = calculate_arrivals(
                state["p_pos"], state["p_vel"], state["x_bc"], targets,
            )
            valid_players = np.linalg.norm(state["p_pos"], axis=-1) < 500
            valid_players[10] = False  # exclude the passer from competing control

            # Then integrate r_i(t)=lambda_control*F_i(t;s_pass) with exponential
            # Euler: one team control probability per destination, plus each
            # receiver's contribution. The function exponentiates the log params.
            raw_pitch_control, individual_control = pitch_control_by_player(
                pass_control_log_parameters, player_tti, ball_arrival_time,
                jnp.asarray(valid_players),
            )
            # There is deliberately no pass_risk_model.calibrate(...) call here.

            # C. BALL-DRIVE RISK from the SAME release state, over a fixed 1 s.
            # The loaded BallDriveModel projects the carrier, evaluates defender
            # F with its own fitted s, evaluates spline lambda_tackle and integrates
            # survival. We take P(loss within 1s), not 1-P(loss).
            drive_prediction = ball_drive_model.predict(
                r_defs=state["r_defs"], v_defs=state["v_defs"],
                x_bc=state["x_bc"], v_bc=state["v_bc"], horizon=1.0,
                physics_kwargs=dict(pitch_length=float(length), pitch_width=float(width)),
            )
            ball_drive_loss = float(drive_prediction["loss_probability"])

            # D. Put the computed quantities into their feature columns.
            # raw_pitch_control varies across cells; ball_drive_loss is repeated.
            feature_matrices[i] = build_features(
                state=state, targets=targets, raw_control=np.asarray(raw_pitch_control),
                drive_loss=ball_drive_loss, xt_surface=xt_surface,
                vocabulary=vocabulary, pass_record=record,
            )

            # E. CNL memberships from the SAME individual raw-control calculation:
            # alpha_jm=PC_jm/sum_m PC_jm. Flat uses the same features, without nests.
            log_memberships[i] = normalized_log_memberships(individual_control, valid_players[:10])

            # F. Only the LABEL uses the observed destination. It never affects B-D.
            chosen_cells.append(int(observed_destination_cell(record.x_target, record.y_target, length, width)))
            if (i + 1) % 25 == 0:
                print("FEATURES", split_name, match_id, i + 1, "/", len(match_passes), flush=True)

        feature_matrices.flush()
        log_memberships.flush()
        metadata = match_passes.copy()
        metadata["chosen_cell"] = chosen_cells
        metadata["n_candidates"] = len(targets)
        metadata.to_parquet(match_folder / "rows.parquet", index=False)

    write_json(folder / "complete.json", dict(n_choices=len(pass_table), split=split_name))


def load_pass_tables(features_folder, matches, family):
    """Load explicit candidate, observed-choice and membership tables.

    candidate_features: (C_tot,F), MultiIndex (pass_index,destination).
    observed_choices: P rows, index pass_index; chosen_cell is the observed y_i.
    log_memberships: (C_tot,N), same index as candidate_features; None for flat.
    N=10 teammate slots at each release, not fixed player identities across passes.
    Numeric features alone enter XGBoost; membership values remain separate.
    """
    if not (features_folder / "complete.json").is_file():
        raise ValueError("Incomplete feature stage.")
    specification = json.loads((features_folder / "feature_spec.json").read_text())
    feature_arrays, choice_tables, membership_arrays, destinations = [], [], [], []
    for match_id in sorted(matches):
        folder = features_folder / match_id
        rows = pd.read_parquet(folder / "rows.parquet")
        values = np.load(folder / "features.npy", mmap_mode="r")  # (P_match,C,F)
        P_match, C, F = values.shape
        if P_match != len(rows) or F != len(specification["names"]):
            raise ValueError("Feature matrix/metadata mismatch.")
        feature_arrays.append(values.reshape(-1,F))
        choice_tables.append(rows[[
            "match_id", "pass_id", "frame_end", "outcome", "chosen_cell", "n_candidates",
        ]])
        destinations.append(np.tile(np.arange(C,dtype=np.int32),P_match))
        if family == "cnl":
            log_alpha = np.load(folder / "log_memberships.npy", mmap_mode="r")
            membership_arrays.append(log_alpha.reshape(-1,log_alpha.shape[-1]))

    # Table 2: pass_index=i identifies the same pass in all three tables.
    # Real match/pass IDs remain columns for reporting and selecting game events.
    observed_choices = pd.concat(choice_tables,ignore_index=True)
    observed_choices.index.name = "pass_index"
    C = observed_choices.n_candidates.to_numpy()
    pass_index = np.repeat(np.arange(len(observed_choices),dtype=np.int32),C)
    candidate_index = pd.MultiIndex.from_arrays(
        [pass_index,np.concatenate(destinations)], names=["pass_index","destination"],
    )

    # Table 1: one row per pass/destination pair; exactly F feature columns.
    # copy=False preserves the numeric buffer rather than creating another copy.
    candidate_features = pd.DataFrame(
        np.concatenate(feature_arrays),index=candidate_index,
        columns=specification["names"],copy=False,
    )

    # Table 3: each row describes those SAME candidate coordinates and its
    # memberships in teammate nests. Log values preserve the accepted inputs.
    log_memberships = None
    if family == "cnl":
        values = np.concatenate(membership_arrays)
        log_memberships = pd.DataFrame(
            values,index=candidate_index,
            columns=[f"log_alpha_{m}" for m in range(values.shape[1])],copy=False,
        )
    return candidate_features,observed_choices,log_memberships
