"""Final pass-probability equations, written as a procedural review script.

Reading order: reconstruct launch -> compute TTI/flight -> integrate control.
The ball ODE and individual TTI are the shared PRESS physics functions.
The likelihood and competing-control integration are explicit below.

Shapes: N passes, T forecast samples, P=22 players, XYZ=3 coordinates.
Player order: teammates 0:10, passer 10, opponents 11:22. Padding is masked.
Outcome: 1=completed pass, 0=failed pass. This model uses MLE.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from PRESS.physics.flight_ball import ball_trajectory, generate_ball_trajectories
from PRESS.physics.times import time_to_intercept


BALL_PARAMETERS = dict(rho=1.22, Cd=0.25, A=0.038, m=0.42, g=9.81)
PHYSICAL_SETTINGS = dict(
    v_max=5.5, a_max=7.0, a_turn=7.0, preserve_initial_overspeed=True,
    dt_ball=0.04, dt_pc=0.01, t_max_flight=10.0, t_max_pc=15.0,
    **BALL_PARAMETERS, z_max=2.0, distance_mode="exact",
    integration_method="exponential", speed_range=[5.0, 25.0],
    speed_candidates=30, elevation_degrees=[0.0, 45.0], elevation_candidates=30,
)
FLIGHT_TIMES = np.arange(0.0, 10.0 + 0.04 / 2, 0.04)
OBSERVATION_TIMES = np.arange(5) / 10.0
LAUNCH_LOWER = np.array([1.0, 0.0, -np.pi])
LAUNCH_UPPER = np.array([40.0, np.pi / 3, np.pi])
FINITE_DIFFERENCE_STEPS = np.array([0.001, 0.0001, 0.0001])
MEAN_BALL_TRAJECTORY = jax.jit(ball_trajectory)


def precision_context(enabled):
    """Use float32 for frozen physical geometry, float64 for MLE/integration."""
    context = getattr(jax, "enable_x64", None)
    if context is None:
        from jax.experimental import enable_x64 as context
    return context(enabled)


# ============================================================
# 1. BALL-FLIGHT INITIAL CONDITIONS FROM FIVE OBSERVED FRAMES
# ============================================================

@jax.jit
def project_ball_flight(launch_parameters, release_position, times):
    """Return XYZ positions from launch parameters [v0, theta, phi].

    v0: initial ball speed [m/s].
    theta: angle above the horizontal plane [rad].
    phi: direction in the pitch XY plane, measured from the x axis [rad].

    PRESS solves horizontal/vertical motion with gravity and drag. Rotate
    the horizontal displacement d(t) into pitch axes:
      x=x0+d*cos(phi), y=y0+d*sin(phi), z=z(t).
    """
    v0, theta, phi = launch_parameters
    flight = ball_trajectory(
        v0, theta, times, z0=release_position[2], **BALL_PARAMETERS,
    )
    horizontal_distance = flight[:, 0]
    return jnp.stack([
        release_position[0] + horizontal_distance * jnp.cos(phi),
        release_position[1] + horizontal_distance * jnp.sin(phi),
        flight[:, 1],
    ], axis=1)


def mean_velocity_trajectory(velocities, release_position):
    """Project a ball path from the mean of the five observed velocity vectors."""
    with precision_context(False):
        vx, vy, vz = velocities.mean(axis=0)
        v0 = float(np.linalg.norm([vx, vy, vz]))
        if v0 <= 1e-8:
            raise ValueError("Mean launch velocity is too small.")
        theta = float(np.arctan2(vz, np.hypot(vx, vy)))
        phi = float(np.arctan2(vy, vx) % (2 * np.pi))
        flight = MEAN_BALL_TRAJECTORY(v0, theta, FLIGHT_TIMES,
                                     z0=release_position[2], **BALL_PARAMETERS)
        trajectory = np.asarray(np.column_stack([
            release_position[0] + flight[:, 0] * np.cos(phi),
            release_position[1] + flight[:, 0] * np.sin(phi),
            flight[:, 1],
        ]), np.float32)
    return trajectory, (v0, theta, phi)


def forecast_position_error(trajectory, tracking, release_frame, reception_frame):
    """Mean 3D position error from .7s onward, before next control, up to 10s.

    Compare both candidates on these same observed frames. If no valid frame
    exists (for example a short pass), return NaN and zero comparison frames.
    """
    frames = np.arange(int(release_frame) + 7,
                       min(int(reception_frame), int(release_frame) + 101))
    if len(frames) == 0:
        return np.nan, 0
    observed = tracking.set_index("frame").reindex(frames)[["ball_x", "ball_y", "ball_z"]].to_numpy()
    times = (frames - int(release_frame)) / 10.0
    predicted = np.column_stack([
        np.interp(times, FLIGHT_TIMES, trajectory[:, axis]) for axis in range(3)
    ])
    valid = np.isfinite(observed).all(axis=1)
    if not valid.any():
        return np.nan, 0
    # e = mean_k ||r_projected(t_k)-r_observed(t_k)||, measured in metres.
    return float(np.linalg.norm(predicted[valid] - observed[valid], axis=1).mean()), int(valid.sum())


def estimate_launch_conditions(tracking, release_frame, target_position):
    """Fit v0,theta,phi to the first five observed ball positions.

    Return ONE projected trajectory, shape (251,3), and fit diagnostics.
    If the position fit is unusable, fall back to the mean of the first five
    observed velocities. The target only initializes phi; it is not a fit constraint.
    """
    frames = np.arange(int(release_frame), int(release_frame) + 5)
    window = tracking.loc[tracking.frame.isin(frames)].sort_values("frame")
    if len(window) != 5 or not np.array_equal(window.frame.to_numpy(), frames):
        raise ValueError("The launch fit requires five consecutive observed frames.")
    if "period" in window and window.period.nunique() != 1:
        raise ValueError("The launch window cannot cross match periods.")
    positions = window[["ball_x", "ball_y", "ball_z"]].to_numpy(float)
    velocities = window[["ball_vx", "ball_vy", "ball_vz"]].to_numpy(float)
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise ValueError("Nonfinite ball observations; do not silently drop this pass.")
    release_position = positions[0]

    with precision_context(False):
        # POSITION FIT: estimate [v0, theta, phi] from five XYZ observations.
        # positions.shape = (5, 3), at t = [0, .1, .2, .3, .4] seconds.
        # Keep the observed release position fixed throughout the fit.

        # 1. Construct a STARTING GUESS.
        # Observed displacement over .4s: Delta_r = r(.4) - r(0).
        displacement = positions[-1] - release_position
        # Horizontal displacement: d_xy = sqrt(Delta_x^2 + Delta_y^2).
        horizontal_distance = np.linalg.norm(displacement[:2])

        # Initial theta [rad]: slope of this displacement relative to XY.
        # Restrict the guess to the fit's bounds [0, pi/3] = [0, 60 degrees].
        # The small denominator floor handles negligible horizontal motion.
        initial_theta = np.clip(
            np.arctan2(displacement[2], max(horizontal_distance, 1e-6)), 0.0, np.pi / 3,
        )

        # Initial v0 [m/s]: d_xy / .4 approximates horizontal speed, and
        # v_horizontal = v0*cos(theta), hence v0 ~ d_xy/(.4*cos(theta)).
        # Clip to [1, 40] m/s, the bounds used in the subsequent fit.
        # This guess ignores gravity/drag; the fitted trajectory includes both.
        initial_v0 = np.clip(
            horizontal_distance / 0.4 / max(np.cos(initial_theta), 1e-6), 1.0, 40.0,
        )

        # Initial phi [rad]: direction from release to the pass destination.
        # The destination ONLY initializes phi; it is not a constraint or a
        # target point in the least-squares objective below.
        target_displacement = np.asarray(target_position) - release_position[:2]
        if np.linalg.norm(target_displacement) < 1e-6:
            # If that direction is undefined, use the observed XY displacement.
            initial_phi = float(np.arctan2(displacement[1], displacement[0]))
        else:
            initial_phi = float(np.arctan2(target_displacement[1], target_displacement[0]))
        initial_parameters = np.array([initial_v0, initial_theta, initial_phi])

        # 2. Define the residuals evaluated at every optimization step.
        def residual(launch_parameters):
            """Return 15 coordinate errors for a trial [v0, theta, phi].

            e_(k,c) = r_model(t_k; v0,theta,phi)_c - r_observed(t_k)_c.
            Least squares minimizes .5*sum(e**2) over the three parameters.
            Only the five observed positions enter this objective.
            """
            # Predicted XYZ at the same five times: shape (5, 3).
            predicted = project_ball_flight(
                jnp.asarray(launch_parameters, jnp.float32),
                jnp.asarray(release_position, jnp.float32),
                jnp.asarray(OBSERVATION_TIMES, jnp.float32),
            )
            # Flatten both matrices in the same order:
            # [dx_0, dy_0, dz_0, ..., dx_4, dy_4, dz_4], shape (15,).
            return np.asarray(predicted, float).ravel() - positions.ravel()

        def launch_jacobian(launch_parameters):
            """Differentiate the 15 residuals with respect to [v0, theta, phi].

            Write q = [v0, theta, phi] and e(q) = residual(q), shape (15,).
            The Jacobian is J[i,j] = d e_i / d q_j, shape (15, 3).
            Its columns are [d e/d v0, d e/d theta, d e/d phi].

            For least-squares loss L(q) = .5 * sum_i e_i(q)**2,
            the gradient is grad L = J.T @ e. The optimizer uses this J.
            """
            columns = []

            # Compute one column at a time: j=0 -> v0, j=1 -> theta, j=2 -> phi.
            # Perturbations h_j are .001 m/s, .0001 rad, .0001 rad. They must
            # remain large enough to produce distinct float32 model outputs.
            for index, step in enumerate(FINITE_DIFFERENCE_STEPS):
                # Start with two identical parameter vectors q^- = q^+ = q.
                # Only component j will change; the other two remain fixed.
                left, right = launch_parameters.copy(), launch_parameters.copy()

                # q^-_j = max(lower_j, q_j - h_j)
                # q^+_j = min(upper_j, q_j + h_j)
                # Keep both evaluations inside the parameter bounds.
                left[index] = max(LAUNCH_LOWER[index], launch_parameters[index] - step)
                right[index] = min(LAUNCH_UPPER[index], launch_parameters[index] + step)

                # Evaluate the 15 residuals at each perturbed vector:
                # J[:,j] ~ [e(q^+) - e(q^-)] / [q^+_j - q^-_j].
                # Away from bounds the denominator is 2*h_j (central
                # difference); at a bound it is h_j (one-sided difference).
                # Use the actual separation because clipping can shorten it.
                columns.append((residual(right) - residual(left)) / (right[index] - left[index]))

            # Stack three length-15 columns into the (15,3) Jacobian:
            # J = [d e/d v0 | d e/d theta | d e/d phi].
            return np.column_stack(columns)

        trajectory_fit = least_squares(
            residual, initial_parameters, jac=launch_jacobian,
            bounds=(LAUNCH_LOWER, LAUNCH_UPPER), method="trf", x_scale="jac",
            ftol=1e-6, xtol=1e-6, gtol=1e-6, max_nfev=100,
        )
        fit_usable = bool(
            trajectory_fit.success and np.isfinite(trajectory_fit.fun).all() and np.isfinite(trajectory_fit.x).all()
        )
        if fit_usable:
            trajectory = np.asarray(project_ball_flight(
                jnp.asarray(trajectory_fit.x, jnp.float32),
                jnp.asarray(release_position, jnp.float32),
                jnp.asarray(FLIGHT_TIMES, jnp.float32),
            ), np.float32)
            fit_usable = trajectory.shape == (251, 3) and np.isfinite(trajectory).all()

        if fit_usable:
            v0, theta, phi = trajectory_fit.x
        else:
            trajectory, (v0, theta, phi) = mean_velocity_trajectory(
                velocities, release_position,
            )

    diagnostics = dict(
        v0=float(v0), theta=float(theta), phi=float(phi),
        fit_usable=bool(fit_usable), used_mean_velocity_fallback=not bool(fit_usable),
        fit_nfev=int(trajectory_fit.nfev),
        fit_initial_cost=float(0.5 * np.sum(residual(initial_parameters) ** 2)),
        fit_final_cost=float(trajectory_fit.cost),
    )
    return trajectory, diagnostics


# ============================================================
# 2. INITIAL PLAYER STATE -> ARRIVALS ALONG THE PATH / AT A TARGET
# ============================================================

@jax.jit
def trajectory_arrival_times(player_positions, player_velocities, ball_xy_path):
    """TTI to each projected path point: (N,P,2),(N,P,2),(N,T,2) -> (N,T,P)."""
    def one_pass(positions, velocities, targets):
        return time_to_intercept(
            positions, velocities, targets, v_max=5.5, a_max=7.0, a_turn=7.0,
            preserve_initial_overspeed=True,
        ).T
    return jax.vmap(one_pass)(player_positions, player_velocities, ball_xy_path)


def prepare_pitch_control_inputs(initial_states):
    """Prepare inputs for pitch control at each candidate pass destination.

    Return player TTI (N,22), ball arrival time (N,) and player mask (N,22).
    This function prepares the physical inputs; it does not integrate control
    probabilities. That calculation is in integrate_pitch_control_logs.

    No observed future trajectory is used. Choose the hypothetical flight
    whose arrival best matches the nearest teammate's TTI at the target.
    Candidate speeds: 30 in [5,25] m/s; elevations: 30 in [0,45] degrees.
    """
    with precision_context(False):
        positions, velocities, targets, carriers = [
            jnp.asarray(initial_states[key], jnp.float32)
            for key in ["p_pos", "p_vel", "r_target", "x_bc"]
        ]
        def one_pass(positions, velocities, target):
            return time_to_intercept(
                positions, velocities, target[None, :], v_max=5.5, a_max=7.0,
                a_turn=7.0, preserve_initial_overspeed=True,
            )[:, 0]
        arrival_times = jax.jit(jax.vmap(one_pass))(positions, velocities, targets)
        # Exact-radius interpolation in the sampled physical flights (no 1m lookup).
        times = jnp.arange(0.0, 10.0 + 0.04, 0.04, dtype=jnp.float32)
        flights = generate_ball_trajectories(
            jnp.linspace(5.0, 25.0, 30, dtype=jnp.float32),
            jnp.linspace(0.0, jnp.pi / 4, 30, dtype=jnp.float32),
            times, **BALL_PARAMETERS,
        )
        target_distance = jnp.linalg.norm(targets - carriers, axis=-1)
        nearest_teammate = jnp.argmin(
            jnp.linalg.norm(positions[:, :10] - targets[:, None, :], axis=-1), axis=1,
        )
        desired_arrival = arrival_times[jnp.arange(len(positions)), nearest_teammate]

        def select_flight(distance, player_time):
            # 1. For each candidate flight k, interpolate the time t_k at
            # which its horizontal displacement reaches the target distance.
            ball_times = jax.vmap(lambda path: jnp.interp(distance, path[:, 0], times))(flights)

            # 2. Evaluate each flight's height z_k at that arrival time t_k.
            heights = jax.vmap(
                lambda path, hit_time: jnp.interp(hit_time, times, path[:, 1]),
            )(flights, ball_times)

            # 3. Keep flights that reach the target and have z_k <= 2m there.
            # Assign infinite arrival time to every unavailable candidate.
            reachable = (heights <= 2.0) & (distance <= flights[:, :, 0].max(axis=1))
            ball_times = jnp.where(reachable, ball_times, jnp.inf)

            # 4. Choose k* = argmin_k |t_k - TTI_receiver|.
            # Return its ball arrival time, not the receiver's arrival time.
            return ball_times[jnp.argmin(abs(ball_times - player_time))]

        # Apply the same selection to each pass, using its target distance
        # and the TTI of the teammate currently nearest that destination.
        ball_arrival_time = np.asarray(jax.jit(jax.vmap(select_flight))(
            target_distance, desired_arrival,
        ))
        valid_players = np.linalg.norm(np.asarray(positions), axis=-1) < 500
        valid_players[:, 10] = False
    if not np.isfinite(ball_arrival_time).all():
        raise ValueError("No valid hypothetical flight for an accepted destination.")
    return np.asarray(arrival_times), ball_arrival_time, valid_players


# ============================================================
# 3. EXPONENTIAL-EULER INTEGRATION OF COMPETING CONTROL
# ============================================================
# For each pass, track three probabilities:
#   U(t)         = ball still uncontrolled;
#   P_attack(t)  = attacking team has controlled the ball;
#   P_defense(t) = defending team has controlled the ball.
# Initially U=1 and P_attack=P_defense=0. Their sum remains 1.
#
# With individual rates r_i(t)=lambda_control*F_i(t), define
#   R_attack = sum_{i in attack} r_i, R_defense = sum_{i in defense} r_i,
#   R = R_attack + R_defense.
# Freeze these rates over one interval dt. Exponential Euler gives
#   dP_total   = U * (1 - exp(-dt*R)),
#   dP_attack  = dP_total * R_attack/R,
#   dP_defense = dP_total * R_defense/R,
#   U_next     = U * exp(-dt*R),
#   P_attack_next = P_attack + dP_attack (similarly for defense).
#
# The code below implements these equations IN LOG SPACE. This prevents
# tiny completion/failure probabilities from rounding to zero in the MLE.
# A zero probability is represented by -inf; log(1)=0.


def add_log_probabilities(log_P, log_dP):
    """Compute log(P+dP) from log(P),log(dP), without exponentiating them.

    logaddexp(a,b) = log(exp(a)+exp(b)). If P=dP=0, return -inf.
    The temporary zeros in that special case avoid undefined AD derivatives
    of logaddexp(-inf,-inf); they do not introduce any probability floor.
    """
    both_zero = jnp.isneginf(log_P) & jnp.isneginf(log_dP)
    log_sum = jnp.logaddexp(
        jnp.where(both_zero, 0.0, log_P),
        jnp.where(both_zero, 0.0, log_dP),
    )
    return jnp.where(both_zero, -jnp.inf, log_sum)


def pitch_control_step(log_control, t, player_tti, ball_can_be_controlled,
                       log_parameters, valid_players, dt):
    """One exponential-Euler step for a batch of N passes.

    log_control = (log_U, log_P_attack, log_P_defense), each shape (N,).
    player_tti and valid_players: (N,22); ball_can_be_controlled: (N,).
    This same update is used along a ball path and at a fixed destination.
    """
    log_U, log_P_attack, log_P_defense = log_control
    log_lambda_control, log_s = log_parameters

    # 1. Individual arrival probabilities F_i(t), in log space.
    # F_i(t) = sigmoid(pi*(t-TTI_i)/(sqrt(3)*s)), s=exp(log_s).
    log_F = jax.nn.log_sigmoid(
        jnp.pi * (t - player_tti) / (jnp.sqrt(3.0) * jnp.exp(log_s))
    )
    # Padded players and the passer cannot control: F_i=0, hence log_F_i=-inf.
    log_F = jnp.where(valid_players, log_F, -jnp.inf)

    # 2. Sum F_i within each team. axis=1 is the PLAYER axis, for every pass.
    # Players 0:10 are teammates, index 10 is the passer, 11:22 opponents.
    log_sum_F_attack = jax.scipy.special.logsumexp(log_F[:, :10], axis=1)
    log_sum_F_defense = jax.scipy.special.logsumexp(log_F[:, 11:], axis=1)
    log_sum_F = jnp.logaddexp(log_sum_F_attack, log_sum_F_defense)

    # 3. Total rate R=lambda_control*sum_i F_i; integrated rate H=dt*R.
    # log(H) = log(dt) + log(lambda_control) + log(sum_i F_i).
    log_H = jnp.log(dt) + log_lambda_control + log_sum_F

    # 4. Fraction of U captured during this interval: f=1-exp(-H).
    # expm1 computes exp(x)-1 accurately near zero, so f=-expm1(-H).
    # For H<exp(-20), use log(f)~log(H). This is small-H
    # approximation; bounding the inactive expression keeps AD finite.
    log_captured_fraction = jnp.where(
        log_H < -20.0,
        log_H,
        jnp.log(-jnp.expm1(-jnp.exp(jnp.maximum(log_H, -20.0)))),
    )
    # dP_total=U*f. When control is inactive, dP_total=0 (log=-inf).
    log_dP_total = jnp.where(
        ball_can_be_controlled, log_U + log_captured_fraction, -jnp.inf,
    )

    # 5. Share the captured mass between teams in proportion to their rates.
    # R_attack/R = sum_attack F_i / sum_all F_i; lambda_control cancels.
    # log(dP_attack) = log(dP_total) + log(sum_attack F) - log(sum_all F).
    log_dP_attack = log_dP_total + log_sum_F_attack - log_sum_F
    log_dP_defense = log_dP_total + log_sum_F_defense - log_sum_F

    # 6. Update U and the accumulated team probabilities.
    # log(U_next) = log(U)-H; leave U unchanged while control is inactive.
    next_log_U = log_U - jnp.where(ball_can_be_controlled, jnp.exp(log_H), 0.0)
    # P_team_next = P_team + dP_team, evaluated as a stable sum in log space.
    next_log_P_attack = add_log_probabilities(log_P_attack, log_dP_attack)
    next_log_P_defense = add_log_probabilities(log_P_defense, log_dP_defense)

    return next_log_U, next_log_P_attack, next_log_P_defense


@partial(jax.jit, static_argnames=("dt",))
def integrate_pass_probability_logs(log_parameters, path_tti, ball_low,
                                    valid_players, dt=0.04):
    """Integrate control along the reconstructed ball path used by the MLE.

    path_tti: (N,T,22), TTI to each successive ball position.
    ball_low: (N,T), True when ball height <=2m, allowing control.
    valid_players: (N,22), excluding padding and the passer.
    Return log(success), log(failure), log(uncontrolled), each shape (N,).
    Preserve all 251 left steps at 0,...,10s of the accepted fit.
    """
    n_passes, n_times, _ = path_tti.shape
    times = jnp.arange(n_times, dtype=jnp.float32) * jnp.float32(dt)

    # Initial probabilities: U(0)=1, P_attack(0)=P_defense(0)=0.
    log_U0 = jnp.zeros(n_passes)
    log_P_attack0 = jnp.full(n_passes, -jnp.inf)
    log_P_defense0 = jnp.full(n_passes, -jnp.inf)
    initial_control = (log_U0, log_P_attack0, log_P_defense0)

    # scan iterates over its first axis. Put time first:
    # path_tti (N,T,22) -> (T,N,22), ball_low (N,T) -> (T,N).
    tti_by_time = jnp.swapaxes(path_tti, 0, 1)
    ball_low_by_time = jnp.swapaxes(ball_low, 0, 1)

    def step_at_time(log_control, time_inputs):
        t, player_tti, ball_can_be_controlled = time_inputs
        next_control = pitch_control_step(
            log_control, t, player_tti, ball_can_be_controlled,
            log_parameters, valid_players, dt,
        )
        return next_control, None

    # A compiled chronological loop: each step receives the previous state.
    # checkpoint only reduces memory during differentiation of the MLE.
    (log_U, log_P_attack, log_P_defense), _ = jax.lax.scan(
        jax.checkpoint(step_at_time), initial_control,
        (times, tti_by_time, ball_low_by_time),
    )

    # Success = attacking control; failure = opponent control OR no control.
    log_success = log_P_attack
    log_failure = jnp.logaddexp(log_P_defense, log_U)
    return log_success, log_failure, log_U


@partial(jax.jit, static_argnames=("dt",))
def integrate_pitch_control_logs(log_parameters, player_tti, ball_arrival_time,
                                 valid_players, dt=0.01):
    """Integrate pitch control at the fixed candidate destination until 15s.

    player_tti: (N,22), fixed TTI to the destination from the release state.
    ball_arrival_time: (N,), hypothetical flight arrival at that destination.
    Unlike the reconstructed-path calculation, the target does not move and
    control starts only when t >= ball_arrival_time. Same update equations.
    Return log(success), log(failure), log(uncontrolled), each shape (N,).
    """
    n_passes = len(player_tti)
    n_times = int(round(15 / dt))
    times = jnp.arange(n_times, dtype=jnp.float32) * jnp.float32(dt)

    # Initial probabilities: U(0)=1, P_attack(0)=P_defense(0)=0.
    log_U0 = jnp.zeros(n_passes)
    log_P_attack0 = jnp.full(n_passes, -jnp.inf)
    log_P_defense0 = jnp.full(n_passes, -jnp.inf)
    initial_control = (log_U0, log_P_attack0, log_P_defense0)

    def step_at_time(log_control, t):
        # Before the ball arrives, all control probabilities stay unchanged.
        ball_can_be_controlled = t >= ball_arrival_time
        next_control = pitch_control_step(
            log_control, t, player_tti, ball_can_be_controlled,
            log_parameters, valid_players, dt,
        )
        return next_control, None

    (log_U, log_P_attack, log_P_defense), _ = jax.lax.scan(
        jax.checkpoint(step_at_time), initial_control, times,
    )

    log_success = log_P_attack
    log_failure = jnp.logaddexp(log_P_defense, log_U)
    return log_success, log_failure, log_U


@jax.jit
def negative_log_likelihood(log_parameters, path_arrivals, ball_low, valid_players, outcome):
    """Natural Bernoulli NLL: -sum[y*log(P_attack)+(1-y)*log(P_not_attack)].

    No prior, probability floor, parameter cap or Jacobian is added. Log
    coordinates only reparameterize this MLE: lambda,s remain positive.
    """
    log_success, log_failure, _ = integrate_pass_probability_logs(
        log_parameters, path_arrivals, ball_low, valid_players,
    )
    return -jnp.sum(jnp.where(outcome == 1, log_success, log_failure))
