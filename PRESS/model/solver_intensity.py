import jax
import jax.numpy as jnp
import math
from PRESS.physics.arrival_probabilities import F
from PRESS.physics.times import rotation_time


def _carry_hazard_components(
    t,
    p_pos,
    p_vel,
    x_bc0,
    v_bc,
    lambda_0,
    beta_v,
    beta_B,
    s=0.45,
    v_max=5.0,
    a_max=7.0,
    pitch_length=105.0,
    pitch_width=68.0,
    a_turn=None,
    preserve_initial_overspeed=False,
    boundary_mode="absolute",
):

    bounds = jnp.array(
        [
            [-pitch_length / 2, pitch_length / 2],
            [-pitch_width / 2, pitch_width / 2],
        ]
    )

    # --------------------------------------------------------
    # Ball carrier position
    # --------------------------------------------------------

    x_bc = (
        x_bc0
        + v_bc * t
    )

    v_bc_norm = jnp.linalg.norm(
        v_bc
    )

    # ========================================================
    # REAL DEFENDERS
    # ========================================================

    F_def = F(
        p_pos,
        p_vel,
        x_bc[None, :],
        t,
        s,
        v_max,
        a_max,
        a_turn=a_turn,
        preserve_initial_overspeed=preserve_initial_overspeed,
    ).squeeze(-1)


    # --------------------------------------------------------
    # Boundary distance
    #
    # Minimum distance from BC to any pitch boundary.
    # --------------------------------------------------------

    margins = jnp.array([
        x_bc[0] - bounds[0, 0], bounds[0, 1] - x_bc[0],
        x_bc[1] - bounds[1, 0], bounds[1, 1] - x_bc[1],
    ])
    if boundary_mode == "absolute":
        x_B = jnp.min(jnp.abs(margins))
    elif boundary_mode == "signed":
        x_B = jnp.min(margins)
    elif boundary_mode == "clipped":
        x_B = jnp.maximum(jnp.min(margins), 0.)
    else:
        raise ValueError("boundary_mode must be absolute, signed or clipped.")

    # --------------------------------------------------------
    # Proportional hazard
    # --------------------------------------------------------

    linear_predictor = beta_v*v_bc_norm + beta_B*x_B
    log_rate = jnp.log(jnp.maximum(lambda_0, 1e-30)) + linear_predictor
    return F_def, log_rate, linear_predictor


def _carry_step_exposure(F_def, log_rate, dt, lambda_0):
    """Integrated total hazard for the same frozen step as the cause solver."""
    availability = jnp.sum(F_def)
    log_exposure = log_rate + jnp.log(jnp.maximum(availability, 1e-30)) + jnp.log(jnp.maximum(dt, 1e-30))
    exposure = jnp.exp(jnp.minimum(log_exposure, 20.))
    return jnp.where((dt > 0) & (lambda_0 > 0) & (availability > 0), exposure, 0.)


def _integrate_log_survival(
    p_pos, p_vel, x_bc0, v_bc0, delta_t, lambda_0, beta_v, beta_B,
    s=0.45, v_max=5.0, a_max=7.0, dt=0.04,
    pitch_length=105.0, pitch_width=68.0, max_time=None,
):
    """Internal exponential-Euler log survival for temporal training likelihood.

    Accumulate -integral(H) using the cause solver's identical gates, rates,
    left endpoints and partial final steps; avoids subtracting loss from one.
    This does not implement the historical Euler observation approximation.
    """
    n_steps = math.ceil((delta_t if max_time is None else max_time)/dt)
    def scan(log_survival, t):
        gates, log_rate, _ = _carry_hazard_components(
            t, p_pos, p_vel, x_bc0, v_bc0, lambda_0, beta_v, beta_B,
            s, v_max, a_max, pitch_length, pitch_width)
        exposure = _carry_step_exposure(gates, log_rate, jnp.clip(delta_t-t, 0., dt), lambda_0)
        return log_survival-exposure, None
    return jax.lax.scan(scan, jnp.asarray(0.), jnp.arange(n_steps)*dt)[0]


def loss_prob_step(
    P, t, p_pos, p_vel, x_bc0, v_bc, lambda_0, beta_v, beta_B,
    s=0.45, dt=0.04, v_max=5.0, a_max=7.0,
    pitch_length=105.0, pitch_width=68.0, integration_method="exponential",
    a_turn=None, preserve_initial_overspeed=False, boundary_mode="absolute",
):
    F_def, log_rate, linear_predictor = _carry_hazard_components(
        t, p_pos, p_vel, x_bc0, v_bc, lambda_0, beta_v, beta_B,
        s, v_max, a_max, pitch_length, pitch_width,
        a_turn, preserve_initial_overspeed, boundary_mode)
    U = jnp.clip(1.0-jnp.sum(P), 0.0, 1.0)

    # --------------------------------------------------------
    # Defensive loss probability increment
    # --------------------------------------------------------

    if integration_method == "exponential":
        # Exact competing-risk update with hazards frozen during this step.
        # The common rate cancels from the cause weights. Compute exposure in
        # log space to avoid exp(eta) overflow during prior/optimizer searches.
        availability = jnp.sum(F_def)
        safe_availability = jnp.maximum(availability, 1e-30)
        exposure = _carry_step_exposure(F_def, log_rate, dt, lambda_0)
        mass = U * (-jnp.expm1(-exposure))
        mass = jnp.where((dt > 0) & (lambda_0 > 0), mass, 0.)
        return P + mass * F_def / safe_availability
    if integration_method != "euler":
        raise ValueError("integration_method must be 'exponential' or 'euler'.")
    # Explicit historical mode retains its overflow/clipping behavior.
    dP = U * F_def * lambda_0 * jnp.exp(linear_predictor)

    # --------------------------------------------------------
    # Update
    # --------------------------------------------------------

    return jnp.clip(
        P + dt * dP,
        0.0,
        1.0,
    )


def integrate_loss_prob(
    p_pos,
    p_vel,
    x_bc0,
    v_bc0,
    delta_t,
    lambda_0,
    beta_v,
    beta_B,
    s=0.45,
    v_max=5.0,
    a_max=7.0,
    dt=0.04,
    tol=1e-6,
    pitch_length=105.0,
    pitch_width=68.0,
    max_time=None,
    integration_method="exponential",
    a_turn=None,
    preserve_initial_overspeed=False,
    boundary_mode="absolute",
):
    """Integrate defender-specific carry-loss probabilities over delta_t seconds.

    p_pos and p_vel have shape (n_defenders, 2); x_bc0 and v_bc0
    have shape (2,). Positions are metres, velocities m/s. lambda_0,
    beta_v and beta_B parameterize defensive intensity; dt is the
    integration step in seconds. Returns shape (n_defenders,), whose
    sum is total drive-loss probability. This low-level solver expects
    valid numerical inputs; use PRESS.compute_ball_drive_loss for frames.
    Exponential Euler preserves shared survival mass for nonnegative finite
    rates, with hazards frozen at each step. 'euler' reproduces the historical
    update and can overshoot total probability for large rates.
    a_turn=None ties the effective turn acceleration to a_max. Explicit
    preserve_initial_overspeed=True retains v0>v_max without accelerating.
    boundary_mode='absolute' reproduces historical B; 'signed' uses the
    minimum inward line margin, and 'clipped' saturates that margin at zero.
    Changing these conventions requires coherently fitted model parameters;
    in particular signed distances are not a validated extrapolation of
    existing weights. No mode stops the BC or turns an exit into sure loss.
    """

    # A fixed maximum schedule makes row-specific durations vmappable/JIT-safe.
    # Without max_time, delta_t must be a static scalar as in frame inference.
    n_steps = math.ceil((delta_t if max_time is None else max_time) / dt)

    time_steps = (
        jnp.arange(n_steps)
        * dt
    )

    P0 = jnp.zeros(
        p_pos.shape[0],
    )

    def scan_body(P, t):
        step_dt = jnp.clip(delta_t - t, 0., dt)

        P_next = loss_prob_step(
            P=P,
            t=t,
            p_pos=p_pos,
            p_vel=p_vel,
            x_bc0=x_bc0,
            v_bc=v_bc0,
            lambda_0=lambda_0,
            beta_v=beta_v,
            beta_B=beta_B,
            s=s,
            dt=step_dt,
            v_max=v_max,
            a_max=a_max,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            integration_method=integration_method,
            a_turn=a_turn,
            preserve_initial_overspeed=preserve_initial_overspeed,
            boundary_mode=boundary_mode,
        )

        active = (
            jnp.sum(P)
            < 1.0 - tol
        ) & (step_dt > 0)

        P = jnp.where(
            active,
            P_next,
            P,
        )

        return P, None

    P_final, _ = jax.lax.scan(
        scan_body,
        P0,
        time_steps,
    )

    return P_final
