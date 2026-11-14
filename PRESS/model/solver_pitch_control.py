"""Competing-control solvers with a shared, mass-preserving survival budget.

Exponential Euler is exact for hazards frozen over each interval. It remains
first order for time-varying arrival gates. Euler is available as an explicit sensitivity mode.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

from PRESS.physics.arrival_probabilities import F
from PRESS.physics.times import time_to_intercept


def _valid_players(p_pos, ball_carrier_idx=None):
    valid = jnp.isfinite(p_pos).all(axis=-1) & (jnp.linalg.norm(p_pos, axis=-1) < 500.0)
    if ball_carrier_idx is not None:
        valid = valid.at[ball_carrier_idx].set(False)
    return valid


def _control_update(P, gates, lambda_control, dt, integration_method):
    rates = lambda_control * gates
    survival = jnp.clip(1.0 - jnp.sum(P, axis=0, keepdims=True), 0.0, 1.0)
    if integration_method == 'euler':
        return P + dt * survival * rates
    if integration_method != 'exponential':
        raise ValueError("integration_method must be 'exponential' or 'euler'.")
    total = jnp.sum(rates, axis=0, keepdims=True)
    mass = survival * -jnp.expm1(-dt * total)
    return P + mass * rates / jnp.where(total > 0.0, total, 1.0)


def pass_prob_step(
    P,
    t,
    p_pos,
    p_vel,
    x_ball,
    lambda_control,
    s,
    v_max,
    a_max,
    dt,
    z_max,
    ball_carrier_idx=None,
    integration_method='exponential',
    *,
    a_turn=None,
    preserve_initial_overspeed=True,
):
    """One frozen-hazard step along a supplied 3D ball path.

    The carrier and finite-coordinate padding are excluded BEFORE allocating
    shared control mass. No control is possible while ball height exceeds z_max.
    """
    gates = F(
        p_pos,
        p_vel,
        x_ball[:2][None, :],
        t,
        s,
        v_max,
        a_max,
        a_turn=a_turn,
        preserve_initial_overspeed=preserve_initial_overspeed,
    ).squeeze(-1)
    gates = jnp.where(_valid_players(p_pos, ball_carrier_idx) & (x_ball[2] <= z_max), gates, 0.0)
    return _control_update(P, gates, lambda_control, dt, integration_method)


def integrate_pass_prob(
    p_pos,
    p_vel,
    attacker_idx,
    ball_carrier_idx,
    ball_trajectory,
    lambda_control=4.3,
    s=0.45,
    v_max=5.0,
    a_max=5.0,
    dt=0.04,
    z_max=2.0,
    tol=1e-06,
    integration_method='exponential',
    *,
    a_turn=None,
    preserve_initial_overspeed=True,
):
    """Attacking control along T ball positions sampled dt seconds apart.

    Each supplied position starts one integration interval, retaining the
    benchmark convention (251 samples at 0..10s imply 251 left steps).
    tol is retained for API compatibility; the full schedule is integrated
    without a parameter-dependent stopping rule. Valid rates and scales must
    be finite and nonnegative/positive respectively.
    """
    P0 = jnp.zeros(
        p_pos.shape[0],
        dtype=jnp.result_type(p_pos, p_vel, ball_trajectory, lambda_control, s),
    )
    times = jnp.arange(ball_trajectory.shape[0], dtype=jnp.float32) * jnp.float32(dt)

    def step(P, inputs):
        t, ball = inputs
        return (
            pass_prob_step(
                P,
                t,
                p_pos,
                p_vel,
                ball,
                lambda_control,
                s,
                v_max,
                a_max,
                dt,
                z_max,
                ball_carrier_idx,
                integration_method,
                a_turn=a_turn,
                preserve_initial_overspeed=preserve_initial_overspeed,
            ),
            None,
        )

    P, _ = jax.lax.scan(step, P0, (times, ball_trajectory))
    return jnp.sum(P[attacker_idx])


def ppcf_step(
    P,
    t,
    p_pos,
    p_vel,
    r_grid,
    T_ball,
    lambda_control,
    s,
    v_max,
    a_max,
    dt,
    ball_carrier_idx=None,
    integration_method='exponential',
    *,
    a_turn=None,
    preserve_initial_overspeed=True,
):
    """One shared-survival control update at hypothetical target locations."""
    gates = F(
        p_pos,
        p_vel,
        r_grid,
        t,
        s,
        v_max,
        a_max,
        a_turn=a_turn,
        preserve_initial_overspeed=preserve_initial_overspeed,
    )
    gates = jnp.where(
        _valid_players(p_pos, ball_carrier_idx)[:, None] & (t >= jnp.asarray(T_ball)[None, :]),
        gates,
        0.0,
    )
    return _control_update(P, gates, lambda_control, dt, integration_method)


def integrate_pitch_control(
    r_grid,
    p_pos,
    p_vel,
    ball_carrier_idx,
    T_ball,
    lambda_control=4.3,
    s=0.45,
    v_max=5.0,
    a_max=5.0,
    t_max_pc=8.0,
    dt_pc=0.01,
    tol=1e-06,
    integration_method='exponential',
    *,
    a_turn=None,
    preserve_initial_overspeed=True,
):
    """Player-specific control, shape (n_players,n_targets).

    Exponential updates preserve sum(control)<=1 for any finite nonnegative
    rates. Infinite ball-arrival time gives zero control. A partial last step
    respects noninteger horizons; tol no longer changes the integration path.
    The carrier is excluded before calculating the total hazard.
    """
    tau = time_to_intercept(
        p_pos,
        p_vel,
        r_grid,
        v_max=v_max,
        a_max=a_max,
        a_turn=a_turn,
        preserve_initial_overspeed=preserve_initial_overspeed,
    )
    valid = jnp.broadcast_to(_valid_players(p_pos, ball_carrier_idx)[:, None], tau.shape)
    return control_from_arrival_times(
        tau,
        T_ball,
        valid,
        lambda_control,
        s,
        t_max_pc,
        dt_pc,
        integration_method,
    )


def control_from_arrival_times(
    tau,
    T_ball,
    valid,
    lambda_control=4.3,
    s=0.45,
    t_max_pc=15.0,
    dt_pc=0.01,
    integration_method='exponential',
):
    """Shared-survival integration from precomputed (players,targets) TTI.

    Valid masks must already exclude the carrier. This also supports many
    independent single-target examples by placing them on the target axis.
    """
    if dt_pc <= 0 or t_max_pc < 0:
        raise ValueError('Require dt_pc>0 and t_max_pc>=0.')
    steps = max(0, math.ceil(t_max_pc / dt_pc - 1e-12))
    times = jnp.arange(steps, dtype=jnp.float32) * jnp.float32(dt_pc)
    widths = jnp.asarray(np.minimum(dt_pc, np.maximum(t_max_pc - np.arange(steps) * dt_pc, 0.0)))
    P0 = jnp.zeros(tau.shape, dtype=jnp.result_type(tau, lambda_control, s, widths))

    def step(P, inputs):
        t, width = inputs
        gates = jax.nn.sigmoid(jnp.pi / (jnp.sqrt(3.0) * s) * (t - tau))
        gates = jnp.where(valid & (t >= jnp.asarray(T_ball)[None, :]), gates, 0.0)
        return (_control_update(P, gates, lambda_control, width, integration_method), None)

    return jax.lax.scan(step, P0, (times, widths))[0]


def isotonic_scaling(probabilities, x_thresholds, y_thresholds):
    """Apply saved validation/OOF-fitted isotonic knots without fitting.

    Linear interpolation and endpoint clipping reproduce sklearn's increasing
    IsotonicRegression(out_of_bounds='clip'). One knot represents a constant.
    """
    x = np.asarray(x_thresholds, float)
    y = np.asarray(y_thresholds, float)
    if (
        x.ndim != 1
        or x.shape != y.shape
        or (not len(x))
        or (not np.isfinite(x).all())
        or (not np.isfinite(y).all())
        or np.any(np.diff(x) <= 0)
        or np.any(np.diff(y) < 0)
        or np.any((y < 0) | (y > 1))
    ):
        raise ValueError('Invalid increasing isotonic thresholds.')
    p = np.asarray(probabilities, float)
    if not np.isfinite(p).all():
        raise ValueError('Probabilities must be finite.')
    return np.interp(p, x, y)


def Platt_scaling(P_attackers, a=1.96, b=0.42):
    """Legacy explicit calibration helper; final pass-risk uses isotonic."""
    p = jnp.clip(P_attackers, 1e-06, 1 - 1e-06)
    return jax.nn.sigmoid(a * (jnp.log(p) - jnp.log1p(-p)) + b)
