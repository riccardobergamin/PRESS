"""Verified BCOS64 conditional hazards; shared exponential survival integration."""

import math
from functools import lru_cache

import numpy as np
import jax
import jax.numpy as jnp

from PRESS.physics.bcos_path import build_path
from PRESS.physics.times import time_to_intercept


def arrival_times(position, velocity, targets, vmax, acceleration, turn):
    """Evaluate the shared TTI convention with initial overspeed preserved."""
    return time_to_intercept(
        jnp.asarray(position, jnp.float32),
        jnp.asarray(velocity, jnp.float32),
        jnp.asarray(targets, jnp.float32),
        v_max=jnp.float32(vmax),
        a_max=jnp.float32(acceleration),
        a_turn=jnp.float32(turn),
        preserve_initial_overspeed=True,
    )


@lru_cache(maxsize=16)
def make_adaptive_kernel(dx, *, radius=50.0, eps=1e-07, dt=0.04, horizon=15.0):
    """Stop only when remaining possible contribution is bounded by eps.

    At each target, additional carrier incidence cannot exceed remaining U.
    This changes computation, not the local hazards or spatial quadrature.
    """
    slots = math.ceil(radius / dx) + 2
    nsteps = round(horizon / dt)

    @jax.jit
    def point(target, bc, velocity, defenders, def_velocity, configuration):
        s, vmax_bc, vmax_def, a_bc, a_def, turn_bc, turn_def = configuration
        path, mask = build_path(target, bc, MAX_POINTS=slots, dx=dx)
        tau_bc = arrival_times(bc, velocity, path, vmax_bc, a_bc, turn_bc).ravel()
        tau_def = arrival_times(defenders, def_velocity, path, vmax_def, a_def, turn_def)
        valid_def = jnp.linalg.norm(defenders, axis=1) < 500
        target_idx = jnp.maximum(mask.sum().astype(jnp.int32) - 1, 0)
        alpha = jnp.pi / (jnp.sqrt(3.0) * s)
        q_bc = jnp.exp(-alpha * tau_bc)
        q_def = jnp.exp(-alpha * tau_def)

        # State: unclaimed mass, carrier incidence, defender incidences, step.
        def condition(state):
            u, c, d, n = state
            return (u > eps) & (n < nsteps)

        def step(state):
            u, c, d, n = state
            t = n * dt
            qt = jnp.exp(-alpha * t)
            raw_bc = q_bc / (q_bc + qt)
            fbc = raw_bc.at[0].set(1.0)
            # Spatial carrier weights are adjacent gate differences, not time derivatives.
            weights = jnp.append(jnp.clip(fbc[:-1] - fbc[1:], 0.0, 1.0), 0.0) * mask
            weights = jnp.where(n == 0, jnp.zeros_like(mask).at[0].set(1.0), weights)
            hc = alpha * raw_bc[target_idx]
            hd = alpha * jnp.sum(q_def / (q_def + qt) * weights[None], axis=1) * valid_def
            total = hc + hd.sum()
            denom = jnp.where(total > 0, total, 1.0)
            released = u * -jnp.expm1(-dt * total)
            return (
                u * jnp.exp(-dt * total),
                c + released * hc / denom,
                d + released * hd / denom,
                n + 1,
            )

        initial = (jnp.array(1.0), jnp.array(0.0), jnp.zeros(len(defenders)), jnp.array(0))
        u, c, d, n = jax.lax.while_loop(condition, step, initial)
        return (c, d.sum(), u, n)

    return jax.jit(jax.vmap(point, in_axes=(0, None, None, None, None, None)))


def compute_bcos_state(
    position,
    velocity,
    *,
    pitch_length=105.0,
    pitch_width=68.0,
    r_max=40.0,
    step=1.0,
    dx=0.03125,
    dt=0.04,
    T_max=15.0,
    stop_eps=1e-09,
    configuration=None,
    targets=None,
):
    """Return field, residual mass and probability-weighted area for one state.

    configuration=(s,vmaxBC,vmaxDef,aBC,aDef,turnBC,turnDef). The model
    acceleration is bounded; observed acceleration is not an input. Targets
    default to carrier-relative metre grid points within the pitch and radius.
    Spatial BC weights are differences of arrival gates, not a differentiated
    product interpreted as an interception CDF.
    """
    position = np.asarray(position)
    velocity = np.asarray(velocity)
    if (
        position.shape != velocity.shape
        or position.shape != (22, 2)
        or (not np.isfinite([position, velocity]).all())
    ):
        raise ValueError('Expected finite 22x2 position and velocity arrays.')
    if configuration is None:
        from .ball_drive import BallDriveModel
        model = BallDriveModel.pretrained().to_dict()
        p = model['physics']
        configuration = [
            np.exp(model['theta'][3]),
            6.6,
            p['v_max'],
            7.0,
            p['a_linear'],
            7.0,
            p['a_turn'],
        ]
    cfg = np.asarray(configuration, float)
    if cfg.shape != (7,) or not np.isfinite(cfg).all() or (cfg <= 0).any():
        raise ValueError('Invalid BCOS configuration.')
    if not all((np.isfinite(z) and z > 0 for z in [
        r_max,
        step,
        dx,
        dt,
        T_max,
        stop_eps,
        pitch_length,
        pitch_width,
    ])):
        raise ValueError('BCOS domain and integration scales must be positive finite values.')
    if abs(round(T_max / dt) * dt - T_max) > 1e-10:
        raise ValueError('T_max must be a multiple of dt.')
    if targets is None:
        axis = np.arange(-r_max, r_max + 1e-05, step)
        xx, yy = np.meshgrid(axis, axis)
        valid = xx ** 2 + yy ** 2 <= r_max ** 2
        targets = np.column_stack([xx[valid], yy[valid]]) + position[10]
        targets = targets[(abs(targets[:, 0]) <= pitch_length / 2) & (abs(targets[:, 1]) <= pitch_width / 2)]
    targets = np.asarray(targets)
    if (
        targets.ndim != 2
        or targets.shape[1] != 2
        or (not len(targets))
        or (not np.isfinite(targets).all())
    ):
        raise ValueError('Expected nonempty finite BCOS targets.')
    if np.any(np.linalg.norm(targets - position[10], axis=1) > r_max + 1e-05):
        raise ValueError('Target outside BCOS domain.')
    context = getattr(jax, 'enable_x64', None)
    if context is None:
        from jax.experimental import enable_x64 as context
    values = []
    defence = []
    survival = []
    steps = []
    with context(True):
        kernel = make_adaptive_kernel(dx, radius=r_max, eps=stop_eps, dt=dt, horizon=T_max)
        for start in range(0, len(targets), 16):
            block = targets[start:start + 16]
            n = len(block)
            if n < 16:
                block = np.pad(block, ((0, 16 - n), (0, 0)), mode='edge')
            c, d, u, k = map(
                np.asarray,
                kernel(block, position[10], velocity[10], position[11:], velocity[11:], cfg),
            )
            values.append(c[:n])
            defence.append(d[:n])
            survival.append(u[:n])
            steps.append(k[:n])
    c, d, u, k = map(np.concatenate, [values, defence, survival, steps])
    error = float(np.max(abs(c + d + u - 1)))
    if not np.isfinite(c).all() or (c < 0).any() or (c > 1).any() or (error > 2e-12):
        raise ArithmeticError('Invalid BCOS probability partition.')
    return {
        'Ball carrier open space': c,
        'Defensive control': d,
        'Unresolved probability': u,
        'BCOS area': float(c.sum() * step ** 2),
        'grid': targets,
        'integration_steps': k,
        'max_mass_error': error,
        'params': dict(
            r_max=r_max,
            step=step,
            dx=dx,
            dt=dt,
            T_max=T_max,
            stop_eps=stop_eps,
            configuration=cfg.tolist(),
            integration='exponential',
            preserve_initial_overspeed=True,
        ),
    }
