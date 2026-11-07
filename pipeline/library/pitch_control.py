"""Batched equivalent of the frozen endpoint PPCF, sharing arrival geometry."""
import jax
import jax.numpy as jnp
from .pass_probability.precision import enable_x64
import numpy as np
from PRESS.physics.flight_ball import generate_ball_trajectories, exact_ball_arrival_times
from PRESS.physics.times import time_to_intercept


def geometry_function(p):
    with enable_x64(False):
        times = jnp.arange(0.0, p['t_max_flight'] + p['dt_ball'], p['dt_ball'], dtype=jnp.float32)
        paths = generate_ball_trajectories(jnp.linspace(*p['speed_range'], p['speed_candidates'], dtype=jnp.float32),
            jnp.linspace(*np.deg2rad(p['elevation_degrees']), p['elevation_candidates'], dtype=jnp.float32),
            times,
            **{k: p[k] for k in ['rho', 'Cd', 'A', 'm', 'g']})

    @jax.jit
    def call(pos, vel, bc, targets):
        with enable_x64(False):
            pos, vel, bc, targets = [jnp.asarray(a, jnp.float32) for a in [pos, vel, bc, targets]]
            tau = time_to_intercept(pos,
                vel,
                targets,
                v_max=p['v_max'],
                a_max=p['a_max'],
                a_turn=p['a_turn'],
                preserve_initial_overspeed=p['preserve_initial_overspeed'])
            nearest = jnp.argmin(jnp.linalg.norm(pos[:10, None, :] - targets[None, :, :], axis=-1), axis=0)
            desired = tau[nearest, jnp.arange(len(targets))]
            ball = exact_ball_arrival_times(jnp.linalg.norm(targets - bc, axis=-1),
                desired,
                paths,
                times,
                p['z_max'])
        return (tau.T, ball)
    return call


def control_one(theta, tau, ball, valid):
    rate, s = (jnp.exp(theta[0]), jnp.exp(theta[1]))
    tau = jnp.asarray(tau, jnp.float64)
    times = jnp.arange(1500, dtype=jnp.float32) * jnp.float32(0.01)

    def step(state, t):
        survival, attack = state
        gates = jax.nn.sigmoid(jnp.pi / (jnp.sqrt(3.0) * s) * (t - tau))
        gates = jnp.where(valid, gates, 0.0)
        ga = gates[:, :10].sum(axis=1)
        total = ga + gates[:, 11:].sum(axis=1)
        exposure = jnp.where(t >= ball, 0.01 * rate * total, 0.0)
        mass = survival * -jnp.expm1(-exposure)
        return ((survival * jnp.exp(-exposure), attack + mass * ga / jnp.where(total > 0, total, 1.0)), None)
    state = (jnp.ones(len(tau), jnp.float64), jnp.zeros(len(tau), jnp.float64))
    return jax.lax.scan(step, state, times)[0][1]
control = jax.jit(control_one)
control_many = jax.jit(jax.vmap(control_one, in_axes=(0, None, None, None)))


def endpoint_scores(raw, theta, geometry, batch=128):
    """Actual observed target is used only for fitting completion transforms."""
    result = []
    call = jax.jit(jax.vmap(lambda p, v, x, r: geometry(p, v, x, r[None, :]), in_axes=(0, 0, 0, 0)))
    for first in range(0, len(raw['x_bc']), batch):
        take = np.arange(first, min(first + batch, len(raw['x_bc'])))
        n = len(take)
        if n < batch:
            take = np.pad(take, (0, batch - n), mode='edge')
        tau, ball = call(*[jnp.asarray(raw[k][take]) for k in ['p_pos', 'p_vel', 'x_bc', 'r_target']])
        valid = np.linalg.norm(raw['p_pos'][take], axis=-1) < 500
        valid[:, 10] = False
        p = np.asarray(control(jnp.asarray(theta), tau[:, 0, :], ball[:, 0], jnp.asarray(valid)))
        result.append(p[:n])
    return np.concatenate(result)
