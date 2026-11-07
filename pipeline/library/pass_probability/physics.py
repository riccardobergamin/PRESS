"""Cache arrival times and hypothetical flight with frozen historical conventions."""
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from PRESS.physics.flight_ball import generate_ball_trajectories, time_flight
from PRESS.physics.times import time_to_intercept
from .precision import enable_x64
FLIGHT = dict(t_max_flight=10.0,
    dt_ball=0.04,
    grid_step=1.0,
    z_max=2.0,
    rho=1.22,
    Cd=0.25,
    A=0.038,
    m=0.42,
    g=9.81)


def candidates():
    return (jnp.linspace(5.0, 25.0, 30, dtype=jnp.float32),
        jnp.linspace(0.0, jnp.pi / 4, 30, dtype=jnp.float32))
TTI_SETTINGS = dict(a_turn=7.0, preserve_initial_overspeed=True)

@partial(jax.jit, static_argnames=('vmax', 'a_turn', 'preserve_initial_overspeed'))


def arrival_batch(pos, vel, target, *, vmax, a_turn=None, preserve_initial_overspeed=False):
    return jax.vmap(lambda p,
        v,
        r: time_to_intercept(p, v, r[None], v_max=vmax, a_max=7.0, a_turn=a_turn, preserve_initial_overspeed=preserve_initial_overspeed).squeeze(-1))(pos,
        vel,
        target)
