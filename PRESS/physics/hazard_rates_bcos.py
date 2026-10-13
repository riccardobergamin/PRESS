"""
Survival analysis utilities.
Defines hazard functions associated with
arrival-time distributions.
"""
import jax
import jax.numpy as jnp
from .arrival_probabilities import F, defender_F


# closed form h = a*F, valid for a single logistic CDF (f = a*F*(1-F), h = f/(1-F) = a*F)


def ballcarrier_hazard(
    x_bc0,          # (dim,)  ball carrier position
    v_bc0,          # (dim,)  ball carrier velocity
    r_target,   # (dim,)  single target point
    t,          # scalar
    s,
    v_max_bc,
    a_max_bc,
):
  
    a = jnp.pi / (jnp.sqrt(3) * s)
    return a * F(x_bc0, v_bc0, r_target, t, s, v_max_bc, a_max_bc).squeeze()


def defender_hazard(
    x_bc0,      # (dim,)          ball carrier starting position
    v_bc0,      # (dim,)          ball carrier starting velocity
    r_def,      # (N_def, dim)    defenders positions
    v_def,      # (N_def, dim)    defenders velocities
    x_path,     # (M, dim)        path points
    mask,       # (M,)            valid path points
    t,          # scalar
    s,
    v_def_max,
    v_bc_max,
    a_def_max,
    a_bc_max,
):

    a = jnp.pi / (jnp.sqrt(3) * s)

    return a * defender_F(
        x_bc0,
        v_bc0,
        r_def,
        v_def,
        x_path,
        mask,
        t,
        s,
        v_def_max,
        v_bc_max,
        a_def_max,
        a_bc_max,
    )

