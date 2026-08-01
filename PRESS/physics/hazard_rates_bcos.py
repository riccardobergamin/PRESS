"""
Survival analysis utilities.
Defines hazard functions associated with
arrival-time distributions.
"""
import jax
import jax.numpy as jnp
from .arrival_probabilities import F, defender_F


def ballcarrier_hazard(
    p,          # (dim,)  ball carrier position
    v,          # (dim,)  ball carrier velocity
    r_target,   # (dim,)  single target point
    t,          # scalar
    s,
    tau_r,
    v_max,
    alpha_rot,
):
    # closed form h = a*F, valid for a single logistic CDF (f = a*F*(1-F),
    # h = f/(1-F) = a*F)
    a = jnp.pi / (jnp.sqrt(3) * s)
    return a * F(p, v, r_target, t, s, tau_r, v_max, alpha_rot).squeeze()


def defender_hazard(
    x_bc0,      # (dim,)          ball carrier starting position
    v_bc0,      # (dim,)          ball carrier starting velocity
    r_def,      # (N_def, dim)    defenders positions
    v_def,      # (N_def, dim)    defenders velocities
    x_path,     # (M, dim)        path points
    mask,       # (M,)            valid path points
    t,          # scalar
    s,
    tau_r_def,
    tau_r_bc,
    v_def_max,
    v_bc_max,
    alpha_rot,
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
        tau_r_def,
        tau_r_bc,
        v_def_max,
        v_bc_max,
        alpha_rot,
    )

