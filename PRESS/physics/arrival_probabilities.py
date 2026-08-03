"""
Arrival-time distributions.
Defines event-time PDFs from physical motion models.
"""
import jax
import jax.numpy as jnp
from .angles import (
    compute_angle,
    rotation_time,
)


def F(
    p,
    v,
    r_target,
    t,
    s=0.45,
    tau_r=0.5,
    v_max=6.0,
    alpha=1.0,
):
    """
    Spearman arrival distribution
    """
    p = jnp.atleast_2d(p)
    v = jnp.atleast_2d(v)
    r_target = jnp.atleast_2d(r_target)
    # rotation
    tau_theta = rotation_time(
        p,
        v,
        r_target,
        alpha,
    )
    # linear dynamics
    p_react = p[:, None, :] + v[:, None, :] * tau_r
    delta_react = r_target[None, :, :] - p_react
    d_react = jnp.linalg.norm(delta_react, axis=2)
    tau_move = d_react / v_max
    # total time
    tau_arrival = tau_r + tau_theta + tau_move
    a = jnp.pi / (jnp.sqrt(3) * s)
    return jax.nn.sigmoid(a * (t - tau_arrival))


def p_bc(
    x_bc0,
    v_bc0,
    x_path,
    mask,
    t,
    s,
    tau_r_bc,
    v_bc_max,
    alpha,
):
    """
    Ball carrier space distribution at time t.
    """
    F_bc = jnp.ravel(
        F(
            p=x_bc0, v=v_bc0, r_target=x_path, t=t,
            s=s, tau_r=tau_r_bc, v_max=v_bc_max, alpha=alpha,
        )
    )
    p_x = jnp.clip(F_bc[:-1] - F_bc[1:], 0.0, 1.0)
    p_x = jnp.append(p_x, 0.0) * mask
    return p_x


def defender_F(
    x_bc0,
    v_bc0,
    r_def,
    v_def,
    x_path,
    mask,
    t,
    s,
    tau_r_bc,
    delta_r_def,
    v_def_max,
    v_bc_max,
    alpha,
):
  
    """
    Defender arrival distribution at bc position
    """
    
    F_def = F(
        r_def,
        v_def,
        x_path,
        t,
        s,
        tau_r_bc + delta_r_def,
        v_def_max,
        alpha,
    )

    p_x = p_bc(
        x_bc0,
        v_bc0,
        x_path,
        mask,
        t,
        s,
        tau_r_bc,
        v_bc_max,
        alpha,
    )

    return jnp.sum(
        F_def * p_x[None, :],
        axis=-1,
    )
