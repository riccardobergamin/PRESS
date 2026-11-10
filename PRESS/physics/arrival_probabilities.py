"""
Arrival-time distributions.
Defines event-time PDFs from physical motion models.
"""

import jax
import jax.numpy as jnp

from .times import time_to_intercept

# def F(
#     p,
#     v,
#     r_target,
#     t,
#     s=0.45,
#     tau_r=0.5,
#     v_max=6.0,
#     alpha=1.0,
# ):
#     """
#     Spearman arrival distribution
#     """
#     p = jnp.atleast_2d(p)
#     v = jnp.atleast_2d(v)
#     r_target = jnp.atleast_2d(r_target)
#     # rotation
#     tau_theta = rotation_time(
#         p,
#         v,
#         r_target,
#         alpha,
#     )
#     # linear dynamics
#     p_react = p[:, None, :] + v[:, None, :] * tau_r
#     delta_react = r_target[None, :, :] - p_react
#     d_react = jnp.linalg.norm(delta_react, axis=2)
#     tau_move = d_react / v_max
#     # total time
#     tau_arrival = tau_r + tau_theta + tau_move
#     a = jnp.pi / (jnp.sqrt(3) * s)
#     return jax.nn.sigmoid(a * (t - tau_arrival))


def F(
    p,
    v,
    r_target,
    t,
    s=0.45,
    v_max=5.0,
    a_max=5.0,
    threshold=0.1,
    a_turn=None,
    preserve_initial_overspeed=True,
):
    """
    Spearman arrival probability.
    """

    tau_int = time_to_intercept(
        p,
        v,
        r_target,
        v_max=v_max,
        a_max=a_max,
        threshold=threshold,
        a_turn=a_turn,
        preserve_initial_overspeed=preserve_initial_overspeed,
    )

    a = (
        jnp.pi
        / (jnp.sqrt(3.0) * s)
    )

    return jax.nn.sigmoid(
        a * (t - tau_int)
    )


def p_bc(
    x_bc0,
    v_bc0,
    x_path,
    mask,
    t,
    s,
    v_bc_max,
    a_bc_max,
):
    """
    Ball carrier space distribution at time t.
    """

    F_bc = jnp.ravel(
        F(
            p=x_bc0,
            v=v_bc0,
            r_target=x_path,
            t=t,
            s=s,
            v_max=v_bc_max,
            a_max=a_bc_max,
        )
    )

    F_bc = F_bc.at[0].set(1.0)

    p_x = jnp.clip(
        F_bc[:-1] - F_bc[1:],
        0.0,
        1.0,
    )

    p_x = jnp.append(
        p_x,
        0.0,
    ) * mask

    p_x = jnp.where(
        t == 0.0,
        jnp.concatenate(
            [
                jnp.ones(1),
                jnp.zeros(
                    p_x.shape[0] - 1
                ),
            ]
        ) * mask,
        p_x,
    )

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
    v_def_max,
    v_bc_max,
    a_def_max,
    a_bc_max
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
        v_def_max,
        a_def_max,
    )

    p_x = p_bc(
        x_bc0,
        v_bc0,
        x_path,
        mask,
        t,
        s,
        v_bc_max,
        a_bc_max,
    )

    return jnp.sum(
        F_def * p_x[None, :],
        axis=-1,
    )
