import jax
import jax.numpy as jnp

from PRESS.physics.arrival_probabilities import F


def interception_probability(
    x_bc0,
    v_bc0,
    r_def,
    v_def,
    s,
    tau_r_bc,
    tau_r_def,
    v_def_max,
    alpha_rot,
    delta_t,
):

    x_target = (
        x_bc0
        + v_bc0 * tau_r_bc
    )[None, :]


    p_int = F(
        r_def,
        v_def,
        x_target,
        delta_t,
        s,
        tau_r_def,
        v_def_max,
        alpha_rot,
    )


    return jnp.squeeze(
        p_int,
        axis=1,
    )
