import jax
import jax.numpy as jnp

from PRESS.physics.arrival_probabilities import F


def interception_probability(
    x_bc0,
    v_bc0,
    r_def,
    v_def,
    t,
    s,
    tau_r_def,
    v_def_max,
    alpha,
):

    x_target = (
        x_bc0
        + v_bc0 * t
    )[None, :]


    p_int = F(
        r_def,
        v_def,
        x_target,
        t,
        s,
        tau_r_def,
        v_def_max,
        alpha,
    )


    return jnp.squeeze(
        p_int,
        axis=1,
    )
