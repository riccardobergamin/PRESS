import jax
import jax.numpy as jnp
from PRESS.physics.arrival_probabilities import F


def interception_probability(
    x_bc0,
    v_bc0,
    r_def,
    v_def,
    x_line,
    t,
    s,
    tau_r_def,
    v_def_max,
    alpha,
):
    """
    Compute individual defender interception probabilities.

    The target is the position reached by the ball carrier
    after time t, or at the pitch exit point if the ball
    carrier reaches the boundary before t.
    """

    # ---------------------------------------------------------
    # Ball carrier speed
    # ---------------------------------------------------------

    v = jnp.linalg.norm(
        v_bc0,
    )

    # ---------------------------------------------------------
    # Time for the ball carrier to reach the pitch boundary
    #
    # If the ball carrier is stationary, the boundary is
    # never reached.
    # ---------------------------------------------------------

    d = jnp.linalg.norm(
        x_line - x_bc0,
    )

    t_line = jnp.where(
        v > 0.0,
        d / v,
        jnp.inf,
    )

    # ---------------------------------------------------------
    # Effective target time
    # ---------------------------------------------------------

    t_eff = jnp.minimum(
        t,
        t_line,
    )

    # ---------------------------------------------------------
    # Ball carrier target position
    # ---------------------------------------------------------

    x_target = (
        x_bc0
        + v_bc0 * t_eff
    )[None, :]

    # ---------------------------------------------------------
    # Defender interception probabilities
    #
    # Pressure is evaluated over the full time horizon t.
    # ---------------------------------------------------------

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
