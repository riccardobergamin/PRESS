"""
Time to intercept

"""

import jax.numpy as jnp

from .angles import compute_angle


def rotation_time(p, v, r_target, a_max):
    """
    Time required to rotate toward target direction.
    """
    theta = compute_angle(p, v, r_target)
    v_mag = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return v_mag * theta / a_max


def time_to_intercept(
    p,
    v,
    r_target,
    v_max=5.0,
    a_max=5.0,
    threshold=0.1,
    a_turn=None,
    preserve_initial_overspeed=False,
):
    """
    Sequential turn-then-accelerate time to a fixed target.

    a_max caps the modelled forward acceleration, not an observed a0.
    a_turn is the effective centripetal/reorientation scale in v*theta/a;
    None uses a_max and exactly retains the historical convention.
    If preserve_initial_overspeed is true, v0 above v_max is maintained
    without further acceleration, instead of instantaneously using v_max.
    The turn remains a fixed-position delay, not an explicit circular arc.
    """

    p = jnp.atleast_2d(p)
    v = jnp.atleast_2d(v)
    r_target = jnp.atleast_2d(r_target)

    # Distance player -> target

    delta = (
        r_target[None, :, :]
        - p[:, None, :]
    )

    d = jnp.linalg.norm(
        delta,
        axis=2,
    )

    # Initial speed

    v0 = jnp.linalg.norm(
        v,
        axis=1,
    )[:, None]

    # Time to reach v_max

    t_acc = jnp.maximum(
        (v_max - v0) / a_max,
        0.0,
    )

    # Distance covered during acceleration

    d_acc = (
        v0 * t_acc
        + 0.5 * a_max * t_acc**2
    )

    # Time to reach target while accelerating

    t_acc_target = (
        -v0
        + jnp.sqrt(
            jnp.maximum(
                v0**2 + 2.0 * a_max * d,
                0.0,
            )
        )
    ) / a_max

    # Time to reach target after v_max

    terminal_speed = jnp.maximum(v0, v_max) if preserve_initial_overspeed else v_max
    t_constant = (
        t_acc
        + (d - d_acc) / terminal_speed
    )

    # Time to intercept

    tau_int = jnp.where(
        d <= d_acc,
        t_acc_target,
        t_constant,
    )

    # Rotation-time contribution

    tau_theta = rotation_time(
        p,
        v,
        r_target,
        a_max if a_turn is None else a_turn,
    )

    tau_int = tau_int + tau_theta

    # Already at target

    tau_int = jnp.where(
        d <= threshold,
        0.0,
        tau_int,
    )

    return tau_int
