import jax.numpy as jnp


def generate_circular_grid(
    r_max,
    step,
):
    """
    Generate circular grid around ball carrier.

    Parameters
    ----------
    r_max : float
        Maximum radius.
    step : float
        Grid resolution.

    Returns
    -------
    jnp.ndarray
        Coordinates relative to ball carrier.
    """

    axis = jnp.arange(
        -r_max,
        r_max + 1e-5,
        step,
    )

    x_c, y_c = jnp.meshgrid(
        axis,
        axis,
    )

    dist_from_center = jnp.sqrt(
        x_c**2 + y_c**2
    )

    mask = dist_from_center <= r_max

    return jnp.stack(
        [
            x_c[mask],
            y_c[mask],
        ],
        axis=-1,
    )
