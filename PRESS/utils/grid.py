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

# def lookup_kappa(
#     speed,
#     distance,
#     kappa_surface,
# ):
#     """
#     Return the data-derived kappa(v,d).
#
#     Parameters
#     ----------
#     speed : float
#         Ball-carrier speed.
#
#     distance : jnp.ndarray (N,)
#         Distance to each target.
#
#     kappa_surface : dict
#         {
#             "speed_edges": ...,
#             "distance_edges": ...,
#             "kappa": ...
#         }
#     """
#
#     speed_edges = kappa_surface["speed_edges"]
#     distance_edges = kappa_surface["distance_edges"]
#
#     K = kappa_surface["kappa"]
#
#     i = jnp.digitize(
#         speed,
#         speed_edges,
#     ) - 1
#
#     j = jnp.digitize(
#         jnp.asarray(distance),
#         distance_edges,
#     ) - 1
#
#     i = jnp.clip(
#         i,
#         0,
#         K.shape[0] - 1,
#     )
#
#     j = jnp.clip(
#         j,
#         0,
#         K.shape[1] - 1,
#     )
#
#     return jnp.asarray(
#         K[i, j]
#     )


def compute_exit_point(x, v, bounds):
    """
    Compute the first pitch boundary reached
    by a straight-line trajectory.

    Parameters
    ----------
    x : array-like, shape (2,)
        Current position [x, y].

    v : array-like, shape (2,)
        Current velocity [vx, vy].

    bounds : array-like, shape (2, 2)
        Pitch boundaries:
        [
            [x_min, x_max],
            [y_min, y_max],
        ]

    Returns
    -------
    jnp.ndarray, shape (2,)
        First boundary point reached.
    """

    x=jnp.asarray(x);v=jnp.asarray(v);bounds=jnp.asarray(bounds)
    moving=v!=0.;safe=jnp.where(moving,v,1.)
    left=(bounds[:,0]-x)/safe;right=(bounds[:,1]-x)/safe
    near=jnp.where(moving,jnp.minimum(left,right),-jnp.inf)
    far=jnp.where(moving,jnp.maximum(left,right),jnp.inf)
    outside=(x<bounds[:,0]) | (x>bounds[:,1])
    near=jnp.where(~moving & outside,jnp.inf,near)
    far=jnp.where(~moving & outside,-jnp.inf,far)
    entry=jnp.max(near);exit=jnp.min(far)
    valid=(exit>=jnp.maximum(entry,0.)) & jnp.isfinite(exit)
    # No forward intersection (including a stationary ray): preserve origin.
    return x+v*jnp.where(valid,exit,0.)
