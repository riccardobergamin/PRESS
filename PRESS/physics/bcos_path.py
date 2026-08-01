import jax.numpy as jnp

def build_path(
    x_target,
    x_start,
    MAX_POINTS=500,
    dx=0.1,
):
    """
    Build a discretized straight-line trajectory for bc dynamics
    """
    delta = x_target - x_start
    distance = jnp.linalg.norm(delta, axis=-1) + 1e-8
    n_needed = (distance / dx).astype(jnp.int32)
    indices = jnp.arange(MAX_POINTS)
    mask = (indices <= n_needed).astype(jnp.float32)
    xi = indices / jnp.maximum(1.0, n_needed.astype(jnp.float32)) 
    x_path = x_start + xi[:, None] * delta
    return x_path, mask
