"""
Geometric and kinematic utilities.
This module contains deterministic quantities:
- directions
- rotation times
"""
import jax.numpy as jnp


def compute_angle(p, v, r_target):
    """
    Angle between velocity vector and target direction.
    """
    target_vector = r_target[None, :, :] - p[:, None, :]
    d_act = jnp.linalg.norm(target_vector, axis=-1, keepdims=True)  # (N, M, 1)
    v_mag = jnp.linalg.norm(v, axis=-1, keepdims=True)              # (N, 1)
    v_dir = v[:, None, :] / (v_mag[:, None, :] + 1e-8)
    target_dir = target_vector / (d_act + 1e-8)
    cos_theta = jnp.clip(
        jnp.sum(v_dir * target_dir, axis=-1),
        -1.0,
        1.0,
    )  # (N, M)
    degenerate = (v_mag < 0.1) | (d_act[..., 0] < 0.1)  # -> (N, M)
    return jnp.where(degenerate, 0.0, jnp.arccos(cos_theta))


def rotation_time(p, v, r_target, alpha):
    """
    Time required to rotate toward target direction.
    """
    theta = compute_angle(p, v, r_target)
    v_mag = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return (v_mag * theta) / (jnp.pi * alpha)
