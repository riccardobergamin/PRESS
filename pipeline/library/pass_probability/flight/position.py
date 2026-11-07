"""Five-frame ball-flight boundary equations; no file-system or CLI state."""
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from PRESS.physics.flight_ball import ball_trajectory
FPS = 10.0
N_FIT_FRAMES = 5
T_FIT = np.arange(N_FIT_FRAMES) / FPS
T_FIT_JAX = jnp.asarray(T_FIT, dtype=jnp.float32)
T_MAX = 10.0
DT = 0.04
V0_MIN = 1.0
V0_MAX = 40.0
THETA_MIN = 0.0
THETA_MAX = np.deg2rad(60.0)
PHI_MIN = -np.pi
PHI_MAX = np.pi
BALL_PARAMS = {'rho': 1.22, 'Cd': 0.25, 'A': 0.038, 'm': 0.42, 'g': 9.81}


def get_initial_phi(r0, target):
    """
    Initial guess for phi.

    phi is the horizontal direction of the pass.

    The target is used ONLY to initialize phi.
    It is not used as an optimization constraint.
    """
    delta = np.asarray(target, dtype=np.float64) - np.asarray(r0[:2], dtype=np.float64)
    distance = np.linalg.norm(delta)
    if distance < 1e-06:
        return None
    return np.arctan2(delta[1], delta[0])


def model_trajectory_3d(r0, v0, theta, phi, t_span, **ball_params):
    """
    Generate the physical ball trajectory and project
    its horizontal component into the XY plane using phi.

    IMPORTANT:
        ball_trajectory() is unchanged.
        Its original ODE and ground boundary condition
        are used exactly as defined in PRESS.
    """
    trajectory = ball_trajectory(v0, theta, t_span, z0=r0[2], **ball_params)
    d = trajectory[:, 0]
    r0 = jnp.asarray(r0, dtype=jnp.float32)
    x = r0[0] + d * jnp.cos(phi)
    y = r0[1] + d * jnp.sin(phi)
    z = trajectory[:, 1]
    return jnp.stack([x, y, z], axis=1)
