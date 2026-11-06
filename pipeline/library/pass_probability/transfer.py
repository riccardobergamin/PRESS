import jax
import jax.numpy as jnp
import numpy as np
from PRESS.physics.flight_ball import generate_ball_trajectories
from PRESS.physics.times import time_to_intercept
from .physics import candidates, FLIGHT, arrival_batch, TTI_SETTINGS
from .precision import enable_x64
METHODS=["mean_velocity","fit_positions"]
def choose_boundary(boundaries, fit_matches):
    """Pick using only fit-match free-flight errors, with equal match weights."""
    selected = boundaries.loc[boundaries.match_id.astype(str).isin(fit_matches)]
    columns = [m + '_forecast_error' for m in METHODS]
    selected = selected.dropna(subset=columns)
    if selected.empty:
        raise ValueError('No paired out-of-window flight observations for boundary selection')
    errors = selected.groupby('match_id')[columns].mean().mean()
    winner = min(METHODS, key=lambda m: errors[m + '_forecast_error'])
    return (winner, dict(selection_matches=sorted(fit_matches), paired_passes=len(selected), match_weighted_mae={m: float(errors[m + '_forecast_error']) for m in METHODS}, criterion='Mean 3D error from .7s until before next controlled touch, equal match weights'))

@jax.jit
def _path_times(pos, vel, path, cap):
    return jax.vmap(lambda p, v, r: time_to_intercept(p, v, r, v_max=cap, a_max=7.0, **TTI_SETTINGS).T)(pos, vel, path)

def transfer_physics(raw, cap):
    with enable_x64(False):
        pos, vel, target, bc = [jnp.asarray(raw[k], dtype=jnp.float32) for k in ['p_pos', 'p_vel', 'r_target', 'x_bc']]
        tau = arrival_batch(pos, vel, target, vmax=cap, **TTI_SETTINGS)
        time = jnp.arange(0.0, 10.0 + 0.04, 0.04, dtype=jnp.float32)
        speed, angle = candidates()
        tr = generate_ball_trajectories(speed, angle, time, **{k: FLIGHT[k] for k in ['rho', 'Cd', 'A', 'm', 'g']})
        distance = jnp.linalg.norm(target - bc, axis=-1)
        nearest = jnp.argmin(jnp.linalg.norm(pos[:, :10] - target[:, None, :], axis=-1), axis=1)
        desired = tau[jnp.arange(len(pos)), nearest]

        def one(radius, wanted):
            times = jax.vmap(lambda t: jnp.interp(radius, t[:, 0], time))(tr)
            height = jax.vmap(lambda t, hit: jnp.interp(hit, time, t[:, 1]))(tr, times)
            times = jnp.where((height <= 2.0) & (radius <= tr[:, :, 0].max(axis=1)), times, jnp.inf)
            return times[jnp.argmin(abs(times - wanted))]
        ball = np.asarray(jax.jit(jax.vmap(one))(distance, desired))
        valid = np.linalg.norm(np.asarray(pos), axis=-1) < 500
        valid[:, 10] = False
    if not np.isfinite(ball).all():
        raise ValueError('Hypothetical flight unavailable')
    return (np.asarray(tau), ball, valid)
