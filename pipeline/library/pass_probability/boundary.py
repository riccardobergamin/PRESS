"""Original five-velocity boundary and position-fit alternative; no target fit."""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares
from .flight import velocity as velocity_boundary
from .flight import position as position_boundary
from .precision import enable_x64
TIMES = np.arange(0.0, 10.0 + 0.04 / 2, 0.04)
FIT_TIMES = np.arange(5) / 10.0
LOWER = np.array([1.0, 0.0, -np.pi])
UPPER = np.array([40.0, np.pi / 3, np.pi])
FD_STEPS = np.array([0.001, 0.0001, 0.0001])
velocity_boundary.ball_trajectory = jax.jit(velocity_boundary.ball_trajectory)

@jax.jit


def project(parameters, r0, times):
    return position_boundary.model_trajectory_3d(r0, *parameters, times, **velocity_boundary.BALL_PARAMS)


def numerical_jacobian(residual, x):
    """Absolute finite-difference steps survive the original float32 solver.

    SciPy's default float64-sized perturbations can round away when parameters
    are cast to float32 in the historical residual. Preserve the original model
    and bounds, but use steps appreciably above float32 resolution.
    """
    columns = []
    for i, h in enumerate(FD_STEPS):
        left = x.copy()
        right = x.copy()
        left[i] = max(LOWER[i], x[i] - h)
        right[i] = min(UPPER[i], x[i] + h)
        columns.append((residual(right) - residual(left)) / (right[i] - left[i]))
    return np.column_stack(columns)


def estimate_both(tracking, row):
    """Fit only release+[0..4]. The target initializes phi, as in old code."""
    with enable_x64(False):
        mean = velocity_boundary.process_pass(tracking, {'frame_end': row.frame_end}, TIMES)
        observed = velocity_boundary.get_ball_observations(tracking, int(row.frame_end))
        r0 = observed[0]
        displacement = observed[-1] - r0
        horizontal = np.linalg.norm(displacement[:2])
        theta = np.clip(np.arctan2(displacement[2], max(horizontal, 1e-06)), 0.0, np.pi / 3)
        speed = np.clip(horizontal / 0.4 / max(np.cos(theta), 1e-06), 1.0, 40.0)
        phi = position_boundary.get_initial_phi(r0, np.array([row.x_target, row.y_target]))
        if phi is None:
            phi = float(np.arctan2(displacement[1], displacement[0]))
        initial = np.array([speed, theta, phi])

        def residual(x):
            return np.asarray(project(jnp.asarray(x, dtype=jnp.float32), jnp.asarray(r0, dtype=jnp.float32), jnp.asarray(FIT_TIMES, dtype=jnp.float32)),
                float).ravel() - observed.ravel()
        result = least_squares(residual,
            initial,
            jac=lambda x: numerical_jacobian(residual, x),
            bounds=(LOWER, UPPER),
            method='trf',
            x_scale='jac',
            ftol=1e-06,
            xtol=1e-06,
            gtol=1e-06,
            max_nfev=100)
        valid = bool(result.success and np.isfinite(result.fun).all() and np.isfinite(result.x).all())
        if valid:
            tr = np.asarray(project(jnp.asarray(result.x, dtype=jnp.float32), jnp.asarray(r0, dtype=jnp.float32), jnp.asarray(TIMES, dtype=jnp.float32)))
            valid = tr.shape == (251, 3) and np.isfinite(tr).all()
        fitted = dict(mean)
        if valid:
            speed, theta, phi = result.x
            horizontal = speed * np.cos(theta)
            fitted.update(v0=float(speed),
                theta=float(theta),
                phi=float(phi),
                vx0=float(horizontal * np.cos(phi)),
                vy0=float(horizontal * np.sin(phi)),
                vz0=float(speed * np.sin(theta)),
                v_horizontal=float(horizontal),
                trajectory_3d=tr.tolist())
            fitted.update(velocity_boundary.compute_trajectory_errors(residual(result.x).reshape(5, 3) + observed, observed))
        fitted.update(fit_usable=bool(valid),
            fit_fallback_to_mean=not valid,
            fit_nfev=int(result.nfev),
            fit_status=int(result.status),
            fit_optimality=float(result.optimality),
            fit_initial_cost=float(0.5 * np.sum(residual(initial) ** 2)),
            fit_final_cost=float(result.cost))
    return (mean, fitted)


def forecast_error(trajectory, tracking, row):
    """Equal-weight per-pass 3D MAE after common .6s support, before next touch."""
    frames = np.arange(int(row.frame_end) + 7, min(int(row.frame_reception), int(row.frame_end) + 101))
    if len(frames) == 0:
        return (np.nan, 0)
    observed = tracking.set_index('frame').reindex(frames)[['ball_x', 'ball_y', 'ball_z']].to_numpy()
    times = (frames - int(row.frame_end)) / 10.0
    predicted = np.column_stack([np.interp(times, TIMES, np.asarray(trajectory)[:, axis]) for axis in range(3)])
    valid = np.isfinite(observed).all(axis=1)
    if not valid.any():
        return (np.nan, 0)
    return (float(np.linalg.norm(predicted[valid] - observed[valid], axis=1).mean()), int(valid.sum()))
