from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from scipy.optimize import least_squares
from PRESS.physics.flight_ball import ball_trajectory
SCRIPTS_DIR = Path('scratch/other/scripts')
sys.path.insert(0, str(SCRIPTS_DIR))
DATA_DIR = Path('/orfeo/cephfs/home/cdslab/rbergamin/scratch/other/datasets')
MATCH_IDS = ['1886347', '1899585', '1925299', '1953632', '1996435', '2006229', '2011166', '2013725', '2015213', '2017461']
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

def get_ball_observations(tracking_data, frame_end):
    """
    Extract exactly five consecutive ball observations.

    Frames:

        frame_end
        frame_end + 1
        ...
        frame_end + 4

    Corresponding to:

        t = 0.0 ... 0.4 s
    """
    frames = np.arange(frame_end, frame_end + N_FIT_FRAMES)
    ball = tracking_data[tracking_data['frame'].isin(frames)][['frame', 'ball_x', 'ball_y', 'ball_z']].drop_duplicates('frame').sort_values('frame')
    if len(ball) != N_FIT_FRAMES:
        return None
    if not np.array_equal(ball['frame'].to_numpy(), frames):
        return None
    positions = ball[['ball_x', 'ball_y', 'ball_z']].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(positions)):
        return None
    return positions

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

def residuals_jax(params, r0, observations):
    """
    JAX residual function.

    Parameters:
        params = [v0, theta, phi]

    Returns:
        15 residuals:
            5 observations x 3 coordinates
    """
    v0 = params[0]
    theta = params[1]
    phi = params[2]
    predicted = model_trajectory_3d(r0=r0, v0=v0, theta=theta, phi=phi, t_span=T_FIT_JAX, **BALL_PARAMS)
    error = predicted - observations
    return jnp.ravel(error)
RESIDUALS_JIT = jax.jit(residuals_jax)

def fit_ball_initial_conditions(observations, r0, phi_guess, **ball_params):
    """
    Estimate v0, theta and phi from the first 0.4 seconds.

    Optimization:
        scipy.optimize.least_squares

    Jacobian:
        SciPy 2-point finite differences

    Forward model:
        JAX JIT

    Physical model:
        original ball_trajectory()
    """
    observations = np.asarray(observations, dtype=np.float64)
    r0 = np.asarray(r0, dtype=np.float64)
    r0_jax = jnp.asarray(r0, dtype=jnp.float32)
    observations_jax = jnp.asarray(observations, dtype=jnp.float32)
    displacement = observations[-1] - observations[0]
    horizontal_distance = np.linalg.norm(displacement[:2])
    vertical_distance = displacement[2]
    total_time = T_FIT[-1]
    if horizontal_distance <= 1e-06 or total_time <= 0.0:
        return None
    theta_guess = np.arctan2(vertical_distance, max(horizontal_distance, 1e-06))
    theta_guess = np.clip(theta_guess, THETA_MIN, THETA_MAX)
    horizontal_speed = horizontal_distance / total_time
    v0_guess = horizontal_speed / max(np.cos(theta_guess), 1e-06)
    v0_guess = np.clip(v0_guess, V0_MIN, V0_MAX)
    x0 = np.array([v0_guess, theta_guess, phi_guess], dtype=np.float64)

    def residuals_numpy(params):
        params_jax = jnp.asarray(params, dtype=jnp.float32)
        return np.asarray(RESIDUALS_JIT(params_jax, r0_jax, observations_jax), dtype=np.float64)
    result = least_squares(residuals_numpy, x0=x0, jac='2-point', bounds=(np.array([V0_MIN, THETA_MIN, PHI_MIN], dtype=np.float64), np.array([V0_MAX, THETA_MAX, PHI_MAX], dtype=np.float64)), method='trf', x_scale='jac', ftol=1e-06, xtol=1e-06, gtol=1e-06, max_nfev=100)
    if not result.success:
        return None
    v0 = float(result.x[0])
    theta = float(result.x[1])
    phi = float(result.x[2])
    predicted = np.asarray(model_trajectory_3d(r0=r0, v0=v0, theta=theta, phi=phi, t_span=T_FIT_JAX, **ball_params), dtype=np.float64)
    error = predicted - observations
    error_3d = np.linalg.norm(error, axis=1)
    fit_rmse_3d = float(np.sqrt(np.mean(error_3d ** 2)))
    fit_mae_3d = float(np.mean(error_3d))
    fit_max_error_3d = float(np.max(error_3d))
    fit_p95_error_3d = float(np.percentile(error_3d, 95))
    error_xy = np.linalg.norm(error[:, :2], axis=1)
    fit_rmse_xy = float(np.sqrt(np.mean(error_xy ** 2)))
    fit_mae_xy = float(np.mean(error_xy))
    fit_max_error_xy = float(np.max(error_xy))
    error_z = np.abs(error[:, 2])
    fit_rmse_z = float(np.sqrt(np.mean(error_z ** 2)))
    fit_mae_z = float(np.mean(error_z))
    fit_max_error_z = float(np.max(error_z))
    fit_sse = float(np.sum(error ** 2))
    v0_at_lower_bound = v0 <= V0_MIN + 0.001
    v0_at_upper_bound = v0 >= V0_MAX - 0.001
    theta_at_lower_bound = theta <= THETA_MIN + 1e-05
    theta_at_upper_bound = theta >= THETA_MAX - 1e-05
    return {'v0': v0, 'theta': theta, 'phi': phi, 'fit_rmse_3d': fit_rmse_3d, 'fit_mae_3d': fit_mae_3d, 'fit_max_error_3d': fit_max_error_3d, 'fit_p95_error_3d': fit_p95_error_3d, 'fit_rmse_xy': fit_rmse_xy, 'fit_mae_xy': fit_mae_xy, 'fit_max_error_xy': fit_max_error_xy, 'fit_rmse_z': fit_rmse_z, 'fit_mae_z': fit_mae_z, 'fit_max_error_z': fit_max_error_z, 'fit_sse': fit_sse, 'fit_cost': float(result.cost), 'fit_nfev': int(result.nfev), 'fit_success': bool(result.success), 'fit_optimality': float(result.optimality), 'v0_at_lower_bound': v0_at_lower_bound, 'v0_at_upper_bound': v0_at_upper_bound, 'theta_at_lower_bound': theta_at_lower_bound, 'theta_at_upper_bound': theta_at_upper_bound}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--match-index', type=int, required=True)
    args = parser.parse_args()
    match_id = MATCH_IDS[args.match_index]
    match_dir = DATA_DIR / 'matches' / match_id
    tracking_file = match_dir / f'{match_id}_tracking.parquet'
    pass_file = DATA_DIR / 'pass_dataset.parquet'
    output_dir = DATA_DIR / 'ball_trajectories'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{match_id}_ball_trajectories.parquet'
    progress_file = output_dir / f'{match_id}_progress.txt'
    print(f'Loading tracking: {tracking_file}', flush=True)
    tracking_data = pd.read_parquet(tracking_file)
    print(f'Loading passes: {pass_file}', flush=True)
    pass_dataset = pd.read_parquet(pass_file)
    passes = pass_dataset[pass_dataset['match_id'].astype(str) == match_id].reset_index(drop=True)
    n_passes = len(passes)
    print('=' * 60, flush=True)
    print(f'MATCH {match_id}', flush=True)
    print(f'Passes: {n_passes}', flush=True)
    print('Fit window: 0.0 -> 0.4 s', flush=True)
    print(f'Full trajectory: 0 -> {T_MAX} s', flush=True)
    print(f'Trajectory dt: {DT} s', flush=True)
    print('Optimizer: scipy.optimize.least_squares', flush=True)
    print('Jacobian: SciPy 2-point finite difference', flush=True)
    print('Forward model: JAX JIT', flush=True)
    print('ODE: original ball_trajectory()', flush=True)
    print('=' * 60, flush=True)
    t_span = jnp.arange(0.0, T_MAX + DT / 2, DT)
    if progress_file.exists():
        start_idx = int(progress_file.read_text().strip())
    else:
        start_idx = 0
    if output_file.exists():
        existing = pd.read_parquet(output_file)
        results = existing.to_dict(orient='records')
        if len(results) < start_idx:
            start_idx = len(results)
    else:
        results = []
    print(f'Starting from {start_idx}/{n_passes}', flush=True)
    for i in range(start_idx, n_passes):
        pass_event = passes.iloc[i]
        print(f'[{match_id}] pass {i + 1}/{n_passes} ({100.0 * (i + 1) / n_passes:.1f}%)', flush=True)
        frame_end = int(pass_event['frame_end'])
        release = tracking_data[tracking_data['frame'] == frame_end][['ball_x', 'ball_y', 'ball_z']].drop_duplicates()
        if len(release) == 0:
            print('  -> missing release position', flush=True)
            continue
        r0 = release.iloc[0].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(r0)):
            print('  -> invalid release position', flush=True)
            continue
        target = np.array([pass_event['x_target'], pass_event['y_target']], dtype=np.float64)
        if not np.all(np.isfinite(target)):
            print('  -> invalid target', flush=True)
            continue
        phi_guess = get_initial_phi(r0, target)
        if phi_guess is None:
            print('  -> invalid target direction', flush=True)
            continue
        observations = get_ball_observations(tracking_data, frame_end)
        if observations is None:
            print('  -> missing one or more required ball frames', flush=True)
            continue
        fit = fit_ball_initial_conditions(observations=observations, r0=r0, phi_guess=phi_guess, **BALL_PARAMS)
        if fit is None:
            print('  -> fit failed', flush=True)
            continue
        trajectory_3d = model_trajectory_3d(r0=r0, v0=fit['v0'], theta=fit['theta'], phi=fit['phi'], t_span=t_span, **BALL_PARAMS)
        trajectory_3d = np.asarray(trajectory_3d, dtype=np.float32)
        result_row = {'pass_idx': i, 'frame_end': frame_end, 'v0': fit['v0'], 'theta': fit['theta'], 'phi': fit['phi'], 'fit_rmse_3d': fit['fit_rmse_3d'], 'fit_mae_3d': fit['fit_mae_3d'], 'fit_max_error_3d': fit['fit_max_error_3d'], 'fit_p95_error_3d': fit['fit_p95_error_3d'], 'fit_rmse_xy': fit['fit_rmse_xy'], 'fit_mae_xy': fit['fit_mae_xy'], 'fit_max_error_xy': fit['fit_max_error_xy'], 'fit_rmse_z': fit['fit_rmse_z'], 'fit_mae_z': fit['fit_mae_z'], 'fit_max_error_z': fit['fit_max_error_z'], 'fit_sse': fit['fit_sse'], 'fit_cost': fit['fit_cost'], 'fit_nfev': fit['fit_nfev'], 'fit_success': fit['fit_success'], 'fit_optimality': fit['fit_optimality'], 'v0_at_lower_bound': fit['v0_at_lower_bound'], 'v0_at_upper_bound': fit['v0_at_upper_bound'], 'theta_at_lower_bound': fit['theta_at_lower_bound'], 'theta_at_upper_bound': fit['theta_at_upper_bound'], 'trajectory_3d': trajectory_3d.tolist()}
        results.append(result_row)
        progress_file.write_text(str(i + 1))
        if (i + 1) % 25 == 0 or i + 1 == n_passes:
            checkpoint = pd.DataFrame(results)
            checkpoint.to_parquet(output_file, index=False)
            print(f'[{match_id}] CHECKPOINT SAVED {i + 1}/{n_passes}', flush=True)
    print('=' * 60, flush=True)
    print(f'MATCH {match_id} COMPLETED', flush=True)
    print(f'Saved: {output_file}', flush=True)
    print(f'Successful fits: {len(results)}/{n_passes}', flush=True)
    print('=' * 60, flush=True)
