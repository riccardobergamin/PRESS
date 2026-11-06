from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
from PRESS.physics.flight_ball import ball_trajectory
SCRIPTS_DIR = Path('scratch/other/scripts')
sys.path.insert(0, str(SCRIPTS_DIR))
DATA_DIR = Path('/orfeo/cephfs/home/cdslab/rbergamin/scratch/other/datasets')
from os import environ as _press_environ
DATA_DIR = Path(_press_environ.get('PRESS_DATA_DIR', str(DATA_DIR))).expanduser().resolve()
MATCH_IDS = ['1886347', '1899585', '1925299', '1953632', '1996435', '2006229', '2011166', '2013725', '2015213', '2017461']
FPS = 10.0
N_FIT_FRAMES = 5
T_FIT = np.arange(N_FIT_FRAMES) / FPS
T_MAX = 10.0
DT = 0.04
BALL_PARAMS = {'rho': 1.22, 'Cd': 0.25, 'A': 0.038, 'm': 0.42, 'g': 9.81}

def get_ball_observations(tracking_data, frame_end):
    """
    Extract exactly five consecutive ball observations.

    Frames:

        frame_end
        frame_end + 1
        ...
        frame_end + 4

    Returns
    -------
    positions : ndarray, shape (5, 3)

        [ball_x, ball_y, ball_z]
    """
    frames = np.arange(frame_end, frame_end + N_FIT_FRAMES)
    ball = tracking_data[tracking_data['frame'].isin(frames)][['frame', 'ball_x', 'ball_y', 'ball_z']].drop_duplicates('frame').sort_values('frame')
    if len(ball) != N_FIT_FRAMES:
        raise ValueError(f"Missing ball observations. frame_end={frame_end}, expected_frames={frames.tolist()}, found_frames={ball['frame'].tolist()}")
    if not np.array_equal(ball['frame'].to_numpy(), frames):
        raise ValueError(f"Invalid ball frame sequence. frame_end={frame_end}, expected_frames={frames.tolist()}, found_frames={ball['frame'].tolist()}")
    positions = ball[['ball_x', 'ball_y', 'ball_z']].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(positions)):
        raise ValueError(f'Non-finite ball positions. frame_end={frame_end}, positions={positions.tolist()}')
    return positions

def get_ball_velocity_observations(tracking_data, frame_end):
    """
    Extract exactly five consecutive ball velocity
    observations.

    Expected columns:

        ball_vx
        ball_vy
        ball_vz

    Frames:

        frame_end ... frame_end + 4
    """
    frames = np.arange(frame_end, frame_end + N_FIT_FRAMES)
    ball = tracking_data[tracking_data['frame'].isin(frames)][['frame', 'ball_vx', 'ball_vy', 'ball_vz']].drop_duplicates('frame').sort_values('frame')
    if len(ball) != N_FIT_FRAMES:
        raise ValueError(f"Missing ball velocity observations. frame_end={frame_end}, expected_frames={frames.tolist()}, found_frames={ball['frame'].tolist()}")
    if not np.array_equal(ball['frame'].to_numpy(), frames):
        raise ValueError(f"Invalid ball velocity frame sequence. frame_end={frame_end}, expected_frames={frames.tolist()}, found_frames={ball['frame'].tolist()}")
    velocities = ball[['ball_vx', 'ball_vy', 'ball_vz']].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(velocities)):
        raise ValueError(f'Non-finite ball velocities. frame_end={frame_end}, velocities={velocities.tolist()}')
    return velocities

def velocity_to_initial_conditions(velocity):
    """
    Convert a 3D velocity vector into:

        v0
        theta
        phi

    Definitions
    -----------

    v0:
        magnitude of velocity.

    theta:
        elevation angle with respect to the horizontal.

        Range:
            [-pi/2, pi/2]

        theta > 0:
            upward

        theta = 0:
            horizontal

        theta < 0:
            downward

    phi:
        horizontal angle in the XY plane.

        Range:
            [0, 2*pi)
    """
    vx = float(velocity[0])
    vy = float(velocity[1])
    vz = float(velocity[2])
    v0 = float(np.linalg.norm(velocity))
    if v0 <= 1e-08:
        raise ValueError(f'Velocity magnitude too small: {v0}')
    v_horizontal = float(np.hypot(vx, vy))
    theta = float(np.arctan2(vz, v_horizontal))
    phi = float(np.arctan2(vy, vx) % (2.0 * np.pi))
    return {'v0': v0, 'theta': theta, 'phi': phi, 'vx': vx, 'vy': vy, 'vz': vz, 'v_horizontal': v_horizontal}

def model_trajectory_3d(r0, v0, theta, phi, t_span, **ball_params):
    """
    Generate the physical 3D ball trajectory.

    r0:
        [x0, y0, z0]

    v0:
        initial speed

    theta:
        elevation angle with horizontal.

        Can be negative.

    phi:
        horizontal angle in XY plane.
    """
    trajectory = ball_trajectory(v0, theta, t_span, z0=r0[2], **ball_params)
    d = trajectory[:, 0]
    r0 = np.asarray(r0, dtype=np.float64)
    x = r0[0] + d * np.cos(phi)
    y = r0[1] + d * np.sin(phi)
    z = trajectory[:, 1]
    return np.column_stack([x, y, z])

def compute_trajectory_errors(predicted, observed):
    """
    Compare predicted and observed trajectories.

    Returns errors in meters.
    """
    error = predicted - observed
    error_3d = np.linalg.norm(error, axis=1)
    error_xy = np.linalg.norm(error[:, :2], axis=1)
    error_z = np.abs(error[:, 2])
    return {'fit_rmse_3d': float(np.sqrt(np.mean(error_3d ** 2))), 'fit_mae_3d': float(np.mean(error_3d)), 'fit_max_error_3d': float(np.max(error_3d)), 'fit_p95_error_3d': float(np.percentile(error_3d, 95)), 'fit_rmse_xy': float(np.sqrt(np.mean(error_xy ** 2))), 'fit_mae_xy': float(np.mean(error_xy)), 'fit_max_error_xy': float(np.max(error_xy)), 'fit_rmse_z': float(np.sqrt(np.mean(error_z ** 2))), 'fit_mae_z': float(np.mean(error_z)), 'fit_max_error_z': float(np.max(error_z)), 'fit_sse': float(np.sum(error ** 2))}

def process_pass(tracking_data, pass_event, t_span):
    """
    Estimate ball initial velocity from the mean of the first
    five measured velocity vectors and generate the physical
    trajectory.

    No optimization.
    No skip.
    """
    frame_end = int(pass_event['frame_end'])
    release = tracking_data[tracking_data['frame'] == frame_end][['ball_x', 'ball_y', 'ball_z']].drop_duplicates()
    if len(release) != 1:
        raise ValueError(f'Invalid release position. frame_end={frame_end}, rows_found={len(release)}')
    r0 = release.iloc[0].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(r0)):
        raise ValueError(f'Non-finite release position. frame_end={frame_end}, r0={r0.tolist()}')
    observations = get_ball_observations(tracking_data, frame_end)
    velocity_observations = get_ball_velocity_observations(tracking_data, frame_end)
    mean_velocity = np.mean(velocity_observations, axis=0)
    initial_conditions = velocity_to_initial_conditions(mean_velocity)
    v0 = initial_conditions['v0']
    theta = initial_conditions['theta']
    phi = initial_conditions['phi']
    trajectory_3d = model_trajectory_3d(r0=r0, v0=v0, theta=theta, phi=phi, t_span=t_span, **BALL_PARAMS)
    trajectory_3d = np.asarray(trajectory_3d, dtype=np.float32)
    predicted_fit = model_trajectory_3d(r0=r0, v0=v0, theta=theta, phi=phi, t_span=T_FIT, **BALL_PARAMS)
    predicted_fit = np.asarray(predicted_fit, dtype=np.float64)
    errors = compute_trajectory_errors(predicted=predicted_fit, observed=observations)
    return {'frame_end': frame_end, 'x0': float(r0[0]), 'y0': float(r0[1]), 'z0': float(r0[2]), 'vx0': float(initial_conditions['vx']), 'vy0': float(initial_conditions['vy']), 'vz0': float(initial_conditions['vz']), 'v0': float(v0), 'theta': float(theta), 'phi': float(phi), 'v_horizontal': float(initial_conditions['v_horizontal']), **errors, 'trajectory_3d': trajectory_3d.tolist()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--match-index', type=int, required=True)
    args = parser.parse_args()
    match_id = MATCH_IDS[args.match_index]
    match_dir = DATA_DIR / 'matches' / match_id
    tracking_file = match_dir / f'{match_id}_tracking.parquet'
    pass_file = DATA_DIR / 'pass_dataset_straight.parquet'
    output_dir = DATA_DIR / 'ball_trajectories'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{match_id}_ball_trajectories.parquet'
    progress_file = output_dir / f'{match_id}_progress.txt'
    print(f'Loading tracking: {tracking_file}', flush=True)
    tracking_data = pd.read_parquet(tracking_file)
    required_tracking_columns = ['frame', 'ball_x', 'ball_y', 'ball_z', 'ball_vx', 'ball_vy', 'ball_vz']
    missing = [c for c in required_tracking_columns if c not in tracking_data.columns]
    if missing:
        raise ValueError(f'Missing required tracking columns: {missing}')
    print(f'Loading passes: {pass_file}', flush=True)
    pass_dataset = pd.read_parquet(pass_file)
    passes = pass_dataset[pass_dataset['match_id'].astype(str) == match_id].reset_index(drop=True)
    n_passes = len(passes)
    print('=' * 60, flush=True)
    print(f'MATCH {match_id}', flush=True)
    print(f'Passes: {n_passes}', flush=True)
    print('Velocity estimation: mean vector of first 5 frames', flush=True)
    print('Fit window: 0.0 -> 0.4 s', flush=True)
    print(f'Full trajectory: 0 -> {T_MAX} s', flush=True)
    print(f'Trajectory dt: {DT} s', flush=True)
    print('No optimization', flush=True)
    print('=' * 60, flush=True)
    t_span = np.arange(0.0, T_MAX + DT / 2, DT)
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
        print(f'[{match_id}] pass {i + 1}/{n_passes}', flush=True)
        result = process_pass(tracking_data=tracking_data, pass_event=pass_event, t_span=t_span)
        result['pass_idx'] = i
        results.append(result)
        print(f"  v0    = {result['v0']:.2f} m/s", flush=True)
        print(f"  theta = {np.degrees(result['theta']):.2f} deg", flush=True)
        print(f"  phi   = {np.degrees(result['phi']):.2f} deg", flush=True)
        print(f"  RMSE  = {result['fit_rmse_3d']:.3f} m", flush=True)
        progress_file.write_text(str(i + 1))
        if (i + 1) % 25 == 0 or i + 1 == n_passes:
            checkpoint = pd.DataFrame(results)
            checkpoint.to_parquet(output_file, index=False)
            print(f'[{match_id}] CHECKPOINT SAVED {i + 1}/{n_passes}', flush=True)
    print('=' * 60, flush=True)
    print(f'MATCH {match_id} COMPLETED', flush=True)
    print(f'Saved: {output_file}', flush=True)
    print(f'Successful trajectories: {len(results)}/{n_passes}', flush=True)
    print('=' * 60, flush=True)
