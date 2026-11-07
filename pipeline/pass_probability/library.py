"""Trajectory MLE, independent endpoint calibration and operational pass model."""
import numpy as np
import pandas as pd
import jax.numpy as jnp
from sklearn.isotonic import IsotonicRegression
from pipeline.library.common import fresh_directory, write_json, read_json, binary_scores
from pipeline.library.states import load_match, initial_state, save_states, folds
from pipeline.library.pass_probability.boundary import estimate_both, forecast_error
from pipeline.library.pass_probability.transfer import choose_boundary, _path_times, transfer_physics
from pipeline.library.pass_probability.trajectory_model import fit
from pipeline.library.pass_probability.stable_likelihood import predict_logs
from pipeline.library.pass_probability.precision import enable_x64
from PRESS.model import PassRiskModel
from .filters import verified_passes


def features(paths, matches):
    for match in matches:
        out = fresh_directory(paths.task('pass_probability', 'features') / match)
        tracking, events, players, metadata = load_match(paths.base, match)
        rows = verified_passes(events, tracking, players, minimum=0.5, maximum=10.0)
        mapping = players.set_index('player_id').player_possession
        states = [initial_state(tracking, r.frame_end, mapping[r.player_id], r.attacking_side) for r in rows.itertuples(index=False)]
        save_states(out / 'states.npz', states, rows[['x_target', 'y_target']].to_numpy())
        rows.to_parquet(out / 'rows.parquet', index=False)
        methods = ['mean_velocity', 'fit_positions']
        trajectories = {m: [] for m in methods}
        records = []
        for row in rows.itertuples(index=False):
            record = {k: getattr(row, k) for k in ['match_id', 'pass_id', 'frame_end', 'outcome']}
            for method, boundary in zip(methods, estimate_both(tracking, row)):
                trajectory = np.asarray(boundary['trajectory_3d'], np.float32)
                trajectories[method].append(trajectory)
                record[method + '_forecast_error'] = forecast_error(trajectory, tracking, row)[0]
            records.append(record)
        pd.DataFrame(records).to_parquet(out / 'boundaries.parquet', index=False)
        np.savez_compressed(out / 'trajectories.npz', **{k: np.asarray(v) for k, v in trajectories.items()})
        with np.load(out / 'states.npz') as z:
            raw = {k: z[k] for k in z.files}
        tau, ball, valid = transfer_physics(raw, 5.5)
        physical = dict(tau=tau, t_ball=ball, valid=valid)
        for method, trajectories in trajectories.items():
            path = np.asarray(trajectories)
            with enable_x64(False):
                physical['tau_path_' + method] = np.asarray(_path_times(jnp.asarray(raw['p_pos'], jnp.float32), jnp.asarray(raw['p_vel'], jnp.float32), jnp.asarray(path[:, :, :2], jnp.float32), 5.5))
            physical['available_' + method] = path[:, :, 2] <= 2.0
        np.savez_compressed(out / 'physics.npz', **physical)
        print('PASS_FEATURES', match, len(rows), flush=True)


def load(paths):
    rows = []
    boundaries = []
    data = {}
    for folder in sorted(paths.task('pass_probability', 'features').iterdir()):
        if not folder.is_dir():
            continue
        rows.append(pd.read_parquet(folder / 'rows.parquet'))
        boundaries.append(pd.read_parquet(folder / 'boundaries.parquet'))
        with np.load(folder / 'physics.npz') as z:
            for key in z.files:
                data.setdefault(key, []).append(z[key])
    return (pd.concat(rows, ignore_index=True),
        pd.concat(boundaries, ignore_index=True),
        {k: np.concatenate(v) for k, v in data.items()})


def estimate(rows, boundaries, data, matches):
    method, selection = choose_boundary(boundaries, matches)
    take = rows.match_id.isin(matches).to_numpy()
    fitted = fit(data['tau_path_' + method][take],
        data['available_' + method][take],
        data['valid'][take],
        rows.outcome.to_numpy()[take])
    fitted.update(boundary=method, boundary_selection=selection, fit_matches=list(matches))
    return fitted


def endpoint(fitted, data, take):
    return np.asarray(jnp.exp(predict_logs(jnp.asarray(fitted['log_parameters']), jnp.asarray(data['tau'][take]), jnp.asarray(data['t_ball'][take]), jnp.asarray(data['valid'][take]), dt=0.01)[0]))


def artifact(fitted, x, y):
    p = dict(v_max=5.5,
        a_max=7.0,
        a_turn=7.0,
        preserve_initial_overspeed=True,
        dt_ball=0.04,
        dt_pc=0.01,
        t_max_flight=10.0,
        t_max_pc=15.0,
        rho=1.22,
        Cd=0.25,
        A=0.038,
        m=0.42,
        g=9.81,
        z_max=2.0,
        distance_mode='exact',
        integration_method='exponential',
        speed_range=[5.0, 25.0],
        speed_candidates=30,
        elevation_degrees=[0.0, 45.0],
        elevation_candidates=30)
    iso = IsotonicRegression(out_of_bounds='clip').fit(x, y)
    result = dict(schema_version=1,
        model_type='pass_risk_trajectory_isotonic',
        model_id='fresh_pass_risk_refit',
        lambda_control=fitted['lambda_control'],
        s=fitted['s'],
        physics=p,
        fit=fitted,
        trajectory_boundary=fitted['boundary'],
        calibration=dict(method='isotonic', x=iso.X_thresholds_.tolist(), y=iso.y_thresholds_.tolist()))
    PassRiskModel(result)
    return result


def train(paths, fold):
    rows, boundaries, data = load(paths)
    split = folds()[str(fold)]
    out = fresh_directory(paths.task('pass_probability', 'training') / f'fold_{fold}')
    outer = sorted(split['fit_matches'] + split['validation_matches'])
    calibration = np.full(len(rows), np.nan)
    inner = []
    six = None
    for i in range(4):
        held = outer[2 * i:2 * i + 2]
        training = [m for m in outer if m not in held]
        fitted = estimate(rows, boundaries, data, training)
        take = rows.match_id.isin(held).to_numpy()
        prediction = endpoint(fitted, data, take)
        calibration[take] = prediction
        inner.append(fitted)
        if set(held) == set(split['validation_matches']):
            six = artifact(fitted, prediction, rows.outcome.to_numpy()[take])
    fitted = estimate(rows, boundaries, data, outer)
    tr = rows.match_id.isin(outer).to_numpy()
    te = rows.match_id.isin(split['test_matches']).to_numpy()
    model = artifact(fitted, calibration[tr], rows.outcome.to_numpy()[tr])
    raw = endpoint(fitted, data, te)
    pred = rows.loc[te].copy()
    pred['raw'] = raw
    pred['probability'] = PassRiskModel(model).calibrate(raw)
    pred['y'] = pred.outcome
    pred['null_probability'] = float(rows.loc[tr, 'outcome'].mean())
    pred['fold'] = fold
    pred.to_parquet(out / 'test_predictions.parquet', index=False)
    write_json(out / 'model.json', model)
    write_json(out / 'six_fit_model.json', six)
    write_json(out / 'inner_fits.json', inner)


def score(paths):
    out = fresh_directory(paths.task('pass_probability', 'scores'))
    pred = pd.concat([pd.read_parquet(paths.task('pass_probability', 'training') / f'fold_{f}' / 'test_predictions.parquet') for f in range(5)],
        ignore_index=True)
    metrics, bins = binary_scores(pred.y, pred.probability, pred.null_probability, floor=1e-06)
    pred.to_parquet(out / 'pooled_predictions.parquet', index=False)
    write_json(out / 'metrics.json', metrics)
    bins.to_csv(out / 'calibration.csv', index=False)


def refit(paths):
    rows, boundaries, data = load(paths)
    fitted = estimate(rows, boundaries, data, sorted(rows.match_id.unique()))
    pred = pd.read_parquet(paths.task('pass_probability', 'scores') / 'pooled_predictions.parquet')
    out = fresh_directory(paths.task('pass_probability', 'production'))
    write_json(out / 'pass_risk_model.json', artifact(fitted, pred.raw, pred.y))
