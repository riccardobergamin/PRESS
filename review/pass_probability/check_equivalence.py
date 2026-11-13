"""Small real-data equivalence checks; does not rerun the complete benchmark.

Only this verification script imports the frozen reference pipeline. The three
review scripts execute their own procedures and reuse PRESS physics directly.
"""
import hashlib
import json
import sys
import importlib.util
import tempfile
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from model import (
    precision_context, prepare_pitch_control_inputs, trajectory_arrival_times,
    integrate_pass_probability_logs, integrate_pitch_control_logs,
    negative_log_likelihood, estimate_launch_conditions,
)
from training import fit_mle, binary_metrics, fit_calibrators, apply_calibrators, collect_results
# Resolve the REFERENCE package rather than the neighbouring pipeline.py CLI.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'PRESS-portfolio-work'))
from pipeline.library.pass_probability.transfer import transfer_physics, _path_times
from pipeline.library.pass_probability.boundary import estimate_both
from pipeline.library.pass_probability.trajectory_model import predict_logs, nll, fit as reference_fit
from pipeline.library.pass_probability.stable_likelihood import predict_logs as reference_endpoint
from PRESS.model import PassRiskModel


ROOT = Path(__file__).resolve().parents[2]
GITHUB = ROOT.parent
DATA = ROOT / 'review/inputs/pass_probability'
TRACKING = ROOT / 'review/inputs/shared/tracking'
PACKAGE = GITHUB / 'PRESS-portfolio-work'
FROZEN = ROOT / 'output/final_pipeline/pass_probability'
OOF = ROOT / 'experiments/41_pass_risk_uniform_tti_20261022/05_results_review/run/report/02_vmax_5_5_oof.parquet'
RESULTS = Path(__file__).parent / 'checks/equivalence.json'


def main():
    jax.config.update('jax_enable_x64', True)
    artifact_path = PACKAGE / 'PRESS/utils/data/pass_risk_model.json'
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    rows = pd.read_parquet(DATA / 'rows.parquet')
    rows['match_id'] = rows.match_id.astype(str)
    with np.load(DATA / 'arrays.npz') as stored:
        raw = {name: stored[name] for name in ['p_pos', 'p_vel', 'r_target', 'x_bc']}
    # Diverse initial states: matches, outcomes, short/long flight and overspeed.
    indices = sorted(set(np.linspace(0, len(rows)-1, 24, dtype=int).tolist()
                         + rows.index[rows.outcome.eq(0)][:8].tolist()
                         + np.argsort(np.linalg.norm(raw['p_vel'], axis=-1).max(axis=1))[-4:].tolist()))
    sample = {name: values[indices] for name, values in raw.items()}
    endpoint, ball, valid = prepare_pitch_control_inputs(sample)
    old_endpoint, old_ball, old_valid = transfer_physics(sample, 5.5)
    np.testing.assert_array_equal(endpoint, old_endpoint)
    np.testing.assert_array_equal(ball, old_ball)
    np.testing.assert_array_equal(valid, old_valid)
    report = dict(n_selected=len(rows), sample_size=len(indices), endpoint_tti_max_difference=0.,
                  hypothetical_flight_max_difference=0.)
    paths = []
    for index in indices:
        row = rows.iloc[index]
        local_rows = rows.loc[rows.match_id.eq(row.match_id)]
        local_index = local_rows.index.get_loc(index)
        with np.load(DATA / 'matches' / row.match_id / 'trajectories.npz') as stored:
            paths.append(stored['fit_positions'][local_index])
    paths = np.asarray(paths)
    with precision_context(False):
        args = [jnp.asarray(sample['p_pos']), jnp.asarray(sample['p_vel']), jnp.asarray(paths[:, :, :2])]
        path_tti = np.asarray(trajectory_arrival_times(*args))
        old_path_tti = np.asarray(_path_times(*args, 5.5))
    np.testing.assert_array_equal(path_tti, old_path_tti)
    report['path_tti_max_difference'] = 0.
    available = paths[:, :, 2] <= 2.
    parameters = jnp.log(jnp.array([5.437829752595985, .29455974908543886]))
    arguments = tuple(jnp.asarray(value) for value in [path_tti, available, valid, rows.iloc[indices].outcome.to_numpy()])
    new_logs = np.asarray(integrate_pass_probability_logs(parameters, *arguments[:3]))
    old_logs = np.asarray(predict_logs(parameters, *arguments[:3]))
    np.testing.assert_allclose(new_logs, old_logs, rtol=0, atol=1e-12)
    report['path_log_probability_max_difference'] = float(abs(new_logs-old_logs).max())
    new_value, new_gradient = jax.value_and_grad(negative_log_likelihood)(parameters, *arguments)
    old_value, old_gradient = jax.value_and_grad(nll)(parameters, *arguments)
    np.testing.assert_allclose(new_value, old_value, atol=1e-12, rtol=0)
    np.testing.assert_allclose(new_gradient, old_gradient, atol=1e-10, rtol=0)
    new_hessian = np.asarray(jax.hessian(negative_log_likelihood)(parameters, *arguments))
    old_hessian = np.asarray(jax.hessian(nll)(parameters, *arguments))
    np.testing.assert_allclose(new_hessian, old_hessian, atol=1e-9, rtol=0)
    report.update(nll_difference=float(abs(new_value-old_value)),
                  gradient_max_difference=float(abs(new_gradient-old_gradient).max()),
                  hessian_max_difference=float(abs(new_hessian-old_hessian).max()))
    new_endpoint = np.asarray(integrate_pitch_control_logs(parameters, endpoint, ball, valid))
    old_endpoint = np.asarray(reference_endpoint(parameters, endpoint, ball, valid, dt=.01))
    np.testing.assert_allclose(new_endpoint, old_endpoint, atol=1e-12, rtol=0)
    np.testing.assert_allclose(np.exp(new_logs[0]) + np.exp(new_logs[1]), 1., atol=1e-12)
    np.testing.assert_allclose(np.exp(new_endpoint[0]) + np.exp(new_endpoint[1]), 1., atol=1e-12)
    report['endpoint_log_probability_max_difference'] = float(abs(new_endpoint-old_endpoint).max())
    print('PHYSICS/LIKELIHOOD/DERIVATIVES OK', flush=True)
    # Actual three-start MLE on this small real cohort, not all 25 nested fits.
    new_fit = fit_mle(path_tti, available, valid, np.asarray(arguments[-1]))
    old_fit = reference_fit(path_tti, available, valid, np.asarray(arguments[-1]))
    np.testing.assert_allclose(new_fit['log_parameters'], old_fit['log_parameters'], atol=1e-8, rtol=0)
    report['small_mle'] = dict(n=len(indices), starts=len(new_fit['attempts']),
                             lambda_control=new_fit['lambda_control'], s=new_fit['s'],
                             log_parameter_max_difference=float(np.max(abs(np.array(new_fit['log_parameters'])-old_fit['log_parameters']))))
    print('THREE-START MLE OK', flush=True)
    # Compare the fixed position-fit path on six actual passes.
    launch_checks = []
    for index in [indices[k] for k in [0, 2, 5, 12, 22, len(indices)-1]]:
        row = rows.iloc[index]
        tracking = pd.read_parquet(TRACKING / 'matches' / row.match_id / f'{row.match_id}_tracking.parquet')
        reference_row = SimpleNamespace(**row.to_dict(), x_target=raw['r_target'][index, 0], y_target=raw['r_target'][index, 1])
        new_path, diagnostics = estimate_launch_conditions(tracking, row.frame_end, raw['r_target'][index])
        old_position_fit = estimate_both(tracking, reference_row)[1]
        old_path = np.asarray(old_position_fit['trajectory_3d'], np.float32)
        difference = float(abs(new_path-old_path).max())
        np.testing.assert_allclose(new_path, old_path, atol=1e-5, rtol=0)
        assert diagnostics['fit_usable'] == old_position_fit['fit_usable']
        launch_checks.append(dict(match_id=row.match_id, pass_id=int(row.pass_id), max_difference=difference))
    report['launch_reconstruction'] = launch_checks
    print('LAUNCH RECONSTRUCTION OK', flush=True)
    # Audit the previous split-level choices. The new per-pass selection is
    # a different protocol; archived score replay does not evaluate that change.
    inner_exceptions = []
    outer_methods = []
    for fold in range(5):
        stored = json.loads((FROZEN / 'training' / f'fold_{fold}' / 'models.json').read_text())
        outer_methods.append(stored['fit']['boundary'])
        for rotation, fitted in enumerate(stored['internal']):
            if fitted['boundary'] != 'fit_positions':
                inner_exceptions.append(dict(fold=fold,rotation=rotation,method=fitted['boundary']))
    assert outer_methods == ['fit_positions'] * 5
    report['archived_method_audit'] = dict(
        outer_position_fits=5,inner_position_fits=20-len(inner_exceptions),
        inner_exceptions=inner_exceptions,
        note='Per-pass trajectory selection can change MLE and calibration; archived score replay is not its benchmark.',
    )
    predictions = pd.read_parquet(OOF)
    predictions['match_id'] = predictions.match_id.astype(str)
    predictions = predictions.sort_values(['match_id', 'pass_id', 'frame_end'], kind='stable').reset_index(drop=True)
    calibration_errors = []
    for fold, group in predictions.groupby('fold'):
        calibration = json.loads((FROZEN / 'training' / f'fold_{fold}' / 'models.json').read_text())['calibration']
        adjusted = apply_calibrators(group.trajectory_raw.to_numpy(), calibration)
        for method in ['trajectory_platt', 'trajectory_isotonic']:
            difference = float(abs(adjusted[method]-group[method].to_numpy()).max())
            np.testing.assert_allclose(adjusted[method], group[method], atol=1e-12, rtol=0)
            calibration_errors.append(difference)
    report['saved_calibration_max_difference'] = max(calibration_errors)
    production = json.loads(artifact_path.read_text())
    calibration = fit_calibrators(predictions.trajectory_raw.to_numpy(), predictions.outcome.to_numpy())
    np.testing.assert_allclose(calibration['isotonic_x'], production['calibration']['x'], atol=1e-12, rtol=0)
    np.testing.assert_allclose(calibration['isotonic_y'], production['calibration']['y'], atol=1e-12, rtol=0)
    model = PassRiskModel(production)
    package_prediction = model.predict_from_times(endpoint.T, ball, valid.T, np.arange(10))
    # Package returns raw/calibrated fields (inspect against accepted API).
    report['package_prediction_keys'] = list(package_prediction)
    np.testing.assert_allclose(package_prediction['raw'], np.exp(new_endpoint[0]), atol=1e-7, rtol=0)
    report['package_raw_max_difference'] = float(abs(package_prediction['raw']-np.exp(new_endpoint[0])).max())
    metrics, _ = binary_metrics(predictions.outcome, predictions.trajectory_isotonic, predictions.null_global)
    expected = json.loads((FROZEN / 'scores/frozen_specification.json').read_text())['metrics']
    for key in ['cross_entropy', 'null_cross_entropy', 'gain_ce', 'brier', 'ECE_quantile', 'ROC_AUC']:
        np.testing.assert_allclose(metrics[key], expected[key], atol=1e-12, rtol=0)
    report['replayed_pooled_isotonic_metrics'] = metrics

    # Exercise actual feature-cache I/O on 20 selected real passes. Temporary
    # fixtures contain only these rows and their unchanged accepted arrays.
    spec = importlib.util.spec_from_file_location('review_pass_pipeline', Path(__file__).with_name('pipeline.py'))
    workflow = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workflow)
    with tempfile.TemporaryDirectory(prefix='press-pass-review-') as temporary:
        temporary = Path(temporary)
        input_folder, output_folder = temporary / 'input', temporary / 'output'
        input_folder.mkdir()
        smoke_indices = []
        for match, group in rows.groupby('match_id', sort=False):
            # One success and one failure per match; keep source row order.
            smoke_indices.extend([group.index[group.outcome.eq(y)][0] for y in [0, 1]])
        smoke_indices = sorted(smoke_indices)
        smoke_rows = rows.iloc[smoke_indices].reset_index(drop=True)
        smoke_rows.to_parquet(input_folder / 'rows.parquet', index=False)
        np.savez(input_folder / 'arrays.npz', **{key: value[smoke_indices] for key, value in raw.items()})
        # No precomputed paths/errors in this fixture: the pipeline must
        # generate them from the shared tracking before calculating TTI.
        trajectories_folder = workflow.generate_trajectories(input_folder, output_folder, TRACKING)
        workflow.prepare_features(input_folder, trajectories_folder, output_folder)
        loaded_rows, loaded_inputs = workflow.load_features(output_folder)
        pd.testing.assert_frame_equal(loaded_rows, smoke_rows)
        assert loaded_inputs['path_tti'].shape == (20, 251, 22)
        # Score collection is replayed from real archived test predictions;
        # no optimized parameters or predictions are fabricated in this check.
        for fold, group in predictions.groupby('fold'):
            destination = output_folder / 'training' / f'fold_{fold}'
            destination.mkdir(parents=True)
            group.to_parquet(destination / 'test_predictions.parquet', index=False)
            (destination / 'complete.json').write_text('{}')
        collected = collect_results(rows, output_folder)
        pooled_iso = collected.loc[collected.model.eq('trajectory_isotonic')].iloc[0]
        np.testing.assert_allclose(pooled_iso.cross_entropy, expected['cross_entropy'], rtol=0, atol=1e-12)
        assert (output_folder / 'scores/calibration_deciles.csv').is_file()
        try:
            workflow.prepare_features(input_folder, trajectories_folder, output_folder)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Feature stage must refuse to overwrite an existing output.')
    report['pipeline_io'] = dict(real_passes=20, matches=10, pooled_score_replay=True, overwrite_rejected=True)
    assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == digest
    report['production_artifact_unchanged_sha256'] = digest
    RESULTS.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
