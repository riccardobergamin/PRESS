"""Export the final spline MAP fit in the public BallDriveModel format."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np


def make_artifact(stored, *, model_id, training, benchmark):
    case = stored['case']
    fit = stored['outer_fits'][0]['fit']
    if not fit['converged'] or not fit['laplace_valid']:
        raise ValueError('Cannot export an invalid fit')
    if case.get('integration') != 'exponential':
        raise ValueError('Export requires fitted exponential Euler')
    return dict(schema_version=1,
        model_type='ball_drive_additive',
        model_id=model_id,
        theta=list(fit['theta']),
        design=deepcopy(stored['design']),
        angle_design=deepcopy(stored['angle_design']),
        active_spline_columns=stored['active_spline_columns'],
        kind=stored['kind'],
        candidate=deepcopy(stored['candidate']),
        physics=dict(**case['dynamics'], pitch_length=105.0, pitch_width=68.0, dt=0.04, integration_method='exponential', boundary_mode=case['boundary_mode']),
        training=dict(**deepcopy(training), target='Bernoulli loss within one second', probability_floor=1e-06, velocity_convention='offline centred seven-frame tracking velocities'),
        benchmark=deepcopy(benchmark),
        fit_diagnostics={k: fit[k] for k in ['objective', 'gradient_max', 'converged', 'laplace_valid']})


def write_artifact(artifact, path):
    path = Path(path)
    if path.exists():
        raise ValueError('Refusing existing model artifact: ' + str(path))
    path.write_text(json.dumps(artifact, indent=2, allow_nan=False) + '\n')
