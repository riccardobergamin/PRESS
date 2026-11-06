"""Carrier covariates and defender-specific approach angles from the initial state."""
from dataclasses import asdict
import numpy as np
from .splines import TIMES, MarginalBasis, SplineDesign

def interception_angle(raw):
    """Angle between BC velocity and defender final approach velocity.

    0 = co-moving, pi = head-on. The predicted final approach to x_B(t)
    follows the historical turn-then-straight routine. At contact <=.1m use
    initial defender velocity. This is a conditional arrival mark, not an
    observed future direction. A BC speed below 1m/s has no reliable heading;
    its angle contribution is zero. Absent defenders are also masked.
    """
    target = raw['x_bc'][:, None, None, :] + raw['v_bc'][:, None, None, :] * TIMES[None, :, None, None]
    delta = target - raw['r_defs'][:, None, :, :]
    distance = np.linalg.norm(delta, axis=-1)
    direction = np.where((distance <= 0.1)[..., None], raw['v_defs'][:, None, :, :], delta)
    norm = np.linalg.norm(direction, axis=-1)
    speed = np.linalg.norm(raw['v_bc'], axis=-1)
    cosine = np.sum(direction * raw['v_bc'][:, None, None, :], axis=-1) / np.maximum(norm * speed[:, None, None], 1e-12)
    angle = np.arccos(np.clip(cosine, -1, 1))
    valid = (speed[:, None, None] >= 1.0) & (norm > 1e-08) & (np.linalg.norm(raw['r_defs'], axis=-1)[:, None, :] < 500)
    return (angle, valid)

def signed_boundary(raw):
    position = raw['x_bc'][:, None, :] + raw['v_bc'][:, None, :] * TIMES[None, :, None]
    x, y = (position[..., 0], position[..., 1])
    return np.minimum.reduce([52.5 - x, 52.5 + x, 34 - y, 34 + y])

def design_split(training, validation, shape):
    """Return independent copies with basis state fitted exclusively on training."""
    spec = SplineDesign.fit(training['speed'], training['boundary'], 'additive')
    angle = MarginalBasis.fit(training['angle'][training['angle_valid']], n_knots=4) if shape == 'additive_angle' else None
    prepared = []
    matrices = []
    for original in [training, validation]:
        data = {k: v for k, v in original.items() if k not in {'angle', 'angle_valid'}}
        if angle is not None:
            data['angle_basis'] = angle.full(original['angle']) * original['angle_valid'][..., None]
        prepared.append(data)
        matrices.append(spec.evaluate(data['speed'], data['boundary']))
    return (spec, angle, prepared, matrices)

def active(spec, angle, candidate):
    count, precision = spec.active(candidate)
    kind = 'none'
    if candidate['angle'] is not None:
        if angle is None:
            raise ValueError('Missing angle basis')
        precision = np.r_[precision, 1.0, candidate['angle'] * np.asarray(angle.penalty)]
        kind = 'individual_angle'
    return (count, precision, kind)
