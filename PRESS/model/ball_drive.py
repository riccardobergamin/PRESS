"""Load and evaluate exported additive ball-drive models without training code."""

from __future__ import annotations

from copy import deepcopy
from functools import partial
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def _basis(spec, values, *, full=False):
    """Evaluate the spline basis with its saved projection and normalization."""
    z = (np.asarray(values, float) - spec['offset']) / spec['scale']
    knots = np.asarray(spec['knots'])
    terms = np.maximum(z[..., None] - knots, 0.0) ** 3
    a = (knots[-1] - knots[:-2]) / (knots[-1] - knots[-2])
    b = (knots[-2] - knots[:-2]) / (knots[-1] - knots[-2])
    raw = terms[..., :-2] - terms[..., -2, None] * a + terms[..., -1, None] * b
    linear = np.stack([np.ones_like(z), z], axis=-1)
    # Remove the fitted linear projection, then apply the saved curvature scaling.
    nonlinear = (raw - linear @ np.asarray(spec['projection'])) @ np.asarray(spec['transform'])
    if not full:
        return nonlinear
    return np.concatenate(
        [((z - spec['linear_mean']) / spec['linear_sd'])[..., None], nonlinear],
        axis=-1,
    )


@partial(jax.jit, static_argnames=('v_max', 'a_linear', 'a_turn', 'preserve_initial_overspeed'))
def _arrivals(r, v, x, vx, times, *, v_max, a_linear, a_turn, preserve_initial_overspeed):
    """Evaluate the shared TTI using the model's saved physical conventions."""
    from PRESS.physics.times import time_to_intercept

    def snapshot(r, v, x, vx):

        def at_time(t):
            target = (x + vx * t)[None, :]
            return time_to_intercept(
                r, v, target,
                v_max=v_max,
                a_max=a_linear,
                a_turn=a_turn,
                preserve_initial_overspeed=preserve_initial_overspeed,
            ).squeeze(-1)

        return jax.vmap(at_time)(times)

    return jax.vmap(snapshot)(r, v, x, vx)


class BallDriveModel:
    """An exported MAP model, its spline bases, physics and provenance.

    Use ``from_json`` to load a refit artifact; this class never fits weights.
    ``predict`` evaluates one initial frame at any nonnegative horizon.
    ``predict_batch`` accepts aligned arrays of independent initial frames.
    Raw inference probabilities have no Bernoulli floor; the benchmark uses
    the explicitly recorded training floor when scoring its one-second target.
    """

    def __init__(self, artifact):
        self._artifact = deepcopy(artifact)
        a = self._artifact
        if a.get('schema_version') != 1 or a.get('model_type') != 'ball_drive_additive':
            raise ValueError('Unsupported ball-drive artifact schema.')
        self.theta = np.asarray(a['theta'], float)
        self.design = a['design']
        self.angle_design = a.get('angle_design')
        self.count = int(a['active_spline_columns'])
        self.kind = a['kind']
        if self.design['variant'] != 'additive' or self.kind not in {'none', 'individual_angle'}:
            raise ValueError('Artifact must describe an additive ball-drive model.')
        expected = 4 + self.count + (3 if self.kind == 'individual_angle' else 0)
        if self.theta.shape != (expected,) or not np.isfinite(self.theta).all():
            raise ValueError('Invalid ball-drive parameter vector.')
        if self.count not in {0, 6}:
            raise ValueError('Unsupported active spline layout.')
        for spec in [self.design['speed_basis'], self.design['boundary_basis'], self.angle_design]:
            if spec is None:
                continue
            knots = np.asarray(spec['knots'], float)
            k = len(knots) - 2
            if (
                spec['scale'] <= 0
                or spec['linear_sd'] <= 0
                or (not np.isfinite(knots).all())
                or np.any(np.diff(knots) <= 0)
                or (np.shape(spec['projection']) != (2, k))
                or (np.shape(spec['transform']) != (k, k))
            ):
                raise ValueError('Invalid spline basis.')
            if (
                not np.isfinite(np.asarray(spec['projection'])).all()
                or not np.isfinite(np.asarray(spec['transform'])).all()
            ):
                raise ValueError('Nonfinite spline basis.')
        if (
            self.kind == 'individual_angle'
            and (
                self.angle_design is None
                or len(self.angle_design['knots']) != 4
            )
        ):
            raise ValueError('Missing four-knot angle spline.')
        if not np.isfinite(np.exp(self.theta[3])) or np.exp(self.theta[3]) <= 0:
            raise ValueError('Arrival dispersion must be positive and finite.')
        self._physics(dict(a['physics']))
        self.theta.setflags(write=False)

    @classmethod
    def from_json(cls, path):
        """Load a self-contained JSON artifact; no pickle or research imports."""
        return cls(json.loads(Path(path).read_text()))

    @classmethod
    def pretrained(cls):
        """Load the bundled ten-match SkillCorner outfield-carrier refit.

        This model uses the documented one-second target and benchmark physics.
        The artifact records the separate pooled OOF evaluation and provenance.
        """
        from importlib.resources import files

        return cls(json.loads(files('PRESS.utils').joinpath('data/ball_drive_model.json').read_text()))

    def to_dict(self):
        """Return a copy of parameters, bases, physics and provenance."""
        return deepcopy(self._artifact)

    @property
    def parameters(self):
        """Named fitted parameters; positive scales are in their natural units.

        gamma_v/gamma_B multiply the saved nonlinear speed/boundary bases.
        beta_angle multiplies the saved standardized linear angle term;
        gamma_angle multiplies its two nonlinear components.
        """
        values = dict(lambda_0=float(np.exp(self.theta[0])),
                      beta_v=float(self.theta[1]), beta_B=float(self.theta[2]),
                      s=float(np.exp(self.theta[3])))
        if self.count:
            values.update(gamma_v=self.theta[4:7].tolist(),
                          gamma_B=self.theta[7:10].tolist())
        if self.kind == 'individual_angle':
            start = 4+self.count
            values.update(beta_angle=float(self.theta[start]),
                          gamma_angle=self.theta[start+1:start+3].tolist())
        return values

    @classmethod
    def from_parameters(cls, parameters, *, base=None):
        """Apply named inputs to a fitted base without refitting or mutating it.

        Missing values/bases are inherited from base (production if omitted).
        A complete exported artifact is also accepted unchanged. Supplying
        another basis requires its corresponding fitted coefficients.
        """
        if not isinstance(parameters, dict):
            raise TypeError('Ball-drive model parameters must be a dictionary.')
        if 'schema_version' in parameters:
            return cls(parameters)
        if base is None:
            base = cls.pretrained()
        if not isinstance(base, cls):
            raise TypeError('base must be a BallDriveModel.')
        allowed = {'lambda_0','beta_v','beta_B','s','gamma_v','gamma_B',
                   'beta_angle','gamma_angle','speed_basis','boundary_basis',
                   'angle_basis','model_id'}
        unknown = set(parameters)-allowed
        if unknown:
            raise ValueError('Unknown ball-drive parameters: '+str(sorted(unknown)))
        artifact = base.to_dict()
        theta = base.theta.copy()
        for key, index in [('lambda_0',0),('beta_v',1),('beta_B',2),('s',3)]:
            if key not in parameters:
                continue
            try:
                value = float(parameters[key])
            except (TypeError, ValueError):
                raise ValueError(f'{key} must be a finite scalar.') from None
            if not np.isfinite(value) or (key in {'lambda_0','s'} and value <= 0):
                raise ValueError(f'{key} must be finite'+(' and positive.' if key in {'lambda_0','s'} else '.'))
            # Keep an exact round trip when the natural-scale input is unchanged.
            if key in {'lambda_0','s'}:
                if value != float(np.exp(theta[index])):
                    theta[index] = np.log(value)
            else:
                theta[index] = value
        for key, start, count in [('gamma_v',4,3),('gamma_B',7,3),
                                  ('gamma_angle',5+base.count,2)]:
            if key not in parameters:
                continue
            if (key in {'gamma_v','gamma_B'} and not base.count
                    or key == 'gamma_angle' and base.kind != 'individual_angle'):
                raise ValueError(f'{key} has no active basis in the selected model.')
            values = np.asarray(parameters[key], float)
            if values.shape != (count,) or not np.isfinite(values).all():
                raise ValueError(f'{key} must contain {count} finite coefficients.')
            theta[start:start+count] = values
        if 'beta_angle' in parameters:
            if base.kind != 'individual_angle':
                raise ValueError('beta_angle has no active basis in the selected model.')
            value = np.asarray(parameters['beta_angle'], float)
            if value.shape != () or not np.isfinite(value):
                raise ValueError('beta_angle must be a finite scalar.')
            theta[4+base.count] = float(value)
        for key, coefficient_keys in [
            ('speed_basis',{'gamma_v'}), ('boundary_basis',{'gamma_B'}),
            ('angle_basis',{'beta_angle','gamma_angle'}),
        ]:
            if key not in parameters:
                continue
            if not coefficient_keys.issubset(parameters):
                raise ValueError(f'{key} also requires coefficients: {sorted(coefficient_keys)}')
            if not isinstance(parameters[key],dict):
                raise ValueError(f'{key} must be a fitted basis dictionary.')
            if key == 'angle_basis':
                artifact['angle_design'] = deepcopy(parameters[key])
            else:
                artifact['design'][key] = deepcopy(parameters[key])
        artifact['theta'] = theta.tolist()
        if parameters:
            identifier = parameters.get('model_id',base.model_id+':custom')
            if not isinstance(identifier,str) or not identifier:
                raise ValueError('model_id must be a nonempty string.')
            artifact['model_id'] = identifier
        return cls(artifact)

    @property
    def model_id(self):
        """Identifier recorded by the exporting training run."""
        return self._artifact['model_id']

    @property
    def physics(self):
        """Return copied physics settings used by this fitted model."""
        return dict(self._artifact['physics'])

    @staticmethod
    def _physics(p):
        """Validate the physical settings supported by this exported model."""
        required = {
            'v_max',
            'a_linear',
            'a_turn',
            'preserve_initial_overspeed',
            'pitch_length',
            'pitch_width',
            'dt',
            'integration_method',
            'boundary_mode',
        }
        if set(p) != required:
            raise ValueError('Unknown or missing fitted physics settings.')
        values = [p[k] for k in [
            'v_max',
            'a_linear',
            'a_turn',
            'pitch_length',
            'pitch_width',
            'dt',
        ]]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('Physics scales must be finite and positive.')
        if (
            p['integration_method'] != 'exponential'
            or p['boundary_mode'] not in {'absolute', 'signed'}
        ):
            raise ValueError('Unsupported fitted integration or boundary convention.')
        if not isinstance(p['preserve_initial_overspeed'], bool):
            raise ValueError('Overspeed convention must be boolean.')
        return p

    def predict(self, r_defs, v_defs, x_bc, v_bc, *, horizon=1.0, physics_kwargs=None):
        """Return per-defender loss, total loss and survival from one frame.

        Coordinates are centred metres and velocities m/s. A shorter horizon
        uses a partial final integration step. The fitted one-second target
        does not establish calibration at arbitrary horizons. Physics overrides
        are explicit sensitivities; the saved pitch is used unless overridden.
        """
        result = self.predict_batch(
            np.asarray(r_defs)[None],
            np.asarray(v_defs)[None],
            np.asarray(x_bc)[None],
            np.asarray(v_bc)[None],
            horizon=horizon,
            physics_kwargs=physics_kwargs,
        )
        return (
            {k: v[0]
            if isinstance(v, np.ndarray)
            and k != 'times'
            else v for k, v in result.items()}
        )

    def predict_batch(self, r_defs, v_defs, x_bc, v_bc, *, horizon=1.0, physics_kwargs=None):
        """Evaluate N independent states: defenders (N,D,2), carrier (N,2).

        Padding at norm(position)>=500m has zero defensive hazard. Real-player
        coordinates and velocities must be finite. Zero horizon or no defenders
        yields zero loss. Cause attribution is model based, not an observed
        tackler identity. ``survival_curve`` includes the initial state.
        """
        p = self._physics(self.physics | dict(physics_kwargs or {}))
        if not np.isfinite(horizon) or horizon < 0:
            raise ValueError('Horizon must be finite and nonnegative.')
        r = np.asarray(r_defs, float)
        v = np.asarray(v_defs, float)
        x = np.asarray(x_bc, float)
        vx = np.asarray(v_bc, float)
        if (
            r.ndim != 3
            or r.shape[-1] != 2
            or v.shape != r.shape
            or (x.shape != (len(r), 2))
            or (vx.shape != x.shape)
        ):
            raise ValueError('Expected defenders (N,D,2) and carrier (N,2).')
        if not all((np.isfinite(z).all() for z in [r, v, x, vx])):
            raise ValueError('State coordinates and velocities must be finite.')
        n, d = r.shape[:2]
        steps = max(0, math.ceil(horizon / p['dt'] - 1e-12))
        logical_times = np.arange(steps, dtype=float) * p['dt']
        times = np.arange(steps, dtype=np.float32) * np.float32(p['dt'])
        widths = np.minimum(p['dt'], np.maximum(horizon - logical_times, 0.0))
        causes = np.zeros((n, d))
        log_survival = np.zeros(n)
        curve = [np.ones(n)]
        if steps and d:
            cfg = {k: p[k] for k in ['v_max', 'a_linear', 'a_turn', 'preserve_initial_overspeed']}
            tti = np.asarray(
                _arrivals(
                    *[jnp.asarray(z, dtype=jnp.float32) for z in [r, v, x, vx]],
                    jnp.asarray(times),
                    **cfg,
                ),
                float,
            )
            target = x[:, None, :] + vx[:, None, :] * times[None, :, None]
            margins = np.stack(
                [
                    p['pitch_length'] / 2 - target[..., 0],
                    p['pitch_length'] / 2 + target[..., 0],
                    p['pitch_width'] / 2 - target[..., 1],
                    p['pitch_width'] / 2 + target[..., 1],
                ],
                axis=-1,
            )
            boundary = np.min(margins if p['boundary_mode'] == 'signed' else abs(margins), axis=-1)
            speed = np.linalg.norm(vx, axis=-1)
            matrix = np.concatenate(
                [
                    _basis(
                        self.design['speed_basis'],
                        np.broadcast_to(speed[:, None], boundary.shape),
                    ),
                    _basis(self.design['boundary_basis'], boundary),
                ],
                axis=-1,
            )[..., :self.count]
            eta = (
                self.theta[0]
                + self.theta[1] * speed[:, None]
                + self.theta[2] * boundary
                + matrix @ self.theta[4:4 + self.count]
            )
            valid = np.linalg.norm(r, axis=-1) < 500
            log_rate = np.broadcast_to(eta[..., None], tti.shape).copy()
            if self.kind == 'individual_angle':
                delta = target[:, :, None, :] - r[:, None, :, :]
                distance = np.linalg.norm(delta, axis=-1)
                direction = np.where((distance <= 0.1)[..., None], v[:, None, :, :], delta)
                norm = np.linalg.norm(direction, axis=-1)
                cosine = (
                    np.sum(direction * vx[:, None, None, :], axis=-1)
                    / np.maximum(norm * speed[:, None, None], 1e-12)
                )
                angle = np.arccos(np.clip(cosine, -1, 1))
                available = (speed[:, None, None] >= 1.0) & (norm > 1e-08) & valid[:, None, :]
                basis = _basis(self.angle_design, angle, full=True) * available[..., None]
                log_rate += basis @ self.theta[4 + self.count:]
            z = np.pi / (np.sqrt(3) * np.exp(self.theta[3])) * (times[None, :, None] - tti)
            gates = np.exp(-np.logaddexp(0.0, -z)) * valid[:, None, :]
            with np.errstate(over='raise', invalid='raise'):
                try:
                    rates = gates * np.exp(log_rate)
                except FloatingPointError as exc:
                    raise ValueError('Fitted rates exceeded the finite numerical range.') from exc
            # Allocate each survival decrement among competing defender causes.
            for i, width in enumerate(widths):
                intensity = rates[:, i]
                total = intensity.sum(axis=1)
                exposure = width * total
                if not np.isfinite(exposure).all():
                    raise ValueError('Nonfinite integrated hazard.')
                # Taylor expansion avoids cancellation for very small exposures.
                small = exposure < 1e-05
                zs = np.where(small, exposure, 0.0)
                factor = width * np.where(
                    small,
                    1 - zs / 2 + zs ** 2 / 6 - zs ** 3 / 24,
                    -np.expm1(-exposure) / np.where(small, 1.0, exposure),
                )
                causes += np.exp(log_survival)[:, None] * factor[:, None] * intensity
                log_survival -= exposure
                curve.append(np.exp(log_survival))
        else:
            curve.extend((np.ones(n) for _ in range(steps)))
        survival = np.exp(log_survival)
        return dict(
            loss_probability=-np.expm1(log_survival),
            survival_probability=survival,
            cause_probabilities=causes,
            survival_curve=np.stack(curve, axis=1),
            times=np.r_[logical_times, horizon],
            physics=p,
            model_id=self.model_id,
            horizon=float(horizon),
        )
