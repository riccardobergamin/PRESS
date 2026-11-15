"""Frozen trajectory-MLE parameters transferred to isotonic-calibrated PPCF."""

from copy import deepcopy
from functools import lru_cache
from importlib.resources import files
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from PRESS.physics.flight_ball import time_flight
from .solver_pitch_control import integrate_pitch_control, isotonic_scaling, control_from_arrival_times


@lru_cache(maxsize=8)
def _flight_candidates(speed_range, speed_candidates, elevation_degrees,
                       elevation_candidates, horizon, step, rho, Cd, area, mass, gravity):
    """Reuse the same physical launch trajectories across match frames.

    These depend on launch/ball settings only, not on the observed players,
    target positions, fitted control rates or calibration. The cache is
    bounded and stores at most eight candidate banks, outside frame results.
    """
    from PRESS.physics.flight_ball import generate_ball_trajectories
    from PRESS.physics.numerics import float64_context
    with float64_context(False):
        return generate_ball_trajectories(
            jnp.linspace(*speed_range, speed_candidates, dtype=jnp.float32),
            jnp.linspace(*np.deg2rad(elevation_degrees), elevation_candidates, dtype=jnp.float32),
            jnp.arange(0., horizon+step, step),
            rho=rho, Cd=Cd, A=area, m=mass, g=gravity,
        )


class PassRiskModel:
    """A self-contained pass-risk artifact; loading never fits a model.

    Parameters lambda_control and s are MLE estimates along reconstructed
    trajectories. The saved isotonic map is fitted to cross-fitted hypothetical
    PPCF scores, not to the trajectory scores. Inference needs only an initial
    player state, carrier position and candidate targets.
    """

    def __init__(self, artifact):
        self._artifact = deepcopy(artifact)
        if (
            artifact.get('schema_version') != 1
            or artifact.get('model_type') != 'pass_risk_trajectory_isotonic'
        ):
            raise ValueError('Unsupported pass-risk model artifact.')
        self.lambda_control = float(artifact['lambda_control'])
        self.s = float(artifact['s'])
        self.physics = deepcopy(artifact['physics'])
        self.calibration = deepcopy(artifact['calibration'])
        scales = [self.lambda_control, self.s] + [self.physics[k] for k in [
            'v_max',
            'a_max',
            'dt_ball',
            'dt_pc',
            't_max_flight',
            't_max_pc',
            'm',
            'rho',
            'A',
            'g',
        ]]
        if not np.isfinite(scales).all() or min(scales) <= 0:
            raise ValueError('Pass-risk scales must be positive and finite.')
        if (
            self.physics.get('distance_mode') != 'exact'
            or self.physics.get('integration_method') != 'exponential'
        ):
            raise ValueError('Final pass-risk requires exact distance and exponential integration.')
        if self.calibration.get('method') != 'isotonic':
            raise ValueError('Expected saved isotonic calibration.')
        isotonic_scaling([0.5], self.calibration['x'], self.calibration['y'])

    @classmethod
    def from_json(cls, path):
        """Load a verified full-refit JSON or an explicitly labelled CV fixture."""
        return cls(json.loads(Path(path).read_text()))

    @classmethod
    def pretrained(cls):
        """Load the bundled full-data refit, with OOF-trained isotonic map."""
        resource = files('PRESS.utils').joinpath('data/pass_risk_model.json')
        if not resource.is_file():
            raise FileNotFoundError(
                'Full-data pass-risk artifact is not installed yet; import the '
                'verified corrected-cohort final fit or supply '
                'PassRiskModel.from_json(path).'
            )
        return cls(json.loads(resource.read_text()))

    @property
    def model_id(self):
        """Return the identifier recorded in the fitted manifest."""
        return self._artifact['model_id']

    def to_dict(self):
        """Return a copy of fitted physics, parameters and calibration."""
        return deepcopy(self._artifact)

    @property
    def parameters(self):
        """Named physical-control parameters in their natural scales."""
        return dict(lambda_control=self.lambda_control, s=self.s)

    @classmethod
    def from_parameters(cls, parameters, *, base=None):
        """Apply named lambda_control/s inputs, inheriting omitted values.

        The existing isotonic map is retained unless calibration is supplied.
        Its parameter reference is preserved, so sensitivity calculations
        cannot be mistaken for predictions with a matching saved calibrator.
        A complete exported artifact is accepted unchanged.
        """
        if not isinstance(parameters,dict):
            raise TypeError('Pass-risk model parameters must be a dictionary.')
        if 'schema_version' in parameters:
            return cls(parameters)
        if base is None:
            base = cls.pretrained()
        if not isinstance(base,cls):
            raise TypeError('base must be a PassRiskModel.')
        unknown = set(parameters)-{'lambda_control','s','calibration','model_id'}
        if unknown:
            raise ValueError('Unknown pass-risk parameters: '+str(sorted(unknown)))
        artifact = base.to_dict()
        for key in ['lambda_control','s']:
            if key in parameters:
                try:
                    value = float(parameters[key])
                except (TypeError,ValueError):
                    raise ValueError(f'{key} must be finite and positive.') from None
                if not np.isfinite(value) or value<=0:
                    raise ValueError(f'{key} must be finite and positive.')
                artifact[key] = value
        if parameters:
            identifier = parameters.get('model_id',base.model_id+':custom')
            if not isinstance(identifier,str) or not identifier:
                raise ValueError('model_id must be a nonempty string.')
            artifact['model_id'] = identifier
            if 'calibration' in parameters:
                artifact['calibration'] = deepcopy(parameters['calibration'])
                artifact['calibration_reference'] = dict(
                    lambda_control=artifact['lambda_control'],s=artifact['s'],
                    physics=deepcopy(artifact['physics']))
            else:
                artifact['calibration_reference'] = deepcopy(base._artifact.get(
                    'calibration_reference',dict(lambda_control=base.lambda_control,
                        s=base.s,physics=base.physics)))
        return cls(artifact)

    @property
    def calibration_matches_parameters(self):
        """Whether lambda_control and s match the saved map's reference."""
        reference = self._artifact.get('calibration_reference',self._artifact)
        return self.lambda_control==reference['lambda_control'] and self.s==reference['s']

    def calibrate(self, p):
        """Apply the frozen monotone map; this performs no fitting."""
        return isotonic_scaling(p, self.calibration['x'], self.calibration['y'])

    def predict_from_times(self, tau, t_ball, valid, attacker_idx):
        """Evaluate precomputed TTI, shape (players,targets), for export checks."""
        p = self.physics
        control = control_from_arrival_times(
            jnp.asarray(tau),
            jnp.asarray(t_ball),
            jnp.asarray(valid),
            np.float64(self.lambda_control),
            np.float64(self.s),
            p['t_max_pc'],
            p['dt_pc'],
        )
        raw = np.asarray(control)[np.asarray(attacker_idx)].sum(axis=0)
        return dict(control=np.asarray(control), raw=raw, calibrated=self.calibrate(raw))

    def predict_field(
        self,
        p_pos,
        p_vel,
        x_bc,
        targets,
        attacker_idx,
        ball_carrier_idx,
        *,
        physics_kwargs=None,
    ):
        """Predict raw and calibrated pass success at candidate pitch targets.

        Initial positions/targets are centred metres, velocities m/s. Physics
        overrides are sensitivities: the saved calibrator was validated only
        with its fitted settings. No observed future ball path is required.
        Padding at norm(position)>=500m is excluded.
        """
        p = deepcopy(self.physics)
        overrides = dict(physics_kwargs or {})
        unknown = set(overrides) - set(p)
        if unknown:
            raise ValueError('Unknown pass physics settings: ' + str(sorted(unknown)))
        p.update(overrides)
        pos = np.asarray(p_pos, float)
        vel = np.asarray(p_vel, float)
        targets = np.asarray(targets, float)
        bc = np.asarray(x_bc, float)
        if (
            pos.ndim != 2
            or pos.shape[-1] != 2
            or vel.shape != pos.shape
            or (targets.ndim != 2)
            or (targets.shape[-1] != 2)
            or (bc.shape != (2,))
        ):
            raise ValueError('Expected players (P,2), targets (M,2) and carrier (2,).')
        if not all((np.isfinite(z).all() for z in [pos, vel, targets, bc])):
            raise ValueError('State inputs must be finite.')
        attackers = np.asarray(attacker_idx, int)
        receivers = attackers[attackers != ball_carrier_idx]
        from PRESS.physics.numerics import float64_context
        with float64_context(False):
            positions = jnp.asarray(pos, jnp.float32)
            velocities = jnp.asarray(vel, jnp.float32)
            grid = jnp.asarray(targets, jnp.float32)
            from PRESS.physics.times import time_to_intercept
            tau = time_to_intercept(
                positions,
                velocities,
                grid,
                v_max=p['v_max'],
                a_max=p['a_max'],
                a_turn=p.get('a_turn', p['a_max']),
                preserve_initial_overspeed=p.get('preserve_initial_overspeed', True),
            )
            candidates = _flight_candidates(
                tuple(p['speed_range']), p['speed_candidates'], tuple(p['elevation_degrees']),
                p['elevation_candidates'], p['t_max_flight'], p['dt_ball'],
                p['rho'], p['Cd'], p['A'], p['m'], p['g'],
            )
            flight = time_flight(
                jnp.asarray(bc, jnp.float32),
                grid,
                positions[receivers],
                velocities[receivers],
                v0_candidates=jnp.linspace(
                    *p['speed_range'],
                    p['speed_candidates'],
                    dtype=jnp.float32,
                ),
                theta_candidates=jnp.linspace(
                    *np.deg2rad(p['elevation_degrees']),
                    p['elevation_candidates'],
                    dtype=jnp.float32,
                ),
                **{k: p[k] for k in [
                    'v_max',
                    'a_max',
                    't_max_flight',
                    'dt_ball',
                    'z_max',
                    'rho',
                    'Cd',
                    'A',
                    'm',
                    'g',
                    'distance_mode',
                ]},
                a_turn=p.get('a_turn', p['a_max']),
                preserve_initial_overspeed=p.get('preserve_initial_overspeed', True),
                trajectories=candidates, receiver_arrival_times=tau[receivers],
            )
        valid = np.broadcast_to((np.linalg.norm(pos, axis=-1) < 500)[:, None], tau.shape).copy()
        valid[ball_carrier_idx] = False
        control = control_from_arrival_times(
            tau,
            flight,
            jnp.asarray(valid),
            np.float64(self.lambda_control),
            np.float64(self.s),
            p['t_max_pc'],
            p['dt_pc'],
            p['integration_method'],
        )
        raw = np.asarray(control)[attackers].sum(axis=0)
        reference = self._artifact.get('calibration_reference',self._artifact)
        physics_match = p == reference['physics']
        parameter_match = self.calibration_matches_parameters
        return dict(
            control=np.asarray(control),
            raw=raw,
            calibrated=self.calibrate(raw),
            flight_time=np.asarray(flight),
            physics=p,
            calibration_matches_physics=physics_match,
            calibration_matches_parameters=parameter_match,
            calibration_matches_model=physics_match and parameter_match,
            model_id=self.model_id,
        )
