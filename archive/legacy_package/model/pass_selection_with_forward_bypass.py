"""Frozen destination-choice ensembles and their training-time feature contract."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
import xgboost as xgb

LEGACY_FEATURE_CONTRACT = 'forward_bypass_v1'
AHEAD_FEATURE_CONTRACT = 'ahead_of_carrier_and_target_v1'


def choice_feature_names(vocabulary, *, feature_contract=LEGACY_FEATURE_CONTRACT):
    """Ordered predictors for the explicitly versioned choice feature contract."""
    names = [
        'pass_distance',
        'pass_angle',
        'passer_speed',
        'target_pitch_control',
        'target_xt_team',
        'target_xt_opp',
        'opp_pressure_line_index',
        'team_pressure_line_index',
        'ball_drive_prob',
        'xt_team_bc',
        'xt_opp_bc',
        'opp_pressure_index_bc',
        'team_pressure_index_bc',
    ]
    for column, categories in vocabulary.items():
        names += [column + '=' + str(value) for value in categories]
    names += ['match_clock_min', 'score_difference_prefix']
    if feature_contract == AHEAD_FEATURE_CONTRACT:
        return names + [
            'opponents_ahead_bc',
            'teammates_ahead_bc',
            'opponents_ahead_target',
            'teammates_ahead_target',
        ]
    if feature_contract == LEGACY_FEATURE_CONTRACT:
        return names + ['opponents_bypassed', 'teammates_bypassed']
    raise ValueError('Unknown pass-selection feature contract.')


def build_choice_features(
    *,
    x_bc,
    v_bc,
    r_atts,
    r_defs,
    targets,
    attacking_side,
    target_pitch_control,
    ball_drive_prob,
    xt_grid,
    vocabulary,
    start_type_raw=None,
    previous_action_raw=None,
    match_clock_min=np.nan,
    score_difference_prefix=np.nan,
    pitch_length=105.0,
    pitch_width=68.0,
    pressure_lines=None,
    feature_contract=LEGACY_FEATURE_CONTRACT,
):
    """Build one choice set using the final benchmark's feature conventions.

    Coordinates are centred metres, velocities m/s. ``r_atts`` excludes the
    carrier. Raw team control must exclude the carrier; calibrated pass
    success is a different quantity. ``ball_drive_prob`` is one-second LOSS.
    Context is known at the frame, not the subsequent receiver or outcome.
    Missing/unseen raw categories retain their explicit trained buckets.
    Optional pressure_lines=(opponent, own) supplies already computed centres.
    ``ahead_of_carrier_and_target_v1`` counts players strictly beyond each
    transverse query line along the CARRIER TEAM's attack direction, for both
    rosters. Current positions are used; backward targets may increase counts.
    The legacy default preserves existing saved ensembles without this field.
    """
    from PRESS.utils.state import compute_dynamic_pressure_lines

    if attacking_side not in ('left_to_right', 'right_to_left'):
        raise ValueError('Unknown attacking side.')
    bc, velocity, targets = [np.asarray(z) for z in (x_bc, v_bc, targets)]
    if (
        bc.shape != (2,)
        or velocity.shape != (2,)
        or targets.ndim != 2
        or (targets.shape[1] != 2)
        or (not len(targets))
    ):
        raise ValueError('Expected carrier (2,) and a nonempty target grid (N,2).')
    if not all((np.isfinite(z).all() for z in (bc, velocity, targets))):
        raise ValueError('Carrier state and targets must be finite.')
    surface = np.asarray(xt_grid)
    if surface.shape != (10, 6) or not np.isfinite(surface).all() or (surface < 0).any():
        raise ValueError('Expected the fitted nonnegative 10x6 xT surface.')
    control = np.asarray(target_pitch_control)
    if (
        control.shape != (len(targets),)
        or not np.isfinite(control).all()
        or np.any((control < 0) | (control > 1))
    ):
        raise ValueError('Expected raw team control per target in [0,1].')
    if not np.isfinite(ball_drive_prob) or not 0 <= ball_drive_prob <= 1:
        raise ValueError('Expected finite one-second ball-drive loss in [0,1].')
    sign = 1 if attacking_side == 'left_to_right' else -1

    def xt(points, direction):
        q = np.asarray(points) * direction
        ix = np.clip(((q[:, 0] + pitch_length / 2) / pitch_length * 10).astype(int), 0, 9)
        iy = np.clip(((q[:, 1] + pitch_width / 2) / pitch_width * 6).astype(int), 0, 5)
        return surface[ix, iy]

    def clean(players):
        a = np.asarray(players)
        if a.size == 0:
            return np.empty((0, 2))
        return a[np.isfinite(a).all(axis=-1) & (np.linalg.norm(a, axis=-1) < 500)]

    own, opponents = (clean(r_atts), clean(r_defs))
    if pressure_lines is None:
        other_side = 'right_to_left' if sign == 1 else 'left_to_right'
        pressure_lines = (
            compute_dynamic_pressure_lines(opponents, attacking_side),
            compute_dynamic_pressure_lines(own, other_side),
        )

    def index(x, centres, direction):
        centres = np.sort(direction * np.asarray(centres, float))
        if len(centres) != 3 or not np.isfinite(centres).all():
            return np.full(len(x), np.nan)
        return (1 + np.argmin(
            abs(direction * np.asarray(x)[:, None] - centres[None]),
            axis=1,
        )).astype(np.float32)

    def bypass(players):
        px = sign * players[:, 0]
        return (
            (px[:, None] > sign * bc[0])
            & (px[:, None] <= sign * targets[None, :, 0])
        ).sum(axis=0).astype(np.float32)

    n = len(targets)
    constant = lambda x: np.full(n, x)
    delta = targets - bc
    speed = np.linalg.norm(velocity)
    angle = np.full(n, np.nan) if speed < 1e-08 else np.arctan2(
        velocity[0] * delta[:, 1] - velocity[1] * delta[:, 0],
        delta @ velocity,
    )
    ol, tl = pressure_lines
    features = [
        np.linalg.norm(delta, axis=1),
        angle,
        constant(speed),
        control,
        xt(targets, sign),
        xt(targets, -sign),
        index(targets[:, 0], ol, sign),
        index(targets[:, 0], tl, -sign),
        constant(ball_drive_prob),
        constant(xt(bc[None], sign)[0]),
        constant(xt(bc[None], -sign)[0]),
        constant(index(bc[:1], ol, sign)[0]),
        constant(index(bc[:1], tl, -sign)[0]),
    ]
    context = dict(start_type_raw=start_type_raw, previous_action_raw=previous_action_raw)
    for column, categories in vocabulary.items():
        value = context[column]
        missing = value is None or (isinstance(value, (float, np.floating)) and np.isnan(value))
        key = '__missing__' if missing else value if value in categories else '__unseen__'
        features += [constant(key == category) for category in categories]
    features += [constant(match_clock_min), constant(score_difference_prefix)]
    if feature_contract == AHEAD_FEATURE_CONTRACT:

        def ahead(players, locations):
            return (sign * players[:, None, 0] > sign * locations[None, :, 0]).sum(axis=0).astype(np.float32)
        features += [
            constant(ahead(opponents, bc[None])[0]),
            constant(ahead(own, bc[None])[0]),
            ahead(opponents, targets),
            ahead(own, targets),
        ]
    elif feature_contract == LEGACY_FEATURE_CONTRACT:
        features += [bypass(opponents), bypass(own)]
    else:
        raise ValueError('Unknown pass-selection feature contract.')
    matrix = np.column_stack(features).astype(np.float32)
    if np.isinf(matrix).any():
        raise ValueError('Features may be missing, but cannot be infinite.')
    return matrix


class PassSelectionModel:
    """Full-data flat or secondary cross-nested destination-choice artifact.

    Loading never trains. ``predict_features`` scores one complete choice set;
    changing its cells changes the categorical denominator. The benchmark uses
    native one-metre pitch cells. Full-data weights are distinct from OOF scores.
    """

    def __init__(self, artifact, booster):
        self._artifact = deepcopy(artifact)
        if (
            artifact.get('schema_version') != 1
            or artifact.get('model_type') != 'pass_destination_choice'
        ):
            raise ValueError('Unsupported pass-selection artifact.')
        if artifact['family'] not in ('flat', 'cnl'):
            raise ValueError('Unknown destination-choice family.')
        self.booster = booster
        self.family = artifact['family']
        self.rho = float(artifact['rho'])
        self.beta = float(artifact['calibration']['inverse_temperature'])
        self.vocabulary = deepcopy(artifact['vocabulary'])
        self.feature_contract = artifact.get('feature_contract', LEGACY_FEATURE_CONTRACT)
        self.feature_names = artifact['feature_names']
        self.xt_grid = np.asarray(artifact['xt_grid'], float)
        if self.feature_names != choice_feature_names(
            self.vocabulary,
            feature_contract=self.feature_contract,
        ):
            raise ValueError('Feature schema does not match the final choice contract.')
        if (
            booster.feature_names != self.feature_names
            or booster.num_features() != len(self.feature_names)
        ):
            raise ValueError('Booster feature names/order differ from the saved schema.')
        if not np.isfinite([self.rho, self.beta]).all() or not 1 <= self.rho <= 10 or self.beta < 0:
            raise ValueError('Invalid nesting or calibration scale.')
        if self.family == 'flat' and self.rho != 1:
            raise ValueError('A flat model must have rho=1.')
        if (
            self.xt_grid.shape != (10, 6)
            or not np.isfinite(self.xt_grid).all()
            or (self.xt_grid < 0).any()
        ):
            raise ValueError('Invalid fitted xT surface.')

    @classmethod
    def from_json(cls, path):
        """Load a JSON manifest and its adjacent, SHA256-checked UBJ ensemble."""
        path = Path(path)
        artifact = json.loads(path.read_text())
        filename = artifact['booster_file']
        if Path(filename).name != filename:
            raise ValueError('Booster must be adjacent to its manifest.')
        model_path = path.parent / filename
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != artifact['booster_sha256']:
            raise ValueError('Pass-selection booster checksum mismatch.')
        booster = xgb.Booster()
        booster.load_model(model_path)
        return cls(artifact, booster)

    @classmethod
    def pretrained(cls):
        """Load the installed operational flat refit, after cluster verification."""
        path = Path(__file__).resolve().parents[1] / 'utils/data/pass_selection_model.json'
        if not path.is_file():
            raise FileNotFoundError(
                'The final pass-selection refit is not installed yet; supply '
                'PassSelectionModel.from_json(path).'
            )
        return cls.from_json(path)

    @property
    def model_id(self):
        """Return the identifier recorded in the fitted manifest."""
        return self._artifact['model_id']

    def to_dict(self):
        """Return a copy of the model manifest."""
        return deepcopy(self._artifact)

    def build_features(self, **state):
        """Use the saved vocabulary/xT; remaining arguments describe one frame."""
        return build_choice_features(
            xt_grid=self.xt_grid,
            vocabulary=self.vocabulary,
            feature_contract=self.feature_contract,
            **state,
        )

    def predict_features(self, matrix, *, log_membership=None, calibrated=True):
        """Return normalized destination probabilities, optionally before temperature.

        CNL log_membership has shape (cells, teammates), normalized across
        teammates; an entirely absent teammate has -inf membership everywhere.
        """
        matrix = np.asarray(matrix, np.float32)
        if (
            matrix.ndim != 2
            or not len(matrix)
            or matrix.shape[1] != len(self.feature_names)
            or np.isinf(matrix).any()
        ):
            raise ValueError('Invalid feature matrix for one nonempty choice set.')
        score = self.booster.predict(
            xgb.DMatrix(matrix, feature_names=self.feature_names),
            output_margin=True,
        ).astype(float)
        if self.family == 'flat':
            logp = score - logsumexp(score)
        else:
            alpha = np.asarray(log_membership, float)
            if (
                alpha.ndim != 2
                or alpha.shape[0] != len(score)
                or (not alpha.shape[1])
                or np.isnan(alpha).any()
                or np.isposinf(alpha).any()
            ):
                raise ValueError('CNL requires valid per-cell log memberships.')
            normalization = logsumexp(alpha, axis=1, keepdims=True)
            if not np.isfinite(normalization).all():
                raise ValueError('Every cell must belong to at least one teammate nest.')
            alpha = alpha - normalization
            active = np.isfinite(alpha).any(axis=0)
            a = self.rho * (score[:, None] + alpha[:, active])
            z = logsumexp(a, axis=0)
            w = z / self.rho
            w -= logsumexp(w)
            logp = logsumexp(w[None] + a - z[None], axis=1)
        if calibrated:
            logp *= self.beta
        return np.exp(logp - logsumexp(logp))
