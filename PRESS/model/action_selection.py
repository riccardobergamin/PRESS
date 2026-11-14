"""Versioned native binary action-choice ensemble and saved feature schema."""

from copy import deepcopy
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import xgboost as xgb

FEATURE_CONTRACT = 'canonical_action_30_v1'
NUMERIC = [
    'match_clock_min',
    'score_difference_prefix',
    'completed_canonical_drives',
    'carrier_x_attack',
    'carrier_y_attack',
    'carrier_speed',
    'carrier_velocity_attack_angle',
    'drive_success_1s',
    'bcos',
    'xt_team_drive_1s',
    'xt_opp_drive_1s',
    'team_line_index_drive_1s',
    'opp_line_index_drive_1s',
    'team_line_distance_drive_1s',
    'opp_line_distance_drive_1s',
    'teammates_ahead_bc',
    'opponents_ahead_bc',
    'pass_success_mean',
    'xt_team_pass_success',
    'xt_opp_pass_failure',
    'team_line_pass_success',
    'team_line_pass_failure',
    'opp_line_pass_success',
    'opp_line_pass_failure',
    'teammates_ahead_pass_success',
    'teammates_ahead_pass_failure',
    'opponents_ahead_pass_success',
    'opponents_ahead_pass_failure',
]
CATEGORICAL = ['start_type_raw', 'previous_action_raw']


def feature_names(vocabulary):
    """Return numeric and one-hot feature names in the fitted order."""
    return list(NUMERIC) + [column + '=' + str(level) for column, levels in vocabulary.items() for level in levels]


def encode(frame, vocabulary):
    """Fit-only categorical buckets and float32 numeric boundary."""
    blocks = [frame[NUMERIC].to_numpy(np.float32)]
    for column, levels in vocabulary.items():
        value = frame[column].astype(object)
        mapped = value.where(value.isin(levels), '__unseen__').where(value.notna(), '__missing__')
        blocks.append(np.column_stack([mapped.eq(level).to_numpy(np.float32) for level in levels]))
    matrix = np.column_stack(blocks).astype(np.float32)
    if np.isinf(matrix).any():
        raise ValueError('Action inputs cannot contain infinities.')
    return matrix


class ActionSelectionModel:
    """Load a checksum-verified feature-contract pass/carry model; no calibration."""

    def __init__(self, artifact, booster):
        self._artifact = deepcopy(artifact)
        self.booster = booster
        if (
            artifact.get('schema_version') != 1
            or artifact.get('model_type') != 'binary_action_choice'
        ):
            raise ValueError('Unsupported action-selection manifest.')
        if (
            artifact.get('feature_contract') != FEATURE_CONTRACT
            or artifact.get('calibration') != {'method': 'none'}
        ):
            raise ValueError('Unsupported action feature/calibration contract.')
        self.vocabulary = deepcopy(artifact['vocabulary'])
        self.feature_names = artifact['feature_names']
        if (
            list(self.vocabulary) != CATEGORICAL
            or self.feature_names != feature_names(self.vocabulary)
        ):
            raise ValueError('Invalid action feature vocabulary/order.')
        if (
            booster.feature_names != self.feature_names
            or booster.num_features() != len(self.feature_names)
        ):
            raise ValueError('Action booster and feature schema differ.')
        if any(('__missing__' not in v or '__unseen__' not in v for v in self.vocabulary.values())):
            raise ValueError('Missing explicit categorical buckets.')
        self.xt_grid = np.asarray(artifact['xt_grid'], float)
        if (
            self.xt_grid.shape != (10, 6)
            or not np.isfinite(self.xt_grid).all()
            or (self.xt_grid < 0).any()
        ):
            raise ValueError('Invalid fitted 10x6 xT surface.')

    @classmethod
    def from_json(cls, path):
        """Read an adjacent UBJ and require its SHA256 and feature-name parity."""
        path = Path(path)
        a = json.loads(path.read_text())
        name = a['booster_file']
        if Path(name).name != name:
            raise ValueError('Booster must be adjacent to its manifest.')
        source = path.parent / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != a['booster_sha256']:
            raise ValueError('Action booster checksum mismatch.')
        booster = xgb.Booster()
        booster.load_model(source)
        return cls(a, booster)

    @classmethod
    def pretrained(cls):
        """Load the installed verified full-data refit, not an outer-fold model."""
        return cls.from_json(Path(__file__).resolve().parents[1] / 'utils/data/action_selection_model.json')

    @property
    def model_id(self):
        """Return the identifier recorded in the fitted manifest."""
        return self._artifact['model_id']

    def to_dict(self):
        """Return a copy of the model manifest."""
        return deepcopy(self._artifact)

    def encode(self, frame):
        """Encode rows using the saved feature order and category vocabulary."""
        return encode(frame, self.vocabulary)

    def build_feature_record(
        self,
        *,
        x_bc,
        v_bc,
        r_atts,
        r_defs,
        attacking_side,
        targets,
        pass_selection_probabilities,
        pass_success_probability,
        bcos,
        ball_drive_loss,
        start_type_raw=None,
        previous_action_raw=None,
        match_clock_min=np.nan,
        score_difference_prefix=np.nan,
        completed_canonical_drives=np.nan,
        pitch_length=105.0,
        pitch_width=68.0,
        pressure_lines=None,
    ):
        """All geometry is current/hypothetical; no realized pass outcome input.

        Pass probabilities are calibrated completion, destination probabilities
        are conditional choices. Own/opponent lines and distances are sampled
        at x_bc+v_bc*1s; current ahead counts retain goalkeepers and exclude BC.
        Context and the integer prefix count must be known at this frame.
        ball_drive_loss is P(loss within 1 s | carry); the builder converts
        it once to the fitted feature drive_success_1s = 1 - ball_drive_loss.
        """
        from .action_features import aggregate
        from PRESS.utils.state import compute_dynamic_pressure_lines

        if attacking_side not in ['left_to_right', 'right_to_left']:
            raise ValueError('Unknown attack direction.')
        direction = 1 if attacking_side == 'left_to_right' else -1
        bc = np.asarray(x_bc)
        velocity = np.asarray(v_bc)
        own = np.asarray(r_atts)
        opponents = np.asarray(r_defs)
        if (
            bc.shape != (2,)
            or velocity.shape != (2,)
            or own.shape != (10, 2)
            or (opponents.shape != (11, 2))
        ):
            raise ValueError(
                'Expected BC(2), ten teammates(10,2), eleven opponents(11,2), '
                'including padding.'
            )
        if not all((np.isfinite(a).all() for a in [bc, velocity, own, opponents])):
            raise ValueError('Nonfinite tracked state.')
        if not np.isfinite(bcos) or bcos < 0:
            raise ValueError('BCOS must be finite and nonnegative.')
        if (
            np.isfinite(completed_canonical_drives)
            and (completed_canonical_drives < 0
            or completed_canonical_drives != int(completed_canonical_drives))
        ):
            raise ValueError('Completed carries must be a nonnegative integer or missing.')
        if pressure_lines is None:
            other = 'right_to_left' if direction == 1 else 'left_to_right'
            pressure_lines = (
                compute_dynamic_pressure_lines(opponents, attacking_side),
                compute_dynamic_pressure_lines(own, other),
            )
        opp_lines, own_lines = pressure_lines
        values = aggregate(
            bc=bc,
            velocity=velocity,
            players=np.vstack([own, bc, opponents]),
            targets=targets,
            direction=direction,
            own_lines=own_lines,
            opp_lines=opp_lines,
            choice=pass_selection_probabilities,
            completion=pass_success_probability,
            drive_loss=ball_drive_loss,
            bcos=bcos,
            xt_grid=self.xt_grid,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            match_clock_min=match_clock_min,
            score_difference_prefix=score_difference_prefix,
            completed_canonical_drives=completed_canonical_drives,
        )
        return dict(
            start_type_raw=start_type_raw,
            previous_action_raw=previous_action_raw,
            **values,
        )

    def build_features(self, **state):
        """Build and encode one current-state feature record."""
        return self.encode(pd.DataFrame([self.build_feature_record(**state)]))

    def predict_features(self, matrix):
        """Return native P(pass); carry is its complement, without temperature."""
        matrix = np.asarray(matrix, np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names) or np.isinf(matrix).any():
            raise ValueError('Invalid action feature matrix.')
        return self.booster.predict(xgb.DMatrix(matrix, feature_names=self.feature_names))
