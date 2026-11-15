"""Versioned terminal individual-possession risk after a successful carry."""

from copy import deepcopy
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from .action_selection import NUMERIC as ACTION_NUMERIC, CATEGORICAL

FEATURE_CONTRACT = 'residual_action31_v1'
NUMERIC = [
    'time_since_start' if c == 'completed_canonical_drives' else c
    for c in ACTION_NUMERIC
] + ['action_carry_probability']


def feature_names(vocabulary):
    """Return numeric and one-hot feature names in the fitted order."""
    return NUMERIC + [c + '=' + str(level) for c, levels in vocabulary.items() for level in levels]


def encode(frame, vocabulary):
    """Encode numeric features and the fitted missing/unseen category buckets."""
    blocks = [frame[NUMERIC].to_numpy(np.float32)]
    for c, levels in vocabulary.items():
        value = frame[c].astype(object)
        mapped = value.where(value.isin(levels), '__unseen__').where(value.notna(), '__missing__')
        blocks.append(np.column_stack([mapped.eq(level).to_numpy(np.float32) for level in levels]))
    matrix = np.column_stack(blocks).astype(np.float32)
    if np.isinf(matrix).any():
        raise ValueError('Residual inputs cannot contain infinities.')
    return matrix


class ResidualRiskModel:
    """Native P(final individual loss | current state, successful 1 s carry).

    Input covariates describe the START of the hypothetical successful carry.
    This is a terminal conditional outcome, not loss during the next second.
    No remaining-time or observed future state is an input. The supported
    feature contract includes native action-choice P(carry), elapsed individual
    possession time, and the other approved action-selection summaries.
    """

    def __init__(self, artifact, booster):
        self._artifact = deepcopy(artifact)
        self.booster = booster
        if (
            artifact.get('schema_version') != 1
            or artifact.get('model_type') != 'conditional_individual_loss'
            or artifact.get('feature_contract') != FEATURE_CONTRACT
            or (artifact.get('calibration') != {'method': 'none'})
        ):
            raise ValueError('Unsupported residual-risk contract.')
        self.vocabulary = deepcopy(artifact['vocabulary'])
        self.feature_names = artifact['feature_names']
        if (
            list(self.vocabulary) != CATEGORICAL
            or self.feature_names != feature_names(self.vocabulary)
        ):
            raise ValueError('Invalid residual feature vocabulary/order.')
        if any(('__missing__' not in v or '__unseen__' not in v for v in self.vocabulary.values())):
            raise ValueError('Residual categories require missing and unseen buckets.')
        if (
            booster.feature_names != self.feature_names
            or booster.num_features() != len(self.feature_names)
        ):
            raise ValueError('Residual booster/schema mismatch.')

    @classmethod
    def from_json(cls, path):
        """Read the manifest and adjacent UBJ with strict SHA256 validation."""
        path = Path(path)
        artifact = json.loads(path.read_text())
        name = artifact['booster_file']
        if Path(name).name != name:
            raise ValueError('Residual booster must be adjacent to manifest.')
        source = path.parent / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != artifact['booster_sha256']:
            raise ValueError('Residual booster checksum mismatch.')
        booster = xgb.Booster()
        booster.load_model(source)
        return cls(artifact, booster)

    @classmethod
    def pretrained(cls):
        """Load the verified deployment refit; raises FileNotFoundError until installed."""
        return cls.from_json(Path(__file__).resolve().parents[1] / 'utils/data/residual_risk_model.json')

    @property
    def model_id(self):
        """Return the identifier recorded in the fitted manifest."""
        return self._artifact['model_id']

    def to_dict(self):
        """Return a copy of the model manifest."""
        return deepcopy(self._artifact)

    def encode(self, frame):
        """Encode an explicit 31-feature data frame in the saved column order."""
        return encode(frame, self.vocabulary)

    def features_from_action_record(self, record, *, time_since_start, action_model,
                                    action_carry_probability=None):
        """Reuse a current-state action record, replacing whole-carry count by age.

        P(carry) comes from the supplied full-data action model. Its booster
        hash must match this residual model's deployment dependency.
        Pass its already computed native probability for the same record via
        action_carry_probability to reuse that prediction. Otherwise predict it.
        """
        if not np.isfinite(time_since_start) or time_since_start < 0:
            raise ValueError('Elapsed possession time must be finite and nonnegative.')
        if action_model.to_dict()['booster_sha256'] != self._artifact['upstream']['action_booster_sha256']:
            raise ValueError('Residual risk requires its matched action-selection model.')
        if action_carry_probability is None:
            action = pd.DataFrame([record])
            carry = 1 - float(action_model.predict_features(action_model.encode(action))[0])
        else:
            carry = float(action_carry_probability)
            if not np.isfinite(carry) or not 0 <= carry <= 1:
                raise ValueError('action_carry_probability must be in [0, 1].')
        residual = dict(
            record,
            time_since_start=float(time_since_start),
            action_carry_probability=carry,
        )
        return self.encode(pd.DataFrame([residual]))

    def build_features(self, *, action_model, time_since_start, **state):
        """Build current geometry/pass summaries through the matched action model."""
        record = action_model.build_feature_record(**state)
        return self.features_from_action_record(
            record,
            time_since_start=time_since_start,
            action_model=action_model,
        )

    def predict_features(self, matrix):
        """Return native terminal loss probability, positive class loss=1."""
        matrix = np.asarray(matrix, np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names) or np.isinf(matrix).any():
            raise ValueError('Invalid residual feature matrix.')
        return self.booster.predict(xgb.DMatrix(matrix, feature_names=self.feature_names))
