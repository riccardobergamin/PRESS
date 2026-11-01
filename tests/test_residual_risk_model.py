"""Protect terminal loss polarity, past-time contract and upstream alignment."""
import hashlib
import json
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from PRESS.model.residual_risk import ResidualRiskModel, NUMERIC, FEATURE_CONTRACT, encode, feature_names


@pytest.fixture
def residual_export(tmp_path):
    vocabulary={c:['pass_reception','__missing__','__unseen__'] for c in ['start_type_raw','previous_action_raw']}
    rows=pd.DataFrame([{**{c:float(i) for c in NUMERIC},'start_type_raw':'pass_reception',
                        'previous_action_raw':None} for i in range(4)])
    matrix=encode(rows,vocabulary);names=feature_names(vocabulary)
    model=xgb.train({'objective':'binary:logistic','nthread':1,'max_depth':1,'min_child_weight':0},
        xgb.DMatrix(matrix,label=[0,0,1,1],feature_names=names),num_boost_round=3)
    model.save_model(tmp_path/'weights.ubj')
    a=dict(schema_version=1,model_type='conditional_individual_loss',feature_contract=FEATURE_CONTRACT,
        model_id='test_residual',calibration={'method':'none'},vocabulary=vocabulary,
        feature_names=names,booster_file='weights.ubj',
        booster_sha256=hashlib.sha256((tmp_path/'weights.ubj').read_bytes()).hexdigest(),
        upstream={'action_booster_sha256':'matched'})
    path=tmp_path/'manifest.json';path.write_text(json.dumps(a))
    return path,rows,matrix,model


def test_reader_preserves_native_loss_probability_and_checks_weight_hash(residual_export):
    path,rows,matrix,reference=residual_export
    model=ResidualRiskModel.from_json(path)
    expected=reference.predict(xgb.DMatrix(matrix,feature_names=model.feature_names))
    np.testing.assert_array_equal(model.predict_features(model.encode(rows)),expected)
    assert expected[-1]>expected[0]  # loss-positive target is not inverted
    with (path.parent/'weights.ubj').open('ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):ResidualRiskModel.from_json(path)


def test_elapsed_replaces_count_and_carry_is_native_complement(residual_export):
    path,rows,_,_=residual_export;model=ResidualRiskModel.from_json(path)
    class Action:
        def to_dict(self):return {'booster_sha256':'matched'}
        def encode(self,frame):return np.zeros((len(frame),1))
        def predict_features(self,matrix):return np.full(len(matrix),.3)
    record=rows.iloc[0].to_dict();record['completed_canonical_drives']=99
    x=model.features_from_action_record(record,time_since_start=2.4,action_model=Action())
    assert 'completed_canonical_drives' not in model.feature_names
    assert x[0,model.feature_names.index('time_since_start')]==np.float32(2.4)
    assert x[0,model.feature_names.index('action_carry_probability')]==np.float32(.7)
    with pytest.raises(ValueError,match='Elapsed'):model.features_from_action_record(record,time_since_start=-1,action_model=Action())
    class Wrong(Action):
        def to_dict(self):return {'booster_sha256':'wrong'}
    with pytest.raises(ValueError,match='matched'):model.features_from_action_record(record,time_since_start=0,action_model=Wrong())
