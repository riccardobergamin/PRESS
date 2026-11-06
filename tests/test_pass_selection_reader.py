"""Checks for the final manifest contract, choice normalization and nesting limit."""
import hashlib
import json
import numpy as np
import pytest
from scipy.special import softmax
import xgboost as xgb
from PRESS.model import PassSelectionModel
from PRESS.model.pass_selection import build_choice_features, choice_feature_names


@pytest.fixture
def artifact(tmp_path):
    vocabulary={c:['pass_reception','__missing__','__unseen__']
                for c in ['start_type_raw','previous_action_raw']}
    names=choice_feature_names(vocabulary)
    x=np.zeros((20,len(names)),np.float32);x[:,0]=np.linspace(0,10,20)
    booster=xgb.train(dict(objective='reg:squarederror',max_depth=2,nthread=1,seed=7),
                      xgb.DMatrix(x,label=np.sin(x[:,0]),feature_names=names),num_boost_round=4)
    booster.save_model(tmp_path/'model.ubj')
    a=dict(schema_version=1,model_type='pass_destination_choice',model_id='TEST_ONLY',family='flat',rho=1.,feature_contract='ahead_of_carrier_and_target_v1',
           calibration=dict(inverse_temperature=.8),vocabulary=vocabulary,feature_names=names,
           xt_grid=np.ones((10,6)).tolist(),booster_file='model.ubj',
           booster_sha256=hashlib.sha256((tmp_path/'model.ubj').read_bytes()).hexdigest())
    path=tmp_path/'model.json';path.write_text(json.dumps(a))
    return path,a,x


def test_manifest_and_temperature_match_explicit_softmax(artifact):
    path,a,x=artifact;model=PassSelectionModel.from_json(path)
    scores=model.booster.predict(xgb.DMatrix(x,feature_names=a['feature_names']),output_margin=True)
    assert np.allclose(model.predict_features(x),softmax(.8*scores.astype(float)),atol=1e-14)
    assert np.allclose(model.predict_features(x,calibrated=False),softmax(scores.astype(float)),atol=1e-14)
    a['family']='cnl';cnl=PassSelectionModel(a,model.booster)
    alpha=np.tile(np.log([.2,.8,1.]),(len(x),1));alpha[:,-1]=-np.inf
    assert np.allclose(cnl.predict_features(x,log_membership=alpha),model.predict_features(x),atol=1e-14)
    with pytest.raises(ValueError,match='memberships'):cnl.predict_features(x)


def test_feature_semantics_and_missing_categories(artifact):
    _,a,_=artifact
    x=build_choice_features(x_bc=np.array([0.,0.]),v_bc=np.array([2.,0.]),
        r_atts=np.array([[2.,0.],[1000.,1000.]]),r_defs=np.array([[3.,0.]]),
        targets=np.array([[5.,0.],[-5.,0.]]),attacking_side='left_to_right',
        target_pitch_control=np.array([.1,.9]),ball_drive_prob=.2,xt_grid=np.ones((10,6)),
        vocabulary=a['vocabulary'],start_type_raw=None,previous_action_raw='never_seen',
        pressure_lines=([0.,10.,20.],[-20.,-10.,0.]))
    columns={name:x[:,i] for i,name in enumerate(a['feature_names'])}
    assert np.all(columns['ball_drive_prob']==np.float32(.2))
    for roster in ['opponents', 'teammates']:
        np.testing.assert_array_equal(columns[roster+'_ahead_bc'], [1,1])
        np.testing.assert_array_equal(columns[roster+'_ahead_target'], [0,1])
    assert np.array_equal(columns['opp_pressure_line_index'],[1,1])
    assert np.all(columns['start_type_raw=__missing__']==1)
    assert np.all(columns['previous_action_raw=__unseen__']==1)


def test_corrupt_artifact_and_column_order_are_rejected(artifact):
    path,a,x=artifact;a['booster_sha256']='0'*64;path.write_text(json.dumps(a))
    with pytest.raises(ValueError,match='checksum'):PassSelectionModel.from_json(path)


@pytest.mark.parametrize('contract', [None, 'forward_bypass_v1'])
def test_missing_or_historical_feature_contract_is_rejected(artifact, contract):
    path, manifest, _ = artifact
    if contract is None:
        manifest.pop('feature_contract')
    else:
        manifest['feature_contract'] = contract
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='feature contract'):
        PassSelectionModel.from_json(path)


@pytest.mark.parametrize('sign', [1, -1])
def test_ahead_counts_use_carrier_attack_axis_for_both_rosters(sign):
    contract='ahead_of_carrier_and_target_v1'
    vocabulary={c:['__missing__','__unseen__'] for c in ['start_type_raw','previous_action_raw']}
    targets=sign*np.array([[-5.,0.],[0.,0.],[5.,0.]])
    matrix=build_choice_features(x_bc=np.array([0.,0.]),v_bc=sign*np.array([2.,0.]),
        r_atts=sign*np.array([[-2.,1.],[0.,1.],[2.,1.],[1000.,1000.]]),
        r_defs=sign*np.array([[-3.,2.],[0.,2.],[3.,2.]]),targets=targets,
        attacking_side='left_to_right' if sign==1 else 'right_to_left',
        target_pitch_control=np.full(3,.5),ball_drive_prob=.2,xt_grid=np.ones((10,6)),
        vocabulary=vocabulary,pressure_lines=([-10.,0.,10.],[-10.,0.,10.]),
        feature_contract=contract)
    names=choice_feature_names(vocabulary,feature_contract=contract)
    columns={name:matrix[:,i] for i,name in enumerate(names)}
    for roster in ['opponents','teammates']:
        np.testing.assert_array_equal(columns[roster+'_ahead_bc'],[1,1,1])
        np.testing.assert_array_equal(columns[roster+'_ahead_target'],[3,1,0])
    assert not any('bypassed' in name for name in names)


def test_installed_default_is_verified_flat_full_refit():
    from pathlib import Path
    from PRESS.model import PassRiskModel, BallDriveModel
    model=PassSelectionModel.pretrained();a=model.to_dict()
    provenance=json.loads((Path(__file__).parents[1]/'docs/pass-selection-final-provenance.json').read_text())
    assert model.family=='flat' and model.model_id=='skillcorner_outfield_pass_selection_flat_v3'
    assert model.booster.num_boosted_rounds()==a['training']['rounds']==1444
    assert model.feature_contract=='ahead_of_carrier_and_target_v1'
    assert len(model.feature_names)==50
    assert all(name in model.feature_names for name in ('opponents_ahead_bc', 'teammates_ahead_bc', 'opponents_ahead_target', 'teammates_ahead_target'))
    assert not any('bypassed' in name for name in model.feature_names)
    assert a['training']['n_choices']==6810 and len(a['training']['match_ids'])==10
    assert model.rho==1 and a['calibration']['temperature']==pytest.approx(1.013295437946196)
    assert a['physics']['pass_risk']==PassRiskModel.pretrained().to_dict()
    assert a['physics']['ball_drive']==BallDriveModel.pretrained().to_dict()
    assert provenance['installed_family']=='flat' and provenance['cnl_installed'] is False
    assert not (Path(__file__).parents[1]/'PRESS/utils/data/pass_selection_cnl_model.json').exists()


def test_frame_api_uses_final_physics_and_granular_context():
    from PRESS import PRESS
    from _fixtures import load_test_match
    from PRESS.model import BallDriveModel
    tracking,possessions,actions,xt=load_test_match()
    tracking['pitch_length']=104.
    press=PRESS(tracking,actions_data=actions,possessions_data=possessions)
    frame=5
    press.compute_ppcf(frame,r_max=12,step=6)
    press.compute_ball_drive_risk(frame)
    p=press.compute_pass_selection(frame,xt,match_clock_min=.5,score_difference_prefix=0)
    state=press.get_state(frame);ppcf=press.get_ppcf_result(frame);pi=press.get_ball_drive_risk_result(frame)
    assert pi['params']['model_id']==BallDriveModel.pretrained().model_id
    assert pi['params']['physics_kwargs']['pitch_length']==104.
    context=ppcf['pass_selection_context'];row=actions.loc[(actions.frame_start<=frame)&(actions.frame_end>=frame)].iloc[0]
    assert context['start_type_raw']==row.start_type and context['previous_action_raw']==row.previous_action
    model=PassSelectionModel.pretrained()
    matrix=model.build_features(x_bc=state['x_bc'],v_bc=state['v_bc'],r_atts=state['r_atts'],r_defs=state['r_defs'],
        targets=ppcf['grid'],attacking_side=state['attacking_side'],target_pitch_control=ppcf['Attacking pitch control field'],
        ball_drive_prob=pi['Ball drive risk'],pitch_length=104.,pitch_width=68.,**context)
    np.testing.assert_array_equal(p,model.predict_features(matrix))
    assert np.isclose(p.sum(),1) and ppcf['params']['choice_grid_matches_benchmark'] is False
    press.compute_ball_drive_risk(frame,t_press=.4)
    with pytest.raises(ValueError,match='one-second'):press.compute_pass_selection(frame,xt)


def test_default_grid_matches_native_training_cells(monkeypatch):
    from PRESS import PRESS
    from _fixtures import load_test_match
    from PRESS.model import PassRiskModel
    tracking,*_=load_test_match();tracking['pitch_length']=104.
    def field(self,pos,vel,bc,targets,attackers,carrier,**kwargs):
        return dict(control=np.zeros((len(pos),len(targets))),raw=np.full(len(targets),.4),
                    calibrated=np.full(len(targets),.5),flight_time=np.zeros(len(targets)),
                    physics=self.physics,calibration_matches_physics=True)
    monkeypatch.setattr(PassRiskModel,'predict_field',field)
    press=PRESS(tracking);press.compute_ppcf(5);result=press.get_ppcf_result(5)
    assert result['grid'].shape==(104*68,2)
    np.testing.assert_array_equal(result['grid'][0],[-51.5,-33.5])
    np.testing.assert_array_equal(result['grid'][-1],[51.5,33.5])
    assert result['params']['choice_grid_matches_benchmark'] is True
