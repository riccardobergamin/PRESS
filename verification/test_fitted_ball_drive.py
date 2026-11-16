"""Export-reader checks using an audited CV fold, never a claimed final refit."""
from pathlib import Path
import numpy as np
import pytest
from PRESS import PRESS
from _fixtures import load_test_match
from PRESS.model import BallDriveModel

from importlib.resources import files
FIXTURE=files('PRESS.utils').joinpath('data/ball_drive_model.json')


def state():
    return dict(r_defs=np.array([[3.,0.],[0.,4.],[1000.,1000.]]),
        v_defs=np.array([[-2.,0.],[0.,-1.],[0.,0.]]),x_bc=np.array([0.,0.]),v_bc=np.array([2.,0.]))


def test_bundled_model_is_verified_full_refit():
    model=BallDriveModel.pretrained()
    artifact=model.to_dict()
    assert artifact['training']['role']=='full_data_refit'
    assert artifact['training']['observations']==9114
    assert artifact['training']['losses']==456
    assert artifact['training']['cohort_policy']=='outfield_no_shot_obe_verified_loss_parents_v1'
    assert len(artifact['training']['match_ids'])==10
    assert artifact['benchmark']['role']=='pooled_out_of_fold'
    assert model.physics['integration_method']=='exponential'
    assert model.physics['v_max']==model.physics['a_linear']==model.physics['a_turn']==7
    assert 0 <= model.predict(**state())['loss_probability'] <= 1


def test_export_roundtrip_and_defender_mass():
    model=BallDriveModel.from_json(FIXTURE);copy=BallDriveModel(model.to_dict())
    a=model.predict(**state());b=copy.predict(**state())
    np.testing.assert_allclose(a['cause_probabilities'],b['cause_probabilities'],atol=1e-14)
    np.testing.assert_allclose(a['survival_probability']+a['cause_probabilities'].sum(),1,atol=1e-14)
    assert a['cause_probabilities'][-1]==0
    assert copy.to_dict()['training']['role']==model.to_dict()['training']['role']


def test_partial_horizons_and_initial_survival():
    model=BallDriveModel.from_json(FIXTURE)
    results=[model.predict(**state(),horizon=h) for h in [0,.017,.17,.4,1.]]
    assert results[0]['loss_probability']==0
    assert np.all(np.diff([r['loss_probability'] for r in results])>=0)
    for r in results:
        assert r['survival_curve'][0]==1
        np.testing.assert_allclose(r['survival_curve'][-1],r['survival_probability'])
        assert len(r['times'])==len(r['survival_curve'])


def test_batch_and_single_predictions_match():
    model=BallDriveModel.from_json(FIXTURE);raw=state()
    batch=model.predict_batch(**{k:np.stack([v,v]) for k,v in raw.items()})
    one=model.predict(**raw)
    np.testing.assert_allclose(batch['loss_probability'],one['loss_probability'],atol=1e-13)


def test_invalid_schema_or_physics_rejected():
    model=BallDriveModel.from_json(FIXTURE);a=model.to_dict();a['theta']=[0]
    with pytest.raises(ValueError):BallDriveModel(a)
    with pytest.raises(ValueError):model.predict(**state(),horizon=-1)
    with pytest.raises(ValueError):model.predict(**state(),physics_kwargs={'a_turn':0})
    with pytest.raises(ValueError):model.predict(**state(),physics_kwargs={'lambda_0':1})


def test_fitted_model_frame_getter_and_pitch_contract():
    tracking,possessions,actions,_=load_test_match()
    fitted=BallDriveModel.from_json(FIXTURE)
    press=PRESS(tracking,possessions_data=possessions,actions_data=actions,ball_drive_model=fitted)
    frame=int(press.get_frames(valid_only=True).frame.iloc[0])
    press.compute_ball_drive_risk(frame,t_press=.4)
    result=press.get_ball_drive_risk_result(frame)
    assert press.get_result_model('BALL_DRIVE_RISK',frame).model_id==fitted.model_id
    assert press.get_result_configuration(frame,'BALL_DRIVE_RISK')['BALL_DRIVE_RISK']['physics_kwargs']['pitch_length']==105
    np.testing.assert_allclose(result['Ball drive risk'],np.sum(result['Individual tackle probabilities']),atol=1e-12)
    # A path can also be supplied per call without changing the constructor.
    press.compute_ball_drive_risk(frame,model=FIXTURE)
    assert press.get_result_configuration(frame,'BALL_DRIVE_RISK')['BALL_DRIVE_RISK']['t_press']==1.
