"""Reuse numerical inputs within a frame without stale data or model changes."""
import numpy as np
import pytest
from PRESS import PRESS
from PRESS.model import ActionSelectionModel, ResidualRiskModel, PassRiskModel
from PRESS.model.pass_risk import _flight_candidates
import PRESS.utils.state as state_module
import PRESS.physics.flight_ball as flight_module
from _fixtures import load_test_match


def match():
    tracking,_,actions,_=load_test_match()
    return PRESS(tracking,actions_data=actions)


def compute(press,**kwargs):
    return press.compute_risk_frame(5,include_residual=True,
        ppcf_kwargs={'r_max':12,'step':6},bcos_kwargs={'r_max':8,'step':4},**kwargs)


def test_one_state_line_pair_action_prediction_and_hazard_per_complete_frame(monkeypatch):
    press=match();calls={}
    def spy(obj,name,key):
        original=getattr(obj,name)
        def count(*args,**kwargs):
            calls[key]=calls.get(key,0)+1
            return original(*args,**kwargs)
        monkeypatch.setattr(obj,name,count)
    spy(state_module,'get_pv_players','state')
    spy(state_module,'compute_dynamic_pressure_lines','lines')
    spy(ActionSelectionModel,'predict_features','action')
    spy(press,'compute_hazard','h')
    r=compute(press)
    assert calls=={'state':1,'lines':2,'action':1,'h':1}
    assert getattr(press,'_frame_inputs',None) is None
    assert press.compute_xloss(5)==r['XLOSS']['xLoss']
    assert calls['h']==1 and calls['action']==1


def test_frame_input_reuse_ends_after_success_and_failure(monkeypatch):
    press=match();compute(press)
    old=press.get_state(5)['x_bc'].copy()
    carrier=press.get_state(5)['player']
    press.tracking_data.loc[press.tracking_data.frame.eq(5),carrier+'_x']+=3
    assert press.get_state(5)['x_bc'][0]==old[0]+3
    original=press.compute_pass_selection
    def fail(*args,**kwargs):raise RuntimeError('injected failure')
    monkeypatch.setattr(press,'compute_pass_selection',fail)
    with pytest.raises(RuntimeError,match='injected failure'):compute(press)
    assert getattr(press,'_frame_inputs',None) is None
    monkeypatch.setattr(press,'compute_pass_selection',original)
    result=compute(press)
    assert np.isfinite(result['XLOSS']['xLoss'])


def test_reuse_carry_matches_model_prediction_and_keeps_dependency_checks():
    press=match();compute(press,keep_details=True)
    record=press.get_action_selection_result(5)['Features']
    action=press.get_result_model('ACTION_SELECTION',5)
    residual=press.get_result_model('RESIDUAL_RISK',5)
    a=residual.features_from_action_record(record,time_since_start=.5,action_model=action)
    b=residual.features_from_action_record(record,time_since_start=.5,action_model=action,
        action_carry_probability=press.get_action_selection_result(5)['Ball drive probability'])
    np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError,match='action_carry_probability'):
        residual.features_from_action_record(record,time_since_start=.5,action_model=action,
            action_carry_probability=1.1)


def test_fixed_ball_trajectories_cached_and_receiver_tti_not_repeated(monkeypatch):
    _flight_candidates.cache_clear()
    calls=[];original=flight_module.generate_ball_trajectories
    def count(*args,**kwargs):
        calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(flight_module,'generate_ball_trajectories',count)
    def forbidden(*args,**kwargs):raise AssertionError('Receiver TTI was recalculated.')
    monkeypatch.setattr(flight_module,'time_to_intercept',forbidden)
    model=PassRiskModel.pretrained()
    positions=np.array([[0.,0.],[3.,1.],[5.,-1.]])
    velocity=np.zeros_like(positions);targets=np.array([[2.,1.],[4.,-1.]])
    physics={'speed_candidates':2,'elevation_candidates':2,'t_max_flight':.4}
    for vmax in [5.5,6.]:
        model.predict_field(positions,velocity,positions[0],targets,[0,1],0,
            physics_kwargs=physics|{'v_max':vmax})
    assert len(calls)==1
    model.predict_field(positions,velocity,positions[0],targets,[0,1],0,
        physics_kwargs=physics|{'Cd':.3})
    assert len(calls)==2
    _flight_candidates.cache_clear()
