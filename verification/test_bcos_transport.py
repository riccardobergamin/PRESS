import numpy as np
from PRESS import PRESS
from PRESS.model import BCOSModel,BCOSConfig,ReservoirBCOSModel
from PRESS.spatial_features import BCOS
from _fixtures import load_test_match


def test_field_bounds_and_conservation():
    model=BCOSModel(BCOSConfig(arrival_order=24,path_order=32))
    result=model.predict_field([0,0],[2,0],[[4,2],[1000,1000]],[[0,0],[0,0]],r_max=4,step=2)
    p=result['Ball carrier open space']
    assert (p>=0).all() and (p<=1).all()
    np.testing.assert_allclose(p+result['Defensive control'],1,atol=1e-14)
    assert result['max_mass_error']<1e-13
    assert result['BCOS area']==p.sum()*4
    assert (result['Defender interception probabilities'][:,1]==0).all()


def test_no_defenders_and_zero_distance():
    result=BCOSModel().predict([0,0],[0,0],np.empty((0,2)),np.empty((0,2)),[[0,0],[10,0]])
    np.testing.assert_allclose(result['Ball carrier open space'],1,atol=1e-14)


def test_action_feature_replaces_standalone_transport_result(monkeypatch):
    tracking,*_=load_test_match();press=PRESS(tracking)
    result=press.compute_bcos(5,r_max=4,step=4,physics_args={'arrival_order':24,'path_order':32})
    assert press.get_result_configuration(5,'BCOS')['BCOS']['formulation']=='transport'
    original=press.compute_bcos
    def small_field(frame_id, **kwargs):
        return original(frame_id,r_max=4,step=4,**kwargs)
    monkeypatch.setattr(press,'compute_bcos',small_field)
    value=press._classifier_bcos(5)
    old=BCOS(tracking).compute_bcos(5,r_max=4,step=4,formulation='effective_hazard')
    assert value==old['BCOS area']
    assert press.get_result_configuration(5,'BCOS')['BCOS']['formulation']=='effective_hazard'
    assert press.get_bcos_result(5)['BCOS area']==value


def test_standalone_action_selection_replaces_transport_result(monkeypatch):
    tracking,*_=load_test_match();press=PRESS(tracking)
    press.compute_ppcf(5,r_max=12,step=6)
    press.compute_ball_drive_risk(5)
    press.compute_pass_selection(5)
    press.compute_bcos(5,r_max=4,step=4)
    original=press.compute_bcos
    def small_field(frame_id, **kwargs):
        return original(frame_id,r_max=4,step=4,**kwargs)
    monkeypatch.setattr(press,'compute_bcos',small_field)
    press.compute_action_selection_probability(5)
    assert press.get_result_configuration(5,'BCOS')['BCOS']['formulation']=='effective_hazard'
    assert press.get_action_selection_result(5)['Features']['bcos']==press.get_bcos_result(5)['BCOS area']


def test_ray_movement_matches_shared_tti_away_from_origin_cutoff():
    from PRESS.physics.transport import RayMotion
    from PRESS.physics.times import time_to_intercept, rotation_time
    from PRESS.physics.numerics import float64_context
    for speed in [0,2,6.6,9]:
        distance=np.array([.2,1,10,40])
        with float64_context():
            expected=np.asarray(time_to_intercept(np.array([[0.,0.]]),np.array([[speed,0.]]),
                np.column_stack([distance,np.zeros_like(distance)]),v_max=6.6,a_max=7.,a_turn=7.))
            turn=np.asarray(rotation_time(np.array([[0.,0.]]),np.array([[speed,0.]]),
                np.column_stack([distance,np.zeros_like(distance)]),7.))
        # The shared angle evaluator uses tiny stabilizers at exact alignment.
        np.testing.assert_allclose(RayMotion(speed,6.6,7.).travel_time(distance),(expected-turn).ravel(),atol=1e-12)


def test_protected_reservoir_remains_selectable_and_bounds_accessibility():
    tracking,*_=load_test_match()
    engine=BCOS(tracking)
    old=engine.compute_bcos(5,r_max=4,step=2,formulation='transport_reservoir')
    new=engine.compute_bcos(5,r_max=4,step=2)
    assert old['params']['formulation']=='transport_reservoir'
    assert old['params']['waiting_interception'] is False
    assert new['params']['waiting_interception'] is True
    np.testing.assert_array_equal(old['grid'],new['grid'])
    assert np.all(new['Ball carrier open space']<=old['Ball carrier open space']+1e-14)
    assert new['BCOS area']<old['BCOS area']


def test_waiting_and_interior_probabilities_form_the_total_survival():
    arguments=([0,0],[2,0],[[1,1]],[[0,0]],[[5,0],[5,3]])
    old=ReservoirBCOSModel().predict(*arguments,time=.5)
    new=BCOSModel().predict(*arguments,time=.5)
    assert np.all(new['Reservoir probability']<old['Reservoir probability'])
    np.testing.assert_allclose(new['Unresolved probability'],
        new['Reservoir probability']+new['Interior probability'],atol=1e-14)
    np.testing.assert_allclose(new['Ball carrier open space']+new['Defensive control']
        +new['Unresolved probability'],1,atol=1e-14)


def test_waiting_interception_is_evaluated_at_the_physical_origin():
    from scipy.special import expit
    from PRESS.physics.transport import RayMotion
    # A defender already at the origin has TTI zero there. Its waiting
    # hazard is kappa*expit(kappa*t), with an analytic cumulative hazard.
    # This checks the actual spatial callback, not just the generic solver.
    model=BCOSModel()
    T=.4
    result=model.predict([0,0],[0,0],[[0,0]],[[0,0]],[[5,0]],time=T)
    kappa=np.pi/(np.sqrt(3)*model.config.s)
    duration=RayMotion(0,model.config.v_bc_max,model.config.a_bc_max).travel_time(5)
    Z=expit(kappa*duration)
    reference_waiting=expit(-kappa*T)/Z
    waiting_survival=np.exp(np.log(2)-np.logaddexp(0,kappa*T))
    np.testing.assert_allclose(result['Reservoir probability'],
        reference_waiting*waiting_survival,atol=2e-14)
