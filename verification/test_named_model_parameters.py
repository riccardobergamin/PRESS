"""Named parameter inputs preserve fitted bases, defaults and numerical models."""
from copy import deepcopy
import numpy as np
import pytest

from PRESS import PRESS
from PRESS.model import BallDriveModel, PassRiskModel, BCOSModel
from _fixtures import load_test_match


def test_ball_drive_named_round_trip_is_exact_and_does_not_mutate_production():
    base=BallDriveModel.pretrained()
    original=base.to_dict()
    named=base.parameters
    reconstructed=BallDriveModel.from_parameters(named)
    np.testing.assert_array_equal(reconstructed.theta,base.theta)
    assert base.to_dict()==original
    named['gamma_v'][0]=999
    assert base.parameters['gamma_v'][0]!=999


def test_named_ball_drive_parameters_match_the_same_explicit_fitted_vector():
    base=BallDriveModel.pretrained()
    values=dict(lambda_0=.4,beta_v=.15,beta_B=-.02,s=.35,
                gamma_v=[.01,.02,.03],gamma_B=[-.02,.01,.02],
                beta_angle=.2,gamma_angle=[-.03,.01])
    named=BallDriveModel.from_parameters(values)
    artifact=base.to_dict()
    artifact['theta']=[np.log(.4),.15,-.02,np.log(.35),
                       .01,.02,.03,-.02,.01,.02,.2,-.03,.01]
    explicit=BallDriveModel(artifact)
    arguments=(np.array([[2.,2.],[5.,0.]]),np.array([[0.,0.],[-1.,0.]]),
               np.array([0.,0.]),np.array([3.,0.]))
    a,b=named.predict(*arguments),explicit.predict(*arguments)
    np.testing.assert_array_equal(a['cause_probabilities'],b['cause_probabilities'])
    assert a['loss_probability']==b['loss_probability']


def test_own_basis_requires_its_coefficients_and_is_preserved():
    base=BallDriveModel.pretrained()
    parameters=base.parameters
    speed_basis=base.to_dict()['design']['speed_basis']
    with pytest.raises(ValueError,match='requires coefficients'):
        BallDriveModel.from_parameters({'speed_basis':speed_basis})
    model=BallDriveModel.from_parameters(dict(speed_basis=speed_basis,
        gamma_v=parameters['gamma_v']))
    assert model.to_dict()['design']['speed_basis']==speed_basis
    assert model.to_dict()['angle_design']==base.to_dict()['angle_design']


@pytest.mark.parametrize('parameters',[
    {'s':0}, {'s':np.nan}, {'lambda_0':-1}, {'beta_v':np.inf},
    {'gamma_v':[0.,1.]}, {'gamma_B':[0.,1.,np.nan]},
    {'beta_angle':[1.,2.]}, {'gamma_angle':[1.]}, {'unknown':1},
])
def test_invalid_ball_drive_named_inputs_are_rejected(parameters):
    with pytest.raises(ValueError):BallDriveModel.from_parameters(parameters)


def test_named_pass_parameters_preserve_and_identify_the_calibration_reference():
    base=PassRiskModel.pretrained()
    original=base.to_dict()
    custom=PassRiskModel.from_parameters({'lambda_control':5.,'s':.3})
    assert custom.parameters=={'lambda_control':5.,'s':.3}
    assert custom.calibration==base.calibration
    assert not custom.calibration_matches_parameters
    further=PassRiskModel.from_parameters({'s':.32},base=custom)
    assert further.to_dict()['calibration_reference']==custom.to_dict()['calibration_reference']
    assert not PassRiskModel(further.to_dict()).calibration_matches_parameters
    assert base.to_dict()==original
    assert PassRiskModel.from_parameters(base.parameters).calibration_matches_parameters


def test_explicit_calibrator_is_kept_with_its_supplied_parameter_reference():
    calibration=dict(method='isotonic',x=[0.,1.],y=[0.,1.])
    custom=PassRiskModel.from_parameters(dict(lambda_control=5.,s=.3,calibration=calibration))
    assert custom.calibration==calibration
    assert custom.calibration_matches_parameters
    np.testing.assert_array_equal(custom.calibrate([.2,.8]),[.2,.8])


@pytest.mark.parametrize('parameters',[{'s':0},{'lambda_control':-1},{'s':np.nan},{'unknown':1}])
def test_invalid_pass_named_inputs_are_rejected(parameters):
    with pytest.raises(ValueError):PassRiskModel.from_parameters(parameters)


def test_same_object_accepts_named_and_complete_artifact_dictionaries():
    tracking,*_=load_test_match()
    press=PRESS(tracking)
    press.compute_ball_drive_risk(5,model={'s':.35,'gamma_v':[0.,0.,0.]})
    parameters=press.get_result_model('BALL_DRIVE_RISK',5).parameters
    assert parameters['s']==pytest.approx(.35,abs=1e-16,rel=0)
    assert parameters['gamma_v']==[0.,0.,0.]
    press.compute_ball_drive_risk(5)
    assert press.get_result_model('BALL_DRIVE_RISK',5).parameters==BallDriveModel.pretrained().parameters
    press.compute_ppcf(5,r_max=8,step=4,model={'lambda_control':5.,'s':.3})
    p=press.get_result_model('PPCF',5).parameters
    diagnostics=press.get_ppcf_result(5)
    assert p['lambda_control']==5. and p['s']==.3
    assert diagnostics['calibration_matches_physics']
    assert not diagnostics['calibration_matches_parameters'] and not diagnostics['calibration_matches_model']
    press.compute_ppcf(5,r_max=8,step=4,model=PassRiskModel.pretrained().to_dict())
    assert press.get_ppcf_result(5)['calibration_matches_model']
    press.compute_ball_drive_risk(5,model=BallDriveModel.pretrained().to_dict())


def test_constructor_and_per_call_override_use_the_supplied_base_model():
    tracking,*_=load_test_match()
    press=PRESS(tracking,ball_drive_model={'s':.35},pass_risk_model={'lambda_control':5.})
    assert press.ball_drive_model.parameters['s']==pytest.approx(.35,abs=1e-16,rel=0)
    assert press.pass_risk_model.lambda_control==5.
    press.compute_ball_drive_risk(5,model={'lambda_0':.4})
    assert press.get_result_model('BALL_DRIVE_RISK',5).parameters['s']==pytest.approx(.35,abs=1e-16,rel=0)
    press.compute_ppcf(5,r_max=8,step=4,model={'s':.32})
    assert press.get_result_model('PPCF',5).lambda_control==5.
    assert press.ball_drive_model.parameters['s']==pytest.approx(.35,abs=1e-16,rel=0)
    assert BCOSModel.from_ball_drive({'s':.35}).config.s==pytest.approx(.35,abs=1e-16,rel=0)
