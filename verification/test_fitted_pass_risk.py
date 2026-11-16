"""Installed pass-model artifact and public default must agree with the export."""
import json
from importlib.resources import files
import numpy as np
from PRESS import PRESS
from _fixtures import load_test_match
from PRESS.model import PassRiskModel


def test_installed_pass_model_is_the_verified_full_refit():
    model=PassRiskModel.pretrained()
    artifact=json.loads(files('PRESS.utils').joinpath('data/pass_risk_model.json').read_text())
    assert model.physics['v_max']==5.5 and model.physics['a_max']==7
    assert model.calibration['method']=='isotonic'
    np.testing.assert_allclose(model.lambda_control,artifact['fit']['lambda_control'])
    assert model.model_id=='skillcorner_outfield_pass_risk_uniform_tti_v4'
    assert model.physics['preserve_initial_overspeed'] is True
    assert model.physics['a_turn']==7.
    np.testing.assert_allclose(model.s,artifact['fit']['s'])
    knots=np.asarray(model.calibration['x'])
    np.testing.assert_allclose(model.calibrate(knots),model.calibration['y'],atol=1e-15)


def test_public_default_uses_saved_isotonic_without_physics_overrides():
    tracking,*_=load_test_match()
    match=PRESS(tracking)
    match.compute_ppcf(5,r_max=12,step=12)
    result=match.get_ppcf_result(5);fitted=PassRiskModel.pretrained()
    assert match.get_result_model('PPCF',5).model_id==fitted.model_id
    assert match.get_result_model('PPCF',5).calibration['method']=='isotonic'
    assert result['calibration_matches_physics']
    np.testing.assert_allclose(result['Pass probability'],fitted.calibrate(result['Attacking pitch control field']))
    assert np.all(np.asarray(result['Pitch control field']).sum(axis=0)<=1+1e-6)
