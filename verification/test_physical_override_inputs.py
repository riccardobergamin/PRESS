"""Physical overrides are per-call inputs on the same PRESS object."""
import numpy as np
from PRESS import PRESS
from _fixtures import load_test_match


def test_default_custom_and_restored_production_results_on_one_object():
    tracking,*_=load_test_match()
    press=PRESS(tracking)

    press.compute_ball_drive_risk(5)
    default_drive=press.get_ball_drive_risk_result(5)
    default_drive_settings=press.get_result_configuration(5,'BALL_DRIVE_RISK')['BALL_DRIVE_RISK']
    press.compute_ball_drive_risk(5,physics_kwargs={'v_max':6.5})
    custom_drive=press.get_ball_drive_risk_result(5)
    assert press.get_result_configuration(5,'BALL_DRIVE_RISK')['BALL_DRIVE_RISK']['physics_kwargs']['v_max']==6.5
    assert default_drive_settings['physics_kwargs']['v_max']==7.
    press.compute_ball_drive_risk(5)
    assert press.get_ball_drive_risk_result(5)['Ball drive risk']==default_drive['Ball drive risk']

    press.compute_ppcf(5,r_max=8,step=4)
    default_pass=press.get_ppcf_result(5)
    default_pass_settings=press.get_result_configuration(5,'PPCF')['PPCF']
    press.compute_ppcf(5,r_max=8,step=4,physics_kwargs={'v_max':6.})
    custom_pass=press.get_ppcf_result(5)
    assert press.get_result_configuration(5,'PPCF')['PPCF']['physics']['v_max']==6.
    assert default_pass_settings['physics']['v_max']==5.5
    assert custom_pass['calibration_matches_physics'] is False
    press.compute_ppcf(5,r_max=8,step=4)
    np.testing.assert_array_equal(press.get_ppcf_result(5)['Pass probability'],default_pass['Pass probability'])

    press.compute_bcos(5,r_max=4,step=4)
    default_bcos=press.get_bcos_result(5)
    default_bcos_settings=press.get_result_configuration(5,'BCOS')['BCOS']
    press.compute_bcos(5,r_max=4,step=4,physics_args={'v_bc_max':6.,'s':.35})
    custom_bcos=press.get_bcos_result(5)
    assert press.get_result_configuration(5,'BCOS')['BCOS']['v_bc_max']==6.
    assert press.get_result_configuration(5,'BCOS')['BCOS']['s']==.35
    assert default_bcos_settings['v_bc_max']==6.6
    press.compute_bcos(5,r_max=4,step=4)
    np.testing.assert_array_equal(press.get_bcos_result(5)['Ball carrier open space'],
                                  default_bcos['Ball carrier open space'])
