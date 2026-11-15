"""A supplied value map must reach every downstream model and result export."""
import numpy as np
import pytest
from PRESS import PRESS
from PRESS.model import PassSelectionModel, ActionSelectionModel
from _fixtures import load_test_match, write_skillcorner_match


def match(grid=None):
    tracking, _, actions, _ = load_test_match()
    return PRESS(tracking, actions_data=actions, xT_grid=grid)


def compute(press, **kwargs):
    return press.compute_risk_frame(5, bcos_value=.5, include_residual=True, keep_details=True,
                               ppcf_kwargs={'r_max':12,'step':6}, **kwargs)


def test_object_map_reaches_choice_action_and_residual(monkeypatch):
    surface=np.full((8,5),.23)
    press=match(surface)
    choice=press._choice_model()
    original=choice.build_features
    seen={}
    def capture(**state):
        values=original(**state)
        seen['features']=values.copy()
        return values
    monkeypatch.setattr(choice,'build_features',capture)
    compute(press)
    for feature in ['target_xt_team','target_xt_opp','xt_team_bc','xt_opp_bc']:
        np.testing.assert_allclose(seen['features'][:,choice.feature_names.index(feature)],.23,atol=1e-8)
    record=press.get_action_selection_result(5)['Features']
    assert record['xt_team_drive_1s']==pytest.approx(.23)
    assert record['xt_opp_drive_1s']==pytest.approx(.23)
    assert record['xt_team_pass_success']==pytest.approx(.23*record['pass_success_mean'])
    assert record['xt_opp_pass_failure']==pytest.approx(.23*(1-record['pass_success_mean']))
    np.testing.assert_array_equal(press.get_xt_grid(5),surface)
    assert np.isfinite(press.get_xloss_result(5)['xLoss'])
    # Input and getter arrays do not share mutable storage with the object.
    surface[:]=0
    retrieved=press.get_xt_grid();retrieved[:]=0
    np.testing.assert_array_equal(press.get_xt_grid(),np.full((8,5),.23))


def test_frame_override_and_separate_calls_use_one_map():
    press=match(np.full((8,5),.1))
    alternate=np.full((12,7),.31)
    press.compute_ppcf(5,r_max=12,step=6)
    press.compute_ball_drive_risk(5)
    press.compute_pass_selection(5,xT_grid=alternate)
    press.compute_action_selection_probability(5,bcos_value=.5)
    assert press.get_action_selection_result(5)['Features']['xt_team_drive_1s']==pytest.approx(.31)
    with pytest.raises(ValueError,match='differs from pass selection'):
        press.compute_action_selection_probability(5,xT_grid=np.ones((8,5))*.1,bcos_value=.5)
    compute(press,xT_grid=alternate)
    np.testing.assert_array_equal(press.get_xt_grid(5),alternate)
    np.testing.assert_array_equal(press.get_xt_grid(),np.full((8,5),.1))


def test_omitted_map_keeps_native_default_and_fitted_manifests():
    first=match()
    second=match(first.get_xt_grid())
    native_choice=PassSelectionModel.pretrained().to_dict()
    native_action=ActionSelectionModel.pretrained().to_dict()
    a,b=compute(first),compute(second)
    np.testing.assert_array_equal(a['PPCF']['Pass selection probabilities'],b['PPCF']['Pass selection probabilities'])
    assert a['ACTION_SELECTION']['Pass probability']==b['ACTION_SELECTION']['Pass probability']
    assert a['XLOSS']['xLoss']==b['XLOSS']['xLoss']
    assert first._choice_model().to_dict()==native_choice
    assert first._action_model().to_dict()==native_action


def test_custom_map_is_checked_on_result_reload(tmp_path):
    surface=np.full((8,5),.23)
    press=match(surface);compute(press)
    folder=press.save_results(tmp_path/'results')
    restored=match(surface);restored.load_results(folder)
    np.testing.assert_array_equal(restored.get_xt_grid(5),surface)
    assert restored.get_xloss_result(5)==press.get_xloss_result(5)
    with pytest.raises(ValueError,match='do not match'):
        match(np.full((8,5),.24)).load_results(folder)


def test_original_skillcorner_loader_accepts_the_map(tmp_path):
    folder=write_skillcorner_match(tmp_path/'123')
    grid=np.full((8,5),.2)
    press=PRESS.from_skillcorner(folder,xT_grid=grid)
    np.testing.assert_array_equal(press.get_xt_grid(),grid)


@pytest.mark.parametrize('grid',[np.array([1,2]),np.empty((0,2)),np.array([[np.nan]]),np.array([[-1.]])])
def test_invalid_map_fails_at_initialization(grid):
    with pytest.raises(ValueError,match='xT_grid'):
        match(grid)
