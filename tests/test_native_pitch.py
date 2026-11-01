"""Native geometry is shared by inference, xT lookup, plots and adapters."""
import numpy as np
import pytest
from PRESS import PRESS
from PRESS.datasets import load_demo_match
from PRESS.utils.state import get_xt_values
from PRESS.utils.frame import resolve_pitch_size


def test_xt_uses_relative_location_on_native_pitch():
    xt = np.arange(100).reshape(10, 10)
    native = get_xt_values(np.array([[26., 17.]]), 'left_to_right', xt,
                           pitch_length=104, pitch_width=68)
    reference = get_xt_values(np.array([[26.25, 17.]]), 'left_to_right', xt)
    np.testing.assert_array_equal(native, reference)
    with pytest.raises(ValueError, match='finite and positive'):
        get_xt_values(np.zeros((1, 2)), 'left_to_right', xt, pitch_length=0)


def test_state_and_plot_dimensions_come_from_tracking():
    tracking, *_ = load_demo_match()
    tracking['pitch_length'] = 104.
    tracking['pitch_width'] = 70.
    state = PRESS(tracking).get_state(5)
    assert (state['pitch_length'], state['pitch_width']) == (104., 70.)
    assert resolve_pitch_size(tracking.iloc[[5]]) == (104., 70.)
    assert resolve_pitch_size(tracking.iloc[[5]], 106, 68) == (106., 68.)


def test_adapter_reads_new_parquet_actions(tmp_path):
    tracking, _, actions, _ = load_demo_match()
    import pandas as pd
    match = tmp_path / '123'
    match.mkdir()
    tracking.to_parquet(match / '123_tracking.parquet', index=False)
    pd.DataFrame([dict(possession_id=1, player_id='home_7', frame_start=0, frame_end=9,
                      end_type='possession_loss', pass_outcome=None, target_pass=None,
                      frame_receipt=None, loss=True)]).to_parquet(match / '123_possession.parquet', index=False)
    actions['match_id'] = '123'
    actions.to_parquet(tmp_path / 'actions.parquet', index=False)
    model = PRESS.from_skillcorner(match, actions_path=tmp_path / 'actions.parquet')
    assert len(model.actions_data) == len(actions)


def test_drive_boundary_distance_uses_native_dimensions():
    from PRESS.model.solver_intensity import integrate_loss_prob
    args = (np.array([[50.,0.]]), np.zeros((1,2)), np.array([51.,0.]), np.zeros(2), 1.)
    kwargs = dict(lambda_0=.36, beta_v=.09, beta_B=.5, s=.43)
    reference = integrate_loss_prob(*args, **kwargs)
    explicit = integrate_loss_prob(*args, **kwargs, pitch_length=105, pitch_width=68)
    np.testing.assert_allclose(reference, explicit)
    narrow = integrate_loss_prob(*args, **kwargs, pitch_length=104, pitch_width=68)
    wide = integrate_loss_prob(*args, **kwargs, pitch_length=106, pitch_width=68)
    assert float(np.asarray(narrow).sum()) < float(np.asarray(wide).sum())
