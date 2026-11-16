"""Behavioral checks for the public match-to-checkpoint workflow."""
import numpy as np
import pandas as pd
import pytest

from PRESS import PRESS
from _fixtures import load_test_match
from PRESS.utils.match import link_team_possessions, standardize_action_context
from PRESS.utils.possession import possession_survival, hazard_to_band
from PRESS.model import PassRiskModel


@pytest.fixture
def pass_risk_model():
    # Explicit synthetic fixture: workflow tests do not require cluster weights
    # and these knots must never be exported as pretrained calibration.
    return PassRiskModel(dict(schema_version=1,model_type='pass_risk_trajectory_isotonic',
        model_id='synthetic-workflow-fixture',lambda_control=4.3,s=.43,
        physics=dict(v_max=5.5,a_max=7.,dt_ball=.04,dt_pc=.01,t_max_flight=10.,t_max_pc=15.,
            rho=1.22,Cd=.25,A=.038,m=.42,g=9.81,z_max=2.,distance_mode='exact',
            integration_method='exponential',speed_range=[5.,25.],speed_candidates=30,
            elevation_degrees=[0.,45.],elevation_candidates=30),
        calibration=dict(method='isotonic',x=[0.,1.],y=[0.,1.])))


@pytest.fixture
def match(pass_risk_model):
    tracking, possessions, actions, xt = load_test_match()
    return PRESS(tracking, possessions_data=possessions, actions_data=actions,
                 pass_risk_model=pass_risk_model), xt


def test_navigation_preserves_team_possession_across_carrier_change(match):
    model, _ = match
    frames = model.get_frames(1)
    assert frames.frame.tolist() == list(range(30))
    assert frames.player_possession.nunique() == 2
    assert model.get_team_possessions(team_id="away").possession_id.tolist() == [2]
    assert model.get_actions(team_possession_id=1, valid_only=True).frame_start.tolist() == [0, 10, 20]
    assert model.get_frame(10).player_possession.item() == "home_8"
    frames.loc[0, "home_7_x"] = 1234
    assert model.get_frame(0).home_7_x.item() != 1234
    with pytest.raises(KeyError):
        model.get_frame(999)
    assert model.get_team_possessions(999).empty


def test_no_team_possessions_are_invented_from_tracking():
    tracking, *_ = load_test_match()
    with pytest.raises(ValueError, match="Supply possessions_data"):
        PRESS(tracking).get_team_possessions()


@pytest.mark.parametrize('release,canonical,all_frames', [(4,[4],[0,4]), (24,[0,10,24],[0,10,20,24])])
def test_pass_release_and_extra_snapshot_sampling(release,canonical,all_frames,monkeypatch):
    tracking,teams,_,xt=load_test_match()
    tracking['player_possession']='home_7'
    teams=teams.iloc[[0]].copy()
    teams['frame_end']=release
    full,tail=divmod(release,10)
    rows=[dict(frame_start=f,frame_end=f+10,action='ball_drive',is_canonical=True,
               start_type='pass_reception',previous_action='pass_reception' if f==0 else 'ball_drive')
          for f in range(0,full*10,10)]
    rows += [dict(frame_start=release,frame_end=release,action='pass',is_canonical=True,
                  start_type='pass_reception',previous_action='ball_drive' if full else 'pass_reception')]
    if tail:
        rows += [dict(frame_start=full*10,frame_end=release,action='snapshot',is_canonical=False,
                      start_type='pass_reception',previous_action='ball_drive' if full else 'pass_reception')]
    model=PRESS(tracking,possessions_data=teams,actions_data=pd.DataFrame(rows))
    assert model.get_actions(team_possession_id=1, valid_only=True).frame_start.tolist()==canonical
    assert model.get_actions(team_possession_id=1,include_snapshots=True, valid_only=True).frame_start.tolist()==all_frames
    seen=[]
    def compute(frame,xT_grid,**kwargs):
        seen.append((frame,kwargs))
        return {'ACTION_SELECTION':{'Pass probability':.5},'HAZARD':{'Hazard':.1}}
    monkeypatch.setattr(model,'compute_risk_frame',compute)
    table=model.compute_possession(1,xt)
    assert table.frame_id.tolist()==all_frames
    assert table.loc[table.is_canonical,'frame_id'].tolist()==canonical
    assert all(kwargs=={} for _,kwargs in seen)
    assert model.compute_possession(1,xt,include_snapshots=False).frame_id.tolist()==canonical


def test_eligibility_exposes_unknown_and_missing_tracking(match):
    model, _ = match
    model.actions_data.loc[0, "start_type"] = "unknown"
    model.tracking_data.loc[model.tracking_data.frame == 10, "home_8_x"] = np.nan
    assert model.get_actions(team_possession_id=1, valid_only=True).frame_start.tolist() == [0, 20]
    all_rows = model.get_actions(team_possession_id=1, valid_only=False)
    assert all_rows.frame_start.tolist() == [0, 10, 20]
    assert standardize_action_context("free_kick_reception") == "reception_set_piece"
    assert standardize_action_context("ball_drive") is None


def player_segments():
    return pd.DataFrame([
        dict(player_id="home_7", frame_start=0, frame_end=9, end_type="pass",
             pass_outcome="successful", target_pass="home_8", frame_receipt=10,
             loss=False, game_interruption_after=None),
        dict(player_id="home_8", frame_start=10, frame_end=29, end_type="possession_loss",
             pass_outcome=None, target_pass=None, frame_receipt=np.nan,
             loss=True, game_interruption_after=None),
    ])


def test_team_link_requires_a_matching_successful_reception():
    players = player_segments()
    teams, annotated = link_team_possessions(players)
    assert len(teams) == 1 and teams.lost.item()
    assert annotated.team_possession_id.tolist() == [1, 1]
    for column, value in (("target_pass", "home_9"), ("pass_outcome", "unsuccessful"),
                          ("frame_receipt", 11), ("game_interruption_after", "foul")):
        changed = players.copy()
        changed.loc[0, column] = value
        assert len(link_team_possessions(changed)[0]) == 2


def test_invalid_intervals_and_duplicate_checkpoints_are_rejected(match):
    tracking, teams, actions, _ = load_test_match()
    teams.loc[1, "frame_start"] = 29
    with pytest.raises(ValueError, match="overlap"):
        PRESS(tracking, possessions_data=teams)
    model, _ = match
    model.actions_data = pd.concat([actions, actions.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Ambiguous"):
        model.get_actions(valid_only=True)


def test_full_checkpoint_and_recompute_invalidate_downstream_results(match):
    model, xt = match
    result = model.compute_risk_frame(0, xt, keep_details=True, ppcf_kwargs={"r_max": 12, "step": 6},
                                 bcos_kwargs={"r_max": 8, "step": 4})
    ppcf = result["PPCF"]
    probabilities = np.asarray(ppcf["Pass selection probabilities"])
    assert np.isclose(probabilities.sum(), 1)
    assert probabilities.shape == np.asarray(ppcf["Pass probability"]).shape
    p_pass = result["ACTION_SELECTION"]["Pass probability"]
    hazard = result["HAZARD"]
    assert hazard["band"] == hazard_to_band(hazard["Hazard"])
    assert 0 <= p_pass <= 1 and 0 <= hazard["Hazard"] <= 1
    assert np.isclose(hazard["Hazard"], (1-p_pass)*hazard["Drive risk"] + p_pass*hazard["Pass failure expected"])
    model.compute_ball_drive_risk(0, t_press=0.5)
    with pytest.raises(KeyError):
        model.get_hazard_result(0)
    assert "Pass selection probabilities" not in model.get_ppcf_result(0)
    with pytest.raises(ValueError, match="pass selection"):
        model.compute_hazard(0)


def test_possession_uses_action_checkpoints_not_every_tracking_frame(match):
    model, xt = match
    table = model.compute_possession(1, xt, ppcf_kwargs={"r_max": 12, "step": 6},
                                     bcos_kwargs={"r_max": 8, "step": 4})
    assert table.frame_id.tolist() == [0, 10, 20]
    assert table.band.tolist() == hazard_to_band(table.Hazard.to_numpy()).tolist()
    summary = possession_survival(table.Hazard)
    assert np.all(np.diff(summary["survival"]) <= 0)
    assert np.isclose(summary["survival"][-1], np.prod(1-table.Hazard))


def test_context_and_xt_errors_happen_before_calculation(match):
    model, xt = match
    with pytest.raises(ValueError, match="xT_grid"):
        model.compute_risk_frame(0, np.array([1, 2]))
    assert not any(model.results.values())
    result = model.compute_risk_frame(0, xt, start_type="unknown", previous_action="ball_drive",
        bcos_kwargs={"r_max":8,"step":4}, ppcf_kwargs={"r_max":12,"step":6})
    assert 0 <= result['ACTION_SELECTION']['Pass probability'] <= 1


def test_scalar_bands_and_absorbing_loss():
    assert hazard_to_band(0.04) == "None"
    assert hazard_to_band(1.0) == "High"
    assert possession_survival([0.1, 1.0, 0.2])["survival"].tolist() == [0.9, 0.0, 0.0]


def test_raw_skillcorner_loader_builds_actions_and_links_players(tmp_path):
    from _fixtures import write_skillcorner_match
    folder=write_skillcorner_match(tmp_path/'demo')
    model=PRESS.from_skillcorner(folder)
    assert model.get_player_possessions(team_possession_id=1).player_id.tolist()==['home_7','home_8']
    assert len(model.get_team_possessions())==2
    assert model.get_actions(team_possession_id=1, valid_only=True).frame_start.tolist()==[9,10,29]
    assert model.get_preparation_report()['n_tracking_frames']==40
    assert not any(model.results.values())


def test_missing_players_can_be_padded_and_invalid_carrier_is_rejected():
    tracking, *_ = load_test_match()
    # Simulate a frame with no detected defenders/teammates.
    player_cols = [c for c in tracking if c.endswith(("_x", "_y"))
                   and c.startswith(("home_", "away_")) and not c.startswith("home_7_")]
    tracking.loc[0, player_cols] = np.nan
    model = PRESS(tracking)
    state = model.get_state(0)
    assert state["r_defs"].shape == (11, 2)
    assert state["r_atts"].shape == (10, 2)
    assert state["defender_ids"] == [None] * 11
    model.tracking_data.loc[0, "home_7_vx"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.get_state(0)


def test_arbitrary_frame_resolves_context_without_checkpoint_membership(match):
    model, xt = match
    assert 5 not in set(model.get_actions(valid_only=True).frame_start)
    assert model.get_frame_context(5, granular=False) == {
        "start_type": "reception_open_play", "previous_action": "reception_open_play",
        "source": "action_interval",
    }
    result = model.compute_risk_frame(5, xt, ppcf_kwargs={"r_max": 12, "step": 6},
                                 bcos_kwargs={"r_max": 8, "step": 4})
    assert 0 <= result["HAZARD"]["Hazard"] <= 1


def test_frame_computation_needs_no_possession_or_segmentation_table(pass_risk_model):
    tracking, _, _, xt = load_test_match()
    model = PRESS(tracking,pass_risk_model=pass_risk_model)
    result = model.compute_risk_frame(5, xt, start_type="pass_reception",
                                 previous_action="ball_drive",
                                 ppcf_kwargs={"r_max": 12, "step": 6},
                                 bcos_kwargs={"r_max": 8, "step": 4})
    assert 0 <= result["HAZARD"]["Hazard"] <= 1


def test_custom_possession_sampling_preserves_single_frame_values(match):
    model, xt = match
    options = dict(ppcf_kwargs={"r_max": 12, "step": 6},
                   bcos_kwargs={"r_max": 8, "step": 4})
    direct = model.compute_risk_frame(5, xt, **options)["HAZARD"]["Hazard"]
    sampled = model.compute_possession(1, xt, frame_ids=[15, 5], **options)
    assert sampled.frame_id.tolist() == [5, 15]
    assert np.isclose(sampled.Hazard.iloc[0], direct)


@pytest.mark.parametrize("frames", [[], [5, 5], [5.5], [30], [[5]], ["5"]])
def test_invalid_custom_sampling_is_rejected_before_calculation(match, frames):
    model, xt = match
    with pytest.raises(ValueError, match="frame_ids"):
        model.compute_possession(1, xt, frame_ids=frames)
    assert not any(model.results.values())


def test_frame_annotations_take_priority_and_boundary_context_is_unambiguous(match):
    model, _ = match
    model.tracking_data["start_type_std"] = "continuation"
    model.tracking_data["previous_action_std"] = "ball_drive"
    assert model.get_frame_context(5, granular=False)["source"] == "tracking"
    del model.tracking_data["start_type_std"]
    del model.tracking_data["previous_action_std"]
    model.actions_data.loc[0, "frame_end"] = 10
    assert model.get_frame_context(10, granular=False)["start_type"] == "reception_open_play"
    conflict = model.actions_data.iloc[[1]].copy()
    conflict["start_type"] = "recovery"
    model.actions_data = pd.concat([model.actions_data, conflict], ignore_index=True)
    with pytest.raises(ValueError, match="Conflicting"):
        model.get_frame_context(10)


def test_pitch_choice_plots_use_stored_probabilities_and_return_figures(match):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PRESS.visualization import plot_pass_selection, plot_action_selection, plot_ppcf, plot_ball_drive_risk, plot_pass_map
    model, xt = match
    result = model.compute_risk_frame(5, xt, keep_details=True, ppcf_kwargs={"r_max": 12, "step": 6},
                                 bcos_kwargs={"r_max": 8, "step": 4})
    original = result["PPCF"]["Pass selection probabilities"].copy()
    fig, ax = plot_pass_selection(model, 5, pass_target=(4., 3.))
    fields = [c for c in ax.collections if c.get_array() is not None]
    assert any(c.norm.vmin == 0 and c.norm.vmax == 100 for c in fields)
    assert "Relative pass preference" in fig.axes[1].get_ylabel()
    np.testing.assert_array_equal(result["PPCF"]["Pass selection probabilities"], original)
    line = next(line for line in ax.lines if line.get_label() == "Observed pass")
    row = model.get_frame(5).iloc[0]
    np.testing.assert_allclose(line.get_xdata(), [row.ball_x, 4.])
    np.testing.assert_allclose(line.get_ydata(), [row.ball_y, 3.])
    assert line.get_linestyle() == "--"
    plt.close(fig)
    fig, ax = plot_action_selection(model, 5)
    expected = result["ACTION_SELECTION"]["Pass probability"]
    assert f"P(pass)={expected:.1%}" in ax.get_title()
    plt.close(fig)
    fig, ax = plot_pass_map(model, 5, field="pass_success")
    assert fig.axes[1].get_ylabel() == "Pass-success probability"
    assert any(c.norm.vmax == 1 for c in ax.collections if c.get_array() is not None)
    plt.close(fig)
    plot_ppcf(model, 5, field="pass_success")
    plt.close("all")
    plot_ball_drive_risk(model, 5)
    assert "P(loss)=" in plt.gca().get_title()
    plt.close("all")
