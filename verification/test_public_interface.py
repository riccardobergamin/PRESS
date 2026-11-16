"""Public data adapters, timing, persistence and composed predictions."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from PRESS import PRESS, MatchInfo
from _fixtures import load_test_match
from PRESS.model import compose_individual_xloss
from PRESS.io import convert_long_tracking
from PRESS.visualization import plot_frame, plot_risk


def long_tables():
    tracking, _, _, _ = load_test_match()
    rows = []
    for side, source_team in [("home", 101), ("away", 202)]:
        for number in range(1, 12):
            alias = f"{side}_{number}"
            for row in tracking.itertuples():
                rows.append(dict(frame=row.frame, team_id=source_team,
                                 player_id=f"source-{alias}",
                                 **{c: getattr(row, f"{alias}_{c}") for c in ["x", "y", "vx", "vy"]}))
    context = tracking[["frame", "player_possession", "attacking_side", "match_clock_min", "score_difference_prefix"]].copy()
    context.player_possession = "source-" + context.player_possession
    ball = tracking[["frame", "ball_x", "ball_y"]].rename(columns={"ball_x": "x", "ball_y": "y"})
    return pd.DataFrame(rows), ball, context


def individual_match(fps=25):
    tracking, _, _, _ = load_test_match()
    tracking.player_possession = "home_7"
    parents = pd.DataFrame([dict(possession_id="control-a", player_id="home_7",
        frame_start=0, frame_end=39, start_type="pass_reception", end_type="possession_loss",
        pass_outcome=None, target_pass=None, frame_receipt=None, loss=True)])
    return PRESS(tracking, player_possessions_data=parents, fps=fps)


def test_long_adapter_retains_original_ids_and_explicit_dimensions():
    players, ball, context = long_tables()
    tracking, roster = convert_long_tracking(players, ball, context, home_team=101, away_team=202)
    press = PRESS(tracking, player_metadata=roster, fps=25,
                  pitch_length=110, pitch_width=72, match_id="other-provider")
    assert press.get_match_info() == MatchInfo(25, 110, 72, "other-provider").to_dict()
    roster = press.get_players(team_id="home")
    carrier = roster.loc[roster.source_player_id.eq("source-home_7"), "player_id"].item()
    assert press.get_frame(5).player_possession.item() == carrier
    expected, *_ = load_test_match()
    state = press.get_state(5)
    np.testing.assert_allclose(state["x_bc"], expected.loc[5, ["home_7_x", "home_7_y"]].to_numpy(float))
    assert len(roster) == 11 and set(roster.source_team_id) == {101}
    roster.loc[:, "source_player_id"] = "changed"
    assert "changed" not in set(press.get_players().source_player_id)


def test_long_adapter_does_not_invent_missing_observations():
    players, ball, context = long_tables()
    players = players[~(players.frame.eq(5) & players.player_id.eq("source-away_3"))]
    tracking, roster = convert_long_tracking(players, ball, context, home_team=101, away_team=202)
    press = PRESS(tracking)
    assert press.get_state(5)["defender_ids"].count(None) == 1
    assert not players.columns.isin(["home_1_x"]).any()
    with pytest.raises(ValueError, match="Duplicate"):
        convert_long_tracking(pd.concat([players, players.iloc[:1]]), ball, context,
                           home_team=101, away_team=202)


def test_ambiguous_source_carriers_require_their_team():
    players, ball, context = long_tables()
    players.player_id = players.player_id.str.rsplit("_", n=1).str[-1]
    context.player_possession = context.player_possession.str.rsplit("_", n=1).str[-1]
    with pytest.raises(ValueError, match="ambiguous"):
        convert_long_tracking(players, ball, context, home_team=101, away_team=202)
    context["team_possession"] = np.where(context.frame < 30, 101, 202)
    tracking, roster = convert_long_tracking(players, ball, context, home_team=101, away_team=202)
    assert tracking.loc[0, "player_possession"].startswith("home_")
    assert tracking.loc[35, "player_possession"].startswith("away_")


def test_optional_speed_norm_and_input_ownership():
    data, *_ = load_test_match()
    data = data.drop(columns=[c for c in data if c.endswith("_v")])
    press = PRESS(data)
    np.testing.assert_allclose(press.get_frame(0).home_7_v, [.4])
    assert "home_7_v" not in data
    data.loc[0, "home_7_x"] = 999
    assert press.get_frame(0).home_7_x.item() != 999
    bad = data.copy()
    bad.loc[1, "away_1_vx"] = np.nan
    with pytest.raises(ValueError, match="finite velocity"):
        PRESS(bad)


@pytest.mark.parametrize("fps", [0, -1, np.nan, np.inf])
def test_invalid_frame_frequency_is_rejected(fps):
    data, *_ = load_test_match()
    with pytest.raises(ValueError, match="fps"):
        PRESS(data, fps=fps)


def test_individual_and_team_getters_are_explicit_and_timing_uses_fps():
    press = individual_match()
    assert press.get_team_possessions(1).frame_start.item() == 0
    assert press.get_player_possessions("control-a").frame_end.item() == 39
    assert len(press.get_player_frames("control-a")) == 40
    assert press.get_elapsed_time(24) == .96
    assert press.get_elapsed_time(25) == 1.0
    assert press.get_frame_context(24)["completed_canonical_drives"] == 0
    assert press.get_frame_context(25)["completed_canonical_drives"] == 1
    assert press.get_player_possessions("missing").empty


def test_results_are_copied_and_persist_arrays_without_pickle(tmp_path):
    data, *_ = load_test_match()
    press = PRESS(data)
    press.results["BALL_DRIVE_RISK"][5] = {"Ball drive risk": .2,
                              "Individual tackle probabilities": np.array([.1, .1], dtype=np.float32),
                              "params": {"unknown": np.nan}}
    result = press.get_ball_drive_risk_result(5)
    result["Individual tackle probabilities"][0] = 99
    assert press.get_ball_drive_risk_result(5)["Individual tackle probabilities"][0] < 1
    folder = press.save_results(tmp_path / "results")
    with np.load(folder / "arrays.npz", allow_pickle=False) as arrays:
        assert arrays.files
    restored = PRESS(data)
    restored.load_results(folder)
    actual = restored.get_ball_drive_risk_result(5)
    np.testing.assert_array_equal(actual["Individual tackle probabilities"], np.array([.1, .1], dtype=np.float32))
    assert actual["Individual tackle probabilities"].dtype == np.float32
    assert np.isnan(actual["params"]["unknown"])
    with pytest.raises(FileExistsError):
        press.save_results(folder)
    with pytest.raises(ValueError, match="already exists"):
        restored.load_results(folder)
    data.loc[0, "home_7_x"] += 1
    other = PRESS(data)
    with pytest.raises(ValueError, match="do not match"):
        other.load_results(folder)
    assert not any(other.results.values())


def test_export_rejects_modified_array_file_before_loading(tmp_path):
    data, *_ = load_test_match()
    press = PRESS(data)
    folder = press.save_results(tmp_path / "export")
    with (folder / "arrays.npz").open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        PRESS(data).load_results(folder)


def test_native_xloss_reuses_current_action_features_and_observed_age():
    press = individual_match(fps=10)
    outputs = press.compute_risk_frame(5, include_residual=True,
                                  ppcf_kwargs={"r_max": 12, "step": 6},
                                  bcos_kwargs={"r_max": 8, "step": 4})
    assert outputs["RESIDUAL_RISK"]["time_since_start"] == .5
    h = outputs["HAZARD"]
    policy = outputs["ACTION_SELECTION"]
    expected = compose_individual_xloss(p_pass=policy["Pass probability"],
        drive_loss=h["Drive risk"], pass_failure=h["Pass failure expected"],
        residual_loss=outputs["RESIDUAL_RISK"]["Residual loss probability"])
    assert outputs["XLOSS"]["xLoss"] == expected
    assert expected >= h["Hazard"]
    press.compute_ball_drive_risk(5)
    assert 5 not in press.get_results("ACTION_SELECTION")
    assert 5 not in press.get_results("RESIDUAL_RISK")
    assert 5 not in press.get_results("XLOSS")


def test_elapsed_time_is_not_silently_imputed():
    tracking, *_ = load_test_match()
    press = PRESS(tracking)
    with pytest.raises(ValueError, match="elapsed"):
        press.get_elapsed_time(5)
    with pytest.raises(ValueError, match="elapsed"):
        press.compute_risk_frame(5, include_residual=True)
    assert not any(press.results.values())


def test_clear_upstream_results_invalidates_all_dependants():
    press = individual_match()
    for component in press.results:
        press.results[component][0] = {"marker": 1}
    press.clear_results("BALL_DRIVE", frame_ids=[0])
    for component in ["BALL_DRIVE_RISK", "ACTION_SELECTION", "HAZARD", "RESIDUAL_RISK", "XLOSS"]:
        assert not press.results[component]
    assert press.results["BCOS"]
    press.clear_results()
    assert not any(press.results.values())


def test_plots_return_explicit_axes_and_allow_inflight_frames_without_carrier():
    tracking, *_ = load_test_match()
    tracking.loc[5, "player_possession"] = None
    press = PRESS(tracking)
    fig, axes = plt.subplots()
    actual_fig, actual_axes = plot_frame(press, 5, ax=axes, show=False, legend=False)
    assert actual_fig is fig and actual_axes is axes
    assert not fig.legends
    plt.close(fig)
    press.results["HAZARD"][0] = {"Hazard": .3}
    fig, axes = plot_risk(press, 0, show=False)
    assert "h=30.00%" in axes.get_title()
    plt.close(fig)


def test_long_adapter_remaps_individual_interval_ids():
    players, ball, context = long_tables()
    context.player_possession = "source-home_7"
    parents = pd.DataFrame([dict(possession_id="source-control", team_id=101,
        player_id="source-home_7", frame_start=0, frame_end=39,
        start_type="pass_reception", end_type="possession_loss", pass_outcome=None,
        target_pass=None, frame_receipt=None, loss=True)])
    tracking, roster = convert_long_tracking(players, ball, context, home_team=101, away_team=202)
    mapping=roster.set_index('source_player_id').player_id
    prepared=parents.copy()
    prepared['player_id']=prepared.player_id.map(mapping)
    prepared['team_id']='home'
    press = PRESS(tracking, player_metadata=roster, player_possessions_data=prepared, fps=25)
    assert press.get_player_possessions("source-control").player_id.item().startswith("home_")
    assert press.get_elapsed_time(25) == 1
    assert parents.player_id.item() == "source-home_7"


def test_individual_sampling_validates_interval_before_computing(monkeypatch):
    press = individual_match()
    seen = []
    def compute(frame, grid, **kwargs):
        seen.append(frame)
        return {"ACTION_SELECTION": {"Pass probability": .4}, "HAZARD": {"Hazard": .2}}
    monkeypatch.setattr(press, "compute_risk_frame", compute)
    table = press.compute_risk_player_possession("control-a", frame_ids=[30, 5])
    assert table.frame_id.tolist() == seen == [5, 30]
    assert table.player_possession_id.eq("control-a").all()
    with pytest.raises(ValueError, match="frame_ids"):
        press.compute_risk_player_possession("control-a", frame_ids=[40])
    assert len(seen) == 2


def test_export_requires_the_same_physical_model(tmp_path):
    from PRESS.model import PassRiskModel
    data, *_ = load_test_match()
    folder = PRESS(data).save_results(tmp_path / "results")
    artifact = PassRiskModel.pretrained().to_dict()
    artifact["lambda_control"] *= 1.1
    press = PRESS(data, pass_risk_model=PassRiskModel(artifact))
    with pytest.raises(ValueError, match="different fitted model"):
        press.load_results(folder)
    assert not any(press.results.values())


def test_fractional_pitch_dimensions_do_not_require_a_provider_specific_grid(monkeypatch):
    from PRESS.model import PassRiskModel
    data, *_ = load_test_match()
    press = PRESS(data, pitch_length=104.5, pitch_width=68.4)
    def predict(self, positions, velocity, bc, targets, attackers, carrier, **kwargs):
        return dict(control=np.zeros((len(positions), len(targets))), raw=np.full(len(targets), .4),
                    calibrated=np.full(len(targets), .5), flight_time=np.zeros(len(targets)),
                    physics=self.physics, calibration_matches_physics=True)
    monkeypatch.setattr(PassRiskModel, "predict_field", predict)
    press.compute_ppcf(0)
    result = press.get_ppcf_result(0)
    assert result["grid"].shape == (104 * 68, 2)
    assert press.get_result_configuration(0,"PPCF")["PPCF"]["choice_grid_matches_benchmark"] is False


@pytest.mark.parametrize("value", [None, np.nan, pd.NA])
def test_explicit_missing_tracking_age_raises_a_clear_error(value):
    data, *_ = load_test_match()
    data["time_since_start"] = value
    with pytest.raises(ValueError, match="elapsed"):
        PRESS(data).get_elapsed_time(5)


@pytest.mark.parametrize("age", [-1, np.nan, np.inf])
def test_invalid_requested_age_is_rejected_before_frame_prediction(age):
    data, *_ = load_test_match()
    press = PRESS(data)
    with pytest.raises(ValueError, match="time_since_start"):
        press.compute_risk_frame(5, include_residual=True, time_since_start=age)
    assert not any(press.results.values())
