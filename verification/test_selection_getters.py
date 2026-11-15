"""Possession/action/checkpoint selection without client-side table slicing."""

import pandas as pd
import pytest

from PRESS import PRESS
from _fixtures import load_test_match, write_skillcorner_match


@pytest.fixture
def match(tmp_path):
    return PRESS.from_skillcorner(write_skillcorner_match(tmp_path / '123'))


def test_individual_ids_are_consistent_across_getters(match):
    assert match.get_player_possessions().possession_id.tolist() == [1, 2, 3]
    assert match.get_player_possessions(2).player_id.tolist() == ['home_8']
    assert match.get_actions(2).frame_start.tolist() == [10, 29]
    assert match.get_actions(2, valid_only=True).frame_start.tolist() == [10, 29]
    assert match.get_player_possessions(2).player_id.item() == 'home_8'


def test_team_player_and_combined_filters(match):
    assert match.get_player_possessions(team_id='home').player_id.tolist() == ['home_7', 'home_8']
    assert match.get_player_possessions(player_id='home_8').possession_id.tolist() == [2]
    assert match.get_actions(team_id='home').frame_start.tolist() == [9, 10, 29]
    assert match.get_actions(player_id='home_8', valid_only=True).frame_start.tolist() == [10, 29]
    assert match.get_actions(2, team_id='home', player_id='home_8').frame_start.tolist() == [10, 29]
    assert match.get_player_possessions(2, team_id='away').empty
    assert match.get_actions(2, player_id='away_7').empty
    assert match.get_actions(2, team_id='away', valid_only=True).empty


def test_team_interval_selection_is_explicit(match):
    assert isinstance(match.get_team_possessions(), pd.DataFrame)
    assert match.get_team_possessions().possession_id.tolist() == [1, 2]
    assert match.get_team_possessions(1).possession_id.tolist() == [1]
    assert match.get_team_possessions(team_id='away').possession_id.tolist() == [2]
    assert match.get_team_possessions(1, team_id='away').empty
    assert match.get_team_possessions('unknown').empty
    assert match.get_player_possessions(team_possession_id=1).possession_id.tolist() == [1, 2]
    assert match.get_actions(team_possession_id=1).frame_start.tolist() == [9, 10, 29]
    assert match.get_actions(team_possession_id=1, valid_only=True).frame_start.tolist() == [9, 10, 29]
    assert match.get_actions(team_possession_id=1, player_id='home_8', valid_only=True).frame_start.tolist() == [10, 29]


def test_canonical_defaults_and_optional_snapshots(match):
    assert match.get_actions().frame_start.tolist() == [9, 10, 29, 30]
    assert match.get_actions(include_snapshots=True).frame_start.tolist() == [0, 9, 10, 20, 29, 30]
    assert match.get_actions(2, include_snapshots=True, valid_only=True).frame_start.tolist() == [10, 20, 29]
    match.tracking_data.loc[match.tracking_data.frame.eq(10), 'home_8_x'] = float('nan')
    assert match.get_actions(2, valid_only=True).frame_start.tolist() == [29]
    assert match.get_actions(2, valid_only=False).frame_start.tolist() == [10, 29]
    assert match.get_actions(2).frame_start.tolist() == [10, 29]


def test_selection_leaves_input_tables_and_result_cache_unchanged(match):
    actions_before = match.actions_data.copy(deep=True)
    individuals_before = match.player_possessions_data.copy(deep=True)
    for table in (match.get_player_possessions(), match.get_actions(), match.get_actions(valid_only=True)):
        table.loc[0, 'frame_start'] = -999
    pd.testing.assert_frame_equal(match.actions_data, actions_before)
    pd.testing.assert_frame_equal(match.player_possessions_data, individuals_before)
    assert not any(match.results.values())


def test_missing_action_identity_can_be_recovered_from_explicit_intervals(match):
    match.actions_data = match.actions_data.drop(
        columns=['possession_id', 'player_id', 'team_id', 'team_possession_id'], errors='ignore')
    assert match.get_actions(2).frame_start.tolist() == [10, 29]
    assert match.get_actions(player_id='away_7', valid_only=True).frame_start.tolist() == [30]


def test_minimal_action_tables_allow_player_filter_but_do_not_invent_possessions():
    tracking, teams, actions, _ = load_test_match()
    match = PRESS(tracking, possessions_data=teams, actions_data=actions)
    assert match.get_actions(player_id='home_8').frame_start.tolist() == [10, 20]
    assert match.get_actions(team_id='away', valid_only=True).frame_start.tolist() == [30]
    with pytest.raises(ValueError, match='Individual selection requires'):
        match.get_actions('unknown')


@pytest.mark.parametrize('getter', ['get_team_possessions', 'get_player_possessions', 'get_actions'])
def test_empty_selection_and_invalid_team(match, getter):
    select = getattr(match, getter)
    assert select('unknown').empty
    with pytest.raises(ValueError, match='team_id must be home or away'):
        select(team_id='other')

