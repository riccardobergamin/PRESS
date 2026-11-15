"""Independent synchronization/segmentation checks, including unknown labels."""
import numpy as np
import pandas as pd
import pytest
from PRESS import PRESS
from PRESS.io import canonical_actions, normalize_events, synchronize_events, segment_possessions
from PRESS.utils.match import link_team_possessions
from _fixtures import load_test_match


def prepare_for_test(tracking, events, *, fps, already_synchronized=True):
    tracking, parents, actions, report = segment_possessions(tracking, events, fps=fps)
    return dict(tracking=tracking,player_possessions=parents,actions=actions,report=report)


def events(records):
    return normalize_events(pd.DataFrame(records))


def test_alignment_obeys_period_tolerance_and_retains_unmatched():
    tracking = pd.DataFrame(dict(frame=[10,11,30,31],period=[1,1,2,2],time_seconds=[0,.1,0,.1]))
    e = events([dict(event_id=i,event_type='pass',period=p,time_seconds=t)
                for i,p,t in [(1,1,.08),(2,2,.08),(3,2,4)]])
    result = synchronize_events(e,tracking,fps=10,max_time_error=.04)
    assert result.frame.tolist()[:2] == [11,31]
    assert pd.isna(result.frame.iloc[2])
    e = events([dict(event_id=1,event_type='unknown',frame=10,period=2)])
    assert not synchronize_events(e,tracking,fps=10,already_synchronized=True).sync_matched.item()


@pytest.mark.parametrize('ending,expected', [('pass',[0,10,24]),('possession_loss',[0,10,20])])
def test_canonical_24_seconds(ending,expected):
    parent = pd.DataFrame([dict(match_id='a',possession_id=1,player_id='home_7',frame_start=0,
        frame_end=24,start_type='pass_reception',end_type=ending,pass_outcome='successful',
        game_interruption_after=None)])
    result = canonical_actions(parent,fps=10)
    canonical = result.loc[result.is_canonical]
    assert canonical.frame_start.tolist()==expected
    assert canonical.previous_action.tolist()==['pass_reception','ball_drive','ball_drive']
    if ending=='possession_loss':assert canonical.outcome.tolist()==[1,1,0]


def controlled_match(success=True,restart='open_play',receiver=True):
    t,*_=load_test_match()
    t.player_possession=None
    t.loc[t.frame.le(24),'player_possession']='home_7'
    if receiver:t.loc[t.frame.ge(28),'player_possession']='home_8' if success else 'away_7'
    e=events([dict(event_id=1,event_type='pass',frame=24,player_id='home_7',team_id='home',
                   receiver_player_id='home_8' if success else None,is_successful=success,restart_type=restart)])
    return t,e


@pytest.mark.parametrize('success,label',[(True,'corner_reception'),(False,'corner_interception')])
def test_set_piece_receiver_and_opponent_context(success,label):
    t,e=controlled_match(success,restart='corner')
    result=prepare_for_test(t,e,fps=10,already_synchronized=True)
    p=result['player_possessions']
    assert p.iloc[0].frame_end==24 and p.iloc[0].outcome_verified
    assert p.iloc[1].frame_start==28 and p.iloc[1].start_type==label
    assert result['report']['one_frame_reception_fallback'] is False


def test_absent_reception_does_not_invent_endpoint():
    t,e=controlled_match(receiver=False)
    result=prepare_for_test(t,e,fps=10,already_synchronized=True)
    first=result['player_possessions'].iloc[0]
    assert pd.isna(first.frame_receipt) and not first.outcome_verified
    assert first.is_player_possession_end_matched  # matched release, unverified reception
    assert len(result['player_possessions'])==1


@pytest.mark.parametrize('kind',['clearance','foul_suffered','shot'])
def test_non_loss_terminal_events_are_not_converted_to_loss(kind):
    t,e=controlled_match(receiver=False)
    e.loc[0,'event_type']=kind
    result=prepare_for_test(t,e,fps=10,already_synchronized=True)
    first=result['player_possessions'].iloc[0]
    assert first.end_type==kind and not first.loss
    assert not result['actions'].loc[result['actions'].is_canonical,'outcome'].eq(0).any()


def test_unknown_outcome_is_not_false_and_complete_core_runs():
    t,e=controlled_match()
    e['is_successful']=pd.array([pd.NA],dtype='boolean')
    t,parents,actions,report=segment_possessions(t,e,fps=10,match_id='external')
    press=PRESS(t,player_possessions_data=parents,actions_data=actions,fps=10,match_id='external')
    first=press.player_possessions_data.iloc[0]
    assert pd.isna(first.loss)
    assert pd.isna(press.possessions_data.iloc[0].lost)
    assert pd.isna(press.actions_data.loc[press.actions_data.action.eq('pass'),'outcome'].iloc[0])
    assert report['n_events']==1
    assert press.get_frame_context(0)['start_type']=='unknown'
    # These categories must be accepted by the unchanged fitted encoders.
    model=press._action_model()
    assert all('unknown' in levels for levels in model.vocabulary.values())


def test_unknown_team_outcome_does_not_join_intervals():
    t,e=controlled_match()
    result=prepare_for_test(t,e,fps=10,already_synchronized=True)
    p=result['player_possessions'];p.loc[0,'loss']=pd.NA
    teams,_=link_team_possessions(p)
    assert len(teams)==2 and pd.isna(teams.lost.iloc[0])


@pytest.mark.parametrize('ending,frames,outcomes', [
    ('pass', [0, 10, 24], [1, 1, 1]),
    ('possession_loss', [0, 10, 20], [1, 1, 0]),
])
def test_constructor_builds_canonical_actions_from_external_possessions(ending, frames, outcomes):
    tracking, _ = controlled_match()
    parents = pd.DataFrame([dict(
        possession_id='external-control', player_id='home_7',
        frame_start=0, frame_end=24, start_type='pass_reception',
        end_type=ending, pass_outcome='successful' if ending == 'pass' else None,
        target_pass='home_8' if ending == 'pass' else None,
        frame_receipt=28 if ending == 'pass' else None,
        loss=ending == 'possession_loss', outcome_verified=True,
    )])
    original = parents.copy(deep=True)
    press = PRESS(tracking, player_possessions_data=parents, fps=10, match_id='external')
    actions = press.get_actions('external-control')
    assert actions.frame_start.tolist() == frames
    assert actions.outcome.tolist() == outcomes
    assert actions.outcome_verified.all()
    assert actions.previous_action.tolist() == ['pass_reception', 'ball_drive', 'ball_drive']
    assert actions.match_id.eq('external').all()
    pd.testing.assert_frame_equal(parents, original)


def test_constructor_preserves_explicit_actions_and_unknown_origin():
    tracking, event_table = controlled_match()
    tracking, parents, actions, _ = segment_possessions(tracking, event_table, fps=10)
    automatic = PRESS(tracking, player_possessions_data=parents, fps=10)
    explicit = PRESS(tracking, player_possessions_data=parents, actions_data=actions, fps=10)
    pd.testing.assert_frame_equal(automatic.actions_data, explicit.actions_data)
    parents = parents.drop(columns='start_type')
    unknown = PRESS(tracking, player_possessions_data=parents, fps=10)
    assert unknown.get_actions().start_type.eq('unknown').all()
