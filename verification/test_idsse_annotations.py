"""Author-defined external-event outcomes, independent of control changes."""

import pandas as pd
import pytest

from PRESS.io import normalize_events, segment_possessions
from examples.preprocessing.idsse_events import read_provider_events
from examples.preprocessing.idsse import align_remaining_events
from _fixtures import load_test_match


def test_provider_semantics_and_foul_deduplication(tmp_path):
    source = tmp_path / 'events.xml'
    source.write_text('''<Events>
      <Event EventId="1" EventTime="2023-05-27T13:00:00+00:00">
        <TacklingGame Winner="w" WinnerTeam="away" Loser="l"
          WinnerResult="ballClaimed" PossessionChange="false"/>
      </Event>
      <Event EventId="2" EventTime="2023-05-27T13:00:01+00:00">
        <TacklingGame Winner="v" WinnerTeam="home" Loser="o" WinnerResult="fouled"/>
      </Event>
      <Event EventId="3" EventTime="2023-05-27T13:00:07+00:00">
        <Foul Fouled="v" TeamFouled="home" Fouler="o" FoulType="foul"/>
      </Event>
      <Event EventId="4" EventTime="2023-05-27T13:00:08+00:00">
        <Foul Fouler="o" FoulType="handBall"/>
      </Event>
      <Event EventId="5" EventTime="2023-05-27T13:00:09+00:00">
        <OtherBallAction Player="l" Team="home" DefensiveClearance="true"/>
      </Event>
      <Event EventId="6" EventTime="2023-05-27T13:00:10+00:00">
        <BallClaiming Player="l" Team="home" Type="BallClaimed"/>
      </Event>
      <Event EventId="7" EventTime="2023-05-27T13:00:11+00:00">
        <BallClaiming Player="w" Team="away" Type="InterceptedBall"/>
      </Event>
    </Events>''')
    rows = {row['event_id']: row for row in read_provider_events(source)}
    assert rows['1']['event_type'] == 'tackle'
    assert rows['1']['winner_player_id'] == 'w'
    assert rows['1']['loser_player_id'] == 'l'
    assert rows['1']['is_successful'] is True
    assert rows['2']['event_type'] == 'foul_suffered'
    assert rows['3']['duplicate_of'] == '2'
    assert rows['4']['excluded_reason'] == 'handball'
    assert rows['5']['event_type'] == 'clearance'
    assert rows['6']['event_type'] == 'recovery'
    assert rows['7']['event_type'] == 'interception'


@pytest.mark.parametrize('loser', ['home_7', 'home_8'])
def test_won_tackle_requires_only_a_different_winner(loser):
    tracking, *_ = load_test_match()
    tracking.player_possession = 'home_7'
    event = normalize_events(pd.DataFrame([dict(
        event_id='duel', event_type='tackle', frame=24,
        player_id='away_7', winner_player_id='away_7', loser_player_id=loser,
        is_successful=True, possession_change=False,
    )]))
    updated, parents, actions, _ = segment_possessions(tracking, event, fps=10)
    first = parents.iloc[0]
    assert first.frame_end == 24
    assert first.end_type == 'possession_loss' and first.loss and first.outcome_verified
    assert first.terminal_event_id == 'duel'
    assert actions.loc[actions.is_canonical, 'outcome'].tolist() == [1, 1, 0]
    assert updated.loc[updated.frame.eq(24), 'estimated_player_possession'].item() == 'home_7'


def test_tackle_won_by_carrier_is_not_its_terminal_loss():
    tracking, *_ = load_test_match()
    tracking.player_possession = 'home_7'
    event = normalize_events(pd.DataFrame([
        dict(event_id='duel', event_type='tackle', frame=24, player_id='home_7',
             winner_player_id='home_7', is_successful=True),
        dict(event_id='shot', event_type='shot', frame=30, player_id='home_7'),
    ]))
    _, parents, _, _ = segment_possessions(tracking, event, fps=10)
    assert parents.iloc[0].end_type == 'shot'
    assert not parents.iloc[0].loss


@pytest.mark.parametrize('kind,label', [('recovery', 'recovery'), ('interception', 'pass_interception')])
def test_annotated_claim_starts_without_estimator_confirmation(kind, label):
    tracking, *_ = load_test_match()
    tracking.player_possession = None
    event = normalize_events(pd.DataFrame([
        dict(event_id='claim', event_type=kind, frame=10, player_id='home_7'),
        dict(event_id='shot', event_type='shot', frame=24, player_id='home_7'),
    ]))
    updated, parents, _, _ = segment_possessions(tracking, event, fps=10)
    assert parents.iloc[0].frame_start == 10
    assert parents.iloc[0].start_type == label
    assert updated.loc[updated.frame.eq(10), 'player_possession'].item() == 'home_7'


def test_clock_interpolation_is_not_a_verified_clearance_contact():
    clock = pd.date_range('2023-05-27T13:00:00Z', periods=11, freq='100ms')
    tracking = pd.DataFrame(dict(frame=range(11), period=1, datetime=clock))
    events = pd.DataFrame(dict(
        event_id=['a', 'clearance', 'b'], event_type=['pass', 'clearance', 'pass'],
        datetime=[clock[0], clock[5], clock[10]],
        is_successful=pd.array([True, pd.NA, True], dtype='boolean'),
    ))
    aligned = pd.DataFrame(dict(original_event_id=['a', 'b'], frame=[0, 10],
                                period_id=[1, 1], datetime=[clock[0], clock[10]], sync_certainty=[0.1, 0.1]))
    result = align_remaining_events(events, aligned, tracking)
    clearance = result.loc[result.event_id.eq('clearance')].iloc[0]
    assert pd.isna(clearance.frame)
    assert not clearance.outcome_verified


def test_event_start_policy_does_not_create_tracking_only_controls():
    tracking, *_ = load_test_match()
    tracking.player_possession = 'home_7'
    event = normalize_events(pd.DataFrame([
        dict(event_id='claim', event_type='recovery', frame=10, player_id='home_7'),
        dict(event_id='shot', event_type='shot', frame=24, player_id='home_7'),
    ]))
    _, parents, _, report = segment_possessions(
        tracking, event, fps=10, include_unannotated_starts=False)
    assert parents.iloc[0].frame_start == 10
    assert parents.iloc[0].boundary_source == 'annotated_control'
    assert report['include_unannotated_starts'] is False


def test_clock_only_actor_does_not_truncate_reception_search():
    tracking, *_ = load_test_match()
    tracking.player_possession = 'home_7'
    tracking.loc[tracking.frame.ge(28), 'player_possession'] = 'home_8'
    event = normalize_events(pd.DataFrame([
        dict(event_id='initial', event_type='recovery', frame=0,
             player_id='home_7', sync_method='databallpy'),
        dict(event_id='pass', event_type='pass', frame=24, player_id='home_7',
             receiver_player_id='home_8', is_successful=True, sync_method='databallpy'),
        dict(event_id='clock', event_type='unknown', frame=26,
             player_id='home_8', sync_method='interpolated_clock'),
        dict(event_id='shot', event_type='shot', frame=35,
             player_id='home_8', sync_method='databallpy'),
    ]))
    _, parents, _, _ = segment_possessions(
        tracking, event, fps=10, include_unannotated_starts=False)
    receipt = parents.loc[parents.player_id.eq('home_8')].iloc[0]
    assert receipt.frame_start == 28
    assert receipt.end_type == 'shot'


def overlapping_tackle_events(*, loser='home_7', claimant='away_7', intervening=False):
    tracking, *_ = load_test_match()
    tracking.player_possession = 'home_7'
    tracking.loc[tracking.frame.ge(20), 'player_possession'] = 'away_7'
    records = [
        dict(event_id='start', event_type='recovery', frame=0, player_id='home_7'),
        dict(event_id='claim', event_type='recovery', frame=20, player_id=claimant),
        dict(event_id='duel', event_type='tackle', frame=22, player_id='away_7',
             winner_player_id='away_7', loser_player_id=loser, is_successful=True),
        dict(event_id='shot', event_type='shot', frame=30, player_id='away_7'),
    ]
    if intervening:
        records.append(dict(event_id='other', event_type='recovery', frame=21, player_id='home_8'))
    return tracking, normalize_events(pd.DataFrame(records))


def test_coherent_claim_tackle_overlap_keeps_both_controls():
    tracking, events = overlapping_tackle_events()
    original = tracking.copy(deep=True)
    updated, parents, actions, report = segment_possessions(
        tracking, events, fps=10, include_unannotated_starts=False)
    assert parents.player_id.tolist() == ['home_7', 'away_7']
    assert parents.frame_start.tolist() == [0, 23]
    assert parents.frame_end.tolist() == [22, 30]
    incoming = parents.iloc[1]
    assert incoming.start_event_id == 'claim'
    assert incoming.original_frame_start == 20
    assert incoming.start_reconciled_with_event_id == 'duel'
    assert incoming.boundary_source == 'reconciled_tackle_recovery'
    assert len(report['overlap_reconciliations']) == 1
    assert report['overlapping_start_annotations'] == []
    assert updated.loc[updated.frame.eq(22), 'player_possession'].item() == 'home_7'
    assert updated.loc[updated.frame.eq(23), 'player_possession'].item() == 'away_7'
    assert actions.loc[actions.possession_id.eq(incoming.possession_id), 'frame_start'].ge(23).all()
    pd.testing.assert_frame_equal(tracking, original)


@pytest.mark.parametrize('changes', [
    dict(loser='home_8'), dict(claimant='away_8'), dict(intervening=True),
])
def test_inconsistent_overlaps_are_not_reconciled(changes):
    tracking, events = overlapping_tackle_events(**changes)
    _, _, _, report = segment_possessions(
        tracking, events, fps=10, include_unannotated_starts=False)
    assert report['overlap_reconciliations'] == []
    assert any(row['frame'] == 20 for row in report['overlapping_start_annotations'])


def missed_receiver_events(*, distance=0.5, actor_frame=24, missing=False):
    tracking, *_ = load_test_match()
    tracking.player_possession = None
    tracking.loc[tracking.frame.le(10), 'player_possession'] = 'home_7'
    tracking.loc[tracking.frame.eq(actor_frame-1), ['home_8_x', 'home_8_y']] = [0.0, 0.0]
    tracking.loc[tracking.frame.eq(actor_frame-1), ['ball_x', 'ball_y']] = [distance, 0.0]
    if missing:
        tracking.loc[tracking.frame.eq(actor_frame-1), 'ball_x'] = float('nan')
    events = normalize_events(pd.DataFrame([
        dict(event_id='start', event_type='recovery', frame=0, player_id='home_7'),
        dict(event_id='delivery', event_type='pass', frame=10, player_id='home_7',
             receiver_player_id='home_8', is_successful=True, restart_type='open_play'),
        dict(event_id='release', event_type='pass', frame=actor_frame, player_id='home_8',
             receiver_player_id='home_9', is_successful=True),
    ]))
    return tracking, events


@pytest.mark.parametrize('distance', [0.0, 0.5, 1.5])
def test_missed_receiver_keeps_only_distance_confirmed_micropossession(distance):
    tracking, events = missed_receiver_events(distance=distance)
    updated, parents, actions, report = segment_possessions(
        tracking, events, fps=10, include_unannotated_starts=False)
    incoming = parents.loc[parents.player_id.eq('home_8')].iloc[0]
    assert incoming.frame_start == 23 and incoming.frame_end == 24
    assert incoming.boundary_source == 'receiver_contact_fallback'
    assert incoming.start_event_id == 'delivery'
    assert incoming.start_support_event_id == 'release'
    assert incoming.start_ball_distance_m == distance
    assert not incoming.outcome_verified  # No next receiver is fabricated.
    assert report['one_frame_reception_fallback'] is True
    assert report['reception_fallback_checks'][0]['reason'] == 'distance_confirmed'
    assert updated.loc[updated.frame.eq(23), 'player_possession'].item() == 'home_8'
    canonical = actions.loc[actions.possession_id.eq(incoming.possession_id) & actions.is_canonical]
    assert canonical.action.tolist() == ['pass']
    assert canonical.frame_start.tolist() == [24]


@pytest.mark.parametrize('kwargs, reason', [
    (dict(distance=1.5001), 'too_far_from_ball'),
    (dict(missing=True), 'missing_positions'),
    (dict(actor_frame=11), 'not_after_release'),
])
def test_unsupported_missed_receiver_is_rejected(kwargs, reason):
    tracking, events = missed_receiver_events(**kwargs)
    _, parents, _, report = segment_possessions(
        tracking, events, fps=10, include_unannotated_starts=False)
    assert not parents.player_id.eq('home_8').any()
    assert report['reception_fallback_checks'][0]['reason'] == reason


def test_missing_candidate_frame_cannot_support_a_reception_fallback():
    tracking, events = missed_receiver_events()
    tracking = tracking.loc[tracking.frame.ne(23)]
    _, parents, _, report = segment_possessions(
        tracking, events, fps=10, include_unannotated_starts=False)
    assert not parents.player_id.eq('home_8').any()
    assert report['reception_fallback_checks'][0]['reason'] == 'missing_candidate_frame'


def test_fallback_threshold_can_be_disabled_or_set_in_metres():
    tracking, events = missed_receiver_events(distance=1.0)
    for threshold in [None, 0.5]:
        _, parents, _, _ = segment_possessions(
            tracking, events, fps=10, include_unannotated_starts=False,
            reception_fallback_max_distance=threshold)
        assert not parents.player_id.eq('home_8').any()
    with pytest.raises(ValueError, match='reception_fallback_max_distance'):
        segment_possessions(tracking, events, fps=10, reception_fallback_max_distance=-1)


@pytest.mark.parametrize('evaluation', ['successful', 'successfullyCompleted'])
def test_success_variants_keep_known_outcome_without_recipient(tmp_path, evaluation):
    source = tmp_path / 'events.xml'
    source.write_text(f'''<Events>
      <Event EventId="claim" EventTime="2023-05-27T13:00:00+00:00">
        <BallClaiming Player="home_7" Team="home" Type="BallClaimed"/>
      </Event>
      <Event EventId="pass" EventTime="2023-05-27T13:00:02.4+00:00">
        <Play Player="home_7" Team="home" Evaluation="{evaluation}" FromOpenPlay="true"/>
      </Event>
    </Events>''')
    rows = pd.DataFrame(read_provider_events(source))
    assert rows.loc[rows.event_id.eq('pass'), 'is_successful'].item() is True
    assert pd.isna(rows.loc[rows.event_id.eq('pass'), 'receiver_player_id'].item())
    rows['frame'] = [0, 24]
    tracking, *_ = load_test_match()
    tracking.player_possession = 'home_7'
    _, parents, actions, _ = segment_possessions(
        tracking, normalize_events(rows), fps=10, include_unannotated_starts=False)
    interval = parents.iloc[0]
    assert interval.end_type == 'pass' and interval.frame_end == 24
    assert interval.pass_outcome == 'successful'
    assert not interval.loss and not interval.outcome_verified
    terminal = actions.loc[actions.action.eq('pass')].iloc[0]
    assert terminal.outcome == 1 and not terminal.outcome_verified
