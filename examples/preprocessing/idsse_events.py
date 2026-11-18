"""Sportec event semantics shared by the optional bridge and PRESS adapter.

This module uses the standard library only, so both Python environments read
the same identities, terminal qualifiers and contact/decision duplicates.
Synchronization categories are deliberately separate from semantic labels.
"""

from datetime import datetime
import xml.etree.ElementTree as ET


def active_event_nodes(path):
    """Apply provider revisions/deletions before reading event annotations."""
    active = {}
    for node in ET.parse(path).getroot().findall('Event'):
        identifier = str(node.get('EventId'))
        if node.find('Delete') is not None:
            active.pop(identifier, None)
        else:
            active[identifier] = node
    return active


def read_provider_events(path):
    """Read annotated actors; do not infer outcomes from the next controller.

    A TacklingGame actor is its winner. Both participants are retained;
    ballClaimed and ballcontactSucceeded identify won tackles. BallClaiming
    is a separate control-start annotation. Foul decisions following a fouled
    contact with the same victim/offender refer to that contact, not a second
    foul. Pair adjacent contact/decision records in chronological foul order,
    consuming each contact once; no temporal tolerance is fitted or imposed.
    """
    records = []
    for identifier, node in active_event_nodes(path).items():
        play, shot = node.find('.//Play'), node.find('.//ShotAtGoal')
        claim, duel = node.find('.//BallClaiming'), node.find('.//TacklingGame')
        foul, contact = node.find('.//Foul'), node.find('.//OtherBallAction')
        record = dict(
            event_id=identifier, datetime=node.get('EventTime'),
            event_type='unknown', player_id=None, team_id=None,
            receiver_player_id=None, is_successful=None,
            restart_type=next((label for tag, label in [
                ('ThrowIn', 'throw_in'), ('CornerKick', 'corner'),
                ('FreeKick', 'free_kick'), ('GoalKick', 'goal_kick'),
                ('KickOff', 'kick_off'),
            ] if node.find('.//' + tag) is not None), 'unknown'),
            original_event=next(iter(node)).tag,
            winner_player_id=None, loser_player_id=None,
            winner_role=None, loser_role=None, possession_change=None,
            fouler_player_id=None, recovery_type=None,
            duplicate_of=None, excluded_reason=None,
        )
        if play is not None:
            record.update(event_type='pass', player_id=play.get('Player'),
                          team_id=play.get('Team'), receiver_player_id=play.get('Recipient'),
                          is_successful={'successful': True, 'successfullyCompleted': True,
                                         'unsuccessful': False}.get(play.get('Evaluation')))
            if record['restart_type'] == 'unknown' and play.get('FromOpenPlay') == 'true':
                record['restart_type'] = 'open_play'
        elif shot is not None:
            record.update(event_type='shot', player_id=shot.get('Player'), team_id=shot.get('Team'))
        elif claim is not None:
            record.update(event_type='interception' if claim.get('Type') == 'InterceptedBall' else 'recovery',
                          player_id=claim.get('Player'), team_id=claim.get('Team'),
                          recovery_type=claim.get('Type'))
        elif duel is not None:
            result = duel.get('WinnerResult')
            record.update(player_id=duel.get('Winner'), team_id=duel.get('WinnerTeam'),
                          original_event=result, winner_player_id=duel.get('Winner'),
                          loser_player_id=duel.get('Loser'), winner_role=duel.get('WinnerRole'),
                          loser_role=duel.get('LoserRole'), possession_change=duel.get('PossessionChange'))
            if result in {'ballClaimed', 'ballcontactSucceeded'}:
                record.update(event_type='tackle', is_successful=True)
            elif result == 'fouled':
                record.update(event_type='foul_suffered', fouler_player_id=duel.get('Loser'))
        elif foul is not None:
            record.update(player_id=foul.get('Fouled'), team_id=foul.get('TeamFouled'),
                          fouler_player_id=foul.get('Fouler'))
            if foul.get('FoulType') == 'handBall':
                record['excluded_reason'] = 'handball'
            else:
                record['event_type'] = 'foul_suffered'
        elif contact is not None:
            record.update(player_id=contact.get('Player'), team_id=contact.get('Team'))
            if contact.get('DefensiveClearance') == 'true':
                record['event_type'] = 'clearance'
        elif node.find('.//FinalWhistle') is not None:
            record['event_type'] = 'stoppage'
        records.append(record)

    pending = None
    for record in sorted(records, key=lambda r: datetime.fromisoformat(r['datetime'])):
        if record['restart_type'] == 'kick_off':
            pending = None
        if record['original_event'] == 'fouled':
            pending = record
        elif record['event_type'] == 'foul_suffered':
            if (pending is not None and record['player_id'] is not None
                    and record['player_id'] == pending['player_id']
                    and record['fouler_player_id'] == pending['fouler_player_id']):
                record['duplicate_of'] = pending['event_id']
            pending = None
    return records
