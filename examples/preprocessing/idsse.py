"""IDSSE Sportec XML -> portable canonical PRESS input tables.

Run databallpy_bridge.py in its optional environment first. This adapter uses
only numpy/pandas plus PRESS's shared tracking derivative. Raw data remain
outside the package; custom segmentation and context labels belong to PRESS.
"""
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PRESS.io import differentiate


def match_files(data_dir, match_id):
    files = {}
    for name, prefix in [('info', 'DFL_02_01_matchinformation_'),
                         ('events', 'DFL_03_02_events_raw_')]:
        candidates = list(Path(data_dir).rglob(f'{prefix}*DFL-MAT-{match_id}.xml'))
        if len(candidates) != 1:
            raise ValueError(f'Expected one {name} XML for {match_id}; found {len(candidates)}.')
        files[name] = candidates[0]
    return files


def read_metadata(path, match_id):
    root = ET.parse(path).getroot()
    general = root.find('.//General').attrib
    environment = root.find('.//Environment').attrib
    players, teams = [], {}
    for team in root.findall('.//Teams/Team'):
        side = 'home' if team.get('Role') == 'home' else 'away'
        teams[team.get('TeamId')] = side
        for player in team.findall('./Players/Player'):
            players.append(dict(
                player_id=f"{side}_{int(player.get('ShirtNumber'))}", team_id=side,
                source_player_id=player.get('PersonId'), source_team_id=team.get('TeamId'),
                full_name=(player.get('FirstName', '') + ' ' + player.get('LastName', '')).strip(),
                is_goalkeeper=(player.get('PlayingPosition') == 'TW'
                               if player.get('PlayingPosition') is not None else pd.NA),
            ))
    roster = pd.DataFrame(players)
    roster['is_goalkeeper'] = roster.is_goalkeeper.astype('boolean')
    return roster, teams, dict(
        match_id=str(match_id), fps=25.0, pitch_length=float(environment['PitchX']),
        pitch_width=float(environment['PitchY']), home_team=general['HomeTeamName'],
        away_team=general['GuestTeamName'],
    )


def read_event_attributes(path, players, teams):
    """Read provider semantics, not estimated outcomes or receiver intentions."""
    aliases = players.set_index('source_player_id').player_id.to_dict()
    # Later revisions of the same event replace earlier ones; Delete removes
    # the event identified by its enclosing EventId.
    active = {}
    for node in ET.parse(path).getroot().findall('Event'):
        identifier = str(node.get('EventId'))
        if node.find('Delete') is not None:
            active.pop(identifier, None)
        else:
            active[identifier] = node
    records, goals, kickoffs = [], [], []
    for identifier, node in active.items():
        timestamp = pd.to_datetime(node.get('EventTime'), utc=True)
        play = node.find('.//Play')
        shot = node.find('.//ShotAtGoal')
        claim = node.find('.//BallClaiming')
        tackle = node.find('.//TacklingGame')
        foul = node.find('.//Foul')
        restart = next((label for tag, label in [
            ('ThrowIn', 'throw_in'), ('CornerKick', 'corner'),
            ('FreeKick', 'free_kick'), ('GoalKick', 'goal_kick'),
            ('KickOff', 'kick_off'),
        ] if node.find('.//' + tag) is not None), None)
        actor = team = receiver = None
        outcome = pd.NA
        kind = 'unknown'
        if play is not None:
            kind, actor, team = 'pass', play.get('Player'), play.get('Team')
            receiver = play.get('Recipient')
            outcome = {'successfullyCompleted': True, 'unsuccessful': False}.get(play.get('Evaluation'), pd.NA)
            if restart is None and play.get('FromOpenPlay') == 'true':
                restart = 'open_play'
        elif shot is not None:
            kind, actor, team = 'shot', shot.get('Player'), shot.get('Team')
        elif claim is not None:
            kind = 'interception' if claim.get('Type') == 'InterceptedBall' else 'recovery'
            actor, team = claim.get('Player'), claim.get('Team')
        elif tackle is not None:
            # Contact alone is not an established recovery.
            kind = 'recovery' if (tackle.get('WinnerResult') == 'ballClaimed'
                    and tackle.get('PossessionChange') == 'true') else 'tackle'
            actor, team = tackle.get('Winner'), tackle.get('WinnerTeam')
        elif foul is not None:
            kind, actor, team = 'foul_suffered', foul.get('Fouled'), foul.get('TeamFouled')
        elif node.find('.//FinalWhistle') is not None:
            kind = 'stoppage'
        else:
            contact = node.find('.//OtherBallAction')
            if contact is not None:
                actor, team = contact.get('Player'), contact.get('Team')
        goal = node.find('.//SuccessfulShot')
        if goal is not None:
            goals.append((timestamp, goal.get('CurrentResult')))
        if node.find('.//KickOff') is not None:
            kickoffs.append(timestamp)
        records.append(dict(
            event_id=identifier, event_type=kind, player_id=aliases.get(actor),
            team_id=teams.get(team), receiver_player_id=aliases.get(receiver),
            is_successful=outcome, datetime=timestamp, restart_type=restart or 'unknown',
            original_event=next(iter(node)).tag,
        ))
    # A score becomes available to active play at its following restart;
    # the shot state itself never receives its own future goal result.
    score_updates = []
    for timestamp, score in goals:
        restarts = [time for time in kickoffs if time > timestamp]
        if score and restarts:
            home, away = map(int, score.split(':'))
            score_updates.append((min(restarts), home-away))
    return pd.DataFrame(records), sorted(score_updates)


def align_remaining_events(events, aligned, tracking, *, max_anchor_gap=10.0):
    """Retain DataBallPy links; interpolate only short bracketed clock gaps.

    Unsupported event types are not passed off as spatially synchronized:
    their clock-interpolated link is explicitly identified. No extrapolation
    or interpolation across periods, missing frame ranges or >10 s gaps.
    sync_certainty is DataBallPy's score, not a calibrated probability.
    """
    aligned = aligned.copy()
    aligned['event_id'] = aligned.original_event_id.astype(str).str.replace(r'\.0$', '', regex=True)
    links = aligned.set_index('event_id')
    result = events.copy()
    result['frame'] = result.event_id.map(links.frame).astype('Int64')
    result['period'] = result.event_id.map(links.period_id)
    result['sync_certainty'] = result.event_id.map(links.sync_certainty)
    result['sync_method'] = np.where(result.frame.notna(), 'databallpy', 'unmatched')
    frame_to_row = tracking.set_index('frame')
    for period, frames in tracking.groupby('period'):
        anchors = aligned.loc[aligned.period_id.eq(period) & aligned.frame.notna()].copy()
        anchors = anchors.sort_values('datetime').drop_duplicates('datetime')
        x = pd.to_datetime(anchors.datetime, utc=True).astype('int64').to_numpy()/1e9
        y = anchors.frame.to_numpy(float)
        if len(x) < 2:
            continue
        if np.any(np.diff(y) < 0):
            raise ValueError('Aligned event frames are not monotone within period.')
        lower = pd.to_datetime(frames.datetime, utc=True).min()
        upper = pd.to_datetime(frames.datetime, utc=True).max()
        use = result.frame.isna() & result.datetime.between(lower, upper)
        for index in result.index[use]:
            time = result.loc[index, 'datetime'].value / 1e9
            right = np.searchsorted(x, time, side='right')
            if right == 0 or right == len(x) or x[right]-x[right-1] > max_anchor_gap:
                continue
            frame = int(round(np.interp(time, x[right-1:right+1], y[right-1:right+1])))
            if frame not in frame_to_row.index or frame_to_row.loc[frame, 'period'] != period:
                continue
            result.loc[index, ['frame', 'period', 'sync_method']] = [frame, period, 'interpolated_clock']
    result['outcome_verified'] = result.is_successful.notna()
    return result


def prepare_idsse(data_dir, match_id, output_dir, *, sync_python=None,
                  synchronize=False, max_anchor_gap=10.0, velocity_window_seconds=0.5):
    """Export a canonical match; raw XML/data paths are supplied by the caller.

    synchronize=True executes the standalone optional bridge. Otherwise its
    alignment outputs must already exist under output_dir/aligned. Position
    derivatives use the same finite-run Savitzky–Golay rule as PRESS's input
    pipeline (quadratic polynomial, no gap/period interpolation).
    """
    output = Path(output_dir)
    aligned_dir = output / 'aligned'
    files = match_files(data_dir, match_id)
    if synchronize:
        if sync_python is None:
            raise ValueError('Supply sync_python for the optional DataBallPy environment.')
        subprocess.run([str(sync_python), str(Path(__file__).with_name('databallpy_bridge.py')),
                        str(data_dir), str(match_id), str(aligned_dir)], check=True)
    report = json.loads((aligned_dir / 'alignment.json').read_text())
    roster, teams, metadata = read_metadata(files['info'], match_id)
    metadata['fps'] = report['fps']
    tracking = pd.read_parquet(aligned_dir / 'tracking.parquet').rename(columns={'period_id': 'period'})
    # Exclude the non-playing halftime/dead-period placeholder. Ordinary
    # stopped-clock rows within either half remain, with no claimed control.
    tracking = tracking.loc[tracking.period.isin([1, 2])].copy().reset_index(drop=True)
    roster = roster.loc[roster.player_id.map(lambda p: p+'_x' in tracking)].copy()
    window = max(3, int(round(velocity_window_seconds*metadata['fps'])))
    window += 1-window % 2
    derivatives = {}
    for player in roster.player_id:
        for axis in ['x', 'y']:
            derivatives[f'{player}_v{axis}'] = differentiate(
                tracking[f'{player}_{axis}'], tracking.frame, tracking.period,
                window=window, fps=metadata['fps'],
            )
    tracking = tracking.drop(columns=list(derivatives), errors='ignore')
    tracking = pd.concat([tracking, pd.DataFrame(derivatives)], axis=1)
    # Finite positions in isolated <3-frame runs have no inferable velocity;
    # mark those player observations as missing; do not replace them with zero.
    for player in roster.player_id:
        valid = np.isfinite(tracking[[f'{player}_vx', f'{player}_vy']]).all(axis=1)
        tracking.loc[~valid, [f'{player}_x', f'{player}_y']] = np.nan
    observed_carrier = tracking.player_possession.map(lambda p: isinstance(p, str) and p in set(roster.player_id))
    tracking.loc[~observed_carrier, 'player_possession'] = None
    for player in roster.player_id:
        invalid = tracking.player_possession.eq(player) & tracking[f'{player}_x'].isna()
        tracking.loc[invalid, 'player_possession'] = None
    tracking['attacking_side'] = np.where(tracking.player_possession.str.startswith('home', na=False),
                                        'left_to_right', 'right_to_left')
    tracking.loc[tracking.player_possession.isna(), 'attacking_side'] = None
    raw_events, scores = read_event_attributes(files['events'], roster, teams)
    events = align_remaining_events(raw_events, pd.read_parquet(aligned_dir/'aligned_events.parquet'),
                                    tracking, max_anchor_gap=max_anchor_gap)
    tracking['match_clock_min'] = np.nan
    for period, group in tracking.groupby('period'):
        tracking.loc[group.index, 'match_clock_min'] = (
            pd.to_datetime(group.datetime, utc=True)-pd.to_datetime(group.datetime, utc=True).min()
        ).dt.total_seconds()/60 + (period-1)*45
    score = np.zeros(len(tracking), dtype=int)
    for timestamp, difference in scores:
        restart = events.loc[events.datetime.eq(timestamp) & events.frame.notna()]
        if len(restart):
            score[tracking.frame.ge(int(restart.frame.iloc[0]))] = difference
    tracking['score_difference_prefix'] = np.where(tracking.player_possession.str.startswith('home', na=False), score, -score)
    tracking['match_id'] = str(match_id)
    for key in ['pitch_length', 'pitch_width']:
        tracking[key] = metadata[key]
    output.mkdir(parents=True, exist_ok=True)
    tracking.to_parquet(output/'tracking.parquet', index=False)
    events.to_parquet(output/'events.parquet', index=False)
    roster.to_parquet(output/'players.parquet', index=False)
    metadata.update(control_source='DataBallPy estimated; not annotated possession ground truth',
                    synchronization=report, velocity_window_frames=window,
                    max_clock_interpolation_gap_seconds=max_anchor_gap)
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2))
    return tracking, events, roster, metadata
