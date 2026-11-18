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
from .idsse_events import active_event_nodes, read_provider_events


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
    """Map explicit provider semantics to canonical IDs, retaining provenance.

    Won tackles retain the winner and loser, independently of subsequent
    control. BallClaiming starts are generic recoveries unless Type explicitly
    says InterceptedBall. Duplicate foul decisions and handballs remain in the
    event audit with exclusion markers; they cannot generate a second target.
    """
    aliases = players.set_index('source_player_id').player_id.to_dict()
    records = pd.DataFrame(read_provider_events(path))
    for column in ['player_id', 'receiver_player_id', 'winner_player_id',
                   'loser_player_id', 'fouler_player_id']:
        records[column] = records[column].map(aliases)
    records['team_id'] = records.team_id.map(teams)
    records['datetime'] = pd.to_datetime(records.datetime, utc=True)
    records['is_successful'] = records.is_successful.astype('boolean')
    goals, kickoffs = [], []
    for node in active_event_nodes(path).values():
        timestamp = pd.to_datetime(node.get('EventTime'), utc=True)
        goal = node.find('.//SuccessfulShot')
        if goal is not None:
            goals.append((timestamp, goal.get('CurrentResult')))
        if node.find('.//KickOff') is not None:
            kickoffs.append(timestamp)
    # A confirmed goal enters score context only at the following restart.
    score_updates = []
    for timestamp, score in goals:
        restarts = [time for time in kickoffs if time > timestamp]
        if score and restarts:
            home, away = map(int, score.split(':'))
            score_updates.append((min(restarts), home-away))
    return records, sorted(score_updates)


def align_remaining_events(events, aligned, tracking, *, max_anchor_gap=10.0):
    """Retain DataBallPy links; interpolate only short bracketed clock gaps.

    Contact events require a direct synchronizer link. Other event types
    may receive an explicitly identified clock-interpolated link for audit. No extrapolation
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
    eligible = result.get('duplicate_of', pd.Series(None, index=result.index)).isna() & result.get(
        'excluded_reason', pd.Series(None, index=result.index)).isna()
    result.loc[~eligible, 'frame'] = pd.NA
    result.loc[~eligible, 'sync_method'] = 'excluded'
    # Clock interpolation is audit information, not a verified contact frame.
    contact_types = {'pass', 'shot', 'tackle', 'recovery', 'interception',
                     'reception', 'foul_suffered', 'clearance'}
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
        use = (eligible & result.frame.isna() & ~result.event_type.isin(contact_types)
               & result.datetime.between(lower, upper))
        for index in result.index[use]:
            time = result.loc[index, 'datetime'].value / 1e9
            right = np.searchsorted(x, time, side='right')
            if right == 0 or right == len(x) or x[right]-x[right-1] > max_anchor_gap:
                continue
            frame = int(round(np.interp(time, x[right-1:right+1], y[right-1:right+1])))
            if frame not in frame_to_row.index or frame_to_row.loc[frame, 'period'] != period:
                continue
            result.loc[index, ['frame', 'period', 'sync_method']] = [frame, period, 'interpolated_clock']
    result['outcome_verified'] = (eligible & result.frame.notna()
        & (result.is_successful.notna() | result.event_type.isin(['foul_suffered', 'clearance'])))
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
    tracking['home_score_difference_prefix'] = score
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
