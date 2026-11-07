"""Native tracking, offline kinematics, state annotation and canonical segmentation.

Task-specific outcome filters are applied by the corresponding task library."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from .possession_selection import matched_individual_mask, matched_pass_pairs, annotate_matched_pass_release_frames
FPS = 10
POLICY = 'one_second_actions_with_exposure_tails_v2'
SCHEMA_VERSION = 'press_base_v4'
TRAINING_COHORT_POLICY = 'outfield_carriers_no_shot_v1'


def annotate_training_cohort(table, players, *, canonical_players=False):
    """Flag carrier eligibility while retaining all intervals for inference.

    Only the carrier role is filtered: goalkeepers remain in tracking and in
    defensive/receiving states. Unknown roles fail rather than default outfield.
    """
    result = table.copy()
    key = 'player_possession' if canonical_players else 'player_id'
    roles = players.set_index(key).is_gk
    result['carrier_is_goalkeeper'] = result.player_id.map(roles)
    if result.carrier_is_goalkeeper.isna().any():
        raise ValueError('Training carrier has no verified goalkeeper/outfield role.')
    result['carrier_is_goalkeeper'] = result.carrier_is_goalkeeper.astype(bool)
    shot = result.end_type.eq('shot')
    keeper = result.carrier_is_goalkeeper
    result['training_cohort_eligible'] = ~shot & ~keeper
    result['training_cohort_exclusion_reason'] = np.select([shot & keeper, shot, keeper],
        ['shot_ending;goalkeeper_carrier', 'shot_ending', 'goalkeeper_carrier'],
        default='')
    result['training_cohort_policy'] = TRAINING_COHORT_POLICY
    return result


def differentiate(values, frames, periods, *, window, fps=FPS):
    """Differentiate finite contiguous runs, never across a gap or period.

    A NaN splits the run instead of contaminating its full filtering window.
    No interpolation is applied. Runs shorter than three samples stay NaN.
    """
    values = np.asarray(values, dtype=float)
    frames = np.asarray(frames)
    periods = np.asarray(periods)
    result = np.full(len(values), np.nan)
    valid = np.isfinite(values) & pd.notna(periods)
    breaks = np.ones(len(values), dtype=bool)
    if len(values) > 1:
        breaks[1:] = (np.diff(frames) != 1) | (periods[1:] != periods[:-1]) | ~valid[1:] | ~valid[:-1]
    starts = np.flatnonzero(breaks)
    for start, stop in zip(starts, np.r_[starts[1:], len(values)]):
        if not valid[start] or stop - start < 3:
            continue
        size = min(window, stop - start if (stop - start) % 2 else stop - start - 1)
        result[start:stop] = savgol_filter(values[start:stop],
            size,
            min(2, size - 1),
            deriv=1,
            delta=1 / fps,
            mode='interp')
    return result


def add_kinematics(table, axes, *, window, prefix=''):
    """Attach velocity and acceleration with explicit units m/s and m/s²."""
    result = table.copy()
    velocities, accelerations = ([], [])
    for axis in axes:
        velocity = f'{prefix}v{axis}'
        acceleration = f'{prefix}a{axis}'
        result[velocity] = differentiate(result[f'{prefix}{axis}'],
            result.frame,
            result.period,
            window=window)
        result[acceleration] = differentiate(result[velocity], result.frame, result.period, window=window)
        velocities.append(velocity)
        accelerations.append(acceleration)
    result[f'{prefix}v'] = np.linalg.norm(result[velocities].to_numpy(), axis=1)
    result[f'{prefix}a'] = np.linalg.norm(result[accelerations].to_numpy(), axis=1)
    return result


def load_raw_tracking(match_dir):
    """Read JSONL once, preserve empty frames, provider carrier and detections."""
    match_dir = Path(match_dir)
    match_id = match_dir.name
    metadata = json.loads((match_dir / f'{match_id}_match.json').read_text())
    players = pd.DataFrame([{'player_id': player['id'], 'team_id': player['team_id'], 'number': player['number'], 'short_name': player['short_name'], 'is_gk': player['player_role']['acronym'] == 'GK', 'player_possession': f"{('home' if player['team_id'] == metadata['home_team']['id'] else 'away')}_{player['number']}"} for player in metadata['players']])
    if players.player_id.duplicated().any() or players.player_possession.duplicated().any():
        raise ValueError('Player IDs and side/jersey labels must be unique within a match.')
    player_map = players.set_index('player_id').player_possession.to_dict()
    frame_records, player_records = ([], [])
    with (match_dir / f'{match_id}_tracking_extrapolated.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            ball, possession = (row.get('ball_data') or {}, row.get('possession') or {})
            frame_records.append((row['frame'], row.get('timestamp'), row.get('period'), ball.get('x'), ball.get('y'), ball.get('z'), ball.get('is_detected'), possession.get('player_id'), possession.get('group')))
            for player in row.get('player_data', []):
                player_records.append((row['frame'], row.get('period'), player['player_id'], player.get('x'), player.get('y'), player.get('is_detected')))
    frames = pd.DataFrame(frame_records,
        columns=['frame', 'timestamp', 'period', 'ball_x', 'ball_y', 'ball_z', 'is_detected_ball', 'provider_player_id', 'provider_possession_group']).sort_values('frame').reset_index(drop=True)
    del frame_records
    if frames.frame.duplicated().any():
        raise ValueError('Duplicate raw tracking frames.')
    frames = add_kinematics(frames, ['x', 'y', 'z'], window=5, prefix='ball_')
    frames['is_detected_ball'] = frames.is_detected_ball.astype('boolean')
    frames['player_possession_provider'] = frames.provider_player_id.map(player_map)
    frames['match_id'] = match_id
    frames['pitch_length'] = float(metadata['pitch_length'])
    frames['pitch_width'] = float(metadata['pitch_width'])
    if frames.pitch_length.iloc[0] <= 0 or frames.pitch_width.iloc[0] <= 0:
        raise ValueError('Pitch dimensions must be positive.')
    long = pd.DataFrame(player_records, columns=['frame', 'period', 'player_id', 'x', 'y', 'is_detected'])
    del player_records
    if long.duplicated(['frame', 'player_id']).any():
        raise ValueError('Duplicate player observations.')
    unknown = set(long.player_id.unique()) - set(player_map)
    if unknown:
        raise ValueError(f'Tracking contains unknown player IDs: {unknown}')
    parts = []
    for player_id, group in long.groupby('player_id', sort=True):
        group = add_kinematics(group.sort_values('frame'), ['x', 'y'], window=7)
        group['player_label'] = player_map[player_id]
        parts.append(group)
    del long
    long = pd.concat(parts, ignore_index=True)
    fields = ['x', 'y', 'vx', 'vy', 'v', 'ax', 'ay', 'a', 'is_detected']
    wide = long.pivot(index='frame', columns='player_label', values=fields)
    wide.columns = [f'{player}_{field}' for field, player in wide.columns]
    frames = frames.merge(wide.reset_index(), on='frame', how='left', validate='one_to_one')
    for column in frames:
        if column.endswith('_is_detected'):
            frames[column] = frames[column].astype('boolean')
        elif column.startswith(('home_', 'away_')):
            frames[column] = pd.to_numeric(frames[column], errors='raise').astype(float)
    return (frames, players, metadata)


def annotate_tracking(tracking, intervals, raw_possessions, player_map):
    """Assign disjoint inclusive individual intervals, then valid pass releases."""
    result = tracking.copy()
    result['player_possession'] = None
    result['attacking_side'] = None
    result['possession_source'] = None
    ordered = intervals.sort_values('frame_start')
    if len(ordered) > 1 and (ordered.frame_start.to_numpy()[1:] <= ordered.frame_end.to_numpy()[:-1]).any():
        raise ValueError('Individual intervals overlap.')
    if not ordered.empty:
        lookup = np.searchsorted(ordered.frame_start.to_numpy(), result.frame.to_numpy(), side='right') - 1
        selected = lookup >= 0
        selected[selected] &= result.frame.to_numpy()[selected] <= ordered.frame_end.to_numpy()[lookup[selected]]
        for column in ('player_id', 'attacking_side'):
            destination = 'player_possession' if column == 'player_id' else column
            result.loc[selected, destination] = ordered[column].to_numpy()[lookup[selected]]
        result.loc[selected, 'possession_source'] = 'matched_individual'
    return annotate_matched_pass_release_frames(result, raw_possessions, player_map)


def annotate_state_quality(tracking):
    """Audit usable carrier/state without removing extrapolated observations."""
    result = tracking.copy()
    carrier_ok = np.zeros(len(result), dtype=bool)
    counts = {side: np.zeros(len(result), dtype=int) for side in ('home', 'away')}
    for column in result.columns:
        if not (column.startswith(('home_', 'away_')) and column.endswith('_x')):
            continue
        label = column[:-2]
        finite = np.isfinite(result[[f'{label}_{axis}' for axis in ('x', 'y', 'vx', 'vy')]].to_numpy(dtype=float)).all(axis=1)
        counts[label.split('_')[0]] += finite
        carrier_ok |= result.player_possession.eq(label).to_numpy() & finite
    for side, count in counts.items():
        result[f'n_{side}_usable'] = count
    result['state_eligible'] = carrier_ok & result.attacking_side.isin(['left_to_right', 'right_to_left']) & (counts['home'] > 0) & (counts['away'] > 0)
    result['state_exclusion_reason'] = np.select([result.player_possession.isna(), ~carrier_ok, ~result.attacking_side.isin(['left_to_right', 'right_to_left']), (counts['home'] == 0) | (counts['away'] == 0)],
        ['no_selected_carrier', 'missing_carrier_kinematics', 'unknown_direction', 'empty_team_state'],
        default='')
    return result


def segment_possessions(intervals, *, fps=FPS, horizon_frames=10):
    """Canonical one-second attempts and passes; separate evaluation snapshots.

    Failed attempts remain canonical regardless of observed duration.
    Non-loss tails are exposure observations, never action-choice examples.
    Full preceding drives are retained for shot/foul/clearance endings too.
    """
    from PRESS.utils.match import standardize_action_context
    records = []
    for row in intervals.itertuples(index=False):
        if row.end_type not in {'pass', 'clearance', 'possession_loss', 'foul_suffered', 'shot'}:
            continue
        duration = row.frame_end - row.frame_start
        if duration < 0:
            raise ValueError('Negative possession duration.')
        previous = row.start_type
        blocks = []
        if row.end_type == 'possession_loss':
            if duration < horizon_frames:
                blocks = [(row.frame_start, row.frame_end, 0)]
            else:
                full, remainder = divmod(duration, horizon_frames)
                blocks = [(row.frame_start + i * horizon_frames, row.frame_start + (i + 1) * horizon_frames, 0 if i == full - 1 and remainder == 0 else 1) for i in range(full)]
                if remainder:
                    blocks.append((row.frame_start + full * horizon_frames, row.frame_end, 0))
        else:
            blocks = [(start, start + horizon_frames, 1) for start in range(row.frame_start, row.frame_end - horizon_frames + 1, horizon_frames)]
        for start, end, outcome in blocks:
            records.append({'match_id': str(row.match_id), 'possession_id': row.possession_id, 'player_id': row.player_id, 'frame_start': start, 'frame_end': end, 'duration_frames': end - start, 'duration': (end - start) / fps, 'action': 'ball_drive', 'previous_action': previous, 'start_type': row.start_type, 'end_type': row.end_type, 'outcome': outcome, 'game_interruption_after': row.game_interruption_after, 'time_since_start': (start - row.frame_start) / fps})
            previous = 'ball_drive'
        if row.end_type == 'pass':
            records.append({'match_id': str(row.match_id), 'possession_id': row.possession_id, 'player_id': row.player_id, 'frame_start': row.frame_end, 'frame_end': row.frame_end, 'duration_frames': 0, 'duration': 0.0, 'action': 'pass', 'previous_action': previous, 'start_type': row.start_type, 'end_type': row.end_type, 'outcome': int(row.pass_outcome == 'successful'), 'game_interruption_after': row.game_interruption_after, 'time_since_start': duration / fps})
    columns = ['match_id',
        'possession_id',
        'player_id',
        'frame_start',
        'frame_end',
        'duration_frames',
        'duration',
        'action',
        'previous_action',
        'start_type',
        'end_type',
        'outcome',
        'game_interruption_after',
        'time_since_start']
    result = pd.DataFrame(records, columns=columns)
    result['action_id'] = np.arange(1, len(result) + 1)
    result['segmentation_policy'] = POLICY
    result['is_canonical'] = True
    result['snapshot_reason'] = None
    result['forecast_horizon'] = np.where(result.action.eq('ball_drive'), 1.0, np.nan)
    extras = []
    for row in intervals.itertuples(index=False):
        if row.end_type not in {'pass', 'shot', 'clearance', 'foul_suffered'}:
            continue
        duration = row.frame_end - row.frame_start
        full, remainder = divmod(duration, horizon_frames)
        common = dict(match_id=str(row.match_id),
            possession_id=row.possession_id,
            player_id=row.player_id,
            action='snapshot',
            start_type=row.start_type,
            end_type=row.end_type,
            outcome=1,
            game_interruption_after=row.game_interruption_after,
            is_canonical=False,
            segmentation_policy=POLICY,
            forecast_horizon=1.0)
        if remainder:
            start = row.frame_start + full * horizon_frames
            extras.append(dict(common, frame_start=start, frame_end=row.frame_end, duration_frames=remainder, duration=remainder / fps, time_since_start=(start - row.frame_start) / fps, previous_action='ball_drive' if full else row.start_type, snapshot_reason='short_control_tail'))
        if row.end_type != 'pass':
            extras.append(dict(common, frame_start=row.frame_end, frame_end=row.frame_end, duration_frames=0, duration=0.0, time_since_start=duration / fps, previous_action='ball_drive' if full else row.start_type, snapshot_reason='terminal_' + row.end_type))
    if extras:
        snapshots = pd.DataFrame(extras)
        snapshots['action_id'] = np.arange(len(result) + 1, len(result) + len(snapshots) + 1)
        result = pd.concat([result, snapshots], ignore_index=True)
    result = result.sort_values(['frame_start', 'frame_end', 'action_id'],
        kind='stable').reset_index(drop=True)
    result['start_type_std'] = result.start_type.map(standardize_action_context)
    result['previous_action_std'] = result.previous_action.map(lambda value: standardize_action_context(value, previous=True))
    return result


def drive_observations(actions):
    """Observed exposure for rate training; distinct from canonical choice rows.

    Failure time is interval-observed at 10Hz. A zero-frame failure is kept
    in the first 0.1s bin, explicitly flagged rather than dropped.
    """
    rows = actions.loc[actions.action.eq('ball_drive') | actions.snapshot_reason.eq('short_control_tail')].copy()
    rows['action'] = 'ball_drive'
    rows['observed_duration'] = rows.duration
    rows['loss_observed'] = rows.outcome.eq(0)
    rows['target_status'] = np.select([rows.loss_observed, ~rows.is_canonical],
        ['failure_interval', 'censored'],
        default='survived')
    rows['observation_upper'] = np.where(rows.loss_observed & rows.duration.eq(0), 1 / FPS, rows.duration)
    rows['observation_lower'] = np.where(rows.loss_observed,
        np.maximum(rows.observation_upper - 1 / FPS, 0),
        rows.observation_upper)
    rows['zero_frame_failure'] = rows.loss_observed & rows.duration.eq(0)
    rows['horizon_label_observed'] = rows.loss_observed | rows.duration.ge(1.0)
    rows['stop_reason'] = np.where(rows.loss_observed,
        'possession_loss',
        np.where(rows.is_canonical, 'one_second_completed', rows.end_type))
    rows['drive_observation_id'] = np.arange(1, len(rows) + 1)
    return rows


def pass_curvature(row, tracking):
    """Maximum lateral deviation (% of endpoint distance), requiring full coverage."""
    start, end = (int(row.frame_end), int(row.frame_reception))
    trajectory = tracking.loc[start:end, ['ball_x', 'ball_y']]
    if len(trajectory) != end - start + 1 or not np.isfinite(trajectory.to_numpy()).all():
        return np.nan
    points = trajectory.to_numpy(dtype=float)
    line = np.array([row.x_target, row.y_target]) - points[0]
    length = np.linalg.norm(line)
    if length == 0:
        return np.nan
    cross = np.abs(line[0] * (points[:, 1] - points[0, 1]) - line[1] * (points[:, 0] - points[0, 0]))
    return float(cross.max() / length ** 2 * 100)
