"""Original SkillCorner files -> canonical match tables, in memory.

Read load_raw_tracking, annotate_possession and load_skillcorner in order.
The numerical movement convention and provider boundary/link checks are shared
with preprocessing. No model fitting or task-specific cohort filtering occurs.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .kinematics import differentiate
from .segmentation import canonical_actions

START_MATCHED = 'is_player_possession_start_matched'
END_MATCHED = 'is_player_possession_end_matched'


def is_matched(value) -> bool:
    """Accept an explicit true flag; missing values and strings are unmatched.

    Numeric 1 is accepted for CSV boolean columns. In particular, NaN
    must not pass through Python's bool(NaN) == True conversion.
    """
    return False if pd.isna(value) else bool(value == True)


def matched_individual_mask(possessions: pd.DataFrame) -> pd.Series:
    """Select individual intervals only when both boundary flags are true.

    This mask does not select pass examples: those require the sender's
    end and the immediate next possession's start, without requiring the
    sender's start or the receiver's end. Input is not modified.
    """
    missing = {START_MATCHED, END_MATCHED} - set(possessions.columns)
    if missing:
        raise ValueError(f'Missing possession boundary flags: {sorted(missing)}')
    return (possessions[START_MATCHED].eq(True) & possessions[END_MATCHED].eq(True)).fillna(False).astype(bool)


def pass_receipt_link_reason(sender, receiver):
    """Explain why the next possession cannot identify this pass's endpoint.

    Optional fields are checked when available. A completed pass's intended
    recipient must be the next carrier; a failed pass must lead to an opponent.
    No outcome is relabelled and no later possession is substituted.
    """
    if pd.notna(sender.get('game_interruption_after')) or pd.notna(receiver.get('game_interruption_before')):
        return 'annotated_interruption'
    outcome = sender.get('pass_outcome')
    if outcome not in ('successful', 'unsuccessful'):
        return ''
    if 'team_id' in sender and 'team_id' in receiver:
        if pd.isna(sender.team_id) or pd.isna(receiver.team_id):
            return 'missing_receipt_team'
        if (sender.team_id == receiver.team_id) != (outcome == 'successful'):
            return 'outcome_receipt_team_mismatch'
    if outcome == 'successful':
        if 'player_targeted_id' in sender and 'player_id' in receiver:
            if pd.isna(sender.player_targeted_id) or sender.player_targeted_id != receiver.player_id:
                return 'completed_pass_receipt_identity_mismatch'
        if 'start_type' in receiver and receiver.start_type != 'pass_reception':
            return 'completed_pass_without_direct_reception'
        if 'is_previous_pass_matched' in receiver and (not is_matched(receiver.is_previous_pass_matched)):
            return 'previous_pass_not_matched'
    return ''


def matched_pass_pairs(events: pd.DataFrame) -> pd.DataFrame:
    """Pair matched pass release with the immediate next matched start.

    Pair on the complete player-possession stream, before selecting fully
    matched individual intervals. Never skip an unmatched intermediate
    possession to find a later receiver. Match and period, when present,
    partition the stream. No pass-target coordinates or outcome labels are
    changed by this boundary selection.
    """
    required = {START_MATCHED, END_MATCHED, 'frame_start', 'frame_end', 'end_type'}
    if not required.issubset(events):
        raise ValueError(f'Missing pass boundary columns: {sorted(required - set(events))}')
    rows = events
    if 'event_type' in rows:
        rows = rows.loc[rows.event_type.eq('player_possession')]
    partitions = [column for column in ('match_id', 'period') if column in rows]
    rows = rows.sort_values(partitions + ['frame_start', 'frame_end'], kind='stable').reset_index(drop=True)
    following = rows.groupby(partitions, dropna=False, sort=False).shift(-1) if partitions else rows.shift(-1)
    keep = (rows.end_type.eq('pass') & rows[END_MATCHED].eq(True) & following[START_MATCHED].eq(True)).fillna(False)
    for index in rows.index[keep]:
        if pass_receipt_link_reason(rows.loc[index], following.loc[index]):
            keep.loc[index] = False
    result = rows.loc[keep].copy()
    result['frame_receipt'] = following.loc[keep, 'frame_start']
    for source, destination in (('event_id', 'receipt_event_id'),
        ('player_id', 'receipt_player_id'),
        (END_MATCHED, 'receipt_end_matched')):
        if source in following:
            result[destination] = following.loc[keep, source]
    result['receipt_start_matched'] = True
    return result.reset_index(drop=True)


def annotate_matched_pass_release_frames(tracking, events, player_map):
    """Retain usable release states without inventing an individual interval.

    tracking already has the annotations of fully matched individual
    intervals. Missing carrier/direction at broader pass release points are
    filled from the raw pass event. The provider carrier is preserved in
    player_possession_provider. Existing conflicting annotations raise
    ValueError; missing player mappings or tracking frames remain unfilled.
    This does not reconstruct receivers or alter the pass target.
    """
    result = tracking.copy()
    if 'player_possession_provider' not in result:
        result['player_possession_provider'] = result['player_possession']
    if 'possession_source' not in result:
        result['possession_source'] = pd.Series(None, index=result.index, dtype=object)
        result.loc[result.player_possession.notna(), 'possession_source'] = 'matched_individual'
    for row in matched_pass_pairs(events).itertuples(index=False):
        player = player_map.get(row.player_id)
        if player is None or pd.isna(player):
            continue
        selected = result.frame.eq(row.frame_end)
        existing = result.loc[selected, 'player_possession'].dropna()
        if existing.ne(player).any():
            raise ValueError(f'Conflicting carrier at matched pass release frame {row.frame_end}.')
        unannotated = selected & result.player_possession.isna()
        result.loc[unannotated, 'player_possession'] = player
        result.loc[unannotated, 'attacking_side'] = row.attacking_side
        result.loc[unannotated, 'possession_source'] = 'matched_pass_release'
    return result


def add_kinematics(table, axes, *, window, prefix='', fps=10.0):
    """Attach velocity and acceleration with explicit units m/s and m/s²."""
    result = table.copy()
    velocities, accelerations = ([], [])
    for axis in axes:
        velocity = f'{prefix}v{axis}'
        acceleration = f'{prefix}a{axis}'
        result[velocity] = differentiate(result[f'{prefix}{axis}'],
            result.frame,
            result.period,
            window=window, fps=fps)
        result[acceleration] = differentiate(result[velocity], result.frame, result.period, window=window, fps=fps)
        velocities.append(velocity)
        accelerations.append(acceleration)
    result[f'{prefix}v'] = np.linalg.norm(result[velocities].to_numpy(), axis=1)
    result[f'{prefix}a'] = np.linalg.norm(result[accelerations].to_numpy(), axis=1)
    return result


def load_raw_tracking(match_dir, *, match_id=None, fps=10.0):
    """Read JSONL once, preserve empty frames, provider carrier and detections."""
    match_dir = Path(match_dir)
    match_id = str(match_id) if match_id is not None else match_dir.name
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
    frames = add_kinematics(frames, ['x', 'y', 'z'], window=5, prefix='ball_', fps=fps)
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
        group = add_kinematics(group.sort_values('frame'), ['x', 'y'], window=7, fps=fps)
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


def annotate_possession(poss):
    """
    Annotate one match's individual possessions with both boundaries matched.

    Receipt candidates are taken from the full chronological event stream
    before filtering, so an excluded intermediate interval is never skipped
    when looking for the next receiver. Pass-model examples are selected
    separately using matched release and subsequent possession start.

    Adds:
    - loss information
    - pass information
    - receipt frame
    - xLoss possession annotations
    """
    if 'match_id' in poss and poss.match_id.nunique(dropna=False) > 1:
        raise ValueError('annotate_possession expects one match at a time.')
    poss = poss.sort_values(['frame_start', 'frame_end']).reset_index(drop=True).copy()
    eligible = matched_individual_mask(poss)
    rows = []
    for idx, p in poss.iterrows():
        if not eligible.loc[idx]:
            continue
        player_id = p['player_id']
        end_type = p.get('end_type')
        pass_outcome = p.get('pass_outcome')
        is_loss = end_type == 'pass' and pass_outcome == 'unsuccessful' or end_type in ['possession_loss',
            'clearance']
        is_non_loss = end_type == 'pass' and pass_outcome == 'successful' or end_type in ['shot',
            'foul_suffered']
        if not (is_loss or is_non_loss):
            continue
        if end_type == 'pass' and pass_outcome == 'unsuccessful':
            loss_reason = 'pass_failed'
        elif end_type == 'possession_loss':
            loss_reason = 'possession_loss'
        elif end_type == 'clearance':
            loss_reason = 'clearance'
        else:
            loss_reason = None
        frame_receipt = None
        if end_type == 'pass' and pass_outcome == 'successful' and (idx + 1 < len(poss)):
            next_possession = poss.iloc[idx + 1]
            same_period = 'period' not in poss.columns or (pd.notna(p['period']) and pd.notna(next_possession['period']) and (p['period'] == next_possession['period']))
            if is_matched(next_possession['is_player_possession_start_matched']) and pd.notna(next_possession['frame_start']) and (next_possession['frame_start'] > p['frame_end']) and same_period:
                frame_receipt = int(next_possession['frame_start'])
        target_pass = None
        if end_type == 'pass':
            target_pass = p.get('player_targeted_id')
        rows.append({'possession_id': p['event_id'], 'player_id': player_id, 'frame_start': int(p['frame_start']), 'frame_end': int(p['frame_end']), 'frame_receipt': frame_receipt, 'time_start': p['time_start'], 'time_end': p['time_end'], 'loss': is_loss, 'loss_reason': loss_reason, 'start_type': p['start_type'], 'end_type': end_type, 'pass_outcome': pass_outcome, 'target_pass': target_pass, 'game_interruption_after': p['game_interruption_after'], 'attacking_side': p['attacking_side'], 'is_player_possession_start_matched': True, 'is_player_possession_end_matched': True, 'receipt_start_matched': frame_receipt is not None, **{column: p[column] for column in ('period', 'match_id') if column in poss.columns}})
    columns = ['possession_id',
        'player_id',
        'frame_start',
        'frame_end',
        'frame_receipt',
        'time_start',
        'time_end',
        'loss',
        'loss_reason',
        'start_type',
        'end_type',
        'pass_outcome',
        'target_pass',
        'game_interruption_after',
        'attacking_side',
        'is_player_possession_start_matched',
        'is_player_possession_end_matched',
        'receipt_start_matched'] + [column for column in ('period', 'match_id') if column in poss.columns]
    result = pd.DataFrame(rows, columns=columns)
    result['target_pass'] = result['target_pass'].astype('Int64')
    return result



def annotate_clock_and_score(tracking, events, metadata, *, fps):
    """Attach observed match clock and score effective by the current frame.

    Retains the existing period-clock offset and confirmed score-prefix
    convention. Future goals are not provided as current contextual features.
    """
    result = tracking.copy()
    parents = events.loc[events.event_type.eq('player_possession')].sort_values(
        ['frame_start', 'frame_end'], kind='stable')
    result['match_clock_min'] = np.nan
    result['score_difference_prefix'] = np.nan
    home_id = metadata['home_team']['id']
    for period, group in parents.groupby('period'):
        def seconds(value):
            return sum(float(part)*60**index for index,part in
                       enumerate(reversed(str(value).split(':'))))
        offset = (group.time_end.map(seconds)-group.frame_end/fps).median()
        selected = result.period.eq(period)
        result.loc[selected, 'match_clock_min'] = (result.loc[selected,'frame']/fps+offset)/60
    home = np.where(parents.team_id.eq(home_id), parents.team_score, parents.opponent_team_score)
    away = np.where(parents.team_id.eq(home_id), parents.opponent_team_score, parents.team_score)
    effective = np.where(parents.game_interruption_before.isin(['goal_for','goal_against']),
                         parents.frame_start, parents.frame_end+1)
    order = np.argsort(effective, kind='stable')
    effective = effective[order]
    # Cumulative maxima reproduce the existing score-prefix calculation.
    home = np.maximum.accumulate(np.nan_to_num(home[order], nan=0))
    away = np.maximum.accumulate(np.nan_to_num(away[order], nan=0))
    index = np.searchsorted(effective, result.frame, side='right')-1
    difference = np.zeros(len(result))
    valid = index >= 0
    difference[valid] = home[index[valid]]-away[index[valid]]
    carrier = result.player_possession
    result.loc[carrier.str.startswith('home_', na=False),'score_difference_prefix'] = difference[carrier.str.startswith('home_', na=False)]
    result.loc[carrier.str.startswith('away_', na=False),'score_difference_prefix'] = -difference[carrier.str.startswith('away_', na=False)]
    return result


def canonical_events(raw_possessions, aliases):
    """Complete matched actor events for the core; retain all raw events in audit."""
    end_types = {'pass','shot','clearance','foul_suffered'}
    selected = raw_possessions.loc[
        raw_possessions.is_player_possession_end_matched.eq(True)
        & raw_possessions.end_type.isin(end_types)
    ].copy()
    result = pd.DataFrame(dict(
        event_id=selected.event_id, frame=selected.frame_end,
        player_id=selected.player_id.map(aliases), event_type=selected.end_type,
        receiver_player_id=selected.player_targeted_id.map(aliases),
    ))
    result['team_id'] = result.player_id.str.split('_').str[0]
    result['is_successful'] = selected.pass_outcome.map({'successful':True,'unsuccessful':False})
    passes = result.event_type.eq('pass')
    complete = ~passes | (result.is_successful.notna() & (
        result.is_successful.eq(False) | result.receiver_player_id.notna()))
    result = result.loc[complete & result.player_id.notna() & result.frame.notna()].copy()
    result['frame'] = result.frame.astype(int)
    return result.reset_index(drop=True)


def load_skillcorner(match_dir, *, match_id=None, fps=10.0):
    """Read original metadata JSON, tracking JSONL and dynamic-event CSV.

    Return in-memory constructor inputs plus full source events and an audit
    report. Player velocities use seven-frame quadratic Savitzky–Golay
    derivatives; ball velocities use five, without bridging gaps or periods.
    Possession boundaries and context come from provider annotations. Whole
    one-second actions and failed micro-carries use the shared convention.
    No Parquet input or intermediate on-disk cache is required.
    """
    match_dir = Path(match_dir)
    match_id = str(match_id) if match_id is not None else match_dir.name
    paths = [match_dir/f'{match_id}_match.json',
             match_dir/f'{match_id}_tracking_extrapolated.jsonl',
             match_dir/f'{match_id}_dynamic_events.csv']
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError('Original SkillCorner match files missing: '+', '.join(missing))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be finite and positive.')
    tracking, players, metadata = load_raw_tracking(match_dir, match_id=match_id, fps=fps)
    events = pd.read_csv(paths[2], low_memory=False)
    events['match_id'] = match_id
    raw_parents = events.loc[events.event_type.eq('player_possession')].copy()
    parents = annotate_possession(raw_parents)
    aliases = players.set_index('player_id').player_possession.to_dict()
    parents['provider_player_id'] = parents.player_id
    parents['player_id'] = parents.player_id.map(aliases)
    parents['target_pass'] = parents.target_pass.map(aliases)
    if parents.player_id.isna().any():
        raise ValueError('A SkillCorner event carrier is absent from match metadata.')
    tracking = annotate_tracking(tracking, parents, raw_parents, aliases)
    tracking = annotate_clock_and_score(tracking, events, metadata, fps=fps)
    actions = canonical_actions(parents, fps=fps)
    roster = players.rename(columns={'player_id':'source_player_id',
                                     'team_id':'source_team_id',
                                     'player_possession':'player_id',
                                     'is_gk':'is_goalkeeper'})
    roster['team_id'] = roster.player_id.str.split('_').str[0]
    absent = set(roster.player_id)-{c[:-2] for c in tracking if c.endswith('_x')}
    roster = roster.loc[~roster.player_id.isin(absent)].copy()
    report = dict(provider='SkillCorner', match_id=match_id, fps=float(fps),
                  n_tracking_frames=len(tracking), n_source_events=len(events),
                  n_individual_possessions=len(parents), n_actions=len(actions),
                  player_velocity_window_frames=7, ball_velocity_window_frames=5,
                  boundary_source='provider matched individual possession annotations')
    inputs = dict(tracking_data=tracking,
                  events_data=canonical_events(raw_parents, aliases),
                  player_possessions_data=parents, actions_data=actions,
                  player_metadata=roster, fps=fps, match_id=match_id,
                  pitch_length=metadata['pitch_length'], pitch_width=metadata['pitch_width'])
    return inputs, events, report
