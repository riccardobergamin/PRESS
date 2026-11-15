"""Individual controls and the canonical one-second/pass action convention."""
import numpy as np
import pandas as pd

POLICY = 'one_second_actions_with_exposure_tails_v2'


def canonical_actions(intervals, *, fps=10.0, horizon_frames=None):
    """Canonical one-second attempts and passes; separate evaluation snapshots.

    Failed attempts remain canonical regardless of observed duration.
    Non-loss tails are exposure observations, never action-choice examples.
    Full preceding drives are retained for shot/foul/clearance endings too.
    """
    from PRESS.utils.match import standardize_action_context
    if horizon_frames is None:
        if not np.isclose(fps,round(fps)):
            raise ValueError("Canonical one-second segmentation requires an integer frame frequency or explicit action annotations.")
        horizon_frames=int(round(fps))
    if not np.isfinite(fps) or fps<=0 or horizon_frames<=0:
        raise ValueError("fps and horizon_frames must be positive.")
    records = []
    for row in intervals.itertuples(index=False):
        if row.end_type not in {'pass', 'clearance', 'possession_loss', 'foul_suffered', 'shot', 'unknown'}:
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
            records.append({'match_id': str(row.match_id), 'possession_id': row.possession_id, 'player_id': row.player_id, 'frame_start': row.frame_end, 'frame_end': row.frame_end, 'duration_frames': 0, 'duration': 0.0, 'action': 'pass', 'previous_action': previous, 'start_type': row.start_type, 'end_type': row.end_type, 'outcome': (1 if row.pass_outcome=='successful' else 0 if row.pass_outcome=='unsuccessful' else pd.NA), 'game_interruption_after': row.game_interruption_after, 'time_since_start': duration / fps})
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



def pass_origin_label(restart_type, *, intercepted=False):
    """Use the existing raw context vocabulary, with an honest unknown fallback."""
    prefix = {'open_play': 'pass', 'throw_in': 'throw_in', 'corner': 'corner',
              'free_kick': 'free_kick', 'goal_kick': 'goal_kick'}.get(restart_type)
    suffix = 'interception' if intercepted else 'reception'
    return prefix + '_' + suffix if prefix else 'unknown'


def segment_controls(tracking,events,*,fps,match_id=None):
    """Recover individual controls from synchronized events and carrier states.

    Reuses reception-to-next-actor-action searches. A successful reception
    requires the identified receiver in the observed/estimated control stream;
    the one-frame-before-next-event fallback is deliberately not used.
    Releases terminate that actor's control. A different tracked controller
    bounds a control when no terminal actor event is available; this ending
    remains unknown unless an explicit opponent recovery verifies a loss.
    """
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be finite and positive.')
    required_events = {'frame','event_type','player_id','is_successful','receiver_player_id','sync_matched'}
    if required_events-set(events):
        raise ValueError('Segmentation requires normalized, synchronized events; missing: '+str(sorted(required_events-set(events))))
    required={'frame','player_possession','period'}
    if required-set(tracking):
        raise ValueError(f'Tracking missing segmentation fields: {sorted(required-set(tracking))}')
    e=events.loc[events.sync_matched].sort_values('frame',kind='stable').copy()
    t=tracking[['frame','player_possession','period']].sort_values('frame').copy()
    indexed=t.set_index('frame')
    seeds=[]
    unverified_receivers=0
    terminal_types={'pass','shot','clearance','foul_suffered'}
    for event in e.itertuples():
        if pd.isna(event.player_id):
            continue
        frame = int(event.frame)
        period = indexed.loc[frame, 'period']
        if event.event_type == 'pass' and pd.notna(event.is_successful):
            successful = bool(event.is_successful)
            # Follow only until the next matched pass, shot or stoppage:
            # a later control must not be attributed across another action.
            stops = e.loc[(e.frame > frame) & e.event_type.isin(
                ['pass', 'shot', 'stoppage', 'restart', 'foul_suffered'])]
            limit = int(stops.frame.iloc[0]) if len(stops) else int(t.loc[t.period.eq(period)].frame.max())
            possible = t.loc[t.frame.gt(frame) & t.frame.le(limit) & t.period.eq(period)
                             & t.player_possession.notna() & t.player_possession.ne(event.player_id)]
            if len(possible):
                control = possible.iloc[0]
                same_team = control.player_possession.split('_')[0] == event.player_id.split('_')[0]
                valid_reception = successful and pd.notna(event.receiver_player_id) and control.player_possession == event.receiver_player_id
                valid_interception = not successful and not same_team
                if valid_reception or valid_interception:
                    restart = getattr(event, 'restart_type', 'unknown')
                    label = pass_origin_label(restart, intercepted=valid_interception)
                    seeds.append((int(control.frame), control.player_possession, label, period,
                                  'receiver_control' if valid_reception else 'opponent_control'))
                elif successful:
                    unverified_receivers += 1
            elif successful:
                unverified_receivers += 1
        if event.event_type in {'recovery', 'interception', 'reception'}:
            label = ('pass_interception' if event.event_type == 'interception' else
                     'recovery' if event.event_type == 'recovery' else 'pass_reception')
            observed = indexed.loc[frame, 'player_possession']
            if pd.notna(observed) and observed == event.player_id:
                seeds.append((frame, event.player_id, label, period, 'event_and_control'))
    # A provider may identify control without a corresponding reception event.
    # Retain the interval with unknown origin instead of assigning a label.
    changes=t.player_possession.ne(t.player_possession.shift()) | t.period.ne(t.period.shift()) | t.frame.diff().ne(1)
    for row in t.loc[changes&t.player_possession.notna()].itertuples():
        seeds.append((int(row.frame),row.player_possession,'unknown',row.period,'control_stream'))
    priority = {'receiver_control': 0, 'opponent_control': 0, 'event_and_control': 1, 'control_stream': 2}
    seeds=sorted(set(seeds),key=lambda r:(r[0],priority[r[4]],r[2],r[1]))
    parents=[]
    covered_until=-np.inf
    for start,player,label,period,source in seeds:
        if start<=covered_until:
            continue
        remaining=t.loc[t.frame.ge(start)&t.period.eq(period)]
        if remaining.empty:
            continue
        actor_events=e.loc[e.frame.ge(start)&e.player_id.eq(player)&e.event_type.isin(terminal_types)]
        actor_events=actor_events.loc[actor_events.frame.isin(remaining.frame)]
        terminal=actor_events.iloc[0] if len(actor_events) else None
        different=remaining.loc[remaining.player_possession.notna()&remaining.player_possession.ne(player)]
        change=int(different.frame.iloc[0]) if len(different) else int(remaining.frame.max())+1
        # Never bridge missing frames or a period boundary in an interval.
        consecutive=remaining.frame.diff().fillna(1).eq(1)
        gaps=remaining.loc[~consecutive]
        cutoff=min(change,int(gaps.frame.iloc[0]) if len(gaps) else int(remaining.frame.max())+1)
        stops=e.loc[e.frame.ge(start)&e.frame.lt(cutoff)&e.event_type.isin(['stoppage','restart'])]
        if len(stops):
            cutoff=int(stops.frame.iloc[0])
        if cutoff<=start:
            continue
        verified=False
        outcome='unknown'
        receiver=None
        receipt=np.nan
        loss=pd.NA
        ending='unknown'
        interruption=None
        if terminal is not None and int(terminal.frame)<cutoff:
            end=int(terminal.frame)
            ending=terminal.event_type
            if ending=='pass':
                value=terminal.is_successful
                outcome='successful' if pd.notna(value) and bool(value) else 'unsuccessful' if pd.notna(value) else 'unknown'
                loss=not bool(value) if pd.notna(value) else pd.NA
                receiver=terminal.receiver_player_id
                after=t.loc[t.frame.gt(end)&t.period.eq(period)&t.player_possession.notna()&t.player_possession.ne(player)]
                if len(after):
                    next_control=after.iloc[0]
                    if outcome=='successful' and pd.notna(receiver) and next_control.player_possession==receiver:
                        receipt=int(next_control.frame)
                        verified=True
                    elif outcome=='unsuccessful' and next_control.player_possession.split('_')[0]!=player.split('_')[0]:
                        receipt=int(next_control.frame)
                        verified=True
                # An annotated actor is legitimate at its matched release frame.
                t.loc[t.frame.eq(end),'player_possession']=player
            else:
                loss=False
                verified=True
                interruption=ending if ending in {'shot','foul_suffered'} else None
        else:
            end=max(start,cutoff-1)
            if cutoff==change:
                recoveries=e.loc[e.frame.eq(change)&e.event_type.isin(['recovery','interception'])]
                if len(recoveries) and pd.notna(recoveries.iloc[0].player_id) and recoveries.iloc[0].player_id.split('_')[0]!=player.split('_')[0]:
                    ending='possession_loss'
                    loss=True
                    verified=True
        if end<start:
            continue
        parents.append(dict(match_id=str(match_id) if match_id is not None else '',possession_id=len(parents)+1,
            player_id=player,team_id=player.split('_')[0],frame_start=start,frame_end=end,period=period,
            start_type=label,end_type=ending,pass_outcome=outcome,target_pass=receiver,frame_receipt=receipt,
            loss=loss,outcome_verified=verified,boundary_source=source,game_interruption_after=interruption,
            is_player_possession_start_matched=True,is_player_possession_end_matched=ending!='unknown'))
        covered_until=end
    result=pd.DataFrame(parents)
    if result.empty:
        raise ValueError('No control interval can be reconstructed from these matched events and carrier states.')
    result['loss']=result.loss.astype('boolean')
    actions=canonical_actions(result,fps=fps)
    actions['outcome_verified']=actions.possession_id.map(result.set_index('possession_id').outcome_verified)
    updated_tracking=tracking.copy()
    updated_tracking['player_possession']=updated_tracking.frame.map(t.set_index('frame').player_possession)
    return updated_tracking,result,actions,dict(n_individual_possessions=len(result),n_actions=len(actions),
        n_unverified_outcomes=int((~result.outcome_verified).sum()),unverified_successful_receptions=unverified_receivers,
        policy=POLICY,control_source='supplied control stream; may be tracking-estimated',
        one_frame_reception_fallback=False)



def segment_possessions(tracking, events, *, fps, match_id=None):
    """Build individual possessions and actions from aligned external data.

    Inputs are canonical tracking and event tables already mapped/synchronized
    by the caller. No files are loaded and no time alignment is inferred.
    Returns (tracking, individual_possessions, actions, report). Supply those
    tables explicitly to PRESS(...). Unknown context/outcomes remain unknown.
    """
    from .events import normalize_events
    from .synchronization import synchronize_events
    if not isinstance(tracking, pd.DataFrame):
        raise TypeError('tracking must be an already prepared DataFrame, not a file path.')
    events = normalize_events(events)
    events = synchronize_events(events, tracking, fps=fps, already_synchronized=True)
    tracking, parents, actions, report = segment_controls(tracking, events, fps=fps, match_id=match_id)
    report.update(n_events=len(events), n_matched_events=int(events.sync_matched.sum()))
    return tracking, parents, actions, report
