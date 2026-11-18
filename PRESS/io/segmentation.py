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


def segment_controls(tracking, events, *, fps, match_id=None,
                     include_unannotated_starts=True,
                     reception_fallback_max_distance=1.5):
    """Recover individual controls from synchronized events and carrier states.

    Reuses reception-to-next-actor-action searches. A successful reception
    requires the identified receiver in the observed/estimated control stream;
    a missed control may use the frame before the receiver's next event only
    when finite ball/player XY positions confirm proximity at that frame.
    Matched terminal events take precedence over the estimated controller.
    A won tackle terminates the current control as a loss when its identified
    winner differs from the carrier, regardless of the next controller or
    PossessionChange. The carrier stream bounds otherwise unknown endings.
    """
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be finite and positive.')
    if (reception_fallback_max_distance is not None and
            (not np.isfinite(reception_fallback_max_distance) or reception_fallback_max_distance < 0)):
        raise ValueError('reception_fallback_max_distance must be nonnegative metres or None.')
    required_events = {'frame','event_type','player_id','is_successful','receiver_player_id','sync_matched'}
    if required_events-set(events):
        raise ValueError('Segmentation requires normalized, synchronized events; missing: '+str(sorted(required_events-set(events))))
    required={'frame','player_possession','period'}
    if required-set(tracking):
        raise ValueError(f'Tracking missing segmentation fields: {sorted(required-set(tracking))}')
    eligible = events.sync_matched.copy()
    if 'sync_method' in events:
        # Clock-only links are audit information, not observed actor contacts.
        eligible &= ~events.sync_method.isin(['interpolated_clock', 'unmatched', 'excluded'])
    for column in ['duplicate_of', 'excluded_reason']:
        if column in events:
            eligible &= events[column].isna()
    e=events.loc[eligible].sort_values('frame',kind='stable').copy()
    if 'winner_player_id' not in e:
        e['winner_player_id'] = None
    # A provider may supply the winner explicitly, or identify a successful
    # tackle's actor. An attempted/unresolved tackle is not a verified loss.
    won = e.event_type.eq('tackle') & e.is_successful.eq(True).fillna(False)
    e.loc[won & e.winner_player_id.isna(), 'winner_player_id'] = e.loc[
        won & e.winner_player_id.isna(), 'player_id']
    t=tracking[['frame','player_possession','period']].sort_values('frame').copy()
    indexed=t.set_index('frame')
    coordinates=tracking.set_index('frame')
    seeds=[]
    seed_events={}
    fallback_support={}
    fallback_checks=[]
    unverified_receivers=0
    terminal_types={'pass','shot','clearance','foul_suffered'}
    for event in e.itertuples():
        if pd.isna(event.player_id):
            continue
        frame = int(event.frame)
        period = indexed.loc[frame, 'period']
        if (event.event_type == 'pass' and pd.notna(event.is_successful)
                and (bool(event.is_successful) or include_unannotated_starts)):
            successful = bool(event.is_successful)
            if successful:
                # Original reception search: look for this identified receiver
                # up to its next synchronized actor event, within the period.
                receiver = event.receiver_player_id
                following = e.loc[e.frame.gt(frame) & e.player_id.eq(receiver)
                                  & e.frame.isin(t.loc[t.period.eq(period), 'frame'])]
                if pd.isna(receiver) or (following.empty and not include_unannotated_starts):
                    unverified_receivers += 1
                    continue
                limit = int(following.frame.iloc[0]) if len(following) else int(t.loc[t.period.eq(period)].frame.max())
                possible = t.loc[t.frame.gt(frame) & t.frame.le(limit) & t.period.eq(period)
                                 & t.player_possession.eq(receiver)]
            else:
                stops = e.loc[e.frame.gt(frame) & e.event_type.isin(
                    ['pass', 'shot', 'stoppage', 'restart', 'foul_suffered'])]
                limit = int(stops.frame.iloc[0]) if len(stops) else int(t.loc[t.period.eq(period)].frame.max())
                possible = t.loc[t.frame.gt(frame) & t.frame.le(limit) & t.period.eq(period)
                                 & t.player_possession.notna() & t.player_possession.ne(event.player_id)]
            # Retain a missed quick touch only with observed spatial support.
            # Its candidate is the original convention: next actor frame - 1.
            # Never place it at/before release, across a gap, or on missing XY.
            if successful and possible.empty and len(following):
                actor_event=following.iloc[0]
                candidate_frame=limit-1
                distance=None
                reason='fallback_disabled'
                if reception_fallback_max_distance is not None:
                    if candidate_frame <= frame:
                        reason='not_after_release'
                    elif candidate_frame not in coordinates.index:
                        reason='missing_candidate_frame'
                    elif coordinates.loc[candidate_frame, 'period'] != period:
                        reason='different_period'
                    else:
                        point=coordinates.loc[candidate_frame]
                        columns=[receiver+'_x', receiver+'_y', 'ball_x', 'ball_y']
                        if not all(column in point.index for column in columns):
                            reason='missing_position_columns'
                        else:
                            xy=point[columns].to_numpy(dtype=float, na_value=np.nan)
                            if not np.isfinite(xy).all():
                                reason='missing_positions'
                            else:
                                distance=float(np.hypot(xy[0]-xy[2], xy[1]-xy[3]))
                                reason=('distance_confirmed' if distance <= reception_fallback_max_distance
                                        else 'too_far_from_ball')
                check=dict(pass_event_id=str(event.event_id), receiver_player_id=receiver,
                    release_frame=frame, actor_event_id=str(actor_event.event_id),
                    actor_event_frame=limit, candidate_frame=candidate_frame,
                    ball_distance_m=distance, reason=reason)
                fallback_checks.append(check)
                if reason == 'distance_confirmed':
                    restart=getattr(event, 'restart_type', 'unknown')
                    label=pass_origin_label(restart)
                    seed=(candidate_frame, receiver, label, period, 'receiver_contact_fallback')
                    seeds.append(seed)
                    seed_events.setdefault(seed, []).append(str(event.event_id))
                    fallback_support[seed]=check
                    continue
            if len(possible):
                control = possible.iloc[0]
                same_team = control.player_possession.split('_')[0] == event.player_id.split('_')[0]
                valid_reception = successful and pd.notna(event.receiver_player_id) and control.player_possession == event.receiver_player_id
                valid_interception = not successful and not same_team
                if valid_reception or valid_interception:
                    restart = getattr(event, 'restart_type', 'unknown')
                    label = pass_origin_label(restart, intercepted=valid_interception)
                    seed=(int(control.frame), control.player_possession, label, period,
                          'receiver_control' if valid_reception else 'opponent_control')
                    seeds.append(seed)
                    seed_events.setdefault(seed, []).append(str(event.event_id))
                elif successful:
                    unverified_receivers += 1
            elif successful:
                unverified_receivers += 1
        if event.event_type in {'recovery', 'interception', 'reception'}:
            label = ('pass_interception' if event.event_type == 'interception' else
                     'recovery' if event.event_type == 'recovery' else 'pass_reception')
            # An annotated BallClaiming/recovery establishes its own start;
            # it does not require simultaneous confirmation by the estimator.
            seed=(frame, event.player_id, label, period, 'annotated_control')
            seeds.append(seed)
            seed_events.setdefault(seed, []).append(str(event.event_id))
    # A provider may identify control without a corresponding reception event.
    # Retain the interval with unknown origin instead of assigning a label.
    changes=t.player_possession.ne(t.player_possession.shift()) | t.period.ne(t.period.shift()) | t.frame.diff().ne(1)
    if include_unannotated_starts:
        for row in t.loc[changes&t.player_possession.notna()].itertuples():
            seeds.append((int(row.frame),row.player_possession,'unknown',row.period,'control_stream'))
    priority = {'receiver_control': 0, 'receiver_contact_fallback': 0, 'opponent_control': 0, 'annotated_control': 1, 'control_stream': 2}
    seeds=sorted(set(seeds),key=lambda r:(r[0],priority[r[4]],r[2],r[1]))
    parents=[]
    covered_until=-np.inf
    overlapping_starts=[]
    reconciliations=[]
    by_event=e.set_index('event_id', drop=False)
    transition_types=terminal_types | {'tackle', 'recovery', 'interception', 'reception', 'stoppage', 'restart'}
    for start,player,label,period,source in seeds:
        original_seed=(start,player,label,period,source)
        supporting_events=seed_events.get(original_seed, [])
        fallback=fallback_support.get(original_seed)
        reconciled_with=None
        observed_start=start
        if start<=covered_until:
            # Two annotations can describe one transfer: recovery by the tackle
            # winner just before the outgoing carrier's synchronized tackle.
            # Match identities and adjacent relevant events, not an arbitrary
            # time tolerance. Preserve the tackle frame for the outgoing player;
            # the incoming interval starts on the next observed sample.
            previous=parents[-1]
            coherent=False
            if (source == 'annotated_control' and previous['end_type'] == 'possession_loss'
                    and previous['frame_start'] <= start and previous['period'] == period
                    and previous['tackle_winner_id'] == player
                    and previous['team_id'] != player.split('_')[0]):
                duel=by_event.loc[previous['terminal_event_id']]
                loser=getattr(duel, 'loser_player_id', None)
                claims=e.loc[e.event_id.astype(str).isin(supporting_events)
                    & e.event_type.isin(['recovery', 'interception']) & e.player_id.eq(player)]
                intervening=e.loc[e.frame.ge(start) & e.frame.le(covered_until)
                    & e.event_type.isin(transition_types)]
                allowed=set(claims.event_id) | {duel.event_id}
                coherent=(len(claims) > 0
                    and (pd.isna(loser) or loser == previous['player_id'])
                    and intervening.event_id.isin(allowed).all())
                aligned_start=int(covered_until)+1
                coherent=(coherent and aligned_start in indexed.index
                    and indexed.loc[aligned_start, 'period'] == period)
                if coherent:
                    reconciled_with=str(duel.event_id)
                    reconciliations.append(dict(original_frame=start, aligned_frame=aligned_start,
                        player_id=player, recovery_event_ids=list(map(str, claims.event_id)),
                        tackle_event_id=reconciled_with,
                        outgoing_possession_id=previous['possession_id']))
                    start=aligned_start
                    source='reconciled_tackle_recovery'
            if not coherent:
                if source != 'control_stream':
                    overlapping_starts.append(dict(frame=start, player_id=player, source=source))
                continue
        remaining=t.loc[t.frame.ge(start)&t.period.eq(period)]
        if remaining.empty:
            continue
        same_period = e.loc[e.frame.isin(remaining.frame)]
        own_terminal = same_period.player_id.eq(player) & same_period.event_type.isin(terminal_types)
        lost_tackle = (same_period.event_type.eq('tackle')
                       & same_period.winner_player_id.notna()
                       & same_period.winner_player_id.ne(player))
        terminal_events = same_period.loc[own_terminal | lost_tackle]
        terminal = terminal_events.iloc[0] if len(terminal_events) else None
        # A different actor's release/shot/foul cannot become this carrier's
        # terminal action. Nor may an interval cross a gap or period boundary.
        other_actions = same_period.loc[
            same_period.event_type.isin(terminal_types) & same_period.player_id.ne(player)]
        consecutive=remaining.frame.diff().fillna(1).eq(1)
        gaps=remaining.loc[~consecutive]
        cutoff=int(gaps.frame.iloc[0]) if len(gaps) else int(remaining.frame.max())+1
        stops=same_period.loc[same_period.event_type.isin(['stoppage','restart'])]
        if len(stops):
            cutoff=min(cutoff,int(stops.frame.iloc[0]))
        if len(other_actions):
            cutoff=min(cutoff,int(other_actions.frame.iloc[0]))
        has_terminal = terminal is not None and int(terminal.frame) < cutoff
        if not has_terminal:
            different=remaining.loc[remaining.player_possession.notna() & remaining.player_possession.ne(player)]
            if len(different):
                cutoff=min(cutoff,int(different.frame.iloc[0]))
        if cutoff<=start:
            continue
        verified=False
        outcome='unknown'
        receiver=None
        receipt=np.nan
        loss=pd.NA
        ending='unknown'
        interruption=None
        if has_terminal:
            end=int(terminal.frame)
            ending=terminal.event_type
            if ending == 'tackle':
                ending='possession_loss'
                loss=True
                verified=True
            elif ending=='pass':
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
        if end<start:
            continue
        parents.append(dict(match_id=str(match_id) if match_id is not None else '',possession_id=len(parents)+1,
            player_id=player,team_id=player.split('_')[0],frame_start=start,frame_end=end,period=period,
            start_type=label,end_type=ending,pass_outcome=outcome,target_pass=receiver,frame_receipt=receipt,
            loss=loss,outcome_verified=verified,boundary_source=source,game_interruption_after=interruption,
            start_event_id=';'.join(dict.fromkeys(supporting_events)) or None,
            original_frame_start=observed_start,
            start_reconciled_with_event_id=reconciled_with,
            start_support_event_id=fallback['actor_event_id'] if fallback else None,
            start_ball_distance_m=fallback['ball_distance_m'] if fallback else np.nan,
            is_player_possession_start_matched=True,is_player_possession_end_matched=ending!='unknown',
            terminal_event_id=terminal.event_id if has_terminal else None,
            terminal_event_type=terminal.event_type if has_terminal else None,
            tackle_winner_id=terminal.winner_player_id if has_terminal and terminal.event_type=='tackle' else None))
        covered_until=end
    result=pd.DataFrame(parents)
    if result.empty:
        raise ValueError('No control interval can be reconstructed from these matched events and carrier states.')
    result['loss']=result.loss.astype('boolean')
    actions=canonical_actions(result,fps=fps)
    actions['outcome_verified']=actions.possession_id.map(result.set_index('possession_id').outcome_verified)
    updated_tracking=tracking.copy()
    # Keep the estimator for audit. The canonical carrier follows the inferred
    # nonoverlapping intervals, so a checkpoint and its possession name the same
    # player even when event/contact timing differs from estimated control.
    updated_tracking['estimated_player_possession'] = tracking.player_possession
    updated_tracking['player_possession'] = updated_tracking.frame.map(t.set_index('frame').player_possession)
    for row in result.itertuples():
        inside = updated_tracking.frame.between(row.frame_start, row.frame_end)
        old_team = updated_tracking.loc[inside, 'player_possession'].astype('string').str.split('_').str[0]
        changed_team = old_team.notna() & old_team.ne(row.team_id)
        if 'attacking_side' in updated_tracking:
            values = updated_tracking.loc[inside, 'attacking_side'].copy()
            values.loc[changed_team] = values.loc[changed_team].map({
                'left_to_right': 'right_to_left', 'right_to_left': 'left_to_right'})
            known = tracking.loc[tracking.period.eq(row.period)
                & tracking.player_possession.astype('string').str.startswith(row.team_id+'_', na=False), 'attacking_side'].dropna().unique()
            if len(known) == 1:
                values = values.fillna(known[0])
            updated_tracking.loc[inside, 'attacking_side'] = values
        if 'home_score_difference_prefix' in updated_tracking:
            updated_tracking.loc[inside, 'score_difference_prefix'] = (
                updated_tracking.loc[inside, 'home_score_difference_prefix'] * (1 if row.team_id == 'home' else -1))
        elif 'score_difference_prefix' in updated_tracking:
            indices = updated_tracking.index[inside][changed_team.to_numpy()]
            updated_tracking.loc[indices, 'score_difference_prefix'] *= -1
        updated_tracking.loc[inside, 'player_possession'] = row.player_id
    return updated_tracking,result,actions,dict(n_individual_possessions=len(result),n_actions=len(actions),
        n_unverified_outcomes=int((~result.outcome_verified).sum()),unverified_successful_receptions=unverified_receivers,
        policy=POLICY,control_source='supplied control stream; may be tracking-estimated',
        one_frame_reception_fallback=bool(result.boundary_source.eq('receiver_contact_fallback').any()),
        reception_fallback_max_distance=reception_fallback_max_distance,
        reception_fallback_checks=fallback_checks,
        overlapping_start_annotations=overlapping_starts,
        overlap_reconciliations=reconciliations,
        tackle_loss_rule='identified winner differs from current carrier; no subsequent-control requirement',
        include_unannotated_starts=include_unannotated_starts, n_segmentation_events=len(e))



def segment_possessions(tracking, events, *, fps, match_id=None,
                        include_unannotated_starts=True,
                        reception_fallback_max_distance=1.5):
    """Build individual possessions and actions from aligned external data.

    Inputs are canonical tracking and event tables already mapped/synchronized
    by the caller. No files are loaded and no time alignment is inferred.
    Returns (tracking, individual_possessions, actions, report). Supply those
    tables explicitly to PRESS(...). Unknown context/outcomes remain unknown.
    include_unannotated_starts=False uses successful receptions and annotated
    recovery/interception starts only, as in the IDSSE example. True also keeps
    tracking-only starts and failed-pass opponent controls, labelled by source.
    reception_fallback_max_distance is the maximum observed ball/receiver XY
    distance in metres for a missed-control start one frame before the next
    actor event (default 1.5; None disables this fallback). The candidate must
    exist in the same period and strictly follow release. Coherent overlapping
    recovery/tackle annotations use tackle+1 for the incoming start. Both rules
    retain source IDs and their diagnostics in the report/interval table.
    The original estimated carrier is retained as estimated_player_possession.
    """
    from .events import normalize_events
    from .synchronization import synchronize_events
    if not isinstance(tracking, pd.DataFrame):
        raise TypeError('tracking must be an already prepared DataFrame, not a file path.')
    events = normalize_events(events)
    events = synchronize_events(events, tracking, fps=fps, already_synchronized=True)
    tracking, parents, actions, report = segment_controls(tracking, events, fps=fps, match_id=match_id,
        include_unannotated_starts=include_unannotated_starts,
        reception_fallback_max_distance=reception_fallback_max_distance)
    report.update(n_events=len(events), n_matched_events=int(events.sync_matched.sum()))
    return tracking, parents, actions, report
