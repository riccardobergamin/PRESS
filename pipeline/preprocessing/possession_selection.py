"""Boundary-quality criteria shared by possession and pass preparation."""
from __future__ import annotations
import pandas as pd
START_MATCHED = 'is_player_possession_start_matched'
END_MATCHED = 'is_player_possession_end_matched'


def pass_duration_eligible(release_frame, receipt_frame, *, fps=10):
    """Training flight duration in the inclusive interval [0.5, 10] seconds."""
    gap = receipt_frame - release_frame
    return (gap >= 0.5 * fps) & (gap <= 10 * fps)


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
