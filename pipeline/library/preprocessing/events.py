"""Individual-control interval annotation with explicitly matched boundaries."""
import numpy as np
import pandas as pd
from .possession_selection import matched_individual_mask, is_matched

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
        is_loss = end_type == 'pass' and pass_outcome == 'unsuccessful' or end_type in ['possession_loss', 'clearance']
        is_non_loss = end_type == 'pass' and pass_outcome == 'successful' or end_type in ['shot', 'foul_suffered']
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
    columns = ['possession_id', 'player_id', 'frame_start', 'frame_end', 'frame_receipt', 'time_start', 'time_end', 'loss', 'loss_reason', 'start_type', 'end_type', 'pass_outcome', 'target_pass', 'game_interruption_after', 'attacking_side', 'is_player_possession_start_matched', 'is_player_possession_end_matched', 'receipt_start_matched'] + [column for column in ('period', 'match_id') if column in poss.columns]
    result = pd.DataFrame(rows, columns=columns)
    result['target_pass'] = result['target_pass'].astype('Int64')
    return result
