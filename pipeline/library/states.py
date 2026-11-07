"""Aligned initial states, known-at-frame context and fixed match partitions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PRESS.utils.state import get_pv_players, compute_dynamic_pressure_lines


def folds():
    return json.loads((Path(__file__).resolve().parents[1] / 'folds.json').read_text())


def initial_state(tracking, frame, carrier=None, direction=None):
    """Carrier authority comes from the matched event, including release-only rows."""
    row = tracking.loc[tracking.frame.eq(frame)].copy()
    if carrier is not None:
        row['player_possession'] = carrier
    if direction is not None:
        row['attacking_side'] = direction
    state = get_pv_players(frame, row)
    state['p_pos'] = np.vstack([state['r_atts'], state['x_bc'], state['r_defs']]).astype(np.float32)
    state['p_vel'] = np.vstack([state['v_atts'], state['v_bc'], state['v_defs']]).astype(np.float32)
    return state


def context(events, metadata, frame, team_id):
    """Match clock and prefix score use only information effective at this frame."""
    raw = events.loc[events.event_type.eq('player_possession')].copy()

    def seconds(value):
        parts = str(value).split(':')
        return sum((float(p) * 60 ** i for i, p in enumerate(reversed(parts))))
    if raw.empty:
        return (np.nan, np.nan)
    before = raw.loc[raw.frame_start.le(frame)].sort_values('frame_start')
    event = before.iloc[-1] if len(before) else raw.iloc[0]
    period = event.period
    same = raw.loc[raw.period.eq(period)]
    offsets = same.time_end.map(seconds) - same.frame_end / 10
    clock = (frame / 10 + float(offsets.median())) / 60
    home = metadata['home_team']['id']
    raw = raw.sort_values(['frame_start', 'frame_end'])
    home_score = np.where(raw.team_id.eq(home), raw.team_score, raw.opponent_team_score)
    away_score = np.where(raw.team_id.eq(home), raw.opponent_team_score, raw.team_score)
    effective = np.where(raw.game_interruption_before.isin(['goal_for', 'goal_against']),
        raw.frame_start,
        raw.frame_end + 1)
    eligible = effective <= frame
    hs = float(np.max(home_score[eligible], initial=0))
    aw = float(np.max(away_score[eligible], initial=0))
    return (clock, hs - aw if team_id == home else aw - hs)


def load_match(base, match):
    folder = Path(base) / 'matches' / str(match)
    return (pd.read_parquet(folder / 'tracking.parquet'),
        pd.read_parquet(folder / 'events.parquet'),
        pd.read_parquet(folder / 'players.parquet'),
        json.loads((folder / 'metadata.json').read_text()))


def save_states(path, states, targets=None):
    keys = ['x_bc', 'v_bc', 'r_defs', 'v_defs', 'p_pos', 'p_vel']
    arrays = {k: np.asarray([s[k] for s in states], np.float32) for k in keys}
    if targets is not None:
        arrays['r_target'] = np.asarray(targets, np.float32)
    np.savez_compressed(path, **arrays)
