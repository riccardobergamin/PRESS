"""Canonical physical state, xT sampling and dynamic pressure-line geometry."""

import re

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


def get_pv_players(frame, tracking_df):
    """Return the current carrier, teammates and opponents in native pitch axes.

    The carrier is a canonical home/away alias, not necessarily a shirt
    number. Missing players are padded at (1000,1000) with zero velocity,
    preserving the fitted ten-teammate/eleven-opponent contracts. Active
    player order follows the tracking columns and is retained in the ID lists.
    More than eleven observed players on a side raises an error instead of
    silently discarding observations. No future positions are consulted.
    """
    rows = tracking_df.loc[tracking_df.frame == frame]
    if len(rows) != 1:
        raise ValueError('Missing or duplicated frame in tracking data.')
    row = rows.iloc[0]
    length, width = (float(row.get('pitch_length', 105.0)), float(row.get('pitch_width', 68.0)))
    if not np.isfinite([length, width]).all() or min(length, width) <= 0:
        raise ValueError('Pitch dimensions must be finite and positive.')
    player = row.player_possession
    if pd.isna(player):
        raise ValueError('Missing player possession information.')
    match = re.fullmatch('(home|away)_(\\d+)', str(player))
    if match is None:
        raise ValueError('Invalid canonical player possession ID.')
    team, identifier = match.groups()
    if row.attacking_side not in ('left_to_right', 'right_to_left'):
        raise ValueError('Missing or invalid attacking direction.')
    bc = row[[f'{player}_x', f'{player}_y']].to_numpy(float)
    velocity = row[[f'{player}_vx', f'{player}_vy']].to_numpy(float)
    if not np.isfinite([bc, velocity]).all() or np.linalg.norm(bc) >= 500:
        raise ValueError('Ball carrier position and velocity must be finite and observed.')

    def side_state(side, count):
        positions, velocities, ids = ([], [], [])
        for column in row.index:
            if not re.fullmatch(side + '_\\d+_x', column):
                continue
            alias = column[:-2]
            if alias == player:
                continue
            xy = row[[f'{alias}_x', f'{alias}_y']].to_numpy(float)
            if not np.isfinite(xy).all() or np.linalg.norm(xy) >= 500:
                continue
            vv = row[[f'{alias}_vx', f'{alias}_vy']].to_numpy(float)
            if not np.isfinite(vv).all():
                raise ValueError(f'Observed player {alias} requires finite velocity.')
            positions.append(xy)
            velocities.append(vv)
            ids.append(alias)
        if len(ids) > count:
            raise ValueError('More than eleven observed players on one team.')
        n = count - len(ids)
        xy = np.asarray(positions, float).reshape(-1, 2)
        vv = np.asarray(velocities, float).reshape(-1, 2)
        return (
            np.vstack([xy, np.full((n, 2), 1000.0)]),
            np.vstack([vv, np.zeros((n, 2))]),
            ids + [None] * n,
        )

    other = 'away' if team == 'home' else 'home'
    own, own_velocity, own_ids = side_state(team, 10)
    opponents, opponent_velocity, opponent_ids = side_state(other, 11)
    return {
        'frame_df': rows.copy(),
        'pitch_length': length,
        'pitch_width': width,
        'player': player,
        'team_id': team,
        'shirt_num': int(identifier),
        'attacking_side': row.attacking_side,
        'x_bc': bc,
        'v_bc': velocity,
        'r_defs': opponents,
        'v_defs': opponent_velocity,
        'defender_ids': opponent_ids,
        'r_atts': own,
        'v_atts': own_velocity,
        'attacker_ids': own_ids,
    }


def validate_xt_grid(grid):
    """Return a copied xT surface: first axis attack x, second lateral y.

    Any nonempty two-dimensional resolution is accepted. The map represents
    cell values (not xT changes), so entries must be finite and nonnegative.
    """
    values = np.asarray(grid, dtype=float)
    if values.ndim != 2 or 0 in values.shape or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('xT_grid must be a finite nonnegative (n_x, n_y) array.')
    return values.copy()


def get_xt_values(targets, attacking_side, xt_grid, *, pitch_length=105.0, pitch_width=68.0):
    """
    Get attacking-team and opponent xT
    for arbitrary target positions.

    Parameters
    ----------
    targets : ndarray, shape (N, 2)
    """
    targets = np.asarray(targets, dtype=float)
    xt_grid = np.asarray(xt_grid, dtype=float)
    n_x, n_y = xt_grid.shape
    x = targets[:, 0]
    y = targets[:, 1]
    if attacking_side == 'left_to_right':
        x_att = x
        y_att = y
    elif attacking_side == 'right_to_left':
        x_att = -x
        y_att = -y
    else:
        raise ValueError(f'Invalid attacking_side: {attacking_side}')
    x_opp = -x_att
    y_opp = -y_att
    PITCH_LENGTH = float(pitch_length)
    PITCH_WIDTH = float(pitch_width)
    if not np.isfinite([PITCH_LENGTH, PITCH_WIDTH]).all() or min(PITCH_LENGTH, PITCH_WIDTH) <= 0:
        raise ValueError('Pitch dimensions must be finite and positive.')
    x_cell_att = np.clip(((x_att + PITCH_LENGTH / 2) / PITCH_LENGTH * n_x).astype(int), 0, n_x - 1)
    y_cell_att = np.clip(((y_att + PITCH_WIDTH / 2) / PITCH_WIDTH * n_y).astype(int), 0, n_y - 1)
    x_cell_opp = np.clip(((x_opp + PITCH_LENGTH / 2) / PITCH_LENGTH * n_x).astype(int), 0, n_x - 1)
    y_cell_opp = np.clip(((y_opp + PITCH_WIDTH / 2) / PITCH_WIDTH * n_y).astype(int), 0, n_y - 1)
    xt_team = xt_grid[x_cell_att, y_cell_att]
    xt_opp = xt_grid[x_cell_opp, y_cell_opp]
    return (xt_team, xt_opp)


def compute_dynamic_pressure_lines(r_defs, attacking_side, n_lines=3):
    """
    Compute defensive pressure lines once per pass.
    """
    r_defs = np.asarray(r_defs, dtype=float)
    if r_defs.ndim != 2:
        return np.full(n_lines, np.nan)
    if r_defs.shape[1] < 2:
        return np.full(n_lines, np.nan)
    valid = np.all(np.isfinite(r_defs[:, :2]), axis=1)
    valid &= np.linalg.norm(r_defs[:, :2], axis=1) < 500
    r_defs = r_defs[valid]
    if len(r_defs) < n_lines + 1:
        return np.full(n_lines, np.nan)
    if attacking_side == 'left_to_right':
        goalkeeper_idx = np.argmax(r_defs[:, 0])
    elif attacking_side == 'right_to_left':
        goalkeeper_idx = np.argmin(r_defs[:, 0])
    else:
        raise ValueError(f'Unknown attacking_side: {attacking_side}')
    r_defs = np.delete(r_defs, goalkeeper_idx, axis=0)
    X = r_defs[:, [0]]
    clustering = AgglomerativeClustering(n_clusters=n_lines, linkage='complete')
    labels = clustering.fit_predict(X)
    lines = np.array([X[labels == cluster_id, 0].mean() for cluster_id in range(n_lines)])
    return np.sort(lines)


def get_target_pressure_line_index(targets, pressure_lines, attacking_side):
    """
    Return the number of pressure lines crossed
    by every candidate destination.

    Returns
    -------
    ndarray, shape (N,)
        0, 1, 2 or 3.
    """
    targets = np.asarray(targets, dtype=float)
    pressure_lines = np.asarray(pressure_lines, dtype=float)
    if len(pressure_lines) != 3 or not np.all(np.isfinite(pressure_lines)):
        return np.full(len(targets), np.nan)
    if attacking_side == 'left_to_right':
        target_x_att = targets[:, 0]
        lines_att = pressure_lines
    elif attacking_side == 'right_to_left':
        target_x_att = -targets[:, 0]
        lines_att = -pressure_lines
    else:
        raise ValueError(f'Invalid attacking_side: {attacking_side}')
    lines_att = np.sort(lines_att)
    return np.sum(target_x_att[:, None] >= lines_att[None, :], axis=1).astype(float)
