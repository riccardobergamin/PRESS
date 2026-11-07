"""Counts ahead of transverse query lines, not forward-only bypass counts."""
import numpy as np
COUNT_COLUMNS = ['opponents_ahead_bc',
    'teammates_ahead_bc',
    'opponents_ahead_target',
    'teammates_ahead_target']


def ahead_counts(players, locations, direction):
    players = np.asarray(players)
    locations = np.asarray(locations)
    assert direction in (-1, 1)
    assert players.ndim == locations.ndim == 2
    assert players.shape[1] == locations.shape[1] == 2
    assert np.isfinite(locations).all()
    valid = np.isfinite(players).all(axis=1) & (np.linalg.norm(players, axis=1) < 500)
    px = direction * players[valid, 0]
    qx = direction * locations[:, 0]
    counts = (px[:, None] > qx[None, :]).sum(axis=0)
    assert (counts <= 11).all()
    return counts.astype(np.uint8)


def state_counts(bc, targets, player_positions, direction):
    """Cache schema: 0..9 teammates, 10 carrier, 11..21 opponents."""
    pos = np.asarray(player_positions)
    assert pos.shape == (22, 2)
    assert np.array_equal(pos[10], bc)
    locations = np.vstack([bc, targets])
    opponents = ahead_counts(pos[11:], locations, direction)
    teammates = ahead_counts(pos[:10], locations, direction)
    paired = np.column_stack([opponents, teammates])
    return (paired[0], paired[1:])
