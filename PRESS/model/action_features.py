"""Aggregate hypothetical pass options; never use actual next action/outcome."""

import numpy as np

from .action_selection import NUMERIC


def ahead(players, targets, direction):
    """Count valid players strictly beyond each target along the attack axis."""
    a = np.asarray(players)
    valid = np.isfinite(a).all(axis=1) & (np.linalg.norm(a, axis=1) < 500)
    return (direction * a[valid, 0, None] > direction * np.asarray(targets)[None, :, 0]).sum(axis=0)


def line_values(points, centres, direction):
    """Return the nearest line index (1–3) and signed longitudinal distance."""
    centres = np.sort(direction * np.asarray(centres, float))
    if len(centres) != 3 or not np.isfinite(centres).all():
        return (np.full(len(points), np.nan), np.full(len(points), np.nan))
    z = direction * np.asarray(points)[:, 0]
    k = np.argmin(abs(z[:, None] - centres[None]), axis=1)
    return (k + 1, centres[k] - z)


def xt(points, direction, length, width, surface):
    """Read an xT cell-value surface in the supplied attack direction."""
    nx, ny = surface.shape
    oriented = direction * np.asarray(points)
    ix = np.clip(((oriented[:, 0] + length / 2) / length * nx).astype(int), 0, nx-1)
    iy = np.clip(((oriented[:, 1] + width / 2) / width * ny).astype(int), 0, ny-1)
    return surface[ix, iy]


def aggregate(
    *,
    bc,
    velocity,
    players,
    targets,
    direction,
    own_lines,
    opp_lines,
    choice,
    completion,
    drive_loss,
    bcos,
    xt_grid,
    pitch_length,
    pitch_width,
    match_clock_min,
    score_difference_prefix,
    completed_canonical_drives,
):
    """Summarize hypothetical pass outcomes in the saved action feature order."""
    bc, velocity, players, targets = map(np.asarray, [bc, velocity, players, targets])
    q, p = (np.asarray(choice, float), np.asarray(completion, float))
    assert q.shape == p.shape == (len(targets),)
    assert np.isfinite(q).all() and np.isfinite(p).all()
    assert (q >= 0).all() and ((p >= 0) & (p <= 1)).all() and np.isclose(q.sum(), 1.0, atol=1e-08)
    assert 0 <= drive_loss <= 1 and np.array_equal(players[10], bc)
    # Joint choice/outcome weights retain branch mass; do not renormalize them.
    success, failure = (q * p, q * (1 - p))
    # State layout: ten teammates, carrier at index 10, then eleven opponents.
    own = players[:10]
    opponents = players[11:]
    carrier_counts = [ahead(g, bc[None], direction)[0] for g in [own, opponents]]
    own_ahead, opp_ahead = [ahead(g, targets, direction) for g in [own, opponents]]
    tl, _ = line_values(targets, own_lines, -direction)
    ol, _ = line_values(targets, opp_lines, direction)
    predicted = (bc + velocity)[None]
    own_index, _ = line_values(predicted, own_lines, -direction)
    opp_index, _ = line_values(predicted, opp_lines, direction)
    own_ordered = np.sort(-direction * np.asarray(own_lines, float))
    own_distance = (
        np.array([direction * (-direction * own_ordered[int(own_index[0]) - 1] - predicted[0, 0])])
        if np.isfinite(own_index[0])
        else np.array([np.nan])
    )
    _, opp_distance = line_values(predicted, opp_lines, direction)
    own_xt = xt(targets, direction, pitch_length, pitch_width, xt_grid)
    opp_xt = xt(targets, -direction, pitch_length, pitch_width, xt_grid)
    speed = float(np.linalg.norm(velocity))
    angle = float(np.arctan2(
        direction * velocity[1],
        direction * velocity[0],
    )) if speed >= 1e-08 else np.nan
    values = [
        match_clock_min,
        score_difference_prefix,
        completed_canonical_drives,
        direction * bc[0],
        direction * bc[1],
        speed,
        angle,
        1 - drive_loss,
        bcos,
        xt(predicted, direction, pitch_length, pitch_width, xt_grid)[0],
        xt(predicted, -direction, pitch_length, pitch_width, xt_grid)[0],
        own_index[0],
        opp_index[0],
        own_distance[0],
        opp_distance[0],
        *carrier_counts,
        success.sum(),
        success @ own_xt,
        failure @ opp_xt,
        success @ tl,
        failure @ tl,
        success @ ol,
        failure @ ol,
        success @ own_ahead,
        failure @ own_ahead,
        success @ opp_ahead,
        failure @ opp_ahead,
    ]
    assert len(values) == len(NUMERIC) and (not np.isinf(values).any())
    return dict(zip(NUMERIC, map(float, values)))
