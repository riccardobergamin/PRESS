"""xT equations: spatial bins, smoothed components and Bellman iteration.

Input is the four accepted EVENT tables, not canonical one-second carries.
This review keeps their selection and coordinate convention unchanged.
No tracking frame is needed: xT is an event-location model.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


PITCH_LENGTH = 105.
PITCH_WIDTH = 68.
EVENT_TYPES = ["carries", "passes", "shots", "losses"]


# 1. Continuous event coordinates -> one spatial origin/destination bin.

def spatial_cells(x, y, grid):
    """Return flat cell indices i=ix*n_y+iy on the fixed 105x68 m pitch.

    First array axis is longitudinal x; second is lateral y. Preserve the
    accepted clipping at pitch limits. This does not introduce event exclusions.
    """
    nx, ny = grid
    ix = np.floor((np.asarray(x) + PITCH_LENGTH / 2) / PITCH_LENGTH * nx).astype(int)
    iy = np.floor((np.asarray(y) + PITCH_WIDTH / 2) / PITCH_WIDTH * ny).astype(int)
    return np.clip(ix, 0, nx - 1) * ny + np.clip(iy, 0, ny - 1)


def bin_events(event_tables, grid):
    """Rebin the SAME continuous event coordinates at the requested resolution.

    Carries/passes have origin and destination. Shots/losses only need an origin.
    The archived bin columns are not reused; row indices are retained for pairing
    predictions across resolutions. No new direction normalization is applied.
    """
    binned = {}
    for kind in EVENT_TYPES:
        events = event_tables[kind].copy()
        if kind in ["passes", "carries"]:
            events["origin"] = spatial_cells(events.x_start, events.y_start, grid)
            events["target"] = spatial_cells(events.x_end, events.y_end, grid)
        else:
            location = "shot" if kind == "shots" else "loss"
            events["origin"] = spatial_cells(events["x_" + location], events["y_" + location], grid)
        binned[kind] = events
    return binned


# 2. Conditional destination distributions T_ij=P(destination=j | origin=i).

def transition_probabilities(events, grid, bandwidth_m):
    """Smooth DESTINATION counts separately for each origin, then normalize.

    counts[i,ix,iy] contains observed arrivals at (ix,iy) from origin i.
    Gaussian sigma=(0,h/dx,h/dy): no smoothing across the origin axis.
    Bandwidth h is in metres, keeping its physical meaning across grids.
    """
    nx, ny = grid
    n_cells = nx * ny
    counts = np.zeros((n_cells, nx, ny))
    np.add.at(counts, (events.origin, events.target // ny, events.target % ny), 1.)
    unseen_origins = counts.sum(axis=(1, 2)) == 0
    if bandwidth_m:
        dx, dy = PITCH_LENGTH / nx, PITCH_WIDTH / ny
        counts = gaussian_filter(counts, sigma=(0, bandwidth_m / dx, bandwidth_m / dy), mode="nearest")

    # Retain the accepted 1e-8 pseudocount per destination. Every row sums to one.
    counts += 1e-8
    transitions = counts.reshape(n_cells, n_cells)
    transitions /= transitions.sum(axis=1, keepdims=True)
    transitions[unseen_origins] = 1. / n_cells
    return transitions


def goal_probabilities(shots, grid, bandwidth_m):
    """Estimate P(goal | shot,cell) from separately smoothed shot/goal counts.

    g_i=(K_h * goal_counts)_i / ((K_h * shot_counts)_i + 1e-8).
    This smooths the two count fields before taking their ratio.
    """
    nx, ny = grid
    n_cells = nx * ny
    shot_counts = np.bincount(shots.origin, minlength=n_cells).reshape(nx, ny).astype(float)
    goal_counts = np.bincount(shots.loc[shots.goal, "origin"], minlength=n_cells).reshape(nx, ny).astype(float)
    if bandwidth_m:
        sigma = (bandwidth_m / (PITCH_LENGTH / nx), bandwidth_m / (PITCH_WIDTH / ny))
        shot_counts = gaussian_filter(shot_counts, sigma=sigma, mode="nearest")
        goal_counts = gaussian_filter(goal_counts, sigma=sigma, mode="nearest")
    return (goal_counts / (shot_counts + 1e-8)).ravel()


# 3. Local action frequencies -> Bellman equation -> fitted value surface.

def fit_xt(binned_tables, grid, bandwidths):
    """Fit the accepted event-based xT model on already selected fitting matches.

    q_i=p_shot,i * p_goal,i
    M_ij=p_pass,i*T_pass,ij + p_carry,i*T_carry,ij
    xT=q+M*xT. Loss is a zero-value terminal branch.

    These are frequencies in the archived event construction, not the PRESS
    action-selection policy or ball-drive classifier probabilities.
    """
    nx, ny = grid
    n_cells = nx * ny

    # a. Counts for all four event types have the SAME per-origin denominator.
    counts = {kind: np.bincount(binned_tables[kind].origin, minlength=n_cells).astype(float)
              for kind in EVENT_TYPES}
    total_counts = sum(counts.values())
    action_probabilities = {
        kind: np.divide(counts[kind], total_counts, out=np.zeros_like(total_counts), where=total_counts > 0)
        for kind in EVENT_TYPES
    }
    # Empty origins have all action frequencies zero. Smoothing below only
    # concerns conditional destination distributions and goal-given-shot counts.

    # b. Fit transition matrices and the goal-given-shot component.
    pass_transitions = transition_probabilities(binned_tables["passes"], grid, bandwidths["passes"])
    carry_transitions = transition_probabilities(binned_tables["carries"], grid, bandwidths["carries"])
    p_goal = goal_probabilities(binned_tables["shots"], grid, bandwidths["shots"])

    # c. Immediate scoring reward and successful-continuation operator.
    operator = (action_probabilities["passes"][:, None] * pass_transitions
                + action_probabilities["carries"][:, None] * carry_transitions)
    reward = action_probabilities["shots"] * p_goal

    # d. Fixed-point iteration from zero; there is no gradient optimizer or LR.
    value = np.zeros(n_cells)
    for iteration in range(10000):
        next_value = reward + operator @ value
        max_change = np.max(abs(next_value - value))
        value = next_value
        if max_change < 1e-12:
            break
    if not (max_change < 1e-12 and np.isfinite(value).all() and value.min() >= 0 and value.max() <= 1):
        raise ValueError("xT Bellman iteration did not converge to a finite probability surface.")

    # Save every component from this SAME fit, so the Bellman identity can be checked.
    return dict(
        xt=value.reshape(nx, ny), p_goal=p_goal.reshape(nx, ny),
        p_pass=action_probabilities["passes"].reshape(nx, ny),
        p_carry=action_probabilities["carries"].reshape(nx, ny),
        p_shot=action_probabilities["shots"].reshape(nx, ny),
        p_loss=action_probabilities["losses"].reshape(nx, ny),
        T_pass=pass_transitions, T_carry=carry_transitions,
        iterations=iteration + 1,
        bellman_residual=float(np.max(abs(value - reward - operator @ value))),
    )


def plot_xt(xt_grid):
    """Conventional pitch heatmap, with exact cell values and no interpolation."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.ticker import PercentFormatter
    from mplsoccer import Pitch

    nx, ny = xt_grid.shape
    pitch = Pitch(pitch_type="skillcorner", pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH,
                  line_color="#d7dce0", line_zorder=3)
    fig, ax = pitch.draw(figsize=(10.5, 7))
    x_edges = np.linspace(-PITCH_LENGTH / 2, PITCH_LENGTH / 2, nx + 1)
    y_edges = np.linspace(-PITCH_WIDTH / 2, PITCH_WIDTH / 2, ny + 1)
    norm = Normalize(0, float(xt_grid.max()))
    mesh = ax.pcolormesh(x_edges, y_edges, xt_grid.T, cmap="magma", norm=norm, shading="flat")
    for ix, iy in np.ndindex(xt_grid.shape):
        ax.text((x_edges[ix] + x_edges[ix + 1]) / 2, (y_edges[iy] + y_edges[iy + 1]) / 2,
                f"{xt_grid[ix, iy]:.2%}", ha="center", va="center", fontsize=9,
                color="white" if norm(xt_grid[ix, iy]) < .65 else "#17212b", zorder=5)
    ax.set_title(f"Expected Threat · {nx} × {ny} · attack →")
    fig.colorbar(mesh, ax=ax, shrink=.8, pad=.025, format=PercentFormatter(1)).set_label("xT")
    return fig, ax
