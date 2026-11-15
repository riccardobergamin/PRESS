"""Shared pitch drawing for observed frames and previously computed fields."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from mplsoccer import Pitch
from scipy.interpolate import griddata

from PRESS.utils.frame import frame_to_long, resolve_pitch_size

TEAM_COLORS = {"home": "steelblue", "away": "indianred"}


def _canvas(press, frame_id, pitch_length, pitch_width, figsize, ax):
    frame = press.tracking_data.loc[press.tracking_data.frame == frame_id].copy()
    if len(frame) != 1:
        raise ValueError(f"Frame {frame_id!r} not found or duplicated.")
    length, width = resolve_pitch_size(frame, pitch_length, pitch_width)
    # mplsoccer's skillcorner pitch means centred metric axes, independent
    # of the provider used to initialize PRESS.
    pitch = Pitch(pitch_type="skillcorner", pitch_length=length, pitch_width=width,
                  pitch_color="#7fbf7f", line_color="white", linewidth=1.3)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    pitch.draw(ax=ax)
    return fig, ax, pitch, frame, frame_to_long(frame)


def _draw_state(pitch, ax, frame, players, *, velocity_scale=1.5, defender_shares=None):
    for team, color in TEAM_COLORS.items():
        selected = players[players.team_id.eq(team)]
        colors = color
        if defender_shares is not None:
            defending_team, shares, cmap = defender_shares
            if team == defending_team:
                colors = [cmap(shares.get(player, 0.0)) for player in selected.player_id]
        pitch.scatter(selected.x, selected.y, ax=ax, s=45, color=colors,
                      edgecolors="white", linewidth=0.6, zorder=3)
    ball = frame.iloc[0]
    if np.isfinite([ball.ball_x, ball.ball_y]).all():
        pitch.scatter(ball.ball_x, ball.ball_y, ax=ax, s=20, color="white",
                      edgecolors="black", linewidth=0.8, zorder=10)
    if velocity_scale:
        for row in players.dropna(subset=["vx", "vy"]).itertuples():
            pitch.arrows(row.x, row.y, row.x + velocity_scale * row.vx,
                         row.y + velocity_scale * row.vy, ax=ax, color="black",
                         width=0.5, headwidth=3, headlength=3, alpha=0.55, zorder=4)


def _finish(fig, ax, title, title_fontsize, legend_fontsize, *, show, legend, pass_line=False):
    ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=12)
    if legend:
        handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                          markeredgecolor="white", markersize=9, label=f"Team {team}")
                   for team, color in TEAM_COLORS.items()]
        handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                              markeredgecolor="black", markersize=5, label="Ball"))
        if pass_line:
            handles.append(Line2D([0], [0], color="#202020", linestyle="--",
                                  linewidth=1.8, label="Observed pass"))
        fig.legend(handles=handles, loc="lower center", fontsize=legend_fontsize,
                   frameon=True, ncol=len(handles))
    fig.subplots_adjust(bottom=0.10, top=0.90)
    if show:
        plt.show()
    return fig, ax


def _field(fig, ax, grid, values, *, label, cmap, legend_fontsize, scale=1.0):
    grid, values = np.asarray(grid, float), np.asarray(values, float)
    if grid.ndim != 2 or grid.shape[1] != 2 or values.shape != (len(grid),) or not len(grid):
        raise ValueError("A field requires nonempty grid (N,2) and values (N,).")
    if not np.isfinite(grid).all() or not np.isfinite(values).all():
        raise ValueError("A plotted field must contain finite values.")
    if len(grid) >= 3 and np.linalg.matrix_rank(grid - grid.mean(axis=0)) == 2:
        xx, yy = np.meshgrid(np.linspace(grid[:, 0].min(), grid[:, 0].max(), 120),
                             np.linspace(grid[:, 1].min(), grid[:, 1].max(), 120))
        surface = griddata(grid, values, (xx, yy), method="linear")
        artist = ax.contourf(xx, yy, surface, levels=np.linspace(0, scale, 11),
                             cmap=cmap, alpha=0.6, zorder=1)
    else:
        artist = ax.scatter(grid[:, 0], grid[:, 1], c=values, cmap=cmap,
                            vmin=0, vmax=scale, alpha=0.6, zorder=1)
    bar = fig.colorbar(artist, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label(label, fontsize=legend_fontsize)
    bar.ax.tick_params(labelsize=legend_fontsize)
    return bar


def plot_frame(press, frame_id, pitch_length=None, pitch_width=None, figsize=(12, 8),
               title_fontsize=9, legend_fontsize=8, *, title=None, ax=None,
               show=True, legend=True, velocity_scale=1.5):
    """Draw an observed frame and return (figure, axes), without inference.

    Accepts a PRESS or BCOS object exposing canonical tracking_data. Optional
    ax draws into an existing panel; show=False supports batch exports. Vector
    length is velocity_scale times velocity, not a predicted future position.
    Missing carriers are allowed; player IDs and extra markers are not drawn.
    """
    fig, ax, pitch, frame, players = _canvas(
        press,
        frame_id,
        pitch_length,
        pitch_width,
        figsize,
        ax,
    )
    _draw_state(pitch, ax, frame, players, velocity_scale=velocity_scale)
    return _finish(fig, ax, title or f"Tracking frame {frame_id}", title_fontsize,
                   legend_fontsize, show=show, legend=legend)


def plot_pass_map(press, frame_id, pitch_length=None, pitch_width=None, figsize=(12, 8),
                  title_fontsize=9, legend_fontsize=8, *, field="pass_success",
                  pass_target=None, title=None, ax=None, show=True, legend=True,
                  velocity_scale=1.5):
    """Draw stored pitch_control, pass_success or pass_selection; return (fig, ax).

    Control/completion use absolute probabilities on 0–1. Choice colors are
    100 * probability / maximum probability, without modifying stored values.
    pass_target=(x,y) adds the observed pass as a dashed line in native metres.
    No model calculation is triggered by this function.
    """
    keys = {"pitch_control": "Attacking pitch control field",
            "pass_success": "Pass probability", "pass_selection": "Pass selection probabilities"}
    if field not in keys:
        raise ValueError("field must be pitch_control, pass_success or pass_selection.")
    if pass_target is not None:
        pass_target = np.asarray(pass_target, float)
        if pass_target.shape != (2,) or not np.isfinite(pass_target).all():
            raise ValueError("pass_target must contain two finite pitch coordinates.")
    result = press.get_ppcf_result(frame_id)
    if keys[field] not in result:
        if result.get('details_available') is False:
            raise ValueError('Spatial details were not retained; rerun compute_risk_frame with keep_details=True.')
        raise ValueError("Compute the requested pass field before plotting it.")
    fig, ax, pitch, frame, players = _canvas(
        press,
        frame_id,
        pitch_length,
        pitch_width,
        figsize,
        ax,
    )
    values = np.asarray(result[keys[field]], float)
    if field == "pass_selection":
        peak = values.max()
        values = 100 * values / peak if peak > 0 else np.zeros_like(values)
        label, default_title, cmap, scale = (
            "Relative pass preference (% of maximum)",
            "Pass destination choice",
            "viridis",
            100,
        )
    else:
        label = (
            "Pass-success probability"
            if field == "pass_success"
            else "Attacking pitch-control probability"
        )
        colors = ["indianred", "#f7f7f7", "steelblue"]
        if frame.iloc[0].player_possession.startswith("away_"):
            colors = colors[::-1]
        cmap = LinearSegmentedColormap.from_list("team_divergent", colors)
        default_title, scale = label, 1
    _field(fig, ax, result["grid"], values, label=label, cmap=cmap,
           legend_fontsize=legend_fontsize, scale=scale)
    _draw_state(pitch, ax, frame, players, velocity_scale=velocity_scale)
    if pass_target is not None:
        ball = frame.iloc[0]
        if not np.isfinite([ball.ball_x, ball.ball_y]).all():
            raise ValueError("A pass line requires a finite observed ball position.")
        pitch.plot([ball.ball_x, pass_target[0]], [ball.ball_y, pass_target[1]],
                   ax=ax, color="#202020", linestyle="--", linewidth=1.8,
                   zorder=6, label="Observed pass")
    return _finish(fig, ax, title or f"{default_title} · frame {frame_id}", title_fontsize,
                   legend_fontsize, show=show, legend=legend, pass_line=pass_target is not None)


def plot_ppcf(press, frame_id, pitch_length=None, pitch_width=None, figsize=(12, 8),
              title_fontsize=9, legend_fontsize=8, *, field="pitch_control", pass_target=None, **kwargs):
    """Compatibility name for plot_pass_map, also returning (figure, axes)."""
    return plot_pass_map(press, frame_id, pitch_length, pitch_width, figsize,
                         title_fontsize, legend_fontsize, field=field,
                         pass_target=pass_target, **kwargs)


def plot_ball_drive_risk(press, frame_id, pitch_length=None, pitch_width=None, figsize=(12, 8),
            title_fontsize=9, legend_fontsize=8, *, title=None, ax=None, show=True,
            legend=True, velocity_scale=1.5):
    """Color defenders by their share of total drive risk; return (fig, ax).

    The absolute loss probability appears in the title. Cause probabilities
    are matched to defender IDs from the computed state, including missing
    player padding. A zero total loss produces zero relative shares.
    """
    result = press.get_ball_drive_risk_result(frame_id)
    if 'Individual tackle probabilities' not in result:
        raise ValueError('Defender details were not retained; rerun compute_risk_frame with keep_details=True.')
    state = press.get_state(frame_id)
    total = float(result["Ball drive risk"])
    probabilities = np.asarray(result["Individual tackle probabilities"], float)
    shares = probabilities / total if total > 0 else np.zeros_like(probabilities)
    by_id = (
        {player: float(value) for player, value in zip(state["defender_ids"], shares)
        if player is not None}
    )
    defending_team = "away" if state["team_id"] == "home" else "home"
    cmap = LinearSegmentedColormap.from_list(
        "ball_drive_risk",
        ["white", TEAM_COLORS[defending_team]],
    )
    fig, ax, pitch, frame, players = _canvas(
        press,
        frame_id,
        pitch_length,
        pitch_width,
        figsize,
        ax,
    )
    _draw_state(pitch, ax, frame, players, velocity_scale=velocity_scale,
                defender_shares=(defending_team, by_id, cmap))
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap), ax=ax,
                       fraction=0.03, pad=0.02)
    bar.set_label("Share of total loss risk", fontsize=legend_fontsize)
    bar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    bar.ax.tick_params(labelsize=legend_fontsize)
    return _finish(fig, ax, title or f"Ball drive risk · frame {frame_id} · P(loss)={total:.3f}",
                   title_fontsize, legend_fontsize, show=show, legend=legend)


def plot_bcos(bcos, frame_id, pitch_length=None, pitch_width=None, figsize=(12, 8),
              title_fontsize=9, legend_fontsize=8, *, title=None, ax=None,
              show=True, legend=True, velocity_scale=1.5):
    """Draw a stored BCOS probability field from PRESS or BCOS; return (fig, ax)."""
    result = (bcos.get_bcos_result(frame_id) if hasattr(bcos, "get_bcos_result")
              else bcos.get_result(frame_id))
    if 'grid' not in result or 'Ball carrier open space' not in result:
        raise ValueError('BCOS map is unavailable; compute_bcos or run compute_risk_frame with keep_details=True.')
    fig, ax, pitch, frame, players = _canvas(bcos, frame_id, pitch_length, pitch_width, figsize, ax)
    _field(fig, ax, result["grid"], result["Ball carrier open space"],
           label="Ball carrier open space", cmap="inferno", legend_fontsize=legend_fontsize)
    _draw_state(pitch, ax, frame, players, velocity_scale=velocity_scale)
    return _finish(fig, ax, title or f"BCOS frame {frame_id}", title_fontsize,
                   legend_fontsize, show=show, legend=legend)


def plot_risk(press, frame_id, *, field="hazard", title=None, **kwargs):
    """Draw a real frame with stored hazard, residual_risk or xloss in the title."""
    fields = {"hazard": ("HAZARD", "Hazard", "h"),
              "residual_risk": ("RESIDUAL_RISK", "Residual loss probability", "Residual risk"),
              "xloss": ("XLOSS", "xLoss", "xLoss")}
    if field not in fields:
        raise ValueError("field must be hazard, residual_risk or xloss.")
    component, key, label = fields[field]
    value = float(press.get_result(component, frame_id)[key])
    return plot_frame(
        press,
        frame_id,
        title=title or f"{label}={value:.2%} · frame {frame_id}",
        **kwargs,
    )
