"""Pitch views of stored destination choice and action-selection results."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import Pitch

from PRESS.utils.frame import frame_to_long, resolve_pitch_size


def _pitch_frame(press, frame_id, pitch_length, pitch_width, figsize):
    frame = press.get_frame(frame_id)
    pitch_length, pitch_width = resolve_pitch_size(frame, pitch_length, pitch_width)
    players = frame_to_long(frame)
    state = press.get_state(frame_id)
    pitch = Pitch(pitch_type="skillcorner", pitch_length=pitch_length,
                  pitch_width=pitch_width, pitch_color="#edf3ed", line_color="#647568")
    fig, ax = plt.subplots(figsize=figsize)
    pitch.draw(ax=ax)
    for team, color in (("home", "steelblue"), ("away", "indianred")):
        selected = players.loc[players.team_id == team]
        pitch.scatter(selected.x, selected.y, ax=ax, s=60, color=color,
                      edgecolors="white", zorder=4, label=team.title())
        for row in selected.itertuples(index=False):
            ax.annotate(str(row.shirt_num), (row.x, row.y), xytext=(4, 4),
                        textcoords="offset points", fontsize=8, color="#25312a", zorder=5)
    pitch.scatter(*state["x_bc"], ax=ax, s=140, facecolors="none",
                  edgecolors="#18231b", linewidths=1.5, zorder=6, label="Carrier")
    frame = press.get_frame(frame_id).iloc[0]
    pitch.scatter(frame.ball_x, frame.ball_y, ax=ax, s=22, color="white",
                  edgecolors="black", zorder=7)
    return fig, ax, pitch, state


def plot_transition(press, frame_id, pitch_length=None, pitch_width=None,
                    figsize=(12, 8), title_fontsize=11, legend_fontsize=8):
    """Plot stored P(destination | pass, state) on its original candidate grid.

    Requires compute_transition_probabilities. Colors show probability mass
    per candidate, without interpolating or renormalizing it. Includes
    tracked players and carrier; returns (figure, pitch_axes).
    """
    result = press.get_ppcf_result(frame_id)
    if "Transition probabilities" not in result:
        raise ValueError("Compute transition probabilities before plotting destination choice.")
    grid = np.asarray(result["grid"])
    probabilities = np.asarray(result["Transition probabilities"])
    fig, ax, pitch, state = _pitch_frame(press, frame_id, pitch_length, pitch_width, figsize)
    field = pitch.scatter(grid[:, 0], grid[:, 1], c=probabilities, cmap="viridis",
                          marker="s", s=120, alpha=0.85, ax=ax, zorder=2)
    colorbar = fig.colorbar(field, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("P(destination | pass, state) per candidate", fontsize=legend_fontsize)
    colorbar.ax.tick_params(labelsize=legend_fontsize)
    best = grid[int(np.argmax(probabilities))]
    pitch.arrows(*state["x_bc"], *best, ax=ax, color="#a67505", width=1.2,
                 zorder=3, label="Highest-probability candidate")
    ax.set_title(f"Pass destination choice · frame {frame_id}", fontsize=title_fontsize)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=legend_fontsize)
    fig.subplots_adjust(bottom=0.12, top=0.92)
    plt.show()
    return fig, ax


def plot_action_selection(press, frame_id, pitch_length=None, pitch_width=None,
                          figsize=(12, 8), title_fontsize=11, legend_fontsize=8):
    """Plot tracked state with the stored scalar pass/carry mixture weights.

    Requires compute_action_selection_probability. The inset bars show
    action probabilities for this frame, not a spatial probability field.
    Returns (figure, pitch_axes).
    """
    result = press.get_action_selection_result(frame_id)
    fig, ax, _, _ = _pitch_frame(press, frame_id, pitch_length, pitch_width, figsize)
    inset = ax.inset_axes([1.04, 0.36, 0.30, 0.28])
    values = [result["Pass probability"], result["Ball drive probability"]]
    inset.barh([0, 1], values, color=["#3172a0", "#a67505"])
    inset.set_yticks([0, 1], labels=["Pass", "Carry"])
    inset.set_xlim(0, 1.16)
    inset.set_xticks([0, 0.5, 1])
    inset.tick_params(labelsize=legend_fontsize)
    for row, probability in enumerate(values):
        inset.text(probability + 0.025, row, f"{probability:.3f}", va="center", fontsize=legend_fontsize)
    inset.set_title("P(action | state)", fontsize=legend_fontsize + 1)
    ax.set_title(f"Pass versus carry choice · frame {frame_id}", fontsize=title_fontsize)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=legend_fontsize)
    fig.subplots_adjust(left=0.05, right=0.73, bottom=0.12, top=0.92)
    plt.show()
    return fig, ax
