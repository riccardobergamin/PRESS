"""Compatibility entry points for the shared package pitch visualizations."""
from __future__ import annotations

import matplotlib.pyplot as plt


def plot_transition(press, frame_id, pitch_length=None, pitch_width=None,
                    figsize=(12, 8), title_fontsize=11, legend_fontsize=8, *,
                    pass_target=None):
    """Compatibility entry point for the shared relative pass-selection map."""
    from .plot import plot_pass_map
    return plot_pass_map(press, frame_id, pitch_length, pitch_width, figsize,
                         title_fontsize, legend_fontsize, field="pass_selection",
                         pass_target=pass_target)


def plot_action_selection(press, frame_id, pitch_length=None, pitch_width=None,
                          figsize=(12, 8), title_fontsize=11, legend_fontsize=8):
    """Draw the real frame with stored P(pass | state) above the pitch."""
    from .plot import plot_frame
    result = press.get_action_selection_result(frame_id)
    probability = float(result["Pass probability"])
    plot_frame(press, frame_id, pitch_length, pitch_width, figsize,
               title_fontsize, legend_fontsize,
               title=f"Action selection · frame {frame_id} · P(pass)={probability:.1%}")
    fig = plt.gcf()
    return fig, fig.axes[0]
