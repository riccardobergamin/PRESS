"""Convenience entry points that reuse the shared package pitch drawing."""

from .plot import plot_frame, plot_pass_map


def plot_pass_selection(press, frame_id, pitch_length=None, pitch_width=None,
                    figsize=(12, 8), title_fontsize=11, legend_fontsize=8, *,
                    pass_target=None, **kwargs):
    """Draw relative destination preference through plot_pass_map; return (fig, ax)."""
    return plot_pass_map(press, frame_id, pitch_length, pitch_width, figsize,
                         title_fontsize, legend_fontsize, field="pass_selection",
                         pass_target=pass_target, **kwargs)


def plot_action_selection(press, frame_id, pitch_length=None, pitch_width=None,
                          figsize=(12, 8), title_fontsize=11, legend_fontsize=8, *,
                          title=None, **kwargs):
    """Draw the observed frame with stored native P(pass); return (fig, ax)."""
    probability = float(press.get_action_selection_result(frame_id)["Pass probability"])
    return plot_frame(press, frame_id, pitch_length, pitch_width, figsize,
                      title_fontsize, legend_fontsize,
                      title=title or f"Action selection · frame {frame_id} · P(pass)={probability:.1%}",
                      **kwargs)
