"""Offline tracking derivatives, respecting gaps and period boundaries."""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def differentiate(values, frames, periods, *, window, fps=10.0):
    """Differentiate finite contiguous runs, never across a gap or period.

    A NaN splits the run instead of contaminating its full filtering window.
    No interpolation is applied. Runs shorter than three samples stay NaN.
    """
    values = np.asarray(values, dtype=float)
    frames = np.asarray(frames)
    periods = np.asarray(periods)
    result = np.full(len(values), np.nan)
    valid = np.isfinite(values) & pd.notna(periods)
    breaks = np.ones(len(values), dtype=bool)
    if len(values) > 1:
        breaks[1:] = (np.diff(frames) != 1) | (periods[1:] != periods[:-1]) | ~valid[1:] | ~valid[:-1]
    starts = np.flatnonzero(breaks)
    for start, stop in zip(starts, np.r_[starts[1:], len(values)]):
        if not valid[start] or stop - start < 3:
            continue
        size = min(window, stop - start if (stop - start) % 2 else stop - start - 1)
        result[start:stop] = savgol_filter(values[start:stop],
            size,
            min(2, size - 1),
            deriv=1,
            delta=1 / fps,
            mode='interp')
    return result

