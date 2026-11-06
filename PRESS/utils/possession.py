"""Possession-level hazard bands, survival summaries, and typologies."""

from __future__ import annotations

import numpy as np

BAND_ORDER = ("None", "Low", "Medium", "High")
_BAND_THRESHOLDS = 1.0 - 2.0 ** (-1.0 / np.array([8.0, 4.0, 2.0]))


def hazard_band_thresholds() -> dict[int, float]:
    """Return theoretical hazard cutoffs keyed by half-life in checkpoints.

    The operational labels use the cutoffs for half-lives of eight, four,
    and two checkpoints. The one-checkpoint cutoff is retained here for
    reference, but does not define a fifth operational band.
    """
    half_lives = (1, 2, 4, 8)
    return {n: float(1.0 - 2.0 ** (-1.0 / n)) for n in half_lives}


def hazard_to_band(hazard) -> str | np.ndarray:
    """Map one hazard or an array of hazards to None/Low/Medium/High.

    Values above the highest operational cutoff remain in High; the
    theoretical one-checkpoint cutoff does not create a fifth label.
    """
    values = np.asarray(hazard, dtype=float)
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Hazard values must be finite and within [0, 1].")
    ranks = np.digitize(values, _BAND_THRESHOLDS)
    labels = np.asarray(BAND_ORDER, dtype=object)[ranks]
    if values.ndim == 0:
        return str(labels)
    return labels


def possession_survival(hazards) -> dict[str, np.ndarray]:
    """Compute checkpoint difficulty, cumulative difficulty, and survival.

    Returns arrays for D_t=-log(1-h_t), H_t=sum_{j<=t}D_j, and
    S_t=exp(-H_t). The calculation is retrospective over the supplied
    sequence of checkpoint hazards.
    """
    h = np.asarray(hazards, dtype=float)
    if h.ndim != 1 or h.size == 0:
        raise ValueError("hazards must be a non-empty one-dimensional sequence.")
    if not np.all(np.isfinite(h)) or np.any((h < 0.0) | (h > 1.0)):
        raise ValueError("Hazard values must be finite and within [0, 1].")
    with np.errstate(divide="ignore"):
        increments = -np.log1p(-h)
    cumulative = np.cumsum(increments)
    survival = np.exp(-cumulative)
    return {
        "difficulty_increment": increments,
        "cumulative_difficulty": cumulative,
        "survival": survival,
        "bands": hazard_to_band(h),
    }
