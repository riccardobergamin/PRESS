"""Optional composition of local risk and individual-possession continuation."""
from __future__ import annotations

import numpy as np


def compose_individual_xloss(*, p_pass, drive_loss, pass_failure, residual_loss):
    """Combine probabilities into an individual-possession loss expectation.

    p_pass is P(pass | state); drive_loss is the current drive's loss
    probability; pass_failure is the destination-weighted pass failure
    probability. residual_loss is P(final individual-possession failure |
    current state, drive selected and successful). A successful pass ends
    the individual possession without failure. This two-action composition
    is separate from the core next-step hazard and does not predict team
    possession loss after a successful pass.

    Inputs are finite probabilities in [0, 1], scalar or broadcastable
    arrays. Returns a float for scalar inputs, otherwise a NumPy array.
    Invalid probabilities or incompatible shapes raise ValueError.
    This function performs composition only; it does not fit/load a model.
    """
    values = []
    for name, value in (("p_pass", p_pass), ("drive_loss", drive_loss),
                        ("pass_failure", pass_failure), ("residual_loss", residual_loss)):
        array = np.asarray(value, dtype=float)
        if not array.size or not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
            raise ValueError(f"{name} must contain finite probabilities in [0, 1].")
        values.append(array)
    try:
        p, d, f, c = np.broadcast_arrays(*values)
    except ValueError as error:
        raise ValueError("Probability inputs must have broadcastable shapes.") from error
    result = p * f + (1 - p) * (d + (1 - d) * c)
    return float(result) if result.ndim == 0 else result
