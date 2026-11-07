import numpy as np


def log_membership(contributions, valid_teammates):
    """Normalize among teammates, excluding carrier and invalid padded slots.

    If ALL controls underflow exactly to zero, retain the cell and allocate
    it uniformly among valid teammates. No smoothing/temperature is learned.
    """
    c = np.asarray(contributions, np.float64)
    valid = np.asarray(valid_teammates, bool)
    assert valid.any() and np.isfinite(c).all() and (c >= 0).all()
    c = np.where(valid, c, 0.0)
    total = c.sum(axis=-1, keepdims=True)
    fallback = total[..., 0] == 0
    with np.errstate(divide='ignore', invalid='ignore'):
        logs = np.log(c) - np.log(total)
    logs = np.where(fallback[..., None], np.where(valid, -np.log(valid.sum()), -np.inf), logs)
    assert not np.isnan(logs).any()
    assert np.allclose(np.exp(logs).sum(axis=-1), 1.0, atol=1e-12)
    return (logs.astype(np.float32), fallback)
