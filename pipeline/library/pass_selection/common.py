"""Small grouped-choice and file helpers used by both final model families."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 ** 2), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def groups(sizes):
    sizes = np.asarray(sizes, dtype=np.int64)
    assert len(sizes) and (sizes > 0).all()
    return (np.r_[0, np.cumsum(sizes)[:-1]], sizes)
