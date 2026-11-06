"""Shared protocol and grouped categorical likelihood; production is read-only."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
CASES = [{'name': '01_raw_fernandez_context_bypass', 'pc': 'raw', 'lines': 'nearest', 'context': True, 'bypass': 2}, {'name': '02_isotonic_fernandez_context_bypass', 'pc': 'isotonic', 'lines': 'nearest', 'context': True, 'bypass': 2}]
TUNING = [{'max_depth': 4, 'reg_lambda': 1.0}]

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

def probabilities(scores, starts, sizes):
    scores = np.asarray(scores, np.float64)
    maximum = np.maximum.reduceat(scores, starts)
    exp = np.exp(scores - np.repeat(maximum, sizes))
    return exp / np.repeat(np.add.reduceat(exp, starts), sizes)

def chosen_nll(scores, starts, sizes, chosen, temperature=1.0):
    scaled = np.asarray(scores, np.float64) / temperature
    maximum = np.maximum.reduceat(scaled, starts)
    normalizer = maximum + np.log(np.add.reduceat(np.exp(scaled - np.repeat(maximum, sizes)), starts))
    return normalizer - scaled[starts + np.asarray(chosen, np.int64)]

def objective(starts, sizes, chosen):
    positive = starts + np.asarray(chosen, np.int64)

    def call(scores, dtrain):
        p = probabilities(scores, starts, sizes)
        h = np.maximum(p * (1 - p), 1e-06)
        p[positive] -= 1.0
        return (p, h)
    return call

def metric(starts, sizes, chosen):

    def call(scores, dtrain):
        return ('choice_nll', float(chosen_nll(scores, starts, sizes, chosen).mean()))
    return call

def grid(length, width):
    nx, ny = (int(round(length)), int(round(width)))
    assert abs(length - nx) < 1e-08 and abs(width - ny) < 1e-08
    x, y = np.meshgrid(np.arange(nx) + 0.5 - length / 2, np.arange(ny) + 0.5 - width / 2, indexing='ij')
    return np.column_stack([x.ravel(), y.ravel()]).astype(np.float32)

def bypass(bc, targets, players, sign):
    valid = np.isfinite(players).all(axis=-1) & (np.linalg.norm(players, axis=-1) < 500)
    px = sign * players[valid, 0]
    bx, tx = (sign * bc[0], sign * targets[:, 0])
    return ((px[:, None] > bx) & (px[:, None] <= tx[None, :])).sum(axis=0).astype(np.float32)

def line_index(x, centers, sign, kind):
    centers = np.sort(sign * np.asarray(centers, float))
    x = sign * np.asarray(x)
    if kind == 'count':
        return (x[:, None] >= centers[None, :]).sum(axis=1).astype(np.float32)
    return (1 + np.argmin(abs(x[:, None] - centers[None, :]), axis=1)).astype(np.float32)

def xt_at(points, sign, length, width, surface):
    q = np.asarray(points) * sign
    ix = np.clip(((q[:, 0] + length / 2) / length * 10).astype(int), 0, 9)
    iy = np.clip(((q[:, 1] + width / 2) / width * 6).astype(int), 0, 5)
    return surface[ix, iy]
