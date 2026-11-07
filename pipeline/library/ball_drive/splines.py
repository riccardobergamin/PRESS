"""Restricted cubic bases, train-only transforms and paired smoothing selection."""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
DT = 0.04
TIMES = np.arange(25, dtype=np.float32) * np.float32(DT)


def _restricted_basis(z, knots, derivative=0):
    """Restricted cubic basis: linear tails, no intercept or linear column."""
    z = np.asarray(z, float)[..., None]
    k = np.asarray(knots, float)
    power = 3 - derivative
    factor = [1.0, 3.0, 6.0][derivative]
    terms = factor * np.maximum(z - k, 0.0) ** power
    a = (k[-1] - k[:-2]) / (k[-1] - k[-2])
    b = (k[-2] - k[:-2]) / (k[-1] - k[-2])
    return terms[..., :-2] - terms[..., -2, None] * a + terms[..., -1, None] * b

@dataclass


class MarginalBasis:
    offset: float
    scale: float
    knots: list
    projection: list
    transform: list
    penalty: list
    linear_mean: float
    linear_sd: float

    @classmethod
    def fit(cls, values, n_knots=5):
        x = np.asarray(values, float).ravel()
        if not np.isfinite(x).all() or len(np.unique(x)) < n_knots:
            raise ValueError('Insufficient distinct finite training values for spline basis')
        offset = float(x.min())
        scale = float(x.max() - offset)
        z = (x - offset) / scale
        knots = np.quantile(z, np.linspace(0, 1, n_knots))
        if np.any(np.diff(knots) <= 1e-08):
            raise ValueError('Tied spline knots; inspect training distribution')
        raw = _restricted_basis(z, knots)
        linear = np.column_stack([np.ones(len(z)), z])
        projection = np.linalg.lstsq(linear, raw, rcond=None)[0]
        h = raw - linear @ projection
        gram = h.T @ h / len(h)
        e, u = np.linalg.eigh(gram)
        if e.min() <= 1e-13:
            raise ValueError('Rank deficient nonlinear spline basis')
        whitening = u @ np.diag(1 / np.sqrt(e))
        nodes, weights = np.polynomial.legendre.leggauss(2)
        roughness = np.zeros_like(gram)
        for left, right in zip(knots[:-1], knots[1:]):
            points = (left + right) / 2 + (right - left) * nodes / 2
            second = _restricted_basis(points, knots, derivative=2)
            roughness += second.T @ ((weights * (right - left) / 2)[:, None] * second)
        penalty, rotation = np.linalg.eigh(whitening.T @ roughness @ whitening)
        if penalty.min() <= 0:
            raise ValueError('Nonpositive curvature penalty')
        return cls(offset,
            scale,
            knots.tolist(),
            projection.tolist(),
            (whitening @ rotation).tolist(),
            (penalty / penalty.min()).tolist(),
            float(z.mean()),
            float(z.std()))

    def evaluate(self, values):
        z = (np.asarray(values, float) - self.offset) / self.scale
        linear = np.stack([np.ones_like(z), z], axis=-1)
        return (_restricted_basis(z, self.knots) - linear @ np.asarray(self.projection)) @ np.asarray(self.transform)

    def full(self, values):
        z = (np.asarray(values, float) - self.offset) / self.scale
        return np.concatenate([((z - self.linear_mean) / self.linear_sd)[..., None], self.evaluate(values)],
            axis=-1)

@dataclass


class SplineDesign:
    variant: str
    speed_basis: dict | None
    boundary_basis: dict | None
    interaction_projection: list | None
    names: list
    main_count: int
    penalty: list
    interaction_rank: int = 3

    @classmethod
    def fit(cls, speed, boundary, variant, n_knots=5, interaction_rank=3):
        if variant not in {'linear', 'speed', 'boundary', 'additive', 'interaction'}:
            raise ValueError('Unknown spline variant')
        speed = np.broadcast_to(np.asarray(speed)[:, None], np.shape(boundary))
        v = MarginalBasis.fit(speed, n_knots) if variant in {'speed', 'additive', 'interaction'} else None
        b = MarginalBasis.fit(boundary,
            n_knots) if variant in {'boundary',
            'additive',
            'interaction'} else None
        names = []
        penalty = []
        parts = []
        for prefix, basis, values in [('speed', v, speed), ('boundary', b, boundary)]:
            if basis is not None:
                value = basis.evaluate(values)
                parts.append(value)
                names.extend((f'{prefix}_nonlinear_{i}' for i in range(value.shape[-1])))
                penalty.extend(basis.penalty)
        count = len(names)
        projection = None
        if variant == 'interaction':
            main = np.concatenate(parts, axis=-1)
            c = np.concatenate([np.ones((*speed.shape, 1)), speed[..., None], np.asarray(boundary)[..., None], main],
                axis=-1)
            raw = (v.full(speed)[..., :interaction_rank, None] * b.full(boundary)[..., None, :interaction_rank]).reshape(*speed.shape,
                -1)
            projection = np.linalg.lstsq(c.reshape(-1, c.shape[-1]),
                raw.reshape(-1, raw.shape[-1]),
                rcond=None)[0].tolist()
            weights = 1 + np.asarray([0] + v.penalty)[:interaction_rank,
                None] + np.asarray([0] + b.penalty)[None,
                :interaction_rank]
            names.extend((f'interaction_{i}_{j}' for i in range(weights.shape[0]) for j in range(weights.shape[1])))
            penalty.extend(weights.ravel().tolist())
        return cls(variant,
            asdict(v) if v else None,
            asdict(b) if b else None,
            projection,
            names,
            count,
            penalty,
            interaction_rank)

    def evaluate(self, speed, boundary):
        speed = np.broadcast_to(np.asarray(speed)[:, None], np.shape(boundary))
        parts = []
        v = MarginalBasis(**self.speed_basis) if self.speed_basis else None
        b = MarginalBasis(**self.boundary_basis) if self.boundary_basis else None
        if v:
            parts.append(v.evaluate(speed))
        if b:
            parts.append(b.evaluate(boundary))
        main = np.concatenate(parts, axis=-1) if parts else np.empty((*speed.shape, 0))
        if self.variant != 'interaction':
            return main
        c = np.concatenate([np.ones((*speed.shape, 1)), speed[..., None], np.asarray(boundary)[..., None], main],
            axis=-1)
        raw = (v.full(speed)[..., :self.interaction_rank, None] * b.full(boundary)[..., None, :self.interaction_rank]).reshape(*speed.shape,
            -1)
        return np.concatenate([main, raw - c @ np.asarray(self.interaction_projection)], axis=-1)

    def active(self, candidate):
        if candidate['main'] is None:
            return (0, np.empty(0))
        count = self.main_count if candidate.get('interaction') is None else len(self.names)
        precision = np.asarray(self.penalty[:count]) * candidate['main']
        if count > self.main_count:
            precision[self.main_count:] = np.asarray(self.penalty[self.main_count:]) * candidate['interaction']
        return (count, precision)


def paired_selection(records, candidate_order):
    """Conservative paired one-SE rule across the eight inner validation matches.

    Match deletion SEs concern paired score differences, not absolute scores
    whose match-to-match difficulty can overwhelm the comparison.
    """
    import pandas as pd
    table = pd.DataFrame(records)
    scores = {}
    for candidate, g in table.groupby('candidate'):
        scores[candidate] = float(np.average(g.nll, weights=g.n))
    best = min(scores, key=scores.get)
    base = table.loc[table.candidate.eq(best)].set_index('match_id')
    eligible = []
    details = []
    for name in candidate_order:
        g = table.loc[table.candidate.eq(name)].set_index('match_id').reindex(base.index)
        if len(g) != 8 or g.nll.isna().any() or (not np.array_equal(g.n, base.n)):
            raise ValueError('Incomplete inner match predictions')
        weights = g.n.to_numpy(float)
        delta = g.nll.to_numpy() - base.nll.to_numpy()
        difference = float(np.average(delta, weights=weights))
        loo = (np.sum(weights * delta) - weights * delta) / (weights.sum() - weights)
        se = float(np.sqrt((len(loo) - 1) / len(loo) * np.sum((loo - loo.mean()) ** 2)))
        accepted = bool(difference <= se + 1e-12)
        details.append(dict(candidate=name, mean_nll=scores[name], difference_to_best=difference, paired_match_jackknife_se=se, within_one_se=accepted))
        if accepted:
            eligible.append(name)
    return (eligible[0],
        dict(best_score_candidate=best, selected=eligible[0], rule='First (strongest shrinkage) candidate within one paired-match SE of best inner OOF score', candidates=details))
