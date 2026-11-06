"""Exact CNL derivatives and the distinct RUMBoost CPU curvature rule.

Here rho >= 1 is the RUMBoost inverse dissimilarity: rho = 1 / mu_run31.
Memberships are fixed state-dependent normalized individual pitch control.
The exact diagonal times J/(J-1) is NOT numerically identical to the
curvature returned by the published CNL objective. Both are exposed and
checked independently; no absolute value or softmax envelope is imposed.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from scipy.optimize import minimize
jax.config.update('jax_enable_x64', True)

def _parts(scores, log_alpha, chosen, rho):
    scores = jnp.asarray(scores, jnp.float64)
    log_alpha = jnp.asarray(log_alpha, jnp.float64)
    log_alpha = log_alpha - logsumexp(log_alpha, axis=-1, keepdims=True)
    active = jnp.isfinite(log_alpha).any(axis=1)
    a = rho * (scores[:, :, None] + log_alpha)
    a = jnp.where(active[:, None, :], a, 0.0)
    log_z = logsumexp(a, axis=1)
    log_w = jnp.where(active, log_z / rho, -jnp.inf)
    log_w = log_w - logsumexp(log_w, axis=1, keepdims=True)
    log_b = a - log_z[:, None, :]
    b = jnp.where(active[:, None, :], jnp.exp(log_b), 0.0)
    joint = log_w + log_b[jnp.arange(len(scores)), chosen, :]
    log_py = logsumexp(joint, axis=1)
    return (-log_py, b, jnp.exp(log_w), jnp.exp(joint - log_py[:, None]))

@jax.jit
def nll(scores, log_alpha, chosen, rho):
    return _parts(scores, log_alpha, chosen, rho)[0]

@jax.jit
def derivatives(scores, log_alpha, chosen, rho):
    loss, b, w, posterior = _parts(scores, log_alpha, chosen, rho)
    p = jnp.sum(b * w[:, None, :], axis=-1)
    c = jnp.sum(b * posterior[:, None, :], axis=-1)
    u = jnp.sum(b * b * w[:, None, :], axis=-1)
    v = jnp.sum(b * b * posterior[:, None, :], axis=-1)
    g = p + (rho - 1.0) * c
    g = g.at[jnp.arange(len(scores)), chosen].add(-rho)
    dp = rho * p - p * p + (1.0 - rho) * u
    dc = rho * c + (1.0 - 2.0 * rho) * v - (1.0 - rho) * c * c
    exact_h = dp + (rho - 1.0) * dc
    count = scores.shape[1]
    return (loss, g, exact_h * count / (count - 1.0), exact_h, p)

@jax.jit
def reference_derivatives(scores, log_alpha, chosen, rho):
    """Exact gradient and literal upstream CNL curvature, in stable form.

This reproduces f_obj_cross_nested from v1.0.2 and the archived main CPU
method. It is intentionally not advertised as the exact Hessian diagonal.
    """
    loss, b, w, posterior = _parts(scores, log_alpha, chosen, rho)
    p = jnp.sum(b * w[:, None, :], axis=-1)
    b2 = b * b
    u = jnp.sum(b2 * w[:, None, :], axis=-1)
    dp = rho * p - p * p + (1.0 - rho) * u
    c = jnp.sum(b * posterior[:, None, :], axis=-1)
    g = p + (rho - 1.0) * c
    g = g.at[jnp.arange(len(scores)), chosen].add(-rho)
    pb = p[:, :, None]
    second_not_chosen = -rho * rho * b + rho * (-b2 + b) + (b - pb) ** 2 - dp[:, :, None]
    second_chosen = rho * rho * (2 * b2 - 3 * b + 1) + rho * (-3 * b2 + 3 * b + 2 * (b * pb - pb)) + b2 - 2 * b * pb + pb * pb - dp[:, :, None]
    mask = jnp.arange(scores.shape[1])[None, :] == chosen[:, None]
    second = jnp.where(mask[:, :, None], second_chosen, second_not_chosen)
    curvature = g * g - jnp.sum(posterior[:, None, :] * second, axis=-1)
    count = scores.shape[1]
    curvature *= count / (count - 1.0)
    return (loss, g, curvature, p)

@jax.jit
def flat_derivatives(scores, chosen):
    scores = jnp.asarray(scores, jnp.float64)
    p = jax.nn.softmax(scores, axis=1)
    g = p.at[jnp.arange(len(scores)), chosen].add(-1.0)
    count = scores.shape[1]
    h = jnp.maximum(p * (1.0 - p) * count / (count - 1.0), 1e-06)
    loss = logsumexp(scores, axis=1) - scores[jnp.arange(len(scores)), chosen]
    return (loss, g, h)

def fit_rho(engine, scores, current, bounds=(1.0, 10.0), relaxation=0.1, restarts=(), scan=()):
    """Fit-only SLSQP with the reference relaxation in inverse scale.

    Common scale and optional restarts are project restrictions. Reject
    failed/nonfinite or uphill relaxed updates rather than silently taking
    an optimizer failure. No validation labels enter this function.
    """
    cache = {}

    def fun(x):
        key = float(x[0])
        if key not in cache:
            cache[key] = float(np.mean(engine.losses(scores, key)))
        return cache[key]
    before = fun([current])
    scan_values = list(dict.fromkeys([float(current)] + [float(x) for x in scan]))
    assert all((bounds[0] <= value <= bounds[1] for value in scan_values))
    profile = [dict(rho=value, fit_ce=fun([value])) for value in scan_values]
    initial = min(profile, key=lambda item: item['fit_ce'])['rho']
    trials = []
    solutions = [(before, float(current))]
    for start in dict.fromkeys([float(initial)] + [float(x) for x in restarts]):
        assert bounds[0] <= start <= bounds[1]
        result = minimize(fun, np.array([start]), bounds=[bounds], method='SLSQP')
        candidate = float(result.x[0])
        value = fun([candidate])
        valid = bool(result.success and np.isfinite(value) and (bounds[0] <= candidate <= bounds[1]))
        trials.append(dict(initial_rho=start, rho=candidate, fit_ce=value, success=bool(result.success), message=str(result.message), evaluations=int(result.nfev)))
        if valid:
            solutions.append((value, candidate))
    candidate_value, candidate = min(solutions)
    relaxed = float(current) + relaxation * (candidate - float(current))
    after = fun([relaxed])
    accepted = bool(np.isfinite(after) and after <= before + 1e-12)
    rho = relaxed if accepted else float(current)
    return (rho, dict(previous_rho=float(current), rho=rho, mu_run31_convention=1.0 / rho, fit_ce_before=before, fit_ce_after=after if accepted else before, candidate_fit_ce=candidate_value, candidate_rho=candidate, relaxation=relaxation, trials=trials, accepted=accepted, evaluations=len(cache), conditional_scan=profile, slsqp_initial_rho=initial, near_lower_bound=bool(rho <= bounds[0] + 0.002), near_upper_bound=bool(rho >= bounds[1] - 0.002)))

@jax.jit
def full_log_probabilities(scores, log_alpha, rho):
    """Stable final CNL distribution, for post-training output calibration only."""
    scores = jnp.asarray(scores, jnp.float64)
    log_alpha = jnp.asarray(log_alpha, jnp.float64)
    log_alpha = log_alpha - logsumexp(log_alpha, axis=-1, keepdims=True)
    active = jnp.isfinite(log_alpha).any(axis=1)
    a = rho * (scores[:, :, None] + log_alpha)
    a = jnp.where(active[:, None, :], a, 0.0)
    log_z = logsumexp(a, axis=1)
    log_w = jnp.where(active, log_z / rho, -jnp.inf)
    log_w = log_w - logsumexp(log_w, axis=1, keepdims=True)
    return logsumexp(log_w[:, None, :] + a - log_z[:, None, :], axis=-1)
