"""Joint spline MAP estimation with shared-survival exponential integration."""
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from .splines import TIMES, DT
DEFAULT_DYNAMICS = dict(v_max=7.0, a_linear=7.0, a_turn=7.0, preserve_initial_overspeed=True)

@partial(jax.jit, static_argnames=('v_max', 'a_linear', 'a_turn', 'preserve_initial_overspeed'))


def _arrival_batch(r, v, x, vx, *, v_max, a_linear, a_turn, preserve_initial_overspeed):
    from PRESS.physics.times import time_to_intercept

    def snapshot(r, v, x, vx):

        def at_time(t):
            target = (x + vx * t)[None, :]
            return time_to_intercept(
                r, v, target,
                v_max=v_max,
                a_max=a_linear,
                a_turn=a_turn,
                preserve_initial_overspeed=preserve_initial_overspeed,
            ).squeeze(-1)
        return jax.vmap(at_time)(jnp.asarray(TIMES))
    return jax.vmap(snapshot)(r, v, x, vx)


def arrival_times(raw, dynamics):
    """Use package TTI arithmetic; alter only the declared physical parameter.

    a_turn is an effective reorientation scale: fixed-position waiting is not
    a literal constant-speed arc. No arbitrary rotation multiplier is added.
    The unchanged case also has an exact cached-TTI control in the runner.
    """
    cfg = {**DEFAULT_DYNAMICS, **dynamics}
    if min(cfg['v_max'], cfg['a_linear'], cfg['a_turn']) <= 0:
        raise ValueError('Invalid arrival dynamics')
    arrays = [jnp.asarray(raw[key], dtype=jnp.float32) for key in ['r_defs', 'v_defs', 'x_bc', 'v_bc']]
    return np.asarray(_arrival_batch(*arrays, **cfg))


def integrate_cause_rates(cause_rates, integration='exponential', dt=DT):
    """Shared survival and cause probabilities, piecewise constant hazards.

    Shape is (observations, 25 time steps, defenders). Exponential Euler
    solves each frozen-hazard step exactly: no independent cause clipping,
    no survival threshold, and no dt*H<1 stability restriction. The returned
    raw probabilities have no Bernoulli floor. Only the final exponential update is exposed here.
    """
    shape = (cause_rates.shape[0], cause_rates.shape[2])
    if integration == 'exponential':

        def step(state, intensities):
            log_survival, causes = state
            total = intensities.sum(axis=1)
            exposure = dt * total
            small = exposure < 1e-05
            z = jnp.where(small, exposure, 0.0)
            series = 1 - z / 2 + z * z / 6 - z * z * z / 24
            ratio = -jnp.expm1(-exposure) / jnp.where(small, 1.0, exposure)
            factor = dt * jnp.where(small, series, ratio)
            increment = jnp.exp(log_survival)[:, None] * factor[:, None] * intensities
            return ((log_survival - exposure, causes + increment), None)
        initial = (jnp.zeros(cause_rates.shape[0]), jnp.zeros(shape))
        (log_survival, causes), _ = jax.lax.scan(step, initial, jnp.swapaxes(cause_rates, 0, 1))
        return (jnp.exp(log_survival), causes)
    raise ValueError('Unknown integration method: ' + str(integration))


def _predict(theta, data, matrix, kind, scale, integration='exponential'):
    k = matrix.shape[-1]
    eta = theta[1] * data['speed'][:, None] + theta[2] * data['boundary'] + matrix @ theta[4:4 + k]
    base = jnp.exp(theta[0]) * jnp.exp(eta)
    gates = jax.nn.sigmoid(jnp.pi / (jnp.sqrt(3) * jnp.exp(theta[3])) * (jnp.asarray(TIMES)[None, :, None] - data['tti']))
    if kind in {'closing', 'transverse'}:
        rates = base[..., None] * jnp.exp(theta[-1] * data[kind] / scale)
    elif kind == 'angular':
        total = gates.sum(axis=2)
        vector = jnp.sum(gates[..., None] * data['unit'], axis=2)
        resultant = jnp.sqrt(jnp.sum(vector ** 2, axis=-1) + 1e-24)
        coverage = jnp.clip(1 - resultant / jnp.maximum(total, 1e-30), 0, 1)
        rates = base[..., None] * jnp.exp(theta[-1] * coverage[..., None])
    elif kind == 'individual_angle':
        rates = base[..., None] * jnp.exp(data['angle_basis'] @ theta[4 + k:])
    elif kind == 'none':
        rates = jnp.broadcast_to(base[..., None], gates.shape)
    else:
        raise ValueError('Unknown lambda geometry')
    cause_rates = gates * rates
    exposure = jnp.max(DT * cause_rates.sum(axis=2))
    nonloss, _ = integrate_cause_rates(cause_rates, integration=integration)
    return (jnp.clip(nonloss, 1e-06, 1 - 1e-06), exposure)
predict_jit = jax.jit(_predict, static_argnames=('kind', 'integration'))


def posterior(theta, data, matrix, precision, ratio, aware, kind, scale, integration='exponential'):
    nonloss, _ = _predict(theta, data, matrix, kind, scale, integration)
    if aware or ratio != 1.0:
        raise ValueError('The final physical model uses natural class proportions.')
    probability = nonloss
    y = data['outcome']
    nll = -jnp.sum(y * jnp.log(probability) + (1 - y) * jnp.log1p(-probability))
    prior = 0.5 * jnp.sum(theta[:3] ** 2) + 5 * jnp.exp(theta[3]) - 2.5 * theta[3]
    return nll + prior + 0.5 * jnp.sum(precision * theta[4:] ** 2)
value_gradient = jax.jit(jax.value_and_grad(posterior), static_argnames=('aware', 'kind', 'integration'))
objective = jax.jit(posterior, static_argnames=('aware', 'kind', 'integration'))
hessian = jax.jit(jax.hessian(posterior), static_argnames=('aware', 'kind', 'integration'))


def fitting_data(data):
    return {key: jnp.asarray(value, dtype=jnp.float64) for key,
        value in data.items() if key != 'valid_defenders'}


def fit_map(data,
    matrix,
    precision,
    *,
    kind,
    scale,
    ratio=1.0,
    aware=False,
    seed=10,
    initializations=32,
    maxiter=2000,
    integration='exponential'):
    if not jax.config.x64_enabled:
        raise ValueError('Set JAX_ENABLE_X64=1')
    batch = fitting_data(data)
    matrix = jnp.asarray(matrix)
    precision = jnp.asarray(precision)
    args = (batch, matrix, precision, ratio, aware, kind, scale, integration)

    def fun(theta):
        val, grad = value_gradient(theta, *args)
        val = float(val)
        grad = np.asarray(grad, float)
        if not np.isfinite(val) or not np.isfinite(grad).all():
            return (1e+100, np.zeros_like(theta))
        return (val, grad)

    def curvature(theta):
        h = np.asarray(hessian(theta, *args))
        h = (h + h.T) / 2
        if not np.isfinite(h).all():
            raise ValueError('Nonfinite follow-up Hessian')
        return h
    fixed = np.r_[np.log(0.4), 0.1, -0.02, np.log(0.4), np.zeros(len(precision))]
    rng = np.random.default_rng(seed)
    starts = [fixed]
    for _ in range(initializations):
        starts.append(np.r_[rng.normal(), rng.normal(), rng.normal(), np.log(rng.gamma(2.5, 0.2)), rng.normal(size=len(precision)) / np.sqrt(precision)])
    scores = [float(objective(theta, *args)) for theta in starts]
    best = starts[int(np.argmin(np.where(np.isfinite(scores), scores, np.inf)))]
    points = [best] if np.array_equal(best, fixed) else [best, fixed]
    attempts = []

    def record(result, method):
        val, grad = fun(result.x)
        return dict(theta=result.x,
            objective=val,
            converged=bool(result.success and val < 1e+99 and (np.max(abs(grad)) < 0.01)),
            gradient_max=float(np.max(abs(grad))),
            iterations=int(result.nit),
            message=str(result.message),
            method=method)

    def optimize(point, recovery_start=False):
        result = minimize(fun,
            point,
            jac=True,
            method='L-BFGS-B',
            options=dict(maxiter=maxiter, ftol=1e-12, gtol=1e-05, maxls=40))
        primary = record(result, 'L-BFGS-B')
        attempts.append(primary)
        primary['recovery_start'] = recovery_start
        if not primary['converged'] and primary['objective'] < 1e+99:
            retry = minimize(fun,
                result.x,
                jac=True,
                hess=curvature,
                method='trust-exact',
                options=dict(maxiter=200, gtol=1e-05))
            refined = record(retry, 'trust-exact stationarity refinement')
            refined['primary_attempt'] = {k: v for k, v in primary.items() if k != 'theta'}
            refined['recovery_start'] = recovery_start
            attempts.append(refined)
    for point in points:
        optimize(point)
    recovery_starts = 0
    if not any((item['converged'] for item in attempts)):
        ranked = np.argsort(np.where(np.isfinite(scores), scores, np.inf), kind='stable')
        for index in ranked:
            point = starts[index]
            if not np.isfinite(scores[index]) or any((np.array_equal(point, p) for p in points)):
                continue
            recovery_starts += 1
            optimize(point, recovery_start=True)
            if any((item['converged'] for item in attempts)) or recovery_starts == 4:
                break
    valid = [item for item in attempts if item['converged']]
    if not valid:
        raise RuntimeError('Follow-up optimization failed: ' + repr([{k: v for k, v in item.items() if k != 'theta'} for item in attempts]))
    fit = min(valid, key=lambda x: x['objective'])
    fit['starts'] = len(points) + recovery_starts
    fit['optimization_attempts'] = len(attempts)
    fit['recovery_starts'] = recovery_starts
    h = curvature(fit['theta'])
    eigenvalues = np.linalg.eigvalsh(h)
    fit['laplace_valid'] = bool(eigenvalues.min() > 0)
    fit['hessian_eigenvalues'] = eigenvalues
    fit['covariance'] = np.linalg.inv(h) if fit['laplace_valid'] else np.full_like(h, np.nan)
    return fit


def predict(data, matrix, theta, kind, scale, integration='exponential'):
    probability, exposure = predict_jit(jnp.asarray(theta),
        fitting_data(data),
        jnp.asarray(matrix),
        kind,
        scale,
        integration)
    probability = np.asarray(probability)
    exposure = float(exposure)
    if not np.isfinite(probability).all() or not np.isfinite(exposure) or (integration == 'euler' and exposure >= 1):
        raise ValueError(f'Invalid fitted {integration} probability/exposure: {exposure}')
    return (probability, exposure)
