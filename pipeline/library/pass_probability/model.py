"""Fixed-target PPCF likelihood; historical Euler and safe competing-risk update."""
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
LAMBDA0 = 4.3
S0 = 0.43
PARAMETER_BOUNDS = np.array([[0.1, 30.0], [0.05, 2.0]])

@partial(jax.jit, static_argnames=('method', 'dt'))
def predict(log_parameters, tau, t_ball, valid, *, method='exponential', dt=0.04):
    rate, dispersion = jnp.exp(log_parameters)
    times = jnp.arange(int(round(15 / dt)), dtype=jnp.float32) * jnp.float32(dt)
    if method == 'euler':

        def step(state, t):
            prob, max_mass = state
            total = prob.sum(axis=1)
            gate = jax.nn.sigmoid(jnp.pi * (t - tau) / (jnp.sqrt(3.0) * dispersion)) * valid
            inc = rate * jnp.maximum(1 - total, 0)[:, None] * gate * (t >= t_ball)[:, None]
            updated = jnp.clip(prob + dt * inc, 0, 1)
            updated = jnp.where((total < 1 - 1e-06)[:, None], updated, prob)
            return ((updated, jnp.maximum(max_mass, updated.sum(axis=1))), None)
        (prob, max_mass), _ = jax.lax.scan(step, (jnp.zeros_like(tau, dtype=log_parameters.dtype), jnp.zeros(len(tau))), times)
        return (prob[:, :10].sum(axis=1), prob.sum(axis=1), max_mass)
    if method != 'exponential':
        raise ValueError('Unknown integration')

    def step(state, t):
        survival, att = state
        gate = jax.nn.sigmoid(jnp.pi * (t - tau) / (jnp.sqrt(3.0) * dispersion)) * valid
        ga = gate[:, :10].sum(axis=1)
        g = gate.sum(axis=1)
        active = t >= t_ball
        exposure = jnp.where(active, dt * rate * g, 0.0)
        mass = survival * -jnp.expm1(-exposure)
        return ((survival * jnp.exp(-exposure), att + mass * ga / jnp.maximum(g, 1e-300)), None)
    (survival, att), _ = jax.lax.scan(jax.checkpoint(step), (jnp.ones(len(tau)), jnp.zeros(len(tau))), times)
    att = jnp.clip(att, 0.0, 1 - survival)
    return (att, 1 - survival, jnp.ones(len(tau)))

def posterior(log_parameters, tau, t_ball, valid, outcome):
    p = predict(log_parameters, tau, t_ball, valid)[0]
    p = jnp.clip(p, 1e-06, 1 - 1e-06)
    nll = -jnp.sum(outcome * jnp.log(p) + (1 - outcome) * jnp.log1p(-p))
    prior = 0.5 * jnp.sum(((log_parameters - jnp.log(jnp.array([LAMBDA0, S0]))) / jnp.array([1.5, 1.0])) ** 2)
    return nll + prior
value_gradient = jax.jit(jax.value_and_grad(posterior))
hessian = jax.jit(jax.hessian(posterior))

def fit(tau, t_ball, valid, outcome, mode):
    if not jax.config.x64_enabled:
        raise ValueError('JAX_ENABLE_X64=1 required')
    free = {'lambda_only': [0], 's_only': [1], 'joint': [0, 1]}[mode]
    fixed = np.log([LAMBDA0, S0])
    args = tuple((jnp.asarray(x) for x in [tau, t_ball, valid, outcome]))
    bounds = np.log(PARAMETER_BOUNDS)

    def expand(x):
        return np.array([x[free.index(i)] if i in free else fixed[i] for i in range(2)])

    def objective(x):
        value, gradient = value_gradient(jnp.asarray(expand(x)), *args)
        return (float(value), np.asarray(gradient)[free])
    starts = [fixed, np.log([1.0, 0.2]), np.log([10.0, 0.8])]
    attempts = []
    for start in starts:
        result = minimize(objective, start[free], jac=True, method='L-BFGS-B', bounds=bounds[free], options=dict(maxiter=1000, ftol=1e-12, gtol=1e-05, maxls=40))
        val, grad = objective(result.x)
        projected = grad.copy()
        for k, (lo, hi) in enumerate(bounds[free]):
            if result.x[k] <= lo + 1e-07 and grad[k] > 0 or (result.x[k] >= hi - 1e-07 and grad[k] < 0):
                projected[k] = 0
        attempts.append(dict(objective=val, log_parameters=expand(result.x).tolist(), success=bool(result.success), message=str(result.message), iterations=int(result.nit), gradient_max=float(np.max(abs(grad))), projected_gradient_max=float(np.max(abs(projected)))))
    accepted = [r for r in attempts if r['success'] and np.isfinite(r['objective']) and (r['projected_gradient_max'] < 0.01)]
    if not accepted:
        raise ValueError('Fit failed: ' + repr(attempts))
    best = min(accepted, key=lambda r: r['objective'])
    theta = np.asarray(best['log_parameters'])
    curvature = np.asarray(hessian(jnp.asarray(theta), *args))[np.ix_(free, free)]
    eigenvalues = np.linalg.eigvalsh((curvature + curvature.T) / 2)
    values = np.exp(theta)
    return dict(**best, mode=mode, lambda_control=float(values[0]), s=float(values[1]), attempts=attempts, hessian_eigenvalues=eigenvalues.tolist(), positive_hessian=bool(eigenvalues.min() > 0), at_parameter_bound=bool(np.any(np.isclose(theta[free], bounds[free, 0], atol=1e-05) | np.isclose(theta[free], bounds[free, 1], atol=1e-05))), prior='Independent weak normals on log(lambda/4.3)/1.5 and log(s/.43)/1; lambda [.1,30], s [.05,2]; numerical bounds, not a plausible physical range')
