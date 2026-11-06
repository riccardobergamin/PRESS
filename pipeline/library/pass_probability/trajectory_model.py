"""Stable competing-control likelihood along the reconstructed ball trajectory."""
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from .stable_likelihood import _add_log_mass

@partial(jax.jit, static_argnames=('dt',))
def predict_logs(theta, tau_path, available, valid, *, dt=0.04):
    times = jnp.arange(tau_path.shape[1], dtype=jnp.float32) * jnp.float32(dt)
    n = len(tau_path)

    def step(state, inputs):
        ls, la, ld = state
        t, tau, on = inputs
        lf = jax.nn.log_sigmoid(jnp.pi * (t - tau) / (jnp.sqrt(3.0) * jnp.exp(theta[1])))
        lf = jnp.where(valid, lf, -jnp.inf)
        ga = jax.scipy.special.logsumexp(lf[:, :10], axis=1)
        gd = jax.scipy.special.logsumexp(lf[:, 11:], axis=1)
        g = jnp.logaddexp(ga, gd)
        lh = jnp.log(dt) + theta[0] + g
        lm_factor = jnp.where(lh < -20.0, lh, jnp.log(-jnp.expm1(-jnp.exp(jnp.maximum(lh, -20.0)))))
        lm = jnp.where(on, ls + lm_factor, -jnp.inf)
        return ((ls - jnp.where(on, jnp.exp(lh), 0.0), _add_log_mass(la, lm + ga - g), _add_log_mass(ld, lm + gd - g)), None)
    (ls, la, ld), _ = jax.lax.scan(jax.checkpoint(step), (jnp.zeros(n), jnp.full(n, -jnp.inf), jnp.full(n, -jnp.inf)), (times, jnp.swapaxes(tau_path, 0, 1), jnp.swapaxes(available, 0, 1)))
    return (la, jnp.logaddexp(ld, ls), ls)

def nll(theta, tau_path, available, valid, outcome):
    la, lf, _ = predict_logs(theta, tau_path, available, valid)
    return -jnp.sum(jnp.where(outcome == 1, la, lf))
value_gradient = jax.jit(jax.value_and_grad(nll))
curvature = jax.jit(jax.hessian(nll))

def fit(tau_path, available, valid, outcome):
    """MLE in log coordinates: no prior and no numerical cap at lambda30."""
    if not jax.config.x64_enabled:
        raise ValueError('Enable JAX x64')
    if not valid[:, :10].any(axis=1).all() or not valid[:, 11:].any(axis=1).all():
        raise ValueError('Both teams require at least one valid player')
    args = tuple((jnp.asarray(x) for x in [tau_path, available, valid, outcome]))

    def objective(theta):
        value, gradient = value_gradient(jnp.asarray(theta), *args)
        return (float(value), np.asarray(gradient))
    attempts = []
    for initial in [[4.3, 0.43], [3.0, 0.3], [5.0, 0.5]]:
        result = minimize(objective, np.log(initial), jac=True, method='BFGS', options={'gtol': 0.0001, 'maxiter': 500})
        value, gradient = objective(result.x)
        attempts.append(dict(initial=initial, objective=value, log_parameters=result.x.tolist(), success=bool(result.success), message=str(result.message), iterations=int(result.nit), gradient_max=float(np.max(abs(gradient)))))
    accepted = [a for a in attempts if np.isfinite(a['objective']) and a['gradient_max'] < 0.01 and np.isfinite(a['log_parameters']).all()]
    if not accepted:
        raise ValueError('Unconverged trajectory MLE: ' + repr(attempts))
    best = min(accepted, key=lambda x: x['objective'])
    theta = np.array(best['log_parameters'])
    eigen = np.linalg.eigvalsh(np.asarray(curvature(jnp.asarray(theta), *args)))
    return dict(**best, lambda_control=float(np.exp(theta[0])), s=float(np.exp(theta[1])), estimator='MLE, natural Bernoulli likelihood; unconstrained positive parameters via log', hessian_eigenvalues=eigen.tolist(), weak_identification=bool(eigen.min() < 1e-05), attempts=attempts)
