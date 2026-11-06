"""Diagnostic PPCF likelihood accumulating both outcomes in log space.

No probability floor and no parameter-dependent integration cutoff. Keep the
submitted model.py and its frozen bundles separate until this audit is closed.
"""
from functools import partial
import jax
import jax.numpy as jnp

def _add_log_mass(left, right):
    both_zero = jnp.isneginf(left) & jnp.isneginf(right)
    value = jnp.logaddexp(jnp.where(both_zero, 0.0, left), jnp.where(both_zero, 0.0, right))
    return jnp.where(both_zero, -jnp.inf, value)

@partial(jax.jit, static_argnames=('dt',))
def predict_logs(theta, tau, t_ball, valid, *, dt=0.04):
    """Return log P(attack), log P(not attack), and log residual survival."""
    log_rate, log_s = theta
    times = jnp.arange(int(round(15 / dt)), dtype=jnp.float32) * jnp.float32(dt)
    n = len(tau)

    def step(state, t):
        log_survival, log_attack, log_defense = state
        log_f = jax.nn.log_sigmoid(jnp.pi * (t - tau) / (jnp.sqrt(3.0) * jnp.exp(log_s)))
        log_f = jnp.where(valid, log_f, -jnp.inf)
        la = jax.scipy.special.logsumexp(log_f[:, :10], axis=1)
        ld = jax.scipy.special.logsumexp(log_f[:, 11:], axis=1)
        lg = jnp.logaddexp(la, ld)
        log_h = jnp.log(dt) + log_rate + lg
        log_mass_factor = jnp.where(log_h < -20.0, log_h, jnp.log(-jnp.expm1(-jnp.exp(jnp.maximum(log_h, -20.0)))))
        active = t >= t_ball
        log_mass = jnp.where(active, log_survival + log_mass_factor, -jnp.inf)
        exposure = jnp.where(active, jnp.exp(log_h), 0.0)
        return ((log_survival - exposure, _add_log_mass(log_attack, log_mass + la - lg), _add_log_mass(log_defense, log_mass + ld - lg)), None)
    (ls, la, ld), _ = jax.lax.scan(jax.checkpoint(step), (jnp.zeros(n), jnp.full(n, -jnp.inf), jnp.full(n, -jnp.inf)), times)
    return (la, jnp.logaddexp(ld, ls), ls)

@jax.jit
def nll_gradient(theta, tau, t_ball, valid, outcome):

    def nll(x):
        la, lf, _ = predict_logs(x, tau, t_ball, valid)
        return -jnp.sum(jnp.where(outcome == 1, la, lf))
    return jax.value_and_grad(nll)(theta)
