"""One-scalar temperature on frozen final choice probabilities; validation only."""
import numpy as np
from scipy.optimize import minimize
from pipeline.library.pass_selection.common import groups

def evaluate(logp, sizes, chosen, beta=1.0, gradient=False):
    logp = np.asarray(logp, np.float64)
    starts, sizes = groups(sizes)
    chosen = np.asarray(chosen, np.int64)
    assert len(logp) == sizes.sum() and beta >= 0 and np.isfinite(logp).all()
    z = beta * logp
    maximum = np.maximum.reduceat(z, starts)
    exp = np.exp(z - np.repeat(maximum, sizes))
    den = np.add.reduceat(exp, starts)
    loss = maximum + np.log(den) - beta * logp[starts + chosen]
    assert np.isfinite(loss).all()
    if not gradient:
        return loss
    grad = np.add.reduceat(exp * logp, starts) / den - logp[starts + chosen]
    return (float(loss.mean()), float(grad.mean()))

def fit_temperature(logp, sizes, chosen):
    before = float(evaluate(logp, sizes, chosen).mean())
    result = minimize(lambda x: evaluate(logp, sizes, chosen, float(x[0]), True), np.array([1.0]), jac=True, method='L-BFGS-B', bounds=[(0.0, None)], options=dict(maxiter=200, gtol=1e-08, ftol=1e-12))
    beta = float(result.x[0])
    after, gradient = evaluate(logp, sizes, chosen, beta, True)
    projected = abs(gradient) if beta > 1e-10 else max(-gradient, 0.0)
    assert result.success and np.isfinite(beta) and (projected < 1e-05) and (after <= before + 1e-10), (result, gradient)
    return dict(method='Temperature on final choice distribution', inverse_temperature=beta, temperature=None if beta == 0 else 1.0 / beta, uniform_limit=beta == 0, initial_inverse_temperature=1.0, bounds='beta>=0; no finite upper bound', validation_ce_before=before, validation_ce_after=after, projected_gradient=projected, success=bool(result.success), iterations=int(result.nit), message=str(result.message), trees_refitted=False, rho_refitted=False, test_labels_used=False)
