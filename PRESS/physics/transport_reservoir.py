"""Previous transport solver: no interception before route entry.

Retained for explicit comparisons with an externally protected reservoir.
Movement and quadrature primitives are shared with the active solver.
"""
import numpy as np
from scipy.special import expit
from .transport import unit_quadrature


def transport_probabilities(duration, turning_delay, dispersion, rates, *,
                            time=np.inf, arrival_order=48, path_order=64):
    """Integrate destination, defender and at-risk probabilities for rays.

    duration : scalar or (R,) -- travel time along each ray, without turning.
    turning_delay : scalar or (R,) -- tau(0), the constant delay on each ray.
    dispersion : arrival-time logistic standard deviation, in seconds.
    rates(zeta, t) : nonnegative intensities with shape (R,Q,K,D).
        zeta and t have shape (R,Q,K), travel-time coordinate and absolute
        time. Q labels no-killing exit times; K labels path quadrature nodes;
        D is the number of competing defenders (possibly zero).
    time : finite nonnegative evaluation time, or infinity for terminal risks.

    Output arrays have leading shape (R,). Initial conditioning is Z=1-F(L,0).
    For each virtual exit a, the active path starts at max(0,a-duration),
    and zeta=t-a+duration. No killing acts before entering the domain.
    Gauss weights are positive time increments for an exponential competing-
    risk update. This gives nonnegative probabilities and exact discrete
    conservation at any order. The hazard integral is Gauss quadrature;
    defender attribution converges with the ordered, frozen-rate updates.
    """
    duration, delay = np.broadcast_arrays(np.atleast_1d(duration).astype(float),
                                         np.atleast_1d(turning_delay).astype(float))
    if duration.ndim != 1 or not np.isfinite(duration).all() or (duration <= 0).any():
        raise ValueError("duration must be a positive finite scalar or vector.")
    if not np.isfinite(delay).all() or (delay < 0).any():
        raise ValueError("turning_delay must be finite and nonnegative.")
    if not np.isfinite(dispersion) or dispersion <= 0:
        raise ValueError("dispersion must be finite and positive.")
    if np.isnan(time) or time < 0:
        raise ValueError("time must be nonnegative or infinity.")
    q, qw = unit_quadrature(arrival_order)
    x, xw = unit_quadrature(path_order)
    kappa = np.pi/(np.sqrt(3)*dispersion)
    tau = delay + duration
    initial_right = expit(-kappa*tau)
    Z = 1-initial_right

    def conditional_cdf(t):
        return np.clip((expit(kappa*(t-tau))-initial_right)/Z, 0, 1)

    arrived = conditional_cdf(time)
    entered = conditional_cdf(time+duration)
    split = conditional_cdf(duration)  # initial-line vs inflow characteristics
    reservoir = (1-entered) if np.isfinite(time) else np.zeros_like(duration)
    absorption = np.zeros_like(duration)
    interior = np.zeros_like(duration)
    interception = None

    def integrate_interval(left, right, complete):
        nonlocal absorption, interior, interception
        width = right-left
        if not np.any(width > 0):
            return
        labels = left[:, None] + width[:, None]*q
        # Survival form of the inverse logistic avoids subtracting near-one CDFs.
        reference_survival = (1-labels)*Z[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            exits = tau[:, None] + (np.log1p(-reference_survival)-np.log(reference_survival))/kappa
        exits = np.where(width[:, None] > 0, np.maximum(exits, 0), 0)
        begin = np.maximum(0, duration[:, None]-exits)
        stop_time = np.minimum(time, exits)
        end = np.clip(duration[:, None]-exits+stop_time, begin, duration[:, None])
        zeta = begin[..., None] + (end-begin)[..., None]*x
        absolute_time = np.maximum(0, exits[..., None]-duration[:, None, None]+zeta)
        intensity = np.asarray(rates(zeta, absolute_time), float)
        if intensity.ndim != 4 or intensity.shape[:3] != zeta.shape:
            raise ValueError("rates must return shape (rays, arrival nodes, path nodes, causes).")
        if not np.isfinite(intensity).all() or (intensity < 0).any():
            raise ValueError("Killing intensities must be finite and nonnegative.")
        total = intensity.sum(axis=-1)
        increments = (end-begin)[..., None]*xw
        dose = total*increments
        cumulative = np.cumsum(dose, axis=-1)
        before = np.exp(-(cumulative-dose))
        removed = before*(-np.expm1(-dose))
        proportions = np.divide(intensity,total[..., None],out=np.zeros_like(intensity),where=total[..., None]>0)
        causes = (removed[..., None]*proportions).sum(axis=-2)
        alive = np.exp(-dose.sum(axis=-1))
        weight = width[:, None]*qw
        contribution = (weight[..., None]*causes).sum(axis=1)
        interception = contribution if interception is None else interception+contribution
        surviving = (weight*alive).sum(axis=1)
        if complete:
            absorption += surviving
        else:
            interior += surviving

    # Splitting at the initial/inflow transition removes a kink in the outer
    # integrand. Splitting at the observation time treats boundary absorption
    # exactly; the future reservoir is integrated analytically.
    for left,right,complete in [(np.zeros_like(arrived),arrived,True),
                                (arrived,entered,False)]:
        middle = np.clip(split,left,right)
        integrate_interval(left,middle,complete)
        integrate_interval(middle,right,complete)
    if interception is None:
        probe = np.zeros((len(duration),1,1))
        n_causes = np.asarray(rates(probe,probe)).shape[-1]
        interception = np.zeros((len(duration),n_causes))
    survival = reservoir+interior
    mass = absorption+interception.sum(axis=-1)+survival
    error = float(np.max(np.abs(mass-1)))
    if error > 2e-12:
        raise ArithmeticError("Transport probability conservation failed.")
    return dict(accessibility=absorption, interception=interception,
                survival=survival,reservoir=reservoir,interior=interior,
                max_mass_error=error,time=float(time),
                arrival_order=arrival_order,path_order=path_order)
