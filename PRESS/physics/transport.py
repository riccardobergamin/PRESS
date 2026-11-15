"""Characteristic transport with competing killing, including latent waiting.

The logistic reference law is conditioned only on not having reached the
right boundary at time zero. Waiting at the physical origin is represented
by negative transport coordinates and remains exposed to interception.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import expit


@lru_cache(maxsize=16)
def unit_quadrature(order):
    """Positive Gauss--Legendre nodes/weights on (0,1)."""
    if not isinstance(order, (int, np.integer)) or order < 2:
        raise ValueError("Quadrature order must be an integer >= 2.")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return (nodes + 1) / 2, weights / 2


@dataclass(frozen=True)
class RayMotion:
    """The appendix's straight-ray movement and its analytic inverse.

    The additive turning delay is handled separately by the transport solver.
    Initial overspeed is preserved.
    The strictly increasing travel law has no near-origin distance cutoff;
    at zero speed its inverse remains regular while tau'(0) is unbounded.
    """
    speed: float
    max_speed: float
    acceleration: float

    def __post_init__(self):
        for name in ("speed",):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        for name in ("max_speed", "acceleration"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")

    def travel_time(self, distance):
        distance = np.asarray(distance, float)
        if not np.isfinite(distance).all() or (distance < 0).any():
            raise ValueError("Ray distance must be finite and nonnegative.")
        if self.speed >= self.max_speed:
            return distance / self.speed
        ta = (self.max_speed - self.speed) / self.acceleration
        da = self.speed * ta + .5 * self.acceleration * ta**2
        return np.where(distance <= da,
                        (np.sqrt(self.speed**2 + 2*self.acceleration*distance)-self.speed)/self.acceleration,
                        ta + (distance-da)/self.max_speed)

    def distance(self, travel_time):
        time = np.asarray(travel_time, float)
        if not np.isfinite(time).all() or (time < -1e-12).any():
            raise ValueError("Travel time must be finite and nonnegative.")
        time = np.maximum(time, 0)
        if self.speed >= self.max_speed:
            return self.speed * time
        ta = (self.max_speed-self.speed)/self.acceleration
        da = self.speed*ta + .5*self.acceleration*ta**2
        return np.where(time <= ta, self.speed*time + .5*self.acceleration*time**2,
                        da + self.max_speed*(time-ta))


def transport_probabilities(duration, turning_delay, dispersion, rates, *,
                            time=np.inf, arrival_order=48, path_order=64,
                            waiting_rates=None):
    """Transport with interception both while waiting and along the route.

    duration : scalar or (R,)
        Positive movement time from the physical origin to each destination.
    turning_delay : scalar or (R,)
        Additive delay tau(0), affecting the logistic reference weights.
    dispersion : float
        Standard deviation of logistic arrival-time uncertainty (seconds).
    rates(zeta, t) : callable
        Nonnegative intensities (R,Q,K,D). zeta and t have shape (R,Q,K).
        zeta is the SIGNED transport-time coordinate: it increases at unit
        speed, and zeta < 0 represents waiting at the observed origin.
        Absolute time t starts at zero, including during waiting.
    waiting_rates(t) : callable, optional
        Equivalent rates while waiting at the physical origin. Allows BCOS
        to reuse its fixed origin TTI instead of evaluating it at every node.
        Omit for generic signed-coordinate killing rates.
    time : float
        Nonnegative finite evaluation time, or infinity for terminal risks.

    A virtual no-killing exit time e labels each initial characteristic.
    Its signed coordinate at time t is zeta=duration-e+t. The conditioned
    exit law has support e>0. Every path starts at time zero: unlike the
    protected-reservoir solver, killing also acts for zeta<0.

    Integrate over the conditional exit CDF, avoiding an artificial spatial
    truncation of the negative half-line. Split paths at entry zeta=0 and
    integrate their waiting and moving segments in chronological order.
    Positive quadrature increments and exponential competing-risk updates
    preserve nonnegativity and discrete probability conservation. No
    probability renormalization or unresolved-tail deletion is applied.
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
    initial_entry_split = conditional_cdf(duration)
    absorption = np.zeros_like(duration)
    interior = np.zeros_like(duration)
    reservoir = np.zeros_like(duration)
    interception = None

    def integrate_interval(left, right, region):
        nonlocal absorption, interior, reservoir, interception
        width = right-left
        if not np.any(width > 0):
            return
        labels = left[:, None] + width[:, None]*q
        reference_survival = (1-labels)*Z[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            exits = tau[:, None] + (np.log1p(-reference_survival)
                                    -np.log(reference_survival))/kappa
        exits = np.where(width[:, None] > 0, np.maximum(exits, 0), 0)
        stop = np.minimum(time, exits)
        entry = np.maximum(exits-duration[:, None], 0)
        waiting_end = np.minimum(entry, stop)
        alive = np.ones_like(exits)
        causes = None

        # In zeta<0 the BCOS callback maps every latent position to the
        # observed origin. A generic callback may depend on signed zeta.
        for waiting, begin, end in [(True, np.zeros_like(stop), waiting_end),
                                    (False, waiting_end, stop)]:
            # A zero-length segment contributes exactly zero exposure.
            if not np.any(end > begin):
                continue
            absolute_time = begin[..., None] + (end-begin)[..., None]*x
            zeta = duration[:, None, None]-exits[..., None]+absolute_time
            intensity = np.asarray(
                waiting_rates(absolute_time) if waiting and waiting_rates is not None
                else rates(zeta, absolute_time), float,
            )
            if intensity.ndim != 4 or intensity.shape[:3] != zeta.shape:
                raise ValueError("rates must return shape (rays, arrival nodes, path nodes, causes).")
            if not np.isfinite(intensity).all() or (intensity < 0).any():
                raise ValueError("Killing intensities must be finite and nonnegative.")
            total = intensity.sum(axis=-1)
            increments = (end-begin)[..., None]*xw
            dose = total*increments
            cumulative = np.cumsum(dose, axis=-1)
            before = alive[..., None]*np.exp(-(cumulative-dose))
            removed = before*(-np.expm1(-dose))
            proportions = np.divide(intensity,total[..., None],
                out=np.zeros_like(intensity),where=total[..., None]>0)
            removed_by_cause = (removed[..., None]*proportions).sum(axis=-2)
            causes = removed_by_cause if causes is None else causes+removed_by_cause
            alive *= np.exp(-dose.sum(axis=-1))

        weight = width[:, None]*qw
        if causes is not None:
            contribution = (weight[..., None]*causes).sum(axis=1)
            interception = contribution if interception is None else interception+contribution
        surviving = (weight*alive).sum(axis=1)
        if region == 'arrived':
            absorption += surviving
        elif region == 'interior':
            interior += surviving
        else:
            reservoir += surviving

    # Exact probability partitions at the observation time. Unlike the old
    # solver, future-entry paths must also accumulate interception risk.
    for left, right, region in [
        (np.zeros_like(arrived), arrived, 'arrived'),
        (arrived, entered, 'interior'),
        (entered, np.ones_like(entered), 'waiting'),
    ]:
        middle = np.clip(initial_entry_split, left, right)
        integrate_interval(left, middle, region)
        integrate_interval(middle, right, region)
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
