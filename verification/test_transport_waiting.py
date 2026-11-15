"""Independent checks of transport with interception during latent waiting."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expit

from PRESS.physics.transport import transport_probabilities
from PRESS.physics.transport_reservoir import transport_probabilities as protected_transport


def constant_rates(values):
    return lambda zeta, time: np.broadcast_to(values, (*zeta.shape, len(values)))


def exit_law(duration, delay, dispersion):
    kappa = np.pi / (np.sqrt(3) * dispersion)
    tau = duration + delay
    Z = expit(kappa * tau)
    cdf = lambda t: (expit(kappa * (t-tau)) - expit(-kappa*tau)) / Z
    density = lambda t: kappa * expit(kappa*(t-tau)) * expit(-kappa*(t-tau)) / Z
    return cdf, density


@pytest.mark.parametrize('time', [0., .1, 1., 2., 5., np.inf])
def test_no_killing_matches_initial_condition_and_reference_flux(time):
    duration, delay, s = 2., .4, .3
    G, _ = exit_law(duration, delay, s)
    result = transport_probabilities(duration, delay, s, constant_rates([]), time=time)
    np.testing.assert_allclose(result['accessibility'], G(time), atol=2e-15)
    np.testing.assert_allclose(result['reservoir'], 1-G(time+duration), atol=2e-15)
    np.testing.assert_allclose(result['interior'], G(time+duration)-G(time), atol=2e-15)
    assert result['max_mass_error'] < 2e-15


@pytest.mark.parametrize('time', [0., .1, 1., 2., 5., np.inf])
def test_constant_killing_matches_independent_exit_flux_and_density(time):
    duration, delay, s, k = 2., .4, .3, .7
    G, density = exit_law(duration, delay, s)
    expected_arrival = quad(lambda t: density(t)*np.exp(-k*t), 0, time,
                            epsabs=2e-12)[0]
    alive = np.exp(-k*time) if np.isfinite(time) else 0.
    result = transport_probabilities(duration, delay, s, constant_rates([k]),
                                    time=time, arrival_order=384)
    np.testing.assert_allclose(result['accessibility'], expected_arrival, atol=2e-8, rtol=0)
    np.testing.assert_allclose(result['survival'], (1-G(time))*alive, atol=2e-14)
    np.testing.assert_allclose(result['reservoir'], (1-G(time+duration))*alive, atol=2e-14)
    np.testing.assert_allclose(result['interception'], 1-expected_arrival-(1-G(time))*alive,
                               atol=2e-8, rtol=0)


def test_waiting_has_interception_and_reduces_boundary_inflow():
    duration, delay, s, T = 1.5, .8, .3, .7
    a, b = .4, .2
    G, _ = exit_law(duration, delay, s)
    rates = lambda z,t: np.where(z < 0, a+b*t, 0)[..., None]
    result = transport_probabilities(duration, delay, s, rates, time=T)
    # Waiting density is removed from the same reference tail at the origin.
    np.testing.assert_allclose(result['reservoir'],
        (1-G(T+duration))*np.exp(-a*T-b*T*T/2), atol=2e-14)
    assert result['interception'].sum() > 0
    # dR/dt = -j_in-k(0,t)R, with j_in attenuated by waiting survival.
    eps = 1e-5
    before = transport_probabilities(duration, delay, s, rates, time=T-eps)['reservoir'][0]
    after = transport_probabilities(duration, delay, s, rates, time=T+eps)['reservoir'][0]
    _, density = exit_law(duration, delay, s)
    incoming = density(T+duration)*np.exp(-a*T-b*T*T/2)
    np.testing.assert_allclose((after-before)/(2*eps),
        -incoming-(a+b*T)*result['reservoir'][0], atol=2e-10)


@pytest.mark.parametrize('time', [.2, 1.5, 4., np.inf])
def test_route_only_killing_reproduces_protected_solver(time):
    active = lambda z,t: np.stack([.2+.1*z, .3+.2*t], axis=-1)
    extended = lambda z,t: active(np.maximum(z, 0), t)*(z >= 0)[..., None]
    old = protected_transport(2., .4, .3, active, time=time)
    new = transport_probabilities(2., .4, .3, extended, time=time)
    for key in ['accessibility', 'interception', 'reservoir', 'interior', 'survival']:
        np.testing.assert_allclose(new[key], old[key], atol=3e-14)


def test_competing_cause_partition_and_finite_time_conservation():
    for time in [0., .1, 1., 4., np.inf]:
        result = transport_probabilities([1., 2.], [0., 2.], .3,
            constant_rates([.2, .6]), time=time)
        np.testing.assert_allclose(result['interception'][:, 1],
            3*result['interception'][:, 0], atol=2e-14)
        np.testing.assert_allclose(result['accessibility']+result['interception'].sum(-1)
            +result['survival'], 1., atol=2e-14)
        assert np.all(result['interception'] >= 0)
        assert np.all(result['survival'] >= 0)


def test_turning_is_exposed_in_the_small_dispersion_limit():
    duration, delay, k = 2., 1., .7
    result = transport_probabilities(duration, delay, .002, constant_rates([k]), arrival_order=96)
    np.testing.assert_allclose(result['accessibility'], np.exp(-k*(duration+delay)), atol=2e-6)
    old = protected_transport(duration, delay, .002, constant_rates([k]), arrival_order=96)
    assert result['accessibility'][0] < old['accessibility'][0]


def test_waiting_time_changes_do_not_affect_motion_along_the_positive_ray():
    calls = []
    def rates(zeta, time):
        calls.append((zeta.copy(), time.copy()))
        return np.ones((*zeta.shape, 1))
    transport_probabilities(2., 1., .3, rates)
    assert any(np.any(z < 0) for z,t in calls)
    for z,t in calls:
        # A signed characteristic has zeta-t constant independently of delay.
        np.testing.assert_allclose(np.ptp(z-t, axis=-1), 0, atol=2e-15)


def test_space_time_killing_matches_independent_current_and_surviving_mass():
    # Unit transport speed: zeta is the signed spatial coordinate. Negative
    # positions map physically to zero, exactly as in BCOS.
    duration, delay, s, T = 2., .4, .3, 1.3
    a, b, c = .1, .2, .3
    G, density = exit_law(duration, delay, s)
    def weight(exit_time, stop):
        z0 = duration-exit_time
        z1 = z0+stop
        dose = a*stop + b*(max(z1,0)**2-max(z0,0)**2)/2 + c*stop*stop/2
        return np.exp(-dose)
    expected_arrival = quad(lambda e:density(e)*weight(e,e),0,T,epsabs=1e-12)[0]
    expected_interior = quad(lambda e:density(e)*weight(e,T),T,T+duration,
                              points=[duration],epsabs=1e-12)[0]
    expected_waiting = (1-G(T+duration))*np.exp(-a*T-c*T*T/2)
    rates = lambda z,t:(a+b*np.maximum(z,0)+c*t)[..., None]
    result = transport_probabilities(duration, delay, s, rates, time=T,
        arrival_order=192, path_order=32)
    np.testing.assert_allclose(result['accessibility'], expected_arrival, atol=3e-10)
    np.testing.assert_allclose(result['interior'], expected_interior, atol=3e-10)
    np.testing.assert_allclose(result['reservoir'], expected_waiting, atol=3e-14)


def test_nonproportional_causes_converge_to_independent_quadrature():
    duration, delay, s = 1.5, .3, .35
    _, density = exit_law(duration, delay, s)
    def cause_given_exit(e):
        z0 = duration-e
        def integrand(t):
            z = max(z0+t, 0)
            dose = .1*t+.05*(z*z-max(z0,0)**2)+.1*t*t
            return (.1+.1*z)*np.exp(-dose)
        entry = max(e-duration, 0)
        return quad(integrand,0,entry,epsabs=1e-10)[0]+quad(integrand,entry,e,epsabs=1e-10)[0]
    expected = quad(lambda e:density(e)*cause_given_exit(e),0,duration,epsabs=2e-10)[0]
    expected += quad(lambda e:density(e)*cause_given_exit(e),duration,np.inf,epsabs=2e-10)[0]
    rates = lambda z,t:np.stack([.1+.1*np.maximum(z,0), .2*t],axis=-1)
    values = [transport_probabilities(duration,delay,s,rates,arrival_order=192,
              path_order=n)['interception'][0,0] for n in [32,64,128]]
    assert abs(values[-1]-expected) < 1e-4
    assert abs(values[-1]-expected) < abs(values[0]-expected)


def test_invalid_inputs_and_negative_intensities_rejected():
    with pytest.raises(ValueError):
        transport_probabilities(0.,0.,.3,constant_rates([1.]))
    with pytest.raises(ValueError):
        transport_probabilities(1.,0.,.3,constant_rates([-1.]))


def test_exit_quadrature_refinement_matches_analytic_constant_killing():
    _, density = exit_law(2., .4, .3)
    expected = quad(lambda e:density(e)*np.exp(-.7*e), 0, 2., epsabs=1e-13)[0]
    errors = [abs(transport_probabilities(2., .4, .3, constant_rates([.7]),
              time=2., arrival_order=n)['accessibility'][0]-expected)
              for n in [48, 96, 192, 384]]
    assert np.all(np.diff(errors) < 0)
    assert errors[0] < 2e-6
    assert errors[-1] < 1e-8


@pytest.mark.parametrize('T', [0., .5, 3., np.inf])
def test_origin_rate_reuse_matches_generic_transport_exactly(T):
    def rates(zeta, t):
        position = np.maximum(zeta, 0)
        return np.stack([.2 + .05*position + .01*t, .3 + .02*position], axis=-1)
    def waiting(t):
        return rates(np.zeros_like(t), t)
    original = transport_probabilities([1., 3.], [0., .7], .3, rates, time=T)
    reused = transport_probabilities([1., 3.], [0., .7], .3, rates, time=T,
                                     waiting_rates=waiting)
    for field in ['accessibility', 'interception', 'survival', 'reservoir', 'interior']:
        np.testing.assert_array_equal(original[field], reused[field])


def test_zero_length_segments_do_not_evaluate_killing():
    calls = []
    def rates(zeta, t):
        calls.append((zeta, t))
        return np.ones((*t.shape, 1))
    result = transport_probabilities(2., 0., .3, rates, time=0.)
    # One shape probe to determine the number of causes; no quadrature-rate calls.
    assert len(calls) == 1 and calls[0][0].shape == (1, 1, 1)
    np.testing.assert_array_equal(result['interception'], [[0.]])
    np.testing.assert_allclose(result['survival'], [1.], atol=1e-15)
