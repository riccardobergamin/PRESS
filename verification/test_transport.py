"""Regression checks for the retained protected-reservoir transport solver."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expit
from PRESS.physics.transport import RayMotion
from PRESS.physics.transport_reservoir import transport_probabilities


def constant_rates(values):
    return lambda zeta,time: np.broadcast_to(values,(*zeta.shape,len(values)))


def test_ray_inverse_acceleration_cruise_rest_and_overspeed():
    for speed in [0.,2.,7.,9.]:
        motion=RayMotion(speed,7.,7.)
        distance=np.array([0,.001,.1,1,10,40])
        np.testing.assert_allclose(motion.distance(motion.travel_time(distance)),distance,atol=1e-13)


@pytest.mark.parametrize('time',[0.,.1,1.,2.,5.,np.inf])
def test_no_killing_initial_condition_inflow_and_absorption(time):
    D,tau0,s=2.,.4,.3;kappa=np.pi/(np.sqrt(3)*s)
    F=lambda x,t:expit(kappa*(t-(tau0+x)))
    result=transport_probabilities(D,tau0,s,constant_rates([]),time=time)
    Z=1-F(D,0)
    np.testing.assert_allclose(result['accessibility'],(F(D,time)-F(D,0))/Z,atol=1e-14)
    np.testing.assert_allclose(result['reservoir'],(1-F(0,time))/Z,atol=1e-14)
    np.testing.assert_allclose(result['survival'],(1-F(D,time))/Z,atol=1e-14)
    assert result['max_mass_error']<1e-14


def test_constant_killing_matches_characteristic_integral():
    D,tau0,s,k=2.,.4,.3,.7;kappa=np.pi/(np.sqrt(3)*s)
    F=lambda t:expit(kappa*(t-(tau0+D)))
    Z=1-F(0)
    expected=quad(lambda t:kappa*F(t)*(1-F(t))*np.exp(-k*t)/Z,0,D,epsabs=1e-13)[0]
    expected+=(1-F(D))/Z*np.exp(-k*D)
    result=transport_probabilities(D,tau0,s,constant_rates([k]),arrival_order=384)
    np.testing.assert_allclose(result['accessibility'],expected,rtol=0,atol=1e-8)
    np.testing.assert_allclose(result['accessibility']+result['interception'].sum(axis=-1),1,atol=1e-14)


def test_proportional_causes_and_finite_time_conservation():
    for time in [0,.1,1,4,np.inf]:
        result=transport_probabilities([1.,2.],[0.,.3],.3,constant_rates([.2,.6]),time=time)
        np.testing.assert_allclose(result['interception'][:,1],3*result['interception'][:,0],atol=1e-14)
        np.testing.assert_allclose(result['accessibility']+result['survival']+result['interception'].sum(-1),1,atol=1e-14)
        assert (result['interception']>=0).all()


def test_stronger_killing_reduces_accessibility_and_deterministic_limit():
    values=[transport_probabilities(2.,1.,.3,constant_rates([k]))['accessibility'][0] for k in [0,.5,2,20]]
    assert np.all(np.diff(values)<0)
    result=transport_probabilities(2.,1.,.002,constant_rates([.7]),arrival_order=96)
    np.testing.assert_allclose(result['accessibility'],np.exp(-1.4),atol=1e-8)


def test_time_varying_killing_converges():
    rates=lambda z,t: np.stack([.1+z*z,.2+np.sin(t)**2],axis=-1)
    values=[transport_probabilities(2.,.4,.3,rates,arrival_order=n,path_order=n) for n in [24,48,96]]
    assert abs(values[-1]['accessibility'][0]-values[-2]['accessibility'][0])<2e-5
    assert all(v['max_mass_error']<1e-13 for v in values)


def test_invalid_intensities_and_parameters():
    with pytest.raises(ValueError):transport_probabilities(0,0,.3,constant_rates([1]))
    with pytest.raises(ValueError):transport_probabilities(1,0,.3,constant_rates([-1]))
    with pytest.raises(ValueError):RayMotion(0,0,7)


def test_space_time_killing_matches_independent_density_and_exit_flux():
    """Directly integrate the appendix's p_k and j_L, independently of labels.

    Constant transport speed 1 makes distance equal to travel-time coordinate.
    k(x,t)=a+b*x+c*t has an analytic line integral on each characteristic.
    This detects wrong initial conditioning, entry times and absolute clocks.
    """
    D, delay, s, T = 2., .4, .3, 1.3
    a,b,c = .1,.2,.3
    kappa = np.pi/(np.sqrt(3)*s)
    F = lambda x,t: expit(kappa*(t-delay-x))
    Z = 1-F(D,0)
    def weight(x,t):
        entry = max(0,t-x)
        start = x-t+entry
        dose = a*(t-entry)+b*(x*x-start*start)/2+c*(t*t-entry*entry)/2
        return np.exp(-dose)
    expected_interior = quad(lambda x:kappa*F(x,T)*(1-F(x,T))/Z*weight(x,T),0,D,epsabs=1e-12)[0]
    expected_arrival = quad(lambda t:kappa*F(D,t)*(1-F(D,t))/Z*weight(D,t),0,T,epsabs=1e-12)[0]
    rates = lambda x,t:(a+b*x+c*t)[...,None]
    result = transport_probabilities(D,delay,s,rates,time=T,arrival_order=192,path_order=32)
    np.testing.assert_allclose(result['interior'],expected_interior,atol=2e-10,rtol=0)
    np.testing.assert_allclose(result['accessibility'],expected_arrival,atol=2e-10,rtol=0)
    np.testing.assert_allclose(result['reservoir'],(1-F(0,T))/Z,atol=1e-14)


def test_boundary_current_and_time_zero_initial_law():
    """At entry the killing path shrinks to zero; J approaches j_in."""
    D,delay,s,T = 2.,.4,.3,.7
    kappa=np.pi/(np.sqrt(3)*s)
    F=lambda x,t: expit(kappa*(t-delay-x))
    Z=1-F(D,0)
    result=transport_probabilities(D,delay,s,constant_rates([2.]),time=0)
    np.testing.assert_allclose(result['interior'],(F(0,0)-F(D,0))/Z,atol=1e-14)
    np.testing.assert_allclose(result['interception'],0,atol=1e-14)
    # On a unit-speed ray, the appendix gives J_k(x,T)=F_t(x,T)/Z exp(-2x).
    inflow=kappa*F(0,T)*(1-F(0,T))/Z
    currents=[kappa*F(x,T)*(1-F(x,T))/Z*np.exp(-2*x) for x in [1e-2,1e-4,1e-6]]
    assert abs(currents[-1]-inflow)<1e-5
    assert abs(currents[-1]-inflow)<abs(currents[0]-inflow)


def test_nonproportional_defender_attribution_converges_to_independent_integral():
    D,delay,s=1.5,.3,.35
    kappa=np.pi/(np.sqrt(3)*s)
    Z=expit(kappa*(delay+D))
    def cause_given_exit(exit_time):
        entry=max(0,exit_time-D)
        begin=max(0,D-exit_time)
        def integrand(t):
            z=begin+t-entry
            dose=.1*(t-entry)+.05*(z*z-begin*begin)+.1*(t*t-entry*entry)
            return (.1+.1*z)*np.exp(-dose)
        return quad(integrand,entry,exit_time,epsabs=1e-11)[0]
    def outer(t):
        density=kappa*expit(kappa*(t-delay-D))*expit(-kappa*(t-delay-D))/Z
        return density*cause_given_exit(t)
    expected=quad(outer,0,D,epsabs=1e-10)[0]+quad(outer,D,np.inf,epsabs=1e-10)[0]
    rates=lambda z,t:np.stack([.1+.1*z,.2*t],axis=-1)
    values=[transport_probabilities(D,delay,s,rates,arrival_order=192,path_order=n)['interception'][0,0]
            for n in [32,64,128]]
    assert abs(values[-1]-expected)<1e-4
    assert abs(values[-1]-expected)<abs(values[0]-expected)
