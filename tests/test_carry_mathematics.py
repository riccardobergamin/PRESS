"""Conservation, exact constant-rate behavior, gradients and turning-time assumptions."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from PRESS.model import solver_intensity as solver
from PRESS.physics.times import time_to_intercept,rotation_time


@pytest.mark.parametrize('duration',[0.,.17,1.])
def test_exponential_euler_matches_constant_cause_hazards(monkeypatch,duration):
    monkeypatch.setattr(solver,'F',lambda *a,**k:jnp.array([[.25],[.75]]))
    result=solver.integrate_loss_prob(jnp.zeros((2,2)),jnp.zeros((2,2)),jnp.zeros(2),jnp.zeros(2),duration,2.,0.,0.)
    expected=(1-np.exp(-2*duration))*np.array([.25,.75])
    np.testing.assert_allclose(result,expected,rtol=2e-6,atol=2e-7)


def test_large_rates_preserve_shared_probability_and_have_finite_gradients():
    positions=jnp.zeros((11,2));velocity=jnp.zeros((11,2));carrier=jnp.zeros(2)
    def risk(beta):
        return solver.integrate_loss_prob(positions,velocity,carrier,jnp.array([3.,0.]),1.,100.,beta,0.).sum()
    for beta in [0.,100.]:
        value,derivative=jax.jit(jax.value_and_grad(risk))(beta)
        assert np.isfinite([value,derivative]).all()
        assert 0<=float(value)<=1+5e-7
    historical=solver.integrate_loss_prob(positions,velocity,carrier,carrier,1.,100.,0.,0.,integration_method='euler')
    assert float(historical.sum())>1  # Reproduction of the archived numerical problem.


def test_legacy_euler_is_explicit_and_keeps_its_previous_update(monkeypatch):
    monkeypatch.setattr(solver,'F',lambda *a,**k:jnp.ones((2,1))*.5)
    result=solver.integrate_loss_prob(jnp.zeros((2,2)),jnp.zeros((2,2)),jnp.zeros(2),jnp.zeros(2),1.,.36,0.,0.,integration_method='euler')
    np.testing.assert_allclose(result.sum(),1-(1-.04*.36)**25,rtol=2e-6)


def test_linear_acceleration_matches_piecewise_equations_and_is_continuous():
    v=3.;maximum=7.;acceleration=7.;ta=(maximum-v)/acceleration;da=v*ta+.5*acceleration*ta**2
    targets=jnp.array([[da-1e-4,0.],[da,0.],[da+1e-4,0.],[10.,0.]])
    got=np.asarray(time_to_intercept(jnp.zeros((1,2)),jnp.array([[v,0.]]),targets,maximum,acceleration))[0]
    # At x64 the historical direction-normalization epsilon induces a tiny
    # aligned-angle delay (~50 microseconds); this is a test of translation.
    turns=np.asarray(rotation_time(jnp.zeros((1,2)),jnp.array([[v,0.]]),targets,acceleration))[0]
    np.testing.assert_allclose(got[1]-turns[1],ta,rtol=1e-6)
    assert np.all(np.diff(got)>0) and got[2]-got[0]<1e-3
    np.testing.assert_allclose(got[-1]-turns[-1],ta+(10-da)/maximum,rtol=1e-6)


def test_rotation_is_constant_speed_centripetal_turning_time():
    p=jnp.zeros((1,2));v=jnp.array([[3.,0.]]);targets=jnp.array([[0.,10.],[-10.,0.]])
    got=np.asarray(rotation_time(p,v,targets,7.))[0]
    np.testing.assert_allclose(got,3*np.array([np.pi/2,np.pi])/7,rtol=1e-6,atol=6e-5)


def test_invalid_integration_method_is_rejected():
    with pytest.raises(ValueError,match='integration_method'):
        solver.integrate_loss_prob(jnp.zeros((1,2)),jnp.zeros((1,2)),jnp.zeros(2),jnp.zeros(2),1.,.36,0.,0.,integration_method='unknown')


@pytest.mark.parametrize('duration',[0.,.17,1.])
def test_log_survival_uses_the_same_partial_steps_and_hazards(monkeypatch,duration):
    monkeypatch.setattr(solver,'F',lambda *a,**k:jnp.array([[.25],[.75]]))
    arguments=(jnp.zeros((2,2)),jnp.zeros((2,2)),jnp.zeros(2),jnp.zeros(2),duration,2.,0.,0.)
    log_s=solver._integrate_log_survival(*arguments)
    np.testing.assert_allclose(log_s,-2*duration,rtol=2e-6,atol=1e-7)
    risk=solver.integrate_loss_prob(*arguments).sum()
    np.testing.assert_allclose(jnp.exp(log_s),1-risk,rtol=3e-6,atol=3e-7)


def test_log_survival_retains_extreme_and_tiny_hazards_with_finite_gradients(monkeypatch):
    monkeypatch.setattr(solver,'F',lambda *a,**k:jnp.array([[1.]]))
    def log_survival(rate):
        return solver._integrate_log_survival(jnp.zeros((1,2)),jnp.zeros((1,2)),jnp.zeros(2),jnp.zeros(2),1.,rate,0.,0.)
    for rate in [1e-10,1000.]:
        value,derivative=jax.jit(jax.value_and_grad(log_survival))(rate)
        np.testing.assert_allclose(value,-rate,rtol=3e-6)
        np.testing.assert_allclose(derivative,-1,rtol=3e-6)
    np.testing.assert_allclose(log_survival(0.),0,atol=1e-30)
