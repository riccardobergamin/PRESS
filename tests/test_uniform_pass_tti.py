"""Final pass artifact uses one overspeed convention through every interface."""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import expit
from PRESS.model import PassRiskModel
from PRESS.physics.times import time_to_intercept
from PRESS.physics.flight_ball import player_arrival_time,time_flight
from PRESS.model.solver_pitch_control import integrate_pass_prob,integrate_pitch_control


def state():
    return (np.array([[0.,0.],[3.,0.],[11.,0.],[8.,3.]],dtype=np.float32),
            np.array([[0.,0.],[8.,0.],[-1.,0.],[4.,-2.]],dtype=np.float32))


def test_final_receiver_tti_keeps_initial_overspeed():
    p,v=state();target=jnp.array([[11.,0.]])
    actual=player_arrival_time(target,jnp.asarray(p[1:2]),jnp.asarray(v[1:2]),5.5,7.,
        a_turn=7.,preserve_initial_overspeed=True)
    np.testing.assert_allclose(actual,[1.],atol=1e-3)
    default=player_arrival_time(target,jnp.asarray(p[1:2]),jnp.asarray(v[1:2]),5.5,7.)
    np.testing.assert_allclose(default,actual,atol=1e-7)
    legacy=player_arrival_time(target,jnp.asarray(p[1:2]),jnp.asarray(v[1:2]),5.5,7.,
        preserve_initial_overspeed=False)
    assert float(legacy[0])-float(actual[0])>.4


def test_low_level_path_control_accepts_same_final_tti():
    p,v=state();p=jnp.asarray(p);v=jnp.asarray(v)
    ball=jnp.array([[11.,0.,0.],[11.,0.,0.],[11.,0.,0.]])
    tau=np.asarray(time_to_intercept(p,v,ball[:1,:2],v_max=5.5,a_max=7.,a_turn=7.,
        preserve_initial_overspeed=True))[:,0]
    mass=np.zeros(4);survival=1.
    for t in np.arange(3)*.04:
        rate=4.3*expit(np.pi*(t-tau)/(np.sqrt(3.)*.3));rate[0]=0.
        increment=survival*(-np.expm1(-.04*rate.sum()))
        mass+=increment*rate/rate.sum();survival*=np.exp(-.04*rate.sum())
    actual=integrate_pass_prob(p,v,jnp.array([0,1]),0,ball,lambda_control=4.3,s=.3,
        v_max=5.5,a_max=7.,a_turn=7.,preserve_initial_overspeed=True,dt=.04)
    np.testing.assert_allclose(actual,mass[:2].sum(),atol=1e-10,rtol=1e-5)


def test_installed_field_uses_final_tti_for_flight_and_target_control():
    model=PassRiskModel.pretrained();settings=model.physics;p,v=state()
    assert settings['preserve_initial_overspeed'] is True
    target=np.array([[11.,0.],[13.,4.]],dtype=np.float32)
    result=model.predict_field(p,v,p[0],target,np.array([0,1]),0)
    context=getattr(jax,'enable_x64',None)
    if context is None:
        from jax.experimental import enable_x64 as context
    with context(False):
        p=jnp.asarray(p);v=jnp.asarray(v);target=jnp.asarray(target)
        flight=time_flight(p[0],target,p[1:2],v[1:2],
            v0_candidates=jnp.linspace(5.,25.,30,dtype=jnp.float32),
            theta_candidates=jnp.linspace(0.,jnp.pi/4,30,dtype=jnp.float32),
            **{k:settings[k] for k in ['v_max','a_max','a_turn','preserve_initial_overspeed',
                't_max_flight','dt_ball','z_max','rho','Cd','A','m','g','distance_mode']})
    expected=integrate_pitch_control(target,p,v,0,flight,lambda_control=model.lambda_control,s=model.s,
        v_max=5.5,a_max=7.,a_turn=7.,preserve_initial_overspeed=True,t_max_pc=15.,dt_pc=.01)
    np.testing.assert_allclose(result['flight_time'],flight,atol=2e-5,rtol=2e-6)
    np.testing.assert_allclose(result['raw'],np.asarray(expected)[:2].sum(axis=0),atol=2e-6,rtol=2e-6)
    assert result['calibration_matches_physics']
