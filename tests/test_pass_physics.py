"""Independent kinematic references, analytic flights and control conservation."""
import math
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import expit
from PRESS.physics.angles import compute_angle
from PRESS.physics.times import time_to_intercept,rotation_time
from PRESS.physics.arrival_probabilities import F
from PRESS.physics.flight_ball import (ball_trajectory,build_flight_lookup,
    exact_ball_arrival_times,match_ball_player_time,player_arrival_time)
from PRESS.model.solver_pitch_control import (_control_update,integrate_pitch_control,
    integrate_pass_prob,isotonic_scaling)
from PRESS.utils.grid import compute_exit_point,generate_circular_grid


def reference_tti(positions,velocities,targets,maximum,acceleration,turn,preserve):
    out=[]
    for pos,vel in zip(positions,velocities):
        speed=math.hypot(*vel);line=[]
        for target in targets:
            delta=target-pos;distance=math.hypot(*delta)
            if distance<=.1:line.append(0.);continue
            angle=0. if speed<.1 else math.acos(np.clip(np.dot(vel,delta)/(speed*distance),-1.,1.))
            ta=max((maximum-speed)/acceleration,0.);da=speed*ta+.5*acceleration*ta*ta
            move=(2*distance/(speed+math.sqrt(speed*speed+2*acceleration*distance)) if distance<=da else
                ta+(distance-da)/(max(speed,maximum) if preserve else maximum))
            line.append(move+speed*angle/turn)
        out.append(line)
    return np.array(out)


@pytest.mark.parametrize('maximum',[5.,5.5,7.])
@pytest.mark.parametrize('turn',[3.5,7.,14.])
@pytest.mark.parametrize('preserve',[False,True])
def test_tti_against_independent_scalar_reference(maximum,turn,preserve):
    rng=np.random.default_rng(20261018)
    p=rng.uniform(-30,30,(24,2));v=rng.normal(0,5,(24,2));v[0]=0.
    targets=np.r_[p[:2],rng.uniform(-40,40,(13,2))]
    actual=np.asarray(time_to_intercept(p,v,targets,v_max=maximum,a_max=7.,a_turn=turn,
        preserve_initial_overspeed=preserve))
    expected=reference_tti(p,v,targets,maximum,7.,turn,preserve)
    np.testing.assert_allclose(actual,expected,atol=1e-3,rtol=1e-6)
    assert np.isfinite(actual).all() and np.all(actual>=0)


def test_geometry_angles_translation_and_rotation():
    p=jnp.array([[0.,0.]]);v=jnp.array([[3.,0.]])
    targets=jnp.array([[10.,0.],[0.,10.],[-10.,0.],[0.,0.]])
    angle=np.asarray(compute_angle(p,v,targets))[0]
    np.testing.assert_allclose(angle,[0,np.pi/2,np.pi,0],atol=3e-4)
    rot=np.array([[0.,-1.],[1.,0.]])
    transformed=compute_angle(p@rot.T+17,v@rot.T,targets@rot.T+17)
    np.testing.assert_allclose(transformed,angle[None],atol=3e-4)
    np.testing.assert_allclose(rotation_time(p,v,targets,7),3*angle[None]/7,atol=1e-7)


@pytest.mark.parametrize('dispersion',[.1,.43,1.])
def test_logistic_arrival_scale_and_derivative(dispersion):
    p=jnp.array([[0.,0.]]);v=jnp.array([[2.,0.]]);target=jnp.array([[5.,0.]])
    tau=float(time_to_intercept(p,v,target,5.,7.)[0,0]);scale=np.pi/(np.sqrt(3)*dispersion)
    times=np.array([tau-1,tau,tau+1])
    actual=np.array([float(F(p,v,target,t,dispersion,5.,7.)[0,0]) for t in times])
    np.testing.assert_allclose(actual,expit(scale*(times-tau)),atol=5e-7)
    derivative=float(jax.grad(lambda t:F(p,v,target,t,dispersion,5.,7.)[0,0])(tau))
    np.testing.assert_allclose(derivative,scale/4,rtol=3e-6)


@pytest.mark.parametrize('speed',[5.,15.,25.])
def test_ground_drag_matches_closed_form(speed):
    t=np.linspace(0,10,251,dtype=np.float32);k=.5*1.22*.25*.038/.42
    tr=np.asarray(ball_trajectory(speed,0.,jnp.asarray(t),rho=1.22,Cd=.25,A=.038,m=.42))
    np.testing.assert_allclose(tr[:,0],np.log1p(k*speed*t)/k,atol=3e-4,rtol=2e-5)
    np.testing.assert_allclose(tr[:,2],speed/(1+k*speed*t),atol=2e-4,rtol=2e-5)
    np.testing.assert_allclose(tr[:,1],0,atol=1e-7)


@pytest.mark.parametrize('theta',[.15,.4,.75])
def test_airborne_no_drag_matches_projectile_equations(theta):
    speed=15.;z0=.5;end=.8*speed*np.sin(theta)/9.81
    t=np.linspace(0,end,41,dtype=np.float32)
    tr=np.asarray(ball_trajectory(speed,theta,jnp.asarray(t),z0=z0,Cd=0.))
    np.testing.assert_allclose(tr[:,0],speed*np.cos(theta)*t,atol=1e-4)
    np.testing.assert_allclose(tr[:,1],z0+speed*np.sin(theta)*t-.5*9.81*t*t,atol=1e-4)


def test_lookup_spacing_and_exact_noninteger_radius():
    times=jnp.arange(31)*.1
    trajectories=jnp.stack([jnp.stack([10*times,jnp.zeros_like(times),10+jnp.zeros_like(times),jnp.zeros_like(times)],axis=1)])
    radii,lookup=build_flight_lookup(12.3,trajectories,times,1.,2.)
    np.testing.assert_allclose(np.diff(radii),1.,atol=1e-7)
    approximate=match_ball_player_time(jnp.array([12.3]),jnp.array([1.23]),lookup,1.)
    exact=exact_ball_arrival_times(jnp.array([12.3]),jnp.array([1.23]),trajectories,times)
    np.testing.assert_allclose(approximate,[1.2],atol=1e-6)
    np.testing.assert_allclose(exact,[1.23],atol=1e-6)
    assert np.isinf(exact_ball_arrival_times(jnp.array([31.]),jnp.array([4.]),trajectories,times)[0])


def test_receiver_is_nearest_not_fastest_and_empty_team_is_unreachable():
    target=jnp.array([[0.,0.]])
    positions=jnp.array([[1.,0.],[2.,0.]]);vel=jnp.array([[5.,0.],[-5.,0.]])
    times=np.asarray(time_to_intercept(positions,vel,target,5.,7.))[:,0]
    assert times[0]>times[1]
    np.testing.assert_allclose(player_arrival_time(target,positions,vel,5.,7.),times[:1])
    assert np.isinf(player_arrival_time(target,jnp.empty((0,2)),jnp.empty((0,2)),5.,7.)[0])


@pytest.mark.parametrize('rate',[0.,4.3,30.,1000.])
def test_constant_cause_exponential_update_matches_analytic_solution(rate):
    gates=jnp.array([[.2,.3],[.8,.7]])
    p=_control_update(jnp.zeros((2,2)),gates,rate,.17,'exponential')
    np.testing.assert_allclose(p,(1-np.exp(-rate*.17))*gates,rtol=3e-6,atol=1e-7)
    assert np.all(np.asarray(p).sum(axis=0)<=1+2e-7)


def test_control_excludes_carrier_before_allocating_high_rate_mass():
    players=jnp.zeros((22,2));grid=jnp.zeros((1,2))
    p=np.asarray(integrate_pitch_control(grid,players,players,10,jnp.zeros(1),
        lambda_control=30.,s=.43,a_max=7.,t_max_pc=.04,dt_pc=.04))
    assert p[10,0]==0 and p.sum()<=1+1e-7
    np.testing.assert_allclose(p[np.arange(22)!=10,0],(1-np.exp(-.04*30*21*.5))/21,atol=1e-7)


def test_unreachable_target_and_airborne_ball_have_zero_control():
    p=jnp.zeros((3,2));grid=jnp.zeros((1,2))
    control=integrate_pitch_control(grid,p,p,0,jnp.array([jnp.inf]),t_max_pc=.2)
    np.testing.assert_allclose(control,0.,atol=0)
    ball=jnp.array([[0.,0.,3.],[0.,0.,3.]])
    np.testing.assert_allclose(integrate_pass_prob(p,p,jnp.array([0,1]),0,ball),0.,atol=0)


def test_isotonic_matches_sklearn_interpolation_and_rejects_invalid_map():
    from sklearn.isotonic import IsotonicRegression
    model=IsotonicRegression(out_of_bounds='clip').fit([.1,.2,.4,.6,.8],[0,1,0,1,1])
    p=np.linspace(0,1,51)
    np.testing.assert_allclose(isotonic_scaling(p,model.X_thresholds_,model.y_thresholds_),model.predict(p),atol=1e-14)
    with pytest.raises(ValueError):isotonic_scaling(p,[0,.5,1],[0,.8,.6])


def test_grid_circle_and_stationary_exit_are_finite():
    grid=np.asarray(generate_circular_grid(3.,.5))
    assert np.isfinite(grid).all() and np.max(np.linalg.norm(grid,axis=1))<=3.+1e-6
    bounds=jnp.array([[-52.5,52.5],[-34.,34.]])
    np.testing.assert_allclose(compute_exit_point(jnp.zeros(2),jnp.array([1.,1.]),bounds),[34.,34.])
    np.testing.assert_allclose(compute_exit_point(jnp.zeros(2),jnp.zeros(2),bounds),[0.,0.])
    np.testing.assert_allclose(compute_exit_point(jnp.array([60.,0.]),jnp.array([1.,0.]),bounds),[60.,0.])
    np.testing.assert_allclose(compute_exit_point(jnp.array([60.,0.]),jnp.array([-1.,0.]),bounds),[-52.5,0.])
