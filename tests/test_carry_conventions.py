"""Closed-form cases for optional carry dynamics, preserving legacy defaults."""
import numpy as np
import jax.numpy as jnp
import pytest
from PRESS.physics.times import time_to_intercept
from PRESS.model.solver_intensity import _carry_hazard_components, integrate_loss_prob


def test_observed_overspeed_is_preserved_without_further_acceleration():
    p=jnp.array([[0.,0.]]);v=jnp.array([[8.,0.]])
    target=jnp.array([[16.,0.]])
    legacy=float(time_to_intercept(p,v,target,v_max=7.,a_max=7.)[0,0])
    kept=float(time_to_intercept(p,v,target,v_max=7.,a_max=7.,preserve_initial_overspeed=True)[0,0])
    # Historical compute_angle adds epsilons to normalized vector norms;
    # float64 retains a tiny nonzero turn even for exactly aligned vectors.
    assert legacy==pytest.approx(16/7,abs=1e-4)
    assert kept==pytest.approx(2.,abs=1e-4)
    assert legacy-kept==pytest.approx(16/7-2.,abs=2e-7)


def test_separate_turn_scale_changes_only_fixed_position_turn_delay():
    p=jnp.array([[0.,0.]]);v=jnp.array([[7.,0.]]);target=jnp.array([[0.,14.]])
    tied=float(time_to_intercept(p,v,target,v_max=7.,a_max=7.)[0,0])
    faster=float(time_to_intercept(p,v,target,v_max=7.,a_max=7.,a_turn=14.)[0,0])
    assert tied==pytest.approx(2+np.pi/2,abs=2e-7)
    assert faster==pytest.approx(2+np.pi/4,abs=2e-7)


def test_forward_acceleration_from_rest_has_analytic_solution():
    got=time_to_intercept(jnp.zeros((1,2)),jnp.zeros((1,2)),jnp.array([[3.5,0.],[10.5,0.]]),v_max=7.,a_max=7.,a_turn=2.,preserve_initial_overspeed=True)
    np.testing.assert_allclose(got,[[1.,2.]],atol=1e-7)


def test_signed_margin_does_not_rebound_after_crossing_touchline():
    p=jnp.array([[0.,30.]]);v=jnp.zeros((1,2));bc=jnp.array([0.,33.]);vb=jnp.array([0.,2.])
    eta={mode:float(_carry_hazard_components(1.,p,v,bc,vb,1.,0.,1.,boundary_mode=mode)[2])
         for mode in ['absolute','signed','clipped']}
    assert eta==pytest.approx(dict(absolute=1.,signed=-1.,clipped=0.))
    gates=[np.asarray(_carry_hazard_components(1.,p,v,bc,vb,1.,0.,1.,boundary_mode=m)[0]) for m in eta]
    np.testing.assert_array_equal(gates[0],gates[1]);np.testing.assert_array_equal(gates[0],gates[2])


def test_options_propagate_to_integrator_and_do_not_force_boundary_loss():
    p=jnp.array([[0.,30.]]);v=jnp.array([[8.,0.]]);bc=jnp.array([0.,33.]);vb=jnp.array([0.,2.])
    for method in ['euler','exponential']:
        for mode in ['absolute','signed','clipped']:
            q=np.asarray(integrate_loss_prob(p,v,bc,vb,1.,.2,.01,-.02,v_max=7.,a_max=7.,a_turn=10.,
                preserve_initial_overspeed=True,boundary_mode=mode,integration_method=method))
            assert np.isfinite(q).all() and 0<=q.sum()<1
    with pytest.raises(ValueError,match='boundary_mode'):
        _carry_hazard_components(0.,p,v,bc,vb,1.,0.,1.,boundary_mode='bad')
