"""Closed-form checks of the shared individual arrival-time dynamics."""
import numpy as np
import jax.numpy as jnp
import pytest
from PRESS.physics.times import time_to_intercept


def test_observed_overspeed_is_preserved_without_further_acceleration():
    p=jnp.array([[0.,0.]]);v=jnp.array([[8.,0.]])
    target=jnp.array([[16.,0.]])
    legacy=float(time_to_intercept(p,v,target,v_max=7.,a_max=7.,preserve_initial_overspeed=False)[0,0])
    kept=float(time_to_intercept(p,v,target,v_max=7.,a_max=7.,preserve_initial_overspeed=True)[0,0])
    default=float(time_to_intercept(p,v,target,v_max=7.,a_max=7.)[0,0])
    assert default == kept
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
