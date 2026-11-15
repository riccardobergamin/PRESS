"""Contracts for the optional individual-possession continuation."""
import numpy as np
import pytest

from PRESS.model import compose_individual_xloss


def test_continuation_counts_only_successful_drive_branch():
    options = dict(p_pass=0.4, drive_loss=0.1, pass_failure=0.2)
    hazard = 0.4 * 0.2 + 0.6 * 0.1
    assert np.isclose(compose_individual_xloss(**options, residual_loss=0), hazard)
    assert np.isclose(compose_individual_xloss(**options, residual_loss=0.3),
                      hazard + 0.6 * 0.9 * 0.3)
    # A completed pass ends this individual's sequence, regardless of c.
    assert compose_individual_xloss(p_pass=1, drive_loss=0.1,
                                   pass_failure=0, residual_loss=1) == 0


def test_composition_supports_broadcasting_and_absorbing_drive_loss():
    actual = compose_individual_xloss(p_pass=0, drive_loss=np.array([0, 1]),
                                     pass_failure=0.5, residual_loss=0.3)
    assert np.allclose(actual, [0.3, 1])
    with pytest.raises(ValueError, match="broadcastable"):
        compose_individual_xloss(p_pass=[0, 1], drive_loss=[0, 0, 0],
                                 pass_failure=0, residual_loss=0)


@pytest.mark.parametrize("name", ["p_pass", "drive_loss", "pass_failure", "residual_loss"])
def test_composition_rejects_invalid_probabilities(name):
    values = dict(p_pass=0.4, drive_loss=0.1, pass_failure=0.2, residual_loss=0.3)
    for invalid in (-0.1, 1.1, np.nan, np.inf, []):
        values[name] = invalid
        with pytest.raises(ValueError, match=name):
            compose_individual_xloss(**values)
