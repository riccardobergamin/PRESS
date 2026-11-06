"""Compose next-action risk from execution probabilities and a pass/carry policy.

h = (1-P(pass))*drive_loss + P(pass)*sum(choice*(1-pass_success)).
Choice is conditional on passing; completion is the calibrated pass field.
There is no post-hoc calibration of h. Individual terminal loss additionally
requires the continuation term exposed by PRESS.compute_xloss().
"""

import numpy as np


def compute_hazard_raw(
    ball_drive_loss,
    pass_selection_probabilities,
    pass_success_probability,
    action_selection_pass_probability,
):
    """
    ball_drive_loss                     : scalar,
                                           self.results["BALL_DRIVE_RISK"][frame_id]["Ball drive risk"]
    pass_selection_probabilities (tr)       : (N,), self.results["PPCF"][frame_id]["Pass selection probabilities"]
    pass_success_probability (pass_prob): (N,), self.results["PPCF"][frame_id]["Pass probability"]
    action_selection_pass_probability   : scalar, self.results["ACTION_SELECTION"][frame_id]["Pass probability"]

    Returns (hazard_h, drive_risk, pass_failure_expected, rppcf) --
    all scalar floats.
    """

    tr = np.asarray(pass_selection_probabilities)
    pass_prob = np.asarray(pass_success_probability)

    # h combines execution-loss probabilities, so retain the stored loss directly.
    drive_risk = float(np.asarray(ball_drive_loss))

    rppcf = float(np.sum(tr * pass_prob))
    pass_failure_expected = 1.0 - rppcf

    p_pass = float(action_selection_pass_probability)
    p_drive = 1.0 - p_pass

    hazard_h = p_drive * drive_risk + p_pass * pass_failure_expected

    return hazard_h, drive_risk, pass_failure_expected, rppcf
