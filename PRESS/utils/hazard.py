"""
PRESS.utils.hazard

Instantaneous hazard h(s_t) -- the marginal (action-unconditional)
expected risk of losing possession at the current checkpoint, before
knowing whether the ball carrier chooses to pass or drive:

    h(s_t) = P(drive|s_t) * drive_risk(s_t)
           + P(pass|s_t)  * pass_failure_expected(s_t)

where:
    drive_risk(s_t) = P(drive fails) = Ball drive loss probability
                       (no inversion needed: drive_risk = 1 -
                       ball_drive_prob, and ball_drive_prob = 1 - loss
                       under the action_features/ convention, so
                       drive_risk simplifies exactly to the raw loss
                       probability -- NOT to be confused with the
                       ball_drive_prob inversion story discussed for
                       action_selection.py / transition.py; here the
                       double inversion cancels out)

    pass_failure_expected(s_t) = 1 - rppcf
                       (rppcf = sum(tr * pass_prob) over the
                       destination grid, already used elsewhere --
                       tr is a softmax that sums to 1, so
                       sum(tr*(1-pass_prob)) = 1 - rppcf, no need to
                       recompute the per-cell arrays)

NOTE (25/9): NO post-hoc calibration here. An isotonic calibrator was
fitted and validated out-of-fold (ECE 0.0201 -> 0.0032), but the four
components feeding h(s_t) are all PRODUCTION models (fit on all 10
matches) -- so both h(s_t) and its calibration were evaluated
in-sample relative to the underlying component training, not
genuinely out-of-fold. Decided not to ship a calibration that can't
be properly validated: h(s_t) is reported raw.
"""

import numpy as np


def compute_hazard_raw(
    ball_drive_loss,
    transition_probabilities,
    pass_success_probability,
    action_selection_pass_probability,
):
    """
    ball_drive_loss                     : scalar,
                                           self.results["PI"][frame_id]["Ball drive loss probability"]
    transition_probabilities (tr)       : (N,), self.results["PPCF"][frame_id]["Transition probabilities"]
    pass_success_probability (pass_prob): (N,), self.results["PPCF"][frame_id]["Pass probability"]
    action_selection_pass_probability   : scalar, self.results["ACTION_SELECTION"][frame_id]["Pass probability"]

    Returns (hazard_h, drive_risk, pass_failure_expected, rppcf) --
    all scalar floats.
    """

    tr = np.asarray(transition_probabilities)
    pass_prob = np.asarray(pass_success_probability)

    # drive_risk = 1 - ball_drive_prob, with ball_drive_prob = 1 - loss
    # (action_features/ convention) -- simplifies to the raw loss
    # probability, no inversion needed here.
    drive_risk = float(np.asarray(ball_drive_loss))

    rppcf = float(np.sum(tr * pass_prob))
    pass_failure_expected = 1.0 - rppcf

    p_pass = float(action_selection_pass_probability)
    p_drive = 1.0 - p_pass

    hazard_h = p_drive * drive_risk + p_pass * pass_failure_expected

    return hazard_h, drive_risk, pass_failure_expected, rppcf
