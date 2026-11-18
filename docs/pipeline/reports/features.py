"""08. Canonical states -> one row per possession -> pressure patterns.

No new model. Every h comes from the outer fold holding out that match.
Input metadata is observed possession timing, identities and terminal labels.
"""

import numpy as np
import pandas as pd

from pipeline.io import PARENT_KEYS
from pipeline.composed_risk.scores import THRESHOLDS, BAND_NAMES


def classify_possessions(states):
    """Return the band-cohort states and one classification row per parent.

    A micropossession with one canonical point uses it as both ends. Longer
    possessions require a checkpoint at the true start for initial-state
    patterns. A pass's final state is release; a direct loss's final state is
    the beginning of the last failed attempt. Never use the post-loss frame.
    """
    states = states.loc[states.band_eligible].copy()
    medium, high = THRESHOLDS[1:]
    states["high"] = states.h.ge(high)
    states["band"] = np.array(BAND_NAMES)[np.searchsorted(THRESHOLDS, states.h, side="right")]
    states["pass_contribution"] = states.p_pass * states.pass_failure
    states["carry_contribution"] = (1-states.p_pass) * states.drive_loss
    rows = []
    for (match, possession), group in states.sort_values("frame_start").groupby(PARENT_KEYS, sort=False):
        first, last = group.iloc[0], group.iloc[-1]
        duration = (first.parent_end_frame-first.parent_start_frame)/10.
        initial = group.loc[group.frame_start.eq(first.parent_start_frame) & group.action.eq("ball_drive")]
        has_true_initial = len(initial) > 0
        micro = duration < 1 and len(group) == 1
        if not has_true_initial and micro:
            initial = group.iloc[[0]]
        initial_h = float(initial.h.iloc[0]) if len(initial) else np.nan
        if first.parent_end_type == "pass":
            terminal_ok = last.action == "pass" and last.frame_start == first.parent_end_frame
        else:
            terminal_ok = (first.parent_end_type == "possession_loss" and last.action == "ball_drive"
                           and last.h_loss == 1 and 0 <= first.parent_end_frame-last.frame_start <= 10)
        if not terminal_ok:
            raise ValueError(f"Parent without terminal canonical action: {match}/{possession}")
        retained = first.terminal_loss == 0
        reception = str(first.parent_start_type).endswith("_reception")
        escape_eligible = len(initial) > 0 and initial_h >= medium
        drawing_eligible = len(initial) > 0 and initial_h < medium and reception
        attempted = group.loc[group.action.eq("ball_drive") & group.h.ge(medium)]
        successful = attempted.loc[attempted.outcome.eq(1) & attempted.duration_frames.eq(10)]
        rows.append(dict(
            match_id=match, possession_id=possession,
            provider_player_id=int(first.provider_player_id), player_id=first.player_id,
            player_name=first.player_name, team_id=int(first.team_id), team_name=first.team_name,
            n_checkpoints=len(group), duration_s=duration,
            parent_start_frame=int(first.parent_start_frame), parent_end_frame=int(first.parent_end_frame),
            next_control_frame=first.get("next_control_frame", np.nan),
            initial_checkpoint_frame=int(initial.frame_start.iloc[0]) if len(initial) else np.nan,
            final_checkpoint_frame=int(last.frame_start), has_initial=bool(len(initial)),
            has_true_initial=has_true_initial, single_micro_checkpoint=micro, has_final=True,
            start_h=initial_h, end_h=float(last.h), maximum_h=float(group.h.max()),
            peak_band=BAND_NAMES[int(np.searchsorted(THRESHOLDS, group.h.max(), side="right"))],
            elevated=bool(group.h.ge(medium).any()), high=bool(group.high.any()),
            retained=retained, terminal_individual_loss=int(first.terminal_loss),
            terminal_end_type=first.parent_end_type, start_type=first.parent_start_type,
            is_reception=reception, escape_eligible=escape_eligible,
            pressure_escape=bool(escape_eligible and duration >= 1 and last.h < medium and retained),
            drawing_eligible=drawing_eligible,
            drawing_pressure=bool(drawing_eligible and duration >= 2 and last.h >= high),
            n_elevated_carry_attempts=len(attempted), n_elevated_successful_carries=len(successful),
            sustained_eligible=len(attempted) > 0, sustained_successful_carrying=len(successful) >= 2,
        ))
    return states, pd.DataFrame(rows)
