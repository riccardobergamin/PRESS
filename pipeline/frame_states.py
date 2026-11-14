"""Read actual tracking frames through the same PRESS getter in every task.

This is only input preparation: no physics model, target or cohort filter.
Caller supplies the matched carrier and the required frame (carry start or
pass release). Provider IDs must first be mapped to tracking aliases.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from PRESS.utils.state import get_pv_players


def state_at_frame(tracking, frame, carrier, attacking_side=None):
    """One actual frame -> PRESS state, with the matched event's carrier.

    The event identifies who executes the action. This matters at a release,
    where a tracking annotation can already name the next carrier.
    Never shift the frame or substitute future positions.
    """
    snapshot = tracking.loc[tracking.frame.eq(frame)].copy()
    snapshot["player_possession"] = carrier
    if attacking_side is not None:
        snapshot["attacking_side"] = attacking_side
    state = get_pv_players(frame, snapshot)
    # Same accepted numerical convention and player order in all review tasks:
    # teammates 0:10, carrier 10, opponents 11:22; missing players padded by PRESS.
    for name in ["x_bc", "v_bc", "r_atts", "v_atts", "r_defs", "v_defs"]:
        state[name] = np.asarray(state[name], np.float32)
    state["p_pos"] = np.vstack([state["r_atts"], state["x_bc"], state["r_defs"]])
    state["p_vel"] = np.vstack([state["v_atts"], state["v_bc"], state["v_defs"]])
    return state


def load_tracking_states(rows, tracking_root, *, frame_column, carrier_column):
    """Read the requested states in exactly the input table's order.

    Ball drive: frame_start and player_id (already a tracking alias).
    Pass probability: frame_end and player (already a tracking alias).
    Pass selection maps its provider player_id, then calls state_at_frame.
    Each match is loaded once. The returned direction is observed at the frame.
    """
    rows = rows.reset_index(drop=True)
    states = {name: np.empty((len(rows), *shape), np.float32) for name, shape in dict(
        x_bc=(2,), v_bc=(2,), r_defs=(11, 2), v_defs=(11, 2),
        p_pos=(22, 2), p_vel=(22, 2),
    ).items()}
    directions = np.empty(len(rows), dtype=object)
    for match_id, match_rows in rows.groupby("match_id", sort=False):
        path = Path(tracking_root) / "matches" / str(match_id) / f"{match_id}_tracking.parquet"
        tracking = pd.read_parquet(path)
        # Keep only requested frames in memory during repeated state extraction.
        tracking = tracking.loc[tracking.frame.isin(match_rows[frame_column])]
        for index, record in match_rows.iterrows():
            state = state_at_frame(tracking, int(record[frame_column]), record[carrier_column])
            for name in states:
                states[name][index] = state[name]
            directions[index] = state["attacking_side"]
    return states, directions
