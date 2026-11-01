"""Evaluate any usable frame, using synthetic or processed local match data."""
from __future__ import annotations

import argparse
import numpy as np

from PRESS import PRESS
from PRESS.datasets import load_demo_match


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-dir", help="Processed SkillCorner match directory")
    parser.add_argument("--actions", help="Multi-match actions.csv")
    parser.add_argument("--xt", help="Match-derived xt_grid.npy, shape (n_x, n_y)")
    parser.add_argument("--possession", type=int, help="Team possession ID from get_possessions()")
    parser.add_argument("--frame", type=int, help="Any usable frame ID inside the selected possession")
    parser.add_argument("--start-type", help="Explicit actual event context, raw or standardized")
    parser.add_argument("--previous-action", help="Explicit actual preceding-action category")
    args = parser.parse_args()
    if args.match_dir:
        if not args.xt:
            parser.error("--match-dir requires --xt")
        model = PRESS.from_skillcorner(args.match_dir, actions_path=args.actions)
        xt = np.load(args.xt)
        options = {}
    else:
        tracking, possessions, actions, xt = load_demo_match()
        model = PRESS(tracking, possessions_data=possessions, actions_data=actions)
        # Coarse grids make the synthetic demo fast; use defaults for research.
        options = {"ppcf_kwargs": {"r_max": 12, "step": 6},
                   "bcos_kwargs": {"r_max": 8, "step": 4}}
    if args.frame is None:
        checkpoints = model.get_checkpoints(args.possession)
        if checkpoints.empty:
            parser.error("No default annotated frame; provide --frame and action context")
        frame = int(checkpoints.frame_id.iloc[0])
    else:
        frame = args.frame
    if frame not in set(model.get_frames(args.possession, valid_only=True).frame):
        parser.error("--frame must be a usable tracking frame inside the selected possession")
    result = model.compute_frame(frame, xt, start_type=args.start_type,
                                 previous_action=args.previous_action, **options)
    print("Frame:", frame)
    print("P(pass):", result["ACTION_SELECTION"]["Pass probability"])
    print("Hazard:", result["HAZARD"])


if __name__ == "__main__":
    main()
