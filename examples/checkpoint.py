"""Evaluate a frame from a real processed match using the bundled models."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from PRESS import PRESS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-dir", type=Path, required=True,
                        help="Processed SkillCorner match directory")
    parser.add_argument("--actions", type=Path, help="Canonical actions parquet or CSV")
    parser.add_argument("--xt", type=Path,
                        help="Optional xT array; otherwise use the fitted model surface")
    parser.add_argument("--possession", type=int,
                        help="Team possession ID from get_possessions()")
    parser.add_argument("--frame", type=int, help="Any usable frame ID")
    parser.add_argument("--start-type", help="Explicit start-event context")
    parser.add_argument("--previous-action", help="Explicit preceding-action context")
    args = parser.parse_args()

    press = PRESS.from_skillcorner(args.match_dir, actions_path=args.actions)
    xt = np.load(args.xt) if args.xt else None
    if args.frame is None:
        if args.actions is None:
            parser.error("Provide --frame or --actions to select an annotated checkpoint")
        checkpoints = press.get_checkpoints(args.possession)
        if checkpoints.empty:
            parser.error("No eligible checkpoints; provide --frame and its action context")
        frame_id = int(checkpoints.frame_id.iloc[0])
    else:
        frame_id = args.frame
    if frame_id not in set(press.get_frames(args.possession, valid_only=True).frame):
        parser.error("--frame must be a usable tracking frame in the selected interval")

    outputs = press.compute_frame(
        frame_id, xt,
        start_type=args.start_type,
        previous_action=args.previous_action,
    )
    print("Frame:", frame_id)
    print("P(pass):", outputs["ACTION_SELECTION"]["Pass probability"])
    print("Hazard:", outputs["HAZARD"])


if __name__ == "__main__":
    main()
