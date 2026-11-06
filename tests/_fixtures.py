"""Deterministic artificial inputs used only by isolated unit tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_test_match():
    """Return fresh tracking, team intervals, actions, and an (n_x, n_y) xT grid.

    All values are synthetic. Tracking has 22 players at 10 Hz, two team
    possessions and a carrier change inside the first team possession.
    The xT grid is illustrative, not learned from SkillCorner events.
    """
    home = np.array([
        [-48, 0], [-30, -20], [-30, -7], [-30, 7], [-30, 20],
        [-10, -20], [-10, -7], [-10, 7], [-10, 20], [12, -10], [14, 12],
    ], dtype=float)
    away = np.array([
        [48, 0], [30, -20], [30, -7], [30, 7], [30, 20],
        [5, -20], [5, -7], [5, 7], [5, 20], [-16, -10], [-18, 12],
    ], dtype=float)
    rows = []
    for frame in range(40):
        carrier = "home_7" if frame < 10 else "home_8" if frame < 30 else "away_7"
        row = {"frame": frame, "timestamp": frame / 10, "period": 1,
               "player_possession": carrier,
               "attacking_side": "left_to_right" if frame < 30 else "right_to_left",
               "match_clock_min": frame / 600, "score_difference_prefix": 0}
        for side, positions in (("home", home), ("away", away)):
            for number, (x, y) in enumerate(positions, 1):
                vx = 0.4 if side == "home" else -0.2
                prefix = f"{side}_{number}"
                row.update({f"{prefix}_x": x + vx * frame / 10,
                            f"{prefix}_y": y, f"{prefix}_vx": vx,
                            f"{prefix}_vy": 0.0, f"{prefix}_v": abs(vx)})
        row["ball_x"] = row[f"{carrier}_x"]
        row["ball_y"] = row[f"{carrier}_y"]
        rows.append(row)
    possessions = pd.DataFrame([
        {"possession_id": 1, "team_id": "home", "frame_start": 0, "frame_end": 29, "lost": True},
        {"possession_id": 2, "team_id": "away", "frame_start": 30, "frame_end": 39, "lost": False},
    ])
    actions = pd.DataFrame([
        {"frame_start": f, "frame_end": min(f + 9, 39), "action": "ball_drive",
         "start_type": "pass_reception" if f < 30 else "recovery",
         "previous_action": "pass_reception" if f in (0, 10) else "ball_drive",
         "time_since_start": 1.0 if f == 20 else 0.0}
        for f in (0, 10, 20, 30)
    ])
    xt = np.repeat(np.linspace(0.001, 0.12, 16)[:, None], 12, axis=1)
    return pd.DataFrame(rows), possessions, actions, xt
