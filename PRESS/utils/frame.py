"""Canonical frame conversion for plotting and inspection."""

import re

import numpy as np
import pandas as pd


def frame_to_long(frame_df):
    """Return observed players from one wide frame as a new long table.

    Ball and frame metadata are repeated for each player. Missing carriers
    are allowed, so in-flight frames can be plotted without inventing one.
    Positions beyond 500 m are internal missing-player padding and omitted.
    """
    if len(frame_df) != 1:
        raise ValueError("frame_df must contain exactly one frame.")
    frame = frame_df.iloc[0]
    carrier = frame.get("player_possession")
    team = str(carrier).split("_")[0] if pd.notna(carrier) else None
    global_columns = [
        "frame",
        "timestamp",
        "period",
        "ball_x",
        "ball_y",
        "ball_z",
        "player_possession",
    ]
    rows = []
    for column in frame.index:
        match = re.fullmatch(r"(home|away)_(\d+)_x", column)
        if match is None:
            continue
        side, number = match.groups()
        player = f"{side}_{number}"
        xy = np.array([frame[f"{player}_x"], frame[f"{player}_y"]], dtype=float)
        if not np.isfinite(xy).all() or np.linalg.norm(xy) >= 500:
            continue
        row = {key: frame[key] for key in global_columns if key in frame}
        row.update(team_possession=team, team_id=side, shirt_num=int(number),
                   player_id=player, x=xy[0], y=xy[1],
                   vx=frame.get(f"{player}_vx", np.nan),
                   vy=frame.get(f"{player}_vy", np.nan),
                   v=frame.get(f"{player}_v", np.nan))
        rows.append(row)
    return pd.DataFrame(rows, columns=global_columns + [
        "team_possession", "team_id", "shirt_num", "player_id", "x", "y", "vx", "vy", "v",
    ])


def resolve_pitch_size(frame_df, pitch_length=None, pitch_width=None):
    """Resolve centred-metre plot dimensions from frame metadata or overrides."""
    row = frame_df.iloc[0]
    length = float(row.get("pitch_length", 105) if pitch_length is None else pitch_length)
    width = float(row.get("pitch_width", 68) if pitch_width is None else pitch_width)
    if not np.isfinite([length, width]).all() or min(length, width) <= 0:
        raise ValueError("Pitch dimensions must be finite and positive.")
    return length, width
