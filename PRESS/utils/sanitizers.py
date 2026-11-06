"""Validate canonical match tables without changing their values."""

import re

import numpy as np
import pandas as pd


def validate_tracking_data(tracking_data):
    """Check wide tracking IDs, numeric columns and observed player states.

    Missing players and missing carriers are allowed. An observed player
    needs finite XY and velocity components; absent/padded players are not
    treated as observed. Coordinates use centred metres. Derived speed is
    optional in the public constructor and normalized before this validator.
    """
    if not isinstance(tracking_data, pd.DataFrame):
        raise ValueError("tracking_data must be a pandas DataFrame.")
    required = {"frame", "player_possession", "attacking_side", "ball_x", "ball_y"}
    missing = required - set(tracking_data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    values = pd.to_numeric(tracking_data.frame, errors="coerce")
    if (not pd.api.types.is_numeric_dtype(tracking_data.frame) or
            not np.isfinite(values).all() or (values != np.floor(values)).any()):
        raise ValueError("frame must contain finite integer frame IDs.")
    if tracking_data.frame.duplicated().any():
        raise ValueError("Frame column contains duplicates.")
    players = {}
    for column in tracking_data:
        match = re.fullmatch(r"(home|away)_(\d+)_(x|y|vx|vy|v)", column)
        if match:
            players.setdefault(f"{match[1]}_{match[2]}", set()).add(match[3])
    if not players:
        raise ValueError("No player columns found.")
    for player, columns in players.items():
        missing = {"x", "y", "vx", "vy", "v"} - columns
        if missing:
            raise ValueError(f"{player} missing columns: {sorted(missing)}")
    if not tracking_data.player_possession.dropna().isin(players).all():
        raise ValueError("player_possession contains unknown players.")
    direction = tracking_data.attacking_side.dropna()
    if not direction.isin(["left_to_right", "right_to_left"]).all():
        raise ValueError("attacking_side must be left_to_right, right_to_left or missing.")
    columns = ["ball_x", "ball_y"] + [f"{p}_{c}" for p, variables in players.items() for c in variables]
    nonnumeric = [c for c in columns if not pd.api.types.is_numeric_dtype(tracking_data[c])]
    if nonnumeric:
        raise ValueError(f"Non numeric tracking columns: {nonnumeric}")
    if np.isinf(tracking_data[columns].to_numpy(dtype=float)).any():
        raise ValueError("Tracking positions and velocities cannot contain infinite values.")
    for player in players:
        xy = tracking_data[[f"{player}_x", f"{player}_y"]].to_numpy(float)
        observed = np.isfinite(xy).all(axis=1) & (np.linalg.norm(xy, axis=1) < 500)
        velocity = tracking_data[[f"{player}_vx", f"{player}_vy"]].to_numpy(float)
        if not np.isfinite(velocity[observed]).all():
            raise ValueError(f"Observed player {player} requires finite velocity components.")


def validate_events_data(events_data):
    """Validate optional canonical events; raw provider events need an adapter."""
    if not isinstance(events_data, pd.DataFrame):
        raise ValueError("events_data must be a pandas DataFrame.")
    required = {"frame", "player_id", "team_id", "event_type"}
    missing = required - set(events_data.columns)
    if missing:
        raise ValueError(f"Missing event columns: {sorted(missing)}")
    frames = pd.to_numeric(events_data.frame, errors="coerce")
    if not np.isfinite(frames).all() or (frames != np.floor(frames)).any():
        raise ValueError("Event frame IDs must be finite integers.")
    players = events_data.player_id.astype(str)
    if not players.str.fullmatch(r"(home|away)_\d+").all():
        raise ValueError("Events require canonical home_<id>/away_<id> player IDs.")
    if not events_data.team_id.isin(["home", "away"]).all():
        raise ValueError("Event team_id must be home or away.")
    if not players.str.split("_").str[0].eq(events_data.team_id).all():
        raise ValueError("player_id and team_id are inconsistent.")
    passes = events_data[events_data.event_type.eq("pass")]
    if not passes.empty:
        if (
            "is_successful" not in passes
            or not passes.is_successful.map(lambda v: isinstance(v, (bool, np.bool_))).all()
        ):
            raise ValueError("Pass events require boolean is_successful.")
        completed = passes[passes.is_successful]
        if not completed.empty:
            if (
                "receiver_player_id" not in completed
                or not completed.receiver_player_id.astype(str).str.fullmatch(r"(home|away)_\d+").all()
            ):
                raise ValueError("Successful pass events require receiver_player_id.")
    return True
