"""Match navigation and adapters for the processed SkillCorner tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

_RAW_TO_STANDARD = {
    "pass_reception": "reception_open_play",
    "throw_in_reception": "reception_set_piece",
    "corner_reception": "reception_set_piece",
    "free_kick_reception": "reception_set_piece",
    "goal_kick_reception": "reception_set_piece",
    "pass_interception": "interception_open_play",
    "recovery": "interception_open_play",
    "throw_in_interception": "interception_set_piece",
    "corner_interception": "interception_set_piece",
    "free_kick_interception": "interception_set_piece",
    "goal_kick_interception": "interception_set_piece",
    "keep_possession": "continuation",
}
START_CATEGORIES = frozenset(_RAW_TO_STANDARD.values())


def standardize_action_context(value, *, previous=False):
    """Return a trained category, or None for unsupported/missing context.

    Accepts raw SkillCorner categories and already standardized categories.
    ``ball_drive`` is allowed only for the previous action. Unknown context
    is never guessed; it was excluded by the original training pipeline.
    """
    if pd.isna(value):
        return None
    if value in START_CATEGORIES or (previous and value == "ball_drive"):
        return value
    return _RAW_TO_STANDARD.get(value)


def validate_intervals(table, required):
    """Validate integral inclusive frame intervals; return a sorted copy."""
    if not isinstance(table, pd.DataFrame):
        raise ValueError("Interval data must be a pandas DataFrame.")
    missing = set(required) - set(table.columns)
    if missing:
        raise ValueError(f"Missing interval columns: {sorted(missing)}")
    result = table.copy()
    for column in ("frame_start", "frame_end"):
        values = pd.to_numeric(result[column], errors="coerce")
        if not np.all(np.isfinite(values)) or np.any(values != np.floor(values)):
            raise ValueError(f"{column} must contain finite integer frame IDs.")
        result[column] = values.astype("int64")
    if (result.frame_end < result.frame_start).any():
        raise ValueError("frame_end must be >= frame_start.")
    return result.sort_values(["frame_start", "frame_end"], kind="stable").reset_index(drop=True)


def link_team_possessions(player_possessions):
    """Link player intervals through verified successful pass receptions.

    Parameters
    ----------
    player_possessions : pandas.DataFrame
        One match, with player_id (home/away shirt label), frame_start,
        frame_end, end_type, pass_outcome, target_pass, frame_receipt, loss.
        Optional interruption and period columns prevent linking across
        stoppages and periods. Loss must be boolean or missing (unverified), never a string.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Team intervals (possession_id, team_id, frame_start, frame_end,
        lost, n_player_possessions) and player intervals annotated with
        team_possession_id. Intervals are inclusive. IDs are local to this
        match. No adjacency-only or carrier-change inference is performed.
    """
    rows = validate_intervals(player_possessions, [
        "player_id", "frame_start", "frame_end", "end_type",
        "pass_outcome", "target_pass", "frame_receipt", "loss",
    ])
    if not rows.player_id.astype(str).str.fullmatch(r"(home|away)_\d+").all():
        raise ValueError("player_id must use home_<number> or away_<number>.")
    if not rows.loss.map(lambda v: pd.isna(v) or isinstance(v, (bool, np.bool_))).all():
        raise ValueError("loss must contain booleans or missing values.")
    if len(rows) > 1 and np.any(rows.frame_start.to_numpy()[1:] <= rows.frame_end.to_numpy()[:-1]):
        raise ValueError("Player possession intervals must not overlap.")
    labels = []
    previous = None
    group = 0
    for row in rows.itertuples(index=False):
        join = False
        if previous is not None:
            interruption = getattr(previous, "game_interruption_after", None)
            join = (
                previous.end_type == "pass"
                and previous.pass_outcome == "successful"
                and pd.notna(previous.loss)
                and not previous.loss
                and pd.isna(interruption)
                and pd.notna(previous.target_pass)
                and previous.target_pass == row.player_id
                and previous.player_id.split("_")[0] == row.player_id.split("_")[0]
                and pd.notna(previous.frame_receipt)
                and previous.frame_receipt == row.frame_start
            )
            if "period" in rows:
                join = join and previous.period == row.period
        if not join:
            group += 1
        labels.append(group)
        previous = row
    rows["team_possession_id"] = labels
    teams = []
    for key, segments in rows.groupby("team_possession_id", sort=False):
        teams.append({
            "possession_id": key,
            "team_id": segments.iloc[0].player_id.split("_")[0],
            "frame_start": int(segments.frame_start.iloc[0]),
            "frame_end": int(segments.frame_end.iloc[-1]),
            "lost": (bool(segments.loss.iloc[-1])
                     if pd.notna(segments.loss.iloc[-1]) else pd.NA),
            "n_player_possessions": len(segments),
        })
    return pd.DataFrame(teams, columns=[
        "possession_id", "team_id", "frame_start", "frame_end", "lost",
        "n_player_possessions",
    ]), rows
