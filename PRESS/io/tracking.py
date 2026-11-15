"""Adapt provider-independent long tables to PRESS's internal wide format."""

import numpy as np
import pandas as pd


def _require(table, columns, name):
    if not isinstance(table, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame.")
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}.")
    if table.columns.duplicated().any():
        raise ValueError(f"{name} contains duplicate column names.")


def convert_long_tracking(players, ball, context, *, home_team="home", away_team="away"):
    """Return canonical wide tracking and a source-to-internal player roster.

    Parameters
    ----------
    players : pandas.DataFrame
        One match. Required columns: ``frame, team_id, player_id, x, y, vx,
        vy``. Source player/team IDs may be strings or numbers. Missing
        players may be omitted from a frame. Positions must already be in
        centred pitch metres and velocities in m/s in the same axes.
    ball : pandas.DataFrame
        One row per frame, with ``frame, x, y``; optional ``z, vx, vy, vz``.
    context : pandas.DataFrame
        One row per frame, with ``frame, player_possession, attacking_side``.
        The carrier uses the original player ID; missing means no carrier.
        Supply ``team_possession`` with the source team ID when player IDs
        are not globally unique. Other context columns are preserved.
    home_team, away_team : str or int
        Explicit source team identifiers. Teams are never guessed from
        location or possession. The adapter does not estimate velocities,
        smooth data, interpolate gaps or infer action labels.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Wide tracking and roster with ``player_id`` (internal home/away
        alias), ``source_player_id``, ``team_id`` and ``source_team_id``.
        Aliases are assigned in first-appearance order within each team;
        they are identifiers, not assumed shirt numbers.
    """
    _require(players, ["frame", "team_id", "player_id", "x", "y", "vx", "vy"], "players")
    _require(ball, ["frame", "x", "y"], "ball")
    _require(context, ["frame", "player_possession", "attacking_side"], "context")
    if home_team == away_team or pd.isna(home_team) or pd.isna(away_team):
        raise ValueError("home_team and away_team must be distinct nonmissing IDs.")
    teams = {home_team: "home", away_team: "away"}
    if players[["player_id", "team_id"]].isna().any().any():
        raise ValueError("Player and team identifiers cannot be missing.")
    if not players.team_id.isin(teams).all():
        raise ValueError("players contains a team other than home_team/away_team.")
    for table, name in [(players, "players"), (ball, "ball"), (context, "context")]:
        values = pd.to_numeric(table.frame, errors="coerce")
        if not np.isfinite(values).all() or (values != np.floor(values)).any():
            raise ValueError(f"{name}.frame must contain finite integer IDs.")
    if players.duplicated(["frame", "team_id", "player_id"]).any():
        raise ValueError("Duplicate player observations within a frame.")
    if ball.frame.duplicated().any() or context.frame.duplicated().any():
        raise ValueError("ball and context require exactly one row per frame.")
    if set(ball.frame) != set(context.frame) or not set(players.frame) <= set(context.frame):
        raise ValueError("ball/context frames must match and cover all player observations.")
    for table, columns in [(players, ["x", "y", "vx", "vy"]), (ball, ["x", "y"])]:
        if any(not pd.api.types.is_numeric_dtype(table[c]) for c in columns):
            raise ValueError("Positions and velocities must be numeric.")

    roster = players[["team_id", "player_id"]].drop_duplicates().copy()
    roster = roster.rename(columns={"team_id": "source_team_id", "player_id": "source_player_id"})
    roster["team_id"] = roster.source_team_id.map(teams)
    roster["player_id"] = [
        f"{side}_{number}"
        for side, number in zip(roster.team_id, roster.groupby("team_id").cumcount() + 1)
    ]
    aliases = {
        (r.source_team_id, r.source_player_id): r.player_id
        for r in roster.itertuples(index=False)
    }
    source_ids = roster.groupby("source_player_id", sort=False).player_id.agg(list).to_dict()
    data = players.copy()
    data["alias"] = [aliases[(t, p)] for t, p in zip(data.team_id, data.player_id)]
    wide = data.pivot(index="frame", columns="alias", values=["x", "y", "vx", "vy"])
    wide.columns = [f"{player}_{variable}" for variable, player in wide.columns]
    for player in roster.player_id:
        wide[f"{player}_v"] = np.hypot(wide[f"{player}_vx"], wide[f"{player}_vy"])
    ball_columns = [c for c in ["x", "y", "z", "vx", "vy", "vz"] if c in ball]
    ball_frame = ball.set_index("frame")[ball_columns].rename(columns=lambda c: "ball_" + c)
    result = context.copy().set_index("frame")
    forbidden = set(result.columns) & (set(wide.columns) | set(ball_frame.columns))
    if forbidden:
        raise ValueError(f"context duplicates tracking columns: {sorted(forbidden)}.")
    carriers = []
    for row in result.itertuples():
        source = row.player_possession
        if pd.isna(source):
            carriers.append(None)
        elif hasattr(row, "team_possession") and pd.notna(row.team_possession):
            alias = aliases.get((row.team_possession, source))
            if alias is None:
                raise ValueError("Carrier/team pair is absent from the player roster.")
            carriers.append(alias)
        else:
            candidates = source_ids.get(source, [])
            if len(candidates) != 1:
                raise ValueError("Carrier ID is unknown or ambiguous; supply team_possession.")
            carriers.append(candidates[0])
    result["player_possession"] = carriers
    if "team_possession" in result:
        result["team_possession"] = result.team_possession.map(teams)
    result = result.join(ball_frame).join(wide).sort_index().reset_index()
    return result, roster.reset_index(drop=True)
