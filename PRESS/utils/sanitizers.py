import re
import pandas as pd


def validate_tracking_data(
    tracking_data: pd.DataFrame,
) -> None:
    """
    Validate canonical wide tracking format.

    The function does not modify data.
    It raises ValueError if the format is invalid.
    """

    required = [
        "frame",
        "player_possession",
        "ball_x",
        "ball_y",
    ]

    missing = [
        col for col in required
        if col not in tracking_data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Frame integrity
    if tracking_data["frame"].duplicated().any():
        raise ValueError(
            "Frame column contains duplicates."
        )

    # Player columns
    pattern = re.compile(
        r"^(home|away)_(\d+)_(x|y|vx|vy)$"
    )

    players = {}

    for col in tracking_data.columns:

        match = pattern.match(col)

        if match:
            player = (
                f"{match.group(1)}_"
                f"{match.group(2)}"
            )

            variable = match.group(3)

            players.setdefault(
                player,
                set()
            ).add(variable)

    expected = {
        "x",
        "y",
        "vx",
        "vy",
    }

    for player, variables in players.items():

        missing = expected - variables

        if missing:
            raise ValueError(
                f"{player} missing columns: {missing}"
            )

    if len(players) == 0:
        raise ValueError(
            "No player columns found."
        )

    # Possession consistency
    valid_players = set(players.keys())

    invalid_possession = (
        tracking_data["player_possession"]
        .dropna()
        .loc[
            lambda x:
            ~x.isin(valid_players)
        ]
    )

    if not invalid_possession.empty:
        raise ValueError(
            "player_possession contains "
            "unknown players."
        )

    # Numeric check
    numeric_cols = [
        c for c in tracking_data.columns
        if (
            c.endswith("_x")
            or c.endswith("_y")
            or c.endswith("_vx")
            or c.endswith("_vy")
        )
    ]

    non_numeric = [
        c for c in numeric_cols
        if not pd.api.types.is_numeric_dtype(
            tracking_data[c]
        )
    ]

    if non_numeric:
        raise ValueError(
            f"Non numeric tracking columns: {non_numeric}"
        )
        


def validate_events_data(
    events_data,
):
    """
    Validate events data format.

    Raises ValueError only if event data
    is incompatible with PRESS conventions.
    """


    if not isinstance(
        events_data,
        pd.DataFrame,
    ):
        raise ValueError(
            "events_data must be a pandas DataFrame."
        )


    # ----------------------------------
    # Required columns
    # ----------------------------------

    required_columns = [
        "frame",
        "event_type",
        "player_id",
        "team_id",
        "outcome",
    ]


    missing_columns = [
        col
        for col in required_columns
        if col not in events_data.columns
    ]


    if missing_columns:

        raise ValueError(
            f"Missing event columns: {missing_columns}"
        )


    # ----------------------------------
    # Required events
    # ----------------------------------

    required_events = {
        "pass",
        "shot",
        "ball_recovery",
        "foul",
        "tackle",
    }


    available_events = set(
        events_data["event_type"]
        .dropna()
        .unique()
    )


    missing_events = (
        required_events
        -
        available_events
    )


    if missing_events:

        raise ValueError(
            f"Missing required events: {missing_events}"
        )


    # ----------------------------------
    # Player id convention
    # ----------------------------------

    invalid_player_ids = events_data[
        ~events_data["player_id"]
        .astype(str)
        .str.match(
            r"^(home|away)_\d+$"
        )
    ]


    if not invalid_player_ids.empty:

        raise ValueError(
            "Invalid player_id convention. "
            "Expected home_<shirt_number> or away_<shirt_number>."
        )


    # ----------------------------------
    # Pass validation
    # ----------------------------------

    passes = events_data[
        events_data["event_type"] == "pass"
    ]


    if "receiver_player_id" not in events_data.columns:

        raise ValueError(
            "Pass events require receiver_player_id."
        )


    if passes["receiver_player_id"].isna().any():

        raise ValueError(
            "Pass events require receiver_player_id."
        )


    if passes["outcome"].isna().any():

        raise ValueError(
            "Pass events require outcome."
        )


    # ----------------------------------
    # Tackle validation
    # ----------------------------------

    tackles = events_data[
        events_data["event_type"] == "tackle"
    ]


    if tackles["outcome"].isna().any():

        raise ValueError(
            "Tackle events require outcome."
        )


    return True
  
