"""
Event preprocessing utilities.
"""

import pandas as pd


def _is_defensive_clearance(qualifier):
    """
    Identify defensive clearance qualifier.
    """

    if isinstance(qualifier, dict):
        return qualifier.get("DefensiveClearance") == "true"

    if isinstance(qualifier, list):
        return any(
            isinstance(q, dict)
            and q.get("DefensiveClearance") == "true"
            for q in qualifier
        )

    return False


def extract_clearances(raw_events):

    events = []

    for half in ["firstHalf", "secondHalf"]:
        for team in ["Home", "Away"]:

            df = raw_events[half][team].events.copy()
            events.append(df)

    df_all = pd.concat(
        events,
        ignore_index=True,
    )

    clearances = df_all[
        df_all.eID == "OtherBallAction"
    ].copy()

    clearances["is_defensive_clearance"] = (
        clearances["qualifier"]
        .apply(_is_defensive_clearance)
    )

    return clearances[
        clearances["is_defensive_clearance"]
    ]
    

def add_player_info_to_events(game):

    import pandas as pd


    home = game.home_players.copy()
    home["side"] = "home"

    away = game.away_players.copy()
    away["side"] = "away"


    players = pd.concat(
        [
            home,
            away,
        ],
        ignore_index=True,
    )


    players["shirt_num"] = (
        players["shirt_num"]
        .astype(int)
    )


    player_map = (
        players
        .set_index("id")
        [
            [
                "full_name",
                "shirt_num",
                "side",
            ]
        ]
        .to_dict(
            orient="index"
        )
    )


    game.event_data["player_name"] = (
        game.event_data["player_id"]
        .map(
            lambda x: player_map.get(x, {})
            .get("full_name")
        )
    )


    game.event_data["shirt_num"] = (
        game.event_data["player_id"]
        .map(
            lambda x: player_map.get(x, {})
            .get("shirt_num")
        )
        .astype("Int64")
    )


    game.event_data["side"] = (
        game.event_data["player_id"]
        .map(
            lambda x: player_map.get(x, {})
            .get("side")
        )
    )


    return game
  

def add_defensive_clearances(
    game,
    raw_events,
):
    """
    Add defensive clearance information.
    """

    clearances = extract_clearances(
        raw_events
    )

    game.event_data["datetime"] = pd.to_datetime(
        game.event_data["datetime"],
        format="mixed",
    )

    game.event_data = game.event_data.merge(
        clearances[
            [
                "timestamp",
                "is_defensive_clearance",
            ]
        ],
        left_on="datetime",
        right_on="timestamp",
        how="left",
    )

    game.event_data["is_defensive_clearance"] = (
        game.event_data["is_defensive_clearance"]
        .fillna(False).astype(bool)
    )

    game.event_data.drop(
        columns=["timestamp"],
        inplace=True,
    )

    return game


def add_pass_receivers(
    game,
    kloppy_events,
):
    """
    Add pass receiver information.
    """

    passes = kloppy_events.dropna(
        subset=["event_id"]
    ).copy()

    passes["event_id"] = (
        passes["event_id"]
        .astype(str)
        .str.strip()
    )

    passes = passes[
        [
            "event_id",
            "receiver_player_id",
            "result",
        ]
    ]

    game.event_data = pd.merge(
        game.event_data,
        passes,
        left_on="original_event_id",
        right_on="event_id",
        how="left",
        suffixes=("", "_kloppy"),
    )

    game.event_data.drop(
        columns=["event_id_kloppy"],
        inplace=True,
        errors="ignore",
    )

    return game
  
  
def process_events(
    game,
    floodlight_events,
    kloppy_events,
):
    """
    Complete event preprocessing pipeline.

    Adds:
    - defensive clearances from Floodlight
    - pass receivers from Kloppy
    """

    game = add_defensive_clearances(
        game,
        floodlight_events,
    )

    game = add_pass_receivers(
        game,
        kloppy_events,
    )

    return game
