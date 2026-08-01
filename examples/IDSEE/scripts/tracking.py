# preprocessing/tracking.py
import pandas as pd
import re

def process_tracking(
    game,
    offset=1.0,
    n_batches="smart",
    verbose=True,
    max_velocity=50.0,
    filter_type="moving_average",
    window_length=12,
):
    """
    Synchronize events and tracking data,
    add possession information,
    compute player velocities,
    and normalize tracking time.
    """

    # Synchronize event and tracking frames
    game.synchronise_tracking_and_event_data(
        n_batches=n_batches,
        verbose=verbose,
        offset=offset,
    )

    # Add player possession labels
    game.tracking_data.add_individual_player_possession()

    # Extract player ids
    player_ids = list({
        col.replace("_x", "").replace("_y", "")
        for col in game.tracking_data.columns
        if "home" in col or "away" in col
    })

    # Compute velocities
    for player in player_ids:
        game.tracking_data.add_velocity(
            column_ids=player,
            max_velocity=max_velocity,
            filter_type=filter_type,
            window_length=window_length,
        )

    # Normalize frame index
    game.tracking_data.frame = (
        game.tracking_data.frame
        - game.tracking_data.frame.iloc[0]
    )

    return game

def tracking_to_long(tracking_df):
    import numpy as np
    import pandas as pd
    import re

    pattern = re.compile(r"^(home|away)_(\d+)_(.+)$")

    player_cols = [c for c in tracking_df.columns if pattern.match(c)]
    frame_cols = [c for c in tracking_df.columns if c not in player_cols]

    groups = {}
    for col in player_cols:
        team, player_id, feature = pattern.match(col).groups()
        groups.setdefault((team, int(player_id)), {})[feature] = col

    standard_features = ("x", "y", "vx", "vy", "velocity")
    defaults = {"x": np.nan, "y": np.nan, "vx": 0.0, "vy": 0.0, "velocity": 0.0}

    players = list(groups.keys())
    n_frames = len(tracking_df)
    n_players = len(players)

    all_features = sorted(set().union(*groups.values()) | set(standard_features))

    # matrice (n_frames, n_players) per ciascuna feature, riempita in un colpo solo
    feature_flat = {}
    for feature in all_features:
        mat = np.full((n_frames, n_players), defaults.get(feature, np.nan))
        for j, key in enumerate(players):
            col = groups[key].get(feature)
            if col is not None:
                mat[:, j] = tracking_df[col].to_numpy()
        # ordine "player-major": tutti i frame del player0, poi del player1, ...
        feature_flat[feature] = mat.reshape(-1, order="F")

    # colonne originali del frame, ripetute per ogni player (un solo concat vettoriale)
    out = pd.concat([tracking_df[frame_cols]] * n_players, ignore_index=True)

    out["team_id"]   = np.repeat([p[0] for p in players], n_frames)
    out["player_id"] = np.repeat([p[1] for p in players], n_frames)
    for feature in all_features:
        out[feature] = feature_flat[feature]

    return out


def long_from_wide(tracking_df):

    velocity_cols = [
        c for c in tracking_df.columns
        if c.endswith("_velocity")
    ]

    vel_all = []

    for col in velocity_cols:

        player_id = col.replace(
            "_velocity",
            ""
        )

        temp = tracking_df[
            [
                "frame",
                "player_possession",
                col,
            ]
        ].copy()

        temp["player_id"] = player_id

        temp = temp.rename(
            columns={
                col: "v"
            }
        )

        temp["is_ball_carrier"] = (
            temp["player_possession"]
            ==
            player_id
        )

        vel_all.append(
            temp[
                [
                    "frame",
                    "player_id",
                    "v",
                    "is_ball_carrier",
                ]
            ]
        )

    return pd.concat(
        vel_all,
        ignore_index=True
    )
