import pandas as pd


def frame_to_long(
    frame_df,
):
    """
    Convert one wide tracking frame
    into long format for visualization.

    Parameters
    ----------
    frame_df : pd.DataFrame
        Single-frame wide tracking dataframe.

    Returns
    -------
    pd.DataFrame
        Long format tracking dataframe.
    """

    if len(frame_df) != 1:
        raise ValueError(
            "frame_df must contain exactly one frame."
        )

    frame = frame_df.iloc[0]

    rows = []


    # --------------------
    # Possession information
    # --------------------

    if "player_possession" not in frame.index:
        raise ValueError(
            "Missing player_possession column."
        )


    player_possession = frame["player_possession"]


    if pd.isna(player_possession):
        raise ValueError(
            "Missing player_possession information."
        )


    team_possession = (
        player_possession.split("_")[0]
    )


    # --------------------
    # Global information
    # --------------------

    global_cols = [
        "frame",
        "timestamp",
        "period",
        "ball_x",
        "ball_y",
        "ball_z",
        "player_possession",
    ]


    # --------------------
    # Players extraction
    # --------------------

    for col in frame.index:

        if (
            not col.endswith("_x")
            or
            (
                not col.startswith("home_")
                and
                not col.startswith("away_")
            )
        ):
            continue


        side, shirt, _ = col.split("_")

        shirt = int(shirt)

        prefix = (
            f"{side}_{shirt}_"
        )


        x = frame[prefix + "x"]
        y = frame[prefix + "y"]


        # player not detected
        if pd.isna(x) or pd.isna(y):
            continue


        row = {
            key: frame[key]
            for key in global_cols
            if key in frame.index
        }


        row.update(
            {
                "team_possession": team_possession,

                "team_id": side,

                "shirt_num": shirt,

                "player_id": (
                    f"{side}_{shirt}"
                ),

                "x": x,

                "y": y,

                "vx": frame.get(
                    prefix + "vx",
                    None,
                ),

                "vy": frame.get(
                    prefix + "vy",
                    None,
                ),

                "velocity": frame.get(
                    prefix + "velocity",
                    None,
                ),
            }
        )


        rows.append(row)


    return pd.DataFrame(rows)
