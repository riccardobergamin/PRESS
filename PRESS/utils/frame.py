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

                "v": frame.get(
                    prefix + "v",
                    None,
                ),
            }
        )


        rows.append(row)


    return pd.DataFrame(rows)


def resolve_pitch_size(frame_df, pitch_length=None, pitch_width=None):
    """Resolve plot dimensions from a frame, with legacy 105×68 fallback."""
    import numpy as np
    row = frame_df.iloc[0]
    length = float(row.get("pitch_length", 105) if pitch_length is None else pitch_length)
    width = float(row.get("pitch_width", 68) if pitch_width is None else pitch_width)
    if not np.isfinite([length, width]).all() or min(length, width) <= 0:
        raise ValueError("Pitch dimensions must be finite and positive.")
    return length, width
