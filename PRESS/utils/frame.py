
import pandas as pd
import numpy as np


def frame_to_long(
    frame_df,
):
    """
    Convert one wide tracking frame
    into long format while keeping ball information.
    """

    frame = frame_df.iloc[0]

    rows = []


    # colonne globali da mantenere
    global_cols = [
        "frame",
        "ball_x",
        "ball_y",
        "ball_z",
        "ball_status",
        "team_possession",
        "ball_vx",
        "ball_vy",
        "ball_velocity",
    ]


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


        prefix = f"{side}_{shirt}_"


        # giocatore assente
        if pd.isna(frame[prefix + "x"]) or pd.isna(frame[prefix + "y"]):
            continue


        row = {
            key: frame[key]
            for key in global_cols
            if key in frame.index
        }


        row.update(
            {

                "team_id": side,

                "shirt_num": int(shirt),

                "player_id": (
                    side + "_" + str(int(shirt))
                ),

                "x": frame[prefix + "x"],

                "y": frame[prefix + "y"],

                "vx": frame[prefix + "vx"],

                "vy": frame[prefix + "vy"],

                "velocity": frame[prefix + "velocity"],

            }
        )


        rows.append(row)


    return pd.DataFrame(rows)
