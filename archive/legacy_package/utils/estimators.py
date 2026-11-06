import pandas as pd
import jax
import jax.numpy as jnp
from jaxopt import LBFGS

def long_from_wide(tracking_df):

    velocity_cols = [
        c for c in tracking_df.columns
        if c.endswith("_v")
    ]

    vel_all = []

    for col in velocity_cols:

        player_id = col.replace(
            "_v",
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


def estimate_player_max_speeds(
    press,
    velocity_threshold = 12.0
):
  
    tracking_long = long_from_wide(
        press.tracking_data
    )

    tracking_long = tracking_long.dropna(
        subset=["v"]
    )

    tracking_long = tracking_long[
        ~tracking_long["player_id"].str.contains("ball")
    ]
    
    tracking_long = tracking_long[
        tracking_long["v"] <= velocity_threshold
    ]

    v_player = (
        tracking_long
        .groupby(
            [
                "player_id",
                "is_ball_carrier",
            ]
        )["v"]
        .max()
        .unstack()
    )

    v_player.columns = [
        "without_ball",
        "with_ball",
    ]

    stats = pd.DataFrame(
        index=[
            "mean",
            "p05",
            "p25",
            "p50",
            "p75",
            "p95",
        ],
        columns=[
            "without_ball",
            "with_ball",
        ],
    )

    stats.loc["mean"] = v_player.mean()

    stats.loc[
        ["p05", "p25", "p50", "p75", "p95"]
    ] = (
        v_player
        .quantile(
            [
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
            ]
        )
        .values
    )

    return stats
