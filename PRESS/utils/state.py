import numpy as np
import pandas as pd


def get_pv_players(
    frame,
    tracking_df,
):
    """
    Extract BCOS players from wide tracking format.

    Parameters
    ----------
    frame : int
        Frame identifier.

    tracking_df : pd.DataFrame
        Canonical wide tracking format.

    Returns
    -------
    dict
        Ball carrier, teammates and defenders state,
        including player IDs.
    """

    def pad_players(
        coords,
        vels,
        n_required,
    ):

        if len(coords) < n_required:

            n_pad = n_required - len(coords)

            coords = np.vstack(
                [
                    coords,
                    np.full(
                        (n_pad, 2),
                        1000.0,
                    ),
                ]
            )

            vels = np.vstack(
                [
                    vels,
                    np.zeros(
                        (n_pad, 2)
                    ),
                ]
            )

        return (
            coords[:n_required],
            vels[:n_required],
        )


    # --------------------
    # Initial frame
    # --------------------

    frame_df = tracking_df[
        tracking_df.frame == frame
    ]

    if len(frame_df) != 1:
        raise ValueError(
            "Missing or duplicated frame in tracking data."
        )

    df = frame_df.iloc[0]


    # --------------------
    # Ball carrier
    # --------------------

    player = df["player_possession"]

    if pd.isna(player):
        raise ValueError(
            "Missing player possession information."
        )


    team_id, shirt_num = player.split("_")

    shirt_num = int(
        shirt_num
    )


    bc_prefix = (
        f"{team_id}_{shirt_num}_"
    )


    x_bc = np.array(
        [
            df[bc_prefix + "x"],
            df[bc_prefix + "y"],
        ]
    )

    v_bc = np.array(
        [
            df[bc_prefix + "vx"],
            df[bc_prefix + "vy"],
        ]
    )


    # --------------------
    # Remaining players
    # --------------------

    teammates = []
    attacker_ids = []
    v_atts = []

    defenders = []
    defender_ids = []
    v_defs = []


    other_team = (
        "away"
        if team_id == "home"
        else "home"
    )


    for side, pos_container, vel_container, id_container in [

        (
            team_id,
            teammates,
            v_atts,
            attacker_ids,
        ),

        (
            other_team,
            defenders,
            v_defs,
            defender_ids,
        ),

    ]:

        for col in df.index:

            if (
                not col.startswith(side + "_")
                or not col.endswith("_x")
            ):
                continue


            shirt = int(
                col.split("_")[1]
            )


            # Exclude ball carrier

            if (
                side == team_id
                and shirt == shirt_num
            ):
                continue


            player_id = (
                f"{side}_{shirt}"
            )


            prefix = (
                f"{player_id}_"
            )


            x = df[prefix + "x"]
            y = df[prefix + "y"]


            if pd.isna(x) or pd.isna(y):
                continue


            pos_container.append(
                [
                    x,
                    y,
                ]
            )

            vel_container.append(
                [
                    df[prefix + "vx"],
                    df[prefix + "vy"],
                ]
            )

            id_container.append(
                player_id
            )


    # --------------------
    # Padding
    # --------------------

    r_defs, v_defs = pad_players(
        np.asarray(defenders),
        np.asarray(v_defs),
        11,
    )


    r_atts, v_atts = pad_players(
        np.asarray(teammates),
        np.asarray(v_atts),
        10,
    )


    defender_ids = (
        defender_ids[:11]
        + [None] * (
            11 - len(defender_ids)
        )
    )


    attacker_ids = (
        attacker_ids[:10]
        + [None] * (
            10 - len(attacker_ids)
        )
    )


    # --------------------
    # Return
    # --------------------

    return {

        "frame_df": frame_df,

        "player": player,

        "team_id": team_id,

        "shirt_num": shirt_num,

        "x_bc": x_bc,

        "v_bc": v_bc,

        "r_defs": r_defs,

        "v_defs": v_defs,

        "defender_ids": defender_ids,

        "r_atts": r_atts,

        "v_atts": v_atts,

        "attacker_ids": attacker_ids,
    }
