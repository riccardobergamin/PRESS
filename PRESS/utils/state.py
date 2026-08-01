import numpy as np
import pandas as pd

def get_pv_players(
    frame,
    tracking_df,
    event_df,
):
    """
    Extract BCOS players from wide tracking format.
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

    player = frame_df.player_possession.iloc[0]
    
    if len(frame_df) != 1:
        raise ValueError(
            "Missing frame in tracking data"
        )
        
    if pd.isna(player):
        raise ValueError(
            "Missing player possession information"
        )    

    frame = frame_df.iloc[0]

    player_dict = (
        event_df[
            ["player_id", "side", "shirt_num"]
        ]
        .dropna()
        .drop_duplicates()
    )

    player_dict = dict(
        zip(
            player_dict["side"] + "_" + player_dict["shirt_num"].astype(int).astype(str),
            player_dict["player_id"],
        )
    )

    player = player_dict.get(
        player
    )
  
    # --------------------
    # Ball carrier metadata
    # --------------------

    player_info = (
        event_df[
            event_df.player_id
            == player
        ][
            [
                "player_id",
                "player_name",
                "side",
                "shirt_num",
            ]
        ]
        .drop_duplicates()
        .iloc[0]
    )


    player_id = player_info.player_id
    player_name = player_info.player_name
    team_id = player_info.side
    shirt_num = int(
        player_info.shirt_num
    )


    # --------------------
    # Ball carrier
    # --------------------

    bc_prefix = (
        f"{team_id}_{shirt_num}_"
    )


    x_bc = np.array(
        [
            frame[bc_prefix + "x"],
            frame[bc_prefix + "y"],
        ]
    )


    v_bc = np.array(
        [
            frame[bc_prefix + "vx"],
            frame[bc_prefix + "vy"],
        ]
    )


    # --------------------
    # Remaining players
    # --------------------

    teammates = []
    v_atts = []

    defenders = []
    v_defs = []


    other_team = (
        "away"
        if team_id == "home"
        else "home"
    )


    for side, container_pos, container_vel in [

        (
            team_id,
            teammates,
            v_atts,
        ),

        (
            other_team,
            defenders,
            v_defs,
        ),

    ]:


        for col in frame.index:


            if (
                not col.startswith(
                    side + "_"
                )
                or
                not col.endswith("_x")
            ):
                continue


            shirt = col.split("_")[1]


            if (
                side == team_id
                and
                int(shirt) == shirt_num
            ):
                continue


            prefix = (
                f"{side}_{shirt}_"
            )


            if np.isnan(
                frame[prefix + "x"]
            ):
                continue


            container_pos.append(

                [
                    frame[prefix + "x"],
                    frame[prefix + "y"],
                ]

            )


            container_vel.append(

                [
                    frame[prefix + "vx"],
                    frame[prefix + "vy"],
                ]

            )


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


    return {

        "frame_df": frame_df,

        "player_name": player_name,

        "player_id": player_id,

        "shirt_num": shirt_num,

        "team_id": team_id,

        "x_bc": x_bc,

        "v_bc": v_bc,

        "r_defs": r_defs,

        "v_defs": v_defs,

        "r_atts": r_atts,

        "v_atts": v_atts,
    }
