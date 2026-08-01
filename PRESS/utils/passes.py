import pandas as pd

def get_passes(game):
    """
    Extract passes with receiver position for successful passes.
    """

    events_df = game.event_data
    tracking_df = game.tracking_data


    passes = events_df[
        events_df["original_event"] == "Pass"
    ].copy()


    passes = passes.rename(
        columns={
            "player_id": "player",
            "receiver_player_id": "receiver",
            "tracking_frame": "frame",
            "result": "pass_result",
        }
    )


    # player_id -> tracking prefix
    dictionary = (
        events_df[
            [
                "player_id",
                "side",
                "shirt_num",
            ]
        ]
        .dropna()
        .drop_duplicates()
    )


    dictionary = dict(
        zip(
            dictionary["player_id"],
            dictionary["side"]
            + "_"
            + dictionary["shirt_num"]
            .astype(int)
            .astype(str)
        )
    )


    receiver_x = []
    receiver_y = []


    for _, row in passes.iterrows():

        # default for unsuccessful passes
        if not row["is_successful"]:
            receiver_x.append(None)
            receiver_y.append(None)
            continue


        receiver_id = row["receiver"]

        if receiver_id not in dictionary:
            receiver_x.append(None)
            receiver_y.append(None)
            continue


        prefix = dictionary[receiver_id]

        frame_row = tracking_df[
            tracking_df["frame"] == row["frame"]
        ]

        if frame_row.empty:
            receiver_x.append(None)
            receiver_y.append(None)
            continue


        receiver_x.append(
            frame_row[f"{prefix}_x"].iloc[0]
        )

        receiver_y.append(
            frame_row[f"{prefix}_y"].iloc[0]
        )


    passes["receiver_x"] = receiver_x
    passes["receiver_y"] = receiver_y


    return passes[
        [
            "player",
            "receiver",
            "frame",
            "pass_result",
            "is_successful",
            "team_id",
            "receiver_x",
            "receiver_y",
        ]
    ].reset_index(drop=True)
    
