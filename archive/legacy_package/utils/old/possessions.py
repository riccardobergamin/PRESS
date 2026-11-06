import pandas as pd

def get_possession(game):
    """
    Segment ball possessions from event and tracking data.
    """

    events_df = game.event_data.dropna(
        subset=["original_event_id"]
    )

    tracking_df = game.tracking_data

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
            + dictionary["shirt_num"].astype(int).astype(str)
        )
    )


    possessions = []

    tracking_indexed = tracking_df.set_index(
        "frame"
    )


    triggers = events_df[
        (
            (events_df["original_event"] == "Pass")
            &
            (events_df["is_successful"] == True)
        )
        |
        (
            events_df["original_event"] == "BallClaiming"
        )
    ].sort_values(
        "tracking_frame"
    ).reset_index(drop=True)



    for _, start_event in triggers.iterrows():

        t_event = int(
            start_event["tracking_frame"]
        )

        current_player = start_event["player_id"]

        used_fallback = False

        # --------------------
        # Find possession start
        # --------------------

        if start_event["databallpy_event"] == "pass":

            receiver_id = start_event["receiver_player_id"]


            receiver_next_event = events_df[
                (events_df["tracking_frame"] > t_event)
                &
                (events_df["player_id"] == receiver_id)
            ].sort_values(
                "tracking_frame"
            ).first_valid_index()


            if receiver_next_event is None:
                continue


            limit_frame = int(
                events_df.loc[
                    receiver_next_event,
                    "tracking_frame"
                ]
            )


            t_start = None



            for t_search in range(
                t_event + 1,
                limit_frame + 1,
            ):

                if t_search in tracking_indexed.index:

                    possession = tracking_indexed.loc[
                        t_search,
                        "player_possession"
                    ]


                    values = (
                        [possession]
                        if isinstance(possession, str)
                        else (
                            possession.values
                            if hasattr(possession, "values")
                            else []
                        )
                    )


                    values = [k for v in values for k, val in dictionary.items() if val == v]


                    if receiver_id in values:
                        t_start = t_search
                        break



            if t_start is None:

                t_start = limit_frame - 1

                used_fallback = True



            current_player = receiver_id


        else:

            t_start = t_event



        # --------------------
        # Find possession end
        # --------------------

        is_terminal = (
            events_df["databallpy_event"].isin(
                [
                    "pass",
                    "shot",
                ]
            )
            |
            events_df["original_event"].isin(
                [
                    "fouled",
                    "ballClaimed",
                    "ballcontactSucceeded",
                ]
            )
            |
            events_df["is_defensive_clearance"]
        )



        future_events = events_df[
            (events_df["tracking_frame"] >= t_start)
            &
            is_terminal
        ].sort_values(
            "tracking_frame"
        )



        if future_events.empty:
            continue



        outcome_event = future_events.iloc[0]



        outcome = (
            outcome_event["databallpy_event"].upper()
            if outcome_event["databallpy_event"]
            in [
                "pass",
                "shot",
            ]
            else outcome_event["original_event"]
        )



        if outcome_event["is_defensive_clearance"]:

            outcome = "LOST_BY_CLEARANCE"


        elif outcome_event["original_event"] == "fouled":

            outcome = "FOULED_SUBITO"



        elif (
            outcome_event["original_event"]
            in [
                "ballClaimed",
                "ballcontactSucceeded",
            ]
            and outcome_event["player_id"] != current_player
        ):

            outcome = "LOST_BY_TACKLE"



        possessions.append(
            {
                "player": current_player,

                "receiver": (
                    outcome_event["receiver_player_id"]
                    if outcome_event["databallpy_event"] == "pass"
                    else None
                ),

                "start_frame": t_start,

                "end_frame": outcome_event["tracking_frame"],

                "outcome": outcome,

                "pass_result": (
                    outcome_event["result"]
                    if outcome_event["databallpy_event"] == "pass"
                    else "N/A"
                ),

                "used_fallback": used_fallback,
            }
        )



    return pd.DataFrame(possessions)
  

# def get_possessions(game):
#     """
#     Extract all frames where a player has possession.
# 
#     One row is returned per frame with an assigned possession --
#     not one row per distinct possession segment. If a player holds
#     the ball for N consecutive frames, this returns N rows for
#     that player, one per frame.
# 
#     Parameters
#     ----------
#     game : object
#         Game object containing tracking and event data.
# 
#     Returns
#     -------
#     pd.DataFrame
#         Contains:
#         - player
#         - frame
#     """
#     tracking_df = game.tracking_data
#     event_df = game.event_data
# 
#     player_dict = (
#         event_df[
#             ["player_id", "side", "shirt_num"]
#         ]
#         .dropna()
#         .drop_duplicates()
#     )
#     player_dict = dict(
#         zip(
#             player_dict["side"] + "_" + player_dict["shirt_num"].astype(int).astype(str),
#             player_dict["player_id"],
#         )
#     )
# 
#     possession = tracking_df[
#         tracking_df["player_possession"].notna()
#     ].copy()
# 
#     possession["player"] = possession["player_possession"].map(player_dict)
#     possession = possession[possession["player"].notna()].copy()
# 
#     return possession[
#         ["player", "frame"]
#     ].reset_index(drop=True)
    

