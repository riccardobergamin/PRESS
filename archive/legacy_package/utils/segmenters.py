import numpy as np
import pandas as pd

def segment_passes(
    press,
):
    """
    Segment passes from PRESS events.

    Results are stored in
    press.passes.
    """

    if press.events_data is None:
        raise ValueError(
            "No events_data available."
        )

    tracking_df = press.tracking_data
    events_df = press.events_data


    passes = events_df[
        events_df["event_type"] == "pass"
    ].copy()


    receiver_x = []
    receiver_y = []

    passer_x = []
    passer_y = []
    passer_v = []


    for _, row in passes.iterrows():

        frame_df = tracking_df[
            tracking_df.frame == row["frame"]
        ]


        if frame_df.empty:
            receiver_x.append(np.nan)
            receiver_y.append(np.nan)

            passer_x.append(np.nan)
            passer_y.append(np.nan)
            passer_v.append(np.nan)

            continue


        frame = frame_df.iloc[0]


        passer = row["player_id"]
        receiver = row["receiver_player_id"]


        if (
            f"{passer}_x" in frame.index
            and f"{passer}_y" in frame.index
        ):
            passer_x.append(
                frame[f"{passer}_x"]
            )
            passer_y.append(
                frame[f"{passer}_y"]
            )
            passer_v.append(
                frame.get(
                    f"{passer}_v",
                    np.nan,
                )
            )

        else:
            passer_x.append(np.nan)
            passer_y.append(np.nan)
            passer_v.append(np.nan)


        if (
            f"{receiver}_x" in frame.index
            and f"{receiver}_y" in frame.index
        ):
            receiver_x.append(
                frame[f"{receiver}_x"]
            )
            receiver_y.append(
                frame[f"{receiver}_y"]
            )

        else:
            receiver_x.append(np.nan)
            receiver_y.append(np.nan)


    passes["passer_x"] = passer_x
    passes["passer_y"] = passer_y
    passes["passer_v"] = passer_v

    passes["receiver_x"] = receiver_x
    passes["receiver_y"] = receiver_y


    return passes.reset_index(
        drop=True
    )
    
    
def segment_possessions(
    press,
):
    """
    Segment player possessions from PRESS tracking
    and PRESS event data.

    Results are stored in
    press.possessions.
    """

    if press.events_data is None:
        raise ValueError(
            "No events_data available."
        )

    tracking_df = press.tracking_data
    events_df = press.events_data

    tracking_indexed = (
        tracking_df
        .set_index("frame")
    )

    # ----------------------------------
    # Possession triggers
    # ----------------------------------

    triggers = events_df[
        (
            (events_df["event_type"] == "pass")
            &
            (events_df["is_successful"])
        )
        |
        (
            events_df["event_type"]
            == "ball_recovery"
        )
    ].sort_values(
        "frame"
    )

    possessions = []

    # ----------------------------------
    # Loop over possessions
    # ----------------------------------

    for _, start_event in triggers.iterrows():

        t_event = int(
            start_event["frame"]
        )

        current_player = (
            start_event["player_id"]
        )

        used_fallback = False

        # --------------------
        # Possession start
        # --------------------

        if (
            start_event["event_type"]
            == "pass"
        ):

            receiver = (
                start_event["receiver_player_id"]
            )

            receiver_events = events_df[
                (
                    events_df["frame"]
                    > t_event
                )
                &
                (
                    events_df["player_id"]
                    == receiver
                )
            ].sort_values(
                "frame"
            )

            if receiver_events.empty:
                continue

            limit_frame = int(
                receiver_events
                .iloc[0]["frame"]
            )

            t_start = None

            for frame in range(
                t_event + 1,
                limit_frame + 1,
            ):

                if (
                    frame
                    not in tracking_indexed.index
                ):
                    continue

                possession = (
                    tracking_indexed.loc[
                        frame,
                        "player_possession",
                    ]
                )

                if isinstance(
                    possession,
                    pd.Series,
                ):
                    possession = (
                        possession.iloc[0]
                    )

                if possession == receiver:

                    t_start = frame

                    break

            if t_start is None:

                t_start = (
                    limit_frame - 1
                )

                used_fallback = True

            current_player = receiver

        else:

            t_start = t_event

        # --------------------
        # Possession end
        # --------------------

        terminal_events = events_df[
            (
                events_df["frame"]
                >= t_start
            )
            &
            (
                events_df["event_type"].isin(
                    [
                        "pass",
                        "shot",
                        "foul",
                        "tackle",
                        "clearance",
                    ]
                )
            )
        ].sort_values(
            "frame"
        )

        if terminal_events.empty:
            continue

        outcome_event = (
            terminal_events.iloc[0]
        )

        event_type = (
            outcome_event["event_type"]
        )

        # --------------------
        # Possession outcome
        # --------------------

        if event_type == "pass":

            loss = not outcome_event["is_successful"]

        elif event_type in [
            "tackle",
            "clearance",
        ]:

            loss = True

        else:

            loss = False
            

        possessions.append(
            {
                "player_id":
                    current_player,

                "team_id":
                    start_event["team_id"],

                "receiver_player_id":
                    outcome_event.get(
                        "receiver_player_id",
                        pd.NA,
                    ),

                "frame_start":
                    t_start,

                "frame_end":
                    int(
                        outcome_event[
                            "frame"
                        ]
                    ),

                "terminal_event":
                    event_type,

                "loss":
                    loss,

                "used_fallback":
                    used_fallback,
            }
        )

    return (
        pd.DataFrame(
            possessions
        )
        .reset_index(
            drop=True
        )
    )
    

def get_possession_frames(press):
  
    """
    Extract all frames where a player has possession.

    One row is returned per frame with an assigned possession --
    not one row per distinct possession segment. If a player holds
    the ball for N consecutive frames, this returns N rows for
    that player, one per frame.

    Parameters
    ----------
    press : object
        PRESS object containing tracking data.

    Returns
    -------
    pd.DataFrame
        Contains:
        - player_possession
        - frame
    """
    tracking_df = press.tracking_data

    possession = tracking_df[
        tracking_df["player_possession"].notna()
    ].copy()


    return possession[
        ["frame","player_possession"]
    ].reset_index(drop=True)
    
    
