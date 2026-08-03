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


    for _, row in passes.iterrows():

        if (
            not row.get(
                "is_successful",
                False,
            )
        ):
            receiver_x.append(np.nan)
            receiver_y.append(np.nan)
            continue


        receiver = row.get(
            "receiver_player_id",
            np.nan,
        )

        if pd.isna(receiver):
            receiver_x.append(np.nan)
            receiver_y.append(np.nan)
            continue


        frame_df = tracking_df[
            tracking_df.frame
            == row.frame
        ]


        if frame_df.empty:
            receiver_x.append(np.nan)
            receiver_y.append(np.nan)
            continue


        frame = frame_df.iloc[0]


        if (
            f"{receiver}_x"
            not in frame.index
        ):
            receiver_x.append(np.nan)
            receiver_y.append(np.nan)
            continue


        receiver_x.append(
            frame[f"{receiver}_x"]
        )

        receiver_y.append(
            frame[f"{receiver}_y"]
        )


    passes["receiver_x"] = receiver_x
    passes["receiver_y"] = receiver_y


    press.passes = passes.reset_index(
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
            start_event.frame
        )

        current_player = (
            start_event.player_id
        )

        used_fallback = False


        # --------------------
        # Possession start
        # --------------------

        if (
            start_event.event_type
            == "pass"
        ):

            receiver = (
                start_event.receiver_player_id
            )

            receiver_events = events_df[
                (events_df.frame > t_event)
                &
                (
                    events_df.player_id
                    == receiver
                )
            ].sort_values(
                "frame"
            )


            if receiver_events.empty:
                continue


            limit_frame = int(
                receiver_events
                .iloc[0]
                .frame
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


                if (
                    isinstance(
                        possession,
                        pd.Series,
                    )
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
            (events_df.frame >= t_start)
            &
            (
                (
                    (events_df.event_type == "pass")
                )
                |
                (
                    events_df.event_type
                    == "shot"
                )
                |
                (
                    events_df.event_type
                    == "foul"
                )
                |
                (
                    (
                        events_df.event_type
                        == "tackle"
                    )
                    &
                    (
                        events_df.is_successful
                    )
                )
                |
                (
                    events_df.event_type
                    == "clearance"
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


        possessions.append(
            {
                "player_id":
                    current_player,

                "team_id":
                    outcome_event.team_id,

                "receiver_player_id":
                    outcome_event.get(
                        "receiver_player_id",
                        pd.NA,
                    ),

                "start_frame":
                    t_start,

                "end_frame":
                    int(
                        outcome_event.frame
                    ),

                "terminal_event":
                    outcome_event.event_type,

                "outcome":
                    outcome_event["outcome"]
                    if outcome_event.event_type == "pass"
                    else pd.NA,

                "used_fallback":
                    used_fallback,
            }
        )


    press.possessions = (
        pd.DataFrame(
            possessions
        )
        .reset_index(
            drop=True
        )
    )
