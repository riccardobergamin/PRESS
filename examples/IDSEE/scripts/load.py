"""
Input utilities for loading Sportec tracking and event data.
"""

from databallpy import get_game
import numpy as np
from floodlight.io.dfl import read_event_data_xml
from kloppy import sportec



def load_game(
    data_dir,
    match_id,
):
    """
    Load Sportec game with databallpy and perform initial event normalization.
    """

    pos_file = (
        f"DFL_04_03_positions_raw_observed_"
        f"DFL-COM-000001_DFL-MAT-{match_id}.xml"
    )

    info_file = (
        f"DFL_02_01_matchinformation_"
        f"DFL-COM-000001_DFL-MAT-{match_id}.xml"
    )

    event_file = (
        f"DFL_03_02_events_raw_"
        f"DFL-COM-000001_DFL-MAT-{match_id}.xml"
    )

    game = get_game(
        tracking_data_provider="sportec",
        tracking_metadata_loc=data_dir + info_file,
        tracking_data_loc=data_dir + pos_file,
        event_data_loc=data_dir + event_file,
        event_metadata_loc=data_dir + info_file,
        event_data_provider="sportec",
    )


    # Normalize Sportec event labels

    pass_list = [
        "Pass",
        "Cross",
    ]

    shot_list = [
        "SuccessfulShot",
        "SavedShot",
        "BlockedShot",
        "ShotWide",
        "ShotWoodWork",
    ]

    tackle_list = [
        "ballClaimed",
        "ballcontactSucceeded",
        "BallClaiming",
        "fouled",
    ]

    conditions = [
        game.event_data["original_event"].isin(pass_list),
        game.event_data["original_event"].isin(shot_list),
        game.event_data["original_event"].isin(tackle_list),
    ]

    choices = [
        "pass",
        "shot",
        "tackle",
    ]

    game.event_data["databallpy_event"] = np.select(
        conditions,
        choices,
        default=game.event_data["databallpy_event"],
    )


    # Clean event identifiers

    game.event_data["original_event_id"] = (
        game.event_data["original_event_id"]
        .astype(str)
        .str.strip()
    )


    return game


def load_Floodlight_events(
    data_dir,
    match_id,
):
    """
    Load raw Sportec events using Floodlight.
    """

    info_file = (
        f"DFL_02_01_matchinformation_"
        f"DFL-COM-000001_DFL-MAT-{match_id}.xml"
    )

    event_file = (
        f"DFL_03_02_events_raw_"
        f"DFL-COM-000001_DFL-MAT-{match_id}.xml"
    )

    events, _, _ = read_event_data_xml(
        data_dir + event_file,
        data_dir + info_file,
    )

    return events


def load_kloppy_events(
    data_dir,
    match_id,
):
    """
    Load Sportec events using Kloppy.
    """

    info_file = (
        f"DFL_02_01_matchinformation_"
        f"DFL-COM-000001_DFL-MAT-{match_id}.xml"
    )

    event_file = (
        f"DFL_03_02_events_raw_"
        f"DFL-COM-000001_DFL-MAT-{match_id}.xml"
    )

    return sportec.load_event(
        meta_data=data_dir + info_file,
        event_data=data_dir + event_file,
    ).to_df()
    
    
