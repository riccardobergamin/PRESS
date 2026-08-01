"""
Preprocessing pipeline.
"""

from load import (
    load_game,
    load_Floodlight_events,
    load_kloppy_events,
)

from tracking import process_tracking
from events import (
    process_events,
    add_player_info_to_events,
)

# from BCOS.utils.possessions import segment_possessions


def preprocess_match(
    data_dir,
    match_id,
):

    game = load_game(
        data_dir,
        match_id,
    )
    
    game = add_player_info_to_events(game)

    # Synchronize tracking and compute player features
    game = process_tracking(
        game,
    )

    # Load external event sources
    floodlight_events = load_Floodlight_events(
        data_dir,
        match_id,
    )

    kloppy_events = load_kloppy_events(
        data_dir,
        match_id,
    )

    # Enrich events
    game = process_events(
        game,
        floodlight_events,
        kloppy_events,
    )

    # possessions = segment_possessions(
    #     game
    # )

    # return game, possessions
    
    return game
