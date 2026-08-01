from pathlib import Path
import os
import sys

# import BCOS
# from BCOS.utils.possessions import segment_possessions


sys.path.append(
    "/Users/riccardobergamin/Documents/GitHub/BCOS/examples/IDSEE/scripts"
)

from pipeline import preprocess_match
from tracking import tracking_to_long


os.chdir(
    "/Users/riccardobergamin/Dropbox/football"
)


data_dir = Path("IDSEE")
output_dir =  "/Users/riccardobergamin/Documents/GitHub/BCOS/examples/IDSEE/datasets/"

matches = [
    "J03WMX",
    "J03WN1",
]


for match_id in matches:

    print(f"Processing {match_id}")


    # game, possessions = preprocess_match(
    #     data_dir=str(data_dir) + "/",
    #     match_id=match_id,
    # )
    
    game = preprocess_match(
        data_dir=str(data_dir) + "/",
        match_id=match_id,
    )

    # Save event data
    game.event_data.to_parquet(
        output_dir + f"{match_id}_events.parquet",
        index=False,
        compression="zstd",
    )


    # Save tracking data
    game.tracking_data.to_parquet(
        output_dir + f"{match_id}_tracking.parquet",
        index=False,
        compression="zstd",
    )


    # Save possessions
    # possessions.to_parquet(
    #     output_dir / f"{match_id}_possessions.parquet",
    #     index=False,
    #     compression="zstd",
    # )


    print(
        f"{match_id} completed"
    )
