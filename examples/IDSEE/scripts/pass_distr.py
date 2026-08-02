# ==============================================================
# PASS DISTANCE AND ANGLE DISTRIBUTION
# Tracking-based passer position and velocity
# Matches: J03WMX + J03WN1
# ==============================================================


import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from PRESS import PRESS
from PRESS.utils.passes import get_passes



# --------------------------------------------------------------
# Setup
# --------------------------------------------------------------

os.chdir(
    "/Users/riccardobergamin/Documents/GitHub/PRESS/examples/IDSEE"
)


data_dir = Path("datasets")


match_ids = [
    "J03WMX",
    "J03WN1"
]



# --------------------------------------------------------------
# Angle function
# --------------------------------------------------------------

def angle_between_vectors(v1, v2):

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)


    if norm1 == 0 or norm2 == 0:
        return np.nan


    cos_angle = np.dot(v1, v2) / (
        norm1 * norm2
    )


    cos_angle = np.clip(
        cos_angle,
        -1,
        1
    )


    return np.degrees(
        np.arccos(cos_angle)
    )



# --------------------------------------------------------------
# Analyse single match
# --------------------------------------------------------------

def analyse_match(match_id):


    print(
        f"Processing {match_id}"
    )


    # -----------------------------
    # Load data
    # -----------------------------

    events_df = pd.read_parquet(
        data_dir / f"{match_id}_events.parquet"
    )


    tracking_df = pd.read_parquet(
        data_dir / f"{match_id}_tracking.parquet"
    )



    # -----------------------------
    # PRESS
    # -----------------------------

    game = PRESS(
        event_data=events_df,
        tracking_data=tracking_df
    )



    # -----------------------------
    # Passes
    # -----------------------------

    passes = get_passes(game)


    passes = passes[
        passes.is_successful
    ].copy()



    # -----------------------------
    # Dictionary player_id -> tracking name
    # -----------------------------

    dictionary_df = (
        events_df[
            [
                "player_id",
                "side",
                "shirt_num"
            ]
        ]
        .dropna()
        .drop_duplicates()
    )


    dictionary = dict(
        zip(
            dictionary_df["player_id"],
            dictionary_df["side"]
            + "_"
            + dictionary_df["shirt_num"]
            .astype(int)
            .astype(str)
        )
    )



    distances = []
    angles = []
    speeds = []



    # -----------------------------
    # Loop passes
    # -----------------------------

    for _, p in passes.iterrows():


        player_id = p["player"]

        frame = p["frame"]



        # player not found

        if player_id not in dictionary:

            distances.append(np.nan)
            angles.append(np.nan)
            speeds.append(np.nan)

            continue



        tracking_player = dictionary[player_id]



        # -------------------------
        # Extract frame
        # -------------------------

        frame_data = tracking_df[
            tracking_df.frame == frame
        ]


        if len(frame_data) == 0:

            distances.append(np.nan)
            angles.append(np.nan)
            speeds.append(np.nan)

            continue



        frame_data = frame_data.iloc[0]



        # -------------------------
        # Passer position
        # -------------------------

        try:

            passer_position = np.array([

                frame_data[
                    f"{tracking_player}_x"
                ],

                frame_data[
                    f"{tracking_player}_y"
                ]

            ])


            velocity = np.array([

                frame_data[
                    f"{tracking_player}_vx"
                ],

                frame_data[
                    f"{tracking_player}_vy"
                ]

            ])


        except KeyError:


            distances.append(np.nan)
            angles.append(np.nan)
            speeds.append(np.nan)

            continue



        # -------------------------
        # Receiver position
        # -------------------------

        receiver_position = np.array([

            p["receiver_x"],

            p["receiver_y"]

        ])



        # -------------------------
        # Pass vector
        # -------------------------

        pass_vector = (
            receiver_position -
            passer_position
        )



        distance = np.linalg.norm(
            pass_vector
        )



        speed = np.linalg.norm(
            velocity
        )



        angle = angle_between_vectors(
            velocity,
            pass_vector
        )



        distances.append(distance)
        angles.append(angle)
        speeds.append(speed)



    # -----------------------------
    # Add variables
    # -----------------------------

    passes["pass_distance"] = distances

    passes["pass_angle_bc"] = angles

    passes["passer_speed"] = speeds

    passes["match_id"] = match_id



    return passes




# ==============================================================
# Run matches
# ==============================================================


all_matches = []


for match in match_ids:

    df = analyse_match(match)

    all_matches.append(df)



passes = pd.concat(
    all_matches,
    ignore_index=True
)



# remove missing values

passes = passes.dropna(
    subset=[
        "pass_distance",
        "pass_angle_bc"
    ]
)



print(
    "\nTotal passes:",
    len(passes)
)



# ==============================================================
# Statistics
# ==============================================================


statistics = passes[
    [
        "pass_distance",
        "pass_angle_bc",
        "passer_speed"
    ]
].describe(
    percentiles=[
        0.05,
        0.25,
        0.50,
        0.75,
        0.95
    ]
)


print(
    statistics
)



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# variabili

corr_data = passes[
    [
        "passer_speed",
        "pass_distance",
        "pass_angle_bc"
    ]
].dropna()



# ------------------------------------------------
# Pearson correlation
# ------------------------------------------------

pearson = corr_data.corr(
    method="pearson"
)


print("Pearson correlation")
print(pearson)



# ------------------------------------------------
# Spearman correlation
# ------------------------------------------------

spearman = corr_data.corr(
    method="spearman"
)


print("\nSpearman correlation")
print(spearman)



# ==============================================================
# Plot distance
# ==============================================================


plt.figure(figsize=(8,5))


plt.hist(
    passes.pass_distance,
    bins=30
)


plt.axvline(
    passes.pass_distance.mean(),
    linestyle="--",
    label="mean"
)


plt.xlabel(
    "Pass distance (m)"
)

plt.ylabel(
    "Number of passes"
)


plt.title(
    "Distribution of pass distance"
)


plt.legend()

plt.show()



# ==============================================================
# Plot angle
# ==============================================================


plt.figure(figsize=(8,5))


plt.hist(
    passes.pass_angle_bc,
    bins=36
)


plt.axvline(
    passes.pass_angle_bc.mean(),
    linestyle="--",
    label="mean"
)


plt.xlabel(
    "Pass angle relative to passer velocity (deg)"
)


plt.ylabel(
    "Number of passes"
)


plt.title(
    "Distribution of pass angle"
)


plt.legend()

plt.show()



# ==============================================================
# Boxplots
# ==============================================================


plt.figure(figsize=(6,5))


plt.boxplot(
    passes.pass_distance
)


plt.ylabel(
    "Distance (m)"
)


plt.title(
    "Pass distance"
)


plt.show()



plt.figure(figsize=(6,5))


plt.boxplot(
    passes.pass_angle_bc
)


plt.ylabel(
    "Angle (deg)"
)


plt.title(
    "Pass angle"
)


plt.show()



# ==============================================================
# Match comparison
# ==============================================================


print("\nMATCH COMPARISON")


print(
    passes.groupby("match_id")
    [
        [
            "pass_distance",
            "pass_angle_bc",
            "passer_speed"
        ]
    ]
    .agg(
        [
            "mean",
            "median",
            "std"
        ]
    )
)



