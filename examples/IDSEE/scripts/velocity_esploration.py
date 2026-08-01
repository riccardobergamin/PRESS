import pandas as pd
import jax
import jax.numpy as jnp
import numpy as np
import math
import os
import sys

os.chdir(
    "/Users/riccardobergamin/Documents/GitHub/BCOS/examples/IDSEE"
)

sys.path.append(
    "/Users/riccardobergamin/Documents/GitHub/BCOS/examples/IDSEE/scripts"
)

from tracking import long_from_wide

# --- DATA LOADING ---
vel1 = pd.read_parquet("datasets/J03WMX_tracking.parquet")
vel2 = pd.read_parquet("datasets/J03WN1_tracking.parquet")

vel1 = long_from_wide(vel1)
vel2 = long_from_wide(vel2)

vel_all = pd.concat(
    [
        vel1,
        vel2,
    ],
    ignore_index=True
)

vel_all = vel_all[
    ~vel_all["player_id"].str.contains("ball")
]


# --- VELOCITY STATS ---
v_stats = (
    vel_all
    .groupby(
        [
            "player_id",
            "is_ball_carrier"
        ]
    )["v"]
    .max()
    .unstack()
)

v_stats.columns = [
    "v_max_off",
    "v_max_on"
]


# --- VELOCITY STATS ---
v_stats = vel_all.groupby(['player_id', 'is_ball_carrier'])['v'].max().unstack()
v_stats.columns = ['v_max_off', 'v_max_on']

# --- CALCOLO STATISTICHE (Media, Quantili specifici, Moda) ---
avg_v = v_stats.mean()
quantiles = v_stats.quantile([0.05, 0.15, 0.5, 0.75, 0.95])

print("--- SUMMARY STATS ---")
print(f"MEDIA:\n{avg_v}\n")
print(f"QUANTILI RICHIESTI:\n{quantiles}\n")


import matplotlib.pyplot as plt
import seaborn as sns

# Set the figure style
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Plot Distributions
sns.histplot(v_stats['v_max_off'], kde=True, color="royalblue", label="Without Ball (Defense)", stat="density", alpha=0.5)
sns.histplot(v_stats['v_max_on'], kde=True, color="crimson", label="With Ball (Attack)", stat="density", alpha=0.5)

# Add Labels and Title
plt.title("Distribution of Maximum Velocities ($V_{max}$)", fontsize=14)
plt.xlabel("Velocity (m/s)", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.legend()

# Show the plot
plt.show()






