# reticulate::use_condaenv("databallpy_env")
# reticulate::repl_python()


import os
from pathlib import Path

import pandas as pd
import jax.numpy as jnp


from PRESS import PRESS

from PRESS.visualization.plot import (
    plot_frame,
    plot_bcos,
    plot_ppcf
)

bcos_args = {

    "MAX_POINTS": 100,
    "dx": 0.5,

    "s": 0.45,

    "tau_r_bc": 0,
    "delta_r_def": 0.5,

    "v_bc_max": 7.62,
    "v_def_max": 8.54,
    
    "alpha": 1.0,

    "dt": 0.04,

    "T_max": 15.0,

    "tol": 1e-7,
}

ppcf_args = {

    "tau_r": 0.5,

    "alpha": 1.0,

    "v_max": 8.54,


    "v0_candidates": jnp.linspace(
        10.0,
        30.0,
        20,
    ),


    "theta_candidates": jnp.linspace(
        0.0,
        jnp.deg2rad(60),
        20,
    ),


    "t_max_flight": 10.0,

    "dt_ball": 0.04,


    "z_max": 2.0,


    "rho": 1.2,

    "Cd": 0.25,

    "A": 0.038,

    "m": 0.43,

    "g": 9.81,


    "t_max_pc": 12.0,

    "dt_pc": 0.04,


    "lambda_control": 4.3,

    "s": 0.45,

    "tol": 1e-6,
}



# ------------------------------------------------------------------
# Working directory
# ------------------------------------------------------------------

os.chdir(
    "/Users/riccardobergamin/Documents/GitHub/PRESS/examples/IDSEE"
)



data_dir = Path("datasets")

match_id = "J03WMX"



# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

event_data = pd.read_parquet(
    data_dir / f"{match_id}_events.parquet"
)

tracking_data = pd.read_parquet(
    data_dir / f"{match_id}_tracking.parquet"
)


# ------------------------------------------------------------------
# PRESS object
# ------------------------------------------------------------------

game = PRESS(
    event_data=event_data,
    tracking_data=tracking_data,
)

# ------------------------------------------------------------------
# Frame
# ------------------------------------------------------------------

game.tracking_data[game.tracking_data.player_possession.notna()].frame[600:700]

frame_id =   1713

plot_frame(
    game,
    frame_id=frame_id,
)


# ------------------------------------------------------------------
# BCOS
# ------------------------------------------------------------------

game.bcos(
    frame_id=frame_id,
    r_max=30,
    step=0.5,
    physics_args=bcos_args,
)

plot_bcos(
    game,
    frame_id,
)


# ------------------------------------------------------------------
# PPCF
# ------------------------------------------------------------------


game.ppcf(
    frame_id=frame_id,
    r_max=50,
    sigma = 14.0,
    k0 = 0.1,
    kv = 0.8,
    kd = 0.08,
    step=0.5,
    physics_kwargs=ppcf_args,
)

plot_ppcf(
    game,
    frame_id,
)


# ------------------------------------------------------------------
# Pressing Intensity
# ------------------------------------------------------------------

game.pressing_intensity(
    frame_id=frame_id,
    t = 1.5,
    s = 0.45,
    tau_r_def = 0.5,
    v_def_max = 8.54,
    alpha = 1.0,
    v_thr=2.0
)

