import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


def get_pv_players(
    frame,
    tracking_df,
):
    """
    Extract player positions and velocities from canonical wide tracking data.

    Parameters
    ----------
    frame : int
        Frame identifier.

    tracking_df : pd.DataFrame
        Canonical wide tracking format.

    Returns
    -------
    dict
        Ball carrier, teammates and defenders state,
        including player IDs.
    """

    def pad_players(
        coords,
        vels,
        n_required,
    ):

        coords = np.asarray(coords, dtype=float).reshape(-1, 2)
        vels = np.asarray(vels, dtype=float).reshape(-1, 2)

        if len(coords) < n_required:

            n_pad = n_required - len(coords)

            coords = np.vstack(
                [
                    coords,
                    np.full(
                        (n_pad, 2),
                        1000.0,
                    ),
                ]
            )

            vels = np.vstack(
                [
                    vels,
                    np.zeros(
                        (n_pad, 2)
                    ),
                ]
            )

        return (
            coords[:n_required],
            vels[:n_required],
        )


    # --------------------
    # Initial frame
    # --------------------

    frame_df = tracking_df[
        tracking_df.frame == frame
    ]

    if len(frame_df) != 1:
        raise ValueError(
            "Missing or duplicated frame in tracking data."
        )

    df = frame_df.iloc[0]
    pitch_length = float(df.get("pitch_length", 105.0))
    pitch_width = float(df.get("pitch_width", 68.0))
    if not np.isfinite([pitch_length, pitch_width]).all() or min(pitch_length, pitch_width) <= 0:
        raise ValueError("Pitch dimensions must be finite and positive.")


    # --------------------
    # Ball carrier
    # --------------------

    player = df["player_possession"]
    
    attacking_side = df["attacking_side"]

    if pd.isna(player):
        raise ValueError(
            "Missing player possession information."
        )


    team_id, shirt_num = player.split("_")

    shirt_num = int(
        shirt_num
    )


    bc_prefix = (
        f"{team_id}_{shirt_num}_"
    )


    x_bc = np.array(
        [
            df[bc_prefix + "x"],
            df[bc_prefix + "y"],
        ]
    )

    v_bc = np.array(
        [
            df[bc_prefix + "vx"],
            df[bc_prefix + "vy"],
        ]
    )

    if attacking_side not in ("left_to_right", "right_to_left"):
        raise ValueError("Missing or invalid attacking direction.")
    if not np.all(np.isfinite(x_bc)) or not np.all(np.isfinite(v_bc)):
        raise ValueError("Ball carrier position and velocity must be finite.")


    # --------------------
    # Remaining players
    # --------------------

    teammates = []
    attacker_ids = []
    v_atts = []

    defenders = []
    defender_ids = []
    v_defs = []


    other_team = (
        "away"
        if team_id == "home"
        else "home"
    )


    for side, pos_container, vel_container, id_container in [

        (
            team_id,
            teammates,
            v_atts,
            attacker_ids,
        ),

        (
            other_team,
            defenders,
            v_defs,
            defender_ids,
        ),

    ]:

        for col in df.index:

            if (
                not col.startswith(side + "_")
                or not col.endswith("_x")
            ):
                continue


            shirt = int(
                col.split("_")[1]
            )


            # Exclude ball carrier

            if (
                side == team_id
                and shirt == shirt_num
            ):
                continue


            player_id = (
                f"{side}_{shirt}"
            )


            prefix = (
                f"{player_id}_"
            )


            x = df[prefix + "x"]
            y = df[prefix + "y"]


            if pd.isna(x) or pd.isna(y):
                continue


            pos_container.append(
                [
                    x,
                    y,
                ]
            )

            vel_container.append(
                [
                    df[prefix + "vx"],
                    df[prefix + "vy"],
                ]
            )

            id_container.append(
                player_id
            )


    # --------------------
    # Padding
    # --------------------

    r_defs, v_defs = pad_players(
        np.asarray(defenders),
        np.asarray(v_defs),
        11,
    )


    r_atts, v_atts = pad_players(
        np.asarray(teammates),
        np.asarray(v_atts),
        10,
    )


    defender_ids = (
        defender_ids[:11]
        + [None] * (
            11 - len(defender_ids)
        )
    )


    attacker_ids = (
        attacker_ids[:10]
        + [None] * (
            10 - len(attacker_ids)
        )
    )


    # --------------------
    # Return
    # --------------------

    return {

        "frame_df": frame_df,
        "pitch_length": pitch_length,
        "pitch_width": pitch_width,

        "player": player,

        "team_id": team_id,

        "shirt_num": shirt_num,
        
        "attacking_side": attacking_side,

        "x_bc": x_bc,

        "v_bc": v_bc,

        "r_defs": r_defs,

        "v_defs": v_defs,

        "defender_ids": defender_ids,

        "r_atts": r_atts,

        "v_atts": v_atts,

        "attacker_ids": attacker_ids,
    }

# ============================================================
# xT
# ============================================================

def get_xt_values(
    targets,
    attacking_side,
    xt_grid,
    *, pitch_length=105.0, pitch_width=68.0,
):
    """
    Get attacking-team and opponent xT
    for arbitrary target positions.

    Parameters
    ----------
    targets : ndarray, shape (N, 2)
    """

    targets = np.asarray(
        targets,
        dtype=float,
    )

    xt_grid = np.asarray(
        xt_grid,
        dtype=float,
    )

    n_x, n_y = xt_grid.shape

    x = targets[:, 0]
    y = targets[:, 1]

    if attacking_side == "left_to_right":

        x_att = x
        y_att = y

    elif attacking_side == "right_to_left":

        x_att = -x
        y_att = -y

    else:

        raise ValueError(
            f"Invalid attacking_side: "
            f"{attacking_side}"
        )

    x_opp = -x_att
    y_opp = -y_att
    
    PITCH_LENGTH = float(pitch_length)
    PITCH_WIDTH = float(pitch_width)
    if not np.isfinite([PITCH_LENGTH, PITCH_WIDTH]).all() or min(PITCH_LENGTH, PITCH_WIDTH) <= 0:
        raise ValueError("Pitch dimensions must be finite and positive.")

    x_cell_att = np.clip(
        (
            (x_att + PITCH_LENGTH / 2)
            / PITCH_LENGTH
            * n_x
        ).astype(int),
        0,
        n_x - 1,
    )

    y_cell_att = np.clip(
        (
            (y_att + PITCH_WIDTH / 2)
            / PITCH_WIDTH
            * n_y
        ).astype(int),
        0,
        n_y - 1,
    )

    x_cell_opp = np.clip(
        (
            (x_opp + PITCH_LENGTH / 2)
            / PITCH_LENGTH
            * n_x
        ).astype(int),
        0,
        n_x - 1,
    )

    y_cell_opp = np.clip(
        (
            (y_opp + PITCH_WIDTH / 2)
            / PITCH_WIDTH
            * n_y
        ).astype(int),
        0,
        n_y - 1,
    )

    xt_team = xt_grid[
        x_cell_att,
        y_cell_att,
    ]

    xt_opp = xt_grid[
        x_cell_opp,
        y_cell_opp,
    ]

    return xt_team, xt_opp


# ============================================================
# DYNAMIC PRESSURE LINES
# ============================================================

def compute_dynamic_pressure_lines(
    r_defs,
    attacking_side,
    n_lines=3,
):
    """
    Compute defensive pressure lines once per pass.
    """

    r_defs = np.asarray(
        r_defs,
        dtype=float,
    )

    if r_defs.ndim != 2:
        return np.full(
            n_lines,
            np.nan,
        )

    if r_defs.shape[1] < 2:
        return np.full(
            n_lines,
            np.nan,
        )

    valid = np.all(
        np.isfinite(
            r_defs[:, :2]
        ),
        axis=1,
    )

    r_defs = r_defs[valid]

    if len(r_defs) < n_lines + 1:
        return np.full(
            n_lines,
            np.nan,
        )

    if attacking_side == "left_to_right":

        goalkeeper_idx = np.argmax(
            r_defs[:, 0]
        )

    elif attacking_side == "right_to_left":

        goalkeeper_idx = np.argmin(
            r_defs[:, 0]
        )

    else:

        raise ValueError(
            f"Unknown attacking_side: "
            f"{attacking_side}"
        )

    r_defs = np.delete(
        r_defs,
        goalkeeper_idx,
        axis=0,
    )

    X = r_defs[:, [0]]

    clustering = AgglomerativeClustering(
        n_clusters=n_lines,
        linkage="complete",
    )

    labels = clustering.fit_predict(X)

    lines = np.array([
        X[
            labels == cluster_id,
            0,
        ].mean()
        for cluster_id in range(n_lines)
    ])

    return np.sort(lines)


def get_target_pressure_line_index(
    targets,
    pressure_lines,
    attacking_side,
):
    """
    Return the number of pressure lines crossed
    by every candidate destination.

    Returns
    -------
    ndarray, shape (N,)
        0, 1, 2 or 3.
    """

    targets = np.asarray(
        targets,
        dtype=float,
    )

    pressure_lines = np.asarray(
        pressure_lines,
        dtype=float,
    )

    if (
        len(pressure_lines) != 3
        or not np.all(
            np.isfinite(
                pressure_lines
            )
        )
    ):

        return np.full(
            len(targets),
            np.nan,
        )

    if attacking_side == "left_to_right":

        target_x_att = targets[:, 0]
        lines_att = pressure_lines

    elif attacking_side == "right_to_left":

        target_x_att = -targets[:, 0]
        lines_att = -pressure_lines

    else:

        raise ValueError(
            f"Invalid attacking_side: "
            f"{attacking_side}"
        )

    lines_att = np.sort(
        lines_att
    )

    return np.sum(
        target_x_att[:, None]
        >= lines_att[None, :],
        axis=1,
    ).astype(float)
