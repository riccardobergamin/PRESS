import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from matplotlib.lines import Line2D
from mplsoccer import Pitch

from PRESS.utils.frame import frame_to_long


def plot_frame(
    press,
    frame_id,
    pitch_length=105,
    pitch_width=68,
    figsize=(12, 8),
    title_fontsize=8,
    legend_fontsize=8,
):
    """
    Plot a tracking frame.

    Parameters
    ----------
    press : PRESS
        PRESS object.

    frame_id : int
        Frame identifier.
    """

    frame_df = press.tracking_data[
        press.tracking_data["frame"] == frame_id
    ].copy()


    if frame_df.empty:
        raise ValueError(
            f"Frame {frame_id} not found."
        )


    frame_df = frame_to_long(
        frame_df
    )


    # ----------------------------------
    # Pitch
    # ----------------------------------

    pitch = Pitch(
        pitch_type="skillcorner",
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        pitch_color="#7fbf7f",
        line_color="white",
        linewidth=1.3,
    )


    fig, ax = plt.subplots(
        figsize=figsize
    )


    pitch.draw(
        ax=ax
    )


    # ----------------------------------
    # Teams
    # ----------------------------------

    team_ids = (
        frame_df["team_id"]
        .dropna()
        .unique()
    )


    if len(team_ids) != 2:
        raise ValueError(
            "Frame must contain two teams."
        )


    team_colors = {

        team_ids[0]: "steelblue",

        team_ids[1]: "indianred",

    }


    players = frame_df.dropna(
        subset=["player_id"]
    )


    for team_id, color in team_colors.items():

        team_players = players[
            players.team_id == team_id
        ]


        pitch.scatter(
            team_players.x,
            team_players.y,
            ax=ax,
            s=45,
            color=color,
            edgecolors="white",
            linewidth=0.6,
            zorder=3,
        )


    # ----------------------------------
    # Ball
    # ----------------------------------

    if (
        "ball_x" in frame_df.columns
        and "ball_y" in frame_df.columns
    ):

        pitch.scatter(
            frame_df["ball_x"].iloc[0],
            frame_df["ball_y"].iloc[0],
            ax=ax,
            s=20,
            color="white",
            edgecolors="black",
            linewidth=0.8,
            zorder=10,
        )


    # ----------------------------------
    # Velocity vectors
    # ----------------------------------

    if (
        "vx" in frame_df.columns
        and "vy" in frame_df.columns
    ):

        for _, player in players.dropna(
            subset=["vx", "vy"]
        ).iterrows():

            pitch.arrows(
                player.x,
                player.y,
                player.x + player.vx * 1.5,
                player.y + player.vy * 1.5,
                ax=ax,
                color="black",
                width=0.5,
                headwidth=3,
                headlength=3,
                alpha=0.55,
                zorder=4,
            )


    # ----------------------------------
    # Title
    # ----------------------------------

    ax.set_title(
        f"Tracking frame {frame_id}",
        fontsize=title_fontsize,
        fontweight="bold",
        pad=12,
    )


    # ----------------------------------
    # Legend
    # ----------------------------------

    legend_elements = [

        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=team_colors[team_ids[0]],
            markeredgecolor="white",
            markersize=9,
            label=f"Team {team_ids[0]}",
        ),


        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=team_colors[team_ids[1]],
            markeredgecolor="white",
            markersize=9,
            label=f"Team {team_ids[1]}",
        ),


        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=5,
            label="Ball",
        ),

    ]


    fig.legend(
        handles=legend_elements,
        loc="lower center",
        fontsize=legend_fontsize,
        frameon=True,
        ncol=3,
    )


    fig.subplots_adjust(
        bottom=0.08,
        top=0.92,
    )


    plt.show()
    
    
def plot_bcos(
    press,
    frame_id,
    pitch_length=105,
    pitch_width=68,
    figsize=(12, 8),
    title_fontsize=9,
    legend_fontsize=8,
):
    """
    Plot tracking frame with BCOS potential overlay.

    Parameters
    ----------
    press : PRESS
        PRESS object.

    frame_id : int
        Frame identifier.
    """

    from matplotlib.colors import LinearSegmentedColormap


    if frame_id not in press.results["BCOS"]:
        raise ValueError(
            f"BCOS frame {frame_id} not found."
        )


    # ----------------------------------
    # Frame
    # ----------------------------------

    frame_df = press.tracking_data[
        press.tracking_data["frame"] == frame_id
    ].copy()


    if frame_df.empty:
        raise ValueError(
            f"Frame {frame_id} not found."
        )


    frame_df = frame_to_long(
        frame_df
    )


    # ----------------------------------
    # Pitch
    # ----------------------------------

    pitch = Pitch(
        pitch_type="skillcorner",
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        pitch_color="#7fbf7f",
        line_color="white",
        linewidth=1.3,
    )


    fig, ax = plt.subplots(
        figsize=figsize
    )


    pitch.draw(
        ax=ax
    )


    # ----------------------------------
    # BCOS potential
    # ----------------------------------

    result = press.results["BCOS"][frame_id]


    grid = result["grid"]

    values = result["Ball carrier open space"]


    xi = np.linspace(
        grid[:, 0].min(),
        grid[:, 0].max(),
        120,
    )

    yi = np.linspace(
        grid[:, 1].min(),
        grid[:, 1].max(),
        120,
    )


    xx, yy = np.meshgrid(
        xi,
        yi,
    )


    potential = griddata(
        grid,
        values,
        (xx, yy),
        method="linear",
    )


    magenta_bcos = LinearSegmentedColormap.from_list(
        "magenta_bcos",
        [
            "#fde0ef",
            "#f768a1",
            "#dd3497",
            "#980043",
        ],
    )


    contour = ax.contourf(
        xx,
        yy,
        potential,
        levels=np.linspace(
            0,
            1,
            10,
        ),
        cmap=magenta_bcos,
        alpha=0.5,
        zorder=1,
    )


    cbar = plt.colorbar(
        contour,
        ax=ax,
        fraction=0.025,
        pad=0.02,
    )


    cbar.set_label(
        "Ball carrier open space",
        fontsize=legend_fontsize,
    )


    cbar.ax.tick_params(
        labelsize=legend_fontsize
    )
    
    # ----------------------------------
    # Teams
    # ----------------------------------

    team_ids = (
        frame_df["team_id"]
        .dropna()
        .unique()
    )


    if len(team_ids) != 2:
        raise ValueError(
            "Frame must contain two teams."
        )


    team_colors = {
        team_ids[0]: "steelblue",
        team_ids[1]: "indianred",
    }


    players = frame_df.dropna(
        subset=["player_id"]
    )


    for team_id, color in team_colors.items():

        team_players = players[
            players.team_id == team_id
        ]


        pitch.scatter(
            team_players.x,
            team_players.y,
            ax=ax,
            s=45,
            color=color,
            edgecolors="white",
            linewidth=0.6,
            zorder=3,
        )


    # ----------------------------------
    # Ball
    # ----------------------------------

    if (
        "ball_x" in frame_df.columns
        and "ball_y" in frame_df.columns
    ):

        pitch.scatter(
            frame_df["ball_x"].iloc[0],
            frame_df["ball_y"].iloc[0],
            ax=ax,
            s=20,
            color="white",
            edgecolors="black",
            linewidth=0.8,
            zorder=10,
        )


    # ----------------------------------
    # Velocity vectors
    # ----------------------------------

    if (
        "vx" in frame_df.columns
        and "vy" in frame_df.columns
    ):

        for _, player in players.dropna(
            subset=["vx", "vy"]
        ).iterrows():

            pitch.arrows(
                player.x,
                player.y,
                player.x + player.vx * 1.5,
                player.y + player.vy * 1.5,
                ax=ax,
                color="black",
                width=0.5,
                headwidth=3,
                headlength=3,
                alpha=0.55,
                zorder=4,
            )


    # ----------------------------------
    # Title
    # ----------------------------------

    ax.set_title(
        f"BCOS frame {frame_id}",
        fontsize=title_fontsize,
        fontweight="bold",
        pad=12,
    )


    # ----------------------------------
    # Legend left
    # ----------------------------------

    legend_elements = [

        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=team_colors[team_ids[0]],
            markeredgecolor="white",
            markersize=9,
            label=f"Team {team_ids[0]}",
        ),

        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=team_colors[team_ids[1]],
            markeredgecolor="white",
            markersize=9,
            label=f"Team {team_ids[1]}",
        ),

        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=5,
            label="Ball",
        ),

    ]


    fig.legend(
        handles=legend_elements,
        loc="lower center",
        fontsize=legend_fontsize,
        frameon=True,
        ncol=3,
    )


    fig.subplots_adjust(
        left=0.10,
        bottom=0.05,
        top=0.92,
    )


    plt.show()    
    

def plot_ppcf(
    press,
    frame_id,
    pitch_length=105,
    pitch_width=68,
    figsize=(12, 8),
    title_fontsize=9,
    legend_fontsize=8,
):
    """
    Plot tracking frame with PPCF potential overlay.

    Parameters
    ----------
    press : PRESS
        PRESS object.

    frame_id : int
        Frame identifier.
    """

    from matplotlib.colors import LinearSegmentedColormap


    if frame_id not in press.results["PPCF"]:
        raise ValueError(
            f"PPCF frame {frame_id} not found."
        )


    # ----------------------------------
    # Frame
    # ----------------------------------

    frame_df = press.tracking_data[
        press.tracking_data["frame"] == frame_id
    ].copy()


    if frame_df.empty:
        raise ValueError(
            f"Frame {frame_id} not found."
        )


    frame_df = frame_to_long(
        frame_df
    )


    # ----------------------------------
    # Pitch
    # ----------------------------------

    pitch = Pitch(
        pitch_type="skillcorner",
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        pitch_color="#7fbf7f",
        line_color="white",
        linewidth=1.3,
    )


    fig, ax = plt.subplots(
        figsize=figsize
    )


    pitch.draw(
        ax=ax
    )


    # ----------------------------------
    # PPCF potential
    # ----------------------------------

    result = press.results["PPCF"][frame_id]


    grid = result["grid"]

    values = result["Pitch control field"]


    xi = np.linspace(
        grid[:, 0].min(),
        grid[:, 0].max(),
        120,
    )

    yi = np.linspace(
        grid[:, 1].min(),
        grid[:, 1].max(),
        120,
    )


    xx, yy = np.meshgrid(
        xi,
        yi,
    )


    potential = griddata(
        grid,
        values,
        (xx, yy),
        method="linear",
    )


    peach_ppcf = LinearSegmentedColormap.from_list(
        "peach_ppcf",
        [
            "#fff5eb",
            "#fee6ce",
            "#fdae6b",
            "#e6550d",
        ],
    )


    contour = ax.contourf(
        xx,
        yy,
        potential,
        levels=np.linspace(
            0,
            1,
            10,
        ),
        cmap=peach_ppcf,
        alpha=0.6,
        zorder=1,
    )


    cbar = plt.colorbar(
        contour,
        ax=ax,
        fraction=0.025,
        pad=0.02,
    )


    cbar.set_label(
        "Pitch control field",
        fontsize=legend_fontsize,
    )


    cbar.ax.tick_params(
        labelsize=legend_fontsize
    )


    # ----------------------------------
    # Teams
    # ----------------------------------

    team_ids = (
        frame_df["team_id"]
        .dropna()
        .unique()
    )


    if len(team_ids) != 2:
        raise ValueError(
            "Frame must contain two teams."
        )


    team_colors = {
        team_ids[0]: "steelblue",
        team_ids[1]: "indianred",
    }


    players = frame_df.dropna(
        subset=["player_id"]
    )


    for team_id, color in team_colors.items():

        team_players = players[
            players.team_id == team_id
        ]


        pitch.scatter(
            team_players.x,
            team_players.y,
            ax=ax,
            s=45,
            color=color,
            edgecolors="white",
            linewidth=0.6,
            zorder=3,
        )


    # ----------------------------------
    # Ball
    # ----------------------------------

    if (
        "ball_x" in frame_df.columns
        and "ball_y" in frame_df.columns
    ):

        pitch.scatter(
            frame_df["ball_x"].iloc[0],
            frame_df["ball_y"].iloc[0],
            ax=ax,
            s=20,
            color="white",
            edgecolors="black",
            linewidth=0.8,
            zorder=10,
        )


    # ----------------------------------
    # Velocity vectors
    # ----------------------------------

    if (
        "vx" in frame_df.columns
        and "vy" in frame_df.columns
    ):

        for _, player in players.dropna(
            subset=["vx", "vy"]
        ).iterrows():

            pitch.arrows(
                player.x,
                player.y,
                player.x + player.vx * 1.5,
                player.y + player.vy * 1.5,
                ax=ax,
                color="black",
                width=0.5,
                headwidth=3,
                headlength=3,
                alpha=0.55,
                zorder=4,
            )


    # ----------------------------------
    # Title
    # ----------------------------------

    ax.set_title(
        f"PPCF frame {frame_id}",
        fontsize=title_fontsize,
        fontweight="bold",
        pad=12,
    )


    # ----------------------------------
    # Legend left
    # ----------------------------------

    legend_elements = [

        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=team_colors[team_ids[0]],
            markeredgecolor="white",
            markersize=9,
            label=f"Team {team_ids[0]}",
        ),

        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=team_colors[team_ids[1]],
            markeredgecolor="white",
            markersize=9,
            label=f"Team {team_ids[1]}",
        ),

        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=5,
            label="Ball",
        ),

    ]


    fig.legend(
        handles=legend_elements,
        loc="lower center",
        fontsize=legend_fontsize,
        frameon=True,
        ncol=3,
    )


    fig.subplots_adjust(
        left=0.10,
        bottom=0.05,
        top=0.92,
    )


    plt.show()
    

