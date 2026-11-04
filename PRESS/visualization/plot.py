import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from matplotlib.lines import Line2D
from mplsoccer import Pitch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
from PRESS.utils.frame import frame_to_long, resolve_pitch_size


def plot_frame(
    press,
    frame_id,
    pitch_length=None,
    pitch_width=None,
    figsize=(12, 8),
    title_fontsize=9,
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


    pitch_length, pitch_width = resolve_pitch_size(frame_df, pitch_length, pitch_width)
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

        "home": "steelblue",

        "away": "indianred",

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
    bcos,
    frame_id,
    pitch_length=None,
    pitch_width=None,
    figsize=(12, 8),
    title_fontsize=9,
    legend_fontsize=8,
):
    if frame_id not in bcos.results:
        raise ValueError(f"BCOS frame {frame_id} not found.")

    frame_df = bcos.tracking_data[bcos.tracking_data["frame"] == frame_id].copy()
    if frame_df.empty:
        raise ValueError(f"Frame {frame_id} not found.")

    pitch_length, pitch_width = resolve_pitch_size(frame_df, pitch_length, pitch_width)
    frame_df = frame_to_long(frame_df)

    pitch = Pitch(
        pitch_type="skillcorner",
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        pitch_color="#7fbf7f",
        line_color="white",
        linewidth=1.3,
    )

    fig, ax = plt.subplots(figsize=figsize)
    pitch.draw(ax=ax)

    result = bcos.results[frame_id]
    grid = result["grid"]
    values = result["Ball carrier open space"]

    xi = np.linspace(grid[:, 0].min(), grid[:, 0].max(), 120)
    yi = np.linspace(grid[:, 1].min(), grid[:, 1].max(), 120)
    xx, yy = np.meshgrid(xi, yi)

    potential = griddata(
        grid,
        values,
        (xx, yy),
        method="linear"
    )

    team_ids = frame_df["team_id"].dropna().unique()
    if len(team_ids) != 2:
        raise ValueError("Frame must contain two teams.")

    t1, t2 = team_ids[0], team_ids[1]
    team_colors = {
        "home": "steelblue",
        "away": "indianred"
    }

    possession_team = frame_df.loc[
        frame_df["team_possession"].notna(),
        "team_id"
    ]

    if not possession_team.empty:
        attacking_team = possession_team.iloc[0]
    else:
        attacking_team = t1

    defending_team = t2 if attacking_team == t1 else t1

    # ----------------------------------
    # BCOS colormap
    # ----------------------------------

    bcos_cmap = plt.get_cmap("inferno")

    contour = ax.contourf(
        xx,
        yy,
        potential,
        levels=np.linspace(0, 1, 11),
        cmap=bcos_cmap,
        alpha=0.5,
        zorder=1,
    )

    cbar = plt.colorbar(
        contour,
        ax=ax,
        fraction=0.03,
        pad=0.02,
    )

    cbar.set_label(
        "Ball carrier open space",
        fontsize=legend_fontsize
    )

    cbar.ax.tick_params(
        labelsize=legend_fontsize
    )

    players = frame_df.dropna(subset=["player_id"])

    for team_id, color in team_colors.items():
        team_players = players[players.team_id == team_id]

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

    if "ball_x" in frame_df.columns and "ball_y" in frame_df.columns:
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

    if "vx" in frame_df.columns and "vy" in frame_df.columns:
        for _, player in players.dropna(subset=["vx", "vy"]).iterrows():

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

    ax.set_title(
        f"BCOS frame {frame_id}",
        fontsize=title_fontsize,
        fontweight="bold",
        pad=12,
    )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=team_colors[t1],
            markeredgecolor="white",
            markersize=9,
            label=f"Team {t1}",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=team_colors[t2],
            markeredgecolor="white",
            markersize=9,
            label=f"Team {t2}",
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

    fig.subplots_adjust(
        left=0.03,
        bottom=0.10,
        right=0.91,
        top=0.92,
    )

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        fontsize=legend_fontsize,
        frameon=True,
        ncol=3,
    )

    plt.show()
  
    

def plot_ppcf(
    press,
    frame_id,
    pitch_length=None,
    pitch_width=None,
    figsize=(12, 8),
    title_fontsize=9,
    legend_fontsize=8,
    *,
    field="pitch_control",
):
    """Plot stored attacking control or calibrated pass success on the pitch.

    field is 'pitch_control' (default) or 'pass_success'; compute_ppcf must
    have been run first. Interpolation is for display only. Returns None.
    """
    if field not in ("pitch_control", "pass_success"):
        raise ValueError("field must be pitch_control or pass_success.")
    if frame_id not in press.results["PPCF"]:
        raise ValueError(f"PPCF frame {frame_id} not found.")

    frame_df = press.tracking_data[press.tracking_data["frame"] == frame_id].copy()
    if frame_df.empty:
        raise ValueError(f"Frame {frame_id} not found.")

    pitch_length, pitch_width = resolve_pitch_size(frame_df, pitch_length, pitch_width)
    frame_df = frame_to_long(frame_df)

    pitch = Pitch(
        pitch_type="skillcorner",
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        pitch_color="#7fbf7f",
        line_color="white",
        linewidth=1.3,
    )

    fig, ax = plt.subplots(figsize=figsize)
    pitch.draw(ax=ax)

    result = press.results["PPCF"][frame_id]
    grid = result["grid"]
    values = result["Pass probability" if field == "pass_success" else "Attacking pitch control field"]

    xi = np.linspace(grid[:, 0].min(), grid[:, 0].max(), 120)
    yi = np.linspace(grid[:, 1].min(), grid[:, 1].max(), 120)
    xx, yy = np.meshgrid(xi, yi)

    potential = griddata(grid, values, (xx, yy), method="linear")

    team_ids = frame_df["team_id"].dropna().unique()
    if len(team_ids) != 2:
        raise ValueError("Frame must contain two teams.")

    t1, t2 = "home", "away"
    
    if frame_df.team_possession.iloc[0] == "home":
        team_divergent_cmap = LinearSegmentedColormap.from_list(
            "team_divergent", ["indianred", "#f7f7f7", "steelblue"]
        )
    else:
        team_divergent_cmap = LinearSegmentedColormap.from_list(
            "team_divergent", ["steelblue", "#f7f7f7", "indianred"]
        )

    contour = ax.contourf(
        xx, yy, potential,
        levels=np.linspace(0, 1, 11),
        cmap=team_divergent_cmap,
        alpha=0.6,
        zorder=1,
    )

    cbar = plt.colorbar(
        contour, ax=ax,
        fraction=0.03,  
        pad=0.02,       
    )
    field_label = "Pass-success probability" if field == "pass_success" else "Attacking pitch-control probability"
    cbar.set_label(field_label, fontsize=legend_fontsize)
    cbar.ax.tick_params(labelsize=legend_fontsize)

    team_colors = {"home": "steelblue", "away": "indianred"}
    players = frame_df.dropna(subset=["player_id"])

    for team_id, color in team_colors.items():
        team_players = players[players.team_id == team_id]
        pitch.scatter(
            team_players.x, team_players.y, ax=ax,
            s=45, color=color, edgecolors="white", linewidth=0.6, zorder=3
        )

    if "ball_x" in frame_df.columns and "ball_y" in frame_df.columns:
        pitch.scatter(
            frame_df["ball_x"].iloc[0], frame_df["ball_y"].iloc[0], ax=ax,
            s=20, color="white", edgecolors="black", linewidth=0.8, zorder=10
        )

    if "vx" in frame_df.columns and "vy" in frame_df.columns:
        for _, player in players.dropna(subset=["vx", "vy"]).iterrows():
            pitch.arrows(
                player.x, player.y,
                player.x + player.vx * 1.5,
                player.y + player.vy * 1.5,
                ax=ax, color="black", width=0.5, headwidth=3, headlength=3,
                alpha=0.55, zorder=4
            )

    ax.set_title(f"{field_label} · frame {frame_id}", fontsize=title_fontsize, fontweight="bold", pad=12)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=team_colors[t1], markeredgecolor="white", markersize=9, label=f"Team {t1}"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=team_colors[t2], markeredgecolor="white", markersize=9, label=f"Team {t2}"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="black", markersize=5, label="Ball"),
    ]

    fig.subplots_adjust(left=0.03, bottom=0.10, right=0.91, top=0.92)

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),  
        fontsize=legend_fontsize,
        frameon=True,
        ncol=3,
    )

    plt.show()

    
def plot_pi(
    press,
    frame_id,
    pitch_length=None,
    pitch_width=None,
    figsize=(12, 8),
    title_fontsize=9,
    legend_fontsize=8,
):
    """
    Plot each defender's share of the stored ball-drive loss probability.

    Colors show cause-specific probabilities divided by the total loss
    probability, on a fixed 0–100% scale. Zero total risk gives zero shares.
    The title reports the absolute drive-loss probability. Returns None.
    """

    if frame_id not in press.results["PI"]:
        raise ValueError(
            f"PI frame {frame_id} not found."
        )


    frame_df = press.tracking_data[
        press.tracking_data["frame"] == frame_id
    ].copy()


    if frame_df.empty:
        raise ValueError(
            f"Frame {frame_id} not found."
        )


    pitch_length, pitch_width = resolve_pitch_size(frame_df, pitch_length, pitch_width)
    frame_df = frame_to_long(
        frame_df
    )


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
    # Pressing intensity
    # ----------------------------------

    result = press.results["PI"][frame_id]
    total_loss_probability = float(result["Ball drive loss probability"])
    cause_probabilities = np.asarray(result["Individual tackle probabilities"], dtype=float)
    pressing_intensity = (
        cause_probabilities / total_loss_probability
        if total_loss_probability > 0.0
        else np.zeros_like(cause_probabilities)
    )


    # ----------------------------------
    # Teams
    # ----------------------------------

    team_colors = {
        "home": "steelblue",
        "away": "indianred",
    }


    possession_team = (
        frame_df["team_possession"]
        .iloc[0]
    )


    defending_team = (
        "away"
        if possession_team == "home"
        else "home"
    )


    players = frame_df.dropna(
        subset=["player_id"]
    )


    # ----------------------------------
    # Pressing intensity colormap
    # ----------------------------------

    base_color = team_colors[
        defending_team
    ]


    pi_cmap = LinearSegmentedColormap.from_list(
        "pressing_intensity",
        [
            "white",
            base_color,
        ],
    )


    norm = plt.Normalize(
        vmin=0.0,
        vmax=1.0,
    )


    # ----------------------------------
    # Defenders
    # ----------------------------------

    defenders = players[
        players.team_id == defending_team
    ]


    for i, (_, player) in enumerate(
        defenders.iterrows()
    ):

        intensity = float(
            pressing_intensity[i]
        )


        pitch.scatter(
            player.x,
            player.y,
            ax=ax,
            s=60,
            color=pi_cmap(
                norm(intensity)
            ),
            edgecolors="white",
            linewidth=0.7,
            zorder=3,
        )

    # ----------------------------------
    # Attacking team
    # ----------------------------------

    attackers = players[
        players.team_id == possession_team
    ]


    pitch.scatter(
        attackers.x,
        attackers.y,
        ax=ax,
        s=45,
        color=team_colors[possession_team],
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
    # Colorbar
    # ----------------------------------

    sm = plt.cm.ScalarMappable(
        cmap=pi_cmap,
        norm=norm,
    )

    sm.set_array([])


    cbar = plt.colorbar(
        sm,
        ax=ax,
        fraction=0.03,
        pad=0.02,
    )


    cbar.set_label(
        "Share of total loss risk",
        fontsize=legend_fontsize,
    )

    cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    cbar.ax.tick_params(
        labelsize=legend_fontsize
    )


    # ----------------------------------
    # Title
    # ----------------------------------

    ax.set_title(
        f"Ball-drive loss · frame {frame_id} · P(loss)={float(press.results['PI'][frame_id]['Ball drive loss probability']):.3f}",
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
            markerfacecolor=team_colors["home"],
            markeredgecolor="white",
            markersize=9,
            label="Home",
        ),

        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=team_colors["away"],
            markeredgecolor="white",
            markersize=9,
            label="Away",
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
        bottom=0.10,
        top=0.92,
        right=0.90,
    )


    plt.show()    
    
    
