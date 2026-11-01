"""Draw the inference data flow without running a match, fit or vignette.

Run from any directory with ``python path/to/scripts/build_model_overview.py``.
The five core models have separate blocks. Shared predictors are shown once;
choice distributions and reused action features have explicit downstream edges.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.path import Path as DrawingPath


ROOT = Path(__file__).resolve().parents[1]
INK = "#203349"
GRAY = "#758495"
BLUE = "#347aa7"
GREEN = "#2d8069"
PURPLE = "#775799"
AMBER = "#b47b26"


def build_overview() -> None:
    """Write the SVG/PNG used by the overview page and README."""
    fig, ax = plt.subplots(figsize=(11.8, 12), layout="constrained")
    ax.set(xlim=(0, 17.8), ylim=(0, 18))
    ax.axis("off")
    fig.patch.set_facecolor("white")

    def box(x, y, w, h, title, detail, *, edge=GRAY, fill="#f4f6f9",
            title_size=14, detail_size=11.5):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
            facecolor=fill, edgecolor=edge, linewidth=1.5, zorder=3,
        ))
        ax.text(x+w/2, y+h*.73, title, ha="center", va="center",
                fontsize=title_size, weight="bold", color=INK, zorder=4)
        ax.text(x+w/2, y+h*.31, detail, ha="center", va="center",
                fontsize=detail_size, color=INK, linespacing=1.45, zorder=4)

    def arrow(points, *, color=GRAY, label=None, at=None, label_size=11):
        path = DrawingPath(points, [DrawingPath.MOVETO]
                           + [DrawingPath.LINETO]*(len(points)-1))
        ax.add_patch(FancyArrowPatch(
            path=path, arrowstyle="-|>", mutation_scale=13,
            linewidth=1.5, color=color, zorder=2,
        ))
        if label:
            ax.text(*at, label, ha="center", va="center", fontsize=label_size,
                    color=color, zorder=5,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

    ax.text(8.9, 17.6, "PRESS model flow", ha="center", va="center",
            fontsize=21, weight="bold", color=INK)

    # Observed data: tracking is distinct from dynamic event annotations.
    box(.8, 15.3, 11.9, 1.35, "2D tracking data",
        "Ball and player positions and velocities", title_size=16,
        detail_size=13)
    box(13.4, 15.3, 3.5, 1.35, "Dynamic events",
        "Possession / action context", title_size=13.5, detail_size=10.5)

    # Physics-based execution and state-derived inputs.
    box(.8, 12.1, 3.5, 1.85, "Ball-drive risk",
        "Carry loss within 1 s\n"+r"$d$", edge=BLUE, fill="#eaf3fa")
    box(5., 12.1, 3.5, 1.85, "Pass risk",
        "Raw pitch control\n"+r"Pass success $p(z)$", edge=BLUE, fill="#eaf3fa")
    box(9.2, 12.1, 3.5, 1.85, "Spatial / tactical state",
        "Carrier state · pressure lines\nxT · BCOS · player counts",
        title_size=12.5, detail_size=10.5)
    box(13.4, 12.1, 3.5, 1.85, "Event context",
        "Start / previous action\nMatch time · score",
        title_size=13, detail_size=11)
    for x in (2.55, 6.75, 10.95):
        arrow([(x, 15.3), (x, 13.95)])
    arrow([(15.15, 15.3), (15.15, 13.95)])

    box(.8, 10.05, 16.1, 1.15, "Shared predictor inputs",
        "Kinematics + spatial / tactical state + event context + physical-model outputs",
        edge=GRAY, fill="#f4f6f9", title_size=14, detail_size=11.5)
    for x, color in [(2.55, BLUE), (6.75, BLUE), (10.95, GRAY), (15.15, GRAY)]:
        arrow([(x, 12.1), (x, 11.2)], color=color)

    # Each learned component has its own node. The action input record is
    # reused by residual risk, with elapsed time replacing whole-carry count.
    box(1.8, 6.6, 4., 2.1, "Pass selection",
        "Destination distribution\n"+r"$\pi(z)=P(z\mid pass)$",
        edge=GREEN, fill="#eaf5ef", title_size=15, detail_size=12)
    box(7., 6.6, 4., 2.1, "Action selection",
        "Pass / carry choice\n"+r"$q=P(pass)$",
        edge=GREEN, fill="#eaf5ef", title_size=15, detail_size=12)
    box(12.8, 6.6, 4., 2.1, "Residual risk",
        "Terminal loss after a\nsuccessful 1 s carry: "+r"$c$",
        edge=GREEN, fill="#eaf5ef", title_size=15, detail_size=11.5)
    arrow([(3.8, 10.05), (3.8, 8.7)], label="State + raw control + d",
          at=(3.8, 9.4), label_size=10.5)
    arrow([(9., 10.05), (9., 8.7)], label=r"State + $d,\ p(z)$",
          at=(9., 9.4), label_size=11)
    arrow([(14.8, 10.05), (14.8, 8.7)], label="State + elapsed time",
          at=(14.8, 9.4), label_size=10.5)
    arrow([(5.8, 7.65), (7., 7.65)], color=GREEN,
          label=r"$\pi(z)$", at=(6.4, 8.05), label_size=12)
    arrow([(11., 7.65), (12.8, 7.65)], color=GREEN,
          label="Features +\nP(carry)", at=(11.9, 8.12), label_size=10)

    # The local mixture needs execution risks AND both choice outputs.
    box(1.2, 3., 8.9, 2., "Hazard function h",
        r"$h=(1-q)d+q\sum_z\pi(z)[1-p(z)]$"
        +"\nCurrent-action possession fragility",
        edge=PURPLE, fill="#f1edf8", title_size=16, detail_size=13.5)
    arrow([(.8, 10.6), (.3, 10.6), (.3, 4.), (1.2, 4.)], color=BLUE,
          label=r"$d$"+"\n"+r"$p(z)$", at=(.4, 5.8), label_size=12)
    arrow([(3.8, 6.6), (3.8, 5.)], color=GREEN,
          label=r"$\pi(z)$", at=(3.8, 5.8), label_size=12)
    arrow([(9., 6.6), (9., 5.)], color=GREEN,
          label=r"$q$", at=(9., 5.8), label_size=12)

    box(1.2, .6, 8.9, 1.35, "Difficulty bands",
        "Minimal · Low · Medium · High", edge=AMBER, fill="#fff6e7",
        title_size=16, detail_size=13)
    arrow([(5.65, 3.), (5.65, 1.95)], color=PURPLE)

    # Successful passing continues the team sequence underlying h. The
    # continuation term instead completes individual-possession expected loss.
    box(12., 3., 5.2, 2., "Expected loss (xLoss)",
        r"$xLoss=h+(1-q)(1-d)c$"+"\nIndividual-possession outcome",
        edge=PURPLE, fill="#f1edf8", title_size=14.5, detail_size=12)
    arrow([(14.8, 6.6), (14.8, 5.)], color=GREEN,
          label=r"$c$", at=(14.8, 5.8), label_size=12)
    arrow([(10.1, 4.), (12., 4.)], color=PURPLE,
          label=r"$h,\ (1-q)(1-d)$", at=(11.05, 4.5), label_size=9.5)

    ax.legend(handles=[
        Patch(facecolor="#eaf3fa", edgecolor=BLUE, label="Physics-based models"),
        Patch(facecolor="#eaf5ef", edgecolor=GREEN, label="Learned models"),
        Patch(facecolor="#f1edf8", edgecolor=PURPLE, label="Composed predictions"),
    ], loc="lower right", bbox_to_anchor=(.98, .025), frameon=False,
        fontsize=10.5, labelspacing=.55)

    for ext in ("svg", "png"):
        fig.savefig(ROOT/"figures"/f"model_overview.{ext}", dpi=180,
                    metadata={"Creator": "PRESS model-overview builder"}
                    if ext == "svg" else None)
    plt.close(fig)


if __name__ == "__main__":
    build_overview()
