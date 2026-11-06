"""Render the approved PRESS logo as SVG/PNG and a square favicon."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, Polygon

ROOT = Path(__file__).resolve().parent
NAVY = "#17354b"
WHITE = "#f8fbfd"
BLUE = "#4aa8d8"
RED = "#e27c79"
PITCH_GREEN = "#327454"


def draw_logo(ax):
    """Draw the pitch, three closing defenders and the escaping carrier."""
    hexagon = [[2, 4.25], [3.82, 3.20], [3.82, 1.10],
               [2, .05], [.18, 1.10], [.18, 3.20]]
    ax.add_patch(Polygon(hexagon, closed=True, facecolor=NAVY, edgecolor=NAVY,
                         linewidth=2.5, joinstyle="round", zorder=0))
    ax.set(xlim=(-.02, 4.02), ylim=(-.12, 4.4), aspect="equal")
    ax.axis("off")
    ax.text(2, .92, "PRESS", ha="center", va="center", weight="bold",
            fontsize=31, color=WHITE, fontfamily="DejaVu Sans")

    ax.add_patch(Polygon([[.78, 1.62], [3.22, 1.62], [3.22, 3.39], [.78, 3.39]],
                         closed=True, facecolor=PITCH_GREEN, edgecolor="none", zorder=1))
    ax.plot([.78, .78, 3.22, 3.22, .78], [1.62, 3.39, 3.39, 1.62, 1.62],
            color=WHITE, linewidth=1.7, alpha=.58, zorder=2)
    ax.plot([2, 2], [1.62, 3.39], color=WHITE, linewidth=1.2, alpha=.41, zorder=2)
    ax.add_patch(Arc((2, 2.50), .64, .64, theta1=0, theta2=360,
                     color=WHITE, linewidth=1.2, alpha=.41, zorder=2))
    for x in [[.78, 1.10, 1.10, .78], [3.22, 2.90, 2.90, 3.22]]:
        ax.plot(x, [2.08, 2.08, 2.92, 2.92], color=WHITE, linewidth=1.1,
                alpha=.5, zorder=2)

    # Every arrow has the same straight tracking-vector style.
    arrow_style = dict(arrowstyle="-|>", mutation_scale=12, linewidth=2,
                       alpha=.9, connectionstyle="arc3,rad=0", zorder=3)
    carrier = (2, 2.51)
    defenders = [(1.52, 2.82), (2.48, 2.82), (1.52, 2.25)]
    for x, y in defenders:
        target = (x + .60 * (carrier[0] - x), y + .60 * (carrier[1] - y))
        ax.add_patch(FancyArrowPatch((x, y), target, color=BLUE, **arrow_style))
        ax.scatter(x, y, s=82, c=BLUE, edgecolors=WHITE, linewidths=.6, zorder=4)
    ax.add_patch(FancyArrowPatch(carrier, (2.58, 2.18), color=RED, **arrow_style))
    ax.scatter(*carrier, s=200, c=RED, edgecolors=WHITE, linewidths=1.4, zorder=4)


def main():
    fig, ax = plt.subplots(figsize=(4, 4.5))
    fig.patch.set_alpha(0)
    draw_logo(ax)
    fig.subplots_adjust(left=.01, right=.99, bottom=.01, top=.99)
    for extension in ["svg", "png"]:
        fig.savefig(ROOT / f"press-logo.{extension}", dpi=180, transparent=True)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    fig.patch.set_alpha(0)
    draw_logo(ax)
    fig.subplots_adjust(left=.01, right=.99, bottom=.01, top=.99)
    fig.savefig(ROOT / "favicon.png", dpi=256 / 4.5, transparent=True)
    plt.close(fig)
    svg = ROOT / "press-logo.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()
