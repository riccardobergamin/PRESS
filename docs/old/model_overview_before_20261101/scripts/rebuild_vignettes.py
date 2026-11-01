"""Execute vignette notebook cells in-process and save outputs and figures."""
from __future__ import annotations

import contextlib
import io
import os
import sys
import traceback
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
from nbformat.v4 import new_output

ROOT = Path(__file__).resolve().parents[1]



def execute_notebook(path: Path) -> None:
    # Each notebook is a standalone vignette; do not let imported package
    # modules or notebook-local monkeypatches leak from the previous one.
    for module_name in tuple(sys.modules):
        if module_name == "PRESS" or module_name.startswith("PRESS."):
            del sys.modules[module_name]
    notebook = nbformat.read(path, as_version=4)
    namespace = {"__name__": "__main__", "__file__": str(path)}
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        cell.execution_count = index + 1
        cell.outputs = []
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                exec(compile(cell.source, str(path), "exec"), namespace)
        except Exception:
            cell.outputs.append(new_output("stream", name="stderr", text=buffer.getvalue()))
            cell.outputs.append(new_output("error", ename="ExecutionError", evalue=f"{path.name}, cell {index + 1}", traceback=traceback.format_exc().splitlines()))
            nbformat.write(notebook, path)
            raise
        if buffer.getvalue():
            cell.outputs.append(new_output("stream", name="stdout", text=buffer.getvalue()))
        for number in plt.get_fignums():
            fig = plt.figure(number)
            png = io.BytesIO()
            fig.savefig(png, format="png", dpi=120)
            cell.outputs.append(new_output("display_data", data={"image/png": __import__("base64").b64encode(png.getvalue()).decode("ascii"), "text/plain": "Generated figure; PNG and SVG saved under figures/."}, metadata={}))
        plt.close("all")
    nbformat.write(notebook, path)


def build_overview() -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis("off")
    boxes = [
        (0.03, 0.61, 0.20, 0.24, "Tracking + events\npositions, velocities,\naction context"),
        (0.29, 0.67, 0.20, 0.19, "Physics branches\ncarry loss · pass control"),
        (0.29, 0.31, 0.20, 0.19, "Choice models\npass vs drive · destination"),
        (0.57, 0.50, 0.18, 0.24, "Next-action hazard\nweighted branch risk"),
        (0.81, 0.50, 0.16, 0.24, "Possession view\nbands · survival ·\ntypologies"),
    ]
    for x, y, w, h, label in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor="#e8f0f7", edgecolor="#315b7d", linewidth=1.5, transform=ax.transAxes))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10, transform=ax.transAxes)
    arrows = [((0.23, 0.73), (0.29, 0.76)), ((0.23, 0.68), (0.29, 0.41)), ((0.49, 0.75), (0.57, 0.65)), ((0.49, 0.40), (0.57, 0.58)), ((0.75, 0.62), (0.81, 0.62))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, xycoords=ax.transAxes, textcoords=ax.transAxes, arrowprops={"arrowstyle": "->", "lw": 1.7, "color": "#315b7d"})
    ax.set_title("PRESS model flow · team possession is the survival unit", fontsize=14, pad=16)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(ROOT / "figures" / f"model_overview.{ext}", dpi=150)
    plt.close(fig)

def main() -> None:
    build_overview()
    notebooks = sorted((ROOT / "vignettes").rglob("*.ipynb"))
    if not notebooks:
        raise SystemExit("No vignette notebooks found.")
    os.chdir(ROOT)
    if str(ROOT) in sys.path:
        sys.path.remove(str(ROOT))
    sys.path.insert(0, str(ROOT))
    for path in notebooks:
        execute_notebook(path)
        print(f"Executed {path.relative_to(ROOT)}")
    expected_figures = [ROOT / "figures" / f"vignette_{i:02d}_{name}.{ext}"
        for i, name in enumerate([
            "drive_risk", "pass_probability", "transition", "action_selection",
            "hazard_bands", "possession_setup", "survival", "typologies",
            "league_report", "bcos", "expected_threat",
        ], 1) for ext in ("png", "svg")]
    expected_figures.extend([ROOT / "figures" / f"model_overview.{ext}" for ext in ("png", "svg")])
    expected_figures.extend([ROOT / "figures" / f"vignette_00_match_workflow.{ext}" for ext in ("png", "svg")])
    expected_figures.extend([ROOT / "figures" / f"vignette_{name}.{ext}"
        for name in ("01_drive_pitch", "02_pass_pitch", "03_transition_pitch", "04_action_pitch", "12_residual_risk")
        for ext in ("png", "svg")])
    missing = [p.relative_to(ROOT) for p in expected_figures if not p.is_file()]
    if missing:
        raise SystemExit(f"Missing generated figures: {missing}")
    # Matplotlib emits insignificant trailing spaces in SVG path data.
    for path in expected_figures:
        if path.suffix == ".svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    print(f"Executed {len(notebooks)} notebooks; generated {len(expected_figures)} figures.")


if __name__ == "__main__":
    main()
