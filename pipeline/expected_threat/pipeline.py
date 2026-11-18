"""Run the reviewed xT workflow from the accepted four event tables.

Reading order: model.py -> training.py -> this file.
Validation compares spatial grids; maps builds training-only six-match maps;
production refits the accepted 10x6 specification on all ten matches.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from pipeline.expected_threat.model import EVENT_TYPES, bin_events, plot_xt
from pipeline.expected_threat.training import fit_on_matches, save_fit, validate_grid, collect_validation


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PIPELINE_ROOT / "inputs" / "expected_threat"
DEFAULT_OUTPUT = PIPELINE_ROOT / "outputs" / "expected_threat"


def load_event_tables(data_root):
    """Load the fixed event cohort without consulting old resolution-bin columns.

    Carries are within-possession displacements; passes are the archived
    completions; shots have a goal label; losses are terminal zero-value events.
    These are not the canonical carry checkpoints or the verified pass-risk
    cohort. This review does not regenerate or filter them differently.
    """
    protocol = json.loads((data_root / "protocol.json").read_text())
    event_tables = {}
    for kind in EVENT_TYPES:
        events = pd.read_parquet(data_root / "events" / (kind + ".parquet"))
        events["match_id"] = events.match_id.astype(str)
        if not set(events.match_id).issubset(protocol["matches"]):
            raise ValueError("Unexpected event-table match.")
        if kind in ["passes", "carries"]:
            coordinates = ["x_start", "y_start", "x_end", "y_end"]
        else:
            location = "shot" if kind == "shots" else "loss"
            coordinates = ["x_" + location, "y_" + location]
        if not np.isfinite(events[coordinates].to_numpy()).all():
            raise ValueError("The accepted event coordinates must be finite.")
        event_tables[kind] = events
    return event_tables, protocol


def prepare_feature_maps(event_tables, protocol, output, fold=None):
    """Build xT for each downstream SIX-match fitting set, with its own smoothing.

    Training, validation and test events will later be sampled on this frozen
    fitting-only map. These maps are not the 8-match fits of the grid benchmark.
    """
    grid = tuple(protocol["selected_grid"])
    tables = bin_events(event_tables, grid)
    for current_fold in (range(5) if fold is None else [fold]):
        fitting_matches = protocol["feature_fitting_matches"][str(current_fold)]
        fitted, bandwidths, tuning, comparisons = fit_on_matches(tables, fitting_matches, grid)
        save_fit(output / "feature_maps" / f"fold_{current_fold}", fitted, grid,
                 fitting_matches, bandwidths, tuning, comparisons)
        print("XT SIX-MATCH MAP", current_fold, bandwidths, flush=True)


def refit_production(event_tables, protocol, output):
    """All ten matches: choose bandwidths by LOMO, then fit all model components.

    Uses the accepted grid, 10x6. The all-match surface is for production and
    descriptive plotting; it never substitutes for a fitting-only feature map.
    """
    grid = tuple(protocol["selected_grid"])
    tables = bin_events(event_tables, grid)
    fitted, bandwidths, tuning, comparisons = fit_on_matches(tables, protocol["matches"], grid)
    folder = output / "production"
    save_fit(folder, fitted, grid, protocol["matches"], bandwidths, tuning, comparisons)
    fig, _ = plot_xt(fitted["xt"])
    for extension in ["png", "svg"]:
        fig.savefig(folder / f"xt_surface.{extension}", dpi=180, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("XT PRODUCTION", bandwidths, "iterations", fitted["iterations"], flush=True)
    return fitted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["validation", "scores", "maps", "production", "all"])
    parser.add_argument("--grid", type=int, nargs=2, metavar=("NX", "NY"),
                        help="With validation: one of the four grids instead of all four.")
    parser.add_argument("--fold", type=int, choices=range(5), help="With maps: one six-match map.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.grid is not None and args.stage != "validation":
        parser.error("--grid only applies to validation.")
    if args.fold is not None and args.stage != "maps":
        parser.error("--fold only applies to maps.")
    event_tables, protocol = load_event_tables(args.data_root)
    if args.grid is not None and args.grid not in protocol["grids"]:
        parser.error("Choose a grid listed in protocol.json.")

    # 1. Rebin the same events at each resolution and evaluate held-out components.
    if args.stage in ["validation", "all"]:
        grids = [args.grid] if args.grid is not None else protocol["grids"]
        for grid in grids:
            validate_grid(event_tables, protocol["test_pairs"], tuple(grid), args.output)

    # 2. Pool the outer-test predictions; compare grids and smoothing choices.
    if args.stage in ["scores", "all"]:
        metrics = collect_validation(args.output, protocol["grids"])
        print(metrics.to_string(index=False), flush=True)

    # 3. Fitting-only xT maps for downstream choice/risk features.
    if args.stage in ["maps", "all"]:
        prepare_feature_maps(event_tables, protocol, args.output, args.fold)

    # 4. Independent all-ten-match operational refit. No package/site replacement.
    if args.stage in ["production", "all"]:
        refit_production(event_tables, protocol, args.output)


if __name__ == "__main__":
    main()
