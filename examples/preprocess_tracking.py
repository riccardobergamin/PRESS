"""Prepare observed tracking and raw possession events without importing PRESS.

Uses the companion analysis checkout's tracking loader, so smoothing and native
coordinate conventions are identical to the research pipeline. Scientific
Python dependencies are NumPy, pandas, SciPy and PyArrow; no fitted model runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-dir", type=Path, required=True)
    parser.add_argument("--analysis-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    match_dir = args.match_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error("Use a new, empty output directory")
    if output_dir == match_dir or match_dir in output_dir.parents:
        parser.error("Keep derived files separate from the raw match directory")

    sys.path.insert(0, str(args.analysis_source.resolve() / "scripts"))
    from utils.base_pipeline import load_raw_tracking

    tracking, players, metadata = load_raw_tracking(match_dir)
    match_id = match_dir.name
    events = pd.read_csv(match_dir / f"{match_id}_dynamic_events.csv", low_memory=False)
    raw_possessions = events.loc[events.event_type.eq("player_possession")].copy()
    raw_possessions["match_id"] = match_id
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "tracking": output_dir / f"{match_id}_tracking_base.parquet",
        "player_map": output_dir / f"{match_id}_player_map.parquet",
        "raw_possessions": output_dir / f"{match_id}_raw_possessions.parquet",
    }
    for key, table in [("tracking", tracking), ("player_map", players),
                       ("raw_possessions", raw_possessions)]:
        table.to_parquet(paths[key], index=False)
    summary = {
        "match_id": match_id, "fps": 10,
        "pitch_length": metadata["pitch_length"], "pitch_width": metadata["pitch_width"],
        "tracking_frames": len(tracking), "player_identities": len(players),
        "raw_individual_possessions": len(raw_possessions),
        "outputs": {key: path.name for key, path in paths.items()},
        "stage": "tracking kinematics and raw event extraction; before canonical segmentation",
    }
    (output_dir / "preprocessing_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
