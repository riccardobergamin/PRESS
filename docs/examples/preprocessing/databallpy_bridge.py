"""Optional preprocessing command, run in a DataBallPy environment.

This file deliberately imports no PRESS modules. Its parquet outputs cross
an environment boundary; the scientific package never imports DataBallPy.
Tested with DataBallPy 0.7.3, whose API is checked explicitly below.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path

import pandas as pd

if __package__:
    from .idsse_events import read_provider_events
else:
    from idsse_events import read_provider_events


def convert(data_dir, match_id, output_dir):
    from databallpy import get_game

    version = importlib.metadata.version("databallpy")
    if version != "0.7.3":
        raise RuntimeError(
            f"This bridge supports DataBallPy 0.7.3; found {version}. "
            "Use that version in a separate environment, or supply aligned tables directly."
        )
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    files = {}
    for kind, prefix in {
        "info": "DFL_02_01_matchinformation_",
        "events": "DFL_03_02_events_raw_",
        "tracking": "DFL_04_03_positions_raw_observed_",
    }.items():
        matches = list(data_dir.rglob(f"{prefix}*DFL-MAT-{match_id}.xml"))
        if len(matches) != 1:
            raise ValueError(f"Expected one {kind} XML for {match_id}; found {len(matches)}.")
        files[kind] = str(matches[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Reading original Sportec XML", flush=True)
    game = get_game(
        tracking_data_provider="sportec", tracking_metadata_loc=files["info"],
        tracking_data_loc=files["tracking"], event_data_provider="sportec",
        event_metadata_loc=files["info"], event_data_loc=files["events"],
    )
    # One semantic reader is shared across both environments. In particular,
    # a foul's actor is its victim, not the offender; handballs and duplicate
    # referee-decision records are not separate synchronization candidates.
    semantics = pd.DataFrame(read_provider_events(files['events'])).set_index('event_id')
    events = game.event_data.copy()
    identifiers = events.original_event_id.astype(str).str.replace(r'\.0$', '', regex=True)
    for column in ['player_id', 'team_id', 'event_type', 'duplicate_of', 'excluded_reason']:
        events[column] = identifiers.map(semantics[column]).to_numpy()
    events = events.loc[
        events.player_id.notna() & events.duplicate_of.isna() & events.excluded_reason.isna()
    ].copy()
    events.loc[events.event_type.eq('pass'), 'databallpy_event'] = 'pass'
    events.loc[events.event_type.eq('shot'), 'databallpy_event'] = 'shot'
    # DataBallPy's generic on-ball contact cost is named "tackle". Reuse that
    # existing cost for the original tackle/recovery/fouled classes and for
    # explicit clearances. Semantic event_type remains separate throughout.
    events.loc[events.event_type.isin([
        'tackle', 'recovery', 'interception', 'foul_suffered', 'clearance',
    ]), 'databallpy_event'] = 'tackle'
    game.event_data = events
    print("Refining event–tracking alignment", flush=True)
    game.synchronise_tracking_and_event_data(n_batches="smart", offset=1.0, verbose=False)
    print("Estimating the tracking control stream", flush=True)
    if "ball_velocity" not in game.tracking_data:
        game.tracking_data.add_velocity(column_ids="ball")
    game.tracking_data.add_individual_player_possession(
        pz_radius=1.5, bv_threshold=5.0, ba_threshold=10.0, min_frames_pz=0,
    )
    # tracking_frame is an index into the loaded tracking table, NOT the
    # provider's frame ID. Preserve the actual frame ID in the exported link.
    events = game.event_data.copy()
    events["frame"] = events.tracking_frame.map(game.tracking_data.frame)
    events.to_parquet(output_dir / "aligned_events.parquet", index=False)
    pd.DataFrame(game.tracking_data).to_parquet(output_dir / "tracking.parquet", index=False)
    for team in ["home", "away"]:
        getattr(game, team + "_players").to_parquet(output_dir / (team + "_players.parquet"), index=False)
    report = dict(
        match_id=match_id, databallpy_version=version,
        fps=float(game.tracking_data.frame_rate),
        pitch_length=float(game.pitch_dimensions[0]), pitch_width=float(game.pitch_dimensions[1]),
        orientation="static_home_away: home attacks positive x in both periods",
        synchronization=dict(n_batches="smart", offset_seconds=1.0),
        control_estimation=dict(pz_radius_metres=1.5, ball_velocity_threshold=5.0,
                                ball_angle_threshold_degrees=10.0, min_frames_pz=0),
        n_tracking_frames=len(game.tracking_data), n_actor_events=len(events),
        n_aligned_events=int(events.frame.notna().sum()),
        aligned_semantic_events=events.loc[events.frame.notna(), 'event_type'].value_counts().to_dict(),
        n_duplicate_foul_decisions=int(semantics.duplicate_of.notna().sum()),
        n_excluded_handballs=int(semantics.excluded_reason.eq('handball').sum()),
    )
    (output_dir / "alignment.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("match_id")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    convert(args.data_dir, args.match_id, args.output_dir)
