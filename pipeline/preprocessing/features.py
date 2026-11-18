"""Raw SkillCorner files -> tracking -> possessions -> canonical task tables.

Read the three preparation functions in order. Numerical tracking derivatives
and segmentation are in tracking.py; endpoint linkage is in passes.py.
Outcome filters remain in each task's filters.py, outside shared preparation.
"""

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from pipeline.io import new_folder, read_table, write_json
from pipeline.preprocessing.tracking import (
    load_raw_tracking, annotate_tracking, annotate_state_quality,
    annotate_training_cohort, segment_possessions, drive_observations,
)
from pipeline.preprocessing.events import annotate_possession
from pipeline.preprocessing.possession_selection import matched_individual_mask
from pipeline.preprocessing.xt_events import prepare_skillcorner_events
from pipeline.ball_drive.filters import select as select_drives, parent_mask
from pipeline.pass_probability.filters import verified_passes


def prepare_tracking(raw_root, output, match_ids):
    """Read native XY(Z), estimate offline v/a, and retain provider events.

    raw_root/<match>/<match>_match.json, *_tracking_extrapolated.jsonl,
    *_dynamic_events.csv. Native pitch dimensions and identities are retained.
    Player derivatives use seven-frame windows; ball derivatives use five.
    Finite contiguous runs are differentiated separately across gaps/periods.
    """
    for match in match_ids:
        folder = new_folder(Path(output) / "matches" / match)
        tracking, players, metadata = load_raw_tracking(Path(raw_root) / match)
        tracking.to_parquet(folder / "tracking_raw.parquet", index=False)
        players.to_parquet(folder / "players.parquet", index=False)
        write_json(folder / "metadata.json", metadata)
        events = pd.read_csv(Path(raw_root) / match / f"{match}_dynamic_events.csv", low_memory=False)
        events["match_id"] = match
        events.to_parquet(folder / "events.parquet", index=False)
        print("TRACKING", match, len(tracking), flush=True)


def prepare_possessions(output, match_ids):
    """Match individual controls, flag usable states, segment one-second actions.

    No task is filtered here. Keep snapshots distinct from canonical actions.
    A 2.4 s pass possession has carries at 0/1 s and a pass at 2.4 s;
    a 2.4 s direct loss also has the failed carry at 2 s. Short failed carries
    remain actions; successful short tails remain exposure-only snapshots.
    """
    from PRESS.utils.match import link_team_possessions
    for match in match_ids:
        folder = Path(output) / "matches" / match
        if (folder / "actions.parquet").exists():
            raise FileExistsError(folder / "actions.parquet")
        events, players = read_table(folder / "events.parquet"), read_table(folder / "players.parquet")
        raw_parents = events.loc[events.event_type.eq("player_possession")].copy()
        parents = annotate_possession(raw_parents)
        aliases = players.set_index("player_id").player_possession.to_dict()
        parents["provider_player_id"] = parents.player_id
        parents["player_id"] = parents.player_id.map(aliases)
        parents["target_pass"] = parents.target_pass.map(aliases)
        if parents.player_id.isna().any():
            raise ValueError("An event carrier has no tracking identity.")
        parents["turnover"] = np.where(parents.end_type.eq("clearance"), np.nan,
            parents.end_type.eq("possession_loss") |
            (parents.end_type.eq("pass") & parents.pass_outcome.eq("unsuccessful")))
        parents["residual_loss"] = parents.loss.astype(bool)
        parents = annotate_training_cohort(parents, players, canonical_players=True)
        raw_parents["individual_boundary_eligible"] = matched_individual_mask(raw_parents)
        raw_parents["individual_eligible"] = raw_parents.event_id.isin(parents.possession_id)
        raw_parents = annotate_training_cohort(raw_parents, players)
        tracking = read_table(folder / "tracking_raw.parquet")
        tracking = annotate_state_quality(annotate_tracking(tracking, parents, raw_parents, aliases))
        teams, parents = link_team_possessions(parents)
        teams["match_id"] = match
        actions = segment_possessions(parents)
        actions["state_eligible"] = actions.frame_start.map(tracking.set_index("frame").state_eligible).fillna(False)
        actions = annotate_training_cohort(actions, players, canonical_players=True)
        actions["start_type_raw"] = actions.start_type
        actions["previous_action_raw"] = actions.previous_action
        actions["time_since_start"] = (actions.frame_start - actions.possession_id.map(
            parents.set_index("possession_id").frame_start)) / 10.
        for name, table in [(f"{match}_tracking", tracking), ("possessions", parents),
                            ("teams", teams), ("actions", actions), ("possession_audit", raw_parents)]:
            table.to_parquet(folder / f"{name}.parquet", index=False)
        print("POSSESSIONS", match, len(parents), len(actions), flush=True)


def frame_context(events, metadata, frame, team_id):
    """Clock and score effective by this frame; future goals are not predictors."""
    parents = events.loc[events.event_type.eq("player_possession")].sort_values(["frame_start", "frame_end"])
    before = parents.loc[parents.frame_start.le(frame)]
    event = before.iloc[-1] if len(before) else parents.iloc[0]
    same_period = parents.loc[parents.period.eq(event.period)]
    def seconds(value):
        return sum(float(p) * 60**i for i, p in enumerate(reversed(str(value).split(":"))))
    offset = (same_period.time_end.map(seconds) - same_period.frame_end / 10).median()
    clock = (frame / 10 + offset) / 60
    home_id = metadata["home_team"]["id"]
    home = np.where(parents.team_id.eq(home_id), parents.team_score, parents.opponent_team_score)
    away = np.where(parents.team_id.eq(home_id), parents.opponent_team_score, parents.team_score)
    effective_frame = np.where(parents.game_interruption_before.isin(["goal_for", "goal_against"]),
                               parents.frame_start, parents.frame_end + 1)
    known = effective_frame <= frame
    difference = float(np.max(home[known], initial=0) - np.max(away[known], initial=0))
    return clock, difference if team_id == home_id else -difference


def order_drive_observations(table):
    """Preserve the benchmark's row order after verifying regenerated keys.

    Seeded sampling selects row indices. The distributed manifest contains
    only observed identifiers, so a raw-data reconstruction can retain the
    same sampling order without loading saved features or predictions.
    """
    path = Path(__file__).resolve().parents[1] / "inputs/ball_drive/observation_order.json"
    specification = json.loads(path.read_text())
    keys = specification["keys"]
    expected = pd.DataFrame(specification["rows"], columns=keys)
    expected = expected.loc[expected.match_id.isin(table.match_id.unique())]
    actual_keys = pd.MultiIndex.from_frame(table[keys].astype(str))
    expected_keys = pd.MultiIndex.from_frame(expected[keys].astype(str))
    if actual_keys.has_duplicates or expected_keys.has_duplicates or set(actual_keys) != set(expected_keys):
        raise ValueError("Regenerated carry keys differ from the observation-order manifest.")
    return table.iloc[actual_keys.get_indexer(expected_keys)].reset_index(drop=True)


def prepare_task_tables(output, match_ids, templates=None):
    """Apply declared task masks and export separate observation/label tables.

    Physical covariates, xT smoothing and learned features are calculated later
    by each task. This function determines observations, never model weights.
    h/xLoss labels cover every verified canonical state; report eligibility is
    an additional parent-level terminal-action mask, not a training exclusion.
    """
    collected = {k: [] for k in ["ball_drive", "pass_probability", "pass_selection",
                                 "action_selection", "residual_risk", "composed_risk"]}
    match_players = {}
    xt_tables = {k: [] for k in ["carries", "passes", "shots", "losses"]}
    for match in match_ids:
        folder = Path(output) / "matches" / match
        tracking = read_table(folder / f"{match}_tracking.parquet")
        events, players = read_table(folder / "events.parquet"), read_table(folder / "players.parquet")
        parents, actions = read_table(folder / "possessions.parquet"), read_table(folder / "actions.parquet")
        metadata = json.loads((folder / "metadata.json").read_text())
        match_players[match] = players
        parents_by_id = parents.set_index("possession_id")
        drives = select_drives(drive_observations(actions), events)
        collected["ball_drive"].append(drives)
        # PS retains <0.5 s; physical trajectory MLE/calibration/test use [0.5,10].
        passed = verified_passes(events, tracking, players)
        aliases = players.set_index("player_id").player_possession
        passed["player"] = passed.player_id.map(aliases)
        passed["pitch_length"], passed["pitch_width"] = metadata["pitch_length"], metadata["pitch_width"]
        passed["start_type_raw"] = passed.start_type
        # previous_action refers to the canonical action BEFORE release,
        # never the pass result. A short initial possession has no full carry.
        start_frames = events.set_index("event_id").frame_start
        elapsed = passed.frame_end-passed.event_id.map(start_frames)
        passed["previous_action_raw"] = np.where(elapsed.ge(10), "ball_drive", passed.start_type)
        # A verified release/receipt can be retained even when the beginning
        # of the passer's possession is unmatched. Its initial context and
        # elapsed-action history are then unknown. Keep the existing missing
        # encoding (__missing__ in the choice model), not a new category.
        matched_starts = events.set_index("event_id").is_player_possession_start_matched
        unknown_start = ~passed.event_id.map(matched_starts).fillna(False).astype(bool)
        passed.loc[unknown_start, ["start_type_raw", "previous_action_raw"]] = None
        teams_at_release = passed.player_id.map(players.set_index("player_id").team_id)
        context = [frame_context(events,metadata,int(frame),team) for frame,team in zip(passed.frame_end,teams_at_release)]
        passed[["match_clock_min","score_difference_prefix"]] = context
        collected["pass_probability"].append(passed.loc[passed.pass_duration.between(.5, 10)].copy())
        collected["pass_selection"].append(passed.loc[passed.pass_duration.le(10)].copy())
        choices = actions.loc[actions.is_canonical & actions.state_eligible & actions.training_cohort_eligible].copy()
        choices["y"] = choices.action.eq("pass").astype(int)
        choices["attacking_side"] = choices.possession_id.map(parents_by_id.attacking_side)
        choices["pitch_length"], choices["pitch_width"] = metadata["pitch_length"], metadata["pitch_width"]
        choices["completed_canonical_drives"] = np.floor(choices.time_since_start).astype(int)
        # The canonical parent table stores home/away aliases, not provider
        # team IDs. Resolve the latter through the observed match metadata.
        team_ids = {side: metadata[side + "_team"]["id"] for side in ["home", "away"]}
        teams = choices.player_id.str.split("_").str[0].map(team_ids)
        context = [frame_context(events, metadata, int(frame), team) for frame, team in zip(choices.frame_start, teams)]
        choices[["match_clock_min", "score_difference_prefix"]] = context
        collected["action_selection"].append(choices)
        valid = parent_mask(choices, events) & (
            choices.end_type.ne("pass") | choices.possession_id.isin(passed.event_id))
        verified = choices.loc[valid].copy()
        verified["terminal_loss"] = verified.possession_id.map(parents_by_id.residual_loss).astype(int)
        verified["h_loss"] = 1 - verified.outcome.astype(int)
        verified["parent_end_type"] = verified.end_type
        verified["terminal_pass_outcome"] = verified.possession_id.map(parents_by_id.pass_outcome)
        verified["band_eligible"] = verified.end_type.isin(["pass", "possession_loss"])
        # Outcome metadata belongs to the label/report table only. It is not
        # included in the predictor list of action selection or residual risk.
        for name, source in [("parent_start_frame","frame_start"),("parent_end_frame","frame_end"),
                             ("parent_start_type","start_type"),("provider_player_id","provider_player_id")]:
            verified[name] = verified.possession_id.map(parents_by_id[source])
        roster = {p["id"]:p for p in metadata["players"]}
        team_names = {metadata[s+"_team"]["id"]:metadata[s+"_team"]["short_name"] for s in ["home","away"]}
        verified["player_name"] = [" ".join(filter(None,[roster[p].get("first_name"),roster[p].get("last_name")]))
                                    or roster[p].get("short_name","") for p in verified.provider_player_id]
        verified["team_id"] = [roster[p]["team_id"] for p in verified.provider_player_id]
        verified["team_name"] = verified.team_id.map(team_names)
        raw_parents = events.loc[events.event_type.eq("player_possession")].sort_values(["frame_start","frame_end"]).copy()
        raw_parents["next_control_frame"] = raw_parents.frame_start.shift(-1)
        verified["next_control_frame"] = verified.possession_id.map(raw_parents.set_index("event_id").next_control_frame)
        collected["composed_risk"].append(verified)
        residual = verified.loc[verified.action.eq("ball_drive") & verified.outcome.eq(1) & verified.duration_frames.eq(10)].copy()
        residual["residual_loss"] = residual.terminal_loss
        collected["residual_risk"].append(residual)
        for kind, table in zip(xt_tables, prepare_skillcorner_events(events)):
            table["match_id"] = match
            xt_tables[kind].append(table)
        print("DATASETS", match, len(choices), len(verified), flush=True)

    root = new_folder(Path(output) / "task_inputs")
    references = {
        "ball_drive": ("ball_drive/rows.parquet", ["match_id","frame_start","frame_end"]),
        "pass_probability": ("pass_probability/rows.parquet", ["match_id","pass_id","frame_end"]),
        "pass_selection": ("pass_selection/rows.parquet", ["match_id","pass_id","frame_end"]),
        "action_selection": ("binary_classification/action_rows.parquet", ["match_id","action_id","frame_start"]),
        "residual_risk": ("binary_classification/residual_labels.parquet", ["match_id","action_id","frame_start"]),
        "composed_risk": ("composed_risk/labels.parquet", ["match_id","action_id","frame_start"]),
    }
    for task, tables in collected.items():
        destination = root / task
        destination.mkdir()
        table = pd.concat(tables, ignore_index=True)
        if task == "ball_drive":
            table = order_drive_observations(table)
        if templates is not None:
            # Preserve observation ordering (hence seeded sampling) only AFTER
            # proving exact cohort equality. No accepted row is inserted,
            # dropped or relabelled by this ordering operation.
            filename,keys = references[task]
            reference = read_table(Path(templates)/filename)
            reference = reference.loc[reference.match_id.isin(match_ids)]
            actual_keys = pd.MultiIndex.from_frame(table[keys].astype(str))
            expected_keys = pd.MultiIndex.from_frame(reference[keys].astype(str))
            if actual_keys.has_duplicates or expected_keys.has_duplicates or set(actual_keys)!=set(expected_keys):
                raise ValueError(f"Regenerated {task} cohort differs from the accepted keys; no automatic promotion.")
            table=table.iloc[actual_keys.get_indexer(expected_keys)].reset_index(drop=True)
            reference=reference.reset_index(drop=True)
            for label in ["outcome","y","residual_loss","terminal_loss","h_loss"]:
                if label in reference and label in table and not np.array_equal(table[label],reference[label]):
                    raise ValueError(f"Regenerated {task} labels differ: {label}")
        table["row_id"] = np.arange(len(table))
        table.to_parquet(destination / "rows.parquet", index=False)
        if task == "pass_probability":
            # The PP pipeline reconstructs player states from tracking. Its
            # only archived array input is the observed pass destination.
            np.savez_compressed(destination / "arrays.npz",r_target=table[["x_target","y_target"]].to_numpy(np.float32))
        if task == "pass_selection":
            for match, players in match_players.items():
                folder=destination/"matches"/match
                folder.mkdir(parents=True)
                players.to_parquet(folder/"players.parquet",index=False)
        if task == "composed_risk":
            table.to_parquet(destination/"labels.parquet",index=False)
    binary = root/"binary_classification"
    binary.mkdir()
    read_table(root/"action_selection/rows.parquet").to_parquet(binary/"action_rows.parquet",index=False)
    label_columns=["match_id","action_id","frame_start","possession_id","time_since_start",
                   "residual_loss","parent_end_type","terminal_pass_outcome"]
    read_table(root/"residual_risk/rows.parquet")[label_columns].to_parquet(binary/"residual_labels.parquet",index=False)
    events_folder = root / "expected_threat" / "events"
    events_folder.mkdir(parents=True)
    for kind, tables in xt_tables.items():
        pd.concat(tables, ignore_index=True).to_parquet(events_folder / f"{kind}.parquet", index=False)
    if templates is not None:
        # This explicit option supplies reviewed protocols and accepted
        # upstream model artifacts, without changing any generated row/label.
        # It does not claim those artifacts were refitted by preprocessing.
        templates=Path(templates)
        for task in ["pass_selection","binary_classification","expected_threat","composed_risk"]:
            shutil.copy2(templates/task/"protocol.json",root/task/"protocol.json")
        for task in ["pass_selection","binary_classification"]:
            shutil.copytree(templates/task/"upstream",root/task/"upstream")
        shutil.copytree(templates/"reports",root/"reports")
        shutil.copytree(templates/"ball_drive/reference_parameters",root/"ball_drive/reference_parameters")
        shutil.copy2(templates/"folds.json",root/"folds.json")
    write_json(root/"provenance.json",dict(matches=match_ids,rows_regenerated_from_raw=True,
        upstream_artifacts="explicit accepted templates" if templates else "not copied",
        observation_order="accepted key order, after exact key/label verification" if templates else "raw preparation order",
        templates=str(templates) if templates else None))
    return root
