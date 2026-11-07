"""Shared raw-data preparation. No outcome-based task cohort is removed here."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PRESS.utils.match import link_team_possessions
from pipeline.library.common import fresh_directory, write_json, finish_stage
from pipeline.library.preprocessing.tracking import load_raw_tracking, annotate_tracking, annotate_state_quality, annotate_training_cohort, segment_possessions, drive_observations
from pipeline.library.preprocessing.events import annotate_possession
from pipeline.library.preprocessing.possession_selection import matched_individual_mask


def tracking(raw_root, base_root, matches):
    """Read native coordinates and compute centred offline tracking derivatives."""
    for match in matches:
        folder = fresh_directory(Path(base_root) / 'matches' / match)
        data, players, metadata = load_raw_tracking(Path(raw_root) / match)
        data.to_parquet(folder / 'tracking_raw.parquet', index=False)
        players.to_parquet(folder / 'players.parquet', index=False)
        write_json(folder / 'metadata.json', metadata)
        events = pd.read_csv(Path(raw_root) / match / f'{match}_dynamic_events.csv', low_memory=False)
        events['match_id'] = str(match)
        events.to_parquet(folder / 'events.parquet', index=False)
        print('TRACKING', match, len(data), flush=True)


def possessions(base_root, matches):
    """Match intervals and annotate all canonical actions plus extra snapshots."""
    for match in matches:
        folder = Path(base_root) / 'matches' / match
        if (folder / 'actions.parquet').exists():
            raise FileExistsError('Possessions already generated: ' + str(folder))
        players = pd.read_parquet(folder / 'players.parquet')
        events = pd.read_parquet(folder / 'events.parquet')
        raw = events.loc[events.event_type.eq('player_possession')].copy()
        individual = annotate_possession(raw)
        mapping = players.set_index('player_id').player_possession.to_dict()
        individual['provider_player_id'] = individual.player_id
        individual['player_id'] = individual.player_id.map(mapping)
        individual['target_pass'] = individual.target_pass.map(mapping)
        if individual.player_id.isna().any():
            raise ValueError('Unmapped selected carrier')
        individual['turnover'] = np.where(individual.end_type.eq('clearance'),
            np.nan,
            individual.end_type.eq('possession_loss') | individual.end_type.eq('pass') & individual.pass_outcome.eq('unsuccessful'))
        individual['residual_loss'] = individual.loss.astype(bool)
        individual = annotate_training_cohort(individual, players, canonical_players=True)
        raw['individual_boundary_eligible'] = matched_individual_mask(raw)
        raw['individual_eligible'] = raw.event_id.isin(individual.possession_id)
        raw = annotate_training_cohort(raw, players)
        data = pd.read_parquet(folder / 'tracking_raw.parquet')
        data = annotate_state_quality(annotate_tracking(data, individual, raw, mapping))
        teams, individual = link_team_possessions(individual)
        teams['match_id'] = str(match)
        actions = segment_possessions(individual)
        actions['match_id'] = str(match)
        actions['state_eligible'] = actions.frame_start.map(data.set_index('frame').state_eligible).fillna(False)
        actions = annotate_training_cohort(actions, players, canonical_players=True)
        actions['start_type_raw'] = actions.start_type
        actions['previous_action_raw'] = actions.previous_action
        actions['time_since_start'] = (actions.frame_start - actions.possession_id.map(individual.set_index('possession_id').frame_start)) / 10.0
        for name, table in [('tracking', data),
            ('possessions', individual),
            ('teams', teams),
            ('actions', actions),
            ('possession_audit', raw)]:
            table.to_parquet(folder / (name + '.parquet'), index=False)
        print('POSSESSIONS', match, len(individual), len(actions), flush=True)


def bases(base_root, matches):
    """Export broad action/drive bases; final endpoint masks belong to each task."""
    tables = []
    for match in matches:
        folder = Path(base_root) / 'matches' / match
        actions = pd.read_parquet(folder / 'actions.parquet')
        drives = drive_observations(actions)
        drives.to_parquet(folder / 'drive_observations.parquet', index=False)
        tables.append(dict(match_id=match, actions=len(actions), canonical=int(actions.is_canonical.sum()), drives=len(drives)))
    pd.DataFrame(tables).to_csv(Path(base_root) / 'coverage.csv', index=False)
    write_json(Path(base_root) / 'manifest.json',
        dict(status='complete', matches=list(matches), policy='one_second_actions_with_exposure_tails_v2', task_filters_applied=False))
