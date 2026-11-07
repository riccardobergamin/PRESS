"""Canonical pass/carry decisions with matching fold-specific upstream features."""
import numpy as np
import pandas as pd
from PRESS.model import PassSelectionModel
from PRESS.model.action_features import aggregate
from PRESS.model.bcos_fine import compute_bcos_state
from pipeline.library.common import fresh_directory, read_json
from pipeline.library.states import load_match, initial_state, context, folds
from pipeline.library.option_features import upstream, options
from pipeline.library import binary_training
from .filters import select


def features(paths, matches, fold=None):
    drive, passing, xt = upstream(paths, fold)
    model_path = paths.task('pass_selection',
        'production') / 'flat' / 'pass_selection_model.json' if fold is None else paths.task('pass_selection',
        'training') / f'fold_{fold}' / 'flat' / 'pass_selection_model.json'
    choice = PassSelectionModel.from_json(model_path)
    expected = sorted(p.name for p in (paths.base / 'matches').iterdir() if p.is_dir()) if fold is None else folds()[str(fold)]['fit_matches']
    if set(choice.to_dict()['training']['match_ids']) != set(expected):
        raise ValueError('Pass-selection model does not match the requested fit split')
    root = paths.task('action_selection', 'features') / ('production' if fold is None else f'fold_{fold}')
    for match in matches:
        root.mkdir(parents=True, exist_ok=True)
        target = root / (match + '.parquet')
        if target.exists():
            raise FileExistsError(target)
        tracking, events, players, metadata = load_match(paths.base, match)
        actions = select(pd.read_parquet(paths.match(match) / 'actions.parquet'))
        raw = events.loc[events.event_type.eq('player_possession')].set_index('event_id')
        records = []
        for row in actions.itertuples(index=False):
            state = initial_state(tracking, row.frame_start, row.player_id, row.attacking_side)
            clock, score = context(events, metadata, row.frame_start, raw.loc[row.possession_id, 'team_id'])
            option = options(state,
                drive,
                passing,
                xt,
                choice.vocabulary,
                start_type=row.start_type,
                previous_action=row.previous_action,
                clock=clock,
                score=score)
            q = choice.predict_features(option['matrix'])
            physical = drive.to_dict()['physics']
            # BCOS keeps the fitted engagement dispersion and defender routine.
            # Carrier movement uses the documented 6.6 m/s cap and 7 m/s² scales.
            cfg = [np.exp(drive.theta[3]), 6.6, physical['v_max'], 7.0,
                   physical['a_linear'], 7.0, physical['a_turn']]
            bcos = compute_bcos_state(state['p_pos'],
                state['p_vel'],
                pitch_length=state['pitch_length'],
                pitch_width=state['pitch_width'],
                configuration=cfg)
            count = int(row.time_since_start // 1)
            record = aggregate(bc=state['x_bc'],
                velocity=state['v_bc'],
                players=state['p_pos'],
                targets=option['targets'],
                direction=1 if row.attacking_side == 'left_to_right' else -1,
                own_lines=option['lines'][1],
                opp_lines=option['lines'][0],
                choice=q,
                completion=option['completion'],
                drive_loss=option['loss'],
                bcos=bcos['BCOS area'],
                xt_grid=xt,
                pitch_length=state['pitch_length'],
                pitch_width=state['pitch_width'],
                match_clock_min=clock,
                score_difference_prefix=score,
                completed_canonical_drives=count)
            record.update(match_id=match,
                action_id=row.action_id,
                frame_start=row.frame_start,
                possession_id=row.possession_id,
                y=int(row.action == 'pass'),
                start_type_raw=row.start_type,
                previous_action_raw=row.previous_action,
                time_since_start=row.time_since_start)
            records.append(record)
        pd.DataFrame(records).to_parquet(target, index=False)
        print('ACTION_FEATURES', fold, match, len(records), flush=True)


def train(paths, fold, threads):
    binary_training.train(paths, 'action_selection', fold, threads)


def score(paths):
    binary_training.score(paths, 'action_selection')


def refit(paths, threads):
    binary_training.refit(paths, 'action_selection', threads)
