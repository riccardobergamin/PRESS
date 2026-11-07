"""Verified continuation outcomes with cross-fitted P(carry) as a predictor."""
import numpy as np
import pandas as pd
import xgboost as xgb
from pipeline.library.common import fresh_directory, read_json, write_json, protocol, vocabulary, encode_categories
from pipeline.library.states import load_match, folds
from pipeline.library import binary_training as binary
from .filters import select
KEYS = ['match_id', 'action_id', 'frame_start']


def predict(model, frame, vocab, threads):
    matrix, names = encode_categories(frame, binary.numeric('action_selection'), vocab)
    return model.predict(xgb.DMatrix(matrix, feature_names=names, nthread=threads))


def features(paths, matches, fold=None, threads=1):
    source = binary.load(paths, 'action_selection', fold)
    scores = np.full(len(source), np.nan)
    p = protocol('action_selection')
    if fold is None:
        config, rounds = binary.full_configuration(paths, 'action_selection')
        split_pairs = [folds()[str(f)]['test_matches'] for f in range(5)]
        groups = [([m for m in matches if m not in held], held) for held in split_pairs]
    else:
        split = folds()[str(fold)]
        selection = read_json(paths.task('action_selection', 'training') / f'fold_{fold}' / 'selection.json')
        config = {k: selection[k] for k in ['max_depth', 'min_child_weight', 'reg_lambda', 'eta']}
        rounds = selection['best_round']
        groups = []
        for i in range(3):
            held = split['fit_matches'][2 * i:2 * i + 2]
            groups.append(([m for m in split['fit_matches'] if m not in held], held))
    # Fit scores are out-of-sample within the outer fit set; validation/test
    # scores come from the matching six-fit action model below.
    for i, (training, held) in enumerate(groups):
        train = source.loc[source.match_id.isin(training)]
        query = source.loc[source.match_id.isin(held)]
        vocab = vocabulary(train)
        matrix, names = encode_categories(train, binary.numeric('action_selection'), vocab)
        seed = p['seed'] + i if fold is None else p['seed'] + 1000 + 100 * fold + i
        model = xgb.train(p['fixed_parameters'] | config | dict(nthread=threads, seed=seed),
            xgb.DMatrix(matrix, label=train.y, feature_names=names, nthread=threads),
            num_boost_round=rounds)
        scores[query.index] = 1 - predict(model, query, vocab, threads)
        out = paths.task('residual_risk',
            'features') / 'stack' / ('production' if fold is None else f'fold_{fold}') / f'pair_{i}'
        out.mkdir(parents=True, exist_ok=False)
        model.save_model(out / 'action_model.ubj')
        write_json(out / 'provenance.json',
            dict(fit_matches=training, held_matches=held, rounds=rounds, fit_only_vocabulary=vocab))
    if fold is not None:
        out = paths.task('action_selection', 'training') / f'fold_{fold}'
        model = xgb.Booster()
        model.load_model(out / 'model.ubj')
        model.set_param({'nthread': threads})
        query = source.loc[source.match_id.isin(split['validation_matches'] + split['test_matches'])]
        scores[query.index] = 1 - predict(model, query, selection['vocabulary'], threads)
    if not np.isfinite(scores).all():
        raise ValueError('Missing cross-fitted action scores')
    source['action_carry_probability'] = scores
    destination = paths.task('residual_risk', 'features') / ('production' if fold is None else f'fold_{fold}')
    destination.mkdir(parents=True, exist_ok=False)
    for match in matches:
        tracking, events, players, metadata = load_match(paths.base, match)
        actions = pd.read_parquet(paths.match(match) / 'actions.parquet')
        labels = select(actions, events, tracking, players)[KEYS + ['possession_id']]
        terminal = pd.read_parquet(paths.match(match) / 'possessions.parquet').set_index('possession_id').residual_loss
        labels['residual_y'] = labels.possession_id.map(terminal).astype(int)
        frame = source.loc[source.match_id.eq(match)].merge(labels[KEYS + ['residual_y']],
            on=KEYS,
            validate='one_to_one')
        frame['y'] = frame.pop('residual_y')
        frame.to_parquet(destination / (match + '.parquet'), index=False)


def train(paths, fold, threads):
    binary.train(paths, 'residual_risk', fold, threads)


def score(paths):
    binary.score(paths, 'residual_risk')


def refit(paths, threads):
    binary.refit(paths, 'residual_risk', threads)
