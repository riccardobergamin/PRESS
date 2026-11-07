"""Native binary XGBoost: fit-only encoding, validation selection, untouched test."""
import numpy as np
import pandas as pd
import xgboost as xgb
from pipeline.library.common import fresh_directory, write_json, read_json, vocabulary, encode_categories, binary_scores, protocol, sha256
from pipeline.library.states import folds
from PRESS.model.action_selection import NUMERIC as AS_NUMERIC, FEATURE_CONTRACT as AS_CONTRACT
from PRESS.model.residual_risk import NUMERIC as RR_NUMERIC, FEATURE_CONTRACT as RR_CONTRACT


def numeric(task):
    return AS_NUMERIC if task == 'action_selection' else RR_NUMERIC


def load(paths, task, fold=None, matches=None):
    root = paths.task(task, 'features') / ('production' if fold is None else f'fold_{fold}')
    frame = pd.concat([pd.read_parquet(p) for p in sorted(root.glob('*.parquet')) if matches is None or p.stem in matches],
        ignore_index=True)
    frame['match_id'] = frame.match_id.astype(str)
    return frame


def train(paths, task, fold, threads):
    p = protocol(task)
    split = folds()[str(fold)]
    frame = load(paths, task, fold, split['fit_matches'] + split['validation_matches'])
    fit = frame.loc[frame.match_id.isin(split['fit_matches'])]
    val = frame.loc[frame.match_id.isin(split['validation_matches'])]
    vocab = vocabulary(fit)
    xf, names = encode_categories(fit, numeric(task), vocab)
    xv, _ = encode_categories(val, numeric(task), vocab)
    df = xgb.DMatrix(xf, label=fit.y, feature_names=names, nthread=threads)
    dv = xgb.DMatrix(xv, label=val.y, feature_names=names, nthread=threads)
    out = fresh_directory(paths.task(task, 'training') / f'fold_{fold}')
    params = p['fixed_parameters'] | dict(nthread=threads, seed=p['seed'] + fold)
    trials = []
    best = None
    for index, candidate in enumerate(p['grid']):
        history = {}
        model = xgb.train(params | candidate,
            df,
            num_boost_round=p['max_rounds'],
            evals=[(df, 'fit'), (dv, 'validation')],
            evals_result=history,
            early_stopping_rounds=p['early_stopping_rounds'],
            verbose_eval=False)
        selection = dict(candidate_index=index,
            **candidate,
            validation_ce=float(model.best_score),
            best_round=model.best_iteration + 1)
        trials.append(selection)
        if best is None or selection['validation_ce'] < best['validation_ce']:
            best = selection
            selected = model[:best['best_round']]
            selected.save_model(out / 'model.ubj')
            write_json(out / 'learning_curve.json', history)
        print(task.upper(), fold, index + 1, len(p['grid']), selection['validation_ce'], flush=True)
    write_json(out / 'selection.json',
        best | dict(vocabulary=vocab, feature_names=names, fit_matches=split['fit_matches']))
    pd.DataFrame(trials).to_csv(out / 'grid_validation.csv', index=False)
    model = xgb.Booster()
    model.load_model(out / 'model.ubj')
    model.set_param({'nthread': threads})
    # The prefix and configuration are frozen before loading the test pair.
    test = load(paths, task, fold, split['test_matches'])
    xt, _ = encode_categories(test, numeric(task), vocab)
    pred = test[['match_id', 'action_id', 'frame_start', 'y']].copy()
    pred['probability'] = model.predict(xgb.DMatrix(xt, feature_names=names, nthread=threads))
    pred['null_probability'] = float(fit.y.mean())
    pred['fold'] = fold
    pred.to_parquet(out / 'test_predictions.parquet', index=False)


def score(paths, task):
    pred = pd.concat([pd.read_parquet(paths.task(task, 'training') / f'fold_{f}' / 'test_predictions.parquet') for f in range(5)],
        ignore_index=True)
    if pred.duplicated(['match_id', 'action_id', 'frame_start']).any():
        raise ValueError('Duplicated held-out observations')
    scores, bins = binary_scores(pred.y, pred.probability, pred.null_probability)
    out = fresh_directory(paths.task(task, 'scores'))
    pred.to_parquet(out / 'pooled_predictions.parquet', index=False)
    bins.to_csv(out / 'calibration.csv', index=False)
    write_json(out / 'metrics.json', scores)


def full_configuration(paths, task):
    tables = []
    for f in range(5):
        frame = pd.read_csv(paths.task(task, 'training') / f'fold_{f}' / 'grid_validation.csv')
        frame['fold'] = f
        tables.append(frame)
    all_trials = pd.concat(tables, ignore_index=True)
    chosen = int(all_trials.groupby('candidate_index').validation_ce.mean().idxmin())
    p = protocol(task)
    config = p['grid'][chosen]
    rounds = int(np.median(all_trials.loc[all_trials.candidate_index.eq(chosen), 'best_round']))
    return (config, rounds)


def refit(paths, task, threads):
    frame = load(paths, task)
    vocab = vocabulary(frame)
    matrix, names = encode_categories(frame, numeric(task), vocab)
    config, rounds = full_configuration(paths, task)
    p = protocol(task)
    params = p['fixed_parameters'] | config | dict(nthread=threads, seed=p['seed'])
    model = xgb.train(params,
        xgb.DMatrix(matrix, label=frame.y, feature_names=names, nthread=threads),
        num_boost_round=rounds)
    out = fresh_directory(paths.task(task, 'production'))
    file = task + '_model.ubj'
    model.save_model(out / file)
    artifact = dict(schema_version=1,
        model_type='binary_action_choice' if task == 'action_selection' else 'conditional_individual_loss',
        feature_contract=AS_CONTRACT if task == 'action_selection' else RR_CONTRACT,
        model_id='fresh_' + task + '_refit',
        vocabulary=vocab,
        feature_names=names,
        booster_file=file,
        booster_sha256=sha256(out / file),
        calibration={'method': 'none'},
        training=dict(n_actions=len(frame), rounds=rounds, parameters=params, role='full_data_refit'))
    if task == 'action_selection':
        choice = read_json(paths.task('pass_selection', 'production') / 'flat' / 'pass_selection_model.json')
        artifact['xt_grid'] = choice['xt_grid']
    else:
        artifact['upstream'] = dict(action_booster_sha256=sha256(paths.task('action_selection', 'production') / 'action_selection_model.ubj'))
    write_json(out / (task + '_model.json'), artifact)
