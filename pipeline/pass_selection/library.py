"""Flat and cross-nested destination choice with one shared fresh feature cache."""
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from PRESS.model.pass_selection import choice_feature_names, AHEAD_FEATURE_CONTRACT
from pipeline.library.common import fresh_directory, read_json, write_json, sha256, vocabulary, protocol
from pipeline.library.states import load_match, initial_state, context, folds
from pipeline.library.option_features import upstream, options
from pipeline.library.pass_selection.engine import Engine
from pipeline.library.pass_selection.training import Trajectory
from pipeline.library.pass_selection.temperature import fit_temperature, evaluate
from pipeline.library.pass_selection.choice_likelihood import fit_rho
from .filters import select
KEYS = ['match_id', 'pass_id', 'frame_end']


def feature_root(paths, fold):
    return paths.task('pass_selection', 'features') / ('production' if fold is None else f'fold_{fold}')


def features(paths, matches, fold=None):
    drive, passing, xt = upstream(paths, fold)
    prepared = {}
    all_matches = sorted((d.name for d in (paths.base / 'matches').iterdir() if d.is_dir()))
    for match in all_matches:
        tracking, events, players, meta = load_match(paths.base, match)
        rows = select(events, tracking, players)
        rows['start_type_raw'] = rows.start_type
        rows['previous_action_raw'] = np.where(rows.frame_end - rows.frame_start >= 10,
            'ball_drive',
            rows.start_type)
        prepared[match] = (tracking, events, players, meta, rows)
    fit_matches = all_matches if fold is None else folds()[str(fold)]['fit_matches']
    vocab = vocabulary(pd.concat([prepared[m][-1] for m in fit_matches], ignore_index=True))
    root = feature_root(paths, fold)
    root.mkdir(parents=True, exist_ok=True)
    spec = dict(vocabulary=vocab,
        feature_names=choice_feature_names(vocab),
        xt_grid=xt.tolist(),
        fit_matches=fit_matches)
    if (root / 'feature_spec.json').exists() and read_json(root / 'feature_spec.json') != spec:
        raise ValueError('Feature contract differs within this run')
    write_json(root / 'feature_spec.json', spec)
    for match, (tracking, events, players, meta, rows) in prepared.items():
        if match not in matches:
            continue
        out = fresh_directory(root / match)
        matrices = []
        memberships = []
        chosen = []
        mapping = players.set_index('player_id').player_possession
        for row in rows.itertuples(index=False):
            state = initial_state(tracking, row.frame_end, mapping[row.player_id], row.attacking_side)
            clock, score = context(events, meta, row.frame_end, row.team_id)
            option = options(state,
                drive,
                passing,
                xt,
                vocab,
                start_type=row.start_type_raw,
                previous_action=row.previous_action_raw,
                clock=clock,
                score=score)
            matrices.append(option['matrix'])
            memberships.append(option['log_alpha'])
            chosen.append(int(np.argmin(np.linalg.norm(option['targets'] - np.array([row.x_target, row.y_target]), axis=1))))
        rows['chosen'] = chosen
        rows['n_candidates'] = [len(m) for m in matrices]
        rows.to_parquet(out / 'rows.parquet', index=False)
        np.save(out / 'matrix.npy', np.asarray(matrices, np.float32))
        if fold is not None:
            folder = paths.task('pass_selection', 'features') / 'memberships' / f'fold_{fold}' / match
            folder.mkdir(parents=True, exist_ok=False)
            rows[KEYS].to_parquet(folder / 'keys.parquet', index=False)
            f = folder / f'log_alpha_fold_{fold}.npy'
            np.save(f, np.asarray(memberships, np.float32))
            write_json(folder / 'completion.json',
                dict(status='complete', smoke=False, protocol_sha256=sha256(Path(__file__).with_name('protocol.json')), files={f.name: sha256(f)}))
        else:
            np.save(out / 'log_alpha.npy', np.asarray(memberships, np.float32))
        print('CHOICE_FEATURES', fold, match, len(rows), flush=True)


def load(paths, fold, matches):
    arrays = []
    rows = []
    for match in matches:
        folder = feature_root(paths, fold) / match
        data = pd.read_parquet(folder / 'rows.parquet')
        data['match_id'] = data.match_id.astype(str)
        rows.append(data)
        arrays.append(np.load(folder / 'matrix.npy', mmap_mode='r').reshape(-1, len(read_json(feature_root(paths, fold) / 'feature_spec.json')['feature_names'])))
    return (np.concatenate(arrays), pd.concat(rows, ignore_index=True))


def make_engine(paths, fold, rows, family):
    return Engine(paths.task('pass_selection', 'features') / 'memberships',
        rows[KEYS],
        rows.n_candidates.to_numpy(),
        rows.chosen.to_numpy(),
        fold,
        family,
        sha256(Path(__file__).with_name('protocol.json')))


def export(paths, out, model, family, rho, calibration, spec):
    file = 'pass_selection_model.ubj'
    model.save_model(out / file)
    write_json(out / 'pass_selection_model.json',
        dict(schema_version=1, model_type='pass_destination_choice', model_id='fresh_pass_selection_' + family, training=dict(match_ids=spec['fit_matches']), family=family, rho=float(rho), vocabulary=spec['vocabulary'], feature_contract=AHEAD_FEATURE_CONTRACT, feature_names=spec['feature_names'], xt_grid=spec['xt_grid'], booster_file=file, booster_sha256=sha256(out / file), calibration=calibration))


def train(paths, fold, threads):
    p = protocol('pass_selection')
    split = folds()[str(fold)]
    spec = read_json(feature_root(paths, fold) / 'feature_spec.json')
    names = spec['feature_names']
    xf, rf = load(paths, fold, split['fit_matches'])
    xv, rv = load(paths, fold, split['validation_matches'])
    for family in ['flat', 'cnl']:
        out = fresh_directory(paths.task('pass_selection', 'training') / f'fold_{fold}' / family)
        ef = make_engine(paths, fold, rf, family)
        ev = make_engine(paths, fold, rv, family)
        labels = np.zeros(len(xf), np.float32)
        starts = np.r_[0, np.cumsum(rf.n_candidates.to_numpy())[:-1]]
        labels[starts + rf.chosen.to_numpy()] = 1
        df = xgb.DMatrix(xf, label=labels, feature_names=names, nthread=threads)
        dv = xgb.DMatrix(xv, feature_names=names, nthread=threads)
        candidate = {k: v for k, v in p['selected_by_family'][family].items() if k != 'name'}
        params = p['fixed_parameters'] | candidate | dict(nthread=threads,
            seed=p['seed'] + fold,
            seed_per_iteration=True,
            disable_default_eval_metric=1)
        trajectory = Trajectory(out / 'trajectory', params, df, dv, ef, ev, p).advance(p['max_rounds'])
        model, rho = trajectory.selected()
        logp = ev.log_probabilities(model.predict(dv, output_margin=True), rho)
        cal = fit_temperature(logp, rv.n_candidates, rv.chosen)
        selection = dict(rounds=trajectory.state['best_round'],
            rho=rho,
            parameters=params,
            validation_ce=float(evaluate(logp, rv.n_candidates, rv.chosen).mean()))
        export(paths, out, model, family, rho, cal, spec)
        write_json(out / 'selection.json', selection)
        xt, rt = load(paths, fold, split['test_matches'])
        et = make_engine(paths, fold, rt, family)
        logp = et.log_probabilities(model.predict(xgb.DMatrix(xt, feature_names=names, nthread=threads), output_margin=True),
            rho)
        pred = rt[KEYS + ['chosen', 'n_candidates']].copy()
        pred['fold'] = fold
        pred['nll_raw'] = evaluate(logp, rt.n_candidates, rt.chosen)
        pred['nll'] = evaluate(logp, rt.n_candidates, rt.chosen, cal['inverse_temperature'])
        pred['nll_null'] = np.log(rt.n_candidates)
        pred.to_parquet(out / 'test_predictions.parquet', index=False)
        np.save(out / 'test_logp.npy', logp)


def score(paths):
    out = fresh_directory(paths.task('pass_selection', 'scores'))
    metrics = []
    for family in ['flat', 'cnl']:
        pred = pd.concat([pd.read_parquet(paths.task('pass_selection', 'training') / f'fold_{f}' / family / 'test_predictions.parquet') for f in range(5)],
            ignore_index=True)
        pred.to_parquet(out / (family + '_pooled_predictions.parquet'), index=False)
        ce = float(pred.nll.mean())
        null = float(pred.nll_null.mean())
        metrics.append(dict(family=family, n=len(pred), cross_entropy=ce, null_cross_entropy=null, gain_over_null=1 - ce / null))
    pd.DataFrame(metrics).to_csv(out / 'metrics.csv', index=False)


def refit(paths, threads):
    p = protocol('pass_selection')
    spec = read_json(feature_root(paths, None) / 'feature_spec.json')
    matches = sorted((d.name for d in feature_root(paths, None).iterdir() if d.is_dir()))
    x, rows = load(paths, None, matches)
    sizes = rows.n_candidates.to_numpy()
    chosen = rows.chosen.to_numpy()
    member_root = paths.task('pass_selection', 'features') / 'production_memberships'
    for match in matches:
        folder = member_root / 'fold_0' / match
        folder.mkdir(parents=True, exist_ok=False)
        source = feature_root(paths, None) / match
        keys = pd.read_parquet(source / 'rows.parquet')[KEYS]
        keys.to_parquet(folder / 'keys.parquet', index=False)
        np.save(folder / 'log_alpha_fold_0.npy', np.load(source / 'log_alpha.npy'))
        write_json(folder / 'completion.json',
            dict(status='complete', smoke=False, protocol_sha256=sha256(Path(__file__).with_name('protocol.json')), files={'log_alpha_fold_0.npy': sha256(folder / 'log_alpha_fold_0.npy')}))
    for family in ['flat', 'cnl']:
        out = fresh_directory(paths.task('pass_selection', 'production') / family)
        selections = [read_json(paths.task('pass_selection', 'training') / f'fold_{f}' / family / 'selection.json') for f in range(5)]
        rounds = int(np.median([s['rounds'] for s in selections]))
        params = p['fixed_parameters'] | {k: v for k,
            v in p['selected_by_family'][family].items() if k != 'name'}
        params.update(nthread=threads, seed=p['seed'], seed_per_iteration=True, disable_default_eval_metric=1)
        engine = Engine(member_root,
            rows[KEYS],
            sizes,
            chosen,
            0,
            family,
            sha256(Path(__file__).with_name('protocol.json')))
        matrix = xgb.DMatrix(x, feature_names=spec['feature_names'], nthread=threads)
        model = xgb.Booster(params=params, cache=[matrix])
        rho = 1.0
        for step in range(rounds):
            model.update(matrix, step, fobj=lambda z, d: engine.gradient(z, rho))
            if family == 'cnl' and (step + 1) % p['nesting']['update_every_trees'] == 0:
                cfg = p['nesting']
                rho, _ = fit_rho(engine,
                    model.predict(matrix, output_margin=True),
                    rho,
                    bounds=tuple(cfg['rho_bounds']),
                    relaxation=cfg['relaxation'],
                    scan=tuple(cfg['conditional_scan']))
        logs = []
        pooled = []
        for f in range(5):
            src = paths.task('pass_selection', 'training') / f'fold_{f}' / family
            logs.append(np.load(src / 'test_logp.npy'))
            pooled.append(pd.read_parquet(src / 'test_predictions.parquet'))
        held = pd.concat(pooled, ignore_index=True)
        cal = fit_temperature(np.concatenate(logs), held.n_candidates, held.chosen)
        export(paths, out, model, family, rho, cal, spec)
        write_json(out / 'selection.json',
            dict(rounds=rounds, parameters=params, calibration_source='outer held-out raw choice probabilities'))
