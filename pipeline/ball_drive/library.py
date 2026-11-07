"""One complete physical ball-drive benchmark and operational MAP refit."""
from dataclasses import asdict
import numpy as np
import pandas as pd
from pipeline.library.common import fresh_directory, write_json, read_json, protocol, binary_scores
from pipeline.library.states import load_match, initial_state, save_states, folds
from pipeline.library.ball_drive import likelihood as engine
from pipeline.library.ball_drive.design import interception_angle, signed_boundary, design_split, active
from pipeline.library.ball_drive.export import make_artifact
from .filters import select


def features(paths, matches):
    for match in matches:
        folder = fresh_directory(paths.task('ball_drive', 'features') / match)
        tracking, events, players, metadata = load_match(paths.base, match)
        rows = select(pd.read_parquet(paths.match(match) / 'drive_observations.parquet'), events)
        rows['match_id'] = match
        states = [initial_state(tracking, r.frame_start, r.player_id, r.attacking_side) for r in rows.itertuples(index=False)]
        rows.to_parquet(folder / 'rows.parquet', index=False)
        save_states(folder / 'states.npz', states)
        print('DRIVE_FEATURES', match, len(rows), flush=True)


def load(paths):
    tables = []
    arrays = {}
    for folder in sorted(paths.task('ball_drive', 'features').iterdir()):
        if not folder.is_dir():
            continue
        rows = pd.read_parquet(folder / 'rows.parquet')
        tables.append(rows)
        with np.load(folder / 'states.npz') as z:
            for key in z.files:
                arrays.setdefault(key, []).append(z[key])
    rows = pd.concat(tables, ignore_index=True)
    raw = {k: np.concatenate(v) for k, v in arrays.items()}
    p = protocol('ball_drive')
    data = dict(outcome=rows.outcome.to_numpy(float),
        speed=np.linalg.norm(raw['v_bc'], axis=-1),
        boundary=signed_boundary(raw),
        tti=engine.arrival_times(raw, p['dynamics']))
    data['angle'], data['angle_valid'] = interception_angle(raw)
    return (rows, data)


def subset(data, mask):
    return {key: value[mask] for key, value in data.items()}


def fit_split(data, train, test, candidate, seed):
    p = protocol('ball_drive')
    spec, angle, (td, vd), (mt, mv) = design_split(subset(data, train), subset(data, test), p['shape'])
    count, precision, kind = active(spec, angle, candidate)
    fit = engine.fit_map(td,
        mt[:, :, :count],
        precision,
        kind=kind,
        scale=1.0,
        seed=seed,
        integration='exponential')
    probability, _ = engine.predict(vd, mv[:, :, :count], fit['theta'], kind, 1.0, integration='exponential')
    stored = dict(case=p,
        design=asdict(spec),
        angle_design=asdict(angle) if angle else None,
        candidate=candidate,
        active_spline_columns=count,
        kind=kind,
        outer_fits=[dict(fit=fit)])
    return (fit, probability, stored, precision)


def artifact(stored, matches):
    return make_artifact(stored,
        model_id='fresh_ball_drive_refit',
        training=dict(match_ids=list(matches), role='operational_or_fold_refit'),
        benchmark={})


def train(paths, fold):
    from pipeline.library.ball_drive.splines import paired_selection
    rows, data = load(paths)
    p = protocol('ball_drive')
    split = folds()[str(fold)]
    folder = fresh_directory(paths.task('ball_drive', 'training') / f'fold_{fold}')
    outer = sorted(split['fit_matches'] + split['validation_matches'])
    records = []
    inner_models = {}
    for inner in range(4):
        held = outer[2 * inner:2 * inner + 2]
        tr = rows.match_id.isin([m for m in outer if m not in held]).to_numpy()
        val = rows.match_id.isin(held).to_numpy()
        for candidate in p['grid']:
            fitted, probability, stored, _ = fit_split(data,
                tr,
                val,
                candidate,
                10 + 10000 * fold + 100 * inner)
            losses = -(data['outcome'][val] * np.log(probability) + (1 - data['outcome'][val]) * np.log1p(-probability))
            for match in held:
                take = rows.loc[val, 'match_id'].eq(match).to_numpy()
                records.append(dict(candidate=candidate['name'], match_id=match, inner=inner, n=int(take.sum()), nll=float(losses[take].mean())))
            if set(held) == set(split['validation_matches']):
                inner_models[candidate['name']] = stored
            print('DRIVE_MAP', fold, inner, candidate['name'], flush=True)
    name, selection = paired_selection(records, [c['name'] for c in p['grid']])
    chosen = next((c for c in p['grid'] if c['name'] == name))
    tr = rows.match_id.isin(outer).to_numpy()
    te = rows.match_id.isin(split['test_matches']).to_numpy()
    fitted, probability, stored, precision = fit_split(data, tr, te, chosen, 10 + 10000 * fold + 900)
    write_json(folder / 'fit.json', stored)
    write_json(folder / 'selection.json', selection)
    write_json(folder / 'six_fit_model.json', artifact(inner_models[name], split['fit_matches']))
    out = rows.loc[te].copy()
    out['y'] = 1 - out.outcome
    out['probability'] = 1 - probability
    out['null_probability'] = float(1 - data['outcome'][tr].mean())
    out['fold'] = fold
    out.to_parquet(folder / 'test_predictions.parquet', index=False)
    pd.DataFrame(records).to_csv(folder / 'inner_scores.csv', index=False)
    np.savez_compressed(folder / 'posterior.npz',
        theta=fitted['theta'],
        precision=precision,
        covariance=fitted['covariance'],
        hessian_eigenvalues=fitted['hessian_eigenvalues'])


def score(paths):
    folder = fresh_directory(paths.task('ball_drive', 'scores'))
    pred = pd.concat([pd.read_parquet(paths.task('ball_drive', 'training') / f'fold_{f}' / 'test_predictions.parquet') for f in range(5)],
        ignore_index=True)
    metrics, bins = binary_scores(pred.y, pred.probability, pred.null_probability, floor=1e-06)
    pred.to_parquet(folder / 'pooled_predictions.parquet', index=False)
    write_json(folder / 'metrics.json', metrics)
    bins.to_csv(folder / 'calibration.csv', index=False)


def refit(paths):
    rows, data = load(paths)
    p = protocol('ball_drive')
    selections = [read_json(paths.task('ball_drive', 'training') / f'fold_{f}' / 'fit.json')['candidate'] for f in range(5)]
    if not all((c == selections[0] for c in selections)):
        raise ValueError('Outer smoothing selections differ: choose the full-data policy explicitly before refitting.')
    take = np.ones(len(rows), bool)
    fitted, _, stored, precision = fit_split(data, take, take, selections[0], p['refit_seed'])
    folder = fresh_directory(paths.task('ball_drive', 'production'))
    model = artifact(stored, sorted(rows.match_id.unique()))
    model['posterior_covariance'] = fitted['covariance'].tolist()
    write_json(folder / 'ball_drive_model.json', model)
    write_json(folder / 'fit.json', stored)
    np.savez_compressed(folder / 'posterior.npz',
        theta=fitted['theta'],
        precision=precision,
        covariance=fitted['covariance'],
        hessian_eigenvalues=fitted['hessian_eigenvalues'])
