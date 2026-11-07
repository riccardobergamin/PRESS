"""Held-out h/xLoss and fixed-threshold match bootstrap; no additional fit."""
import numpy as np
import pandas as pd
import xgboost as xgb
from pipeline.library import binary_training as binary
from pipeline.library.common import fresh_directory, read_json, write_json, binary_scores, encode_categories
from pipeline.library.states import load_match, folds
from pipeline.ball_drive.filters import parent_mask
from pipeline.pass_probability.filters import verified_passes


def compute(paths, threads=1):
    """Compose the matching outer-fold models on verified canonical states."""
    records = []
    for fold, split in folds().items():
        frame = binary.load(paths, 'action_selection', int(fold), split['test_matches'])
        models = {}
        for task in ['action_selection', 'residual_risk']:
            folder = paths.task(task, 'training') / f'fold_{fold}'
            selection = read_json(folder / 'selection.json')
            model = xgb.Booster()
            model.load_model(folder / 'model.ubj')
            model.set_param({'nthread': threads})
            models[task] = (model, selection)
        action, selection = models['action_selection']
        matrix, names = encode_categories(frame, binary.numeric('action_selection'), selection['vocabulary'])
        p = action.predict(xgb.DMatrix(matrix, feature_names=names, nthread=threads))
        frame['action_carry_probability'] = 1 - p
        residual, selection = models['residual_risk']
        matrix, names = encode_categories(frame, binary.numeric('residual_risk'), selection['vocabulary'])
        r = residual.predict(xgb.DMatrix(matrix, feature_names=names, nthread=threads))
        drive = 1 - frame.drive_success_1s.to_numpy()
        failed_pass = 1 - frame.pass_success_mean.to_numpy()
        frame['h'] = p * failed_pass + (1 - p) * drive
        frame['xloss'] = p * failed_pass + (1 - p) * (drive + (1 - drive) * r)
        frame['fold'] = int(fold)
        for match in split['test_matches']:
            tracking, events, players, metadata = load_match(paths.base, match)
            actions = pd.read_parquet(paths.match(match) / 'actions.parquet')
            passes = verified_passes(events, tracking, players)
            valid = parent_mask(actions,
                events) & (actions.end_type.ne('pass') | actions.possession_id.isin(passes.event_id))
            labels = actions.loc[valid, ['action_id', 'frame_start', 'possession_id', 'outcome', 'end_type']]
            parents = pd.read_parquet(paths.match(match) / 'possessions.parquet').set_index('possession_id')
            labels['terminal_loss'] = labels.possession_id.map(parents.residual_loss).astype(int)
            labels['current_failure'] = 1 - labels.outcome
            labels['band_eligible'] = labels.end_type.isin(['pass', 'possession_loss'])
            selected = frame.loc[frame.match_id.eq(match)].merge(labels.drop(columns='possession_id'),
                on=['action_id', 'frame_start'],
                validate='one_to_one')
            records.append(selected)
    out = fresh_directory(paths.output / 'composed_risk' / 'features')
    pd.concat(records, ignore_index=True).to_parquet(out / 'held_out.parquet', index=False)


def score(paths):
    """Score all verified states; bootstrap risk bands on terminally covered parents."""
    frame = pd.read_parquet(paths.output / 'composed_risk' / 'features' / 'held_out.parquet')
    out = fresh_directory(paths.output / 'composed_risk' / 'scores')
    results = {}
    for name, label in [('h', 'current_failure'), ('xloss', 'terminal_loss')]:
        null = np.full(len(frame), np.nan)
        for fold, split in folds().items():
            training = frame.loc[frame.match_id.isin(split['fit_matches']), label]
            null[frame.fold.eq(int(fold))] = training.mean()
        metrics, calibration = binary_scores(frame[label], frame[name], null)
        results[name] = metrics
        calibration.to_csv(out / f'{name}_calibration.csv', index=False)
    write_json(out / 'metrics.json', results)
    bands = frame.loc[frame.band_eligible].copy()
    thresholds = np.array([0.082996, 0.159104, 0.292893])
    bands['band'] = np.searchsorted(thresholds, bands.h, side='right')
    matches = sorted(bands.match_id.unique())
    n = np.zeros((len(matches), 4))
    predicted = n.copy()
    observed = n.copy()
    for i, match in enumerate(matches):
        for b in range(4):
            rows = bands.loc[bands.match_id.eq(match) & bands.band.eq(b)]
            n[i, b], predicted[i, b], observed[i, b] = (len(rows), rows.h.sum(), rows.current_failure.sum())
    rng = np.random.default_rng(20261030)
    draws = rng.integers(0, len(matches), size=(2000, len(matches)))
    counts = n[draws].sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        means = predicted[draws].sum(axis=1) / counts
        frequencies = observed[draws].sum(axis=1) / counts
    summaries = []
    for b, name in enumerate(['None', 'Low', 'Medium', 'High']):
        summaries.append(dict(band=name, actions=int(n[:, b].sum()), predicted=float(predicted[:, b].sum() / n[:, b].sum()), observed=float(observed[:, b].sum() / n[:, b].sum()), predicted_interval=np.nanquantile(means[:, b], [0.025, 0.975]).tolist(), observed_interval=np.nanquantile(frequencies[:, b], [0.025, 0.975]).tolist(), empty_replicates=int((counts[:, b] == 0).sum()), min_actions=int(counts[:, b].min()), max_actions=int(counts[:, b].max())))
    write_json(out / 'bands.json',
        dict(bands=summaries, seed=20261030, replicates=2000, thresholds=thresholds, very_high_states=int(bands.h.ge(0.5).sum()), empty_band_policy='Undefined replicate ratios omitted from that band percentile interval'))
    np.savez_compressed(out / 'band_bootstrap.npz', counts=counts, predicted=means, observed=frequencies)
