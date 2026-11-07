"""Paths, checked files, match splits and scores shared by the research stages.

The output root belongs to one complete run. Feature generation, fitting,
held-out scoring and production refitting have separate directories. Nothing
in this module installs weights into the inference package or submits jobs.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
REPOSITORY = Path(__file__).resolve().parents[2]
TASKS = ('ball_drive', 'pass_probability', 'pass_selection', 'action_selection', 'residual_risk')
ACTION_KEYS = ['match_id', 'action_id', 'frame_start']
PASS_KEYS = ['match_id', 'pass_id', 'frame_end']


def sha256(path):
    """Hash a file without loading a large feature matrix into memory."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def json_value(value):
    """Convert numerical metadata, retaining nonfinite values as JSON null."""
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(item) for item in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and (not np.isfinite(value)):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    """Write readable metadata atomically; no pickle or executable objects."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def fresh_directory(path):
    """Protect completed stages from accidental overwrite; use a new run root."""
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'Stage already contains outputs: {path}. Choose a new output root.')
    path.mkdir(parents=True, exist_ok=True)
    return path

@dataclass(frozen=True)


class Paths:
    """Explicit roots; scientific code never searches previous experiments."""
    raw: Path
    base: Path
    output: Path

    def task(self, name, stage):
        return self.output / name / stage

    def match(self, match_id):
        return self.base / 'matches' / str(match_id)


def parser(task, stage, description):
    """Common command-line contract for the numbered standalone scripts."""
    result = argparse.ArgumentParser(description=description)
    result.add_argument('--raw-root', type=Path, default=os.environ.get('PRESS_RAW_ROOT'))
    result.add_argument('--base-root', type=Path, default=os.environ.get('PRESS_BASE_ROOT'))
    result.add_argument('--output-root',
        type=Path,
        required=True,
        help='Dedicated root for this complete run')
    result.add_argument('--fold', type=int, choices=range(5), help='One fold; omit to run all five')
    result.add_argument('--match', dest='matches', action='append', help='Optional match shard; repeatable')
    result.add_argument('--threads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    result.add_argument('--production-features',
        action='store_true',
        help='Use full-data upstream models; feature stage only')
    result.set_defaults(task=task, stage=stage)
    return result


def paths_from(args, *, require_base=True, require_raw=False):
    raw = Path(args.raw_root).expanduser().resolve() if args.raw_root else None
    base = Path(args.base_root).expanduser().resolve() if args.base_root else None
    if require_base and (base is None or not base.is_dir()):
        raise ValueError('Supply an existing --base-root (or PRESS_BASE_ROOT).')
    if require_raw and (raw is None or not raw.is_dir()):
        raise ValueError('Supply an existing --raw-root (or PRESS_RAW_ROOT).')
    output = args.output_root.expanduser().resolve()
    return Paths(raw, base, output)


def protocol(task):
    return read_json(REPOSITORY / 'pipeline' / task / 'protocol.json')


def selected_folds(args):
    return range(5) if args.fold is None else [args.fold]


def match_ids(paths, requested=None):
    available = sorted((path.name for path in (paths.base / 'matches').iterdir() if path.is_dir()))
    if requested:
        missing = set(requested) - set(available)
        if missing:
            raise ValueError(f'Unknown matches: {sorted(missing)}')
        return [name for name in available if name in requested]
    return available


def read_table(path):
    frame = pd.read_parquet(path)
    if 'match_id' in frame:
        frame['match_id'] = frame.match_id.astype(str)
    return frame


def finish_stage(folder, *, task, stage, smoke=False, **metadata):
    """Inventory output hashes after a stage completes successfully."""
    folder = Path(folder)
    files = {str(path.relative_to(folder)): sha256(path) for path in sorted(folder.rglob('*')) if path.is_file() and path.name != 'manifest.json' and (not path.name.endswith('.tmp'))}
    write_json(folder / 'manifest.json',
        dict(status='complete', task=task, stage=stage, smoke=smoke, files=files, **metadata))


def check_stage(folder, *, allow_smoke=False):
    """Require a complete stage and exact input identities before consumption."""
    folder = Path(folder)
    metadata = read_json(folder / 'manifest.json')
    if metadata['status'] != 'complete' or (metadata.get('smoke') and (not allow_smoke)):
        raise ValueError(f'Incomplete or smoke input stage: {folder}')
    for name, expected in metadata['files'].items():
        if sha256(folder / name) != expected:
            raise ValueError(f'Changed stage input: {folder / name}')
    return metadata


def binary_scores(y, probability, null_probability, *, floor=1e-15):
    """Natural-prevalence scores; the null is supplied by the matching fit."""
    y = np.asarray(y, int)
    p = np.asarray(probability, float)
    null = np.broadcast_to(np.asarray(null_probability, float), p.shape)
    if not len(y) or p.shape != y.shape or (not np.isfinite(p).all()):
        raise ValueError('Expected aligned, finite held-out predictions.')

    def ce(q):
        q = np.clip(q, floor, 1 - floor)
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log1p(-q)))
    labels = pd.qcut(pd.Series(p), q=10, duplicates='drop')
    bins = pd.DataFrame({'probability': p, 'observed': y, 'bin': labels}).groupby('bin',
        observed=True).agg(n=('observed', 'size'),
        predicted=('probability', 'mean'),
        observed=('observed', 'mean')).reset_index(drop=True)
    if bins.empty:
        bins = pd.DataFrame([dict(n=len(y), predicted=float(p.mean()), observed=float(y.mean()))])
    cross_entropy, null_ce = (ce(p), ce(null))
    scores = dict(n=len(y),
        positives=int(y.sum()),
        cross_entropy=cross_entropy,
        null_cross_entropy=null_ce,
        gain_over_null=1 - cross_entropy / null_ce,
        brier=float(np.mean((p - y) ** 2)),
        ece_quantile=float((bins.n * abs(bins.predicted - bins.observed)).sum() / len(y)),
        mean_predicted=float(p.mean()),
        observed_frequency=float(y.mean()))
    if len(np.unique(y)) == 2:
        scores.update(roc_auc=float(roc_auc_score(y, p)), pr_auc=float(average_precision_score(y, p)))
    return (scores, bins)


def encode_categories(frame, numeric, vocabulary):
    """Fit-only category levels; preserve explicit missing and unseen buckets."""
    columns = list(numeric)
    blocks = [frame[numeric].to_numpy(np.float32)]
    for column, levels in vocabulary.items():
        values = frame[column].astype(object)
        mapped = values.where(values.isin(levels), '__unseen__').where(values.notna(), '__missing__')
        columns += [column + '=' + str(level) for level in levels]
        blocks.append(np.column_stack([mapped.eq(level).to_numpy(np.float32) for level in levels]))
    matrix = np.column_stack(blocks).astype(np.float32)
    if np.isinf(matrix).any():
        raise ValueError('Feature values may be missing but cannot be infinite.')
    return (matrix, columns)


def vocabulary(frame):
    return {column: sorted(frame[column].dropna().unique().tolist()) + ['__missing__', '__unseen__'] for column in ['start_type_raw', 'previous_action_raw']}
