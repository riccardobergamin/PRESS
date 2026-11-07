"""Thin commands for numbered stages; scientific logic stays in task libraries."""
import importlib
from pathlib import Path
from pipeline.library.common import parser, paths_from, match_ids, selected_folds, finish_stage


def task_main(task, stage):
    args = parser(task, stage, f'{task}: {stage} for one fresh run').parse_args()
    paths = paths_from(args)
    library = importlib.import_module('pipeline.' + task + '.library')
    if args.production_features and stage != 'features':
        raise ValueError('--production-features applies only to 01_features.py')
    if stage == 'features':
        matches = match_ids(paths, args.matches)
        if task in ['ball_drive', 'pass_probability']:
            library.features(paths, matches)
        elif task == 'residual_risk':
            for fold in [None] if args.production_features else selected_folds(args):
                library.features(paths, matches, fold, threads=args.threads)
        else:
            for fold in [None] if args.production_features else selected_folds(args):
                library.features(paths, matches, fold)
    elif stage == 'training':
        for fold in selected_folds(args):
            if task in ['ball_drive', 'pass_probability']:
                library.train(paths, fold)
            else:
                library.train(paths, fold, args.threads)
            finish_stage(paths.task(task, stage) / f'fold_{fold}', task=task, stage=stage)
    elif stage == 'scores':
        library.score(paths)
        finish_stage(paths.task(task, stage), task=task, stage=stage)
    elif stage == 'production':
        if task in ['ball_drive', 'pass_probability']:
            library.refit(paths)
        else:
            library.refit(paths, args.threads)
        finish_stage(paths.task(task, stage), task=task, stage=stage)
