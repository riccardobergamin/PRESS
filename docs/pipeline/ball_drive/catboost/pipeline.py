"""Final CatBoost comparison: features -> nested validation/refit -> pooled test.

The physical ball-drive model remains the deployed model. This pipeline
reproduces the natural/loss10 CatBoost benchmark, without installing its fits.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pipeline.ball_drive.catboost.features import prepare_features, load_features
from pipeline.ball_drive.catboost.training import train_validate_test, compute_pooled_metrics


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PIPELINE_ROOT / "inputs/ball_drive"
DEFAULT_TRACKING = PIPELINE_ROOT / "inputs/shared/tracking"
DEFAULT_PHYSICAL_SPLITS = PIPELINE_ROOT / "outputs/ball_drive/physical/physical_splits.json"
DEFAULT_OUTPUT = PIPELINE_ROOT / "outputs/ball_drive/catboost"


def train_folds(rows, spatial_features, tti, valid, physical_splits, output,
                samplings=("natural", "loss10"), folds=range(5), threads=4):
    """Use already calculated geometry for both arms; every tree fit is fresh."""
    for sampling in samplings:
        for fold in folds:
            train_validate_test(
                rows, spatial_features, tti, valid, physical_splits[str(fold)],
                sampling, fold, threads, output,
            )


def run_pipeline(data_root=DEFAULT_INPUT, tracking_root=DEFAULT_TRACKING,
                 physical_splits=DEFAULT_PHYSICAL_SPLITS, output=DEFAULT_OUTPUT,
                 samplings=("natural", "loss10"), threads=4):
    """Complete fresh comparison, with one feature calculation for both arms."""
    # 1. Reconstruct actual frames through PRESS getters; calculate TTI once.
    rows, spatial_features, tti, valid, splits = prepare_features(
        data_root, tracking_root, physical_splits, output,
    )

    # 2. Each fold/arm: four 6+2 inner rotations choose configuration/trees;
    # inner OOF scores fit Platt/isotonic; three fresh eight-match models
    # predict the untouched two test matches. Only fitting rows are sampled.
    train_folds(rows, spatial_features, tti, valid, splits, output, samplings, threads=threads)

    # 3. Pool the five natural-prevalence test sets, then compute metrics.
    return compute_pooled_metrics(output, samplings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["features", "train", "scores", "all"], default="all")
    parser.add_argument("--data-root", "--cache", dest="data_root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--tracking-root", type=Path, default=DEFAULT_TRACKING)
    parser.add_argument("--physical-splits", type=Path, default=DEFAULT_PHYSICAL_SPLITS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fold", type=int, choices=range(5), help="One training fold; features are shared by all folds.")
    parser.add_argument("--sampling", choices=["natural", "loss10"])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.fold is not None and args.stage != "train":
        parser.error("--fold is only for --stage train; prepare features once beforehand.")
    samplings = (args.sampling,) if args.sampling else ("natural", "loss10")
    if args.stage == "all":
        print(run_pipeline(args.data_root, args.tracking_root, args.physical_splits,
                           args.output, samplings, args.threads).to_string(index=False))
    elif args.stage == "features":
        prepare_features(args.data_root, args.tracking_root, args.physical_splits, args.output)
    elif args.stage == "train":
        rows, spatial_features, tti, valid, splits = load_features(args.output)
        folds = range(5) if args.fold is None else [args.fold]
        train_folds(rows, spatial_features, tti, valid, splits, args.output, samplings, folds, args.threads)
    elif args.stage == "scores":
        print(compute_pooled_metrics(args.output, samplings).to_string(index=False))


if __name__ == "__main__":
    main()
