"""Procedural action-selection/residual-risk workflow; start at run_pipeline.

Features -> five native 6+2+2 fits -> pooled scores -> production features/refit.
Action selection precedes residual risk because the latter uses P(carry).
Training P(carry) is cross-fitted inside the six-match fit set. Validation
and test use the matching six-fit action model; production uses five 8+2 fits.
"""

import argparse
import json
from pathlib import Path
import sys

# Permit direct execution from any working directory, using this review tree.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.binary_classification.features import prepare_action_features, prepare_residual_features
from pipeline.binary_classification.training import (
    train_validate_test, compute_pooled_metrics, train_production_model,
)


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PIPELINE_ROOT / "inputs/binary_classification"
DEFAULT_TRACKING = PIPELINE_ROOT / "inputs/shared/tracking"
DEFAULT_OUTPUT = PIPELINE_ROOT / "outputs/binary_classification"


def run_pipeline(data_root=DEFAULT_INPUT, tracking_root=DEFAULT_TRACKING,
                 output=DEFAULT_OUTPUT, tasks=("action_selection", "residual_risk"), threads=1):
    """Complete fresh run, with features reused between the two tasks."""
    protocol = json.loads((Path(data_root) / "protocol.json").read_text())
    metrics = {}
    for task in tasks:
        # 1. Features for every outer fold. Residual features reuse the action
        # tables and add P(carry) with the approved classifier-level cross-fitting.
        for fold in range(5):
            if task == "action_selection":
                prepare_action_features(data_root, tracking_root, output, f"fold_{fold}")
            else:
                prepare_residual_features(data_root, output, f"fold_{fold}", threads)

        # 2. Full grid in each fixed 6+2+2 split; native objective and optimizer.
        for fold in range(5):
            train_validate_test(output, task, fold, protocol, threads)

        # 3. Pool untouched outer-test predictions, then compute metrics.
        metrics[task] = compute_pooled_metrics(data_root, output, task)

        # 4. New all-ten-match features and fixed-length production refit.
        if task == "action_selection":
            prepare_action_features(data_root, tracking_root, output, "production")
        else:
            prepare_residual_features(data_root, output, "production", threads)
        train_production_model(data_root, output, task, protocol, threads)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["features", "train", "scores", "refit", "all"], default="all")
    parser.add_argument("--task", choices=["action_selection", "residual_risk", "both"], default="both")
    parser.add_argument("--fold", type=int, choices=range(5), help="One fold for features/train; otherwise all five.")
    parser.add_argument("--production-features", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--tracking-root", type=Path, default=DEFAULT_TRACKING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.fold is not None and args.stage not in ["features", "train"]:
        parser.error("--fold is only for features/train.")
    if args.production_features and (args.stage != "features" or args.fold is not None):
        parser.error("--production-features requires stage=features and no --fold.")
    tasks = ("action_selection", "residual_risk") if args.task == "both" else (args.task,)
    if args.stage == "all":
        print(run_pipeline(args.data_root, args.tracking_root, args.output, tasks, args.threads))
        return

    protocol = json.loads((args.data_root / "protocol.json").read_text())
    folds = range(5) if args.fold is None else [args.fold]
    for task in tasks:
        if args.stage == "features":
            splits = ["production"] if args.production_features else [f"fold_{f}" for f in folds]
            for split_name in splits:
                if task == "action_selection":
                    prepare_action_features(args.data_root, args.tracking_root, args.output, split_name)
                else:
                    prepare_residual_features(args.data_root, args.output, split_name, args.threads)
        elif args.stage == "train":
            for fold in folds:
                train_validate_test(args.output, task, fold, protocol, args.threads)
        elif args.stage == "scores":
            print(task, compute_pooled_metrics(args.data_root, args.output, task))
        elif args.stage == "refit":
            train_production_model(args.data_root, args.output, task, protocol, args.threads)


if __name__ == "__main__":
    main()
