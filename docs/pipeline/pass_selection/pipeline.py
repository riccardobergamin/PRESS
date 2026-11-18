"""Run the reviewed pass-selection workflow, one explicit stage at a time.

Start with run_pipeline below for the complete sequence, then read
training.train_validate_test and training.train_model for one fold and its trees.
Screening compares five configurations per family on fold 0 (6+2), without
temperature scaling. The selected configurations then enter five 6+2+2 folds.
Training can also reproduce the accepted configurations without re-screening.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import pandas as pd

from pipeline.pass_selection.features import prepare_features
from pipeline.pass_selection.training import (
    SCREENING_FOLD, SCREENING_CONFIGURATIONS, SELECTED_CONFIGURATIONS,
    compare_configurations, select_best_configuration, train_validate_test,
    compute_pooled_metrics, train_production_model, collect_feature_importance,
)


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PIPELINE_ROOT / "inputs" / "pass_selection"
DEFAULT_TRACKING = PIPELINE_ROOT / "inputs" / "shared" / "tracking"
DEFAULT_OUTPUT = PIPELINE_ROOT / "outputs" / "pass_selection"


def run_pipeline(data_root=DEFAULT_INPUT, tracking_root=DEFAULT_TRACKING,
                 output=DEFAULT_OUTPUT, families=("flat", "cnl"), threads=1):
    """Features -> screening -> five fit/validation/test splits -> production.

    Each family uses the same saved features. Every training starts from zero
    utilities; no fitted trees are reused between configurations or folds.
    Outputs go to a new folder, without replacing the benchmark or package.
    """
    jax.config.update("jax_enable_x64", True)
    protocol = json.loads((data_root / "protocol.json").read_text())
    pass_table = pd.read_parquet(data_root / "rows.parquet")
    pass_table["match_id"] = pass_table.match_id.astype(str)

    # 1. Calculate features with the upstream physical models of each fold.
    for fold in range(5):
        prepare_features(data_root, tracking_root, output, f"fold_{fold}")

    # 2. Compare five configurations on the same six fitting + two validation
    # matches. Choose each family's configuration using native validation CE.
    screening_split = protocol["folds"][str(SCREENING_FOLD)]
    configurations = {}
    for family in families:
        compare_configurations(
            output / "features" / f"fold_{SCREENING_FOLD}", screening_split,
            family, output, threads=threads,
        )
        configurations[family] = select_best_configuration(output, family)

    # 3. For each fold: fit trees/rho on six matches, select the checkpoint and
    # calibrate temperature on two validation matches, then evaluate two tests.
    for family in families:
        configuration = configurations[family]
        for fold in range(5):
            train_validate_test(
                output / "features" / f"fold_{fold}",
                protocol["folds"][str(fold)], family, fold, output,
                threads=threads, configuration=configuration,
                max_rounds=configuration["max_rounds"],
            )

    # 4. Pool the five held-out test sets before computing benchmark metrics.
    benchmark_metrics = {}
    for family in families:
        benchmark_metrics[family] = compute_pooled_metrics(output, family, pass_table)

    # 5. Prepare all-ten-match features and train fresh production models.
    # Their tree counts and calibration come from the completed five folds.
    prepare_features(data_root, tracking_root, output, "production")
    for family in families:
        train_production_model(
            output / "features" / "production", data_root, output, family,
            threads=threads,
        )
    return benchmark_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=[
        "features", "screen", "screen-report", "train", "scores", "refit", "importance", "all",
    ])
    parser.add_argument("--family", default="both", choices=["flat", "cnl", "both"])
    parser.add_argument("--fold", type=int, choices=range(5), help="One fold for features or train; omit for all five.")
    parser.add_argument("--production-features", action="store_true",
                        help="With stage=features: use all-ten upstream weights instead of fold weights.")
    parser.add_argument("--configuration", choices=[candidate["name"] for candidate in SCREENING_CONFIGURATIONS],
                        help="With stage=screen: fit only this configuration, for separate jobs.")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--tracking-root", type=Path, default=DEFAULT_TRACKING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.fold is not None and args.stage not in ["features", "train"]:
        parser.error("--fold is only for features or train.")
    if args.production_features and (args.stage != "features" or args.fold is not None):
        parser.error("--production-features requires stage=features and no --fold.")
    if args.configuration is not None and args.stage != "screen":
        parser.error("--configuration is only for stage=screen.")
    if args.threads < 1:
        parser.error("--threads must be positive.")
    families = ["flat", "cnl"] if args.family == "both" else [args.family]
    if args.stage == "all":
        benchmark_metrics = run_pipeline(
            args.data_root, args.tracking_root, args.output, families, args.threads,
        )
        for family, metrics in benchmark_metrics.items():
            print(family, metrics.to_string(index=False), flush=True)
        return

    jax.config.update("jax_enable_x64", True)
    protocol = json.loads((args.data_root / "protocol.json").read_text())
    pass_table = pd.read_parquet(args.data_root / "rows.parquet")
    pass_table["match_id"] = pass_table.match_id.astype(str)
    folds = list(range(5)) if args.fold is None else [args.fold]

    # 1. Same covariates and memberships are calculated ONCE for both families.
    if args.stage == "features":
        split_names = ["production"] if args.production_features else [f"fold_{fold}" for fold in folds]
        for split_name in split_names:
            prepare_features(args.data_root, args.tracking_root, args.output, split_name)

    # 2. Exploratory screening: SAME six fitting and two validation matches for
    # all five candidates and both families. Features are reused, trees are not.
    # No temperature fit and no outer-test evaluation are performed at this stage.
    if args.stage == "screen":
        split = protocol["folds"][str(SCREENING_FOLD)]
        for family in families:
            compare_configurations(args.output / "features" / f"fold_{SCREENING_FOLD}",
                          split, family, args.output, threads=args.threads,
                          configuration_name=args.configuration)
            # Separate candidate jobs are collected only after all five finish.
            if args.configuration is None:
                print(family, select_best_configuration(args.output, family), flush=True)
    if args.stage == "screen-report":
        for family in families:
            print(family, select_best_configuration(args.output, family), flush=True)

    # 3. Fresh six-match fits. Use this run's screening winner when available;
    # otherwise reproduce the already accepted family-specific configuration.
    # The two validation matches select prefix/rho, then temperature. Test is last.
    if args.stage == "train":
        for family in families:
            screening_folder = args.output / "screening" / family
            selection_file = screening_folder / "selection.json"
            if selection_file.is_file():
                configuration = json.loads(selection_file.read_text())["configuration"]
                configuration_source = "this run's native 6+2 screening"
            elif screening_folder.exists():
                raise ValueError("Complete screen-report before training with a partial screening.")
            else:
                configuration = SELECTED_CONFIGURATIONS[family]
                configuration_source = "accepted configuration (no new screening)"
            print("CONFIGURATION", family, configuration_source, configuration, flush=True)
            for fold in folds:
                split = protocol["folds"][str(fold)]
                train_validate_test(args.output / "features" / f"fold_{fold}", split, family, fold,
                         args.output, threads=args.threads, configuration=configuration,
                         max_rounds=configuration.get("max_rounds", protocol["max_rounds"]))

    # 4. Concatenate each pass's single outer-test prediction to obtain the benchmark.
    if args.stage == "scores":
        for family in families:
            print(family, compute_pooled_metrics(args.output, family, pass_table).to_string(index=False), flush=True)

    # 5. Full-data deployment refit. Its temperature uses the saved raw OOF maps.
    if args.stage == "refit":
        for family in families:
            train_production_model(args.output / "features" / "production", args.data_root,
                             args.output, family, threads=args.threads)

    # Importance is already calculated after every fit. This stage only rebuilds
    # its summary, useful after reviewing fold and production tables separately.
    if args.stage == "importance":
        for family in families:
            collect_feature_importance(args.output, family)


if __name__ == "__main__":
    main()
