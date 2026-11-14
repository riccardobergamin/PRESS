"""Numbered reading/execution order for the procedural review pipelines.

Default: print commands, do not run expensive fits. --execute is explicit.
Every module's own pipeline.py shows its scientific steps in procedural order.
The review starts from frozen accepted inputs, or from the preprocessing stage.
Regenerated task tables must be reconciled before replacing frozen inputs;
this runner never silently switches a benchmark to a different cohort.
"""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


REVIEW = Path(__file__).resolve().parent


def commands(input_root, output_root, raw_root, threads, tracking_root=None):
    """One visible dependency order; shared input/output roots are explicit."""
    py = sys.executable
    tracking = tracking_root or input_root / "shared/tracking"
    def script(name, *args):
        return [py, str(REVIEW / name / "pipeline.py"), *map(str, args)]
    return {
        "01": script("preprocessing", "--raw-root", raw_root or "/path/to/opendata/data/matches",
                     "--output", output_root / "preprocessing"),
        "02": script("expected_threat", "--stage", "all", "--data-root", input_root / "expected_threat",
                     "--output", output_root / "expected_threat"),
        "03": script("ball_drive/physical", "--cache", input_root / "ball_drive", "--refit",
                     "--tracking-root", tracking, "--output", output_root / "ball_drive/physical"),
        "04": script("ball_drive/catboost", "--stage", "all", "--data-root", input_root / "ball_drive",
                     "--physical-splits", output_root / "ball_drive/physical/physical_splits.json",
                     "--tracking-root", tracking, "--output", output_root / "ball_drive/catboost", "--threads", threads),
        "05": script("pass_probability", "--stage", "all", "--data-root", input_root / "pass_probability",
                     "--tracking-root", tracking, "--output", output_root / "pass_probability"),
        "06": script("pass_selection", "--stage", "all", "--data-root", input_root / "pass_selection",
                     "--tracking-root", tracking, "--output", output_root / "pass_selection", "--threads", threads),
        "07": script("binary_classification", "--stage", "all", "--data-root", input_root / "binary_classification",
                     "--tracking-root", tracking, "--output", output_root / "binary_classification", "--threads", threads),
        "08": script("composed_risk", "--stage", "all", "--data-root", input_root / "composed_risk",
                     "--binary-root", output_root / "binary_classification", "--output", output_root / "composed_risk",
                     "--threads", threads),
        "09": script("reports", "--stage", "all", "--states", output_root / "composed_risk/features/held_out.parquet",
                     "--case-file", input_root / "reports/cases.json", "--output", output_root / "reports",
                     "--tracking-root", tracking),
        "10": [py, str(REVIEW / "benchmarks.py"), "--output-root", str(output_root)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", choices=[f"{n:02d}" for n in range(1,11)], action="append",
                        help="Repeat to select steps. No step: show all commands.")
    parser.add_argument("--input-root", type=Path, default=REVIEW / "inputs")
    parser.add_argument("--output-root", type=Path, default=REVIEW / "outputs")
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--tracking-root",type=Path,help="Override shared tracking, e.g. the new preprocessing output.")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    steps = commands(args.input_root.resolve(), args.output_root.resolve(), args.raw_root, args.threads,args.tracking_root)
    if args.execute and not args.step:
        parser.error("Choose --step explicitly; default is a reading plan, not an expensive full rerun.")
    for number, command in steps.items():
        if args.step and number not in args.step:
            continue
        print(number, shlex.join(command), flush=True)
        if args.execute:
            if number == "01" and args.raw_root is None:
                parser.error("Step 01 requires --raw-root.")
            subprocess.run(command, check=True, cwd=REVIEW.parent)


if __name__ == "__main__":
    main()
