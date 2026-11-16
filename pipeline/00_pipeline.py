"""Run the research stages in order, starting from raw SkillCorner matches.

Without --execute, print the commands. Each fit writes to a new output root;
the handoff commands supply fixed protocols and split-specific upstream models.
"""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


REVIEW = Path(__file__).resolve().parent


def commands(input_root, output_root, raw_root, threads, tracking_root=None):
    """Return all commands, including the dependencies between fitted tasks."""
    py = sys.executable
    tracking = tracking_root or output_root / "preprocessing"
    def script(name, *args):
        return [py, str(REVIEW / name / "pipeline.py"), *map(str, args)]
    def handoff(stage):
        return [py, str(REVIEW / "handoff.py"), stage, "--data-root", str(input_root),
                "--output-root", str(output_root)]
    released_pass = REVIEW / "released/pass_probability/pipeline.py"
    pass_script = released_pass if released_pass.is_file() else REVIEW / "pass_probability/pipeline.py"
    return {
        "01": [script("preprocessing", "--raw-root", raw_root or "/path/to/opendata/data/matches",
                      "--output", output_root / "preprocessing"), handoff("protocols")],
        "02": [script("expected_threat", "--stage", "all", "--data-root", input_root / "expected_threat",
                      "--output", output_root / "expected_threat")],
        "03": [script("ball_drive/physical", "--cache", input_root / "ball_drive", "--refit",
                      "--tracking-root", tracking, "--output", output_root / "ball_drive/physical")],
        "04": [script("ball_drive/catboost", "--stage", "all", "--data-root", input_root / "ball_drive",
                      "--physical-splits", output_root / "ball_drive/physical/physical_splits.json",
                      "--tracking-root", tracking, "--output", output_root / "ball_drive/catboost", "--threads", threads)],
        "05": [[py, str(pass_script), "--stage", "all", "--data-root", str(input_root / "pass_probability"),
                "--tracking-root", str(tracking), "--output", str(output_root / "pass_probability")]],
        "06": [handoff("physical"), script("pass_selection", "--stage", "all",
                      "--data-root", input_root / "pass_selection", "--tracking-root", tracking,
                      "--output", output_root / "pass_selection", "--threads", threads)],
        "07": [handoff("choice"), script("binary_classification", "--stage", "all",
                      "--data-root", input_root / "binary_classification", "--tracking-root", tracking,
                      "--output", output_root / "binary_classification", "--threads", threads)],
        "08": [script("composed_risk", "--stage", "all", "--data-root", input_root / "composed_risk",
                      "--binary-root", output_root / "binary_classification", "--output", output_root / "composed_risk",
                      "--threads", threads)],
        "09": [script("reports", "--stage", "all", "--states", output_root / "composed_risk/features/held_out.parquet",
                      "--case-file", input_root / "reports/cases.json", "--output", output_root / "reports",
                      "--tracking-root", tracking)],
        "10": [[py, str(REVIEW / "benchmarks.py"), "--output-root", str(output_root)]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", choices=[f"{n:02d}" for n in range(1,11)], action="append",
                        help="Repeat to select steps. No step: show all commands.")
    parser.add_argument("--input-root", type=Path,
                        help="Prepared task inputs; default: <output-root>/preprocessing/task_inputs.")
    parser.add_argument("--output-root", type=Path, default=REVIEW / "outputs")
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--tracking-root",type=Path,help="Override shared tracking, e.g. the new preprocessing output.")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    input_root = args.input_root.resolve() if args.input_root else output_root / "preprocessing/task_inputs"
    steps = commands(input_root, output_root, args.raw_root, args.threads, args.tracking_root)
    if args.execute and not args.step:
        parser.error("Choose --step explicitly; default is a reading plan, not an expensive full rerun.")
    for number, stage_commands in steps.items():
        if args.step and number not in args.step:
            continue
        for command in stage_commands:
            print(number, shlex.join(command), flush=True)
            if args.execute:
                if number == "01" and args.raw_root is None:
                    parser.error("Step 01 requires --raw-root.")
                subprocess.run(command, check=True, cwd=REVIEW.parent)


if __name__ == "__main__":
    main()
