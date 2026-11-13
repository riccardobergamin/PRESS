"""01. Raw files -> kinematics -> matched possessions -> task observation tables."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from review.preprocessing.features import prepare_tracking, prepare_possessions, prepare_task_tables
from review.io import REVIEW_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["tracking", "possessions", "datasets", "all"], default="all")
    parser.add_argument("--raw-root", type=Path, help="Public SkillCorner data/matches directory.")
    parser.add_argument("--output", type=Path, default=REVIEW_ROOT / "outputs/preprocessing")
    parser.add_argument("--match", action="append", dest="matches", help="Repeat for a subset of matches.")
    parser.add_argument("--templates",type=Path,help="Explicit reviewed protocols/upstream artifacts; no observation rows are copied.")
    args = parser.parse_args()
    if args.stage in ["tracking", "all"] and args.raw_root is None:
        parser.error("--raw-root is required to read the raw observations.")
    source = args.raw_root if args.stage in ["tracking", "all"] else args.output / "matches"
    matches = args.matches or sorted(p.name for p in source.iterdir() if p.is_dir() and p.name.isdigit())
    if args.stage in ["tracking", "all"]:
        prepare_tracking(args.raw_root, args.output, matches)
    if args.stage in ["possessions", "all"]:
        prepare_possessions(args.output, matches)
    if args.stage in ["datasets", "all"]:
        prepare_task_tables(args.output, matches,args.templates)


if __name__ == "__main__":
    main()
