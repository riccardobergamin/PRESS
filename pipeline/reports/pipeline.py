"""08. Possession classification -> player/team scores -> selected real cases."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.io import PIPELINE_ROOT, new_folder, read_table, write_json
from pipeline.reports.features import classify_possessions
from pipeline.reports.scores import compute_reports
from pipeline.reports.cases import export_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["classify", "scores", "cases", "all"], default="all")
    parser.add_argument("--states", type=Path, default=PIPELINE_ROOT / "outputs/composed_risk/features/held_out.parquet")
    parser.add_argument("--case-file", type=Path, default=PIPELINE_ROOT / "inputs/reports/cases.json")
    parser.add_argument("--output", type=Path, default=PIPELINE_ROOT / "outputs/reports")
    parser.add_argument("--tracking-root", type=Path, help="Also render the real case frames using PRESS.")
    args = parser.parse_args()
    if args.stage in ["classify", "all"]:
        states, possessions = classify_possessions(read_table(args.states))
        folder = new_folder(args.output / "features")
        states.to_parquet(folder / "states.parquet", index=False)
        possessions.to_parquet(folder / "possessions.parquet", index=False)
        write_json(folder / "cohort.json", dict(n_states=len(states), n_possessions=len(possessions),
            n_initial=int(possessions.has_initial.sum()), micropossessions=int(possessions.single_micro_checkpoint.sum()),
            terminal_coverage=True, predictions="matching outer held-out fold", new_fit=False))
    if args.stage in ["scores", "cases", "all"]:
        states = read_table(args.output / "features/states.parquet")
        possessions = read_table(args.output / "features/possessions.parquet")
        if args.stage in ["scores", "all"]:
            compute_reports(states, possessions, args.output)
        if args.stage in ["cases", "all"]:
            export_cases(states, possessions, args.case_file, args.output, args.tracking_root)


if __name__ == "__main__":
    main()
