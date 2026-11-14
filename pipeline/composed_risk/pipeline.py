"""07. Components -> held-out h/xLoss -> pooled metrics -> band bootstrap."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.io import PIPELINE_ROOT, read_table
from pipeline.composed_risk.features import compute_held_out, replay_accepted_predictions
from pipeline.composed_risk.scores import compute_metrics, bootstrap_bands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["compute", "scores", "bands", "all"], default="all")
    parser.add_argument("--data-root", type=Path, default=PIPELINE_ROOT / "inputs/composed_risk")
    parser.add_argument("--binary-root", type=Path, default=PIPELINE_ROOT / "outputs/binary_classification")
    parser.add_argument("--output", type=Path, default=PIPELINE_ROOT / "outputs/composed_risk")
    parser.add_argument("--replay-accepted", action="store_true", help="Verify archived components; do not train.")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.stage in ["compute", "all"]:
        if args.replay_accepted:
            replay_accepted_predictions(args.data_root, args.output)
        else:
            compute_held_out(args.data_root, args.binary_root, args.output, args.threads)
    if args.stage in ["scores", "bands", "all"]:
        states = read_table(args.output / "features/held_out.parquet")
        if args.stage in ["scores", "all"]:
            print(compute_metrics(states, args.output).to_string(index=False))
        if args.stage in ["bands", "all"]:
            print(bootstrap_bands(states, args.output).to_string(index=False))


if __name__ == "__main__":
    main()
