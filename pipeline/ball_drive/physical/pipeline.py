"""Execute the final physical ball-drive benchmark, using model.py and training.py."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import jax
import pandas as pd

from pipeline.ball_drive.physical.model import prepare_covariates
from pipeline.ball_drive.physical.training import run_fold, collect_results, refit_production

from pipeline.frame_states import load_tracking_states


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PIPELINE_ROOT / "inputs" / "ball_drive"
DEFAULT_TRACKING = PIPELINE_ROOT / "inputs" / "shared" / "tracking"
DEFAULT_OUTPUT = PIPELINE_ROOT / "outputs" / "ball_drive" / "physical"


TEST_MATCHES = {
    0: ["1925299", "2015213"],
    1: ["2006229", "2011166"],
    2: ["1899585", "1953632"],
    3: ["1886347", "2013725"],
    4: ["1996435", "2017461"],
}


def main(cache, output, fold=None, collect_only=False, refit=False, tracking_root=DEFAULT_TRACKING):
    """Load the selected cohort, execute requested folds and save new outputs.

    --collect-only aggregates finished folds; --refit additionally estimates
    the all-match production model. Neither installs weights into PRESS.
    """
    jax.config.update("jax_enable_x64", True)
    cache, output = Path(cache), Path(output)
    output.mkdir(parents=True, exist_ok=True)

    # Input: the final, already selected 9,114 one-second observations.
    # Early failures keep their one-second Bernoulli target; no successful
    # micro-tail is added, and no new filtering is introduced in this pipeline.
    rows = pd.read_parquet(cache / "rows.parquet")
    rows["match_id"] = rows.match_id.astype(str)
    if rows.duplicated(["match_id", "frame_start", "frame_end"]).any():
        raise ValueError("Duplicate initial observations.")
    # Use the same package getter as pass probability and pass selection.
    # No observed endpoint is needed: the physical model projects this start state.
    raw, _ = load_tracking_states(rows, tracking_root, frame_column="frame_start", carrier_column="player_id")
    covariates = prepare_covariates(raw, rows.outcome.to_numpy())

    if not collect_only:
        selected_folds = range(5) if fold is None else [fold]
        for current in selected_folds:
            run_fold(rows, covariates, current, TEST_MATCHES[current], output)

    # A single-fold cluster task only writes its own folder. Pool once after
    # all jobs finish, using --collect-only.
    if fold is not None and not collect_only and not refit:
        return

    if all((output / f"fold_{f}" / "complete.json").is_file() for f in range(5)):
        print("POOLED", collect_results(output), flush=True)
        if refit:
            refit_production(rows, covariates, output)
    elif collect_only or refit:
        raise ValueError("All five completed folds are required for pooling/refitting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_INPUT,
                        help="Selected final states (default: pipeline/inputs/ball_drive).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="New output folder (default: pipeline/outputs/ball_drive/physical).")
    parser.add_argument("--tracking-root", type=Path, default=DEFAULT_TRACKING)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--refit", action="store_true")
    args = parser.parse_args()
    main(args.cache, args.output, args.fold, args.collect_only, args.refit, args.tracking_root)
