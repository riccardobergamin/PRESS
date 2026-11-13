"""Explicit release allowlist: procedural code and selected statistical tables.

No datasets, posterior draws, feature matrices, fold trees, histories or test
experiments are copied. Production inference weights remain in PRESS itself.
The source folders are enumerated so future local experiments cannot leak
into the release merely because they were added beneath review/.
"""

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CODE_FOLDERS = ["preprocessing", "expected_threat", "ball_drive/physical", "ball_drive/catboost",
                "pass_probability", "pass_selection", "binary_classification", "composed_risk", "reports"]
CODE_SUFFIXES = {".py", ".md"}
EXCLUDE_PARTS = {"old", "checks", "__pycache__"}


def export_code(destination):
    """Copy readable source only into a NEW destination, never the history."""
    destination.mkdir(parents=True, exist_ok=False)
    for filename in ["00_pipeline.py", "frame_states.py", "io.py", "benchmarks.py", "release.py", "README.md"]:
        shutil.copy2(ROOT / filename, destination / filename)
    for name in ["ball_drive/README.md", "ball_drive/TRAINING.md", "inputs/README.md",
                 "inputs/binary_classification/README.md", "inputs/pass_probability/README.md"]:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    for name in CODE_FOLDERS:
        for source in sorted((ROOT / name).rglob("*")):
            relative = source.relative_to(ROOT)
            if not source.is_file() or source.suffix not in CODE_SUFFIXES or EXCLUDE_PARTS.intersection(relative.parts):
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    # Each pipeline needs its declared split/grid protocol but no fitted trees.
    for name in ["pass_selection", "binary_classification", "expected_threat", "composed_risk"]:
        source = ROOT / "inputs" / name / "protocol.json"
        if source.exists():
            target = destination / "inputs" / name / "protocol.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    # Two task-specific cohort filters live one level above model variants.
    for name in ["ball_drive/filters.py", "pass_probability/filters.py"]:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    # The new smoke checks depend only on review code, PRESS and explicit
    # local inputs. Older archive-dependent equivalence checks stay local.
    for source in (ROOT / "checks").iterdir():
        if source.name in {"README.md", "run.py", "pass_selection_steps.py", "physical_steps.py",
                           "xt_steps.py", "binary_steps.py", "catboost_steps.py",
                           "compare_reports.py", "preprocessing_rows.py"}:
            target = destination / "checks" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def export_tables(output_root, destination):
    """Only approved performance, parameter-statistic and report tables.

    Physical fitting routines may save full posteriors beside summaries.
    They are NEVER recursively copied. State-level case tables contain
    observed identifiers and predicted scores for the selected examples only.
    """
    destination.mkdir(parents=True, exist_ok=False)
    approved = {
        "composed_risk": ["scores/pooled_metrics.csv", "scores/metrics_by_fold.csv",
                          "scores/metrics_by_match.csv", "scores/calibration.csv", "bands/summary.csv"],
        "reports": ["scores/pattern_counts.csv", "scores/player_summary.csv", "scores/team_summary.csv",
                    "scores/team_pattern_shares.csv", "scores/player_pattern_shares.csv",
                    "scores/team_peak_bands.csv", "scores/endpoint_transitions.csv",
                    "scores/team_match_bootstrap_sensitivity.csv", "cases/checkpoints.csv", "cases/possessions.csv"],
        "binary_classification": [f"{task}/scores/{name}" for task in ["action_selection", "residual_risk"]
                                  for name in ["metrics.json", "calibration.csv"]],
        "statistics": ["physical_parameters.csv", "performance.csv"],
        "ball_drive/physical": ["pooled_metrics.json", "pooled_calibration.csv"],
        "ball_drive/catboost": ["pooled_metrics.csv"],
        "pass_probability": ["scores/metrics.csv", "scores/calibration_deciles.csv"],
        "pass_selection": [f"{family}/scores/{name}" for family in ["flat", "cnl"]
                           for name in ["pooled_metrics.csv", "metrics_by_fold.csv", "metrics_by_match_id.csv"]],
    }
    copied = []
    for task, names in approved.items():
        for name in names:
            source = output_root / task / name
            if source.is_file():
                target = destination / task / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                copied.append(str(target.relative_to(destination)))
    (destination / "README.md").write_text("# Published statistical tables\n\n" + "\n".join("- `"+p+"`" for p in copied) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-destination", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--tables-destination", type=Path)
    args = parser.parse_args()
    if args.code_destination:
        export_code(args.code_destination)
    if args.tables_destination:
        export_tables(args.output_root, args.tables_destination)
    if not args.code_destination and not args.tables_destination:
        parser.error("Choose a NEW code and/or tables destination.")


if __name__ == "__main__":
    main()
