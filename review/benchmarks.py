"""10. Collect small parameter/performance statistics, never fitted objects.

Reads this run's own outputs. Missing tasks are listed, not substituted with
another experiment. CE/ECE conventions remain those of the task that produced
the table; this collector does not recompute or average pooled fold metrics.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from review.io import REVIEW_ROOT, new_folder, write_json


def collect_statistics(output_root, destination):
    folder = new_folder(destination)
    sources = {
        "ball_drive_physical": "ball_drive/physical/pooled_metrics.json",
        "ball_drive_catboost": "ball_drive/catboost/pooled_metrics.csv",
        "pass_probability": "pass_probability/scores/metrics.csv",
        "pass_selection_flat": "pass_selection/flat/scores/pooled_metrics.csv",
        "pass_selection_cnl": "pass_selection/cnl/scores/pooled_metrics.csv",
        "action_selection": "binary_classification/action_selection/scores/metrics.json",
        "residual_risk": "binary_classification/residual_risk/scores/metrics.json",
        "composed_risk": "composed_risk/scores/pooled_metrics.csv",
    }
    tables, missing = [], []
    for task, name in sources.items():
        source = output_root / name
        if not source.is_file():
            missing.append(task)
            continue
        table = pd.read_csv(source) if source.suffix == ".csv" else pd.DataFrame([json.loads(source.read_text())])
        table.insert(0, "task", task)
        tables.append(table)
    if tables:
        pd.concat(tables, ignore_index=True).to_csv(folder / "performance.csv", index=False)
    parameters = []
    physical = output_root / "ball_drive/physical"
    for fit_name in [f"fold_{f}" for f in range(5)] + ["production"]:
        source = physical / fit_name / "fit.json"
        if source.is_file():
            fitted = json.loads(source.read_text())
            theta, covariance = np.asarray(fitted["theta"]), np.asarray(fitted["covariance"])
            for name, index in [("lambda_0", 0), ("s", 3)]:
                sd = np.sqrt(covariance[index,index])
                parameters.append(dict(task="ball_drive", fit=fit_name, parameter=name,
                    transformed_joint_map=np.exp(theta[index]),
                    laplace_mean=np.exp(theta[index]+.5*covariance[index,index]),
                    laplace_lower95=np.exp(theta[index]-1.95996398454*sd),
                    laplace_upper95=np.exp(theta[index]+1.95996398454*sd)))
        pass_file = output_root / "pass_probability" / (f"training/{fit_name}/models.json" if fit_name != "production" else "production/pass_risk_model.json")
        if pass_file.is_file():
            fitted = json.loads(pass_file.read_text())
            if fit_name != "production":
                fitted = fitted["fit"]
            for name in ["lambda_control", "s"]:
                parameters.append(dict(task="pass_probability", fit=fit_name, parameter=name,
                                       mle=fitted[name]))
    if parameters:
        pd.DataFrame(parameters).to_csv(folder / "physical_parameters.csv", index=False)
    write_json(folder / "coverage.json", dict(included_tasks=[k for k in sources if k not in missing],
        missing_tasks=missing, no_fallback_to_archived_runs=True,
        laplace_intervals="Conditional on selected ball-drive model/regularization; lognormal transform of Gaussian approximation."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=REVIEW_ROOT / "outputs")
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    collect_statistics(args.output_root, args.destination or args.output_root / "statistics")


if __name__ == "__main__":
    main()
