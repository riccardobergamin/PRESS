"""Connect fitted pipeline stages without using archived model objects.

The preprocessing stage creates observation tables. This module supplies its
fixed protocols and exports fitted upstream models from completed training
stages. Every fold is checked against the declared six/two/two match split.
Existing input files are never replaced.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


PIPELINE_ROOT = Path(__file__).resolve().parent
PROTOCOL_TASKS = ("expected_threat", "pass_selection", "binary_classification", "composed_risk")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def copy_new(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def install_protocols(data_root, protocol_root=PIPELINE_ROOT / "inputs"):
    """Add fixed split/grid protocols to tables regenerated from raw matches."""
    data_root, protocol_root = Path(data_root), Path(protocol_root)
    if not (data_root / "ball_drive/rows.parquet").is_file():
        raise FileNotFoundError("Run preprocessing before installing protocols.")
    for task in PROTOCOL_TASKS:
        copy_new(protocol_root / task / "protocol.json", data_root / task / "protocol.json")
    copy_new(protocol_root / "folds.json", data_root / "folds.json")
    copy_new(protocol_root / "reports/cases.json", data_root / "reports/cases.json")


def declared_folds(data_root):
    """The choice and binary classifiers must use identical outer splits."""
    data_root = Path(data_root)
    choice = read_json(data_root / "pass_selection/protocol.json")["folds"]
    binary = read_json(data_root / "binary_classification/protocol.json")["folds"]
    if choice != binary or set(choice) != {str(i) for i in range(5)}:
        raise ValueError("Pass-selection and binary fold protocols disagree.")
    for fold, split in choice.items():
        fitting = set(split["fit_matches"])
        validation = set(split["validation_matches"])
        test = set(split["test_matches"])
        if len(fitting) != 6 or len(validation) != 2 or len(test) != 2:
            raise ValueError(f"Fold {fold} does not have a 6+2+2 split.")
        if fitting & validation or fitting & test or validation & test:
            raise ValueError(f"Fold {fold} has overlapping match roles.")
    return choice


def six_match_drive_model(fold_root, split, fold):
    """Export the selected inner MAP whose training matches equal this fold's six."""
    fold_root = Path(fold_root)
    outer_model = read_json(fold_root / "model.json")
    selected = outer_model["candidate"]
    designs = read_json(fold_root / "inner_designs.json")
    fits = read_json(fold_root / "inner_fits.json")
    matching_designs = [item for item in designs
                        if set(item["fitting_matches"]) == set(split["fit_matches"])
                        and set(item["validation_matches"]) == set(split["validation_matches"])]
    if len(matching_designs) != 1:
        raise ValueError(f"Fold {fold}: no unique physical six-match design.")
    design = matching_designs[0]
    matching_fits = [item["fit"] for item in fits
                     if item["inner"] == design["inner"] and item["candidate"] == selected["name"]]
    if len(matching_fits) != 1:
        raise ValueError(f"Fold {fold}: no unique physical fit for {selected['name']}.")
    fitted = matching_fits[0]
    bases = design["design"]
    model = deepcopy(outer_model)
    model["model_id"] = f"ball_drive_six_match_fold_{fold}"
    model["theta"] = fitted["theta"]
    model["design"]["speed_basis"] = bases["speed"]
    model["design"]["boundary_basis"] = bases["boundary"]
    model["design"]["penalty"] = bases["speed"]["penalty"] + bases["boundary"]["penalty"]
    model["angle_design"] = bases["angle"]
    model["training"]["match_ids"] = list(split["fit_matches"])
    model["training"]["role"] = "six_match_feature_fit"
    model["fit_diagnostics"] = {name: fitted[name]
                                for name in ("objective", "gradient_max", "converged", "laplace_valid")}
    model["posterior_covariance"] = fitted["covariance"]
    return model


def six_match_pass_models(fold_root, production_model, split, fold):
    """Return raw and two-match-calibrated versions of one six-match MLE."""
    fold_root = Path(fold_root)
    internal = read_json(fold_root / "models.json")["internal"]
    matches = [fit for fit in internal
               if set(fit["fit_matches"]) == set(split["fit_matches"])
               and set(fit["calibration_matches"]) == set(split["validation_matches"])]
    if len(matches) != 1:
        raise ValueError(f"Fold {fold}: no unique pass-probability six-match fit.")
    fitted = matches[0]
    raw = deepcopy(production_model)
    raw["model_id"] = f"pass_probability_six_match_fold_{fold}"
    raw["lambda_control"] = fitted["lambda_control"]
    raw["s"] = fitted["s"]
    raw["fit"] = fitted
    raw["trajectory_boundary"] = fitted["boundary"]
    raw["training"] = dict(role="six_match_feature_fit", match_ids=list(split["fit_matches"]))
    raw["calibration"] = dict(method="isotonic", x=[0.0, 1.0], y=[0.0, 1.0],
                              role="Raw pitch control for pass-selection features")
    raw.pop("calibration_reference", None)
    raw.pop("provenance", None)

    calibration_rows = pd.read_parquet(fold_root / "inner_calibration.parquet")
    calibration_rows = calibration_rows.loc[
        calibration_rows.match_id.astype(str).isin(split["validation_matches"])
    ]
    if set(calibration_rows.match_id.astype(str)) != set(split["validation_matches"]):
        raise ValueError(f"Fold {fold}: incomplete pass-probability validation pair.")
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(
        calibration_rows.raw.to_numpy(), calibration_rows.outcome.to_numpy(),
    )
    calibrated = deepcopy(raw)
    calibrated["calibration"] = dict(
        method="isotonic", x=isotonic.X_thresholds_.tolist(),
        y=isotonic.y_thresholds_.tolist(), role="Two validation matches; physical fit held fixed",
        match_ids=list(split["validation_matches"]),
    )
    return fitted, raw, calibrated


def install_physical_upstream(data_root, output_root):
    """Feed six-match physical fits and xT maps to the two choice tasks."""
    data_root, output_root = Path(data_root), Path(output_root)
    splits = declared_folds(data_root)
    pass_production = read_json(output_root / "pass_probability/production/pass_risk_model.json")
    for fold in range(5):
        split = splits[str(fold)]
        physical = output_root / "ball_drive/physical" / f"fold_{fold}"
        passes = output_root / "pass_probability/training" / f"fold_{fold}"
        xt = output_root / "expected_threat/feature_maps" / f"fold_{fold}"
        drive_model = six_match_drive_model(physical, split, fold)
        fitted_pass, raw_pass, calibrated_pass = six_match_pass_models(
            passes, pass_production, split, fold,
        )
        xt_manifest = read_json(xt / "manifest.json")
        if set(xt_manifest["fitting_matches"]) != set(split["fit_matches"]):
            raise ValueError(f"Fold {fold}: xT map uses the wrong matches.")
        for task, pass_model in (("pass_selection", raw_pass),
                                 ("binary_classification", calibrated_pass)):
            destination = data_root / task / "upstream" / f"fold_{fold}"
            write_json(destination / "ball_drive_model.json", drive_model)
            write_json(destination / "pass_risk_model.json", pass_model)
            copy_new(xt / "xt.npy", destination / "xt.npy")
            copy_new(xt / "manifest.json", destination / "xt_provenance.json")
            if task == "pass_selection":
                write_json(destination / "pass_fit.json", fitted_pass)

    production_matches = read_json(data_root / "pass_selection/protocol.json")["matches"]
    if not pass_production.get("fit", {}).get("log_parameters"):
        raise ValueError("Production pass-probability export lacks its physical fit.")
    for task in ("pass_selection", "binary_classification"):
        destination = data_root / task / "upstream/production"
        copy_new(output_root / "ball_drive/physical/production/ball_drive_model.json",
                 destination / "ball_drive_model.json")
        if task == "pass_selection":
            # The choice feature builder checks that the fitted pass physics
            # uses all ten matches. The physical export itself has no match
            # provenance field, so the handoff supplies the declared cohort.
            production_for_choice = deepcopy(pass_production)
            production_for_choice["provenance"] = dict(matches=production_matches,
                                                       role="full_data_feature_fit")
            write_json(destination / "pass_risk_model.json", production_for_choice)
        else:
            copy_new(output_root / "pass_probability/production/pass_risk_model.json",
                     destination / "pass_risk_model.json")
        copy_new(output_root / "expected_threat/production/xt.npy", destination / "xt.npy")
        copy_new(output_root / "expected_threat/production/manifest.json",
                 destination / "xt_manifest.json")


def install_choice_upstream(data_root, output_root):
    """Export the fitted flat choice model for each binary-classifier split."""
    data_root, output_root = Path(data_root), Path(output_root)
    splits = declared_folds(data_root)
    production = output_root / "pass_selection/flat/production"
    template = read_json(production / "pass_selection_model.json")
    for fold in range(5):
        split = splits[str(fold)]
        fitted = output_root / "pass_selection/flat/training" / f"fold_{fold}"
        specification = read_json(output_root / "pass_selection/features" / f"fold_{fold}" / "feature_spec.json")
        selection = read_json(fitted / "selection.json")
        calibration = read_json(fitted / "temperature.json")
        if set(selection["fit_matches"]) != set(split["fit_matches"]):
            raise ValueError(f"Fold {fold}: choice fit uses the wrong matches.")
        if set(calibration["fit_matches"]) != set(split["validation_matches"]):
            raise ValueError(f"Fold {fold}: choice temperature uses the wrong matches.")
        destination = data_root / "binary_classification/upstream" / f"fold_{fold}"
        booster = fitted / "selected_model.ubj"
        artifact = deepcopy(template)
        artifact["model_id"] = f"pass_selection_six_match_fold_{fold}"
        artifact["booster_file"] = "pass_selection_model.ubj"
        artifact["booster_sha256"] = hashlib.sha256(booster.read_bytes()).hexdigest()
        artifact["rho"], artifact["mu"] = selection["rho"], selection["mu"]
        artifact["feature_names"] = specification["names"]
        artifact["vocabulary"] = specification["vocabulary"]
        artifact["xt_grid"] = np.load(destination / "xt.npy").tolist()
        artifact["calibration"] = calibration
        artifact["training"] = selection | dict(role="six_match_feature_fit",
                                               match_ids=list(split["fit_matches"]))
        artifact["physics"] = dict(
            pass_risk=read_json(destination / "pass_risk_model.json"),
            ball_drive=read_json(destination / "ball_drive_model.json"),
        )
        artifact.pop("benchmark", None)
        copy_new(booster, destination / "pass_selection_model.ubj")
        write_json(destination / "pass_selection_model.json", artifact)

    destination = data_root / "binary_classification/upstream/production"
    copy_new(production / "pass_selection_model.ubj", destination / "pass_selection_model.ubj")
    copy_new(production / "pass_selection_model.json", destination / "pass_selection_model.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("protocols", "physical", "choice"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.stage == "protocols":
        install_protocols(args.data_root)
    elif args.output_root is None:
        parser.error("physical and choice stages require --output-root")
    elif args.stage == "physical":
        install_physical_upstream(args.data_root, args.output_root)
    else:
        install_choice_upstream(args.data_root, args.output_root)


if __name__ == "__main__":
    main()
