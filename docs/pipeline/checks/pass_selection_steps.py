"""Local pass-selection checks on ten real passes; never the full benchmark.

Features -> three aligned tables -> probabilities/derivatives -> screening ->
tree update/rho -> early stopping -> temperature -> held-out scores -> export.
Only the first two development configurations and <=22 trees are used.
The complete candidate grids are retained; candidate cells are not subsampled.
"""

import argparse
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline/pass_selection"))
import features
import training
from PRESS.model.pass_selection import build_choice_features
from PRESS.model import PassSelectionModel
from pipeline.io import new_folder, write_json


def build_real_feature_tables(inputs, output):
    """Evaluate the reviewed features and compare with the package evaluator."""
    protocol = json.loads((inputs / "protocol.json").read_text())
    cohort = pd.read_parquet(inputs / "rows.parquet")
    cohort["match_id"] = cohort.match_id.astype(str)
    split = protocol["folds"]["0"]
    vocabulary = features.fit_category_vocabulary(cohort, split["fit_matches"])
    names = features.feature_names(vocabulary)
    folder = new_folder(output / "features")
    write_json(folder / "feature_spec.json", dict(names=names, vocabulary=vocabulary))
    xt = np.load(inputs / "smoke/xt.npy")
    for match in protocol["matches"]:
        source = inputs / "smoke" / match
        rows = pd.read_parquet(source / "rows.parquet")
        targets = np.load(source / "targets.npy")
        raw = np.load(source / "raw_control.npy")
        with np.load(source / "state.npz") as arrays:
            state = {k: arrays[k][0] for k in arrays.files}
        state.update(v_bc=state["p_vel"][10], r_atts=state["p_pos"][:10], r_defs=state["p_pos"][11:])
        row = rows.iloc[0]
        values = features.build_features(state, targets, raw[0], row.drive_loss_fold_0, xt, vocabulary, row)
        expected = build_choice_features(
            x_bc=state["x_bc"], v_bc=state["v_bc"], r_atts=state["r_atts"], r_defs=state["r_defs"],
            targets=targets, attacking_side=row.attacking_side, target_pitch_control=raw[0],
            ball_drive_prob=row.drive_loss_fold_0, xt_grid=xt, vocabulary=vocabulary,
            start_type_raw=row.start_type_raw, previous_action_raw=row.previous_action_raw,
            match_clock_min=row.match_clock_min, score_difference_prefix=row.score_difference_prefix,
            pitch_length=row.pitch_length, pitch_width=row.pitch_width,
            pressure_lines=(state["opp_lines"], state["team_lines"]))
        np.testing.assert_allclose(values, expected, atol=2e-6, rtol=0, equal_nan=True)
        selected = features.observed_destination_cell(rows.x_target.to_numpy(), rows.y_target.to_numpy(),
                                                     row.pitch_length, row.pitch_width)
        np.testing.assert_array_equal(selected, rows.chosen_cell)
        rows["n_candidates"] = len(targets)
        target = folder / match
        target.mkdir()
        rows.to_parquet(target / "rows.parquet", index=False)
        np.save(target / "features.npy", values[None])
        np.save(target / "log_memberships.npy", np.load(source / "log_memberships.npy"))
    write_json(folder / "complete.json", dict(scope="ten real passes; verification only"))
    return folder, protocol


def check_math(folder, split):
    """Autodiff verifies gradients; curvature is the declared boosting rule."""
    X, observed, memberships = features.load_pass_tables(folder, split["fit_matches"], "cnl")
    assert memberships.index.equals(X.index)
    first = observed.iloc[0]
    C = int(first.n_candidates)
    alpha = jnp.asarray(memberships.iloc[:C].to_numpy()[None])
    u = jnp.linspace(-1., 1., C)[None]
    y = jnp.array([int(first.chosen_cell)])
    flat = training.flat_log_probabilities(u)
    nested = training.cnl_log_probabilities(u, alpha, 1.)
    np.testing.assert_allclose(flat, nested, atol=1e-12, rtol=0)
    for rho in [1., 1.7, 3.]:
        log_p = training.cnl_log_probabilities(u, alpha, rho)
        np.testing.assert_allclose(np.exp(log_p).sum(axis=1), 1., atol=1e-12)
        loss, gradient, curvature = training.cnl_derivatives(u, alpha, y, rho)
        exact = jax.grad(lambda v: -training.cnl_log_probabilities(v, alpha, rho)[0,y[0]])(u)
        np.testing.assert_allclose(gradient, exact, atol=1e-11, rtol=0)
        assert np.isfinite(curvature).all()  # CNL reference curvature is not an exact Hessian.
    _, g, h = training.flat_derivatives(u, y)
    p = np.exp(flat)
    np.testing.assert_allclose(h, np.maximum(p*(1-p)*C/(C-1), 1e-6), atol=1e-12)
    np.testing.assert_allclose(g.sum(axis=1), 0., atol=1e-12)


def check_optimization(folder, protocol, output):
    """Short fits exercise the real native XGBoost optimization and artifacts."""
    split = protocol["folds"]["0"]
    report = {}
    for family in ["flat", "cnl"]:
        X, observed, membership = features.load_pass_tables(folder, split["fit_matches"], family)
        Xv, validation, mv = features.load_pass_tables(folder, split["validation_matches"], family)
        D, Dv = training.build_xgboost_input(X, observed), training.build_xgboost_input(Xv)
        parameters = training.booster_parameters(family, training.SEED, 1, dict(min_child_weight=.1))
        # Fixed-budget fit reaches tree 20 and therefore the rho update.
        model, summary, history, updates = training.train_model(
            D, observed, parameters, memberships_fit=membership, max_rounds=22,
            conditional_multistart=False)
        if family == "cnl":
            assert len(updates) == 1 and updates[0]["after_tree"] == 20
            step = updates[0]
            expected_rho = step["previous_rho"]+.1*(step["candidate_rho"]-step["previous_rho"])
            assert abs(step["rho"]-expected_rho) < 1e-12 or not step["accepted"]
        # Early stopping must restore BOTH the selected tree prefix and rho.
        _, selected, curve, _ = training.train_model(D, observed, parameters,
            D_val=Dv, observed_val=validation, memberships_fit=membership, memberships_val=mv,
            max_rounds=22, patience=3, conditional_multistart=False)
        best = min(curve, key=lambda point: point["validation_ce"])
        assert selected["rounds"] == best["round"] and selected["rho"] == best["rho"]
        # Reduced screening does not evaluate test or fit temperature.
        training.compare_configurations(folder, split, family, output / "screening", threads=1,
            configuration_name=training.SCREENING_CONFIGURATIONS[0]["name"], tree_limit=2)
        scored = training.train_validate_test(folder, split, family, 0, output / "folds", threads=1,
            max_rounds=22, configuration=dict(min_child_weight=.1))
        saved = output / "folds" / family / "training/fold_0"
        reloaded = xgb.Booster()
        reloaded.load_model(saved / "selected_model.ubj")
        assert reloaded.num_boosted_rounds() == json.loads((saved / "selection.json").read_text())["rounds"]
        calibration = json.loads((saved / "temperature.json").read_text())
        assert calibration["validation_ce_after"] <= calibration["validation_ce_before"] + 1e-10
        training.compute_metrics(scored)
        # Exercise production export/reload without fabricating five full fits.
        specification = json.loads((folder / "feature_spec.json").read_text())
        artifact = training.save_production_model(output / "exports" / family, model, summary,
            calibration, observed, specification, ROOT / "pipeline/inputs/pass_selection", protocol["matches"], family)
        assert artifact["training"]["initial_utilities"] == 0.
        filename = "pass_selection_model.json" if family == "flat" else "pass_selection_cnl_model.json"
        loaded = PassSelectionModel.from_json(output / "exports" / family / filename)
        np.testing.assert_array_equal(loaded.booster.predict(Dv,output_margin=True), model.predict(Dv,output_margin=True))
        # Run the OTHER four folds with the same tiny interface fixture, so
        # pooling and the actual median-length/OOF-temperature refit functions
        # execute too. Frozen fold-0 physics are only smoke-test covariates;
        # none of these short-fit scores is a scientific fold benchmark.
        for fold in range(1,5):
            training.train_validate_test(folder,protocol["folds"][str(fold)],family,fold,output/"folds",
                threads=1,max_rounds=22,configuration=dict(min_child_weight=.1))
        smoke_cohort=pd.concat([pd.read_parquet(folder/match/"rows.parquet") for match in protocol["matches"]],ignore_index=True)
        training.compute_pooled_metrics(output/"folds",family,smoke_cohort)
        production=training.train_production_model(folder,ROOT/"pipeline/inputs/pass_selection",output/"folds",family,threads=1)
        selected_counts=[json.loads((output/"folds"/family/"training"/f"fold_{f}"/"selection.json").read_text())["rounds"] for f in range(5)]
        assert production["training"]["rounds"]==int(np.median(selected_counts))
        production_model=PassSelectionModel.from_json(output/"folds"/family/"production"/filename)
        assert production_model.booster.num_boosted_rounds()==int(np.median(selected_counts))
        report[family] = dict(n_fit_passes=len(observed), fixed_trees=22, rho_updates=len(updates),
            selected_trees=selected["rounds"], saved_test_passes=len(scored), export_reloaded=True,
            short_outer_folds=5,pooled_test_passes=10,production_trees=int(np.median(selected_counts)),
            production_oof_temperature_checked=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    folder, protocol = build_real_feature_tables(ROOT / "pipeline/inputs/pass_selection", args.output)
    check_math(folder, protocol["folds"]["0"])
    report = check_optimization(folder, protocol, args.output)
    write_json(args.output / "verification.json", dict(scope="Local smoke checks, not benchmark scores", **report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
