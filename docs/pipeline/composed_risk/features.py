"""07. Compose h and xLoss from matching held-out fold predictions.

No additional fit is performed. The observed action/outcome is a label only:
it never chooses a branch of the predicted mixture.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.io import ACTION_KEYS, new_folder, read_table
from pipeline.binary_classification.training import build_xgboost_input


def compose_probabilities(p_pass, drive_loss, pass_failure, residual_loss):
    """Four aligned state arrays -> current-action h and individual xLoss.

    q=P(pass|state), d=P(loss in the next carry), f=expected pass failure,
    r=P(eventual individual loss|successful next carry,state).
    h=q*f+(1-q)*d; xLoss=h+(1-q)*(1-d)*r.
    Residual r has no role after a completed pass: that individual possession
    has ended. No event from the actual future is a predictor here.
    """
    q, d, f, r = np.broadcast_arrays(*[np.asarray(v, float) for v in
                                      [p_pass, drive_loss, pass_failure, residual_loss]])
    for probability in [q, d, f, r]:
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError("Component probabilities must be finite and in [0,1].")
    h = q*f + (1-q)*d
    xloss = h + (1-q)*(1-d)*r
    return h, xloss


def add_null_probabilities(states, splits):
    """Each outer test uses target prevalence in its SIX fitting matches.

    Both nulls use the full verified-outcome cohort, even when bands or a
    retrospective sensitivity subsequently restrict the evaluated rows.
    No prevalence is estimated from the held-out target of the scored match.
    """
    states = states.copy()
    for target, column in [("h_loss", "null_h"), ("terminal_loss", "null_xloss")]:
        states[column] = np.nan
        for fold, split in splits.items():
            fitting = states.match_id.isin(split["fit_matches"])
            states.loc[states.fold.eq(int(fold)), column] = states.loc[fitting, target].mean()
    return states


def predict_with_fold_model(feature_table, binary_root, task, fold, threads):
    """Use saved native tree prefix and the SAME fold's fitting vocabulary."""
    folder = Path(binary_root) / task / "training" / f"fold_{fold}"
    spec = json.loads((folder / "feature_spec.json").read_text())
    model = xgb.Booster()
    model.load_model(folder / "model.ubj")
    model.set_param({"nthread": threads})
    D = build_xgboost_input(feature_table, task, spec["vocabulary"], threads, with_labels=False)
    return model.predict(D)


def compute_held_out(input_root, binary_root, output, threads=1):
    """Read action features once; evaluate both native classifiers on outer tests.

    input_root/labels.parquet contains only observed labels and metadata for
    the verified canonical cohort. It is joined by (match,action,frame), never
    by row position. Upstream physical/choice features already belong to the
    same six-fit/two-validation stack as the two binary classifiers.
    """
    input_root, binary_root = Path(input_root), Path(binary_root)
    protocol = json.loads((input_root / "protocol.json").read_text())
    labels = read_table(input_root / "labels.parquet")
    records = []
    for fold, split in protocol["folds"].items():
        tables = [read_table(binary_root / "action_selection/features" / f"fold_{fold}" / f"{m}.parquet")
                  for m in split["test_matches"]]
        feature_table = pd.concat(tables, ignore_index=True)
        # Keep only the verified parents, but do not remove foul/clearance
        # parents from overall h/xLoss metrics. Terminal coverage is for bands.
        observed = labels.loc[labels.match_id.isin(split["test_matches"])]
        features = feature_table.merge(observed, on=ACTION_KEYS, how="inner",
                                       validate="one_to_one", suffixes=("", "_label"))
        q = predict_with_fold_model(features, binary_root, "action_selection", fold, threads)
        features["action_carry_probability"] = 1-q
        r = predict_with_fold_model(features, binary_root, "residual_risk", fold, threads)
        d, f = 1-features.drive_success_1s.to_numpy(), 1-features.pass_success_mean.to_numpy()
        # Match the merged feature order explicitly before attaching predictions.
        selected = features[ACTION_KEYS].merge(observed, on=ACTION_KEYS, validate="one_to_one")
        selected["p_pass"], selected["drive_loss"], selected["pass_failure"] = q, d, f
        selected["residual_loss_probability"] = r
        selected["h"], selected["xloss"] = compose_probabilities(q, d, f, r)
        selected["fold"] = int(fold)
        records.append(selected)
    states = pd.concat(records, ignore_index=True)
    if states.duplicated(ACTION_KEYS).any() or len(states) != len(labels):
        raise ValueError("Every verified canonical state needs exactly one held-out prediction.")
    states = add_null_probabilities(states, protocol["folds"])
    destination = new_folder(Path(output) / "features")
    states.to_parquet(destination / "held_out.parquet", index=False)
    return states


def replay_accepted_predictions(input_root, output):
    """Recompute mixtures from the accepted component predictions, without fits.

    Explicit replay mode for verifying this refactor. It never claims that
    archived component predictions were generated by a fresh training run.
    """
    input_root = Path(input_root)
    states = read_table(input_root / "reference/components.parquet")
    labels = read_table(input_root / "labels.parquet")
    states = states.merge(labels, on=ACTION_KEYS, validate="one_to_one")
    states["h"], states["xloss"] = compose_probabilities(
        states.p_pass, states.drive_loss, states.pass_failure, states.residual_loss_probability)
    protocol = json.loads((input_root / "protocol.json").read_text())
    states = add_null_probabilities(states, protocol["folds"])
    destination = new_folder(Path(output) / "features")
    states.to_parquet(destination / "held_out.parquet", index=False)
    return states
