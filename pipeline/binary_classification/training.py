"""Native binary XGBoost for action selection and residual risk.

Read train_model first, then select_configuration and train_validate_test.
The same Bernoulli likelihood and native tree optimizer serve both tasks.
Only predictors and the meaning of y differ. Production fitting and metrics
come after the training/validation functions; file helpers are at the end.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

from PRESS.model.action_selection import NUMERIC as ACTION_FEATURES
from PRESS.model.residual_risk import NUMERIC as RESIDUAL_FEATURES


KEYS = ["match_id", "action_id", "frame_start"]
CONTEXT_COLUMNS = ["start_type_raw", "previous_action_raw"]
NUMERIC_FEATURES = {
    "action_selection": ACTION_FEATURES,
    "residual_risk": RESIDUAL_FEATURES,
}


# ============================================================
# 1. BERNOULLI LIKELIHOOD AND NATIVE TREE FITTING
# ============================================================

def bernoulli_loss(y, probability):
    """One NLL per state: ell_i=-y_i log(p_i)-(1-y_i) log(1-p_i).

    y, probability: arrays (S,), with S observed states. This function is
    used for reporting; XGBoost differentiates its own binary:logistic loss.
    y=1 means PASS for action selection, TERMINAL LOSS for residual risk.
    """
    p = np.clip(np.asarray(probability, float), 1e-15, 1 - 1e-15)
    y = np.asarray(y, int)
    return -y * np.log(p) - (1 - y) * np.log1p(-p)


def train_model(D_fit, parameters, *, D_val=None, max_rounds=2000, patience=100):
    """Build native binary-logistic trees, optionally selecting a prefix.

    D_fit/D_val: XGBoost DMatrix, S rows x F encoded features, with binary labels.
    Each tree updates the score f; p=sigmoid(f). Native derivatives are
    g=p-y and h=p(1-p). Splits and leaf weights use XGBoost's regularized
    quadratic objective. No custom gradient, resampling or class weighting.

    With validation: retain the prefix at minimum native validation logloss.
    Without validation: fit exactly max_rounds trees, for cross-fitting/refit.
    """
    learning_curve = {}
    evaluation_sets = [(D_fit, "fit")]
    if D_val is not None:
        evaluation_sets.append((D_val, "validation"))

    model = xgb.train(
        parameters,
        D_fit,
        num_boost_round=max_rounds,
        evals=evaluation_sets,
        evals_result=learning_curve,
        early_stopping_rounds=patience if D_val is not None else None,
        verbose_eval=False,
    )
    if D_val is None:
        return model, learning_curve

    # best_iteration is zero-based. Trees beyond this prefix only establish
    # that validation has not improved for patience=100 further rounds.
    best_tree_count = model.best_iteration + 1
    selected_model = model[:best_tree_count]
    return selected_model, learning_curve


# ============================================================
# 2. FIT-ONLY CATEGORY ENCODING AND FEATURE TABLES
# ============================================================

def fit_category_vocabulary(fitting_states):
    """Categories learned on fitting matches only; no validation/test refitting."""
    return {
        column: sorted(fitting_states[column].dropna().unique().tolist())
        + ["__missing__", "__unseen__"]
        for column in CONTEXT_COLUMNS
    }


def encode_features(states, task, vocabulary):
    """State DataFrame -> X (S,F) and ordered feature names.

    Numerical NaNs stay missing for native XGBoost routing. Categories become
    one-hot columns, with separate missing and unseen buckets. The label y
    and identifying columns never enter X.
    """
    names = list(NUMERIC_FEATURES[task])
    blocks = [states[names].to_numpy(np.float32)]
    for column, levels in vocabulary.items():
        value = states[column].astype(object)
        mapped = value.where(value.isin(levels), "__unseen__")
        mapped = mapped.where(value.notna(), "__missing__")
        blocks.append(np.column_stack([mapped.eq(level).to_numpy(np.float32) for level in levels]))
        names.extend(f"{column}={level}" for level in levels)
    X = np.column_stack(blocks).astype(np.float32)
    if np.isinf(X).any():
        raise ValueError("Infinite feature value; inspect the physical feature calculation.")
    return X, names


def build_xgboost_input(states, task, vocabulary, threads=1, *, with_labels=True):
    """Create DMatrix from predictor columns; optionally attach observed y."""
    X, names = encode_features(states, task, vocabulary)
    return xgb.DMatrix(
        X, label=states.y if with_labels else None,
        feature_names=names, nthread=threads,
    )


def load_feature_tables(output, task, split_name, matches):
    """Concatenate one state-feature parquet per requested match, in that order."""
    folder = Path(output) / task / "features" / split_name
    tables = [pd.read_parquet(folder / f"{match}.parquet") for match in matches]
    states = pd.concat(tables, ignore_index=True)
    states["match_id"] = states.match_id.astype(str)
    return states


# ============================================================
# 3. GRID SELECTION, THEN VALIDATION-FROZEN OUTER TEST
# ============================================================

def select_configuration(D_fit, D_val, fixed_parameters, configurations,
                         max_rounds=2000, patience=100, curve_folder=None):
    """Compare the full grid on ONE fixed validation pair.

    Each candidate starts a new native model. Native validation CE chooses
    its best prefix and then the winning configuration. Exact ties keep the
    first candidate, as in the accepted benchmark. Test data are not inputs.
    curve_folder optionally retains every candidate history for later audits.
    """
    trials = []
    best_ce = np.inf
    for index, configuration in enumerate(configurations):
        model, curve = train_model(
            D_fit, fixed_parameters | configuration, D_val=D_val,
            max_rounds=max_rounds, patience=patience,
        )
        # Retain each candidate history before the next candidate replaces it.
        # This is an audit artifact; it does not affect training or selection.
        if curve_folder is not None:
            write_json(Path(curve_folder) / f"learning_curve_{index:03d}.json", curve)
        best_tree_count = model.num_boosted_rounds()
        validation_ce = float(curve["validation"]["logloss"][best_tree_count - 1])
        trial = dict(
            candidate_index=index, **configuration, validation_ce=validation_ce,
            best_round=best_tree_count,
            trained_rounds=len(curve["validation"]["logloss"]),
            fit_ce_at_selection=float(curve["fit"]["logloss"][best_tree_count - 1]),
        )
        trials.append(trial)
        if validation_ce < best_ce:
            best_ce = validation_ce
            selected_model, selected_configuration, selected_curve = model, trial, curve
        print("GRID", index + 1, "/", len(configurations), validation_ce, flush=True)
    return selected_model, selected_configuration, selected_curve, pd.DataFrame(trials)


def train_validate_test(output, task, fold, protocol, threads=1):
    """Six fitting -> two fixed validation -> two held-out test matches.

    Category encoding, grid and early stopping use no test observations.
    Both retained models use native probabilities, with no post-hoc scaling.
    No eight-match tree refit is performed before the outer test.
    """
    split_name = f"fold_{fold}"
    folder = Path(output) / task / "training" / split_name
    folder.mkdir(parents=True, exist_ok=False)
    split = protocol["folds"][str(fold)]

    # 1. Fit-only vocabulary and matrices for the six/two development matches.
    fitting_states = load_feature_tables(output, task, split_name, split["fit_matches"])
    validation_states = load_feature_tables(output, task, split_name, split["validation_matches"])
    vocabulary = fit_category_vocabulary(fitting_states)
    D_fit = build_xgboost_input(fitting_states, task, vocabulary, threads)
    D_val = build_xgboost_input(validation_states, task, vocabulary, threads)

    # 2. Fix the configuration and tree prefix using native validation CE.
    parameters = protocol["fixed_parameters"] | dict(nthread=threads, seed=protocol["seed"] + fold)
    model, selection, curve, grid_scores = select_configuration(
        D_fit, D_val, parameters, protocol["grid"],
        protocol["max_rounds"], protocol["early_stopping_rounds"], curve_folder=folder,
    )
    model.save_model(folder / "model.ubj")
    write_json(folder / "selection.json", selection | split | dict(fold=fold, calibration="none"))
    write_json(folder / "feature_spec.json", dict(
        vocabulary=vocabulary, feature_names=model.feature_names,
        numeric=NUMERIC_FEATURES[task], encoding_fit_matches=split["fit_matches"],
    ))
    write_json(folder / "learning_curve.json", curve)
    grid_scores.to_csv(folder / "grid_validation.csv", index=False)

    # 3. Only now read and score the two test matches, using frozen encoding.
    test_states = load_feature_tables(output, task, split_name, split["test_matches"])
    D_test = build_xgboost_input(test_states, task, vocabulary, threads, with_labels=False)
    predictions = test_states[KEYS + ["y"]].copy()
    predictions["probability"] = model.predict(D_test)
    predictions["null_probability"] = float(fitting_states.y.mean())
    predictions["fold"] = fold
    predictions.to_parquet(folder / "test_predictions.parquet", index=False)
    save_feature_importance(model, folder)
    return predictions


# ============================================================
# 4. PRODUCTION: VALIDATION-CHOSEN CONFIGURATION AND FIXED LENGTH
# ============================================================

def select_production_configuration(output, task):
    """Lowest MEAN validation CE across five grids; no outer-test selection.

    The tree count is the median of the five best prefixes of THAT SAME
    configuration, not the median of the five fold winners' tree counts.
    """
    grids = [pd.read_csv(Path(output) / task / "training" / f"fold_{f}" / "grid_validation.csv")
             .assign(fold=f) for f in range(5)]
    scores = pd.concat(grids, ignore_index=True)
    means = scores.groupby("candidate_index", sort=True).validation_ce.mean()
    chosen_index = int(means.idxmin())
    chosen_rows = scores.loc[scores.candidate_index.eq(chosen_index)]
    configuration = {name: chosen_rows.iloc[0][name].item()
                     for name in ["max_depth", "min_child_weight", "reg_lambda", "eta"]}
    configuration["max_depth"] = int(configuration["max_depth"])
    tree_count = int(np.median(chosen_rows.best_round))
    return configuration, tree_count


def train_production_model(data_root, output, task, protocol, threads=1):
    """Fresh all-ten-match fit, fixed tree count, no early stopping/scaling."""
    folder = Path(output) / task / "production"
    folder.mkdir(parents=True, exist_ok=False)
    states = load_feature_tables(output, task, "production", protocol["matches"])
    vocabulary = fit_category_vocabulary(states)
    D_fit = build_xgboost_input(states, task, vocabulary, threads)
    configuration, tree_count = select_production_configuration(output, task)
    parameters = protocol["fixed_parameters"] | configuration | dict(nthread=threads, seed=protocol["seed"])
    model, curve = train_model(D_fit, parameters, max_rounds=tree_count)

    # Export a public-reader-compatible artifact beside the new booster.
    booster_file = task + "_model.ubj"
    model.save_model(folder / booster_file)
    artifact = dict(
        schema_version=1,
        model_type="binary_action_choice" if task == "action_selection" else "conditional_individual_loss",
        feature_contract="canonical_action_30_v1" if task == "action_selection" else "residual_action31_v1",
        model_id="review_" + task, vocabulary=vocabulary,
        feature_names=model.feature_names, booster_file=booster_file,
        booster_sha256=hashlib.sha256((folder / booster_file).read_bytes()).hexdigest(),
        calibration={"method": "none"},
        training=dict(n_actions=len(states), rounds=tree_count, parameters=parameters,
                      match_ids=protocol["matches"], role="full_data_refit"),
    )
    if task == "action_selection":
        choice = json.loads((Path(data_root) / "upstream/production/pass_selection_model.json").read_text())
        artifact["xt_grid"] = choice["xt_grid"]
    else:
        action_file = Path(output) / "action_selection/production/action_selection_model.ubj"
        artifact["upstream"] = dict(action_booster_sha256=hashlib.sha256(action_file.read_bytes()).hexdigest())
    write_json(folder / (task + "_model.json"), artifact)
    write_json(folder / "learning_curve.json", curve)
    save_feature_importance(model, folder)
    return artifact


# ============================================================
# 5. POOLED METRICS AND TOTAL SPLIT-GAIN IMPORTANCE
# ============================================================

def binary_metrics(y, probability, null_probability):
    """CE, null gain and ten equal-count RANK bins, for pooled held-out states."""
    y, p = np.asarray(y, int), np.asarray(probability, float)
    ce = float(bernoulli_loss(y, p).mean())
    null_ce = float(bernoulli_loss(y, null_probability).mean())
    bins = []
    for group in np.array_split(np.argsort(p, kind="stable"), min(10, len(p))):
        bins.append(dict(n=len(group), predicted=float(p[group].mean()), observed=float(y[group].mean())))
    calibration = pd.DataFrame(bins)
    result = dict(
        n=len(y), positives=int(y.sum()), cross_entropy=ce, null_cross_entropy=null_ce,
        gain_on_null=1 - ce / null_ce, brier=float(np.mean((p - y) ** 2)),
        ece_quantile10=float((calibration.n * abs(calibration.predicted - calibration.observed)).sum() / len(y)),
    )
    if len(np.unique(y)) == 2:
        result.update(roc_auc=float(roc_auc_score(y, p)), pr_auc=float(average_precision_score(y, p)))
    return result, calibration


def compute_pooled_metrics(data_root, output, task):
    """One held-out prediction per accepted state; metrics AFTER concatenation."""
    tables = [pd.read_parquet(Path(output) / task / "training" / f"fold_{f}" / "test_predictions.parquet")
              for f in range(5)]
    predictions = pd.concat(tables, ignore_index=True)
    expected = pd.read_parquet(Path(data_root) / ("action_rows.parquet" if task == "action_selection" else "residual_labels.parquet"))
    keys = pd.MultiIndex.from_frame(predictions[KEYS])
    if keys.has_duplicates or set(keys) != set(pd.MultiIndex.from_frame(expected[KEYS])):
        raise ValueError("Pooled tests must cover each accepted state exactly once.")
    metrics, calibration = binary_metrics(predictions.y, predictions.probability, predictions.null_probability)
    folder = Path(output) / task / "scores"
    folder.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(folder / "pooled_predictions.parquet", index=False)
    calibration.to_csv(folder / "calibration.csv", index=False)
    write_json(folder / "metrics.json", metrics)
    return metrics


def save_feature_importance(model, folder):
    """Sum each feature's split gains across ALL trees, then normalize.

    G_f=sum_{splits on f} gain(split). Share_f=G_f/sum_k G_k.
    XGBoost total_gain implements the sum; gain alone would be the mean per
    split. One-hot columns are also summed under their original variable.
    """
    gain = model.get_score(importance_type="total_gain")
    table = pd.DataFrame(dict(feature=model.feature_names, total_gain=[gain.get(f, 0.) for f in model.feature_names]))
    total = table.total_gain.sum()
    table["share"] = table.total_gain / total if total else 0.
    table.to_csv(Path(folder) / "feature_importance_encoded.csv", index=False)
    grouped = table.assign(feature=table.feature.str.split("=").str[0]).groupby("feature", sort=False).total_gain.sum().reset_index()
    grouped["share"] = grouped.total_gain / total if total else 0.
    grouped.sort_values("share", ascending=False).to_csv(Path(folder) / "feature_importance.csv", index=False)


def write_json(path, content):
    """Save small summaries; arrays and state tables stay in parquet/UBJ files."""
    def convert(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    Path(path).write_text(json.dumps(content, default=convert, indent=2, allow_nan=False) + "\n")
