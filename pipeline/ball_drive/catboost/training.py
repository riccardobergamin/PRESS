"""Native CatBoost fitting, full validation grid, calibration and pooled scores.

Only the fit indices can be resampled. Every validation and test row keeps
its natural prevalence. No custom CatBoost gradient or Hessian is used.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score

from pipeline.ball_drive.catboost.features import (
    PARAMETERS, CANDIDATES, TREE_CHECKPOINTS, SEEDS, build_features, bernoulli_nll,
)


# ============================================================
# NATIVE BERNOULLI BOOSTING STEP (SHARED BY ALL FITS BELOW)
# ============================================================

def train_model(X_fit, y_fit, configuration, seed, threads):
    """X_fit=(S,F), y_fit=(S,): fit the native CatBoost binary classifier.

    p=sigmoid(f); Logloss=-sum[y*log(p)+(1-y)*log(1-p)]. CatBoost computes
    its derivatives and builds symmetric trees internally. configuration
    fixes depth, learning rate, L2 and (after selection) number of trees.
    The caller, not this function, selects any training-only resampled rows.
    No validation/test rows are passed to this optimizer.
    """
    from catboost import CatBoostClassifier

    parameters = PARAMETERS | configuration | dict(thread_count=threads, random_seed=seed)
    model = CatBoostClassifier(**parameters)
    model.fit(X_fit, y_fit)
    return model


# ============================================================
# 1. TRAINING-ONLY SAMPLING AND ANALYTIC PRIOR CORRECTION
# ============================================================

def select_training_rows(indices, loss, sampling, seed):
    """Return fitting indices only: natural, or all losses + nine successes/loss.

    Successes are sampled without replacement. Validation/test indices never
    enter this function. enrichment_ratio is the inverse success inclusion
    probability; it is used for analytic prior correction, not a class weight.
    """
    indices = np.asarray(indices, int)
    failed = indices[loss[indices] == 1]
    successful = indices[loss[indices] == 0]
    if sampling == "natural":
        keep = indices.copy()
    elif sampling == "loss10":
        n_success = min(len(successful), 9 * len(failed))
        rng = np.random.default_rng(seed)
        selected = rng.choice(successful, n_success, replace=False)
        keep = np.sort(np.r_[failed, selected])
    else:
        raise ValueError("Choose natural or loss10.")
    kept_success = int((loss[keep] == 0).sum())
    ratio = len(successful) / kept_success
    return keep, dict(
        before_n=len(indices), after_n=len(keep), losses=len(failed),
        enrichment_ratio=ratio, success_inclusion=kept_success / len(successful),
    )


def correct_sampling_prior(probability, enrichment_ratio):
    """p_natural=sigmoid(logit(p_sampled)-log(enrichment_ratio))."""
    probability = np.clip(np.asarray(probability, float), 1e-15, 1 - 1e-15)
    return expit(logit(probability) - np.log(enrichment_ratio))


# ============================================================
# 2. COMPLETE GRID: 36 CONFIGURATIONS x 4 INNER 6+2 SPLITS
# ============================================================

def select_configuration(rows, spatial_features, tti, valid_defenders, physical_splits, sampling, fold, threads, folder):
    """36 configurations x four 6+2 rotations within eight development matches.

    For each candidate: fit 3,000 trees, evaluate prefixes every 25 trees, and
    pool validation NLL across the four pairs. For loss10 selection uses
    analytic prior-corrected predictions. No Platt/isotonic fit or test scores
    choose the configuration/tree count here.
    """
    loss = rows.loss_observed.to_numpy(int)
    totals = np.zeros((len(CANDIDATES), len(TREE_CHECKPOINTS)))
    audit = []
    n_validation = sum(int(rows.match_id.isin(split["validation_matches"]).sum()) for split in physical_splits["inner"])
    for candidate_index, candidate in enumerate(CANDIDATES):
        for inner, split in enumerate(physical_splits["inner"]):
            features = build_features(spatial_features, tti, valid_defenders, split["s"])
            fitting = np.flatnonzero(rows.match_id.isin(split["fit_matches"]))
            validation = np.flatnonzero(rows.match_id.isin(split["validation_matches"]))
            seed = 19 + 10000 * fold + 100 * inner
            selected, sampling_info = select_training_rows(fitting, loss, sampling, seed)
            model = train_model(features.iloc[selected], loss[selected], candidate, seed, threads)

            count = 0
            for index, probabilities in enumerate(model.staged_predict_proba(
                features.iloc[validation], ntree_start=0,
                ntree_end=PARAMETERS["iterations"], eval_period=25, thread_count=threads,
            )):
                corrected = correct_sampling_prior(probabilities[:, 1], sampling_info["enrichment_ratio"])
                totals[candidate_index, index] += bernoulli_nll(loss[validation], corrected).sum()
                count += 1
            if count != len(TREE_CHECKPOINTS):
                raise ValueError("Incomplete staged-validation curve.")
            audit.append(dict(candidate=candidate_index, inner=inner, seed=seed, sampling=sampling_info))
        print("CATBOOST_SEARCH", sampling, fold, candidate_index + 1, len(CANDIDATES), flush=True)

    # Deterministic first minimum, over the pooled natural validation outcomes.
    candidate_index, checkpoint = np.unravel_index(np.argmin(totals / n_validation), totals.shape)
    records = []
    for i, candidate in enumerate(CANDIDATES):
        for j, trees in enumerate(TREE_CHECKPOINTS):
            records.append(dict(
                candidate=i, **candidate, trees=trees,
                validation_ce=float(totals[i, j] / n_validation),
            ))
    pd.DataFrame(records).to_csv(folder / "search_scores.csv", index=False)
    write_json(folder / "search_audit.json", audit)
    return dict(CANDIDATES[candidate_index], iterations=TREE_CHECKPOINTS[checkpoint])


# ============================================================
# 3. SELECTED MODEL: THREE SEEDS, INNER OOF CALIBRATION, OUTER 8+2
# ============================================================

def train_validate_test(rows, spatial_features, tti, valid_defenders, physical_splits, sampling, fold, threads, output):
    """Complete outer fold: grid -> inner OOF calibration -> eight-match refit -> test.

    rows: accepted state/label DataFrame; spatial_features: spatial features DataFrame;
    tti=(S,11 times,11 defenders), valid_defenders=(S,11 defenders). physical_splits contains
    four inner-six physical s values and outer-eight s. Targets are loss=1.
    Three native fits (seeds 19/41/73) are averaged, with natural test prevalence.
    """
    import catboost

    if catboost.__version__ != "1.2.10":
        raise RuntimeError("The accepted benchmark used CatBoost 1.2.10.")
    folder = Path(output) / sampling / f"fold_{fold}"
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    loss = rows.loss_observed.to_numpy(int)
    test = np.flatnonzero(rows.fold.eq(fold))
    training = np.flatnonzero(rows.fold.ne(fold))
    test_matches = set(rows.iloc[test].match_id)
    for split in physical_splits["inner"]:
        fit_matches = set(split["fit_matches"])
        val_matches = set(split["validation_matches"])
        if len(fit_matches) != 6 or len(val_matches) != 2:
            raise ValueError("Expected inner six-fit/two-validation split.")
        if fit_matches & val_matches or test_matches & (fit_matches | val_matches):
            raise ValueError("Match leakage across fitting, validation or test.")
        if fit_matches | val_matches != set(rows.iloc[training].match_id):
            raise ValueError("Inner split does not cover the eight development matches.")

    chosen = select_configuration(
        rows, spatial_features, tti, valid_defenders, physical_splits, sampling, fold, threads, folder,
    )
    # Configuration and tree count are fixed before any test prediction.
    write_json(folder / "selection.json", chosen)
    outer_features = build_features(spatial_features, tti, valid_defenders, physical_splits["outer_s"])
    calibration_scores, raw_test, corrected_test = [], [], []
    importance, audits, sampled_indices = [], [], {}

    # For each seed, obtain four disjoint validation-pair predictions covering
    # the eight development matches, then fit a fresh eight-match model.
    for seed in SEEDS:
        inner_prediction = np.full(len(rows), np.nan)
        for inner, split in enumerate(physical_splits["inner"]):
            features = build_features(spatial_features, tti, valid_defenders, split["s"])
            fitting = np.flatnonzero(rows.match_id.isin(split["fit_matches"]))
            validation = np.flatnonzero(rows.match_id.isin(split["validation_matches"]))
            fit_seed = seed + 10000 * fold + 100 * inner
            selected, sampling_info = select_training_rows(fitting, loss, sampling, fit_seed)
            model = train_model(features.iloc[selected], loss[selected], chosen, fit_seed, threads)
            inner_prediction[validation] = model.predict_proba(features.iloc[validation])[:, 1]
            sampled_indices[f"inner_{inner}_seed_{seed}"] = selected
            audits.append(dict(seed=seed, inner=inner, sampling=sampling_info, physical_s=split["s"]))

        if not np.isfinite(inner_prediction[training]).all() or np.isfinite(inner_prediction[test]).any():
            raise ValueError("Invalid inner calibration predictions.")
        calibration_scores.append(inner_prediction[training])

        fit_seed = seed + 10000 * fold + 900
        selected, sampling_info = select_training_rows(training, loss, sampling, fit_seed)
        model = train_model(outer_features.iloc[selected], loss[selected], chosen, fit_seed, threads)
        model.save_model(str(folder / f"seed_{seed}.cbm"))
        probability = model.predict_proba(outer_features.iloc[test])[:, 1]
        raw_test.append(probability)
        corrected_test.append(correct_sampling_prior(probability, sampling_info["enrichment_ratio"]))
        importance.append(model.get_feature_importance())
        sampled_indices[f"outer_seed_{seed}"] = selected
        audits.append(dict(seed=seed, outer_refit=True, sampling=sampling_info, physical_s=physical_splits["outer_s"]))

    # Mean the three seeds BEFORE fitting calibration on inner OOF scores.
    # Both calibrators use natural-label inner OOF predictions only.
    inner_raw = np.mean(calibration_scores, axis=0)
    platt = LogisticRegression(C=1000.0, max_iter=2000, tol=1e-9)
    platt.fit(logit(np.clip(inner_raw, 1e-12, 1 - 1e-12))[:, None], loss[training])
    isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    isotonic.fit(inner_raw, loss[training])

    # Freeze calibrators and evaluate the same untouched test states in four
    # forms: native, analytic correction, Platt and isotonic. No test tuning.
    probability = np.mean(raw_test, axis=0)
    prediction = rows.iloc[test].copy()
    prediction["p_raw"] = probability
    prediction["p_prior_corrected"] = np.mean(corrected_test, axis=0)
    prediction["p_platt"] = platt.predict_proba(logit(np.clip(probability, 1e-12, 1 - 1e-12))[:, None])[:, 1]
    prediction["p_isotonic"] = isotonic.predict(probability)
    prediction["p_null"] = float(loss[training].mean())
    prediction.to_parquet(folder / "test_predictions.parquet", index=False)
    pd.DataFrame(dict(
        row_id=rows.iloc[training].row_id, loss_observed=loss[training], p_inner_oof_raw=inner_raw,
    )).to_parquet(folder / "calibration_oof.parquet", index=False)
    pd.DataFrame(dict(feature=outer_features.columns, importance=np.mean(importance, axis=0))).to_csv(
        folder / "feature_importance.csv", index=False,
    )
    np.savez_compressed(folder / "fit_indices.npz", **sampled_indices)
    write_json(folder / "calibrators.json", dict(
        platt=dict(coef=float(platt.coef_[0, 0]), intercept=float(platt.intercept_[0]), C=1000.0),
        isotonic=dict(x=isotonic.X_thresholds_.tolist(), y=isotonic.y_thresholds_.tolist()),
        fitted_on="Natural-label inner OOF of eight development matches; no test outcomes",
    ))
    write_json(folder / "fit_audit.json", audits)
    write_json(folder / "complete.json", dict(
        fold=fold, sampling=sampling, n=len(test), elapsed_seconds=time.perf_counter() - started,
        selected_parameters=chosen, catboost_version=catboost.__version__,
        test_matches=sorted(test_matches),
    ))


# ============================================================
# 4. NATURAL-PREVALENCE HELD-OUT METRICS
# ============================================================

def evaluate_predictions(loss, probability, null_probability):
    loss = np.asarray(loss, int)
    probability = np.asarray(probability, float)
    success, outcome = 1 - probability, 1 - loss
    null = np.broadcast_to(np.asarray(null_probability, float), probability.shape)
    ce = float(log_loss(outcome, success, labels=[0, 1]))
    null_ce = float(log_loss(outcome, 1 - null, labels=[0, 1]))
    edges = np.unique(np.quantile(success, np.linspace(0, 1, 11)))
    if len(edges) < 2:
        edges = np.array([0.0, 1.0])
    edges[0], edges[-1] = 0.0, 1.0
    bins = []
    for index in range(len(edges) - 1):
        selected = success >= edges[index]
        selected &= success <= edges[index + 1] if index == len(edges) - 2 else success < edges[index + 1]
        if selected.any():
            bins.append(dict(
                n=int(selected.sum()), predicted_loss=float(probability[selected].mean()),
                observed_loss=float(loss[selected].mean()),
            ))
    table = pd.DataFrame(bins)
    ece = float((table.n * abs(table.predicted_loss - table.observed_loss)).sum() / len(loss))
    result = dict(
        n=len(loss), losses=int(loss.sum()), cross_entropy=ce, null_cross_entropy=null_ce,
        gain_on_null=1 - ce / null_ce, brier=float(np.mean((loss - probability) ** 2)),
        ece_quantile10=ece, mean_predicted_loss=float(probability.mean()),
        observed_loss=float(loss.mean()), zero_probability_losses=int(((probability == 0) & (loss == 1)).sum()),
    )
    if len(np.unique(loss)) == 2:
        result.update(roc_auc=float(roc_auc_score(loss, probability)), pr_auc_loss=float(average_precision_score(loss, probability)))
    return result, table


def compute_pooled_metrics(output, samplings=("natural", "loss10")):
    metrics = []
    for sampling in samplings:
        parts = []
        for fold in range(5):
            folder = Path(output) / sampling / f"fold_{fold}"
            json.loads((folder / "complete.json").read_text())
            parts.append(pd.read_parquet(folder / "test_predictions.parquet"))
        pooled = pd.concat(parts, ignore_index=True)
        if pooled.duplicated(["match_id", "frame_start", "frame_end"]).any():
            raise ValueError("Duplicated held-out states.")
        pooled.to_parquet(Path(output) / f"{sampling}_pooled_predictions.parquet", index=False)
        for version in ["raw", "prior_corrected", "platt", "isotonic"]:
            scores, bins = evaluate_predictions(pooled.loss_observed, pooled["p_" + version], pooled.p_null)
            metrics.append(dict(sampling=sampling, version=version, **scores))
            bins.to_csv(Path(output) / f"{sampling}_{version}_calibration.csv", index=False)
    table = pd.DataFrame(metrics)
    table.to_csv(Path(output) / "pooled_metrics.csv", index=False)
    return table


# Small output helper; outside the mathematical/training flow.
def write_json(path, value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, default=convert, indent=2, allow_nan=False) + "\n")


