"""MLE -> nested match calibration -> production refit -> pooled evaluation.

The two physical parameters are estimated ONLY from reconstructed trajectories.
Calibrators subsequently map endpoint PPCF scores to completion probabilities.
No tree model, prior, early stopping or fixed learning rate enters this task.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from model import PHYSICAL_SETTINGS, negative_log_likelihood, integrate_pitch_control_logs


# ============================================================
# 1. OPTIMIZATION: THREE COMPLETE MLE FITS, THEN KEEP LOWEST NLL
# ============================================================

INITIAL_PARAMETERS = [[4.3, 0.43], [3.0, 0.3], [5.0, 0.5]]
VALUE_AND_GRADIENT = jax.jit(jax.value_and_grad(negative_log_likelihood))
LIKELIHOOD_HESSIAN = jax.jit(jax.hessian(negative_log_likelihood))


def fit_mle(path_tti, ball_low, valid_players, outcomes):
    """Estimate lambda_control,s > 0 in theta=(log(lambda),log(s)).

    BFGS chooses steps by line search: there is no fixed learning rate.
    Every start receives a complete optimization, up to 500 iterations.
    A finite result with max|gradient| < .01 is accepted, even when SciPy's
    success flag is False. Keep the accepted result with lowest TRAIN NLL.
    The Hessian is diagnostic; this MLE does not define a Laplace posterior.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable JAX float64 before MLE.")
    if not valid_players[:, :10].any(axis=1).all() or not valid_players[:, 11:].any(axis=1).all():
        raise ValueError("Each pass needs at least one valid player per team.")
    likelihood_inputs = tuple(jnp.asarray(value) for value in
                      [path_tti, ball_low, valid_players, outcomes])

    def objective(log_parameters):
        value, gradient = VALUE_AND_GRADIENT(jnp.asarray(log_parameters), *likelihood_inputs)
        return float(value), np.asarray(gradient)

    attempts = []
    for initial in INITIAL_PARAMETERS:
        optimization_result = minimize(
            objective, np.log(initial), jac=True, method="BFGS",
            options={"gtol": 1e-4, "maxiter": 500},
        )
        value, gradient = objective(optimization_result.x)
        attempts.append(dict(
            initial=initial, objective=value, log_parameters=optimization_result.x.tolist(),
            success=bool(optimization_result.success), message=str(optimization_result.message),
            iterations=int(optimization_result.nit), gradient_max=float(np.max(abs(gradient))),
        ))
    converged_attempts = [attempt for attempt in attempts
                if np.isfinite(attempt["objective"])
                and np.isfinite(attempt["log_parameters"]).all()
                and attempt["gradient_max"] < 0.01]
    if not converged_attempts:
        raise ValueError(f"No converged MLE among the three starts: {attempts}")
    best_attempt = min(converged_attempts, key=lambda attempt: attempt["objective"])
    log_parameters = np.asarray(best_attempt["log_parameters"])
    eigenvalues = np.linalg.eigvalsh(np.asarray(
        LIKELIHOOD_HESSIAN(jnp.asarray(log_parameters), *likelihood_inputs),
    ))
    return dict(
        **best_attempt, lambda_control=float(np.exp(log_parameters[0])),
        s=float(np.exp(log_parameters[1])), attempts=attempts,
        hessian_eigenvalues=eigenvalues.tolist(),
        weak_identification=bool(eigenvalues.min() < 1e-5),
        estimator="MLE; natural Bernoulli likelihood; positive parameters via logs; no prior",
    )


# ============================================================
# 2. ESTIMATE CONTROL PARAMETERS ON TRAINING MATCHES
# ============================================================


def fit_on_matches(pass_table, model_inputs, training_matches):
    """Estimate lambda_control and s using only the requested training matches.

    pass_table: one row per pass, with match_id and observed outcome (1=success).
    model_inputs: precomputed arrays, aligned row by row with pass_table:
      path_tti: player TTI along the already chosen ball path, shape (N,T,22);
      ball_low: ball height <=2m at each time, shape (N,T);
      valid_players: eligible players, shape (N,22), passer excluded.
    N is the full number of passes, T=251. These are input data, not fitted
    parameters. The output contains parameter estimates and fit diagnostics.
    """
    # 1. Select passes belonging to the current training matches.
    # Apply the same row mask to model inputs and observed outcomes.
    training_mask = pass_table.match_id.isin(training_matches).to_numpy()

    # 2. Extract the already chosen paths' inputs for the N_train passes.
    # TTI of all 22 players to the ball position at each time: (N_train,T,22).
    training_player_tti = model_inputs["path_tti"][training_mask]
    # Control is allowed only when the ball is low enough: (N_train,T).
    training_ball_low = model_inputs["ball_low"][training_mask]
    # Player eligibility is fixed at release: (N_train,22).
    training_valid_players = model_inputs["valid_players"][training_mask]
    # Observed pass completion: (N_train,), with 1=success and 0=failure.
    training_outcomes = pass_table.outcome.to_numpy()[training_mask]

    # 3. MLE: integrate along these training paths and optimize lambda,s.
    parameter_estimates = fit_mle(
        path_tti=training_player_tti,
        ball_low=training_ball_low,
        valid_players=training_valid_players,
        outcomes=training_outcomes,
    )

    # 4. Record the upstream trajectory-selection rule and training matches.
    parameter_estimates.update(
        boundary="per_pass_forecast_error",
        fit_matches=sorted(training_matches),
        n_fit=int(training_mask.sum()),
    )
    return parameter_estimates


def predict_pitch_control(parameter_estimates, model_inputs, prediction_rows, batch_size=256):
    """Return raw attacking-team pitch control using fitted lambda and s.

    Only release-state TTI and hypothetical flight times are consulted.
    This is the PPCF at the pass destination, not the probability integrated
    along a reconstructed trajectory. It has not yet been calibrated.
    Batches reduce memory; each pass has its own independent integration.
    """
    # prediction_rows can be a Boolean row mask or explicit integer indices.
    if np.asarray(prediction_rows).dtype == bool:
        prediction_indices = np.flatnonzero(prediction_rows)
    else:
        prediction_indices = np.asarray(prediction_rows)

    success_probability_batches = []
    for batch_start in range(0, len(prediction_indices), batch_size):
        batch_indices = prediction_indices[batch_start:batch_start + batch_size]
        log_success, _, _ = integrate_pitch_control_logs(
            jnp.asarray(parameter_estimates["log_parameters"]),
            jnp.asarray(model_inputs["endpoint_arrivals"][batch_indices]),
            jnp.asarray(model_inputs["ball_arrival_time"][batch_indices]),
            jnp.asarray(model_inputs["valid_players"][batch_indices]),
        )
        success_probability_batches.append(np.asarray(jnp.exp(log_success)))
    return np.concatenate(success_probability_batches) if success_probability_batches else np.empty(0)


# ============================================================
# 3. INDEPENDENT CALIBRATION OF CROSS-FITTED ENDPOINT PPCF SCORES
# ============================================================

def clipped_logit(probabilities):
    """Accepted Platt convention: clip at 1e-8 before taking log odds."""
    probabilities = np.clip(probabilities, 1e-8, 1 - 1e-8)
    return np.log(probabilities) - np.log1p(-probabilities)


def fit_calibrators(raw_probabilities, outcomes):
    """Fit isotonic (main) and Platt (comparison) on natural held-out labels.

    Platt uses sklearn's accepted default L2 penalty (C=1), solver lbfgs.
    Calibration never updates lambda or s and never sees the outer test.
    """
    platt = LogisticRegression(solver="lbfgs").fit(
        clipped_logit(raw_probabilities)[:, None], outcomes,
    )
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(raw_probabilities, outcomes)
    return dict(
        platt_a=float(platt.coef_[0, 0]), platt_b=float(platt.intercept_[0]),
        isotonic_x=isotonic.X_thresholds_.tolist(), isotonic_y=isotonic.y_thresholds_.tolist(),
    )


def apply_calibrators(raw_probabilities, calibration):
    """Apply previously estimated maps; no fitting on these observations."""
    return dict(
        trajectory_raw=np.asarray(raw_probabilities),
        trajectory_platt=expit(calibration["platt_a"] * clipped_logit(raw_probabilities)
                               + calibration["platt_b"]),
        trajectory_isotonic=np.interp(raw_probabilities, calibration["isotonic_x"],
                                      calibration["isotonic_y"]),
    )


def run_fold(pass_table, model_inputs, fold, test_matches, output):
    """Run one outer fold, keeping its two test matches untouched until step 5.

    10 matches = 8 development + 2 test.
    Within development: four rotations of 6 for MLE + 2 for calibration scores.
    Pool those scores to fit calibrators, then refit lambda,s on all eight.
    Apply that physical model and the fixed calibrators to the two test matches.
    This is the final fixed specification; no hyperparameter grid is searched.
    """
    folder = Path(output) / "training" / f"fold_{fold}"
    folder.mkdir(parents=True, exist_ok=False)

    # --------------------------------------------------------
    # 1. SEPARATE THE EIGHT DEVELOPMENT AND TWO TEST MATCHES
    # --------------------------------------------------------
    # The masks refer to rows of pass_table and the aligned model-input arrays.
    test_mask = pass_table.match_id.isin(test_matches).to_numpy()
    development_mask = ~test_mask
    development_matches = sorted(
        pass_table.loc[development_mask, "match_id"].unique()
    )
    if len(development_matches) != 8 or len(set(test_matches)) != 2:
        raise ValueError("The accepted split requires eight development / two test matches.")

    # --------------------------------------------------------
    # 2. GET HELD-OUT PPCF SCORES THROUGH FOUR INTERNAL 6+2 FITS
    # --------------------------------------------------------
    # Reserve one slot per pass. Fill only the development rows below.
    # Each receives a raw PPCF score from a model that did NOT fit its match.
    cross_fitted_pitch_control = np.full(len(pass_table), np.nan)
    six_match_fits = []

    # Sorted development matches are split into four nonoverlapping pairs:
    # [0:2], [2:4], [4:6], [6:8]. Each pair is held out exactly once.
    for pair_start in range(0, 8, 2):
        calibration_matches = development_matches[pair_start:pair_start + 2]
        training_matches = [
            match for match in development_matches
            if match not in calibration_matches
        ]

        # A. Estimate lambda,s on SIX matches using the already chosen paths.
        inner_parameter_estimates = fit_on_matches(
            pass_table, model_inputs, training_matches,
        )

        # B. Predict raw pitch control on the OTHER TWO matches, without
        # updating lambda,s or using their outcomes in the physical fit.
        calibration_mask = pass_table.match_id.isin(calibration_matches).to_numpy()
        pair_pitch_control = predict_pitch_control(
            inner_parameter_estimates, model_inputs, calibration_mask,
        )
        cross_fitted_pitch_control[calibration_mask] = pair_pitch_control

        # C. Keep the fit diagnostics and the identities of the held-out pair.
        inner_parameter_estimates["calibration_matches"] = calibration_matches
        six_match_fits.append(inner_parameter_estimates)
        print(
            "INNER", fold, calibration_matches,
            inner_parameter_estimates["lambda_control"],
            inner_parameter_estimates["s"], flush=True,
        )

    # All eight development matches now have held-out scores. Test slots
    # must still be empty: no test information has entered calibration.
    assert np.isfinite(cross_fitted_pitch_control[development_mask]).all()
    assert np.isnan(cross_fitted_pitch_control[test_mask]).all()

    # --------------------------------------------------------
    # 3. FIT CALIBRATORS ON THE EIGHT MATCHES' HELD-OUT SCORES
    # --------------------------------------------------------
    # Each score above was obtained by a different six-match physical fit.
    # Pool all four pairs, with natural outcomes, to fit ONE isotonic and ONE
    # Platt map for this outer fold. This does not re-estimate lambda or s.
    calibration_probabilities = cross_fitted_pitch_control[development_mask]
    calibration_outcomes = pass_table.outcome.to_numpy()[development_mask]
    calibration_models = fit_calibrators(
        calibration_probabilities, calibration_outcomes,
    )

    # --------------------------------------------------------
    # 4. REFIT THE PHYSICAL PARAMETERS ON ALL EIGHT DEVELOPMENT MATCHES
    # --------------------------------------------------------
    # This is the physical fit used to predict the external test pair.
    # The calibration maps from step 3 remain fixed.
    development_parameter_estimates = fit_on_matches(
        pass_table, model_inputs, development_matches,
    )

    # --------------------------------------------------------
    # 5. PREDICT ON THE TWO EXTERNAL TEST MATCHES
    # --------------------------------------------------------
    # Compute raw PPCF with the eight-match estimates, then apply the maps.
    # The returned table contains raw, Platt and isotonic predictions.
    test_pitch_control = predict_pitch_control(
        development_parameter_estimates, model_inputs, test_mask,
    )
    calibrated_test_probabilities = apply_calibrators(
        test_pitch_control, calibration_models,
    )
    test_predictions = pass_table.loc[test_mask].copy()
    for probability_name, probabilities in calibrated_test_probabilities.items():
        test_predictions[probability_name] = probabilities

    # Null model: constant completion frequency from the EIGHT development
    # matches. The test outcomes do not determine this reference probability.
    test_predictions["null_global"] = float(
        pass_table.loc[development_mask, "outcome"].mean()
    )
    test_predictions["fold"] = int(fold)

    # --------------------------------------------------------
    # 6. SAVE TEST PREDICTIONS, CALIBRATION SCORES AND FIT DIAGNOSTICS
    # --------------------------------------------------------
    # Metrics are pooled later by collect_results, after all five folds finish.
    test_predictions.to_parquet(folder / "test_predictions.parquet", index=False)

    calibration_table = pass_table.loc[
        development_mask, ["match_id", "pass_id", "frame_end", "outcome"]
    ].copy()
    calibration_table["raw"] = calibration_probabilities
    calibration_table.to_parquet(folder / "inner_calibration.parquet", index=False)

    write_json(folder / "models.json", dict(
        fit=development_parameter_estimates,
        internal=six_match_fits,
        calibration=calibration_models,
        outer_test_matches=list(test_matches),
    ))
    write_json(folder / "complete.json", dict(n_test=len(test_predictions), fold=int(fold)))
    return test_predictions

# ============================================================
# 4. PRODUCTION REFIT: TEN-MATCH MLE + OUTER-CROSS-FITTED ISOTONIC
# ============================================================

def refit_production(pass_table, model_inputs, pooled_predictions, output):
    """Build ONE operational model after the five-fold benchmark is complete.

    Physical fit: estimate one lambda,s pair from all ten matches' trajectories.
    Calibration: fit one isotonic map to pooled raw OUTER-HELD-OUT PPCF scores.
    Each calibration score was predicted by a model fitted on the other eight
    matches. These are not in-sample scores of the ten-match physical fit.

    Save both pieces in one JSON. Benchmark metrics still use the original
    held-out predictions; they are not replaced by production predictions.
    """
    # --------------------------------------------------------
    # 1. ALIGN EACH PASS WITH ITS RAW HELD-OUT PPCF SCORE
    # --------------------------------------------------------
    # pooled_predictions is collected from the five external test pairs.
    # trajectory_raw contains PPCF BEFORE isotonic/Platt calibration.
    # Match by identifiers, since fold concatenation can change row order.
    pass_keys = ["match_id", "pass_id", "frame_end"]
    calibration_passes = pass_table[pass_keys + ["outcome"]].merge(
        pooled_predictions[pass_keys + ["outcome", "trajectory_raw"]],
        on=pass_keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_source", ""),
        indicator=True,
    )
    # Require exactly the same passes and observed outcomes on both sides.
    all_passes_present = calibration_passes._merge.eq("both").all()
    outcomes_match = calibration_passes.outcome_source.eq(calibration_passes.outcome).all()
    if not all_passes_present or not outcomes_match:
        raise ValueError("Production calibration must cover exactly the accepted cohort.")

    # --------------------------------------------------------
    # 2. ESTIMATE ONE lambda,s PAIR FROM ALL TEN MATCHES
    # --------------------------------------------------------
    # Use all available trajectories in one new MLE. The five fold estimates
    # are not averaged. The trajectory for each pass was chosen upstream.
    all_training_matches = sorted(pass_table.match_id.unique())
    production_parameter_estimates = fit_on_matches(
        pass_table, model_inputs, all_training_matches,
    )

    # --------------------------------------------------------
    # 3. FIT ONE PRODUCTION ISOTONIC MAP FROM RAW HELD-OUT SCORES
    # --------------------------------------------------------
    # For every pass i, we have (p_i_OOF, y_i):
    #   p_i_OOF = raw PPCF from the eight-match fit that excluded its match;
    #   y_i     = observed completion label, 1=success and 0=failure.
    # Fit g to these pairs. Do not apply the existing fold calibrators first,
    # and do not recompute calibration scores with the ten-match estimates.
    calibration_probabilities = calibration_passes["trajectory_raw"]
    calibration_outcomes = calibration_passes["outcome"]
    production_isotonic = IsotonicRegression(out_of_bounds="clip").fit(
        calibration_probabilities, calibration_outcomes,
    )

    # --------------------------------------------------------
    # 4. PACKAGE PARAMETERS, PHYSICAL SETTINGS AND CALIBRATION
    # --------------------------------------------------------
    # For a new state x, the operational prediction is
    #   p_success(x) = g(PPCF(x; lambda_10, s_10)).
    # Store the isotonic knots/values so inference needs no calibration fit.
    production_model = dict(
        schema_version=1,
        model_type="pass_risk_trajectory_isotonic",
        model_id="review_pass_probability_final_refit",
        physics=PHYSICAL_SETTINGS,
        lambda_control=production_parameter_estimates["lambda_control"],
        s=production_parameter_estimates["s"],
        fit=production_parameter_estimates,
        trajectory_boundary=production_parameter_estimates["boundary"],
        calibration=dict(
            method="isotonic",
            x=production_isotonic.X_thresholds_.tolist(),
            y=production_isotonic.y_thresholds_.tolist(),
        ),
    )

    # --------------------------------------------------------
    # 5. CHECK THE PUBLIC READER AND SAVE A NEW PRODUCTION MODEL
    # --------------------------------------------------------
    from PRESS.model import PassRiskModel
    PassRiskModel(production_model)
    production_folder = Path(output) / "production"
    production_folder.mkdir(parents=True, exist_ok=False)
    write_json(production_folder / "pass_risk_model.json", production_model)
    return production_model

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


# ============================================================
# 5. FINAL EVALUATION: POOL FIVE UNTOUCHED TEST PAIRS, THEN SCORE
# ============================================================

def binary_metrics(outcomes, probabilities, null_probabilities):
    """Evaluate fixed predictions; return a metric dictionary and calibration bins.

    Inputs have one value per pass, in the same row order:
      outcomes: 1 for completion, 0 for failure;
      probabilities: predicted completion probability;
      null_probabilities: completion frequency from that pass's training fold.

    Rows must already follow canonical match_id/pass_id/frame_end ordering.
    This fixes the order of tied scores when building the calibration bins.
    Only cross-entropy uses clipping; Brier, ECE and AUC use raw predictions.
    """
    success_labels = np.asarray(outcomes, float)
    success_probabilities = np.asarray(probabilities, float)
    null_success_probabilities = np.asarray(null_probabilities, float)
    n_passes = len(success_labels)

    # --------------------------------------------------------
    # 1. CROSS-ENTROPY OF THE MODEL
    # --------------------------------------------------------
    # For pass i: loss_i = -y_i*log(p_i) - (1-y_i)*log(1-p_i).
    # CE = mean_i loss_i. Lower is better.
    # Clip ONLY for logarithms, to keep exact probabilities 0/1 finite.
    ce_probabilities = np.clip(success_probabilities, 1e-6, 1 - 1e-6)
    model_log_losses = (
        -success_labels * np.log(ce_probabilities)
        - (1 - success_labels) * np.log1p(-ce_probabilities)
    )
    cross_entropy = float(np.mean(model_log_losses))

    # --------------------------------------------------------
    # 2. CROSS-ENTROPY OF THE NULL MODEL AND RELATIVE GAIN
    # --------------------------------------------------------
    # The null probability is constant within each outer test fold: it is
    # the completion rate observed in that fold's eight development matches.
    # Across the pooled test rows it can therefore differ between folds.
    null_ce_probabilities = np.clip(null_success_probabilities, 1e-6, 1 - 1e-6)
    null_log_losses = (
        -success_labels * np.log(null_ce_probabilities)
        - (1 - success_labels) * np.log1p(-null_ce_probabilities)
    )
    null_cross_entropy = float(np.mean(null_log_losses))
    # Gain = (CE_null - CE_model)/CE_null = 1 - CE_model/CE_null.
    # Return a fraction: .68 means a 68% reduction relative to the null.
    ce_gain = 1 - cross_entropy / null_cross_entropy

    # --------------------------------------------------------
    # 3. BRIER SCORE
    # --------------------------------------------------------
    # Brier = mean_i (y_i-p_i)^2, using the original probabilities.
    brier_score = float(np.mean((success_labels - success_probabilities) ** 2))

    # --------------------------------------------------------
    # 4. TEN EQUAL-COUNT CALIBRATION BINS AND ECE
    # --------------------------------------------------------
    # Sort from low to high predicted completion probability, then divide the
    # indices into ten groups whose sizes differ by at most one pass.
    # Stable sorting retains the input order for identical probabilities.
    # Tied isotonic scores can span several bins; no qcut/deduplication is used.
    sorted_pass_indices = np.argsort(success_probabilities, kind="stable")
    indices_per_bin = np.array_split(sorted_pass_indices, 10)

    calibration_bins = []
    weighted_calibration_gaps = []
    for bin_number, pass_indices in enumerate(indices_per_bin):
        n_in_bin = len(pass_indices)
        if n_in_bin == 0:
            continue

        # In bin b: p_bar_b = mean predicted probability;
        #           y_bar_b = fraction of passes actually completed.
        mean_predicted = float(success_probabilities[pass_indices].mean())
        observed_frequency = float(success_labels[pass_indices].mean())
        calibration_bins.append(dict(
            bin=bin_number, n=n_in_bin,
            predicted=mean_predicted, observed=observed_frequency,
        ))

        # ECE = sum_b (n_b/N) * |p_bar_b - y_bar_b|.
        # Weight each absolute calibration gap by its fraction of all passes.
        bin_weight = n_in_bin / n_passes
        calibration_gap = abs(mean_predicted - observed_frequency)
        weighted_calibration_gaps.append(bin_weight * calibration_gap)

    expected_calibration_error = float(sum(weighted_calibration_gaps))

    # --------------------------------------------------------
    # 5. ROC AUC: RANKING OF COMPLETIONS VERSUS FAILURES
    # --------------------------------------------------------
    # AUC measures discrimination, independently of the ten calibration bins.
    # It is undefined when the evaluated sample contains only one class.
    if len(np.unique(success_labels)) > 1:
        roc_auc = float(roc_auc_score(success_labels, success_probabilities))
    else:
        roc_auc = None

    # --------------------------------------------------------
    # 6. RETURN METRICS AND THE POINTS FOR THE CALIBRATION PLOT
    # --------------------------------------------------------
    metrics = dict(
        n=n_passes,
        losses=int((1 - success_labels).sum()),
        cross_entropy=cross_entropy,
        null_cross_entropy=null_cross_entropy,
        gain_ce=ce_gain,
        brier=brier_score,
        ECE_quantile=expected_calibration_error,
        mean_predicted_success=float(success_probabilities.mean()),
        observed_success=float(success_labels.mean()),
        ROC_AUC=roc_auc,
    )
    return metrics, calibration_bins

def collect_results(pass_table, output):
    """Pool the five outer test folds and evaluate each prediction version.

    Each pass must appear exactly once, predicted while its match was held out.
    Metrics are computed on the combined pass-level predictions, instead of
    by averaging five fold metrics. Calibration bins are also built anew on
    this pooled sample. No physical parameter or calibrator is fitted here.
    """
    # --------------------------------------------------------
    # 1. READ THE TWO-MATCH TEST PREDICTIONS FROM EACH FOLD
    # --------------------------------------------------------
    test_tables_by_fold = []
    for fold in range(5):
        fold_folder = Path(output) / "training" / f"fold_{fold}"
        if not (fold_folder / "complete.json").is_file():
            raise ValueError(f"Fold {fold} is incomplete.")

        fold_test_predictions = pd.read_parquet(
            fold_folder / "test_predictions.parquet"
        )
        assert fold_test_predictions.fold.eq(fold).all()
        test_tables_by_fold.append(fold_test_predictions)

    # --------------------------------------------------------
    # 2. COMBINE THE FIVE TEST TABLES AND FIX THEIR ROW ORDER
    # --------------------------------------------------------
    # Concatenation gives one table covering all ten matches. Sort by pass
    # identity so results do not depend on the order in which jobs finished.
    pooled_test_predictions = pd.concat(test_tables_by_fold, ignore_index=True)
    pooled_test_predictions = pooled_test_predictions.sort_values(
        ["match_id", "pass_id", "frame_end"], kind="stable",
    ).reset_index(drop=True)

    # Isotonic can assign identical probabilities to several passes.
    # binary_metrics sorts scores stably: this canonical input order fixes
    # how tied scores are divided between equal-count calibration bins.

    # --------------------------------------------------------
    # 3. CHECK THAT THE POOLED SAMPLE IS EXACTLY THE SELECTED COHORT
    # --------------------------------------------------------
    pass_keys = ["match_id", "pass_id", "frame_end"]
    coverage_check = pass_table[pass_keys + ["outcome"]].merge(
        pooled_test_predictions[pass_keys + ["outcome"]],
        on=pass_keys,
        how="outer",
        validate="one_to_one",  # Reject duplicated pass identities.
        suffixes=("_source", ""),
        indicator=True,
    )
    # Every source pass must have one prediction, with no extra passes.
    all_passes_present = coverage_check._merge.eq("both").all()
    # The success/failure labels must be unchanged from the selected input.
    outcomes_unchanged = coverage_check.outcome_source.eq(coverage_check.outcome).all()
    assert all_passes_present and outcomes_unchanged

    # --------------------------------------------------------
    # 4. CALCULATE POOLED METRICS AND CALIBRATION BINS
    # --------------------------------------------------------
    score_folder = Path(output) / "scores"
    score_folder.mkdir(parents=True, exist_ok=False)
    metric_rows = []
    calibration_rows = []
    probability_columns = [
        "trajectory_raw",        # Raw PPCF, using fitted lambda,s.
        "trajectory_platt",      # Same scores, with fold-specific Platt maps.
        "trajectory_isotonic",   # Same scores, with fold-specific isotonic maps.
    ]
    for probability_column in probability_columns:
        # Every row keeps its own fold's null probability, obtained from
        # that fold's development matches. Do not refit a pooled null model.
        model_metrics, model_calibration_bins = binary_metrics(
            outcomes=pooled_test_predictions.outcome,
            probabilities=pooled_test_predictions[probability_column],
            null_probabilities=pooled_test_predictions.null_global,
        )
        metric_rows.append(dict(model=probability_column, **model_metrics))
        for bin_summary in model_calibration_bins:
            calibration_rows.append(dict(model=probability_column, **bin_summary))

    # --------------------------------------------------------
    # 5. SAVE THE POOLED PREDICTIONS, METRIC TABLE AND PLOT POINTS
    # --------------------------------------------------------
    metrics_table = pd.DataFrame(metric_rows)
    calibration_table = pd.DataFrame(calibration_rows)
    pooled_test_predictions.to_parquet(
        score_folder / "pooled_predictions.parquet", index=False,
    )
    metrics_table.to_csv(score_folder / "metrics.csv", index=False)
    calibration_table.to_csv(score_folder / "calibration_deciles.csv", index=False)
    return metrics_table
