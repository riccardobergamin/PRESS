"""Physical ball-drive training, in order from optimization to final evaluation.

Reading order:
  1. MAP optimization, convergence checks and Laplace covariance.
  2. Training-set fit and validation/test prediction with fixed coefficients.
  3. Smoothing grid and validation-based regularization selection.
  4. Five-fold benchmark with inner six-fit/two-validation rotations.
  5. Final production refit on all ten matches, after validation is complete.
  6. Model/diagnostic output helpers.
  7. Metrics, calibration bins and pooled held-out evaluation.

The equations and spline construction are in model.py. Function order is for
reading: Python loads all definitions before the pipeline calls any of them.
Production-refit predictions do not enter the held-out benchmark metrics.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score

from model import (
    PHYSICS, build_spline_basis, prepare_model_data, predict_survival,
    build_prior_precision, negative_log_posterior,
)


# ============================================================
# 1. MAP OPTIMIZATION AND LAPLACE UNCERTAINTY
# ============================================================


# Objective J(theta) = -log L(theta) -log p(theta).
# JAX differentiates the complete likelihood/prior, including the survival
# integration. jit compiles each array shape; SciPy receives value/gradient.
VALUE_GRADIENT = jax.jit(jax.value_and_grad(negative_log_posterior))
# H(theta) = second derivative of J, used for refinement and Laplace covariance.
HESSIAN = jax.jit(jax.hessian(negative_log_posterior))


def fit_map(training_data, prior_precision, seed, initializations=32, maxiter=2000):
    """Estimate the MAP from training data and its local Laplace covariance.

    Optimization coordinates:
      theta = [alpha, beta_v, beta_B, ell, spline/angle coefficients],
      lambda0 = exp(alpha), s = exp(ell).

    prior_precision contains inverse Gaussian variances for the extra effects.
    The returned theta and covariance remain in the optimization coordinates.
    The five numbered steps below implement initialization, optimization,
    fallback, MAP selection and Laplace; validation outcomes never enter here.
    """

    if not jax.config.x64_enabled:
        raise ValueError("Set JAX_ENABLE_X64=1 before fitting.")
    prior_precision = jnp.asarray(prior_precision, dtype=jnp.float64)

    # SciPy adapters: J(theta) = -log likelihood -log prior.
    def objective_and_gradient(parameters):
        objective_value, gradient = VALUE_GRADIENT(
            jnp.asarray(parameters), training_data, prior_precision,
        )
        objective_value = float(objective_value)
        gradient = np.asarray(gradient, float)
        if not np.isfinite(objective_value) or not np.isfinite(gradient).all():
            # Mark numerical overflow with a cost that cannot be accepted.
            return 1e100, np.zeros_like(parameters)
        return objective_value, gradient

    def posterior_hessian(parameters):
        hessian = np.asarray(HESSIAN(
            jnp.asarray(parameters), training_data, prior_precision,
        ))
        hessian = (hessian + hessian.T) / 2
        if not np.isfinite(hessian).all():
            raise ValueError("Nonfinite Hessian.")
        return hessian

    def optimize_from_start(initial_parameters):
        """One L-BFGS-B fit; refine with the Hessian only if needed."""
        start_parameters = initial_parameters
        solver_attempts = []
        for solver in ["L-BFGS-B", "trust-exact"]:
            if solver == "L-BFGS-B":
                # Steps are chosen by line search, not a fixed learning rate.
                # ftol/gtol control convergence; maxls limits line-search steps.
                result = minimize(
                    objective_and_gradient, start_parameters,
                    jac=True, method="L-BFGS-B",
                    options=dict(maxiter=maxiter, ftol=1e-12, gtol=1e-5, maxls=40),
                )
                method_name = "L-BFGS-B"
            else:
                # Same training objective, starting from the first result.
                result = minimize(
                    objective_and_gradient, start_parameters,
                    jac=True, hess=posterior_hessian, method="trust-exact",
                    options=dict(maxiter=200, gtol=1e-5),
                )
                method_name = "trust-exact stationarity refinement"

            final_objective, final_gradient = objective_and_gradient(result.x)
            max_gradient = float(abs(final_gradient).max())
            converged = bool(
                result.success and final_objective < 1e99 and max_gradient < 0.01
            )
            solver_attempts.append(dict(
                theta=result.x, objective=final_objective, converged=converged,
                gradient_max=max_gradient, iterations=int(result.nit),
                message=str(result.message), method=method_name,
            ))
            # A successful first fit needs no refinement. Overflow also stops
            # this start; only finite unsuccessful results reach trust-exact.
            if converged or final_objective >= 1e99:
                break
            start_parameters = result.x
        return solver_attempts

    # 1. INITIAL POINTS: fixed point plus 32 prior draws by default.
    # These 33 points are only prescored; they are not 33 full optimizations.
    fixed_start = np.r_[
        np.log(0.4), 0.1, -0.02, np.log(0.4), np.zeros(len(prior_precision)),
    ]
    random_generator = np.random.default_rng(seed)
    initial_points = [fixed_start]
    for _ in range(initializations):
        # alpha,beta_v,beta_B ~ N(0,1); s ~ Gamma(2.5,rate=5).
        # Extra coefficients gamma_k ~ N(0,1/prior_precision_k).
        alpha = random_generator.normal()
        beta_v = random_generator.normal()
        beta_boundary = random_generator.normal()
        log_dispersion = np.log(random_generator.gamma(2.5, 0.2))
        extra_coefficients = (
            random_generator.normal(size=len(prior_precision))
            / np.sqrt(np.asarray(prior_precision))
        )
        initial_points.append(np.r_[
            alpha, beta_v, beta_boundary, log_dispersion, extra_coefficients,
        ])

    initial_objectives = [
        float(negative_log_posterior(point, training_data, prior_precision))
        for point in initial_points
    ]
    ranked_start_indices = np.argsort(
        np.where(np.isfinite(initial_objectives), initial_objectives, np.inf),
        kind="stable",
    )
    best_initial_point = initial_points[int(ranked_start_indices[0])]
    primary_starts = [best_initial_point]
    if not np.array_equal(best_initial_point, fixed_start):
        primary_starts.append(fixed_start)

    # 2. OPTIMIZE from the best initial point and, if different, the fixed one.
    optimization_attempts = []
    for initial_point in primary_starts:
        optimization_attempts.extend(optimize_from_start(initial_point))

    # 3. FALLBACK: only if no result converged, try at most four further
    # prescored points, in objective order. Stop at the first accepted result.
    extra_start_count = 0
    if not any(attempt["converged"] for attempt in optimization_attempts):
        for start_index in ranked_start_indices:
            initial_point = initial_points[int(start_index)]
            if not np.isfinite(initial_objectives[start_index]):
                continue
            if any(np.array_equal(initial_point, used) for used in primary_starts):
                continue
            optimization_attempts.extend(optimize_from_start(initial_point))
            extra_start_count += 1
            if (any(attempt["converged"] for attempt in optimization_attempts)
                    or extra_start_count == 4):
                break

    # 4. MAP: keep the accepted solution with the lowest posterior cost.
    converged_solutions = [
        attempt for attempt in optimization_attempts if attempt["converged"]
    ]
    if not converged_solutions:
        raise RuntimeError("MAP did not satisfy the accepted convergence criteria.")
    map_fit = dict(min(converged_solutions, key=lambda fit: fit["objective"]))

    # 5. LAPLACE: theta|data approximately N(theta_MAP, H(theta_MAP)^(-1)).
    hessian_at_map = posterior_hessian(map_fit["theta"])
    hessian_eigenvalues = np.linalg.eigvalsh(hessian_at_map)
    if hessian_eigenvalues.min() <= 0:
        raise RuntimeError("Selected MAP has no positive-definite Laplace Hessian.")
    map_fit.update(
        covariance=np.linalg.inv(hessian_at_map),
        hessian_eigenvalues=hessian_eigenvalues, laplace_valid=True,
        starts=len(primary_starts) + extra_start_count,
        optimization_attempts=optimization_attempts,
    )
    return map_fit


# ============================================================
# 2. FIT A TRAINING SET AND PREDICT ITS VALIDATION/TEST SET
# ============================================================


def fit_split(covariates, training_indices, evaluation_indices,
              spline_bases, model_configuration, seed):
    """Fit on training rows, then predict validation/test rows with fixed theta.

    ``spline_bases`` was learned using only ``training_indices``.
    ``evaluation_indices`` selects validation rows during tuning, or test
    rows after all tuning choices have been frozen.
    Here we apply them to both sets; only the training data enter the MAP.
    """

    # 1. Which effect blocks are included? The numerical rho values do not
    # enter the covariate matrices; they only enter the prior below.
    use_main_splines = model_configuration["main"] is not None
    use_angle = model_configuration["angle"] is not None

    # 2. Apply the SAME training transformations to both observation sets.
    training_data = prepare_model_data(
        covariates, training_indices, spline_bases,
        use_main_splines=use_main_splines, use_angle=use_angle,
    )
    evaluation_data = prepare_model_data(
        covariates, evaluation_indices, spline_bases,
        use_main_splines=use_main_splines, use_angle=use_angle,
    )

    # 3. Build the prior once and estimate theta using training outcomes only.
    prior_precision = build_prior_precision(spline_bases, model_configuration)
    map_fit = fit_map(training_data, prior_precision, seed)

    # 4. Validation/test prediction needs the fitted theta and covariates,
    # not a new prior or an optimization on the held-out observations.
    evaluation_survival = np.asarray(
        predict_survival(jnp.asarray(map_fit["theta"]), evaluation_data),
    )
    return map_fit, evaluation_survival, prior_precision


# ============================================================
# 3. REGULARIZATION GRID AND VALIDATION SELECTION
# ============================================================


def regularization_grid():
    """Return 16 model/regularization configurations in fixed selection order.

    model_configuration['main'] is the shared speed/boundary curvature rho;
    None keeps only their linear effects. model_configuration['angle'] is
    the angle rho; None removes that complete effect. Its linear prior has
    precision 1 whenever the angle is included.
    """

    # Four carrier configurations: linear, or curvature rho=100,1,.01.
    # Start with all four WITHOUT the angle effect.
    carrier_configurations = [dict(name="linear", main=None, angle=None)]
    for carrier_rho in [100.0, 1.0, 0.01]:
        carrier_configurations.append(dict(
            name=f"main_{carrier_rho:g}", main=carrier_rho, angle=None,
        ))
    model_configurations = [dict(model_configuration) for model_configuration in carrier_configurations]
    # Add those same four configurations for each angle rho: 4 + 3*4 = 16.
    # select_regularization uses this declared order to break one-SE choices.
    for angle_rho in [100.0, 1.0, 0.01]:
        for model_configuration in carrier_configurations:
            model_configurations.append(dict(
                name=model_configuration["name"] + f"_angle_{angle_rho:g}",
                main=model_configuration["main"], angle=angle_rho,
            ))
    return model_configurations


def select_regularization(validation_records, model_configurations):
    """Choose a configuration from inner-validation scores, paired by match.

    Input records contain one mean CE per configuration and validation match.
    The four 6+2 rotations give eight match scores for every configuration.
    Return the first eligible configuration in the grid's declared order,
    with the pooled CE, paired difference and jackknife SE for the audit.
    """

    validation_scores = pd.DataFrame(validation_records)

    # 1. POOLED CE: CE_c = sum_m n_m*CE_cm / sum_m n_m.
    pooled_ce_by_configuration = {}
    for configuration_name, match_scores in validation_scores.groupby("candidate"):
        pooled_ce_by_configuration[configuration_name] = float(
            np.average(match_scores.nll, weights=match_scores.n)
        )
    best_configuration_name = min(
        pooled_ce_by_configuration, key=pooled_ce_by_configuration.get,
    )
    best_match_scores = validation_scores.loc[
        validation_scores.candidate.eq(best_configuration_name)
    ].set_index("match_id")

    comparison_records = []
    eligible_configurations = []
    for model_configuration in model_configurations:
        configuration_name = model_configuration["name"]
        match_scores = validation_scores.loc[
            validation_scores.candidate.eq(configuration_name)
        ].set_index("match_id")
        match_scores = match_scores.reindex(best_match_scores.index)
        if len(match_scores) != 8 or match_scores.nll.isna().any():
            raise ValueError("Expected one paired validation score for each of eight matches.")
        if not np.array_equal(match_scores.n, best_match_scores.n):
            raise ValueError("Candidates were evaluated on different observations.")

        # 2. PAIRED DIFFERENCE: d_m=CE_cm-CE_best,m.
        # delta_c=sum_m n_m*d_m / sum_m n_m.
        match_counts = match_scores.n.to_numpy(float)
        match_ce_differences = match_scores.nll.to_numpy() - best_match_scores.nll.to_numpy()
        pooled_ce_difference = float(np.average(match_ce_differences, weights=match_counts))

        # 3. JACKKNIFE: leave one whole match out, not one drive.
        # delta_(-m)=(sum_j n_j*d_j - n_m*d_m)/(sum_j n_j - n_m).
        leave_one_match_out_differences = (
            (np.sum(match_counts * match_ce_differences) - match_counts * match_ce_differences)
            / (match_counts.sum() - match_counts)
        )
        # SE=sqrt((M-1)/M * sum_m (delta_(-m)-mean(delta_(-m)))^2), M=8.
        jackknife_standard_error = float(np.sqrt(
            7 / 8 * np.sum((
                leave_one_match_out_differences - leave_one_match_out_differences.mean()
            ) ** 2)
        ))

        # 4. ELIGIBILITY: accept if delta_c <= one standard error.
        # The 1e-12 tolerance handles only floating-point roundoff.
        within_one_se = bool(pooled_ce_difference <= jackknife_standard_error + 1e-12)
        comparison_records.append(dict(
            candidate=configuration_name,
            mean_ce=pooled_ce_by_configuration[configuration_name],
            difference_to_best=pooled_ce_difference,
            paired_match_se=jackknife_standard_error, within_one_se=within_one_se,
        ))
        if within_one_se:
            eligible_configurations.append(model_configuration)

    # 5. PARSIMONY: first eligible entry in the PREDEFINED model hierarchy.
    # No sorting by test scores or estimated effective degrees of freedom.
    selected_configuration = eligible_configurations[0]
    return selected_configuration, dict(
        best=best_configuration_name, selected=selected_configuration["name"],
        comparison=comparison_records,
    )


# ============================================================
# 4. FIVE-FOLD BENCHMARK: INNER 6+2, THEN OUTER 8+2
# ============================================================


def run_fold(rows, covariates, fold, test_matches, output):
    """Tune within eight development matches, then predict two held-out matches.

    Each inner fit learns its bases on six matches; the remaining two are
    used to compare smoothing candidates. Four rotations cover all eight
    development matches. The selected candidate is refitted on those eight
    with newly learned bases, before evaluating the untouched test pair.
    """
    fold_output = Path(output) / f"fold_{fold}"
    fold_output.mkdir(parents=True, exist_ok=False)
    # A. OUTER SPLIT: reserve two complete matches for test. All tuning and
    # training below use only the remaining eight development matches.
    development_matches = sorted(set(rows.match_id) - set(test_matches))
    if len(development_matches) != 8 or len(test_matches) != 2:
        raise ValueError("The accepted benchmark uses ten matches and five fixed test pairs.")
    model_configurations = regularization_grid()
    validation_records, inner_fit_results, inner_basis_definitions = [], [], []

    # B. INNER VALIDATION: four fixed pairs rotate through validation.
    # Each development match is evaluated once, with coefficients and spline
    # transformations learned on the other six matches.
    for inner_split_index in range(4):
        validation_matches = development_matches[2 * inner_split_index:2 * inner_split_index + 2]
        training_matches = [match for match in development_matches if match not in validation_matches]
        training_indices = np.flatnonzero(rows.match_id.isin(training_matches))
        validation_indices = np.flatnonzero(rows.match_id.isin(validation_matches))
        # These indices select rows by whole match, not random individual states.
        # Learn the three bases explicitly, using only this fit's training rows.
        # v(t)=v(0): repeat speed across forecast steps as in the final benchmark.
        training_boundary = covariates["boundary"][training_indices]
        training_speed_by_time = np.broadcast_to(
            covariates["speed"][training_indices, None], training_boundary.shape,
        )
        training_angles = covariates["angle"][training_indices]
        valid_training_angles = covariates["angle_available"][training_indices]
        # Five/five/four knots; only available headings define the angle basis.
        # Reuse these definitions for every candidate in this inner split.
        spline_bases = {
            "speed": build_spline_basis(training_speed_by_time, 5),
            "boundary": build_spline_basis(training_boundary, 5),
            "angle": build_spline_basis(training_angles[valid_training_angles], 4),
        }
        inner_basis_definitions.append(dict(
            inner=inner_split_index, design=spline_bases,
            fitting_matches=training_matches, validation_matches=validation_matches,
        ))
        for model_configuration in model_configurations:
            # New MAP per candidate, using the candidate's own prior precision.
            # All candidates get the same declared seed in this inner split.
            map_fit, predicted_survival, _ = fit_split(
                covariates, training_indices, validation_indices, spline_bases, model_configuration,
                seed=10 + 10000 * fold + 100 * inner_split_index,
            )
            observed_success = covariates["outcome"][validation_indices]
            # Per-state validation cost: -[y*log(S)+(1-y)*log(1-S)].
            validation_state_cost = (
                -observed_success * np.log(predicted_survival)
                - (1 - observed_success) * np.log1p(-predicted_survival)
            )
            for match in validation_matches:
                # Keep a separate mean CE and count for each match. These
                # paired values are needed by the one-SE selection rule.
                match_rows = rows.iloc[validation_indices].match_id.eq(match).to_numpy()
                validation_records.append(dict(
                    candidate=model_configuration["name"], match_id=match,
                    n=int(match_rows.sum()), nll=float(validation_state_cost[match_rows].mean()),
                ))
            inner_fit_results.append(dict(
                inner=inner_split_index, candidate=model_configuration["name"], fit=map_fit,
            ))
            pd.DataFrame(validation_records).to_csv(fold_output / "inner_scores.csv", index=False)
            print("PHYSICAL", fold, inner_split_index, model_configuration["name"], flush=True)

    # C. SELECT REGULARIZATION on inner validation, then save the audit trail.
    model_configuration, selection_diagnostics = select_regularization(
        validation_records, model_configurations,
    )
    write_json(fold_output / "selection.json", selection_diagnostics)
    write_json(fold_output / "inner_designs.json", inner_basis_definitions)
    write_json(fold_output / "inner_fits.json", inner_fit_results)

    # D. OUTER REFIT: freeze the selected model/rhos, rebuild bases on all
    # eight development matches, and refit their coefficients from scratch.
    training_indices = np.flatnonzero(rows.match_id.isin(development_matches))
    test_indices = np.flatnonzero(rows.match_id.isin(test_matches))
    # Learn the three bases explicitly, using only this fit's training rows.
    # v(t)=v(0): repeat speed across forecast steps as in the final benchmark.
    training_boundary = covariates["boundary"][training_indices]
    training_speed_by_time = np.broadcast_to(
        covariates["speed"][training_indices, None], training_boundary.shape,
    )
    training_angles = covariates["angle"][training_indices]
    valid_training_angles = covariates["angle_available"][training_indices]
    spline_bases = {
        "speed": build_spline_basis(training_speed_by_time, 5),
        "boundary": build_spline_basis(training_boundary, 5),
        "angle": build_spline_basis(training_angles[valid_training_angles], 4),
    }
    map_fit, predicted_survival, prior_precision = fit_split(
        covariates, training_indices, test_indices, spline_bases, model_configuration,
        seed=10 + 10000 * fold + 900,
    )
    # E. HELD-OUT PREDICTION: p_loss=1-S(1). The constant null probability
    # is the loss frequency of these eight TRAINING matches, not the test pair.
    prediction = rows.iloc[test_indices].copy()
    prediction["p_loss"] = 1 - predicted_survival
    prediction["p_null"] = float(1 - covariates["outcome"][training_indices].mean())
    prediction["fold"] = fold
    prediction.to_parquet(fold_output / "test_predictions.parquet", index=False)
    write_json(fold_output / "fit.json", map_fit)
    write_json(
        fold_output / "model.json",
        export_model(map_fit, spline_bases, model_configuration, development_matches),
    )
    np.savez_compressed(
        fold_output / "posterior.npz", theta=map_fit["theta"], precision=prior_precision,
        covariance=map_fit["covariance"], hessian_eigenvalues=map_fit["hessian_eigenvalues"],
    )
    # Score the test pair only after all choices and fits are frozen. These
    # probabilities are later pooled across five folds for the main benchmark.
    scores, bins = evaluate_predictions(1 - prediction.outcome, prediction.p_loss, prediction.p_null)
    write_json(fold_output / "metrics.json", scores)
    bins.to_csv(fold_output / "calibration.csv", index=False)
    write_json(fold_output / "complete.json", dict(fold=fold, test_matches=test_matches, n=len(test_indices)))


# ============================================================
# 5. FINAL PRODUCTION REFIT ON ALL TEN MATCHES
# ============================================================


def refit_production(rows, covariates, output):
    """Fit deployment parameters/bases on all ten matches after validation.

    This is a production artifact, not a new held-out benchmark result.
    Requires the same smoothing choice in all five accepted outer folds.
    """
    # A. Read the already selected configuration from each benchmark fold.
    # This procedure requires their model/rho choices to agree; otherwise it
    # stops instead of silently choosing a new production configuration.
    fold_configurations = [
        json.loads((Path(output) / f"fold_{fold}" / "model.json").read_text())["candidate"]
        for fold in range(5)
    ]
    if not all(configuration == fold_configurations[0] for configuration in fold_configurations):
        raise ValueError("Fold smoothing choices differ: review the full-data selection before refitting.")
    model_configuration = fold_configurations[0]
    all_observation_indices = np.arange(len(rows))
    # B. Relearn knots/transformations on ALL ten matches for deployment.
    # Learn the three bases explicitly, using only this fit's training rows.
    # v(t)=v(0): repeat speed across forecast steps as in the final benchmark.
    training_boundary = covariates["boundary"][all_observation_indices]
    training_speed_by_time = np.broadcast_to(
        covariates["speed"][all_observation_indices, None], training_boundary.shape,
    )
    training_angles = covariates["angle"][all_observation_indices]
    valid_training_angles = covariates["angle_available"][all_observation_indices]
    spline_bases = {
        "speed": build_spline_basis(training_speed_by_time, 5),
        "boundary": build_spline_basis(training_boundary, 5),
        "angle": build_spline_basis(training_angles[valid_training_angles], 4),
    }
    # C. Estimate the production MAP and Laplace covariance using the same
    # optimizer/priors as before. The fixed seed makes initialization repeatable.
    # fit_split also predicts on these rows, but that return value is discarded:
    # in-sample production predictions are not held-out benchmark evidence.
    production_fit, _, prior_precision = fit_split(
        covariates, all_observation_indices, all_observation_indices,
        spline_bases, model_configuration, seed=20261017,
    )
    # D. Write a separate production artifact; do not install it into PRESS.
    production_output = Path(output) / "production"
    production_output.mkdir(exist_ok=False)
    write_json(
        production_output / "ball_drive_model.json",
        export_model(production_fit, spline_bases, model_configuration,
                     sorted(rows.match_id.unique())),
    )
    write_json(production_output / "fit.json", production_fit)
    np.savez_compressed(
        production_output / "posterior.npz", theta=production_fit["theta"],
        covariance=production_fit["covariance"], precision=prior_precision,
    )


# ============================================================
# 6. SAVE FITTED MODELS AND DIAGNOSTICS
# ============================================================


def write_json(path, value):
    """Save diagnostics/parameters as JSON, converting NumPy arrays to lists."""
    def convert(item):
        # JSON has no NumPy arrays/scalars or Path objects; convert their
        # representation without changing the values stored in the fit.
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, default=convert, indent=2, allow_nan=False) + "\n")


def export_model(fitted, spline_bases, model_configuration, training_matches):
    """Save MAP coefficients, training basis definitions and physical settings.

    Dictionary keys follow the package's model schema. ``spline_bases``
    contains transformations, not observation-specific design matrices.
    """
    # theta/covariance use the optimizer's parameter order. The basis matrices
    # and physical settings let the package reproduce predictions on new frames.
    # Keep the package schema's field names, including its unused interaction
    # metadata, so the public reader can load the artifact directly.
    n_main = 0 if model_configuration["main"] is None else 6
    names = [f"{kind}_nonlinear_{i}" for kind in ["speed", "boundary"] for i in range(3)]
    main_penalty = spline_bases["speed"]["penalty"] + spline_bases["boundary"]["penalty"]
    return dict(
        schema_version=1, model_type="ball_drive_additive", model_id="review_final_ball_drive",
        theta=np.asarray(fitted["theta"]).tolist(),
        design=dict(
            variant="additive", speed_basis=spline_bases["speed"], boundary_basis=spline_bases["boundary"],
            interaction_projection=None, names=names, main_count=6,
            penalty=main_penalty, interaction_rank=3,
        ),
        angle_design=spline_bases["angle"], active_spline_columns=n_main,
        kind="none" if model_configuration["angle"] is None else "individual_angle",
        candidate=model_configuration, physics=PHYSICS,
        training=dict(match_ids=list(training_matches), target="Bernoulli loss within one second", probability_floor=1e-6),
        fit_diagnostics={key: fitted[key] for key in ["objective", "gradient_max", "converged", "laplace_valid"]},
        posterior_covariance=np.asarray(fitted["covariance"]).tolist(),
    )


# ============================================================
# 7. FINAL EVALUATION: METRICS, CALIBRATION AND POOLED HELD-OUT SCORES
# ============================================================


def evaluate_predictions(loss, probability, null_probability):
    """Score fixed predictions; this function does not train or calibrate them.

    loss: observed loss labels (1=loss, 0=success).
    probability: corresponding predicted LOSS probabilities.
    null_probability: loss prevalence from each row's training fold.
    Returns a metric dictionary and the reliability-bin table for plotting.
    """

    # Use success coordinates for CE/bin construction, as in the benchmark;
    # p_success=1-p_loss and y_success=1-y_loss describe the same Bernoulli task.
    loss = np.asarray(loss, int)
    probability = np.asarray(probability, float)
    success_probability = 1 - probability
    success_outcome = 1 - loss
    null = np.broadcast_to(np.asarray(null_probability, float), probability.shape)

    # CE=mean(-y*log(p)-(1-y)*log(1-p)); gain=1-CE_model/CE_null.
    ce = float(log_loss(success_outcome, success_probability, labels=[0, 1]))
    null_ce = float(log_loss(success_outcome, 1 - null, labels=[0, 1]))

    # Quantile bins target roughly equal counts. Tied probabilities can merge
    # cutpoints, so the number of distinct bins can be lower than ten.
    edges = np.unique(np.quantile(success_probability, np.linspace(0, 1, 11)))
    if len(edges) < 2:
        edges = np.array([0.0, 1.0])
    edges[0], edges[-1] = 0.0, 1.0
    bins = []
    for index in range(len(edges) - 1):
        selected = success_probability >= edges[index]
        if index == len(edges) - 2:
            # Include p=1 in the final bin; all other bins are [left,right).
            selected &= success_probability <= edges[index + 1]
        else:
            selected &= success_probability < edges[index + 1]
        if selected.any():
            # Report in loss coordinates for the reliability plot. Absolute
            # calibration error is identical in success/loss coordinates.
            bins.append(dict(
                n=int(selected.sum()),
                predicted_loss=float(probability[selected].mean()),
                observed_loss=float(loss[selected].mean()),
            ))
    table = pd.DataFrame(bins)
    # ECE=sum_bins (n_bin/N)*abs(mean_prediction_bin-observed_rate_bin).
    ece = float((table.n * abs(table.predicted_loss - table.observed_loss)).sum() / len(loss))

    # Brier=mean((y_loss-p_loss)^2). Gain/ECE are returned as fractions:
    # multiply by 100 for percentages in a report.
    scores = dict(
        n=len(loss), losses=int(loss.sum()), cross_entropy=ce,
        null_cross_entropy=null_ce, gain_on_null=1 - ce / null_ce,
        brier=float(np.mean((loss - probability) ** 2)), ece_quantile10=ece,
        mean_predicted_loss=float(probability.mean()), observed_loss=float(loss.mean()),
    )
    if len(np.unique(loss)) == 2:
        # Ranking metrics are defined here only when both classes are present.
        scores["roc_auc"] = float(roc_auc_score(loss, probability))
        scores["pr_auc_loss"] = float(average_precision_score(loss, probability))
    return scores, table


def collect_results(output):
    """Pool each state's single held-out prediction and evaluate it once.

    Also export the physical dispersion estimates from six/eight-match
    fits, so the CatBoost comparison can use the matching training splits.
    """
    parts = []
    splits = {}
    for fold in range(5):
        # Read ONLY saved outer-test predictions. Every match was test in
        # exactly one fold; never mix in production-refit predictions.
        folder = Path(output) / f"fold_{fold}"
        complete = json.loads((folder / "complete.json").read_text())
        parts.append(pd.read_parquet(folder / "test_predictions.parquet"))
        artifact = json.loads((folder / "model.json").read_text())
        fits = json.loads((folder / "inner_fits.json").read_text())
        designs = json.loads((folder / "inner_designs.json").read_text())
        candidate = artifact["candidate"]["name"]
        inner = []
        for item in designs:
            # Recover s=exp(theta[3]) for the selected configuration's inner
            # six-match fit. CatBoost needs that same split-specific s.
            fit = next(f for f in fits if f["inner"] == item["inner"] and f["candidate"] == candidate)
            inner.append(dict(
                fit_matches=item["fitting_matches"], validation_matches=item["validation_matches"],
                s=float(np.exp(fit["fit"]["theta"][3])),
            ))
        splits[str(fold)] = dict(
            outer_s=float(np.exp(artifact["theta"][3])), inner=inner,
            test_matches=complete["test_matches"],
        )
    # Concatenate probabilities and labels, then recompute all metrics once.
    # This is pooled-out scoring, not an unweighted mean of five fold scores.
    pooled = pd.concat(parts, ignore_index=True)
    if pooled.duplicated(["match_id", "frame_start", "frame_end"]).any():
        raise ValueError("A held-out observation appeared more than once.")
    scores, bins = evaluate_predictions(1 - pooled.outcome, pooled.p_loss, pooled.p_null)
    pooled.to_parquet(Path(output) / "pooled_predictions.parquet", index=False)
    write_json(Path(output) / "pooled_metrics.json", scores)
    bins.to_csv(Path(output) / "pooled_calibration.csv", index=False)
    write_json(Path(output) / "physical_splits.json", dict(splits=splits))
    return scores
