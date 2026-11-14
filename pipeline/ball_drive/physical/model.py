"""Final physical ball-drive model: covariates, spline bases, likelihood, priors.

No classes or pipeline dependencies. Shapes used below:
  N = initial frames; T = 25 forecast steps; D = defenders.
  outcome = 1 means the one-second carry succeeds, 0 means loss within 1 s.

Reading order:
  prepare_covariates: initial tracking state -> projected physical covariates.
  build_spline_basis: one training covariate -> knots and basis transformations.
  prepare_model_data: apply those bases to training, validation or test values.
  build_prior_precision: smoothing strengths -> Gaussian prior precisions.
  predict_survival / negative_log_posterior: prediction / MAP objective.

Spline notation in the comments:
  C = raw basis, L = [1,z], H = residualized basis, G = empirical Gram matrix,
  W = whitening matrix, R = curvature matrix, Q = curvature rotation.
"""

import numpy as np
import jax
import jax.numpy as jnp

from PRESS.physics.times import time_to_intercept


DT = 0.04
TIMES = np.arange(25, dtype=np.float32) * np.float32(DT)

PHYSICS = dict(
    v_max=7.0,
    a_linear=7.0,
    a_turn=7.0,
    preserve_initial_overspeed=True,
    pitch_length=105.0,
    pitch_width=68.0,
    boundary_mode="signed",
    dt=DT,
    integration_method="exponential",
)


# ============================================================
# 1. INITIAL-STATE COVARIATES
# ============================================================

def prepare_covariates(tracking_state, outcome):
    """Project the carrier and evaluate defender arrivals on the 1 s horizon.

    tracking_state contains defender positions/velocities (N,D,2), and
    carrier position/velocity (N,2). outcome contains the N success labels.
    Returns speed (N), boundary distance (N,T), and defender-specific
    arrival times, approach angles and validity masks (N,T,D).
    All future positions below are projections of the initial state.
    """

    # Preserve the final benchmark's float32 arrival geometry.
    defender_positions32 = jnp.asarray(tracking_state["r_defs"], dtype=jnp.float32)
    defender_velocities32 = jnp.asarray(tracking_state["v_defs"], dtype=jnp.float32)
    carrier_position32 = jnp.asarray(tracking_state["x_bc"], dtype=jnp.float32)
    carrier_velocity32 = jnp.asarray(tracking_state["v_bc"], dtype=jnp.float32)

    def arrivals_for_frame(defender_positions, defender_velocities,
                           carrier_position, carrier_velocity):
        """One initial frame: evaluate each defender against x_bc(t)."""
        def arrivals_at_time(t):
            # x_bc(t) = x_bc(0) + v_bc(0)*t; TTI_i(t) targets this position.
            projected_carrier_position = (carrier_position + carrier_velocity * t)[None, :]
            return time_to_intercept(
                defender_positions, defender_velocities, projected_carrier_position,
                v_max=7.0,
                a_max=7.0,
                a_turn=7.0,
                preserve_initial_overspeed=True,
            )[:, 0]

        return jax.vmap(arrivals_at_time)(jnp.asarray(TIMES))

    # vmap repeats the calculation over times and initial frames.
    defender_arrival_times = jax.jit(jax.vmap(arrivals_for_frame))(
        defender_positions32, defender_velocities32,
        carrier_position32, carrier_velocity32,
    )

    # The saved speed covariate was calculated from the float32 tracking state.
    carrier_speed = np.linalg.norm(
        np.asarray(tracking_state["v_bc"], np.float32), axis=1,
    ).astype(float)
    carrier_position = np.asarray(tracking_state["x_bc"], float)
    carrier_velocity = np.asarray(tracking_state["v_bc"], float)
    defender_positions = np.asarray(tracking_state["r_defs"], float)
    defender_velocities = np.asarray(tracking_state["v_defs"], float)
    projected_carrier_position = (
        carrier_position[:, None, :]
        + carrier_velocity[:, None, :] * TIMES[None, :, None]
    )

    # B(t) = min(L/2-x(t), L/2+x(t), W/2-y(t), W/2+y(t)).
    # Signed distance: positive inside the pitch, negative beyond a boundary.
    boundary_distance = np.minimum.reduce([
        52.5 - projected_carrier_position[..., 0],
        52.5 + projected_carrier_position[..., 0],
        34.0 - projected_carrier_position[..., 1],
        34.0 + projected_carrier_position[..., 1],
    ])

    # Approach angle: 0=co-moving, pi=head-on. Below 1 m/s carrier heading
    # is unreliable, so the entire angle effect is omitted.
    approach_vector = projected_carrier_position[:, :, None, :] - defender_positions[:, None, :, :]
    defender_distance = np.linalg.norm(approach_vector, axis=-1)
    # u_i(t) points from defender i to the projected carrier. At contact,
    # use the defender velocity because the positional direction is unstable.
    approach_direction = np.where(
        (defender_distance <= 0.1)[..., None],
        defender_velocities[:, None, :, :], approach_vector,
    )
    approach_norm = np.linalg.norm(approach_direction, axis=-1)
    carrier_heading_speed = np.linalg.norm(carrier_velocity, axis=1)
    # phi_i(t) = arccos( u_i(t).v_bc / (||u_i(t)||*||v_bc||) ).
    approach_cosine = np.sum(approach_direction * carrier_velocity[:, None, None, :], axis=-1)
    approach_cosine /= np.maximum(approach_norm * carrier_heading_speed[:, None, None], 1e-12)
    approach_angle = np.arccos(np.clip(approach_cosine, -1, 1))
    angle_available = (
        (carrier_heading_speed[:, None, None] >= 1.0)
        & (approach_norm > 1e-8)
        & (np.linalg.norm(defender_positions, axis=-1)[:, None, :] < 500)
    )

    return dict(
        outcome=np.asarray(outcome, float),
        speed=carrier_speed,
        boundary=boundary_distance,
        angle=approach_angle,
        angle_available=angle_available,
        tti=np.asarray(defender_arrival_times),
    )


# ============================================================
# 2. RESTRICTED CUBIC SPLINES: TRAINING-ONLY TRANSFORMATIONS
# ============================================================

def restricted_cubic_basis(scaled_values, knots, derivative=0):
    """Evaluate raw restricted-cubic columns or their derivatives in z.

    K knots give K-2 nonlinear columns. With q_k(z)=(z-knot_k)_+^3,
    C_k(z)=q_k(z)-a_k*q_{K-1}(z)+b_k*q_K(z), where the tail weights
    below enforce linear tails. derivative=0/1/2 returns C/C'/C''.
    """

    scaled_values = np.asarray(scaled_values, float)[..., None]
    knots = np.asarray(knots, float)
    derivative_factor = [1.0, 3.0, 6.0][derivative]
    # q_k^(d)(z) = factor_d * (z-knot_k)_+^(3-d), factors = 1,3,6.
    truncated_power_terms = (
        derivative_factor * np.maximum(scaled_values - knots, 0.0) ** (3 - derivative)
    )
    right_tail_weights = (knots[-1] - knots[:-2]) / (knots[-1] - knots[-2])
    left_tail_weights = (knots[-2] - knots[:-2]) / (knots[-1] - knots[-2])
    return (
        truncated_power_terms[..., :-2]
        - truncated_power_terms[..., -2, None] * right_tail_weights
        + truncated_power_terms[..., -1, None] * left_tail_weights
    )


def build_spline_basis(training_values, n_knots):
    """Learn one spline basis from training values; no rate coefficients yet.

    Returns knots and transformations to evaluate the SAME basis on any set.
    Steps: rescale -> raw cubics -> remove constant/linear projection ->
    whiten -> rotate to diagonal curvature -> normalize curvature eigenvalues.
    Saved dictionary keys retain the package's production artifact format.
    """

    training_values = np.asarray(training_values, float).ravel()
    if not np.isfinite(training_values).all() or len(np.unique(training_values)) < n_knots:
        raise ValueError("Insufficient distinct finite training values.")

    # 1. z = (x-min_train)/(max_train-min_train); knots = training quantiles.
    training_minimum = float(training_values.min())
    training_range = float(training_values.max() - training_minimum)
    scaled_training_values = (training_values - training_minimum) / training_range
    knots = np.quantile(scaled_training_values, np.linspace(0, 1, n_knots))
    if np.any(np.diff(knots) <= 1e-8):
        raise ValueError("Tied training knots.")

    # 2. C = raw cubic columns, L = [1,z], A = argmin_A ||C-LA||_F^2.
    # H = C-LA has L.T@H=0 on the training observations.
    raw_basis = restricted_cubic_basis(scaled_training_values, knots)
    linear_design = np.column_stack([np.ones(len(scaled_training_values)), scaled_training_values])
    linear_projection = np.linalg.lstsq(linear_design, raw_basis, rcond=None)[0]
    residual_basis = raw_basis - linear_design @ linear_projection
    # 3. G=H.T@H/n=U diag(g) U.T; W=U diag(g^(-1/2)).
    # Z=H@W then satisfies Z.T@Z/n=I: unit variance, orthogonal columns.
    gram_matrix = residual_basis.T @ residual_basis / len(residual_basis)
    gram_eigenvalues, gram_eigenvectors = np.linalg.eigh(gram_matrix)
    if gram_eigenvalues.min() <= 1e-13:
        raise ValueError("Rank-deficient nonlinear basis.")
    whitening_matrix = gram_eigenvectors @ np.diag(1 / np.sqrt(gram_eigenvalues))

    # 4. R_kl = integral C_k''(z)*C_l''(z) dz.
    # Subtracting LA does not change second derivatives, so H''=C''.
    # Two Gauss-Legendre nodes integrate these quadratic products exactly
    # within each knot interval, apart from floating-point roundoff.
    quadrature_nodes, quadrature_weights = np.polynomial.legendre.leggauss(2)
    curvature_matrix = np.zeros_like(gram_matrix)
    for interval_start, interval_end in zip(knots[:-1], knots[1:]):
        # Map nodes u in [-1,1] to z=(left+right)/2+(right-left)*u/2.
        quadrature_points = (
            (interval_start + interval_end) / 2
            + (interval_end - interval_start) * quadrature_nodes / 2
        )
        second_derivatives = restricted_cubic_basis(quadrature_points, knots, derivative=2)
        # R += C''(z).T @ diag(quadrature_weight * dz/du) @ C''(z).
        curvature_matrix += second_derivatives.T @ (
            (quadrature_weights * (interval_end - interval_start) / 2)[:, None]
            * second_derivatives
        )

    # 5. W.T@R@W = Q diag(eta) Q.T. Final basis B=H@W@Q keeps
    # B.T@B/n=I, with integral (B(z)@gamma)''^2 dz = sum eta_k*gamma_k^2.
    curvature_eigenvalues, curvature_rotation = np.linalg.eigh(
        whitening_matrix.T @ curvature_matrix @ whitening_matrix,
    )
    if curvature_eigenvalues.min() <= 0:
        raise ValueError("Nonpositive curvature penalty.")

    # Store A and WQ for evaluation; Omega=diag(eta/min(eta)) is the
    # normalized curvature precision used later when choosing rho.
    # mean(z),sd(z) are used only if the linear effect is standardized (angle).
    return dict(
        offset=training_minimum,
        scale=training_range,
        knots=knots.tolist(),
        projection=linear_projection.tolist(),
        transform=(whitening_matrix @ curvature_rotation).tolist(),
        penalty=(curvature_eigenvalues / curvature_eigenvalues.min()).tolist(),
        linear_mean=float(scaled_training_values.mean()),
        linear_sd=float(scaled_training_values.std()),
    )


def evaluate_spline_basis(covariate_values, fitted_basis, include_linear=False):
    """Evaluate a saved training basis on training, validation or test values.

    No knots, projections, means or scales are learned in this function.
    Output: same observation axes as covariate_values, plus K-2 basis columns.
    include_linear=True prepends the standardized linear column (angle only).
    """

    # Use TRAINING bounds: z=(x-min_train)/range_train. New z may be outside
    # [0,1]; restricted-cubic linear tails provide the extrapolation.
    scaled_values = (np.asarray(covariate_values, float) - fitted_basis["offset"]) / fitted_basis["scale"]
    linear_design = np.stack([np.ones_like(scaled_values), scaled_values], axis=-1)
    nonlinear_columns = restricted_cubic_basis(scaled_values, fitted_basis["knots"])
    # H_new=C(z_new)-[1,z_new]@A_train; B_new=H_new@W_train@Q_train.
    nonlinear_columns -= linear_design @ np.asarray(fitted_basis["projection"])
    nonlinear_columns = nonlinear_columns @ np.asarray(fitted_basis["transform"])
    if include_linear:
        # (z-mean_train(z))/sd_train(z) = (x-mean_train(x))/sd_train(x).
        standardized_linear_column = (scaled_values - fitted_basis["linear_mean"]) / fitted_basis["linear_sd"]
        return np.concatenate([standardized_linear_column[..., None], nonlinear_columns], axis=-1)
    return nonlinear_columns


def prepare_model_data(covariates, observation_indices, spline_bases,
                       use_main_splines=True, use_angle=True):
    """Apply fitted bases and assemble arrays for the likelihood/prediction.

    Use this same function for training, validation and test. ``spline_bases``
    contains three definitions built by build_spline_basis on TRAINING values.
    The three calls are written explicitly in training.py.
    This function does not fit coefficients or build priors.

    N = selected states, T = forecast steps, D = defenders.
    main_matrix: (N,T,6), or zero columns for linear speed/boundary effects.
    angle_matrix: (N,T,D,3), or zero columns when the angle is excluded.
    outcome is carried alongside the covariates for likelihood/evaluation;
    predict_survival uses only the covariates and fitted coefficients.
    """

    carrier_speed = covariates["speed"][observation_indices]
    boundary_distance = covariates["boundary"][observation_indices]
    n_states = len(carrier_speed)

    # Three nonlinear speed columns followed by three boundary columns.
    # M_bc(t) = [B_v(v), B_boundary(B(t))]. These are only nonlinear columns;
    # the ordinary linear speed/boundary terms remain separate in model_data.
    if not use_main_splines:
        carrier_spline_columns = np.empty((n_states, len(TIMES), 0))
    else:
        carrier_speed_by_time = np.broadcast_to(carrier_speed[:, None], boundary_distance.shape)
        carrier_spline_columns = np.concatenate([
            evaluate_spline_basis(carrier_speed_by_time, spline_bases["speed"]),
            evaluate_spline_basis(boundary_distance, spline_bases["boundary"]),
        ], axis=-1)

    # Each defender gets its own angle effect: one standardized linear
    # column followed by two nonlinear columns: M_angle=[z_phi, B_phi(phi)].
    # Unavailable headings make
    # the entire angle contribution zero, including the linear column.
    if not use_angle:
        angle_effect_columns = np.empty((*covariates["angle"][observation_indices].shape, 0))
    else:
        angle_effect_columns = evaluate_spline_basis(
            covariates["angle"][observation_indices], spline_bases["angle"], include_linear=True,
        )
        angle_effect_columns *= covariates["angle_available"][observation_indices, ..., None]

    model_data = dict(
        speed=carrier_speed,
        boundary=boundary_distance,
        tti=covariates["tti"][observation_indices],
        outcome=covariates["outcome"][observation_indices],
        main_matrix=carrier_spline_columns,
        angle_matrix=angle_effect_columns,
    )
    return {name: jnp.asarray(value, dtype=jnp.float64) for name, value in model_data.items()}


# ============================================================
# 3. SURVIVAL WITH COMPETING DEFENDERS AND EXPONENTIAL EULER
# ============================================================

@jax.jit
def predict_survival(parameters, model_data):
    """Evaluate P(success within 1 s) for already prepared observation arrays.

    Parameter order: log(lambda0), beta_v, beta_B, log(s), then carrier
    nonlinear coefficients and angle coefficients, in matrix-column order.
    Returns one success probability per initial state; loss = 1-success.
    No outcome labels or prior precisions enter the prediction.
    """

    n_carrier_spline_terms = model_data["main_matrix"].shape[-1]
    # Positive parameters are optimized on their log scale.
    # lambda0=exp(parameters[0]); s=exp(parameters[3]).
    lambda_0 = jnp.exp(parameters[0])
    arrival_dispersion = jnp.exp(parameters[3])

    # g_bc(t) = beta_v*v + beta_B*B(t) + M_bc(t)@gamma_bc.
    carrier_log_rate_effect = (
        parameters[1] * model_data["speed"][:, None]
        + parameters[2] * model_data["boundary"]
        + model_data["main_matrix"] @ parameters[4:4 + n_carrier_spline_terms]
    )
    tackle_rate = lambda_0 * jnp.exp(carrier_log_rate_effect)
    # g_angle_i(t)=M_angle_i(t)@gamma_angle.
    # lambda_i(t)=lambda0*exp(g_bc(t)+g_angle_i(t)).
    angle_log_rate_effect = model_data["angle_matrix"] @ parameters[4 + n_carrier_spline_terms:]
    tackle_rate = tackle_rate[..., None] * jnp.exp(angle_log_rate_effect)

    # F_i(t) = sigmoid(pi*(t-TTI_i(t))/(sqrt(3)*s)).
    arrival_probability = jax.nn.sigmoid(
        jnp.pi * (jnp.asarray(TIMES)[None, :, None] - model_data["tti"])
        / (jnp.sqrt(3.0) * arrival_dispersion)
    )
    # r_i(t)=F_i(t)*lambda_i(t); competing rate r_total(t)=sum_i r_i(t).
    defender_loss_rates = arrival_probability * tackle_rate

    def step(log_survival, defender_rates_at_time):
        """One exponential-Euler step with rates frozen at the left endpoint."""
        total_loss_rate = defender_rates_at_time.sum(axis=1)
        # S(t+dt)=S(t)*exp(-dt*r_total(t)); update in log coordinates.
        return log_survival - DT * total_loss_rate, None

    # S(0)=1, hence log(S(0))=0. Scan chronologically across forecast steps.
    initial_log_survival = jnp.zeros(len(model_data["speed"]))
    log_survival, _ = jax.lax.scan(
        step, initial_log_survival, jnp.swapaxes(defender_loss_rates, 0, 1),
    )
    # S(1)=exp(-dt*sum_t sum_i r_i(t)); keep the benchmark Bernoulli floor.
    return jnp.clip(jnp.exp(log_survival), 1e-6, 1 - 1e-6)


# ============================================================
# 4. BERNOULLI LIKELIHOOD AND PRIORS
# ============================================================

def build_prior_precision(spline_bases, candidate):
    """Construct Gaussian prior precisions for the optional effect blocks.

    This is used for MAP fitting, not for validation/test prediction.
    The basis-specific curvature penalties come from the training design.
    candidate['main'] is the shared speed/boundary rho; candidate['angle']
    is the angle rho. None excludes the corresponding block.

    Coefficient order must match prepare_model_data: speed nonlinear,
    boundary nonlinear, angle linear, angle nonlinear. The four baseline
    parameter priors are fixed separately in negative_log_prior below.
    """

    prior_precision = np.empty(0)
    if candidate["main"] is not None:
        # p_k=rho_main*Omega_kk for the six carrier nonlinear coefficients.
        prior_precision = candidate["main"] * np.r_[
            spline_bases["speed"]["penalty"], spline_bases["boundary"]["penalty"],
        ]
    if candidate["angle"] is not None:
        # p_angle_linear=1; p_angle_nonlinear=rho_angle*Omega_angle_kk.
        # gamma_k ~ N(0, 1/p_k). Larger precision means stronger shrinkage.
        prior_precision = np.r_[
            prior_precision, 1.0, candidate["angle"] * np.asarray(spline_bases["angle"]["penalty"]),
        ]
    return np.asarray(prior_precision)


@jax.jit
def negative_log_likelihood(parameters, model_data):
    """Bernoulli NLL summed over states; y=1 means successful carry."""
    success_probability = predict_survival(parameters, model_data)
    observed_success = model_data["outcome"]
    # -log L = -sum_n [y_n*log(S_n) + (1-y_n)*log(1-S_n)].
    return -jnp.sum(
        observed_success * jnp.log(success_probability)
        + (1 - observed_success) * jnp.log1p(-success_probability)
    )


@jax.jit
def negative_log_prior(parameters, prior_precision):
    """Final priors, up to parameter-independent constants.

    log(lambda0), beta_v, beta_B ~ Normal(0,1).
    s ~ Gamma(shape=2.5, rate=5), represented by ell=log(s).
    The log-coordinate Jacobian gives 5*exp(ell) - 2.5*ell.
    Spline coefficients have diagonal Gaussian precision supplied above;
    the standardized linear angle coefficient has precision 1.
    """

    # Normal(0,1): -log p = (log(lambda0)^2 + beta_v^2 + beta_B^2)/2 + const.
    linear_prior_cost = 0.5 * jnp.sum(parameters[:3] ** 2)
    # Gamma(2.5,rate=5) with ell=log(s), INCLUDING ds/dell=exp(ell):
    # -log p(ell) = 5*exp(ell) - 2.5*ell + const.
    dispersion_prior_cost = 5 * jnp.exp(parameters[3]) - 2.5 * parameters[3]
    # Independent Gaussian coefficients: -log p(gamma)=sum_k p_k*gamma_k^2/2.
    spline_prior_cost = 0.5 * jnp.sum(prior_precision * parameters[4:] ** 2)
    return linear_prior_cost + dispersion_prior_cost + spline_prior_cost


@jax.jit
def negative_log_posterior(parameters, model_data, prior_precision):
    """MAP minimizes -log posterior = -log likelihood -log prior + const."""
    return (
        negative_log_likelihood(parameters, model_data)
        + negative_log_prior(parameters, prior_precision)
    )
