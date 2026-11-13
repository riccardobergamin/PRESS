"""Pass selection: formulas -> tree/rho updates -> validation -> evaluation.

Screening, five-fold validation and production refit follow the same fit loop.
File writing and deployment metadata are collected at the end of this script.
"""


import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as jax_logsumexp
import numpy as np
import pandas as pd
# SciPy minimize estimates scalar rho or beta=1/T; it does not fit trees.
from scipy.optimize import minimize
# XGBoost owns DMatrix (input data) and Booster (tree ensemble).
# Our code supplies the grouped categorical loss, g,h, and the CNL rho update.
import xgboost as xgb

from features import load_pass_tables, write_json


SEED = 20261019
SELECTED_CONFIGURATIONS = {
    "flat": dict(max_depth=4, min_child_weight=20., reg_lambda=1.),
    "cnl": dict(max_depth=6, min_child_weight=5., reg_lambda=3.),
}

# Final screening grid: depth 6 uses L2=3; depth 3 gets 3,000 trees.
SCREENING_FOLD = 0
SCREENING_CONFIGURATIONS = [
    dict(name="01_depth6_child5", max_depth=6, min_child_weight=5., reg_lambda=3., eta=.05, max_rounds=2000),
    dict(name="02_depth3_child5", max_depth=3, min_child_weight=5., reg_lambda=1., eta=.05, max_rounds=3000),
    dict(name="03_depth4_child20", max_depth=4, min_child_weight=20., reg_lambda=1., eta=.05, max_rounds=2000),
    dict(name="04_depth4_child1", max_depth=4, min_child_weight=1., reg_lambda=1., eta=.05, max_rounds=2000),
    dict(name="05_depth4_child5_baseline", max_depth=4, min_child_weight=5., reg_lambda=1., eta=.05, max_rounds=2000),
]


# ============================================================
# 1. CHOICE LIKELIHOOD AND ITS DERIVATIVES
# ============================================================
# NOTATION USED THROUGHOUT THE FILE:
#   i = observed pass; j = candidate destination cell; m = receiver nest.
#   z_ij = covariates of cell j for pass i; u_ij = f_theta(z_ij).
#   y_i = observed destination cell (an index, not a one-hot vector).
#   alpha_ijm = fixed fractional membership, sum_m alpha_ijm=1.
#   rho=1/mu = common inverse dissimilarity; it is estimated only for CNL.
#   p_ij = model probability of choosing cell j; sum_j p_ij=1.
#   ell_i = -log(p_i,y_i); L=sum_i ell_i; reported CE=mean_i ell_i.
# CNL auxiliaries: S_im=sum_j (alpha_ijm*exp(u_ij))**rho;
# w_im=P(m), b_ijm=P(j|m), q_im=P(m|observed y_i).
# One rectangular block has P_b passes, C cells per pass, N=10 receiver nests:
# u,p,g,h: (P_b,C); y,ell: (P_b,); alpha,b: (P_b,C,N); w,q,S: (P_b,N).
# Cells of a pass belong to ONE categorical choice, not separate binary trials.

# THE THREE INPUT TABLES (loaded by features.load_pass_tables):
# candidate_features: MultiIndex (pass_index,destination), C_tot rows x F features.
# observed_choices: index pass_index, P rows, with chosen_cell=y_i and actual pass IDs.
# log_memberships: SAME MultiIndex as candidate_features, C_tot rows x N teammates;
#                 values are log(alpha_ijm). None for flat. Not input features.
# P=passes, C=cells per pass, F=features, N=teammate nests, T=temperature.
# C_tot=sum_i C_i. Inside L/loss_gradient, u is (P,C), y is (P,), alpha is (P,C,N).
# Flattening/memory groups belong only to the XGBoost interface in section 2.

# 1A. FLAT: SOFTMAX -> NLL -> GRADIENT AND CURVATURE

# JAX jit compiles repeated numerical evaluation; this decorator performs no fit.
@jax.jit
def flat_log_probabilities(u):
    """Flat log p_ij=u_ij-log(sum_k exp(u_ik)); input/output shape (P,C)."""
    u = jnp.asarray(u, jnp.float64)
    return u - jax_logsumexp(u, axis=1, keepdims=True)


@jax.jit
def flat_derivatives(u, y):
    """Flat loss, gradient and boosting curvature.

    Input: u (P,C), y (P,) integer destination indices 0<=y_i<C.
    Output: ell (P,), g (P,C), h (P,C), as JAX arrays.
    g_ij=p_ij-1[j=y_i]; each row is one pass, each column a candidate cell.
    """
    u = jnp.asarray(u, jnp.float64)
    # Softmax normalizes across candidate CELLS of each pass, axis=1.
    p = jax.nn.softmax(u, axis=1)
    # JAX .at[i,y_i].add(-1) subtracts 1 at the observed destination of each pass.
    # Starting from p therefore gives g_ij=p_ij-1[j=y_i].
    g = p.at[jnp.arange(len(y)), y].add(-1.)
    C = u.shape[1]
    # Curvature supplied to XGBoost: Friedman factor C/(C-1), floor 1e-6.
    h = jnp.maximum(p * (1. - p) * C / (C - 1.), 1e-6)
    # ell_i=log(sum_j exp(u_ij))-u_i,y_i = -log p_i,y_i.
    # logsumexp evaluates the normalizer without exponentiating large utilities.
    ell = jax_logsumexp(u, axis=1) - u[jnp.arange(len(y)), y]
    return ell, g, h


# 1B. CROSS-NESTED: RECEIVER COMPONENTS -> NLL -> DERIVATIVES

def cross_nested_components(u, log_alpha, rho):
    """CNL components before marginalizing over receiver nests.

    Input: u (P,C), log_alpha (P,C,N), rho scalar; N=10 receiver nests.
    Output: log_w (P,N), log_b and b (P,C,N), as JAX arrays.
    S_m=sum_j (alpha_jm*exp(u_j))**rho; w_m proportional to S_m**(1/rho).
    b_jm=P(cell j|nest m); w_m=P(nest m).
    """
    u = jnp.asarray(u, jnp.float64)
    # Input u: (P,C); log_alpha: (P,C,N). alpha=0 is stored as log_alpha=-inf.
    log_alpha = jnp.asarray(log_alpha, jnp.float64)

    # 1. Normalize memberships across NESTS for each pass and candidate cell:
    # log(alpha_jm) <- log(alpha_jm)-log(sum_l alpha_jl). Shape stays (P,C,N).
    log_alpha -= jax_logsumexp(log_alpha, axis=-1, keepdims=True)

    # axis=1 is the CELL axis. An inactive nest must get final weight zero.
    active_nests = jnp.isfinite(log_alpha).any(axis=1)

    # 2. a_jm=rho*(u_j+log(alpha_jm)). In probability space exp(a)=
    # (alpha_jm*exp(u_j))**rho. Insert the receiver axis: (P,C)->(P,C,1).
    a = rho * (u[:, :, None] + log_alpha)
    # Empty nests use temporary finite values; log_w=-inf removes them below.
    a = jnp.where(active_nests[:, None, :], a, 0.)

    # 3. S_m=sum_j exp(a_jm), the sum over candidate CELLS. log_S: (P,N).
    log_S = jax_logsumexp(a, axis=1)

    # 4. w_m=S_m**(1/rho)/sum_l S_l**(1/rho), the nest probability.
    log_w = jnp.where(active_nests, log_S / rho, -jnp.inf)
    log_w -= jax_logsumexp(log_w, axis=1, keepdims=True)

    # 5. b_jm=exp(a_jm)/S_m=P(cell j | nest m). Shape: (P,C,N), 
    # destination probability conditioned on nest choice.
    log_b = a - log_S[:, None, :]
    b = jnp.where(active_nests[:, None, :], jnp.exp(log_b), 0.)
    return log_w, log_b, b


@jax.jit
def cnl_log_probabilities(u, log_alpha, rho):
    """Marginal CNL log probabilities, log p_ij=log(sum_m w_im*b_ijm).

    Input: u (P,C), log_alpha (P,C,N), rho scalar.
    Output: JAX array (P,C), retaining all candidate destinations.
    """
    log_w, log_b, _ = cross_nested_components(u, log_alpha, rho)
    # log(w_m*b_jm)=log(w_m)+log(b_jm); sum over receiver nests in log space.
    return jax_logsumexp(log_w[:, None, :] + log_b, axis=-1)


@jax.jit
def cnl_derivatives(u, log_alpha, y, rho):
    """Exact CNL gradient and RUMBoost CPU curvature (not the exact Hessian).

    Input: u (P,C), log_alpha (P,C,N), y (P,) integer indices, rho scalar.
    Output: ell (P,), g (P,C), h (P,C), as JAX arrays.
    q_m=w_m*b_y,m/p_y; cbar_j=sum_m q_m*b_jm.
    g_j=p_j+(rho-1)*cbar_j-rho*1[j=y].
    """
    log_w, log_b, b = cross_nested_components(u, log_alpha, rho)

    # 1. Joint probability of observed cell AND nest: w_m*b_y,m, shape (P,N).
    log_joint_y = log_w + log_b[jnp.arange(len(y)), y]
    # p_y=sum_m w_m*b_y,m, so ell=-log(p_y).
    log_p_y = jax_logsumexp(log_joint_y, axis=-1)
    # q_m=(w_m*b_y,m)/p_y. These are algebraic responsibilities, not new parameters.
    q = jnp.exp(log_joint_y - log_p_y[:, None])
    w = jnp.exp(log_w)

    # 2. p_j=sum_m w_m*b_jm; cbar_j=sum_m q_m*b_jm. Both have shape (P,C).
    p = jnp.sum(b * w[:, None, :], axis=-1)
    cbar = jnp.sum(b * q[:, None, :], axis=-1)
    # Exact gradient g_j=p_j+(rho-1)*cbar_j-rho*1[j=y].
    g = p + (rho - 1.) * cbar
    g = g.at[jnp.arange(len(y)), y].add(-rho)

    # 3. Reference curvature, using b, p and q defined above.
    # U_j=sum_m w_m*b_jm**2.
    b2 = b * b
    U = jnp.sum(b2 * w[:, None, :], axis=-1)
    # dp_j=d p_j/d u_j=rho*p_j-p_j**2+(1-rho)*U_j.
    dp = rho * p - p**2 + (1. - rho) * U
    # Expand p_j across receivers: (P,C)->(P,C,1), to combine it with b_jm.
    p3 = p[:, :, None]

    # Q_other(j,m)=-rho**2*b + rho*(b-b**2) + (b-p)**2 - dp_j.
    # Q_y=rho**2*(2b**2-3b+1)+rho*(-3b**2+3b+2*(b*p-p))+(b-p)**2-dp.
    Q_other = (
        -rho**2 * b
        + rho * (-b2 + b)
        + (b - p3)**2
        - dp[:, :, None]
    )
    Q_y = (
        rho**2 * (2 * b2 - 3 * b + 1)
        + rho * (-3 * b2 + 3 * b
                 + 2 * (b * p3 - p3))
        + b2 - 2 * b * p3
        + p3 * p3
        - dp[:, :, None]
    )
    # The reference curvature has one expression at j=y_i and another elsewhere.
    # is_y is (P,C); inserting a final axis applies that choice to every nest.
    is_y = jnp.arange(u.shape[1])[None, :] == y[:, None]
    Q = jnp.where(is_y[:, :, None], Q_y, Q_other)
    # h_ref,j = [g_j**2 - sum_m q_m*Q(j,m)] * C/(C-1).
    # Select Q_y only for j=y, Q_other elsewhere. Sum over NESTS, axis=-1.
    h = g * g - jnp.sum(q[:, None, :] * Q, axis=-1)
    C = u.shape[1]
    # Apply the reference factor after the nest sum; retain the curvature sign.
    h *= C / (C - 1.)
    return -log_p_y, g, h


# 1C. MATHEMATICAL FUNCTIONS: rectangular arrays, no tables/dictionaries
# L selects observed destinations when condition=True, all cells otherwise.
# loss_gradient returns the gradient/curvature matrices used to build a tree.

def L(u, y, rho=1., log_alpha=None, condition=True):
    """Negative log probabilities for P passes with C destinations each.

    u: (P,C) utilities; y: (P,) observed destination indices; rho: scalar.
    log_alpha: (P,C,N) memberships for CNL, or None for the flat model.
    condition=True returns ell_i=-log p_i,y_i, shape (P,).
    condition=False returns -log p_ij for all destinations, shape (P,C).
    """
    # 1. Same output from either model: log p_ij for ALL destinations, (P,C).
    if log_alpha is None:
        log_p = flat_log_probabilities(u)
    else:
        log_p = cnl_log_probabilities(u, log_alpha, rho)

    # 2. One negative log probability per pass/destination pair.
    ell = -log_p                                      # (P,C)

    # 3. Optionally select the observed destination y_i in EACH pass row.
    if condition:
        ell = ell[jnp.arange(len(y)), y]              # (P,)
    return ell  # no sum here: conditioned .mean() gives CE, .sum() gives total NLL


def loss_gradient(u, y, rho=1., log_alpha=None):
    """Gradient and boosting curvature of total NLL=sum_i ell_i.

    Input: u (P,C), y (P,), rho scalar; log_alpha (P,C,N) for CNL or None.
    Output: g,h, each (P,C). All arguments are mathematical arrays/scalars.
    """
    if log_alpha is None:
        _, g, h = flat_derivatives(u, y)
    else:
        _, g, h = cnl_derivatives(u, log_alpha, y, rho)
    return g, h


# ============================================================
# 2. OPTIMIZATION: INITIALIZATION -> ONE STEP -> FULL ENSEMBLE
# ============================================================

def booster_parameters(family, seed, threads, configuration=None):
    """Build a native XGBoost parameter dictionary.

    Input: family string, seed/threads integers; configuration optional dictionary
    of scalar hyperparameters. No numerical arrays are passed here.
    Output: dictionary passed to xgboost.Booster.
    """
    # base_score=0 initializes every utility u_ij to zero. Together with rho=1,
    # this gives p_ij=1/C_i in both families, regardless of fixed memberships.
    # eta: learning rate; reg_lambda: L2 on leaf weights; min_child_weight:
    # minimum summed curvature in a child. hist chooses splits from feature bins.
    # subsample/colsample sample candidate rows/features when building trees.
    parameters = dict(
        **SELECTED_CONFIGURATIONS[family], eta=.05, subsample=.8, colsample_bytree=.8,
        max_bin=256, tree_method="hist", device="cpu", base_score=0.,
        seed=seed, seed_per_iteration=True, nthread=threads, disable_default_eval_metric=1,
    )
    # Override capacity/regularization with the screening candidate or winner.
    # name and max_rounds are experiment metadata, not Booster parameters.
    if configuration is not None:
        parameters.update({name: value for name, value in configuration.items()
                           if name not in ["name", "max_rounds"]})
    return parameters


def build_xgboost_input(candidate_features, observed_choices=None, threads=1):
    """Build XGBoost's DMatrix from the feature table and optional fitting labels.

    Input tables are aligned by pass_index and destination, as loaded together.
    candidate_features: C_tot rows x F numeric feature columns.
    observed_choices: P rows, including chosen_cell=y_i and n_candidates=C_i.
    Output D: native XGBoost container, with X (C_tot,F) and optional labels (C_tot,).
    The third membership table is used by our likelihood, not placed in X.
    """
    # 1. Take ONLY the feature values. The index contains pass/cell identifiers
    # and does not become a predictor. Keep the original feature-column order.
    X = candidate_features.to_numpy(copy=False)       # (C_tot,F)
    feature_names = list(candidate_features.columns) # F strings

    # 2. Fitting labels: mark the observed cell in each pass's candidate rows.
    # XGBoost receives a flat label vector; our custom loss still treats each
    # whole pass as one categorical observation, using y from observed_choices.
    labels = None
    if observed_choices is not None:
        C = observed_choices.n_candidates.to_numpy() # (P,), counts C_i
        y = observed_choices.chosen_cell.to_numpy()  # (P,), local cell indices
        first_row = np.r_[0, np.cumsum(C)[:-1]]       # first candidate row of pass i
        observed_row = first_row + y                 # global row of its chosen cell
        labels = np.zeros(len(X), np.float32)        # (C_tot,)
        labels[observed_row] = 1.
        # Example: C=[3,3], y=[2,0] -> observed rows [2,3]
        # and flat labels [0,0,1, 1,0,0]. No labels are added to X itself.

    # 3. Native XGBoost constructor. For validation/test, labels=None: the
    # observed destinations are supplied separately to our evaluation functions.
    D = xgb.DMatrix(X, label=labels, feature_names=feature_names, nthread=threads)
    return D


def pass_row_ranges(observed_choices, group_size=32):
    """Locate pass rows and their corresponding candidate rows in the tables.

    Input: observed_choices, one row per pass, in the loader's match/pass order.
    Each iteration yields (pass_rows, candidate_rows, C), where the first two
    are Python slices: slice(a,b) selects positions a,...,b-1.
    These groups only limit memory; all passes contribute to the same fit.
    """
    # Two tables have different row counts: one per pass versus one per cell.
    # Offsets locate the first pass/candidate row of the CURRENT match.
    pass_offset = 0
    candidate_offset = 0
    for _, match_choices in observed_choices.groupby("match_id", sort=False):
        P_match = len(match_choices)
        C = int(match_choices.n_candidates.iloc[0])  # same grid throughout a match

        for local_start in range(0, P_match, group_size):
            local_stop = min(local_start + group_size, P_match)
            # First select pass rows; multiplying local positions by C selects
            # their complete candidate rows. Group boundaries never split a pass.
            pass_rows = slice(pass_offset + local_start, pass_offset + local_stop)
            candidate_rows = slice(candidate_offset + local_start*C,
                                   candidate_offset + local_stop*C)
            yield pass_rows, candidate_rows, C

        # The next match starts after all passes/candidate rows of this match.
        pass_offset += P_match
        candidate_offset += P_match*C


def xgboost_losses(u, observed_choices, rho, log_memberships=None, condition=True):
    """Restore pass matrices from XGBoost's flat u, then evaluate L.

    Input: u (C_tot,), observed_choices (P rows), rho scalar;
    optional log_memberships (C_tot,N), indexed in the SAME order as feature rows.
    Output: loss (P,) if condition=True, or (C_tot,) for all candidate cells.
    """
    # 1. Allocate the output and read all observed destination indices.
    if condition:
        losses = np.empty(len(observed_choices), float)  # one value per pass
    else:
        losses = np.empty(len(u), float)                # one value per candidate
    y_all = observed_choices.chosen_cell.to_numpy(dtype=np.int64)  # (P,)

    for pass_rows, candidate_rows, C in pass_row_ranges(observed_choices):
        # 2. Restore a rectangular group of passes from the flat candidate rows.
        y = y_all[pass_rows]                          # (P_b,)
        P_b = len(y)
        u_ij = np.asarray(u[candidate_rows]).reshape(P_b, C)  # (P_b,C)

        # 3. If present, restore the matching membership rows to (P_b,C,N).
        # These come from the third table and never become feature columns.
        log_alpha = None
        if log_memberships is not None:
            N = log_memberships.shape[1]
            values = log_memberships.iloc[candidate_rows].to_numpy(copy=False)
            log_alpha = values.reshape(P_b, C, N)

        # 4. Call the mathematical likelihood. It handles only rectangular arrays.
        ell = L(u_ij, y, rho, log_alpha, condition=condition)

        # 5. Put results back in their original pass/candidate-row order.
        if condition:
            losses[pass_rows] = np.asarray(ell)              # (P_b,)
        else:
            losses[candidate_rows] = np.asarray(ell).ravel() # (P_b*C,)
    return losses


def xgboost_gradients(u, observed_choices, rho, log_memberships=None):
    """Restore pass matrices, evaluate loss_gradient, flatten g,h for XGBoost.

    Same inputs and row ordering as xgboost_losses.
    Output: float32 vectors g,h, each (C_tot,), one pair per candidate row.
    XGBoost's native tree builder consumes these two vectors.
    """
    # 1. XGBoost expects one gradient/curvature per feature row in its DMatrix.
    g = np.empty(len(u), np.float32)
    h = np.empty(len(u), np.float32)
    y_all = observed_choices.chosen_cell.to_numpy(dtype=np.int64)

    for pass_rows, candidate_rows, C in pass_row_ranges(observed_choices):
        # 2. Restore the same (pass,cell) layout used by the likelihood formulas.
        y = y_all[pass_rows]                          # (P_b,)
        P_b = len(y)
        u_ij = np.asarray(u[candidate_rows]).reshape(P_b, C)  # (P_b,C)

        # 3. Recover this group's memberships from the separate third table.
        log_alpha = None
        if log_memberships is not None:
            N = log_memberships.shape[1]
            values = log_memberships.iloc[candidate_rows].to_numpy(copy=False)
            log_alpha = values.reshape(P_b, C, N)

        # 4. Mathematical derivatives: g_ij,h_ij both have shape (P_b,C).
        g_ij, h_ij = loss_gradient(u_ij, y, rho, log_alpha)

        # 5. Flatten in exactly the same order as X. Assignment to the allocated
        # float32 arrays performs the conversion expected by the native objective.
        g[candidate_rows] = np.asarray(g_ij).ravel()
        h[candidate_rows] = np.asarray(h_ij).ravel()
    return g, h


def update_nesting_scale(u, observed_fit, current_rho, memberships_fit=None, conditional_multistart=True):
    """Estimate rho at fixed utilities: rho_new=rho_old+.1*(rho_star-rho_old).

    Input: u (C_fit_tot,), observed_fit DataFrame with P_fit rows; current_rho scalar,
    memberships_fit optional DataFrame (C_fit_tot,N), conditional_multistart boolean.
    Trees are fixed during this optimization.
    Output: updated rho scalar and a dictionary of scalar-fit diagnostics.
    Screening starts at current rho; final fits first scan [1,1.5,2,3,5,10].
    Reject failed/uphill moves. Validation never enters the scalar fit.
    """
    # u stays fixed throughout this scalar optimization. Cache repeated CE(rho)
    # evaluations made by SciPy; this dictionary contains no fitted ensembles.
    cache = {}

    def fit_ce(x):
        # SciPy supplies a one-element parameter vector x; extract its scalar rho.
        rho = float(x[0])
        if rho not in cache:
            cache[rho] = float(xgboost_losses(u, observed_fit, rho, memberships_fit).mean())
        return cache[rho]

    # Start at current rho (screening) or at the best scanned value (final fits).
    before = fit_ce([current_rho])
    starting_values = [float(current_rho)]
    if conditional_multistart:
        starting_values += [1., 1.5, 2., 3., 5., 10.]
    # Remove duplicate starting values, evaluate fit CE at each, choose the best.
    # This chooses the start of ONE SLSQP optimization at the current fixed u.
    profile = [dict(rho=rho, fit_ce=fit_ce([rho])) for rho in dict.fromkeys(starting_values)]
    initial = min(profile, key=lambda point: point["fit_ce"])["rho"]
    # 2. Conditional optimum rho_star from bounded SLSQP, with trees unchanged.
    # scipy.optimize.minimize estimates rho_star in [1,10], using numerical
    # derivatives and an adaptive step. Only fitting labels enter fit_ce.
    optimization = minimize(fit_ce, np.array([initial]), method="SLSQP", bounds=[(1., 10.)])
    proposed_rho = float(optimization.x[0])
    proposed_ce = fit_ce([proposed_rho])
    # Keep the current rho as a fallback. Add the proposed optimum only when
    # SciPy succeeds and the result is finite/in bounds; select the lower CE.
    solutions = [(before, float(current_rho))]
    if optimization.success and np.isfinite(proposed_ce) and 1 <= proposed_rho <= 10:
        solutions.append((proposed_ce, proposed_rho))
    candidate_ce, candidate_rho = min(solutions)
    # 3. Relax in RHO space: rho_next=.9*rho_old+.1*rho_star.
    relaxed_rho = current_rho + .1 * (candidate_rho - current_rho)
    after = fit_ce([relaxed_rho])
    # Accept the relaxed move only if fit CE remains finite and non-increasing.
    # Otherwise leave rho unchanged. Returned diagnostics describe this update.
    accepted = bool(np.isfinite(after) and after <= before + 1e-12)
    updated_rho = float(relaxed_rho if accepted else current_rho)
    return updated_rho, dict(
        previous_rho=float(current_rho), rho=updated_rho, mu=1. / updated_rho,
        candidate_rho=candidate_rho, candidate_mu=1. / candidate_rho,
        fit_ce_before=before, candidate_fit_ce=candidate_ce,
        fit_ce_after=after if accepted else before,
        conditional_scan=profile, slsqp_initial_rho=initial,
        slsqp_success=bool(optimization.success), relaxation=.1, accepted=accepted,
    )


def train_model(D_fit, observed_fit, parameters, *, D_val=None,
                observed_val=None, memberships_fit=None, memberships_val=None,
                max_rounds=2000, patience=100, conditional_multistart=True):
    """Train the tree model in explicit chronological order.

    D_fit/D_val: XGBoost feature matrices (C_tot,F).
    observed_fit/val: observed-choice tables, one row per pass.
    memberships_fit/val: optional log-membership tables (C_tot,N), for CNL.
    Return the selected model, its parameters, tree history and rho updates.
    """
    if (D_val is None) != (observed_val is None):
        raise ValueError("Supply both validation matrix and observed choices, or neither.")

    # INITIAL STATE: no trees, u=0, rho=1 (mu=1). Booster owns the fitted trees.
    cached_matrices = [D_fit] if D_val is None else [D_fit, D_val]
    model = xgb.Booster(params=parameters, cache=cached_matrices)
    rho = 1.
    u_fit = model.predict(D_fit, output_margin=True, training=True)  # (C_fit_tot,)
    use_validation = observed_val is not None
    best_ce, best_tree_count, best_rho = np.inf, 0, 1.
    rounds_without_improvement = 0
    history, scale_updates = [], []
    start_time = time.monotonic()

    for iteration in range(max_rounds):
        n_trees = iteration + 1

        # 1. CURRENT u -> probabilities -> likelihood derivatives g,h.
        # xgboost_gradients restores (P,C) matrices and calls loss_gradient;
        # flat_derivatives/cnl_derivatives contain the probability formulas.
        g, h = xgboost_gradients(u_fit, observed_fit, rho, memberships_fit)

        # 2. BUILD ONE TREE from g,h with native XGBoost.
        # Greedy splits improve the regularized quadratic objective; a leaf A
        # receives w_A=-sum_A(g)/(sum_A(h)+reg_lambda). XGBoost applies eta.
        # boost is the same native operation used by update(fobj=...) after its
        # callback computes g,h; here those arrays are supplied explicitly.
        model.boost(D_fit, iteration=iteration, grad=g, hess=h)

        # 3. NEW UTILITIES: u(z)=sum_{k=1}^{n_trees} eta*b_k(z).
        # predict evaluates the updated ensemble; raw margins are utilities.
        u_fit = model.predict(D_fit, output_margin=True, training=True)

        # 4. CNL ONLY: every 20 trees, hold u fixed and update rho on fitting CE.
        if memberships_fit is not None and n_trees % 20 == 0:
            rho, update = update_nesting_scale(
                u_fit, observed_fit, rho, memberships_fit,
                conditional_multistart=conditional_multistart,
            )
            update["after_tree"] = n_trees
            scale_updates.append(update)

        # 5. VALIDATION: predict with the current trees/rho, then compute native CE.
        # These labels select a checkpoint; they never determine g,h or tree splits.
        validation_ce = None
        if use_validation:
            u_val = model.predict(D_val, output_margin=True)
            validation_ce = float(xgboost_losses(u_val, observed_val, rho, memberships_val).mean())
            if not np.isfinite(validation_ce):
                raise ValueError("Nonfinite validation likelihood.")
            if validation_ce < best_ce:
                best_ce = validation_ce
                best_tree_count = n_trees
                best_rho = rho
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1

        # 6. EARLY STOPPING: stop after patience trees without a new validation minimum.
        # Progress fitting CE is descriptive and does not choose the checkpoint.
        record = dict(round=n_trees, rho=rho, mu=1. / rho, validation_ce=validation_ce)
        history.append(record)
        stopped = use_validation and rounds_without_improvement >= patience
        if n_trees % 50 == 0 or stopped or n_trees == max_rounds:
            record["fit_ce"] = float(xgboost_losses(u_fit, observed_fit, rho, memberships_fit).mean())
            family = "cnl" if memberships_fit is not None else "flat"
            print("TREE", family, n_trees, "fit", record["fit_ce"],
                  "validation", validation_ce, "mu", 1. / rho, flush=True)
        if stopped:
            break

    # 7. RETURN THE SELECTED TREES AND THEIR MATCHING rho.
    # Production has no validation: retain all requested trees and the last rho.
    if not use_validation:
        best_tree_count, best_rho = n_trees, rho
    selected_model = model[:best_tree_count]
    training_summary = dict(
        rounds=best_tree_count, trained_rounds=n_trees, rho=best_rho, mu=1. / best_rho,
        parameters=parameters, seconds=time.monotonic() - start_time,
        early_stopping=use_validation, conditional_multistart=conditional_multistart,
    )
    if use_validation:
        training_summary["validation_ce"] = best_ce
    return selected_model, training_summary, history, scale_updates


# ============================================================
# 3. TEMPERATURE: CALIBRATE THE SELECTED, FROZEN CHOICE DISTRIBUTION
# ============================================================

def temperature_losses(log_p, C, y, beta=1., gradient=False):
    """Evaluate p_T,ij=p_ij**beta/sum_k p_ik**beta, with beta=1/T.

    Input: log_p (C_tot,) in candidate-row order; C (P,) candidate counts and
    y (P,) integer destination indices, with C_tot=sum_i C_i and 0<=y_i<C_i.
    beta is a scalar; gradient is a boolean.
    Output: ell (P,) when gradient=False; otherwise (mean CE, dCE/dbeta),
    two scalars. ell_i=log(sum_j exp(beta*log_p_ij))-beta*log_p_i,y_i.
    """
    C = np.asarray(C, np.int64)
    y = np.asarray(y, np.int64)
    # C_i is the number of destinations in pass i. starts locates that pass
    # inside the flattened log_p array; y_i is a LOCAL destination index.
    starts = np.r_[0, np.cumsum(C)[:-1]]
    log_p = np.asarray(log_p, float)
    if len(log_p) != C.sum() or beta < 0 or not np.isfinite(log_p).all():
        raise ValueError("Invalid grouped log probabilities.")
    beta_log_p = beta * log_p
    # Stable log-sum-exp: a_i=max_j(beta*l_ij),
    # log Z_i=a_i+log(sum_j exp(beta*l_ij-a_i)). The shift cancels exactly.
    # reduceat operates separately on each pass segment. Here it finds the
    # largest beta*log_p in each choice set; repeat broadcasts it to its cells.
    a = np.maximum.reduceat(beta_log_p, starts)
    exponentials = np.exp(beta_log_p - np.repeat(a, C))
    # One shifted exponential sum per pass. The shift a cancels in ell_i.
    Z = np.add.reduceat(exponentials, starts)
    ell = a + np.log(Z) - beta * log_p[starts + y]
    if not gradient:
        return ell
    # Analytic derivative: E_{p_T}[log p] - log p_observed, averaged over passes.
    # Return (mean CE, derivative) when the optimizer requests gradient=True.
    dCE_dbeta = np.add.reduceat(exponentials * log_p, starts) / Z - log_p[starts + y]
    return float(ell.mean()), float(dCE_dbeta.mean())


def fit_temperature(log_p, C, y):
    """Fit the single scalar beta=1/T with L-BFGS-B, leaving trees/rho fixed.

    Input: log_p (C_tot,), C (P,) candidate counts, y (P,) integer choices.
    C_tot=sum_i C_i. All arrays describe the same P passes in the same order.
    Output: calibration dictionary, containing scalar beta, T and CE diagnostics.
    Start beta=1, constrain beta>=0; beta=0 denotes the uniform limit.
    """
    before = float(temperature_losses(log_p, C, y).mean())
    # SciPy L-BFGS-B estimates the single beta=1/T. jac=True means our function
    # returns both CE and its analytic derivative. Start beta=1 (T=1); constrain
    # beta>=0. This optimization leaves trees, rho and memberships fixed.
    optimization = minimize(
        lambda x: temperature_losses(log_p, C, y, float(x[0]), gradient=True),
        np.array([1.]), jac=True, method="L-BFGS-B", bounds=[(0., None)],
        options=dict(maxiter=200, gtol=1e-8, ftol=1e-12),
    )
    beta = float(optimization.x[0])
    after, dCE_dbeta = temperature_losses(log_p, C, y, beta, gradient=True)
    # Interior optimum: derivative=0. At beta=0, a positive derivative is valid
    # because the constraint forbids a further move to negative beta.
    projected_gradient = abs(dCE_dbeta) if beta > 1e-10 else max(-dCE_dbeta, 0.)
    if not (optimization.success and np.isfinite(beta) and projected_gradient < 1e-5 and after <= before + 1e-10):
        raise ValueError(f"Temperature fit did not converge: {optimization.message}")
    # beta=0 is the uniform-distribution limit T=infinity, recorded explicitly.
    # The returned dictionary contains the fitted calibrator and its CE change.
    return dict(method="Temperature on final choice distribution", inverse_temperature=beta,
                temperature=None if beta == 0 else 1. / beta, uniform_limit=beta == 0,
                validation_ce_before=before, validation_ce_after=after,
                projected_gradient=projected_gradient, trees_refitted=False, rho_refitted=False)


# ============================================================
# 4. COMPLETE WORKFLOW: SCREENING -> FIVE 6+2+2 FOLDS -> PRODUCTION
# ============================================================

def compare_configurations(features_folder, split, family, output, threads=1,
                  configuration_name=None, tree_limit=None):
    """Five fresh fits on six fitting and two validation matches; native CE only.

    The loader returns three tables: candidate features, observed choices, and
    log memberships (None for flat). Test matches are not opened here.
    """
    X_fit, observed_fit, memberships_fit = load_pass_tables(features_folder, split["fit_matches"], family)
    X_val, observed_val, memberships_val = load_pass_tables(features_folder, split["validation_matches"], family)
    D_fit = build_xgboost_input(X_fit, observed_fit, threads)
    D_val = build_xgboost_input(X_val, threads=threads)
    del X_fit, X_val

    configurations = SCREENING_CONFIGURATIONS
    if configuration_name is not None:
        configurations = [c for c in configurations if c["name"] == configuration_name]
        if not configurations:
            raise ValueError("Unknown screening configuration.")

    records = []
    for configuration in configurations:
        folder = output / "screening" / family / configuration["name"]
        if folder.exists():
            raise FileExistsError(folder)
        parameters = booster_parameters(family, SEED, threads, configuration)
        K = configuration["max_rounds"] if tree_limit is None else tree_limit
        model, summary, history, updates = train_model(
            D_fit, observed_fit, parameters, D_val=D_val, observed_val=observed_val,
            memberships_fit=memberships_fit, memberships_val=memberships_val,
            max_rounds=K, conditional_multistart=False,
        )
        summary.update(
            family=family, fold=SCREENING_FOLD, configuration=configuration,
            fit_matches=split["fit_matches"], validation_matches=split["validation_matches"],
            validation_ce=float(xgboost_losses(model.predict(D_val, output_margin=True), observed_val,
                                               summary["rho"], memberships_val).mean()),
            fit_ce=float(xgboost_losses(model.predict(D_fit, output_margin=True), observed_fit,
                                       summary["rho"], memberships_fit).mean()),
            max_rounds=K, verification_only=tree_limit is not None,
            temperature_scaling=False, test_evaluated=False,
        )
        save_fit(folder, dict(selection=summary, history=history, scale_updates=updates))
        records.append(summary)
    return records


def select_best_configuration(output, family):
    """Select the lowest native validation CE separately for each family.

    Input: output Path and family string, locating the five saved summaries.
    Output: winning hyperparameter dictionary. The five-row comparison is saved
    as comparison.csv. No feature arrays or models are fitted here.
    """
    folder = output / "screening" / family
    if (folder / "selection.json").exists():
        raise FileExistsError(folder / "selection.json")
    records = []
    # Read one completed summary for each of the five candidates in this family.
    for configuration in SCREENING_CONFIGURATIONS:
        summary = json.loads((folder / configuration["name"] / "selection.json").read_text())
        if summary["verification_only"]:
            raise ValueError("Screening selection requires full native-validation fits.")
        records.append(dict(
            configuration=configuration["name"], max_depth=configuration["max_depth"],
            min_child_weight=configuration["min_child_weight"], reg_lambda=configuration["reg_lambda"],
            eta=configuration["eta"], max_rounds=summary["max_rounds"],
            best_round=summary["rounds"], trained_rounds=summary["trained_rounds"],
            validation_ce=summary["validation_ce"], fit_ce=summary["fit_ce"],
            rho=summary["rho"], mu=summary["mu"],
        ))
    # Rank by lowest native validation CE. Exact ties prefer lower depth, then
    # higher child weight, then candidate name; choose separately for flat/CNL.
    comparison = pd.DataFrame(records).sort_values(
        ["validation_ce", "max_depth", "min_child_weight", "configuration"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    # The first sorted row is the winner. Differences are descriptive CE gaps.
    comparison["delta_from_best"] = comparison.validation_ce - comparison.validation_ce.iloc[0]
    winner = next(c for c in SCREENING_CONFIGURATIONS if c["name"] == comparison.configuration.iloc[0])
    # Write the comparison and winning settings for the subsequent five-fold fit.
    comparison.to_csv(folder / "comparison.csv", index=False)
    write_json(folder / "selection.json", dict(
        family=family, fold=SCREENING_FOLD, configuration=winner,
        criterion="lowest native validation CE within each family",
    ))
    return dict(winner)


def train_validate_test(features_folder, split, family, fold, output, threads=1, max_rounds=2000, configuration=None):
    """6 fit + 2 validation + 2 test, using the same explicit three-table inputs.

    X: candidate feature DataFrame (C_tot,F), indexed by (pass_index,destination).
    observed: P rows, observed destination and real pass IDs.
    memberships: DataFrame (C_tot,N), same index as X; None for flat.
    Return the NLL table of the two held-out test matches.
    """
    folder = output / family / "training" / f"fold_{fold}"
    if folder.exists():
        raise FileExistsError(folder)

    # 1. Load the tables and convert only the feature columns to XGBoost inputs.
    X_fit, observed_fit, memberships_fit = load_pass_tables(features_folder, split["fit_matches"], family)
    X_val, observed_val, memberships_val = load_pass_tables(features_folder, split["validation_matches"], family)
    D_fit = build_xgboost_input(X_fit, observed_fit, threads)
    D_val = build_xgboost_input(X_val, threads=threads)
    del X_fit, X_val

    # 2. Select the joint (tree prefix,rho) checkpoint by native validation CE.
    parameters = booster_parameters(family, SEED + fold, threads, configuration)
    model, summary, history, updates = train_model(
        D_fit, observed_fit, parameters, D_val=D_val, observed_val=observed_val,
        memberships_fit=memberships_fit, memberships_val=memberships_val,
        max_rounds=max_rounds,
    )
    rho = summary["rho"]

    # 3. Freeze trees/rho; estimate T on the complete validation distributions.
    log_p_val = -xgboost_losses(model.predict(D_val, output_margin=True), observed_val,
                               rho, memberships_val, condition=False)
    calibration = fit_temperature(log_p_val, observed_val.n_candidates, observed_val.chosen_cell)
    calibration["fit_matches"] = split["validation_matches"]

    # 4. Open the two outer-test matches only after fitting/calibration.
    X_test, observed_test, memberships_test = load_pass_tables(features_folder, split["test_matches"], family)
    D_test = build_xgboost_input(X_test, threads=threads)
    del X_test
    test, log_p_test = compute_pass_losses(
        model, D_test, observed_test, rho, calibration["inverse_temperature"], memberships_test,
    )
    test["fold"] = fold

    # 5. Save results; feature vocabulary is metadata, outside the likelihood.
    specification = json.loads((features_folder / "feature_spec.json").read_text())
    save_fit(folder, dict(
        selection=summary | dict(family=family, fold=fold, **split),
        temperature=calibration, history=history, scale_updates=updates,
        feature_spec=dict(names=D_fit.feature_names, vocabulary=specification["vocabulary"]),
    ), model=model, test=test, log_p=log_p_test)
    save_feature_importance(model, folder)
    return test


def train_production_model(features_folder, data_root, output, family, threads=1):
    """Fresh ten-match fit: three input tables, median tree count, OOF temperature."""
    folder = output / family / "production"
    if folder.exists():
        raise FileExistsError(folder)
    protocol = json.loads((data_root / "protocol.json").read_text())
    selections, tables, log_p_oof = [], [], []
    for fold in range(5):
        fitted = output / family / "training" / f"fold_{fold}"
        selections.append(json.loads((fitted / "selection.json").read_text()))
        tables.append(pd.read_parquet(fitted / "test_predictions.parquet"))
        log_p_oof.append(np.load(fitted / "test_log_probabilities.npy"))

    # One raw outer-OOF distribution per pass, for the deployment temperature.
    held_out = pd.concat(tables, ignore_index=True)
    calibration = fit_temperature(np.concatenate(log_p_oof), held_out.n_candidates, held_out.chosen_cell)
    calibration["source"] = "raw outer-OOF choice distributions"
    K = int(np.median([s["rounds"] for s in selections]))
    names = ["max_depth", "min_child_weight", "reg_lambda", "eta"]
    configuration = {name: selections[0]["parameters"][name] for name in names}
    if any(any(s["parameters"][name] != value for name,value in configuration.items()) for s in selections):
        raise ValueError("Use the same family configuration in all five folds.")

    # All ten matches; fresh u=0,rho=1; fixed K and no early stopping.
    X, observed, memberships = load_pass_tables(features_folder, protocol["matches"], family)
    D = build_xgboost_input(X, observed, threads)
    del X
    model, summary, history, updates = train_model(
        D, observed, booster_parameters(family, SEED, threads, configuration),
        memberships_fit=memberships, max_rounds=K,
    )
    specification = json.loads((features_folder / "feature_spec.json").read_text())
    artifact = save_production_model(
        folder, model, summary, calibration, observed, specification, data_root,
        protocol["matches"], family,
    )
    write_json(folder / "history.json", history)
    write_json(folder / "scale_updates.json", updates)
    save_feature_importance(model, folder)
    collect_feature_importance(output, family)
    return artifact


# ============================================================
# 5. FINAL EVALUATION: PREDICTIONS, METRICS AND POOLED TEST SCORES
# ============================================================

def compute_pass_losses(model, D, observed_choices, rho, beta, log_memberships=None):
    """Per-pass null/native/calibrated NLL from a frozen fitted model.

    D: candidate-feature DMatrix (C_tot,F); observed_choices: P-row label table;
    log_memberships: optional third table (C_tot,N). rho,beta=1/T are scalars.
    Return a P-row NLL table and full native log p (C_tot,).
    """
    u = model.predict(D, output_margin=True)
    log_p = -xgboost_losses(u, observed_choices, rho, log_memberships, condition=False)
    pass_scores = observed_choices[[
        "match_id", "pass_id", "frame_end", "outcome", "chosen_cell", "n_candidates",
    ]].copy()
    y, C = pass_scores.chosen_cell, pass_scores.n_candidates
    pass_scores["nll_null"] = np.log(C)
    pass_scores["nll_raw"] = temperature_losses(log_p, C, y)
    pass_scores["nll_temperature"] = temperature_losses(log_p, C, y, beta)
    return pass_scores.reset_index(drop=True), log_p


def compute_metrics(pass_scores):
    """Aggregate per-pass losses into native/calibrated CE and null-relative gain.

    Input: pass_scores DataFrame with P rows and NLL columns nll_null,
    nll_raw, nll_temperature. Each column is a length-P vector.
    Output: list of 2 dictionaries (native and temperature), with scalar metrics.
    The number of candidate cells does not change the weight of a pass.
    """
    CE_null = float(pass_scores.nll_null.mean())  # mean_i log C_i
    metrics = []
    for version, loss_column in [("native", "nll_raw"), ("temperature", "nll_temperature")]:
        CE = float(pass_scores[loss_column].mean())  # mean_i[-log p_i,y_i]
        gain = 1. - CE / CE_null
        metrics.append(dict(
            calibration=version, n=len(pass_scores), cross_entropy=CE,
            null_cross_entropy=CE_null, gain_over_null=gain,
        ))
    return metrics


def compute_pooled_metrics(output, family, cohort):
    """Pool the five outer-test score tables to calculate the final benchmark.

    Input: output Path, family string; cohort DataFrame with P_all accepted
    passes and ID columns match_id, pass_id, frame_end. Each stored fold table
    has P_test_fold rows. Their total must cover those P_all passes exactly once.
    Output: pooled metrics DataFrame (2,5), one row each for native/calibrated p.
    No model is fitted or recalibrated during this aggregation.
    """
    # 1. Each fold supplies predictions for its two test matches only.
    test_tables = []
    for fold in range(5):
        fold_folder = output / family / "training" / f"fold_{fold}"
        test_tables.append(pd.read_parquet(fold_folder / "test_predictions.parquet"))
    held_out_scores = pd.concat(test_tables, ignore_index=True)

    # 2. Every accepted pass must occur exactly once among the five outer tests.
    id_columns = ["match_id", "pass_id", "frame_end"]
    predicted_ids = held_out_scores[id_columns].astype(str).sort_values(id_columns).reset_index(drop=True)
    cohort_ids = cohort[id_columns].astype(str).sort_values(id_columns).reset_index(drop=True)
    if held_out_scores.duplicated(id_columns).any() or not predicted_ids.equals(cohort_ids):
        raise ValueError("Pooled test set must cover every accepted pass exactly once.")

    # 3. Main benchmark: compute CE on ALL held-out passes together.
    # CE_pooled=sum_f N_f*CE_f/sum_f N_f; gain=1-CE_pooled/CE_null,pooled.
    # Each pass has weight one, so this is not an unweighted average of fold gains.
    pooled_metrics = pd.DataFrame(compute_metrics(held_out_scores))

    # 4. Descriptive breakdowns: apply the same calculation per fold and per match.
    breakdowns = {}
    for group_column in ["fold", "match_id"]:
        group_metrics = []
        for group_id, group_scores in held_out_scores.groupby(group_column):
            for metric in compute_metrics(group_scores):
                group_metrics.append({group_column: group_id, **metric})
        breakdowns[group_column] = pd.DataFrame(group_metrics)

    # 5. Save this evaluation once; existing fitted trees/predictions remain fixed.
    scores_folder = output / family / "scores"
    scores_folder.mkdir(parents=True, exist_ok=True)
    if (scores_folder / "pooled_predictions.parquet").exists():
        raise FileExistsError("Pooled scores already exist; choose a new output folder.")
    held_out_scores.to_parquet(scores_folder / "pooled_predictions.parquet", index=False)
    pooled_metrics.to_csv(scores_folder / "pooled_metrics.csv", index=False)
    for group_column, group_metrics in breakdowns.items():
        group_metrics.to_csv(scores_folder / f"metrics_by_{group_column}.csv", index=False)
    collect_feature_importance(output, family)  # summarize gain tables from existing models
    return pooled_metrics


# ============================================================
# 6. FEATURE IMPORTANCE FROM THE SELECTED ENSEMBLES
# ============================================================

def save_feature_importance(ensemble, folder):
    """Normalize total split gain, grouping one-hot columns by variable.

    Input: ensemble is a fitted Booster with F feature names; folder is a Path.
    Output: grouped importance DataFrame (G,4), with G<=F variable groups and
    columns feature, total_gain, gain_share, rank. Also save column-wise (F,4)
    and grouped tables. Gain describes the training splits, not causal effects.
    """
    # For each split s, XGBoost stores its gain: improvement in the local
    # quadratic objective built from g,h and leaf regularization during training.
    # I_k=sum_{s using feature k} gain_s; importance_k=I_k/sum_l I_l.
    # Temperature acts on final probabilities and leaves these tree gains unchanged.
    # Native Booster.get_score returns the SUM of split gains per feature.
    # Include unused feature columns below with gain=0.
    total_gain = ensemble.get_score(importance_type="total_gain")
    columns = pd.DataFrame(dict(
        feature=ensemble.feature_names,
        total_gain=[total_gain.get(name, 0.) for name in ensemble.feature_names],
    ))
    # Normalize each feature's total gain by the ensemble-wide total.
    # A constant ensemble with no splits gets zero shares.
    total = columns.total_gain.sum()
    columns["gain_share"] = columns.total_gain / total if total > 0 else 0.
    columns = columns.sort_values(["gain_share", "feature"], ascending=[False, True]).reset_index(drop=True)
    columns["rank"] = np.arange(1, len(columns) + 1)
    columns.to_csv(folder / "importance_columns.csv", index=False)
    # Collapse one-hot names such as start_type=pass_reception to start_type.
    # Sum their total gains, then normalize by the same ensemble-wide total.
    grouped = columns.assign(feature=columns.feature.str.split("=", n=1).str[0])
    grouped = grouped.groupby("feature", sort=True).total_gain.sum().reset_index()
    grouped["gain_share"] = grouped.total_gain / total if total > 0 else 0.
    grouped = grouped.sort_values(["gain_share", "feature"], ascending=[False, True]).reset_index(drop=True)
    grouped["rank"] = np.arange(1, len(grouped) + 1)
    grouped.to_csv(folder / "importance_grouped.csv", index=False)
    return grouped


def collect_feature_importance(output, family):
    """Summarize fold importance, displaying production importance separately.

    Input: output Path and family string. Read five grouped-importance tables,
    each with one row per variable group, and optionally the production table.
    Output: DataFrame with one row per group: mean/min/max share over folds,
    number of folds, and production share/gain when available.
    """
    tables = []
    for fold in range(5):
        path = output / family / "training" / f"fold_{fold}" / "importance_grouped.csv"
        table = pd.read_csv(path)
        table["fold"] = fold
        tables.append(table)
    folds = pd.concat(tables, ignore_index=True)
    # For each feature, summarize its gain share over the five selected models.
    # Normalize each ensemble separately, then take the arithmetic mean of shares.
    # Min/max describe fold variability; they are not confidence intervals.
    summary = folds.groupby("feature").agg(
        fold_mean_share=("gain_share", "mean"), fold_min_share=("gain_share", "min"),
        fold_max_share=("gain_share", "max"), n_folds=("fold", "nunique"),
    ).reset_index()
    production_path = output / family / "production" / "importance_grouped.csv"
    # Append production importance separately when the full-data fit exists.
    # Its different training sample/tree count can put it outside the fold range.
    if production_path.is_file():
        production = pd.read_csv(production_path)[["feature", "gain_share", "total_gain"]]
        production = production.rename(columns={"gain_share": "production_share", "total_gain": "production_total_gain"})
        summary = summary.merge(production, on="feature", how="outer")
    folder = output / family / "scores"
    folder.mkdir(parents=True, exist_ok=True)
    folds.to_csv(folder / "importance_by_fold.csv", index=False)
    summary.to_csv(folder / "importance_summary.csv", index=False)
    return summary


# 7. OUTPUT FILES — not part of the likelihood or learning procedure

def save_fit(folder, records, model=None, test=None, log_p=None):
    """Write completed fit results once, refusing to overwrite an existing folder.

    Input: folder Path; records dict mapping file names to JSON-compatible data.
    Optional model is a Booster, test a DataFrame with P_test rows, and log_p
    a flattened vector (C_test_tot,) of full native test log probabilities.
    Output: files only; return None. Screening omits model/test/log_p.
    """
    # Create one NEW output folder for this fit. An existing folder raises an
    # error immediately, preserving previous results.
    folder.mkdir(parents=True, exist_ok=False)
    if model is not None:
        model.save_model(folder / "selected_model.ubj")
    # records maps filenames to JSON content: selection, temperature, history,
    # etc. Each is written once after the fit, not after every tree.
    for name, value in records.items():
        write_json(folder / f"{name}.json", value)
    if test is not None:
        test.to_parquet(folder / "test_predictions.parquet", index=False)
    # Retain full native test log p because production T needs all alternatives,
    # not only the observed-destination NLL stored in the prediction table.
    if log_p is not None:
        np.save(folder / "test_log_probabilities.npy", log_p)


def save_production_model(folder, model, summary, calibration, observed_choices, specification, data_root, matches, family):
    """Export the fitted model and its dependencies in the PRESS artifact format.

    Input: folder/data_root Paths; fitted model Booster with F feature names;
    summary/calibration dictionaries; observed_choices with P_all rows and feature specification;
    matches list of ten IDs and family string. Numeric estimates are scalars.
    Output: JSON-compatible metadata dictionary; save the Booster in a .ubj file
    and record its feature order, vocabulary, xT and physical-model dependencies.
    """
    folder.mkdir(parents=True, exist_ok=False)
    filename = "pass_selection_model" if family == "flat" else "pass_selection_cnl_model"
    # The .ubj file stores XGBoost trees. The accompanying JSON stores rho, T,
    # feature definitions and physical dependencies needed to evaluate them.
    booster_path = folder / (filename + ".ubj")
    model.save_model(booster_path)
    upstream = data_root / "upstream" / "production"
    # Package metadata: identify the model/tree file and its checksum; preserve
    # feature order, category vocabulary, xT and the fitted calibrator.
    # Embed pass-risk/ball-drive weights so prediction uses consistent physics.
    artifact = dict(
        schema_version=1, model_type="pass_destination_choice", model_id=f"review_pass_selection_{family}",
        family=family, feature_contract="ahead_of_carrier_and_target_v1", booster_file=booster_path.name,
        booster_sha256=hashlib.sha256(booster_path.read_bytes()).hexdigest(),
        rho=summary["rho"], mu=summary["mu"], feature_names=specification["names"],
        vocabulary=specification["vocabulary"], xt_grid=np.load(upstream / "xt.npy").tolist(), calibration=calibration,
        training=summary | dict(match_ids=matches, n_choices=len(observed_choices),
                                tree_count_rule="median of five selected outer-fold prefixes",
                                initial_utilities=0., initial_rho=1., post_training_rho_refit=False),
        physics=dict(pass_risk=json.loads((upstream / "pass_risk_model.json").read_text()),
                     ball_drive=json.loads((upstream / "ball_drive_model.json").read_text())),
        preferred_for_package=family == "flat",
    )
    write_json(folder / (filename + ".json"), artifact)
    return artifact
