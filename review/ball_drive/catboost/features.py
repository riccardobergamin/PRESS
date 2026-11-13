"""Final CatBoost ball-drive benchmark: 50 state-only features and Logloss.

The target is loss within one second. CatBoost optimizes its native Bernoulli
Logloss; there is no custom objective or explicit Bayesian parameter prior.
Leaf L2 regularization is selected in training.py.
Read prepare_features at the end for the input-to-feature workflow.
"""

from itertools import product
from pathlib import Path
import json

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from scipy.special import log_expit

from PRESS.physics.times import time_to_intercept
from review.frame_states import load_tracking_states


TIMES = np.arange(11, dtype=np.float32) * np.float32(0.1)

FEATURE_NAMES = (
    ["bc_speed"]
    + [f"boundary_distance_t{k:02d}" for k in [0, 5, 10]]
    + [f"tti_rank{rank}_t{k:02d}" for k in [0, 5, 10] for rank in [1, 2, 3]]
    + [f"encounter_t{k:02d}" for k in range(11)]
    + [f"{name}_{suffix}" for suffix in ["start", "projected_1s"] for name in [
        "bc_x", "bc_y", "opponent_goal_distance", "opponent_goal_bearing",
    ]]
    + [f"nearest_defender{rank}_{name}" for rank in [1, 2, 3] for name in [
        "distance", "speed", "closing", "transverse", "front_cosine",
    ]]
    + ["defenders_within_2m", "defenders_within_5m", "nearest_two_position_cosine"]
)

PARAMETERS = dict(
    loss_function="Logloss",
    iterations=3000,
    thread_count=4,
    boosting_type="Plain",
    grow_policy="SymmetricTree",
    bootstrap_type="Bayesian",
    bagging_temperature=1.0,
    random_strength=1.0,
    border_count=254,
    nan_mode="Min",
    allow_writing_files=False,
    verbose=False,
    use_best_model=False,
)
CANDIDATES = [
    dict(depth=depth, learning_rate=rate, l2_leaf_reg=penalty)
    for depth, rate, penalty in product([4, 6, 8], [0.02, 0.05, 0.1], [1.0, 3.0, 10.0, 30.0])
]
TREE_CHECKPOINTS = list(range(25, 3001, 25))
SEEDS = [19, 41, 73]


# ============================================================
# 1. CONTEXT KNOWN AT THE INITIAL FRAME
# ============================================================

# ============================================================
# 2. TTI AT 0, 0.1, ..., 1 SECOND
# ============================================================

def calculate_defender_tti(states):
    """Final PRESS TTI: vmax=7, linear/turn acceleration=7, overspeed retained."""

    def one_frame(r, v, x, vx):
        def one_time(t):
            return time_to_intercept(
                r, v, (x + vx * t)[None, :],
                v_max=7.0, a_max=7.0, a_turn=7.0,
                preserve_initial_overspeed=True,
            )[:, 0]
        return jax.vmap(one_time)(jnp.asarray(TIMES))

    arguments = [jnp.asarray(states[key], jnp.float32) for key in ["r_defs", "v_defs", "x_bc", "v_bc"]]
    tti = np.asarray(jax.jit(jax.vmap(one_frame))(*arguments))
    valid = np.linalg.norm(states["r_defs"], axis=-1) < 500
    return tti, valid


# ============================================================
# 3. DETERMINISTIC SPATIAL AND KINEMATIC FEATURES
# ============================================================

def build_spatial_features(states, direction, tti, valid):
    """39 covariates independent of the fitted arrival dispersion s.

    TTI ranks are re-evaluated at each checkpoint. The three defenders used
    for distance/relative-motion descriptors are instead fixed by their
    initial distance to the carrier. Neither ordering uses a future tackler.
    """

    x = states["x_bc"]
    vx = states["v_bc"]
    r = states["r_defs"]
    v = states["v_defs"]
    data = {}

    # The benchmark's scalar carrier speed/context were evaluated in float64.
    data["bc_speed"] = np.linalg.norm(vx.astype(float), axis=1)
    ranked_tti = np.sort(np.where(valid[:, None, :], tti, np.inf), axis=-1)
    for k in [0, 5, 10]:
        projected = x + vx * (k / 10.0)
        data[f"boundary_distance_t{k:02d}"] = np.minimum(
            52.5 - abs(projected[:, 0]), 34.0 - abs(projected[:, 1]),
        )
        for rank in range(3):
            data[f"tti_rank{rank + 1}_t{k:02d}"] = ranked_tti[:, k, rank]

    # Coordinates are oriented toward the opponent's goal. All contexts use
    # the same fixed 105x68 convention as the accepted CatBoost experiment.
    start = x.astype(float) * direction[:, None]
    projected = start + vx.astype(float) * direction[:, None]
    for suffix, position in [("start", start), ("projected_1s", projected)]:
        delta = np.array([52.5, 0.0]) - position
        data[f"bc_x_{suffix}"] = position[:, 0]
        data[f"bc_y_{suffix}"] = position[:, 1]
        data[f"opponent_goal_distance_{suffix}"] = np.linalg.norm(delta, axis=1)
        data[f"opponent_goal_bearing_{suffix}"] = np.arctan2(delta[:, 1], delta[:, 0])

    # These pair descriptors deliberately preserve the final float32 inputs.
    delta = r - x[:, None, :]
    distance = np.linalg.norm(delta, axis=-1)
    nearest = np.argsort(np.where(valid, distance, np.inf), axis=1, kind="stable")[:, :3]
    relative_velocity = v - vx[:, None, :]
    unit = delta / np.maximum(distance[..., None], 1e-12)
    closing = -np.sum(unit * relative_velocity, axis=-1)
    transverse = np.sqrt(np.maximum(np.sum(relative_velocity ** 2, axis=-1) - closing ** 2, 0.0))
    speed = np.linalg.norm(vx, axis=1)
    front = np.sum(unit * vx[:, None, :], axis=-1) / np.maximum(speed[:, None], 1e-12)
    defined = valid & (distance > 1e-8)
    closing = np.where(defined, closing, np.nan)
    transverse = np.where(defined, transverse, np.nan)
    front = np.where(defined & (speed[:, None] >= 1.0), np.clip(front, -1, 1), np.nan)

    values = dict(
        distance=distance, speed=np.linalg.norm(v, axis=-1),
        closing=closing, transverse=transverse, front_cosine=front,
    )
    row = np.arange(len(x))
    for rank in range(3):
        defender = nearest[:, rank]
        for name, array in values.items():
            data[f"nearest_defender{rank + 1}_{name}"] = np.where(
                valid[row, defender], array[row, defender], np.nan,
            )
    for radius in [2, 5]:
        data[f"defenders_within_{radius}m"] = np.sum(valid & (distance <= radius), axis=1)

    first, second = nearest[:, 0], nearest[:, 1]
    cosine = np.sum(unit[row, first] * unit[row, second], axis=1)
    data["nearest_two_position_cosine"] = np.where(
        defined[row, first] & defined[row, second], np.clip(cosine, -1, 1), np.nan,
    )
    result = pd.DataFrame(data)
    if np.isinf(result.to_numpy()).any():
        raise ValueError("Insufficient valid defenders for the declared feature set.")
    return result


# ============================================================
# 4. ENCOUNTER CURVE: s COMES FROM THE MATCHING PHYSICAL FIT
# ============================================================

def build_features(base, tti, valid, s):
    """Add 11 Baker-style encounter probabilities to the deterministic table.

    s is the physical ball-drive dispersion estimated on these fit matches:
    six for inner validation, eight for outer-test feature calculation.
    """

    if not np.isfinite(s) or s <= 0:
        raise ValueError("Arrival dispersion must be positive.")
    z = np.pi * (TIMES.astype(float)[None, :, None] - tti.astype(float)) / (np.sqrt(3.0) * s)
    log_no_encounter = np.where(valid[:, None, :], log_expit(-z), 0.0)
    encounter = -np.expm1(log_no_encounter.sum(axis=-1))
    features = base.copy()
    for k in range(11):
        features[f"encounter_t{k:02d}"] = encounter[:, k]
    return features[FEATURE_NAMES]


# ============================================================
# 5. BERNOULLI LIKELIHOOD (OPTIMIZED NATIVELY BY CATBOOST)
# ============================================================

def bernoulli_nll(loss, probability):
    probability = np.clip(np.asarray(probability, float), 1e-15, 1 - 1e-15)
    loss = np.asarray(loss, int)
    return -loss * np.log(probability) - (1 - loss) * np.log1p(-probability)


# ============================================================
# 6. PREPARE ONCE, THEN LOAD FOR EVERY TRAINING ARM/FOLD
# ============================================================

def prepare_features(data_root, tracking_root, physical_splits, output):
    """Real frames -> defender TTI -> spatial features, saved once.

    Encounter probabilities are added later using the physical s belonging
    to each inner-six or outer-eight fit. This stage never fits CatBoost.
    """
    folder = Path(output) / "features"
    folder.mkdir(parents=True, exist_ok=False)
    rows = pd.read_parquet(Path(data_root) / "rows.parquet").reset_index(drop=True)
    rows["match_id"] = rows.match_id.astype(str)
    rows["row_id"] = np.arange(len(rows))
    if not np.array_equal(rows.loss_observed, rows.outcome.eq(0)):
        raise ValueError("CatBoost loss target disagrees with the physical cohort.")
    states, sides = load_tracking_states(
        rows, tracking_root, frame_column="frame_start", carrier_column="player_id",
    )
    direction = np.where(sides == "left_to_right", 1., -1.)
    tti, valid = calculate_defender_tti(states)
    spatial_features = build_spatial_features(states, direction, tti, valid)
    rows.to_parquet(folder / "rows.parquet", index=False)
    spatial_features.to_parquet(folder / "spatial_features.parquet", index=False)
    np.savez_compressed(folder / "defender_tti.npz", tti=tti, valid=valid)
    splits = json.loads(Path(physical_splits).read_text())["splits"]
    (folder / "physical_splits.json").write_text(json.dumps(splits, indent=2) + "\n")
    return rows, spatial_features, tti, valid, splits


def load_features(output):
    """Read the prepared tables/arrays, without reopening tracking frames."""
    folder = Path(output) / "features"
    rows = pd.read_parquet(folder / "rows.parquet")
    spatial_features = pd.read_parquet(folder / "spatial_features.parquet")
    with np.load(folder / "defender_tti.npz") as arrays:
        tti, valid = arrays["tti"], arrays["valid"]
    splits = json.loads((folder / "physical_splits.json").read_text())
    return rows, spatial_features, tti, valid, splits
