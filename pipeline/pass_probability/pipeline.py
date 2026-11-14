"""Run the final pass-probability workflow in explicit, independent stages.

Reading order: model.py -> training.py -> this file.
Input is the FINAL SELECTED pass cohort, not raw dynamical events. Association
checks and selection stay upstream; this rewrite introduces no new exclusions.
"""

import argparse
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from model import (
    PHYSICAL_SETTINGS, estimate_launch_conditions,
    mean_velocity_trajectory, forecast_position_error,
    prepare_pitch_control_inputs, trajectory_arrival_times, precision_context,
)
from training import run_fold, collect_results, refit_production, write_json

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.frame_states import load_tracking_states


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PIPELINE_ROOT / "inputs" / "pass_probability"
DEFAULT_TRACKING = PIPELINE_ROOT / "inputs" / "shared" / "tracking"
DEFAULT_OUTPUT = PIPELINE_ROOT / "outputs" / "pass_probability"


TEST_MATCHES = {
    0: ["1925299", "2015213"],
    1: ["2006229", "2011166"],
    2: ["1899585", "1953632"],
    3: ["1886347", "2013725"],
    4: ["1996435", "2017461"],
}
MODEL_INPUT_NAMES = [
    "endpoint_arrivals", "ball_arrival_time", "valid_players", "path_tti", "ball_low",
]


def load_selected_cohort(data_root, tracking_root=DEFAULT_TRACKING):
    """Read accepted rows and reconstruct release states through get_pv_players.

    Only the observed destination is loaded from the archived arrays. It is
    the training label/flight endpoint, not a future player state.
    """
    pass_table = pd.read_parquet(data_root / "rows.parquet")
    pass_table["match_id"] = pass_table.match_id.astype(str)
    keys = ["match_id", "pass_id", "frame_end"]
    if pass_table.duplicated(keys).any() or not pass_table.outcome.isin([0, 1]).all():
        raise ValueError("Invalid selected pass keys/outcomes.")
    if set(pass_table.match_id) != set(sum(TEST_MATCHES.values(), [])):
        raise ValueError("Input matches differ from the accepted five test pairs.")
    states, _ = load_tracking_states(pass_table, tracking_root, frame_column="frame_end", carrier_column="player")
    with np.load(data_root / "arrays.npz") as stored:
        states["r_target"] = stored["r_target"]
    if states["p_pos"].shape != (len(pass_table), 22, 2) or states["p_vel"].shape != (len(pass_table), 22, 2):
        raise ValueError("Expected 22-player initial states aligned to rows.")
    if any(len(values) != len(pass_table) or not np.isfinite(values).all() for values in states.values()):
        raise ValueError("Nonfinite or misaligned accepted initial states.")
    return pass_table, states


def generate_trajectories(data_root, output, tracking_root=DEFAULT_TRACKING):
    """Compare two paths per pass against tracking; save ONLY the chosen path.

    Selection uses mean 3D forecast error on the same observed flight frames.
    With no valid comparison frames, prefer the position fit if usable.
    No trajectory-method choice is repeated during lambda,s training.
    """
    pass_table, initial_states = load_selected_cohort(data_root, tracking_root)
    trajectories_folder = output / "trajectories"
    trajectories_folder.mkdir(parents=True, exist_ok=False)
    pass_keys = ["match_id", "pass_id", "frame_end", "outcome"]

    for match_id in pass_table.match_id.unique():
        match_mask = pass_table.match_id.eq(match_id).to_numpy()
        match_passes = pass_table.loc[match_mask]
        target_positions = initial_states["r_target"][match_mask]
        tracking = pd.read_parquet(
            tracking_root / "matches" / match_id / f"{match_id}_tracking.parquet"
        )
        trajectories = []
        fit_records = []
        for pass_record, target in zip(match_passes.itertuples(index=False), target_positions):
            # 1. Produce the position-fit candidate (or its fallback).
            fitted_path, diagnostics = estimate_launch_conditions(
                tracking, pass_record.frame_end, target,
            )

            # 2. Produce the mean-velocity candidate from the same five frames.
            window = tracking.loc[
                tracking.frame.between(pass_record.frame_end, pass_record.frame_end + 4)
            ].sort_values("frame")
            velocities = window[["ball_vx", "ball_vy", "ball_vz"]].to_numpy(float)
            release_position = window[["ball_x", "ball_y", "ball_z"]].iloc[0].to_numpy(float)
            mean_path, mean_parameters = mean_velocity_trajectory(velocities, release_position)

            # 3. Compare the mean position errors against the actual ball path.
            fit_error, n_frames = forecast_position_error(
                fitted_path, tracking, pass_record.frame_end, pass_record.frame_reception,
            )
            mean_error, _ = forecast_position_error(
                mean_path, tracking, pass_record.frame_end, pass_record.frame_reception,
            )
            comparison_available = np.isfinite(fit_error) and np.isfinite(mean_error)
            use_mean = (not diagnostics["fit_usable"]
                        or (comparison_available and mean_error < fit_error))
            if use_mean:
                chosen_path = mean_path
                chosen_method = "mean_velocity"
            else:
                chosen_path = fitted_path
                chosen_method = "fit_positions"

            # Save one path per pass. Retain the errors/method as diagnostics.
            trajectories.append(chosen_path)
            record = {key: getattr(pass_record, key) for key in pass_keys}
            record.update(diagnostics)
            if use_mean:
                v0, theta, phi = mean_parameters
                record.update(v0=v0, theta=theta, phi=phi)
            record.update(
                trajectory_method=chosen_method,
                fit_positions_forecast_error=fit_error,
                mean_velocity_forecast_error=mean_error,
                forecast_points=n_frames,
            )
            fit_records.append(record)

        match_folder = trajectories_folder / match_id
        match_folder.mkdir()
        np.save(match_folder / "trajectory.npy", np.asarray(trajectories))
        pd.DataFrame(fit_records).to_parquet(match_folder / "diagnostics.parquet", index=False)
        print("TRAJECTORIES", match_id, len(match_passes), flush=True)

    write_json(trajectories_folder / "complete.json", dict(
        n=len(pass_table), trajectory_method="per_pass_forecast_error",
        no_comparison_rule="position_fit_if_usable_else_mean_velocity",
        launch_source="reconstructed from tracking",
    ))
    return trajectories_folder


def prepare_features(data_root, trajectories_folder, output, tracking_root=DEFAULT_TRACKING):
    """Load the already chosen paths and release states; calculate model inputs."""
    if not (trajectories_folder / "complete.json").is_file():
        raise ValueError("Run generate_trajectories before prepare_features.")
    pass_table, initial_states = load_selected_cohort(data_root, tracking_root)
    features_folder = output / "features"
    features_folder.mkdir(parents=True, exist_ok=False)

    n_passes = len(pass_table)
    model_inputs = dict(
        endpoint_arrivals=np.empty((n_passes, 22), np.float32),
        ball_arrival_time=np.empty(n_passes, np.float32),
        valid_players=np.empty((n_passes, 22), bool),
        path_tti=np.empty((n_passes, 251, 22), np.float32),
        ball_low=np.empty((n_passes, 251), bool),
    )
    pass_keys = ["match_id", "pass_id", "frame_end", "outcome"]
    for match_id in pass_table.match_id.unique():
        # 1. Select release states and load one generated path per pass.
        match_mask = pass_table.match_id.eq(match_id).to_numpy()
        match_passes = pass_table.loc[match_mask].reset_index(drop=True)
        match_states = {name: values[match_mask] for name, values in initial_states.items()}
        match_folder = trajectories_folder / match_id
        diagnostics = pd.read_parquet(match_folder / "diagnostics.parquet")
        if not match_passes[pass_keys].equals(diagnostics[pass_keys].reset_index(drop=True)):
            raise ValueError(f"Trajectory row order does not match initial states: {match_id}")
        ball_path = np.load(match_folder / "trajectory.npy")
        if ball_path.shape != (len(match_passes), 251, 3) or not np.isfinite(ball_path).all():
            raise ValueError(f"Unusable generated trajectory: {match_id}")

        # 2. At the destination: player TTI, hypothetical ball time and mask.
        tti_target, ball_time, valid_players = prepare_pitch_control_inputs(match_states)
        model_inputs["endpoint_arrivals"][match_mask] = tti_target
        model_inputs["ball_arrival_time"][match_mask] = ball_time
        model_inputs["valid_players"][match_mask] = valid_players

        # 3. Along the already chosen path: player TTI and height-control mask.
        with precision_context(False):
            tti_path = trajectory_arrival_times(
                jnp.asarray(match_states["p_pos"], jnp.float32),
                jnp.asarray(match_states["p_vel"], jnp.float32),
                jnp.asarray(ball_path[:, :, :2], jnp.float32),
            )
        model_inputs["path_tti"][match_mask] = np.asarray(tti_path)
        model_inputs["ball_low"][match_mask] = ball_path[:, :, 2] <= 2.0
        print("FEATURES", match_id, len(match_passes), flush=True)

    # 4. Save aligned inputs. Trajectory-fit diagnostics stay in trajectories/.
    pass_table.to_parquet(features_folder / "rows.parquet", index=False)
    for name, values in model_inputs.items():
        np.save(features_folder / f"{name}.npy", values)
    write_json(features_folder / "complete.json", dict(
        n=n_passes, losses=int(pass_table.outcome.eq(0).sum()), physics=PHYSICAL_SETTINGS,
        trajectory_method="per_pass_forecast_error", launch_source="reconstructed from tracking",
    ))


def load_features(output):
    """Load the pass table and its aligned precomputed model inputs.

    Every input array's first axis follows pass_table row order. Arrays are
    opened read-only; neither trajectories nor TTI are recalculated here.
    """
    folder = output / "features"
    if not (folder / "complete.json").is_file():
        raise ValueError("The feature stage must finish before training/refitting.")
    pass_table = pd.read_parquet(folder / "rows.parquet")
    pass_table["match_id"] = pass_table.match_id.astype(str)
    model_inputs = {name: np.load(folder / f"{name}.npy", mmap_mode="r") for name in MODEL_INPUT_NAMES}
    return pass_table, model_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["features", "train", "scores", "refit", "all"])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_INPUT,
                        help="Accepted cohort (default: pipeline/inputs/pass_probability).")
    parser.add_argument("--tracking-root", type=Path, default=DEFAULT_TRACKING,
                        help="Tracking for trajectory estimation (default: pipeline/inputs/shared/tracking).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="NEW output folder (default: pipeline/outputs/pass_probability).")
    parser.add_argument("--fold", type=int, choices=range(5), help="Only with --stage train; omit for all folds.")
    args = parser.parse_args()
    if args.fold is not None and args.stage != "train":
        parser.error("--fold only applies to --stage train.")
    if args.stage in ["features", "all"] and args.data_root is None:
        parser.error("--data-root is required for feature construction.")
    jax.config.update("jax_enable_x64", True)
    if args.stage in ["features", "all"]:
        trajectories_folder = generate_trajectories(
            args.data_root, args.output, args.tracking_root,
        )
        prepare_features(args.data_root, trajectories_folder, args.output, args.tracking_root)
    pass_table, model_inputs = load_features(args.output)
    if args.stage in ["train", "all"]:
        folds = range(5) if args.fold is None else [args.fold]
        for fold in folds:
            run_fold(pass_table, model_inputs, fold, TEST_MATCHES[fold], args.output)
    if args.stage in ["scores", "all"]:
        print(collect_results(pass_table, args.output).to_string(index=False), flush=True)
    if args.stage in ["refit", "all"]:
        pooled = pd.read_parquet(args.output / "scores" / "pooled_predictions.parquet")
        refit_production(pass_table, model_inputs, pooled, args.output)


if __name__ == "__main__":
    main()
