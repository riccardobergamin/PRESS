"""Exercise real physical-model inputs, short optimization and serialization.

Run one task per process so the two deliberately simple model.py/training.py
modules do not share Python's unqualified module cache. No complete CV grid.
"""

import argparse
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from review.io import new_folder, write_json


def ball_drive(output):
    sys.path.insert(0, str(ROOT / "review/ball_drive/physical"))
    import model
    import training
    rows = pd.read_parquet(ROOT / "review/inputs/ball_drive/rows.parquet")
    # Diverse states plus actual losses; basis construction needs variability.
    take = np.unique(np.r_[np.linspace(0,len(rows)-1,32,dtype=int), np.flatnonzero(rows.outcome.eq(0))[:8]])
    with np.load(ROOT / "review/inputs/ball_drive/arrays.npz") as a:
        states = {k:a[k][take] for k in ["x_bc","v_bc","r_defs","v_defs"]}
    covariates = model.prepare_covariates(states, rows.outcome.to_numpy()[take])
    boundary, speed = covariates["boundary"], covariates["speed"]
    repeated_speed = np.broadcast_to(speed[:,None], boundary.shape)
    bases = dict(speed=model.build_spline_basis(repeated_speed,5),
        boundary=model.build_spline_basis(boundary,5),
        angle=model.build_spline_basis(covariates["angle"][covariates["angle_available"]],4))
    candidate = dict(main=1.,angle=100.)
    data = model.prepare_model_data(covariates, np.arange(len(take)), bases)
    precision = model.build_prior_precision(bases, candidate)
    assert data["main_matrix"].shape[-1] == 6 and data["angle_matrix"].shape[-1] == 3
    assert np.all(np.asarray(data["angle_matrix"])[~covariates["angle_available"]] == 0)
    # A single small MAP, not the 16-configuration nested search. The prior,
    # likelihood, optimizer, Laplace calculation and export are the real ones.
    fitted = training.fit_map(data, precision, seed=20261030, initializations=0, maxiter=150)
    prediction = np.asarray(model.predict_survival(jnp.asarray(fitted["theta"]), data))
    assert np.isfinite(prediction).all() and np.all((prediction>=0)&(prediction<=1))
    artifact = training.export_model(fitted,bases,candidate,sorted(rows.iloc[take].match_id.astype(str).unique()))
    write_json(output / "ball_drive_model.json",artifact)
    from PRESS.model import BallDriveModel
    loaded = BallDriveModel.from_json(output / "ball_drive_model.json")
    from_package = loaded.predict_batch(**states)["survival_probability"]
    # Both use float32 arrival geometry, but one constructs the time vector
    # in float64 before conversion and the package constructs it in float32.
    # Report the actual error; allow two float32 probability epsilons here.
    np.testing.assert_allclose(prediction,from_package,atol=2*np.finfo(np.float32).eps,rtol=0)
    metrics,bins=training.evaluate_predictions(1-rows.outcome.to_numpy()[take],1-prediction,
                                               float(rows.outcome.eq(0).mean()))
    bins.to_csv(output / "calibration.csv",index=False)
    return dict(states=len(take),spline_shapes_checked=True,map_and_laplace=True,
                export_prediction_max_difference=float(np.max(abs(prediction-from_package))),metrics=metrics)


def pass_probability(output):
    sys.path.insert(0,str(ROOT / "review/pass_probability"))
    import model
    import training
    rows=pd.read_parquet(ROOT / "review/inputs/pass_probability/rows.parquet")
    rows["match_id"]=rows.match_id.astype(str)
    take=np.unique(np.r_[np.linspace(0,len(rows)-1,8,dtype=int),np.flatnonzero(rows.outcome.eq(0))[:4]])
    with np.load(ROOT / "review/inputs/pass_probability/arrays.npz") as a:
        states={k:a[k][take] for k in ["p_pos","p_vel","x_bc","r_target"]}
    endpoint,ball_time,valid=model.prepare_pitch_control_inputs(states)
    paths=[]
    for index in take:
        row=rows.iloc[index]
        local=np.flatnonzero(rows.match_id.eq(row.match_id)).tolist().index(index)
        with np.load(ROOT / "review/inputs/pass_probability/matches" / row.match_id / "trajectories.npz") as a:
            paths.append(a["fit_positions"][local])
    paths=np.asarray(paths)
    with model.precision_context(False):
        tti=model.trajectory_arrival_times(jnp.asarray(states["p_pos"]),jnp.asarray(states["p_vel"]),jnp.asarray(paths[:,:,:2]))
    low=paths[:,:,2]<=2.
    fitted=training.fit_mle(tti,low,valid,rows.outcome.to_numpy()[take])
    inputs=dict(endpoint_arrivals=endpoint,ball_arrival_time=ball_time,valid_players=valid)
    raw=training.predict_pitch_control(fitted,inputs,np.arange(len(take)))
    calibrators=training.fit_calibrators(raw,rows.outcome.to_numpy()[take])
    calibrated=training.apply_calibrators(raw,calibrators)
    # The trajectory-fit step also runs on one actual release window.
    first=rows.iloc[take[0]]
    tracking=pd.read_parquet(ROOT / "review/inputs/shared/tracking/matches" / first.match_id / f"{first.match_id}_tracking.parquet")
    trajectory,diagnostics=model.estimate_launch_conditions(tracking,first.frame_end,states["r_target"][0])
    assert trajectory.shape==(251,3) and np.isfinite(trajectory).all()
    write_json(output / "mle.json",fitted)
    write_json(output / "calibration.json",calibrators)
    saved=json.loads((output / "calibration.json").read_text())
    reloaded=training.apply_calibrators(raw,saved)
    for name in calibrated:
        np.testing.assert_array_equal(calibrated[name],reloaded[name])
    metrics,bins=training.binary_metrics(rows.outcome.to_numpy()[take],raw,float(rows.outcome.mean()))
    pd.DataFrame(bins).to_csv(output / "calibration_bins.csv",index=False)
    return dict(states=len(take),trajectory_fitted=True,three_mle_initializations=len(fitted["attempts"]),
                calibrators_serialized=True,metrics=metrics)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task",choices=["ball_drive","pass_probability"],required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    jax.config.update("jax_enable_x64",True)
    folder=new_folder(args.output)
    report=ball_drive(folder) if args.task=="ball_drive" else pass_probability(folder)
    write_json(folder / "verification.json",dict(scope="Short real-data checks, not a benchmark",**report))
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
