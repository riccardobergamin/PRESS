"""Real CatBoost features/sampling; optional 12-tree native smoke fit."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import jax
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from review.ball_drive.catboost import features,training
from review.io import new_folder,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--require-native",action="store_true")
    args=parser.parse_args()
    jax.config.update("jax_enable_x64",True)
    folder=new_folder(args.output)
    rows=pd.read_parquet(ROOT/"review/inputs/ball_drive/rows.parquet")
    with np.load(ROOT/"review/inputs/ball_drive/arrays.npz") as a:
        states={k:a[k] for k in ["r_defs","v_defs","x_bc","v_bc"]}
    direction=np.empty(len(rows))
    for match,group in rows.groupby("match_id",sort=False):
        file=ROOT/"review/inputs/shared/tracking/matches"/str(match)/f"{match}_tracking.parquet"
        tracking=pd.read_parquet(file,columns=["frame","attacking_side"]).set_index("frame")
        sides=tracking.attacking_side.reindex(group.frame_start).to_numpy()
        if not np.isin(sides,["left_to_right","right_to_left"]).all():
            raise ValueError("An observed carry lacks attack direction.")
        direction[group.index]=np.where(sides=="left_to_right",1.,-1.)
    tti,valid=features.calculate_defender_tti(states)
    spatial=features.build_spatial_features(states,direction,tti,valid)
    X=features.build_features(spatial,tti,valid,.2996318904783872)
    y=rows.loss_observed.to_numpy(int)
    tests=json.loads((ROOT/"review/inputs/folds.json").read_text())["0"]
    test_mask=rows.match_id.astype(str).isin(tests)
    fitting=np.flatnonzero(~test_mask);test=np.flatnonzero(test_mask)
    for sampling in ["natural","loss10"]:
        selected,audit=training.select_training_rows(fitting,y,sampling,19)
        assert not np.intersect1d(selected,test).size
        assert len(selected)==len(np.unique(selected))
        if sampling=="loss10":
            np.testing.assert_array_equal(np.sort(selected[y[selected]==1]),np.sort(fitting[y[fitting]==1]))
    present=importlib.util.find_spec("catboost") is not None
    if present:
        take=np.unique(np.r_[fitting[:80],fitting[y[fitting]==1][:20]])
        native=training.train_model(X.iloc[take],y[take],dict(depth=4,learning_rate=.05,l2_leaf_reg=3,iterations=12),19,1)
        p=native.predict_proba(X.iloc[test[:20]])[:,1]
        assert np.isfinite(p).all()
        native.save_model(str(folder/"smoke.cbm"))
        from catboost import CatBoostClassifier
        loaded=CatBoostClassifier();loaded.load_model(str(folder/"smoke.cbm"))
        np.testing.assert_array_equal(p,loaded.predict_proba(X.iloc[test[:20]])[:,1])
    report=dict(feature_rows=len(X),features=X.shape[1],training_only_sampling=True,
        native_fit="passed" if present else "NOT RUN: catboost not installed in this interpreter")
    write_json(folder/"verification.json",report)
    print(report)
    if args.require_native and not present:
        raise SystemExit("Native CatBoost check still required in an environment with CatBoost.")


if __name__=="__main__":
    main()
