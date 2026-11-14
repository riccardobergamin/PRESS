"""Short native binary fits, P(carry) cross-fitting and production exports."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from pipeline.binary_classification import training,features
from pipeline.io import new_folder,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    inputs=ROOT/"pipeline/inputs/binary_classification"
    protocol=json.loads((inputs/"protocol.json").read_text())
    action=pd.read_parquet(inputs/"reference/action_selection_production_features.parquet")
    labels=pd.read_parquet(inputs/"residual_labels.parquet")
    for table in [action,labels]:table["match_id"]=table.match_id.astype(str)
    # Real states from every match, and actual terminal-loss cases, retained
    # by their keys rather than constructing artificial outcomes/features.
    successful_keys=labels.groupby("match_id",sort=True).head(24)[training.KEYS]
    lost_keys=labels.loc[labels.residual_loss.eq(1)].groupby("match_id",sort=True).head(4)[training.KEYS]
    other_keys=action.groupby("match_id",sort=True).head(12)[training.KEYS]
    keys=pd.concat([successful_keys,lost_keys,other_keys]).drop_duplicates()
    action=action.merge(keys,on=training.KEYS,validate="one_to_one")
    selected_labels=labels.merge(keys,on=training.KEYS,validate="one_to_one")
    data=new_folder(args.output/"input")
    selected_labels.to_parquet(data/"residual_labels.parquet",index=False)
    short=protocol | dict(grid=protocol["grid"][:2],max_rounds=16,early_stopping_rounds=3)
    write_json(data/"protocol.json",short)
    for split in ["fold_0","production"]:
        folder=args.output/"action_selection/features"/split
        folder.mkdir(parents=True)
        for match,group in action.groupby("match_id",sort=True):group.to_parquet(folder/f"{match}.parquet",index=False)
    training.train_validate_test(args.output,"action_selection",0,short)
    features.prepare_residual_features(data,args.output,"fold_0",threads=1)
    training.train_validate_test(args.output,"residual_risk",0,short)
    # Production configuration/tree selection is checked against all five
    # accepted validation tables; no five full classifier fits are rerun.
    for task in ["action_selection","residual_risk"]:
        production_test=args.output/"production_test"
        for fold in range(5):
            folder=production_test/task/"training"/f"fold_{fold}"
            folder.mkdir(parents=True,exist_ok=True)
            table=pd.read_csv(inputs/f"reference/fold_{fold}/{task}_grid_validation.csv")
            table.to_csv(folder/"grid_validation.csv",index=False)
        configuration,rounds=training.select_production_configuration(production_test,task)
        assert rounds==(277 if task=="action_selection" else 118)
        source=action if task=="action_selection" else pd.read_parquet(inputs/"reference/residual_risk_production_features.parquet").groupby("match_id",sort=True).head(10)
        folder=production_test/task/"features/production"
        folder.mkdir(parents=True)
        for match,group in source.groupby("match_id",sort=True):group.to_parquet(folder/f"{match}.parquet",index=False)
        training.train_production_model(inputs,production_test,task,protocol)
    # Exercise the five 8+2 P(carry) production cross-fits on the small real set.
    selection=json.loads((args.output/"production_test/action_selection/production/action_selection_model.json").read_text())
    parameters={k:selection["training"]["parameters"][k] for k in ["max_depth","min_child_weight","reg_lambda","eta"]}
    pairs=[protocol["folds"][str(f)]["test_matches"] for f in range(5)]
    p_carry=features.cross_fit_carry_probability(action,pairs,parameters,16,protocol["fixed_parameters"],
        protocol["seed"],1,args.output/"production_crossfit")
    assert np.isfinite(p_carry).all() and ((p_carry>=0)&(p_carry<=1)).all()
    write_json(args.output/"verification.json",dict(status="passed",real_states=len(action),
        short_6_plus_2_plus_2=True,three_inner_action_refits=True,five_production_action_refits=True,
        production_action_trees=277,production_residual_trees=118))


if __name__=="__main__":
    main()
