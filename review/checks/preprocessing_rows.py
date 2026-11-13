"""Compare reconstructed match cohorts with accepted observation keys/labels."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from review.io import read_table,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--match",default="1925299")
    args=parser.parse_args()
    inputs=ROOT/"review/inputs"
    tasks=[
        ("ball_drive","ball_drive/rows.parquet",["match_id","frame_start","frame_end"]),
        ("pass_probability","pass_probability/rows.parquet",["match_id","pass_id","frame_end"]),
        ("pass_selection","pass_selection/rows.parquet",["match_id","pass_id","frame_end"]),
        ("action_selection","binary_classification/action_rows.parquet",["match_id","action_id","frame_start"]),
        ("residual_risk","binary_classification/residual_labels.parquet",["match_id","action_id","frame_start"]),
        ("composed_risk","composed_risk/labels.parquet",["match_id","action_id","frame_start"]),
    ]
    report={}
    for task,reference,keys in tasks:
        accepted=read_table(inputs/reference)
        accepted=accepted.loc[accepted.match_id.eq(args.match)]
        actual=read_table(args.output/"task_inputs"/task/"rows.parquet")
        a=set(map(tuple,accepted[keys].astype(str).to_numpy()))
        b=set(map(tuple,actual[keys].astype(str).to_numpy()))
        assert a==b and len(a)==len(actual)==len(accepted),(task,len(a-b),len(b-a))
        ordered=accepted[keys].astype(str).reset_index(drop=True).equals(actual[keys].astype(str).reset_index(drop=True))
        provenance=args.output/"task_inputs/provenance.json"
        if provenance.is_file():
            import json
            if json.loads(provenance.read_text()).get("templates"):
                assert ordered,(task,"Accepted observation order changed despite explicit templates")
        paired=accepted.merge(actual,on=keys,suffixes=("_ref","_new"),validate="one_to_one")
        for label in ["outcome","y","residual_loss","terminal_loss","h_loss"]:
            if label in accepted and label in actual:
                np.testing.assert_array_equal(paired[label+"_ref"],paired[label+"_new"])
        report[task]=dict(n=len(actual),keys_identical=True,labels_identical=True,accepted_order_preserved=ordered)
    write_json(args.output/"cohort_verification.json",report)
    print(report)


if __name__=="__main__":
    main()
