"""Step-by-step local verification plan; no complete ML hyperparameter search.

Default prints the commands. --execute runs selected checks and stores logs.
Use a NEW output directory. The installed/checkout PRESS is selected explicitly;
this script neither installs a package nor changes the pretrained weights.
"""

import argparse
import ast
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, help="Optional source checkout; otherwise use the installed PRESS package")
    parser.add_argument("--raw-root", type=Path, help="Public SkillCorner raw match directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", action="append", choices=["syntax","cli","preprocessing","xt","ball_drive","catboost","pass_probability","pass_selection","binary","composed","reports","release"])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    py, out = sys.executable, args.output.resolve()
    def call(path,*values):
        return [py,str(PIPELINE_ROOT/path),*map(str,values)]
    steps = {
        "preprocessing": [call("preprocessing/pipeline.py","--raw-root",args.raw_root or "/path/to/opendata/data/matches","--match","1925299","--templates",PIPELINE_ROOT/"inputs","--output",out/"preprocessing"),
                          call("checks/preprocessing_rows.py","--output",out/"preprocessing")],
        "xt": [call("checks/xt_steps.py","--output",out/"xt")],
        "ball_drive": [call("checks/physical_steps.py","--task","ball_drive","--output",out/"physical_drive")],
        "catboost": [call("checks/catboost_steps.py","--output",out/"catboost")],
        "pass_probability": [call("checks/physical_steps.py","--task","pass_probability","--output",out/"physical_pass")],
        "pass_selection": [call("checks/pass_selection_steps.py","--output",out/"pass_selection")],
        "binary": [call("checks/binary_steps.py","--output",out/"binary")],
        "composed": [call("composed_risk/pipeline.py","--replay-accepted","--output",out/"composed_risk"),
                     call("composed_risk/pipeline.py","--binary-root",PIPELINE_ROOT/"inputs/composed_risk/reference/fold_stack",
                          "--output",out/"composed_from_models")],
        "reports": [call("reports/pipeline.py","--states",out/"composed_risk/features/held_out.parquet","--output",out/"reports"),
                    call("checks/compare_reports.py","--output-root",out)],
        "release": [call("release.py","--code-destination",out/"code_release","--output-root",out,"--tables-destination",out/"public_tables")],
    }
    selected = args.step or ["syntax","cli",*steps]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), *([str(args.package_root.resolve())] if args.package_root else []), env.get("PYTHONPATH","")])
    env.update(MPLBACKEND="Agg",JAX_ENABLE_X64="1")
    if args.execute and "preprocessing" in (args.step or steps) and args.raw_root is None:
        parser.error("The preprocessing check requires --raw-root.")
    if args.execute:
        out.mkdir(parents=True,exist_ok=True)
    results = []
    if "syntax" in selected:
        paths = [p for p in PIPELINE_ROOT.rglob("*.py") if not {"old","local_history","experiments","outputs","inputs"}.intersection(p.relative_to(PIPELINE_ROOT).parts)]
        if args.execute:
            for path in paths:
                ast.parse(path.read_text(),filename=str(path))
            results.append(dict(step="syntax",files=len(paths),status="passed"))
        print("SYNTAX",len(paths),"current source files")
    if "cli" in selected:
        pipelines = ["preprocessing","expected_threat","ball_drive/physical","ball_drive/catboost","pass_probability","pass_selection","binary_classification","composed_risk","reports"]
        steps = {"cli":[call(name+"/pipeline.py","--help") for name in pipelines],**steps}
    for step, commands in steps.items():
        if step not in selected:
            continue
        for number, command in enumerate(commands):
            print(step,shlex.join(command),flush=True)
            if args.execute:
                log = out / f"{step}_{number}.log"
                with log.open("w") as stream:
                    result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                results.append(dict(step=step,command=command,exit_code=result.returncode,log=str(log)))
                (out/"checks.json").write_text(json.dumps(results,indent=2)+"\n")
                if result.returncode:
                    print(log.read_text()[-5000:])
                    raise SystemExit(result.returncode)
    if args.execute:
        (out/"checks.json").write_text(json.dumps(results,indent=2)+"\n")


if __name__ == "__main__":
    main()
