"""One explicit raw-to-benchmark-to-production run, without old-run recovery."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
TASKS = ['ball_drive', 'pass_probability', 'pass_selection', 'action_selection', 'residual_risk']


def plan(raw, base, output, threads):
    common = ['--raw-root',
        str(raw),
        '--base-root',
        str(base),
        '--output-root',
        str(output),
        '--threads',
        str(threads)]
    stages = []
    for script in ['01_tracking', '02_possessions', '03_bases', '04_context_inputs']:
        stages.append([sys.executable, str(ROOT / 'pipeline/preprocessing' / (script + '.py')), *common])
    for task in TASKS:
        for script in ['01_features', '02_train', '03_scores']:
            stages.append([sys.executable, str(ROOT / 'pipeline' / task / (script + '.py')), *common])
    for script in ['01_compute', '02_scores']:
        stages.append([sys.executable, str(ROOT / 'pipeline/composed_risk' / (script + '.py')), *common])
    for task in TASKS:
        if task not in ['ball_drive', 'pass_probability']:
            stages.append([sys.executable, str(ROOT / 'pipeline' / task / '01_features.py'), *common, '--production-features'])
        stages.append([sys.executable, str(ROOT / 'pipeline' / task / '04_refit.py'), *common])
    return stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--execute',
        action='store_true',
        help='Run every stage; otherwise print the exact plan')
    args = parser.parse_args()
    base = args.output_root / 'base'
    stages = plan(args.raw_root, base, args.output_root, args.threads)
    if args.execute and args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError('A complete run requires a new, empty output root.')
    for command in stages:
        print(shlex.join(command), flush=True)
        if args.execute:
            env = os.environ | dict(JAX_ENABLE_X64='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            subprocess.run(command, cwd=ROOT, env=env, check=True)
if __name__ == '__main__':
    main()
