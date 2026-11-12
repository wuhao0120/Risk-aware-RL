"""Reproduce the final SimpleButton appendix experiments.

Setup (Linux, Python 3.10, CUDA-capable NVIDIA GPU for formal experiments):
    python -m venv .venv
    .venv/bin/python -m pip install --no-deps -r requirements.lock
    .venv/bin/python pipeline.py check
    .venv/bin/python pipeline.py smoke --output runs/smoke

Formal pipeline, sequential on one explicitly selected GPU:
    python pipeline.py all --gpu 0 --output runs/formal
    python pipeline.py all --gpu 0 --method dca --seeds 0 --output runs/formal
    python pipeline.py all --dry-run
    python pipeline.py summarize --output runs/formal

`all` trains, evaluates final checkpoints, and writes tables and learning curves.
`train` and `evaluate` run the corresponding stages independently. Training
automatically resumes from latest.pt, enforcing the saved configuration and
source signature. Use a new output directory after changing scientific code.

Protocol: P(sum(cost)>15)<=0.2, H=1000, stochastic held-out evaluation.
DCA: seeds 0..4, 10M steps including 20 warmup rollouts, 2048 test episodes.
QCPO: seeds 0..3, 10M steps, 512 test episodes; reward/cost gamma=1.
TQPO: seeds 0..4, 10.008M steps, 2048 test episodes; training gamma=.99.
The TQPO original strict history-buffer guard is intentionally preserved.
CPU smoke checks are software checks, not replicas of GPU training results.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT / 'configs' / 'protocol.json').read_text())


def commands(stage, method, seed, output):
    """Build argument lists without a shell; run paths are independent of cwd."""
    run = output / f'{method}_seed{seed}'
    if stage == 'train':
        script = {'dca': 'run.py', 'qcpo': 'qcpo_run.py', 'tqpo': 'run_tqpo.py'}[method]
        return [sys.executable, str(ROOT / script), '--seed', str(seed), '--output', str(run)]
    script = {'dca': 'dca_evaluate.py', 'qcpo': 'qcpo_evaluate.py', 'tqpo': 'tqpo_evaluate.py'}[method]
    return [sys.executable, str(ROOT / script), '--run-dir', str(run), '--out-dir', str(run / 'evaluation')]


def check():
    """Check the executed runtime versions, local imports and CUDA availability."""
    import importlib.metadata as metadata
    import base_runner
    base_runner.setup()
    import torch
    import run, qcpo_run, run_tqpo, dca_evaluate, qcpo_evaluate, tqpo_evaluate
    import envs.safety_env_simple_dynamic_mjwarp
    expected = json.loads((ROOT / 'configs' / 'runtime.json').read_text())
    actual = {name: metadata.version(name) for name in expected}
    differences = {name: [version, actual[name]] for name, version in expected.items() if version != actual[name]}
    print(json.dumps(dict(python=sys.version.split()[0], cuda_available=torch.cuda.is_available(),
                         versions=actual, differences=differences), indent=2))
    if sys.version_info[:2] != (3, 10) or differences:
        raise SystemExit('Runtime differs from the frozen environment. Install requirements.lock with Python 3.10.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['check', 'smoke', 'train', 'evaluate', 'summarize', 'all'])
    parser.add_argument('--method', choices=['all', 'dca', 'qcpo', 'tqpo'], default='all')
    parser.add_argument('--seeds', type=int, nargs='+', help='Default: exact paper seeds for each method')
    parser.add_argument('--gpu', default='0', help='One CUDA device index or UUID; runs are sequential')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs' / 'formal')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without training or creating run files')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage == 'check':
        check()
        return
    if args.stage == 'smoke':
        subprocess.run([sys.executable, str(ROOT / 'smoke.py'), '--output', str(output)], check=True, cwd=ROOT)
        return
    methods = list(PROTOCOL['methods']) if args.method == 'all' else [args.method]
    selected = {m: PROTOCOL['methods'][m]['seeds'] if args.seeds is None else args.seeds for m in methods}
    if any(s < 0 for seeds in selected.values() for s in seeds):
        parser.error('Seeds must be nonnegative')
    if any(len(seeds) != len(set(seeds)) for seeds in selected.values()):
        parser.error('Duplicate seeds are not allowed')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, MUJOCO_GL='disable', WANDB_MODE='disabled')
    for method, seeds in selected.items():
        for seed in seeds:
            stages = ['train', 'evaluate'] if args.stage == 'all' else [args.stage]
            for stage in stages:
                if stage == 'summarize':
                    continue
                command = commands(stage, method, seed, output)
                print(json.dumps({'method': method, 'seed': seed, 'command': command}), flush=True)
                if not args.dry_run:
                    subprocess.run(command, check=True, cwd=ROOT, env=env)
    if args.stage in ['all', 'summarize'] and not args.dry_run:
        from summarize import summarize
        summarize(output, selected, output / 'results')


if __name__ == '__main__':
    main()
