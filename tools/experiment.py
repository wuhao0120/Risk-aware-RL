#!/usr/bin/env python3
"""Portable launcher for frozen experiment sources; no machine-specific paths."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = json.loads((ROOT / 'experiments/registry.json').read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify():
    snapshots = json.loads((ROOT / 'provenance/source_snapshots.json').read_text())
    count = 0
    for snapshot in snapshots:
        source = ROOT / snapshot['export_source']
        for name, expected in snapshot['files'].items():
            if digest(source / name) != expected:
                raise RuntimeError(f'Frozen source changed: {source / name}')
            count += 1
    return count


def runtime_env(gpu, gamma, phase, output):
    cache = Path(os.environ.get('RISK_CACHE_DIR', str(ROOT / '.cache'))).resolve()
    for name in ('warp', 'mpl', 'tmp'):
        (cache / name).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
               SIMPLEBUTTON_REWARD_GAMMA=str(gamma), SIMPLEBUTTON_TRAIN_GAMMA=str(gamma),
               RISK_EVAL_SEED='81000000' if phase == 'smoke' else '92000000',
               MUJOCO_GL='disable', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', WANDB_MODE='disabled',
               XDG_CACHE_HOME=str(cache), WARP_CACHE_PATH=str(cache / 'warp'),
               MPLCONFIGDIR=str(cache / 'mpl'), TMPDIR=str(cache / 'tmp'))
    return env


def plan(key, seed, output, phase):
    s = REGISTRY[key]
    method = s['method']
    batches = (22 if method == 'dca' else 10 if method == 'qcpo' else 2) if phase == 'smoke' else s['batches']
    source = ROOT / s['source']
    if method == 'dca':
        entry = 'run.py'
        extra = ['--arm', s['arm'], '--reward-scale', str(s['divisor']), '--main-batches', str(batches - 20)]
    elif method == 'qcpo':
        entry = 'qcpo_run.py'
        extra = ['--main-batches', str(batches)]
    else:
        entry = 'run_tqpo.py'
        extra = ['--smoke'] if phase == 'smoke' else ['--batches', str(batches)] if s.get('supports_batches') else []
    steps = batches * s['batch'] * s['horizon']
    command = [sys.executable, '-u', str(source / entry), '--seed', str(seed), '--output', str(output), *extra]
    return command, steps, [steps] if phase == 'smoke' else s['checkpoints']


def call(command, env, log):
    """Forward interruption only to this launcher's child process group."""
    with log.open('a') as f:
        child = subprocess.Popen(command, env=env, cwd=ROOT, stdout=f,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait()
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait()  # runner saves after the current complete batch
            raise
    if code:
        raise RuntimeError(f'Command failed ({code}); see {log}')


def run(key, seed, gpu, output, phase, dry_run=False):
    spec = REGISTRY[key]
    command, steps, checkpoints = plan(key, seed, output, phase)
    print(json.dumps(dict(experiment=key, seed=seed, gpu=gpu, phase=phase,
                          steps=steps, command=command, checkpoints=checkpoints), ensure_ascii=False), flush=True)
    if dry_run:
        return
    verify()
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'launcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = dict(experiment=key, seed=seed, phase=phase, spec=spec)
        record = output / 'launch.json'
        if record.exists() and json.loads(record.read_text()) != identity:
            raise RuntimeError('Output belongs to a different experiment/seed/phase; choose another directory')
        record.write_text(json.dumps(identity, indent=2) + '\n')
        env = runtime_env(gpu, spec['reward_gamma'], phase, output)
        receipt_path = output / 'training_receipt.json'
        if not receipt_path.exists():
            call(command, env, output / 'training.log')
        if not receipt_path.exists():
            raise RuntimeError('Training stopped without completion; rerun the same command to resume')
        receipt = json.loads(receipt_path.read_text())
        config = json.loads((output / 'config.json').read_text())
        if receipt['actual_env_steps'] != steps or config['planned_env_steps'] != steps or not receipt['checkpoint_roundtrip']:
            raise RuntimeError('Training receipt has an incorrect budget or checkpoint roundtrip')
        if receipt['signature'] != config['signature'] or config['signature']['seed'] != seed:
            raise RuntimeError('Training signature mismatch')
        signature = config['signature']
        if spec['method'] == 'tqpo':
            identity = receipt['tqpo_identity']
            actual = (identity['environment'], identity['gamma'], identity['cost_gamma'],
                      identity['dual_gamma'], identity['failure_alpha'], identity['num_envs'])
            if not receipt.get('cdf_guard_fixed'):
                raise RuntimeError('Expected fixed TQPO CDF guard')
        else:
            actual = (config['configuration']['env_name'], signature['gamma_reward'],
                      signature['gamma_cost'], signature['dual_gamma'], signature['epsilon'],
                      config['configuration']['num_envs'])
            if spec['method'] == 'dca' and signature['mean_divisor'] != spec['divisor']:
                raise RuntimeError('DCA divisor mismatch')
        expected_protocol = (spec['environment'], spec['reward_gamma'], spec['cost_gamma'],
                             spec['dual_gamma'], spec['epsilon'], spec['batch'])
        if actual != expected_protocol:
            raise RuntimeError(f'Protocol mismatch: {actual} != {expected_protocol}')
        source = ROOT / spec['source']
        for name, expected in config['signature']['source'].items():
            if digest(source / name) != expected:
                raise RuntimeError(f'Checkpoint source mismatch: {name}')
        episodes = 16 if phase == 'smoke' else 2048
        for step in checkpoints:
            checkpoint = output / f'checkpoint_{step}.pt'
            if not checkpoint.exists() and step == steps:
                shutil.copy2(output / 'final.pt', checkpoint)
            dest = output / f'validation_{step}'
            result_path = dest / 'result.json'
            if not result_path.exists():
                call([sys.executable, '-u', str(source / 'validate_gpu.py'), spec['method'],
                      str(output), str(step), str(episodes), str(dest)], env, output / f'eval_{step}.log')
            result = json.loads(result_path.read_text())
            if (result['steps'], result['episodes'], result['base_seed'], result['checkpoint_sha256']) != (
                    step, episodes, int(env['RISK_EVAL_SEED']), digest(checkpoint)):
                raise RuntimeError('Evaluation identity mismatch')
        (output / 'COMPLETE.json').write_text(json.dumps(dict(experiment=key, seed=seed,
                phase=phase, steps=steps, evaluation_episodes=episodes, checkpoints=checkpoints), indent=2) + '\n')
        print(f'COMPLETE {key} seed={seed} output={output}', flush=True)


def campaign(args):
    # Each lane owns one explicitly selected physical GPU.
    queue = [(key, seed) for seed in args.seeds for key in args.experiments]
    running = {}
    try:
        while queue or running:
            for gpu, child in list(running.items()):
                code = child.poll()
                if code is None:
                    continue
                del running[gpu]
                if code:
                    raise RuntimeError(f'GPU {gpu} worker failed with exit code {code}')
            for gpu in args.gpus:
                if gpu in running or not queue:
                    continue
                key, seed = queue.pop(0)
                command = [sys.executable, str(Path(__file__).resolve()), 'run', '--experiment', key,
                           '--seed', str(seed), '--gpu', str(gpu), '--phase', args.phase,
                           '--output', str(args.output.resolve() / key / f'seed{seed}')]
                if args.dry_run:
                    command.append('--dry-run')
                running[gpu] = subprocess.Popen(command, start_new_session=True)
            time.sleep(0.2)
    finally:
        for child in running.values():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in running.values():
            child.wait()  # worker propagates SIGTERM and waits for a saved batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list')
    sub.add_parser('verify')
    one = sub.add_parser('run')
    one.add_argument('--experiment', choices=REGISTRY, required=True)
    one.add_argument('--seed', type=int, required=True)
    one.add_argument('--gpu', type=int, default=0)
    one.add_argument('--output', type=Path, required=True)
    many = sub.add_parser('campaign')
    many.add_argument('--experiments', nargs='+', choices=REGISTRY, required=True)
    many.add_argument('--seeds', nargs='+', type=int, required=True)
    many.add_argument('--gpus', nargs='+', type=int, required=True)
    many.add_argument('--output', type=Path, required=True)
    for p in (one, many):
        p.add_argument('--phase', choices=['smoke', 'train'], default='train')
        p.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.command == 'list':
        for key, s in REGISTRY.items():
            print(f"{key:48} {s['environment']:12} B{s['batch']:2} gamma={s['reward_gamma']} eps={s['epsilon']} {s['status']}")
    elif args.command == 'verify':
        print(f'PASS: {verify()} source files match the original experiment snapshots')
    elif args.command == 'run':
        run(args.experiment, args.seed, args.gpu, args.output.resolve(), args.phase, args.dry_run)
    else:
        if len(set(args.gpus)) != len(args.gpus) or len(set(args.experiments)) != len(args.experiments) or len(set(args.seeds)) != len(args.seeds):
            parser.error('Duplicate GPUs, experiments or seeds are not allowed')
        campaign(args)


if __name__ == '__main__':
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    main()
