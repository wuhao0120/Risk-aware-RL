#!/usr/bin/env python3
"""Check pinned packages, frozen sources and real headless CUDA physics."""
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from experiment import ROOT, REGISTRY, runtime_env, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/doctor.json')
    parser.add_argument('--versions-only', action='store_true')
    parser.add_argument('--image-torch', action='store_true', help='Keep image-provided torch 2.3.1 / CUDA 12.1')
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError('Use the pinned Python 3.10 environment (not 3.11/3.12)')
    versions = {}
    for line in (ROOT / 'requirements.lock').read_text().splitlines():
        if not line or line.startswith(('#', '--')):
            continue
        name, expected = line.strip().split('==')
        if args.image_torch and (name in {'torch', 'triton'} or name.startswith('nvidia-')):
            continue
        actual = metadata.version(name)
        if actual != expected:
            raise RuntimeError(f'{name}: installed {actual}, expected {expected}')
        versions[name] = actual
    os.environ.update(runtime_env(args.gpu, 1, 'smoke', args.output.parent))
    import torch
    if torch.__version__.split('+')[0] != '2.3.1' or torch.version.cuda != '12.1':
        raise RuntimeError('Expected PyTorch 2.3.1 built for CUDA 12.1')
    report = dict(image_torch=args.image_torch, torch_runtime=torch.__version__,
                  torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), python=sys.version, platform=platform.platform(), packages=versions,
                  source_files_verified=verify(), gpu_tests=[])
    if not args.versions_only:
        env = runtime_env(args.gpu, 1, 'smoke', args.output.parent)
        os.environ.update(env)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable: check NVIDIA driver and GPU/container access')
        report['gpu'] = torch.cuda.get_device_name(0)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for key in ('simplebutton-main_r099_c1_p10-dca', 'dynamic-dca_div25'):
            destination = args.output.parent / f'doctor-{key}.json'
            subprocess.run([sys.executable, str(ROOT / 'tools/gpu_probe.py'), key,
                            str(destination.resolve())], env=env, check=True)
            report['gpu_tests'].append(json.loads(destination.read_text()))
    report['passed'] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'PASS: {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
