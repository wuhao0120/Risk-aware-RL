"""
DQC-AC Hyperparameter Sweep
============================
Parallel training with different hyperparameters.
Each process gets its own wandb run, no conflicts.

Usage: python run_sweep.py
"""

import os
import sys
import time
import torch
import numpy as np
import random
import multiprocessing as mp
import wandb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import DQCAC
from envs import RiskSensitiveEnv


class DQC_AC_Args:
    """Baseline parameters (same as notebook). Use **overrides to change any field."""
    def __init__(self, **overrides):
        self.env_name = "RiskSensitiveEnv"
        self.seed = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.max_episode = 10000
        self.log_interval = 100
        self.est_interval = 100
        self.gamma = 0.99

        self.q_alpha = 0.25
        self.quantile_threshold = 4
        self.outer_interval = 10

        self.init_std = float(np.sqrt(1e-1))
        self.theta_a = (10000 ** 0.9) * 1e-3
        self.theta_b = 10000
        self.theta_c = 0.9

        self.q_a = (10000 ** 0.6) * 1e-2
        self.q_b = 10000
        self.q_c = 0.6

        self.lambda_a = 0.3
        self.lambda_b = 5000
        self.lambda_c = 0.6

        self.emb_dim = [64, 64]
        self.critic_lr = 1e-3
        self.tau = 0.005
        self.target_update_interval = 100

        self.num_quantiles = 32
        self.huber_kappa = 1.0
        self.density_bandwidth = 0.01

        self.buffer_capacity = 100000
        self.batch_size = 64
        self.min_buffer_size = 1000
        self.updates_per_episode = 10

        self.wandb_dir = None
        self.algo_name = "DQCAC"

        for k, v in overrides.items():
            setattr(self, k, v)


def run_experiment(task):
    """Run a single experiment in a subprocess."""
    run_name, overrides = task
    print(f"[{run_name}] PID={os.getpid()} Starting | {overrides}")

    args = DQC_AC_Args(**overrides)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = RiskSensitiveEnv(n=10)
    agent = DQCAC(args, env)

    # Rename wandb run so dashboard can distinguish experiments
    wandb.run.name = run_name
    # wandb auto-syncs name change, no save needed

    t0 = time.time()
    agent.train()
    elapsed = time.time() - t0

    wandb.finish()
    print(f"[{run_name}] Done in {elapsed / 60:.1f} min")


EXPERIMENTS = [
    # 1. num_quantiles
    ("nq64",    {"num_quantiles": 64}),
    ("nq200",   {"num_quantiles": 200}),
    # 2. max_episode (10x longer)
    ("ep100k",  {"max_episode": 100000}),
    # 3. batch_size
    ("bs256",   {"batch_size": 256}),
    # 4. updates_per_episode
    ("upe1",    {"updates_per_episode": 1}),
    ("upe20",   {"updates_per_episode": 20}),
    ("upe100",  {"updates_per_episode": 100}),
    # 5. target_update_interval
    ("tui10",   {"target_update_interval": 10}),
    ("tui1000", {"target_update_interval": 1000}),
]


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    MAX_WORKERS = 4

    print("=" * 60)
    print(f"DQC-AC Sweep: {len(EXPERIMENTS)} experiments, {MAX_WORKERS} workers")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 60)
    for name, ov in EXPERIMENTS:
        print(f"  {name:10s} -> {ov}")
    print("=" * 60)

    t_start = time.time()

    with mp.Pool(processes=MAX_WORKERS) as pool:
        pool.map(run_experiment, EXPERIMENTS)

    t_total = time.time() - t_start
    print(f"\nAll {len(EXPERIMENTS)} experiments done in {t_total / 60:.1f} min")
