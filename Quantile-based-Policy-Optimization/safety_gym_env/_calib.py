# -*- coding: utf-8 -*-
"""
Deprecated filename retained for compatibility; now delegates to reward-side random calibration.
It reports raw G_T distribution only and never selects a formal q automatically.
"""
import os
import sys
from types import SimpleNamespace
import torch
from run_experiment import calibrate_random_policy

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("WANDB_MODE", "disabled")
env_id = sys.argv[1] if len(sys.argv) > 1 else "SimpleButton"
num_envs = int(sys.argv[2]) if len(sys.argv) > 2 else 16
horizon = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 2
args = SimpleNamespace(
    env_id=env_id, num_envs=num_envs, horizon=horizon,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    seed=0, vec_backend="mp", gamma=0.99, failure_alpha=0.2)
calibrate_random_policy(args, num_episodes=num_envs * rounds)
