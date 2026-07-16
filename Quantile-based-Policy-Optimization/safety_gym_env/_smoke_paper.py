# -*- coding: utf-8 -*-
"""临时冒烟: QCPO/DQCAC 两算法在 4 个论文环境上各跑 3 迭代 (tiny), 验证接线后无报错/NaN。"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('WANDB_MODE', 'disabled')
import sys
sys.path.insert(0, '.')
import numpy as np
import torch
from types import SimpleNamespace
from envs import SafetyEnv, PAPER_ENV_IDS
from agents import QCPOGPU, DQCACBetaGPU


def make_args(algo, device, env_key):
    a = SimpleNamespace()
    a.device = device; a.seed = 0; a.algo_name = algo; a.env_name = env_key
    a.gamma = 0.99; a.cost_gamma = 0.99; a.q_alpha = 0.2; a.cost_limit = 2.0
    a.num_envs = 4; a.num_iterations = 3; a.horizon = 30; a.init_std = 0.5
    a.actor_hidden = [64, 64]; a.log_interval = 1
    a.wandb_project = 'safety_gym_qcrl'; a.wandb_dir = None
    a.theta_a = 0.1; a.theta_b = 1000; a.theta_c = 0.9
    a.lambda_a = 1.0; a.lambda_b = 1000; a.lambda_c = 0.1
    a.lambda_max = 50.0; a.lambda_min = 0.0; a.outer_interval = 1
    a.updates_per_episode = 2; a.warmup_rms_iters = 1
    if algo == 'QCPO':
        a.theta_optimizer = 'adam'; a.norm_ema_decay = 0.01; a.actor_grad_clip = 1.0
    else:
        a.beta = 0.95; a.num_quantiles = 8; a.huber_kappa = 0.1; a.target_tau = 0.05
        a.critic_step_feature = False; a.num_action_samples = 2; a.advantage_norm = 'qcpo'
        a.norm_ema_decay = 0.1; a.warmup_iters = 1; a.critic_lr = 1e-3; a.critic_hidden = [64, 64]
        a.entropy_coef = 0.0; a.actor_grad_clip = 100.0; a.critic_grad_clip = 10.0
        a.n_step = 1; a.target_update_interval = 1
    return a


dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
for env_key in PAPER_ENV_IDS:
    for algo, Cls in [('QCPO', QCPOGPU), ('DQCAC', DQCACBetaGPU)]:
        print(f"\n==== {env_key} / {algo}")
        env = SafetyEnv(env_key)
        ag = Cls(make_args(algo, dev, env_key), env)
        ag.train()
        lam = float(ag.lambda_dual.item())
        assert np.isfinite(lam), f"{env_key}/{algo} lambda NaN/Inf!"
        print(f"OK {env_key}/{algo} lambda={lam:.4f} outage={ag.last_outage_prob:.3f}")
        ag.vec_env.close(); env.close()
print("\nALL PAPER-ENV SMOKE OK")
