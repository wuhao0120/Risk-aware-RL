# -*- coding: utf-8 -*-
"""临时冒烟测试: 两算法完整 train() 循环各跑 3 迭代 (tiny), 验证无报错/NaN、λ/critic 路径通。"""
import os, sys
sys.path.insert(0, '.')
import numpy as np
import torch
from types import SimpleNamespace
from envs import SafetyEnv
from agents import QCPOGPU, DQCACBetaGPU


def make_args(algo, device):
    a = SimpleNamespace()
    a.device = device; a.seed = 0; a.algo_name = algo; a.env_name = 'SafetyPointGoal1-v0'
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


for algo, Cls in [('QCPO', QCPOGPU), ('DQCAC', DQCACBetaGPU)]:
    print(f"\n{'='*60}\n==== {algo}")
    env = SafetyEnv('SafetyPointGoal1-v0')
    ag = Cls(make_args(algo, torch.device('cuda:0')), env)
    ag.train()
    lam = float(ag.lambda_dual.item())
    print(f"[{algo}] DONE lambda={lam:.4f} last_outage={ag.last_outage_prob:.4f} "
          f"dual_prob={getattr(ag,'last_dual_prob',None)}")
    assert np.isfinite(lam), f"{algo} lambda is NaN/Inf!"
    print(f"OK {algo}")
print("\nALL SMOKE TRAIN OK")
