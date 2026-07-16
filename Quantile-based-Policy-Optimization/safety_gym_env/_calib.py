# -*- coding: utf-8 -*-
"""
CALIB 探针: 用随机(未训练)策略在【整段 horizon=1000】上测 reward 回报 R 与 cost 回报 C 的
分布 → 选约束阈值 d 与目标 ω 的依据 (对应 risk/portfolio 的 _noisy_calib.py 方法论)。
同时测 CPU 同步 VecEnv 的 rollout 耗时 → 决定是否需要 async 多进程加速。
"""
import os, sys, time
sys.path.insert(0, '.')
import numpy as np
import torch
from types import SimpleNamespace
from envs import SafetyEnv, SafetyVecEnv
from agents.vec_base import VecAgentBase

ENV_ID = sys.argv[1] if len(sys.argv) > 1 else 'SafetyPointGoal1-v0'
B = int(sys.argv[2]) if len(sys.argv) > 2 else 16
T = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
ROUNDS = int(sys.argv[4]) if len(sys.argv) > 4 else 2

dev = torch.device('cuda:0')
args = SimpleNamespace(device=dev, gamma=0.99, cost_gamma=0.99, q_alpha=0.2, cost_limit=1.0,
                       log_interval=999, num_envs=B, num_iterations=1, horizon=T, init_std=0.5,
                       actor_hidden=[256, 256], seed=0, wandb_project='safety_gym_qcrl', algo_name='CALIB')
env = SafetyEnv(ENV_ID)
base = VecAgentBase(args, env)                              # 未训练随机策略
vec = SafetyVecEnv(ENV_ID, num_envs=B, horizon=T, device=dev, ref_env=env, seed=100)

Rs, Cs, Cus = [], [], []
t0 = time.time()
with torch.no_grad():
    for _ in range(ROUNDS):
        s = vec.reset()
        R = torch.zeros(B, device=dev); C = torch.zeros(B, device=dev); Cu = torch.zeros(B, device=dev)
        dr = dc = 1.0
        for t in range(T):
            a = base._sample_actions(s)
            s, r, c, done = vec.step(a)
            R += dr * r; dr *= 0.99
            C += dc * c; dc *= 0.99
            Cu += c
        Rs.append(R); Cs.append(C); Cus.append(Cu)
dt = time.time() - t0
R = torch.cat(Rs).cpu().numpy(); C = torch.cat(Cs).cpu().numpy(); Cu = torch.cat(Cus).cpu().numpy()

print(f"\n===== CALIB {ENV_ID} (random policy) =====")
print(f"timing: {dt:.1f}s total, {dt/ROUNDS:.2f}s / (B={B},T={T}) rollout, "
      f"{B*T*ROUNDS/dt:.0f} env-steps/s")
print(f"R (reward 回报):   mean={R.mean():.3f} std={R.std():.3f} min={R.min():.3f} max={R.max():.3f}")
print(f"C_disc (折扣cost): mean={C.mean():.3f} std={C.std():.3f} "
      f"pct[50,80,90,95]={np.round(np.percentile(C,[50,80,90,95]),2)}")
print(f"C_undisc (Σcost):  mean={Cu.mean():.3f} std={Cu.std():.3f} "
      f"pct[50,80,90,95]={np.round(np.percentile(Cu,[50,80,90,95]),2)}")
print("outage 网格 (选 d 依据):")
for d in [0.5, 1, 2, 3, 5, 8, 12, 20, 30]:
    print(f"  d={d:>4g}: P(C_disc>=d)={np.mean(C>=d):.3f}   P(C_undisc>=d)={np.mean(Cu>=d):.3f}")
