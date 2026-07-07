# -*- coding: utf-8 -*-
"""
_sweep.py —— 用 MC 直接刻画【随机策略类】在 ∞-horizon 环境上的 mean(r) / Q_alpha(r) 曲线。

目的: 在选 quantile_threshold q 之前, 先量出环境的"理论最优值":
  - QPO 收敛到的点 = argmax_r Q_alpha(r) (最保守, 分位数天花板 Q_max)
  - 任何约束 P(Z<=q)<=alpha 等价于 Q_alpha(r) >= q → 可行要求 q <= Q_max
  - QCPO/DQCAC 收敛到: 满足 Q_alpha(r)>=q 的【最大 mean】策略 (即最大 r)
策略与训练/评估一致: a_t ~ N(w, sigma^2) 每步独立采样, sigma=init_std=sqrt(0.1), log_std 固定。
"""
import os, sys
os.environ['WANDB_MODE'] = 'disabled'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import numpy as np
import torch
from envs import RiskSensitiveVecTorch, RiskSensitiveEnv

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ref = RiskSensitiveEnv(n=100)                 # 取奖励参数 (与训练同分布)
gamma = 0.9                                   # ∞-horizon 折扣
sigma = float(np.sqrt(0.1))                   # 策略动作噪声 (= init_std, 固定 log_std)
alpha = 0.25                                  # 约束水平
n = ref.n                                     # 截断长度 T=100
B = 40000                                     # 每个 r 的 MC 轨迹数 (分位数估计够紧)


@torch.no_grad()
def eval_w(mu_w):
    """固定均值动作 w 的随机策略 a~N(w,sigma^2): 采 B 条轨迹, 返回截断折扣回报 Z [B]。"""
    env = RiskSensitiveVecTorch(num_envs=B, device=dev, ref_env=ref)
    env.reset()
    Z = torch.zeros(B, device=dev)
    disc = 1.0
    for _ in range(n):
        a = mu_w + sigma * torch.randn(B, 1, device=dev)   # a~N(w, sigma^2)
        _, r, _ = env.step(a)
        Z += disc * r
        disc *= gamma
    return Z


grid = np.round(np.arange(-0.10, 1.31, 0.10), 2)
print(f"sigma={sigma:.3f} gamma={gamma} T={n} B={B} alpha={alpha}")
print(f"{'r(w)':>6} {'mean':>8} {'std':>8} {'Q0.25':>8} {'P(<=5)':>7} {'P(<=5.5)':>8} {'P(<=6)':>7} {'P(<=6.5)':>8}")
rows = []
for w in grid:
    Z = eval_w(float(w)).cpu().numpy()
    mean = Z.mean(); std = Z.std(); q25 = np.percentile(Z, 25)
    p5 = (Z <= 5).mean(); p55 = (Z <= 5.5).mean(); p6 = (Z <= 6).mean(); p65 = (Z <= 6.5).mean()
    rows.append((float(w), mean, std, q25, p5, p55, p6, p65))
    print(f"{w:6.2f} {mean:8.2f} {std:8.2f} {q25:8.3f} {p5:7.3f} {p55:8.3f} {p6:7.3f} {p65:8.3f}")

# QPO 天花板 = 最大化 Q_0.25 的点
qpo = max(rows, key=lambda x: x[3])
print(f"\n[QPO ceiling]  r*={qpo[0]:.2f}  Q0.25_max={qpo[3]:.3f}  mean={qpo[1]:.2f}  (最保守, 分位数最高)")
print(f"[feasible q ]  约束 P(Z<=q)<=0.25 可行需 q <= Q0.25_max={qpo[3]:.3f}; 留余量取 q≈{qpo[3]-1.5:.1f}~{qpo[3]-0.5:.1f}")

# 对每个候选 q, 找满足 Q_0.25(r)>=q 的最大 r (= 约束最优), 报其 mean
print("\n候选 q 的约束最优 (满足 Q0.25(r)>=q 的最大 r → 最大 mean):")
for q in [5.0, 5.5, 6.0, 6.5, 7.0]:
    feas = [row for row in rows if row[3] >= q]           # Q0.25(r) >= q 的 r
    if feas:
        best = max(feas, key=lambda x: x[0])              # 最大 r → 最大 mean
        print(f"  q={q:4.1f}: 约束最优 r≈{best[0]:.2f}, mean≈{best[1]:.2f}, Q0.25≈{best[3]:.3f}, P(Z<=q)≈{(np.nan if False else 0):.0f}".replace(' P(Z<=q)≈0',''))
    else:
        print(f"  q={q:4.1f}: 不可行 (无 r 使 Q0.25(r)>=q; q 超过天花板 {qpo[3]:.2f})")
