# -*- coding: utf-8 -*-
"""生成 risk_sensitive_env_inf/train_evaluate.ipynb (∞-horizon 正式实验 notebook)。一次性脚本。"""
import json, os

BASE = os.path.dirname(os.path.abspath(__file__))
cells = []
def md(cid, text):   cells.append({"cell_type": "markdown", "id": cid, "metadata": {}, "source": text})
def code(cid, text): cells.append({"cell_type": "code", "execution_count": None, "id": cid, "metadata": {}, "outputs": [], "source": text})

md("title", r'''# ∞-horizon (continuing) 正式实验 — risk_sensitive_env_inf

无状态 continuing 风险赌局: obs≡常量 (价值平稳 V(s)≡V), 截断长度 T=`ENV_N`, 折扣 γ=`GAMMA`。
约束优化: **max E[Z]  s.t.  P(Z≤q) ≤ α=0.25**,  Z = Σ γ^t r_t。

**尺度说明 (为什么 100 步、回报还是 ~10 量级)**: 有效视界 = 1/(1−γ)。γ=0.9 → **10**, 折扣几何和 Σγ^t=10 →
回报 ≈ 10·μ (≈10~20), 不是 100·μ。T=100 只是让截断尾项 γ^100≈3e-5 可忽略。
约束要 bind 必须 **γ < 0.957** (否则多步噪声被 CLT 平均掉、分位数≈均值、约束失效)。

**怎么选 q**: 先跑「QPO 天花板扫描」cell。约束 P(Z≤q)≤α 可行需 **q ≤ Q₀.₂₅ 天花板** (γ=0.9 实测≈7.45@r0.1,mean10.8)。
q→约束最优(mean): q5.0→17.9, q5.5→16.4, **q6.0→15.5(推荐)**, q6.5→14.9, q7.0→13.2。

> 改了任何 `.py` 后**重启内核**, 否则跑的是缓存的旧模块。''')

code("imports", r'''import os, sys
# os.environ['WANDB_MODE'] = 'disabled'   # 取消注释可关闭 wandb 云日志 (无需登录)
import torch
import numpy as np
import random
import wandb

# 确保导入【本目录 risk_sensitive_env_inf】的 envs/agents/utils (而非同名的有限步版)
BASE_DIR = os.getcwd()
if not os.path.isdir(os.path.join(BASE_DIR, 'agents')):
    BASE_DIR = r"E:\Rist-aware RL\Quantile-based-Policy-Optimization\risk_sensitive_env_inf"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from agents import DQCACBetaGPU, QCPOGPU
from envs import RiskSensitiveEnv, RiskSensitiveVecTorch
from utils.evaluation import monte_carlo_evaluate_constraint

# ===================== 共享实验配置 (只改这里, DQCAC 与 QCPO 自动对齐) =====================
GAMMA       = 0.9     # ∞-horizon 折扣; 有效视界 1/(1-γ)=10; 必须 < 0.957 否则约束失效
ENV_N       = 100     # 截断长度 T (continuing); γ^100≈3e-5 → 尾部可忽略
Q_THRESHOLD = 6.0     # 约束阈值 q; 天花板≈7.45; q=6.0 → mean≈15.5 (+43% vs QPO≈10.8)
SEED        = 0
WANDB_PROJECT = 'risk_sensitive_inf'   # ∞-horizon 独立 wandb project (不与有限步 RiskSensitiveEnv 混)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"BASE_DIR={BASE_DIR}")
print(f"device={DEVICE}  gamma={GAMMA}  T(ENV_N)={ENV_N}  q={Q_THRESHOLD}  seed={SEED}")''')

md("sweep-md", r'''## 0. QPO 天花板扫描 (先跑这个, 据此定 q)

直接 MC 刻画【随机策略类】的 mean(r) / Q₀.₂₅(r) 曲线 (策略 a~N(r, σ²), σ=init_std 固定)。
**QPO 收敛点 = argmax_r Q₀.₂₅(r)** (最保守, 分位数天花板)。约束可行需当前 `Q_THRESHOLD` ≤ 该天花板。
比训练 QPO agent 更快更全 (给出整条权衡曲线)。''')

code("sweep", r'''_sigma = float(np.sqrt(1e-1))   # 策略动作噪声 (= init_std, 固定 log_std, 与训练一致)
_B = 40000                      # 每个 r 的 MC 轨迹数

@torch.no_grad()
def _eval_w(w):
    """固定均值动作 w 的随机策略 a~N(w,σ²): 采 _B 条轨迹, 返回截断折扣回报 Z [_B]。"""
    e = RiskSensitiveVecTorch(num_envs=_B, device=DEVICE, ref_env=RiskSensitiveEnv(n=ENV_N))
    e.reset(); Z = torch.zeros(_B, device=DEVICE); disc = 1.0
    for _ in range(ENV_N):
        a = w + _sigma * torch.randn(_B, 1, device=DEVICE)
        _, r, _ = e.step(a); Z += disc * r; disc *= GAMMA
    return Z

print(f"{'r':>6}{'mean':>8}{'std':>8}{'Q0.25':>8}{'P(<=q)':>8}   (q={Q_THRESHOLD}, sigma={_sigma:.3f})")
_rows = []
for w in np.round(np.arange(-0.1, 1.31, 0.1), 2):
    Z = _eval_w(float(w)).cpu().numpy()
    row = (float(w), float(Z.mean()), float(Z.std()), float(np.percentile(Z, 25)), float((Z <= Q_THRESHOLD).mean()))
    _rows.append(row)
    print(f"{row[0]:>6.2f}{row[1]:>8.2f}{row[2]:>8.2f}{row[3]:>8.3f}{row[4]:>8.3f}")
_qpo = max(_rows, key=lambda x: x[3])
print(f"\n[QPO 天花板] r*={_qpo[0]:.2f}  Q0.25_max={_qpo[3]:.3f}  mean={_qpo[1]:.2f} (最保守)")
print(f"[可行性]    约束需 q <= {_qpo[3]:.2f}; 当前 q={Q_THRESHOLD} -> "
      + ("可行 OK" if Q_THRESHOLD < _qpo[3] else "不可行! 调低 Q_THRESHOLD"))''')

md("dqc-md", r'''## 1. DQC-AC-β GPU (∞-horizon)

无状态平稳价值 + 截断 bootstrap (`critic_step_feature=False`, `_rollout_vec` 恒 bootstrap 截断次态)。
critic-dual 定版旋钮: `huber_kappa=0.1` (critic 无偏) · `target_tau=0.05` (消除滞后) · `warmup_iters=30` ·
`advantage_norm='qcpo'` · `n_step=1` · `beta=0.90`。总 env-step = num_iterations × num_envs × ENV_N。
> B=512/1500 迭代约 ~100min; 想快可 `num_envs=256` (约 ~30min) 或 `num_iterations=800`。''')

code("dqc-train", r'''class DQCACBetaGPUArgs:
    def __init__(self):
        # 基础
        self.env_name = 'RiskSensitiveEnv'; self.seed = SEED; self.device = DEVICE
        self.algo_name = 'DQCACBetaGPU'; self.wandb_name = 'DQCACBetaGPU_inf'; self.wandb_dir = None
        self.wandb_project = WANDB_PROJECT
        # 向量化规模
        self.num_envs = 512
        self.num_iterations = 1500       # critic 校准+mean 收敛需 ~1200+ (B=256 可减半墙钟)
        self.log_interval = 50; self.est_interval = 100
        # ∞-horizon 尺度 (共享常量)
        self.gamma = GAMMA               # 0.9
        self.beta = 0.90                 # Abel 风险折扣 β (仅 actor 约束项)
        # 约束
        self.q_alpha = 0.25; self.quantile_threshold = Q_THRESHOLD
        # Actor LR (每迭代内层多次更新, 保守)
        self.init_std = float(np.sqrt(1e-1))
        self.theta_a = (10000 ** 0.9) * 2e-4; self.theta_b = 10000; self.theta_c = 0.9
        self.q_a = (10000 ** 0.6) * 1e-2; self.q_b = 10000; self.q_c = 0.6   # 兼容字段
        # Dual λ (概率约束, 不乘 β; dual 用 critic 估计的 P(Z≤q))
        self.lambda_a = 0.3; self.lambda_b = 5000; self.lambda_c = 0.1
        self.lambda_max = 50.0; self.lambda_min = 0.0; self.outer_interval = 1
        # 分布式 critic / QRTD (定版旋钮)
        self.num_quantiles = 32
        self.huber_kappa = 0.1           # 【关键】近纯分位数回归 → critic 无偏
        self.critic_hidden = [64, 64]; self.critic_lr = 1e-3
        self.target_tau = 0.05           # 消除 critic 滞后 → 避免 λ 极限环
        self.target_update_interval = 1
        self.n_step = 1                  # 原生 per-transition; ≥2 失控, 别改
        self.critic_step_feature = False # ∞-horizon 无状态 → 价值平稳 → step-blind 正确
        self.num_action_samples = 4; self.updates_per_episode = 10
        # 优势归一化 / 数值稳定
        self.advantage_norm = 'qcpo'; self.norm_ema_decay = 0.1; self.warmup_iters = 30
        self.entropy_coef = 0.0; self.actor_grad_clip = 100.0; self.critic_grad_clip = 10.0

args_dqc = DQCACBetaGPUArgs()
random.seed(args_dqc.seed); np.random.seed(args_dqc.seed); torch.manual_seed(args_dqc.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args_dqc.seed)

env_dqc = RiskSensitiveEnv(n=ENV_N)
agent_DQCAC = DQCACBetaGPU(args_dqc, env_dqc)
agent_DQCAC.train()
wandb.finish()''')

md("qcpo-md", r'''## 2. QCPO GPU 参考 (∞-horizon)

MC / 轨迹级对偶的 QCPO 参考 (与 DQCAC 同 q, 公平对比)。无 critic、无 bootstrap (截断回报≈∞)。
预期: 在约束边界**来回震荡** (方差大于 DQCAC 的 per-transition TD 信号), 单快照可能略越界。''')

code("qcpo-train", r'''class QCPOGPUArgs:
    def __init__(self):
        self.env_name = 'RiskSensitiveEnv'; self.seed = SEED; self.device = DEVICE
        self.algo_name = 'QCPOGPU'; self.wandb_name = 'QCPOGPU_inf'; self.wandb_dir = None
        self.wandb_project = WANDB_PROJECT
        self.num_envs = 512; self.num_iterations = 800
        self.log_interval = 50; self.est_interval = 100
        self.gamma = GAMMA
        self.q_alpha = 0.25; self.quantile_threshold = Q_THRESHOLD
        self.init_std = float(np.sqrt(1e-1))
        self.theta_a = (10000 ** 0.9) * 1e-3; self.theta_b = 10000; self.theta_c = 0.9   # QCPO 原始系数
        self.lambda_a = 0.3; self.lambda_b = 5000; self.lambda_c = 0.1
        self.outer_interval = 1; self.updates_per_episode = 10
        self.norm_ema_decay = 0.01       # return_rms decay (QCPO 默认)
        self.actor_grad_clip = 0.0       # QCPO 不裁 actor 梯度
        self.warmup_rms_iters = 2

args_qcpo = QCPOGPUArgs()
random.seed(args_qcpo.seed); np.random.seed(args_qcpo.seed); torch.manual_seed(args_qcpo.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args_qcpo.seed)

env_qcpo = RiskSensitiveEnv(n=ENV_N)
agent_QCPO = QCPOGPU(args_qcpo, env_qcpo)
agent_QCPO.train()
wandb.finish()''')

md("eval-md", r'''## 3. 评估 (后训练 3000 条 eval 轨迹, 真值 + critic 校准)''')

code("dqc-eval", r'''res_dqc = monte_carlo_evaluate_constraint(agent_DQCAC, env_dqc, num_episodes=3000, quantile_threshold=Q_THRESHOLD)
print(f"[DQCAC-beta inf]  mean={res_dqc['mean']:.3f}   Q0.25={res_dqc['quantile']:.3f}  (应 >= q={Q_THRESHOLD})")
print(f"                  P(Z<=q)={res_dqc['empirical_prob']:.3f}  (eval真值, 应 <= alpha=0.25)   std={res_dqc['std']:.3f}")
print(f"                  critic cdf_init={res_dqc['cdf_initial']:.3f}  (critic预测违约率, 校准好时 ≈ P(Z<=q))")''')

code("qcpo-eval", r'''res_qcpo = monte_carlo_evaluate_constraint(agent_QCPO, env_qcpo, num_episodes=3000, quantile_threshold=Q_THRESHOLD)
print(f"[QCPO inf ref]    mean={res_qcpo['mean']:.3f}   Q0.25={res_qcpo['quantile']:.3f}")
print(f"                  P(Z<=q)={res_qcpo['empirical_prob']:.3f}  (QCPO 在边界震荡, 单快照可能略越界)   std={res_qcpo['std']:.3f}")''')

code("compare", r'''print(f"{'algo':<16}{'mean':>8}{'Q0.25':>9}{'P(Z<=q)':>10}{'std':>8}    (q={Q_THRESHOLD}, alpha=0.25)")
print(f"{'DQCAC-beta inf':<16}{res_dqc['mean']:>8.2f}{res_dqc['quantile']:>9.2f}{res_dqc['empirical_prob']:>10.3f}{res_dqc['std']:>8.2f}")
print(f"{'QCPO inf ref':<16}{res_qcpo['mean']:>8.2f}{res_qcpo['quantile']:>9.2f}{res_qcpo['empirical_prob']:>10.3f}{res_qcpo['std']:>8.2f}")
print(f"\n对照 QPO 天花板(扫描 cell): mean≈10.8, Q0.25≈7.45 (最保守)。")
print(f"约束方法应: mean 大幅 > QPO, Q0.25 钉在 ≈q, P(Z<=q) <= 0.25; DQCAC 应比 QCPO 更稳/更可行。")''')

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
out = os.path.join(BASE, "train_evaluate.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("written:", out, "(", len(cells), "cells )")
