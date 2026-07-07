# -*- coding: utf-8 -*-
"""
_debug_probe.py —— QPO/QCPO 退化机制探针 (临时诊断脚本, 用完即删)。

假设: 策略线性层 W 在含噪 est 特征 (obs 前 20 维) 上随机游走增长 →
      状态噪声 × W 放大成 logits 方差 → 组合权重乱跳 → 回报方差爆炸、分位数崩塌。
探针: 每 25 迭代打印 W 各特征块的范数、策略 logits 的状态致方差、q_est 追踪误差、
      batch 回报统计 → 确认/否证。
用法: python _debug_probe.py QPO 300
"""
import os, sys
os.environ['WANDB_MODE'] = 'disabled'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import numpy as np
import torch
import random
import wandb

# wandb 全程打空 (探针不需要日志; init 也空防 service 竞争)
wandb.log = lambda *a, **k: None
wandb.init = lambda *a, **k: None
wandb.define_metric = lambda *a, **k: None

from run_experiment import base_args
from envs import PortfolioEnv
from agents import QPOGPU, QCPOGPU, QPPOGPU

algo = sys.argv[1] if len(sys.argv) > 1 else 'QPO'
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 300

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = base_args(algo, seed=0, device=device)
args.num_iterations = iters
args.log_interval = 10 ** 9                       # 关掉常规打印

# 额外 key=value 覆盖 (如 theta_lr0=0.2 → 重算 theta_a)
# 注意: theta_lr0 必须在所有覆盖之后处理 (否则用旧 theta_b 算 theta_a → lr(0) 不等于给定值,
#       本探针曾因此把 lr0=0.6 实际跑成 0.12, 见 DESIGN.md)
from run_experiment import cast
_lr0 = None
for kv in sys.argv[3:]:
    if '=' not in kv:
        continue
    k, v = kv.split('=', 1)
    if k == 'theta_lr0':                          # 便捷: 直接给 lr(0), 最后重算 theta_a
        _lr0 = float(v)
    else:
        setattr(args, k, cast(v))
if _lr0 is not None:
    args.theta_a = (args.theta_b ** args.theta_c) * _lr0
print(f"[probe] effective theta lr(0) = {args.theta_a / (args.theta_b ** args.theta_c):.4f}")

random.seed(0); np.random.seed(0); torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

env = PortfolioEnv(n=args.env_n)
agent = {'QPO': QPOGPU, 'QCPO': QCPOGPU, 'QPPO': QPPOGPU}[algo](args, env)

# ---- monkeypatch _log_core: 注入探针 ----
orig_log = agent._log_core
K = agent.action_dim


def probe_log(it, z_np, q_est_value, extra=None):
    out = orig_log(it, z_np, q_est_value, extra)
    if it % 25 == 0:
        W = agent.actor.model[0].weight.detach()              # [K, in_dim] (weights 模式: [K,K])
        bias = agent.actor.model[0].bias.detach()             # [K]
        # 状态致 logits 方差: 用一批观测算 μ(s) 的跨样本 std (均值 over K 维)
        with torch.no_grad():
            s = agent.vec_env.reset()                          # [B, sd] 新鲜观测
            logits = agent.actor(agent._actor_in(s))           # [B, K]
            state_std = float(logits.std(dim=0).mean().item()) # 状态噪声引起的 logits 波动
            mean_w = torch.softmax(logits, dim=1).mean(dim=0).cpu().numpy()  # 平均目标权重
        emp_q = float(np.percentile(z_np, agent.q_alpha * 100))
        lam = float(agent.lambda_dual.item()) if hasattr(agent, 'lambda_dual') else float('nan')
        qe = float(q_est_value)
        wstr = '/'.join(f"{x:.2f}" for x in mean_w)
        print(f"it={it:4d} |W|={W.norm():.3f} |b|={bias.norm():.3f} stateStd={state_std:.3f} "
              f"q_est={qe:.3f} empQ={emp_q:.3f} mean={np.mean(z_np):.3f} "
              f"std={np.std(z_np):.3f} lam={lam:.3f} w={wstr}")
    return out


agent._log_core = probe_log
agent.train()
