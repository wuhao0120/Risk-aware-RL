# -*- coding: utf-8 -*-
"""
_noisy_calib.py —— 带探索噪声的参考策略校准 (临时诊断脚本, 用完即删)。

动机 (v3 诊断): calibration.json 是【确定性】参考策略的可行性表 (hedge45 Q=2.49 可行),
但训练/评估的策略类是 logits 上加 σ=0.35 高斯噪声的随机策略 —— QPO (分位数最大化器,
即策略类天花板探测器) 只到 Q=1.55, DQCAC λ 顶满仍 P=0.285 —— 怀疑 q=2.0 在
σ=0.35 噪声策略类下不可行 (risk env 的 q-direction 教训: 必须在同噪声策略类下先标定)。

方法: tilt45 网格 w4∈[0,0.5] (w5=1-w4) + uniform/全仓股5, 在 σ∈{0, 0.35} 下
用与 evaluate_policy_vec 完全相同的协议测 mean/Q_0.25/P(Z≤2)。
用法: python -u _noisy_calib.py [num_episodes]
"""
import os, sys
os.environ['WANDB_MODE'] = 'disabled'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import numpy as np
import torch

from envs import PortfolioEnv, PortfolioVecTorch
from utils import evaluate_policy_vec

E = int(sys.argv[1]) if len(sys.argv) > 1 else 8192            # 每策略评估轨迹数
GAMMA, ALPHA, Q = 0.9, 0.25, 2.0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0); np.random.seed(0)


class _NoisyConstPolicy:
    """常量 logits + σ·ε 随机策略 (与训练策略类同形: softmax(N(log w, σ²)))。"""

    def __init__(self, w_ref, sigma, device):
        w = np.asarray(w_ref, dtype=np.float64)
        w = np.maximum(w, 1e-8)
        self._logits = torch.as_tensor(np.log(w / w.sum()), dtype=torch.float32,
                                       device=device)
        self.sigma = float(sigma)
        self.device = device

    def _sample_actions(self, states):
        mu = self._logits.unsqueeze(0).expand(states.shape[0], -1)
        if self.sigma <= 0:
            return mu
        return mu + torch.randn_like(mu) * self.sigma


def _collect_Z(agent, vec, E):
    """与 evaluate_policy_vec 同协议收全部回报 Z numpy (q 网格逐点算 P 用)。"""
    import math
    B, n = vec.B, vec.n
    Z_all = []
    with torch.no_grad():
        for _ in range(max(1, math.ceil(E / B))):
            s = vec.reset()
            z = torch.zeros(B, device=s.device)
            disc = 1.0
            for _t in range(n):
                a = agent._sample_actions(s)
                s, r, _ = vec.step(a)
                z += disc * r
                disc *= GAMMA
            Z_all.append(z)
    return torch.cat(Z_all).cpu().numpy()


env = PortfolioEnv(n=100)
refs = [('uniform', np.full(5, 0.2)), ('stock4(maxmu)', np.eye(5)[4])]
for w4 in (0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20):
    w = np.zeros(5); w[3] = w4; w[4] = 1.0 - w4
    refs.append((f'tilt45_a{w4:.2f}', w))
# 频率前沿 sanity check: 在 binding 区掺少量对冲对 (2,3) 看能否扩张前沿
refs.append(('mix_h23_10%', np.array([0, .05, .05, .35, .55])))
refs.append(('mix_h23_20%', np.array([0, .10, .10, .30, .50])))

QGRID = (0.5, 0.75, 1.0, 1.25, 1.5)
sigma = 0.35
print(f"\n===== sigma={sigma} (E={E}, gamma={GAMMA}, alpha={ALPHA}) =====")
print(f"{'policy':16s} {'mean':>7s} {'std':>7s} {'Q_a':>7s} " +
      ' '.join(f"P<={q:<4g}" for q in QGRID))
for name, w in refs:
    vec = PortfolioVecTorch(num_envs=4096, device=device, ref_env=env)
    agent = _NoisyConstPolicy(w, sigma, device)
    zs = _collect_Z(agent, vec, E)
    qa = float(np.percentile(zs, ALPHA * 100))
    probs = ' '.join(f"{float(np.mean(zs <= q)):7.3f}" for q in QGRID)
    print(f"{name:16s} {zs.mean():7.3f} {zs.std():7.3f} {qa:7.3f} {probs}")
