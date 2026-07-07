# -*- coding: utf-8 -*-
"""
_calib2.py —— 实验A (2资产反相关对) 的带噪声 q 校准 (临时脚本, 用完即删)。

背景: 实验A 把 5 资产砍到仅 股4/股5 反相关对 (μ=[.2,.6], ρ≈-.95)。5 资产的 q=1.0 是在
σ=0.35 噪声【5维 softmax】策略类下标的; 2 资产无"权重漏给低均值股0~3"的污染, 同名义倾斜
a 下回报更干净 (左尾更轻) → 噪声天花板 Q 可能更高、可行域更宽 → 必须重标 q。

方法 (与 _noisy_calib.py / evaluate_policy_vec 同协议): tilt 网格 a=股4权重∈[0,0.5]
(a=0 即全仓股5=风险中性最优, a 越大越保守), σ=0.35, 测 mean/Q_0.25/P(Z≤q) 网格。
用法: python -u _calib2.py [num_episodes]
"""
import os, sys, math
os.environ['WANDB_MODE'] = 'disabled'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import numpy as np
import torch

from envs import PortfolioEnv, PortfolioVecTorch

E = int(sys.argv[1]) if len(sys.argv) > 1 else 8192            # 每策略评估轨迹数
GAMMA, ALPHA = 0.9, 0.25
SIGMA = 0.35
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0); np.random.seed(0)

# 2资产配置 = 5资产的 股4(idx3)/股5(idx4): μ=[.20,.60], Σ=[[.16,-.19],[-.19,.25]] (ρ≈-.95)
MU2 = [0.20, 0.60]
SIGMA2 = [[0.16, -0.19], [-0.19, 0.25]]


class _NoisyConstPolicy:
    """常量 logits + σ·ε 随机策略 (与训练策略类同形)。"""

    def __init__(self, w_ref, sigma, device):
        w = np.maximum(np.asarray(w_ref, dtype=np.float64), 1e-8)
        self._logits = torch.as_tensor(np.log(w / w.sum()), dtype=torch.float32, device=device)
        self.sigma = float(sigma)

    def _sample_actions(self, states):
        mu = self._logits.unsqueeze(0).expand(states.shape[0], -1)
        return mu if self.sigma <= 0 else mu + torch.randn_like(mu) * self.sigma


def _collect_Z(agent, vec, E):
    Z_all = []
    with torch.no_grad():
        for _ in range(max(1, math.ceil(E / vec.B))):
            s = vec.reset(); z = torch.zeros(vec.B, device=s.device); disc = 1.0
            for _t in range(vec.n):
                s, r, _ = vec.step(agent._sample_actions(s)); z += disc * r; disc *= GAMMA
            Z_all.append(z)
    return torch.cat(Z_all).cpu().numpy()


env = PortfolioEnv(n=100, mu=MU2, sigma=SIGMA2)               # a = 股4(idx0)权重
refs = [('stock5(maxmu,a=0)', [0.0, 1.0])]
for a in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
    refs.append((f'tilt_a{a:.2f}', [a, 1.0 - a]))

QGRID = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
print(f"\n===== 2asset sigma={SIGMA} (E={E}, gamma={GAMMA}, alpha={ALPHA}) =====")
print(f"{'policy':18s} {'mean':>7s} {'std':>7s} {'Q_a':>7s} " +
      ' '.join(f"P<={q:<4g}" for q in QGRID))
for name, w in refs:
    vec = PortfolioVecTorch(num_envs=4096, device=device, ref_env=env)
    zs = _collect_Z(_NoisyConstPolicy(w, SIGMA, device), vec, E)
    qa = float(np.percentile(zs, ALPHA * 100))
    probs = ' '.join(f"{float(np.mean(zs <= q)):7.3f}" for q in QGRID)
    print(f"{name:18s} {zs.mean():7.3f} {zs.std():7.3f} {qa:7.3f} {probs}")
