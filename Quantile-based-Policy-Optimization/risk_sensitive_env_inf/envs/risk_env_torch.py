# -*- coding: utf-8 -*-
"""
RiskSensitiveVecTorch (∞-horizon / continuing 变体) —— 全 GPU、B 路并行向量化「无状态风险赌局」。

仅供 DQCACBetaGPU / QCPOGPU 的全 GPU 向量化训练使用。与有限步版
(risk_sensitive_env/envs/risk_env_torch.py) 奖励【逐式一致】, 区别仅两点 (与同目录 risk_env.py 对应):
    1. 【无状态】: 状态恒为常量 ones(B,1) → state_dim=1 → 价值平稳 V(s)≡V (critic step-blind 即正确)。
    2. 【continuing/截断】: 固定 n 步为一段 rollout, 第 n 步是【截断 truncated】而非终止;
       次态恒返回有效常量 (含截断点 s_T) → 供 GPU 智能体在截断点 bootstrap V(s_T)。
奖励模型 (复刻有限步版 _compute_reward):
    risk  = action[:, 0]
    mu    = mu_base + k_mean * risk
    sigma = softplus(sigma_base + k_std * risk)
    reward ~ N(mu, sigma)
参数默认从一个 RiskSensitiveEnv 实例 (ref_env) 读取, 保证与 baseline 完全同分布。
"""
import torch
import torch.nn.functional as F


class _Space:
    """轻量 shape 容器, 使父类 np.prod(env.observation_space.shape) 可用。"""

    def __init__(self, shape):
        self.shape = tuple(shape)


class RiskSensitiveVecTorch:
    """B 路并行、全 GPU 的 ∞-horizon 无状态风险赌局 (与 RiskSensitiveEnv 同分布)。"""

    def __init__(self, n=100, num_envs=256, device=None, ref_env=None,
                 mu_base=1.0, k_mean=0.8, sigma_base=0.02, k_std=8.0):
        # 若提供参考 env, 直接读取其奖励参数, 保证与 baseline 完全一致
        if ref_env is not None:
            n = ref_env.n
            mu_base, k_mean = ref_env.mu_base, ref_env.k_mean
            sigma_base, k_std = ref_env.sigma_base, ref_env.k_std

        self.n = int(n)                                    # 一段 rollout 步长 T (截断长度)
        self.B = int(num_envs)                             # 并行 env 数
        self.device = device if device is not None else torch.device('cpu')
        self.mu_base = float(mu_base)                      # 奖励均值基线 μ_base
        self.k_mean = float(k_mean)                        # 风险溢价系数 k_mean
        self.sigma_base = float(sigma_base)                # 波动基线 σ_base
        self.k_std = float(k_std)                          # 风险放大系数 k_std

        # 单 env 的 space shape: 无状态 → state_dim=1; action_dim=1
        self.observation_space = _Space((1,))
        self.action_space = _Space((1,))

        self.step_count = 0                                # 当前步 (B 路锁步)
        self._risk_buf = []                                # 每步 batch 平均 risk, 供 render()

    def reset(self):
        """重置全部 B 个 env, 返回常量初始状态 ones(B,1) (GPU)。"""
        self.step_count = 0
        self._risk_buf = []
        return torch.ones(self.B, 1, device=self.device)   # 无状态: 常量

    def step(self, actions):
        """
        actions: [B, 1] torch (GPU)。返回 (next_state [B,1], reward [B], done: bool)。
        奖励逐式复刻 RiskSensitiveEnv, 全程 GPU; 次态恒为常量 (含截断点 s_T, 供 bootstrap)。
        """
        risk = actions[:, 0]                                       # [B] 动作即风险等级 r
        self._risk_buf.append(risk.detach())
        mu = self.mu_base + self.k_mean * risk                     # [B] 单步均值 μ(r)
        sigma = F.softplus(self.sigma_base + self.k_std * risk)    # [B] 单步标准差 σ(r)
        reward = mu + sigma * torch.randn_like(mu)                 # ~ N(μ, σ), 全 GPU

        self.step_count += 1
        done = (self.step_count == self.n)                         # 截断点 (truncated, 非终止)
        next_state = torch.ones(self.B, 1, device=self.device)     # 次态恒为有效常量 (含 s_T)
        return next_state, reward, done

    def render(self, mode=None):
        """返回本段 rollout 每步的 batch 平均 risk 序列 [T] (numpy), 供日志。"""
        if not self._risk_buf:
            return None
        return torch.stack(self._risk_buf, dim=0).mean(dim=1).cpu().numpy()

    def close(self):
        return None
