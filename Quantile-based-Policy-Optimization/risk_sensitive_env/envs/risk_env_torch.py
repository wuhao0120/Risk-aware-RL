# -*- coding: utf-8 -*-
"""
RiskSensitiveVecTorch —— RiskSensitiveEnv 的【全 GPU、B 路并行】torch 向量化版本。

仅供 DQCACBetaGPU 的全 GPU 向量化训练使用; 【不改动】envs/risk_env.py。
奖励模型【逐式复刻】RiskSensitiveEnv._compute_reward:
    risk  = action[:, 0]
    mu    = mu_base + k_mean * risk
    sigma = softplus(sigma_base + k_std * risk)      # 与 env 的数值稳定 softplus 等价
    reward ~ N(mu, sigma)
状态: n 维 one-hot, 按每个 env 独立洗牌的 order 给出当前步位置; 固定 n 步、B 路锁步。
参数默认从一个 RiskSensitiveEnv 实例 (ref_env) 读取, 保证与 baseline 完全同分布。
"""
import torch
import torch.nn.functional as F


class _Space:
    """轻量 shape 容器, 使父类 np.prod(env.observation_space.shape) 可用。"""

    def __init__(self, shape):
        self.shape = tuple(shape)


class RiskSensitiveVecTorch:
    """B 路并行、全 GPU 的风险敏感环境 (与 RiskSensitiveEnv 同分布)。"""

    def __init__(self, n=10, num_envs=256, device=None, ref_env=None,
                 mu_base=1.0, k_mean=0.8, sigma_base=0.02, k_std=8.0):
        # 若提供参考 env, 直接读取其奖励参数, 保证与 baseline 完全一致
        if ref_env is not None:
            n = ref_env.n
            mu_base, k_mean = ref_env.mu_base, ref_env.k_mean
            sigma_base, k_std = ref_env.sigma_base, ref_env.k_std

        self.n = int(n)                                    # episode 步长 (固定)
        self.B = int(num_envs)                             # 并行 env 数
        self.device = device if device is not None else torch.device('cpu')
        self.mu_base = float(mu_base)                      # 奖励均值基线 μ_base
        self.k_mean = float(k_mean)                        # 风险溢价系数 k_mean
        self.sigma_base = float(sigma_base)                # 波动基线 σ_base
        self.k_std = float(k_std)                          # 风险放大系数 k_std

        # 单 env 的 space shape (供父类推断 state_dim=n, action_dim=1)
        self.observation_space = _Space((self.n,))
        self.action_space = _Space((1,))

        self.step_count = 0                                # 当前步 (B 路锁步)
        self.order = None                                  # [B, n] 每个 env 的位置洗牌
        self._risk_buf = []                                # 每步 batch 平均 risk, 供 render()

    def reset(self):
        """重置全部 B 个 env, 返回初始 one-hot 状态 [B, n] (GPU)。"""
        self.step_count = 0
        self._risk_buf = []
        # argsort(rand) 等价于每个 env 独立随机置换 (复刻 np.random.shuffle(order))
        self.order = torch.argsort(
            torch.rand(self.B, self.n, device=self.device), dim=1
        )                                                  # [B, n]
        return self._one_hot(self.step_count)

    def _one_hot(self, t):
        """第 t 步的 one-hot 状态 [B, n]: 每行在 order[:, t] 处置 1。"""
        state = torch.zeros(self.B, self.n, device=self.device)
        rows = torch.arange(self.B, device=self.device)
        state[rows, self.order[:, t]] = 1.0
        return state

    def step(self, actions):
        """
        actions: [B, 1] torch (GPU)。返回 (next_state [B, n], reward [B], done: bool)。
        奖励逐式复刻 RiskSensitiveEnv, 全程 GPU。
        """
        risk = actions[:, 0]                                       # [B] 动作即风险等级 r
        self._risk_buf.append(risk.detach())
        mu = self.mu_base + self.k_mean * risk                     # [B] 单步均值 μ(r)
        sigma = F.softplus(self.sigma_base + self.k_std * risk)    # [B] 单步标准差 σ(r)
        reward = mu + sigma * torch.randn_like(mu)                 # ~ N(μ, σ), 全 GPU

        self.step_count += 1
        done = (self.step_count == self.n)                         # 固定 n 步, B 路锁步
        if done:
            next_state = torch.zeros(self.B, self.n, device=self.device)  # 终止态全 0 (同 baseline)
        else:
            next_state = self._one_hot(self.step_count)
        return next_state, reward, done

    def render(self, mode=None):
        """返回本 episode 每步的 batch 平均 risk 序列 [T] (numpy), 供日志。"""
        if not self._risk_buf:
            return None
        return torch.stack(self._risk_buf, dim=0).mean(dim=1).cpu().numpy()

    def close(self):
        return None
