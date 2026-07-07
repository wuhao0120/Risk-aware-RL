import gymnasium as gym
from gymnasium import spaces
import numpy as np


class RiskSensitiveEnv(gym.Env):
    """
    RiskSensitiveEnv (∞-horizon / continuing 变体) —— 风险敏感「无状态重复赌局」。

    与有限步版 (risk_sensitive_env/envs/risk_env.py) 的关系:
    ====================================================
    奖励分布【完全一致】(同 μ(r)/σ(r) 公式), 只改两点以表达 infinite-horizon:
      1. 【无状态】: 状态恒为常量 [1.0] (state_dim=1), 不再是 one-hot 时钟。
         → 回报-to-go 分布 G_t 与 t 无关 ⇒ 价值平稳 V(s)≡V ⇒ critic 用 step-blind 即正确
           (有限步版需要 one-hot 时钟 + critic_step_feature 来区分步数; 这里不需要)。
      2. 【continuing / 截断】: 任务不终止, 仅在第 n 步【截断】(truncated, 非 terminal)。
         numpy 版的 done=(step_count==n) 只用来【界定一段评估 rollout 的长度】, 使 MC 评估
         计算截断折扣回报 Z≈Σ_{t<n} γ^t r_t (γ=0.9, n=100 ⇒ 尾项 γ^100≈3e-5, ≈∞ 回报)。
         训练侧 (DQCACBetaGPU) 在截断点【bootstrap】 V(s_T) 而非当作终止 —— 见 GPU 智能体。

    奖励模型 (逐式复刻有限步版):
    ===========================
    动作 a ∈ R 直接作为风险等级 r:
      - 单步均值:   μ(r) = μ_base + k_mean · r
      - 单步标准差: σ(r) = softplus(σ_base + k_std · r)   (数值稳定写法, 见 _compute_reward)
      - 单步奖励:   reward_t ~ N(μ(r), σ(r)²)
    默认参数: μ_base=1.0, k_mean=0.8, σ_base=0.02, k_std=8.0 (与有限步版同分布)。

    解析真值 (常数动作 r, γ=0.9, 用于验收 critic 校准):
    ===================================================
    Z = Σ_{t≥0} γ^t r_t,  r_t ~ N(μ(r), σ(r)²) i.i.d.:
      - E[Z]   = μ(r) / (1-γ)        = 10 · (1 + 0.8r)
      - Std[Z] = σ(r) / √(1-γ²)      = 2.294 · softplus(0.02 + 8r)
      - Q_0.25[Z] ≈ E[Z] − 0.674·Std[Z]   (T=100 项, CLT 近似为正态)
    约束 P(Z≤q)≤α=0.25 在 Q_0.25=q 处恰好激活:
      - QPO (max 分位数) → r→0 → mean≈10, Q_0.25≈8.9 (最保守)
      - QCPO/DQCAC (max mean s.t. 约束), 例 q=8 → r≈0.45 → mean≈13.6 (满足约束, +36%)

    环境配置:
    =========
    - n: 一段 rollout 的步数 (continuing 任务的截断长度 T), 默认 100
    - 状态空间: 1 维常量 [1.0] (无状态; 价值平稳)
    - 动作空间: 1 维连续 (直接视为风险等级 r)
    """

    def __init__(self, n=100, mu_base=1.0, k_mean=0.8, sigma_base=0.02, k_std=8.0):
        """
        初始化环境

        Args:
            n: 一段 rollout 的步数 (continuing 截断长度 T, 默认 100)
            mu_base: 基础均值 (r=0 时的期望奖励)
            k_mean: 均值增长系数 (风险溢价)
            sigma_base: 基础标准差 (r=0 时的波动性)
            k_std: 标准差增长系数 (风险放大因子)
        """
        super().__init__()

        self.n = n                                         # rollout 截断长度 T (continuing)

        # 奖励分布参数 (与有限步版同名同值 → 同分布)
        self.mu_base = mu_base                             # μ_base
        self.k_mean = k_mean                               # k_mean (风险溢价)
        self.sigma_base = sigma_base                       # σ_base
        self.k_std = k_std                                 # k_std (风险放大)

        # 动作空间: 实数 (直接视为风险等级 r)
        self.action_space = spaces.Box(
            low=-np.inf * np.ones((1,)),
            high=np.inf * np.ones((1,)),
            dtype=np.float32
        )

        # 状态空间: 1 维常量 (无状态 → 价值平稳 V(s)≡V)
        self.observation_space = spaces.Box(
            low=np.zeros((1,)),
            high=np.ones((1,)),
            dtype=np.float32
        )

        # 运行时变量
        self.step_count = None                             # 当前步 (用于截断)
        self.risk_buf = None                               # 记录每步选择的风险等级 (供 render 日志)

    def _action_to_risk(self, action):
        """直接使用动作作为风险等级 r。"""
        return float(action[0])

    def _compute_reward(self, risk_level):
        """
        根据风险等级计算单步奖励 reward ~ N(μ(r), σ(r)²)。
        softplus 用数值稳定写法 (与有限步版逐式一致): 大 x 时 ≈x, 小 x 时不溢出。
        """
        mu = self.mu_base + self.k_mean * risk_level                       # μ(r)
        sigma_linear = self.sigma_base + self.k_std * risk_level           # σ 的线性前体
        sigma = np.log1p(np.exp(-np.abs(sigma_linear))) + np.maximum(sigma_linear, 0.0)  # 稳定 softplus
        return np.random.normal(mu, sigma)                                # 采样 N(μ, σ²)

    def step(self, action):
        """
        执行一步动作 (4 值 gym API)。

        Returns:
            state: 新状态 (恒为常量 [1.0])
            reward: 单步奖励 ~ N(μ(r), σ(r)²)
            done:  是否到截断点 (step_count==n); continuing 任务里仅用于界定评估 rollout 长度
            info:  {}
        """
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        risk_level = self._action_to_risk(action)          # 动作 → 风险等级 r
        self.risk_buf.append(risk_level)                   # 记录 (供 render)
        reward = self._compute_reward(risk_level)          # 单步奖励

        self.step_count += 1                               # 步数 +1
        done = (self.step_count == self.n)                 # 到达截断长度 T (非真正终止)
        state = np.ones((1,), dtype=np.float32)            # 次态恒为有效常量 (含截断点 s_T)
        return state, reward, done, {}

    def reset(self):
        """
        重置环境 (开始一段新的 rollout)。

        Returns:
            state: 初始状态 (常量 [1.0])
        """
        self.step_count = 0                                # 步数清零
        self.risk_buf = []                                 # 风险记录清零
        return np.ones((1,), dtype=np.float32)             # 常量初始状态

    def render(self, mode=None):
        """返回本段 rollout 的风险等级序列 (供日志记录)。"""
        return np.array(self.risk_buf) if self.risk_buf else None

    def close(self):
        return None
