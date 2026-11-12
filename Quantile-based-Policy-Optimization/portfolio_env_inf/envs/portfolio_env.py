# -*- coding: utf-8 -*-
"""
PortfolioEnv (∞-horizon / continuing 变体) —— 平稳化改造的投资组合管理环境 (numpy 参考版)。

与原版 portfolio_management/envs/portfolio_management.py 的关系与差异
=====================================================================
本环境保留原版的核心结构 (5 只股票 + 两对强负相关对冲组 + 交易成本 + softmax 权重动作 +
滚动窗口估计 μ/Σ 作为观测), 但为了适配 ∞-horizon (continuing) 风险约束算法
(QPO/QCPO/QPPO/DQCAC, 见 risk_sensitive_env_inf), 做了 5 处修改:

【差异 1: 奖励平稳化】 绝对价值增量 → 单步百分比收益
    原版: reward_t = V_{t+1} - V_t  (绝对量, 财富 V 复合增长 ⇒ 奖励尺度随时间漂移,
          无限视野下发散, 违反"价值平稳 V(s)≡V"前提)
    本版: reward_t = 100·(κ_t·Σ_i u_i·g_i - 1)  (百分比单步收益, 只依赖权重与价格比,
          与财富水平无关 ⇒ 奖励分布平稳)。κ_t 为交易成本因子, g_i 为单步毛收益。
    可选 reward_mode='log' 用 100·log(κ·Σug) (对数收益, 可加性好); 默认 'pct':
          因为 max E[Σ pct] 的风险中性最优 = 全仓最高 μ 股票 (最大方差),
          约束的"塑形"效果最清晰; 而 log 收益自带 -σ²/2 拖累 (Kelly), 未加约束就
          已经偏向对冲, 会稀释约束算法的对比效果。

【差异 2: 观测平稳化】 去掉原始价格与持仓股数
    原版 obs = (est_mu, tril(est_sigma), position(股数), price(原始价))  共 30 维
          —— price 是几何随机游走 (水平漂移发散), position 随财富复合增长, 两者非平稳。
    本版 obs = (est_mu(%/步), tril(est_cov)((%/步)²·0.1), w(当前权重))  共 25 维
          —— est_mu/est_cov 是收益率统计 (天然平稳, 与原版同源),
             持仓改用权重 w∈单纯形 (平稳; 因有交易成本, 持仓状态仍必须保留)。
    缩放: est_cov 各项 ∈ ~[-19, 25], 统一乘 cov_scale=0.1 拉到 O(1), 利于网络条件数。

【差异 3: 真 GBM 模拟】 Euler 加性更新 → 精确几何布朗运动
    原版: dS = S·N(μ·dt, Σ·dt) (Euler 离散, 价格可能为负, 价格比分布依赖路径)
    本版: g = exp((μ - diag(Σ)/2)·dt + √dt·L·z), z~N(0,I), L=cholesky(Σ)
          (精确 GBM 解 ⇒ 单步毛收益 g 独立同分布、恒为正 ⇒ 环境严格平稳,
           一阶矩 E[g_i]-1 ≈ μ_i·dt 与原版一致)。

【差异 4: 漂移参数 μ 放大】 [0.01..0.05] → [0.02, 0.06, 0.10, 0.20, 0.60]
    原版年化漂移 1%~5%: 单步收益 信噪比 μ·dt/(σ·√dt) ≈ 0.01 —— 均值差异完全淹没
    在噪声里 (B=512 条轨迹的均值标准误就有 ~0.5, 而全仓股5 vs 对冲组合的均值差仅 0.05),
    RL 可训练样本量内"均值最大化"信号不可辨, 无法展示约束算法"均值更高"的卖点。
    放大后 (Σ 与负相关结构【完全保留】), 解析参考 (γ=0.9, Z=Σγ^t r_t):
        全仓股5:        E[Z]=6.0,  std[Z]=11.5,  P(Z≤2)≈0.36  (风险中性最优, 违约)
        对冲(4,5)各半:  E[Z]=4.0,  std[Z]=2.0,   P(Z≤2)≈0.16  (QPO 风格保守解)
        倾斜 w4≈0.40:   E[Z]≈4.4,  std[Z]≈3.6,   P(Z≤2)≈0.25  (约束最优, 均值+10%)
    ⇒ q=2.0, α=0.25 下约束恰好把策略从全仓股5"塑形"到对冲倾斜, 对比阶梯清晰。

【差异 5: API 与终止语义】 旧 gym 4 值 → gymnasium 5 值 + 纯截断
    原版: done = (t == max_steps) (旧 gym, done 混淆终止/截断)
    本版: terminated 恒 False, truncated = (t == n) —— 与 risk_sensitive_env_inf 相同的
          continuing 语义: 第 n 步只是【截断】一段 rollout (供 MC 评估 / 训练 bootstrap),
          环境本身永续。转移动力学时齐 (GBM + 窗口统计 + 权重漂移都与 t 无关) ⇒
          配合差异 1/2, 状态-奖励过程严格平稳 ⇒ critic 可用 step-blind (无需 step 特征)。

保留不变的原版要素: Σ 协方差矩阵 (两对 ρ≈-0.92/-0.95 的对冲组)、dt=1/100、
est_window=25 (24 个收益样本)、transaction_cost=0.001 (只对买入收费)、
softmax 动作→权重、episode 长度 100。
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PortfolioEnv(gym.Env):
    """
    平稳化 continuing 投资组合环境 (numpy 单 env 参考实现)。

    职责:
      1. 作为评估环境 (monte_carlo_evaluate* 走 select_action, 与 risk_sensitive_env_inf 同口径);
      2. 作为 PortfolioVecTorch(ref_env=env) 的参数来源 (保证 GPU 向量化版与本版完全同分布)。
    """

    # 默认市场参数 (差异 3/4 的注释里有完整动机)
    DEFAULT_MU = np.array([0.02, 0.06, 0.10, 0.20, 0.60])     # 年化漂移 (μ 放大版, 见差异 4)
    DEFAULT_SIGMA = np.array([                                # 年化协方差 Σ (与原版逐项一致)
        [0.01,  0.00,   0.00,  0.00,  0.00],
        [0.00,  0.04,  -0.055, 0.00,  0.00],                  # 股2-股3: ρ≈-0.92 对冲组
        [0.00, -0.055,  0.09,  0.00,  0.00],
        [0.00,  0.00,   0.00,  0.16, -0.19],                  # 股4-股5: ρ≈-0.95 对冲组
        [0.00,  0.00,   0.00, -0.19,  0.25],
    ])

    def __init__(self, n=100, mu=None, sigma=None, dt=0.01, est_window=25,
                 transaction_cost=0.001, reward_mode='pct', cov_scale=0.1):
        """
        Args:
            n:                一段 rollout 的截断长度 T (continuing, 默认 100, 同原版 max_steps)
            mu:               年化漂移向量 [K] (默认 DEFAULT_MU)
            sigma:            年化协方差矩阵 [K,K] (默认 DEFAULT_SIGMA, 与原版一致)
            dt:               单步时间 (默认 1/100, 同原版)
            est_window:       估计窗口价格数 (默认 25 ⇒ 24 个收益样本, 同原版)
            transaction_cost: 买入交易费率 (默认 0.001, 同原版)
            reward_mode:      'pct' 单步百分比收益 (默认) / 'log' 单步百分比对数收益
            cov_scale:        观测中 est_cov 各项的缩放系数 (默认 0.1, 拉到 O(1))
        """
        super().__init__()

        # -------------------- 市场参数 --------------------
        self.mu = np.array(self.DEFAULT_MU if mu is None else mu, dtype=np.float64)      # [K] 年化漂移
        self.sigma = np.array(self.DEFAULT_SIGMA if sigma is None else sigma,
                              dtype=np.float64)                                          # [K,K] 年化协方差
        self.K = self.mu.shape[0]                              # 股票数 K (默认 5)
        self.dt = float(dt)                                    # 单步时间 Δt
        self.n = int(n)                                        # 截断长度 T (continuing)
        self.est_window = int(est_window)                      # 窗口价格数 (同原版语义)
        self.M = self.est_window - 1                           # 收益样本数 M=24 (25 价 → 24 收益)
        self.tc = float(transaction_cost)                      # 买入费率 c
        self.reward_mode = str(reward_mode)                    # 'pct' / 'log'
        self.cov_scale = float(cov_scale)                      # est_cov 观测缩放

        # GBM 单步采样的预计算量 (差异 3: 精确 GBM)
        #   log g = (μ - diag(Σ)/2)·dt + √dt · L·z,  L = cholesky(Σ) (下三角), z~N(0,I)
        self._drift = (self.mu - 0.5 * np.diag(self.sigma)) * self.dt   # [K] 对数漂移项
        self._chol = np.linalg.cholesky(self.sigma) * np.sqrt(self.dt)  # [K,K] √dt·L (噪声混合矩阵)

        # 观测中 est_cov 下三角的索引 (含对角, K(K+1)/2=15 项, 同原版 tril_mask 语义)
        self._tril_idx = np.tril_indices(self.K)

        # -------------------- 空间定义 --------------------
        # 动作: K 维实数 logits, env 内部 softmax → 权重 (同原版 is_rl=True 路径)
        self.action_space = spaces.Box(low=-np.inf * np.ones((self.K,)),
                                       high=np.inf * np.ones((self.K,)), dtype=np.float64)
        # 观测: est_mu [K] + tril(est_cov) [K(K+1)/2] + w [K] = 25 维 (差异 2)
        obs_dim = self.K + self.K * (self.K + 1) // 2 + self.K
        self.observation_space = spaces.Box(low=-np.inf * np.ones((obs_dim,)),
                                            high=np.inf * np.ones((obs_dim,)), dtype=np.float64)

        # -------------------- 运行时状态 --------------------
        self.step_count = None                                 # 当前步 (仅用于截断判定)
        self.w = None                                          # 当前持仓权重 [K] (单纯形)
        self.hist = None                                       # 滚动收益窗口 [K, M] (% / 步)
        self._maxw_buf = None                                  # 每步 max 权重序列 (render 日志用)

    # ============================================================ 内部工具 ============================================================
    def _sample_gross_returns(self):
        """
        采一步 GBM 毛收益 g [K]: g_i = exp((μ_i-Σ_ii/2)dt + √dt·(L·z)_i), z~N(0,I_K)。
        E[g_i]-1 ≈ μ_i·dt (一阶与原版 Euler 一致), 恒为正 (差异 3)。
        """
        z = np.random.standard_normal(self.K)                  # [K] 独立标准正态
        return np.exp(self._drift + self._chol @ z)            # [K] 毛收益 (i.i.d. 跨步)

    def _make_obs(self):
        """
        组装观测 (差异 2):
            est_mu  = 窗口收益均值 [K]      (% / 步, 量级 ~0.5±1)
            est_cov = 窗口收益协方差 [K,K]  ((%/步)², ddof=1, 同原版 np.cov)
            obs     = [est_mu, tril(est_cov)·cov_scale, w]  共 25 维, 全平稳
        """
        est_mu = self.hist.mean(axis=1)                        # [K] 窗口均值
        est_cov = np.cov(self.hist, rowvar=True, ddof=1)       # [K,K] 窗口协方差 (同原版)
        tril = est_cov[self._tril_idx] * self.cov_scale        # [15] 下三角(含对角), 缩放到 O(1)
        return np.concatenate([est_mu, tril, self.w]).astype(np.float64)

    # ============================================================ gymnasium API ============================================================
    def reset(self, seed=None, options=None):
        """
        重置: 预热 M 步 GBM 填满收益窗口 + 随机初始权重 (softmax(N(0,1)), 同原版随机初仓语义)。
        返回 (obs, info) —— gymnasium 5 值 API (差异 5)。
        """
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)                               # 与项目其它 env 一致: 全局 numpy RNG

        self.step_count = 0
        self._maxw_buf = []

        # 预热窗口: M 步纯模拟器收益 (对应原版 reset 里先走 est_window-1 步价格)
        self.hist = np.stack(
            [100.0 * (self._sample_gross_returns() - 1.0) for _ in range(self.M)],
            axis=1)                                            # [K, M] 百分比收益样本

        # 随机初始权重: softmax 标准正态 logits (原版用 action_space.sample() 后 softmax;
        # inf 边界的 Box.sample 行为依赖 gym 版本, 这里用显式 N(0,1) logits, 语义相同且可控)
        logits = np.random.standard_normal(self.K)
        e = np.exp(logits - logits.max())                      # 数值稳定 softmax
        self.w = e / e.sum()                                   # [K] 初始权重 ∈ 单纯形

        return self._make_obs(), {}

    def step(self, action):
        """
        一步交互 (全部公式只依赖权重与价格比 ⇒ 与财富水平无关, 平稳; 差异 1):

          1. 目标权重     u = softmax(action)                       (同原版 is_rl 路径)
          2. 交易成本因子 κ = 1 - c·Σ_i max(u_i - w_i, 0)           (只对买入部分收费,
             与原版 "delta_position>0 时乘 (1-c)" 同一阶等价; 权重空间记账保持环境无标度)
          3. 毛收益       g ~ GBM 一步;  组合毛收益 G_p = Σ_i u_i·g_i
          4. 奖励         pct: 100·(κ·G_p - 1)   /   log: 100·log(κ·G_p)
          5. 权重漂移     w' = u·g / Σ_j u_j·g_j                    (相对价格变动自然漂移)
          6. 更新滚动窗口 → est_mu/est_cov → obs
          7. truncated = (step_count == n), terminated ≡ False     (continuing, 差异 5)
        """
        action = np.asarray(action, dtype=np.float64).flatten()
        assert action.shape == (self.K,), f"action shape {action.shape} != ({self.K},)"

        # ---- 1. softmax → 目标权重 u (数值稳定: 减 max) ----
        e = np.exp(action - action.max())
        u = e / e.sum()                                        # [K] 目标权重

        # ---- 2. 交易成本: 只对买入量收费 (同原版语义, 权重空间记账) ----
        turnover_buy = np.maximum(u - self.w, 0.0).sum()       # 买入权重总量 ∈ [0,1]
        kappa = 1.0 - self.tc * turnover_buy                   # 成本因子 κ ∈ (1-c, 1]

        # ---- 3. GBM 一步 + 组合毛收益 ----
        g = self._sample_gross_returns()                       # [K] 单步毛收益
        Gp = float(u @ g)                                      # 组合毛收益 Σ u_i g_i

        # ---- 4. 奖励 (差异 1: 百分比收益, 平稳) ----
        if self.reward_mode == 'log':
            reward = 100.0 * np.log(kappa * Gp)                # 百分比对数收益
        else:
            reward = 100.0 * (kappa * Gp - 1.0)                # 百分比简单收益 (默认)

        # ---- 5. 权重随相对价格变动漂移 (成本按比例摊薄, 不改变权重构成) ----
        self.w = (u * g) / Gp                                  # [K] 新持仓权重

        # ---- 6. 滚动窗口更新 + 观测 ----
        x = 100.0 * (g - 1.0)                                  # [K] 本步各股百分比收益
        self.hist = np.roll(self.hist, -1, axis=1)             # 窗口左移一格
        self.hist[:, -1] = x                                   # 末位写入本步收益
        obs = self._make_obs()

        # ---- 7. continuing 截断语义 (差异 5) ----
        self.step_count += 1
        self._maxw_buf.append(float(self.w.max()))             # 记录持仓集中度 (render 日志)
        truncated = (self.step_count == self.n)                # 第 n 步: 截断 (非终止)

        return obs, float(reward), False, truncated, {}

    # ============================================================ 日志辅助 ============================================================
    def render(self, mode=None):
        """
        返回本段 rollout 每步的 max 权重序列 [t] (numpy) —— 持仓集中度 (∈[1/K,1]) 作为
        "风险水平"代理 (对应 risk_sensitive_env_inf 中 render 返回 risk 序列的接口语义)。
        """
        if not self._maxw_buf:
            return None
        return np.asarray(self._maxw_buf, dtype=np.float64)

    def close(self):
        return None
