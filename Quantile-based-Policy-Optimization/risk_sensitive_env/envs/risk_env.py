import gymnasium as gym
from gymnasium import spaces
import numpy as np


# ==================================================== 默认体制参数 ====================================================
# 三个体制的默认 risk-return 参数
# 每个体制: k_mean 控制风险溢价 (μ 随 r 增长的斜率), k_std 控制波动放大 (σ 随 r 增长的斜率)
DEFAULT_REGIME_PARAMS = {
    0: {'name': 'calm',     'k_mean': 0.3,  'k_std': 4.0},   # 正常: 适度风险溢价, 低波动
    1: {'name': 'volatile', 'k_mean': 0.5,  'k_std': 8.0},   # 波动: 高风险溢价, 中等波动
    2: {'name': 'crisis',   'k_mean': -0.2, 'k_std': 15.0},  # 危机: 风险溢价反转(!), 极端波动
}

# 默认转移矩阵 P[i,j] = P(regime_{t+1}=j | regime_t=i)
# 平稳分布约: calm ~64%, volatile ~24%, crisis ~12%
DEFAULT_TRANSITION_MATRIX = np.array([
    [0.88, 0.09, 0.03],   # calm     → calm 88%, volatile 9%, crisis 3%
    [0.25, 0.60, 0.15],   # volatile → calm 25%, volatile 60%, crisis 15%
    [0.15, 0.30, 0.55],   # crisis   → calm 15%, volatile 30%, crisis 55%
], dtype=np.float64)


# ==================================================== WIDEGAP 体制参数 ====================================================
# 设计目标: 让 Q_0.25 随风险等级 r 递减 (k_mean/k_std < 0.055)
# 这样 QPO 学保守策略 (r≈0)，QCPO 才能通过冒险获得更高均值
#
# 推导: dQ_0.25/dr = G·k_mean − 4.77·k_std
#        Q递减条件: k_mean/k_std < 4.77/G ≈ 4.77/86.7 ≈ 0.055
#
# 对比默认参数:
#   calm   0.3/4.0 =0.075 >0.055 → Q随r递增(QPO冒险!) ← 问题所在
#   volatile 0.5/8.0=0.063 >0.055 → Q微增
#
# WIDEGAP参数:
#   calm   0.3/8.0 =0.038 <0.055 → Q随r递减 ✓
#   volatile 0.5/12.0=0.042 <0.055 → Q随r递减 ✓
WIDEGAP_REGIME_PARAMS = {
    0: {'name': 'calm',     'k_mean': 0.3,  'k_std': 8.0},   # 波动加倍, 风险溢价不变
    1: {'name': 'volatile', 'k_mean': 0.5,  'k_std': 12.0},  # 波动1.5倍
    2: {'name': 'crisis',   'k_mean': -0.3, 'k_std': 20.0},  # 更强的风险惩罚
}

# WIDEGAP 转移矩阵: crisis 更粘 (0.55→0.65), volatile→crisis 增加
WIDEGAP_TRANSITION_MATRIX = np.array([
    [0.88, 0.09, 0.03],   # calm     不变
    [0.20, 0.60, 0.20],   # volatile → crisis: 0.15→0.20
    [0.10, 0.25, 0.65],   # crisis   自环: 0.55→0.65
], dtype=np.float64)


def _compute_stationary(P):
    """
    [函数简介]: 计算 Markov 链的平稳分布 π，满足 πP = π

    [算法逻辑]:
    求解 (P^T - I)π = 0 加上约束 Σπ_i = 1
    等价于解增广线性系统: 把最后一个方程替换为 Σπ_i = 1

    [输入]: P — 转移矩阵 [K, K]
    [输出]: π — 平稳分布 [K]
    """
    K = P.shape[0]
    A = P.T - np.eye(K)       # (P^T - I), 零空间含平稳分布
    A[-1, :] = 1.0            # 替换最后一行为归一化约束
    b = np.zeros(K)
    b[-1] = 1.0               # Σπ_i = 1
    return np.linalg.solve(A, b)


class RiskSensitiveEnv(gym.Env):
    """
    RiskSensitiveEnv: Markov Regime-Switching 风险敏感环境

    核心设计思想:
    ==============
    环境包含 K 个体制 (regime)，通过 Markov 链随机切换。
    每个体制有不同的 risk-return 参数，使得:
    - 最优策略依赖于当前体制 (状态依赖)
    - Distributional Critic ψ(s,a) 需要学习不同体制的回报分布

    动作 a ∈ R 直接作为风险等级 r。
    奖励服从正态分布，参数由 **当前体制** 决定:

    - 单步均值: μ(r, regime) = μ_base + k_mean[regime] * r
    - 单步标准差: σ(r, regime) = softplus(σ_base + k_std[regime] * r)
    - 单步奖励: reward_t ~ N(μ, σ²)

    默认三体制 (K=3):
    - calm (体制0):     k_mean=0.3,  k_std=4.0   → 正常的风险溢价
    - volatile (体制1): k_mean=0.5,  k_std=8.0   → 高风险溢价, 中等波动
    - crisis (体制2):   k_mean=-0.2, k_std=15.0  → **风险溢价反转**, 极端波动

    关键设计: crisis 体制中 k_mean < 0
    =========================================
    冒险(r>0)反而降低期望收益, 同时方差爆炸。
    最优策略必须学会在 crisis 中保守 (r→0)。

    DQC-AC 优势体现:
    ================
    crisis 体制稀有 (~12%) 但对尾部风险至关重要:
    - DQC-AC: replay buffer 反复学习 crisis transitions → 快速学会在 crisis 中保守
    - QCPO: 等完整轨迹, crisis 信息被 200 步平均稀释 → 学习慢

    Infinite Horizon 设计:
    =====================
    环境永不终止 (terminated=False), 仅在 max_steps 处截断 (truncated=True)。
    截断时 TD target 需要 bootstrap: y = r + γ·V(s')。

    状态空间:
    =========
    one-hot 编码当前体制: [1,0,0]=calm, [0,1,0]=volatile, [0,0,1]=crisis
    Actor W @ s 等价于学 K 个独立的风险等级 (每个体制一个权重)。

    预期行为:
    =========
    【QPO】 所有体制 r→0 (最大化分位数 = 最小化方差)
    【QCPO】 calm/volatile 中 r>0 (利用风险溢价), crisis 中 r→0
    【DQC-AC】 同 QCPO, 但收敛更快 (尤其在 crisis 学习上)
    """

    def __init__(self, max_steps=200, mu_base=1.0, sigma_base=0.02,
                 num_regimes=3, transition_matrix=None, regime_params=None,
                 preset='default'):
        """
        初始化环境

        Args:
            max_steps: 每个 episode 的最大步数 (截断点, 近似 infinite horizon)
            mu_base: 基础均值 (r=0 时的期望奖励, 所有体制共享)
            sigma_base: 基础标准差 (r=0 时的波动性, 所有体制共享)
            num_regimes: 体制数量 K (默认 3: calm/volatile/crisis)
            transition_matrix: [K, K] 转移矩阵, None 则用默认值
            regime_params: dict {i: {'name': str, 'k_mean': float, 'k_std': float}},
                           None 则用默认值
            preset: 预设参数集, 'default' 或 'widegap'
                    'widegap' 使 Q_0.25 随 r 递减, 拉大 QPO/QCPO 差距
                    注意: 显式传入 regime_params/transition_matrix 会覆盖 preset
        """
        super().__init__()

        self.max_steps = max_steps
        self.num_regimes = num_regimes

        # 奖励分布的共享基础参数
        self.mu_base = mu_base
        self.sigma_base = sigma_base

        # ==================================================== 选择 preset ====================================================
        if preset == 'widegap':
            _default_params = WIDEGAP_REGIME_PARAMS
            _default_transition = WIDEGAP_TRANSITION_MATRIX
        else:
            _default_params = DEFAULT_REGIME_PARAMS
            _default_transition = DEFAULT_TRANSITION_MATRIX

        # ==================================================== 体制参数 ====================================================
        if regime_params is not None:
            self.regime_params = regime_params
        elif num_regimes == 3:
            self.regime_params = {k: dict(v) for k, v in _default_params.items()}
        elif num_regimes == 1:
            # 单体制: 等价于旧版本的 k_mean=0.3, k_std=8.0
            self.regime_params = {0: {'name': 'single', 'k_mean': 0.3, 'k_std': 8.0}}
        else:
            raise ValueError(
                f"num_regimes={num_regimes} 但未提供 regime_params。"
                f"仅 num_regimes=1 或 3 有内置默认值。"
            )

        # ==================================================== 转移矩阵 ====================================================
        if transition_matrix is not None:
            self.transition_matrix = np.asarray(transition_matrix, dtype=np.float64)
        elif num_regimes == 3:
            self.transition_matrix = _default_transition.copy()
        elif num_regimes == 1:
            self.transition_matrix = np.array([[1.0]], dtype=np.float64)  # 单体制: 不切换
        else:
            raise ValueError(
                f"num_regimes={num_regimes} 但未提供 transition_matrix。"
            )

        # 验证转移矩阵维度和行和
        assert self.transition_matrix.shape == (num_regimes, num_regimes), \
            f"transition_matrix shape {self.transition_matrix.shape} != ({num_regimes}, {num_regimes})"
        row_sums = self.transition_matrix.sum(axis=1)
        assert np.allclose(row_sums, 1.0), f"transition_matrix 行和不为 1: {row_sums}"

        # 计算平稳分布 (用于 reset 时采样初始体制)
        self.stationary_dist = _compute_stationary(self.transition_matrix)

        # ==================================================== 动作空间 ====================================================
        # 动作: 实数 (直接视为风险等级 r)
        self.action_space = spaces.Box(
            low=np.float32(-np.inf) * np.ones(1, dtype=np.float32),
            high=np.float32(np.inf) * np.ones(1, dtype=np.float32),
            dtype=np.float32
        )

        # ==================================================== 状态空间 ====================================================
        # one-hot 编码当前体制: [1,0,0]=calm, [0,1,0]=volatile, [0,0,1]=crisis
        # Actor W @ one_hot 等价于每个体制学一个独立的风险等级
        self.observation_space = spaces.Box(
            low=np.zeros(num_regimes, dtype=np.float32),
            high=np.ones(num_regimes, dtype=np.float32),
            dtype=np.float32
        )

        # ==================================================== 运行时变量 ====================================================
        self.step_count = None
        self.current_regime = None
        self.risk_buf = None        # 记录每步选择的风险等级
        self.regime_buf = None      # 记录每步的体制 (用于诊断日志)

    # ==================================================== 体制切换 ====================================================
    def _transition_regime(self):
        """
        [函数简介]: Markov 链体制切换

        [算法逻辑]:
        从当前体制 i 出发，按转移矩阵第 i 行的概率分布采样下一个体制。
        P(regime_{t+1}=j | regime_t=i) = transition_matrix[i, j]

        [副作用]: 更新 self.current_regime
        """
        probs = self.transition_matrix[self.current_regime]
        self.current_regime = int(self.np_random.choice(self.num_regimes, p=probs))

    def _action_to_risk(self, action):
        """直接使用动作作为风险等级"""
        return float(action[0])

    def _compute_reward(self, risk_level):
        """
        [函数简介]: 根据当前体制和风险等级计算奖励

        [公式]:
        μ = μ_base + k_mean[regime] * r
        σ = softplus(σ_base + k_std[regime] * r)
        reward ~ N(μ, σ²)

        [关键]: k_mean 和 k_std 取决于 self.current_regime
        - calm:     k_mean=0.3  → r 增大, μ 增大 (正常风险溢价)
        - crisis:   k_mean=-0.2 → r 增大, μ 反而减小 (风险惩罚)

        [输入]: risk_level (float), 即动作值 r
        [输出]: float, 采样的奖励值
        """
        params = self.regime_params[self.current_regime]
        k_mean = params['k_mean']
        k_std = params['k_std']

        mu = self.mu_base + k_mean * risk_level
        sigma_linear = self.sigma_base + k_std * risk_level
        # softplus: log(1 + exp(x)), 数值稳定写法
        sigma = np.log1p(np.exp(-np.abs(sigma_linear))) + np.maximum(sigma_linear, 0.0)

        return float(self.np_random.normal(mu, sigma))

    def _get_obs(self):
        """
        [函数简介]: 返回 one-hot 编码的当前体制观测

        [输出]: np.ndarray, shape=(num_regimes,)
        - [1,0,0] = calm, [0,1,0] = volatile, [0,0,1] = crisis
        """
        obs = np.zeros(self.num_regimes, dtype=np.float32)
        obs[self.current_regime] = 1.0
        return obs

    # ==================================================== 核心 API ====================================================
    def step(self, action):
        """
        [函数简介]: 执行一步动作

        [流程]:
        1. 在 当前体制 下, 用 risk_level 计算奖励
        2. 切换到下一个体制 (Markov 转移)
        3. 返回 新体制的观测 (下一步 agent 看到的是新体制)

        [注意]: 奖励由「旧体制」决定, 观测返回「新体制」
        这与标准 MDP 语义一致: r_t = R(s_t, a_t), s_{t+1} ~ P(·|s_t, a_t)

        Returns (gymnasium 5-value API):
            obs: 新体制的 one-hot 观测
            reward: 在旧体制下采样的奖励
            terminated: 始终 False (infinite horizon)
            truncated: 达到 max_steps 时为 True
            info: 包含当前步体制信息
        """
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        risk_level = self._action_to_risk(action)
        self.risk_buf.append(risk_level)

        # 在当前体制下计算奖励
        reward_regime = self.current_regime                    # 记录产生奖励的体制
        reward = self._compute_reward(risk_level)

        # 体制切换 (Markov 转移)
        self._transition_regime()
        self.regime_buf.append(self.current_regime)

        self.step_count += 1

        # 返回新体制的观测
        obs = self._get_obs()

        # infinite horizon: 永不终止, 只在 max_steps 处截断
        terminated = False
        truncated = (self.step_count >= self.max_steps)

        info = {'reward_regime': reward_regime}

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        """
        [函数简介]: 重置环境 (gymnasium API)

        [流程]:
        1. 从平稳分布采样初始体制
        2. 返回初始体制的 one-hot 观测

        Returns:
            obs: 初始体制的 one-hot 观测
            info: 包含初始体制信息
        """
        super().reset(seed=seed, options=options)

        self.step_count = 0
        self.risk_buf = []
        self.regime_buf = []

        # 从平稳分布采样初始体制
        self.current_regime = int(
            self.np_random.choice(self.num_regimes, p=self.stationary_dist)
        )
        self.regime_buf.append(self.current_regime)

        return self._get_obs(), {'initial_regime': self.current_regime}

    def render(self, mode=None):
        """返回风险等级序列 (兼容旧接口, 用于 agent 日志)"""
        return np.array(self.risk_buf) if self.risk_buf else None

    def get_regime_history(self):
        """返回体制序列 (用于诊断日志)"""
        return np.array(self.regime_buf) if self.regime_buf else None

    def close(self):
        return None
