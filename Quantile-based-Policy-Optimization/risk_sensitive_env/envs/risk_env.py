import gymnasium as gym
from gymnasium import spaces
import numpy as np


class RiskSensitiveEnv(gym.Env):
    """
    RiskSensitiveEnv: 风险敏感环境，用于验证QCPO相比QPO的优越性。
    
    核心设计思想:
    ==============
    动作 a ∈ R 直接作为风险等级 r
    奖励服从正态分布，其均值和标准差都随风险等级 r 变化:
    
    - 单步均值: μ(r) = μ_base + k_mean * r
    - 单步标准差: σ(r) = σ_base + k_std * r
    - 单步奖励: reward_t ~ N(μ(r), σ(r)²)
    
    默认参数设置:
    - μ_base = 1.0   (基础均值)
    - k_mean = 0.8   (均值增长系数)
    - σ_base = 0.02  (基础标准差)  
    - k_std = 8.0    (标准差增长系数)
    
    数学分析 (累积回报):
    ====================
    设每步奖励独立同分布(r固定)，折扣因子γ=0.99，共n=10步
    
    reward_t ~N(1 + 0.8r_t, softplus(0.02 + 8r_t)²)
    
    25%分位数 (z_{0.25} ≈ -0.674):
    quantile_0.25_t = 1 + 0.8r_t - 0.674 * softplus(0.02 + 8r_t)
    mean_t = 1 + 0.8r_t

    

    预期行为:
    =========
    【QPO (最大化分位数)】
    - 由于 Q(r) 随 r 单调递减，QPO 会选择 r → 0
    
    【QCPO (最大化均值 s.t. 分位数约束 Q_α ≥ C)】
    - 目标: 在满足 Q_α ≥ C 约束下最大化均值
    - QCPO 会增大 r 直到分位数约束被激活
    - 例如约束 C = 6.0:
        - 解 9.43 - 13.8r = 6.0 → r ≈ 0.25
        - 此时 E[U] ≈ 11.5, Q_0.25 ≈ 6.0
    
    对比结论 (约束C=6.0时):
    =======================
    - QPO:  Mean ≈ 9.56,  Q_0.25 ≈ 9.43
    - QCPO: Mean ≈ 11.5,  Q_0.25 ≈ 6.0 (满足约束)
    
    **QCPO 的均值提高约30%，同时分位数仍满足约束。**
    这清晰地展示了 QCPO 在风险约束下最大化均值的优越性。
    
    环境配置:
    =========
    - horizon (n): 每个episode的步数，默认10
    - 状态空间: n维one-hot向量表示当前步数
    - 动作空间: 1维连续动作（直接裁剪到风险等级）
    - 折扣因子 γ = 0.99 时:
        - Σγ^t ≈ 9.56 (均值系数)
        - √(Σγ^(2t)) ≈ 3.02 (标准差系数)
    """
    
    def __init__(self, n=10, mu_base=1.0, k_mean=0.8, sigma_base=0.02, k_std=8.0):
        """
        初始化环境
        
        Args:
            n: episode长度（步数）
            mu_base: 基础均值（r=0时的期望奖励）
            k_mean: 均值增长系数（风险溢价）
            sigma_base: 基础标准差（r=0时的波动性）
            k_std: 标准差增长系数（风险放大因子）
        """
        super().__init__()
        
        self.n = n
        
        # 奖励分布参数
        self.mu_base = mu_base
        self.k_mean = k_mean
        self.sigma_base = sigma_base
        self.k_std = k_std
        
        # 动作空间: 实数 (直接视为风险等级)
        self.action_space = spaces.Box(
            low=-np.inf * np.ones((1,)), 
            high=np.inf * np.ones((1,)), 
            dtype=np.float32
        )
        
        # 状态空间: n维one-hot向量
        self.observation_space = spaces.Box(
            low=np.zeros((n,)), 
            high=np.ones((n,)), 
            dtype=np.float32
        )
        
        # 用于随机化状态转移顺序
        self.order = np.arange(n)
        
        # 运行时变量
        self.step_count = None
        self.risk_buf = None  # 记录每步选择的风险等级
        
    def _action_to_risk(self, action):
        """
        直接使用动作作为风险等级
        """
        return float(action[0])
    
    def _compute_reward(self, risk_level):
        """
        根据风险等级计算奖励
        reward ~ N(μ(r), σ(r))
        """
        mu = self.mu_base + self.k_mean * risk_level
        sigma_linear = self.sigma_base + self.k_std * risk_level
        sigma = np.log1p(np.exp(-np.abs(sigma_linear))) + np.maximum(sigma_linear, 0.0)  # 数值稳定写法，当x很大时exp(x)会溢出，当很小时exp(x)接近0会产生精度损失
                                                                                          
        return np.random.normal(mu, sigma)
    
    def step(self, action):
        """
        执行一步动作
        
        Returns:
            state: 新状态 (n维one-hot向量)
            reward: 奖励值
            done: 是否结束
            info: 额外信息
        """
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        
        # 映射动作到风险等级
        risk_level = self._action_to_risk(action)
        self.risk_buf.append(risk_level)
        
        # 计算奖励
        reward = self._compute_reward(risk_level)
        
        # 更新步数
        self.step_count += 1
        state = np.zeros((self.n,), dtype=np.float32)
        
        done = False
        if self.step_count == self.n:
            done = True
        else:
            state[self.order[self.step_count]] = 1.0
            
        return state, reward, done, {}
    
    def reset(self):
        """
        重置环境
        
        Returns:
            state: 初始状态
        """
        self.step_count = 0
        self.risk_buf = []
        np.random.shuffle(self.order)
        
        state = np.zeros((self.n,), dtype=np.float32)
        state[self.order[self.step_count]] = 1.0
        
        return state
    
    def render(self, mode=None):
        """
        渲染环境信息 (返回风险等级序列用于日志记录)
        """
        return np.array(self.risk_buf) if self.risk_buf else None
    
    def close(self):
        return None
    
