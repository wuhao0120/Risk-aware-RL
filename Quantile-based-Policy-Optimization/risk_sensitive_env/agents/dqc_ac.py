# ==================================================== DQC-AC Agent (v5) ====================================================
# DQC-AC (Distributional Quantile-Constrained Actor-Critic) for Risk-Sensitive Environment
#
# 严格实现论文 Algorithm 3 (Distributional Quantile-Constrained Actor-Critic)
# 核心思想: 用 TD targets 完全替代 Monte Carlo episode returns，实现 per-transition 更新
#
# ===== v5: 严格 Algorithm 3 + 正确的 truncation 处理 (infinite horizon) =====
# 环境每 episode 走 max_steps 步 (截断近似 infinite horizon)
# 关键改进: 区分 terminated 与 truncated，TD target 只在真正终止时置零
# 算法完全基于 TD targets，不依赖 episode returns:
#   - Critic ψ(s,a;ω) 输出 M 个分位数，学习 Z(s,a) 的分布
#   - Q update 用 TD targets: Q ← Q + (1/M)Σ_j β(α - I{y_j ≤ Q})
#   - Actor 用 TD targets:   D̃ = (1/M)Σ_j [(y_j - λ·I{y_j≤Q}/f̂_Z) · ∇θ log π]
#
# 关键公式 (论文 Section 3):
# - TD targets:     y_j = r + γ·ψ_j(s', a'; ω_target)      (Algorithm 3 Step 12)
# - Critic Loss:    Σ_i Σ_j ρ_{τ̂_i}^κ(y_j - ψ_i(s,a;ω))   (Algorithm 2 / Step 13)
# - Q Update:       Q ← Q + (1/M)Σ_j β(α - I{y_j ≤ Q})     (公式 25 / Step 15)
# - Actor Weight:   w = (1/M)Σ_j [y_j - λ·I{y_j≤Q}/f̂_Z]    (公式 24 / Step 18)
# - Density:        f̂_Z = 2δ / (Q(α+δ) - Q(α-δ))            (公式 13 / Step 17)
# - Dual Update:    λ ← [λ - ε(Q - q)]₊                     (标准 dual descent)
# ==================================================================================================================

import numpy as np                                     # 数值计算库
import torch                                           # PyTorch深度学习框架
import torch.nn as nn                                  # 神经网络模块
import torch.nn.functional as F                        # 函数式API
from torch.optim import Adam                           # Adam优化器
import wandb                                           # Weights & Biases日志系统
from torch.optim.lr_scheduler import LambdaLR         # 学习率调度器
from collections import deque                          # 双端队列
import random                                          # 随机数

from utils import Actor                                # 导入Actor网络


# ==================================================== 学习率调度函数 ====================================================
def lr_lambda(k, a, b, c):
    """
    [函数简介]: 学习率因子函数，用于LambdaLR调度器
    [输入]: k(int, 当前轮数), a, b, c(float, 超参数)
    [算法逻辑]: lr(k) = a / (b + k)^c
    [返回值]: 标量float, 学习率衰减因子
    """
    return a / ((b + k) ** c)


# ==================================================== 示性函数 ====================================================
def indicator(threshold, values: torch.Tensor):
    """
    [函数简介]: 示性函数 I(values <= threshold)
    [输入]: threshold(标量或张量), values(张量)
    [输出]: 与values形状相同的张量, values<=threshold的位置是1, 否则是0
    """
    # 用布尔比较后直接转float，减少额外中间张量分配
    return (values <= threshold).to(values.dtype)


# ==================================================== Replay Buffer ====================================================
class ReplayBuffer:
    """
    [类简介]: 经验回放缓冲区，存储 transition (s, a, r, s', terminated)

    关键: 只存 terminated 标志（真正的终止），不存 truncated。
    Infinite horizon 环境中 terminated 始终为 False，截断由外部 max_steps 控制。
    TD target 只用 terminated 做 bootstrap mask:
      y = r + γ·V(s') · (1 - terminated)
    截断时仍然 bootstrap（因为 MDP 并未真正结束）。
    """

    def __init__(self, capacity, device):
        """
        [初始化]

        [参数]:
        - capacity: 缓冲区最大容量
        - device: 计算设备
        """
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, terminated):
        """
        [函数简介]: 添加一个 transition 到缓冲区

        [参数]:
        - state: 当前状态 (np.ndarray)
        - action: 采取的动作 (np.ndarray)
        - reward: 获得的奖励 (float)
        - next_state: 下一个状态 (np.ndarray)
        - terminated: 是否真正终止 (bool/float), 截断不算终止
        """
        self.buffer.append((state, action, reward, next_state, terminated))

    def sample(self, batch_size):
        """
        [函数简介]: 从缓冲区随机采样 batch_size 个 transitions

        [输出]: dict，包含以下 tensor:
        - states: [batch_size, state_dim]
        - actions: [batch_size, action_dim]
        - rewards: [batch_size]
        - next_states: [batch_size, state_dim]
        - terminateds: [batch_size], 只有真正终止才为1，截断为0
        """
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, terminateds = zip(*batch)

        return {
            'states': torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device),
            'actions': torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=self.device),
            'rewards': torch.as_tensor(np.asarray(rewards), dtype=torch.float32, device=self.device),
            'next_states': torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=self.device),
            'terminateds': torch.as_tensor(np.asarray(terminateds), dtype=torch.float32, device=self.device),
        }

    def __len__(self):
        return len(self.buffer)


# ==================================================== Distributional Critic ====================================================
class DistributionalCritic(nn.Module):
    """
    [类简介]: 分布式价值网络，输出 M 个分位数估计

    ψ(s, a; ω) = [ψ_1(s,a), ψ_2(s,a), ..., ψ_M(s,a)]
    其中 ψ_j 对应分位数水平 τ_j = (j - 0.5) / M

    用于 QR-DQN 风格的分布式 TD Learning
    """

    def __init__(self, state_dim, action_dim, num_quantiles, hidden_dims=[64, 64]):
        """
        [初始化]

        [参数]:
        - state_dim: 状态维度
        - action_dim: 动作维度
        - num_quantiles: 分位数数量 M
        - hidden_dims: 隐藏层维度列表
        """
        super(DistributionalCritic, self).__init__()

        self.num_quantiles = num_quantiles

        # 输入: 状态 + 动作 的拼接
        input_dim = state_dim + action_dim

        # 构建网络
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim)) # nn.Linear默认bias=True
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # 输出层: M 个分位数
        layers.append(nn.Linear(prev_dim, num_quantiles))

        self.network = nn.Sequential(*layers)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """正交初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, state, action):
        """
        [前向传播]

        [输入]:
        - state: [batch_size, state_dim]
        - action: [batch_size, action_dim]

        [输出]:
        - quantiles: [batch_size, num_quantiles], M 个分位数估计
        """
        # 拼接状态和动作
        x = torch.cat([state, action], dim=-1) # dim=-1表示在最后一个维度上拼接
        return self.network(x)


# ==================================================== DQC-AC Agent ====================================================
class DQCAC(object):
    """
    [类简介]: DQC-AC (Distributional Quantile-Constrained Actor-Critic)

    [算法目标]:
    最大化累积回报的期望 E[R]
    约束: Q_α(R) >= C (分位数约束)

    [核心组件]:
    - Actor: 策略网络 π(a|s; θ)
    - Distributional Critic: 输出 M 个分位数 ψ(s,a; ω)
    - Global Quantile Estimate: 三个分位数 Q(α-δ), Q(α), Q(α+δ)
    - Lagrange Multiplier: λ 用于约束优化

    [更新流程] (严格 Algorithm 3):
    1. 收集 episode，transitions 直接存入 Replay Buffer (不需要 episode return)
    2. 从 Buffer 采样 batch，计算 TD targets y_j = r + γψ_j(s',a'; ω_target):
       - Critic Update: Distributional TD (Quantile Huber Loss)
       - Q Update: Q ← Q + (1/M)Σ_j β(α - I{y_j ≤ Q})     ← 用 TD targets
       - Actor Update: w = (1/M)Σ_j [y_j - λ·I{y_j≤Q}/f̂_Z] ← 用 TD targets
    3. 每 I 步更新 λ
    """

    def __init__(self, args, env):
        """
        [初始化方法]
        """
        # ==================================================== 基础参数 ====================================================
        self.device = args.device                      # 计算设备 (CPU/GPU)
        self.log_interval = args.log_interval          # 日志记录间隔 (episodes)
        self.est_interval = args.est_interval          # 滚动估计窗口大小
        self.q_alpha = args.q_alpha                    # 目标分位数水平 α (默认0.25)
        self.gamma = args.gamma                        # 折扣因子 γ
        self.max_episode = args.max_episode            # 最大训练轮数

        # ==================================================== DQC-AC特有参数 ====================================================
        self.quantile_threshold = args.quantile_threshold    # 分位数约束阈值 C
        self.outer_interval = args.outer_interval            # λ更新间隔

        # 密度估计带宽 δ（增大到 0.1，使密度估计更稳定，λ/density 不易爆炸）
        self.delta = getattr(args, 'density_bandwidth', 0.1)
        self.alpha_lower = max(0.001, self.q_alpha - self.delta)
        self.alpha_upper = min(0.999, self.q_alpha + self.delta)

        # 约束项安全上界: clamp(λ·I/f̂_Z, max=max_constraint)，防止约束项主导 Actor 梯度
        self.max_constraint = getattr(args, 'max_constraint', 50.0)

        # ==================================================== TD Learning 参数 ====================================================
        # 分位数数量 M (Distributional Critic 输出维度)
        self.num_quantiles = getattr(args, 'num_quantiles', 32)

        # 分位数水平 τ_j = (j - 0.5) / M，j = 1, ..., M
        self.quantile_taus = torch.FloatTensor(
            [(j + 0.5) / self.num_quantiles for j in range(self.num_quantiles)]
        ).to(self.device)
        # quantile_taus: [M] tensor, 每个元素是对应的分位数水平

        # Replay Buffer 参数
        self.buffer_capacity = getattr(args, 'buffer_capacity', 100000)
        self.batch_size = getattr(args, 'batch_size', 64)
        self.min_buffer_size = getattr(args, 'min_buffer_size', 1000)  # 开始学习前的最小样本数

        # Quantile Huber Loss 的 κ 参数
        self.huber_kappa = getattr(args, 'huber_kappa', 1.0)

        # 每episode更新次数
        self.updates_per_episode = getattr(args, 'updates_per_episode', 10)

        # ==================================================== 环境引用 ====================================================
        self.env = env
        self.env_name = args.env_name

        # 维度
        state_dim = int(np.prod(self.env.observation_space.shape))
        action_dim = int(np.prod(self.env.action_space.shape))
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._log_2pi = float(np.log(2.0 * np.pi))     # 高斯log_prob常量项，避免重复计算

        # ==================================================== Actor网络 ====================================================
        actor_hidden = getattr(args, 'actor_hidden_dims', None)
        self.actor = Actor(state_dim, action_dim, args.init_std, hidden_dims=actor_hidden).to(self.device)

        # Actor优化器
        self.actor_optimizer = Adam(self.actor.parameters(), lr=1.0, eps=1e-5)
        self.actor_scheduler = LambdaLR(
            self.actor_optimizer,
            lr_lambda=lambda k: lr_lambda(k, args.theta_a, args.theta_b, args.theta_c)
        )

        # ==================================================== Distributional Critic ====================================================
        critic_hidden = getattr(args, 'emb_dim', [64, 64])
        self.online_critic = DistributionalCritic(
            state_dim, action_dim, self.num_quantiles, critic_hidden
        ).to(self.device)

        # Target Critic (用于稳定 TD targets)
        self.target_critic = DistributionalCritic(
            state_dim, action_dim, self.num_quantiles, critic_hidden
        ).to(self.device)
        self.target_critic.load_state_dict(self.online_critic.state_dict())

        # Critic 优化器
        self.critic_lr = getattr(args, 'critic_lr', 1e-3)
        self.critic_optimizer = Adam(self.online_critic.parameters(), lr=self.critic_lr, eps=1e-5)

        # Target 网络软更新参数
        self.tau = getattr(args, 'tau', 0.005)

        # ==================================================== Replay Buffer ====================================================
        self.replay_buffer = ReplayBuffer(self.buffer_capacity, self.device)

        # ==================================================== 全局分位数估计 ====================================================
        # 初始化为0，通过warm_up估计
        self.q_est_lower = torch.tensor(
            [0.0], dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.q_est = torch.tensor(
            [0.0], dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.q_est_upper = torch.tensor(
            [0.0], dtype=torch.float32, device=self.device, requires_grad=True
        )

        # 分位数优化器
        self.q_optimizer = Adam(
            [self.q_est_lower, self.q_est, self.q_est_upper], lr=1.0, eps=1e-5
        )
        self.q_scheduler = LambdaLR(
            self.q_optimizer,
            lr_lambda=lambda k: lr_lambda(k, args.q_a, args.q_b, args.q_c)
        )

        # ==================================================== 拉格朗日乘子λ ====================================================
        self.lambda_dual = torch.tensor(
            [0.0], dtype=torch.float32, device=self.device, requires_grad=True
        )

        lambda_a = getattr(args, 'lambda_a', 1.0)
        lambda_b = getattr(args, 'lambda_b', 100.0)
        lambda_c = getattr(args, 'lambda_c', 0.6)

        self.lambda_optimizer = Adam([self.lambda_dual], lr=1.0, eps=1e-5)
        self.lambda_scheduler = LambdaLR(
            self.lambda_optimizer,
            lr_lambda=lambda k: lr_lambda(k, lambda_a, lambda_b, lambda_c)
        )

        # ==================================================== Crossing监控 ====================================================
        self.crossing_history = deque(maxlen=max(100, self.log_interval))

        # ==================================================== 训练计数器 ====================================================
        self.learning_steps = 0                        # 累计更新次数，用来控制target网络的更新
        self.target_update_interval = getattr(args, 'target_update_interval', 100)  # target 更新间隔

        # ==================================================== Wandb日志初始化 ====================================================
        wandb.init(
            project=args.env_name,
            name=getattr(args, 'wandb_name', f"DQCAC_{args.seed}"),
            config={
                **vars(args),
                'algorithm': 'DQC-AC-v5-Algorithm3-truncation',
                'num_quantiles': self.num_quantiles,
                'buffer_capacity': self.buffer_capacity,
                'batch_size': self.batch_size,
            },
            reinit=True,
            id=wandb.util.generate_id(),
            group="DQCAC",
            dir=getattr(args, 'wandb_dir', None)
        )

        # ==================================================== 预热 ====================================================
        self._warmup()

        # ==================================================== 打印初始化信息 ====================================================
        print(f"DQC-AC (v5): α={self.q_alpha}, δ={self.delta}")
        print(f"DQC-AC (v5): Distributional Critic M={self.num_quantiles} quantiles")
        print(f"DQC-AC (v5): Constraint Q >= {self.quantile_threshold}")
        print(f"DQC-AC (v5): Truncation-aware TD targets (infinite horizon)")
        print(f"DQC-AC (v5): Q estimates: {self.q_est_lower.item():.3f}, "
              f"{self.q_est.item():.3f}, {self.q_est_upper.item():.3f}")

    # ==================================================== 预热 ====================================================
    def _warmup(self):
        """
        [函数简介]: 预热阶段，收集初始数据填充 Replay Buffer

        Algorithm 3 不需要 episode return，transitions 可以直接 push
        初始分位数通过收集到的 episode returns (仅用于初始化) 来估计
        """
        print("DQC-AC: Starting warmup...")

        # 仅用于估计初始分位数，不存入 buffer
        episode_returns = []

        while len(self.replay_buffer) < self.min_buffer_size:
            state = self._reset_env()
            episode_return = 0.0
            disc_factor = 1.0

            while True:
                state_flat = state.reshape(-1)
                action = self.choose_action(state_flat, store_memory=False)
                next_state, reward, terminated, truncated, _ = self._step_env(action)

                # 只存 terminated，截断时仍需 bootstrap
                self.replay_buffer.push(
                    state_flat, action, reward, next_state.reshape(-1), float(terminated)
                )

                episode_return += disc_factor * reward
                disc_factor *= self.gamma
                state = next_state

                if terminated or truncated:
                    break

            episode_returns.append(episode_return)

        # 用 warmup 期间的 episode returns 估计初始分位数
        # 后续训练中 Q 完全由 TD targets 驱动更新
        if len(episode_returns) > 0:
            q_low = np.percentile(episode_returns, self.alpha_lower * 100)
            q_mid = np.percentile(episode_returns, self.q_alpha * 100)
            q_high = np.percentile(episode_returns, self.alpha_upper * 100)

            with torch.no_grad():
                self.q_est_lower.fill_(q_low)
                self.q_est.fill_(q_mid)
                self.q_est_upper.fill_(q_high)

        print(f"DQC-AC: Warmup done. Buffer size: {len(self.replay_buffer)}")

    # ==================================================== 主训练循环 ====================================================
    def train(self):
        """
        [函数简介]: 主训练循环 (严格 Algorithm 3)

        for each episode:
            1. 收集 episode，transitions 直接存入 Replay Buffer
            2. 执行多次更新 (全部基于 TD targets):
               - Critic Update (Distributional TD)
               - Q Update (TD targets)
               - Actor Update (TD targets)
            3. 每 outer_interval 更新 λ
        """
        disc_epi_rewards = []  # 仅用于日志监控，不参与算法
        total_steps = 0        # 累计环境交互次数
        update_counter = 0     # 内层累计更新次数，控制 λ 更新

        for i_episode in range(self.max_episode + 1):
            # ==================== 1. 收集 Episode ====================
            state = self._reset_env()
            episode_reward = 0
            disc_epi_reward = 0
            disc_factor = 1.0
            episode_steps = 0

            while True:
                state_flat = state.reshape(-1)
                action = self.choose_action(state_flat, store_memory=False)
                next_state, reward, terminated, truncated, _ = self._step_env(action)

                # 只存 terminated，截断时仍需 bootstrap
                self.replay_buffer.push(
                    state_flat, action, reward, next_state.reshape(-1), float(terminated)
                )

                episode_reward += reward
                disc_epi_reward += disc_factor * reward
                disc_factor *= self.gamma
                total_steps += 1
                episode_steps += 1

                state = next_state

                if terminated or truncated:
                    break

            # disc_epi_reward 仅用于日志监控，不参与算法更新
            disc_epi_rewards.append(disc_epi_reward)

            # 获取平均风险等级
            avg_risk_episode = self._compute_episode_avg_risk()

            # ==================== 2. 执行更新 ====================
            if len(self.replay_buffer) >= self.batch_size:
                for _ in range(self.updates_per_episode):
                    self._learn()
                    update_counter += 1

                    # ==================== 3. 外层更新 (λ) ====================
                    if update_counter % self.outer_interval == 0:
                        self._update_dual()
                        self.lambda_scheduler.step()

                    # 学习率调度
                    self.actor_scheduler.step()
                    self.q_scheduler.step()

            # ==================== 4. 日志记录 ====================
            density_est = self.estimate_density()
            crossing_violation = self._check_crossing_violation()
            self.crossing_history.append(crossing_violation)
            crossing_rate = sum(self.crossing_history) / len(self.crossing_history)

            log_dict = {
                'disc_reward/raw_reward': disc_epi_reward,
                'lambda/value': self.lambda_dual.item(),
                'action/avg_risk_episode': avg_risk_episode,
                'density/estimate': density_est.item(),
                'quantile/crossing_rate': crossing_rate,
                'buffer/size': len(self.replay_buffer),
                'training/total_steps': total_steps,
                'training/episode_steps': episode_steps,
                'training/learning_steps': self.learning_steps,
            }

            # ==================== 5. 定期详细日志 ====================
            if i_episode % self.log_interval == 0 and i_episode != 0:
                lb = max(0, len(disc_epi_rewards) - self.est_interval)
                disc_a_reward = np.mean(disc_epi_rewards[lb:])
                disc_q_reward = np.percentile(
                    disc_epi_rewards[lb:], self.q_alpha * 100
                )

                constraint_margin = disc_q_reward - self.quantile_threshold

                log_dict.update({
                    'disc_reward/aver_reward': disc_a_reward,
                    'disc_reward/quantile_reward': disc_q_reward,
                    'quantile/q_est': self.q_est.item(),
                    'quantile/q_est_lower': self.q_est_lower.item(),
                    'quantile/q_est_upper': self.q_est_upper.item(),
                    'constraint/margin': constraint_margin,
                })

                print(f'Epi:{i_episode:05d} || disc_a_r:{disc_a_reward:.03f} '
                      f'disc_q_r:{disc_q_reward:.03f} λ:{self.lambda_dual.item():.04f}')
                print(f'Epi:{i_episode:05d} || Q_low:{self.q_est_lower.item():.03f} '
                      f'Q_mid:{self.q_est.item():.03f} Q_high:{self.q_est_upper.item():.03f} '
                      f'density:{density_est.item():.04f}', '\n')

            wandb.log(log_dict, step=i_episode)

    # ==================================================== 核心学习函数 ====================================================
    def _learn(self):
        """
        [函数简介]: 从 Replay Buffer 采样并执行一次更新 (严格 Algorithm 3)

        核心: 计算 TD targets 一次，三个更新步骤共享同一组 td_targets

        [三步更新] (Algorithm 3 Step 10-19):
        1. Critic Update: Distributional TD (Quantile Huber Loss)
           - ω ← ω - ξ∇ω Σ_i Σ_j ρ^κ_{τ̂_i}(y_j - ψ_i(s,a;ω))

        2. Global Quantile Estimate Update (公式 25):
           - Q ← Q + (1/M) Σ_j β(α - I{y_j ≤ Q})
           - 用 TD targets y_j 替代 Monte Carlo 的 U(τ)

        3. Actor Update (公式 24):
           - w = (1/M) Σ_j [y_j - λ·I{y_j ≤ Q}/f̂_Z]
           - loss = -E[log π(a|s) · w]
        """
        self.learning_steps += 1

        # ==================== 采样 Batch ====================
        batch = self.replay_buffer.sample(self.batch_size)
        states = batch['states']               # [B, state_dim]
        actions = batch['actions']             # [B, action_dim]
        rewards = batch['rewards']             # [B]
        next_states = batch['next_states']     # [B, state_dim]
        terminateds = batch['terminateds']     # [B], 只有真正终止才为1

        # ==================== 计算 TD targets (三步共享) ====================
        # Algorithm 3 Step 11-12: a' ~ π(·|s';θ), y_j = r + γ·ψ_j(s',a';ω_target)
        with torch.no_grad():
            next_actions = self._sample_actions(next_states)   # [B, action_dim]
            next_quantiles = self.target_critic(next_states, next_actions)  # [B, M]

            # y_j = r + γ·ψ_j(s', a'; ω_target) · (1 - terminated)
            # 关键: 用 terminated 而非 done 做 mask
            # 截断 (truncated) 时 terminated=0，仍然 bootstrap → 正确的 infinite horizon 行为
            # 真正终止时 terminated=1，不 bootstrap → y_j = r
            td_targets = rewards.unsqueeze(1) + self.gamma * next_quantiles * (1 - terminateds.unsqueeze(1))
            # td_targets: [B, M], 论文中的 distributional Bellman target

        # ==================== 1. Critic Update (Algorithm 3 Step 13) ====================
        current_quantiles = self.online_critic(states, actions)  # [B, M]
        critic_loss = self._compute_quantile_huber_loss(current_quantiles, td_targets)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        # ==================== 2. Q Update (Algorithm 3 Step 15, 公式 25) ====================
        # Q ← Q + (1/M) Σ_j β(α - I{y_j ≤ Q})
        self._update_global_quantiles(td_targets)

        # ==================== 3. Actor Update (Algorithm 3 Step 18, 公式 24) ====================
        # w = (1/M) Σ_j [y_j - λ·I{y_j ≤ Q}/f̂_Z]
        self._update_actor(states, actions, td_targets)

        # ==================== Target 网络软更新 ====================
        if self.learning_steps % self.target_update_interval == 0:
            self._soft_update_target()

    # ==================================================== Quantile Huber Loss ====================================================
    def _compute_quantile_huber_loss(self, current_quantiles, target_quantiles):
        """
        [函数简介]: 计算 Quantile Huber Loss (QR-DQN 核心损失函数)

        [公式]:
        ρ^κ_τ(u) = |τ - I{u < 0}| · L_κ(u) / κ
        其中 L_κ(u) = 0.5u² if |u| <= κ else κ(|u| - 0.5κ)

        [输入]:
        - current_quantiles: [B, M] 当前分位数估计 ψ_i(s,a)
        - target_quantiles: [B, M] 目标分位数 y_j = r + γψ_j(s',a')

        [输出]: 标量 loss
        """
        B, M = current_quantiles.shape

        # 扩展维度用于广播: td_error[b, i, j] = y_j[b] - ψ_i[b]
        current_expanded = current_quantiles.unsqueeze(2)  # [B, M, 1]
        target_expanded = target_quantiles.unsqueeze(1)    # [B, 1, M]

        # TD error: u = y_j - ψ_i
        td_errors = target_expanded - current_expanded     # [B, M, M]

        # Huber Loss: L_κ(u) = 0.5u² if |u| <= κ else κ(|u| - 0.5κ)
        abs_td_errors = td_errors.abs()
        huber_loss = torch.where(
            abs_td_errors <= self.huber_kappa,
            0.5 * td_errors ** 2,
            self.huber_kappa * (abs_td_errors - 0.5 * self.huber_kappa)
        )
        # huber_loss: [B, M, M]

        # 分位数权重: |τ_i - I{u < 0}|
        taus = self.quantile_taus.view(1, M, 1)            # [1, M, 1]
        quantile_weight = torch.abs(
            taus - (td_errors.detach() < 0).to(td_errors.dtype)
        )  # [B, M, M]

        # element_wise: ρ^κ_τ(u) = |τ - I{u<0}| · L_κ(u) / κ
        element_wise_loss = quantile_weight * huber_loss / self.huber_kappa  # [B, M, M]

        loss = element_wise_loss.sum(dim=2).mean(dim=1).mean()

        return loss

    # ==================================================== 全局分位数更新 ====================================================
    def _update_global_quantiles(self, td_targets):
        """
        [函数简介]: 用 TD targets 更新三个全局分位数估计 (论文公式 25)

        Q_{k+1} ← Q_k + (1/M) Σ_{j=1}^{M} β_k (α - I{y_j ≤ Q_k})

        [输入]:
        - td_targets: [B, M] tensor, 每个元素 y_{b,j} = r_b + γ·ψ_j(s'_b, a'_b; ω_target)

        [与 Algorithm 1 的区别]:
        Algorithm 1 用 episode return U(τ) 更新 Q → 需要等 episode 结束
        Algorithm 3 用 TD targets y_j 更新 Q → per-transition 更新，数据效率更高

        [batch 处理]:
        对 batch 内所有 B×M 个 TD target 取均值:
        grad = -(α - (1/(B·M)) Σ_{b,j} I{y_{b,j} ≤ Q})
        """
        self.q_optimizer.zero_grad(set_to_none=True)

        # 将 [B, M] 展平为 [B*M]，所有 TD targets 作为样本
        td_flat = td_targets.detach().reshape(-1)  # [B*M]

        # Q(α-δ) 梯度: -(α-δ) + (1/(B·M)) Σ_{b,j} I{y_{b,j} ≤ Q(α-δ)}
        ind_lower = indicator(self.q_est_lower.detach(), td_flat)
        grad_lower = -(self.alpha_lower - ind_lower.mean())
        self.q_est_lower.grad = grad_lower.view(1)

        # Q(α) 梯度: -α + (1/(B·M)) Σ_{b,j} I{y_{b,j} ≤ Q(α)}
        ind_center = indicator(self.q_est.detach(), td_flat)
        grad_center = -(self.q_alpha - ind_center.mean())
        self.q_est.grad = grad_center.view(1)

        # Q(α+δ) 梯度: -(α+δ) + (1/(B·M)) Σ_{b,j} I{y_{b,j} ≤ Q(α+δ)}
        ind_upper = indicator(self.q_est_upper.detach(), td_flat)
        grad_upper = -(self.alpha_upper - ind_upper.mean())
        self.q_est_upper.grad = grad_upper.view(1)

        self.q_optimizer.step()

    # ==================================================== Actor 更新 ====================================================
    def _update_actor(self, states, actions, td_targets):
        """
        [函数简介]: Actor 更新 (严格论文公式 24 / Algorithm 3 Step 18)

        D̃ = (1/M) Σ_j [(y_j - λ·I{y_j ≤ Q}/f̂_Z) · ∇θ log π(a|s; θ)]

        [数学推导]:
        Lagrangian: L = E[U(τ)] + λ(Q_α - C)
        对 θ 求梯度 (公式 12):
          ∇θJ = E[(U(τ) - λ·I{U(τ)≤Q}/f_Z) · Σ_t ∇θ log π]
        Algorithm 3 用 TD targets 替代 U(τ)，对 M 个分位数取均值:
          D̃ = (1/M) Σ_j [(y_j - λ·I{y_j≤Q}/f̂_Z) · ∇θ log π(a|s)]

        [关键: per-quantile indicator]
        每个 TD target y_j 有各自的 indicator I{y_j ≤ Q}:
        - 不同的 ψ_j 给出不同的 TD target → 各自独立判断是否 ≤ Q
        - 这是 Algorithm 3 与 Algorithm 1 的核心区别

        [输入]:
        - states: [B, state_dim]
        - actions: [B, action_dim]
        - td_targets: [B, M] tensor, y_j = r + γ·ψ_j(s', a'; ω_target)
        """
        # ==================================================== 计算 log π(a|s) ====================================================
        log_probs = self._compute_log_probs(states, actions)
        # log_probs: [B]

        # ==================================================== 计算 per-quantile 梯度权重 ====================================================
        density = self.estimate_density()

        with torch.no_grad():
            # I{y_j ≤ Q}: 每个 TD target 独立判断 (per-quantile indicator)
            # td_targets: [B, M], q_est: 标量 → indicators: [B, M]
            indicators = indicator(self.q_est.detach(), td_targets)  # [B, M]

            # λ · I{y_j ≤ Q} / f̂_Z: 约束项 [B, M]
            constraint_term = self.lambda_dual.detach() * indicators / density.detach()
            constraint_term = torch.clamp(constraint_term, max=self.max_constraint)

            # w_{b,j} = y_j - λ·I{y_j ≤ Q}/f̂_Z
            per_quantile_weights = td_targets - constraint_term  # [B, M]

            # (1/M) Σ_j w_{b,j}: 对 M 个分位数取均值，得到每个 transition 的标量权重
            gradient_weights = per_quantile_weights.mean(dim=1)  # [B]

        # ==================================================== Actor loss ====================================================
        # loss = -E_b[log π(a_b|s_b) · w_b]，负号把梯度上升转为梯度下降
        actor_loss = -(log_probs * gradient_weights).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

    # ==================================================== Dual 更新 ====================================================
    def _update_dual(self):
        """
        [函数简介]: 更新拉格朗日乘子 λ

        λ ← ϕ(λ + ε_k(Q_k - q))

        其中 q 是约束阈值，ϕ 是投影到 [0, ∞)
        """
        self.lambda_optimizer.zero_grad(set_to_none=True)

        # Dual loss: λ · (Q - C)
        dual_loss = self.lambda_dual * (self.q_est.detach() - self.quantile_threshold)

        dual_loss.backward()
        self.lambda_optimizer.step()

        # 投影到 [0, ∞)
        with torch.no_grad():
            self.lambda_dual.clamp_(min=0.0)

    # ==================================================== 辅助方法 ====================================================
    def _sample_actions(self, states):
        """
        [函数简介]: 从策略采样动作

        [输入]: states [B, state_dim]
        [输出]: actions [B, action_dim]
        """
        means = self.actor(states)                                  # [B, action_dim]
        std = torch.exp(self.actor.log_std).unsqueeze(0)           # [1, action_dim]
        std = std.expand_as(means)                                 # [B, action_dim]

        noise = torch.randn_like(means)
        return means + noise * std

    def _compute_log_probs(self, states, actions):
        """
        [函数简介]: 计算动作的对数概率

        [输入]: states [B, state_dim], actions [B, action_dim]
        [输出]: log_probs [B]
        """
        means = self.actor(states)                                 # [B, action_dim]
        std = torch.exp(self.actor.log_std).unsqueeze(0)           # [1, action_dim]
        std = std.expand_as(means)                                 # [B, action_dim]
        var = std.pow(2)                                           # [B, action_dim]

        # 对角高斯分布 log_prob:
        # log N(a|μ,σ²) = -0.5 * [((a-μ)^2 / σ²) + 2logσ + log(2π)]
        log_probs_per_dim = -0.5 * (
            ((actions - means) ** 2) / var + 2.0 * torch.log(std) + self._log_2pi
        )
        return log_probs_per_dim.sum(dim=-1)                       # [B]

    def _soft_update_target(self):
        """Target 网络软更新"""
        with torch.no_grad():
            for target_param, online_param in zip(
                self.target_critic.parameters(),
                self.online_critic.parameters()
            ):
                target_param.data.lerp_(online_param.data, self.tau)

    def estimate_density(self):
        """
        [函数简介]: 密度估计 (公式13)

        f̂_Z = 2δ / (Q(α+δ) - Q(α-δ))
        """
        quantile_diff = self.q_est_upper.detach() - self.q_est_lower.detach()
        quantile_diff_safe = torch.clamp(quantile_diff.abs(), min=1e-6)
        density = 2 * self.delta / quantile_diff_safe
        return torch.clamp(density, max=100.0)

    def choose_action(self, state, store_memory=False):
        """
        [函数简介]: 选择动作

        [输入]: state (np.ndarray)
        [输出]: action (np.ndarray)
        """
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            mean = self.actor(state_tensor)
            std = torch.exp(self.actor.log_std)
            action = mean + torch.randn_like(mean) * std
        return action.cpu().numpy()

    def select_action(self, state):
        """评估时使用"""
        return self.choose_action(state, store_memory=False)

    def _reset_env(self):
        """兼容 Gym/Gymnasium 的 reset"""
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]
        return state

    def _step_env(self, action):
        """
        兼容 Gym/Gymnasium 的 step, 始终返回 5 值:
        (state, reward, terminated, truncated, info)

        terminated: MDP 真正结束 (infinite horizon 环境中始终 False)
        truncated: 达到 max_steps 的人为截断
        """
        outcome = self.env.step(action)
        if len(outcome) == 5:
            state, reward, terminated, truncated, info = outcome
        elif len(outcome) == 4:
            state, reward, done, info = outcome
            terminated = done
            truncated = False
        else:
            raise ValueError("env.step() 返回值格式异常")
        return state, reward, terminated, truncated, info

    def _compute_episode_avg_risk(self):
        """返回当前 episode 的平均风险等级"""
        if hasattr(self.env, 'render'):
            risk_series = self.env.render()
            if risk_series is not None:
                try:
                    if len(risk_series) > 0:
                        return float(np.mean(risk_series))
                except TypeError:
                    pass
        return 0.0

    def _check_crossing_violation(self):
        """检查分位数单调性违反"""
        q_low = self.q_est_lower.item()
        q_mid = self.q_est.item()
        q_high = self.q_est_upper.item()
        return (q_low > q_mid) or (q_mid > q_high) or (q_high - q_low < 0)
