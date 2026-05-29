import numpy as np
import torch
from torch.distributions import MultivariateNormal
from torch.optim import Adam
import wandb
from torch.optim.lr_scheduler import LambdaLR

from utils import Memory, Actor, RunningMeanStd


def lr_lambda(k, a, b, c):
    """
    学习率衰减函数
    lr(k) = a / (b + k)^c
    """
    lr = a / ((b + k) ** c)
    return lr


def indicator(x, y: torch.Tensor):
    """
    示性函数: I(y <= x)

    Args:
        x: 阈值 (标量或tensor)
        y: 输入tensor

    Returns:
        与y形状相同的tensor, y<=x处为1, 否则为0
    """
    return torch.where(y <= x, torch.ones_like(y), torch.zeros_like(y))


class QCPO(object):
    """
    QCPO (Quantile-Constrained Policy Optimization) — Probability-Based Constraint

    优化问题 (公式10):
        max_θ E[U(τ)]  s.t. P_θ(Z ≤ q) ≤ α

    其中 q 是预设阈值, α 是允许的违反概率水平。

    Lagrangian (公式31):
        L(θ, λ) = E_θ[Z] - λ(P_θ(Z ≤ q) - α),  λ ≥ 0

    策略梯度 (公式22):
        D = (U(τ) - λ·1{U(τ) ≤ q}) Σ ∇_θ log π(a|s;θ)
        - indicator直接使用阈值q, 不需要估计密度

    Dual更新 (公式21):
        λ ← φ(λ + ε_k(1{U(τ) ≤ q} - α))
        - 基于经验概率, 约束违反时λ增大, 满足时λ减小

    分位数估计 (公式20, 仅监控用):
        Q_{k+1} = Q_k + β_k(α - 1{U(τ_k) ≤ Q_k})
    """

    def __init__(self, args, env):
        self.device = args.device

        # 训练参数
        self.log_interval = args.log_interval              # 日志记录间隔
        self.est_interval = args.est_interval              # 滚动窗口大小
        self.q_alpha = args.q_alpha                        # 分位数水平 α
        self.gamma = args.gamma                            # 折扣因子
        self.max_episode = args.max_episode                # 最大训练轮数

        # QCPO特有参数
        self.quantile_threshold = args.quantile_threshold  # 约束阈值 q
        self.outer_interval = args.outer_interval          # 外层更新间隔

        # 环境
        self.env = env
        self.env_name = args.env_name

        # 策略网络
        state_dim = np.prod(self.env.observation_space.shape)
        action_dim = np.prod(self.env.action_space.shape)
        self.actor = Actor(state_dim, action_dim, args.init_std)

        # Return 归一化器 (Reward Normalization)
        # 使梯度权重中的 U(τ) 项归一化到 ~O(1), 与约束项量级匹配
        self.return_rms = RunningMeanStd()

        # 策略优化器和学习率调度器
        self.optimizer = Adam(self.actor.parameters(), 1., eps=1e-5)
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda k: lr_lambda(k, args.theta_a, args.theta_b, args.theta_c)
        )

        # 经验回放
        self.memory = Memory()

        # wandb日志
        wandb.init(
            project=args.env_name,
            name=f"{args.algo_name}_{args.seed}",
            config=vars(args),
            reinit=True,
            group=args.algo_name,
            dir=getattr(args, 'wandb_dir', None)
        )

        # 预热估计初始分位数 (仅用于监控)
        q = self.warm_up(10 * self.est_interval)
        self.q_est = torch.autograd.Variable(q * torch.ones((1,))).to(self.device)

        # 分位数估计器的优化器 (仅用于监控Q_est的追踪)
        self.q_optimizer = Adam([self.q_est], 1., eps=1e-5)
        self.q_scheduler = LambdaLR(
            self.q_optimizer,
            lr_lambda=lambda k: lr_lambda(k, args.q_a, args.q_b, args.q_c)
        )

        # 初始化拉格朗日乘子λ
        self.lambda_dual = torch.tensor([0.0], device=self.device, requires_grad=True)

        # λ的优化器
        self.lambda_optimizer = Adam([self.lambda_dual], lr=1.0, eps=1e-5)
        self.lambda_scheduler = LambdaLR(
            self.lambda_optimizer,
            lr_lambda=lambda k: lr_lambda(k, args.lambda_a, args.lambda_b, args.lambda_c)
        )

    def warm_up(self, max_episode):
        """
        预热阶段：运行轨迹估计初始分位数 + 初始化 return 归一化统计量

        采样max_episode条轨迹, 计算经验α-分位数作为Q_est的初始值。
        同时用这些 return 预热 RunningMeanStd, 确保训练开始时归一化已稳定。
        """
        disc_epi_rewards = []
        for _ in range(max_episode):
            disc_epi_reward, disc_factor, state = 0, 1, self._reset_env()
            while True:
                state = state.flatten()
                action = self.choose_action(state)
                state, reward, done, _ = self._step_env(action)
                disc_epi_reward += disc_factor * reward
                disc_factor *= self.gamma
                if done:
                    break
            disc_epi_rewards.append(disc_epi_reward)
            self.return_rms.update(disc_epi_reward)         # 预热归一化统计量

        q = np.percentile(disc_epi_rewards, self.q_alpha * 100)
        print(f'QCPO warm up || n_epi:{max_episode:04d} {self.q_alpha:.2f}-quantile:{q:.3f}'
              f' return_mean:{self.return_rms.mean:.3f} return_std:{self.return_rms.std:.3f}')
        self.memory.clear()
        return q

    def train(self):
        """
        主训练循环 (Primal-Dual, Probability-Based Constraint)

        每个episode:
          1. 采样轨迹, 计算U(τ)
          2. 内层更新: θ (策略) 和 Q_est (监控用分位数)
          3. 外层更新 (每outer_interval次): λ (拉格朗日乘子)
        """
        disc_epi_rewards = []                              # 记录所有episode的折扣回报
        inner_step_counter = 0                             # 内层步数计数器

        for i_episode in range(self.max_episode + 1):
            # ========== 采样轨迹 ==========
            disc_epi_reward, disc_factor, state = 0, 1, self._reset_env()
            episode_reward = 0                             # 无折扣累积奖励
            while True:
                action = self.choose_action(state)
                state, reward, done, _ = self._step_env(action)
                episode_reward += reward
                disc_epi_reward += disc_factor * reward    # U(τ) = Σ γ^t r_t
                disc_factor *= self.gamma
                self.memory.rewards.append(reward)
                self.memory.is_terminals.append(done)
                if done:
                    break

            avg_risk_episode = self._compute_episode_avg_risk()

            # ========== 更新 return 归一化统计量 ==========
            self.return_rms.update(disc_epi_reward)

            # ========== 内层更新 (θ, Q_est) ==========
            self.update_inner()
            self.memory.clear()

            inner_step_counter += 1

            # ========== 外层更新 (λ), 每outer_interval次 ==========
            if inner_step_counter >= self.outer_interval:
                # 取最近outer_interval条轨迹的回报, 计算经验概率
                recent_rewards = disc_epi_rewards[-self.outer_interval:]
                self.update_dual(recent_rewards)
                inner_step_counter = 0

            disc_epi_rewards.append(disc_epi_reward)

            # ========== 每episode日志 ==========
            # 计算当前轨迹的约束违反指标: 1{U(τ) ≤ q}
            violation = 1.0 if disc_epi_reward <= self.quantile_threshold else 0.0

            wandb.log({
                'disc_reward/raw_reward': episode_reward,
                'disc_reward/discounted_reward': disc_epi_reward,
                'lambda/value': self.lambda_dual.item(),
                'action/avg_risk_episode': avg_risk_episode,
                'constraint/violation': violation,
                'normalize/return_mean': self.return_rms.mean,
                'normalize/return_std': self.return_rms.std,
            }, step=i_episode)

            # ========== 周期性详细日志 ==========
            if i_episode % self.log_interval == 0 and i_episode != 0:
                lb = max(0, len(disc_epi_rewards) - self.est_interval)
                disc_a_reward = np.mean(disc_epi_rewards[lb:])         # 滚动平均回报
                disc_q_reward = np.percentile(                         # 经验α-分位数
                    disc_epi_rewards[lb:], self.q_alpha * 100
                )

                # 经验概率 P̂(U(τ) ≤ q) over 滚动窗口
                recent = disc_epi_rewards[lb:]
                empirical_prob = np.mean(
                    [1.0 if r <= self.quantile_threshold else 0.0 for r in recent]
                )

                # 约束余量: α - P̂(U≤q), 正值表示约束满足
                constraint_margin = self.q_alpha - empirical_prob

                wandb.log({
                    'disc_reward/aver_reward': disc_a_reward,
                    'disc_reward/quantile_reward': disc_q_reward,
                    'quantile/q_est': self.q_est.item(),
                    'constraint/empirical_prob': empirical_prob,
                    'constraint/margin': constraint_margin,
                }, step=i_episode)

                print(f'Epi:{i_episode:05d} || disc_a_r:{disc_a_reward:.03f} '
                      f'disc_q_r:{disc_q_reward:.03f} λ:{self.lambda_dual.item():.04f}')
                print(f'Epi:{i_episode:05d} || P(U≤q):{empirical_prob:.03f} '
                      f'α:{self.q_alpha:.03f} margin:{constraint_margin:.03f}')
                print(f'Epi:{i_episode:05d} || lr:{self.scheduler.get_last_lr()[0]:.2e} '
                      f'q_lr:{self.q_scheduler.get_last_lr()[0]:.2e} '
                      f'λ_lr:{self.lambda_scheduler.get_last_lr()[0]:.2e}\n')

            # ========== 学习率调度器步进 ==========
            self.scheduler.step()                          # 策略学习率衰减
            self.q_scheduler.step()                        # Q_est学习率衰减
            self.lambda_scheduler.step()                   # λ学习率衰减

    def choose_action(self, state):
        """
        根据当前策略采样动作（训练时使用，会存储到memory）
        """
        state = torch.from_numpy(state).float()
        mean = self.actor(state)
        var = torch.diag(torch.exp(2 * self.actor.log_std))
        dist = MultivariateNormal(mean, var)
        action = dist.sample()

        self.memory.states.append(state)
        self.memory.actions.append(action)

        return action.detach().data.cpu().numpy()

    def select_action(self, state):
        """
        选择动作（评估时使用，不存储memory）
        """
        state = torch.from_numpy(state).float()
        mean = self.actor(state)
        var = torch.diag(torch.exp(2 * self.actor.log_std))
        dist = MultivariateNormal(mean, var)
        action = dist.sample()
        return action.detach().data.cpu().numpy()

    def _reset_env(self):
        """兼容Gym/Gymnasium reset接口"""
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]
        return state

    def _step_env(self, action):
        """兼容Gym/Gymnasium step接口"""
        outcome = self.env.step(action)
        if isinstance(outcome, tuple):
            if len(outcome) == 5:
                state, reward, terminated, truncated, info = outcome
                done = terminated or truncated
            elif len(outcome) == 4:
                state, reward, done, info = outcome
            else:
                raise ValueError("env.step() 返回值格式异常")
        else:
            raise ValueError("env.step() 必须返回tuple")
        return state, reward, done, info

    def evaluate(self, state, action):
        """计算动作的对数概率和熵"""
        mean = self.actor(state)
        var = torch.diag(torch.exp(2 * self.actor.log_std))
        dist = MultivariateNormal(mean, var)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, dist_entropy

    def compute_discounted_epi_reward(self):
        """
        计算折扣累积奖励

        返回:
          disc_reward: [T]张量, 每个timestep的值都等于其所属轨迹的U(τ)
          disc_reward_short: [N]张量, 每条轨迹一个标量U(τ)
        """
        memory_len = self.memory.get_len()
        disc_reward = np.zeros(memory_len, dtype=float)
        disc_reward_short = []
        pre_r_sum, p1, p2 = 0, 0, 0

        for i in range(memory_len - 1, -1, -1):
            if self.memory.is_terminals[i]:
                if p1 > 0:
                    disc_reward[memory_len - p1: memory_len - p2] += pre_r_sum
                    disc_reward_short.insert(0, pre_r_sum)
                pre_r_sum, p2 = 0, p1
            pre_r_sum = self.memory.rewards[i] + self.gamma * pre_r_sum
            p1 += 1

        disc_reward[memory_len - p1: memory_len - p2] += pre_r_sum
        disc_reward_short.insert(0, pre_r_sum)

        disc_reward = torch.from_numpy(disc_reward).to(self.device).float()
        disc_reward_short = torch.tensor(disc_reward_short).to(self.device).float()

        return disc_reward, disc_reward_short

    def update_inner(self):
        """
        内层更新: 策略参数θ 和 分位数估计Q_est(监控用)

        策略梯度 (公式22 + Reward Normalization):
            D = ((U(τ)-μ)/σ - λ·1{U(τ) ≤ q}) Σ ∇_θ log π(a|s;θ)

        Reward Normalization:
            - 目标项归一化: (U(τ)-μ)/σ ~ O(1), 使 λ 只需 O(1) 即可平衡
            - indicator 仍用原始尺度: 1{U_raw(τ) ≤ q}, 约束语义不变
            - 减均值 = baseline (无偏), 除std = 自适应步长 (实际无偏)

        分位数更新 (公式20, 仅监控):
            Q_{k+1} = Q_k + β_k(α - 1{U(τ_k) ≤ Q_k})
        """
        self.actor.to(self.device)

        # ====== 计算折扣奖励 ======
        disc_reward, disc_reward_short = self.compute_discounted_epi_reward()

        # ====== 计算示性函数 (原始尺度!) ======
        # 1{U(τ) ≤ q}: 使用固定阈值q和原始尺度的U(τ)
        # 约束 P(Z≤q) ≤ α 定义在原始回报空间, indicator 不能归一化
        ind = indicator(self.quantile_threshold, disc_reward)

        # ====== 归一化目标项 (Reward Normalization) ======
        # (U(τ) - μ) / σ → ~O(1), 使 λ·1{...} 只需 λ~O(1) 即可平衡
        return_mean = self.return_rms.mean
        return_std = self.return_rms.std
        disc_reward_normalized = (disc_reward - return_mean) / return_std

        # ====== 获取旧状态和动作 ======
        old_states = torch.stack(self.memory.states).to(self.device).detach()
        old_actions = torch.stack(self.memory.actions).to(self.device).detach()

        # ====== 计算对数概率 ======
        logprobs, _ = self.evaluate(old_states, old_actions)

        # ====== 计算策略梯度权重 ======
        # gradient_weights = (U(τ)-μ)/σ - λ · 1{U_raw(τ) ≤ q}
        # 归一化后的 return ~O(1), λ ~O(1), 两项量级匹配
        gradient_weights = disc_reward_normalized - self.lambda_dual * ind

        # ====== 更新策略网络 ======
        # loss = -E[D], optimizer做梯度下降等价于梯度上升
        loss = -torch.mean(logprobs * gradient_weights)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ====== 更新分位数估计 (公式20, 仅监控用) ======
        # Q_{k+1} = Q_k + β_k(α - 1{U(τ_k) ≤ Q_k})
        # 手动设置梯度: grad = -(α - 1{U(τ)≤Q}), 因为optimizer做梯度下降
        self.q_optimizer.zero_grad()
        self.q_est.grad = -torch.mean(
            self.q_alpha - indicator(self.q_est.detach(), disc_reward_short),
            dim=0, keepdim=True
        )
        self.q_optimizer.step()

        self.actor.to(torch.device('cpu'))

    def update_dual(self, disc_epi_rewards_window):
        """
        外层更新: 拉格朗日乘子 λ (公式21)

        基于经验概率的dual更新:
            λ ← φ(λ + ε_k · (P̂(U(τ) ≤ q) - α))

        由于 actor 使用了 Reward Normalization, 梯度权重中 U_norm ~O(1),
        λ 只需 O(1) 即可平衡, dual 梯度 (P̂-α) ∈ [-α, 1-α] 量级恰好匹配。
        """
        # 计算经验违反概率: P̂(U(τ) ≤ q)
        if len(disc_epi_rewards_window) > 0:
            empirical_prob = np.mean(
                [1.0 if r <= self.quantile_threshold else 0.0
                 for r in disc_epi_rewards_window]
            )
        else:
            empirical_prob = 0.0

        self.lambda_optimizer.zero_grad()

        # Dual loss 设计:
        # loss = -λ * (P̂ - α)
        # ∂loss/∂λ = -(P̂ - α)
        # optimizer step: λ ← λ - lr*(-(P̂-α)) = λ + lr*(P̂-α)  (公式21)
        constraint_violation = empirical_prob - self.q_alpha
        dual_loss = -self.lambda_dual * constraint_violation

        dual_loss.backward()
        self.lambda_optimizer.step()

        # 投影到 [0, ∞)
        with torch.no_grad():
            self.lambda_dual.clamp_(min=0.0)

    def _compute_episode_avg_risk(self):
        """返回当前episode的平均风险等级"""
        if hasattr(self.env, 'render'):
            risk_series = self.env.render()
            if risk_series is not None:
                try:
                    if len(risk_series) > 0:
                        return float(np.mean(risk_series))
                except TypeError:
                    pass
        return 0.0
