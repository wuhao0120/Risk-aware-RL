import numpy as np
import torch
from torch.distributions import MultivariateNormal
from torch.optim import Adam
import wandb
from torch.optim.lr_scheduler import LambdaLR

from utils import Memory, Actor


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
    QCPO-Density (Quantile-Constrained Policy Optimization with Density Estimation)
    
    目标: 最大化累积回报的期望 E[U(τ)]
    约束: Q_α(U(τ)) >= C (分位数约束)
    
    使用Primal-Dual方法求解，支持两种梯度估计方式:
    
    1. 原始版本 (density_estimate=False):
       D(τ, θ, Q) = (U(τ) - ν·λ·I{U(τ) ≤ Q}) Σ ∇_θ log π(a|s;θ)
       
    2. 密度估计版本 (density_estimate=True):
       维护三个分位数 Q_{τ-h}, Q_τ, Q_{τ+h}
       估计密度: π̂(Q_τ) = 2h_n / (Q̂_{τ+h} - Q̂_{τ-h})
       D(τ, θ, Q) = (U(τ) - λ·I{U(τ) ≤ Q}/π̂(Q_τ)) Σ ∇_θ log π(a|s;θ)
    """
    
    def __init__(self, args, env):
        self.device = args.device

        # 训练参数
        self.log_interval = args.log_interval
        self.est_interval = args.est_interval
        self.q_alpha = args.q_alpha
        self.gamma = args.gamma
        self.max_episode = args.max_episode

        # QCPO特有参数
        self.density_estimate = args.density_estimate  # 是否使用密度估计
        self.quantile_threshold = args.quantile_threshold   # 分位数约束阈值 C
        self.outer_interval = args.outer_interval           # 外层更新间隔

        # ==== 修复方案参数 (解决 λ/density 比值失控问题) ====
        # 方案B: clamp constraint_term 使约束项不超过 |U(τ)|*ratio
        #   constraint_clamp_ratio > 0 时启用, 例如 2.0 表示约束项最大为 2*|U|
        self.constraint_clamp_ratio = getattr(args, 'constraint_clamp_ratio', 0.0)
        # 方案C: 归一化 gradient_weights, 使 U(τ) 和 constraint_term 量级可比
        #   'none'=不归一化(默认), 'scale'=按max(|U|,|C|)缩放, 'standardize'=均值0方差1
        self.normalize_gradient = getattr(args, 'normalize_gradient', 'none')

        if not self.density_estimate:
            # 原始版本使用nu
            self.nu = args.nu
        else:
            # 密度估计版本的参数
            self.h_n = args.h_n # 分位数间隔，默认0.01
            self.q_alpha_low = self.q_alpha - self.h_n   # τ - h_n = 0.24
            self.q_alpha_high = self.q_alpha + self.h_n  # τ + h_n = 0.26

        # 环境
        self.env = env
        self.env_name = args.env_name

        # 策略网络
        state_dim = np.prod(self.env.observation_space.shape)
        action_dim = np.prod(self.env.action_space.shape)
        actor_hidden = getattr(args, 'actor_hidden_dims', None)
        self.actor = Actor(state_dim, action_dim, args.init_std, hidden_dims=actor_hidden).to(self.device)

        # 策略优化器和学习率调度器
        self.optimizer = Adam(self.actor.parameters(), 1., eps=1e-5)
        self.scheduler = LambdaLR(
            self.optimizer, 
            lr_lambda=lambda k: lr_lambda(k, args.theta_a, args.theta_b, args.theta_c)
        )
        
        # 经验回放
        self.memory = Memory()
        
        # wandb日志
        algo_suffix = "_density" if self.density_estimate else "_nu"
        wandb.init(
            project=args.env_name, 
            name=f"{args.algo_name}{algo_suffix}_{args.seed}", 
            config={**vars(args), 'density_estimate': self.density_estimate}, 
            reinit=True, 
            id=wandb.util.generate_id(),
            group=f"{args.algo_name}{algo_suffix}",
            dir=getattr(args, 'wandb_dir', None)
        )

        # 预热估计初始分位数
        if self.density_estimate:
            # 密度估计版本：需要估计三个分位数
            q_low, q_mid, q_high = self.warm_up_density(10 * self.est_interval)
            self.q_est_low = torch.autograd.Variable(q_low * torch.ones((1,))).to(self.device)
            self.q_est = torch.autograd.Variable(q_mid * torch.ones((1,))).to(self.device)
            self.q_est_high = torch.autograd.Variable(q_high * torch.ones((1,))).to(self.device)
        else:
            # 原始版本：只需要一个分位数
            q = self.warm_up(10 * self.est_interval)
            self.q_est = torch.autograd.Variable(q * torch.ones((1,))).to(self.device)
        
        # 分位数估计器的优化器
        if self.density_estimate:
            self.q_optimizer = Adam([self.q_est_low, self.q_est, self.q_est_high], 1., eps=1e-5)
        else:
            self.q_optimizer = Adam([self.q_est], 1., eps=1e-5)
            
        self.q_scheduler = LambdaLR(
            self.q_optimizer, 
            lr_lambda=lambda k: lr_lambda(k, args.q_a, args.q_b, args.q_c)
        )
        
        # 初始化拉格朗日乘子λ
        self.lambda_dual = torch.tensor([0.0], device=self.device, requires_grad=True)
        
        # λ的优化器
        lambda_a = args.lambda_a
        lambda_b = args.lambda_b
        lambda_c = args.lambda_c
        
        self.lambda_optimizer = Adam([self.lambda_dual], lr=1.0, eps=1e-5)
        self.lambda_scheduler = LambdaLR(
            self.lambda_optimizer,
            lr_lambda=lambda k: lr_lambda(k, lambda_a, lambda_b, lambda_c)
        )
        
    def warm_up(self, max_episode):
        """
        预热阶段：运行轨迹估计初始分位数（原始版本）
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
        
        q = np.percentile(disc_epi_rewards, self.q_alpha * 100)
        print(f'QCPO warm up || n_epi:{max_episode:04d} {self.q_alpha:.2f}-quantile:{q:.3f}')
        self.memory.clear()
        return q

    def warm_up_density(self, max_episode):
        """
        预热阶段：运行轨迹估计三个初始分位数（密度估计版本）
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
        
        q_low = np.percentile(disc_epi_rewards, self.q_alpha_low * 100)
        q_mid = np.percentile(disc_epi_rewards, self.q_alpha * 100)
        q_high = np.percentile(disc_epi_rewards, self.q_alpha_high * 100)
        
        print(f'QCPO-Density warm up || n_epi:{max_episode:04d}')
        print(f'  Q_{self.q_alpha_low:.2f}={q_low:.3f}, Q_{self.q_alpha:.2f}={q_mid:.3f}, Q_{self.q_alpha_high:.2f}={q_high:.3f}')
        self.memory.clear()
        return q_low, q_mid, q_high

    def train(self):
        """主训练循环 (Primal-Dual)"""
        disc_epi_rewards = []
        inner_step_counter = 0
                                    
        for i_episode in range(self.max_episode + 1):
            # 采样轨迹
            disc_epi_reward, disc_factor, state = 0, 1, self._reset_env()
            episode_reward = 0
            while True:
                action = self.choose_action(state)
                state, reward, done, _ = self._step_env(action)
                episode_reward += reward
                disc_epi_reward += disc_factor * reward
                disc_factor *= self.gamma
                self.memory.rewards.append(reward)
                self.memory.is_terminals.append(done)
                if done:
                    break
            
            avg_risk_episode = self._compute_episode_avg_risk()
            regime_stats = self._compute_regime_stats()

            # 内层更新(θ, Q)
            self.update_inner()
            self.memory.clear()

            inner_step_counter += 1

            # 外层更新(λ)，每outer_interval次执行一次
            if inner_step_counter >= self.outer_interval:
                self.update_dual()
                inner_step_counter = 0

            disc_epi_rewards.append(disc_epi_reward)

            log_dict = {
                'disc_reward/raw_reward': disc_epi_reward,
                'lambda/value': self.lambda_dual.item(),
                'action/avg_risk_episode': avg_risk_episode
            }
            log_dict.update(regime_stats)

            if self.density_estimate:
                # 计算密度估计及 λ/f 比值 (诊断量级失控)
                density_est = self.compute_density_estimate()
                log_dict['density/estimate'] = density_est.item()
                lambda_over_density = self.lambda_dual.item() / max(density_est.item(), 1e-8)
                log_dict['debug/lambda_over_density'] = lambda_over_density
                log_dict['debug/ratio_to_reward'] = lambda_over_density / max(abs(disc_epi_reward), 1.0)
            
            # 日志记录
            if i_episode % self.log_interval == 0 and i_episode != 0:
                lb = max(0, len(disc_epi_rewards) - self.est_interval)
                disc_a_reward = np.mean(disc_epi_rewards[lb:])
                disc_q_reward = np.percentile(disc_epi_rewards[lb:], self.q_alpha * 100)
                
                # 约束满足余量: Q - C
                constraint_margin = disc_q_reward - self.quantile_threshold
                
                interval_log_dict = {
                    'disc_reward/aver_reward': disc_a_reward,
                    'disc_reward/quantile_reward': disc_q_reward,
                    'quantile/q_est': self.q_est.item(),
                    'constraint/margin': constraint_margin
                }
                
                if self.density_estimate:
                    interval_log_dict['quantile/q_est_low'] = self.q_est_low.item()
                    interval_log_dict['quantile/q_est_high'] = self.q_est_high.item()
                
                log_dict.update(interval_log_dict)
                
                print(f'Epi:{i_episode:05d} || disc_a_r:{disc_a_reward:.03f} '
                      f'disc_q_r:{disc_q_reward:.03f} λ:{self.lambda_dual.item():.04f}')
                print(f'Epi:{i_episode:05d} || avg_risk:{avg_risk_episode:.03f} '
                      f'constraint_margin:{constraint_margin:.03f}')
                
                if self.density_estimate:
                    density_est = self.compute_density_estimate()
                    print(f'Epi:{i_episode:05d} || Q_low:{self.q_est_low.item():.03f} '
                          f'Q_mid:{self.q_est.item():.03f} Q_high:{self.q_est_high.item():.03f} '
                          f'density:{density_est.item():.04f}')
                
                print(f'Epi:{i_episode:05d} || lr:{self.scheduler.get_last_lr()[0]:.2e} '
                      f'q_lr:{self.q_scheduler.get_last_lr()[0]:.2e} '
                      f'λ_lr:{self.lambda_scheduler.get_last_lr()[0]:.2e}\n')

            wandb.log(log_dict, step=i_episode)
            
            self.scheduler.step()
            self.q_scheduler.step()
            self.lambda_scheduler.step()

    def choose_action(self, state):
        """
        根据当前策略采样动作（训练时使用，会存储到memory）
        """
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(self.device)
            mean = self.actor(state)
            var = torch.diag(torch.exp(2 * self.actor.log_std))
            dist = MultivariateNormal(mean, var)
            action = dist.sample()

        self.memory.states.append(state)
        self.memory.actions.append(action)

        return action.cpu().numpy()

    def select_action(self, state):
        """
        选择动作（评估时使用，不存储memory）
        """
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(self.device)
            mean = self.actor(state)
            var = torch.diag(torch.exp(2 * self.actor.log_std))
            dist = MultivariateNormal(mean, var)
            action = dist.sample()
        return action.cpu().numpy()

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
        """计算折扣累积奖励"""
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

    def compute_density_estimate(self):
        """
        计算reward密度函数在Q_τ分位数上的估计值
        
        公式: π̂{Q_Y(τ | x) | x} = 2h_n / (Q̂_Y(τ + h_n | x) - Q̂_Y(τ - h_n | x))
        
        其中:
        - h_n 是分位数间隔
        - Q̂_Y(τ + h_n | x) 是 τ+h_n 分位数的估计
        - Q̂_Y(τ - h_n | x) 是 τ-h_n 分位数的估计
        """
        denominator = self.q_est_high.detach() - self.q_est_low.detach()
        # 避免除零，添加小的epsilon
        denominator = torch.clamp(denominator, min=1e-6)
        density = (2.0 * self.h_n) / denominator
        return density

    def update_inner(self):
        """
        内层更新: 策略参数θ 和 分位数估计Q
        
        两种版本的策略梯度:
        1. 原始版本 (density_estimate=False):
           D(τ, θ, Q) = (U(τ) - ν·λ·I{U(τ) ≤ Q}) Σ ∇_θ log π(a|s;θ)
           
        2. 密度估计版本 (density_estimate=True):
           D(τ, θ, Q) = (U(τ) - λ·I{U(τ) ≤ Q}/π̂(Q_τ)) Σ ∇_θ log π(a|s;θ)
        """
        # 计算折扣奖励
        disc_reward, disc_reward_short = self.compute_discounted_epi_reward()

        # 计算示性函数: I{U(τ) ≤ Q}
        ind = indicator(self.q_est.detach(), disc_reward)

        # 获取旧状态和动作 (已在device上)
        old_states = torch.stack(self.memory.states).detach()
        old_actions = torch.stack(self.memory.actions).detach()

        # 计算对数概率
        logprobs, _ = self.evaluate(old_states, old_actions)
        
        # 计算策略梯度权重
        if self.density_estimate:
            # 密度估计版本: U(τ) - λ·I{U(τ) ≤ Q}/π̂(Q_τ)
            density = self.compute_density_estimate()
            constraint_term = self.lambda_dual * ind / density
        else:
            # 原始版本: U(τ) - ν·λ·I{U(τ) ≤ Q}
            constraint_term = self.nu * self.lambda_dual * ind

        # ==== 方案B: Clamp constraint_term ====
        # 防止 λ/density 失控导致约束项淹没奖励项
        if self.constraint_clamp_ratio > 0:
            reward_scale = disc_reward.abs().mean().detach()          # |U(τ)| 的平均量级
            clamp_max = self.constraint_clamp_ratio * reward_scale    # 约束项上界
            constraint_term = torch.clamp(constraint_term, max=clamp_max)

        gradient_weights = disc_reward - constraint_term

        # ==== 方案C: 归一化 gradient_weights ====
        if self.normalize_gradient == 'scale':
            # 按 max(|U|, |C|) 缩放, 使两项量级相当
            scale = torch.max(disc_reward.abs().mean(), constraint_term.abs().mean()).detach()
            scale = torch.clamp(scale, min=1.0)                      # 防除零
            gradient_weights = gradient_weights / scale
        elif self.normalize_gradient == 'standardize':
            # 标准化为均值0方差1
            gw_std = gradient_weights.std().detach()
            gw_std = torch.clamp(gw_std, min=1.0)                    # 防除零
            gradient_weights = gradient_weights / gw_std

        # 损失函数 (负号因为我们做梯度上升)
        loss = -torch.mean(logprobs * gradient_weights)

        # 更新策略网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新分位数估计
        self.q_optimizer.zero_grad()
        
        if self.density_estimate:
            # 更新三个分位数
            self.q_est_low.grad = -torch.mean(
                self.q_alpha_low - indicator(self.q_est_low.detach(), disc_reward_short), 
                dim=0, keepdim=True
            )
            self.q_est.grad = -torch.mean(
                self.q_alpha - indicator(self.q_est.detach(), disc_reward_short), 
                dim=0, keepdim=True
            )
            self.q_est_high.grad = -torch.mean(
                self.q_alpha_high - indicator(self.q_est_high.detach(), disc_reward_short), 
                dim=0, keepdim=True
            )
        else:
            # 只更新一个分位数
            self.q_est.grad = -torch.mean(
                self.q_alpha - indicator(self.q_est.detach(), disc_reward_short), 
                dim=0, keepdim=True
            )
        
        self.q_optimizer.step()

    def update_dual(self):
        """
        外层更新: 拉格朗日乘子 λ
        
        目标: min_λ L(λ) ≈ E[U] + λ(Q - C)  (s.t. λ >= 0)
        
        梯度下降: λ ← λ - lr * (Q - C)
        - 约束满足 (Q > C): λ 减小
        - 约束违反 (Q < C): λ 增大
        """
        self.lambda_optimizer.zero_grad()
        
        # Dual loss
        dual_loss = self.lambda_dual * (self.q_est.detach() - self.quantile_threshold)  
        
        dual_loss.backward()
        self.lambda_optimizer.step()
        
        # 投影到 [0, inf)
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

    def _compute_regime_stats(self):
        """计算当前episode的per-regime统计"""
        stats = {}
        if not hasattr(self.env, 'get_regime_history'):
            return stats
        regime_hist = self.env.get_regime_history()
        risk_series = self.env.render()
        if regime_hist is None or risk_series is None or len(risk_series) == 0:
            return stats
        step_regimes = regime_hist[:-1] if len(regime_hist) > len(risk_series) else regime_hist
        step_regimes = np.array(step_regimes[:len(risk_series)])
        risk_arr = np.array(risk_series)
        regime_names = {0: 'calm', 1: 'volatile', 2: 'crisis'}
        total_steps = len(risk_arr)
        for rid, rname in regime_names.items():
            mask = (step_regimes == rid)
            count = mask.sum()
            stats[f'regime/{rname}_frac'] = float(count) / total_steps
            if count > 0:
                stats[f'action/risk_{rname}'] = float(risk_arr[mask].mean())
        return stats
