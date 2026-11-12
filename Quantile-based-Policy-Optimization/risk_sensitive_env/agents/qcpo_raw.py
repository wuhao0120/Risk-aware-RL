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
    QCPO (Quantile-Constrained Policy Optimization)
    
    目标: 最大化累积回报的期望 E[U(τ)]
    约束: Q_α(U(τ)) >= C (分位数约束)
    
    使用Primal-Dual方法求解:
    - Primal: 更新策略参数θ和分位数估计Q
    - Dual: 更新拉格朗日乘子λ
    
    核心算法:
    1. 内层更新(θ, Q): 使用带约束惩罚项的策略梯度
       D(τ, θ, Q) = (U(τ) - ν·λ·I{U(τ) ≤ Q}) Σ ∇_θ log π(a|s;θ)
    2. 外层更新(λ): 基于约束违反程度更新
       λ ← λ - lr * (Q - C)
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
        self.nu = args.nu                                   # 约束惩罚系数
        self.quantile_threshold = args.quantile_threshold   # 分位数约束阈值 C
        self.outer_interval = args.outer_interval           # 外层更新间隔

        # 环境
        self.env = env
        self.env_name = args.env_name

        # 策略网络
        state_dim = np.prod(self.env.observation_space.shape)
        action_dim = np.prod(self.env.action_space.shape)
        actor_hidden = getattr(args, 'actor_hidden_dims', None)
        self.actor = Actor(state_dim, action_dim, args.init_std, hidden_dims=actor_hidden)

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
            name=f"{args.algo_name}_pd_{args.seed}", 
            config=vars(args), 
            reinit=True, 
            group=args.algo_name,
            dir=getattr(args, 'wandb_dir', None)
        )

        # 预热估计初始分位数
        q = self.warm_up(10 * self.est_interval)
        self.q_est = torch.autograd.Variable(q * torch.ones((1,))).to(self.device)
        
        # 分位数估计器的优化器
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
        预热阶段：运行轨迹估计初始分位数
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
        print(f'QCPO (Primal-Dual) warm up || n_epi:{max_episode:04d} {self.q_alpha:.2f}-quantile:{q:.3f}')
        self.memory.clear()
        return q

    def train(self):
        """主训练循环 (Primal-Dual)"""
        disc_epi_rewards = []
        inner_step_counter = 0
                                    
        for i_episode in range(self.max_episode + 1):
            # 采样轨迹
            disc_epi_reward, disc_factor, state = 0, 1, self._reset_env()
            while True:
                action = self.choose_action(state)
                state, reward, done, _ = self._step_env(action)
                disc_epi_reward += disc_factor * reward
                disc_factor *= self.gamma
                self.memory.rewards.append(reward)
                self.memory.is_terminals.append(done)
                if done:
                    break
            
            avg_risk_episode = self._compute_episode_avg_risk()
            
            # 内层更新(θ, Q)
            self.update_inner()
            self.memory.clear()
            
            inner_step_counter += 1
            
            # 外层更新(λ)，每outer_interval次执行一次
            if inner_step_counter >= self.outer_interval:
                self.update_dual()
                inner_step_counter = 0
            
            disc_epi_rewards.append(disc_epi_reward)

            wandb.log({
                'disc_reward/raw_reward': disc_epi_reward,
                'lambda/value': self.lambda_dual.item(),
                'action/avg_risk_episode': avg_risk_episode
            }, step=i_episode)
            
            # 日志记录
            if i_episode % self.log_interval == 0 and i_episode != 0:
                lb = max(0, len(disc_epi_rewards) - self.est_interval)
                disc_a_reward = np.mean(disc_epi_rewards[lb:])
                disc_q_reward = np.percentile(disc_epi_rewards[lb:], self.q_alpha * 100)
                
                # 约束满足余量: Q - C
                constraint_margin = disc_q_reward - self.quantile_threshold
                
                wandb.log({
                    'disc_reward/aver_reward': disc_a_reward,
                    'disc_reward/quantile_reward': disc_q_reward,
                    'quantile/q_est': self.q_est.item(),
                    'constraint/margin': constraint_margin
                }, step=i_episode)
                
                print(f'Epi:{i_episode:05d} || disc_a_r:{disc_a_reward:.03f} '
                      f'disc_q_r:{disc_q_reward:.03f} λ:{self.lambda_dual.item():.04f}')
                print(f'Epi:{i_episode:05d} || avg_risk:{avg_risk_episode:.03f} '
                      f'constraint_margin:{constraint_margin:.03f}')
                print(f'Epi:{i_episode:05d} || lr:{self.scheduler.get_last_lr()[0]:.2e} '
                      f'q_lr:{self.q_scheduler.get_last_lr()[0]:.2e} '
                      f'λ_lr:{self.lambda_scheduler.get_last_lr()[0]:.2e} '
                      f'q_est:{self.q_est.item():.03f}\n')
            
            self.scheduler.step()
            self.q_scheduler.step()
            self.lambda_scheduler.step()

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

    def update_inner(self):
        """
        内层更新: 策略参数θ 和 分位数估计Q
        
        策略梯度方向:
        D(τ, θ, Q) = (U(τ) - ν·λ·I{U(τ) ≤ Q}) Σ ∇_θ log π(a|s;θ)
        """
        self.actor.to(self.device)

        # 计算折扣奖励
        disc_reward, disc_reward_short = self.compute_discounted_epi_reward()
        
        # 计算示性函数: I{U(τ) ≤ Q}
        ind = indicator(self.q_est.detach(), disc_reward)

        # 获取旧状态和动作
        old_states = torch.stack(self.memory.states).to(self.device).detach()
        old_actions = torch.stack(self.memory.actions).to(self.device).detach()

        # 计算对数概率
        logprobs, _ = self.evaluate(old_states, old_actions)
        
        # QCPO策略梯度: U(τ) - ν·λ·I{U(τ) ≤ Q}
        constraint_term = self.nu * self.lambda_dual * ind
        gradient_weights = disc_reward - constraint_term

        # 损失函数 (负号因为我们做梯度上升)
        loss = -torch.mean(logprobs * gradient_weights)

        # 更新策略网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新分位数估计
        self.q_optimizer.zero_grad()
        self.q_est.grad = -torch.mean(
            self.q_alpha - indicator(self.q_est.detach(), disc_reward_short), 
            dim=0, keepdim=True
        )
        self.q_optimizer.step()

        self.actor.to(torch.device('cpu'))

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
