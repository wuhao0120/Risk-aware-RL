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


class QPO(object):
    """
    QPO (Quantile Policy Optimization)
    
    目标: 最大化累积回报的α分位数
    
    核心算法:
    1. 使用示性函数 I{U(τ) <= Q} 作为梯度权重
    2. 同时更新策略参数θ和分位数估计Q
    """
    
    def __init__(self, args, env):
        self.device = args.device
        self.log_interval = args.log_interval
        self.est_interval = args.est_interval
        self.q_alpha = args.q_alpha
        self.gamma = args.gamma
        self.max_episode = args.max_episode

        # 环境
        self.env = env
        self.env_name = args.env_name

        # 策略网络
        state_dim = np.prod(self.env.observation_space.shape)
        action_dim = np.prod(self.env.action_space.shape)
        actor_hidden = getattr(args, 'actor_hidden_dims', None)
        self.actor = Actor(state_dim, action_dim, args.init_std, hidden_dims=actor_hidden).to(self.device)

        # 优化器和学习率调度器
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
            id=wandb.util.generate_id(),
            group=args.algo_name,
            dir=getattr(args, 'wandb_dir', None)
        )

        # 预热估计初始分位数
        q = self.warm_up(5 * self.est_interval)
        self.q_est = torch.tensor([q], dtype=torch.float32, device=self.device, requires_grad=True)
        
        # 分位数估计的优化器
        self.q_optimizer = Adam([self.q_est], 1., eps=1e-5)
        self.q_scheduler = LambdaLR(
            self.q_optimizer, 
            lr_lambda=lambda k: lr_lambda(k, args.q_a, args.q_b, args.q_c)
        )

    def warm_up(self, max_episode): 
        """
        预热阶段：用当前策略采样估计初始分位数
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
        print(f'QPO warm up || n_epi:{max_episode:04d} {self.q_alpha:.2f}-quantile:{q:.3f}')
        self.memory.clear()
        return q

    def train(self):
        """主训练循环"""
        disc_epi_rewards = []
                                                                         
        for i_episode in range(self.max_episode + 1):
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
            regime_stats = self._compute_regime_stats()
            self.update()
            self.memory.clear()

            disc_epi_rewards.append(disc_epi_reward)

            log_dict = {
                'disc_reward/raw_reward': disc_epi_reward,
                'action/avg_risk_episode': avg_risk_episode
            }
            log_dict.update(regime_stats)

            if i_episode % self.log_interval == 0 and i_episode != 0:
                lb = max(0, len(disc_epi_rewards) - self.est_interval)
                disc_a_reward = np.mean(disc_epi_rewards[lb:])
                disc_q_reward = np.percentile(disc_epi_rewards[lb:], self.q_alpha * 100)
                
                log_dict.update({
                    'disc_reward/aver_reward': disc_a_reward,
                    'disc_reward/quantile_reward': disc_q_reward,
                    'quantile/q_est': self.q_est.item()
                })
                
                print(f'Epi:{i_episode:05d} || disc_a_r:{disc_a_reward:.03f} disc_q_r:{disc_q_reward:.03f}')
                print(f'Epi:{i_episode:05d} || avg_risk:{avg_risk_episode:.03f}')
                print(f'Epi:{i_episode:05d} || lr:{self.scheduler.get_last_lr()[0]:.2e} '
                      f'q_lr:{self.q_scheduler.get_last_lr()[0]:.2e} '
                      f'q_est:{self.q_est.item():.03f}\n')

            wandb.log(log_dict, step=i_episode)
                      
            self.scheduler.step()
            self.q_scheduler.step()

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

        disc_reward, disc_reward_short = np.zeros(memory_len, dtype=float), []
        pre_r_sum, p1, p2 = 0, 0, 0
        
        for i in range(memory_len - 1, -1, -1):
            if self.memory.is_terminals[i]:
                if p1 > 0:
                    disc_reward[memory_len-p1: memory_len-p2] += pre_r_sum
                    disc_reward_short.insert(0, pre_r_sum)
                pre_r_sum, p2 = 0, p1
            pre_r_sum = self.memory.rewards[i] + self.gamma * pre_r_sum
            p1 += 1
            
        disc_reward[memory_len-p1: memory_len-p2] += pre_r_sum
        disc_reward_short.insert(0, pre_r_sum)

        disc_reward = torch.from_numpy(disc_reward).to(self.device).float()
        disc_reward_short = torch.tensor(disc_reward_short).to(self.device).float()

        return disc_reward, disc_reward_short

    def update(self):
        """更新策略网络和分位数估计"""
        # 计算折扣奖励
        disc_reward, disc_reward_short = self.compute_discounted_epi_reward()

        # 计算示性函数
        ind = indicator(self.q_est.detach(), disc_reward)

        # 获取旧状态和动作 (已在device上)
        old_states = torch.stack(self.memory.states).detach()
        old_actions = torch.stack(self.memory.actions).detach()

        # 计算对数概率
        logprobs, _ = self.evaluate(old_states, old_actions)

        # QPO策略损失: 最大化分位数
        loss = torch.mean(logprobs * ind)

        # 更新策略网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新分位数估计
        self.q_optimizer.zero_grad()
        self.q_est.grad = -torch.mean(
            self.q_alpha - indicator(self.q_est, disc_reward_short),
            dim=0, keepdim=True
        )
        self.q_optimizer.step()

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
        """计算当前episode的per-regime统计, 用于诊断训练动态"""
        stats = {}
        if not hasattr(self.env, 'get_regime_history'):
            return stats

        regime_hist = self.env.get_regime_history()     # [T+1] 体制序列
        risk_series = self.env.render()                 # [T]   动作序列
        if regime_hist is None or risk_series is None:
            return stats
        if len(risk_series) == 0:
            return stats

        # regime_hist[0] 是初始体制, regime_hist[t+1] 是 step t 之后的体制
        # risk_series[t] 是 step t 的动作
        # step t 的奖励由 regime_hist[t] (旧体制) 决定
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
