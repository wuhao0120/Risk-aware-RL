import numpy as np
import torch
from torch.distributions import MultivariateNormal
from torch.optim import Adam
# from torch.utils.tensorboard import SummaryWriter
import wandb
from torch.optim.lr_scheduler import LambdaLR

from utils import Memory, Actor


def lr_lambda(k, a, b, c):  
    """
    给LambdaLR调度器用的学习率因子函数, 输入标量, 输出标量, 都是float/int
    计算学习率衰减, 使用公式 a/(b+k)^c,k是当前训练轮数,由LambdaLR调度器传进来,a,b,c是超参数
    返回一个标量lr,作为学习率衰减因子,乘到基础学习率上
    """                      
    lr = a / ((b + k) ** c) 
    return lr      


def indicator(x, y:torch.Tensor):
    """
    输入x一个标量或者张量, 输入y一个张量
    输出一个与y形状相同的张量, y<=x的位置是1, 否则是0
    """
    return torch.where(y <= x, torch.ones_like(y), torch.zeros_like(y))  # torch.where(condition:bool tensor, x, y),当condition为True时, 输出x, 否则输出y
                                                                         # torch.ones_like(y)创建一个与y形状相同的张量, 元素为1
                                                                         # torch.zeros_like(y)创建一个与y形状相同的张量, 元素为0
                                                                         # x可以是标量也可以是张量,y必须是张量
                                                                         # 如果x是标量,会自动广播为与y形状相同的tensor
                                                                         # 先判断y<=x, 返回一个bool tensor, 然后在bool tensor True的位置填x对应位置的元素, False的位置填y对应位置的元素


class QCPO(object):  # QPO类封装了QPO算法,负责管理整个训练过程
                     # 负责管理如何从环境走一条轨迹, train(), choose_action()
                     # 负责管理如何计算分位数回报, compute_discounted_epi_reward()
                     # 如何根据回报来更新Actor和分位数估计, update()
    def __init__(self, args, env):
        self.device = args.device               # 设置设备CPU/GPU和路径
        
        # 设置训练参数
        self.log_interval = args.log_interval   # 日志记录间隔
        self.est_interval = args.est_interval   # 估计间隔
        self.q_alpha = args.q_alpha             # 分位数水平
        self.gamma = args.gamma                 # 折扣因子
        self.max_episode = args.max_episode     # 最大训练轮数
        
        # QCPO特有参数
        self.nu = args.nu                                           # 约束项权重系数
        self.quantile_threshold = args.quantile_threshold           # 分位数阈值q，用于计算对偶函数
        self.outer_interval = getattr(args, 'outer_interval', 1)    # 外层更新间隔，默认1（即即时更新）

        # 初始化环境
        self.env = env                         # 算法要与环境交互产生trajectory,所以需要环境对象
                                               # 在train.py训练开始时创建env对象,并作为参数在实例化时传入给QPO类
        self.env_name = args.env_name          # 环境名称

        # 创建策略网络（Actor）
        state_dim = np.prod(self.env.observation_space.shape)       # 状态空间维度, np.prod计算数组元素的乘积
                                                                    # env.observation_space是一个Box空间,形状为(n,), 即10维向量, state_dim=10
                                                                    # self.env.observation_space是传入的env对象的属性, env对象是ToyEnv类的实例, ToyEnv类在toy_env.py当中定义

        action_dim = np.prod(self.env.action_space.shape)           # 动作空间维度, np.prod计算数组元素的乘积
                                                                    # env.action_space是一个Box空间,形状为(1,), 即1维向量, action_dim=1
                                                                    # self.env.action_space是传入的env对象的属性, env对象是ToyEnv类的实例, ToyEnv类在toy_env.py当中定义

        self.actor = Actor(state_dim, action_dim, args.init_std)    # 创建Actor网络, Actor函数在model.py当中定义, 输入参数自动定义一个网络
                                                                    # Actor网络输出动作高斯分布的均值, std固定

        # 设置优化器和学习率调度器
        self.optimizer = Adam(self.actor.parameters(), 1., eps=1e-5)
        self.scheduler = LambdaLR(
            self.optimizer, 
            lr_lambda=lambda k: lr_lambda(k, args.theta_a, args.theta_b, args.theta_c)
        )
        
        # 创建经验回放内存和tensorboard记录器
        self.memory = Memory()
        wandb.init(project=args.env_name, name=f"{args.algo_name}_pd_{args.seed}", config=vars(args), reinit=True, group=args.algo_name)

        # warm up 估计初始Q值
        q = self.warm_up(5 * self.est_interval)
        self.q_est = torch.autograd.Variable(q * torch.ones((1,))).to(self.device)
        
        # 设置分位数估计器的优化器和学习率调度器
        self.q_optimizer = Adam([self.q_est], 1., eps=1e-5)
        self.q_scheduler = LambdaLR(
            self.q_optimizer, 
            lr_lambda=lambda k: lr_lambda(k, args.q_a, args.q_b, args.q_c)
        )
        
        self.lambda_dual = torch.tensor([0.0], device=self.device, requires_grad=True)  # 初始化拉格朗日乘子λ
                                                                                        # 使用params而非单一tensor，以便优化器管理
        
                                                                                        
                                                                                         
        lambda_a = getattr(args, 'lambda_a', getattr(args, 'q_a', 1.0))                 # 设置λ的优化器和学习率调度器
        lambda_b = getattr(args, 'lambda_b', getattr(args, 'q_b', 100.0))               # 获取调度参数，如果未指定则使用q_est的参数或默认值
        lambda_c = getattr(args, 'lambda_c', getattr(args, 'q_c', 0.5))
        
        self.lambda_optimizer = Adam([self.lambda_dual], lr=1.0, eps=1e-5) # 基础lr设为1，由scheduler控制实际lr
        self.lambda_scheduler = LambdaLR(
            self.lambda_optimizer,
            lr_lambda=lambda k: lr_lambda(k, lambda_a, lambda_b, lambda_c)
        )
        
    def warm_up(self, max_episode):
        """
        预热阶段：运行一些轨迹，估计初始的Q值
        """
        disc_epi_rewards = []
        for _ in range(max_episode):
            disc_epi_reward, disc_factor, state = 0, 1, self.env.reset()  # 初始化episode,cumulative return为0,discount factor为1,状态由env.reset()方法给出,
                                                                          # self.env.reset()是ToyEnv类的方法,返回初始状态,形状为(10,)的one-hot张量
            while True:
                state = state.flatten()                           # state.flatten()是一种更稳健的写法,确保输入进choose_action()的总是一个一维向量,虽然这里state本来就是长度为10的向量了     
                                                                  # 形如array([0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])
 
                action = self.choose_action(state)                # action是一个numpy数组,形状为(1,)
                                                                  # 形如array([0.60672134], dtype=float32)

                state, reward, done, _ = self.env.step(action)    # 返回新state,reward是一个标量,一个数,done是bool类型
                                                                  # self.env.step(action)是ToyEnv类的方法,返回新state,reward,done,其他信息
                disc_epi_reward += disc_factor * reward
                disc_factor *= self.gamma                         # 之后折扣因子开始变成gamma, gamma^2...
                if done:
                    break
            disc_epi_rewards.append(disc_epi_reward)              # disc_epi_reward是一个一维的numpy数组,只用来记录日志,不参与到梯度计算
                                                                  # 一个episode结束之后,把cumulative return放到disc_epi_rewards列表中
                                                                  # 最终的disc_epi_rewards列表形状为(max_episode,)
        
        # 计算指定分位数
        q = np.percentile(disc_epi_rewards, self.q_alpha * 100)   # 用Monte Carlo方法估计初始Actor权重下的分位数
        print(f'QCPO (Primal-Dual) warm up || n_epi:{max_episode:04d} {self.q_alpha:.2f}-quantile:{q:.3f}')
        self.memory.clear()
        return q

    def train(self):
        """主训练循环 (Primal-Dual 版本)"""
        disc_epi_rewards = []       # 创建一个空列表,储存每个episode的cumulative return
        inner_step_counter = 0      # 计数器控制outer update
                                    
        for i_episode in range(self.max_episode + 1):
            # 采样一条轨迹
            disc_epi_reward, disc_factor, state = 0, 1, self.env.reset()
            while True:
                action = self.choose_action(state)                 # choose_action()返回一个numpy数组,并且将state和采样到的action存到对应的memory列表当中  
                state, reward, done, _ = self.env.step(action)     # step()方法返回新state,形状为(10,)的numpy数组,reward是一个标量,done是bool类型
                disc_epi_reward += disc_factor * reward     
                disc_factor *= self.gamma
                self.memory.rewards.append(reward)
                self.memory.is_terminals.append(done)
                if done:
                    break
            
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
                'lambda/value': self.lambda_dual.item()
            }, step=i_episode)
            
            # 日志记录
            if i_episode % self.log_interval == 0 and i_episode != 0:
                lb = max(0, len(disc_epi_rewards) - self.est_interval)
                disc_a_reward = np.mean(disc_epi_rewards[lb:])
                disc_q_reward = np.percentile(disc_epi_rewards[lb:], self.q_alpha * 100)
                error_w = np.mean(
                    (self.actor.model[0].weight.data.cpu().numpy().squeeze(0) - 
                     np.ones((self.env.n,))) ** 2
                )

                wandb.log({
                    'error/weights': error_w,
                    'disc_reward/aver_reward': disc_a_reward,
                    'disc_reward/quantile_reward': disc_q_reward,
                    'quantile/q_est': self.q_est.item()
                }, step=i_episode)
                
                print(f'Epi:{i_episode:05d} || disc_a_r:{disc_a_reward:.03f} '
                      f'disc_q_r:{disc_q_reward:.03f} λ:{self.lambda_dual.item():.04f}')
                print(f'Epi:{i_episode:05d} || model Updated with lr:{self.scheduler.get_last_lr()[0]:.2e} '
                      f'q_lr:{self.q_scheduler.get_last_lr()[0]:.2e} λ_lr:{self.lambda_scheduler.get_last_lr()[0]:.2e} '
                      f'q_est:{self.q_est.item():.03f}\n')
            
            self.scheduler.step()
            self.q_scheduler.step()
            self.lambda_scheduler.step()

    def choose_action(self, state):
        """
        根据当前策略选择动作
        输入state一个numpy数组，形状为(10,)
        输出action一个numpy数组，形状为(1,)
        """
        state = torch.from_numpy(state).float()                  # 将state从numpy数组转换为torch张量,形状为(10,)

        mean = self.actor(state)                                 # mean是Actor网络的输出,形状为(1,)的tensor
                                                                 # 形如tensor([0.4515], grad_fn=<SqueezeBackward4>)

        var = torch.diag(torch.exp(2 * self.actor.log_std))      # self.actor.log_std是一个形状为(1,)的tensor
                                                                 # torch.diag把这个tensor放到一个矩阵的对角线上,创建一个对角阵,形状为(1,1)
                                                                 # 形如tensor([[0.1000]])

        dist = MultivariateNormal(mean, var)
        action = dist.sample()
        
        # 保存状态和动作到memory
        self.memory.states.append(state)  
        
        self.memory.actions.append(action)  
        
        
        return action.detach().data.cpu().numpy()

    def select_action(self, state):
        """
        用于评估的动作选择，不存储memory
        """
        state = torch.from_numpy(state).float()
        mean = self.actor(state)
        var = torch.diag(torch.exp(2 * self.actor.log_std))
        dist = MultivariateNormal(mean, var)
        action = dist.sample()
        return action.detach().data.cpu().numpy()
        
    def evaluate(self, state, action):
        """
        计算动作的对数概率和熵
        输入state和action都是torch张量,形状为(10,)和(1,)
        输出action_logprobs和dist_entropy都是标量张量,维度为0
        """
        mean = self.actor(state)
        var = torch.diag(torch.exp(2 * self.actor.log_std))
        dist = MultivariateNormal(mean, var)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, dist_entropy                      # 返回actoin_logprobs是一个标量张量，维度为0

    def compute_discounted_epi_reward(self):
        """
        计算折扣累积奖励
        输出disc_reward是一个张量, 形状为(memory_len,), 每个位置元素是该timestep所在trajectory的cumulative return
        disc_reward_short是一个张量, 形状为(num_episodes,), 每个位置元素是该episode的cumulative return
        """
        memory_len = self.memory.get_len()
        disc_reward = np.zeros(memory_len, dtype=float)
                                                            # 如果有多条episode
                                                            # self.memory.rewards = [r0, r1, r2, r3, r4]; 
                                                            # self.memory.is_terminals = [False, False, True, False, True]
                                                            # disc_reward长度为5,储存每个timestep所在trajectory的cumulative return, 形如[G1, G1, G1, G2, G2]
                                                            # disc_reward_short长度为2,储存每个episode的cumulative return, 形如[G1, G2]
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
        更新策略参数θ、分位数估计Q 和 拉格朗日乘子λ
        """
        self.actor.to(self.device) 

        # 计算折扣奖励
        disc_reward, disc_reward_short = self.compute_discounted_epi_reward()  
        
        # 计算示性函数的值：1{U(τ_k) ≤ Q_k}
        ind = indicator(self.q_est.detach(), disc_reward)  # 计算示性函数的值：1{U(τ_k) ≤ Q_k}
                                                           # disc_reward是一个张量,形状为(memory_len,), 每个位置元素是该timestep所在trajectory的cumulative return
                                                           # self.q_est是一个张量,形状为(1,),会自动广播成(memory_len,), 具体看indicator注释
                                                           # 返回ind形状为(memory_len,), 与disc_reward形状相同,元素为1或0,表示disc_reward是否小于等于q_est
                                                           # 在这里由于disc_reward的值就是cumulative reward, 所以实际上输出全1或者全0向量, 决定这条trajectory是否被用来更新策略网络

        # 获取旧状态和动作
        old_states = torch.stack(self.memory.states).to(self.device).detach()    # 将self.memory.states列表中的所有tensor堆叠成一个形状为(T,10)的张量
        old_actions = torch.stack(self.memory.actions).to(self.device).detach()  # 将self.memory.actions列表中的所有tensor堆叠成一个形状为(T,1)的张量

        # 计算动作的对数概率
        logprobs, dist_entropy = self.evaluate(old_states, old_actions)
        
        # 计算策略梯度方向（算法第12步）
        # D(τ_k, θ_k, Q_k) = (U(τ) - ν_t·λ·1{U(τ_k) ≤ Q_k}) Σ ∇_θ log π(a_t|s_t;θ_k)
        # 由于我们使用梯度上升，损失函数为：
        # loss = -(U(τ) - ν·λ·1{U(τ_k) ≤ Q_k}) * log_prob
        # 注意这里已经修复了符合论文的符号：使用减号
        constraint_term = self.nu * self.lambda_dual * ind  # self.nu是一个标量，self.lambda_dual是一个标量张量，ind是一个形状为(memory_len,)的张量
                                                            # 返回constraint_term形状为(memory_len,), 与ind形状相同,元素为self.nu * self.lambda_dual * ind

        gradient_weights = disc_reward - constraint_term  # disc_reward是一个形状为(memory_len,)的张量，constraint_term是一个形状为(memory_len,)的张量
                                                          # 返回gradient_weights形状为(memory_len,), 与disc_reward和constraint_term形状相同,元素为disc_reward - constraint_term

        loss = -torch.mean(logprobs * gradient_weights)   # 计算不求导的梯度，下面对loss求导之后变成文章中的梯度D，取个负号放进Adam做梯度下降，就是文章中梯度上升优化theta

        # 更新策略网络
        self.optimizer.zero_grad()  # 将Actor所有参数的梯度清零, PyTorch中梯度会累积, 如果不清零, 新一轮的更新会叠加到之前的梯度上
                                    # 通常在每次参数更新前调用
        loss.backward()             # 计算loss对Actor所有参数的梯度(实际上就是对theta的梯度，就是文章中的D)
        self.optimizer.step()       # 根据每一个参数的梯度执行参数更新

        # 更新分位数估计（算法第11步）
        # Q_{k+1} = Q_k + β_k(α - 1{U(τ_k) ≤ Q_k})
        self.q_optimizer.zero_grad()
        self.q_est.grad = -torch.mean(self.q_alpha - indicator(self.q_est.detach(), disc_reward_short), dim=0, keepdim=True)
        self.q_optimizer.step()

        self.actor.to(torch.device('cpu'))

    def update_dual(self):
        """
        更新拉格朗日乘子 λ (Dual Update)
        目标：min_λ L(λ) ≈ E[U] + λ(Q - q)  (s.t. λ >= 0)
        
        Loss = λ * (Q - q)
        Adam会自动做梯度下降： λ <- λ - lr * ∇λ = λ - lr * (Q - q)
        如果是约束满足 (Q > q)， (Q-q) > 0, λ 减小。
        如果是约束违反 (Q < q)， (Q-q) < 0, λ 增大。
        """
        self.lambda_optimizer.zero_grad()
        
        # 计算Loss
        # 使用当前的Q估计值 (detach, 视为常数)
        # 我们希望最小化 Lagragian 对 λ 的部分
        dual_loss = self.lambda_dual * (self.q_est.detach() - self.quantile_threshold)
        
        dual_loss.backward()
        self.lambda_optimizer.step()
        
        # 投影到 [0, inf)
        with torch.no_grad():
            self.lambda_dual.clamp_(min=0.0)
