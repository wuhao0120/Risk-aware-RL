import numpy as np
import torch
from torch.distributions import MultivariateNormal
from torch.optim import Adam
import wandb
from torch.optim.lr_scheduler import LambdaLR # 导入学习率调度器
# from torch.utils.tensorboard import SummaryWriter # 导入tensorboard

from utils import Memory, Actor  # Memory类用于储存一整个trajectory的状态,动作和奖励等


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
                                                                         
                                                                         


class QPO(object):  # QPO类封装了QPO算法,负责管理整个训练过程
                    # 负责管理如何从环境走一条轨迹, train(), choose_action()
                    # 负责管理如何计算分位数回报, compute_discounted_epi_reward()
                    # 如何根据回报来更新Actor和分位数估计, update()
    def __init__(self, args, env):
        self.device = args.device              # 设置设备CPU/GPU和路径
        # self.path = args.path
        self.log_interval = args.log_interval  # 每隔多少个episode打印/记录一次统计信息
        self.est_interval = args.est_interval  # 做滚动估计时,用最近多少个episode的平均值作为估计值
        self.q_alpha = args.q_alpha            # 分位数水平
        self.gamma = args.gamma                # 折扣因子
        self.max_episode = args.max_episode    # 最大训练轮数

        # 初始化环境
        self.env = env                         # 算法要与环境交互产生trajectory,所以需要环境对象
                                               # 在train.py训练开始时创建env对象,并作为参数在实例化时传入给QPO类
        self.env_name = args.env_name          # 环境名称

        # 创建策略网络（Actor）
        state_dim = np.prod(self.env.observation_space.shape)     # 状态空间维度, np.prod计算数组元素的乘积
                                                                  # env.observation_space是一个Box空间,形状为(n,), 即10维向量, state_dim=10
                                                                  # self.env.observation_space是传入的env对象的属性, env对象是ToyEnv类的实例, ToyEnv类在toy_env.py当中定义

        action_dim = np.prod(self.env.action_space.shape)         # 动作空间维度, np.prod计算数组元素的乘积
                                                                  # env.action_space是一个Box空间,形状为(1,), 即1维向量, action_dim=1
                                                                  # self.env.action_space是传入的env对象的属性, env对象是ToyEnv类的实例, ToyEnv类在toy_env.py当中定义

        # Actor网络输出动作高斯分布的均值, std固定
        self.actor = Actor(state_dim, action_dim, args.init_std)  # 创建Actor网络, Actor函数在model.py当中定义, 输入参数自动定义一个网络
                                                                  # 输入state_dim=10, action_dim=1, agrs.init_std由参数指定, 默认1e-1
                                                                  # 返回一个Actor网络实例, 包含一个线性层, 权重矩阵形状(1,10), 无偏置

        # 设置优化器和学习率调度器
        self.optimizer = Adam(self.actor.parameters(), 1., eps=1e-5)  # 创建Adam优化器,self.actor.parameters()是nn.Module类提供的方法
                                                                      # self.actor.parameters()返回一个生成器,里面是Actor的所有可训练参数, 这里只有fc0.weight, shape=(1,10) 
                                                                      # Adam优化器默认学习率设置为1.0, 实际有效学习率要乘上LambdaLR调度器给的系数

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda k: lr_lambda(k, args.theta_a, args.theta_b, args.theta_c))  # 学习率调度器
        self.MSELoss = torch.nn.MSELoss()  # 没用到

        # 创建经验回放内存和tensorboard记录器
        self.memory = Memory()  # 在这里实例化一个Memory对象, 作为self的memory属性, actions, states, logprobs, rewards, is_terminals等列表,用来储存trajectory
        
        # self.writer = SummaryWriter(log_dir=args.path)
        wandb.init(project=args.env_name, name=f"{args.algo_name}_{args.seed}", config=vars(args), reinit=True, group=args.algo_name)

        # warm up 估计初始Q值
        q = self.warm_up(5*self.est_interval)  # warm_up方法返回一个标量q,作为初始的分位数估计
        self.q_est = torch.tensor([q], dtype=torch.float32, device=self.device, requires_grad=True)                      
        #self.q_est = torch.autograd.Variable(q*torch.ones((1,))).to(self.device) 
        # 创建一个一维张量,元素为q,移动到指定设备并包装自动求导变量self.q_est
        # 创建一个指定形状的张量元素为q,移动到指定设备并包装自动求导变量self.q_est
        # 在pytorch版本大于等于0.4之后,autograd.Variable被弃用,和torch.tensor.requires_grad=True相同效果
        # 等价的命令是self.q_est = torch.tensor([q], dtype=torch.float32, device=self.devide, requires_grad=True)
        
        # 设置分位数估计器的优化器和学习率调度器
        self.q_optimizer = Adam([self.q_est], 1., eps=1e-5)  # 单独用一个Adam优化器和调度器来更新q_est,和actor的参数分开
        self.q_scheduler = LambdaLR(self.q_optimizer, lr_lambda=lambda k: lr_lambda(k, args.q_a, args.q_b, args.q_c))

    def warm_up(self, max_episode): 
        '''
        输入一个整数max_episode, 表示要跑多少个episode估计初始Q值
        输出q, 标量float, 作为初始分位数估计
        策略不动, 重复初始化环境跑episode

        **用Monte Carlo方法估计初始Actor权重下的Q值**
        '''
        disc_epi_rewards = []  # 用来储存每个episode的cumulative return
        for _ in range(max_episode):
            # 初始化回合
            disc_epi_reward, disc_factor, state = 0, 1, self.env.reset()  # 初始化episode,cumulative return为0,discount factor为1,状态由env.reset()方法给出,
                                                                          # self.env.reset()是ToyEnv类的方法,返回初始状态,形状为(10,)的one-hot张量
            while True:
                state = state.flatten()                         # state.flatten()是一种更稳健的写法,确保输入进choose_action()的总是一个一维向量,虽然这里state本来就是长度为10的向量了     
                                                                # 形如array([0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])

                action = self.choose_action(state)              # action是一个numpy数组,形状为(1,)
                                                                # 形如array([0.60672134], dtype=float32)

                state, reward, done, _ = self.env.step(action)  # 返回新state,reward是一个标量,一个数,done是bool类型
                disc_epi_reward += disc_factor*reward           # 从r0开始累加
                disc_factor *= self.gamma                       # 之后折扣因子开始变成gamma, gamma^2...
                if done:
                    break
            disc_epi_rewards.append(disc_epi_reward)            # disc_epi_reward是一个一维的numpy数组,只用来记录日志,不参与到梯度计算
                                                                # 一个episode结束之后,把cumulative return放到disc_epi_rewards列表中
                                                                # 最终的disc_epi_rewards列表形状为(max_episode,)
        
        q = np.percentile(disc_epi_rewards, self.q_alpha*100)   # 用Monte Carlo方法估计初始Actor权重下的分位数
        print(f'QPO warm up || n_epi:{max_episode:04d} {self.q_alpha:.2f}-quantile:{q:.3f}')
        self.memory.clear()
        return q

    def train(self):
        disc_epi_rewards = []       # 创建一个空列表,储存每个episode的cumulative return
                                    # memory.rewards列表储存一个episode当中每个step的reward
                                    # disc_epi_rewards在后面用最近est_interval个episode的平均值估计平均奖励和分位数奖励
                                    # 只是用来记录日志,不参与梯度计算
                                                                         
        for i_episode in range(self.max_episode+1):
            disc_epi_reward, disc_factor, state = 0, 1, self.env.reset()
            while True:
                action = self.choose_action(state)              # choose_action()返回一个numpy数组,并且将state和采样到的action存到对应的memory列表当中 
                state, reward, done, _ = self.env.step(action)  # step()方法返回新state,形状为(10,)的numpy数组,reward是一个标量,done是bool类型
                disc_epi_reward += disc_factor * reward
                disc_factor *= self.gamma
                self.memory.rewards.append(reward)              # self.memory.reward列表收集一个episode当中每个step的reward
                self.memory.is_terminals.append(done)           # self.memory.is_terminals列表收集一个episode当中每个step的done
                if done:
                    break
            self.update()                                       # 一个episode结束之后,用整条trajectory数据做一次更新
            self.memory.clear()                                 # 更新之后,将这一个episode的memory清空,准备下一个episode

            disc_epi_rewards.append(disc_epi_reward)            # 每个episode结束之后,dis_epi_reward变量是一个一维numpy数组,储存这个episode的cumulative return,存到disc_epi_rewards列表中
                                                                # 最终的disc_epi_rewards列表形状为(max_episode,)
                                                                # 用来记录日志,不参与梯度计算

            wandb.log({'disc_reward/raw_reward': disc_epi_reward}, step=i_episode) # 记录日志,每个episode的cumulative return

            if i_episode % self.log_interval == 0 and i_episode != 0:
                lb = max(0,len(disc_epi_rewards)-self.est_interval)
                disc_a_reward, disc_q_reward = np.mean(disc_epi_rewards[lb:]), np.percentile(disc_epi_rewards[lb:], self.q_alpha*100)  # 用最近est_interval个episode的数据做mean和quantile的Monte Carlo估计
                '''
                Action从高斯分布当中采样, 均值和方差由Actor给出, 在toy_env中, 固定Actor.log_std属性, 且设置require_grad=False, 方差的梯度不回传, 相当于固定方差
                mean是Actor网络的输出, 形状为(1,), std是固定值1, 因此action的分布是均值为mean, 方差为1的高斯分布

                Reward从高斯分布中采样, 均值为0, 方差为std=(action[0]-1)**2+0.1
                为了让reward的0.25分位数最大, 应该使reward分布的方差尽可能小, 也就是action[0]=1, 那就要让action分布的均值为1, 也就是最优策略应该使action[0]=1, 无论当前的状态
                '''
                # 最优策略应该使action[0]=1, 此时reward的std=0.1最小
                # Actor输入状态向量(长度为n, one-hot编码), 经过一个线性层输出一个动作均值mean向量(长度为1), 权重矩阵形状(1,n), 无偏置
                # Actor只有一个线性层, 权重矩阵为self.actor.model[0].weight, 形状(1,n), 放到cpu上后squeeze(0)去掉第一个维度, 得到形状(n,)的向量
                # 状态是one-hot向量,输出是one-hot向量与权重向量做内积,为了让无论什么状态,Actor都能输出1,weight应该为全1向量
                error_w = np.mean((self.actor.model[0].weight.data.cpu().numpy().squeeze(0)-np.ones((self.env.n,)))**2)
                # self.writer.add_scalar('error/weights', error_w, i_episode) # 记录权重误差,越小越好 
                # self.writer.add_scalar('disc_reward/aver_reward', disc_a_reward, i_episode) # 记录平均奖励,越大越好
                # self.writer.add_scalar('disc_reward/quantile_reward', disc_q_reward, i_episode) # 记录分位数奖励,越大越好
                wandb.log({
                    'error/weights': error_w,
                    'disc_reward/aver_reward': disc_a_reward,      # 最近100个episode做均值的Monte Carlo估计
                    'disc_reward/quantile_reward': disc_q_reward,  # 最近100个episode做quantile的Monte Carlo估计
                    'quantile/q_est': self.q_est.item()            # 记录当前的quantile估计值
                }, step=i_episode)
                print(f'Epi:{i_episode:05d} || disc_a_r:{disc_a_reward:.03f} disc_q_r:{disc_q_reward:.03f}')
                print(f'Epi:{i_episode:05d} || model Updated with lr:{self.scheduler.get_last_lr()[0]:.2e} '
                      + f'q_lr:{self.q_scheduler.get_last_lr()[0]:.2e}\n')
            self.scheduler.step()    # 更新actor的学习率
            self.q_scheduler.step()  # 更新q_est的学习率

    def choose_action(self, state):
        """
        给定状态,从策略中采样动作,动作从高斯分布中采样
        创建多元正态分布, 均值和方差由当前状态决定
        根据当前状态从多元正态分布中采样动作
        """
        state = torch.from_numpy(state).float()  # torch.from_numpy(state)创建一个与state(np.array)共享内存的张量,然后转换为float32类型,大部分神经网络默认使用的精度
                                                 # 而torch.tensor(state)创建一个新张量,不共享内存
                                                 # state是numpy数组,本来是int,转化为float类型

        mean = self.actor(state)                           # mean是Actor网络的输出,形状为(1,)的tensor
                                                           # 形如tensor([0.4515], grad_fn=<SqueezeBackward4>)

        var = torch.diag(torch.exp(2*self.actor.log_std))  # self.actor.log_std是一个形状为(1,)的tensor
                                                           # torch.diag把这个tensor放到一个矩阵的对角线上,创建一个对角阵,形状为(1,1)
                                                           # 形如tensor([[0.1000]])

        dist = MultivariateNormal(mean, var)      # 创建一个多元正态分布示例,在这里就是一维正态
        action = dist.sample()                    # 从分布中采样动作,返回一个形状为(1,)的tensor
                                                  # 形如tensor([0.1416])
                                                
        self.memory.states.append(state)          # 保存状态到self.memory.states列表中
        self.memory.actions.append(action)        # 保存动作到self.memory.actions列表中
        return action.detach().data.cpu().numpy() # action是一个pytorch张量, detach将其计算图中分离, 忽略梯度信息保存为新tensor, data获取张量数据
                                                  # cpu将其移动到cpu上, numpy将其转化为numpy数组便于交互
                                                  
    # def select_action(self, state):
    #     """
    #     用于评估的动作选择，不存储memory
    #     """
    #     state = torch.from_numpy(state).float()
    #     mean = self.actor(state)
    #     var = torch.diag(torch.exp(2 * self.actor.log_std))
    #     dist = MultivariateNormal(mean, var)
    #     action = dist.sample()
    #     return action.detach().data.cpu().numpy()

    def evaluate(self, state, action):
        """
        计算动作的对数概率和熵
        """
        mean = self.actor(state)                           # 会自动调用Actor网络的forward方法,走一遍前向传播,mean是形状为(1,)的tensor

        var = torch.diag(torch.exp(2*self.actor.log_std))  # torch.exp(2*self.actor.log_std)算方差
                                                           # torch.diag()创建对角阵,但这里是1*1矩阵
        dist = MultivariateNormal(mean, var)
        action_logprobs = dist.log_prob(action)            # dist是一个多元正态分布实例, 该实例提供了对数概率方法,返回一个0维张量
                                                           # 形如tensor(-0.2479, grad_fn=<SubBackward0>) 维度为0,是一个标量
                                                           # tensor([-0.2479]) 维度为1,是一个向量
        dist_entropy = dist.entropy()                      # 也提供了计算熵的方法, 熵用来衡量策略的不确定性程度,在这里没用到
        return action_logprobs, dist_entropy

    def compute_discounted_epi_reward(self):
        """
        代码写的比较通用,支持多个episode拼接,用一批trajectory做多次epoch更新
        算法从后往前扫memory,用is_terminal识别多个episode的边界,  
        代码从后往前扫,每识别到一个is.terminal,就把改episode内所有时间步对应的disc_reward位置元素赋值为这条trajectory的cumulative return
        在QPO算法当中, train()每个episode后就调用一次update再memory.clear(),实际上每次memory里只有一个episode
        但代码可以无缝迁移到ppo, qppo等算法中,写成了高级版本,这里直接用过来

        前面train()里面的disc_epi_rewards列表,每个元素是一个一维numpy数组,只用来记录日志,打印输出
        参与梯度计算所需的cumulative reward用compute_discounted_epi_reward()函数计算并输出为所需要的tensor形状
        """
        memory_len = self.memory.get_len()                 # 返回当前这批trajectory的timestep总数量
                                                           # 如果一次只跑一个episode, 那就是episode长度
                                                           # 如果打算多条轨迹拼在一起,那就是多条轨迹的timestep总数量,代码支持多episode拼接
                                                           # 在PPO等算法中,会先收集一批trajectory, 可能会包含多条episode,全部塞进一个memory,再统一调用一次update,用一批轨迹做多次epoch更新
                                                           
                                                           # 如果有多条episode
                                                           # self.memory.rewards = [r0, r1, r2, r3, r4]; 
                                                           # self.memory.is_terminals = [False, False, True, False, True]
                                                           # disc_reward长度为5,储存每个timestep所在trajectory的cumulative reward, 形如[G1, G1, G1, G2, G2]
                                                           # disc_reward_short长度为2,储存每个episode的cumulative return, 形如[G1, G2]

        disc_reward, disc_reward_short = np.zeros(memory_len, dtype=float), []  # disc_reward是一个numpy数组,形状为(memory_len,), 储存每个timestep的对应episode的cumulative return
                                                                                # disc_reward_short是一个列表,长度等于episode数量,元素是每一个episode的cumulative return,这里只有一个episode,所以长度为1
        pre_r_sum, p1, p2 = 0, 0, 0
        for i in range(memory_len - 1, -1, -1):                                     # ====================================================
            if self.memory.is_terminals[i]:                                         # ====================================================       
                if p1 > 0:                                                          # ====================================================        
                    disc_reward[memory_len-p1: memory_len-p2] += pre_r_sum          # ====================================================
                    disc_reward_short.insert(0, pre_r_sum)                          # ====================================================
                pre_r_sum, p2 = 0, p1                                               # 计算cumulative return，并根据公式需要输出成需要的形状
            pre_r_sum = self.memory.rewards[i] + self.gamma * pre_r_sum             # ====================================================
            p1 += 1                                                                 # ====================================================
        disc_reward[memory_len-p1: memory_len-p2] += pre_r_sum                      # ====================================================
        disc_reward_short.insert(0, pre_r_sum)                                      # ====================================================

        disc_reward = torch.from_numpy(disc_reward).to(self.device).float()
        disc_reward_short = torch.tensor(disc_reward_short).to(self.device).float()

        return disc_reward, disc_reward_short                                       # disc_reward, shape (memory_len,), 储存每个timestep所在的trajectory的cumulative return
                                                                                    # disc_reward_short, shape (n_episodes,), 元素是每个episode的cumulative return

    def update(self):
        """
        更新策略网络和分位数估计
        """
        self.actor.to(self.device)

        # 计算折扣奖励
        disc_reward, disc_reward_short = self.compute_discounted_epi_reward()
        # 计算示性函数的值
        ind = indicator(self.q_est.detach(), disc_reward)  # disc_reward是一个张量,形状为(T,)
                                                           # self.q_est是一个张量,形状为(1,),会自动广播成(T,), 具体看indicator注释
                                                           # 返回ind形状为(T,), 与disc_reward形状相同,元素为1或0,表示disc_reward是否小于等于q_est
                                                           # 在这里由于disc_reward的值就是cumulative reward, 所以实际上输出全1或者全0向量, 决定这条trajectory是否被用来更新策略网络

        # 获取旧状态和动作
        old_states = torch.stack(self.memory.states).to(self.device).detach()    # 将self.memory.states列表中的所有tensor堆叠成一个形状为(T,10)的张量
        old_actions = torch.stack(self.memory.actions).to(self.device).detach()  # 将self.memory.actions列表中的所有tensor堆叠成一个形状为(T,1)的张量

        # 计算动作的对数概率和熵
        logprobs, dist_entropy = self.evaluate(old_states, old_actions)          # 计算对数概率和熵，返回logprobs是一个tensor标量
        # 计算策略损失
        loss = torch.mean(logprobs * ind)                                        # 计算对数概率和示性函数乘积，这里还没有求导，loss是一个包含梯度信息的标量张量
                                                                                 # 文章中是求和，这里求平均也无所谓，前面的1/T会被吸收到学习率当中去
                                                                                 # 这里是不带负号的，文章中的D是负梯度方向，用的是+grad更新的写法，等价于这里不带负号用Adam做梯度下降-grad更新

        # 更新策略网络
        self.optimizer.zero_grad()  # 将Actor所有参数的梯度清零, PyTorch中梯度会累积, 如果不清零, 新一轮的更新会叠加到之前的梯度上
                                    # 通常在每次参数更新前调用
        loss.backward()             # 计算对数概率和示性函数乘积对Actor所有参数的梯度(实际上就是对theta的梯度，就是文章中的-D)
        self.optimizer.step()       # 根据每一个参数的梯度执行参数更新

        # 更新分位数估计
        self.q_optimizer.zero_grad()
        self.q_est.grad = -torch.mean(self.q_alpha - indicator(self.q_est, disc_reward_short), dim=0, keepdim=True)  # 手动计算分位数的梯度，加上负号是因为要做梯度上升，Adam默认做梯度下降，朝梯度方向更新，所以加上负号朝梯度反方向做梯度上升
        self.q_optimizer.step()

        self.actor.to(torch.device('cpu'))
