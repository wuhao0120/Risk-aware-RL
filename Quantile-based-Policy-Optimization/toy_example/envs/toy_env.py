import gymnasium as gym
from gym import spaces
import numpy as np


class ToyEnv(gym.Env): # 继承gym.Env接口，gym.Env是Gym库中的一个基类，定义了强化学习环境的标准接口。
    """
    Toy Env
    """
    def __init__(self, n=10):
        self.n = n
        # spaces.Box(low, high, dtype=None) 定义了一个连续动作空间，low和high是动作空间的上下界，dtype是数据类型
        self.action_space = spaces.Box(-np.inf*np.ones((1,)), np.inf*np.ones((1,)), dtype=np.float32)  # 定义动作空间，动作空间是一维的，范围从-np.inf到np.inf
        
        self.observation_space = spaces.Box(np.zeros((n,)), np.ones((n,)), dtype=np.float32)  # 定义状态空间，维度为10，范围从[0]*10到[1]*10，定义了一个10维的连续状态空间
                                                                                              # 理论上状态应该是连续值，但是下面并没有用self.observation_space.sample()来采样状态
                                                                                              # 而是直接定义了状态只在10个状态之间随机转移，就是one-hot向量表示的10个状态
        # 创建一个[0,1,...,n-1]的数组
        self.order = np.arange(n)
        # 初始化步数计数和标准差缓冲区
        self.step_count = None 
        self.std_buf = None 

    def step(self, action): # 执行一步动作，返回(state: 状态，reward: 奖励，done: 是否结束，info: 额外信息)
        # 检查动作是否合法，action!r 将action转换为字符串，type(action) 返回action的类型，assert 断言，如果动作不在定义的状态空间内，则抛出异常
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        std = (action[0] - 1)**2 + 0.1
        self.std_buf.append(std) # 在__init__时初始化为None，实际先调用reset()再调用step()，在reset()方法中已经将std_buf初始化为空列表
        ############################################
        # 原文里的例子是uniform，但这里用的是normal #
        ############################################
        reward = np.random.normal(0, std) 

        #print(f"action: {action}")
        #print(f"reward: {reward}")  


        self.step_count += 1
        state = np.zeros((self.n,))
        if self.step_count==self.n:
            return state, reward, True, {} # 终止时返回0状态，最终转移步的reward，done=True，info=空字典
        else:
            # 先reset再step，在reset方法中，self.order是一个被打乱的[0,1,..,n-1]数组，
            state[self.order[self.step_count]] += 1. # 非终止返回更新后的状态，将状态的第self.order[self.step_count]位置的元素置为1，其他位置为0，one-hot编码表示当前位于self.order[self.step_count]状态
            return state, reward, False, {} 

    def reset(self): # 每一个episode开始时，重置环境。重置步数计数器和标准差缓冲区，将order数组打乱，随机选择初始状态
        self.step_count = 0
        self.std_buf = []
        np.random.shuffle(self.order)
        state = np.zeros((self.n,))
        state[self.order[self.step_count]] += 1. 
        return state

    def close(self):
        return None

    def render(self, mode=None):
        return np.array(self.std_buf)

# env = ToyEnv()
# observation = env.reset() # 重置环境，随机选择初始状态 
# while True:
#     print(observation)
#     action = env.action_space.sample() # 随机选择动作
#     print(action)
#     observation, reward, done, info = env.step(action) # 执行动作，更新状态，计算奖励，判断是否终止
#     print(reward)
#     if done:
#         break




