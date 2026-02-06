class Memory(object):
    """
    Memory: 经验回放缓冲区
    用于存储一个或多个episode的轨迹数据
    """
    def __init__(self):
        self.actions = []      # 动作列表，每个元素是一个形状为(action_dim,)的tensor
        self.states = []       # 状态列表，每个元素是一个形状为(state_dim,)的tensor
        self.logprobs = []     # 动作对数概率列表
        self.rewards = []      # 奖励列表，每个元素是一个标量
        self.is_terminals = [] # 终止标志列表，True/False
        self.values = []       # 状态价值列表（用于Actor-Critic）
        self.last_values = []  # 最后状态价值列表
        self.t = []            # 时间步列表

    def clear(self):
        """清空所有存储的数据"""
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()
        self.values.clear()
        self.last_values.clear()
        self.t.clear()

    def get_len(self):
        """返回当前存储的数据长度"""
        return len(self.is_terminals)
