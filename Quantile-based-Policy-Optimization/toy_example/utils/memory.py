class Memory(object):
    """
    Memory
    """
    def __init__(self):
        self.actions = []      # 储存动作列表，每个动作是一个形状为(1,)的tensor
        self.states = []       # 储存状态列表，每个状态是一个形状为(10,)的tensor
        self.logprobs = []     # 储存动作的对数概率列表，每个动作的对数概率是一个形状为(1,)的tensor
        self.rewards = []      # 储存奖励列表，每个元素是一个标量，self.rewards列表长度为T，轨迹长度，每个episode后清空，用来做更新用
        self.is_terminals = [] # 储存是否终止列表，True/False
        self.values = []       # 储存状态价值列表
        self.last_values = []  # 储存最后一个状态价值列表
        self.t = []            # 储存时间步列表，每个时间步是一个int

    def clear(self):
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()
        self.values.clear()
        self.last_values.clear()
        self.t.clear()

    def get_len(self):
        return len(self.is_terminals)