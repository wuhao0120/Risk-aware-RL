import torch.nn as nn
import torch
import numpy as np


def init_weights(module: nn.Module, gain: float = 1):
    """
    正交初始化权重
    有助于训练稳定性，防止梯度消失/爆炸
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    return module


class Actor(nn.Module):
    """
    Actor Network (策略网络)

    用于输出动作分布的均值参数
    在QPO/QCPO中，动作服从高斯分布：π(a|s) = N(μ(s), σ²)
    本网络输出均值 μ(s)，标准差 σ 作为可学习参数（默认不训练）

    支持两种架构:
    - hidden_dims=None: 线性策略 μ(s) = W @ s  (原始行为, 3个参数)
    - hidden_dims=[64,64]: MLP策略 μ(s) = W₂·tanh(W₁·s + b₁) + b₂
    """
    def __init__(self, state_dim, action_dim, init_std=1.0, hidden_dims=None):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            init_std: 初始标准差（用于探索）
            hidden_dims: 隐藏层维度列表, 如 [64, 64]。None 则用线性策略
        """
        super(Actor, self).__init__()

        self.model = nn.Sequential()

        if hidden_dims is None or len(hidden_dims) == 0:
            # 线性策略: μ(s) = W @ s (向后兼容)
            self.model.add_module('fc0', init_weights(nn.Linear(state_dim, action_dim, bias=False)))
        else:
            # MLP策略: state_dim → hidden[0] → ... → hidden[-1] → action_dim
            layer_dims = [state_dim] + list(hidden_dims) + [action_dim]
            for i in range(len(layer_dims) - 1):
                is_last = (i == len(layer_dims) - 2)
                self.model.add_module(
                    f'fc{i}',
                    init_weights(nn.Linear(layer_dims[i], layer_dims[i + 1], bias=True))
                )
                if not is_last:
                    self.model.add_module(f'act{i}', nn.Tanh())

        # 对数标准差作为可学习参数（默认不训练）
        self.log_std = nn.Parameter(torch.full((action_dim,), np.log(init_std)))
        self.log_std.requires_grad = False

    def forward(self, x):
        """
        前向传播

        Args:
            x: 状态输入 [batch_size, state_dim] 或 [state_dim]

        Returns:
            动作均值 [batch_size, action_dim] 或 [action_dim]
        """
        return self.model(x)


class Critic(nn.Module):
    """
    Critic Network (价值网络)
    
    用于估计状态价值函数 V(s)
    在基本的QPO中不需要，但在QPPO等算法中使用
    """
    def __init__(self, state_dim, emb_dim):
        """
        Args:
            state_dim: 状态维度
            emb_dim: 隐藏层维度列表，如 [64, 64]
        """
        super(Critic, self).__init__()
        
        # 构建网络层维度: [state_dim, *emb_dim, 1]
        self.model_size = [state_dim] + emb_dim + [1]
        self.model = nn.Sequential()
        
        # 构建多层神经网络
        for i in range(len(self.model_size) - 2):
            self.model.add_module(
                f'fc{i}', 
                init_weights(nn.Linear(self.model_size[i], self.model_size[i + 1], bias=True))
            )
            self.model.add_module(f'act{i}', nn.Tanh())
        
        # 输出层
        self.model.add_module(
            f'fc{len(self.model_size) - 1}', 
            init_weights(nn.Linear(self.model_size[-2], self.model_size[-1], bias=True))
        )

    def forward(self, x): 
        """
        前向传播
        
        Args:
            x: 状态输入
            
        Returns:
            状态价值估计 V(s)
        """
        return self.model(x)
