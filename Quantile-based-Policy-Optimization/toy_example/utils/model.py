import torch.nn as nn
import torch
import numpy as np


def init_weights(module: nn.Module, gain: float = 1):
    """
    Orthogonal Initialization
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)  # 把权重矩阵初始化为正交矩阵，有助于训练稳定
        if module.bias is not None:
            module.bias.data.fill_(0.0)                # 偏置初始化为0
    return module                                      # 输入一个定义好的模块，返回权重初始化之后的模块


class Actor(nn.Module): 
    # nn.Module是所有神经网络模型的基类，任何自定义网络都继承它
    # 继承它之后，可以在函数里定义子模块，nn.Linear, nn.Conv2d, nn.ReLU, nn.Tanh, nn.Sigmoid, nn.Softmax等
    # 可以重写forward函数来定义前向传播过程，前向传播就是给输入x，输出y的过程
    # 可以使用model.parameters()方法来获取模型所有需要训练的参数，返回一个生成器，生成器是一个迭代器，每次迭代返回一个参数
    # 一般用for param in model.parameters()来遍历所有参数，model.parameters()返回一个迭代器/生成器，里面依次给出所有nn.Parameter且requires_grad=True的张量
    # 会递归地逐级往下给出所有子模块的参数，例如self.actor里有self.model（一个nn.Sequential容器），self.model里有fc0（一个nn.Linear层），fc0里有weight和bias两个参数
    
    # 关联到qpo.py当中的48-50行，创建self.optimizer时，会自动把所有需要训练的参数，比如fc0.weight和fc0.bias都交给Adam优化器管理
    # 虽然self.actor.log_std是nn.Parameter，但是设置了requires_grad=False，所以不会被Adam优化器更新
    # 每次loss.backward()后，调用optimizer.step()时，会自动更新这些参数

    # 通过self.fc0.weight是一个torch.nn.Parameter对象，Parameter是Tensor的子类，但是多了requires_grad属性
    # 常用的属性和方法有
    # .data: 返回底层tensor数据，仍是一个tensor，忽略梯度信息，不推荐直接修改.data
    # .grad: 获取tensor的梯度，默认为None，在backward()后才有值
    # .requires_grad: 获取tensor是否需要梯度，默认为True，可以直接write
    # .detach(): 从计算图中分离，忽略梯度信息，返回一个新的tensor，不共享内存
    # .detach.numpy(): 从计算图中分离，返回一个numpy数组，忽略梯度信息
    # .mean(), .sum(), .reshape(), 
    # .t()二维转置,.transpose()交换两个维度，但返回的tensor可能不是内存连续的需要使用.contiguous()转为内存连续的tensor，.permute()任意维度重排
    # .unsqueeze()在指定位置添加维度, .squeeze()移除所有大小为1的维度或者指定维度, .repeat(), .expand(), .roll()等方法
    """
    Actor Network
    策略网络（演员网络）
    在QPO当中用于生成动作的高斯分布的均值
    """
    def __init__(self, state_dim, action_dim, init_std=1e0):
        super(Actor, self).__init__()  # super是用来调用父类的方法，Actor是子类，self是子类实例，必须调用，否则parameters()等方法无法正常工作
                                       # 在python3中等价的写法是super().__init__()

        self.model = nn.Sequential()  # nn.Sequantial会创建一个把多个层按顺序串起来的“容器”，它本身也是一个nn.Module，forward过程会顺序调用内部的子模块
                                      # 此处先创建一个空的Sequential容器，然后往里面添加子模块

        self.model.add_module(f'fc0', init_weights(nn.Linear(state_dim, action_dim, bias=False)))  # 添加一个名字叫做'fc0'的线性层，权重初始化使用自定义的init_weights函数
                                                                                                   # 输入维度为state_dim，输出维度为action_dim，无偏置

        self.log_std = nn.Parameter(torch.full((action_dim,), np.log(init_std)))  # torch.full(size, fill_value)，生成一个全为fill_value的tensor，形状为size: tuple of ints (eg.(2,))
                                                                                  # nn.Parameter把任意一个torch.tensor包装成“可训练参数”
                                                                                  # 自动被添加到model.parameters()中，被优化器自动更新
                                                                                  # 创建一个形状为(action_dim,)，值为log(init_std)的tensor，并注册为一个模型参数，命名为log_std
                                                                                  # 于是在创建Actor实例时，这个Actor网络就拥有一个可训练的log_std参数，用于存储动作分布的标准差
        self.log_std.requires_grad = False  # 固定标准差，不进行训练，默认nn.Parameters.requires_grad=True
                                            # 虽然这是个参数，但是我们不希望在训练中更新它（固定标准差）

    def forward(self, x):  # 所有nn.Module子类都要实现forward函数，是必要的，它定义了模型输入x时如何计算输出
        return self.model(x)  # forward函数定义了模型的前向传播过程，输入x，输出self.model(x)，也就是走一遍nn.Sequential()之后的结果，这里只有一个线性层，所以输出就是线性层的输出


class Critic(nn.Module):
    """
    Critic Network
    在QPO当中没用，在QPPO等算法中用于估计状态的价值
    """
    def __init__(self, state_dim, emb_dim):
        super(Critic, self).__init__()
        # 网络层的维度
        self.model_size = [state_dim] + emb_dim + [1]
        self.model = nn.Sequential()
        
        # 构建多层神经网络
        for i in range(len(self.model_size) - 2):
            self.model.add_module(f'fc{i}', init_weights(nn.Linear(self.model_size[i], self.model_size[i + 1], bias=True)))
            # 使用tanh激活函数
            self.model.add_module(f'act{i}', nn.Tanh())
        # 添加输出层
        self.model.add_module(f'fc{len(self.model_size) - 1}', init_weights(nn.Linear(self.model_size[-2], self.model_size[-1], bias=True)))

    def forward(self, x): 
        return self.model(x) # 返回状态价值估计


