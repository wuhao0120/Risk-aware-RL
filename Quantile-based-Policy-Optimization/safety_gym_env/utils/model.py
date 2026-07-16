# -*- coding: utf-8 -*-
"""
网络结构 (safety_gym_env 专用)。

与 portfolio_env_inf/utils/model.py 的差异 (源于环境从"softmax 权重"变为"连续 2D 力矩"):
  1. Actor 输出经 **tanh 压缩** 到 (-1,1): safety-gym 点机器人动作值域为 [-1,1] (力矩),
     与 portfolio 的"logits→env 内 softmax"不同。做法对齐 NIPS QcpoModel 的 mu_nonlinearity=tanh:
     μ(s)=tanh(net(s)) ∈(-1,1), 采样 a=μ+σ·ε (普通高斯噪声, 越界由 env clip 兜底)。
     ⇒ logπ 仍是【普通对角高斯】(不做 tanh 变量替换修正), 复用用户 vec_base 的 logπ/熵机制。
  2. 默认 **MLP** ([256,256] tanh): 观测 60-76 维 (lidar+传感器), 线性策略不足以表达。
  3. log_std 默认【不训练】(固定探索, 四算法/两算法公平对比, 与 risk/portfolio 系列一致);
     learn_std=True 可开 (对齐 NIPS 的可学习 log_std)。
"""
import numpy as np
import torch
import torch.nn as nn


def init_weights(module, gain=1.0):
    """正交初始化权重 + 零偏置 (与项目其它网络风格一致, 利于训练稳定)。"""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    return module


class ObservationNormalizer(nn.Module):
    """
    逐维 observation running mean/variance，与 QCPO_refs 的 RunningMeanStdModel 同公式。

    rollout 内统计保持冻结；rollout 完成后由 agent 批量调用 update()。这样采样期间的
    行为策略是固定分布，同时 actor、value 与 distributional critics 可共享同一输入尺度。
    """

    def __init__(self, state_dim, var_clip=1e-6, value_clip=10.0):
        """初始化均值=0、方差=1、样本数=0；三个量注册为 buffer，自动跟随 device。"""
        super().__init__()
        self.var_clip = float(var_clip)                         # 防止近常数维除以接近 0 的标准差
        self.value_clip = float(value_clip)                     # 对齐 QCPO_refs 的 [-10,10] 截断
        self.register_buffer('mean', torch.zeros(state_dim))
        self.register_buffer('var', torch.ones(state_dim))
        self.register_buffer('count', torch.zeros(()))

    @torch.no_grad()
    def update(self, observations):
        """把 [*,state_dim] 扁平化后，用 Chan 并行方差公式合并当前批统计。"""
        x = observations.reshape(-1, self.mean.numel())
        batch_mean = x.mean(dim=0)                              # 当前 rollout 各观测维均值
        batch_var = x.var(dim=0, unbiased=False)                # population variance，与 ref 一致
        batch_count = x.shape[0]

        if self.count.item() == 0:
            self.mean.copy_(batch_mean)
            self.var.copy_(batch_var)
            self.count.fill_(batch_count)
            return

        delta = batch_mean - self.mean
        total = self.count + batch_count
        m_a = self.var * self.count                             # 历史平方离差和
        m_b = batch_var * batch_count                           # 当前批平方离差和
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total
        self.mean.copy_(self.mean + delta * batch_count / total)
        self.var.copy_(m2 / total)
        self.count.copy_(total)

    def forward(self, observations):
        """返回 clip((obs-mean)/sqrt(var), -value_clip, value_clip)，不修改统计。"""
        variance = self.var.clamp_min(self.var_clip)
        normalized = (observations - self.mean) / variance.sqrt()
        return torch.clamp(normalized, -self.value_clip, self.value_clip)


class Actor(nn.Module):
    """
    高斯策略网络: π(a|s)=N(μ(s), σ²I), 动作 ∈ (-1,1) (tanh 压缩均值)。

    μ(s): MLP(hidden, tanh) → Linear → tanh  (输出层 gain 小 → 初始 μ≈0 → 初始动作≈0);
          hidden=None 时退化为单线性层 + tanh。
    σ:    log_std 可学习参数, 默认 requires_grad=False (固定探索幅度)。
    """

    def __init__(self, state_dim, action_dim, init_std=0.5, hidden=None,
                 bias=True, learn_std=False):
        """
        Args:
            state_dim:  观测维度 (safety-gym: 60/76)
            action_dim: 动作维度 (safety-gym 点机器人: 2)
            init_std:   初始探索标准差 (动作空间 [-1,1]; 0.3~1.0 合理)
            hidden:     隐藏层维度列表 (默认 [256,256]); None → 单线性层
            bias:       是否带偏置 (默认 True)
            learn_std:  log_std 是否可训练 (默认 False, 固定探索)
        """
        super().__init__()
        if hidden is None:
            hidden = [256, 256]                             # 默认 MLP (观测高维, 需非线性)
        if len(hidden) == 0:
            # 单线性层 + tanh (退化, 一般不用)
            self.model = nn.Sequential(
                init_weights(nn.Linear(state_dim, action_dim, bias=bias), gain=0.01),
                nn.Tanh())
        else:
            # MLP: [Linear-Tanh]×L + 输出 Linear + tanh (输出 gain 小 → 初始动作居中)
            dims = [state_dim] + list(hidden)
            layers = []
            for i in range(len(dims) - 1):
                layers.append(init_weights(nn.Linear(dims[i], dims[i + 1], bias=bias)))
                layers.append(nn.Tanh())
            layers.append(init_weights(nn.Linear(dims[-1], action_dim, bias=bias), gain=0.01))
            layers.append(nn.Tanh())                        # 压缩均值到 (-1,1)
            self.model = nn.Sequential(*layers)

        # 对数标准差 (各动作维独立同 σ), 默认不训练 (固定探索, 公平对比)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(np.log(init_std))))
        self.log_std.requires_grad = bool(learn_std)

    def forward(self, x):
        """前向: x [*, state_dim] → 动作均值 μ(s)=tanh(net(x)) ∈(-1,1), [*, action_dim]。"""
        return self.model(x)
