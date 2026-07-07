# -*- coding: utf-8 -*-
"""
网络结构 (portfolio_env_inf 专用)。

与 risk_sensitive_env_inf/utils/model.py 的差异:
  1. Actor 增加 bias 与可选 MLP 隐藏层:
     - 原版面向"无状态 [1.0]"环境, 线性无偏置 μ(s)=W·s 即可 (W 本身充当常量 logits)。
     - 本环境观测 25 维且最优策略 ≈ 常量目标权重 (μ/Σ 恒定 ⇒ Markowitz 解与状态无关,
       仅交易成本引入轻微状态依赖) ⇒ 需要 bias 项表达"常量 logits"; est_mu/est_cov
       特征噪声大, 默认仍用线性层 (hidden=None), 需要时可开 MLP。
  2. 新增 Critic (QPPO 的 V(s) baseline 用; DQCAC 的分布式 critic 在 agents/common.py)。
"""
import numpy as np
import torch
import torch.nn as nn


def init_weights(module: nn.Module, gain: float = 1.0):
    """正交初始化权重 + 零偏置 (与项目其它网络风格一致, 利于训练稳定)。"""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    return module


class Actor(nn.Module):
    """
    高斯策略网络: π(a|s) = N(μ(s), σ²I), 输出 K 维动作 logits (env 内部 softmax → 权重)。

    μ(s): hidden=None → 线性 (含 bias); hidden=[h1,h2,...] → MLP (Tanh 激活)。
    σ:    log_std 可学习参数, 默认【不训练】(requires_grad=False, 与 risk_sensitive 系列一致;
          固定探索幅度 ⇒ 四算法公平对比, 熵恒定)。
    """

    def __init__(self, state_dim, action_dim, init_std=0.5, hidden=None, bias=True):
        """
        Args:
            state_dim:  状态维度 (本环境 25)
            action_dim: 动作维度 (本环境 K=5, logits)
            init_std:   初始探索标准差 (logits 空间; softmax 敏感度下 0.3~0.7 合理)
            hidden:     None=线性策略 (默认); [64,64] 等 → MLP
            bias:       是否带偏置 (默认 True, 表达常量 logits, 见模块注释)
        """
        super().__init__()
        if hidden:
            # MLP: Linear-Tanh 堆叠 + 输出层 (输出层 gain 取小, 初始接近均匀权重)
            dims = [state_dim] + list(hidden)
            layers = []
            for i in range(len(dims) - 1):
                layers.append(init_weights(nn.Linear(dims[i], dims[i + 1], bias=bias)))
                layers.append(nn.Tanh())
            layers.append(init_weights(nn.Linear(dims[-1], action_dim, bias=bias), gain=0.01))
            self.model = nn.Sequential(*layers)
        else:
            # 线性策略: μ(s) = W·s + b (gain 取小 → 初始 logits≈0 → 初始权重≈均匀)
            self.model = nn.Sequential(
                init_weights(nn.Linear(state_dim, action_dim, bias=bias), gain=0.01))

        # 对数标准差 (各动作维独立同 σ, 默认不训练)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(np.log(init_std))))
        self.log_std.requires_grad = False

    def forward(self, x):
        """前向: x [*, state_dim] → 动作均值 μ(s) [*, action_dim] (logits)。"""
        return self.model(x)


class Critic(nn.Module):
    """
    标量价值网络 V(s) (QPPO 的 baseline 用): MLP [state_dim → hidden → 1]。
    QPPO 中回归目标是 -𝟙{U(τ)≤q} (轨迹级), V(s) 仅作方差缩减 baseline。
    """

    def __init__(self, state_dim, hidden=(64, 64)):
        super().__init__()
        dims = [state_dim] + list(hidden)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(init_weights(nn.Linear(dims[i], dims[i + 1])))
            layers.append(nn.Tanh())
        layers.append(init_weights(nn.Linear(dims[-1], 1), gain=1.0))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """前向: x [*, state_dim] → V(s) [*, 1]。"""
        return self.model(x)
