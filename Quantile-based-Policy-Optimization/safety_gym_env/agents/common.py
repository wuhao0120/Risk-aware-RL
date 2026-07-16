# -*- coding: utf-8 -*-
"""
common.py —— ∞-horizon setting 内 GPU 智能体共享的可复用件。

【来源】逐字提取自有限步版 risk_sensitive_env/agents/dqc_ac_beta.py 的 3 个无依赖件
(lr_lambda / indicator / DistributionalCritic), 使本目录自成体系、无需 import 那个
(已知 stale/buggy 的) CPU 智能体文件。算法语义与原件完全一致。
"""

import torch
import torch.nn as nn


def lr_lambda(k, a, b, c):
    """
    学习率衰减因子 (与 qcpo.py / dqc_ac_beta.py 完全一致)
    lr(k) = a / (b + k)^c
    """
    return a / ((b + k) ** c)


def indicator(threshold, values):
    """
    示性函数 I(values <= threshold), 返回与 values 同形状的 0/1 张量
    (与 qcpo.py / dqc_ac.py 语义一致)
    """
    return torch.where(values <= threshold,
                       torch.ones_like(values), torch.zeros_like(values))


def indicator_ge(threshold, values):
    """
    示性函数 I(values >= threshold), 返回同形状 0/1 张量 —— 【上尾】版本。

    CMDP 约束 P(C ≥ d) ≤ ω (cost 超限 = outage) 是【上尾】事件, 与用户 risk/portfolio
    环境的 P(Z ≤ q) (下尾) 方向相反, 故需要这个上尾示性函数:
        outage 判定 𝟙{C ≥ d};  cost 分布式 critic 的约束 CDF Ψ̂ = (1/N)Σ 𝟙{ψ_c,i ≥ b}。
    """
    return torch.where(values >= threshold,
                       torch.ones_like(values), torch.zeros_like(values))


class DistributionalCritic(nn.Module):
    """
    QR-DQN 风格分布式 critic: 把 (s,a) 映射到 N 个分位数 ψ_i(s,a)。

    分位数 ψ_i 对应水平 τ_i=(i-0.5)/N, 一起刻画了回报 Z(s,a) 的整条分布。
    """

    def __init__(self, state_dim, action_dim, num_quantiles, hidden=[64, 64]):
        """
        Args:
            state_dim:     状态维度 (本 ∞-horizon 环境为无状态, =1)
            action_dim:    动作维度 (本环境为 1)
            num_quantiles: 分位数个数 N (critic 输出维度)
            hidden:        隐藏层维度列表
        """
        super().__init__()
        # 输入维度 = 状态 + 动作 拼接; 逐层堆 Linear+ReLU
        dims = [state_dim + action_dim] + list(hidden)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))  # 全连接层
            layers.append(nn.ReLU())                        # 非线性激活
        layers.append(nn.Linear(dims[-1], num_quantiles))   # 输出层: N 个分位数
        self.net = nn.Sequential(*layers)

        # 正交初始化 (与项目其它网络风格一致, 利于训练稳定)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, state, action):
        """
        前向传播

        输入: state [B, state_dim], action [B, action_dim]
        输出: [B, N] —— 每行是该 (s,a) 的 N 个分位数估计
        """
        x = torch.cat([state, action], dim=-1)              # 在最后一维拼接 → [B, sd+ad]
        return self.net(x)                                  # → [B, N]
