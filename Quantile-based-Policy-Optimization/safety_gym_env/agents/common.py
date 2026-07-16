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


class ImplicitQuantileCritic(nn.Module):
    """
    IQN 风格 action-conditioned critic：输入 (s,a,τ)，输出 Q(s,a,τ)。

    与固定输出头 QR critic 的区别是：τ 不再绑定在第 i 个输出神经元上，而是先经
    cosine embedding，再与 (s,a) 特征逐元素相乘。因此训练可以连续采样 τ，
    查询 CDF 时也可以使用比训练更多的确定性 uniform τ 点。IQN 仍估计 quantile
    function，不是直接输出 CDF；上尾概率由调用方对 uniform-τ 查询结果做积分。
    """

    def __init__(self, state_dim, action_dim, hidden=None,
                 num_cosines=64, default_query_quantiles=128):
        """
        Args:
            state_dim: cost critic 条件状态/历史特征维度。
            action_dim: 连续动作维度。
            hidden: (s,a) MLP 宽度；最后一层宽度也是 τ embedding 维度。
            num_cosines: cosine basis 数量，使用 cos(π i τ), i=0,...,K-1。
            default_query_quantiles: 未显式传 τ 时使用的 deterministic midpoint 数。
        """
        super().__init__()
        hidden = [256, 256] if hidden is None else list(hidden)
        if not hidden:
            raise ValueError("IQN hidden must contain at least one layer")
        if int(num_cosines) <= 0:
            raise ValueError("IQN num_cosines must be positive")
        if int(default_query_quantiles) <= 0:
            raise ValueError("IQN default_query_quantiles must be positive")

        # (s,a) 主干与固定 QR critic 相同：Linear+ReLU，便于把差异限制在输出表示。
        dims = [int(state_dim) + int(action_dim)] + hidden
        feature_layers = []
        for input_dim, output_dim in zip(dims[:-1], dims[1:]):
            feature_layers.append(nn.Linear(input_dim, output_dim))
            feature_layers.append(nn.ReLU())
        self.feature_net = nn.Sequential(*feature_layers)

        self.num_cosines = int(num_cosines)
        feature_dim = int(hidden[-1])
        self.tau_embedding = nn.Linear(self.num_cosines, feature_dim)
        self.quantile_head = nn.Linear(feature_dim, 1)

        # buffer 随 module.to()/state_dict 迁移；midpoint grid 不含端点，避免 τ=0/1。
        query_count = int(default_query_quantiles)
        self.register_buffer(
            "default_query_taus",
            (torch.arange(query_count, dtype=torch.float32) + 0.5) / query_count)
        self.register_buffer(
            "cosine_indices",
            torch.arange(self.num_cosines, dtype=torch.float32) * torch.pi)

        # 与 DistributionalCritic 对齐使用正交初始化；不单独缩小输出层，避免把
        # “IQN 表示”与“更小初始 cost 尺度”混成同一个实验变量。
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, state, action, taus=None):
        """
        Args:
            state: [B,state_dim]。
            action: [B,action_dim]。
            taus: None、[N] 或 [B,N]；None 使用 deterministic 查询网格。

        Returns:
            [B,N]，第 j 列为对应 τ_j 的 quantile-function 估计。
        """
        if state.ndim != 2 or action.ndim != 2:
            raise ValueError("IQN state/action must both be rank-2 tensors")
        if state.shape[0] != action.shape[0]:
            raise ValueError("IQN state/action batch sizes must match")

        batch_size = state.shape[0]
        selected_taus = self.default_query_taus if taus is None else taus
        selected_taus = selected_taus.to(device=state.device, dtype=state.dtype)
        if selected_taus.ndim == 1:
            selected_taus = selected_taus.unsqueeze(0).expand(batch_size, -1)
        elif selected_taus.ndim == 2:
            if selected_taus.shape[0] not in {1, batch_size}:
                raise ValueError("IQN taus batch dimension must be 1 or match state batch")
            if selected_taus.shape[0] == 1 and batch_size != 1:
                selected_taus = selected_taus.expand(batch_size, -1)
        else:
            raise ValueError("IQN taus must be rank 1 or 2")
        if bool(((selected_taus <= 0.0) | (selected_taus >= 1.0)).any()):
            raise ValueError("IQN taus must lie strictly inside (0,1)")

        state_action = torch.cat([state, action], dim=-1)
        state_action_feature = self.feature_net(state_action)       # [B,H]
        cosine = torch.cos(
            selected_taus.unsqueeze(-1)
            * self.cosine_indices.to(dtype=state.dtype))            # [B,N,K]
        tau_feature = torch.relu(self.tau_embedding(cosine))         # [B,N,H]
        joint_feature = state_action_feature.unsqueeze(1) * tau_feature
        return self.quantile_head(joint_feature).squeeze(-1)         # [B,N]


class ScalarValueCritic(nn.Module):
    """
    状态价值网络 V(s)，专供 DQCAC 的 reward GAE/PPO 可选主干使用。

    为什么不直接复用 DistributionalCritic:
        - GAE 需要低方差的 V(s)，而不是用少量动作采样近似 E_a[mean ψ(s,a)]；
        - V(s) 不接动作，避免 Q(s,a)-E_a Q(s,a) 两个相近估计相减后的 critic/采样噪声；
        - cost 约束仍由原 distributional critic 负责，所以该网络不改变风险分布建模。
    """

    def __init__(self, state_dim, hidden=None):
        """
        Args:
            state_dim: 原始观测维度；当前 Safety-Gym 为 60 或 76。
            hidden:    隐藏层宽度列表；None 时使用 [256, 256]。

        输出:
            任意前缀形状 `[..., state_dim]` → `[...]` 的标量状态价值。
        """
        super().__init__()
        hidden = [256, 256] if hidden is None else list(hidden)
        dims = [state_dim] + hidden
        layers = []

        # 与 actor 的平滑 Tanh 主干保持相近表达能力，减少“网络容量不同”这个额外变量。
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(dims[-1], 1))                 # V(s) 只有一个标量输出
        self.net = nn.Sequential(*layers)

        # 隐藏层 gain=1；输出层用较小 gain，令初始 V≈0，避免首批 GAE 被随机大偏置支配。
        linear_layers = [module for module in self.modules() if isinstance(module, nn.Linear)]
        for module in linear_layers:
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        nn.init.orthogonal_(linear_layers[-1].weight, gain=0.01)

    def forward(self, state):
        """调用 MLP 并移除末尾长度为 1 的 value 维度。"""

        return self.net(state).squeeze(-1)
