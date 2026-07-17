# -*- coding: utf-8 -*-
"""
common.py —— ∞-horizon setting 内 GPU 智能体共享的可复用件。

【来源】逐字提取自有限步版 risk_sensitive_env/agents/dqc_ac_beta.py 的 3 个无依赖件
(lr_lambda / indicator / DistributionalCritic), 使本目录自成体系、无需 import 那个
(已知 stale/buggy 的) CPU 智能体文件。算法语义与原件完全一致。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward_features(self, state, action):
        """
        返回最后一个quantile Linear之前的共享action-conditioned表示。

        默认forward不调用本函数，所以新增接口不会改变历史QR前向、state_dict或
        浮点运算。Weibull消融显式二次前向同一组Linear+ReLU；无Dropout/BatchNorm，
        因此feature数值与quantile head实际接收的表示逐元素相同。
        """
        x = torch.cat([state, action], dim=-1)              # [B,state_dim+action_dim]
        # 最后一层是N维quantile输出；这里只执行共享hidden trunk并保留其计算图。
        for layer in list(self.net.children())[:-1]:
            x = layer(x)
        return x                                             # [B,hidden[-1]]


class NonCrossingDistributionalCritic(DistributionalCritic):
    """
    NQ-Net 风格固定网格 critic：显式预测均值与非负相邻分位数 gap。

    为了与现有 QR critic 做同容量因果对照，最后一层仍只输出 N 个 raw 值：
    第一个是 N 个 quantile 的均值 v，其余 N-1 个是相邻 gap 的 pre-activation。
    累积 gap 后减去行均值，保证输出均值严格等于 v，同时保证相邻输出不下降。
    这是论文 K+1 输出公式去掉完全不可辨识首 gap 后的非冗余等价形式。
    """

    def __init__(self, state_dim, action_dim, num_quantiles, hidden=None,
                 gap_activation='relu'):
        """
        构造与 DistributionalCritic 完全同形状、同初始化顺序的 NQ critic。

        gap_activation='relu' 对齐论文用于离散 Atari 回报的 NQ-Net*；'elu1'
        对齐一般 NQ-Net 的 ELU(x)+1 严格正 gap。两者必须独立消融，不能混用。
        """
        selected_hidden = [64, 64] if hidden is None else list(hidden)
        super().__init__(state_dim, action_dim, num_quantiles, selected_hidden)
        self.num_quantiles = int(num_quantiles)
        self.gap_activation = str(gap_activation).lower()
        if self.num_quantiles <= 0:
            raise ValueError("num_quantiles must be positive")
        if self.gap_activation not in {'relu', 'elu1'}:
            raise ValueError("NQ gap_activation must be 'relu' or 'elu1'")

    def _activate_gaps(self, raw_gaps):
        """把 [B,N-1] pre-activation 映射为非负相邻 quantile 差值。"""
        if self.gap_activation == 'relu':
            # 离散 cost 允许相邻 quantile 完全相等；ReLU 精确表示这些原子区间。
            return torch.relu(raw_gaps)
        # ELU(x)+1 在 x<0 时等于 exp(x)，因此 gap 严格大于 0。
        return F.elu(raw_gaps) + 1.0

    def forward(self, state, action):
        """
        输入 state/action 与 QR 相同，返回 [B,N] 单调非降 quantile。

        positions 的第一列固定为 0，后续列为相邻 gap 的累计和；按行中心化只
        平移整条 quantile 曲线，不改变任何 gap，最后再由 v 决定分布位置。
        """
        raw = super().forward(state, action)                   # [B,N]，与 QR raw 同初始化
        mean = raw[:, :1]                                      # [B,1]，目标 quantile 均值
        gaps = self._activate_gaps(raw[:, 1:])                 # [B,N-1]，逐项非负
        first = torch.zeros_like(mean)                         # 第一个 quantile 的相对位置
        positions = torch.cat([first, torch.cumsum(gaps, dim=-1)], dim=-1)
        centered = positions - positions.mean(dim=-1, keepdim=True)
        return mean + centered                                 # mean(output)=v，q_i<=q_{i+1}


class WeibullTailHead(nn.Module):
    """
    QCPO_refs式两参数Weibull尾部头，只读取cost critic共享(s,a) feature。

    原实现使用alpha=4*sigmoid(linear)与beta=exp(linear)。训练loss只需要
    log(beta)，因此直接返回未指数化的log_beta，避免exp后再log的溢出/下溢；
    在有限值区域两者数学和梯度完全等价。Linear保留PyTorch默认初始化，与ref一致。
    """

    def __init__(self, feature_dim):
        """构造独立shape与log-scale标量头；不拥有或复制critic trunk参数。"""
        super().__init__()
        self.alpha = nn.Linear(int(feature_dim), 1)          # sigmoid后限制shape∈(0,4)
        self.log_beta = nn.Linear(int(feature_dim), 1)       # beta的自然对数，无需显式exp

    def forward(self, feature):
        """输入[B,H]，返回alpha/log_beta两个[B]张量。"""
        alpha = 4.0 * torch.sigmoid(self.alpha(feature).squeeze(-1))
        log_beta = self.log_beta(feature).squeeze(-1)
        return alpha, log_beta


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


class ExceedanceProbabilityCritic(nn.Module):
    """
    直接查询点 critic：估计 P(C_remaining >= budget | state, action, budget)。

    QR/IQN 先学习整条 quantile function，再在给定 budget 处计数/积分；本网络只学习
    DQCAC actor 真正使用的 Bernoulli 条件概率。budget 是显式条件变量，因此同一个
    (state, action) 可以查询不同剩余安全预算，而不需要为每个阈值重训输出头。
    """

    def __init__(self, state_dim, action_dim, hidden=None, budget_scale=1.0):
        """
        Args:
            state_dim: cost critic 已预处理的状态/step-feature 维度。
            action_dim: 连续动作维度。
            hidden: MLP 隐藏层宽度；None 时使用 [256, 256]。
            budget_scale: 固定正尺度，网络实际接收 budget / budget_scale。

        Returns:
            forward 返回未过 sigmoid 的 [B] logits；训练用 BCEWithLogitsLoss，
            查询方显式 sigmoid 得到概率，避免训练时重复 sigmoid 降低数值稳定性。
        """
        super().__init__()
        hidden = [256, 256] if hidden is None else list(hidden)
        if not hidden:
            raise ValueError("direct CDF hidden must contain at least one layer")
        if float(budget_scale) <= 0.0:
            raise ValueError("direct CDF budget_scale must be positive")

        # budget_scale 是模型语义的一部分，注册为buffer后可随checkpoint/device迁移。
        self.register_buffer(
            "budget_scale", torch.tensor(float(budget_scale), dtype=torch.float32))
        dims = [int(state_dim) + int(action_dim) + 1] + hidden
        layers = []
        for input_dim, output_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

        # 隐藏层与QR critic统一用正交初始化；末层小gain令初始logit接近0，避免
        # 随机大logit把首批BCE推入饱和区。bias=0对应中性先验p=0.5。
        linear_layers = [
            module for module in self.modules() if isinstance(module, nn.Linear)]
        for module in linear_layers:
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        nn.init.orthogonal_(linear_layers[-1].weight, gain=0.01)

    def forward(self, state, action, budget):
        """
        Args:
            state: [B,state_dim]。
            action: [B,action_dim]。
            budget: 标量、[B] 或 [B,1]。

        Returns:
            [B] Bernoulli logits，对应每行查询点的超阈概率。
        """
        if state.ndim != 2 or action.ndim != 2:
            raise ValueError("direct CDF state/action must both be rank-2 tensors")
        if state.shape[0] != action.shape[0]:
            raise ValueError("direct CDF state/action batch sizes must match")

        if torch.is_tensor(budget):
            budget_values = budget.to(device=state.device, dtype=state.dtype)
            if budget_values.ndim == 0:
                budget_values = budget_values.expand(state.shape[0]).unsqueeze(1)
            elif budget_values.ndim == 1:
                budget_values = budget_values.unsqueeze(1)
            elif budget_values.ndim != 2 or budget_values.shape[1] != 1:
                raise ValueError("direct CDF budget must be scalar, [B], or [B,1]")
            if budget_values.shape[0] not in {1, state.shape[0]}:
                raise ValueError("direct CDF budget batch must be 1 or match state")
            if budget_values.shape[0] == 1 and state.shape[0] != 1:
                budget_values = budget_values.expand(state.shape[0], 1)
        else:
            budget_values = torch.full(
                (state.shape[0], 1), float(budget),
                dtype=state.dtype, device=state.device)

        scaled_budget = budget_values / self.budget_scale.to(dtype=state.dtype)
        inputs = torch.cat([state, action, scaled_budget], dim=-1)
        return self.net(inputs).squeeze(-1)


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
