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

class RecurrentActorValue(nn.Module):
    """
    与 QCPO_refs policy/reward-value 主干逐层同构的 MLP+LSTM 适配器。

    输入协议严格保留参考实现：
        augmented_observation_t = concat(raw_observation_t, previous_cost_t)
        lstm_input_t = concat(MLP(augmented_observation_t), previous_action_t,
                              previous_reward_t)

    输出协议：
        mean_t = tanh(policy_head(feature_t))
        value_t = reward_value_head(feature_t)
        log_std 是可学习的逐动作全局参数
        feature_t = MLP_feature_t + LSTM_output_t（lstm_skip=True 时）

    默认不创建 cost distribution，严格保留历史 DQCAC 的 action-conditioned
    Z_c(history,a)。显式启用 QCPO quantile-GAE 消融时，才额外挂接参考实现同形的
    state-value cost head exp(Linear(feature))；原 action-conditioned critic 仍保留，
    因而可以在同一 rollout 上比较两种风险信用，而不是永久偷换算法语义。
    """

    def __init__(self, observation_dim, action_dim, hidden=None, lstm_size=512,
                 lstm_skip=True, init_std=1.0, learn_std=True,
                 normalize_observation=True, var_clip=1e-6,
                 hidden_nonlinearity='tanh', cost_adapter_width=0,
                 cost_adapter_scale=1.0, cost_value_quantiles=0):
        """
        Args:
            observation_dim: 已追加 previous_cost 后的维度，即 raw_state_dim+1。
            action_dim:      连续动作维度；DynamicButton 为 2。
            hidden:          MLP 隐层；公平对比默认 [512,512]。
            lstm_size:       recurrent hidden size；公平对比默认 512。
            lstm_skip:       True 时使用 MLP feature + LSTM output 残差。
            init_std:        初始高斯探索标准差；QCPO_refs 为 1。
            learn_std:       是否训练 log_std；公平对比应为 True。
            cost_adapter_width: 正数时在共享feature后增加H→W→H零初始化残差adapter；
                                0保持QCPO与历史DQCAC逐式结构不变。
            cost_adapter_scale: adapter残差的固定缩放系数。
            cost_value_quantiles: 正数时创建QCPO_refs同形的状态cost quantile头；
                                  0保持历史state_dict与forward路径完全不变。
        """
        super().__init__()
        hidden = [512, 512] if hidden is None else list(hidden)
        nonlinearities = {'tanh': nn.Tanh, 'relu': nn.ReLU}
        if hidden_nonlinearity not in nonlinearities:
            raise ValueError("hidden_nonlinearity must be 'tanh' or 'relu'")
        nonlinearity = nonlinearities[hidden_nonlinearity]

        # MLP body 与 QcpoRefModel._mlp_body 相同：每层 Linear 后接相同激活，
        # 不调用本文件的正交初始化，保留参考实现的 PyTorch 默认初始化。
        layers = []
        last_size = int(observation_dim)
        for width in hidden:
            layers.append(nn.Linear(last_size, int(width)))
            layers.append(nonlinearity())
            last_size = int(width)
        self.body = nn.Sequential(*layers)

        # LSTM 输入额外拼 previous_action 与 previous_reward；previous_cost 已在 observation 中。
        self.lstm_size = int(lstm_size)
        lstm_input_size = last_size + int(action_dim) + 1
        self.lstm = nn.LSTM(lstm_input_size, self.lstm_size)
        self.lstm_skip = bool(lstm_skip)
        if self.lstm_skip and last_size != self.lstm_size:
            raise ValueError("lstm_skip requires last MLP width == lstm_size")

        # policy/value 两头与 QCPO_refs 同形状；均值用 tanh，value 保持无界标量。
        self.mu = nn.Sequential(nn.Linear(self.lstm_size, int(action_dim)), nn.Tanh())
        self.value = nn.Linear(self.lstm_size, 1)
        self.log_std = nn.Parameter(
            torch.full((int(action_dim),), float(np.log(init_std))))
        self.log_std.requires_grad = bool(learn_std)

        # RMS 直接覆盖 augmented observation（含 previous_cost），与参考实现输入完全一致。
        self.normalize_observation = bool(normalize_observation)
        self.obs_rms = ObservationNormalizer(
            int(observation_dim), var_clip=float(var_clip), value_clip=10.0)

        # cost adapter是DQCAC共享监督的可选小参数子空间。输出层严格零初始化，
        # 所以启用时初始policy/value/cost feature仍与无adapter网络逐元素相同。
        # 构造前后恢复CPU RNG，避免新增模块推进随后reward/cost critic初始化随机流。
        self.cost_adapter_width = int(cost_adapter_width)
        self.cost_adapter_scale = float(cost_adapter_scale)
        if self.cost_adapter_width < 0:
            raise ValueError("cost_adapter_width must be non-negative")
        if self.cost_adapter_scale <= 0.0:
            raise ValueError("cost_adapter_scale must be positive")
        self.cost_adapter = None
        if self.cost_adapter_width > 0:
            rng_state = torch.get_rng_state()
            try:
                self.cost_adapter = nn.Sequential(
                    nn.Linear(self.lstm_size, self.cost_adapter_width),
                    nn.Tanh(),
                    nn.Linear(self.cost_adapter_width, self.lstm_size))
                nn.init.zeros_(self.cost_adapter[-1].weight)
                nn.init.zeros_(self.cost_adapter[-1].bias)
            finally:
                torch.set_rng_state(rng_state)

        # QCPO_refs 的 constraint head 直接输出 cost/scale 单位的正 quantiles：
        # c_dist=exp(Linear(feature))。默认0时既没有模块/state_dict键，也不改变
        # actor.parameters()顺序。显式启用时恢复CPU RNG，防止新增头推进随后
        # reward/cost critic初始化与Gaussian动作噪声的全局随机流。
        self.cost_value_quantiles = int(cost_value_quantiles)
        if self.cost_value_quantiles < 0:
            raise ValueError("cost_value_quantiles must be non-negative")
        self.cost_value = None
        if self.cost_value_quantiles > 0:
            rng_state = torch.get_rng_state()
            try:
                self.cost_value = nn.Linear(
                    self.lstm_size, self.cost_value_quantiles)
            finally:
                torch.set_rng_state(rng_state)

    def apply_cost_adapter(self, base_feature, detach_input=False):
        """
        返回base+scale*adapter(base)，可选阻断梯度进入MLP/LSTM基础表示。

        policy/value路径传detach_input=False，因此PPO与V loss仍训练完整网络；
        adapter-only cost辅助路径传True，使cost loss只能更新adapter参数。
        """
        source = base_feature.detach() if detach_input else base_feature
        if self.cost_adapter is None:
            return source
        residual = self.cost_adapter(source)
        return source + self.cost_adapter_scale * residual

    def cost_distribution(self, feature, detach_feature=False):
        """
        由历史条件feature预测状态cost quantiles，数值单位由Agent的cost scale定义。

        detach_feature=True只训练cost head，不让其loss写回policy MLP/LSTM；False则
        复现QCPO_refs的共享多任务梯度。exp保证输出为正，逐式对齐参考QcpoModel。
        """
        if self.cost_value is None:
            raise RuntimeError(
                "state cost distribution requested without cost_value_quantiles")
        source = feature.detach() if detach_feature else feature
        return torch.exp(self.cost_value(source))

    def initial_state(self, batch_size, device=None):
        """返回零初始化 (h,c)，形状均为 [1,B,H]。"""
        device = self.log_std.device if device is None else device
        h = torch.zeros(1, int(batch_size), self.lstm_size, device=device)
        c = torch.zeros(1, int(batch_size), self.lstm_size, device=device)
        return h, c

    def forward(self, observation, prev_action, prev_reward, init_rnn_state=None,
                return_features=False, return_base_features=False):
        """
        前向调用。

        输入:
            observation: [T,B,raw_state_dim+1]，最后一维是 previous_cost。
            prev_action: [T,B,action_dim]。
            prev_reward: [T,B]。
            init_rnn_state: (h,c)，各 [1,B,H]；None 表示零状态。
            return_features: True 时额外返回产生 policy/value 的历史条件特征。
            return_base_features: True时再返回adapter之前的MLP+LSTM基础特征；
                                  仅供DQCAC隔离cost梯度，要求return_features=True。

        返回:
            默认返回 mean [T,B,A]、log_std [T,B,A]、value [T,B]、(h_n,c_n)。
            return_features=True 时再追加 feature [T,B,H]；该接口供 DQCAC
            的 C-H0.5 cost-history 消融复用，不改变已有调用的四元返回值。
        """
        T, B = observation.shape[:2]
        policy_observation = self.obs_rms(observation)             if self.normalize_observation else observation

        # MLP 先在 [T*B,D] 上计算，再恢复时间主序供 LSTM 处理。
        mlp_feature = self.body(policy_observation.reshape(T * B, -1))
        recurrent_input = torch.cat([
            mlp_feature.view(T, B, -1),
            prev_action.reshape(T, B, -1),
            prev_reward.reshape(T, B, 1),
        ], dim=2)

        # nn.LSTM 返回所有时刻输出与末状态；显式 tuple 保证兼容参考实现 namedtuple 状态。
        recurrent_output, final_state = self.lstm(recurrent_input, init_rnn_state)
        recurrent_flat = recurrent_output.reshape(T * B, self.lstm_size)
        base_feature = (
            mlp_feature + recurrent_flat if self.lstm_skip else recurrent_flat)
        feature = self.apply_cost_adapter(base_feature, detach_input=False)

        mean = self.mu(feature).view(T, B, -1)
        log_std = self.log_std.repeat(T * B, 1).view(T, B, -1)
        value = self.value(feature).squeeze(-1).view(T, B)
        if return_base_features and not return_features:
            raise ValueError("return_base_features requires return_features=True")
        if return_features:
            feature_view = feature.view(T, B, -1)
            if return_base_features:
                return (
                    mean, log_std, value, final_state, feature_view,
                    base_feature.view(T, B, -1))
            return mean, log_std, value, final_state, feature_view
        return mean, log_std, value, final_state

    @torch.no_grad()
    def update_obs_rms(self, augmented_observation):
        """在 rollout 边界批量合并 augmented observation moments。"""
        if self.normalize_observation:
            self.obs_rms.update(augmented_observation)


class RecurrentCostEncoder(nn.Module):
    """
    DQCAC cost distribution critic 的独立 MLP+LSTM 历史编码器。

    输入协议与 QCPO_refs/RecurrentActorValue 完全相同：observation 已追加
    previous_cost，MLP 特征再拼 previous_action 与 previous_reward 后送入 LSTM。
    与 actor_feature 消融的关键区别是本编码器由 cost quantile loss 独立训练，
    因而 cost 表示不会随 PPO actor/value 的联合更新被动漂移。

    observation 的 running mean/variance 不在本类重复维护。Agent 在调用前使用
    actor.obs_rms 做同一数值变换；这只共享输入尺度统计，不共享任何可学习表示。
    """

    def __init__(self, observation_dim, action_dim, hidden=None, lstm_size=512,
                 lstm_skip=True, hidden_nonlinearity='tanh'):
        """
        构造与参考策略同形的 MLP+LSTM，但不创建 policy/value 输出头。

        Args:
            observation_dim: raw state 加 previous_cost 后的输入维度。
            action_dim:      previous_action 的维度。
            hidden:          MLP 隐层宽度，公平消融默认 [512,512]。
            lstm_size:       LSTM hidden/cell 宽度，公平消融默认 512。
            lstm_skip:       是否把 MLP feature 残差加到 LSTM output。
        """
        super().__init__()
        hidden = [512, 512] if hidden is None else list(hidden)
        nonlinearities = {'tanh': nn.Tanh, 'relu': nn.ReLU}
        if hidden_nonlinearity not in nonlinearities:
            raise ValueError("hidden_nonlinearity must be 'tanh' or 'relu'")
        nonlinearity = nonlinearities[hidden_nonlinearity]

        # 保留 QCPO_refs 的 PyTorch 默认 Linear 初始化，避免额外初始化差异。
        layers = []
        last_size = int(observation_dim)
        for width in hidden:
            layers.append(nn.Linear(last_size, int(width)))
            layers.append(nonlinearity())
            last_size = int(width)
        self.body = nn.Sequential(*layers)

        # previous_cost 已在 observation；这里只再拼 previous action/reward。
        self.lstm_size = int(lstm_size)
        lstm_input_size = last_size + int(action_dim) + 1
        self.lstm = nn.LSTM(lstm_input_size, self.lstm_size)
        self.lstm_skip = bool(lstm_skip)
        if self.lstm_skip and last_size != self.lstm_size:
            raise ValueError("lstm_skip requires last MLP width == lstm_size")

    def initial_state(self, batch_size, device):
        """返回 episode 起点的零 (hidden, cell)，形状均为 [1,B,H]。"""
        hidden = torch.zeros(
            1, int(batch_size), self.lstm_size, device=device)
        cell = torch.zeros(
            1, int(batch_size), self.lstm_size, device=device)
        return hidden, cell

    def forward(self, observation, prev_action, prev_reward,
                init_rnn_state=None):
        """
        把 [T,B,*] 历史输入编码成 [T,B,H] cost feature。

        observation 必须已经用共享 RMS 标准化；返回 final_state 供相邻 TBPTT
        chunk 继续传播，Agent 会在 chunk 边界 detach，限制反向图长度。
        """
        T, B = observation.shape[:2]
        mlp_feature = self.body(observation.reshape(T * B, -1))
        recurrent_input = torch.cat([
            mlp_feature.view(T, B, -1),
            prev_action.reshape(T, B, -1),
            prev_reward.reshape(T, B, 1),
        ], dim=2)
        recurrent_output, final_state = self.lstm(
            recurrent_input, init_rnn_state)
        recurrent_flat = recurrent_output.reshape(T * B, self.lstm_size)
        feature = (
            mlp_feature + recurrent_flat if self.lstm_skip else recurrent_flat)
        return feature.view(T, B, -1), final_state
