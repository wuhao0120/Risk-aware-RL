# -*- coding: utf-8 -*-
"""
qcpo_ref_model.py —— NIPS'22 QCPO 参考实现的网络【逐行移植】(QcpoModel)。

来源: QCPO_nips_ref/qcpo/qcpo_model.py (wyjung0625/QCPO)。
移植原则: 结构/数学逐式保留 (MLP body → LSTM(skip) → μ/V/c_dist/Weibull α,β 五头,
obs 归一化 + clip(-10,10), c_dist=exp(linear), α=4·sigmoid, β=exp), 仅把 rlpyt 的
MlpModel / RunningMeanStdModel / infer_leading_dims / namedarraytuple 换成本地等价物。

张量约定: forward 输入均为时间主序 [T, B, *] (采样时 T=1); init_rnn_state=(h,c) [1,B,H]。
"""
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn

ValueInfo = namedtuple("ValueInfo", ["r_value", "c_dist", "c_w_alpha", "c_w_beta"])
RnnState = namedtuple("RnnState", ["h", "c"])


class RunningMeanStdModel(nn.Module):
    """rlpyt.models.running_mean_std.RunningMeanStdModel 等价: obs 逐维 running mean/var。"""

    def __init__(self, shape):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.zeros(()))

    @torch.no_grad()
    def update(self, x):
        """x: [..., *shape] → 展平 leading 维后并行 moments 合并 (Chan et al.)。"""
        x = x.reshape(-1, *self.mean.shape)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        if self.count == 0:
            self.mean.copy_(batch_mean)
            self.var.copy_(batch_var)
            self.count.fill_(batch_count)
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / total)
        self.count.copy_(total)


def _mlp_body(input_size, hidden_sizes, nonlinearity):
    """rlpyt MlpModel(output_size=None) 等价: [Linear+act]×L, 输出 = 最后隐层。"""
    layers, last = [], input_size
    for h in hidden_sizes:
        layers += [nn.Linear(last, h), nonlinearity()]
        last = h
    return nn.Sequential(*layers), last


class QcpoRefModel(nn.Module):
    """QcpoModel 移植版: 共享 body + LSTM(skip) + μ/log_std/V/c_dist/Weibull(α,β) 头。"""

    def __init__(
            self,
            observation_shape,
            action_size,
            n_quantile,
            hidden_sizes=None,
            lstm_size=None,
            lstm_skip=True,
            constraint=True,
            hidden_nonlinearity="tanh",       # or "relu"
            mu_nonlinearity="tanh",
            init_log_std=0.,
            normalize_observation=True,
            var_clip=1e-6,
            ):
        super().__init__()
        self._n_quantile = n_quantile

        nl = {"tanh": nn.Tanh, "relu": nn.ReLU}[hidden_nonlinearity]
        mu_nl = {"tanh": nn.Tanh, "relu": nn.ReLU}[mu_nonlinearity]
        self._obs_ndim = len(tuple(observation_shape))
        input_size = int(np.prod(observation_shape))
        self.body, last_size = _mlp_body(input_size, hidden_sizes or [256, 256], nl)
        if lstm_size:
            lstm_input_size = last_size + action_size + 1     # + prev_action + prev_reward
            self.lstm = nn.LSTM(lstm_input_size, lstm_size)
            last_size = lstm_size
        else:
            self.lstm = None
        self.mu = nn.Sequential(nn.Linear(last_size, action_size), mu_nl())
        self.value = nn.Linear(last_size, 1)

        # 分位水平 τ_i=(i+0.5)/N; Weibull 对数尺度 c_tau=-log(1-τ) (buffer: 随 .to(device) 迁移)
        self.register_buffer("tau", torch.arange(n_quantile).float() / n_quantile + 1 / 2 / n_quantile)
        self.register_buffer("c_tau", -1 * torch.log(1. - self.tau))

        if constraint:
            self.constraint = nn.Linear(last_size, n_quantile)           # c_dist = exp(linear)
            self.w_alpha = nn.Sequential(nn.Linear(last_size, 1), nn.Sigmoid())  # α = 4·sigmoid
            self.w_beta = nn.Linear(last_size, 1)                        # β = exp(linear)
        else:
            self.constraint = self.w_alpha = self.w_beta = None
        self.log_std = nn.Parameter(init_log_std * torch.ones(action_size))
        self._lstm_skip = lstm_skip
        if normalize_observation:
            self.obs_rms = RunningMeanStdModel(tuple(observation_shape))
            self.var_clip = var_clip
        self.normalize_observation = normalize_observation

    def forward(self, observation, prev_action, prev_reward, init_rnn_state=None):
        """
        observation [T,B,obs] / prev_action [T,B,ad] / prev_reward [T,B] →
        (mu [T,B,ad], log_std [T,B,ad], ValueInfo([T,B],[T,B,N],[T,B],[T,B]), RnnState)。
        """
        T, B = observation.shape[0], observation.shape[1]
        if self.normalize_observation:
            obs_var = self.obs_rms.var
            if self.var_clip is not None:
                obs_var = torch.clamp(obs_var, min=self.var_clip)
            observation = torch.clamp((observation - self.obs_rms.mean) /
                                      obs_var.sqrt(), -10, 10)
        fc_x = self.body(observation.reshape(T * B, -1))
        if self.lstm is not None:
            lstm_input = torch.cat([fc_x.view(T, B, -1),
                                    prev_action.view(T, B, -1),
                                    prev_reward.view(T, B, 1)], dim=2)
            init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)
            lstm_out, (hn, cn) = self.lstm(lstm_input, init_rnn_state)
            lstm_out = lstm_out.reshape(T * B, -1)
            fc_x = fc_x + lstm_out if self._lstm_skip else lstm_out

        mu = self.mu(fc_x).view(T, B, -1)
        log_std = self.log_std.repeat(T * B, 1).view(T, B, -1)
        r_value = self.value(fc_x).squeeze(-1).view(T, B)

        if self.constraint is None:
            value = ValueInfo(r_value=r_value, c_dist=None, c_w_alpha=None, c_w_beta=None)
        else:
            c_dist = torch.exp(self.constraint(fc_x)).view(T, B, -1)
            c_w_alpha = (4 * self.w_alpha(fc_x)).squeeze(-1).view(T, B)
            c_w_beta = torch.exp(self.w_beta(fc_x)).squeeze(-1).view(T, B)
            value = ValueInfo(r_value=r_value, c_dist=c_dist,
                              c_w_alpha=c_w_alpha, c_w_beta=c_w_beta)

        outputs = (mu, log_std, value)
        if self.lstm is not None:
            outputs += (RnnState(h=hn, c=cn),)
        return outputs

    def update_obs_rms(self, observation):
        if not self.normalize_observation:
            return
        self.obs_rms.update(observation)
