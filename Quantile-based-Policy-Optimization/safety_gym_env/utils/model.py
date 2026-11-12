# -*- coding: utf-8 -*-
"""
Safety Gym 共用 recurrent Gaussian policy。

Shape contract:
    - raw observations: [T,B,D]，单步便捷接口接受 [B,D]；
    - recurrent state: h/c 均为 [1,B,H]，每个 vector slot 独立；
    - Gaussian mean/log_std/action: [T,B,A] 或单步 [B,A]；
    - episode_end mask: [B]，只清零对应 slot 的 h/c；
    - observation RMS 最后一维固定为 D，并作为 actor buffers checkpoint。

Update contract:
    - rollout 期间和 PPO epochs 期间 RMS 固定；
    - 完成当前 actor update 后才合并该 rollout 的 raw observations；
    - policy likelihood 对 latent Gaussian action 求值；
    - SafetyEnv 执行 clip 不改变保存的 action/log probability；
    - log_std 默认可学习且初始 sigma=1；
    - cost 从未进入 encoder、RMS 或 recurrent state。


四个算法共享同一策略容量:
    raw observation -> running normalization -> FC512 -> FC512 -> LSTM512
                    -> tanh(mean), learnable log_std

策略采样保存的是未裁剪 latent Gaussian action。SafetyEnv 只在执行前裁剪动作，
因此 PPO likelihood ratio 始终对同一个 latent action 求值，不把 many-to-one 的
环境裁剪误当成可逆 tanh transformation。
"""
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

RnnState = Tuple[torch.Tensor, torch.Tensor]


def init_weights(module, gain=1.0):
    """对 Linear/LSTM 做正交初始化，并把所有偏置置零。"""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        for name, parameter in module.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)
    return module


class ObservationRMS(nn.Module):
    """可 checkpoint 的 observation running mean/variance。"""

    def __init__(self, shape, epsilon=1e-4, var_floor=1e-6):
        super().__init__()
        self.var_floor = float(var_floor)
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(float(epsilon), dtype=torch.float64))

    @torch.no_grad()
    def update(self, observations):
        """用 parallel-Welford 公式合并 [*,D] raw observations。"""
        x = observations.detach().reshape(-1, self.mean.numel()).float()
        if x.numel() == 0:
            return
        # 批统计用 population variance，随后转为 M2 合并。
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = torch.tensor(float(x.shape[0]), device=x.device, dtype=torch.float64)
        old_count = self.count.to(x.device)
        total_count = old_count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * (batch_count / total_count).float()
        old_m2 = self.var * old_count.float()
        batch_m2 = batch_var * batch_count.float()
        correction = delta.square() * (old_count * batch_count / total_count).float()
        new_var = (old_m2 + batch_m2 + correction) / total_count.float()
        self.mean.copy_(new_mean)
        self.var.copy_(torch.clamp(new_var, min=self.var_floor))
        self.count.copy_(total_count.to(self.count.device))

    def normalize(self, observations):
        """用 detached moments 做 affine normalization。"""
        std = torch.sqrt(torch.clamp(self.var.detach(), min=self.var_floor))
        return (observations - self.mean.detach()) / std


class RecurrentGaussianActor(nn.Module):
    """FC×2 + LSTM 的统一 latent Gaussian policy。"""

    def __init__(self, state_dim, action_dim, hidden=(512, 512), lstm_size=512,
                 init_std=1.0, learn_std=True, normalize_observation=True):
        super().__init__()
        hidden = tuple(int(x) for x in hidden)
        if len(hidden) != 2:
            raise ValueError("公共 actor 要求恰好两层 FC，例如 hidden=(512, 512)")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.lstm_size = int(lstm_size)
        self.normalize_observation = bool(normalize_observation)
        # RMS 是 actor buffer，checkpoint 自动包含 normalization state。
        self.obs_rms = ObservationRMS((self.state_dim,))
        self.encoder = nn.Sequential(
            init_weights(nn.Linear(self.state_dim, hidden[0])), nn.Tanh(),
            init_weights(nn.Linear(hidden[0], hidden[1])), nn.Tanh(),
        )
        self.lstm = init_weights(nn.LSTM(hidden[1], self.lstm_size))
        self.mean_head = init_weights(nn.Linear(self.lstm_size, self.action_dim), gain=0.01)
        self.log_std = nn.Parameter(torch.full(
            (self.action_dim,), float(np.log(float(init_std)))))
        self.log_std.requires_grad = bool(learn_std)

    def initial_state(self, batch_size, device=None):
        """返回全零 (h,c)[1,B,H]，作为 episode 初始 recurrent state。"""
        parameter = next(self.parameters())
        device = parameter.device if device is None else device
        shape = (1, int(batch_size), self.lstm_size)
        return torch.zeros(shape, device=device), torch.zeros(shape, device=device)

    @staticmethod
    def reset_state(state: RnnState, episode_end):
        """仅对 episode_end=True 的 vector slots 清零 h/c。"""
        keep = (~episode_end.bool()).to(state[0].dtype).view(1, -1, 1)
        return state[0] * keep, state[1] * keep

    def update_obs_rms(self, observations):
        """rollout 后显式更新；PPO epochs 内不改变 normalization。"""
        if self.normalize_observation:
            self.obs_rms.update(observations)

    def forward(self, observations, state: Optional[RnnState] = None):
        """[T,B,D] -> mean/log_std [T,B,A] 与 next state [1,B,H]。"""
        if observations.ndim != 3:
            raise ValueError(f"expected observations [T,B,D], got {tuple(observations.shape)}")
        T, B, D = observations.shape
        if D != self.state_dim:
            raise ValueError(f"expected state_dim={self.state_dim}, got {D}")
        if state is None:
            state = self.initial_state(B, observations.device)
        x = self.obs_rms.normalize(observations) if self.normalize_observation else observations
        x = self.encoder(x.reshape(T * B, D)).reshape(T, B, -1)
        recurrent, next_state = self.lstm(x, state)
        mean = torch.tanh(self.mean_head(recurrent))
        log_std = self.log_std.view(1, 1, -1).expand_as(mean)
        return mean, log_std, next_state

    def step(self, observations, state: Optional[RnnState] = None):
        """[B,D] 单步接口。"""
        mean, log_std, next_state = self.forward(observations.unsqueeze(0), state)
        return mean[0], log_std[0], next_state


# 旧导入名兼容；实现已经是 recurrent actor。
Actor = RecurrentGaussianActor
