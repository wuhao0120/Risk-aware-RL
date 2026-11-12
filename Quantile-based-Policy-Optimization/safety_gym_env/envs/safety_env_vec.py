# -*- coding: utf-8 -*-
"""
同步 Safety Gym vector adapter，完整保留 episode boundary 语义。

Step contract:
    - observation: auto-reset 后可供下一 episode 使用的 [B,D] state；
    - final_observation: 旧 episode 边界的 [B,D] state；
    - terminated/truncated: 两个独立 [B] bool masks；
    - final_observation_mask: 明确哪些 final rows 有效；
    - reward/cost: 原生环境当步标量，cost 只允许进入 diagnostic 日志。

Adapter horizon 的语义是 sampling truncation，不是真实 terminal。达到 horizon 时，若环境
没有 terminated，则 adapter 设置 truncated=True。collector 因而从 final_observation
bootstrap critic，却仍只把实际采到的 reward 累加进 G_T。auto-reset observation 从不替代
旧 episode TD next state，也不会被 collector 拼接进同一 trajectory。

Additional invariants:
    - 每次显式 reset 会清空 segment step counter 与 diagnostic buffers；
    - first reset 使用 seed+i，之后 reset 让环境自身 RNG 连续演化；
    - force truncation 不覆盖同一步已发生的 true termination；
    - final mask 与 terminated|truncated 逐元素一致；
    - action 只在 SafetyEnv 内 clip，adapter 同时返回其报告的 executed action；
    - 返回 tensors 的 dtype/device 在 sync 和 MP 后端保持一致。

本同步实现是 MP 版本的语义基准：两者共享 VecStepResult，算法不包含 backend 分支。
"""
from dataclasses import dataclass
import numpy as np
import torch
from .safety_env import SafetyEnv


@dataclass
class VecStepResult:
    """结构化 step；reset observation 与旧 episode final observation 分离。"""
    observation: torch.Tensor
    reward: torch.Tensor
    cost: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    final_observation: torch.Tensor
    final_observation_mask: torch.Tensor
    # Optional default keeps old custom adapters/tests source-compatible. Production sync/MP
    # adapters always populate the exact action that SafetyEnv passed to native physics.
    executed_action: torch.Tensor = None

    @property
    def episode_end(self):
        """逐环境 episode boundary：[B] bool。"""
        return self.terminated | self.truncated


class _Space:
    """只提供 shape 的兼容容器。"""
    def __init__(self, shape):
        self.shape = tuple(shape)


class SafetyVecEnv:
    """B 路同步 CPU environments，输出 device tensors。"""

    def __init__(self, env_id="SafetyPointGoal1-v0", num_envs=8, horizon=1000,
                 device=None, ref_env=None, seed=0):
        self.env_id = ref_env.env_id if ref_env is not None else str(env_id)
        self.B, self.n = int(num_envs), int(horizon)
        self.device = device if device is not None else torch.device("cpu")
        self._seed = int(seed)
        self.envs = [SafetyEnv(self.env_id, seed=self._seed + i) for i in range(self.B)]
        self.obs_dim, self.act_dim = self.envs[0].obs_dim, self.envs[0].act_dim
        self.observation_space, self.action_space = _Space((self.obs_dim,)), _Space((self.act_dim,))
        self.step_count, self._first_reset = 0, True
        self._cost_buf, self._reward_buf = [], []

    def reset(self):
        """开始新 vector segment，返回 [B,D] reset observations。"""
        self.step_count = 0
        self._cost_buf, self._reward_buf = [], []
        obs = []
        for i, env in enumerate(self.envs):
            seed = self._seed + i if self._first_reset else None
            obs.append(env.reset(seed=seed)[0])
        self._first_reset = False
        return torch.as_tensor(np.stack(obs), dtype=torch.float32, device=self.device)

    def step(self, actions):
        """step、标记 adapter horizon truncation，并 auto-reset ended slots。"""
        actions_np = actions.detach().cpu().numpy().astype(np.float32)
        self.step_count += 1
        force_truncation = self.step_count >= self.n
        obs, finals, rewards, costs = [], [], [], []
        terms, truncs, masks, executed_actions = [], [], [], []
        for i, env in enumerate(self.envs):
            final, reward, cost, term, trunc, info = env.step(actions_np[i])
            trunc = bool(trunc or (force_truncation and not term))
            ended = bool(term or trunc)
            # final 保持为旧 episode observation；obs 可安全承载 reset state。
            finals.append(final)
            masks.append(ended)
            obs.append(env.reset()[0] if ended else final)
            rewards.append(reward)
            costs.append(cost)
            terms.append(bool(term))
            truncs.append(bool(trunc))
            # SafetyEnv is the single clipping authority. The fallback supports a custom legacy
            # wrapper that does not yet publish executed_action while preserving identical bounds.
            reported_action = info.get("executed_action")
            if reported_action is None:
                reported_action = env.clip_action(actions_np[i])
            executed_actions.append(np.asarray(reported_action, dtype=np.float32))
        self._reward_buf.append(float(np.mean(rewards)))
        self._cost_buf.append(float(np.mean(costs)))
        return VecStepResult(
            torch.as_tensor(np.stack(obs), dtype=torch.float32, device=self.device),
            torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
            torch.as_tensor(costs, dtype=torch.float32, device=self.device),
            torch.as_tensor(terms, dtype=torch.bool, device=self.device),
            torch.as_tensor(truncs, dtype=torch.bool, device=self.device),
            torch.as_tensor(np.stack(finals), dtype=torch.float32, device=self.device),
            torch.as_tensor(masks, dtype=torch.bool, device=self.device),
            torch.as_tensor(np.stack(executed_actions), dtype=torch.float32, device=self.device))

    def render(self, mode=None):
        """返回 environment cost batch mean 序列，仅作诊断。"""
        return None if not self._cost_buf else np.asarray(self._cost_buf, dtype=np.float64)

    def stats(self):
        """返回环境侧诊断统计。"""
        out = {}
        if self._cost_buf:
            out["avg_step_cost"] = float(np.mean(self._cost_buf))
        if self._reward_buf:
            out["avg_step_reward"] = float(np.mean(self._reward_buf))
        return out

    def close(self):
        """释放全部 MuJoCo resources。"""
        for env in self.envs:
            env.close()
