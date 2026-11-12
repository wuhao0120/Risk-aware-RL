# -*- coding: utf-8 -*-
"""
多进程 Safety Gym vector adapter；字段语义与同步版完全一致。

Concurrency protocol:
    - 使用 spawn，避免父进程 CUDA/EGL 状态通过 fork 污染 MuJoCo worker；
    - 每个 worker 永久持有一个 SafetyEnv，父进程只发送 reset/step/close；
    - step worker 先返回 native final observation，不在子进程内提前覆盖；
    - 父进程收齐 B 个结果后统一识别 native/adapter truncation；
    - 只有 ended slots 收到 reset 命令，其他 worker episode 不受影响；
    - reset observation 与 final observation 在 VecStepResult 中分别传递；
    - reward/cost/flags 批量搬到 policy device，物理始终留在 CPU；
    - close 优先走协议退出，只有 join 超时才 terminate 异常 worker。

Algorithm contract:
    - terminated critic mask=0；
    - truncated critic mask=1 且 next state=final_observation；
    - auto-reset state 只可能成为下一 episode 的起点；
    - Safety cost 仅是 diagnostic payload，不参与任何风险约束计算；
    - sync/mp 差异被限制在环境执行层，不进入 agent 数学；
    - worker 返回顺序固定为 remote 列表顺序，保持 vector slot identity；
    - parent 先广播全部 actions 再 recv，形成真正的并行 MuJoCo window；
    - adapter horizon counter 是全 vector 锁步 counter，不替代 native flags；
    - diagnostic buffers 只存 batch mean，不参与返回或训练状态。
"""
import multiprocessing as mp
import numpy as np
import torch
from .safety_env_vec import VecStepResult


def _worker(remote, parent_remote, env_id, seed):
    """worker 保留 native boundary，不在发送 final observation 前 reset。"""
    parent_remote.close()
    from envs.safety_env import SafetyEnv
    env = SafetyEnv(env_id, seed=seed)
    try:
        while True:
            command, data = remote.recv()
            if command == "reset":
                remote.send(env.reset(seed=data)[0])
            elif command == "step":
                observation, reward, cost, term, trunc, info = env.step(data)
                # Return the exact post-clip action alongside the transition. The parent never
                # reconstructs it from reset state or from a potentially different bound source.
                remote.send((
                    observation, reward, cost, term, trunc,
                    np.asarray(info["executed_action"], dtype=np.float32)))
            elif command == "close":
                remote.send(None)
                break
            else:
                raise RuntimeError(f"unknown worker command: {command}")
    finally:
        env.close()


class _Space:
    """只提供 shape 的兼容容器。"""
    def __init__(self, shape):
        self.shape = tuple(shape)


class SafetyVecEnvMP:
    """spawn-based B-worker vector environment。"""

    def __init__(self, env_id="SafetyPointGoal1-v0", num_envs=8, horizon=1000,
                 device=None, ref_env=None, seed=0):
        self.env_id = ref_env.env_id if ref_env is not None else str(env_id)
        self.B, self.n = int(num_envs), int(horizon)
        self.device, self._seed = device or torch.device("cpu"), int(seed)
        if ref_env is not None:
            self.obs_dim, self.act_dim = ref_env.obs_dim, ref_env.act_dim
        else:
            from .safety_env import SafetyEnv
            probe = SafetyEnv(self.env_id)
            self.obs_dim, self.act_dim = probe.obs_dim, probe.act_dim
            probe.close()
        self.observation_space, self.action_space = _Space((self.obs_dim,)), _Space((self.act_dim,))
        context = mp.get_context("spawn")
        self._remotes, self._procs = [], []
        for i in range(self.B):
            remote, work_remote = context.Pipe()
            process = context.Process(target=_worker,
                args=(work_remote, remote, self.env_id, self._seed + i), daemon=True)
            process.start()
            work_remote.close()
            self._remotes.append(remote)
            self._procs.append(process)
        self._first_reset, self.step_count = True, 0
        self._cost_buf, self._reward_buf = [], []

    def _reset_indices(self, indices):
        """只 reset ended slots，不改变其他 worker episode。"""
        for i in indices:
            self._remotes[i].send(("reset", None))
        return {i: self._remotes[i].recv() for i in indices}

    def reset(self):
        """reset 全部 workers，返回 [B,D] observations。"""
        self.step_count = 0
        self._cost_buf, self._reward_buf = [], []
        for i, remote in enumerate(self._remotes):
            seed = self._seed + i if self._first_reset else None
            remote.send(("reset", seed))
        self._first_reset = False
        obs = np.stack([remote.recv() for remote in self._remotes])
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device)

    def step(self, actions):
        """并行 step，并分离 reset/final observations。"""
        actions_np = actions.detach().cpu().numpy().astype(np.float32)
        for i, remote in enumerate(self._remotes):
            remote.send(("step", actions_np[i]))
        raw = [remote.recv() for remote in self._remotes]
        self.step_count += 1
        forced = self.step_count >= self.n
        finals, rewards, costs = [x[0] for x in raw], [x[1] for x in raw], [x[2] for x in raw]
        terms = [bool(x[3]) for x in raw]
        truncs = [bool(x[4] or (forced and not x[3])) for x in raw]
        executed_actions = [x[5] for x in raw]
        ended = [terms[i] or truncs[i] for i in range(self.B)]
        reset_map = self._reset_indices([i for i, flag in enumerate(ended) if flag])
        obs = [reset_map.get(i, finals[i]) for i in range(self.B)]
        self._reward_buf.append(float(np.mean(rewards)))
        self._cost_buf.append(float(np.mean(costs)))
        return VecStepResult(
            torch.as_tensor(np.stack(obs), dtype=torch.float32, device=self.device),
            torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
            torch.as_tensor(costs, dtype=torch.float32, device=self.device),
            torch.as_tensor(terms, dtype=torch.bool, device=self.device),
            torch.as_tensor(truncs, dtype=torch.bool, device=self.device),
            torch.as_tensor(np.stack(finals), dtype=torch.float32, device=self.device),
            torch.as_tensor(ended, dtype=torch.bool, device=self.device),
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
        """正常关闭 workers；只在 join 超时后 terminate。"""
        for remote in self._remotes:
            try:
                remote.send(("close", None))
                remote.recv()
                remote.close()
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self._procs:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()


def make_vec_env(env_id, num_envs, horizon, device, ref_env=None, seed=0, backend="mp"):
    """统一工厂；sync/mp 都返回 VecStepResult。"""
    if str(backend).lower() == "sync":
        from .safety_env_vec import SafetyVecEnv
        return SafetyVecEnv(env_id, num_envs, horizon, device, ref_env, seed)
    return SafetyVecEnvMP(env_id, num_envs, horizon, device, ref_env, seed)
