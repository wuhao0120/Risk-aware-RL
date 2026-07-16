# -*- coding: utf-8 -*-
"""
SafetyVecEnvMP —— B 路【CPU 多进程】向量化 safety-gymnasium 环境 (与 SafetyVecEnv 同接口)。

为什么需要多进程 (对应用户要求"CPU 多核并行仿真 + GPU update"):
    SafetyVecEnv 是单进程 for 循环, B 个 mujoco env 串行 step → 吞吐被单核钉死。
    本类每个 worker 进程持有 1 个 SafetyEnv, 父进程广播动作/回收 (obs,r,c) → B 路
    物理仿真真并行 (机器 128 核)。【每个 env 内部仍是串行 mujoco step】—— 只是把
    独立 env 摊到多核, 不改任何单 env 动力学, 与论文 rlpyt CpuSampler 的并行语义一致。

实现要点:
    - multiprocessing 【spawn】上下文: 父进程已初始化 CUDA/EGL, fork 不安全;
      spawn 子进程干净重 import (envs 包 import 即注册论文环境, 短名可解析)。
    - worker 协议: ('reset', seed|None) → obs;  ('step', action) → (obs, r, c, term, trunc);
      ('close', None) → 退出。env 自身 terminated/truncated 时 worker 内部立即 reset
      (返回新初始 obs), 与 SafetyVecEnv 的兜底语义一致。
    - 段长 horizon / done 语义 / render / stats 与 SafetyVecEnv 完全一致 (锁步段截断)。

接口: reset()->obs[B,obs_dim];  step(a[B,act_dim])->(obs[B,obs_dim], reward[B], cost[B], done:bool)。
"""
import multiprocessing as mp

import numpy as np
import torch


def _worker(remote, parent_remote, env_id, seed):
    """worker 进程主循环: 持有 1 个 SafetyEnv, 按命令 reset/step/close。"""
    parent_remote.close()
    # 延迟 import: spawn 子进程内重新加载 (触发论文环境注册, EGL 各进程独立)
    from envs.safety_env import SafetyEnv
    env = SafetyEnv(env_id, seed=seed)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'reset':
                obs, _info = env.reset(seed=data)
                remote.send(obs)
            elif cmd == 'step':
                obs, r, c, term, trunc, _info = env.step(data)
                if term or trunc:                            # env 自身终止/截断 → 重开兜底
                    obs, _info = env.reset()
                remote.send((obs, r, c))
            elif cmd == 'close':
                remote.send(None)
                break
    finally:
        env.close()


class _Space:
    """轻量 shape 容器 (使 np.prod(env.observation_space.shape) 可用)。"""

    def __init__(self, shape):
        self.shape = tuple(shape)


class SafetyVecEnvMP:
    """B 路多进程向量化 safety-gymnasium 环境 (返回 device 张量, 含 cost; 接口=SafetyVecEnv)。"""

    def __init__(self, env_id='SafetyPointGoal1-v0', num_envs=8, horizon=1000,
                 device=None, ref_env=None, seed=0):
        """
        Args:
            env_id:   safety-gymnasium 环境 id 或论文短名 (ref_env 提供时以 ref_env.env_id 为准)
            num_envs: 并行 env 数 B = worker 进程数
            horizon:  一段 rollout 步长 T (段截断长度)
            device:   torch 设备 (张量搬到此设备供 GPU 网络)
            ref_env:  SafetyEnv 实例 (参数来源: env_id/维度; 避免子进程起来前无维度信息)
            seed:     基础种子 (第 i 个 worker 的 env 用 seed+i)
        """
        self.env_id = ref_env.env_id if ref_env is not None else str(env_id)
        self.B = int(num_envs)
        self.n = int(horizon)
        self.device = device if device is not None else torch.device('cpu')
        self._seed = int(seed)

        # 维度信息: 有 ref_env 直接用; 否则临时建一个探测 (仅父进程, 用后即弃)
        if ref_env is not None:
            self.obs_dim, self.act_dim = ref_env.obs_dim, ref_env.act_dim
        else:
            from .safety_env import SafetyEnv
            _probe = SafetyEnv(self.env_id)
            self.obs_dim, self.act_dim = _probe.obs_dim, _probe.act_dim
            _probe.close()
        self.observation_space = _Space((self.obs_dim,))
        self.action_space = _Space((self.act_dim,))

        # spawn B 个 worker (spawn: 干净子进程, 不继承父进程 CUDA/EGL 状态)
        ctx = mp.get_context('spawn')
        self._remotes, self._procs = [], []
        for i in range(self.B):
            remote, work_remote = ctx.Pipe()
            p = ctx.Process(target=_worker,
                            args=(work_remote, remote, self.env_id, self._seed + i),
                            daemon=True)
            p.start()
            work_remote.close()
            self._remotes.append(remote)
            self._procs.append(p)
        self._first_reset = True                             # 首次 reset 用确定种子, 之后随机

        self.step_count = 0                                  # 当前段内步数 (B 路锁步)
        self._cost_buf = []                                  # 每步 batch 平均 cost (render 日志)
        self._creward_buf = []                               # 每步 batch 平均 reward (日志)

    def reset(self):
        """重置全部 B 个 env, 返回 [B, obs_dim] (device 张量)。每段开始调用一次。"""
        self.step_count = 0
        self._cost_buf, self._creward_buf = [], []
        for i, r in enumerate(self._remotes):                # 广播 (首段确定种子去相关)
            r.send(('reset', self._seed + i if self._first_reset else None))
        self._first_reset = False
        obs = np.stack([r.recv() for r in self._remotes], axis=0)   # [B, obs_dim]
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device)

    def step(self, actions):
        """
        actions: [B, act_dim] torch。广播动作 → B 个 worker 并行 step → 回收。
        返回 (next_obs [B,obs_dim], reward [B], cost [B], done:bool); 语义与 SafetyVecEnv 一致。
        """
        a_np = actions.detach().cpu().numpy().astype(np.float32)     # [B, act_dim]
        for i, r in enumerate(self._remotes):                # 先全部发送 (并行窗口)
            r.send(('step', a_np[i]))
        obs_l, r_l, c_l = [], [], []
        for r in self._remotes:                              # 再统一回收
            o, rew, c = r.recv()
            obs_l.append(o); r_l.append(rew); c_l.append(c)

        self.step_count += 1
        self._creward_buf.append(float(np.mean(r_l)))        # batch 平均 reward
        self._cost_buf.append(float(np.mean(c_l)))           # batch 平均 cost (风险水平代理)
        done = (self.step_count == self.n)                   # 段截断 (非环境终止)

        obs = torch.as_tensor(np.stack(obs_l, axis=0), dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(np.asarray(r_l, dtype=np.float32), device=self.device)
        cost = torch.as_tensor(np.asarray(c_l, dtype=np.float32), device=self.device)
        return obs, reward, cost, done

    def render(self, mode=None):
        """返回本段每步 batch 平均 cost 序列 [T] (numpy) —— 与 SafetyVecEnv 一致。"""
        if not self._cost_buf:
            return None
        return np.asarray(self._cost_buf, dtype=np.float64)

    def stats(self):
        """附加统计 (本段 batch 平均单步 reward/cost), 供智能体日志。"""
        out = {}
        if self._cost_buf:
            out['avg_step_cost'] = float(np.mean(self._cost_buf))
        if self._creward_buf:
            out['avg_step_reward'] = float(np.mean(self._creward_buf))
        return out

    def close(self):
        """通知全部 worker 退出并回收进程。"""
        for r in self._remotes:
            try:
                r.send(('close', None))
                r.recv()
                r.close()
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


def make_vec_env(env_id, num_envs, horizon, device, ref_env=None, seed=0, backend='mp'):
    """
    统一工厂: backend='mp' (默认, 多进程并行) / 'sync' (单进程串行, 调试/对拍用)。
    两后端接口与语义完全一致 (同 seed 下逐 env 轨迹一致, 见 _probe_vec_equiv.py)。
    """
    if str(backend).lower() == 'sync':
        from .safety_env_vec import SafetyVecEnv
        return SafetyVecEnv(env_id, num_envs=num_envs, horizon=horizon,
                            device=device, ref_env=ref_env, seed=seed)
    return SafetyVecEnvMP(env_id, num_envs=num_envs, horizon=horizon,
                          device=device, ref_env=ref_env, seed=seed)
