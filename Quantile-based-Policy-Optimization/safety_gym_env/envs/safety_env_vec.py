# -*- coding: utf-8 -*-
"""
SafetyVecEnv —— B 路 CPU 同步向量化 safety-gymnasium 环境 (供 GPU 智能体 rollout 复用)。

与 portfolio_env_inf/envs/portfolio_env_torch.py::PortfolioVecTorch 的关系:
    【接口对齐, 后端不同】。PortfolioVecTorch 是全 GPU 张量物理 (可 B 并行);
    safety-gym 是 mujoco CPU 物理, 无法 GPU 并行 ⇒ 本类用 B 个 SafetyEnv 的
    【同步 for 循环】逐个 step (对应用户早年"CPU VecEnv 50-worker"计划), 再把结果
    搬成 device 张量供 GPU 上的策略/critic 使用。GPU 只跑网络与更新, 不跑物理。

关键差异 (相对 PortfolioVecTorch):
    1. step 额外返回 **cost** (CMDP: 目标用 reward, 约束用 cost)。
    2. 段长 horizon 由【本类】统一控制: step_count 到 horizon → done=True (段截断);
       下一次 reset 重开全部 B 个 env。对 safety-gym 点机器人任务 (reach-goal 不终止、
       默认 1000 步截断), horizon≤1000 时段内一般不触发 env 自身终止 (罕见 terminated
       兜底: 段内 reset 该 env)。horizon=1000 ⇒ 一段=一整 episode (C=整段折扣 cost 回报)。
    3. 返回 torch 张量 (搬到 device)。

接口: reset()->obs[B,obs_dim];  step(a[B,act_dim])->(obs[B,obs_dim], reward[B], cost[B], done:bool)。
"""
import numpy as np
import torch

from .safety_env import SafetyEnv


class _Space:
    """轻量 shape 容器 (使 np.prod(env.observation_space.shape) 可用)。"""

    def __init__(self, shape):
        self.shape = tuple(shape)


class SafetyVecEnv:
    """B 路 CPU 同步向量化 safety-gymnasium 环境 (返回 device 张量, 含 cost)。"""

    def __init__(self, env_id='SafetyPointGoal1-v0', num_envs=8, horizon=1000,
                 device=None, ref_env=None, seed=0):
        """
        Args:
            env_id:   safety-gymnasium 环境 id (ref_env 提供时以 ref_env.env_id 为准, 保证同分布)
            num_envs: 并行 env 数 B (CPU 同步, 别开太大: mujoco 逐个 step)
            horizon:  一段 rollout 步长 T (段截断长度; =1000 时一段=一 episode, 对得上论文口径)
            device:   torch 设备 (物理在 CPU, 张量搬到此设备供 GPU 网络)
            ref_env:  SafetyEnv 实例 (参数来源: env_id)
            seed:     基础种子 (第 i 个 env 用 seed+i)
        """
        self.env_id = ref_env.env_id if ref_env is not None else str(env_id)
        self.B = int(num_envs)                              # 并行 env 数
        self.n = int(horizon)                               # 段长 T
        self.device = device if device is not None else torch.device('cpu')
        # B 个独立 SafetyEnv (不同种子 → 去相关的初始布局)
        self.envs = [SafetyEnv(self.env_id, seed=seed + i) for i in range(self.B)]
        self.obs_dim = self.envs[0].obs_dim                 # 观测维度 (60/76)
        self.act_dim = self.envs[0].act_dim                 # 动作维度 (2)
        self.observation_space = _Space((self.obs_dim,))
        self.action_space = _Space((self.act_dim,))

        self.step_count = 0                                 # 当前段内步数 (B 路锁步)
        self._cost_buf = []                                 # 每步 batch 平均 cost (render 日志)
        self._creward_buf = []                              # 每步 batch 平均 reward (日志)

    def reset(self):
        """重置全部 B 个 env, 返回 [B, obs_dim] (device 张量)。每段开始调用一次。"""
        self.step_count = 0
        self._cost_buf, self._creward_buf = [], []
        obs = np.stack([e.reset()[0] for e in self.envs], axis=0)   # [B, obs_dim]
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device)

    def step(self, actions):
        """
        actions: [B, act_dim] torch。同步逐个 step B 个 env。
        返回 (next_obs [B,obs_dim], reward [B], cost [B], done:bool)。
          - reward/cost 为本步各 env 的标量, 堆成 [B];
          - done = (段内步数==horizon) 的段截断标志 (统一由本类控制 horizon);
          - 段内若某 env 自身 terminated/truncated (点机器人任务罕见), 立即 reset 该 env 兜底
            (返回其新初始 obs, 保持 [B] 结构对齐; 段截断 bootstrap 语义见 vec_base)。
        """
        a_np = actions.detach().cpu().numpy().astype(np.float32)     # [B, act_dim]
        obs_l, r_l, c_l = [], [], []
        for i, e in enumerate(self.envs):
            o, r, c, term, trunc, _info = e.step(a_np[i])
            if term or trunc:                               # env 自身终止/截断 (罕见) → 重开兜底
                o = e.reset()[0]
            obs_l.append(o); r_l.append(r); c_l.append(c)

        self.step_count += 1
        self._creward_buf.append(float(np.mean(r_l)))       # batch 平均 reward
        self._cost_buf.append(float(np.mean(c_l)))          # batch 平均 cost (风险水平代理)
        done = (self.step_count == self.n)                  # 段截断 (非环境终止)

        obs = torch.as_tensor(np.stack(obs_l, axis=0), dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(np.asarray(r_l, dtype=np.float32), device=self.device)
        cost = torch.as_tensor(np.asarray(c_l, dtype=np.float32), device=self.device)
        return obs, reward, cost, done

    def render(self, mode=None):
        """返回本段每步 batch 平均 cost 序列 [T] (numpy) —— "风险水平"代理, 供统一日志。"""
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
        """释放全部 mujoco 资源。"""
        for e in self.envs:
            e.close()
