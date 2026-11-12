# -*- coding: utf-8 -*-
"""
SafetyEnv —— safety-gymnasium 点机器人【安全约束】环境的薄封装 (numpy 单 env 参考版)。

为什么要这层封装 (对齐 portfolio_env_inf/envs/portfolio_env.py 的"numpy 参考 env"角色):
    NIPS'22 QCPO 的 SimpleButton/Dynamic/Gremlin 环境 = safety-gymnasium 的点机器人
    Goal/Button 任务 (连续 2D 动作、mujoco 物理、lidar 观测)。本项目算法栈 (QCPO/DQCAC)
    要的是统一的 (obs, reward, cost, done) 接口 + 维度/空间信息, 因此这里只做【薄封装】:
      1. 统一 reset/step 返回 (cost 从 safety-gymnasium 的 6 元组第 3 位取标量);
      2. 暴露 observation_space / action_space / obs_dim / act_dim / n(建议 horizon);
      3. 动作 clip 到 [-1,1] (env 期望值域; 高斯策略 μ=tanh 已在 (-1,1), 加噪后 clip 兜底);
      4. 作为 SafetyVecEnv 的构造参数来源 + numpy 串行评估参考 (交叉校验向量化路径)。
    不重写任何物理 (与 portfolio 需自写 GBM 不同) —— 直接用 safety_gymnasium.make。

本目录当前研究对象是 reward-side 风险约束：max E[G] s.t. P(G<=q)<=alpha。
    reward r 构成唯一训练随机变量 G；Safety Gym cost 仍按原生 API 返回，但只允许进入
    diagnostic/environment_cost_* 日志，不进入 observation、loss、target、dual 或模型选择。
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')                  # headless: EGL 无需显示 (osmesa 兜底)

import numpy as np
import safety_gymnasium                                     # import 即注册 Safety* 环境到 gymnasium

from . import paper_envs                                    # 注册论文 config0-3 复刻环境
from .paper_envs import PAPER_ENV_IDS                       # 短名 (SimpleButton 等) → env id


class SafetyEnv:
    """safety-gymnasium 点机器人安全约束环境的薄封装 (numpy 单 env)。"""

    def __init__(self, env_id='SafetyPointGoal1-v0', seed=None):
        """
        Args:
            env_id: safety-gymnasium 环境 id。最小 pipeline 候选:
                    'SafetyPointGoal1-v0'   (goal+hazards, obs 60)  —— 对应论文 Simple/Dynamic 族
                    'SafetyPointButton1-v0' (buttons+hazards+gremlins, obs 76) —— 对应 Button/Gremlin 族
            seed:   默认随机种子 (reset 未显式给 seed 时用)
        """
        # 论文短名 (SimpleButton/Dynamic/Gremlin/DynamicButton) → 注册的 env id; 其余原样
        self.env_id = PAPER_ENV_IDS.get(str(env_id), str(env_id))
        self.env = safety_gymnasium.make(self.env_id)       # 真实 mujoco 环境 (6 元组 step)
        self.observation_space = self.env.observation_space # gymnasium Box (连续观测)
        self.action_space = self.env.action_space           # gymnasium Box [-1,1]^2 (连续 2D 动作)
        self.obs_dim = int(np.prod(self.observation_space.shape))   # 观测维度 (60 / 76)
        self.act_dim = int(np.prod(self.action_space.shape))        # 动作维度 (2)
        # 建议 horizon: safety-gym 点机器人任务默认 1000 步截断 (reach-goal 不终止, goal 重定位);
        # 段长由 VecEnv 侧统一控制, 这里仅暴露默认值供评估/日志参考。
        self.n = int(getattr(self.env.unwrapped, 'num_steps', 1000) or 1000)
        self._seed = seed                                   # 默认种子
        # 动作值域 (clip 兜底用), 转 float32
        self._act_low = np.asarray(self.action_space.low, dtype=np.float32)
        self._act_high = np.asarray(self.action_space.high, dtype=np.float32)

    def reset(self, seed=None):
        """重置: 返回 (obs[obs_dim] float32, info)。gymnasium 5 值 reset 语义。"""
        obs, info = self.env.reset(seed=seed if seed is not None else self._seed)
        self._seed = None                                   # 种子只用一次 (后续 reset 随机)
        return np.asarray(obs, dtype=np.float32).flatten(), info

    def clip_action(self, action):
        """
        将策略 latent action 映射为物理环境真正执行的 Box action。

        输入会转成扁平 float32 ``[act_dim]``；逐维 clip 使用 action_space 的原生上下界，
        而不是硬编码 ``[-1, 1]``，从而使 collector、同步/MP adapter 与底层环境共享同一语义。
        """
        latent = np.asarray(action, dtype=np.float32).flatten()
        return np.clip(latent, self._act_low, self._act_high)

    def step(self, action):
        """
        一步交互。safety-gymnasium step 返回【6 元组】(obs, reward, cost, terminated, truncated, info)。
        本封装:
          - 动作 clip 到 [-1,1] (env 期望值域);
          - obs 转 float32 扁平; reward/cost 转标量 float。
        返回 (obs[obs_dim], reward:float, cost:float, terminated:bool, truncated:bool, info:dict)。
        """
        a = self.clip_action(action)                         # 高斯 latent → 实际 Box action
        obs, reward, cost, terminated, truncated, info = self.env.step(a)
        # Copy rather than mutate the native mapping in place: wrappers may retain/reference it.
        # ``executed_action`` is the ground truth consumed by vector adapters and DQC replay.
        info = dict(info)
        info["executed_action"] = a.copy()
        return (np.asarray(obs, dtype=np.float32).flatten(),
                float(reward), float(cost), bool(terminated), bool(truncated), info)

    def close(self):
        """释放 mujoco 资源。"""
        try:
            self.env.close()
        except Exception:
            pass
