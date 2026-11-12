# -*- coding: utf-8 -*-
"""
SafetyEnv —— safety-gymnasium 点机器人【安全约束】环境的薄封装 (numpy 单 env 参考版)。

为什么要这层封装 (对齐 portfolio_env_inf/envs/portfolio_env.py 的"numpy 参考 env"角色):
    NIPS'22 QCPO 的 SimpleButton/Dynamic/Gremlin 环境 = safety-gymnasium 的点机器人
    Goal/Button 任务 (连续 2D 动作、mujoco 物理、lidar 观测)。本项目算法栈 (QCPO/DQCAC)
    要的是统一的 (obs, reward, cost, done) 接口 + 维度/空间信息, 因此这里只做【薄封装】:
      1. 统一 reset/step 返回 (cost 从 safety-gymnasium 的 6 元组第 3 位取标量);
      2. 暴露 observation_space / action_space / obs_dim / act_dim / n(建议 horizon);
      3. 显式声明 action forwarding 语义：默认由本 wrapper 做 Box clip；作者对齐实验则把
         float32 扁平 raw Gaussian action 原样交给 native env，让其内部 actuator clamp 生效；
      4. 作为 SafetyVecEnv 的构造参数来源 + numpy 串行评估参考 (交叉校验向量化路径)。
    不重写任何物理 (与 portfolio 需自写 GBM 不同) —— 直接用 safety_gymnasium.make。

本目录当前研究对象是 reward-side 风险约束：max E[G] s.t. P(G<=q)<=alpha。
    reward r 构成唯一训练随机变量 G；Safety Gym cost 仍按原生 API 返回，但只允许进入
    diagnostic/environment_cost_* 日志，不进入 observation、loss、target、dual 或模型选择。
"""
import copy
import os
os.environ.setdefault('MUJOCO_GL', 'egl')                  # headless: EGL 无需显示 (osmesa 兜底)

import mujoco                                                   # Select one explicit native integrator profile.
import numpy as np
import safety_gymnasium                                     # import 即注册 Safety* 环境到 gymnasium

from . import paper_envs                                    # 注册论文 config0-3 复刻环境
from .paper_envs import PAPER_ENV_IDS                       # 短名 (SimpleButton 等) → env id


class SafetyEnv:
    """safety-gymnasium 点机器人安全约束环境的薄封装 (numpy 单 env)。"""

    # String identities are stable and checkpoint-signable; a bool cannot distinguish a missing
    # legacy default from an explicitly frozen author-compatible forwarding contract.
    EXPLICIT_BOX_CLIP = "explicit_box_clip"
    AUTHOR_RAW_ENV_CLAMP = "author_raw_env_clamp"
    ACTION_MODES = frozenset((EXPLICIT_BOX_CLIP, AUTHOR_RAW_ENV_CLAMP))

    # Point's low-inertia yaw velocity actuator is discretely unstable under the
    # upstream explicit-Euler default.  Tiny float64/float32 backend differences
    # enter different bounded chatter phases and corrupt accelerometer/gyro parity.
    # The fixed profile changes only the integrator; the historical profile remains
    # available under an explicit, checkpoint-visible identity.
    LEGACY_EULER = "legacy_euler_v1"
    STABLE_IMPLICITFAST = "point_velocity_implicitfast_v1"
    DEFAULT_DYNAMICS_PROFILE = LEGACY_EULER
    DYNAMICS_PROFILES = frozenset((LEGACY_EULER, STABLE_IMPLICITFAST))

    def __init__(self, env_id='SafetyPointSimpleButton0-v0', seed=None,
                 action_mode=EXPLICIT_BOX_CLIP,
                 dynamics_profile=DEFAULT_DYNAMICS_PROFILE):
        """
        Args:
            env_id: safety-gymnasium 环境 id。最小 pipeline 候选:
                    'SafetyPointGoal1-v0'   (goal+hazards, obs 60)  —— 对应论文 Simple/Dynamic 族
                    'SafetyPointButton1-v0' (buttons+hazards+gremlins, obs 76) —— 对应 Button/Gremlin 族
            seed:   默认随机种子 (reset 未显式给 seed 时用)
            action_mode:
                    ``explicit_box_clip`` 在本 wrapper 按原生 Box bounds clip（历史默认）；
                    ``author_raw_env_clamp`` 仅做 float32+flatten，复刻作者 sampler 将
                    raw Gaussian action 交给环境、由 native dynamics clamp 的边界位置。
            dynamics_profile:
                    ``point_velocity_implicitfast_v1`` 使用 CPU/MJWarp 均支持的
                    implicit-fast 积分，避免 Point yaw 速度伺服放大后端舍入差异；
                    ``legacy_euler_v1`` 仅用于复现历史 CPU MDP。
        """
        # 论文短名 (SimpleButton/Dynamic/Gremlin/DynamicButton) → 注册的 env id; 其余原样
        self.env_id = PAPER_ENV_IDS.get(str(env_id), str(env_id))
        if self.env_id != 'SafetyPointDynamic0-v0':
            raise ValueError('This experiment supports only Dynamic')
        self.dynamics_profile = str(dynamics_profile)
        if self.dynamics_profile not in self.DYNAMICS_PROFILES:
            raise ValueError(
                f"unsupported dynamics_profile={self.dynamics_profile!r}; "
                f"expected one of {sorted(self.DYNAMICS_PROFILES)}")
        self.env = safety_gymnasium.make(self.env_id)       # 真实 mujoco 环境 (6 元组 step)
        self.observation_space = self.env.observation_space # gymnasium Box (连续观测)
        self.action_space = self.env.action_space           # gymnasium Box [-1,1]^2 (连续 2D 动作)
        self.obs_dim = int(np.prod(self.observation_space.shape))   # 观测维度 (60 / 76)
        self.act_dim = int(np.prod(self.action_space.shape))        # 动作维度 (2)
        # 建议 horizon: safety-gym 点机器人任务默认 1000 步截断 (reach-goal 不终止, goal 重定位);
        # 段长由 VecEnv 侧统一控制, 这里仅暴露默认值供评估/日志参考。
        self.n = int(getattr(self.env.unwrapped, 'num_steps', 1000) or 1000)
        self._seed = seed                                   # 默认种子
        self.action_mode = str(action_mode)
        if self.action_mode not in self.ACTION_MODES:
            raise ValueError(
                f"unsupported action_mode={self.action_mode!r}; "
                f"expected one of {sorted(self.ACTION_MODES)}")
        # 动作值域 (clip 兜底用), 转 float32
        self._act_low = np.asarray(self.action_space.low, dtype=np.float32)
        self._act_high = np.asarray(self.action_space.high, dtype=np.float32)

    def _apply_dynamics_profile(self):
        """Install the selected integrator after native reset compiles/rebuilds mjModel.

        Safety-Gymnasium creates ``task.model`` lazily during the first reset and may
        rebuild it on later resets.  Applying the profile at this boundary guarantees
        that every following physics step uses the signed MDP identity.
        """
        task = self.env.unwrapped.task
        integrator = (
            mujoco.mjtIntegrator.mjINT_EULER
            if self.dynamics_profile == self.LEGACY_EULER
            else mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        )
        task.model.opt.integrator = integrator

    def reset(self, seed=None):
        """重置: 返回 (obs[obs_dim] float32, info)。gymnasium 5 值 reset 语义。"""
        obs, info = self.env.reset(seed=seed if seed is not None else self._seed)
        self._apply_dynamics_profile()                        # Configure the compiled model before step().
        self._seed = None                                   # 种子只用一次 (后续 reset 随机)
        return np.asarray(obs, dtype=np.float32).flatten(), info

    def get_rng_state(self):
        """Deep-copy every RNG that can affect a later native reset or sampled action.

        Safety-Gymnasium's Builder uses ``task.random_generator.random_generator`` (legacy
        ``RandomState``) for layouts/goals/gremlins, rather than Gymnasium's ``np_random``.
        Both are retained: the task state is physically authoritative today, while the native
        Generator and action-space Generator keep the wrapper correct for other registered envs.
        """
        unwrapped = self.env.unwrapped
        native_rng = getattr(unwrapped, "np_random", None)
        action_rng = getattr(self.action_space, "np_random", None)
        task_holder = getattr(getattr(unwrapped, "task", None), "random_generator", None)
        task_rng = getattr(task_holder, "random_generator", None)
        return {
            "version": 1,
            "backend": "cpu",
            "action_mode": self.action_mode,
            # Objects restored/constructed by pre-migration fixtures have no
            # attribute and necessarily represent the historical Euler MDP.
            "dynamics_profile": getattr(
                self, "dynamics_profile", self.LEGACY_EULER),
            "wrapper_seed": copy.deepcopy(self._seed),
            "native_seed": copy.deepcopy(getattr(unwrapped, "_seed", None)),
            "native_np_random": (
                None if native_rng is None or not hasattr(native_rng, "bit_generator")
                else copy.deepcopy(native_rng.bit_generator.state)),
            "action_space_np_random": (
                None if action_rng is None or not hasattr(action_rng, "bit_generator")
                else copy.deepcopy(action_rng.bit_generator.state)),
            "task_random_state": (
                None if task_rng is None or not hasattr(task_rng, "get_state")
                else copy.deepcopy(task_rng.get_state())),
        }

    def set_rng_state(self, state):
        """Restore a state produced by :meth:`get_rng_state` without performing a reset."""
        if not isinstance(state, dict) or int(state.get("version", -1)) != 1:
            raise ValueError("SafetyEnv RNG state must be a version-1 mapping")
        if str(state.get("backend", "cpu")) != "cpu":
            raise ValueError("cannot restore a non-CPU RNG state into SafetyEnv")
        saved_mode = str(state.get("action_mode", ""))
        if saved_mode != self.action_mode:
            raise ValueError(
                f"SafetyEnv RNG action_mode mismatch: {saved_mode!r} != {self.action_mode!r}")
        # Version-1 states written before the GPU migration have no profile field
        # and unambiguously belong to the historical Euler MDP.
        saved_profile = str(state.get("dynamics_profile", self.LEGACY_EULER))
        current_profile = str(getattr(
            self, "dynamics_profile", self.LEGACY_EULER))
        if saved_profile != current_profile:
            raise ValueError(
                "SafetyEnv RNG dynamics profile mismatch: "
                f"{saved_profile!r} != {current_profile!r}")
        unwrapped = self.env.unwrapped
        self._seed = copy.deepcopy(state.get("wrapper_seed"))
        if hasattr(unwrapped, "_seed"):
            unwrapped._seed = copy.deepcopy(state.get("native_seed"))

        native_state = state.get("native_np_random")
        if native_state is not None:
            native_rng = getattr(unwrapped, "np_random", None)
            if native_rng is None or not hasattr(native_rng, "bit_generator"):
                raise ValueError("native environment cannot restore a Generator RNG state")
            native_rng.bit_generator.state = copy.deepcopy(native_state)

        action_state = state.get("action_space_np_random")
        if action_state is not None:
            action_rng = getattr(self.action_space, "np_random", None)
            if action_rng is None or not hasattr(action_rng, "bit_generator"):
                raise ValueError("action space cannot restore a Generator RNG state")
            action_rng.bit_generator.state = copy.deepcopy(action_state)

        task_state = state.get("task_random_state")
        if task_state is not None:
            task_holder = getattr(getattr(unwrapped, "task", None), "random_generator", None)
            if task_holder is None:
                raise ValueError("native environment cannot restore its task RandomState")
            task_rng = getattr(task_holder, "random_generator", None)
            if task_rng is None:
                task_rng = np.random.RandomState()
                task_holder.random_generator = task_rng
            if not hasattr(task_rng, "set_state"):
                raise ValueError("task random generator lacks RandomState.set_state")
            task_rng.set_state(copy.deepcopy(task_state))

    def clip_action(self, action):
        """
        按冻结的 action mode 准备传给 native environment 的动作。

        两种模式都先转成扁平 float32 ``[act_dim]``。历史模式逐维使用 action_space 原生
        bounds clip；作者模式不在本层 clip，使 raw action 到达 native actuator clamp。

        方法名为兼容已有 collector 保留；作者模式的返回值有意不是 Box-clipped action。
        """
        latent = np.asarray(action, dtype=np.float32).flatten()
        if self.action_mode == self.AUTHOR_RAW_ENV_CLAMP:
            return latent
        return np.clip(latent, self._act_low, self._act_high)

    def step(self, action):
        """
        一步交互。safety-gymnasium step 返回【6 元组】(obs, reward, cost, terminated, truncated, info)。
        本封装:
          - 按 action_mode 在 wrapper Box clip，或 raw-forward 后由 native env clamp;
          - obs 转 float32 扁平; reward/cost 转标量 float。
        返回 (obs[obs_dim], reward:float, cost:float, terminated:bool, truncated:bool, info:dict)。
        """
        a = self.clip_action(action)                         # 高斯 raw → 冻结边界位置后的 native input
        obs, reward, cost, terminated, truncated, info = self.env.step(a)
        # Copy rather than mutate the native mapping in place: wrappers may retain/reference it.
        # ``executed_action`` is this wrapper's exact forwarded array: post-clip in the legacy
        # mode, intentionally raw in author mode because PPO likelihood/history use the raw draw.
        # The native environment remains the physical actuator clamp authority in author mode.
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
