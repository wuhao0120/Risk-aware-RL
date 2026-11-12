# -*- coding: utf-8 -*-
"""
GPU-native Gremlin vector environment backed by MuJoCo Warp.

This module is intentionally imported lazily by ``make_vec_env``.  The historical
CPU environment is pinned to MuJoCo 2.3.3, while MuJoCo Warp requires a newer
MuJoCo release; keeping the import optional lets old checkpoints and CPU campaigns
continue to run in their frozen environment.

The steady-state step path stays on CUDA:
    policy action -> actuator clamp -> 10 MuJoCo Warp substeps -> reward/cost ->
    pseudo-lidar -> ``VecStepResult``.

Only process control and logging remain on the host.  Layout reset, goal resampling,
Gremlin mocap motion, physics, observations, rewards, and diagnostic costs all run on
the selected CUDA device.  The simulator keeps the existing ``VecStepResult`` API so
agents do not contain backend-specific mathematics.

Numerical contract:
    - classic MuJoCo uses float64 internally;
    - MuJoCo Warp uses float32 and therefore is not bitwise trajectory-equivalent;
    - reset distributions and task formulas match the Gremlin task, but Warp owns a
      counter-based per-world RNG rather than NumPy ``RandomState``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

try:
    import mujoco_warp as mjw                         # GPU MuJoCo implementation
    import warp as wp                                 # CUDA kernels + Torch zero-copy views
except ImportError as exc:                            # Lazy import produces an actionable error
    raise ImportError(
        "vec_backend=mjwarp requires an isolated environment with "
        "mujoco-warp==3.12.0 and warp-lang==1.16.0"
    ) from exc

from .safety_env import SafetyEnv
from .safety_env_vec import VecStepResult


# Gremlin paper task constants.  Keeping these next to the kernels makes the GPU
# implementation auditable against ``paper_envs.GremlinLevel0`` and upstream
# Safety-Gymnasium's ``RandomGenerator.sample_layout``.
_NUM_LAYOUT_OBJECTS = 10                              # agent, goal, 5 hazards, 3 gremlins
_AGENT_LAYOUT_INDEX = 0
_GOAL_LAYOUT_INDEX = 1
_FIRST_HAZARD_LAYOUT_INDEX = 2
_FIRST_GREMLIN_LAYOUT_INDEX = 7
_NUM_HAZARDS = 5
_NUM_GREMLINS = 3
_EXTENT = 2.0
_AGENT_KEEPOUT = 0.4
_GOAL_KEEPOUT = 0.305
_HAZARD_KEEPOUT = 0.305
_GREMLIN_KEEPOUT = 0.4
_GOAL_SIZE = 0.3
_HAZARD_SIZE = 0.3
_GREMLIN_TRAVEL = 0.35
_GREMLIN_Z = 0.1
_LIDAR_BINS = 16
_LIDAR_MAX_DIST = 5.0
_PHYSICS_SUBSTEPS = 10


@wp.func
def _layout_keepout(index: int) -> float:
    """Return the Safety-Gymnasium placement keepout for one layout object."""
    keepout = _HAZARD_KEEPOUT
    if index == _AGENT_LAYOUT_INDEX:
        keepout = _AGENT_KEEPOUT
    elif index == _GOAL_LAYOUT_INDEX:
        keepout = _GOAL_KEEPOUT
    elif index >= _FIRST_GREMLIN_LAYOUT_INDEX:
        keepout = _GREMLIN_KEEPOUT
    return keepout


@wp.func
def _candidate_is_valid(
    world: int,
    candidate: wp.vec2,
    candidate_keepout: float,
    placed_count: int,
    skip_index: int,
    layout: wp.array2d(dtype=wp.vec2),
) -> bool:
    """Apply the upstream pairwise Euclidean keepout test on device."""
    valid = bool(True)
    for other in range(_NUM_LAYOUT_OBJECTS):
        # ``placed_count`` limits initial sequential layout construction.  During
        # goal resampling it is 10 and ``skip_index`` excludes the old goal.
        if other < placed_count and other != skip_index:
            required = candidate_keepout + _layout_keepout(other)
            if wp.length(candidate - layout[world, other]) < required:
                valid = bool(False)
    return valid


@wp.kernel
def _sample_layout_kernel(
    base_seed: int,
    reset_mask: wp.array(dtype=wp.bool),
    reset_count: wp.array(dtype=wp.int32),
    goal_count: wp.array(dtype=wp.int32),
    layout: wp.array2d(dtype=wp.vec2),
    rotations: wp.array2d(dtype=float),
    failures: wp.array(dtype=wp.int32),
):
    """Sample a complete Gremlin layout independently for every selected world.

    The nested retry structure mirrors Safety-Gymnasium: each object gets 100
    attempts; a conflicted object restarts the complete layout; a final goal is
    then resampled after all other objects have been placed.
    """
    world = wp.tid()
    if not reset_mask[world]:
        return

    # The world index is the PCG stream offset.  ``reset_count`` selects a new,
    # deterministic substream after every reset and is checkpointed by the adapter.
    launch_seed = base_seed + reset_count[world] * 104729
    rng = wp.rand_init(launch_seed, world)
    failures[world] = 0
    layout_found = bool(False)

    for _layout_attempt in range(10000):
        layout_ok = bool(True)
        for obj in range(_NUM_LAYOUT_OBJECTS):
            keepout = _layout_keepout(obj)
            low = -_EXTENT + keepout
            high = _EXTENT - keepout
            object_found = bool(False)

            for _object_attempt in range(100):
                candidate = wp.vec2(
                    wp.randf(rng, low, high),
                    wp.randf(rng, low, high),
                )
                if _candidate_is_valid(
                    world, candidate, keepout, obj, -1, layout
                ):
                    layout[world, obj] = candidate
                    object_found = bool(True)
                    break

            if not object_found:
                layout_ok = bool(False)
                break

        if layout_ok:
            layout_found = bool(True)
            break

    if not layout_found:
        failures[world] = 1
        return

    # Upstream world construction draws independent rotations after placement.
    # Only the agent and contact-capable gremlins affect dynamics, but retaining
    # all ten angles keeps one explicit task-state representation.
    for obj in range(_NUM_LAYOUT_OBJECTS):
        rotations[world, obj] = wp.randf(rng, 0.0, 2.0 * math.pi)

    # GoalLevel0.update_world deletes the initially placed goal and draws it again
    # against the current agent/hazard/gremlin layout.  Reproduce that final state.
    goal_found = bool(False)
    goal_low = -_EXTENT + _GOAL_KEEPOUT
    goal_high = _EXTENT - _GOAL_KEEPOUT
    for _goal_attempt in range(10000):
        candidate = wp.vec2(
            wp.randf(rng, goal_low, goal_high),
            wp.randf(rng, goal_low, goal_high),
        )
        if _candidate_is_valid(
            world,
            candidate,
            _GOAL_KEEPOUT,
            _NUM_LAYOUT_OBJECTS,
            _GOAL_LAYOUT_INDEX,
            layout,
        ):
            layout[world, _GOAL_LAYOUT_INDEX] = candidate
            goal_found = bool(True)
            break

    if not goal_found:
        failures[world] = 1
        return

    # A successful reset advances this world's counter exactly once and starts a
    # fresh per-episode goal-event counter.
    reset_count[world] = reset_count[world] + 1
    goal_count[world] = 0


@wp.kernel
def _resample_goal_kernel(
    base_seed: int,
    reached: wp.array(dtype=wp.bool),
    reset_count: wp.array(dtype=wp.int32),
    goal_count: wp.array(dtype=wp.int32),
    layout: wp.array2d(dtype=wp.vec2),
    failures: wp.array(dtype=wp.int32),
):
    """Resample reached goals on CUDA without synchronizing through the host."""
    world = wp.tid()
    # A retry-budget failure is a native terminal.  Keep its flag sticky until
    # the rollout boundary so inactive slots cannot silently resume an old goal.
    if not reached[world] or failures[world] != 0:
        return
    failures[world] = 0

    # Mix reset and goal counters into a stable per-world PCG substream.  Batch
    # size changes do not alter any existing slot's random sequence.
    launch_seed = (
        base_seed
        + reset_count[world] * 104729
        + goal_count[world] * 130363
        + 7919
    )
    rng = wp.rand_init(launch_seed, world)
    goal_low = -_EXTENT + _GOAL_KEEPOUT
    goal_high = _EXTENT - _GOAL_KEEPOUT
    goal_found = bool(False)

    for _attempt in range(10000):
        candidate = wp.vec2(
            wp.randf(rng, goal_low, goal_high),
            wp.randf(rng, goal_low, goal_high),
        )
        if _candidate_is_valid(
            world,
            candidate,
            _GOAL_KEEPOUT,
            _NUM_LAYOUT_OBJECTS,
            _GOAL_LAYOUT_INDEX,
            layout,
        ):
            layout[world, _GOAL_LAYOUT_INDEX] = candidate
            goal_found = bool(True)
            break

    if not goal_found:
        failures[world] = 1
        return

    goal_count[world] = goal_count[world] + 1


@wp.kernel
def _move_gremlins_kernel(
    time: wp.array(dtype=float),
    mocap_pos: wp.array2d(dtype=wp.vec3),
):
    """Apply the exact Gremlins.move circular mocap target before a physics step."""
    world, gremlin = wp.tid()
    phase = time[world]
    mocap_pos[world, gremlin] = wp.vec3(
        wp.sin(phase) * _GREMLIN_TRAVEL,
        wp.cos(phase) * _GREMLIN_TRAVEL,
        _GREMLIN_Z,
    )


class _Space:
    """Minimal Gym-compatible shape container used by the existing agents."""

    def __init__(self, shape):
        self.shape = tuple(shape)


@dataclass(frozen=True)
class _ModelIds:
    """Resolved MuJoCo indices; names are checked once instead of in the step loop."""

    agent_body: int
    goal_body: int
    hazard_bodies: tuple[int, ...]
    gremlin_bodies: tuple[int, ...]
    gremlin_mocap_bodies: tuple[int, ...]
    gremlin_mocap_geoms: tuple[int, ...]
    agent_geoms: tuple[int, ...]
    gremlin_geoms: tuple[int, ...]
    gremlin_qpos_addresses: tuple[int, ...]


class SafetyVecEnvMJWarp:
    """Batched, GPU-resident implementation of the paper Gremlin environment."""

    def __init__(
        self,
        env_id="SafetyPointGremlin0-v0",
        num_envs=32,
        horizon=1000,
        device=None,
        ref_env=None,
        seed=0,
        action_mode=None,
        dynamics_profile=None,
    ):
        """Compile one host model, then replicate its mutable fields across CUDA worlds."""
        self.env_id = ref_env.env_id if ref_env is not None else str(env_id)
        if "Gremlin" not in self.env_id:
            raise NotImplementedError(
                "vec_backend=mjwarp currently supports only the paper Gremlin environment"
            )

        # The backend is meaningful only when both learner and simulator share a
        # CUDA device; silently falling back to CPU would recreate the original bug.
        self.device = torch.device(device if device is not None else "cuda")
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("vec_backend=mjwarp requires an available CUDA device")
        self._last_caller_stream_t = torch.cuda.current_stream(self.device)

        self.B = int(num_envs)
        self.n = int(horizon)
        self._seed = int(seed)
        self.obs_dim, self.act_dim = 60, 2
        self.observation_space = _Space((self.obs_dim,))
        self.action_space = _Space((self.act_dim,))
        inherited_mode = (
            getattr(ref_env, "action_mode", SafetyEnv.EXPLICIT_BOX_CLIP)
            if ref_env is not None
            else SafetyEnv.EXPLICIT_BOX_CLIP
        )
        self.action_mode = str(inherited_mode if action_mode is None else action_mode)
        if self.action_mode not in SafetyEnv.ACTION_MODES:
            raise ValueError(f"unsupported action_mode={self.action_mode!r}")
        inherited_profile = (
            getattr(ref_env, "dynamics_profile", SafetyEnv.DEFAULT_DYNAMICS_PROFILE)
            if ref_env is not None else SafetyEnv.DEFAULT_DYNAMICS_PROFILE
        )
        if (ref_env is not None and dynamics_profile is not None
                and str(dynamics_profile) != str(inherited_profile)):
            raise ValueError(
                "ref_env/vector dynamics profile mismatch: "
                f"{inherited_profile!r} != {dynamics_profile!r}")
        self.dynamics_profile = str(
            inherited_profile if dynamics_profile is None else dynamics_profile
        )
        if self.dynamics_profile not in SafetyEnv.DYNAMICS_PROFILES:
            raise ValueError(f"unsupported dynamics_profile={self.dynamics_profile!r}")

        # Warp and Torch share the exact CUDA stream and memory through zero-copy
        # views.  All subsequent kernels are launched under this device context.
        wp.config.quiet = True
        wp.init()                                      # Populate Warp's CUDA device registry once.
        self._warp_device = wp.device_from_torch(self.device)
        self._cost_buf = []
        self._reward_buf = []
        self.step_count = 0

        # One CPU instance is used only as a model compiler/name oracle at startup;
        # no CPU SafetyEnv exists in the rollout or reset path.
        probe = SafetyEnv(
            self.env_id,
            seed=self._seed,
            action_mode=self.action_mode,
            dynamics_profile=self.dynamics_profile,
        )
        try:
            probe.reset(seed=self._seed)
            task = probe.env.unwrapped.task

            # This backend freezes the exact paper-task execution contract.  A
            # future Safety-Gymnasium configuration change must fail loudly rather
            # than silently alter frameskip, inject unimplemented action noise, or
            # switch the public cost from an indicator to continuous shaping.
            frameskip_is_supported = (
                int(task.sim_conf.frameskip_binom_n) == _PHYSICS_SUBSTEPS
                and float(task.sim_conf.frameskip_binom_p) == 1.0
            )
            if not frameskip_is_supported or float(task.action_noise) != 0.0:
                raise NotImplementedError(
                    "MJWarp Gremlin requires deterministic frameskip=10 and action_noise=0"
                )
            if not bool(task.cost_conf.constrain_indicator):
                raise NotImplementedError(
                    "MJWarp Gremlin currently implements constrain_indicator=True"
                )

            self._native_horizon = int(task.num_steps)       # Builder's own timeout boundary.
            host_model = task.model
            self._ids = self._resolve_ids(host_model)
            self._initialize_warp(host_model)
        finally:
            probe.close()

    @staticmethod
    def _resolve_ids(model):
        """Resolve every body/geom/joint index needed by the device task logic."""
        body_id = lambda name: int(model.body(name).id)
        geom_id = lambda name: int(model.geom(name).id)
        joint_qpos = lambda name: int(model.jnt_qposadr[model.joint(name).id])

        return _ModelIds(
            agent_body=body_id("agent"),
            goal_body=body_id("goal"),
            hazard_bodies=tuple(body_id(f"hazard{i}") for i in range(_NUM_HAZARDS)),
            gremlin_bodies=tuple(body_id(f"gremlin{i}obj") for i in range(_NUM_GREMLINS)),
            gremlin_mocap_bodies=tuple(
                body_id(f"gremlin{i}mocap") for i in range(_NUM_GREMLINS)
            ),
            gremlin_mocap_geoms=tuple(
                geom_id(f"gremlin{i}mocap") for i in range(_NUM_GREMLINS)
            ),
            agent_geoms=(geom_id("agent"), geom_id("pointarrow")),
            gremlin_geoms=tuple(geom_id(f"gremlin{i}obj") for i in range(_NUM_GREMLINS)),
            gremlin_qpos_addresses=tuple(
                joint_qpos(f"gremlin{i}obj") for i in range(_NUM_GREMLINS)
            ),
        )

    def _initialize_warp(self, host_model):
        """Create batched Model/Data objects and persistent Torch views on one CUDA stream."""
        # Mutable inputs and set_const_0 outputs need one leading row per world.
        # Shared topology/material arrays retain a single row and consume little memory.
        batched_fields = {
            name: self.B
            for name in (
                "body_pos", "body_quat", "body_ipos", "body_iquat",
                "geom_pos", "geom_quat", "eq_data", "qpos0", "qpos_spring",
                "body_invweight0", "dof_invweight0", "cam_pos0", "cam_poscom0",
                "cam_mat0", "light_pos0", "light_poscom0", "light_dir0",
                "actuator_acc0",
            )
        }

        with wp.ScopedDevice(self._warp_device):
            self._model = mjw.put_model(host_model, batch_sizes=batched_fields)
            # Empirically verified for Gremlin: <=12 contacts and <=138 constraints
            # per world.  Headroom avoids silent truncation while keeping B=2048 small.
            self._data = mjw.make_data(
                host_model,
                nworld=self.B,
                nconmax=32,
                njmax=256,
            )

            # Per-world task state and counter-based RNG clocks remain on CUDA.
            self._layout = wp.zeros(
                (self.B, _NUM_LAYOUT_OBJECTS), dtype=wp.vec2, device=self._warp_device
            )
            self._rotations = wp.zeros(
                (self.B, _NUM_LAYOUT_OBJECTS), dtype=float, device=self._warp_device
            )
            self._reset_count = wp.zeros(self.B, dtype=wp.int32, device=self._warp_device)
            self._goal_count = wp.zeros(self.B, dtype=wp.int32, device=self._warp_device)
            self._layout_failures = wp.zeros(
                self.B, dtype=wp.int32, device=self._warp_device
            )

            # A native Warp stream can own a CUDA Graph, while its Torch wrapper
            # supplies event-based dependencies to the learner's current stream.
            # This removes hundreds of Python/CUDA launches per environment step
            # without inserting a host synchronization between policy and physics.
            self._physics_stream_w = wp.get_stream(self._warp_device)
            self._physics_stream_t = wp.stream_to_torch(self._physics_stream_w)
            self._physics_graph = None

        # ``wp.to_torch`` is zero-copy: assignments below mutate Warp-owned arrays
        # on the current Torch CUDA stream without NumPy or PCIe transfers.
        self._layout_t = wp.to_torch(self._layout)
        self._rotations_t = wp.to_torch(self._rotations)
        self._reset_count_t = wp.to_torch(self._reset_count)
        self._goal_count_t = wp.to_torch(self._goal_count)
        self._failure_t = wp.to_torch(self._layout_failures)
        self._body_pos_t = wp.to_torch(self._model.body_pos)
        self._body_quat_t = wp.to_torch(self._model.body_quat)
        self._body_ipos_t = wp.to_torch(self._model.body_ipos)
        self._body_iquat_t = wp.to_torch(self._model.body_iquat)
        self._geom_pos_t = wp.to_torch(self._model.geom_pos)
        self._geom_quat_t = wp.to_torch(self._model.geom_quat)
        self._eq_data_t = wp.to_torch(self._model.eq_data)
        self._qpos0_t = wp.to_torch(self._model.qpos0)
        self._qpos_spring_t = wp.to_torch(self._model.qpos_spring)
        self._qpos_t = wp.to_torch(self._data.qpos)
        self._ctrl_t = wp.to_torch(self._data.ctrl)
        self._time_t = wp.to_torch(self._data.time)
        self._xpos_t = wp.to_torch(self._data.xpos)
        self._xmat_t = wp.to_torch(self._data.xmat)
        self._sensordata_t = wp.to_torch(self._data.sensordata)
        self._nacon_t = wp.to_torch(self._data.nacon)
        self._overflow_t = wp.to_torch(self._data.overflow)
        self._contact_world_t = wp.to_torch(self._data.contact.worldid)
        self._contact_geom_t = wp.to_torch(self._data.contact.geom)
        self._contact_capacity_row_t = torch.arange(
            self._contact_world_t.numel(), dtype=torch.int32, device=self.device
        )

        # Reusable masks avoid one allocation at every explicit reset.
        self._all_reset_t = torch.ones(self.B, dtype=torch.bool, device=self.device)
        self._all_reset_w = wp.from_torch(self._all_reset_t)

    def _warp_stream(self):
        """Return a Warp wrapper around the caller's current Torch CUDA stream."""
        return wp.stream_from_torch(torch.cuda.current_stream(self.device))

    def _chain_caller_stream(self):
        """Order environment state access when callers switch Torch CUDA streams.

        The common default-stream path performs no operation.  On a stream
        change, ``wait_stream`` records a CUDA event dependency without blocking
        the host, preserving the asynchronous rollout path and shared buffers.
        """
        current = torch.cuda.current_stream(self.device)
        previous = self._last_caller_stream_t
        if int(current.cuda_stream) != int(previous.cuda_stream):
            current.wait_stream(previous)
        self._last_caller_stream_t = current

    @staticmethod
    def _z_quaternion(angles):
        """Convert [B,N] z rotations to MuJoCo scalar-first quaternions [B,N,4]."""
        half = 0.5 * angles
        zeros = torch.zeros_like(half)
        return torch.stack((torch.cos(half), zeros, zeros, torch.sin(half)), dim=-1)

    def _apply_layout_to_model(self):
        """Write sampled layouts into all model fields used by Gremlin dynamics."""
        ids = self._ids
        quat = self._z_quaternion(self._rotations_t)

        # Agent base transform.  Its slide/hinge qpos remains zero at reset, exactly
        # as in the Safety-Gymnasium-compiled model.
        self._body_pos_t[:, ids.agent_body, :2] = self._layout_t[:, 0]
        self._body_quat_t[:, ids.agent_body] = quat[:, 0]

        # Static goal/hazards are collision-disabled visualization/lidar bodies.
        self._body_pos_t[:, ids.goal_body, :2] = self._layout_t[:, 1]
        self._body_quat_t[:, ids.goal_body] = quat[:, 1]
        for offset, body in enumerate(ids.hazard_bodies):
            layout_index = _FIRST_HAZARD_LAYOUT_INDEX + offset
            self._body_pos_t[:, body, :2] = self._layout_t[:, layout_index]
            self._body_quat_t[:, body] = quat[:, layout_index]

        # Each gremlin has a free body welded to a mocap body.  Its sampled center
        # and rotation appear in four model locations plus qpos0/qpos_spring.
        for offset, (body, mocap_body, mocap_geom, qadr) in enumerate(zip(
            ids.gremlin_bodies,
            ids.gremlin_mocap_bodies,
            ids.gremlin_mocap_geoms,
            ids.gremlin_qpos_addresses,
        )):
            layout_index = _FIRST_GREMLIN_LAYOUT_INDEX + offset
            xy = self._layout_t[:, layout_index]
            q = quat[:, layout_index]

            self._body_pos_t[:, body, :2] = xy
            self._body_quat_t[:, body] = q
            self._geom_pos_t[:, mocap_geom, :2] = xy
            self._geom_quat_t[:, mocap_geom] = q
            self._body_ipos_t[:, mocap_body, :2] = xy
            self._body_iquat_t[:, mocap_body] = q

            self._eq_data_t[:, offset, 3:5] = xy
            self._eq_data_t[:, offset, 6:10] = q
            self._qpos0_t[:, qadr:qadr + 2] = xy
            self._qpos0_t[:, qadr + 3:qadr + 7] = q
            self._qpos_spring_t[:, qadr:qadr + 2] = xy
            self._qpos_spring_t[:, qadr + 3:qadr + 7] = q

    def _sample_full_layout(self):
        """Generate and install one complete layout for every world on CUDA."""
        self._failure_t.zero_()
        stream = self._warp_stream()
        with wp.ScopedStream(stream, sync_enter=False):
            wp.launch(
                _sample_layout_kernel,
                dim=self.B,
                inputs=[
                    self._seed,
                    self._all_reset_w,
                    self._reset_count,
                    self._goal_count,
                    self._layout,
                    self._rotations,
                    self._layout_failures,
                ],
                device=self._warp_device,
            )

        # Torch assignments execute on the same CUDA stream and therefore observe
        # the kernel's layout without a device/host synchronization.
        self._apply_layout_to_model()

        with wp.ScopedStream(stream, sync_enter=False):
            mjw.reset_data(self._model, self._data)
            # qpos0/body transforms changed per world; recompute the corresponding
            # inertia, camera, and actuator reference quantities before rollout.
            mjw.set_const_0(self._model, self._data, restore=True)
            mjw.forward(self._model, self._data)

        # Layout failure is exceptional and checked only once per 1000-step reset,
        # so this synchronization does not reintroduce a per-step CPU bottleneck.
        if bool(torch.any(self._failure_t).item()):
            raise RuntimeError("MJWarp Gremlin layout sampling exhausted its retry budget")

    def _sample_reached_goals(self, reached):
        """Advance only reached goals using per-world GPU RNG counters."""
        # Builder.update_layout refreshes the agent entry from current xpos before
        # Goal.update_world samples against the placement keepouts.  Keeping the
        # reset-time agent coordinate would permit a new goal on top of the agent.
        agent_xy = self._xpos_t[:, self._ids.agent_body, :2]
        self._layout_t[:, _AGENT_LAYOUT_INDEX] = torch.where(
            reached[:, None],
            agent_xy,
            self._layout_t[:, _AGENT_LAYOUT_INDEX],
        )
        stream = self._warp_stream()
        reached_w = wp.from_torch(reached.contiguous())
        with wp.ScopedStream(stream, sync_enter=False):
            wp.launch(
                _resample_goal_kernel,
                dim=self.B,
                inputs=[
                    self._seed,
                    reached_w,
                    self._reset_count,
                    self._goal_count,
                    self._layout,
                    self._layout_failures,
                ],
                device=self._warp_device,
            )

        # The goal is a static, collision-disabled body.  Updating its batched
        # body_pos is sufficient; the next physics forward sees the new location.
        self._body_pos_t[:, self._ids.goal_body, :2] = self._layout_t[:, 1]

    def _capture_physics_graph(self):
        """Capture ten complete MuJoCo substeps once on a native CUDA stream.

        One MuJoCo-Warp step expands to many small kernels.  Launching those
        kernels through Python on every transition leaves an A100 idle between
        launches, especially at the historical B=32.  CUDA Graph replay submits
        the entire ten-substep sequence as one operation while reading current
        ctrl, time, mocap, and simulation state from persistent buffers.
        """
        # Reset/forward and the action copy may have been queued on PyTorch's
        # caller stream.  This one-time synchronization gives graph capture a
        # quiescent device; steady-state replay below uses stream dependencies.
        torch.cuda.synchronize(self.device)

        # MuJoCo-Warp creates short-lived collision/solver workspaces during an
        # eager call.  Prime its CUDA allocator before capture (allocation is not
        # legal while a stream is being captured).  Capture immediately while
        # those blocks are reusable, then restore the exact reset state.
        saved_ctrl = self._ctrl_t.clone()
        warm_stream = self._warp_stream()
        with wp.ScopedStream(warm_stream, sync_enter=False):
            for _ in range(_PHYSICS_SUBSTEPS):
                wp.launch(
                    _move_gremlins_kernel,
                    dim=(self.B, _NUM_GREMLINS),
                    inputs=[self._data.time, self._data.mocap_pos],
                    device=self._warp_device,
                )
                mjw.step(self._model, self._data)
            # Safety-Gymnasium calls mj_forward after the frameskip loop so xpos,
            # xmat, contacts, and sensors correspond to the final integrated qpos.
            mjw.forward(self._model, self._data)
        wp.synchronize_device(self._warp_device)

        with wp.ScopedCapture(
            device=self._warp_device,
            stream=self._physics_stream_w,
        ) as capture:
            for _ in range(_PHYSICS_SUBSTEPS):
                wp.launch(
                    _move_gremlins_kernel,
                    dim=(self.B, _NUM_GREMLINS),
                    inputs=[self._data.time, self._data.mocap_pos],
                    device=self._warp_device,
                )
                mjw.step(self._model, self._data)
            # This final forward is part of the environment-step semantics, not
            # optional observation post-processing; reward/cost read its outputs.
            mjw.forward(self._model, self._data)

        self._physics_graph = capture.graph

        # Graph capture does not advance Data, but the eager allocator warmup
        # did.  Rebuild the same episode start before the first graph replay.
        reset_stream = self._warp_stream()
        with wp.ScopedStream(reset_stream, sync_enter=False):
            mjw.reset_data(self._model, self._data)
            mjw.set_const_0(self._model, self._data, restore=True)
            mjw.forward(self._model, self._data)
        self._ctrl_t.copy_(saved_ctrl)
        torch.cuda.synchronize(self.device)

    def _physics_step(self, actions):
        """Execute actuator preparation and a graphed ten-substep physics update."""
        raw = actions.to(device=self.device, dtype=torch.float32)
        if raw.shape != (self.B, self.act_dim):
            raise ValueError(
                f"MJWarp actions must have shape {(self.B, self.act_dim)}, got {tuple(raw.shape)}"
            )

        # Safety-Gymnasium BaseAgent.apply_action clips data.ctrl even when the
        # author-facing wrapper reports the raw Gaussian draw as executed_action.
        self._ctrl_t.copy_(torch.clamp(raw, -1.0, 1.0))
        if self._physics_graph is None:
            self._capture_physics_graph()

        # Producer: policy/action copy on the current Torch stream.  Consumer:
        # graph replay on the native Warp stream.  The reciprocal wait makes all
        # reward/observation Torch kernels consume completed physics without a
        # CPU-side synchronize in the recurring path.
        caller_stream = torch.cuda.current_stream(self.device)
        self._physics_stream_t.wait_stream(caller_stream)
        wp.capture_launch(self._physics_graph, stream=self._physics_stream_w)
        caller_stream.wait_stream(self._physics_stream_t)

        reported = (
            raw
            if self.action_mode == SafetyEnv.AUTHOR_RAW_ENV_CLAMP
            else torch.clamp(raw, -1.0, 1.0)
        )
        return reported

    def _pseudo_lidar(self, object_xy):
        """Vectorized Torch implementation of Safety-Gymnasium pseudo-lidar."""
        agent_pos = self._xpos_t[:, self._ids.agent_body]
        agent_mat = self._xmat_t[:, self._ids.agent_body]

        # Upstream forms a world xyz vector with object z=0, multiplies the row
        # vector by agent xmat, then retains ego XY.
        world = torch.zeros(
            (self.B, object_xy.shape[1], 3),
            dtype=torch.float32,
            device=self.device,
        )
        world[:, :, :2] = object_xy - agent_pos[:, None, :2]
        world[:, :, 2] = -agent_pos[:, None, 2]
        ego = torch.einsum("bni,bij->bnj", world, agent_mat)[:, :, :2]

        distance = torch.linalg.vector_norm(ego, dim=-1)
        angle = torch.remainder(torch.atan2(ego[:, :, 1], ego[:, :, 0]), 2.0 * math.pi)
        bin_size = (2.0 * math.pi) / _LIDAR_BINS
        scaled_angle = angle / bin_size
        # Float32 remainder may round a tiny negative angle to exactly 2*pi,
        # yielding floor(scaled_angle)==16.  Clamp only that boundary case to
        # the final central bin; alias=1 then correctly wraps its neighbor to 0.
        bins = torch.clamp(
            torch.floor(scaled_angle).to(torch.int64),
            min=0,
            max=_LIDAR_BINS - 1,
        )
        sensor = torch.clamp((_LIDAR_MAX_DIST - distance) / _LIDAR_MAX_DIST, min=0.0)

        # ``scatter_reduce_(amax)`` exactly expresses closest-object occlusion and
        # the two neighboring alias bins used by upstream pseudo-lidar.
        observation = torch.zeros(
            (self.B, _LIDAR_BINS), dtype=torch.float32, device=self.device
        )
        observation.scatter_reduce_(1, bins, sensor, reduce="amax", include_self=True)
        alias = torch.clamp(scaled_angle - bins.to(angle.dtype), min=0.0, max=1.0)
        observation.scatter_reduce_(
            1,
            torch.remainder(bins + 1, _LIDAR_BINS),
            alias * sensor,
            reduce="amax",
            include_self=True,
        )
        observation.scatter_reduce_(
            1,
            torch.remainder(bins - 1, _LIDAR_BINS),
            (1.0 - alias) * sensor,
            reduce="amax",
            include_self=True,
        )
        return observation

    def _observation(self):
        """Assemble the exact 60-field flattened observation ordering on CUDA."""
        goal = self._layout_t[:, 1:2]
        hazards = self._layout_t[:, 2:7]
        gremlins = self._xpos_t[:, list(self._ids.gremlin_bodies), :2]

        # OrderedDict flatten order in Safety-Gymnasium is sensors (12), goal
        # lidar (16), hazards lidar (16), then gremlins lidar (16).
        return torch.cat(
            (
                self._sensordata_t[:, :12],
                self._pseudo_lidar(goal),
                self._pseudo_lidar(hazards),
                self._pseudo_lidar(gremlins),
            ),
            dim=1,
        ).to(torch.float32)

    def _diagnostic_cost(self):
        """Compute hazard and Gremlin contact costs with the native task formulas."""
        agent_xy = self._xpos_t[:, self._ids.agent_body, :2]
        hazard_xy = self._layout_t[:, 2:7]
        hazard_distance = torch.linalg.vector_norm(
            hazard_xy - agent_xy[:, None, :], dim=-1
        )
        hazard_cost = torch.clamp(_HAZARD_SIZE - hazard_distance, min=0.0).sum(dim=1)

        # MJWarp reuses its over-allocated contact buffer and may leave old geom
        # IDs beyond nacon.  Mask those rows on device so a past Gremlin collision
        # cannot be charged again after the contact disappears.
        geom0 = self._contact_geom_t[:, 0]
        geom1 = self._contact_geom_t[:, 1]
        is_agent = torch.zeros_like(geom0, dtype=torch.bool)
        is_gremlin = torch.zeros_like(geom0, dtype=torch.bool)
        for geom in self._ids.agent_geoms:
            is_agent |= (geom0 == geom) | (geom1 == geom)
        for geom in self._ids.gremlin_geoms:
            is_gremlin |= (geom0 == geom) | (geom1 == geom)

        valid_contact = self._contact_capacity_row_t < torch.clamp(
            self._nacon_t[0], max=self._contact_world_t.numel()
        )
        contact = is_agent & is_gremlin & valid_contact
        gremlin_cost = torch.zeros(self.B, dtype=torch.float32, device=self.device)
        gremlin_cost.scatter_add_(
            0,
            self._contact_world_t.to(torch.int64),
            contact.to(torch.float32),
        )
        # Builder._cost applies CostConf.constrain_indicator=True separately to
        # every source and to cost_sum.  Since all sources are non-negative, the
        # externally returned cost_sum is exactly the indicator of their union.
        return ((hazard_cost + gremlin_cost) > 0.0).to(torch.float32)

    def _raise_if_physics_overflow(self):
        """Fail closed if MJWarp exhausted a contact or constraint buffer.

        ``Data.overflow`` accumulates a per-world bitmask until ``reset_data``.
        Checking only after reset would therefore erase the evidence.  This method
        is called at rollout/reset boundaries, never in the recurring physics path,
        so its one host synchronization cannot recreate the original CPU bottleneck.
        """
        overflowed = self._overflow_t != 0                 # [B] device-side bitmask predicate
        if not bool(torch.any(overflowed).item()):          # One boundary-only CUDA synchronization
            return

        # Copy only exceptional rows for an actionable error.  Continuing after
        # an overflow would make contact/cost/solver output scientifically invalid.
        slots = torch.nonzero(overflowed, as_tuple=False).flatten()
        bitmasks = self._overflow_t[slots]
        raise RuntimeError(
            "MJWarp physics buffer overflow; increase nconmax/njmax before rerunning: "
            f"slots={slots.detach().cpu().tolist()}, "
            f"bitmasks={bitmasks.detach().cpu().tolist()}"
        )

    def _raise_if_goal_resample_failed(self):
        """Reject a rollout containing the task's exceptional resampling terminal."""
        failed = self._failure_t != 0                       # Sticky until full reset_data/layout.
        if not bool(torch.any(failed).item()):               # Boundary-only CUDA synchronization.
            return
        slots = torch.nonzero(failed, as_tuple=False).flatten().detach().cpu().tolist()
        raise RuntimeError(
            "MJWarp goal resampling exhausted 10,000 attempts; "
            f"terminated slots={slots}. Refusing to continue an invalid task state."
        )

    def reset(self):
        """Start a vector segment with a fully GPU-sampled Gremlin layout."""
        self._chain_caller_stream()                          # Preserve state order across Torch streams.
        # Author-aligned mode defers its horizon reset; inspect the cumulative
        # simulator flag before ``reset_data`` clears it for the next segment.
        self._raise_if_physics_overflow()
        self._raise_if_goal_resample_failed()
        self.step_count = 0
        self._cost_buf = []
        self._reward_buf = []
        self._sample_full_layout()

        observation = self._observation()
        agent_xy = self._xpos_t[:, self._ids.agent_body, :2]
        self._last_goal_distance = torch.linalg.vector_norm(
            self._layout_t[:, 1] - agent_xy, dim=1
        )
        return observation

    def step(self, actions):
        """Advance all CUDA worlds and return the existing structured vector API."""
        self._chain_caller_stream()                          # Usually a no-op on the default stream.
        executed_action = self._physics_step(actions)
        self.step_count += 1

        agent_xy = self._xpos_t[:, self._ids.agent_body, :2]
        old_goal = self._layout_t[:, 1].clone()
        current_distance = torch.linalg.vector_norm(old_goal - agent_xy, dim=1)
        reached = current_distance <= _GOAL_SIZE

        # GoalLevel0 reward: distance progress plus one on goal achievement.
        reward = self._last_goal_distance - current_distance
        reward = reward + reached.to(reward.dtype)
        cost = self._diagnostic_cost()

        # The kernel is launched for every step but returns immediately for worlds
        # without an event; no reached-mask transfer or Python branch is involved.
        self._sample_reached_goals(reached)
        next_goal_distance = torch.linalg.vector_norm(
            self._layout_t[:, 1] - agent_xy, dim=1
        )
        self._last_goal_distance = torch.where(
            reached, next_goal_distance, current_distance
        )

        final_observation = self._observation()
        # CPU Builder always applies task.num_steps even if an adapter requests a
        # longer segment.  The earlier of native and adapter horizons is authoritative.
        force_truncation = self.step_count >= min(self.n, self._native_horizon)
        # The paper task sets terminate_resample_failure=True.  The failure flag
        # is produced by the CUDA goal kernel; ordinary runs remain all-false.
        terminated = self._failure_t.to(torch.bool)
        truncated = torch.full(
            (self.B,), force_truncation, dtype=torch.bool, device=self.device
        ) & ~terminated
        final_mask = terminated | truncated

        # Legacy explicit-clip mode auto-resets at the synchronized horizon.  The
        # author path intentionally defers reset so checkpoint code can snapshot
        # the pre-reset RNG counters, matching the CPU adapters' boundary contract.
        observation = final_observation
        if force_truncation and self.action_mode != SafetyEnv.AUTHOR_RAW_ENV_CLAMP:
            # Legacy mode resets inside step(), so preserve the overflow audit
            # before ``_sample_full_layout`` calls ``reset_data``.
            self._raise_if_physics_overflow()
            self._raise_if_goal_resample_failed()
            self._sample_full_layout()
            observation = self._observation()
            reset_agent_xy = self._xpos_t[:, self._ids.agent_body, :2]
            self._last_goal_distance = torch.linalg.vector_norm(
                self._layout_t[:, 1] - reset_agent_xy, dim=1
            )

        # Match the CPU adapters' O(T) diagnostic storage.  Retaining [B]
        # tensors here would waste O(B*T) GPU memory at B=2048 even though both
        # render() and stats() consume only the per-step batch means.
        self._reward_buf.append(reward.detach().mean())
        self._cost_buf.append(cost.detach().mean())
        return VecStepResult(
            observation=observation,
            reward=reward,
            cost=cost,
            terminated=terminated,
            truncated=truncated,
            final_observation=final_observation,
            final_observation_mask=final_mask,
            executed_action=executed_action,
        )

    def render(self, mode=None):
        """Return the per-step batch-mean diagnostic cost sequence on demand."""
        del mode
        self._chain_caller_stream()                          # Order a possible device-to-host copy.
        if not self._cost_buf:
            return None
        return torch.stack(self._cost_buf).cpu().numpy().astype(np.float64)

    def stats(self):
        """Transfer two rollout-level diagnostic scalars rather than every transition."""
        self._chain_caller_stream()                          # Order reductions after the last step.
        self._raise_if_physics_overflow()                   # Validate physics before logging results.
        self._raise_if_goal_resample_failed()               # Do not log a failed task layout.
        out = {}
        if self._cost_buf:
            out["avg_step_cost"] = float(torch.stack(self._cost_buf).mean().item())
        if self._reward_buf:
            out["avg_step_reward"] = float(torch.stack(self._reward_buf).mean().item())
        return out

    def checkpoint_rng_state(self):
        """Serialize per-world GPU RNG clocks in the existing list envelope."""
        self._chain_caller_stream()                          # Counters may have changed on another stream.
        reset_count = self._reset_count_t.detach().cpu().tolist()
        goal_count = self._goal_count_t.detach().cpu().tolist()
        return [
            {
                "version": 1,
                "action_mode": self.action_mode,
                "dynamics_profile": self.dynamics_profile,
                "backend": "mjwarp",
                "seed": self._seed,
                "slot": slot,
                "reset_count": int(reset_count[slot]),
                "goal_count": int(goal_count[slot]),
            }
            for slot in range(self.B)
        ]

    def restore_rng_state(self, states):
        """Restore GPU RNG counters; physical state is rebuilt by the next reset."""
        self._chain_caller_stream()                          # Serialize copies with prior environment work.
        if not isinstance(states, (list, tuple)) or len(states) != self.B:
            raise ValueError(f"MJWarp RNG state must contain exactly {self.B} slots")

        reset_count, goal_count = [], []
        for slot, state in enumerate(states):
            if not isinstance(state, dict) or int(state.get("version", -1)) != 1:
                raise ValueError("MJWarp RNG state must use the version-1 mapping")
            if state.get("backend") != "mjwarp":
                raise ValueError("cannot restore a non-MJWarp RNG state into MJWarp")
            if state.get("action_mode") != self.action_mode:
                raise ValueError("MJWarp RNG action_mode mismatch")
            saved_profile = str(state.get("dynamics_profile", SafetyEnv.LEGACY_EULER))
            if saved_profile != self.dynamics_profile:
                raise ValueError("MJWarp RNG dynamics profile mismatch")
            if int(state.get("seed", -1)) != self._seed or int(state.get("slot", -1)) != slot:
                raise ValueError("MJWarp RNG seed/slot mismatch")
            reset_count.append(int(state["reset_count"]))
            goal_count.append(int(state["goal_count"]))

        self._reset_count_t.copy_(
            torch.tensor(reset_count, dtype=torch.int32, device=self.device)
        )
        self._goal_count_t.copy_(
            torch.tensor(goal_count, dtype=torch.int32, device=self.device)
        )

    def close(self):
        """Synchronize outstanding GPU work before releasing Warp-owned arrays."""
        with wp.ScopedDevice(self._warp_device):
            wp.synchronize_device(self._warp_device)
