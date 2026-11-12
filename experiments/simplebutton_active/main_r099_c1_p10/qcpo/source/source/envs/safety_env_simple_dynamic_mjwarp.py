# -*- coding: utf-8 -*-
"""GPU-native MJWarp backends for the paper SimpleButton and Dynamic tasks.

The Gremlin backend in :mod:`safety_env_mjwarp` owns the validated MuJoCo-Warp
model replication, CUDA-stream chaining, graph replay, pseudo-vector API, overflow
audits, and checkpoint RNG envelope.  This module reuses that infrastructure and
implements only the task-state differences required by paper config0/config1:

``SimpleButton``
    two fixed contact buttons, one visual goal, three fixed hazards, a ten-step
    button visibility/cost timer, and contact-based goal achievement;

``Dynamic``
    one distance-based goal and three randomly placed hazards.

All recurring dynamics and task calculations remain on the learner's CUDA device.
The one host synchronization at reset/rollout boundaries audits exceptional layout,
contact, or solver overflows; it is not part of the steady-state step path.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .safety_env import SafetyEnv
from .safety_env_mjwarp import SafetyVecEnvMJWarp, _Space, mjw, wp
from .safety_env_vec import VecStepResult


# ============================================================================
# Frozen paper-task constants
# ============================================================================
# ``task_kind`` is passed to Warp kernels as an integer because one compiled
# module must serve both task layouts without Python branches in the rollout path.
_TASK_DYNAMIC = 0
_TASK_SIMPLE_BUTTON = 1

# Parent ``_initialize_warp`` allocates ten layout columns for Gremlin.  These
# tasks use only the leading five/seven columns, so no common allocator changes
# (and therefore no Gremlin regression surface) are required.
_MAX_TASK_LAYOUT_OBJECTS = 7
_DYNAMIC_LAYOUT_OBJECTS = 5
_SIMPLE_LAYOUT_OBJECTS = 7

_AGENT_LAYOUT_INDEX = 0
_DYNAMIC_GOAL_LAYOUT_INDEX = 1
_DYNAMIC_FIRST_HAZARD_INDEX = 2
_SIMPLE_FIRST_BUTTON_INDEX = 1
_SIMPLE_GOAL_LAYOUT_INDEX = 3
_SIMPLE_FIRST_HAZARD_INDEX = 4

_NUM_HAZARDS = 3
_NUM_SIMPLE_BUTTONS = 2
_EXTENT = 1.5
_AGENT_KEEPOUT = 0.4
_DYNAMIC_GOAL_KEEPOUT = 0.305
_SIMPLE_BUTTON_KEEPOUT = 0.2
_SIMPLE_VISUAL_GOAL_KEEPOUT = 0.4
_HAZARD_KEEPOUT = 0.305
_GOAL_SIZE = 0.3
_HAZARD_SIZE = 0.3
_BUTTON_RESAMPLING_DELAY = 10
_LIDAR_BINS = 16
_LIDAR_MAX_DIST = 3.0
_PHYSICS_SUBSTEPS = 10


@wp.func
def _task_layout_keepout(task_kind: int, index: int) -> float:
    """Return the placement keepout for one task/layout column."""
    keepout = _HAZARD_KEEPOUT
    if index == _AGENT_LAYOUT_INDEX:
        keepout = _AGENT_KEEPOUT
    elif task_kind == _TASK_DYNAMIC and index == _DYNAMIC_GOAL_LAYOUT_INDEX:
        keepout = _DYNAMIC_GOAL_KEEPOUT
    elif task_kind == _TASK_SIMPLE_BUTTON:
        if index == _SIMPLE_FIRST_BUTTON_INDEX or index == _SIMPLE_FIRST_BUTTON_INDEX + 1:
            keepout = _SIMPLE_BUTTON_KEEPOUT
        elif index == _SIMPLE_GOAL_LAYOUT_INDEX:
            keepout = _SIMPLE_VISUAL_GOAL_KEEPOUT
    return keepout


@wp.func
def _task_candidate_is_valid(
    task_kind: int,
    world: int,
    candidate: wp.vec2,
    candidate_keepout: float,
    placed_count: int,
    skip_index: int,
    layout: wp.array2d(dtype=wp.vec2),
) -> bool:
    """Apply Safety-Gymnasium's ordered pairwise Euclidean keepout test."""
    valid = bool(True)
    for other in range(_MAX_TASK_LAYOUT_OBJECTS):
        # Initial construction only compares already placed objects.  Dynamic
        # goal resampling compares every object except the obsolete goal slot.
        if other < placed_count and other != skip_index:
            required = candidate_keepout + _task_layout_keepout(task_kind, other)
            if wp.length(candidate - layout[world, other]) < required:
                valid = bool(False)
    return valid


@wp.kernel
def _sample_task_layout_kernel(
    task_kind: int,
    base_seed: int,
    reset_mask: wp.array(dtype=wp.bool),
    reset_count: wp.array(dtype=wp.int32),
    goal_count: wp.array(dtype=wp.int32),
    layout: wp.array2d(dtype=wp.vec2),
    rotations: wp.array2d(dtype=float),
    goal_button: wp.array(dtype=wp.int32),
    button_timer: wp.array(dtype=wp.int32),
    failures: wp.array(dtype=wp.int32),
):
    """Sample a complete paper-task reset state independently per CUDA world.

    The 10,000 complete-layout attempts and 100 per-object attempts match the
    upstream ``RandomGenerator.build_layout/sample_layout`` retry structure.
    SimpleButton fixed locations are exactly representable in float32; upstream
    samples only a +/-1e-9 interval around those coordinates.
    """
    world = wp.tid()
    if not reset_mask[world]:
        return

    # Slot is the independent PCG sequence offset; reset_count chooses a stable
    # checkpointable episode substream and is incremented only after success.
    launch_seed = base_seed + reset_count[world] * 104729
    rng = wp.rand_init(launch_seed, world)
    failures[world] = 0
    object_count = _DYNAMIC_LAYOUT_OBJECTS
    if task_kind == _TASK_SIMPLE_BUTTON:
        object_count = _SIMPLE_LAYOUT_OBJECTS

    layout_found = bool(False)
    for _layout_attempt in range(10000):
        layout_ok = bool(True)
        for obj in range(_MAX_TASK_LAYOUT_OBJECTS):
            if obj < object_count:
                keepout = _task_layout_keepout(task_kind, obj)
                low = -_EXTENT + keepout
                high = _EXTENT - keepout
                fixed = bool(False)
                fixed_xy = wp.vec2(0.0, 0.0)

                # Paper config0 fixes both buttons and all hazards.  A conflict
                # with an earlier random agent/dummy goal restarts the layout.
                if task_kind == _TASK_SIMPLE_BUTTON:
                    if obj == 1:
                        fixed = bool(True)
                        fixed_xy = wp.vec2(-1.0, -1.0)
                    elif obj == 2:
                        fixed = bool(True)
                        fixed_xy = wp.vec2(1.0, 1.0)
                    elif obj == 4:
                        fixed = bool(True)
                        fixed_xy = wp.vec2(0.0, 0.0)
                    elif obj == 5:
                        fixed = bool(True)
                        fixed_xy = wp.vec2(-1.0, 1.0)
                    elif obj == 6:
                        fixed = bool(True)
                        fixed_xy = wp.vec2(0.5, -0.5)

                object_found = bool(False)
                for _object_attempt in range(100):
                    candidate = fixed_xy
                    if not fixed:
                        candidate = wp.vec2(
                            wp.randf(rng, low, high),
                            wp.randf(rng, low, high),
                        )
                    if _task_candidate_is_valid(
                        task_kind,
                        world,
                        candidate,
                        keepout,
                        obj,
                        -1,
                        layout,
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

    # World construction draws agent and obstacle rotations after placement.
    # Goal resampling changes only XY and intentionally preserves its rotation.
    for obj in range(_MAX_TASK_LAYOUT_OBJECTS):
        if obj < object_count:
            rotations[world, obj] = wp.randf(rng, 0.0, 2.0 * math.pi)

    if task_kind == _TASK_DYNAMIC:
        # GoalLevel0.reset first lays out a goal and then update_world deletes and
        # resamples it against the current agent and hazards.
        goal_found = bool(False)
        goal_low = -_EXTENT + _DYNAMIC_GOAL_KEEPOUT
        goal_high = _EXTENT - _DYNAMIC_GOAL_KEEPOUT
        for _goal_attempt in range(10000):
            candidate = wp.vec2(
                wp.randf(rng, goal_low, goal_high),
                wp.randf(rng, goal_low, goal_high),
            )
            if _task_candidate_is_valid(
                task_kind,
                world,
                candidate,
                _DYNAMIC_GOAL_KEEPOUT,
                object_count,
                _DYNAMIC_GOAL_LAYOUT_INDEX,
                layout,
            ):
                layout[world, _DYNAMIC_GOAL_LAYOUT_INDEX] = candidate
                goal_found = bool(True)
                break
        if not goal_found:
            failures[world] = 1
            return
    else:
        # ButtonLevel0.update_world picks one physical button, moves the visual
        # goal there, then specific_reset immediately sets timer back to zero.
        selected = int(0)
        if wp.randf(rng, 0.0, 1.0) >= 0.5:
            selected = 1
        goal_button[world] = selected
        layout[world, _SIMPLE_GOAL_LAYOUT_INDEX] = layout[
            world, _SIMPLE_FIRST_BUTTON_INDEX + selected
        ]
        button_timer[world] = 0

    reset_count[world] = reset_count[world] + 1
    goal_count[world] = 0


@wp.kernel
def _resample_dynamic_goal_kernel(
    base_seed: int,
    reached: wp.array(dtype=wp.bool),
    reset_count: wp.array(dtype=wp.int32),
    goal_count: wp.array(dtype=wp.int32),
    layout: wp.array2d(dtype=wp.vec2),
    failures: wp.array(dtype=wp.int32),
):
    """Resample reached Dynamic goals against current agent/hazard positions."""
    world = wp.tid()
    if not reached[world] or failures[world] != 0:
        return
    failures[world] = 0

    # The counter mix matches the proven Gremlin backend and makes each slot's
    # sequence invariant to batch-size changes and checkpoint/resume boundaries.
    launch_seed = (
        base_seed
        + reset_count[world] * 104729
        + goal_count[world] * 130363
        + 7919
    )
    rng = wp.rand_init(launch_seed, world)
    goal_low = -_EXTENT + _DYNAMIC_GOAL_KEEPOUT
    goal_high = _EXTENT - _DYNAMIC_GOAL_KEEPOUT
    goal_found = bool(False)

    for _attempt in range(10000):
        candidate = wp.vec2(
            wp.randf(rng, goal_low, goal_high),
            wp.randf(rng, goal_low, goal_high),
        )
        if _task_candidate_is_valid(
            _TASK_DYNAMIC,
            world,
            candidate,
            _DYNAMIC_GOAL_KEEPOUT,
            _DYNAMIC_LAYOUT_OBJECTS,
            _DYNAMIC_GOAL_LAYOUT_INDEX,
            layout,
        ):
            layout[world, _DYNAMIC_GOAL_LAYOUT_INDEX] = candidate
            goal_found = bool(True)
            break

    if not goal_found:
        failures[world] = 1
        return
    goal_count[world] = goal_count[world] + 1


@wp.kernel
def _select_next_button_kernel(
    base_seed: int,
    reached: wp.array(dtype=wp.bool),
    reset_count: wp.array(dtype=wp.int32),
    goal_count: wp.array(dtype=wp.int32),
    layout: wp.array2d(dtype=wp.vec2),
    goal_button: wp.array(dtype=wp.int32),
    button_timer: wp.array(dtype=wp.int32),
):
    """Choose a new goal button and reset its visibility/cost delay on CUDA."""
    world = wp.tid()
    if not reached[world]:
        return

    launch_seed = (
        base_seed
        + reset_count[world] * 104729
        + goal_count[world] * 130363
        + 7919
    )
    rng = wp.rand_init(launch_seed, world)
    selected = int(0)
    if wp.randf(rng, 0.0, 1.0) >= 0.5:
        selected = 1

    goal_button[world] = selected
    layout[world, _SIMPLE_GOAL_LAYOUT_INDEX] = layout[
        world, _SIMPLE_FIRST_BUTTON_INDEX + selected
    ]
    button_timer[world] = _BUTTON_RESAMPLING_DELAY
    goal_count[world] = goal_count[world] + 1


@dataclass(frozen=True)
class _StaticTaskModelIds:
    """Resolved MuJoCo indices used by task logic after one startup lookup."""

    agent_body: int
    goal_body: int
    hazard_bodies: tuple[int, ...]
    button_bodies: tuple[int, ...]
    agent_geoms: tuple[int, ...]
    button_geoms: tuple[int, ...]


class SafetyVecEnvSimpleDynamicMJWarp(SafetyVecEnvMJWarp):
    """Batched GPU implementation of paper SimpleButton and Dynamic environments."""

    def __init__(
        self,
        env_id="SafetyPointDynamic0-v0",
        num_envs=32,
        horizon=1000,
        device=None,
        ref_env=None,
        seed=0,
        action_mode=None,
        dynamics_profile=None,
    ):
        """Compile one CPU model oracle, then replicate all mutable state on CUDA."""
        self.env_id = ref_env.env_id if ref_env is not None else str(env_id)
        if "DynamicButton" in self.env_id:
            raise NotImplementedError(
                "DynamicButton has a separate MJWarp adapter and must not use this class"
            )
        if "SimpleButton" in self.env_id:
            self._task_kind = _TASK_SIMPLE_BUTTON
            self._goal_layout_index = _SIMPLE_GOAL_LAYOUT_INDEX
            self._hazard_layout_start = _SIMPLE_FIRST_HAZARD_INDEX
            self.obs_dim = 60
        elif "Dynamic" in self.env_id:
            self._task_kind = _TASK_DYNAMIC
            self._goal_layout_index = _DYNAMIC_GOAL_LAYOUT_INDEX
            self._hazard_layout_start = _DYNAMIC_FIRST_HAZARD_INDEX
            self.obs_dim = 44
        else:
            raise NotImplementedError(
                "this MJWarp adapter supports only paper SimpleButton and Dynamic"
            )

        # Learner and simulator must share one CUDA device.  A fallback to CPU
        # would preserve API shape while invalidating the claimed GPU campaign.
        self.device = torch.device(device if device is not None else "cuda")
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("vec_backend=mjwarp requires an available CUDA device")
        self._last_caller_stream_t = torch.cuda.current_stream(self.device)

        self.B = int(num_envs)
        self.n = int(horizon)
        self._seed = int(seed)
        self.act_dim = 2
        self.observation_space = _Space((self.obs_dim,))
        self.action_space = _Space((self.act_dim,))

        # Preserve the reference environment's action-reporting and simulator
        # profile identities unless the caller explicitly supplies a match.
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
            if ref_env is not None
            else SafetyEnv.DEFAULT_DYNAMICS_PROFILE
        )
        if (
            ref_env is not None
            and dynamics_profile is not None
            and str(dynamics_profile) != str(inherited_profile)
        ):
            raise ValueError(
                "ref_env/vector dynamics profile mismatch: "
                f"{inherited_profile!r} != {dynamics_profile!r}"
            )
        self.dynamics_profile = str(
            inherited_profile if dynamics_profile is None else dynamics_profile
        )
        if self.dynamics_profile not in SafetyEnv.DYNAMICS_PROFILES:
            raise ValueError(f"unsupported dynamics_profile={self.dynamics_profile!r}")

        # Warp owns persistent CUDA arrays while Torch consumes zero-copy views.
        # Parent helpers rely on these fields but do not require parent __init__.
        wp.config.quiet = True
        wp.init()
        self._warp_device = wp.device_from_torch(self.device)
        self._cost_buf = []
        self._reward_buf = []
        self.step_count = 0

        # CPU construction occurs once only to obtain the canonical XML-derived
        # model and verify the installed task constants.  No CPU env survives init.
        probe = SafetyEnv(
            self.env_id,
            seed=self._seed,
            action_mode=self.action_mode,
            dynamics_profile=self.dynamics_profile,
        )
        try:
            probe.reset(seed=self._seed)
            task = probe.env.unwrapped.task
            self._validate_native_task(task)
            self._native_horizon = int(task.num_steps)
            host_model = task.model
            self._ids = self._resolve_static_ids(host_model)

            # Reuse the Gremlin-proven batched model/data allocation, stream,
            # contact buffer, task RNG clocks, and zero-copy Torch views.
            self._initialize_warp(host_model)
            with wp.ScopedDevice(self._warp_device):
                self._goal_button = wp.zeros(
                    self.B, dtype=wp.int32, device=self._warp_device
                )
                self._button_timer = wp.zeros(
                    self.B, dtype=wp.int32, device=self._warp_device
                )
            self._goal_button_t = wp.to_torch(self._goal_button)
            self._button_timer_t = wp.to_torch(self._button_timer)
        finally:
            probe.close()

    def _validate_native_task(self, task):
        """Fail loudly if installed Safety-Gymnasium drifts from paper config0/1."""
        frameskip_ok = (
            int(task.sim_conf.frameskip_binom_n) == _PHYSICS_SUBSTEPS
            and float(task.sim_conf.frameskip_binom_p) == 1.0
        )
        if not frameskip_ok or float(task.action_noise) != 0.0:
            raise NotImplementedError(
                "MJWarp SimpleButton/Dynamic requires frameskip=10 and action_noise=0"
            )
        if not bool(task.cost_conf.constrain_indicator):
            raise NotImplementedError("MJWarp task backend requires constrain_indicator=True")
        if not bool(task.mechanism_conf.continue_goal):
            raise NotImplementedError("MJWarp task backend requires continue_goal=True")

        extents = np.asarray(task.placements_conf.extents, dtype=np.float64)
        if not np.array_equal(extents, np.asarray([-1.5, -1.5, 1.5, 1.5])):
            raise NotImplementedError(f"unsupported placement extents={extents.tolist()}")
        if int(task.lidar_conf.num_bins) != _LIDAR_BINS:
            raise NotImplementedError("MJWarp task backend requires lidar_num_bins=16")
        if float(task.lidar_conf.max_dist) != _LIDAR_MAX_DIST:
            raise NotImplementedError("MJWarp task backend requires lidar_max_dist=3")

        hazards_ok = (
            int(task.hazards.num) == _NUM_HAZARDS
            and float(task.hazards.size) == _HAZARD_SIZE
            and float(task.hazards.keepout) == _HAZARD_KEEPOUT
        )
        if not hazards_ok:
            raise NotImplementedError("unsupported hazards configuration for MJWarp task")

        if self._task_kind == _TASK_DYNAMIC:
            goal_ok = (
                float(task.goal.size) == _GOAL_SIZE
                and float(task.goal.keepout) == _DYNAMIC_GOAL_KEEPOUT
            )
            if not goal_ok:
                raise NotImplementedError("unsupported Dynamic goal configuration")
            return

        buttons_ok = (
            int(task.buttons.num) == _NUM_SIMPLE_BUTTONS
            and float(task.buttons.size) == 0.1
            and float(task.buttons.keepout) == _SIMPLE_BUTTON_KEEPOUT
            and bool(task.buttons.is_constrained)
            and int(task.buttons.resampling_delay) == _BUTTON_RESAMPLING_DELAY
        )
        expected_buttons = np.asarray([[-1.0, -1.0], [1.0, 1.0]])
        expected_hazards = np.asarray([[0.0, 0.0], [-1.0, 1.0], [0.5, -0.5]])
        fixed_locations_ok = (
            np.array_equal(np.asarray(task.buttons.locations), expected_buttons)
            and np.array_equal(np.asarray(task.hazards.locations), expected_hazards)
        )
        if not buttons_ok or not fixed_locations_ok:
            raise NotImplementedError("unsupported SimpleButton paper configuration")

    def _resolve_static_ids(self, model):
        """Resolve names once so recurring device calculations use integer indexing."""
        body_id = lambda name: int(model.body(name).id)
        geom_id = lambda name: int(model.geom(name).id)
        button_count = _NUM_SIMPLE_BUTTONS if self._task_kind == _TASK_SIMPLE_BUTTON else 0
        return _StaticTaskModelIds(
            agent_body=body_id("agent"),
            goal_body=body_id("goal"),
            hazard_bodies=tuple(body_id(f"hazard{i}") for i in range(_NUM_HAZARDS)),
            button_bodies=tuple(body_id(f"button{i}") for i in range(button_count)),
            agent_geoms=(geom_id("agent"), geom_id("pointarrow")),
            button_geoms=tuple(geom_id(f"button{i}") for i in range(button_count)),
        )

    def _initialize_warp(self, host_model):
        """Create batched Warp state without viewing absent equality arrays.

        Gremlin's parent initializer exposes ``model.eq_data`` to Torch because
        moving welded gremlins update equality anchors at every reset.  These two
        tasks have ``neq == 0``; MuJoCo-Warp correctly leaves that zero-length
        array on host, and asking Torch for a CUDA view of its null/host pointer is
        invalid.  The model/data batching itself remains identical, while this
        specialization exports only arrays consumed by static-task methods.
        """
        batched_fields = {
            name: self.B
            for name in (
                "body_pos",
                "body_quat",
                "body_ipos",
                "body_iquat",
                "geom_pos",
                "geom_quat",
                "eq_data",
                "qpos0",
                "qpos_spring",
                "body_invweight0",
                "dof_invweight0",
                "cam_pos0",
                "cam_poscom0",
                "cam_mat0",
                "light_pos0",
                "light_poscom0",
                "light_dir0",
                "actuator_acc0",
            )
        }

        with wp.ScopedDevice(self._warp_device):
            self._model = mjw.put_model(host_model, batch_sizes=batched_fields)
            # Point + static buttons have fewer contacts/constraints than the
            # validated Gremlin maximum.  Retain its conservative capacities so
            # campaign identities share the same fail-closed overflow policy.
            self._data = mjw.make_data(
                host_model,
                nworld=self.B,
                nconmax=32,
                njmax=256,
            )

            # Parent methods and formal checkpoint validators expect this exact
            # task-RNG clock schema, independent of the number of layout columns
            # actively used by a particular paper environment.
            self._layout = wp.zeros(
                (self.B, 10), dtype=wp.vec2, device=self._warp_device
            )
            self._rotations = wp.zeros(
                (self.B, 10), dtype=float, device=self._warp_device
            )
            self._reset_count = wp.zeros(
                self.B, dtype=wp.int32, device=self._warp_device
            )
            self._goal_count = wp.zeros(
                self.B, dtype=wp.int32, device=self._warp_device
            )
            self._layout_failures = wp.zeros(
                self.B, dtype=wp.int32, device=self._warp_device
            )

            # A native Warp stream owns the captured graph.  Its Torch wrapper
            # supplies event dependencies to whichever learner stream calls step.
            self._physics_stream_w = wp.get_stream(self._warp_device)
            self._physics_stream_t = wp.stream_to_torch(self._physics_stream_w)
            self._physics_graph = None

        # Every conversion below is zero-copy and points at a nonempty CUDA array.
        # Equality, mocap, and free-joint model arrays are intentionally omitted:
        # no static-task method reads or mutates them directly.
        self._layout_t = wp.to_torch(self._layout)
        self._rotations_t = wp.to_torch(self._rotations)
        self._reset_count_t = wp.to_torch(self._reset_count)
        self._goal_count_t = wp.to_torch(self._goal_count)
        self._failure_t = wp.to_torch(self._layout_failures)
        self._body_pos_t = wp.to_torch(self._model.body_pos)
        self._body_quat_t = wp.to_torch(self._model.body_quat)
        self._ctrl_t = wp.to_torch(self._data.ctrl)
        self._xpos_t = wp.to_torch(self._data.xpos)
        self._xmat_t = wp.to_torch(self._data.xmat)
        self._sensordata_t = wp.to_torch(self._data.sensordata)
        self._nacon_t = wp.to_torch(self._data.nacon)
        self._overflow_t = wp.to_torch(self._data.overflow)
        self._contact_world_t = wp.to_torch(self._data.contact.worldid)
        self._contact_geom_t = wp.to_torch(self._data.contact.geom)
        self._contact_capacity_row_t = torch.arange(
            self._contact_world_t.numel(),
            dtype=torch.int32,
            device=self.device,
        )

        # Explicit reset always addresses every vector slot, so one persistent
        # bool mask avoids allocating and converting a new mask each episode.
        self._all_reset_t = torch.ones(
            self.B, dtype=torch.bool, device=self.device
        )
        self._all_reset_w = wp.from_torch(self._all_reset_t)

    # ========================================================================
    # Reset/layout installation
    # ========================================================================
    def _apply_task_layout_to_model(self):
        """Write sampled task layout/rotations into the batched model fields."""
        ids = self._ids
        quat = self._z_quaternion(self._rotations_t)

        # Point's slide/hinge qpos remains zero; changing its model base transform
        # exactly represents Safety-Gymnasium's randomized agent start pose.
        self._body_pos_t[:, ids.agent_body, :2] = self._layout_t[:, _AGENT_LAYOUT_INDEX]
        self._body_quat_t[:, ids.agent_body] = quat[:, _AGENT_LAYOUT_INDEX]

        self._body_pos_t[:, ids.goal_body, :2] = self._layout_t[
            :, self._goal_layout_index
        ]
        self._body_quat_t[:, ids.goal_body] = quat[:, self._goal_layout_index]

        for offset, body in enumerate(ids.hazard_bodies):
            layout_index = self._hazard_layout_start + offset
            self._body_pos_t[:, body, :2] = self._layout_t[:, layout_index]
            self._body_quat_t[:, body] = quat[:, layout_index]

        for offset, body in enumerate(ids.button_bodies):
            layout_index = _SIMPLE_FIRST_BUTTON_INDEX + offset
            self._body_pos_t[:, body, :2] = self._layout_t[:, layout_index]
            self._body_quat_t[:, body] = quat[:, layout_index]

    def _sample_full_layout(self):
        """Generate, install, and forward one complete reset state per world."""
        self._failure_t.zero_()
        stream = self._warp_stream()
        with wp.ScopedStream(stream, sync_enter=False):
            wp.launch(
                _sample_task_layout_kernel,
                dim=self.B,
                inputs=[
                    self._task_kind,
                    self._seed,
                    self._all_reset_w,
                    self._reset_count,
                    self._goal_count,
                    self._layout,
                    self._rotations,
                    self._goal_button,
                    self._button_timer,
                    self._layout_failures,
                ],
                device=self._warp_device,
            )

        # Warp and Torch execute on one CUDA stream, so these zero-copy writes
        # consume the sampled layout without a device/host transfer.
        self._apply_task_layout_to_model()
        with wp.ScopedStream(stream, sync_enter=False):
            mjw.reset_data(self._model, self._data)
            mjw.set_const_0(self._model, self._data, restore=True)
            mjw.forward(self._model, self._data)

        # A retry exhaustion is exceptional and checked once per reset only.
        if bool(torch.any(self._failure_t).item()):
            task_name = "SimpleButton" if self._task_kind == _TASK_SIMPLE_BUTTON else "Dynamic"
            raise RuntimeError(f"MJWarp {task_name} layout sampling exhausted its retry budget")

    # ========================================================================
    # Physics graph and task observations
    # ========================================================================
    def _capture_physics_graph(self):
        """Capture the ten static-task MuJoCo substeps as one CUDA Graph replay."""
        torch.cuda.synchronize(self.device)

        # Prime MuJoCo-Warp's collision/solver allocator before capture because
        # stream capture forbids dynamic CUDA allocation.
        saved_ctrl = self._ctrl_t.clone()
        warm_stream = self._warp_stream()
        with wp.ScopedStream(warm_stream, sync_enter=False):
            for _ in range(_PHYSICS_SUBSTEPS):
                mjw.step(self._model, self._data)
            mjw.forward(self._model, self._data)
        wp.synchronize_device(self._warp_device)

        with wp.ScopedCapture(
            device=self._warp_device,
            stream=self._physics_stream_w,
        ) as capture:
            for _ in range(_PHYSICS_SUBSTEPS):
                mjw.step(self._model, self._data)
            # Builder calls mj_forward after frameskip; reward, contacts, sensors,
            # and lidar transforms must all observe this final integrated state.
            mjw.forward(self._model, self._data)
        self._physics_graph = capture.graph

        # Eager warmup advanced Data.  Rebuild the exact sampled episode start
        # while retaining the graph's already allocated workspaces.
        reset_stream = self._warp_stream()
        with wp.ScopedStream(reset_stream, sync_enter=False):
            mjw.reset_data(self._model, self._data)
            mjw.set_const_0(self._model, self._data, restore=True)
            mjw.forward(self._model, self._data)
        self._ctrl_t.copy_(saved_ctrl)
        torch.cuda.synchronize(self.device)

    def _pseudo_lidar(self, object_xy):
        """Vectorize upstream pseudo-lidar with the paper config0/1 max distance."""
        agent_pos = self._xpos_t[:, self._ids.agent_body]
        agent_mat = self._xmat_t[:, self._ids.agent_body]

        # Object z is zero in upstream lidar.  Multiplication by Point's xmat
        # converts the world-space row vector into the agent frame.
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
        bins = torch.clamp(
            torch.floor(scaled_angle).to(torch.int64),
            min=0,
            max=_LIDAR_BINS - 1,
        )
        sensor = torch.clamp(
            (_LIDAR_MAX_DIST - distance) / _LIDAR_MAX_DIST,
            min=0.0,
        )

        # Closest-object reduction plus left/right aliases matches upstream's
        # three-bin soft angular encoding without Python object loops.
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
        """Assemble the task-specific flattened OrderedDict observation on CUDA."""
        goal = self._layout_t[:, self._goal_layout_index:self._goal_layout_index + 1]
        hazards = self._layout_t[
            :, self._hazard_layout_start:self._hazard_layout_start + _NUM_HAZARDS
        ]
        sensors = self._sensordata_t[:, :12]

        if self._task_kind == _TASK_DYNAMIC:
            # GoalLevel0 order: Point sensors, goal lidar, hazards lidar.
            return torch.cat(
                (sensors, self._pseudo_lidar(goal), self._pseudo_lidar(hazards)),
                dim=1,
            ).to(torch.float32)

        buttons = self._layout_t[
            :, _SIMPLE_FIRST_BUTTON_INDEX:_SIMPLE_FIRST_BUTTON_INDEX + _NUM_SIMPLE_BUTTONS
        ]
        button_lidar = self._pseudo_lidar(buttons)
        # ButtonLevel0 hides all physical buttons while the resampling timer is
        # nonzero; goal/hazard lidar and proprioception remain visible.
        button_lidar = torch.where(
            self._button_timer_t[:, None] == 0,
            button_lidar,
            torch.zeros_like(button_lidar),
        )
        return torch.cat(
            (
                sensors,
                button_lidar,
                self._pseudo_lidar(goal),
                self._pseudo_lidar(hazards),
            ),
            dim=1,
        ).to(torch.float32)

    # ========================================================================
    # Reward, cost, and goal-state transitions
    # ========================================================================
    def _hazard_violation(self):
        """Return the indicator-equivalent union of all three hazard costs."""
        agent_xy = self._xpos_t[:, self._ids.agent_body, :2]
        hazard_xy = self._layout_t[
            :, self._hazard_layout_start:self._hazard_layout_start + _NUM_HAZARDS
        ]
        distance = torch.linalg.vector_norm(hazard_xy - agent_xy[:, None, :], dim=-1)
        # Upstream sums max(size-distance, 0), then Builder applies >0 indicator.
        shaped_cost = torch.clamp(_HAZARD_SIZE - distance, min=0.0).sum(dim=1)
        return shaped_cost > 0.0

    def _button_contact_flags(self):
        """Return per-world (goal-button contact, wrong-button contact) flags."""
        geom0 = self._contact_geom_t[:, 0]
        geom1 = self._contact_geom_t[:, 1]
        world_raw = self._contact_world_t.to(torch.int64)

        # MJWarp reuses over-allocated contact rows.  nacon marks the active
        # prefix; world bounds additionally protect scatter from stale row IDs.
        active_count = torch.clamp(
            self._nacon_t[0], min=0, max=self._contact_world_t.numel()
        )
        valid_row = self._contact_capacity_row_t < active_count
        valid_world = (world_raw >= 0) & (world_raw < self.B)
        safe_world = torch.clamp(world_raw, min=0, max=self.B - 1)

        is_agent = torch.zeros_like(geom0, dtype=torch.bool)
        for geom in self._ids.agent_geoms:
            is_agent |= (geom0 == geom) | (geom1 == geom)

        # Encode which physical button appears in each active contact.  The
        # other geom must be an agent geom before the row can become a task event.
        button_index = torch.full_like(world_raw, -1)
        for index, geom in enumerate(self._ids.button_geoms):
            has_button = (geom0 == geom) | (geom1 == geom)
            button_index = torch.where(
                has_button,
                torch.full_like(button_index, index),
                button_index,
            )

        task_contact = valid_row & valid_world & is_agent & (button_index >= 0)
        selected = self._goal_button_t[safe_world].to(torch.int64)
        goal_contact = task_contact & (button_index == selected)
        wrong_contact = task_contact & (button_index != selected)

        # Multiple contact points with one button still represent one per-world
        # indicator after Builder's constrain_indicator transformation.
        goal_hits = torch.zeros(self.B, dtype=torch.int32, device=self.device)
        wrong_hits = torch.zeros(self.B, dtype=torch.int32, device=self.device)
        goal_hits.scatter_add_(0, safe_world, goal_contact.to(torch.int32))
        wrong_hits.scatter_add_(0, safe_world, wrong_contact.to(torch.int32))
        return goal_hits > 0, wrong_hits > 0

    def _advance_goal_state(self, reached):
        """Apply Builder's specific_step then update_world ordering on CUDA."""
        reached_contiguous = reached.contiguous()
        reached_w = wp.from_torch(reached_contiguous)
        stream = self._warp_stream()

        if self._task_kind == _TASK_DYNAMIC:
            # update_layout refreshes moving agent XY before a new goal is tested
            # against placement keepouts; hazards are static and already current.
            agent_xy = self._xpos_t[:, self._ids.agent_body, :2]
            self._layout_t[:, _AGENT_LAYOUT_INDEX] = torch.where(
                reached[:, None],
                agent_xy,
                self._layout_t[:, _AGENT_LAYOUT_INDEX],
            )
            with wp.ScopedStream(stream, sync_enter=False):
                wp.launch(
                    _resample_dynamic_goal_kernel,
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
        else:
            # Builder computes reward/cost first, then timer_tick, then tests
            # goal contact and resets timer to ten if a goal was achieved.
            self._button_timer_t.sub_(1)
            self._button_timer_t.clamp_(min=0)
            with wp.ScopedStream(stream, sync_enter=False):
                wp.launch(
                    _select_next_button_kernel,
                    dim=self.B,
                    inputs=[
                        self._seed,
                        reached_w,
                        self._reset_count,
                        self._goal_count,
                        self._layout,
                        self._goal_button,
                        self._button_timer,
                    ],
                    device=self._warp_device,
                )

        # Goal is a collision-disabled visualization body.  Task calculations
        # consume layout directly and the next physics forward sees this body_pos.
        self._body_pos_t[:, self._ids.goal_body, :2] = self._layout_t[
            :, self._goal_layout_index
        ]

    def _raise_if_goal_resample_failed(self):
        """Reject Dynamic rollouts whose exceptional 10,000-try resample failed."""
        failed = self._failure_t != 0
        if not bool(torch.any(failed).item()):
            return
        slots = torch.nonzero(failed, as_tuple=False).flatten().detach().cpu().tolist()
        raise RuntimeError(
            "MJWarp task goal resampling exhausted 10,000 attempts; "
            f"terminated slots={slots}. Refusing to continue an invalid state."
        )

    # ========================================================================
    # Public vector API
    # ========================================================================
    def reset(self):
        """Start a vector segment with one fully GPU-sampled task state per slot."""
        self._chain_caller_stream()
        self._raise_if_physics_overflow()
        self._raise_if_goal_resample_failed()
        self.step_count = 0
        self._cost_buf = []
        self._reward_buf = []
        self._sample_full_layout()

        observation = self._observation()
        agent_xy = self._xpos_t[:, self._ids.agent_body, :2]
        self._last_goal_distance = torch.linalg.vector_norm(
            self._layout_t[:, self._goal_layout_index] - agent_xy,
            dim=1,
        )
        return observation

    def step(self, actions):
        """Advance all CUDA worlds while preserving Builder's task-step ordering."""
        self._chain_caller_stream()
        executed_action = self._physics_step(actions)
        self.step_count += 1

        agent_xy = self._xpos_t[:, self._ids.agent_body, :2]
        current_goal = self._layout_t[:, self._goal_layout_index].clone()
        current_distance = torch.linalg.vector_norm(current_goal - agent_xy, dim=1)

        if self._task_kind == _TASK_DYNAMIC:
            reached = current_distance <= _GOAL_SIZE
            wrong_button = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        else:
            reached, wrong_button = self._button_contact_flags()

        # Both upstream tasks use distance progress plus one sparse goal reward.
        reward = self._last_goal_distance - current_distance
        reward = reward + reached.to(reward.dtype)

        hazard_violation = self._hazard_violation()
        button_violation = wrong_button & (self._button_timer_t == 0)
        cost = (hazard_violation | button_violation).to(torch.float32)

        self._advance_goal_state(reached)
        next_goal_distance = torch.linalg.vector_norm(
            self._layout_t[:, self._goal_layout_index] - agent_xy,
            dim=1,
        )
        self._last_goal_distance = torch.where(
            reached,
            next_goal_distance,
            current_distance,
        )

        final_observation = self._observation()
        force_truncation = self.step_count >= min(self.n, self._native_horizon)
        terminated = self._failure_t.to(torch.bool)
        truncated = torch.full(
            (self.B,),
            force_truncation,
            dtype=torch.bool,
            device=self.device,
        ) & ~terminated
        final_mask = terminated | truncated

        # Historical explicit-clip mode resets synchronously inside step.  The
        # author-aligned mode defers reset so checkpoint code snapshots pre-reset
        # RNG counters using the same contract as the CPU vector adapters.
        observation = final_observation
        if force_truncation and self.action_mode != SafetyEnv.AUTHOR_RAW_ENV_CLAMP:
            self._raise_if_physics_overflow()
            self._raise_if_goal_resample_failed()
            self._sample_full_layout()
            observation = self._observation()
            reset_agent_xy = self._xpos_t[:, self._ids.agent_body, :2]
            self._last_goal_distance = torch.linalg.vector_norm(
                self._layout_t[:, self._goal_layout_index] - reset_agent_xy,
                dim=1,
            )

        # Retain only per-step batch means: diagnostics stay O(T), not O(B*T).
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
