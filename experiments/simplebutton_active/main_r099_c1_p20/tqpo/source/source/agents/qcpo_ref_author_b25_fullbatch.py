# -*- coding: utf-8 -*-
"""QRefAA 的独立 B25-only-B 执行适配器。

作者对齐实现 :class:`QCPORefAuthorAlignedGPU` 有意把公开身份冻结在
``T=1000, B=12 -> new_T=100, new_B=120``。正式 B25 对照不能直接修改该类，
否则历史 B12 checkpoint 会与新实验共享类名和签名。本模块因此注册一个独立身份：

``QCPO_REF_AUTHOR_B25``

它只改变原始 rollout 的轨迹数及由此必然派生的几何/预算字段：

* 原始采样从 ``[1000, 12]`` 改为 ``[1000, 25]``；
* 作者的双射变换从 ``[100, 120]`` 改为 ``[100, 250]``；
* 400 个 rollout 恰好产生 10,000,000 次物理 transition；
* 第一个 rollout 仍只更新 observation RMS 与 dual，不执行 Adam；
* 其余 399 个 rollout 各执行固定 E8、单个完整 recurrent batch，
  因而总计 ``399 * 8 = 3192`` 次 Adam step。
* 正式执行信封固定 MJWarp-3.12 train/CRN555 validation，并由孤立
  MuJoCo-3.12.0 CPU/MP implicit-fast 栈执行 CRN777 终测；这些字段进入 factor tuple 与
  checkpoint 身份，且与 DQC B25 使用同一运行 profile。

网络、损失、target、PPO clipping、GAE、学习率、PID ``Ki``、deque-100、
动作边界、Weibull 与 RMS 时序全部继承父类。父类的完整 B12 validator 仍会执行：
适配器仅在校验期间把身份/几何临时投影回 B12，随后无条件恢复真实 B25 状态。
"""

from __future__ import annotations

import copy  # 浅复制参数命名空间；校验投影不得修改调用方的真实 B25 对象。
import json  # 以稳定 JSON 序列化 factor tuple，供 runner/checkpoint 重复签名。
from typing import Dict, Tuple  # 明确 checkpoint 与 factor tuple 的公开返回类型。

from .qcpo_ref_author_aligned import (
    QCPORefAuthorAlignedGPU,
    QCPO_AUTHOR_DENSITY_RULE,
    QCPO_AUTHOR_DUAL_RULE,
    QCPO_AUTHOR_HISTORY_TIMING,
    QCPO_AUTHOR_IMPLEMENTATION_ID,
    QCPO_AUTHOR_LAYOUT,
    QCPO_AUTHOR_POLICY_CLIP_RULE,
    QCPO_AUTHOR_QUANTILE_GRID,
    QCPO_AUTHOR_RMS_WARMUP,
    QCPO_AUTHOR_TARGET_ADAPTER,
    QCPO_AUTHOR_UPSTREAM_SHA,
    QCPO_AUTHOR_WEIBULL_IDENTITY,
    QCPO_FORMAL_DYNAMICS_PROFILE,
    QCPO_FORMAL_TRAIN_WARP_DETERMINISM,
    QCPO_FORMAL_TRAIN_WARP_DETERMINISTIC_MAX_RECORDS,
    QCPO_FORMAL_WANDB_PROJECT,
)


# ============================================================================
# 注册身份：B25 checkpoint 不得伪装成原生 B12 或历史 matched-B32
# ============================================================================

QCPO_REF_AUTHOR_B25_ALGO = "QCPO_REF_AUTHOR_B25"
QCPO_B25_IMPLEMENTATION_ID = (
    "qcpo_author_executable_aligned_reward_outage_shifted_weibull_"
    "B25_only_batch_geometry_mjwarp312_implicitfast_"
    "train_cpu312_implicitfast_eval_v3"
)
QCPO_B25_LAYOUT = (
    "T1000_B25_transform_T100_B250_chunk_bootstrap_early_end_fail_v1"
)
QCPO_B25_TRAINING_BUDGET_SEMANTICS = (
    "author_B25_exact_10m_first_batch_rms_only_v1"
)


# ============================================================================
# 精确几何：只把物理 trajectory batch 从 12 改成 25
# ============================================================================

QCPO_B25_RAW_TIME = 1000                    # 每条轨迹仍覆盖完整 Gremlin 截断时域。
QCPO_B25_RAW_BATCH = 25                     # 每个 rollout 收集 25 条完整轨迹。
QCPO_B25_ACTOR_TIME = 100                   # 作者 truncated-BPTT 时间长度保持不变。
QCPO_B25_ACTOR_BATCH = 250                  # 25 条轨迹各拆成 10 个有序 recurrent chunk。
QCPO_B25_TRANSITIONS = (
    QCPO_B25_RAW_TIME * QCPO_B25_RAW_BATCH
)                                             # 每个 rollout 恰好 25,000 transitions。


# ============================================================================
# 精确预算：400 * 25,000 = 10M，首批依旧属于主预算但不进入 Adam
# ============================================================================

QCPO_B25_NOMINAL_ENV_STEPS = 10_000_000     # 论文比较采用的物理交互总预算。
QCPO_B25_MAIN_BATCHES = 400                 # 10M / 25k，无需向上或向下取整。
QCPO_B25_MAIN_ENV_STEPS = (
    QCPO_B25_MAIN_BATCHES * QCPO_B25_TRANSITIONS
)                                             # 主时钟终点严格等于 10,000,000。
QCPO_B25_ACTUAL_ENV_STEPS = QCPO_B25_MAIN_ENV_STEPS

QCPO_B25_RMS_ONLY_BATCHES = 1               # batch-1 保留作者 RMS/dual-only 协议。
QCPO_B25_OPTIMIZED_MAIN_BATCHES = (
    QCPO_B25_MAIN_BATCHES - QCPO_B25_RMS_ONLY_BATCHES
)                                             # 只有后续 399 批进入共享 Adam。
QCPO_B25_OPTIMIZER_TRANSITION_COUNT = (
    QCPO_B25_OPTIMIZED_MAIN_BATCHES * QCPO_B25_TRANSITIONS
)                                             # 399 * 25k = 9,975,000 unique transitions。

QCPO_B25_OPTIMIZER_EPOCHS = 8               # 与作者 B12 完全相同的数据复用次数。
QCPO_B25_MINIBATCHES = 1                    # 每个 epoch 仍是一个完整 25k batch。
QCPO_B25_EXPECTED_OPTIMIZER_STEPS = (
    QCPO_B25_OPTIMIZED_MAIN_BATCHES * QCPO_B25_OPTIMIZER_EPOCHS
)                                             # 399 * 8 = 3192 次共享 Adam.step()。
QCPO_B25_EXPECTED_DUAL_UPDATES = QCPO_B25_MAIN_BATCHES
                                              # dual 包含首批，因此严格更新 400 次。

QCPO_B25_FORCED_MAIN_BATCHES = (200, 400)    # 5M 中点与 10M 终点的强制事件边界。
QCPO_B25_EVENT_SCHEDULE_OFFSET_BATCHES = 0   # 首批在 main clock 内，不存在隐藏 warmup 偏移。

# 正式 DQC/QRefAA 共同两阶段信封：训练/CRN555 validation 固定 MJWarp，
# 论文 CRN777 终测由孤立 MuJoCo-3.12.0 CPU/MP implicit-fast 进程执行。
# simulator 数值路径、Warp 确定性披露和 W&B destination 均属于可恢复身份。
QCPO_B25_FORMAL_ENDPOINT_PRIMARY = True
QCPO_B25_WANDB_PROJECT = QCPO_FORMAL_WANDB_PROJECT
QCPO_B25_FORMAL_EXECUTION_PROFILE = (
    "gremlin_mjwarp312_implicitfast_train_cpu312_final_eval_v2"
)
QCPO_B25_DYNAMICS_PROFILE = QCPO_FORMAL_DYNAMICS_PROFILE
QCPO_B25_TRAIN_VEC_BACKEND = "mjwarp"
QCPO_B25_VALIDATION_VEC_BACKEND = "mjwarp"
QCPO_B25_VALIDATION_NUM_ENVS = 25
QCPO_B25_EVAL_VEC_BACKEND = "mp"
QCPO_B25_EVAL_NUM_ENVS = 25
QCPO_B25_TRAIN_SIMULATOR_IDENTITY = "mjwarp-implicitfast-v1"
QCPO_B25_TRAIN_MUJOCO_VERSION = "3.12.0"
QCPO_B25_TRAIN_MUJOCO_WARP_VERSION = "3.12.0"
QCPO_B25_TRAIN_WARP_VERSION = "1.16.0"
QCPO_B25_TRAIN_WARP_DETERMINISM = QCPO_FORMAL_TRAIN_WARP_DETERMINISM
QCPO_B25_TRAIN_WARP_DETERMINISTIC_MAX_RECORDS = (
    QCPO_FORMAL_TRAIN_WARP_DETERMINISTIC_MAX_RECORDS
)
QCPO_B25_TRAIN_MUJOCO_GL = "disable"
QCPO_B25_CANONICAL_EVAL_STACK = (
    "zprl_mujoco_3.12.0_cpu_mp_implicitfast_v1"
)
QCPO_B25_FORMAL_EVALUATION_PROTOCOL = (
    "gremlin_mjwarp312_implicitfast_validation_B25_crn555_"
    "cpu312_implicitfast_final_B25_crn777_v2"
)


# 该字符串只用于调用父类 validator 时的临时历史投影；任何公开 B25 状态均不得暴露它。
_PARENT_B12_BUDGET_SEMANTICS = (
    "author_nominal_10m_round_up_log_interval_first_batch_rms_only_v1"
)


def qcpo_b25_factor_tuple() -> Tuple[Tuple[str, object], ...]:
    """返回稳定的 B25 科学身份，显式区分继承项与唯一变化项。

    返回值不依赖命令行或主机路径。正式 W&B project 作为固定 provenance 显式签名；
    config、checkpoint signature 和结果清单，从而防止 B12/B25/B32 交叉恢复。
    """

    return (
        # ------------------------------ 完整继承的作者科学定义
        ("algorithm", QCPO_REF_AUTHOR_B25_ALGO),
        ("implementation", QCPO_B25_IMPLEMENTATION_ID),
        ("parent_implementation", QCPO_AUTHOR_IMPLEMENTATION_ID),
        ("upstream_sha", QCPO_AUTHOR_UPSTREAM_SHA),
        ("target_adapter", QCPO_AUTHOR_TARGET_ADAPTER),
        ("history_timing", QCPO_AUTHOR_HISTORY_TIMING),
        ("density_rule", QCPO_AUTHOR_DENSITY_RULE),
        ("dual_rule", QCPO_AUTHOR_DUAL_RULE),
        ("quantile_grid", QCPO_AUTHOR_QUANTILE_GRID),
        ("weibull_identity", QCPO_AUTHOR_WEIBULL_IDENTITY),
        ("policy_clip_rule", QCPO_AUTHOR_POLICY_CLIP_RULE),
        ("rms_warmup", QCPO_AUTHOR_RMS_WARMUP),

        # ------------------------------ 唯一科学变化及其确定性派生量
        ("raw_rollout_layout", "T1000_B25"),
        ("actor_sequence_layout", "T100_B250"),
        ("transitions_per_rollout", QCPO_B25_TRANSITIONS),
        ("optimizer_epochs", QCPO_B25_OPTIMIZER_EPOCHS),
        ("minibatches_per_epoch", QCPO_B25_MINIBATCHES),
        ("first_batch", "dual_and_rms_only_included_in_main_budget_v1"),
        ("main_rollout_batches", QCPO_B25_MAIN_BATCHES),
        ("optimized_main_batches", QCPO_B25_OPTIMIZED_MAIN_BATCHES),
        ("main_env_steps", QCPO_B25_MAIN_ENV_STEPS),
        ("actual_env_steps", QCPO_B25_ACTUAL_ENV_STEPS),
        ("optimizer_transition_count", QCPO_B25_OPTIMIZER_TRANSITION_COUNT),
        ("expected_optimizer_steps", QCPO_B25_EXPECTED_OPTIMIZER_STEPS),
        ("expected_dual_updates", QCPO_B25_EXPECTED_DUAL_UPDATES),
        ("forced_main_batches", QCPO_B25_FORCED_MAIN_BATCHES),
        ("event_schedule_offset_batches", QCPO_B25_EVENT_SCHEDULE_OFFSET_BATCHES),
        ("formal_endpoint_primary", QCPO_B25_FORMAL_ENDPOINT_PRIMARY),
        ("wandb_project", QCPO_B25_WANDB_PROJECT),
        ("formal_execution_profile", QCPO_B25_FORMAL_EXECUTION_PROFILE),
        ("dynamics_profile", QCPO_B25_DYNAMICS_PROFILE),
        ("train_vec_backend", QCPO_B25_TRAIN_VEC_BACKEND),
        ("validation_vec_backend", QCPO_B25_VALIDATION_VEC_BACKEND),
        ("validation_num_envs", QCPO_B25_VALIDATION_NUM_ENVS),
        ("eval_vec_backend", QCPO_B25_EVAL_VEC_BACKEND),
        ("eval_num_envs", QCPO_B25_EVAL_NUM_ENVS),
        ("train_simulator_identity", QCPO_B25_TRAIN_SIMULATOR_IDENTITY),
        ("train_mujoco_version", QCPO_B25_TRAIN_MUJOCO_VERSION),
        ("train_mujoco_warp_version", QCPO_B25_TRAIN_MUJOCO_WARP_VERSION),
        ("train_warp_version", QCPO_B25_TRAIN_WARP_VERSION),
        ("train_warp_determinism", QCPO_B25_TRAIN_WARP_DETERMINISM),
        ("train_warp_deterministic_max_records",
         QCPO_B25_TRAIN_WARP_DETERMINISTIC_MAX_RECORDS),
        ("train_mujoco_gl", QCPO_B25_TRAIN_MUJOCO_GL),
        ("canonical_eval_stack", QCPO_B25_CANONICAL_EVAL_STACK),
        ("formal_evaluation_protocol", QCPO_B25_FORMAL_EVALUATION_PROTOCOL),
        ("registered_change", "raw_B12_to_B25_and_derived_budget_only_v1"),
    )


def canonical_qcpo_b25_factor_tuple() -> str:
    """以无空白、ASCII、稳定顺序编码 B25 factor tuple。"""

    return json.dumps(
        qcpo_b25_factor_tuple(),
        separators=(",", ":"),
        ensure_ascii=True,
    )


def apply_qcpo_ref_author_b25_configuration(args):
    """在完整 B12 namespace 上安装 B25 几何、预算和独立身份。

    调用者必须先通过 ``base_args('QCPO_REF_AUTHOR_ALIGNED', ...)`` 获得作者 B12
    的全部配置。本函数不会设置 W&B、文件路径、device 或 seed，也不会创建模型/环境；
    它只替换 B、new_B 与这些字段数学上必然决定的预算/签名信息。
    """

    # ------------------------------ 独立算法/实现身份：禁止加载原生 B12 checkpoint
    args.algo_name = QCPO_REF_AUTHOR_B25_ALGO
    args.training_budget_semantics = QCPO_B25_TRAINING_BUDGET_SEMANTICS
    args.qcpo_author_implementation_identity = QCPO_B25_IMPLEMENTATION_ID
    args.qcpo_author_layout = QCPO_B25_LAYOUT
    args.qcpo_ref_formulation = QCPO_B25_IMPLEMENTATION_ID

    # ------------------------------ 采样与 recurrent transform：T/new_T 均保持不变
    args.horizon = QCPO_B25_RAW_TIME
    args.num_envs = QCPO_B25_RAW_BATCH
    args.new_T = QCPO_B25_ACTOR_TIME
    args.new_B = QCPO_B25_ACTOR_BATCH

    # E8/M1 虽与父配置数值相同，仍显式回写以让 B25 公共校验拒绝半配置 namespace。
    args.epochs = QCPO_B25_OPTIMIZER_EPOCHS
    args.minibatches = QCPO_B25_MINIBATCHES

    # ------------------------------ 10M 预算：首批算物理交互，但不算神经优化数据
    args.num_iterations = QCPO_B25_MAIN_BATCHES
    args.qcpo_author_nominal_budget_steps = QCPO_B25_NOMINAL_ENV_STEPS
    args.qcpo_author_actual_env_steps = QCPO_B25_ACTUAL_ENV_STEPS
    args.qcpo_author_rollout_batches = QCPO_B25_MAIN_BATCHES
    args.qcpo_author_optimizer_transition_count = (
        QCPO_B25_OPTIMIZER_TRANSITION_COUNT
    )
    args.qcpo_author_milestone_batch = QCPO_B25_FORCED_MAIN_BATCHES[0]
    args.forced_event_batches = list(QCPO_B25_FORCED_MAIN_BATCHES)
    args.event_schedule_offset_batches = QCPO_B25_EVENT_SCHEDULE_OFFSET_BATCHES

    # ------------------------------ 两算法共同的正式执行/评估协议
    # adapter 主动安装 MJWarp 训练后端；CPU 终测必须由另一进程加载 checkpoint。
    args.formal_endpoint_primary = QCPO_B25_FORMAL_ENDPOINT_PRIMARY
    args.wandb_project = QCPO_B25_WANDB_PROJECT
    args.formal_execution_profile = QCPO_B25_FORMAL_EXECUTION_PROFILE
    args.dynamics_profile = QCPO_B25_DYNAMICS_PROFILE
    args.vec_backend = QCPO_B25_TRAIN_VEC_BACKEND
    args.train_vec_backend = QCPO_B25_TRAIN_VEC_BACKEND
    args.validation_vec_backend = QCPO_B25_VALIDATION_VEC_BACKEND
    args.validation_num_envs = QCPO_B25_VALIDATION_NUM_ENVS
    args.eval_vec_backend = QCPO_B25_EVAL_VEC_BACKEND
    args.eval_num_envs = QCPO_B25_EVAL_NUM_ENVS
    args.train_simulator_identity = QCPO_B25_TRAIN_SIMULATOR_IDENTITY
    args.train_mujoco_version = QCPO_B25_TRAIN_MUJOCO_VERSION
    args.train_mujoco_warp_version = QCPO_B25_TRAIN_MUJOCO_WARP_VERSION
    args.train_warp_version = QCPO_B25_TRAIN_WARP_VERSION
    args.train_warp_determinism = QCPO_B25_TRAIN_WARP_DETERMINISM
    args.train_warp_deterministic_max_records = (
        QCPO_B25_TRAIN_WARP_DETERMINISTIC_MAX_RECORDS
    )
    args.train_mujoco_gl = QCPO_B25_TRAIN_MUJOCO_GL
    args.canonical_eval_stack = QCPO_B25_CANONICAL_EVAL_STACK
    args.formal_evaluation_protocol = QCPO_B25_FORMAL_EVALUATION_PROTOCOL

    # ------------------------------ 冗余注册字段：外层 signature 变弱时仍无法串 checkpoint
    factor_tuple = canonical_qcpo_b25_factor_tuple()
    args.qcpo_b25_implementation_identity = QCPO_B25_IMPLEMENTATION_ID
    args.qcpo_b25_factor_tuple = factor_tuple
    args.qcpo_b25_layout = QCPO_B25_LAYOUT
    args.qcpo_b25_raw_layout = "T1000_B25"
    args.qcpo_b25_actor_layout = "T100_B250"
    args.qcpo_b25_transitions_per_rollout = QCPO_B25_TRANSITIONS
    args.qcpo_b25_optimizer_epochs = QCPO_B25_OPTIMIZER_EPOCHS
    args.qcpo_b25_minibatches = QCPO_B25_MINIBATCHES
    args.qcpo_b25_first_batch = "dual_and_rms_only_included_in_main_budget_v1"
    args.qcpo_b25_rms_only_batches = QCPO_B25_RMS_ONLY_BATCHES
    args.qcpo_b25_main_batches = QCPO_B25_MAIN_BATCHES
    args.qcpo_b25_optimized_main_batches = QCPO_B25_OPTIMIZED_MAIN_BATCHES
    args.qcpo_b25_main_env_steps = QCPO_B25_MAIN_ENV_STEPS
    args.qcpo_b25_actual_env_steps = QCPO_B25_ACTUAL_ENV_STEPS
    args.qcpo_b25_optimizer_transition_count = QCPO_B25_OPTIMIZER_TRANSITION_COUNT
    args.qcpo_b25_expected_optimizer_steps = QCPO_B25_EXPECTED_OPTIMIZER_STEPS
    args.qcpo_b25_expected_dual_updates = QCPO_B25_EXPECTED_DUAL_UPDATES
    args.qcpo_b25_forced_main_batches = list(QCPO_B25_FORCED_MAIN_BATCHES)
    args.qcpo_b25_event_schedule_offset_batches = (
        QCPO_B25_EVENT_SCHEDULE_OFFSET_BATCHES
    )
    return args


class QCPORefAuthorB25GPU(QCPORefAuthorAlignedGPU):
    """仅改变 B12 采样几何与派生预算的 author-aligned QRefAA。"""

    # VecAgentBase 在分配 vector workers 之前验证时钟语义；这里只接受唯一 B25 身份。
    SUPPORTED_TRAINING_BUDGET_SEMANTICS = (
        QCPO_B25_TRAINING_BUDGET_SEMANTICS,
    )

    def __init__(self, args, env):
        """先拒绝不完整 B25 namespace，再让父类构造真实 25-slot runtime。"""

        # 公共字段校验发生在环境、模型和 W&B 副作用之前。
        self._validate_public_b25_configuration(args)

        # 父构造器将读取真实 B25 几何；只有其内部 author validator 会看到临时 B12 投影。
        super().__init__(args, env)

        # 以下字段不含可学习参数，只为 checkpoint/result 审计保留重复的运行时证据。
        self.b25_raw_time = QCPO_B25_RAW_TIME
        self.b25_raw_batch = QCPO_B25_RAW_BATCH
        self.b25_actor_time = QCPO_B25_ACTOR_TIME
        self.b25_actor_batch = QCPO_B25_ACTOR_BATCH
        self.b25_transitions = QCPO_B25_TRANSITIONS
        self.b25_main_batches = QCPO_B25_MAIN_BATCHES
        self.b25_optimized_main_batches = QCPO_B25_OPTIMIZED_MAIN_BATCHES
        self.b25_main_env_steps = QCPO_B25_MAIN_ENV_STEPS
        self.b25_actual_env_steps = QCPO_B25_ACTUAL_ENV_STEPS
        self.b25_expected_optimizer_steps = QCPO_B25_EXPECTED_OPTIMIZER_STEPS
        self.b25_expected_dual_updates = QCPO_B25_EXPECTED_DUAL_UPDATES
        self.b25_factor_tuple = canonical_qcpo_b25_factor_tuple()
        self.formal_endpoint_primary = QCPO_B25_FORMAL_ENDPOINT_PRIMARY
        self.formal_wandb_project = QCPO_B25_WANDB_PROJECT
        self.formal_execution_profile = QCPO_B25_FORMAL_EXECUTION_PROFILE
        self.dynamics_profile = QCPO_B25_DYNAMICS_PROFILE
        self.train_vec_backend = QCPO_B25_TRAIN_VEC_BACKEND
        self.validation_vec_backend = QCPO_B25_VALIDATION_VEC_BACKEND
        self.validation_num_envs = QCPO_B25_VALIDATION_NUM_ENVS
        self.eval_vec_backend = QCPO_B25_EVAL_VEC_BACKEND
        self.eval_num_envs = QCPO_B25_EVAL_NUM_ENVS
        self.train_simulator_identity = QCPO_B25_TRAIN_SIMULATOR_IDENTITY
        self.train_mujoco_version = QCPO_B25_TRAIN_MUJOCO_VERSION
        self.train_mujoco_warp_version = QCPO_B25_TRAIN_MUJOCO_WARP_VERSION
        self.train_warp_version = QCPO_B25_TRAIN_WARP_VERSION
        self.train_warp_determinism = QCPO_B25_TRAIN_WARP_DETERMINISM
        self.train_warp_deterministic_max_records = (
            QCPO_B25_TRAIN_WARP_DETERMINISTIC_MAX_RECORDS
        )
        self.train_mujoco_gl = QCPO_B25_TRAIN_MUJOCO_GL
        self.canonical_eval_stack = QCPO_B25_CANONICAL_EVAL_STACK
        self.formal_evaluation_protocol = (
            QCPO_B25_FORMAL_EVALUATION_PROTOCOL
        )
        self.event_schedule_offset_batches = QCPO_B25_EVENT_SCHEDULE_OFFSET_BATCHES

    @staticmethod
    def _validate_public_b25_configuration(args) -> None:
        """逐项验证 B25 身份、几何、预算以及固定 E8/M1 协议。"""

        required = {
            # -------------------------- 公共算法与父实现谱系
            "algorithm": (
                str(getattr(args, "algo_name", "")),
                QCPO_REF_AUTHOR_B25_ALGO,
            ),
            "training_budget_semantics": (
                getattr(args, "training_budget_semantics", None),
                QCPO_B25_TRAINING_BUDGET_SEMANTICS,
            ),
            "author_implementation": (
                getattr(args, "qcpo_author_implementation_identity", None),
                QCPO_B25_IMPLEMENTATION_ID,
            ),
            "author_layout": (
                getattr(args, "qcpo_author_layout", None),
                QCPO_B25_LAYOUT,
            ),
            "author_formulation": (
                getattr(args, "qcpo_ref_formulation", None),
                QCPO_B25_IMPLEMENTATION_ID,
            ),
            "registered_implementation": (
                getattr(args, "qcpo_b25_implementation_identity", None),
                QCPO_B25_IMPLEMENTATION_ID,
            ),
            "registered_factor_tuple": (
                getattr(args, "qcpo_b25_factor_tuple", None),
                canonical_qcpo_b25_factor_tuple(),
            ),
            "registered_layout": (
                getattr(args, "qcpo_b25_layout", None),
                QCPO_B25_LAYOUT,
            ),

            # -------------------------- 唯一改变的 B 与保持不变的 T/new_T/E/M
            "horizon": (
                int(getattr(args, "horizon", -1)),
                QCPO_B25_RAW_TIME,
            ),
            "num_envs": (
                int(getattr(args, "num_envs", -1)),
                QCPO_B25_RAW_BATCH,
            ),
            "new_T": (
                int(getattr(args, "new_T", -1)),
                QCPO_B25_ACTOR_TIME,
            ),
            "new_B": (
                int(getattr(args, "new_B", -1)),
                QCPO_B25_ACTOR_BATCH,
            ),
            "epochs": (
                int(getattr(args, "epochs", -1)),
                QCPO_B25_OPTIMIZER_EPOCHS,
            ),
            "minibatches": (
                int(getattr(args, "minibatches", -1)),
                QCPO_B25_MINIBATCHES,
            ),

            # -------------------------- 400-rollout/10M 预算与父类兼容字段
            "main_batches": (
                int(getattr(args, "num_iterations", -1)),
                QCPO_B25_MAIN_BATCHES,
            ),
            "author_nominal_steps": (
                int(getattr(args, "qcpo_author_nominal_budget_steps", -1)),
                QCPO_B25_NOMINAL_ENV_STEPS,
            ),
            "author_rollout_batches": (
                int(getattr(args, "qcpo_author_rollout_batches", -1)),
                QCPO_B25_MAIN_BATCHES,
            ),
            "author_actual_steps": (
                int(getattr(args, "qcpo_author_actual_env_steps", -1)),
                QCPO_B25_ACTUAL_ENV_STEPS,
            ),
            "author_optimizer_transitions": (
                int(getattr(
                    args, "qcpo_author_optimizer_transition_count", -1)),
                QCPO_B25_OPTIMIZER_TRANSITION_COUNT,
            ),
            "author_milestone": (
                int(getattr(args, "qcpo_author_milestone_batch", -1)),
                QCPO_B25_FORCED_MAIN_BATCHES[0],
            ),
            "forced_batches": (
                tuple(int(value) for value in getattr(
                    args, "forced_event_batches", ())),
                QCPO_B25_FORCED_MAIN_BATCHES,
            ),
            "event_offset": (
                int(getattr(args, "event_schedule_offset_batches", -1)),
                QCPO_B25_EVENT_SCHEDULE_OFFSET_BATCHES,
            ),
            # -------------------------- DQC/QRefAA 共同执行信封
            "formal_endpoint_primary": (
                getattr(args, "formal_endpoint_primary", None),
                QCPO_B25_FORMAL_ENDPOINT_PRIMARY,
            ),
            "wandb_project": (
                str(getattr(args, "wandb_project", "")),
                QCPO_B25_WANDB_PROJECT,
            ),
            "formal_execution_profile": (
                str(getattr(args, "formal_execution_profile", "")),
                QCPO_B25_FORMAL_EXECUTION_PROFILE,
            ),
            "dynamics_profile": (
                str(getattr(args, "dynamics_profile", "")),
                QCPO_B25_DYNAMICS_PROFILE,
            ),
            "train_vec_backend": (
                str(getattr(args, "train_vec_backend", "")),
                QCPO_B25_TRAIN_VEC_BACKEND,
            ),
            "runtime_vec_backend": (
                str(getattr(args, "vec_backend", "")),
                QCPO_B25_TRAIN_VEC_BACKEND,
            ),
            "train_backend_matches_vec_backend": (
                str(getattr(args, "train_vec_backend", "")),
                str(getattr(args, "vec_backend", "")),
            ),
            "validation_vec_backend": (
                str(getattr(args, "validation_vec_backend", "")),
                QCPO_B25_VALIDATION_VEC_BACKEND,
            ),
            "validation_num_envs": (
                int(getattr(args, "validation_num_envs", -1)),
                QCPO_B25_VALIDATION_NUM_ENVS,
            ),
            "eval_vec_backend": (
                str(getattr(args, "eval_vec_backend", "")),
                QCPO_B25_EVAL_VEC_BACKEND,
            ),
            "eval_num_envs": (
                int(getattr(args, "eval_num_envs", -1)),
                QCPO_B25_EVAL_NUM_ENVS,
            ),
            "train_simulator_identity": (
                str(getattr(args, "train_simulator_identity", "")),
                QCPO_B25_TRAIN_SIMULATOR_IDENTITY,
            ),
            "train_mujoco_version": (
                str(getattr(args, "train_mujoco_version", "")),
                QCPO_B25_TRAIN_MUJOCO_VERSION,
            ),
            "train_mujoco_warp_version": (
                str(getattr(args, "train_mujoco_warp_version", "")),
                QCPO_B25_TRAIN_MUJOCO_WARP_VERSION,
            ),
            "train_warp_version": (
                str(getattr(args, "train_warp_version", "")),
                QCPO_B25_TRAIN_WARP_VERSION,
            ),
            "train_warp_determinism": (
                str(getattr(args, "train_warp_determinism", "")),
                QCPO_B25_TRAIN_WARP_DETERMINISM,
            ),
            "train_warp_deterministic_max_records": (
                (type(getattr(args, "train_warp_deterministic_max_records", None)),
                 getattr(args, "train_warp_deterministic_max_records", None)),
                (int, QCPO_B25_TRAIN_WARP_DETERMINISTIC_MAX_RECORDS),
            ),
            "train_mujoco_gl": (
                str(getattr(args, "train_mujoco_gl", "")),
                QCPO_B25_TRAIN_MUJOCO_GL,
            ),
            "canonical_eval_stack": (
                str(getattr(args, "canonical_eval_stack", "")),
                QCPO_B25_CANONICAL_EVAL_STACK,
            ),
            "formal_evaluation_protocol": (
                str(getattr(args, "formal_evaluation_protocol", "")),
                QCPO_B25_FORMAL_EVALUATION_PROTOCOL,
            ),

            # -------------------------- B25 冗余审计字段
            "registered_raw_layout": (
                getattr(args, "qcpo_b25_raw_layout", None),
                "T1000_B25",
            ),
            "registered_actor_layout": (
                getattr(args, "qcpo_b25_actor_layout", None),
                "T100_B250",
            ),
            "registered_transitions": (
                int(getattr(args, "qcpo_b25_transitions_per_rollout", -1)),
                QCPO_B25_TRANSITIONS,
            ),
            "registered_epochs": (
                int(getattr(args, "qcpo_b25_optimizer_epochs", -1)),
                QCPO_B25_OPTIMIZER_EPOCHS,
            ),
            "registered_minibatches": (
                int(getattr(args, "qcpo_b25_minibatches", -1)),
                QCPO_B25_MINIBATCHES,
            ),
            "registered_first_batch": (
                getattr(args, "qcpo_b25_first_batch", None),
                "dual_and_rms_only_included_in_main_budget_v1",
            ),
            "registered_rms_only_batches": (
                int(getattr(args, "qcpo_b25_rms_only_batches", -1)),
                QCPO_B25_RMS_ONLY_BATCHES,
            ),
            "registered_main_batches": (
                int(getattr(args, "qcpo_b25_main_batches", -1)),
                QCPO_B25_MAIN_BATCHES,
            ),
            "registered_optimized_batches": (
                int(getattr(args, "qcpo_b25_optimized_main_batches", -1)),
                QCPO_B25_OPTIMIZED_MAIN_BATCHES,
            ),
            "registered_main_steps": (
                int(getattr(args, "qcpo_b25_main_env_steps", -1)),
                QCPO_B25_MAIN_ENV_STEPS,
            ),
            "registered_actual_steps": (
                int(getattr(args, "qcpo_b25_actual_env_steps", -1)),
                QCPO_B25_ACTUAL_ENV_STEPS,
            ),
            "registered_optimizer_transitions": (
                int(getattr(
                    args, "qcpo_b25_optimizer_transition_count", -1)),
                QCPO_B25_OPTIMIZER_TRANSITION_COUNT,
            ),
            "registered_optimizer_steps": (
                int(getattr(args, "qcpo_b25_expected_optimizer_steps", -1)),
                QCPO_B25_EXPECTED_OPTIMIZER_STEPS,
            ),
            "registered_dual_updates": (
                int(getattr(args, "qcpo_b25_expected_dual_updates", -1)),
                QCPO_B25_EXPECTED_DUAL_UPDATES,
            ),
            "registered_forced_batches": (
                tuple(int(value) for value in getattr(
                    args, "qcpo_b25_forced_main_batches", ())),
                QCPO_B25_FORCED_MAIN_BATCHES,
            ),
            "registered_event_offset": (
                int(getattr(
                    args,
                    "qcpo_b25_event_schedule_offset_batches",
                    -1,
                )),
                QCPO_B25_EVENT_SCHEDULE_OFFSET_BATCHES,
            ),
        }

        # 汇总所有漂移一次性报错，避免修复一个字段后才暴露下一个错误。
        mismatches = [
            f"{name}={actual!r} (required {expected!r})"
            for name, (actual, expected) in required.items()
            if actual != expected
        ]
        if mismatches:
            raise ValueError(
                "QRefAA B25-only-B identity mismatch: "
                + ", ".join(mismatches)
            )

    def _validate_frozen_configuration(self, args) -> None:
        """用临时 B12 投影执行父类完整科学配置校验，并保证异常时恢复。

        此方法只替换父 validator 明确硬锁的身份与几何标量。真实 vector env 已按 B25
        创建，且这里不会重建环境、模型、optimizer、RNG 或 W&B 对象。学习率、clip、
        GAE、Ki、deque、网络和所有公式仍由未修改的父 validator 逐项检查。
        """

        # 参数对象使用浅复制即可：这里只覆写标量/字符串，不修改其中任何可变容器。
        historical_args = copy.copy(args)
        historical_args.training_budget_semantics = _PARENT_B12_BUDGET_SEMANTICS
        historical_args.qcpo_author_implementation_identity = (
            QCPO_AUTHOR_IMPLEMENTATION_ID
        )
        historical_args.qcpo_author_layout = QCPO_AUTHOR_LAYOUT
        historical_args.qcpo_ref_formulation = QCPO_AUTHOR_IMPLEMENTATION_ID
        historical_args.num_envs = 12
        historical_args.new_T = 100
        historical_args.new_B = 120

        # 父 validator 从 self 读取以下字段；保存真实 B25 值后才可做临时投影。
        projected = {
            "implementation_identity": QCPO_AUTHOR_IMPLEMENTATION_ID,
            "layout_identity": QCPO_AUTHOR_LAYOUT,
            "training_budget_semantics": _PARENT_B12_BUDGET_SEMANTICS,
            "num_envs": 12,
            "new_T": 100,
            "new_B": 120,
        }
        saved = {name: getattr(self, name) for name in projected}

        # finally 是 checkpoint 身份安全边界：即使父校验失败，也不得遗留伪 B12 runtime。
        try:
            for name, value in projected.items():
                setattr(self, name, value)
            QCPORefAuthorAlignedGPU._validate_frozen_configuration(
                self, historical_args
            )
        finally:
            for name, value in saved.items():
                setattr(self, name, value)

    def _checkpoint_identity(self) -> Dict[str, object]:
        """在父身份上追加 B25 几何、预算与次数证据。"""

        identity = super()._checkpoint_identity()
        identity.update({
            # 算法名与实现名使 B25 无法被 B12/B32 runner 接收。
            "algorithm": QCPO_REF_AUTHOR_B25_ALGO,
            "qcpo_b25_implementation_identity": QCPO_B25_IMPLEMENTATION_ID,
            "qcpo_b25_factor_tuple": canonical_qcpo_b25_factor_tuple(),
            "qcpo_b25_layout": QCPO_B25_LAYOUT,

            # 几何字段重复签名，防止只校验外层 algorithm 的弱 resume 路径。
            "qcpo_b25_raw_layout": "T1000_B25",
            "qcpo_b25_actor_layout": "T100_B250",
            "qcpo_b25_transitions_per_rollout": QCPO_B25_TRANSITIONS,
            "qcpo_b25_optimizer_epochs": QCPO_B25_OPTIMIZER_EPOCHS,
            "qcpo_b25_minibatches": QCPO_B25_MINIBATCHES,

            # 预算字段同时区分 main、actual、unique optimizer exposure 与 step 次数。
            "qcpo_b25_first_batch": (
                "dual_and_rms_only_included_in_main_budget_v1"
            ),
            "qcpo_b25_rms_only_batches": QCPO_B25_RMS_ONLY_BATCHES,
            "qcpo_b25_main_batches": QCPO_B25_MAIN_BATCHES,
            "qcpo_b25_optimized_main_batches": QCPO_B25_OPTIMIZED_MAIN_BATCHES,
            "qcpo_b25_main_env_steps": QCPO_B25_MAIN_ENV_STEPS,
            "qcpo_b25_actual_env_steps": QCPO_B25_ACTUAL_ENV_STEPS,
            "qcpo_b25_optimizer_transition_count": (
                QCPO_B25_OPTIMIZER_TRANSITION_COUNT
            ),
            "qcpo_b25_expected_optimizer_steps": (
                QCPO_B25_EXPECTED_OPTIMIZER_STEPS
            ),
            "qcpo_b25_expected_dual_updates": QCPO_B25_EXPECTED_DUAL_UPDATES,
            "qcpo_b25_forced_main_batches": QCPO_B25_FORCED_MAIN_BATCHES,
            "qcpo_b25_event_schedule_offset_batches": (
                QCPO_B25_EVENT_SCHEDULE_OFFSET_BATCHES
            ),
            "formal_endpoint_primary": QCPO_B25_FORMAL_ENDPOINT_PRIMARY,
            "wandb_project": QCPO_B25_WANDB_PROJECT,
            "formal_execution_profile": QCPO_B25_FORMAL_EXECUTION_PROFILE,
            "dynamics_profile": QCPO_B25_DYNAMICS_PROFILE,
            "train_vec_backend": QCPO_B25_TRAIN_VEC_BACKEND,
            "validation_vec_backend": QCPO_B25_VALIDATION_VEC_BACKEND,
            "validation_num_envs": QCPO_B25_VALIDATION_NUM_ENVS,
            "eval_vec_backend": QCPO_B25_EVAL_VEC_BACKEND,
            "eval_num_envs": QCPO_B25_EVAL_NUM_ENVS,
            "train_simulator_identity": QCPO_B25_TRAIN_SIMULATOR_IDENTITY,
            "train_mujoco_version": QCPO_B25_TRAIN_MUJOCO_VERSION,
            "train_mujoco_warp_version": QCPO_B25_TRAIN_MUJOCO_WARP_VERSION,
            "train_warp_version": QCPO_B25_TRAIN_WARP_VERSION,
            "train_warp_determinism": QCPO_B25_TRAIN_WARP_DETERMINISM,
            "train_warp_deterministic_max_records": (
                QCPO_B25_TRAIN_WARP_DETERMINISTIC_MAX_RECORDS),
            "train_mujoco_gl": QCPO_B25_TRAIN_MUJOCO_GL,
            "canonical_eval_stack": QCPO_B25_CANONICAL_EVAL_STACK,
            "formal_evaluation_protocol": QCPO_B25_FORMAL_EVALUATION_PROTOCOL,
        })
        return identity

    def get_training_summary(self) -> Dict[str, object]:
        """发布 B25 固定协议与真实运行计数，供结果清单独立审计。"""

        summary = super().get_training_summary()
        summary.update({
            # 公共身份与 factor tuple 可直接进入论文实验清单。
            "algorithm_identity": QCPO_B25_IMPLEMENTATION_ID,
            "registered_algorithm": QCPO_REF_AUTHOR_B25_ALGO,
            "b25_factor_tuple": canonical_qcpo_b25_factor_tuple(),

            # 采样/优化几何明确区分 25 条物理轨迹和 250 个 recurrent chunks。
            "raw_rollout_layout": "T1000_B25",
            "actor_sequence_layout": "T100_B250",
            "transitions_per_rollout": QCPO_B25_TRANSITIONS,
            "optimizer_epochs": QCPO_B25_OPTIMIZER_EPOCHS,
            "minibatches_per_epoch": QCPO_B25_MINIBATCHES,

            # 首批不隐藏为 initialization；main 与 actual 因而都严格终止于 10M。
            "first_batch_protocol": (
                "dual_and_rms_only_included_in_main_budget_v1"
            ),
            "rms_only_batches": QCPO_B25_RMS_ONLY_BATCHES,
            "main_rollout_batches": QCPO_B25_MAIN_BATCHES,
            "optimized_main_batches": QCPO_B25_OPTIMIZED_MAIN_BATCHES,
            "b25_main_env_steps": QCPO_B25_MAIN_ENV_STEPS,
            "b25_actual_env_steps": QCPO_B25_ACTUAL_ENV_STEPS,
            "optimizer_transition_count": QCPO_B25_OPTIMIZER_TRANSITION_COUNT,
            "expected_optimizer_steps": QCPO_B25_EXPECTED_OPTIMIZER_STEPS,
            "expected_dual_updates": QCPO_B25_EXPECTED_DUAL_UPDATES,
            "forced_main_batches": QCPO_B25_FORCED_MAIN_BATCHES,
            "event_schedule_offset_batches": (
                QCPO_B25_EVENT_SCHEDULE_OFFSET_BATCHES
            ),
            "formal_endpoint_primary": QCPO_B25_FORMAL_ENDPOINT_PRIMARY,
            "wandb_project": QCPO_B25_WANDB_PROJECT,
            "formal_execution_profile": QCPO_B25_FORMAL_EXECUTION_PROFILE,
            "dynamics_profile": QCPO_B25_DYNAMICS_PROFILE,
            "train_vec_backend": QCPO_B25_TRAIN_VEC_BACKEND,
            "validation_vec_backend": QCPO_B25_VALIDATION_VEC_BACKEND,
            "validation_num_envs": QCPO_B25_VALIDATION_NUM_ENVS,
            "eval_vec_backend": QCPO_B25_EVAL_VEC_BACKEND,
            "eval_num_envs": QCPO_B25_EVAL_NUM_ENVS,
            "train_simulator_identity": QCPO_B25_TRAIN_SIMULATOR_IDENTITY,
            "train_mujoco_version": QCPO_B25_TRAIN_MUJOCO_VERSION,
            "train_mujoco_warp_version": QCPO_B25_TRAIN_MUJOCO_WARP_VERSION,
            "train_warp_version": QCPO_B25_TRAIN_WARP_VERSION,
            "train_warp_determinism": QCPO_B25_TRAIN_WARP_DETERMINISM,
            "train_warp_deterministic_max_records": (
                QCPO_B25_TRAIN_WARP_DETERMINISTIC_MAX_RECORDS),
            "train_mujoco_gl": QCPO_B25_TRAIN_MUJOCO_GL,
            "canonical_eval_stack": QCPO_B25_CANONICAL_EVAL_STACK,
            "formal_evaluation_protocol": QCPO_B25_FORMAL_EVALUATION_PROTOCOL,
        })
        return summary


# 明确公开接口，便于 agents/__init__.py 与 run_experiment.py 后续单点接线。
__all__ = [
    "QCPORefAuthorB25GPU",
    "QCPO_B25_ACTOR_BATCH",
    "QCPO_B25_ACTOR_TIME",
    "QCPO_B25_ACTUAL_ENV_STEPS",
    "QCPO_B25_CANONICAL_EVAL_STACK",
    "QCPO_B25_DYNAMICS_PROFILE",
    "QCPO_B25_EVENT_SCHEDULE_OFFSET_BATCHES",
    "QCPO_B25_EVAL_NUM_ENVS",
    "QCPO_B25_EVAL_VEC_BACKEND",
    "QCPO_B25_EXPECTED_DUAL_UPDATES",
    "QCPO_B25_EXPECTED_OPTIMIZER_STEPS",
    "QCPO_B25_FORMAL_ENDPOINT_PRIMARY",
    "QCPO_B25_FORMAL_EVALUATION_PROTOCOL",
    "QCPO_B25_FORMAL_EXECUTION_PROFILE",
    "QCPO_B25_FORCED_MAIN_BATCHES",
    "QCPO_B25_IMPLEMENTATION_ID",
    "QCPO_B25_LAYOUT",
    "QCPO_B25_MAIN_BATCHES",
    "QCPO_B25_MAIN_ENV_STEPS",
    "QCPO_B25_MINIBATCHES",
    "QCPO_B25_NOMINAL_ENV_STEPS",
    "QCPO_B25_OPTIMIZED_MAIN_BATCHES",
    "QCPO_B25_OPTIMIZER_EPOCHS",
    "QCPO_B25_OPTIMIZER_TRANSITION_COUNT",
    "QCPO_B25_RAW_BATCH",
    "QCPO_B25_RAW_TIME",
    "QCPO_B25_RMS_ONLY_BATCHES",
    "QCPO_B25_TRAINING_BUDGET_SEMANTICS",
    "QCPO_B25_TRAIN_MUJOCO_VERSION",
    "QCPO_B25_TRAIN_MUJOCO_GL",
    "QCPO_B25_TRAIN_MUJOCO_WARP_VERSION",
    "QCPO_B25_TRAIN_SIMULATOR_IDENTITY",
    "QCPO_B25_TRAIN_VEC_BACKEND",
    "QCPO_B25_TRAIN_WARP_DETERMINISM",
    "QCPO_B25_TRAIN_WARP_DETERMINISTIC_MAX_RECORDS",
    "QCPO_B25_TRAIN_WARP_VERSION",
    "QCPO_B25_TRANSITIONS",
    "QCPO_B25_VALIDATION_NUM_ENVS",
    "QCPO_B25_VALIDATION_VEC_BACKEND",
    "QCPO_B25_WANDB_PROJECT",
    "QCPO_REF_AUTHOR_B25_ALGO",
    "apply_qcpo_ref_author_b25_configuration",
    "canonical_qcpo_b25_factor_tuple",
    "qcpo_b25_factor_tuple",
]
