# -*- coding: utf-8 -*-
"""DQC C1/no-outer 的正式 B25、物理 10M 交互适配器。

这个模块只注册一个新的、与历史 checkpoint 不兼容的实验身份。算法主体仍然是
``DQCACC1NoRewardOuterGPU``：独立 recurrent actor、标量 GAE value critic、
action-conditioned quantile critic、C1 split clipping，以及 reward/risk 均无额外
outer discount。这里仅冻结正式实验所需的采样与优化协议：

* 每个 rollout 收集 ``T=1000, B=25``，即 25,000 个环境 transition；
* actor 将 rollout 双射重排为 ``[T=100, B=250]`` 连续 recurrent chunks；
  每个 epoch 独立打乱 250 个 chunk，并以 25 个 1000-transition minibatch 做 SGD；
* 两个 critic 保留历史 nominal minibatch=1024。父类使用 ``tensor_split``，因此
  25,000 行被均匀分成 25 份，实际每份恰好 1,000 行；
* main 阶段 value/quantile 固定 E8；actor 至多 E8，PPO ratio clip 保留为 0.1，
  每个完整 actor epoch 后若全 rollout behavior KL 达到 0.01，则保留该轮并提前停止；
* 前 20 个 rollout 只训练 value/quantile critic，各自 E4；actor、dual、RMS、
  W&B 与 main clock 均不更新；
* 物理预算严格为 10M：20 个 warmup + 380 个 main，一共 400*25*1000。
* 训练与 CRN555 validation 固定 MJWarp 3.12 + implicit-fast，CRN777 论文终测
  固定为孤立 MuJoCo-3.12 CPU/MP + implicit-fast；两阶段执行身份都进入 checkpoint。

注意：父类的历史 finite-GAE 身份会强制检查 warmup=10 与 target_kl=0.01。
构造时因此先用一个私有历史 namespace 完成全部 seeded 参数分配，再安装本模块的
allocation-free 运行时字段。这一做法保持历史初始化顺序，同时阻止旧 checkpoint
被静默解释成新的正式实验。
"""

from __future__ import annotations

import copy                                  # 复制私有构造 namespace，绝不改写调用者对象
import json                                  # 生成稳定、无空白的 checkpoint 身份字符串
import math                                  # 按父类 ceil(N/M) 规则推导 minibatch 数量
from numbers import Integral                 # 识别 Python/NumPy 整数而显式排除 bool clock
from typing import Dict, Mapping, Tuple      # 标注 rollout/身份映射与有序 factor tuple

import torch                                 # recurrent rechunk、独立 shuffle RNG 与 checkpoint

from .dqc_ac_c1_no_reward_outer import (
    C1_NO_REWARD_OUTER_BASE_ARM,
    DQCACC1NoRewardOuterGPU,
    apply_c1_no_reward_outer_configuration,
    c1_no_reward_outer_factor_tuple,
)
from .dqc_ac_independent_factorial import (
    INDEPENDENT_FACTORIAL_TRAINING_BUDGET,
)


# =============================================================================
# 正式算法身份
# =============================================================================

FORMAL_B25_ALGO = "DQCAC_FORMAL_B25"
# implementation ID 同时进入 args、agent state 与 checkpoint；任何协议变化都必须换 ID。
FORMAL_B25_IMPLEMENTATION_ID = (
    "dqc_independent_c1_no_outer_formal_b25_raw_t1000_actor_t100_"
    "chunk_shuffle_minibatch1024_bptt100_targetkl0p01_epochstop_"
    "physical10m_mjwarp312_implicitfast_train_cpu312_implicitfast_eval_v5"
)
FORMAL_B25_WANDB_PROJECT = "Gremlin_full_gpu"
FORMAL_B25_EXECUTION_PROFILE = (
    "gremlin_mjwarp312_implicitfast_train_cpu312_final_eval_v2"
)
FORMAL_B25_TRAIN_VEC_BACKEND = "mjwarp"       # 10M 正式训练后端。
FORMAL_B25_VALIDATION_VEC_BACKEND = "mjwarp"  # 训练期 CRN555 validation 后端。
FORMAL_B25_VALIDATION_NUM_ENVS = 25            # validation 仍以 25 轨迹并行。
FORMAL_B25_EVAL_VEC_BACKEND = "mp"            # 孤立 CPU 3.12 implicit-fast canonical 终测。
FORMAL_B25_DYNAMICS_PROFILE = "point_velocity_implicitfast_v1"
FORMAL_B25_TRAIN_SIMULATOR_IDENTITY = "mjwarp-implicitfast-v1"
FORMAL_B25_TRAIN_MUJOCO_VERSION = "3.12.0"
FORMAL_B25_TRAIN_MUJOCO_WARP_VERSION = "3.12.0"
FORMAL_B25_TRAIN_WARP_VERSION = "1.16.0"
# Warp 1.16 在当前配置下不承诺 deterministic replay；max_records=0 明确记录关闭状态。
FORMAL_B25_TRAIN_WARP_DETERMINISM = "NOT_GUARANTEED"
FORMAL_B25_TRAIN_WARP_DETERMINISTIC_MAX_RECORDS = 0
FORMAL_B25_TRAIN_MUJOCO_GL = "disable"
FORMAL_B25_CANONICAL_EVAL_STACK = (
    "zprl_mujoco_3.12.0_cpu_mp_implicitfast_v1"
)
FORMAL_B25_EVALUATION_PROTOCOL = (
    "gremlin_mjwarp312_implicitfast_validation_B25_crn555_"
    "cpu312_implicitfast_final_B25_crn777_v2"
)
FORMAL_B25_ACTION_MODE = "explicit_box_clip"  # DQC critic 使用实际 Box-clipped action。

# MJWarp v1 将每个 world 的随机过程压缩为两个 int32 clock。这个
# 字段集是 checkpoint schema，不接受缺失或额外字段；未来扩展必须提升 version。
_FORMAL_B25_MJWARP_RNG_FIELDS = frozenset({
    "version", "backend", "action_mode", "dynamics_profile", "seed", "slot",
    "reset_count", "goal_count",
})


# =============================================================================
# 采样几何与严格物理预算
# =============================================================================

FORMAL_B25_HORIZON = 1000                 # 每条 Gremlin 轨迹的固定截断长度 T
FORMAL_B25_NUM_ENVS = 25                  # 每个 rollout 并行收集的完整轨迹数 B
FORMAL_B25_TRANSITIONS = (                # 每个 rollout 的物理环境交互数 N=B*T
    FORMAL_B25_HORIZON * FORMAL_B25_NUM_ENVS
)

FORMAL_B25_WARMUP_BATCHES = 20            # critic-only 初始化 rollout 数
FORMAL_B25_MAIN_BATCHES = 380             # actor/critic/dual 正式更新 rollout 数
FORMAL_B25_TOTAL_ROLLOUT_BATCHES = (       # 所有物理 rollout 都计入 sample budget
    FORMAL_B25_WARMUP_BATCHES + FORMAL_B25_MAIN_BATCHES
)

FORMAL_B25_INITIALIZATION_ENV_STEPS = (    # 20*25,000 = 500,000
    FORMAL_B25_WARMUP_BATCHES * FORMAL_B25_TRANSITIONS
)
FORMAL_B25_MAIN_ENV_STEPS = (              # 380*25,000 = 9,500,000
    FORMAL_B25_MAIN_BATCHES * FORMAL_B25_TRANSITIONS
)
FORMAL_B25_ACTUAL_ENV_STEPS = (            # 物理总量严格等于 10,000,000
    FORMAL_B25_TOTAL_ROLLOUT_BATCHES * FORMAL_B25_TRANSITIONS
)

# validation/checkpoint 调度使用 main-batch clock；190/380 是 main 的中点与终点。
FORMAL_B25_FORCED_MAIN_BATCHES = (190, 380)
FORMAL_B25_EVENT_SCHEDULE_OFFSET_BATCHES = 0


# =============================================================================
# 优化预算：每个 rollout 内部允许算法保留自己的 optimizer cadence
# =============================================================================

FORMAL_B25_CRITIC_MINIBATCH_SIZE = 1024   # 用户可见、checkpoint-signed 的 nominal 上限
FORMAL_B25_CRITIC_MINIBATCHES = int(math.ceil(
    FORMAL_B25_TRANSITIONS / float(FORMAL_B25_CRITIC_MINIBATCH_SIZE)
))
FORMAL_B25_ACTOR_CHUNK_LENGTH = 100       # chunk 内保留时间顺序，BPTT 长度为 100
FORMAL_B25_ACTOR_CHUNKS = (               # 25 条 T1000 轨迹切成 250 个连续片段
    FORMAL_B25_TRANSITIONS // FORMAL_B25_ACTOR_CHUNK_LENGTH
)
FORMAL_B25_ACTOR_MINIBATCH_SIZE = 1024    # 与 critic 相同的用户可见 nominal 上限
FORMAL_B25_ACTOR_MINIBATCHES = int(math.ceil(
    FORMAL_B25_TRANSITIONS / float(FORMAL_B25_ACTOR_MINIBATCH_SIZE)
))
FORMAL_B25_ACTOR_CHUNKS_PER_MINIBATCH = (
    FORMAL_B25_ACTOR_CHUNKS // FORMAL_B25_ACTOR_MINIBATCHES
)
FORMAL_B25_EFFECTIVE_ACTOR_MINIBATCH_SIZE = (
    FORMAL_B25_ACTOR_CHUNK_LENGTH * FORMAL_B25_ACTOR_CHUNKS_PER_MINIBATCH
)
# 单独的 CPU Generator 避免 actor shuffle 改变后续 CUDA action noise 序列。
FORMAL_B25_ACTOR_SHUFFLE_SEED_OFFSET = 25_100_003
FORMAL_B25_ACTOR_SHUFFLE_PROTOCOL = (
    "cpu_generator_randperm_250_chunks_no_replacement_each_epoch_v1"
)

# 父类调用 torch.tensor_split(randperm, ceil(N/M))，而不是固定 1024 后留 remainder。
# 25,000/25 可整除，所以每个实际 minibatch 都恰好包含 1,000 个 transition。
FORMAL_B25_EFFECTIVE_CRITIC_MINIBATCH_SIZE = (
    FORMAL_B25_TRANSITIONS // FORMAL_B25_CRITIC_MINIBATCHES
)
FORMAL_B25_MAIN_EPOCHS = 8                # actor/value/quantile 正式阶段均为 E8
FORMAL_B25_WARMUP_CRITIC_EPOCHS = 4       # 两个 critic 的初始化阶段均为 E4
FORMAL_B25_ACTOR_TARGET_KL = 0.01         # 每个完整 actor epoch 后按全 rollout KL 早停
FORMAL_B25_ACTOR_EPOCH_PROTOCOL = (
    "shuffle_chunks_minibatch25_up_to_E8_whole_behavior_kl0p01_"
    "early_stop_v1"
)

# actor 每个已执行 epoch 做 25 个 Adam step；两个 critic 固定 E8，dual 按 rollout 更新。
FORMAL_B25_EXPECTED_ACTOR_OPTIMIZER_STEPS = (
    FORMAL_B25_MAIN_BATCHES * FORMAL_B25_MAIN_EPOCHS * FORMAL_B25_ACTOR_MINIBATCHES
)
FORMAL_B25_MINIMUM_ACTOR_OPTIMIZER_STEPS = (
    FORMAL_B25_MAIN_BATCHES * FORMAL_B25_ACTOR_MINIBATCHES
)
FORMAL_B25_EXPECTED_CRITIC_MAIN_STEPS = (
    FORMAL_B25_MAIN_BATCHES
    * FORMAL_B25_MAIN_EPOCHS
    * FORMAL_B25_CRITIC_MINIBATCHES
)
FORMAL_B25_EXPECTED_CRITIC_WARMUP_STEPS = (
    FORMAL_B25_WARMUP_BATCHES
    * FORMAL_B25_WARMUP_CRITIC_EPOCHS
    * FORMAL_B25_CRITIC_MINIBATCHES
)
FORMAL_B25_EXPECTED_CRITIC_TOTAL_STEPS = (
    FORMAL_B25_EXPECTED_CRITIC_MAIN_STEPS
    + FORMAL_B25_EXPECTED_CRITIC_WARMUP_STEPS
)

# 这些 import-time 检查不是测试替代品，而是防止未来改常量时产生自相矛盾的身份。
if FORMAL_B25_ACTUAL_ENV_STEPS != 10_000_000:
    raise RuntimeError("formal B25 physical interaction budget must equal exactly 10M")
if FORMAL_B25_EFFECTIVE_CRITIC_MINIBATCH_SIZE != 1000:
    raise RuntimeError("formal B25 tensor_split minibatches must each contain 1000 rows")
if FORMAL_B25_HORIZON % FORMAL_B25_ACTOR_CHUNK_LENGTH != 0:
    raise RuntimeError("formal B25 actor chunk length must divide the rollout horizon")
if FORMAL_B25_ACTOR_CHUNKS % FORMAL_B25_ACTOR_MINIBATCHES != 0:
    raise RuntimeError("formal B25 actor chunks must divide evenly across minibatches")
if FORMAL_B25_EFFECTIVE_ACTOR_MINIBATCH_SIZE != 1000:
    raise RuntimeError("formal B25 actor minibatches must each contain 1000 rows")
if (FORMAL_B25_ACTOR_CHUNKS_PER_MINIBATCH * FORMAL_B25_ACTOR_MINIBATCHES
        != FORMAL_B25_ACTOR_CHUNKS):
    raise RuntimeError("formal B25 actor epoch must cover every chunk exactly once")


def formal_b25_factor_tuple() -> Tuple[Tuple[str, object], ...]:
    """返回完整 C1/no-outer 语义及所有 B25 正式协议字段。

    父 factor tuple 中 ``warmup`` 与 ``actor_reuse`` 已经存在，必须原位替换；
    若在尾部再次追加同名 key，序列化身份会同时宣称两个冲突协议。
    """

    inherited_factors = []
    for name, value in c1_no_reward_outer_factor_tuple():
        # warmup 仍是 fitted-hard Bellman generation，只把 rollout 数从 10 改为 20。
        if name == "warmup":
            value = (
                "20_batches_500_trajectories_minibatch1024_effective1000_"
                "value_and_quantile_E4_hard_target_no_actor_no_dual_no_rms_"
                "no_wandb_physical_budget_v1"
            )
        # minibatch loss 只按本批有效 transition 归约，不能沿用整批固定 25,000。
        elif name == "actor_reduction":
            value = "fixed_effective_recurrent_minibatch_1000_v1"
        # ratio clipping 和 E8 不变；每个 epoch 固定遍历 25 个 shuffled minibatch。
        elif name == "actor_reuse":
            value = (
                "ppo_clip_up_to_E8_raw_T1000_B25_actor_T100_B250_chunk_shuffle_"
                "minibatch25_effective1000_ratio0p1_whole_kl0p01_epoch_stop_v1"
            )
        inherited_factors.append((name, value))

    # 以下字段完整描述“采多少、怎样分、更新多少、预算如何计数”。
    return tuple(inherited_factors) + (
        ("raw_rollout_layout", "T1000_B25_v1"),
        ("actor_sequence_layout", "T100_B250_contiguous_chunks_v1"),
        ("actor_bptt_length", FORMAL_B25_ACTOR_CHUNK_LENGTH),
        ("actor_chunk_length", FORMAL_B25_ACTOR_CHUNK_LENGTH),
        ("actor_chunks_per_rollout", FORMAL_B25_ACTOR_CHUNKS),
        ("actor_minibatch_size_nominal", FORMAL_B25_ACTOR_MINIBATCH_SIZE),
        ("actor_minibatches_per_epoch", FORMAL_B25_ACTOR_MINIBATCHES),
        ("actor_chunks_per_minibatch",
         FORMAL_B25_ACTOR_CHUNKS_PER_MINIBATCH),
        ("actor_minibatch_size_effective",
         FORMAL_B25_EFFECTIVE_ACTOR_MINIBATCH_SIZE),
        ("actor_shuffle_protocol", FORMAL_B25_ACTOR_SHUFFLE_PROTOCOL),
        ("actor_shuffle_seed_offset", FORMAL_B25_ACTOR_SHUFFLE_SEED_OFFSET),
        ("critic_minibatch_size_nominal", FORMAL_B25_CRITIC_MINIBATCH_SIZE),
        ("critic_minibatches_per_epoch", FORMAL_B25_CRITIC_MINIBATCHES),
        ("critic_minibatch_size_effective",
         FORMAL_B25_EFFECTIVE_CRITIC_MINIBATCH_SIZE),
        ("main_optimizer_epochs", FORMAL_B25_MAIN_EPOCHS),
        ("warmup_critic_epochs", FORMAL_B25_WARMUP_CRITIC_EPOCHS),
        ("actor_target_kl", FORMAL_B25_ACTOR_TARGET_KL),
        ("actor_epoch_protocol", FORMAL_B25_ACTOR_EPOCH_PROTOCOL),
        ("actor_kl_logging_protocol",
         "fixed_epoch_1_to_8_carry_forward_with_applied_mask_v1"),
        ("actor_epoch_evidence",
         "checkpointed_formal_b25_actor_epochs_applied_total_v1"),
        ("minimum_actor_optimizer_steps",
         FORMAL_B25_MINIMUM_ACTOR_OPTIMIZER_STEPS),
        ("maximum_actor_optimizer_steps",
         FORMAL_B25_EXPECTED_ACTOR_OPTIMIZER_STEPS),
        ("expected_critic_main_steps_per_critic",
         FORMAL_B25_EXPECTED_CRITIC_MAIN_STEPS),
        ("expected_critic_warmup_steps_per_critic",
         FORMAL_B25_EXPECTED_CRITIC_WARMUP_STEPS),
        ("expected_critic_total_steps_per_critic",
         FORMAL_B25_EXPECTED_CRITIC_TOTAL_STEPS),
        ("warmup_rollout_batches", FORMAL_B25_WARMUP_BATCHES),
        ("main_rollout_batches", FORMAL_B25_MAIN_BATCHES),
        ("actual_rollout_batches", FORMAL_B25_TOTAL_ROLLOUT_BATCHES),
        ("initialization_env_steps", FORMAL_B25_INITIALIZATION_ENV_STEPS),
        ("main_env_steps", FORMAL_B25_MAIN_ENV_STEPS),
        ("actual_env_steps", FORMAL_B25_ACTUAL_ENV_STEPS),
        ("budget_axis", "physical_actual_env_steps_v1"),
        ("event_schedule_offset_batches",
         FORMAL_B25_EVENT_SCHEDULE_OFFSET_BATCHES),
        ("formal_endpoint_primary", True),
        ("formal_execution_profile", FORMAL_B25_EXECUTION_PROFILE),
        ("dynamics_profile", FORMAL_B25_DYNAMICS_PROFILE),
        ("train_vec_backend", FORMAL_B25_TRAIN_VEC_BACKEND),
        ("validation_vec_backend", FORMAL_B25_VALIDATION_VEC_BACKEND),
        ("validation_num_envs", FORMAL_B25_VALIDATION_NUM_ENVS),
        ("eval_vec_backend", FORMAL_B25_EVAL_VEC_BACKEND),
        ("eval_num_envs", FORMAL_B25_NUM_ENVS),
        ("train_simulator_identity", FORMAL_B25_TRAIN_SIMULATOR_IDENTITY),
        ("train_mujoco_version", FORMAL_B25_TRAIN_MUJOCO_VERSION),
        ("train_mujoco_warp_version", FORMAL_B25_TRAIN_MUJOCO_WARP_VERSION),
        ("train_warp_version", FORMAL_B25_TRAIN_WARP_VERSION),
        ("train_warp_determinism", FORMAL_B25_TRAIN_WARP_DETERMINISM),
        ("train_warp_deterministic_max_records",
         FORMAL_B25_TRAIN_WARP_DETERMINISTIC_MAX_RECORDS),
        ("train_mujoco_gl", FORMAL_B25_TRAIN_MUJOCO_GL),
        ("canonical_eval_stack", FORMAL_B25_CANONICAL_EVAL_STACK),
        ("formal_evaluation_protocol", FORMAL_B25_EVALUATION_PROTOCOL),
        ("dual_batch_trajectories", FORMAL_B25_NUM_ENVS),
    )


def canonical_formal_b25_factor_tuple() -> str:
    """把正式 factor tuple 序列化为跨 locale 稳定的 checkpoint 字符串。"""

    return json.dumps(
        formal_b25_factor_tuple(),
        separators=(",", ":"),
        ensure_ascii=True,
    )


def apply_dqc_formal_b25_configuration(args):
    """在普通 DQC namespace 上安装唯一的正式 B25 科学配置。

    调用方可以从 ``base_args('DQCACBeta', ...)`` 开始。本函数先完整重建历史
    C1/no-outer，再覆盖且仅覆盖本实验注册的采样、warmup、KL 与物理预算字段。
    """

    # 先恢复历史高表现 C1/no-outer 的网络、loss、target、LR 与 optimizer 语义。
    apply_c1_no_reward_outer_configuration(args)

    # -------------------- 原始 B25/T1000 采样；actor 另建 T100 连续 chunk view --------------------
    args.algo_name = FORMAL_B25_ALGO
    args.horizon = FORMAL_B25_HORIZON
    args.num_envs = FORMAL_B25_NUM_ENVS
    args.critic_minibatch_size = FORMAL_B25_CRITIC_MINIBATCH_SIZE

    # 10M 训练与训练期 validation 都固定 MJWarp/B25/implicit-fast。论文终测由
    # 独立 MuJoCo-3.12 CPU/MP implicit-fast evaluator 执行；profile 不能依赖默认值。
    args.formal_execution_profile = FORMAL_B25_EXECUTION_PROFILE
    args.dynamics_profile = FORMAL_B25_DYNAMICS_PROFILE
    args.vec_backend = FORMAL_B25_TRAIN_VEC_BACKEND
    args.train_vec_backend = FORMAL_B25_TRAIN_VEC_BACKEND
    args.validation_vec_backend = FORMAL_B25_VALIDATION_VEC_BACKEND
    args.validation_num_envs = FORMAL_B25_VALIDATION_NUM_ENVS
    args.eval_vec_backend = FORMAL_B25_EVAL_VEC_BACKEND
    args.eval_num_envs = FORMAL_B25_NUM_ENVS
    args.train_simulator_identity = FORMAL_B25_TRAIN_SIMULATOR_IDENTITY
    args.train_mujoco_version = FORMAL_B25_TRAIN_MUJOCO_VERSION
    args.train_mujoco_warp_version = FORMAL_B25_TRAIN_MUJOCO_WARP_VERSION
    args.train_warp_version = FORMAL_B25_TRAIN_WARP_VERSION
    args.train_warp_determinism = FORMAL_B25_TRAIN_WARP_DETERMINISM
    args.train_warp_deterministic_max_records = (
        FORMAL_B25_TRAIN_WARP_DETERMINISTIC_MAX_RECORDS
    )
    args.train_mujoco_gl = FORMAL_B25_TRAIN_MUJOCO_GL
    args.canonical_eval_stack = FORMAL_B25_CANONICAL_EVAL_STACK
    args.formal_endpoint_primary = True
    args.formal_evaluation_protocol = FORMAL_B25_EVALUATION_PROTOCOL

    # 两个 critic 固定 E8；actor 最多 E8，并仅在完整 epoch 边界检查 whole-rollout KL。
    args.critic_epochs = FORMAL_B25_MAIN_EPOCHS
    args.dqc_value_epochs = FORMAL_B25_MAIN_EPOCHS
    args.actor_epochs = FORMAL_B25_MAIN_EPOCHS
    args.dqc_actor_reuse_mode = "ppo_clip"
    args.dqc_ratio_clip = 0.1
    args.dqc_target_kl = FORMAL_B25_ACTOR_TARGET_KL

    # warmup 使用历史 fitted-hard/E4，只增加到 20 个 B25 rollout。
    args.warmup_iters = FORMAL_B25_WARMUP_BATCHES
    args.critic_warmup_mode = "fitted_hard"
    args.warmup_critic_epochs = FORMAL_B25_WARMUP_CRITIC_EPOCHS

    # main 只跑 380 批；加上 warmup 后物理交互正好 10M，而不是 10M+warmup。
    args.num_iterations = FORMAL_B25_MAIN_BATCHES
    args.training_budget_semantics = INDEPENDENT_FACTORIAL_TRAINING_BUDGET
    args.forced_event_batches = list(FORMAL_B25_FORCED_MAIN_BATCHES)
    args.event_schedule_offset_batches = (
        FORMAL_B25_EVENT_SCHEDULE_OFFSET_BATCHES
    )

    # 正式 Gremlin reward-outage 工作点与历史对照保持完全一致。
    args.failure_alpha = 0.2
    args.q_alpha = args.failure_alpha
    args.return_threshold = 1.85

    # -------------------- 冗余、checkpoint-signed 的公共身份 --------------------
    factor_tuple = canonical_formal_b25_factor_tuple()
    # 父类 checkpoint validator 会读取这两组历史命名字段；全部替换成新身份，
    # 可确保历史 C1 或 B32 checkpoint 在任何 tensor mutation 前失败。
    args.dqc_independent_factorial_implementation_identity = (
        FORMAL_B25_IMPLEMENTATION_ID
    )
    args.dqc_independent_factorial_factor_tuple = factor_tuple
    args.dqc_independent_factorial_reward_outer_discount = "none"
    args.dqc_c1_no_reward_outer_implementation_identity = (
        FORMAL_B25_IMPLEMENTATION_ID
    )
    args.dqc_c1_no_reward_outer_factor_tuple = factor_tuple

    # ``dqc_formal_b25_*`` 是 runner 后续加入 checkpoint signature 时的稳定接口。
    args.dqc_formal_b25_implementation_identity = FORMAL_B25_IMPLEMENTATION_ID
    args.dqc_formal_b25_factor_tuple = factor_tuple
    args.dqc_formal_b25_dynamics_profile = FORMAL_B25_DYNAMICS_PROFILE
    args.dqc_formal_b25_raw_layout = "T1000_B25"
    args.dqc_formal_b25_actor_layout = "T100_B250_contiguous_chunks"
    args.dqc_formal_b25_actor_bptt_length = FORMAL_B25_ACTOR_CHUNK_LENGTH
    args.dqc_formal_b25_actor_chunk_length = FORMAL_B25_ACTOR_CHUNK_LENGTH
    args.dqc_formal_b25_actor_chunks = FORMAL_B25_ACTOR_CHUNKS
    args.dqc_formal_b25_actor_minibatch_size = (
        FORMAL_B25_ACTOR_MINIBATCH_SIZE
    )
    args.dqc_formal_b25_actor_minibatches_per_epoch = (
        FORMAL_B25_ACTOR_MINIBATCHES
    )
    args.dqc_formal_b25_actor_chunks_per_minibatch = (
        FORMAL_B25_ACTOR_CHUNKS_PER_MINIBATCH
    )
    args.dqc_formal_b25_effective_actor_minibatch_size = (
        FORMAL_B25_EFFECTIVE_ACTOR_MINIBATCH_SIZE
    )
    args.dqc_formal_b25_actor_shuffle_protocol = (
        FORMAL_B25_ACTOR_SHUFFLE_PROTOCOL
    )
    args.dqc_formal_b25_actor_shuffle_seed = (
        FORMAL_B25_ACTOR_SHUFFLE_SEED_OFFSET + int(args.seed)
    )
    args.dqc_formal_b25_critic_minibatch_size = (
        FORMAL_B25_CRITIC_MINIBATCH_SIZE
    )
    args.dqc_formal_b25_critic_minibatches_per_epoch = (
        FORMAL_B25_CRITIC_MINIBATCHES
    )
    args.dqc_formal_b25_effective_critic_minibatch_size = (
        FORMAL_B25_EFFECTIVE_CRITIC_MINIBATCH_SIZE
    )
    args.dqc_formal_b25_main_epochs = FORMAL_B25_MAIN_EPOCHS
    args.dqc_formal_b25_warmup_critic_epochs = (
        FORMAL_B25_WARMUP_CRITIC_EPOCHS
    )
    args.dqc_formal_b25_actor_target_kl = FORMAL_B25_ACTOR_TARGET_KL
    args.dqc_formal_b25_actor_epoch_protocol = FORMAL_B25_ACTOR_EPOCH_PROTOCOL
    args.dqc_formal_b25_minimum_actor_optimizer_steps = (
        FORMAL_B25_MINIMUM_ACTOR_OPTIMIZER_STEPS
    )
    args.dqc_formal_b25_expected_actor_optimizer_steps = (
        FORMAL_B25_EXPECTED_ACTOR_OPTIMIZER_STEPS
    )
    args.dqc_formal_b25_maximum_actor_optimizer_steps = (
        FORMAL_B25_EXPECTED_ACTOR_OPTIMIZER_STEPS)
    args.dqc_formal_b25_expected_critic_main_steps = (
        FORMAL_B25_EXPECTED_CRITIC_MAIN_STEPS
    )
    args.dqc_formal_b25_expected_critic_warmup_steps = (
        FORMAL_B25_EXPECTED_CRITIC_WARMUP_STEPS
    )
    args.dqc_formal_b25_expected_critic_total_steps = (
        FORMAL_B25_EXPECTED_CRITIC_TOTAL_STEPS
    )
    args.dqc_formal_b25_warmup_batches = FORMAL_B25_WARMUP_BATCHES
    args.dqc_formal_b25_main_batches = FORMAL_B25_MAIN_BATCHES
    args.dqc_formal_b25_actual_rollout_batches = (
        FORMAL_B25_TOTAL_ROLLOUT_BATCHES
    )
    args.dqc_formal_b25_initialization_env_steps = (
        FORMAL_B25_INITIALIZATION_ENV_STEPS
    )
    args.dqc_formal_b25_main_env_steps = FORMAL_B25_MAIN_ENV_STEPS
    args.dqc_formal_b25_actual_env_steps = FORMAL_B25_ACTUAL_ENV_STEPS
    args.dqc_formal_b25_budget_axis = "physical_actual_env_steps_v1"
    args.dqc_formal_b25_forced_main_batches = list(
        FORMAL_B25_FORCED_MAIN_BATCHES
    )
    args.dqc_formal_b25_event_schedule_offset_batches = (
        FORMAL_B25_EVENT_SCHEDULE_OFFSET_BATCHES
    )

    # 默认用 physical clock 画曲线。正式 launcher 仍可覆盖 W&B 名称/ID/路径等操作字段。
    args.defer_wandb_init = True
    args.wandb_project = FORMAL_B25_WANDB_PROJECT
    args.wandb_step_metric = "progress/actual_env_steps"
    args.wandb_id = ""
    args.wandb_resume = "never"
    args.wandb_group = "formal-b25-gremlin-target-kl-0p01-10seed-v4"
    args.wandb_name = f"dqcac-formal-b25-gremlin-seed{int(args.seed)}"
    args.wandb_lineage = args.wandb_name
    args.experiment_variant = args.wandb_name

    # 去除旧正式匹配标签，避免 W&B UI 把 raw-T1000 新实验误归入 B32/T100。
    tags = [
        tag.strip()
        for tag in str(getattr(args, "wandb_tags", "") or "").split(",")
        if tag.strip() and tag.strip() not in {
            "raw-T1000-B32", "actor-T100-B320", "full-minibatch-32000",
            "actor-full-E8-no-kl-stop", "fresh-uninterrupted-10m",
            "single-factor-ablation", "actor-full-T1000",
            "actor-minibatch-E8-no-kl-stop",
        }
    ]
    for tag in (
            "formal-b25-10seed", "dqcac-c1-no-outer", "raw-T1000-B25",
            "actor-T100-B250", "actor-chunk-shuffle",
            "actor-minibatch-1024-effective-1000",
            "critic-minibatch-1024-effective-1000", "physical-10m",
            "actor-target-kl-0p01-epoch-stop", "implicitfast-v1"):
        if tag not in tags:
            tags.append(tag)
    args.wandb_tags = ",".join(tags)
    return args


class DQCACFormalB25GPU(DQCACC1NoRewardOuterGPU):
    """历史 C1/no-outer DQC 加正式 B25/physical-10M 运行协议。"""

    def __init__(self, args, env):
        """先校验公开身份，再按历史 RNG 顺序分配模型并安装新运行时字段。"""

        # 所有字段在调用父构造器前检查；非法任务不会创建环境 worker、模型、Adam 或 W&B。
        required = {
            "algorithm": (
                str(getattr(args, "algo_name", "")), FORMAL_B25_ALGO),
            "implementation": (
                getattr(args, "dqc_formal_b25_implementation_identity", None),
                FORMAL_B25_IMPLEMENTATION_ID),
            "factor_tuple": (
                getattr(args, "dqc_formal_b25_factor_tuple", None),
                canonical_formal_b25_factor_tuple()),
            "independent_implementation": (
                getattr(
                    args,
                    "dqc_independent_factorial_implementation_identity",
                    None),
                FORMAL_B25_IMPLEMENTATION_ID),
            "independent_factor_tuple": (
                getattr(args, "dqc_independent_factorial_factor_tuple", None),
                canonical_formal_b25_factor_tuple()),
            "c1_implementation": (
                getattr(
                    args,
                    "dqc_c1_no_reward_outer_implementation_identity",
                    None),
                FORMAL_B25_IMPLEMENTATION_ID),
            "c1_factor_tuple": (
                getattr(args, "dqc_c1_no_reward_outer_factor_tuple", None),
                canonical_formal_b25_factor_tuple()),
            "horizon": (
                int(getattr(args, "horizon", -1)), FORMAL_B25_HORIZON),
            "num_envs": (
                int(getattr(args, "num_envs", -1)), FORMAL_B25_NUM_ENVS),
            "critic_minibatch": (
                int(getattr(args, "critic_minibatch_size", -1)),
                FORMAL_B25_CRITIC_MINIBATCH_SIZE),
            "critic_epochs": (
                int(getattr(args, "critic_epochs", -1)), FORMAL_B25_MAIN_EPOCHS),
            "value_epochs": (
                int(getattr(args, "dqc_value_epochs", -1)),
                FORMAL_B25_MAIN_EPOCHS),
            "actor_epochs": (
                int(getattr(args, "actor_epochs", -1)), FORMAL_B25_MAIN_EPOCHS),
            "actor_reuse": (
                getattr(args, "dqc_actor_reuse_mode", None), "ppo_clip"),
            "ratio_clip": (
                float(getattr(args, "dqc_ratio_clip", float("nan"))), 0.1),
            "actor_target_kl": (
                float(getattr(args, "dqc_target_kl", float("nan"))),
                FORMAL_B25_ACTOR_TARGET_KL),
            "warmup_batches": (
                int(getattr(args, "warmup_iters", -1)),
                FORMAL_B25_WARMUP_BATCHES),
            "warmup_mode": (
                getattr(args, "critic_warmup_mode", None), "fitted_hard"),
            "warmup_epochs": (
                int(getattr(args, "warmup_critic_epochs", -1)),
                FORMAL_B25_WARMUP_CRITIC_EPOCHS),
            "main_batches": (
                int(getattr(args, "num_iterations", -1)),
                FORMAL_B25_MAIN_BATCHES),
            "budget_semantics": (
                getattr(args, "training_budget_semantics", None),
                INDEPENDENT_FACTORIAL_TRAINING_BUDGET),
            "failure_alpha": (
                float(getattr(args, "failure_alpha", float("nan"))), 0.2),
            "return_threshold": (
                float(getattr(args, "return_threshold", float("nan"))), 1.85),
            "reward_outer": (
                getattr(args, "dqc_reward_outer_discount", None), "none"),
            "forced_batches": (
                tuple(int(value) for value in getattr(
                    args, "forced_event_batches", ())),
                FORMAL_B25_FORCED_MAIN_BATCHES),
            "event_offset": (
                int(getattr(args, "event_schedule_offset_batches", -1)),
                FORMAL_B25_EVENT_SCHEDULE_OFFSET_BATCHES),
            "formal_endpoint_primary": (
                bool(getattr(args, "formal_endpoint_primary", False)), True),
            "formal_execution_profile": (
                str(getattr(args, "formal_execution_profile", "")),
                FORMAL_B25_EXECUTION_PROFILE),
            # args 与实际 reference env 必须在任何模型/optimizer 分配前都声明同一 MDP。
            "dynamics_profile": (
                str(getattr(args, "dynamics_profile", "")),
                FORMAL_B25_DYNAMICS_PROFILE),
            "environment_dynamics_profile": (
                str(getattr(env, "dynamics_profile", "")),
                FORMAL_B25_DYNAMICS_PROFILE),
            "registered_dynamics_profile": (
                str(getattr(args, "dqc_formal_b25_dynamics_profile", "")),
                FORMAL_B25_DYNAMICS_PROFILE),
            "vec_backend": (
                str(getattr(args, "vec_backend", "")),
                FORMAL_B25_TRAIN_VEC_BACKEND),
            "train_vec_backend": (
                str(getattr(args, "train_vec_backend", "")),
                FORMAL_B25_TRAIN_VEC_BACKEND),
            # 分别要求 MJWarp 后再做等值检查，阻止元数据与真实 backend 分叉。
            "train_backend_matches_vec_backend": (
                str(getattr(args, "train_vec_backend", "")),
                str(getattr(args, "vec_backend", ""))),
            "validation_vec_backend": (
                str(getattr(args, "validation_vec_backend", "")),
                FORMAL_B25_VALIDATION_VEC_BACKEND),
            "validation_num_envs": (
                int(getattr(args, "validation_num_envs", -1)),
                FORMAL_B25_VALIDATION_NUM_ENVS),
            "eval_vec_backend": (
                str(getattr(args, "eval_vec_backend", "")),
                FORMAL_B25_EVAL_VEC_BACKEND),
            "eval_num_envs": (
                int(getattr(args, "eval_num_envs", -1)),
                FORMAL_B25_NUM_ENVS),
            "train_simulator_identity": (
                str(getattr(args, "train_simulator_identity", "")),
                FORMAL_B25_TRAIN_SIMULATOR_IDENTITY),
            "train_mujoco_version": (
                str(getattr(args, "train_mujoco_version", "")),
                FORMAL_B25_TRAIN_MUJOCO_VERSION),
            "train_mujoco_warp_version": (
                str(getattr(args, "train_mujoco_warp_version", "")),
                FORMAL_B25_TRAIN_MUJOCO_WARP_VERSION),
            "train_warp_version": (
                str(getattr(args, "train_warp_version", "")),
                FORMAL_B25_TRAIN_WARP_VERSION),
            "train_warp_determinism": (
                str(getattr(args, "train_warp_determinism", "")),
                FORMAL_B25_TRAIN_WARP_DETERMINISM),
            "train_warp_deterministic_max_records": (
                int(getattr(
                    args, "train_warp_deterministic_max_records", -1)),
                FORMAL_B25_TRAIN_WARP_DETERMINISTIC_MAX_RECORDS),
            "train_mujoco_gl": (
                str(getattr(args, "train_mujoco_gl", "")),
                FORMAL_B25_TRAIN_MUJOCO_GL),
            "canonical_eval_stack": (
                str(getattr(args, "canonical_eval_stack", "")),
                FORMAL_B25_CANONICAL_EVAL_STACK),
            "action_execution_mode": (
                str(getattr(args, "action_execution_mode", "")),
                FORMAL_B25_ACTION_MODE),
            "formal_evaluation_protocol": (
                getattr(args, "formal_evaluation_protocol", None),
                FORMAL_B25_EVALUATION_PROTOCOL),
            "registered_actor_layout": (
                getattr(args, "dqc_formal_b25_actor_layout", None),
                "T100_B250_contiguous_chunks"),
            "registered_actor_bptt": (
                int(getattr(args, "dqc_formal_b25_actor_bptt_length", -1)),
                FORMAL_B25_ACTOR_CHUNK_LENGTH),
            "registered_actor_chunks": (
                int(getattr(args, "dqc_formal_b25_actor_chunks", -1)),
                FORMAL_B25_ACTOR_CHUNKS),
            "registered_actor_minibatch": (
                int(getattr(
                    args, "dqc_formal_b25_actor_minibatch_size", -1)),
                FORMAL_B25_ACTOR_MINIBATCH_SIZE),
            "registered_actor_minibatches": (
                int(getattr(
                    args, "dqc_formal_b25_actor_minibatches_per_epoch", -1)),
                FORMAL_B25_ACTOR_MINIBATCHES),
            "registered_actor_chunks_per_minibatch": (
                int(getattr(
                    args, "dqc_formal_b25_actor_chunks_per_minibatch", -1)),
                FORMAL_B25_ACTOR_CHUNKS_PER_MINIBATCH),
            "registered_effective_actor_minibatch": (
                int(getattr(
                    args, "dqc_formal_b25_effective_actor_minibatch_size", -1)),
                FORMAL_B25_EFFECTIVE_ACTOR_MINIBATCH_SIZE),
            "registered_actor_shuffle": (
                getattr(args, "dqc_formal_b25_actor_shuffle_protocol", None),
                FORMAL_B25_ACTOR_SHUFFLE_PROTOCOL),
            "registered_actor_shuffle_seed": (
                int(getattr(args, "dqc_formal_b25_actor_shuffle_seed", -1)),
                FORMAL_B25_ACTOR_SHUFFLE_SEED_OFFSET + int(args.seed)),
            "registered_actor_epoch_protocol": (
                getattr(args, "dqc_formal_b25_actor_epoch_protocol", None),
                FORMAL_B25_ACTOR_EPOCH_PROTOCOL),
            "registered_minimum_actor_optimizer_steps": (
                int(getattr(
                    args,
                    "dqc_formal_b25_minimum_actor_optimizer_steps",
                    -1)),
                FORMAL_B25_MINIMUM_ACTOR_OPTIMIZER_STEPS),
            "registered_expected_actor_optimizer_steps": (
                int(getattr(
                    args,
                    "dqc_formal_b25_expected_actor_optimizer_steps",
                    -1)),
                FORMAL_B25_EXPECTED_ACTOR_OPTIMIZER_STEPS),
            "registered_maximum_actor_optimizer_steps": (
                int(getattr(
                    args,
                    "dqc_formal_b25_maximum_actor_optimizer_steps",
                    -1)),
                FORMAL_B25_EXPECTED_ACTOR_OPTIMIZER_STEPS),
            "registered_effective_minibatch": (
                int(getattr(
                    args,
                    "dqc_formal_b25_effective_critic_minibatch_size",
                    -1)),
                FORMAL_B25_EFFECTIVE_CRITIC_MINIBATCH_SIZE),
            "registered_actual_steps": (
                int(getattr(args, "dqc_formal_b25_actual_env_steps", -1)),
                FORMAL_B25_ACTUAL_ENV_STEPS),
            "registered_budget_axis": (
                getattr(args, "dqc_formal_b25_budget_axis", None),
                "physical_actual_env_steps_v1"),
        }
        mismatches = [
            f"{name}={actual!r} (required {expected!r})"
            for name, (actual, expected) in required.items()
            if actual != expected
        ]
        if mismatches:
            raise ValueError(
                "DQC formal B25 identity mismatch: " + ", ".join(mismatches))

        # -------------------- 仅用于 seeded allocation 的历史私有 namespace --------------------
        # 历史父类必须看到 main=400、warmup=10、target_kl=.01 才会通过其完整身份检查。
        construction_args = copy.copy(args)
        apply_c1_no_reward_outer_configuration(construction_args)
        # 保留新的公共算法名，使 VecAgentBase/W&B/console 不会显示成历史 seed0 ablation。
        construction_args.algo_name = FORMAL_B25_ALGO
        construction_args.horizon = FORMAL_B25_HORIZON
        construction_args.num_envs = FORMAL_B25_NUM_ENVS
        construction_args.dynamics_profile = FORMAL_B25_DYNAMICS_PROFILE
        super().__init__(construction_args, env)

        # -------------------- 正式运行时协议与私有 shuffle RNG --------------------
        # 循环字段不创建参数；CPU Generator 有独立 state，不消耗全局 Torch/CUDA RNG。
        self.num_iterations = FORMAL_B25_MAIN_BATCHES
        self.warmup_iters = FORMAL_B25_WARMUP_BATCHES
        self.critic_minibatch_size = FORMAL_B25_CRITIC_MINIBATCH_SIZE
        self.critic_epochs = FORMAL_B25_MAIN_EPOCHS
        self.dqc_value_epochs = FORMAL_B25_MAIN_EPOCHS
        self.actor_epochs = FORMAL_B25_MAIN_EPOCHS
        self.dqc_target_kl = FORMAL_B25_ACTOR_TARGET_KL
        self.dynamics_profile = FORMAL_B25_DYNAMICS_PROFILE
        self.event_schedule_offset_batches = (
            FORMAL_B25_EVENT_SCHEDULE_OFFSET_BATCHES
        )

        self.formal_b25_actor_layout = "T100_B250_contiguous_chunks"
        self.formal_b25_actor_shuffle_seed = int(
            args.dqc_formal_b25_actor_shuffle_seed
        )
        self.formal_b25_actor_shuffle_generator = torch.Generator(device="cpu")
        self.formal_b25_actor_shuffle_generator.manual_seed(
            self.formal_b25_actor_shuffle_seed
        )
        self.formal_b25_actor_epochs_applied_total = 0
        # Runtime actor geometry lives on the instance. The historical M25
        # identity receives exactly its original constants; incompatible
        # adapters may replace these fields before the first optimizer step.
        self.formal_b25_actor_minibatches_per_epoch = (
            FORMAL_B25_ACTOR_MINIBATCHES)
        self.formal_b25_actor_chunk_counts_per_minibatch = tuple(
            [FORMAL_B25_ACTOR_CHUNKS_PER_MINIBATCH]
            * FORMAL_B25_ACTOR_MINIBATCHES)
        self.formal_b25_actor_effective_minibatch_sizes = tuple(
            [FORMAL_B25_EFFECTIVE_ACTOR_MINIBATCH_SIZE]
            * FORMAL_B25_ACTOR_MINIBATCHES)
        self.formal_b25_factor_tuple = canonical_formal_b25_factor_tuple()
        self.formal_b25_expected_actor_optimizer_steps = (
            FORMAL_B25_EXPECTED_ACTOR_OPTIMIZER_STEPS
        )
        self.formal_b25_expected_critic_optimizer_steps = (
            FORMAL_B25_EXPECTED_CRITIC_TOTAL_STEPS
        )
        # 父类 checkpoint identity 从此读取新 factor，而不是历史 C1 字符串。
        self.independent_factorial_factor_tuple = self.formal_b25_factor_tuple

    def _transform_actor_tensor(self, value: torch.Tensor) -> torch.Tensor:
        """把 ``[1000,25,*]`` 双射为 250 个连续 ``[100,*]`` chunk。

        转换先把 environment 轴移到最前，再按原轨迹的连续内存顺序切成十段。
        因此输出第 ``10*b+k`` 列精确对应原 ``b`` 轨迹的
        ``[100*k:100*(k+1)]``，chunk 内绝不打乱时间顺序。
        """

        if not torch.is_tensor(value) or value.ndim < 2:
            raise TypeError("formal B25 actor transform requires a rank>=2 tensor")
        old_time, old_batch = int(value.shape[0]), int(value.shape[1])
        if (
                old_time == FORMAL_B25_ACTOR_CHUNK_LENGTH
                and old_batch == FORMAL_B25_ACTOR_CHUNKS):
            return value
        if old_time != FORMAL_B25_HORIZON or old_batch != FORMAL_B25_NUM_ENVS:
            raise ValueError(
                "formal B25 actor transform requires [1000,25,*], got "
                f"[{old_time},{old_batch},*]")

        # [T,B,*] -> [B,T,*] 后 reshape 只切连续时间，不混合不同原始 environment。
        trailing = tuple(value.shape[2:])
        return (
            value.transpose(0, 1)
            .reshape(
                FORMAL_B25_ACTOR_CHUNKS,
                FORMAL_B25_ACTOR_CHUNK_LENGTH,
                *trailing,
            )
            .transpose(0, 1)
            .contiguous()
        )

    def _formal_actor_training_view(
            self,
            rollout: Mapping[str, torch.Tensor],
            combined: torch.Tensor,
    ) -> Tuple[Dict[str, object], torch.Tensor, Dict[str, object]]:
        """一次性构造 actor chunk view，并冻结 C1 的全量归一化尺度。

        ``combined/mean_term/risk_term`` 等逐 decision 标签采用与 observation
        完全相同的双射。标量 ``combined_normalization_scale`` 保持由完整 25,000
        transition population 计算的值，不能在 minibatch 内重新估计。
        """

        required_fields = (
            "S", "A_raw", "valid", "logp", "mean_old", "log_std_old",
            "h0", "c0",
        )
        missing = [name for name in required_fields if name not in rollout]
        if missing:
            raise ValueError(
                "formal B25 actor rollout is missing fields: "
                + ", ".join(missing))

        original_terms = self._independent_factorial_actor_terms
        if not isinstance(original_terms, dict):
            raise RuntimeError(
                "formal B25 C1 minibatches require frozen actor terms from "
                "actor_advantages()")

        # 只转换 actor 更新真正读取的字段；critic/dual 继续持有原始 [1000,25] rollout。
        transformed = dict(rollout)
        for name in required_fields:
            transformed[name] = self._transform_actor_tensor(rollout[name])
        transformed_combined = self._transform_actor_tensor(combined)

        # h0/c0 是 collector 在动作前保存的 recurrent state。转换后的第零行
        # 正好是每个 100-step chunk 的边界状态，[250,H] -> LSTM 的 [1,250,H]。
        transformed["actor_initial_state"] = (
            transformed["h0"][0].unsqueeze(0).contiguous(),
            transformed["c0"][0].unsqueeze(0).contiguous(),
        )

        transformed_terms = dict(original_terms)
        for name, value in original_terms.items():
            # 只对 leading shape 恰为原始 [T,B] 的逐 decision tensor 做同一双射。
            # 全局 scalar/vector metadata 原样复用，避免 minibatch normalization 漂移。
            if (
                    torch.is_tensor(value)
                    and value.ndim >= 2
                    and tuple(value.shape[:2])
                    == (FORMAL_B25_HORIZON, FORMAL_B25_NUM_ENVS)):
                transformed_terms[name] = self._transform_actor_tensor(value)
        return transformed, transformed_combined, transformed_terms

    def _actor_minibatch_view(
            self,
            rollout: Mapping[str, object],
            combined: torch.Tensor,
            terms: Mapping[str, object],
            chunk_indices: torch.Tensor,
    ) -> Tuple[Dict[str, object], torch.Tensor, Dict[str, object]]:
        """用同一组 chunk ID 切出 rollout、combined 与全部 C1 split 标签。"""

        indices = torch.as_tensor(chunk_indices, dtype=torch.long)
        chunk_counts, _effective_sizes = self._formal_actor_minibatch_geometry()
        if indices.ndim != 1 or int(indices.numel()) not in set(chunk_counts):
            if chunk_counts == (FORMAL_B25_ACTOR_CHUNKS_PER_MINIBATCH,) * (
                    FORMAL_B25_ACTOR_MINIBATCHES):
                raise ValueError(
                    "formal B25 actor minibatch requires exactly ten chunk IDs")
            raise ValueError(
                "formal B25 actor minibatch chunk count is outside the "
                f"registered geometry {chunk_counts!r}")
        if int(torch.unique(indices.cpu()).numel()) != int(indices.numel()):
            raise ValueError("formal B25 actor minibatch contains duplicate chunks")

        minibatch_rollout = dict(rollout)
        for name, value in rollout.items():
            if (
                    torch.is_tensor(value)
                    and value.ndim >= 2
                    and tuple(value.shape[:2]) == (
                        FORMAL_B25_ACTOR_CHUNK_LENGTH,
                        FORMAL_B25_ACTOR_CHUNKS)):
                minibatch_rollout[name] = value.index_select(
                    1, indices.to(value.device))

        # actor_initial_state 的 batch 轴是 dim=1；它必须与 observation chunk 同序。
        initial_state = rollout.get("actor_initial_state")
        if not isinstance(initial_state, tuple) or len(initial_state) != 2:
            raise ValueError("formal B25 actor view lacks chunk boundary h0/c0")
        minibatch_rollout["actor_initial_state"] = tuple(
            state.index_select(1, indices.to(state.device))
            for state in initial_state
        )

        minibatch_combined = combined.index_select(
            1, indices.to(combined.device))
        minibatch_terms = dict(terms)
        for name, value in terms.items():
            if (
                    torch.is_tensor(value)
                    and value.ndim >= 2
                    and tuple(value.shape[:2]) == (
                        FORMAL_B25_ACTOR_CHUNK_LENGTH,
                        FORMAL_B25_ACTOR_CHUNKS)):
                minibatch_terms[name] = value.index_select(
                    1, indices.to(value.device))
        return minibatch_rollout, minibatch_combined, minibatch_terms

    def _formal_actor_minibatch_geometry(self):
        """Return and validate registered per-minibatch chunk and row counts.

        tensor_split may produce unequal pieces. For M12 over 250 chunks this
        is exactly ten 21-chunk pieces followed by two 20-chunk pieces. The
        complete tuple prevents a variable schedule being labelled as fixed.
        """

        minibatches = int(getattr(
            self, "formal_b25_actor_minibatches_per_epoch",
            FORMAL_B25_ACTOR_MINIBATCHES))
        chunk_counts = tuple(int(value) for value in getattr(
            self, "formal_b25_actor_chunk_counts_per_minibatch",
            tuple([FORMAL_B25_ACTOR_CHUNKS_PER_MINIBATCH] * minibatches)))
        effective_sizes = tuple(int(value) for value in getattr(
            self, "formal_b25_actor_effective_minibatch_sizes",
            tuple([FORMAL_B25_EFFECTIVE_ACTOR_MINIBATCH_SIZE] * minibatches)))
        required_sizes = tuple(
            count * FORMAL_B25_ACTOR_CHUNK_LENGTH for count in chunk_counts)
        if (
                minibatches <= 0
                or len(chunk_counts) != minibatches
                or len(effective_sizes) != minibatches
                or any(count <= 0 for count in chunk_counts)
                or sum(chunk_counts) != FORMAL_B25_ACTOR_CHUNKS
                or effective_sizes != required_sizes):
            raise RuntimeError(
                "formal B25 actor minibatch geometry is internally inconsistent")
        return chunk_counts, effective_sizes

    def _replay_actor_for_update(self, rollout):
        """从每个 chunk 保存的 behavior-policy 边界状态重放 T100 序列。"""

        initial_state = rollout.get("actor_initial_state")
        if not isinstance(initial_state, tuple) or len(initial_state) != 2:
            raise ValueError(
                "formal B25 actor replay requires (h0,c0) chunk initial state")
        return self.actor(rollout["S"], initial_state)

    def _actor_loss_denominator(self, rollout):
        """Use the registered tensor population as each minibatch denominator."""

        valid = rollout.get("valid")
        if not torch.is_tensor(valid):
            raise ValueError("formal B25 actor minibatch lacks valid mask")
        _chunk_counts, effective_sizes = self._formal_actor_minibatch_geometry()
        if int(valid.numel()) not in set(effective_sizes):
            if effective_sizes == (FORMAL_B25_EFFECTIVE_ACTOR_MINIBATCH_SIZE,) * (
                    FORMAL_B25_ACTOR_MINIBATCHES):
                raise RuntimeError(
                    "formal B25 actor minibatch must contain exactly 1000 rows")
            raise RuntimeError(
                "formal B25 actor minibatch row count is outside the "
                f"registered geometry {effective_sizes!r}")
        # Invalid rows are zero-masked in the surrogate.  The denominator is the
        # registered tensor population (2,100 or 2,000 for M12), not the number
        # of valid episode rows, so early endings cannot reweight minibatches.
        return float(valid.numel())

    def _whole_actor_behavior_kl(
            self, rollout, behavior_mean, behavior_log_std) -> float:
        """在完整 25,000-row chunk view 上测量一个 epoch 后的 policy KL。"""

        with torch.no_grad():
            current_mean, current_log_std, _next_state = (
                self._replay_actor_for_update(rollout)
            )
            policy_kl = self._masked_diagonal_gaussian_kl(
                behavior_mean, behavior_log_std,
                current_mean, current_log_std, rollout["valid"])
        if not bool(torch.isfinite(policy_kl)):
            raise FloatingPointError(
                "non-finite formal B25 whole-rollout actor KL")
        return float(policy_kl.item())

    def update_actor_epochs(self, rollout, combined):
        """执行至多 E8 × 25 个 shuffled recurrent minibatch PPO Adam step。

        每个 epoch 的 CPU ``randperm(250)`` 无放回覆盖所有 chunk。chunk 内保持
        100-step 时间顺序，且每个 minibatch 使用其十个边界 h0/c0；dual 与两个
        critic 的更新路径不经过这个 optimizer-only view。
        每个完整 epoch 后测量全 rollout behavior KL；命中轮保留，只跳过后续轮。
        第 E8 轮已无后续工作，因此即使命中阈值也不记为 early stop。
        """

        epochs = int(getattr(self, "actor_epochs", -1))
        if epochs != FORMAL_B25_MAIN_EPOCHS:
            raise RuntimeError("formal B25 actor must request exactly E8")
        if str(getattr(self, "dqc_actor_reuse_mode", "")) != "ppo_clip":
            raise RuntimeError("formal B25 actor minibatches require PPO clipping")
        target_kl = float(getattr(self, "dqc_target_kl", float("nan")))
        if not math.isfinite(target_kl) or target_kl < 0.0:
            raise RuntimeError("formal B25 target KL must be finite and non-negative")

        chunk_counts, effective_sizes = self._formal_actor_minibatch_geometry()
        minibatches_per_epoch = len(chunk_counts)

        full_rollout, full_combined, full_terms = (
            self._formal_actor_training_view(rollout, combined)
        )
        behavior_mean = full_rollout["mean_old"].detach().clone()
        behavior_log_std = full_rollout["log_std_old"].detach().clone()
        step_metrics = []
        epoch_behavior_kls = []
        epoch_coverages = []
        minibatch_valid_counts = []
        early_stopped = False
        early_stop_epoch = 0
        starting_applied_epochs = int(
            self.formal_b25_actor_epochs_applied_total)

        try:
            for epoch_index in range(epochs):
                # permutation 在 CPU 私有 generator 上生成，再一次性复制到 policy device。
                # 这既不消耗 action noise RNG，也让 checkpoint/resume 精确续接下一顺序。
                permutation_cpu = torch.randperm(
                    FORMAL_B25_ACTOR_CHUNKS,
                    generator=self.formal_b25_actor_shuffle_generator,
                    device="cpu",
                )
                permutation_hook = getattr(
                    self, "_formal_actor_permutation_hook", None)
                if callable(permutation_hook):
                    permutation_hook(epoch_index, permutation_cpu.clone())

                # tensor_split deterministically places remainder chunks in
                # the earliest pieces; verify its exact registered size vector.
                permutation = permutation_cpu.to(full_rollout["S"].device)
                minibatches = torch.tensor_split(
                    permutation, minibatches_per_epoch)
                observed_chunk_counts = tuple(
                    int(batch.numel()) for batch in minibatches)
                if observed_chunk_counts != chunk_counts:
                    raise RuntimeError("formal B25 actor minibatch count drifted")
                coverage = torch.cat([batch.cpu() for batch in minibatches])
                unique_coverage = int(torch.unique(coverage).numel())
                if (
                        int(coverage.numel()) != FORMAL_B25_ACTOR_CHUNKS
                        or unique_coverage != FORMAL_B25_ACTOR_CHUNKS):
                    raise RuntimeError(
                        "formal B25 actor epoch did not cover every chunk once")
                epoch_coverages.append(unique_coverage)

                for minibatch_index, chunk_indices in enumerate(minibatches):
                    minibatch_rollout, minibatch_combined, minibatch_terms = (
                        self._actor_minibatch_view(
                            full_rollout, full_combined, full_terms, chunk_indices)
                    )
                    # C1 split surrogate reads this side-channel; install the exactly aligned
                    # minibatch immediately before the parent likelihood/optimizer primitive.
                    self._independent_factorial_actor_terms = minibatch_terms
                    valid_count = int(minibatch_rollout["valid"].sum().item())
                    minibatch_valid_counts.append(valid_count)

                    minibatch_hook = getattr(
                        self, "_formal_actor_minibatch_hook", None)
                    if callable(minibatch_hook):
                        minibatch_hook(
                            epoch_index, minibatch_index, chunk_indices.detach().cpu(),
                            minibatch_rollout, minibatch_combined, minibatch_terms)

                    step_metrics.append(self.update_actor_ppo_epoch(
                        minibatch_rollout,
                        minibatch_combined,
                        minibatch_rollout["mean_old"],
                        minibatch_rollout["log_std_old"],
                    ))

                # 仅在 25 个 minibatch 全部提交后测量一次全 rollout KL。
                self._independent_factorial_actor_terms = full_terms
                epoch_kl = self._whole_actor_behavior_kl(
                    full_rollout, behavior_mean, behavior_log_std)
                epoch_behavior_kls.append(epoch_kl)
                self.formal_b25_actor_epochs_applied_total += 1

                # 触发 epoch 保留；E8 已无待跳过 epoch，所以命中阈值只作诊断。
                completed_epoch = epoch_index + 1
                if (
                        completed_epoch < epochs
                        and target_kl > 0.0
                        and epoch_kl >= target_kl):
                    early_stopped = True
                    early_stop_epoch = completed_epoch
                    self.actor_kl_early_stop_count = int(getattr(
                        self, "actor_kl_early_stop_count", 0)) + 1
                    break
        finally:
            # 不让 25,000-row actor labels 跨 rollout 存活，避免下一次采集期间占用显存。
            self._independent_factorial_actor_terms = None

        applied_epochs = len(epoch_behavior_kls)
        expected_steps = applied_epochs * minibatches_per_epoch
        if len(step_metrics) != expected_steps:
            raise RuntimeError("formal B25 actor optimizer-step count drifted")
        if self.formal_b25_actor_epochs_applied_total != (
                starting_applied_epochs + applied_epochs):
            raise RuntimeError("formal B25 actor applied-epoch counter drifted")
        last = dict(step_metrics[-1])

        # 兼容历史 key，同时把 minibatch 与 whole-epoch 聚合边界明确写入诊断。
        losses = [item["actor/loss"] for item in step_metrics]
        grad_norms = [
            item["diagnostic/dqc/actor_grad_norm_pre_clip"]
            for item in step_metrics
        ]
        step_kls = [
            item["diagnostic/dqc/policy_kl_after_step"]
            for item in step_metrics
        ]
        last.update({
            "diagnostic/dqc/actor_epochs_requested": float(epochs),
            "diagnostic/dqc/actor_epochs_applied": float(applied_epochs),
            "diagnostic/dqc/actor_minibatches_per_epoch": float(
                minibatches_per_epoch),
            "diagnostic/dqc/actor_minibatches_applied": float(expected_steps),
            "diagnostic/dqc/actor_chunks_per_minibatch_min": float(
                min(chunk_counts)),
            "diagnostic/dqc/actor_chunks_per_minibatch_max": float(
                max(chunk_counts)),
            "diagnostic/dqc/actor_minibatch_size_effective_min": float(
                min(effective_sizes)),
            "diagnostic/dqc/actor_minibatch_size_effective_max": float(
                max(effective_sizes)),
            "diagnostic/dqc/actor_minibatch_valid_rows_min": float(
                min(minibatch_valid_counts)),
            "diagnostic/dqc/actor_minibatch_valid_rows_max": float(
                max(minibatch_valid_counts)),
            "diagnostic/dqc/actor_epoch_chunk_coverage_min": float(
                min(epoch_coverages)),
            "diagnostic/dqc/actor_epoch_chunk_coverage_max": float(
                max(epoch_coverages)),
            "diagnostic/dqc/actor_loss_epoch_mean": float(
                sum(losses) / len(losses)),
            "diagnostic/dqc/actor_grad_norm_epoch_mean": float(
                sum(grad_norms) / len(grad_norms)),
            "diagnostic/dqc/policy_kl_step_mean": float(
                sum(step_kls) / len(step_kls)),
            "diagnostic/dqc/policy_kl_step_max": float(max(step_kls)),
            "diagnostic/dqc/policy_kl_behavior_after_epoch": float(
                epoch_behavior_kls[-1]),
            "diagnostic/dqc/policy_kl_behavior_epoch_mean": float(
                sum(epoch_behavior_kls) / len(epoch_behavior_kls)),
            "diagnostic/dqc/policy_kl_behavior_epoch_max": float(
                max(epoch_behavior_kls)),
            "diagnostic/dqc/policy_kl_after_epochs": float(
                epoch_behavior_kls[-1]),
            "diagnostic/dqc/actor_kl_early_stop": float(early_stopped),
            "diagnostic/dqc/actor_kl_early_stop_count": float(
                getattr(self, "actor_kl_early_stop_count", 0)),
            "diagnostic/dqc/actor_kl_early_stop_epoch": float(early_stop_epoch),
            "diagnostic/dqc/actor_kl_early_stop_enabled": float(
                target_kl > 0.0),
            "diagnostic/dqc/actor_kl_target": float(target_kl),
            "diagnostic/dqc/actor_epochs_applied_total": float(
                self.formal_b25_actor_epochs_applied_total),
        })

        # Retain singular metrics only for genuinely uniform schedules. M12
        # therefore never claims one fixed chunk or transition count.
        if len(set(chunk_counts)) == 1:
            last["diagnostic/dqc/actor_chunks_per_minibatch"] = float(
                chunk_counts[0])
            last["diagnostic/dqc/actor_minibatch_size_effective"] = float(
                effective_sizes[0])

        # 固定八槽便于跨 seed 对齐；未执行槽携带最后一个真实 KL，并以 mask=0 区分。
        last_epoch_kl = float(epoch_behavior_kls[-1])
        carried_epoch_kls = list(epoch_behavior_kls)
        carried_epoch_kls.extend(
            [last_epoch_kl] * (epochs - applied_epochs))
        for epoch_number, epoch_kl in enumerate(carried_epoch_kls, start=1):
            applied = float(epoch_number <= applied_epochs)
            metric_name = f"policy_kl_behavior_epoch_{epoch_number}"
            mask_name = f"{metric_name}_applied"
            last[metric_name] = float(epoch_kl)
            last[f"diagnostic/dqc/{metric_name}"] = float(epoch_kl)
            last[mask_name] = applied
            last[f"diagnostic/dqc/{mask_name}"] = applied
        return last

    def _validate_vec_env_rng_state(self, states) -> None:
        """在任何 live restore 前完整验证 B25 MJWarp world clocks。

        父类的 CPU/MP envelope 保留 NumPy/Task RNG 对象；本正式身份只允许
        MJWarp v1 的 ``(dynamics_profile, seed, slot, reset_count, goal_count)``。
        严格键集检查阻止 MP/Euler state、另一 seed 或不同 slot 的 clock 被静默解释。
        """

        if not isinstance(states, (list, tuple)) or len(states) != self.num_envs:
            raise ValueError(
                "DQC formal B25 MJWarp vec_env_rng_state must contain exactly "
                f"{self.num_envs} slots")

        expected_seed = int(self.seed)
        max_counter = (1 << 31) - 1              # backend 使用 wp.int32 存储两个 clock。
        for slot_index, slot_state in enumerate(states):
            if not isinstance(slot_state, dict):
                raise ValueError(
                    "DQC formal B25 MJWarp vec_env_rng_state slot "
                    f"{slot_index} is not a mapping")
            if set(slot_state) != _FORMAL_B25_MJWARP_RNG_FIELDS:
                missing = sorted(_FORMAL_B25_MJWARP_RNG_FIELDS - set(slot_state))
                extra = sorted(set(slot_state) - _FORMAL_B25_MJWARP_RNG_FIELDS)
                raise ValueError(
                    "DQC formal B25 MJWarp vec_env_rng_state slot "
                    f"{slot_index} schema mismatch: missing={missing}, extra={extra}")
            version = slot_state["version"]
            if isinstance(version, bool) or not isinstance(version, Integral) or int(version) != 1:
                raise ValueError(
                    "DQC formal B25 MJWarp vec_env_rng_state slot "
                    f"{slot_index} is not a version-1 mapping")
            if slot_state["backend"] != FORMAL_B25_TRAIN_VEC_BACKEND:
                raise ValueError(
                    "DQC formal B25 MJWarp vec_env_rng_state slot "
                    f"{slot_index} backend mismatch")
            if slot_state["action_mode"] != FORMAL_B25_ACTION_MODE:
                raise ValueError(
                    "DQC formal B25 MJWarp vec_env_rng_state slot "
                    f"{slot_index} action mode mismatch")
            if slot_state["dynamics_profile"] != FORMAL_B25_DYNAMICS_PROFILE:
                raise ValueError(
                    "DQC formal B25 MJWarp vec_env_rng_state slot "
                    f"{slot_index} dynamics profile mismatch")

            # ``bool`` 是 ``int`` 子类，必须显式排除，否则 True 会被当成 seed/slot=1。
            for field, expected in (("seed", expected_seed), ("slot", slot_index)):
                value = slot_state[field]
                if (
                        isinstance(value, bool)
                        or not isinstance(value, Integral)
                        or int(value) != expected):
                    raise ValueError(
                        "DQC formal B25 MJWarp vec_env_rng_state slot "
                        f"{slot_index} {field} mismatch")
            for field in ("reset_count", "goal_count"):
                value = slot_state[field]
                if (
                        isinstance(value, bool)
                        or not isinstance(value, Integral)
                        or not 0 <= int(value) <= max_counter):
                    raise ValueError(
                        "DQC formal B25 MJWarp vec_env_rng_state slot "
                        f"{slot_index} {field} is not a non-negative int32 counter")

    def _checkpoint_identity(self) -> Dict[str, object]:
        """返回所有不可被 transfer/resume 重新解释的正式 B25 字段。"""

        identity = super()._checkpoint_identity()
        factor_tuple = canonical_formal_b25_factor_tuple()
        identity.update({
            # 替换两层历史身份，保证历史 C1/B32 checkpoint 在 tensor load 前失败。
            "dqc_independent_factorial_implementation_identity": (
                FORMAL_B25_IMPLEMENTATION_ID),
            "dqc_independent_factorial_factor_tuple": factor_tuple,
            "dqc_independent_factorial_reward_outer_discount": "none",
            "dqc_c1_no_reward_outer_implementation_identity": (
                FORMAL_B25_IMPLEMENTATION_ID),
            "dqc_c1_no_reward_outer_factor_tuple": factor_tuple,
            "dqc_c1_no_reward_outer_base_arm": C1_NO_REWARD_OUTER_BASE_ARM,
            # 新模块专属身份与优化/预算冗余校验。
            "dqc_formal_b25_algorithm": FORMAL_B25_ALGO,
            "dqc_formal_b25_implementation_identity": (
                FORMAL_B25_IMPLEMENTATION_ID),
            "dqc_formal_b25_factor_tuple": factor_tuple,
            "dqc_formal_b25_raw_layout": "T1000_B25",
            "dqc_formal_b25_actor_layout": "T100_B250_contiguous_chunks",
            "dqc_formal_b25_actor_bptt_length": (
                FORMAL_B25_ACTOR_CHUNK_LENGTH),
            "dqc_formal_b25_actor_chunk_length": (
                FORMAL_B25_ACTOR_CHUNK_LENGTH),
            "dqc_formal_b25_actor_chunks": FORMAL_B25_ACTOR_CHUNKS,
            "dqc_formal_b25_actor_minibatch_size": (
                FORMAL_B25_ACTOR_MINIBATCH_SIZE),
            "dqc_formal_b25_actor_minibatches_per_epoch": (
                FORMAL_B25_ACTOR_MINIBATCHES),
            "dqc_formal_b25_actor_chunks_per_minibatch": (
                FORMAL_B25_ACTOR_CHUNKS_PER_MINIBATCH),
            "dqc_formal_b25_effective_actor_minibatch_size": (
                FORMAL_B25_EFFECTIVE_ACTOR_MINIBATCH_SIZE),
            "dqc_formal_b25_actor_shuffle_protocol": (
                FORMAL_B25_ACTOR_SHUFFLE_PROTOCOL),
            "dqc_formal_b25_actor_shuffle_seed": int(
                self.formal_b25_actor_shuffle_seed),
            "critic_minibatch_size": FORMAL_B25_CRITIC_MINIBATCH_SIZE,
            "dqc_formal_b25_critic_minibatches_per_epoch": (
                FORMAL_B25_CRITIC_MINIBATCHES),
            "dqc_formal_b25_effective_critic_minibatch_size": (
                FORMAL_B25_EFFECTIVE_CRITIC_MINIBATCH_SIZE),
            "critic_epochs": FORMAL_B25_MAIN_EPOCHS,
            "dqc_value_epochs": FORMAL_B25_MAIN_EPOCHS,
            "actor_epochs": FORMAL_B25_MAIN_EPOCHS,
            "warmup_iters": FORMAL_B25_WARMUP_BATCHES,
            "warmup_critic_epochs": FORMAL_B25_WARMUP_CRITIC_EPOCHS,
            "dqc_target_kl": FORMAL_B25_ACTOR_TARGET_KL,
            "dqc_formal_b25_actor_epoch_protocol": (
                FORMAL_B25_ACTOR_EPOCH_PROTOCOL),
            "dqc_formal_b25_minimum_actor_optimizer_steps": (
                FORMAL_B25_MINIMUM_ACTOR_OPTIMIZER_STEPS),
            "dqc_formal_b25_expected_actor_optimizer_steps": (
                FORMAL_B25_EXPECTED_ACTOR_OPTIMIZER_STEPS),
            "dqc_formal_b25_maximum_actor_optimizer_steps": (
                FORMAL_B25_EXPECTED_ACTOR_OPTIMIZER_STEPS),
            "dqc_formal_b25_expected_critic_main_steps": (
                FORMAL_B25_EXPECTED_CRITIC_MAIN_STEPS),
            "dqc_formal_b25_expected_critic_warmup_steps": (
                FORMAL_B25_EXPECTED_CRITIC_WARMUP_STEPS),
            "dqc_formal_b25_expected_critic_total_steps": (
                FORMAL_B25_EXPECTED_CRITIC_TOTAL_STEPS),
            "dqc_formal_b25_warmup_batches": FORMAL_B25_WARMUP_BATCHES,
            "dqc_formal_b25_main_batches": FORMAL_B25_MAIN_BATCHES,
            "dqc_formal_b25_actual_rollout_batches": (
                FORMAL_B25_TOTAL_ROLLOUT_BATCHES),
            "dqc_formal_b25_initialization_env_steps": (
                FORMAL_B25_INITIALIZATION_ENV_STEPS),
            "dqc_formal_b25_main_env_steps": FORMAL_B25_MAIN_ENV_STEPS,
            "dqc_formal_b25_actual_env_steps": FORMAL_B25_ACTUAL_ENV_STEPS,
            "dqc_formal_b25_budget_axis": "physical_actual_env_steps_v1",
            "dqc_formal_b25_forced_main_batches": (
                FORMAL_B25_FORCED_MAIN_BATCHES),
            "dqc_formal_b25_event_schedule_offset_batches": (
                FORMAL_B25_EVENT_SCHEDULE_OFFSET_BATCHES),
            "formal_endpoint_primary": True,
            "formal_execution_profile": FORMAL_B25_EXECUTION_PROFILE,
            "dynamics_profile": FORMAL_B25_DYNAMICS_PROFILE,
            "dqc_formal_b25_dynamics_profile": FORMAL_B25_DYNAMICS_PROFILE,
            "train_vec_backend": FORMAL_B25_TRAIN_VEC_BACKEND,
            "validation_vec_backend": FORMAL_B25_VALIDATION_VEC_BACKEND,
            "validation_num_envs": FORMAL_B25_VALIDATION_NUM_ENVS,
            "eval_vec_backend": FORMAL_B25_EVAL_VEC_BACKEND,
            "eval_num_envs": FORMAL_B25_NUM_ENVS,
            "train_simulator_identity": FORMAL_B25_TRAIN_SIMULATOR_IDENTITY,
            "train_mujoco_version": FORMAL_B25_TRAIN_MUJOCO_VERSION,
            "train_mujoco_warp_version": FORMAL_B25_TRAIN_MUJOCO_WARP_VERSION,
            "train_warp_version": FORMAL_B25_TRAIN_WARP_VERSION,
            "train_warp_determinism": FORMAL_B25_TRAIN_WARP_DETERMINISM,
            "train_warp_deterministic_max_records": (
                FORMAL_B25_TRAIN_WARP_DETERMINISTIC_MAX_RECORDS),
            "train_mujoco_gl": FORMAL_B25_TRAIN_MUJOCO_GL,
            "canonical_eval_stack": FORMAL_B25_CANONICAL_EVAL_STACK,
            "formal_evaluation_protocol": (
                FORMAL_B25_EVALUATION_PROTOCOL),
        })
        return identity

    def validate_checkpoint_state(self, state):
        """在父 tensor 校验前验证独立 actor-shuffle RNG state。"""

        if not isinstance(state, dict):
            raise ValueError("formal B25 checkpoint state must be a dictionary")
        rng_state = state.get("dqc_formal_b25_actor_shuffle_rng_state")
        if rng_state is None:
            raise ValueError(
                "formal B25 checkpoint is missing actor shuffle RNG state")

        # 在临时 CPU Generator 上试载；live generator 此时仍保持原 state。
        candidate = torch.Generator(device="cpu")
        try:
            candidate.set_state(torch.as_tensor(rng_state, device="cpu"))
        except (RuntimeError, TypeError, ValueError) as error:
            raise ValueError(
                "formal B25 checkpoint has invalid actor shuffle RNG state"
            ) from error
        applied_epochs = state.get(
            "formal_b25_actor_epochs_applied_total")
        if (
                isinstance(applied_epochs, bool)
                or not isinstance(applied_epochs, Integral)
                or not 0 <= int(applied_epochs) <= (
                    FORMAL_B25_MAIN_BATCHES * FORMAL_B25_MAIN_EPOCHS)):
            raise ValueError(
                "formal B25 checkpoint has invalid applied actor epoch count")

        actor_steps = state.get("actor_optimizer_steps")
        if (
                isinstance(actor_steps, bool)
                or not isinstance(actor_steps, Integral)
                or int(actor_steps) < 0):
            raise ValueError(
                "formal B25 checkpoint has invalid actor optimizer-step count")

        if int(actor_steps) != (
                int(applied_epochs) * FORMAL_B25_ACTOR_MINIBATCHES):
            raise ValueError(
                "formal B25 checkpoint actor steps do not match applied epochs")

        completed_rollouts = state.get("dual_update_count")
        early_stops = state.get("actor_kl_early_stop_count")
        if (
                isinstance(completed_rollouts, bool)
                or not isinstance(completed_rollouts, Integral)
                or not 0 <= int(completed_rollouts) <= FORMAL_B25_MAIN_BATCHES):
            raise ValueError(
                "formal B25 checkpoint has invalid completed actor rollout count")
        if (
                isinstance(early_stops, bool)
                or not isinstance(early_stops, Integral)
                or not 0 <= int(early_stops) <= int(completed_rollouts)):
            raise ValueError(
                "formal B25 checkpoint has invalid actor KL early-stop count")

        # Every completed rollout applies E1..E8. A counted stop occurs only
        # before E8, so each stop skips between one and seven later epochs.
        minimum_epochs = (
            FORMAL_B25_MAIN_EPOCHS * int(completed_rollouts)
            - (FORMAL_B25_MAIN_EPOCHS - 1) * int(early_stops)
        )
        maximum_epochs = (
            FORMAL_B25_MAIN_EPOCHS * int(completed_rollouts)
            - int(early_stops)
        )
        if not minimum_epochs <= int(applied_epochs) <= maximum_epochs:
            raise ValueError(
                "formal B25 checkpoint applied epochs are inconsistent with "
                "completed rollouts and KL early stops")
        super().validate_checkpoint_state(state)

    def checkpoint_state(self):
        """序列化父状态并追加 actor chunk permutation 的私有 CPU RNG。"""

        state = super().checkpoint_state()
        state["dqc_formal_b25_actor_shuffle_rng_state"] = (
            self.formal_b25_actor_shuffle_generator.get_state().clone()
        )
        state["formal_b25_actor_epochs_applied_total"] = int(
            self.formal_b25_actor_epochs_applied_total)
        return state

    def load_checkpoint_state(self, state):
        """原子恢复父状态与 shuffle RNG，任一提交失败则整体回滚。"""

        self.validate_checkpoint_state(state)
        rollback_state = copy.deepcopy(self.checkpoint_state())
        try:
            # 父类先恢复模型、三个 Adam、scheduler、clock 与 MJWarp world RNG。
            super().load_checkpoint_state(state)
            # permutation RNG 最后提交；validation 已在临时 Generator 上证明可载入。
            self.formal_b25_actor_shuffle_generator.set_state(
                torch.as_tensor(
                    state["dqc_formal_b25_actor_shuffle_rng_state"],
                    device="cpu",
                )
            )
            self.formal_b25_actor_epochs_applied_total = int(
                state["formal_b25_actor_epochs_applied_total"])
        except BaseException as error:
            try:
                # 父 load 自身具备事务保护；这里额外覆盖“父已成功、RNG 提交失败”。
                super().load_checkpoint_state(rollback_state)
                self.formal_b25_actor_shuffle_generator.set_state(
                    torch.as_tensor(
                        rollback_state[
                            "dqc_formal_b25_actor_shuffle_rng_state"],
                        device="cpu",
                    )
                )
                self.formal_b25_actor_epochs_applied_total = int(
                    rollback_state["formal_b25_actor_epochs_applied_total"])
            except BaseException as rollback_error:
                raise RuntimeError(
                    "formal B25 checkpoint rollback failed"
                ) from rollback_error
            if isinstance(error, (RuntimeError, TypeError, ValueError)):
                raise ValueError(
                    "formal B25 checkpoint commit failed"
                ) from error
            raise

    def get_training_summary(self):
        """发布 actor、两个 critic 与三套 transition clock 的可审计终点预算。"""

        summary = super().get_training_summary()
        actual_actor_steps = int(summary.get("actor_optimizer_steps", 0))
        actual_quantile_steps = int(summary.get("critic_optimizer_steps", 0))
        actual_value_steps = int(summary.get("value_optimizer_steps", 0))
        early_stop_count = int(summary.get("actor_kl_early_stop_count", 0))
        completed_actor_rollouts = int(self.dual_update_count)
        applied_epochs = int(self.formal_b25_actor_epochs_applied_total)
        skipped_epochs = (
            FORMAL_B25_MAIN_EPOCHS * completed_actor_rollouts - applied_epochs
        )
        minimum_applied_epochs = (
            FORMAL_B25_MAIN_EPOCHS * completed_actor_rollouts
            - (FORMAL_B25_MAIN_EPOCHS - 1) * early_stop_count
        )
        maximum_applied_epochs = (
            FORMAL_B25_MAIN_EPOCHS * completed_actor_rollouts
            - early_stop_count
        )
        actor_target_kl_protocol_consistent = bool(
            0 <= completed_actor_rollouts <= FORMAL_B25_MAIN_BATCHES
            and 0 <= early_stop_count <= completed_actor_rollouts
            and minimum_applied_epochs <= applied_epochs <= maximum_applied_epochs
            and skipped_epochs >= 0
            and actual_actor_steps
            == applied_epochs * FORMAL_B25_ACTOR_MINIBATCHES
        )
        full_actor_target_kl_protocol_complete = bool(
            actor_target_kl_protocol_consistent
            and completed_actor_rollouts == FORMAL_B25_MAIN_BATCHES
        )

        # 同时报告 expected 与 observed；launcher 可据此 fail closed，而无需推测 optimizer 布局。
        summary.update({
            "algorithm_identity": FORMAL_B25_IMPLEMENTATION_ID,
            "registered_algorithm": FORMAL_B25_ALGO,
            "raw_rollout_layout": "T1000_B25",
            "actor_sequence_layout": "T100_B250_contiguous_chunks",
            "actor_bptt_length": FORMAL_B25_ACTOR_CHUNK_LENGTH,
            "actor_rechunking_enabled": True,
            "actor_chunk_length": FORMAL_B25_ACTOR_CHUNK_LENGTH,
            "actor_chunks_per_rollout": FORMAL_B25_ACTOR_CHUNKS,
            "actor_minibatch_size_nominal": (
                FORMAL_B25_ACTOR_MINIBATCH_SIZE),
            "actor_minibatches_per_epoch": FORMAL_B25_ACTOR_MINIBATCHES,
            "actor_chunks_per_minibatch": (
                FORMAL_B25_ACTOR_CHUNKS_PER_MINIBATCH),
            "actor_minibatch_size_effective": (
                FORMAL_B25_EFFECTIVE_ACTOR_MINIBATCH_SIZE),
            "actor_shuffle_protocol": FORMAL_B25_ACTOR_SHUFFLE_PROTOCOL,
            "actor_shuffle_seed": int(self.formal_b25_actor_shuffle_seed),
            "actor_shuffle_rng_isolated": True,
            "transitions_per_rollout": FORMAL_B25_TRANSITIONS,
            "critic_minibatch_size_nominal": (
                FORMAL_B25_CRITIC_MINIBATCH_SIZE),
            "critic_minibatches_per_epoch": FORMAL_B25_CRITIC_MINIBATCHES,
            "critic_minibatch_size_effective": (
                FORMAL_B25_EFFECTIVE_CRITIC_MINIBATCH_SIZE),
            "main_optimizer_epochs": FORMAL_B25_MAIN_EPOCHS,
            "warmup_critic_epochs": FORMAL_B25_WARMUP_CRITIC_EPOCHS,
            "actor_target_kl": FORMAL_B25_ACTOR_TARGET_KL,
            "actor_epoch_protocol": FORMAL_B25_ACTOR_EPOCH_PROTOCOL,
            "actor_kl_early_stop_enabled": True,
            "minimum_actor_optimizer_steps": (
                FORMAL_B25_MINIMUM_ACTOR_OPTIMIZER_STEPS),
            "maximum_actor_optimizer_steps": (
                FORMAL_B25_EXPECTED_ACTOR_OPTIMIZER_STEPS),
            "expected_actor_optimizer_steps": (
                FORMAL_B25_EXPECTED_ACTOR_OPTIMIZER_STEPS),
            "actor_epochs_applied_total": applied_epochs,
            "actor_epochs_skipped_total": skipped_epochs,
            "dual_update_count": completed_actor_rollouts,
            "actor_target_kl_protocol_consistent": (
                actor_target_kl_protocol_consistent),
            "full_actor_target_kl_protocol_complete": (
                full_actor_target_kl_protocol_complete),
            "expected_critic_main_steps_per_critic": (
                FORMAL_B25_EXPECTED_CRITIC_MAIN_STEPS),
            "expected_critic_warmup_steps_per_critic": (
                FORMAL_B25_EXPECTED_CRITIC_WARMUP_STEPS),
            "expected_critic_total_steps_per_critic": (
                FORMAL_B25_EXPECTED_CRITIC_TOTAL_STEPS),
            # Historical field name retained for launcher compatibility. Completion
            # now means every rollout obeyed the signed target-KL protocol, not E8 always.
            "full_actor_epoch_execution_complete": (
                full_actor_target_kl_protocol_complete),
            "full_quantile_optimizer_budget_complete": bool(
                actual_quantile_steps == FORMAL_B25_EXPECTED_CRITIC_TOTAL_STEPS),
            "full_value_optimizer_budget_complete": bool(
                actual_value_steps == FORMAL_B25_EXPECTED_CRITIC_TOTAL_STEPS),
            "warmup_rollout_batches": FORMAL_B25_WARMUP_BATCHES,
            "main_rollout_batches": FORMAL_B25_MAIN_BATCHES,
            "actual_rollout_batches": FORMAL_B25_TOTAL_ROLLOUT_BATCHES,
            "formal_initialization_env_steps": (
                FORMAL_B25_INITIALIZATION_ENV_STEPS),
            "formal_main_env_steps": FORMAL_B25_MAIN_ENV_STEPS,
            "formal_actual_env_steps": FORMAL_B25_ACTUAL_ENV_STEPS,
            "budget_axis": "physical_actual_env_steps_v1",
            "forced_main_batches": FORMAL_B25_FORCED_MAIN_BATCHES,
            "event_schedule_offset_batches": (
                FORMAL_B25_EVENT_SCHEDULE_OFFSET_BATCHES),
            "dual_batch_trajectories": FORMAL_B25_NUM_ENVS,
            "formal_endpoint_primary": True,
            "formal_execution_profile": FORMAL_B25_EXECUTION_PROFILE,
            "dynamics_profile": FORMAL_B25_DYNAMICS_PROFILE,
            "train_vec_backend": FORMAL_B25_TRAIN_VEC_BACKEND,
            "validation_vec_backend": FORMAL_B25_VALIDATION_VEC_BACKEND,
            "validation_num_envs": FORMAL_B25_VALIDATION_NUM_ENVS,
            "eval_vec_backend": FORMAL_B25_EVAL_VEC_BACKEND,
            "eval_num_envs": FORMAL_B25_NUM_ENVS,
            "train_simulator_identity": FORMAL_B25_TRAIN_SIMULATOR_IDENTITY,
            "train_mujoco_version": FORMAL_B25_TRAIN_MUJOCO_VERSION,
            "train_mujoco_warp_version": FORMAL_B25_TRAIN_MUJOCO_WARP_VERSION,
            "train_warp_version": FORMAL_B25_TRAIN_WARP_VERSION,
            "train_warp_determinism": FORMAL_B25_TRAIN_WARP_DETERMINISM,
            "train_warp_deterministic_max_records": (
                FORMAL_B25_TRAIN_WARP_DETERMINISTIC_MAX_RECORDS),
            "train_mujoco_gl": FORMAL_B25_TRAIN_MUJOCO_GL,
            "canonical_eval_stack": FORMAL_B25_CANONICAL_EVAL_STACK,
            "formal_evaluation_protocol": (
                FORMAL_B25_EVALUATION_PROTOCOL),
        })
        summary["full_formal_optimizer_budget_complete"] = bool(
            summary["full_actor_target_kl_protocol_complete"]
            and summary["full_quantile_optimizer_budget_complete"]
            and summary["full_value_optimizer_budget_complete"]
        )
        return summary


__all__ = [
    "DQCACFormalB25GPU",
    "FORMAL_B25_ACTOR_EPOCH_PROTOCOL",
    "FORMAL_B25_ACTOR_TARGET_KL",
    "FORMAL_B25_ACTOR_CHUNK_LENGTH",
    "FORMAL_B25_ACTOR_CHUNKS",
    "FORMAL_B25_ACTOR_CHUNKS_PER_MINIBATCH",
    "FORMAL_B25_ACTOR_MINIBATCHES",
    "FORMAL_B25_ACTOR_MINIBATCH_SIZE",
    "FORMAL_B25_ACTOR_SHUFFLE_PROTOCOL",
    "FORMAL_B25_ACTOR_SHUFFLE_SEED_OFFSET",
    "FORMAL_B25_EFFECTIVE_ACTOR_MINIBATCH_SIZE",
    "FORMAL_B25_ACTION_MODE",
    "FORMAL_B25_ACTUAL_ENV_STEPS",
    "FORMAL_B25_ALGO",
    "FORMAL_B25_CRITIC_MINIBATCHES",
    "FORMAL_B25_CRITIC_MINIBATCH_SIZE",
    "FORMAL_B25_DYNAMICS_PROFILE",
    "FORMAL_B25_EFFECTIVE_CRITIC_MINIBATCH_SIZE",
    "FORMAL_B25_EVALUATION_PROTOCOL",
    "FORMAL_B25_EVAL_VEC_BACKEND",
    "FORMAL_B25_EXECUTION_PROFILE",
    "FORMAL_B25_EVENT_SCHEDULE_OFFSET_BATCHES",
    "FORMAL_B25_EXPECTED_ACTOR_OPTIMIZER_STEPS",
    "FORMAL_B25_EXPECTED_CRITIC_MAIN_STEPS",
    "FORMAL_B25_EXPECTED_CRITIC_TOTAL_STEPS",
    "FORMAL_B25_EXPECTED_CRITIC_WARMUP_STEPS",
    "FORMAL_B25_FORCED_MAIN_BATCHES",
    "FORMAL_B25_HORIZON",
    "FORMAL_B25_IMPLEMENTATION_ID",
    "FORMAL_B25_INITIALIZATION_ENV_STEPS",
    "FORMAL_B25_MAIN_BATCHES",
    "FORMAL_B25_MAIN_ENV_STEPS",
    "FORMAL_B25_MAIN_EPOCHS",
    "FORMAL_B25_MINIMUM_ACTOR_OPTIMIZER_STEPS",
    "FORMAL_B25_NUM_ENVS",
    "FORMAL_B25_TOTAL_ROLLOUT_BATCHES",
    "FORMAL_B25_TRAIN_MUJOCO_VERSION",
    "FORMAL_B25_TRAIN_MUJOCO_GL",
    "FORMAL_B25_TRAIN_MUJOCO_WARP_VERSION",
    "FORMAL_B25_TRAIN_SIMULATOR_IDENTITY",
    "FORMAL_B25_TRAIN_VEC_BACKEND",
    "FORMAL_B25_TRAIN_WARP_DETERMINISM",
    "FORMAL_B25_TRAIN_WARP_DETERMINISTIC_MAX_RECORDS",
    "FORMAL_B25_TRAIN_WARP_VERSION",
    "FORMAL_B25_TRANSITIONS",
    "FORMAL_B25_VALIDATION_NUM_ENVS",
    "FORMAL_B25_VALIDATION_VEC_BACKEND",
    "FORMAL_B25_WARMUP_BATCHES",
    "FORMAL_B25_WARMUP_CRITIC_EPOCHS",
    "FORMAL_B25_CANONICAL_EVAL_STACK",
    "apply_dqc_formal_b25_configuration",
    "canonical_formal_b25_factor_tuple",
    "formal_b25_factor_tuple",
]
