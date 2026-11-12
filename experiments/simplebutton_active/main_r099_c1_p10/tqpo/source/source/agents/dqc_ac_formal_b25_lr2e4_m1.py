# -*- coding: utf-8 -*-
"""DQC 正式 B25 的 policy-LR2e-4、full-actor M1 两个冻结实验身份。

本模块恢复 ``f415ad4`` 已完成实验使用的 full-actor 语义，同时只把 policy
初始学习率固定为 ``2e-4``。两个公开算法除 target-KL 因素外完全相同：

* ``DQCAC_FORMAL_B25_LR2E4_M1_KL``：每个完整 25,000-row actor epoch 后
  测量 behavior-policy KL；达到 0.01 的 E1..E7 保留当前 Adam step 并跳过
  后续 epoch，E8 命中因没有待跳工作而只记录诊断；
* ``DQCAC_FORMAL_B25_LR2E4_M1_NOKL``：target-KL 固定为零，但仍计算并发布
  每个 epoch 的 behavior KL，且每个 rollout 必须执行完整 E8。

两个 arm 都直接把原始 ``[T=1000, B=25]`` rollout 输入 recurrent actor，
不执行当前正式 M25 adapter 的 T100 rechunk。因而每个 epoch 恰好是一个
25,000-transition、BPTT1000 的 full-batch Adam step。quantile/value critic
仍使用 nominal1024、``tensor_split`` M25×effective1000，warmup/main 仍为
20+380 个 rollout，物理预算仍严格为 10M，PPO ratio clip 仍为 0.1。

父正式类为了保持历史 seeded allocation 会以 policy LR=2e-4 构造 Adam。
本 adapter 仅在确认 Adam 尚无 moment 且 step counter 为零之后，同时修正
optimizer defaults、每个 param group 的 ``lr/initial_lr``，并重建 LambdaLR。
该过程不分配模型参数、不消耗 RNG，也不会把一个旧 2e-4 checkpoint 伪装成
新实验；独立 implementation ID、factor tuple 和 scheduler base LR 会共同
拒绝 M25、M10、KL/noKL 之间的交叉加载。
"""

from __future__ import annotations

import copy                                  # 私有旧身份 namespace 与原子校验副本
import json                                  # 稳定序列化有序 factor tuple
import math                                  # 精确检查 LR、KL 与公开浮点字段
from numbers import Integral                 # clock 校验接受整数但显式拒绝 bool
from typing import Dict, Mapping, Tuple      # 标注身份、rollout 与 factor tuple

import torch                                 # optimizer/scheduler 和 full rollout 张量

from .dqc_ac_c1_no_reward_outer import DQCACC1NoRewardOuterGPU
from .dqc_ac_formal_b25 import (
    DQCACFormalB25GPU,
    FORMAL_B25_ACTUAL_ENV_STEPS,
    FORMAL_B25_CRITIC_MINIBATCHES,
    FORMAL_B25_CRITIC_MINIBATCH_SIZE,
    FORMAL_B25_DYNAMICS_PROFILE,
    FORMAL_B25_EFFECTIVE_CRITIC_MINIBATCH_SIZE,
    FORMAL_B25_EXPECTED_CRITIC_MAIN_STEPS,
    FORMAL_B25_EXPECTED_CRITIC_TOTAL_STEPS,
    FORMAL_B25_EXPECTED_CRITIC_WARMUP_STEPS,
    FORMAL_B25_HORIZON,
    FORMAL_B25_MAIN_BATCHES,
    FORMAL_B25_MAIN_EPOCHS,
    FORMAL_B25_NUM_ENVS,
    FORMAL_B25_TRANSITIONS,
    apply_dqc_formal_b25_configuration,
    formal_b25_factor_tuple,
)


# =============================================================================
# 共同 full-batch 几何与真实 policy optimizer 参数
# =============================================================================

# 两个算法名是 runner/launcher 应注册的唯一公开入口。
DQC_LR2E4_M1_KL_ALGO = "DQCAC_FORMAL_B25_LR2E4_M1_KL"
DQC_LR2E4_M1_NOKL_ALGO = "DQCAC_FORMAL_B25_LR2E4_M1_NOKL"

# implementation ID 明确包含 no-rechunk/BPTT1000、KL arm 与物理预算。
DQC_LR2E4_M1_KL_IMPLEMENTATION_ID = (
    "dqc_independent_c1_no_outer_formal_b25_raw_t1000_actor_t1000_"
    "no_rechunk_full_m1_effective25000_bptt1000_actorlr2e-4_"
    "criticvaluelr3e-4_targetkl0p01_epochstop_physical10m_"
    "mjwarp312_implicitfast_train_cpu312_implicitfast_eval_v1"
)
DQC_LR2E4_M1_NOKL_IMPLEMENTATION_ID = (
    "dqc_independent_c1_no_outer_formal_b25_raw_t1000_actor_t1000_"
    "no_rechunk_full_m1_effective25000_bptt1000_actorlr2e-4_"
    "criticvaluelr3e-4_targetkl0_fixedE8_kl_diag_physical10m_"
    "mjwarp312_implicitfast_train_cpu312_implicitfast_eval_v1"
)

# ``theta_a`` 只是旧参数化的等价元数据；真实 LambdaLR base 是 actor_lr。
DQC_LR2E4_M1_ACTOR_LR = 2e-4
DQC_LR2E4_M1_ACTOR_LR_B = 10_000.0
DQC_LR2E4_M1_ACTOR_LR_C = 0.9
DQC_LR2E4_M1_THETA_A = (
    DQC_LR2E4_M1_ACTOR_LR_B ** DQC_LR2E4_M1_ACTOR_LR_C
) * DQC_LR2E4_M1_ACTOR_LR

# 一个 recurrent sequence 对应一条完整 T1000 trajectory；25 条一起做一个 step。
DQC_LR2E4_M1_ACTOR_BPTT_LENGTH = FORMAL_B25_HORIZON
DQC_LR2E4_M1_ACTOR_CHUNK_LENGTH = FORMAL_B25_HORIZON
DQC_LR2E4_M1_ACTOR_CHUNKS = FORMAL_B25_NUM_ENVS
DQC_LR2E4_M1_ACTOR_MINIBATCHES = 1
DQC_LR2E4_M1_ACTOR_CHUNKS_PER_MINIBATCH = FORMAL_B25_NUM_ENVS
DQC_LR2E4_M1_ACTOR_MINIBATCH_SIZE = FORMAL_B25_TRANSITIONS
DQC_LR2E4_M1_EFFECTIVE_ACTOR_MINIBATCH_SIZE = FORMAL_B25_TRANSITIONS
DQC_LR2E4_M1_ACTOR_LAYOUT = "T1000_B25_no_rechunk"
DQC_LR2E4_M1_ACTOR_SHUFFLE_PROTOCOL = (
    "none_full_batch_T1000_B25_no_rechunk_v1"
)
# 字段保持整数类型以兼容已有 runner signature；协议明确说明该 seed 不被消费。
DQC_LR2E4_M1_ACTOR_SHUFFLE_SEED = 0

# KL arm 最少每个 rollout 执行 E1；noKL arm 的最小值就是固定 E8 上界。
DQC_LR2E4_M1_KL_TARGET = 0.01
DQC_LR2E4_M1_NOKL_TARGET = 0.0
DQC_LR2E4_M1_KL_MINIMUM_ACTOR_OPTIMIZER_STEPS = FORMAL_B25_MAIN_BATCHES
DQC_LR2E4_M1_EXPECTED_ACTOR_OPTIMIZER_STEPS = (
    FORMAL_B25_MAIN_BATCHES * FORMAL_B25_MAIN_EPOCHS
)
DQC_LR2E4_M1_NOKL_MINIMUM_ACTOR_OPTIMIZER_STEPS = (
    DQC_LR2E4_M1_EXPECTED_ACTOR_OPTIMIZER_STEPS
)

DQC_LR2E4_M1_KL_ACTOR_EPOCH_PROTOCOL = (
    "full_batch_m1_up_to_E8_whole_behavior_kl0p01_epoch_stop_v1"
)
DQC_LR2E4_M1_NOKL_ACTOR_EPOCH_PROTOCOL = (
    "full_batch_m1_fixed_E8_target_kl_disabled_kl_diagnostic_v1"
)

# 这些等式是实验身份的一部分；未来改 rollout 几何时必须提升 implementation ID。
if DQC_LR2E4_M1_ACTOR_BPTT_LENGTH != 1_000:
    raise RuntimeError("DQC LR2e-4/M1 actor BPTT must remain exactly 1000")
if DQC_LR2E4_M1_EFFECTIVE_ACTOR_MINIBATCH_SIZE != 25_000:
    raise RuntimeError("DQC LR2e-4/M1 actor batch must contain 25000 rows")
if DQC_LR2E4_M1_EXPECTED_ACTOR_OPTIMIZER_STEPS != 3_040:
    raise RuntimeError("DQC LR2e-4/M1 fixed-E8 actor budget must equal 3040")
if FORMAL_B25_ACTUAL_ENV_STEPS != 10_000_000:
    raise RuntimeError("DQC LR2e-4/M1 physical budget must remain exactly 10M")


# =============================================================================
# 两个 arm 的不可变规格与 factor tuple
# =============================================================================

_ARM_SPECS = {
    "kl": {
        "algorithm": DQC_LR2E4_M1_KL_ALGO,
        "implementation": DQC_LR2E4_M1_KL_IMPLEMENTATION_ID,
        "target_kl": DQC_LR2E4_M1_KL_TARGET,
        "epoch_protocol": DQC_LR2E4_M1_KL_ACTOR_EPOCH_PROTOCOL,
        "actor_reuse": (
            "ppo_clip_up_to_E8_raw_T1000_B25_no_rechunk_full_m1_"
            "effective25000_ratio0p1_whole_kl0p01_epoch_stop_v1"),
        "minimum_steps": DQC_LR2E4_M1_KL_MINIMUM_ACTOR_OPTIMIZER_STEPS,
        "wandb_group": "formal-b25-lr2e4-full-target-kl-0p01-10seed-v1",
        "wandb_stem": "dqcac-formal-b25-lr2e4-full-kl-gremlin",
        "wandb_tag": "actor-target-kl-0p01-epoch-stop",
    },
    "nokl": {
        "algorithm": DQC_LR2E4_M1_NOKL_ALGO,
        "implementation": DQC_LR2E4_M1_NOKL_IMPLEMENTATION_ID,
        "target_kl": DQC_LR2E4_M1_NOKL_TARGET,
        "epoch_protocol": DQC_LR2E4_M1_NOKL_ACTOR_EPOCH_PROTOCOL,
        "actor_reuse": (
            "ppo_clip_fixed_E8_raw_T1000_B25_no_rechunk_full_m1_"
            "effective25000_ratio0p1_target_kl_disabled_kl_diag_v1"),
        "minimum_steps": DQC_LR2E4_M1_NOKL_MINIMUM_ACTOR_OPTIMIZER_STEPS,
        "wandb_group": "formal-b25-lr2e4-full-no-kl-10seed-v1",
        "wandb_stem": "dqcac-formal-b25-lr2e4-full-nokl-gremlin",
        "wandb_tag": "actor-target-kl-disabled-fixed-e8",
    },
}


def _arm_spec(arm: str) -> Mapping[str, object]:
    """返回内部 arm 规格；拒绝拼写回退，避免 KL 身份被静默混用。"""

    try:
        return _ARM_SPECS[str(arm)]
    except KeyError as error:
        raise ValueError(f"unknown DQC LR2e-4/M1 arm {arm!r}") from error


def _factor_replacements(arm: str) -> Dict[str, object]:
    """只返回与当前 M25 正式 identity 不同的 actor 因素。"""

    spec = _arm_spec(arm)
    return {
        "actor_optimizer": "adam_lr2e-4_eps1e-5_absolute_batch_schedule_v1",
        "actor_reduction": "fixed_full_rollout_25000_v1",
        "actor_reuse": spec["actor_reuse"],
        "actor_sequence_layout": "T1000_B25_no_rechunk_v1",
        "actor_bptt_length": DQC_LR2E4_M1_ACTOR_BPTT_LENGTH,
        "actor_chunk_length": DQC_LR2E4_M1_ACTOR_CHUNK_LENGTH,
        "actor_chunks_per_rollout": DQC_LR2E4_M1_ACTOR_CHUNKS,
        "actor_minibatch_size_nominal": DQC_LR2E4_M1_ACTOR_MINIBATCH_SIZE,
        "actor_minibatches_per_epoch": DQC_LR2E4_M1_ACTOR_MINIBATCHES,
        "actor_chunks_per_minibatch": (
            DQC_LR2E4_M1_ACTOR_CHUNKS_PER_MINIBATCH),
        "actor_minibatch_size_effective": (
            DQC_LR2E4_M1_EFFECTIVE_ACTOR_MINIBATCH_SIZE),
        "actor_shuffle_protocol": DQC_LR2E4_M1_ACTOR_SHUFFLE_PROTOCOL,
        "actor_shuffle_seed_offset": None,
        "actor_target_kl": spec["target_kl"],
        "actor_epoch_protocol": spec["epoch_protocol"],
        "minimum_actor_optimizer_steps": spec["minimum_steps"],
        "maximum_actor_optimizer_steps": (
            DQC_LR2E4_M1_EXPECTED_ACTOR_OPTIMIZER_STEPS),
    }


def _dqc_lr2e4_m1_factor_tuple(arm: str) -> Tuple[Tuple[str, object], ...]:
    """原位替换正式 factor tuple，确保没有重复或遗漏的科学 key。"""

    replacements = _factor_replacements(arm)
    replaced = set()
    factors = []
    for name, value in formal_b25_factor_tuple():
        if name in replacements:
            value = replacements[name]
            replaced.add(name)
        factors.append((name, value))

    # 父 identity 字段一旦改名，本 adapter 必须 import-time/test-time 显式失败。
    if replaced != set(replacements):
        missing = sorted(set(replacements) - replaced)
        raise RuntimeError(
            "DQC LR2e-4/M1 factor replacement keys are missing: "
            + ", ".join(missing)
        )
    names = [name for name, _value in factors]
    if len(names) != len(set(names)):
        raise RuntimeError("DQC LR2e-4/M1 factor tuple contains duplicate keys")
    return tuple(factors)


def dqc_lr2e4_m1_kl_factor_tuple() -> Tuple[Tuple[str, object], ...]:
    """返回 target-KL=0.01 full-batch arm 的有序因素。"""

    return _dqc_lr2e4_m1_factor_tuple("kl")


def dqc_lr2e4_m1_nokl_factor_tuple() -> Tuple[Tuple[str, object], ...]:
    """返回 target-KL=0、固定 E8 full-batch arm 的有序因素。"""

    return _dqc_lr2e4_m1_factor_tuple("nokl")


def _canonical_factor_tuple(arm: str) -> str:
    """生成无空白、跨 locale 稳定的 checkpoint factor 字符串。"""

    return json.dumps(
        _dqc_lr2e4_m1_factor_tuple(arm),
        separators=(",", ":"),
        ensure_ascii=True,
    )


def canonical_dqc_lr2e4_m1_kl_factor_tuple() -> str:
    """序列化 target-KL full-batch factor tuple。"""

    return _canonical_factor_tuple("kl")


def canonical_dqc_lr2e4_m1_nokl_factor_tuple() -> str:
    """序列化 noKL/fixed-E8 full-batch factor tuple。"""

    return _canonical_factor_tuple("nokl")


def _apply_lr2e4_m1_overrides(args, arm: str):
    """在 canonical formal namespace 上安装一个且仅一个 full-batch arm。"""

    spec = _arm_spec(arm)
    factor_tuple = _canonical_factor_tuple(arm)
    args.algo_name = spec["algorithm"]

    # 真实 Adam 输入、公开初值与旧 theta 参数化必须共同指向 2e-4。
    args.actor_lr = DQC_LR2E4_M1_ACTOR_LR
    args.actor_lr0 = DQC_LR2E4_M1_ACTOR_LR
    args.theta_b = DQC_LR2E4_M1_ACTOR_LR_B
    args.theta_c = DQC_LR2E4_M1_ACTOR_LR_C
    args.theta_a = DQC_LR2E4_M1_THETA_A

    # 三层历史 validator 都读取自己的旧命名空间；统一换成当前 arm identity。
    args.dqc_independent_factorial_implementation_identity = (
        spec["implementation"])
    args.dqc_independent_factorial_factor_tuple = factor_tuple
    args.dqc_c1_no_reward_outer_implementation_identity = (
        spec["implementation"])
    args.dqc_c1_no_reward_outer_factor_tuple = factor_tuple
    args.dqc_formal_b25_implementation_identity = spec["implementation"]
    args.dqc_formal_b25_factor_tuple = factor_tuple

    # full actor 直接消费原始 T1000×B25；以下冗余字段供 runner/checkpoint 审计。
    args.dqc_formal_b25_actor_layout = DQC_LR2E4_M1_ACTOR_LAYOUT
    args.dqc_formal_b25_actor_bptt_length = DQC_LR2E4_M1_ACTOR_BPTT_LENGTH
    args.dqc_formal_b25_actor_chunk_length = DQC_LR2E4_M1_ACTOR_CHUNK_LENGTH
    args.dqc_formal_b25_actor_chunks = DQC_LR2E4_M1_ACTOR_CHUNKS
    args.dqc_formal_b25_actor_minibatch_size = (
        DQC_LR2E4_M1_ACTOR_MINIBATCH_SIZE)
    args.dqc_formal_b25_actor_minibatches_per_epoch = (
        DQC_LR2E4_M1_ACTOR_MINIBATCHES)
    args.dqc_formal_b25_actor_chunks_per_minibatch = (
        DQC_LR2E4_M1_ACTOR_CHUNKS_PER_MINIBATCH)
    args.dqc_formal_b25_effective_actor_minibatch_size = (
        DQC_LR2E4_M1_EFFECTIVE_ACTOR_MINIBATCH_SIZE)
    args.dqc_formal_b25_actor_chunk_counts_per_minibatch = (
        DQC_LR2E4_M1_ACTOR_CHUNKS_PER_MINIBATCH,)
    args.dqc_formal_b25_actor_effective_minibatch_sizes = (
        DQC_LR2E4_M1_EFFECTIVE_ACTOR_MINIBATCH_SIZE,)
    args.dqc_formal_b25_actor_shuffle_protocol = (
        DQC_LR2E4_M1_ACTOR_SHUFFLE_PROTOCOL)
    args.dqc_formal_b25_actor_shuffle_seed = DQC_LR2E4_M1_ACTOR_SHUFFLE_SEED

    # ``target=0`` disables only the break; PPO clipping and all KL measurements remain active。
    args.dqc_target_kl = spec["target_kl"]
    args.dqc_formal_b25_actor_target_kl = spec["target_kl"]
    args.dqc_formal_b25_actor_epoch_protocol = spec["epoch_protocol"]
    args.dqc_formal_b25_minimum_actor_optimizer_steps = spec["minimum_steps"]
    args.dqc_formal_b25_expected_actor_optimizer_steps = (
        DQC_LR2E4_M1_EXPECTED_ACTOR_OPTIMIZER_STEPS)
    args.dqc_formal_b25_maximum_actor_optimizer_steps = (
        DQC_LR2E4_M1_EXPECTED_ACTOR_OPTIMIZER_STEPS)

    # 新前缀让集成方不必推断旧 formal 字段当前属于哪个 adapter arm。
    args.dqc_formal_b25_lr2e4_m1_arm = arm
    args.dqc_formal_b25_lr2e4_m1_implementation_identity = (
        spec["implementation"])
    args.dqc_formal_b25_lr2e4_m1_factor_tuple = factor_tuple
    args.dqc_formal_b25_lr2e4_m1_actor_lr = DQC_LR2E4_M1_ACTOR_LR
    args.dqc_formal_b25_lr2e4_m1_target_kl = spec["target_kl"]

    # 两个默认 W&B namespace 永不相撞；campaign 仍可覆盖纯 operational 字段。
    args.wandb_group = spec["wandb_group"]
    args.wandb_name = f"{spec['wandb_stem']}-seed{int(args.seed)}"
    args.wandb_lineage = args.wandb_name
    args.experiment_variant = args.wandb_name
    obsolete_tags = {
        "actor-minibatch-1024-effective-1000",
        "actor-target-kl-0p01-epoch-stop",
    }
    tags = [
        tag.strip()
        for tag in str(getattr(args, "wandb_tags", "") or "").split(",")
        if tag.strip() and tag.strip() not in obsolete_tags
    ]
    for tag in (
            "dqcac-policy-lr-2e-4", "actor-full-batch-M1",
            "actor-bptt-1000", "actor-no-rechunk",
            "actor-minibatch-effective-25000", "critic-value-M25",
            spec["wandb_tag"]):
        if tag not in tags:
            tags.append(tag)
    args.wandb_tags = ",".join(tags)
    return args


def apply_dqc_formal_b25_lr2e4_m1_kl_configuration(args):
    """冻结 LR2e-4/full-M1/target-KL=0.01 的完整正式配置。"""

    apply_dqc_formal_b25_configuration(args)
    return _apply_lr2e4_m1_overrides(args, "kl")


def apply_dqc_formal_b25_lr2e4_m1_nokl_configuration(args):
    """冻结 LR2e-4/full-M1/target-KL=0/fixed-E8 的完整正式配置。"""

    apply_dqc_formal_b25_configuration(args)
    return _apply_lr2e4_m1_overrides(args, "nokl")


# 路径、W&B 名称与展示字段可由 launcher 覆盖；其余字段必须在构造前 fail closed。
_PUBLIC_SCIENTIFIC_FIELDS = (
    "algo_name", "gamma", "horizon", "num_envs", "num_iterations",
    "failure_alpha", "q_alpha", "return_threshold", "action_execution_mode",
    "actor_lr", "actor_lr0", "theta_a", "theta_b", "theta_c",
    "critic_lr", "dqc_value_lr", "critic_minibatch_size",
    "critic_epochs", "dqc_value_epochs", "actor_epochs",
    "dqc_actor_reuse_mode", "dqc_ratio_clip", "dqc_target_kl",
    "warmup_iters", "critic_warmup_mode", "warmup_critic_epochs",
    "training_budget_semantics", "dqc_reward_outer_discount",
    "formal_execution_profile", "dynamics_profile", "vec_backend",
    "train_vec_backend", "validation_vec_backend", "validation_num_envs",
    "eval_vec_backend", "eval_num_envs", "train_simulator_identity",
    "train_mujoco_version", "train_mujoco_warp_version", "train_warp_version",
    "train_warp_determinism", "train_warp_deterministic_max_records",
    "train_mujoco_gl", "canonical_eval_stack", "formal_evaluation_protocol",
    "dqc_independent_factorial_implementation_identity",
    "dqc_independent_factorial_factor_tuple",
    "dqc_c1_no_reward_outer_implementation_identity",
    "dqc_c1_no_reward_outer_factor_tuple",
    "dqc_formal_b25_implementation_identity", "dqc_formal_b25_factor_tuple",
    "dqc_formal_b25_raw_layout", "dqc_formal_b25_actor_layout",
    "dqc_formal_b25_actor_bptt_length", "dqc_formal_b25_actor_chunk_length",
    "dqc_formal_b25_actor_chunks", "dqc_formal_b25_actor_minibatch_size",
    "dqc_formal_b25_actor_minibatches_per_epoch",
    "dqc_formal_b25_actor_chunks_per_minibatch",
    "dqc_formal_b25_effective_actor_minibatch_size",
    "dqc_formal_b25_actor_chunk_counts_per_minibatch",
    "dqc_formal_b25_actor_effective_minibatch_sizes",
    "dqc_formal_b25_actor_shuffle_protocol",
    "dqc_formal_b25_actor_shuffle_seed", "dqc_formal_b25_actor_epoch_protocol",
    "dqc_formal_b25_actor_target_kl",
    "dqc_formal_b25_minimum_actor_optimizer_steps",
    "dqc_formal_b25_expected_actor_optimizer_steps",
    "dqc_formal_b25_maximum_actor_optimizer_steps",
    "dqc_formal_b25_critic_minibatches_per_epoch",
    "dqc_formal_b25_effective_critic_minibatch_size",
    "dqc_formal_b25_expected_critic_main_steps",
    "dqc_formal_b25_expected_critic_warmup_steps",
    "dqc_formal_b25_expected_critic_total_steps",
    "dqc_formal_b25_warmup_batches", "dqc_formal_b25_main_batches",
    "dqc_formal_b25_actual_rollout_batches",
    "dqc_formal_b25_initialization_env_steps",
    "dqc_formal_b25_main_env_steps", "dqc_formal_b25_actual_env_steps",
    "dqc_formal_b25_budget_axis", "dqc_formal_b25_forced_main_batches",
    "dqc_formal_b25_event_schedule_offset_batches", "forced_event_batches",
    "event_schedule_offset_batches", "formal_endpoint_primary",
    "dqc_formal_b25_lr2e4_m1_arm",
    "dqc_formal_b25_lr2e4_m1_implementation_identity",
    "dqc_formal_b25_lr2e4_m1_factor_tuple",
    "dqc_formal_b25_lr2e4_m1_actor_lr",
    "dqc_formal_b25_lr2e4_m1_target_kl",
)


class _DQCACFormalB25LR2E4M1GPU(DQCACFormalB25GPU):
    """共享 full actor 实现；公开子类只选择被冻结的 KL arm。"""

    ARM = ""                                # 子类必须覆写；空值永不回退到某个 arm

    @classmethod
    def _spec(cls) -> Mapping[str, object]:
        """返回类级冻结规格，并让误实例化私有基类立即失败。"""

        return _arm_spec(cls.ARM)

    @classmethod
    def _apply_configuration(cls, args):
        """在临时 namespace 上重建本类唯一合法的公共配置。"""

        apply_dqc_formal_b25_configuration(args)
        return _apply_lr2e4_m1_overrides(args, cls.ARM)

    @classmethod
    def _validate_public_configuration(cls, args, env) -> None:
        """在任何 worker、模型、Adam 或 W&B 分配前拒绝科学字段漂移。"""

        expected = copy.copy(args)
        cls._apply_configuration(expected)
        missing = object()
        mismatches = []
        for name in _PUBLIC_SCIENTIFIC_FIELDS:
            actual = getattr(args, name, missing)
            required = getattr(expected, name, missing)
            if actual != required:
                mismatches.append(f"{name}={actual!r} (required {required!r})")

        # reference env 是构造时唯一真实 MDP 证据，不能只信 args 的字符串标签。
        env_profile = str(getattr(env, "dynamics_profile", ""))
        if env_profile != FORMAL_B25_DYNAMICS_PROFILE:
            mismatches.append(
                "environment_dynamics_profile="
                f"{env_profile!r} (required {FORMAL_B25_DYNAMICS_PROFILE!r})"
            )
        if mismatches:
            raise ValueError(
                f"DQC formal B25 LR2e-4/M1/{cls.ARM} identity mismatch: "
                + ", ".join(mismatches)
            )

    def _install_actor_lr2e4_before_first_step(self) -> None:
        """在无 Adam moment 状态下原子修正 optimizer 与 LambdaLR 的 LR 身份。"""

        if int(getattr(self, "actor_optimizer_steps", -1)) != 0:
            raise RuntimeError(
                "DQC LR2e-4/M1 actor optimizer changed before LR installation")
        if self.actor_optimizer.state:
            raise RuntimeError(
                "DQC LR2e-4/M1 actor Adam has state before its first update")
        if not self.actor_optimizer.param_groups:
            raise RuntimeError("DQC LR2e-4/M1 actor Adam has no parameter groups")

        # LambdaLR 读取 group.initial_lr 作为 base；三个位置必须一起替换。
        self.actor_optimizer.defaults["lr"] = DQC_LR2E4_M1_ACTOR_LR
        for group in self.actor_optimizer.param_groups:
            group["lr"] = DQC_LR2E4_M1_ACTOR_LR
            group["initial_lr"] = DQC_LR2E4_M1_ACTOR_LR
        self.actor_scheduler = self._new_actor_scheduler()
        self._set_actor_lr_batch_index(0)

        # 校验所有 group，而不是仅检查常见的第一个 group。
        expected_base_lrs = [
            DQC_LR2E4_M1_ACTOR_LR
        ] * len(self.actor_optimizer.param_groups)
        if list(self.actor_scheduler.base_lrs) != expected_base_lrs:
            raise RuntimeError("DQC LR2e-4/M1 scheduler base LR installation failed")
        for group in self.actor_optimizer.param_groups:
            for field in ("lr", "initial_lr"):
                if not math.isclose(
                        float(group.get(field, float("nan"))),
                        DQC_LR2E4_M1_ACTOR_LR,
                        rel_tol=0.0,
                        abs_tol=1e-15):
                    raise RuntimeError(
                        f"DQC LR2e-4/M1 actor optimizer {field} installation failed")

    def __init__(self, args, env):
        """复用旧 seeded allocation，首个 optimizer step 前安装新 LR/full identity。"""

        self._validate_public_configuration(args, env)

        # 当前正式父类只接受自己的 M25/LR2e-4 namespace；私有副本保证其模型
        # 初始化次序完全不变。这里不修改调用者持有的新 arm args。
        allocation_args = copy.copy(args)
        apply_dqc_formal_b25_configuration(allocation_args)
        super().__init__(allocation_args, env)
        self._install_actor_lr2e4_before_first_step()

        spec = self._spec()
        factor_tuple = _canonical_factor_tuple(self.ARM)

        # 以下赋值不分配 tensor、不消耗 RNG；父类的 shuffle generator 保留但永不使用。
        self.algo_name = spec["algorithm"]
        self.actor_lr0 = DQC_LR2E4_M1_ACTOR_LR
        self.actor_lr_b = DQC_LR2E4_M1_ACTOR_LR_B
        self.actor_lr_c = DQC_LR2E4_M1_ACTOR_LR_C
        self.theta_a = DQC_LR2E4_M1_THETA_A
        self.dqc_target_kl = spec["target_kl"]
        self.formal_b25_actor_minibatches_per_epoch = (
            DQC_LR2E4_M1_ACTOR_MINIBATCHES)
        self.formal_b25_actor_chunk_counts_per_minibatch = (
            DQC_LR2E4_M1_ACTOR_CHUNKS_PER_MINIBATCH,)
        self.formal_b25_actor_effective_minibatch_sizes = (
            DQC_LR2E4_M1_EFFECTIVE_ACTOR_MINIBATCH_SIZE,)
        self.formal_b25_actor_chunks_per_minibatch = (
            DQC_LR2E4_M1_ACTOR_CHUNKS_PER_MINIBATCH)
        self.formal_b25_effective_actor_minibatch_size = (
            DQC_LR2E4_M1_EFFECTIVE_ACTOR_MINIBATCH_SIZE)
        self.formal_b25_actor_shuffle_seed = DQC_LR2E4_M1_ACTOR_SHUFFLE_SEED
        self.formal_b25_expected_actor_optimizer_steps = (
            DQC_LR2E4_M1_EXPECTED_ACTOR_OPTIMIZER_STEPS)
        self.formal_b25_factor_tuple = factor_tuple
        self.independent_factorial_factor_tuple = factor_tuple

    @staticmethod
    def _require_full_actor_population(
            rollout: Mapping[str, object], combined: torch.Tensor) -> None:
        """验证 actor 看到原始 T1000×B25，而不是 T100 rechunk 或子集。"""

        required = ("S", "A_raw", "valid", "logp", "mean_old", "log_std_old")
        missing = [name for name in required if name not in rollout]
        if missing:
            raise ValueError(
                "DQC LR2e-4/M1 actor rollout is missing fields: "
                + ", ".join(missing))
        for name in required:
            value = rollout[name]
            if (
                    not torch.is_tensor(value)
                    or value.ndim < 2
                    or tuple(value.shape[:2]) != (
                        FORMAL_B25_HORIZON, FORMAL_B25_NUM_ENVS)):
                raise ValueError(
                    f"DQC LR2e-4/M1 {name} must start with [1000,25]")
        if (
                not torch.is_tensor(combined)
                or combined.ndim < 2
                or tuple(combined.shape[:2]) != (
                    FORMAL_B25_HORIZON, FORMAL_B25_NUM_ENVS)):
            raise ValueError(
                "DQC LR2e-4/M1 combined weight must start with [1000,25]")

    def _replay_actor_for_update(self, rollout):
        """从隐式零状态直接重放完整 T1000 序列，即 f415ad4 的 full actor。"""

        observations = rollout.get("S")
        if (
                not torch.is_tensor(observations)
                or tuple(observations.shape[:2]) != (
                    FORMAL_B25_HORIZON, FORMAL_B25_NUM_ENVS)):
            raise ValueError(
                "DQC LR2e-4/M1 actor replay requires raw [1000,25,*] states")
        # 不传 actor_initial_state；LSTM 在完整 trajectory 起点使用自己的零状态。
        return self.actor(observations)

    def _actor_loss_denominator(self, rollout):
        """返回固定 25,000 population 分母，不按有效 episode 长度重加权。"""

        valid = rollout.get("valid")
        if (
                not torch.is_tensor(valid)
                or tuple(valid.shape) != (
                    FORMAL_B25_HORIZON, FORMAL_B25_NUM_ENVS)
                or int(valid.numel()) != FORMAL_B25_TRANSITIONS):
            raise ValueError(
                "DQC LR2e-4/M1 actor loss requires a [1000,25] valid mask")
        return float(FORMAL_B25_TRANSITIONS)

    def update_actor_epochs(self, rollout, combined):
        """执行 E1..E8 full-batch PPO step，并只在完整 25k epoch 边界判 KL。"""

        spec = self._spec()
        epochs = int(getattr(self, "actor_epochs", -1))
        if epochs != FORMAL_B25_MAIN_EPOCHS:
            raise RuntimeError("DQC LR2e-4/M1 actor must request exactly E8")
        if str(getattr(self, "dqc_actor_reuse_mode", "")) != "ppo_clip":
            raise RuntimeError("DQC LR2e-4/M1 actor requires PPO clipping")
        target_kl = float(getattr(self, "dqc_target_kl", float("nan")))
        if not math.isclose(
                target_kl, float(spec["target_kl"]),
                rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError("DQC LR2e-4/M1 target KL drifted at runtime")
        self._require_full_actor_population(rollout, combined)

        # rollout-time Gaussian 参数是真实 behavior policy；expanded log_std 必须 clone，
        # 否则 Adam 原地更新参数后旧 view 会同步改变并低估 KL。
        behavior_mean = rollout["mean_old"].detach().clone()
        behavior_log_std = rollout["log_std_old"].detach().clone()
        epoch_metrics = []
        behavior_kls = []
        early_stopped = False
        early_stop_epoch = 0
        starting_applied_epochs = int(self.formal_b25_actor_epochs_applied_total)

        try:
            for epoch_index in range(epochs):
                # 每次调用恰好提交一个 25k Adam step；本循环绝不触碰 LambdaLR，
                # scheduler 的 k 只由 _update_rollout 在 rollout 边界推进。
                epoch_info = self.update_actor_ppo_epoch(
                    rollout, combined, behavior_mean, behavior_log_std)
                behavior_kl = float(
                    epoch_info["diagnostic/dqc/policy_kl_behavior_after_epoch"])
                if not math.isfinite(behavior_kl):
                    raise FloatingPointError(
                        "non-finite DQC LR2e-4/M1 full-population behavior KL")
                epoch_metrics.append(epoch_info)
                behavior_kls.append(behavior_kl)
                self.formal_b25_actor_epochs_applied_total += 1

                # 命中 epoch 保留。E8 后没有待跳 epoch，因此不虚增 early-stop count。
                completed_epoch = epoch_index + 1
                if (
                        completed_epoch < epochs
                        and target_kl > 0.0
                        and behavior_kl >= target_kl):
                    early_stopped = True
                    early_stop_epoch = completed_epoch
                    self.actor_kl_early_stop_count = int(getattr(
                        self, "actor_kl_early_stop_count", 0)) + 1
                    break
        finally:
            # C1 split surrogate 的标签只能属于当前 rollout；及时释放 25k tensor 引用。
            self._independent_factorial_actor_terms = None

        applied_epochs = len(epoch_metrics)
        if applied_epochs <= 0:
            raise RuntimeError("DQC LR2e-4/M1 actor applied no epoch")
        if self.formal_b25_actor_epochs_applied_total != (
                starting_applied_epochs + applied_epochs):
            raise RuntimeError("DQC LR2e-4/M1 applied-epoch counter drifted")

        # 父 primitive 已检查 PPO ratio、loss、gradient 和逐步 KL 的 finite 性。
        last = dict(epoch_metrics[-1])
        losses = [item["actor/loss"] for item in epoch_metrics]
        grad_norms = [
            item["diagnostic/dqc/actor_grad_norm_pre_clip"]
            for item in epoch_metrics]
        step_kls = [
            item["diagnostic/dqc/policy_kl_after_step"]
            for item in epoch_metrics]
        valid_rows = float(rollout["valid"].sum().item())
        last.update({
            "diagnostic/dqc/actor_epochs_requested": float(epochs),
            "diagnostic/dqc/actor_epochs_applied": float(applied_epochs),
            "diagnostic/dqc/actor_minibatches_per_epoch": 1.0,
            "diagnostic/dqc/actor_minibatches_applied": float(applied_epochs),
            "diagnostic/dqc/actor_chunks_per_minibatch": float(
                DQC_LR2E4_M1_ACTOR_CHUNKS_PER_MINIBATCH),
            "diagnostic/dqc/actor_minibatch_size_effective": float(
                DQC_LR2E4_M1_EFFECTIVE_ACTOR_MINIBATCH_SIZE),
            "diagnostic/dqc/actor_minibatch_valid_rows_min": valid_rows,
            "diagnostic/dqc/actor_minibatch_valid_rows_max": valid_rows,
            "diagnostic/dqc/actor_epoch_chunk_coverage_min": float(
                DQC_LR2E4_M1_ACTOR_CHUNKS),
            "diagnostic/dqc/actor_epoch_chunk_coverage_max": float(
                DQC_LR2E4_M1_ACTOR_CHUNKS),
            "diagnostic/dqc/actor_loss_epoch_mean": float(
                sum(losses) / applied_epochs),
            "diagnostic/dqc/actor_grad_norm_epoch_mean": float(
                sum(grad_norms) / applied_epochs),
            "diagnostic/dqc/policy_kl_step_mean": float(
                sum(step_kls) / applied_epochs),
            "diagnostic/dqc/policy_kl_step_max": float(max(step_kls)),
            "diagnostic/dqc/policy_kl_behavior_after_epoch": float(
                behavior_kls[-1]),
            "diagnostic/dqc/policy_kl_behavior_epoch_mean": float(
                sum(behavior_kls) / applied_epochs),
            "diagnostic/dqc/policy_kl_behavior_epoch_max": float(
                max(behavior_kls)),
            "diagnostic/dqc/policy_kl_after_epochs": float(behavior_kls[-1]),
            "diagnostic/dqc/actor_kl_early_stop": float(early_stopped),
            "diagnostic/dqc/actor_kl_early_stop_count": float(getattr(
                self, "actor_kl_early_stop_count", 0)),
            "diagnostic/dqc/actor_kl_early_stop_epoch": float(early_stop_epoch),
            "diagnostic/dqc/actor_kl_early_stop_enabled": float(target_kl > 0.0),
            "diagnostic/dqc/actor_kl_target": target_kl,
            "diagnostic/dqc/actor_epochs_applied_total": float(
                self.formal_b25_actor_epochs_applied_total),
        })

        # 固定八槽让 10-seed 聚合无需动态列；未执行槽 carry-forward 但 mask=0。
        carried_kls = list(behavior_kls)
        carried_kls.extend([behavior_kls[-1]] * (epochs - applied_epochs))
        for epoch_number, epoch_kl in enumerate(carried_kls, start=1):
            applied = float(epoch_number <= applied_epochs)
            metric = f"policy_kl_behavior_epoch_{epoch_number}"
            last[metric] = float(epoch_kl)
            last[f"diagnostic/dqc/{metric}"] = float(epoch_kl)
            last[f"{metric}_applied"] = applied
            last[f"diagnostic/dqc/{metric}_applied"] = applied
        return last

    def _checkpoint_identity(self) -> Dict[str, object]:
        """覆盖父 M25 字段并冗余签名 LR、full geometry 与 KL arm。"""

        identity = super()._checkpoint_identity()
        spec = self._spec()
        factor_tuple = _canonical_factor_tuple(self.ARM)
        identity.update({
            "dqc_independent_factorial_implementation_identity": (
                spec["implementation"]),
            "dqc_independent_factorial_factor_tuple": factor_tuple,
            "dqc_c1_no_reward_outer_implementation_identity": (
                spec["implementation"]),
            "dqc_c1_no_reward_outer_factor_tuple": factor_tuple,
            "dqc_formal_b25_algorithm": spec["algorithm"],
            "dqc_formal_b25_implementation_identity": spec["implementation"],
            "dqc_formal_b25_factor_tuple": factor_tuple,
            "dqc_formal_b25_raw_layout": "T1000_B25",
            "dqc_formal_b25_actor_layout": DQC_LR2E4_M1_ACTOR_LAYOUT,
            "dqc_formal_b25_actor_bptt_length": (
                DQC_LR2E4_M1_ACTOR_BPTT_LENGTH),
            "dqc_formal_b25_actor_chunk_length": (
                DQC_LR2E4_M1_ACTOR_CHUNK_LENGTH),
            "dqc_formal_b25_actor_chunks": DQC_LR2E4_M1_ACTOR_CHUNKS,
            "dqc_formal_b25_actor_minibatch_size": (
                DQC_LR2E4_M1_ACTOR_MINIBATCH_SIZE),
            "dqc_formal_b25_actor_minibatches_per_epoch": 1,
            "dqc_formal_b25_actor_chunks_per_minibatch": (
                DQC_LR2E4_M1_ACTOR_CHUNKS_PER_MINIBATCH),
            "dqc_formal_b25_effective_actor_minibatch_size": (
                DQC_LR2E4_M1_EFFECTIVE_ACTOR_MINIBATCH_SIZE),
            "dqc_formal_b25_actor_chunk_counts_per_minibatch": (
                DQC_LR2E4_M1_ACTOR_CHUNKS_PER_MINIBATCH,),
            "dqc_formal_b25_actor_effective_minibatch_sizes": (
                DQC_LR2E4_M1_EFFECTIVE_ACTOR_MINIBATCH_SIZE,),
            "dqc_formal_b25_actor_shuffle_protocol": (
                DQC_LR2E4_M1_ACTOR_SHUFFLE_PROTOCOL),
            "dqc_formal_b25_actor_shuffle_seed": (
                DQC_LR2E4_M1_ACTOR_SHUFFLE_SEED),
            "dqc_target_kl": spec["target_kl"],
            "dqc_formal_b25_actor_target_kl": spec["target_kl"],
            "dqc_formal_b25_actor_epoch_protocol": spec["epoch_protocol"],
            "dqc_formal_b25_minimum_actor_optimizer_steps": (
                spec["minimum_steps"]),
            "dqc_formal_b25_expected_actor_optimizer_steps": (
                DQC_LR2E4_M1_EXPECTED_ACTOR_OPTIMIZER_STEPS),
            "dqc_formal_b25_maximum_actor_optimizer_steps": (
                DQC_LR2E4_M1_EXPECTED_ACTOR_OPTIMIZER_STEPS),
            "dqc_formal_b25_actor_lr0": DQC_LR2E4_M1_ACTOR_LR,
            "dqc_formal_b25_actor_lr_b": DQC_LR2E4_M1_ACTOR_LR_B,
            "dqc_formal_b25_actor_lr_c": DQC_LR2E4_M1_ACTOR_LR_C,
            "dqc_formal_b25_actor_theta_a": DQC_LR2E4_M1_THETA_A,
            "dqc_formal_b25_critic_lr": 3e-4,
            "dqc_formal_b25_value_lr": 3e-4,
            "dqc_formal_b25_lr2e4_m1_arm": self.ARM,
            "dqc_formal_b25_lr2e4_m1_implementation_identity": (
                spec["implementation"]),
            "dqc_formal_b25_lr2e4_m1_factor_tuple": factor_tuple,
            "dqc_formal_b25_lr2e4_m1_actor_lr": DQC_LR2E4_M1_ACTOR_LR,
            "dqc_formal_b25_lr2e4_m1_target_kl": spec["target_kl"],
        })
        return identity

    def validate_checkpoint_state(self, state):
        """先拒绝 cross-arm/LR/step 漂移，再进入共享 C1 tensor/loadability 校验。"""

        if not isinstance(state, dict):
            raise ValueError("DQC formal B25 LR2e-4/M1 checkpoint must be a dictionary")
        spec = self._spec()
        factor_tuple = _canonical_factor_tuple(self.ARM)

        # 独立 arm 证据在任何临时模型/optimizer load 之前比较。
        early_identity = {
            "dqc_formal_b25_lr2e4_m1_arm": self.ARM,
            "dqc_formal_b25_lr2e4_m1_implementation_identity": (
                spec["implementation"]),
            "dqc_formal_b25_lr2e4_m1_factor_tuple": factor_tuple,
        }
        differences = [
            f"{name}: checkpoint={state.get(name, '<missing>')!r}, required={value!r}"
            for name, value in early_identity.items()
            if state.get(name, "<missing>") != value
        ]
        if differences:
            raise ValueError(
                "DQC formal B25 LR2e-4/M1 checkpoint arm identity mismatch: "
                + "; ".join(differences))

        # 父 formal checkpoint/load 仍保存一个未使用的 shuffle Generator；验证其 payload
        # 仅为保持事务式恢复格式，不代表 full actor 会消费这个 RNG。
        rng_state = state.get("dqc_formal_b25_actor_shuffle_rng_state")
        if rng_state is None:
            raise ValueError(
                "DQC formal B25 LR2e-4/M1 checkpoint lacks formal RNG state")
        candidate = torch.Generator(device="cpu")
        try:
            candidate.set_state(torch.as_tensor(rng_state, device="cpu"))
        except (RuntimeError, TypeError, ValueError) as error:
            raise ValueError(
                "DQC formal B25 LR2e-4/M1 checkpoint has invalid formal RNG state"
            ) from error

        applied_epochs = state.get("formal_b25_actor_epochs_applied_total")
        actor_steps = state.get("actor_optimizer_steps")
        if (
                isinstance(applied_epochs, bool)
                or not isinstance(applied_epochs, Integral)
                or not 0 <= int(applied_epochs) <= (
                    self.num_iterations * FORMAL_B25_MAIN_EPOCHS)):
            raise ValueError("DQC LR2e-4/M1 checkpoint has invalid applied epochs")
        if (
                isinstance(actor_steps, bool)
                or not isinstance(actor_steps, Integral)
                or int(actor_steps) != int(applied_epochs)):
            raise ValueError(
                "DQC LR2e-4/M1 actor steps must equal applied full-batch epochs")

        # 当前 group.lr 会按 rollout clock 衰减；不可变 LR 证据是 initial_lr/base_lrs。
        optimizer_state = state.get("actor_optimizer")
        parameter_groups = (
            optimizer_state.get("param_groups", [])
            if isinstance(optimizer_state, dict) else [])
        if not parameter_groups or any(
                not math.isclose(
                    float(group.get("initial_lr", float("nan"))),
                    DQC_LR2E4_M1_ACTOR_LR,
                    rel_tol=0.0,
                    abs_tol=1e-15)
                for group in parameter_groups):
            raise ValueError(
                "DQC LR2e-4/M1 checkpoint has wrong actor initial LR")
        scheduler_state = state.get("actor_scheduler")
        scheduler_base_lrs = (
            scheduler_state.get("base_lrs", [])
            if isinstance(scheduler_state, dict) else [])
        if (
                len(scheduler_base_lrs) != len(parameter_groups)
                or any(not math.isclose(
                    float(value), DQC_LR2E4_M1_ACTOR_LR,
                    rel_tol=0.0, abs_tol=1e-15)
                    for value in scheduler_base_lrs)):
            raise ValueError(
                "DQC LR2e-4/M1 checkpoint has wrong scheduler base LR")

        completed_rollouts = state.get("dual_update_count")
        early_stops = state.get("actor_kl_early_stop_count")
        if (
                isinstance(completed_rollouts, bool)
                or not isinstance(completed_rollouts, Integral)
                or not 0 <= int(completed_rollouts) <= self.num_iterations):
            raise ValueError("DQC LR2e-4/M1 has invalid completed rollout count")
        if (
                isinstance(early_stops, bool)
                or not isinstance(early_stops, Integral)
                or not 0 <= int(early_stops) <= int(completed_rollouts)):
            raise ValueError("DQC LR2e-4/M1 has invalid KL early-stop count")

        if self.ARM == "nokl":
            # target=0 必须执行 R×E8，且任何 early-stop count 都代表身份污染。
            if int(early_stops) != 0 or int(applied_epochs) != (
                    FORMAL_B25_MAIN_EPOCHS * int(completed_rollouts)):
                raise ValueError(
                    "DQC LR2e-4/M1 noKL checkpoint violates fixed E8 protocol")
        else:
            # 每个 counted stop 跳过 1..7 轮；未 stop rollout 必须完整 E8。
            minimum_epochs = (
                FORMAL_B25_MAIN_EPOCHS * int(completed_rollouts)
                - (FORMAL_B25_MAIN_EPOCHS - 1) * int(early_stops))
            maximum_epochs = (
                FORMAL_B25_MAIN_EPOCHS * int(completed_rollouts)
                - int(early_stops))
            if not minimum_epochs <= int(applied_epochs) <= maximum_epochs:
                raise ValueError(
                    "DQC LR2e-4/M1 KL checkpoint conflicts with epoch stops")

        # 跳过 formal M25 写死的 steps=epochs×25；更底层的 identity、tensor、
        # optimizer moment、scheduler 和 vec-env RNG 校验仍完整执行。
        DQCACC1NoRewardOuterGPU.validate_checkpoint_state(self, state)

    def get_training_summary(self):
        """发布 M1 actor、M25 critics、动态 KL/fixed-E8 与 10M 联合完成证据。"""

        summary = super().get_training_summary()
        spec = self._spec()
        actual_actor_steps = int(summary.get("actor_optimizer_steps", 0))
        early_stop_count = int(summary.get("actor_kl_early_stop_count", 0))
        completed_rollouts = int(self.dual_update_count)
        applied_epochs = int(self.formal_b25_actor_epochs_applied_total)
        total_requested_epochs = FORMAL_B25_MAIN_EPOCHS * completed_rollouts
        skipped_epochs = total_requested_epochs - applied_epochs

        if self.ARM == "nokl":
            actor_protocol_consistent = bool(
                0 <= completed_rollouts <= FORMAL_B25_MAIN_BATCHES
                and early_stop_count == 0
                and applied_epochs == total_requested_epochs
                and actual_actor_steps == applied_epochs)
        else:
            minimum_epochs = (
                total_requested_epochs
                - (FORMAL_B25_MAIN_EPOCHS - 1) * early_stop_count)
            maximum_epochs = total_requested_epochs - early_stop_count
            actor_protocol_consistent = bool(
                0 <= completed_rollouts <= FORMAL_B25_MAIN_BATCHES
                and 0 <= early_stop_count <= completed_rollouts
                and minimum_epochs <= applied_epochs <= maximum_epochs
                and skipped_epochs >= 0
                and actual_actor_steps == applied_epochs)
        actor_protocol_complete = bool(
            actor_protocol_consistent
            and completed_rollouts == FORMAL_B25_MAIN_BATCHES)

        summary.update({
            "algorithm_identity": spec["implementation"],
            "registered_algorithm": spec["algorithm"],
            "raw_rollout_layout": "T1000_B25",
            "actor_sequence_layout": DQC_LR2E4_M1_ACTOR_LAYOUT,
            "actor_bptt_length": DQC_LR2E4_M1_ACTOR_BPTT_LENGTH,
            "actor_rechunking_enabled": False,
            "actor_chunk_length": DQC_LR2E4_M1_ACTOR_CHUNK_LENGTH,
            "actor_chunks_per_rollout": DQC_LR2E4_M1_ACTOR_CHUNKS,
            "actor_optimizer_initial_lr": DQC_LR2E4_M1_ACTOR_LR,
            "actor_lr_schedule_b": DQC_LR2E4_M1_ACTOR_LR_B,
            "actor_lr_schedule_c": DQC_LR2E4_M1_ACTOR_LR_C,
            "actor_minibatch_size_nominal": DQC_LR2E4_M1_ACTOR_MINIBATCH_SIZE,
            "actor_minibatches_per_epoch": DQC_LR2E4_M1_ACTOR_MINIBATCHES,
            "actor_chunks_per_minibatch": (
                DQC_LR2E4_M1_ACTOR_CHUNKS_PER_MINIBATCH),
            "actor_minibatch_size_effective": (
                DQC_LR2E4_M1_EFFECTIVE_ACTOR_MINIBATCH_SIZE),
            "actor_shuffle_protocol": DQC_LR2E4_M1_ACTOR_SHUFFLE_PROTOCOL,
            "actor_shuffle_seed": DQC_LR2E4_M1_ACTOR_SHUFFLE_SEED,
            "actor_shuffle_rng_isolated": False,
            "actor_epoch_protocol": spec["epoch_protocol"],
            "actor_target_kl": spec["target_kl"],
            "actor_kl_diagnostics_enabled": True,
            "actor_kl_early_stop_enabled": self.ARM == "kl",
            "minimum_actor_optimizer_steps": spec["minimum_steps"],
            "maximum_actor_optimizer_steps": (
                DQC_LR2E4_M1_EXPECTED_ACTOR_OPTIMIZER_STEPS),
            "expected_actor_optimizer_steps": (
                DQC_LR2E4_M1_EXPECTED_ACTOR_OPTIMIZER_STEPS),
            "actor_epochs_applied_total": applied_epochs,
            "actor_epochs_skipped_total": skipped_epochs,
            "actor_target_kl_protocol_consistent": actor_protocol_consistent,
            "full_actor_target_kl_protocol_complete": actor_protocol_complete,
            "full_actor_epoch_execution_complete": actor_protocol_complete,
            # 明确证明两个 critic 没有随 actor full-batch 改为 M1。
            "critic_optimizer_lr": 3e-4,
            "value_optimizer_lr": 3e-4,
            "critic_minibatch_size_nominal": FORMAL_B25_CRITIC_MINIBATCH_SIZE,
            "critic_minibatches_per_epoch": FORMAL_B25_CRITIC_MINIBATCHES,
            "critic_minibatch_size_effective": (
                FORMAL_B25_EFFECTIVE_CRITIC_MINIBATCH_SIZE),
            "expected_critic_main_steps_per_critic": (
                FORMAL_B25_EXPECTED_CRITIC_MAIN_STEPS),
            "expected_critic_warmup_steps_per_critic": (
                FORMAL_B25_EXPECTED_CRITIC_WARMUP_STEPS),
            "expected_critic_total_steps_per_critic": (
                FORMAL_B25_EXPECTED_CRITIC_TOTAL_STEPS),
        })
        summary["full_formal_optimizer_budget_complete"] = bool(
            actor_protocol_complete
            and summary.get("full_quantile_optimizer_budget_complete", False)
            and summary.get("full_value_optimizer_budget_complete", False))
        return summary


class DQCACFormalB25LR2E4M1KLGPU(_DQCACFormalB25LR2E4M1GPU):
    """公开 target-KL=0.01、full-batch M1、policy-LR2e-4 agent。"""

    ARM = "kl"


class DQCACFormalB25LR2E4M1NoKLGPU(_DQCACFormalB25LR2E4M1GPU):
    """公开 target-KL=0、固定 E8、full-batch M1、policy-LR2e-4 agent。"""

    ARM = "nokl"


# 全大写 NOKL 别名降低 runner 集成时 acronym 大小写造成的脆弱性。
DQCACFormalB25LR2E4M1NOKLGPU = DQCACFormalB25LR2E4M1NoKLGPU


__all__ = [
    "DQCACFormalB25LR2E4M1KLGPU",
    "DQCACFormalB25LR2E4M1NoKLGPU",
    "DQCACFormalB25LR2E4M1NOKLGPU",
    "DQC_LR2E4_M1_ACTOR_BPTT_LENGTH",
    "DQC_LR2E4_M1_ACTOR_CHUNKS",
    "DQC_LR2E4_M1_ACTOR_CHUNKS_PER_MINIBATCH",
    "DQC_LR2E4_M1_ACTOR_LAYOUT",
    "DQC_LR2E4_M1_ACTOR_LR",
    "DQC_LR2E4_M1_ACTOR_MINIBATCHES",
    "DQC_LR2E4_M1_ACTOR_MINIBATCH_SIZE",
    "DQC_LR2E4_M1_EFFECTIVE_ACTOR_MINIBATCH_SIZE",
    "DQC_LR2E4_M1_EXPECTED_ACTOR_OPTIMIZER_STEPS",
    "DQC_LR2E4_M1_KL_ACTOR_EPOCH_PROTOCOL",
    "DQC_LR2E4_M1_KL_ALGO",
    "DQC_LR2E4_M1_KL_IMPLEMENTATION_ID",
    "DQC_LR2E4_M1_KL_MINIMUM_ACTOR_OPTIMIZER_STEPS",
    "DQC_LR2E4_M1_KL_TARGET",
    "DQC_LR2E4_M1_NOKL_ACTOR_EPOCH_PROTOCOL",
    "DQC_LR2E4_M1_NOKL_ALGO",
    "DQC_LR2E4_M1_NOKL_IMPLEMENTATION_ID",
    "DQC_LR2E4_M1_NOKL_MINIMUM_ACTOR_OPTIMIZER_STEPS",
    "DQC_LR2E4_M1_NOKL_TARGET",
    "DQC_LR2E4_M1_THETA_A",
    "apply_dqc_formal_b25_lr2e4_m1_kl_configuration",
    "apply_dqc_formal_b25_lr2e4_m1_nokl_configuration",
    "canonical_dqc_lr2e4_m1_kl_factor_tuple",
    "canonical_dqc_lr2e4_m1_nokl_factor_tuple",
    "dqc_lr2e4_m1_kl_factor_tuple",
    "dqc_lr2e4_m1_nokl_factor_tuple",
]
