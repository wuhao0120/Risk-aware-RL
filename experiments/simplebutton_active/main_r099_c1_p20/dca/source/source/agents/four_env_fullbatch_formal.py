# -*- coding: utf-8 -*-
"""Frozen four-task B25/full-batch comparison agents.

This module is the checkpoint-incompatible outer envelope for the paper matrix
requested on 2026-09-04.  It combines three already-audited actor implementations
with one B25 adaptation of the DQC rolling-latest-100 multiplier used by the prior
B12 comparison.  QRefAA uses the pre-minibatch author-aligned B25 implementation.

The environment adapter changes only task identity, the preregistered raw-return
threshold, and simulator/W&B provenance.  The DQC arms differ only in actor LR and
target-KL; critics, rollout geometry, physical budget, PPO clip, and dual estimator
are shared.  Every extra field is serialized so old current-B25 checkpoints fail
before any live model, optimizer, deque, or worker RNG state is mutated.
"""
from __future__ import annotations

import copy                                  # Build validator-only Gremlin projections.
from collections import deque               # Own the exact latest-100 return sample.
from dataclasses import dataclass            # Freeze task and arm registries.
import json                                  # Stable checkpoint factor serialization.
import math                                  # Validate finite probabilities/thresholds.
from numbers import Integral, Real           # Reject bool while accepting numeric scalars.
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np                           # Reproduce float32 threshold comparisons.
import torch                                 # Training tensors and no-grad dual update.

from .dqc_ac_formal_b25 import (
    FORMAL_B25_MAIN_BATCHES,
    FORMAL_B25_NUM_ENVS,
    FORMAL_B25_TOTAL_ROLLOUT_BATCHES,
    FORMAL_B25_TRANSITIONS,
    FORMAL_B25_WARMUP_BATCHES,
)
from .dqc_ac_formal_b25_lr2e4_m1 import (
    DQCACFormalB25LR2E4M1KLGPU,
    DQCACFormalB25LR2E4M1NoKLGPU,
    apply_dqc_formal_b25_lr2e4_m1_kl_configuration,
    apply_dqc_formal_b25_lr2e4_m1_nokl_configuration,
)
from .qcpo_ref_author_b25_fullbatch import (
    QCPORefAuthorB25GPU as QCPORefAuthorB25FullBatchBaseGPU,
    apply_qcpo_ref_author_b25_configuration as apply_qref_b25_fullbatch_base,
)


# =============================================================================
# Frozen public matrix identity
# =============================================================================

DQC_LR2_KL_ALGO = "DQCAC_FOUR_ENV_B25_M1_LR2E4_KL_ROLL100"
DQC_LR1_KL_ALGO = "DQCAC_FOUR_ENV_B25_M1_LR1E4_KL_ROLL100"
DQC_LR2_NOKL_ALGO = "DQCAC_FOUR_ENV_B25_M1_LR2E4_NOKL_ROLL100"
QREF_FULL_ALGO = "QCPO_REF_FOUR_ENV_B25_M1_LR1E4_NOKL"

FOUR_ENV_ALGORITHMS = (
    DQC_LR2_KL_ALGO,
    QREF_FULL_ALGO,
)
FOUR_ENV_DQC_ALGORITHMS = (DQC_LR2_KL_ALGO,)

DUAL_WINDOW_SIZE = 100
DUAL_WINDOW_SOURCE = "latest_100_completed_discounted_reward_returns_v1"
DUAL_WARMUP_PROTOCOL = (
    "warmup20_append25_after_success_keep_latest100_no_lambda_step_v1"
)
DUAL_UPDATE_PROTOCOL = (
    "post_actor_append25_latest100_single_projected_step_per_main_rollout_v1"
)
DUAL_FAILURE_RULE = "inclusive_discounted_reward_return_G_le_q_v1"
EXPECTED_DUAL_UPDATES = FORMAL_B25_MAIN_BATCHES
BASE_DUAL_REPLACEMENT = (
    "private_current_b25_validator_scaffold_replaced_by_outer_rolling100_v1"
)


@dataclass(frozen=True)
class TaskSpec:
    """Task-only factors around the shared algorithm implementations."""

    environment: str
    env_id: str
    slug: str
    return_threshold: float
    threshold_source: str
    wandb_project: str
    execution_profile: str
    evaluation_protocol: str


TASKS = {
    'SimpleButton': TaskSpec('SimpleButton', 'SafetyPointSimpleButton0-v0', 'simplebutton', 2.6, 'user_requested_simplebutton_q2p6_20260923_v1', 'SimpleButton_full_gpu', 'simplebutton_mjwarp312_implicitfast_train_cpu312_final_eval_v1', 'simplebutton_mjwarp312_implicitfast_validation_B25_crn555_cpu312_implicitfast_final_B25_crn777_v1'),
}


@dataclass(frozen=True)
class ArmSpec:
    """Public arm identity plus its private audited base configurator."""

    algorithm: str
    slug: str
    implementation: str
    actor_lr: float
    target_kl: float
    base_configurer: object


ARMS = {
    DQC_LR2_KL_ALGO: ArmSpec(DQC_LR2_KL_ALGO, 'dqc-b25-m1-lr2e4-kl-roll100', 'dqc_b25_full_m1_lr2e4_kl0p01_rolling100_fourenv_v1', 0.0002, 0.01, apply_dqc_formal_b25_lr2e4_m1_kl_configuration),
    QREF_FULL_ALGO: ArmSpec(QREF_FULL_ALGO, 'qrefaa-b25-m1-lr1e4-nokl', 'qrefaa_author_aligned_b25_full_m1_lr1e4_nokl_fourenv_v1', 0.0001, 0.0, apply_qref_b25_fullbatch_base),
}


def task_spec(environment: str) -> TaskSpec:
    """Resolve an explicit short task name; never fall back to Gremlin."""

    try:
        return TASKS[str(environment)]
    except KeyError as error:
        raise ValueError(f"unsupported four-env task {environment!r}") from error


def arm_spec(algorithm: str) -> ArmSpec:
    """Resolve an explicit arm name; typos cannot silently select a baseline."""

    try:
        return ARMS[str(algorithm)]
    except KeyError as error:
        raise ValueError(f"unsupported four-env arm {algorithm!r}") from error


def canonical_outer_factor(environment: str, algorithm: str) -> str:
    """Serialize all factors that differ outside the audited private base."""

    task = task_spec(environment)
    arm = arm_spec(algorithm)
    payload = (
        ("schema", "four_env_b25_fullbatch_outer_v1"),
        ("environment", task.environment),
        ("env_id", task.env_id),
        ("return_threshold", task.return_threshold),
        ("threshold_source", task.threshold_source),
        ("algorithm", arm.algorithm),
        ("implementation", arm.implementation),
        ("actor_lr", arm.actor_lr),
        ("actor_minibatches_per_epoch", 1),
        ("actor_epochs", 8),
        ("actor_target_kl", arm.target_kl),
        ("raw_rollout", "T1000_B12"),
        ("physical_training_steps", 10_008_000),
        ("dual_window_size", DUAL_WINDOW_SIZE if algorithm in FOUR_ENV_DQC_ALGORITHMS else 100),
        ("dual_protocol", DUAL_UPDATE_PROTOCOL if algorithm in FOUR_ENV_DQC_ALGORITHMS
         else "qrefaa_author_latest100_order_statistic_first_batch_rms_dual_only_v1"),
        ("base_dual_replaced_by_outer",
         algorithm in FOUR_ENV_DQC_ALGORITHMS),
    )
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def _install_task_and_remote_identity(args, public_algorithm: str) -> None:
    """Install task/q plus collision-free defaults; the manifest may refine IDs."""

    task = task_spec(args.env_name)
    arm = arm_spec(public_algorithm)
    args.algo_name = public_algorithm
    args.return_threshold = task.return_threshold
    args.formal_environment = task.environment
    args.formal_threshold_source = task.threshold_source
    args.formal_execution_profile = task.execution_profile
    args.formal_evaluation_protocol = task.evaluation_protocol
    args.wandb_project = task.wandb_project
    args.wandb_step_metric = "progress/actual_env_steps"
    args.defer_wandb_init = True
    args.four_env_implementation_identity = arm.implementation
    args.four_env_factor_tuple = canonical_outer_factor(task.environment, public_algorithm)
    args.wandb_group = "formal-four-env-b25-fullbatch-10seed-v1"
    args.wandb_name = f"{arm.slug}-{task.slug}-s{int(args.seed)}-physical10m-v1"
    args.wandb_lineage = args.wandb_name
    args.experiment_variant = args.wandb_name

    # Tags are diagnostics, not identity; remove stale minibatch labels inherited
    # from the generic namespace before recording the real full-batch protocol.
    stale = {"actor-minibatch-1024-effective-1000", "actor-M25", "qrefaa-M25"}
    tags = [item.strip() for item in str(getattr(args, "wandb_tags", "")).split(",")
            if item.strip() and item.strip() not in stale]
    for item in (task.slug, "formal-four-env-v1", "actor-full-batch-M1",
                 "raw-T1000-B25", "physical-10m"):
        if item not in tags:
            tags.append(item)
    args.wandb_tags = ",".join(tags)


def _install_rolling_fields(args) -> None:
    """Add redundant public fields checked before DQC model allocation."""

    args.dual_window_size = DUAL_WINDOW_SIZE
    args.dual_window_source = DUAL_WINDOW_SOURCE
    args.dual_warmup_protocol = DUAL_WARMUP_PROTOCOL
    args.dual_update_protocol = DUAL_UPDATE_PROTOCOL
    args.dual_failure_rule = DUAL_FAILURE_RULE
    args.expected_dual_updates = EXPECTED_DUAL_UPDATES
    args.four_env_base_dual_replaced_by_outer = True
    args.four_env_base_dual_replacement = BASE_DUAL_REPLACEMENT


def _strip_private_factor_tuples(args) -> None:
    """Keep Gremlin-validator tuples out of public W&B/scientific signatures.

    The copied namespace constructed by the outer envelope reinstalls these
    tuples solely for inherited fail-closed validation. Public provenance is
    the task-neutral individual optimizer fields plus four_env_factor_tuple.
    """

    for field in (
            "dqc_independent_factorial_factor_tuple",
            "dqc_c1_no_reward_outer_factor_tuple",
            "dqc_formal_b25_factor_tuple",
            "dqc_formal_b25_lr1e4_m1_factor_tuple",
            "dqc_formal_b25_lr2e4_m1_factor_tuple",
            "qcpo_b25_factor_tuple"):
        if hasattr(args, field):
            delattr(args, field)
    args.four_env_private_base_factor_metadata_projected_by_outer = True
    args.four_env_private_base_factor_scope = (
        "private_gremlin_validation_metadata_only_actor_critic_optimizer_core_"
        "executed_task_and_dqc_dual_defined_by_outer_v1"
    )
    if not hasattr(args, "four_env_base_dual_replaced_by_outer"):
        args.four_env_base_dual_replaced_by_outer = False
        args.four_env_base_dual_replacement = "none"


def apply_four_env_configuration(args, algorithm: str):
    """Build one exact arm and task namespace before runner/environment setup."""

    public_algorithm = str(algorithm)
    arm = arm_spec(public_algorithm)
    task_spec(args.env_name)                 # Fail before any remote/environment side effect.
    arm.base_configurer(args)               # Install the audited private optimizer implementation.
    if public_algorithm in FOUR_ENV_DQC_ALGORITHMS:
        _install_rolling_fields(args)
    _strip_private_factor_tuples(args)
    _install_task_and_remote_identity(args, public_algorithm)
    return args


# =============================================================================
# B25 rolling-latest-100 DQC mixin
# =============================================================================

class DQCACB25Rolling100Mixin:
    """Replace rollout-local B25 dual updates with a durable latest-100 sample."""

    def __init__(self, args, env):
        """Validate protocol fields, allocate the base agent, then create the deque."""

        required = {
            "dual_window_size": (int(getattr(args, "dual_window_size", -1)), 100),
            "dual_window_source": (getattr(args, "dual_window_source", None), DUAL_WINDOW_SOURCE),
            "dual_warmup_protocol": (getattr(args, "dual_warmup_protocol", None), DUAL_WARMUP_PROTOCOL),
            "dual_update_protocol": (getattr(args, "dual_update_protocol", None), DUAL_UPDATE_PROTOCOL),
            "dual_failure_rule": (getattr(args, "dual_failure_rule", None), DUAL_FAILURE_RULE),
            "expected_dual_updates": (int(getattr(args, "expected_dual_updates", -1)), 814),
        }
        mismatches = [f"{name}={actual!r} (required {wanted!r})"
                      for name, (actual, wanted) in required.items() if actual != wanted]
        if mismatches:
            raise ValueError("B25 rolling100 identity mismatch: " + ", ".join(mismatches))
        super().__init__(args, env)

        # Python floats are exact copies of completed float32 returns.  The deque
        # owns no device tensor and consumes no Torch/NumPy/environment RNG state.
        self.dual_window_size = DUAL_WINDOW_SIZE
        self.dual_return_window = deque(maxlen=DUAL_WINDOW_SIZE)
        self.dual_total_observations = 0
        self.last_dual_prob = 0.0
        self.last_dual_window_probability = 0.0
        self.last_batch_failure_probability = 0.0

    def _validated_completed_returns(self, returns: torch.Tensor) -> Tuple[float, ...]:
        """Copy exactly 25 finite completed returns to immutable host scalars."""

        if not torch.is_tensor(returns):
            raise TypeError("B25 rolling100 returns must be a tensor")
        flat = returns.detach().cpu().reshape(-1)
        if flat.numel() != FORMAL_B25_NUM_ENVS:
            raise ValueError("B25 rolling100 requires exactly 25 completed returns")
        if not bool(torch.isfinite(flat).all()):
            raise ValueError("B25 rolling100 returns contain NaN/Inf")
        return tuple(float(value) for value in flat.tolist())

    def _empirical_failure_probability(self, values: Iterable[float]) -> float:
        """Evaluate inclusive G<=q in the training return's float32 precision."""

        array = np.asarray(tuple(values), dtype=np.float32).reshape(-1)
        if array.size == 0:
            return 0.0
        return float(np.mean(array <= np.float32(self.return_threshold)))

    def _append_completed_returns(self, values: Iterable[float]) -> float:
        """Append one B25 rollout and return the probability over latest 100."""

        values = tuple(float(value) for value in values)
        if len(values) != FORMAL_B25_NUM_ENVS:
            raise ValueError("B25 rolling100 append requires one complete B25 rollout")
        self.dual_return_window.extend(values)
        self.dual_total_observations += len(values)
        probability = self._empirical_failure_probability(self.dual_return_window)
        self.last_dual_prob = probability
        self.last_dual_window_probability = probability
        return probability

    @torch.no_grad()
    def update_dual(self, returns):
        """After actor update, append B25 and take one latest-100 projected step."""

        if len(self.dual_return_window) != DUAL_WINDOW_SIZE:
            raise RuntimeError("main dual update requires a warmup-prefilled 100-return window")
        probability = self._append_completed_returns(
            self._validated_completed_returns(returns))
        if len(self.dual_return_window) != DUAL_WINDOW_SIZE:
            raise RuntimeError("rolling dual lost its fixed 100-return denominator")
        residual = probability - self.failure_alpha
        step = float(self.dual_step_size())
        self.lambda_probability.add_(step * residual).clamp_(0.0, self.lambda_max)
        self.dual_update_count += 1
        return probability, float(residual), step

    def _update_rollout(self, iteration, rollout, initialization=False):
        """Prefill only after successful warmup critic updates; delegate main order."""

        if not initialization:
            # Parent main transaction is actor -> dual -> schedulers/checkpoint row.
            # Dynamic dispatch reaches this mixin's update_dual exactly once.
            return super()._update_rollout(iteration, rollout, initialization=False)
        values = self._validated_completed_returns(
            rollout["diagnostic_raw_cost"] if getattr(self, "raw_cost_constraint", False)
            else rollout["disc_return"])
        info = super()._update_rollout(iteration, rollout, initialization=True)
        self._append_completed_returns(values)
        return info

    def _log_core(self, iteration, returns, diagnostic_disc_cost,
                  diagnostic_raw_cost, extra=None):
        """Keep B25 noise diagnostics separate from the actual dual statistic."""

        batch_probability = self._empirical_failure_probability(returns)
        self.last_batch_failure_probability = batch_probability
        if len(self.dual_return_window) != DUAL_WINDOW_SIZE:
            raise RuntimeError("main logging requires a ready rolling100 window")
        dual_probability = float(self.last_dual_prob)
        additions = dict(extra or {})
        additions.update({
            "constraint/batch_failure_probability": batch_probability,
            "constraint/dual_window_failure_probability": dual_probability,
            "constraint/dual_window_residual":
                dual_probability - self.failure_alpha,
            "constraint/dual_window_feasible":
                float(dual_probability <= self.failure_alpha),
            "constraint/dual_window_count": 100.0,
            "constraint/dual_window_ready": 1.0,
            "lambda/dual_window_count": 100.0,
            "lambda/dual_update_applied": 1.0,
            "lambda/dual_updates": float(self.dual_update_count),
            "lambda/residual": dual_probability - self.failure_alpha,
        })
        metrics = super()._log_core(
            iteration, returns, diagnostic_disc_cost, diagnostic_raw_cost, extra=additions)
        return metrics

    def _checkpoint_identity(self) -> Dict[str, object]:
        """Append a checkpoint-incompatible rolling estimator identity."""

        identity = super()._checkpoint_identity()
        identity.update({
            "dqc_b25_dual_window_size": DUAL_WINDOW_SIZE,
            "dqc_b25_dual_window_source": DUAL_WINDOW_SOURCE,
            "dqc_b25_dual_warmup_protocol": DUAL_WARMUP_PROTOCOL,
            "dqc_b25_dual_update_protocol": DUAL_UPDATE_PROTOCOL,
            "dqc_b25_dual_failure_rule": DUAL_FAILURE_RULE,
            "dqc_b25_expected_dual_updates": EXPECTED_DUAL_UPDATES,
        })
        return identity

    def _validate_dual_checkpoint_state(self, state: Mapping[str, object]) -> None:
        """Validate deque content and redundant rollout clocks before mutation."""

        required = {"dual_window_size", "dual_return_window", "dual_total_observations",
                    "dual_window_ready", "dual_update_count", "last_dual_prob"}
        missing = sorted(required - set(state))
        if missing:
            raise ValueError("B25 rolling100 checkpoint missing: " + ", ".join(missing))
        if state["dual_window_size"] != DUAL_WINDOW_SIZE:
            raise ValueError("B25 rolling100 checkpoint window size mismatch")
        raw_window = state["dual_return_window"]
        if not isinstance(raw_window, (list, tuple)) or len(raw_window) > DUAL_WINDOW_SIZE:
            raise ValueError("B25 rolling100 checkpoint window is invalid")
        window = []
        for index, value in enumerate(raw_window):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                raise ValueError(f"B25 rolling100 return {index} is invalid")
            window.append(float(value))

        total = state["dual_total_observations"]
        if isinstance(total, bool) or not isinstance(total, Integral) or int(total) < 0:
            raise ValueError("B25 rolling100 observation clock is invalid")
        cursor = int(state.get("initialization_cursor", -1))
        initialization_steps = int(state.get("initialization_env_steps", -1))
        main_steps = int(state.get("main_env_steps", -1))
        if not 0 <= cursor <= FORMAL_B25_WARMUP_BATCHES:
            raise ValueError("B25 rolling100 warmup cursor is invalid")
        if initialization_steps != cursor * FORMAL_B25_TRANSITIONS:
            raise ValueError("B25 rolling100 warmup step clock mismatch")
        if main_steps < 0 or main_steps % FORMAL_B25_TRANSITIONS:
            raise ValueError("B25 rolling100 main step clock mismatch")
        main_batches = main_steps // FORMAL_B25_TRANSITIONS
        if main_batches > self.num_iterations:
            raise ValueError("B25 rolling100 exceeds registered main batches")
        if main_batches and cursor != FORMAL_B25_WARMUP_BATCHES:
            raise ValueError("B25 rolling100 main phase started before warmup")
        expected_total = (cursor + main_batches) * FORMAL_B25_NUM_ENVS
        expected_length = min(DUAL_WINDOW_SIZE, expected_total)
        if int(total) != expected_total or len(window) != expected_length:
            raise ValueError("B25 rolling100 deque/observation clock mismatch")
        if int(state["dual_update_count"]) != main_batches:
            raise ValueError("B25 rolling100 dual update clock mismatch")
        ready = state["dual_window_ready"]
        if not isinstance(ready, bool) or ready != (expected_length == DUAL_WINDOW_SIZE):
            raise ValueError("B25 rolling100 readiness mismatch")
        expected_probability = self._empirical_failure_probability(window)
        if not math.isclose(float(state["last_dual_prob"]), expected_probability,
                            rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("B25 rolling100 probability does not match deque")

    def validate_checkpoint_state(self, state):
        """Reject corrupt rolling state before delegating base tensor validation."""

        if not isinstance(state, dict):
            raise ValueError("B25 rolling100 checkpoint must be a dictionary")
        self._validate_dual_checkpoint_state(state)
        super().validate_checkpoint_state(state)

    def checkpoint_state(self):
        """Serialize the entire dual sample alongside model/optimizer/worker RNG."""

        state = super().checkpoint_state()
        state.update({
            "dual_window_size": DUAL_WINDOW_SIZE,
            "dual_return_window": list(self.dual_return_window),
            "dual_total_observations": int(self.dual_total_observations),
            "dual_window_ready": len(self.dual_return_window) == DUAL_WINDOW_SIZE,
            "dual_update_count": int(self.dual_update_count),
            "last_dual_prob": float(self.last_dual_prob),
        })
        return state

    def load_checkpoint_state(self, state):
        """Commit deque values only after the parent transactional restore succeeds."""

        self.validate_checkpoint_state(state)
        super().load_checkpoint_state(state)
        self.dual_return_window.clear()
        self.dual_return_window.extend(float(value) for value in state["dual_return_window"])
        self.dual_total_observations = int(state["dual_total_observations"])
        self.last_dual_prob = float(state["last_dual_prob"])
        self.last_dual_window_probability = self.last_dual_prob
        self.last_empirical_prob = self.last_dual_prob
        self.last_outage_prob = self.last_dual_prob

    def get_training_summary(self):
        """Expose explicit completion evidence for the rolling controller."""

        summary = super().get_training_summary()
        updates = int(self.dual_update_count)
        observations = int(self.dual_total_observations)
        rolling_complete = bool(
            updates == EXPECTED_DUAL_UPDATES
            and observations == FORMAL_B25_TOTAL_ROLLOUT_BATCHES * FORMAL_B25_NUM_ENVS
            and len(self.dual_return_window) == DUAL_WINDOW_SIZE)
        summary.update({
            "dual_window_size": DUAL_WINDOW_SIZE,
            "dual_window_source": DUAL_WINDOW_SOURCE,
            "dual_warmup_protocol": DUAL_WARMUP_PROTOCOL,
            "dual_update_protocol": DUAL_UPDATE_PROTOCOL,
            "dual_failure_rule": DUAL_FAILURE_RULE,
            "expected_dual_updates": EXPECTED_DUAL_UPDATES,
            "dual_update_count": updates,
            "dual_total_observations": observations,
            "dual_window_count": len(self.dual_return_window),
            "dual_window_probability": float(self.last_dual_prob),
            "full_dual_protocol_complete": rolling_complete,
        })
        summary["full_formal_training_protocol_complete"] = bool(
            summary.get("full_formal_optimizer_budget_complete", False)
            and rolling_complete)
        return summary


# =============================================================================
# Four-environment outer envelopes and concrete public classes
# =============================================================================

class _FourEnvEnvelopeMixin:
    """Project task metadata around one private Gremlin-validated base class."""

    PUBLIC_ALGORITHM = ""

    def __init__(self, args, env):
        """Validate public task identity, construct private base, restore live q."""


        task = task_spec(getattr(args, "env_name", ""))
        arm = arm_spec(self.PUBLIC_ALGORITHM)
        expected_factor = canonical_outer_factor(task.environment, self.PUBLIC_ALGORITHM)
        required = {
            "algorithm": (getattr(args, "algo_name", None), self.PUBLIC_ALGORITHM),
            "threshold": (
                float(getattr(args, "return_threshold", float("nan"))),
                task.return_threshold,
            ),
            "implementation": (
                getattr(args, "four_env_implementation_identity", None),
                arm.implementation,
            ),
            "factor_tuple": (
                getattr(args, "four_env_factor_tuple", None), expected_factor,
            ),
            "wandb_project": (
                getattr(args, "wandb_project", None), task.wandb_project,
            ),
        }
        mismatches = [
            f"{name}={actual!r} (required {wanted!r})"
            for name, (actual, wanted) in required.items()
            if actual != wanted
        ]
        public_env_id = str(getattr(args, "env_id", ""))
        if public_env_id not in (task.environment, task.env_id):
            mismatches.append(f"env_id={public_env_id!r}")
        if str(getattr(env, "env_id", "")) != task.env_id:
            mismatches.append(f"reference_env={getattr(env, 'env_id', None)!r}")
        if mismatches:
            raise ValueError(
                "four-env public identity mismatch: " + ", ".join(mismatches)
            )

        # Private classes retain their complete fail-closed validators.  Only a
        # copied namespace is projected; the caller's public args stay intact.
        projected = copy.copy(args)
        arm.base_configurer(projected)
        # All private base validators are the historical Gremlin q=1.85
        # implementation.  The public task threshold is restored immediately
        # after construction and signed by the outer checkpoint envelope.
        projected.return_threshold = 1.85
        if self.PUBLIC_ALGORITHM in FOUR_ENV_DQC_ALGORITHMS:
            _install_rolling_fields(projected)
        super().__init__(projected, env)

        # These assignments allocate no tensor and consume no RNG.  They are the
        # live task values used by reward tests, receipts, and W&B metadata.
        self.algo_name = self.PUBLIC_ALGORITHM
        self.return_threshold = task.return_threshold
        if hasattr(self, "cost_scale"):
            self.cost_threshold = -task.return_threshold
            self.cost_threshold_scaled = -task.return_threshold / self.cost_scale
        # The author QRef parent caches both threshold representations during
        # its private Gremlin validation construction. They remain diagnostic
        # until the first rollout updates the empirical order statistic, but
        # restoring them here also keeps initialization checkpoints and any
        # pre-rollout summary faithful to the selected public task.
        if hasattr(self, "last_dual_cost_quantile"):
            self.last_dual_cost_quantile = -task.return_threshold
        if hasattr(self, "last_dual_reward_quantile"):
            self.last_dual_reward_quantile = task.return_threshold
        self.formal_environment = task.environment
        self.formal_threshold_source = task.threshold_source
        self.formal_wandb_project = task.wandb_project
        self.formal_execution_profile = task.execution_profile
        self.formal_evaluation_protocol = task.evaluation_protocol
        self.four_env_factor_tuple = expected_factor
        # QRef caches this value during its private parent construction and
        # serializes it in checkpoint_identity. Restore the public stable ID so
        # no non-Gremlin checkpoint claims the validator-only Gremlin lineage.
        self.wandb_lineage = str(args.wandb_lineage)

    def _checkpoint_identity(self) -> Dict[str, object]:
        """Append outer task/arm evidence without weakening private validators."""

        identity = super()._checkpoint_identity()
        task = task_spec(self.env_name)
        arm = arm_spec(self.PUBLIC_ALGORITHM)
        identity.update({
            "four_env_schema": "four_env_b25_fullbatch_outer_v1",
            "four_env_environment": task.environment,
            "four_env_env_id": task.env_id,
            "four_env_return_threshold": task.return_threshold,
            "four_env_threshold_source": task.threshold_source,
            "four_env_algorithm": self.PUBLIC_ALGORITHM,
            "four_env_implementation_identity": arm.implementation,
            "four_env_factor_tuple": canonical_outer_factor(
                task.environment, self.PUBLIC_ALGORITHM
            ),
            "four_env_base_dual_replaced_by_outer": bool(
                self.PUBLIC_ALGORITHM in FOUR_ENV_DQC_ALGORITHMS
            ),
            "four_env_base_dual_replacement": (
                BASE_DUAL_REPLACEMENT
                if self.PUBLIC_ALGORITHM in FOUR_ENV_DQC_ALGORITHMS else "none"
            ),
            "four_env_private_base_factor_metadata_projected_by_outer": True,
            "wandb_project": task.wandb_project,
            "formal_execution_profile": task.execution_profile,
            "formal_evaluation_protocol": task.evaluation_protocol,
        })
        return identity

    def validate_checkpoint_state(self, state):
        """Reject cross-task/cross-arm checkpoints before private mutation."""

        task = task_spec(self.env_name)
        expected = {
            "four_env_schema": "four_env_b25_fullbatch_outer_v1",
            "four_env_environment": task.environment,
            "four_env_env_id": task.env_id,
            "four_env_return_threshold": task.return_threshold,
            "four_env_algorithm": self.PUBLIC_ALGORITHM,
            "four_env_implementation_identity": (
                arm_spec(self.PUBLIC_ALGORITHM).implementation
            ),
            "four_env_factor_tuple": canonical_outer_factor(
                task.environment, self.PUBLIC_ALGORITHM
            ),
            "four_env_base_dual_replaced_by_outer": bool(
                self.PUBLIC_ALGORITHM in FOUR_ENV_DQC_ALGORITHMS
            ),
            "four_env_private_base_factor_metadata_projected_by_outer": True,
        }
        # DQC spreads its identity into the state mapping, while the literal
        # author QRef format keeps the same payload under checkpoint_identity.
        # Read whichever native container the private class serializes.
        nested = state.get("checkpoint_identity")
        identity = (
            nested
            if isinstance(nested, Mapping) and "four_env_schema" in nested
            else state
        )
        mismatches = [
            name for name, value in expected.items() if identity.get(name) != value
        ]
        if mismatches:
            raise ValueError(
                "four-env checkpoint outer mismatch: " + ", ".join(mismatches)
            )
        super().validate_checkpoint_state(state)

    def get_training_summary(self):
        """Publish task/public-arm identity with private completion counters."""

        summary = super().get_training_summary()
        task = task_spec(self.env_name)
        arm = arm_spec(self.PUBLIC_ALGORITHM)
        summary.update({
            "algorithm_identity": arm.implementation,
            "registered_algorithm": self.PUBLIC_ALGORITHM,
            "formal_environment": task.environment,
            "return_threshold": task.return_threshold,
            "formal_threshold_source": task.threshold_source,
            "wandb_project": task.wandb_project,
            "formal_execution_profile": task.execution_profile,
            "formal_evaluation_protocol": task.evaluation_protocol,
            "four_env_factor_tuple": canonical_outer_factor(
                task.environment, self.PUBLIC_ALGORITHM
            ),
        })
        return summary


class DQCACFourEnvB25LR2E4M1KLRolling100GPU(
        _FourEnvEnvelopeMixin, DQCACB25Rolling100Mixin,
        DQCACFormalB25LR2E4M1KLGPU):
    """DQC B25/full, actor LR2e-4, target-KL .01, latest-100 dual."""

    PUBLIC_ALGORITHM = DQC_LR2_KL_ALGO






class QCPORefFourEnvB25FullBatchGPU(
        _FourEnvEnvelopeMixin, QCPORefAuthorB25FullBatchBaseGPU):
    """Author-aligned QRefAA B25/full/no-KL/shared-Adam LR1e-4."""

    PUBLIC_ALGORITHM = QREF_FULL_ALGO


FOUR_ENV_CLASSES = {
    DQC_LR2_KL_ALGO: DQCACFourEnvB25LR2E4M1KLRolling100GPU,
    QREF_FULL_ALGO: QCPORefFourEnvB25FullBatchGPU,
}


__all__ = [
    "ARMS", "TASKS", "FOUR_ENV_ALGORITHMS", "FOUR_ENV_CLASSES",
    "FOUR_ENV_DQC_ALGORITHMS", "DQC_LR2_KL_ALGO", "DQC_LR1_KL_ALGO",
    "DQC_LR2_NOKL_ALGO", "QREF_FULL_ALGO", "DQCACB25Rolling100Mixin",
    "apply_four_env_configuration", "arm_spec", "task_spec",
]
