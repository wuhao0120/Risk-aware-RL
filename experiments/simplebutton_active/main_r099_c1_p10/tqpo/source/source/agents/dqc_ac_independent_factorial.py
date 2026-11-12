# -*- coding: utf-8 -*-
"""Clean clipping factorial built on the original independent-network DQCAC.

The three registered arms preserve the historical finite-GAE DQC architecture
and training state:

* recurrent actor, scalar value critic, and action-conditioned quantile critic
  are independent modules with three disjoint Adam optimizers;
* reward credit is multiplied by gamma**t while finite-horizon CDF credit
  receives no outer discount;
* the scalar value target is frozen GAE, the distributional target is frozen
  n-step-100 QR, and there is no quantile-mean auxiliary loss.

Only PPO clipping changes. I0 is the historical combined-advantage surrogate.
C1 clips reward with min and the already-lambda-weighted constraint with max.
C2 is exactly C1 divided once by 1 + lambda.
"""

from __future__ import annotations

import copy
import json
from typing import Dict, Tuple

import numpy as np
import torch

from .dqc_ac_beta_gpu import DQCACBetaGPU


# ============================================================================
# Registered scientific identities
# ============================================================================

INDEPENDENT_FACTORIAL_ARMS = ("I0", "C1", "C2")
INDEPENDENT_FACTORIAL_IMPLEMENTATION_ID = (
    "dqc_original_independent_n100_gae_clipping_factorial_v1")
INDEPENDENT_FACTORIAL_TRAINING_BUDGET = "main_excludes_initialization_v1"
INDEPENDENT_FACTORIAL_ACTION_MODE = "explicit_box_clip"
COMMON_ADVANTAGE_SCALE = (
    "std_pop_valid_reward_gamma_t_minus_lambda_cdf_divide_only_no_center_v1")

_ARM_SPEC = {
    "I0": {
        "clip_mode": "combined_min_v1",
        "loss_normalization": "none",
        "description": "original-independent-combined-clipping",
    },
    "C1": {
        "clip_mode": "split_reward_min_constraint_max_v1",
        "loss_normalization": "none",
        "description": "independent-split-clipping-no-one-plus-lambda-division",
    },
    "C2": {
        "clip_mode": "split_reward_min_constraint_max_v1",
        "loss_normalization": "divide_entire_loss_by_one_plus_lambda_v1",
        "description": "independent-split-clipping-divide-by-one-plus-lambda",
    },
}

# This ordered tuple is the executable algorithm definition. The arm-specific
# clipping fields are appended by independent_factorial_factor_tuple().
_COMMON_FACTORS = (
    ("gamma", 0.99),
    ("actor_architecture", "fc512x2_lstm512_gaussian_learned_std_v1"),
    ("observation_normalization", "running_mean_std_v1"),
    ("dual", "empirical_failure_adam_free_a_over_b_plus_k_pow_c_v1"),
    ("dual_schedule", "lr0_0p01_b1000_c0p5_cap50_init0_v1"),
    ("representation", "independent_actor_value_quantile_networks_v1"),
    ("optimizer_layout", "three_disjoint_adam_actor_value_quantile_v1"),
    ("actor_optimizer", "adam_lr2e-4_eps1e-5_absolute_batch_schedule_v1"),
    ("value_optimizer", "adam_lr3e-4_eps1e-5_v1"),
    ("quantile_optimizer", "adam_lr3e-4_eps1e-5_v1"),
    ("actor_input", "observation_history_fc512x2_lstm512_v1"),
    (
        "critic_input",
        "observation_remaining_fraction_detached_actor_h0c0_executed_action_v1",
    ),
    ("reward_outer_discount", "gamma_t"),
    ("finite_risk_outer_discount", "none"),
    ("advantage_scale", COMMON_ADVANTAGE_SCALE),
    ("lambda_placement", "constraint_weight_once_before_clipping_v1"),
    ("actor_reduction", "fixed_num_envs_times_horizon_v1"),
    ("action_baseline_samples", 4),
    ("actor_reuse", "ppo_clip_e8_ratio0p1_target_kl0p01_v1"),
    ("value_target", "frozen_finite_gae_lambda0p97_v1"),
    ("quantile_target", "frozen_nstep100_hard_target_v1"),
    ("quantiles", "32_midpoints_huber_kappa0p1_unsorted_v1"),
    ("quantile_mean_auxiliary_loss", "none"),
    ("action_execution_mode", INDEPENDENT_FACTORIAL_ACTION_MODE),
    ("action_boundary", "raw_gaussian_likelihood_explicit_box_clipped_critic_v1"),
    ("warmup", "10_batches_value_and_quantile_e4_hard_target_no_actor_v1"),
)


def require_independent_factorial_arm(arm: str) -> str:
    """Return one registered arm or reject before model/W&B construction."""

    arm = str(arm)
    if arm not in _ARM_SPEC:
        raise ValueError(
            "DQCAC_INDEPENDENT_FACTORIAL requires independent_factorial_arm in "
            f"{INDEPENDENT_FACTORIAL_ARMS!r}, got {arm!r}")
    return arm


def independent_factorial_factor_tuple(arm: str) -> Tuple[Tuple[str, object], ...]:
    """Return the complete ordered factor tuple for one clipping arm."""

    arm = require_independent_factorial_arm(arm)
    spec = _ARM_SPEC[arm]
    return _COMMON_FACTORS + (
        ("policy_clip_mode", spec["clip_mode"]),
        ("actor_loss_normalization", spec["loss_normalization"]),
    )


def canonical_independent_factorial_factor_tuple(arm: str) -> str:
    """Serialize the factor tuple without whitespace or locale ambiguity."""

    return json.dumps(
        independent_factorial_factor_tuple(arm),
        separators=(",", ":"),
        ensure_ascii=True,
    )


def apply_independent_factorial_configuration(args, arm: str):
    """Install original independent-DQC constants plus one clipping arm.

    The helper overwrites every interpretation-changing inherited field. Thus
    programmatic callers and the CLI execute the same identity, and no stale
    no-outer/shared setting can leak into this campaign.
    """

    arm = require_independent_factorial_arm(arm)
    spec = _ARM_SPEC[arm]

    # Fix the finite task discount/horizon before any network field is read.
    args.gamma = 0.99
    args.horizon = 1000
    # The recurrent Gaussian policy is byte-compatible with historical DQC.
    args.actor_hidden = [512, 512]
    args.lstm_size = 512
    args.init_std = 1.0
    args.learn_std = True
    args.normalize_observation = True
    # DQC likelihood remains raw Gaussian, while every critic/target receives
    # the action actually executed after the explicit local Box projection.
    args.action_execution_mode = INDEPENDENT_FACTORIAL_ACTION_MODE
    # The empirical multiplier keeps its original momentum-free schedule.
    args.penalty_init = 0.0
    args.lambda_max = 50.0
    args.lambda_b = 1000.0
    args.lambda_c = 0.5
    args.lambda_a = (args.lambda_b ** args.lambda_c) * 1e-2
    # -------------------- finite critic state/target identity --------------------
    args.dqc_update_mode = "finite_nstep_v1"
    args.horizon_mode = "finite"
    args.critic_target_source = "n_step"
    args.critic_n_step = 100
    args.critic_time_feature = "remaining_fraction"
    args.critic_actor_context = True
    args.target_update_mode = "hard"
    args.beta = 1.0

    args.num_quantiles = 32
    # Distribution geometry and action marginalization remain unchanged.
    args.num_action_samples = 4
    args.huber_kappa = 0.1
    args.target_tau = 0.005
    args.critic_hidden = [512, 512]
    args.critic_minibatch_size = 1024
    # The QR optimizer owns only the independent action-conditioned critic.
    args.critic_lr = 3e-4
    args.critic_grad_clip = 10.0
    args.critic_epochs = 8

    # The scalar GAE critic is independent and supplies reward credit only.
    # The quantile critic supplies CDF credit only; there is no centre/mean aux.
    args.dqc_mean_estimator = "state_value_gae_v1"
    args.dqc_gae_lambda = 0.97
    args.dqc_value_hidden = [512, 512]
    args.dqc_value_lr = 3e-4
    args.dqc_value_epochs = 8
    args.normalize_combined_advantage = True

    args.actor_lr = 2e-4
    # Preserve the historical actor LR schedule before fixing PPO reuse.
    # theta_a makes the schedule start at actor_lr0 exactly.
    args.actor_lr0 = 2e-4
    args.theta_b = 10000.0
    args.theta_c = 0.9
    args.theta_a = (args.theta_b ** args.theta_c) * args.actor_lr0
    args.actor_grad_clip = 10.0
    # Eight clipped passes retain the existing global gradient/KL guards.
    args.actor_epochs = 8
    args.dqc_actor_reuse_mode = "ppo_clip"
    args.dqc_ratio_clip = 0.1
    args.dqc_target_kl = 0.01

    # This is the historical high-performing Gremlin identity, not the later
    # transfer preset: reward keeps gamma**t and finite CDF risk has no outer.
    args.dqc_reward_outer_discount = "gamma_t"
    args.warmup_iters = 10
    args.critic_warmup_mode = "fitted_hard"
    args.warmup_critic_epochs = 4
    args.training_budget_semantics = INDEPENDENT_FACTORIAL_TRAINING_BUDGET

    # -------------------- new immutable public identity --------------------
    args.independent_factorial_arm = arm
    args.dqc_independent_factorial_implementation_identity = (
        INDEPENDENT_FACTORIAL_IMPLEMENTATION_ID)
    args.dqc_independent_factorial_factor_tuple = (
        canonical_independent_factorial_factor_tuple(arm))
    args.dqc_independent_factorial_policy_clip_mode = spec["clip_mode"]
    args.dqc_independent_factorial_loss_normalization = spec["loss_normalization"]
    args.dqc_independent_factorial_reward_outer_discount = "gamma_t"
    args.dqc_independent_factorial_risk_outer_discount = "none"
    args.dqc_independent_factorial_quantile_mean_loss_coeff = 0.0
    args.dqc_independent_factorial_optimizer_layout = (
        "three_disjoint_adam_actor_value_quantile_v1")

    # W&B opens only after read-only checkpoint preflight and successful load.
    args.defer_wandb_init = True
    args.experiment_variant = (
        f"dqcac-independent-factorial-{arm.lower()}-seed{int(args.seed)}")

    # Keep programmatic construction readable; the runner adds q/alpha/budget.
    env_slug = str(getattr(args, "env_name", "unknown")).lower().replace("_", "-")
    args.wandb_group = f"dqcac-independent-factorial-{env_slug}-v1"
    args.wandb_name = (
        f"dqcac-independent-{spec['description']}-{env_slug}-"
        f"seed{int(args.seed)}")
    args.wandb_lineage = args.wandb_name

    tags = [
        item.strip()
        for item in str(getattr(args, "wandb_tags", "") or "").split(",")
        if item.strip() and not item.strip().startswith("independent-factorial-arm-")
    ]
    for tag in (
            "dqcac-independent-factorial",
            f"independent-factorial-arm-{arm.lower()}"):
        if tag not in tags:
            tags.append(tag)
    args.wandb_tags = ",".join(tags)
    return args


# ============================================================================
# Executable agent
# ============================================================================

class DQCACIndependentFactorialGPU(DQCACBetaGPU):
    """Original independent DQC with a registered PPO clipping factorial."""

    SUPPORTED_TRAINING_BUDGET_SEMANTICS = (
        INDEPENDENT_FACTORIAL_TRAINING_BUDGET,)

    def __init__(self, args, env):
        """Validate the arm before allocating the unchanged independent model."""

        arm = require_independent_factorial_arm(
            getattr(args, "independent_factorial_arm", ""))
        spec = _ARM_SPEC[arm]
        expected = {
            "implementation": (
                getattr(
                    args,
                    "dqc_independent_factorial_implementation_identity",
                    None,
                ),
                INDEPENDENT_FACTORIAL_IMPLEMENTATION_ID,
            ),
            "factor_tuple": (
                getattr(args, "dqc_independent_factorial_factor_tuple", None),
                canonical_independent_factorial_factor_tuple(arm),
            ),
            "policy_clip_mode": (
                getattr(args, "dqc_independent_factorial_policy_clip_mode", None),
                spec["clip_mode"],
            ),
            "loss_normalization": (
                getattr(
                    args,
                    "dqc_independent_factorial_loss_normalization",
                    None,
                ),
                spec["loss_normalization"],
            ),
            "reward_outer": (
                getattr(args, "dqc_reward_outer_discount", None),
                "gamma_t",
            ),
            "registered_reward_outer": (
                getattr(
                    args,
                    "dqc_independent_factorial_reward_outer_discount",
                    None,
                ),
                "gamma_t",
            ),
            "risk_outer": (
                getattr(
                    args,
                    "dqc_independent_factorial_risk_outer_discount",
                    None,
                ),
                "none",
            ),
            "mean_aux": (
                float(getattr(
                    args,
                    "dqc_independent_factorial_quantile_mean_loss_coeff",
                    -1.0,
                )),
                0.0,
            ),
            "optimizer_layout": (
                getattr(
                    args,
                    "dqc_independent_factorial_optimizer_layout",
                    None,
                ),
                "three_disjoint_adam_actor_value_quantile_v1",
            ),
            "action_execution_mode": (
                getattr(args, "action_execution_mode", None),
                INDEPENDENT_FACTORIAL_ACTION_MODE,
            ),
            "deferred_wandb": (
                bool(getattr(args, "defer_wandb_init", False)),
                True,
            ),
        }
        # The compact factor string is redundant by design: these direct
        # checks reject a programmatic caller that edits inherited fields while
        # leaving the serialized tuple untouched.
        expected.update({
            "gamma": (float(getattr(args, "gamma", -1.0)), 0.99),
            "horizon": (int(getattr(args, "horizon", -1)), 1000),
            "actor_hidden": (
                list(getattr(args, "actor_hidden", [])), [512, 512]),
            "lstm_size": (int(getattr(args, "lstm_size", -1)), 512),
            "init_std": (float(getattr(args, "init_std", -1.0)), 1.0),
            "learn_std": (bool(getattr(args, "learn_std", False)), True),
            "normalize_observation": (
                bool(getattr(args, "normalize_observation", False)), True),
            "training_budget_semantics": (
                getattr(args, "training_budget_semantics", None),
                INDEPENDENT_FACTORIAL_TRAINING_BUDGET),
            "dqc_update_mode": (
                getattr(args, "dqc_update_mode", None), "finite_nstep_v1"),
            "horizon_mode": (
                getattr(args, "horizon_mode", None), "finite"),
            "critic_target_source": (
                getattr(args, "critic_target_source", None), "n_step"),
            "critic_n_step": (
                int(getattr(args, "critic_n_step", -1)), 100),
            "critic_time_feature": (
                getattr(args, "critic_time_feature", None),
                "remaining_fraction",
            ),
            "critic_actor_context": (
                bool(getattr(args, "critic_actor_context", False)), True),
            "target_update_mode": (
                getattr(args, "target_update_mode", None), "hard"),
            "beta": (float(getattr(args, "beta", -1.0)), 1.0),
            "num_quantiles": (
                int(getattr(args, "num_quantiles", -1)), 32),
            "num_action_samples": (
                int(getattr(args, "num_action_samples", -1)), 4),
            "huber_kappa": (
                float(getattr(args, "huber_kappa", -1.0)), 0.1),
            "target_tau": (
                float(getattr(args, "target_tau", -1.0)), 0.005),
            "critic_minibatch_size": (
                int(getattr(args, "critic_minibatch_size", -1)), 1024),
            "critic_grad_clip": (
                float(getattr(args, "critic_grad_clip", -1.0)), 10.0),
            "dqc_mean_estimator": (
                getattr(args, "dqc_mean_estimator", None),
                "state_value_gae_v1",
            ),
            "dqc_gae_lambda": (
                float(getattr(args, "dqc_gae_lambda", -1.0)), 0.97),
            "critic_hidden": (
                list(getattr(args, "critic_hidden", [])), [512, 512]),
            "dqc_value_hidden": (
                list(getattr(args, "dqc_value_hidden", [])), [512, 512]),
            "actor_lr": (
                float(getattr(args, "actor_lr", -1.0)), 2e-4),
            "actor_lr0": (
                float(getattr(args, "actor_lr0", -1.0)), 2e-4),
            "theta_b": (float(getattr(args, "theta_b", -1.0)), 10000.0),
            "theta_c": (float(getattr(args, "theta_c", -1.0)), 0.9),
            "theta_a": (
                float(getattr(args, "theta_a", -1.0)),
                (10000.0 ** 0.9) * 2e-4),
            "actor_grad_clip": (
                float(getattr(args, "actor_grad_clip", -1.0)), 10.0),
            "penalty_init": (
                float(getattr(args, "penalty_init", -1.0)), 0.0),
            "critic_lr": (
                float(getattr(args, "critic_lr", -1.0)), 3e-4),
            "dqc_value_lr": (
                float(getattr(args, "dqc_value_lr", -1.0)), 3e-4),
            "lambda_max": (
                float(getattr(args, "lambda_max", -1.0)), 50.0),
            "lambda_b": (float(getattr(args, "lambda_b", -1.0)), 1000.0),
            "lambda_c": (float(getattr(args, "lambda_c", -1.0)), 0.5),
            "lambda_a": (
                float(getattr(args, "lambda_a", -1.0)),
                (1000.0 ** 0.5) * 1e-2),
            "critic_epochs": (
                int(getattr(args, "critic_epochs", -1)), 8),
            "actor_epochs": (
                int(getattr(args, "actor_epochs", -1)), 8),
            "dqc_value_epochs": (
                int(getattr(args, "dqc_value_epochs", -1)), 8),
            "actor_reuse": (
                getattr(args, "dqc_actor_reuse_mode", None), "ppo_clip"),
            "ratio_clip": (
                float(getattr(args, "dqc_ratio_clip", -1.0)), 0.1),
            "target_kl": (
                float(getattr(args, "dqc_target_kl", -1.0)), 0.01),
            "combined_scale": (
                bool(getattr(args, "normalize_combined_advantage", False)),
                True,
            ),
            "warmup_iters": (
                int(getattr(args, "warmup_iters", -1)), 10),
            "critic_warmup_mode": (
                getattr(args, "critic_warmup_mode", None), "fitted_hard"),
            "warmup_critic_epochs": (
                int(getattr(args, "warmup_critic_epochs", -1)), 4),
        })

        mismatches = [
            f"{name}={actual!r} (required {required!r})"
            for name, (actual, required) in expected.items()
            if actual != required
        ]
        if mismatches:
            raise ValueError(
                "DQC independent-factorial identity mismatch: "
                + ", ".join(mismatches))

        # This call owns every seeded allocation. Storing metadata below
        # allocates no parameters and consumes no Python/NumPy/Torch RNG.
        super().__init__(args, env)
        self.action_execution_mode = INDEPENDENT_FACTORIAL_ACTION_MODE
        self.independent_factorial_arm = arm
        self.independent_factorial_factor_tuple = (
            canonical_independent_factorial_factor_tuple(arm))
        self.independent_factorial_policy_clip_mode = spec["clip_mode"]
        self.independent_factorial_loss_normalization = spec["loss_normalization"]
        self._independent_factorial_actor_terms = None

    @torch.no_grad()
    def actor_advantages(self, rollout, gae_batch=None):
        """Materialize unchanged DQC terms and retain detached split views."""

        terms = super().actor_advantages(rollout, gae_batch)
        # Tensors are already detached. Keeping references consumes no RNG and
        # makes all eight PPO epochs reuse one frozen set of labels.
        self._independent_factorial_actor_terms = (
            None if self.independent_factorial_arm == "I0" else terms)
        return terms

    def update_actor_epochs(self, rollout, combined):
        """Reuse parent PPO epochs, then release frozen split-label views."""

        try:
            return super().update_actor_epochs(rollout, combined)
        finally:
            # Do not retain the prior T-by-B rollout while the next rollout is
            # collected. Clearing references changes no tensor or RNG state.
            self._independent_factorial_actor_terms = None

    @staticmethod
    def split_ppo_surrogate(
            ratio, clipped_ratio, reward_weight, constraint_weight,
            valid, denominator, multiplier, divide_by_one_plus_lambda):
        """Return split reward-min/constraint-max loss and components.

        constraint_weight already contains lambda exactly once. multiplier is
        used only by C2's positive outer normalization; it is never multiplied
        into the constraint a second time.
        """

        reward_unclipped = ratio * reward_weight
        reward_clipped = clipped_ratio * reward_weight
        constraint_unclipped = ratio * constraint_weight
        constraint_clipped = clipped_ratio * constraint_weight

        reward_surrogate = torch.minimum(
            reward_unclipped, reward_clipped)
        constraint_surrogate = torch.maximum(
            constraint_unclipped, constraint_clipped)
        reward_loss = -(reward_surrogate * valid).sum() / denominator
        constraint_loss = (
            constraint_surrogate * valid).sum() / denominator

        normalization = torch.ones(
            (), dtype=reward_loss.dtype, device=reward_loss.device)
        if divide_by_one_plus_lambda:
            normalization = normalization + multiplier.detach().to(
                dtype=reward_loss.dtype, device=reward_loss.device)
        loss = (reward_loss + constraint_loss) / normalization

        valid_count = valid.sum().clamp(min=1.0)
        reward_clip_fraction = (
            (reward_clipped < reward_unclipped).to(valid.dtype) * valid
        ).sum() / valid_count
        constraint_clip_fraction = (
            (constraint_clipped > constraint_unclipped).to(valid.dtype) * valid
        ).sum() / valid_count
        return {
            "loss": loss,
            "reward_loss": reward_loss,
            "constraint_loss": constraint_loss,
            "normalization": normalization,
            "reward_clip_fraction": reward_clip_fraction,
            "constraint_clip_fraction": constraint_clip_fraction,
        }

    def _ppo_actor_surrogate(
            self, ratio, clipped_ratio, weight, valid, denominator):
        """Dispatch I0 combined clipping or C1/C2 split clipping."""

        if self.independent_factorial_arm == "I0":
            loss, metrics = super()._ppo_actor_surrogate(
                ratio, clipped_ratio, weight, valid, denominator)
            metrics.update({
                "policy/combined_clip_loss": float(loss.detach().item()),
                "policy/reward_split_loss": 0.0,
                "policy/constraint_split_loss": 0.0,
                "policy/reward_clip_fraction": 0.0,
                "policy/constraint_clip_fraction": 0.0,
                "policy/split_loss_normalization_denominator": 1.0,
                "policy/combined_clipping_active": 1.0,
                "policy/split_clipping_active": 0.0,
            })
            return loss, metrics

        terms = self._independent_factorial_actor_terms
        if not isinstance(terms, dict):
            raise RuntimeError(
                "split clipping requires actor_advantages() from the same rollout")
        scale = terms["combined_normalization_scale"].detach()
        reward_weight = terms["mean_term"].detach() / scale
        constraint_weight = terms["risk_term"].detach() / scale
        result = self.split_ppo_surrogate(
            ratio,
            clipped_ratio,
            reward_weight,
            constraint_weight,
            valid,
            denominator,
            self.lambda_probability,
            divide_by_one_plus_lambda=(
                self.independent_factorial_arm == "C2"),
        )
        metrics = {
            "policy/combined_clip_loss": 0.0,
            "policy/reward_split_loss": float(
                result["reward_loss"].detach().item()),
            "policy/constraint_split_loss": float(
                result["constraint_loss"].detach().item()),
            "policy/reward_clip_fraction": float(
                result["reward_clip_fraction"].detach().item()),
            "policy/constraint_clip_fraction": float(
                result["constraint_clip_fraction"].detach().item()),
            "policy/split_loss_normalization_denominator": float(
                result["normalization"].detach().item()),
            "policy/combined_clipping_active": 0.0,
            "policy/split_clipping_active": 1.0,
        }
        return result["loss"], metrics

    def _validate_vec_env_rng_state(self, states) -> None:
        """Fully validate CPU-worker or MJWarp counters before live state can mutate."""

        if not isinstance(states, (list, tuple)) or len(states) != self.num_envs:
            raise ValueError(
                "DQC independent-factorial vec_env_rng_state must contain exactly "
                f"{self.num_envs} slots")
        cpu_required = {
            "version", "action_mode", "wrapper_seed", "native_seed",
            "native_np_random", "action_space_np_random", "task_random_state",
        }
        mjwarp_required = {
            "version", "backend", "action_mode", "dynamics_profile",
            "seed", "slot", "reset_count", "goal_count",
        }
        expected_backend = (
            "mjwarp"
            if str(getattr(self, "vec_backend", "mp")).lower() == "mjwarp"
            else "cpu"
        )
        expected_profile = str(getattr(
            self, "dynamics_profile", "legacy_euler_v1"))
        bit_generators = {
            generator.__name__: generator
            for generator in (
                np.random.MT19937, np.random.PCG64, np.random.Philox,
                np.random.SFC64,
            )
        }
        for slot_index, slot_state in enumerate(states):
            if (
                    not isinstance(slot_state, dict)
                    or int(slot_state.get("version", -1)) != 1):
                raise ValueError(
                    "DQC independent-factorial vec_env_rng_state slot "
                    f"{slot_index} is not a version-1 mapping")
            if str(slot_state.get("action_mode", "")) != self.action_execution_mode:
                raise ValueError(
                    "DQC independent-factorial vec_env_rng_state slot "
                    f"{slot_index} action mode mismatch")
            saved_backend = str(slot_state.get("backend", "cpu"))
            if saved_backend != expected_backend:
                raise ValueError(
                    "DQC independent-factorial vec_env_rng_state slot "
                    f"{slot_index} backend mismatch")
            saved_profile = str(slot_state.get(
                "dynamics_profile", "legacy_euler_v1"))
            if saved_profile != expected_profile:
                raise ValueError(
                    "DQC independent-factorial vec_env_rng_state slot "
                    f"{slot_index} dynamics profile mismatch")
            required = (
                mjwarp_required if expected_backend == "mjwarp" else cpu_required)
            missing = sorted(required - set(slot_state))
            if missing:
                raise ValueError(
                    "DQC independent-factorial vec_env_rng_state slot "
                    f"{slot_index} is missing fields: " + ", ".join(missing))
            if expected_backend == "mjwarp":
                for field in ("seed", "slot", "reset_count", "goal_count"):
                    if not isinstance(slot_state[field], (int, np.integer)):
                        raise ValueError(
                            "DQC independent-factorial vec_env_rng_state slot "
                            f"{slot_index} {field} is invalid")
                if int(slot_state["slot"]) != slot_index:
                    raise ValueError(
                        "DQC independent-factorial vec_env_rng_state slot "
                        f"{slot_index} slot identity mismatch")
                vector_seed = getattr(self.vec_env, "_seed", None)
                if (vector_seed is not None
                        and int(slot_state["seed"]) != int(vector_seed)):
                    raise ValueError(
                        "DQC independent-factorial vec_env_rng_state slot "
                        f"{slot_index} seed mismatch")
                if (int(slot_state["reset_count"]) < 0
                        or int(slot_state["goal_count"]) < 0):
                    raise ValueError(
                        "DQC independent-factorial vec_env_rng_state slot "
                        f"{slot_index} counter is invalid")
                continue
            for field in ("wrapper_seed", "native_seed"):
                value = slot_state[field]
                if value is not None and not isinstance(value, (int, np.integer)):
                    raise ValueError(
                        "DQC independent-factorial vec_env_rng_state slot "
                        f"{slot_index} {field} is invalid")
            for field in ("native_np_random", "action_space_np_random"):
                nested = slot_state[field]
                if nested is None:
                    continue
                if not isinstance(nested, dict):
                    raise ValueError(
                        "DQC independent-factorial vec_env_rng_state slot "
                        f"{slot_index} {field} is invalid")
                generator_type = bit_generators.get(
                    str(nested.get("bit_generator", "")))
                if generator_type is None:
                    raise ValueError(
                        "DQC independent-factorial vec_env_rng_state slot "
                        f"{slot_index} {field} has unsupported bit generator")
                try:
                    temporary = np.random.Generator(generator_type())
                    temporary.bit_generator.state = copy.deepcopy(nested)
                except (IndexError, TypeError, ValueError) as error:
                    raise ValueError(
                        "DQC independent-factorial vec_env_rng_state slot "
                        f"{slot_index} {field} is invalid") from error
            task_state = slot_state["task_random_state"]
            if task_state is not None:
                try:
                    np.random.RandomState().set_state(copy.deepcopy(task_state))
                except (IndexError, TypeError, ValueError) as error:
                    raise ValueError(
                        "DQC independent-factorial vec_env_rng_state slot "
                        f"{slot_index} task_random_state is invalid") from error

    def _capture_vec_env_rng_state(self):
        """Deep-copy native worker RNGs at the closed pre-next-reset boundary."""

        checkpoint_rng_state = getattr(self.vec_env, "checkpoint_rng_state", None)
        if not callable(checkpoint_rng_state):
            raise RuntimeError(
                "DQC independent-factorial vec_env lacks checkpoint_rng_state; "
                "the next rollout reset cannot be resumed reproducibly")
        states = copy.deepcopy(checkpoint_rng_state())
        self._validate_vec_env_rng_state(states)
        return states

    def _checkpoint_identity(self) -> Dict[str, object]:
        """Return all arm fields that must match before tensor mutation."""

        return {
            "independent_factorial_arm": self.independent_factorial_arm,
            "action_execution_mode": self.action_execution_mode,
            "dqc_independent_factorial_implementation_identity": (
                INDEPENDENT_FACTORIAL_IMPLEMENTATION_ID),
            "dqc_independent_factorial_factor_tuple": (
                self.independent_factorial_factor_tuple),
            "dqc_independent_factorial_policy_clip_mode": (
                self.independent_factorial_policy_clip_mode),
            "dqc_independent_factorial_loss_normalization": (
                self.independent_factorial_loss_normalization),
            "dqc_independent_factorial_reward_outer_discount": "gamma_t",
            "dqc_independent_factorial_risk_outer_discount": "none",
            "dqc_independent_factorial_quantile_mean_loss_coeff": 0.0,
            "dqc_independent_factorial_optimizer_layout": (
                "three_disjoint_adam_actor_value_quantile_v1"),
        }

    @staticmethod
    def _validate_optimizer_loadability(optimizer, saved_state, label) -> None:
        """Load one Adam payload on a private copy and verify moment shapes."""

        try:
            candidate = copy.deepcopy(optimizer)
            candidate.load_state_dict(copy.deepcopy(saved_state))
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            raise ValueError(
                f"DQC independent-factorial {label} state is invalid") from error
        for parameter, slots in candidate.state.items():
            if not isinstance(slots, dict):
                raise ValueError(
                    f"DQC independent-factorial {label} slots are invalid")
            for slot_name, value in slots.items():
                if not torch.is_tensor(value) or slot_name == "step":
                    continue
                if tuple(value.shape) != tuple(parameter.shape):
                    raise ValueError(
                        "DQC independent-factorial optimizer moment shape "
                        f"mismatch: {label}.{slot_name}")

    def _validate_parent_state_loadability(self, state) -> None:
        """Prove every parent load on private copies before live state changes."""

        modules = [
            ("actor", self.actor),
            ("reward_critic", self.reward_critic),
            ("reward_target_critic", self.reward_target_critic),
        ]
        optimizers = [
            ("actor_optimizer", self.actor_optimizer),
            ("critic_optimizer", self.critic_optimizer),
        ]
        if getattr(
                self, "dqc_mean_estimator", self.DISTRIBUTIONAL_MEAN) == self.GAE_MEAN:
            modules.append(("value_critic", self.value_critic))
            optimizers.append(("value_optimizer", self.value_optimizer))

        for label, module in modules:
            try:
                candidate = copy.deepcopy(module)
                candidate.load_state_dict(copy.deepcopy(state[label]))
            except (KeyError, RuntimeError, TypeError, ValueError) as error:
                raise ValueError(
                    f"DQC independent-factorial {label} state is invalid") from error
        for label, optimizer in optimizers:
            self._validate_optimizer_loadability(
                optimizer, state[label], label)
        try:
            # LambdaLR stores the bound ``self._actor_lr_ratio`` callback.  A
            # direct deepcopy therefore walks from the scheduler back through
            # the complete agent and, on a real Safety-Gym process, into an
            # environment Builder that is intentionally not deepcopyable.
            # Build the same scheduler around a private optimizer instead: the
            # saved state is exercised by ``load_state_dict`` while neither the
            # live optimizer/scheduler nor the environment can be mutated.
            scheduler_optimizer = copy.deepcopy(self.actor_optimizer)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                scheduler_optimizer,
                lr_lambda=self._actor_lr_ratio,
            )
            scheduler.load_state_dict(copy.deepcopy(state["actor_scheduler"]))
            lambda_copy = self.lambda_probability.detach().clone()
            lambda_copy.copy_(state["lambda_probability"].to(lambda_copy.device))
            for name in (
                    "dual_update_count", "actor_optimizer_steps",
                    "critic_optimizer_steps", "value_optimizer_steps",
                    "target_update_count", "actor_kl_early_stop_count"):
                if name in state:
                    int(state[name])
            float(state.get("last_dual_prob", 0.0))
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as error:
            raise ValueError(
                "DQC independent-factorial scalar/scheduler state is invalid") from error

    def validate_checkpoint_state(self, state):
        """Reject old-class/cross-arm state before parent tensor validation."""

        if not isinstance(state, dict):
            raise ValueError(
                "DQC independent-factorial checkpoint state must be a dictionary")
        differences = [
            f"{name}: checkpoint={state.get(name, '<missing>')!r}, required={value!r}"
            for name, value in self._checkpoint_identity().items()
            if state.get(name, "<missing>") != value
        ]
        if differences:
            raise ValueError(
                "DQC independent-factorial checkpoint identity mismatch: "
                + "; ".join(differences))
        restore_rng_state = getattr(self.vec_env, "restore_rng_state", None)
        if not callable(restore_rng_state):
            raise ValueError(
                "DQC independent-factorial vec_env lacks restore_rng_state; "
                "the next rollout reset cannot be resumed reproducibly")
        if "vec_env_rng_state" not in state:
            raise ValueError(
                "DQC independent-factorial checkpoint is missing vec_env_rng_state")
        self._validate_vec_env_rng_state(state["vec_env_rng_state"])
        super().validate_checkpoint_state(state)
        self._validate_parent_state_loadability(state)

    def checkpoint_state(self):
        """Serialize DQC state, clipping identity, and native worker RNGs."""

        vec_env_rng_state = self._capture_vec_env_rng_state()
        state = super().checkpoint_state()
        state.update(self._checkpoint_identity())
        state["vec_env_rng_state"] = vec_env_rng_state
        return state

    def load_checkpoint_state(self, state):
        """Commit parent state first, then worker RNG, with full rollback."""

        self.validate_checkpoint_state(state)
        rollback_state = copy.deepcopy(self.checkpoint_state())
        try:
            # Commit actor/critics/Adam/scheduler/scalars before sending any MP
            # worker a restore command. Private-copy validation above makes a
            # malformed serialized payload fail before this boundary.
            super().load_checkpoint_state(state)
            self.vec_env.restore_rng_state(copy.deepcopy(state["vec_env_rng_state"]))
        except BaseException as error:
            try:
                # Restore every parent tensor/counter and worker slot if an
                # unexpected runtime/IPC error occurs during the live commit.
                super().load_checkpoint_state(rollback_state)
                self.vec_env.restore_rng_state(copy.deepcopy(
                    rollback_state["vec_env_rng_state"]))
            except BaseException as rollback_error:
                raise RuntimeError(
                    "DQC independent-factorial checkpoint rollback failed") from rollback_error
            if isinstance(error, (RuntimeError, TypeError, ValueError)):
                raise ValueError(
                    "DQC independent-factorial checkpoint commit failed") from error
            raise

    def get_training_summary(self):
        """Expose clipping identity and the endpoint's three step clocks."""

        summary = super().get_training_summary()
        summary.update({
            "algorithm_identity": INDEPENDENT_FACTORIAL_IMPLEMENTATION_ID,
            "independent_factorial_arm": self.independent_factorial_arm,
            "policy_clip_mode": self.independent_factorial_policy_clip_mode,
            "actor_loss_normalization": (
                self.independent_factorial_loss_normalization),
            "quantile_mean_auxiliary_loss_coeff": 0.0,
            "optimizer_layout": "three_disjoint_adam_actor_value_quantile_v1",
            "action_execution_mode": self.action_execution_mode,
            # These clocks already belong to the checkpoint state. Publishing
            # the same integer values in the summary lets the runner verify the
            # endpoint before either exact-N=1000 evaluation consumes GPU time.
            "main_env_steps": int(self.main_env_steps),
            "initialization_env_steps": int(self.initialization_env_steps),
            "actual_env_steps": int(self.actual_env_steps),
        })
        return summary


__all__ = [
    "COMMON_ADVANTAGE_SCALE",
    "DQCACIndependentFactorialGPU",
    "INDEPENDENT_FACTORIAL_ACTION_MODE",
    "INDEPENDENT_FACTORIAL_ARMS",
    "INDEPENDENT_FACTORIAL_IMPLEMENTATION_ID",
    "INDEPENDENT_FACTORIAL_TRAINING_BUDGET",
    "apply_independent_factorial_configuration",
    "canonical_independent_factorial_factor_tuple",
    "independent_factorial_factor_tuple",
    "require_independent_factorial_arm",
]
