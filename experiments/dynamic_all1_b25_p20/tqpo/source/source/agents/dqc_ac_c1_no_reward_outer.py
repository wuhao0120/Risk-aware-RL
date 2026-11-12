# -*- coding: utf-8 -*-
"""Single-factor C1 ablation: remove reward advantage outer gamma**t.

Everything else is the registered independent-network C1 implementation:
three separate networks/Adam optimizers, scalar frozen-GAE reward credit,
32-atom n-step-100 CDF critic, split PPO clipping, and no finite-risk outer
discount.  Only the actor reward term changes from gamma**t * A_GAE to A_GAE.
"""

from __future__ import annotations

import copy
import json
from typing import Dict

from .dqc_ac_independent_factorial import (
    DQCACIndependentFactorialGPU,
    apply_independent_factorial_configuration,
    independent_factorial_factor_tuple,
)


C1_NO_REWARD_OUTER_IMPLEMENTATION_ID = (
    "dqc_independent_c1_reward_outer_none_v1")
C1_NO_REWARD_OUTER_BASE_ARM = "C1"


def c1_no_reward_outer_factor_tuple():
    """Return C1's complete factor tuple with exactly one replaced value."""

    factors = []
    for name, value in independent_factorial_factor_tuple(
            C1_NO_REWARD_OUTER_BASE_ARM):
        if name == "reward_outer_discount":
            value = "none"
        factors.append((name, value))
    return tuple(factors)


def canonical_c1_no_reward_outer_factor_tuple() -> str:
    """Serialize the immutable single-factor identity without whitespace."""

    return json.dumps(
        c1_no_reward_outer_factor_tuple(),
        separators=(",", ":"), ensure_ascii=True)


def apply_c1_no_reward_outer_configuration(args):
    """Install C1 and replace only its reward outer-discount identity."""

    apply_independent_factorial_configuration(
        args, C1_NO_REWARD_OUTER_BASE_ARM)
    args.algo_name = "DQCAC_C1_NO_REWARD_OUTER"
    args.num_iterations = 400
    args.forced_event_batches = [200, 390, 400]

    # The base DQC actor implementation dispatches this exact string.  GAE
    # itself still contains ordinary temporal discounting; only the additional
    # policy-gradient multiplier gamma**t is absent.
    args.dqc_reward_outer_discount = "none"
    args.dqc_independent_factorial_reward_outer_discount = "none"
    args.dqc_independent_factorial_implementation_identity = (
        C1_NO_REWARD_OUTER_IMPLEMENTATION_ID)
    args.dqc_independent_factorial_factor_tuple = (
        canonical_c1_no_reward_outer_factor_tuple())

    args.dqc_c1_no_reward_outer_implementation_identity = (
        C1_NO_REWARD_OUTER_IMPLEMENTATION_ID)
    args.dqc_c1_no_reward_outer_factor_tuple = (
        canonical_c1_no_reward_outer_factor_tuple())
    args.dqc_c1_no_reward_outer_base_arm = C1_NO_REWARD_OUTER_BASE_ARM
    args.dqc_c1_no_reward_outer_reward_term = "A_GAE_no_outer_gamma_t_v1"
    args.dqc_c1_no_reward_outer_risk_term = (
        "lambda_times_A_CDF_no_outer_discount_v1")

    # One fresh 10M process, one collision-resistant remote identity.  W&B is
    # created only immediately before logging the first complete main row.
    args.defer_wandb_init = True
    args.wandb_id = "dqcc1nogamma-gremlin-s0-10m-v1"
    args.wandb_resume = "never"
    args.wandb_group = "dqcac-c1-reward-outer-gamma-ablation-gremlin-v1"
    args.wandb_name = "C1-no-reward-outer-gamma-seed0-fresh-10m"
    args.wandb_lineage = args.wandb_id
    args.experiment_variant = args.wandb_id
    tags = [
        tag.strip()
        for tag in str(getattr(args, "wandb_tags", "") or "").split(",")
        if tag.strip()
    ]
    for tag in (
            "dqcac-independent", "base-c1", "reward-outer-none",
            "fresh-uninterrupted-10m", "single-factor-ablation"):
        if tag not in tags:
            tags.append(tag)
    args.wandb_tags = ",".join(tags)
    return args


class DQCACC1NoRewardOuterGPU(DQCACIndependentFactorialGPU):
    """C1 with A_GAE used directly in the split reward PPO surrogate."""

    def __init__(self, args, env):
        expected = {
            "implementation": (
                getattr(
                    args, "dqc_c1_no_reward_outer_implementation_identity", None),
                C1_NO_REWARD_OUTER_IMPLEMENTATION_ID),
            "factor_tuple": (
                getattr(args, "dqc_c1_no_reward_outer_factor_tuple", None),
                canonical_c1_no_reward_outer_factor_tuple()),
            "base_arm": (
                getattr(args, "dqc_c1_no_reward_outer_base_arm", None),
                C1_NO_REWARD_OUTER_BASE_ARM),
            "reward_outer": (
                getattr(args, "dqc_reward_outer_discount", None), "none"),
            "risk_outer": (
                getattr(
                    args, "dqc_independent_factorial_risk_outer_discount", None),
                "none"),
            "main_batches": (int(getattr(args, "num_iterations", -1)), 400),
        }
        mismatches = [
            f"{name}={actual!r} (required {required!r})"
            for name, (actual, required) in expected.items()
            if actual != required
        ]
        if mismatches:
            raise ValueError(
                "DQC C1-no-reward-outer identity mismatch: "
                + ", ".join(mismatches))

        # Parent C1 validates its historical gamma-outer identity before model
        # construction.  Give it a private namespace containing that metadata,
        # then restore the public no-outer identity on this agent.  All seeded
        # allocations and initial parameters are therefore bitwise C1-equivalent.
        construction_args = copy.copy(args)
        apply_independent_factorial_configuration(
            construction_args, C1_NO_REWARD_OUTER_BASE_ARM)
        construction_args.algo_name = args.algo_name
        super().__init__(construction_args, env)

        self.dqc_reward_outer_discount = "none"
        self.independent_factorial_factor_tuple = (
            canonical_c1_no_reward_outer_factor_tuple())

    def _checkpoint_identity(self) -> Dict[str, object]:
        """Replace C1's gamma metadata and add the no-outer experiment ID."""

        identity = super()._checkpoint_identity()
        identity.update({
            "dqc_independent_factorial_implementation_identity": (
                C1_NO_REWARD_OUTER_IMPLEMENTATION_ID),
            "dqc_independent_factorial_factor_tuple": (
                canonical_c1_no_reward_outer_factor_tuple()),
            "dqc_independent_factorial_reward_outer_discount": "none",
            "dqc_c1_no_reward_outer_implementation_identity": (
                C1_NO_REWARD_OUTER_IMPLEMENTATION_ID),
            "dqc_c1_no_reward_outer_factor_tuple": (
                canonical_c1_no_reward_outer_factor_tuple()),
            "dqc_c1_no_reward_outer_base_arm": C1_NO_REWARD_OUTER_BASE_ARM,
            "dqc_c1_no_reward_outer_reward_term": (
                "A_GAE_no_outer_gamma_t_v1"),
            "dqc_c1_no_reward_outer_risk_term": (
                "lambda_times_A_CDF_no_outer_discount_v1"),
        })
        return identity

    def get_training_summary(self):
        """Expose the exact changed equation alongside the inherited C1 state."""

        summary = super().get_training_summary()
        summary.update({
            "algorithm_identity": C1_NO_REWARD_OUTER_IMPLEMENTATION_ID,
            "base_arm": C1_NO_REWARD_OUTER_BASE_ARM,
            "reward_actor_term": "A_GAE",
            "reward_outer_discount": "none",
            "risk_actor_term": "lambda_times_A_CDF",
            "finite_risk_outer_discount": "none",
            "changed_factor_count": 1,
        })
        return summary


__all__ = [
    "C1_NO_REWARD_OUTER_IMPLEMENTATION_ID",
    "DQCACC1NoRewardOuterGPU",
    "apply_c1_no_reward_outer_configuration",
    "canonical_c1_no_reward_outer_factor_tuple",
    "c1_no_reward_outer_factor_tuple",
]
