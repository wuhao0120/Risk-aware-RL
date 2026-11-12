# -*- coding: utf-8 -*-
"""
QCPO reference agent aligned to the authors' executable NIPS-2022 implementation.

This class is deliberately isolated from :mod:`agents.qcpo_ref`.  The existing public
``QCPO_REF`` remains the repository's finite-context reward-side baseline; this module owns the
new public identity ``QCPO_REF_AUTHOR_ALIGNED`` and cannot load the old class's checkpoints.

The executable behavior copied from upstream is:

* one shared MLP(512, 512, Tanh)-LSTM(512) model for policy and every value/distribution head;
* raw Gaussian PPO with separately clipped reward/min and constraint/max surrogates;
* 25 independent midpoint quantile heads, the 0.78 atom for alpha=0.2, and the top eight atoms
  for the Weibull density correction;
* time-major 1000x12 sampling transformed into 100x120 recurrent chunks;
* one Adam optimizer, one full recurrent minibatch, and eight fixed-target epochs;
* a 100-trajectory discrete order-statistic dual updated before optimization, including batch 1;
* batch 1 updates the dual and observation RMS but performs no neural optimizer step.

Only the frozen task adapter differs from upstream: the internal cost is ``C=-G`` (per-step
``-reward``), its fixed threshold is ``-1.85``, signed atoms replace ``exp(linear)``, and a learned
location extends the Weibull support.  Native Safety-Gymnasium cost is diagnostic-only.
"""

from collections import deque
import copy
import math
from numbers import Integral
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch.optim import Adam

from .vec_base import VecAgentBase
from .qcpo_ref_author_aligned_model import QcpoRefAuthorAlignedModel, RnnState
from .dist_rl_utils_author_aligned import (
    compute_prob_ratio,
    gae_quantile_simple,
    generalized_advantage_estimation,
    quantile_huber_loss,
    quantile_target_estimation,
    shifted_weibull_log_pdf,
    valid_from_done,
    valid_mean,
    weibull_tail_loss,
)
from utils.risk_metrics import linear_empirical_quantile, return_statistics


# The version strings are checkpoint state as well as runner-signature fields.  Changing any of
# them creates a new scientific implementation instead of silently altering an existing lineage.
QCPO_AUTHOR_UPSTREAM_SHA = "9e633849154e7f2471bf98709bcc8ba83514708e"
QCPO_AUTHOR_IMPLEMENTATION_ID = (
    "qcpo_author_executable_aligned_reward_outage_shifted_weibull_v1"
)
QCPO_AUTHOR_TARGET_ADAPTER = "discounted_reward_outage_C_equals_negative_G_v1"
QCPO_AUTHOR_HISTORY_TIMING = (
    "wrapper_old_stored_internal_cost_extra_one_step_delay_"
    "collector_carries_terminal_action_reward_v1"
)
QCPO_AUTHOR_DENSITY_RULE = "center_TB_divide_5std_no_epsilon_clip_0p5_done_after_outside_support_final_neutral_v1"
QCPO_AUTHOR_DUAL_RULE = "deque100_immediate_discrete_floor_n_times_1_minus_alpha_Ki0p1_v1"
QCPO_AUTHOR_QUANTILE_GRID = "midpoint_25_0p02_to_0p98_target_index19_tail_index17_v1"
QCPO_AUTHOR_WEIBULL_IDENTITY = (
    "signed_independent_atoms_shifted_three_parameter_location_below_detached_min_v1"
)
QCPO_AUTHOR_POLICY_CLIP_RULE = "split_reward_min_constraint_max_sum_norm_v1"
QCPO_AUTHOR_RMS_WARMUP = "first_batch_dual_and_rms_no_optimizer_count0_v1"
QCPO_AUTHOR_LAYOUT = "T1000_B12_transform_T100_B120_chunk_bootstrap_early_end_fail_v1"

# Formal B12 is an explicit opt-in execution profile.  Keeping these fields outside the ordinary
# author identity preserves historical Euler B12 runs, while a formal run must sign the same
# stable dynamics, Warp reproducibility disclosure, and W&B destination as formal B25.
QCPO_FORMAL_DYNAMICS_PROFILE = "point_velocity_implicitfast_v1"
QCPO_FORMAL_TRAIN_WARP_DETERMINISM = "NOT_GUARANTEED"
QCPO_FORMAL_TRAIN_WARP_DETERMINISTIC_MAX_RECORDS = 0
QCPO_FORMAL_WANDB_PROJECT = "Gremlin_full_gpu"

# A checkpoint may be written immediately after construction, before the first environment reset,
# or after a complete rollout has installed CpuResetCollector-compatible carried AgentInputs.
# Serializing this distinction prevents a pristine resume from accidentally executing an eager
# reset and then entering the first rollout as if terminal action/reward history already existed.
QCPO_SAMPLER_PHASE_PRISTINE = "pristine_before_first_rollout_v1"
QCPO_SAMPLER_PHASE_CARRIED = "post_rollout_reset_carried_inputs_v1"

# MJWarp v1 reconstructs stochastic layouts from two per-world int32 counters.  The exact key set
# is checkpoint schema: accepting an unknown extra clock would make replay dependent on state this
# implementation did not validate or restore.
_MJWARP_RNG_FIELDS = frozenset({
    "version", "backend", "action_mode", "dynamics_profile", "seed", "slot",
    "reset_count", "goal_count",
})


class QCPORefAuthorAlignedGPU(VecAgentBase):
    """Literal upstream QCPO execution with the frozen reward-outage/shifted-Weibull adapter."""

    # The first 12k rollout is physical campaign budget even though no neural optimizer step occurs.
    # Keeping this permission on the subclass leaves every legacy agent's base-class gate unchanged.
    SUPPORTED_TRAINING_BUDGET_SEMANTICS = (
        "author_nominal_10m_round_up_log_interval_first_batch_rms_only_v1",
    )

    # ---------------------------------------------------------------------- construction / identity
    def __init__(self, args, env):
        """Allocate the isolated shared model, one Adam optimizer, and exact dual state."""

        # Formal identity is checked before VecAgentBase can create workers or initialize W&B.
        # Ordinary native B12 namespaces do not enter this gate and retain their historical setup.
        self._validate_formal_execution_identity(args)

        # VecAgentBase owns vector-environment construction, W&B, clocks, and common logging.  Its
        # ordinary actor is replaced below; it is never queried or checkpointed by this subclass.
        # Preserve Torch RNG across that unused allocation so seed-0 initializes the aligned shared
        # model exactly as a direct construction of the author's model would.
        torch_rng_before_base = torch.get_rng_state()
        cuda_rng_before_base = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        super().__init__(args, env)
        torch.set_rng_state(torch_rng_before_base)
        if cuda_rng_before_base is not None:
            torch.cuda.set_rng_state_all(cuda_rng_before_base)
        self._require_return_threshold()

        # Keep the actual simulator/action envelope next to the sampler state.  CPU MP/SYNC
        # checkpoints remain supported, while an MJWarp run can now require its stronger schema.
        self.vec_backend = str(getattr(args, "vec_backend", "mp")).lower()
        self.action_execution_mode = str(getattr(
            args, "action_execution_mode", "author_raw_env_clamp"))
        self.formal_endpoint_primary = bool(getattr(
            args, "formal_endpoint_primary", False))
        if self.formal_endpoint_primary:
            self.train_warp_determinism = str(args.train_warp_determinism)
            self.train_warp_deterministic_max_records = int(
                args.train_warp_deterministic_max_records)
            self.formal_wandb_project = str(args.wandb_project)

        # Preserve these identities in both agent state and the runner's flat checkpoint signature.
        self.author_upstream_sha = str(getattr(
            args, "qcpo_author_upstream_sha", QCPO_AUTHOR_UPSTREAM_SHA))
        self.implementation_identity = str(getattr(
            args, "qcpo_author_implementation_identity", QCPO_AUTHOR_IMPLEMENTATION_ID))
        self.target_adapter = str(getattr(
            args, "qcpo_author_target_adapter", QCPO_AUTHOR_TARGET_ADAPTER))
        self.history_timing = str(getattr(
            args, "qcpo_author_history_timing", QCPO_AUTHOR_HISTORY_TIMING))
        self.density_rule = str(getattr(
            args, "qcpo_author_density_rule", QCPO_AUTHOR_DENSITY_RULE))
        self.dual_rule = str(getattr(
            args, "qcpo_author_dual_rule", QCPO_AUTHOR_DUAL_RULE))
        self.quantile_grid_identity = str(getattr(
            args, "qcpo_author_quantile_grid", QCPO_AUTHOR_QUANTILE_GRID))
        self.weibull_identity = str(getattr(
            args, "qcpo_author_weibull_identity", QCPO_AUTHOR_WEIBULL_IDENTITY))
        self.policy_clip_rule = str(getattr(
            args, "qcpo_author_policy_clip_rule", QCPO_AUTHOR_POLICY_CLIP_RULE))
        self.rms_warmup_identity = str(getattr(
            args, "qcpo_author_rms_warmup", QCPO_AUTHOR_RMS_WARMUP))
        self.layout_identity = str(getattr(
            args, "qcpo_author_layout", QCPO_AUTHOR_LAYOUT))
        self.wandb_lineage = str(getattr(args, "wandb_lineage", ""))

        # Algorithm hyperparameters mirror config_qcpo.py / launch_qcpo.py.  They are intentionally
        # read separately rather than borrowing schedule fields from the old finite-context agent.
        self.learning_rate = float(getattr(args, "ref_lr", 1e-4))
        self.epochs = int(getattr(args, "epochs", 8))
        self.minibatches = int(getattr(args, "minibatches", 1))
        self.ratio_clip = float(getattr(args, "ratio_clip", 0.1))
        self.gae_lambda = float(getattr(args, "gae_lambda", 0.97))
        self.cost_discount = float(getattr(args, "cost_discount", self.gamma))
        self.cost_gae_lambda = float(getattr(args, "cost_gae_lambda", self.gae_lambda))
        self.cost_scale = float(getattr(args, "cost_scale", 10.0))
        self.cost_value_loss_coeff = float(getattr(args, "cost_value_loss_coeff", 0.5))
        self.value_loss_coeff = float(getattr(args, "value_loss_coeff", 1.0))
        self.quantile_loss_coeff = float(getattr(args, "quantile_loss_coeff", 1.0))
        self.weibull_loss_coeff = float(getattr(args, "weibull_loss_coeff", 1.0))
        self.entropy_loss_coeff = float(getattr(args, "entropy_loss_coeff", 0.0))
        self.clip_grad_norm = float(getattr(args, "clip_grad_norm", 1e4))
        self.pid_Ki = float(getattr(args, "pid_Ki", 0.1))
        self.penalty_init = float(getattr(args, "penalty_init", 0.0))
        self.sum_norm = bool(getattr(args, "sum_norm", True))
        self.diff_norm = bool(getattr(args, "diff_norm", False))
        self.n_quantile = int(getattr(args, "n_quantile", 25))
        self.weibull_tail_prob = float(getattr(args, "weibull_tail_prob", 0.3))
        self.weibull_location_epsilon = float(
            getattr(args, "weibull_location_epsilon", 1e-6))
        self.new_T = int(getattr(args, "new_T", 100))
        self.new_B = int(getattr(
            args, "new_B", (self.n * self.num_envs) // self.new_T))

        # The target is expressed in internal raw-cost units, then divided by the same official
        # cost_scale as the critic's per-step -reward.  Thus d_C=-q=-1.85 and d_C/10=-0.185.
        self.cost_threshold = -float(self.return_threshold)
        self.cost_threshold_scaled = self.cost_threshold / self.cost_scale
        self.target_outage_prob = float(self.failure_alpha)
        self.target_outage_ind = int(np.floor(
            self.n_quantile * (1.0 - self.target_outage_prob) - 0.5))
        self.weibull_tail_ind = int(np.floor(
            self.n_quantile * (1.0 - self.weibull_tail_prob) - 0.5))

        # Fail before allocating the aligned model if a nominal run has drifted from the frozen
        # executable identity.  These checks are deliberately strict: ablations need a new class ID.
        self._validate_frozen_configuration(args)

        # The collector appends one lagged internal-cost coordinate to the native observation.  The
        # model then consumes [phi(obs_aug), previous raw action, previous reward] exactly upstream.
        self.obs_dim_aug = self.state_dim + 1
        self.model = QcpoRefAuthorAlignedModel(
            observation_shape=(self.obs_dim_aug,),
            action_size=self.action_dim,
            n_quantile=self.n_quantile,
            hidden_sizes=(512, 512),
            lstm_size=512,
            lstm_skip=True,
            constraint=True,
            init_log_std=0.0,
            normalize_observation=True,
            var_clip=1e-6,
            location_epsilon=self.weibull_location_epsilon,
            raw_cost_support=getattr(self, "raw_cost_constraint", False),
        ).to(self.device)

        # Keep ``actor`` as a compatibility alias for introspection only.  This subclass overrides
        # sampling, evaluation, and checkpoint methods, so VecAgentBase never calls actor.step().
        self.actor = self.model
        self.lstm_size = self.model.lstm_size
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

        # The upstream PID implementation is integral-only under this configuration.  sum_norm=True
        # deliberately imposes no upper cap on lambda.
        self.current_ep_costs = deque(maxlen=100)
        self.pid_i = self.penalty_init
        self.cost_penalty = max(0.0, self.pid_i)
        self.lambda_dual = torch.tensor(
            [self.cost_penalty], dtype=torch.float32, device=self.device)
        self.dual_update_count = 0
        self.update_counter = 0
        self.rms_only_batch_count = 0
        self.last_dual_cost_quantile = self.cost_threshold
        self.last_dual_reward_quantile = float(self.return_threshold)
        self.last_dual_residual = 0.0
        self.last_loss_info: Dict[str, float] = {}
        # CpuResetCollector owns one persistent AgentInputs tuple across optimize_agent() calls.
        # Initialize lazily on the first rollout, then carry reset observation plus terminal
        # action/reward across every later rollout and serialize it at checkpoint boundaries.
        self._sampler_state = None
        self._sampler_pre_reset_rng_state = None
        self._sampler_phase = QCPO_SAMPLER_PHASE_PRISTINE

    @staticmethod
    def _validate_formal_execution_identity(args) -> None:
        """Reject incomplete formal B12/B25 execution envelopes before side effects."""

        if not bool(getattr(args, "formal_endpoint_primary", False)):
            return

        # ``type(...) is int`` deliberately rejects bool, which otherwise aliases integer zero.
        max_records = getattr(args, "train_warp_deterministic_max_records", None)
        expected = {
            "dynamics_profile": (
                getattr(args, "dynamics_profile", None),
                QCPO_FORMAL_DYNAMICS_PROFILE,
            ),
            "train_warp_determinism": (
                getattr(args, "train_warp_determinism", None),
                QCPO_FORMAL_TRAIN_WARP_DETERMINISM,
            ),
            "train_warp_deterministic_max_records": (
                max_records,
                QCPO_FORMAL_TRAIN_WARP_DETERMINISTIC_MAX_RECORDS,
            ),
            "wandb_project": (
                getattr(args, "wandb_project", None),
                QCPO_FORMAL_WANDB_PROJECT,
            ),
        }
        mismatches = [
            f"{name}={actual!r} (expected {wanted!r})"
            for name, (actual, wanted) in expected.items()
            if actual != wanted
        ]
        if type(max_records) is not int:
            mismatches.append(
                "train_warp_deterministic_max_records must be an exact int")
        if mismatches:
            raise ValueError(
                "QCPO formal execution identity mismatch:\n  "
                + "\n  ".join(mismatches))

    def _formal_execution_identity(self) -> Dict[str, object]:
        """Return formal-only checkpoint fields without changing ordinary B12 identity."""

        if not bool(getattr(self, "formal_endpoint_primary", False)):
            return {}
        return {
            "dynamics_profile": self.dynamics_profile,
            "train_warp_determinism": self.train_warp_determinism,
            "train_warp_deterministic_max_records": (
                self.train_warp_deterministic_max_records
            ),
            "wandb_project": self.formal_wandb_project,
        }

    def _validate_frozen_configuration(self, args) -> None:
        """Reject every structural or numerical deviation from the approved aligned identity."""

        expected = {
            "author_upstream_sha": (self.author_upstream_sha, QCPO_AUTHOR_UPSTREAM_SHA),
            "implementation_identity": (
                self.implementation_identity, QCPO_AUTHOR_IMPLEMENTATION_ID),
            "target_adapter": (self.target_adapter, QCPO_AUTHOR_TARGET_ADAPTER),
            "history_timing": (self.history_timing, QCPO_AUTHOR_HISTORY_TIMING),
            "density_rule": (self.density_rule, QCPO_AUTHOR_DENSITY_RULE),
            "dual_rule": (self.dual_rule, QCPO_AUTHOR_DUAL_RULE),
            "quantile_grid_identity": (
                self.quantile_grid_identity, QCPO_AUTHOR_QUANTILE_GRID),
            "weibull_identity": (
                self.weibull_identity, QCPO_AUTHOR_WEIBULL_IDENTITY),
            "policy_clip_rule": (
                self.policy_clip_rule, QCPO_AUTHOR_POLICY_CLIP_RULE),
            "rms_warmup_identity": (
                self.rms_warmup_identity, QCPO_AUTHOR_RMS_WARMUP),
            "layout_identity": (self.layout_identity, QCPO_AUTHOR_LAYOUT),
            "training_budget_semantics": (
                self.training_budget_semantics,
                "author_nominal_10m_round_up_log_interval_first_batch_rms_only_v1",
            ),
            "num_iterations_vs_author_rollout_batches": (
                self.num_iterations,
                int(getattr(args, "qcpo_author_rollout_batches", -1)),
            ),
            "gamma": (self.gamma, 0.99),
            "cost_gamma": (float(getattr(args, "cost_gamma", 0.99)), 0.99),
            "failure_alpha": (self.failure_alpha, 0.1),
            "return_threshold": (self.return_threshold, 15.),
            "num_envs": (self.num_envs, 12),
            "horizon": (self.n, 1000),
            "new_T": (self.new_T, 100),
            "new_B": (self.new_B, 120),
            "n_quantile": (self.n_quantile, 25),
            "epochs": (self.epochs, 8),
            "minibatches": (self.minibatches, 1),
            "ref_lr": (self.learning_rate, 1e-4),
            "ratio_clip": (self.ratio_clip, 0.1),
            "gae_lambda": (self.gae_lambda, 0.97),
            "cost_discount": (self.cost_discount, 0.99),
            "cost_gae_lambda": (self.cost_gae_lambda, 0.97),
            "cost_scale": (self.cost_scale, 10.0),
            "cost_value_loss_coeff": (self.cost_value_loss_coeff, 0.5),
            "value_loss_coeff": (self.value_loss_coeff, 1.0),
            "quantile_loss_coeff": (self.quantile_loss_coeff, 1.0),
            "weibull_loss_coeff": (self.weibull_loss_coeff, 1.0),
            "entropy_loss_coeff": (self.entropy_loss_coeff, 0.0),
            "clip_grad_norm": (self.clip_grad_norm, 1e4),
            "pid_Ki": (self.pid_Ki, 0.1),
            "sum_norm": (self.sum_norm, True),
            "diff_norm": (self.diff_norm, False),
            "target_outage_ind": (self.target_outage_ind, 22),
            "weibull_tail_ind": (self.weibull_tail_ind, 17),
            "weibull_tail_prob": (self.weibull_tail_prob, 0.3),
            "weibull_location_epsilon": (self.weibull_location_epsilon, 1e-6),
            "actor_hidden": (
                tuple(getattr(args, "actor_hidden", (512, 512))), (512, 512)),
            "configured_lstm_size": (int(getattr(args, "lstm_size", 512)), 512),
            "init_std": (float(getattr(args, "init_std", 1.0)), 1.0),
            "learn_std": (bool(getattr(args, "learn_std", True)), True),
            "normalize_observation": (
                bool(getattr(args, "normalize_observation", True)), True),
            "normalize_advantage": (
                bool(getattr(args, "normalize_advantage", False)), False),
            "normalize_cost_advantage": (
                bool(getattr(args, "normalize_cost_advantage", False)), False),
            "normalize_combined_advantage": (
                bool(getattr(args, "normalize_combined_advantage", False)), False),
            "lambda_max": (getattr(args, "lambda_max", None), None),
            "penalty_init": (self.penalty_init, 0.0),
        }
        mismatches = [
            f"{name}={actual!r} (expected {wanted!r})"
            for name, (actual, wanted) in expected.items()
            if actual != wanted
        ]
        if self.n * self.num_envs != self.new_T * self.new_B:
            mismatches.append(
                f"layout {self.n}*{self.num_envs} != {self.new_T}*{self.new_B}")
        if self.n % self.new_T != 0:
            mismatches.append(f"horizon {self.n} is not divisible by new_T {self.new_T}")
        action_mode = str(getattr(args, "action_execution_mode", ""))
        if action_mode != "author_raw_env_clamp":
            mismatches.append(
                f"action_execution_mode={action_mode!r} (expected 'author_raw_env_clamp')")
        if mismatches:
            raise ValueError(
                "QCPO_REF_AUTHOR_ALIGNED frozen configuration mismatch:\n  "
                + "\n  ".join(mismatches))

    # --------------------------------------------------------------------------- Gaussian utilities
    def _logp(self, action: torch.Tensor, mean: torch.Tensor,
              log_std: torch.Tensor) -> torch.Tensor:
        """Return upstream raw-Gaussian log likelihood, including its ``1e-8`` denominator."""

        std = torch.exp(log_std)
        standardized = (action - mean) / (std + 1e-8)
        return -(
            (log_std + 0.5 * standardized.square()).sum(dim=-1)
            + 0.5 * action.shape[-1] * self._log_2pi
        )

    def _gaussian_entropy(self, log_std: torch.Tensor) -> torch.Tensor:
        """Return diagonal-Gaussian entropy for diagnostics; its loss coefficient is frozen at 0."""

        return (log_std + 0.5 * (1.0 + self._log_2pi)).sum(dim=-1)

    @staticmethod
    def split_clipped_policy_losses(
        ratio: torch.Tensor,
        reward_advantage: torch.Tensor,
        constraint_advantage: torch.Tensor,
        valid: torch.Tensor,
        clip_ratio: float,
        multiplier: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the official reward/min and constraint/max PPO clipping independently."""

        clipped = torch.clamp(ratio, 1.0 - float(clip_ratio), 1.0 + float(clip_ratio))
        reward_loss = -valid_mean(
            torch.minimum(ratio * reward_advantage, clipped * reward_advantage), valid)
        constraint_loss = valid_mean(
            torch.maximum(
                ratio * constraint_advantage,
                clipped * constraint_advantage,
            ),
            valid,
        )
        combined = (
            (reward_loss + float(multiplier) * constraint_loss)
            / (1.0 + float(multiplier))
        )
        clipped_rows = (ratio - 1.0).abs() > float(clip_ratio)
        clip_fraction = valid_mean(clipped_rows.to(ratio.dtype), valid)
        return combined, reward_loss, constraint_loss, clip_fraction

    # ------------------------------------------------------------------------- sampling / histories
    def _initial_rnn_state(self, batch: int) -> RnnState:
        """Allocate the single-layer LSTM state used at reset and at ended vector slots."""

        shape = (1, int(batch), self.lstm_size)
        zeros = torch.zeros(shape, dtype=torch.float32, device=self.device)
        return RnnState(h=zeros, c=zeros.clone())

    def _rollout_author_aligned(self) -> Dict[str, torch.Tensor]:
        """Collect exactly ``[T=1000,B=12]`` while preserving the wrapper's cost-delay quirk."""

        time_steps, batch = self.n, self.num_envs
        if getattr(self, "_sampler_state", None) is None:
            sampler_phase = getattr(
                self, "_sampler_phase", QCPO_SAMPLER_PHASE_PRISTINE)
            if sampler_phase != QCPO_SAMPLER_PHASE_PRISTINE:
                raise RuntimeError(
                    "QCPO author-aligned sampler phase/state mismatch before first reset")
            # Only sampler construction supplies all-zero history.  At later batch boundaries the
            # official collector retains terminal action/reward while the environment and LSTM have
            # already reset for the next trajectory.
            raw_observation = self.vec_env.reset()
            observation_cost = torch.zeros(batch, 1, device=self.device)
            stored_internal_cost = torch.zeros_like(observation_cost)
            previous_action = torch.zeros(batch, self.action_dim, device=self.device)
            previous_reward = torch.zeros(batch, device=self.device)
            recurrent = self._initial_rnn_state(batch)
        else:
            sampler_phase = getattr(
                self, "_sampler_phase", QCPO_SAMPLER_PHASE_CARRIED)
            if sampler_phase != QCPO_SAMPLER_PHASE_CARRIED:
                raise RuntimeError(
                    "QCPO author-aligned carried sampler phase/state mismatch")
            carried = self._sampler_state
            raw_observation = carried["raw_observation"].clone()
            observation_cost = carried["observation_cost"].clone()
            stored_internal_cost = carried["stored_internal_cost"].clone()
            previous_action = carried["previous_action"].clone()
            previous_reward = carried["previous_reward"].clone()
            recurrent = RnnState(
                h=carried["recurrent_h"].clone(),
                c=carried["recurrent_c"].clone(),
            )

        # ``observation_cost`` is what the agent currently sees. ``stored_internal_cost`` is the
        # wrapper's private previous cost.  After each step the wrapper first constructs next_obs
        # using the old stored value and only then stores -r_t, yielding the intentional extra lag.
        active = torch.ones(batch, dtype=torch.bool, device=self.device)

        names = (
            "obs", "prev_action", "prev_reward", "action", "mu_old", "log_std_old",
            "reward", "diagnostic_cost", "done", "valid", "r_value", "c_dist",
            "c_w_location", "c_w_alpha", "c_w_beta", "h0", "c0",
        )
        fields = {name: [] for name in names}
        discounted_return = torch.zeros(batch, device=self.device)
        diagnostic_discounted_cost = torch.zeros_like(discounted_return)
        diagnostic_raw_cost = torch.zeros_like(discounted_return)

        with torch.no_grad():
            for time_index in range(time_steps):
                augmented = torch.cat((raw_observation, observation_cost), dim=-1)
                fields["h0"].append(recurrent.h[0].clone())
                fields["c0"].append(recurrent.c[0].clone())

                mean, log_std, value, next_recurrent = self.model(
                    augmented, previous_action, previous_reward, recurrent)
                raw_action = mean + torch.exp(log_std) * torch.randn_like(mean)
                step = self.vec_env.step(raw_action)

                valid = active.clone()
                ended = step.episode_end & valid
                if time_index + 1 < time_steps and bool(ended.any()):
                    raise RuntimeError(
                        "QCPO_REF_AUTHOR_ALIGNED fixed-horizon rollout received "
                        f"episode_end before step {time_steps}: t={time_index}")
                reward = step.reward * valid.to(step.reward.dtype)
                diagnostic_cost = step.cost * valid.to(step.cost.dtype)

                fields["obs"].append(augmented)
                fields["prev_action"].append(previous_action.clone())
                fields["prev_reward"].append(previous_reward.clone())
                fields["action"].append(raw_action)
                fields["mu_old"].append(mean)
                fields["log_std_old"].append(log_std)
                fields["reward"].append(reward)
                fields["diagnostic_cost"].append(diagnostic_cost)
                fields["done"].append(ended.to(reward.dtype))
                fields["valid"].append(valid.to(reward.dtype))
                fields["r_value"].append(value.r_value)
                fields["c_dist"].append(value.c_dist)
                fields["c_w_location"].append(value.c_w_location)
                fields["c_w_alpha"].append(value.c_w_alpha)
                fields["c_w_beta"].append(value.c_w_beta)

                discount = self.gamma ** time_index
                discounted_return.add_(discount * reward)
                diagnostic_discounted_cost.add_(discount * diagnostic_cost)
                diagnostic_raw_cost.add_(diagnostic_cost)

                # Every valid native interaction belongs to the advertised main budget, including
                # the RMS-only first batch.  This algorithm has no separate initialization phase.
                valid_steps = int(valid.sum().item())
                self.main_env_steps += valid_steps
                self.actual_env_steps = self.main_env_steps + self.initialization_env_steps
                self.total_env_steps = self.actual_env_steps

                # Literal wrapper order: expose old stored cost on next observation, then save -r_t.
                ended_column = ended.unsqueeze(-1)
                next_observation_cost = torch.where(
                    ended_column, torch.zeros_like(stored_internal_cost), stored_internal_cost)
                stored_internal_cost = torch.where(
                    ended_column,
                    torch.zeros_like(stored_internal_cost),
                    (step.cost if getattr(self, "raw_cost_constraint", False) else -reward).unsqueeze(-1),
                )
                observation_cost = next_observation_cost
                previous_action = raw_action
                previous_reward = reward
                raw_observation = step.observation

                # Gremlin normally ends only at the fixed adapter timeout.  The mask keeps an
                # unexpected earlier native termination from contaminating reward/value objectives.
                active = active & ~ended
                keep = active.to(next_recurrent.h.dtype).view(1, batch, 1)
                recurrent = RnnState(
                    h=next_recurrent.h * keep,
                    c=next_recurrent.c * keep,
                )

            # The author-mode vector adapter defers the synchronized horizon reset until this exact
            # point. Save pre-reset native RNGs, reset once (as CpuResetCollector does immediately
            # after terminal transition recording), and use that reset observation for bootstrap.
            self._sampler_pre_reset_rng_state = self.vec_env.checkpoint_rng_state()
            raw_observation = self.vec_env.reset()
            observation_cost = torch.zeros_like(observation_cost)
            stored_internal_cost = torch.zeros_like(stored_internal_cost)

            # Upstream requests a final bootstrap for every original sampler trajectory.  Although
            # done=1 later nulls ordinary returns, the bootstrap Weibull heads still participate in
            # the literal pre-mask density normalization, so the reset observation must be exact.
            final_augmented = torch.cat((raw_observation, observation_cost), dim=-1)
            _mean, _log_std, final_value, _next = self.model(
                final_augmented, previous_action, previous_reward, recurrent)

        self._sampler_state = {
            "raw_observation": raw_observation.detach().clone(),
            "observation_cost": observation_cost.detach().clone(),
            "stored_internal_cost": stored_internal_cost.detach().clone(),
            "previous_action": previous_action.detach().clone(),
            "previous_reward": previous_reward.detach().clone(),
            "recurrent_h": recurrent.h.detach().clone(),
            "recurrent_c": recurrent.c.detach().clone(),
        }
        self._sampler_phase = QCPO_SAMPLER_PHASE_CARRIED

        rollout = {name: torch.stack(values, dim=0) for name, values in fields.items()}
        rollout.update({
            "r_bootstrap_original": final_value.r_value,
            "c_bootstrap_original": final_value.c_dist,
            "location_bootstrap_original": final_value.c_w_location,
            "alpha_bootstrap_original": final_value.c_w_alpha,
            "beta_bootstrap_original": final_value.c_w_beta,
            "disc_return": discounted_return,
            "internal_trajectory_cost": (diagnostic_raw_cost if getattr(self, "raw_cost_constraint", False) else -discounted_return),
            "diagnostic_disc_cost": diagnostic_discounted_cost,
            "diagnostic_raw_cost": diagnostic_raw_cost,
        })
        return rollout

    # ---------------------------------------------------------------------- official sample transform
    def _transform(self, value: torch.Tensor) -> torch.Tensor:
        """Apply upstream ``transform_sample``: ``[1000,12,*] -> [100,120,*]`` bijectively."""

        old_time, old_batch = int(value.shape[0]), int(value.shape[1])
        if old_time == self.new_T and old_batch == self.new_B:
            return value
        if old_time * old_batch != self.new_T * self.new_B:
            raise ValueError(
                "transform_sample size mismatch: "
                f"{old_time}*{old_batch} != {self.new_T}*{self.new_B}")
        trailing = tuple(value.shape[2:])
        return (
            value.transpose(0, 1)
            .reshape(self.new_B, self.new_T, *trailing)
            .transpose(0, 1)
            .contiguous()
        )

    def _chunk_bootstrap(
        self,
        transformed_value: torch.Tensor,
        final_original_value: torch.Tensor,
    ) -> torch.Tensor:
        """Use the next chunk's first value except at each original trajectory's final chunk."""

        chunks_per_environment = self.n // self.new_T
        trailing = tuple(transformed_value.shape[2:])
        bootstrap = torch.empty(
            (self.new_B, *trailing),
            dtype=transformed_value.dtype,
            device=transformed_value.device,
        )
        for environment_index in range(self.num_envs):
            begin = environment_index * chunks_per_environment
            end = begin + chunks_per_environment
            # Chunk j bootstraps from state (j+1)*new_T, stored as time zero of chunk j+1.
            bootstrap[begin:end - 1] = transformed_value[0, begin + 1:end]
            # The original 1000-step bootstrap is used only by the last chunk and then masked by
            # timeout done=1, exactly matching the author's sampler/algorithm boundary.
            bootstrap[end - 1] = final_original_value[environment_index]
        return bootstrap

    @torch.no_grad()
    def _fixed_rollout_targets(self, rollout: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Build every target/advantage once; all eight PPO epochs reuse these detached tensors."""

        transform = self._transform
        reward = transform(rollout["reward"])
        internal_cost = -reward / self.cost_scale
        done = transform(rollout["done"])
        valid = valid_from_done(done)

        reward_value = transform(rollout["r_value"])
        cost_distribution = transform(rollout["c_dist"])
        location = transform(rollout["c_w_location"])
        shape = transform(rollout["c_w_alpha"])
        scale = transform(rollout["c_w_beta"])

        reward_bootstrap = self._chunk_bootstrap(
            reward_value, rollout["r_bootstrap_original"])
        cost_bootstrap = self._chunk_bootstrap(
            cost_distribution, rollout["c_bootstrap_original"])
        location_bootstrap = self._chunk_bootstrap(
            location, rollout["location_bootstrap_original"])
        shape_bootstrap = self._chunk_bootstrap(
            shape, rollout["alpha_bootstrap_original"])
        scale_bootstrap = self._chunk_bootstrap(
            scale, rollout["beta_bootstrap_original"])

        # Ordinary reward GAE uses no normalization and no extra gamma^t actor weighting.
        reward_advantage, reward_return = generalized_advantage_estimation(
            reward,
            reward_value,
            done,
            reward_bootstrap,
            self.gamma,
            self.gae_lambda,
        )

        # The auxiliary mean-cost target is the scalar GAE return of the mean of 25 signed atoms.
        cost_value = cost_distribution.mean(dim=-1)
        cost_bootstrap_value = cost_bootstrap.mean(dim=-1)
        _cost_advantage, cost_return = generalized_advantage_estimation(
            internal_cost,
            cost_value,
            done,
            cost_bootstrap_value,
            self.cost_discount,
            self.cost_gae_lambda,
        )

        # Density ratios use the sorted top-eight current atoms.  The helper performs per-tail-atom
        # T/B centering, division by exactly 5*std (no epsilon), clipping, then the done mask.
        sorted_cost = torch.sort(cost_distribution, dim=-1).values.detach()
        tail = sorted_cost[..., self.weibull_tail_ind:]
        probability_ratio = compute_prob_ratio(
            internal_cost,
            tail,
            location,
            shape,
            scale,
            done,
            location_bootstrap,
            shape_bootstrap,
            scale_bootstrap,
            self.gamma,
            log_clip_range=0.5,
            normalize=True,
        )

        # First compute ordinary quantile GAE, then apply the current-time density log-ratio weight.
        quantile_advantage, _quantile_return = gae_quantile_simple(
            internal_cost,
            cost_distribution,
            done,
            cost_bootstrap,
            self.gamma,
            self.cost_gae_lambda,
        )
        weighted_tail_advantage = (
            1.0 + torch.log(probability_ratio)
        ) * quantile_advantage[..., self.weibull_tail_ind:]
        constraint_advantage = weighted_tail_advantage[
            ..., self.target_outage_ind - self.weibull_tail_ind]

        # The helper freezes the one-step target and sorts its 25 atoms before QR regression.  QR's
        # pairwise reduction would be permutation-invariant, but the explicit sort is signature-tested.
        cost_distribution_target = quantile_target_estimation(
            internal_cost,
            cost_distribution,
            done,
            cost_bootstrap,
            self.cost_discount,
        )

        return {
            "obs": transform(rollout["obs"]),
            "prev_action": transform(rollout["prev_action"]),
            "prev_reward": transform(rollout["prev_reward"]),
            "action": transform(rollout["action"]),
            "mu_old": transform(rollout["mu_old"]),
            "log_std_old": transform(rollout["log_std_old"]),
            "h0": transform(rollout["h0"])[0],
            "c0": transform(rollout["c0"])[0],
            "reward_return": reward_return.detach(),
            "reward_advantage": reward_advantage.detach(),
            "cost_return": cost_return.detach(),
            "constraint_advantage": constraint_advantage.detach(),
            "cost_distribution_target": cost_distribution_target.detach(),
            "valid": valid.detach(),
            "probability_ratio": probability_ratio.detach(),
            "sorted_cost": sorted_cost,
        }

    # -------------------------------------------------------------------------------------- dual
    @staticmethod
    def discrete_order_statistic(values: Iterable[float], alpha: float) -> float:
        """Return upstream ``sort(C)[floor(n*(1-alpha))]`` without interpolation."""

        ordered = np.sort(np.asarray(list(values), dtype=np.float64).reshape(-1))
        if ordered.size == 0:
            raise ValueError("dual order statistic requires at least one trajectory")
        index = int(np.floor(ordered.size * (1.0 - float(alpha))))
        index = min(index, ordered.size - 1)
        return float(ordered[index])

    @torch.no_grad()
    def update_dual(self, internal_trajectory_cost: torch.Tensor) -> Tuple[float, float]:
        """Update immediately from the rolling deque; sum_norm deliberately applies no upper cap."""

        costs = internal_trajectory_cost.detach().cpu().numpy().reshape(-1)
        self.current_ep_costs.extend(float(value) for value in costs)
        empirical_cost_quantile = self.discrete_order_statistic(
            self.current_ep_costs, self.target_outage_prob)
        scaled_residual = (
            empirical_cost_quantile / self.cost_scale - self.cost_threshold_scaled)

        self.pid_i = max(0.0, self.pid_i + self.pid_Ki * scaled_residual)
        self.cost_penalty = max(0.0, self.pid_i)
        self.lambda_dual.fill_(self.cost_penalty)
        self.dual_update_count += 1

        self.last_dual_cost_quantile = empirical_cost_quantile
        self.last_dual_reward_quantile = -empirical_cost_quantile
        self.last_dual_residual = scaled_residual
        return empirical_cost_quantile, scaled_residual

    # -------------------------------------------------------------------------------------- loss
    def _loss(
        self,
        fixed: Dict[str, torch.Tensor],
        indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Evaluate one full recurrent minibatch and assemble the exact joint objective."""

        initial_state = RnnState(
            h=fixed["h0"][indices].unsqueeze(0).contiguous(),
            c=fixed["c0"][indices].unsqueeze(0).contiguous(),
        )
        mean, log_std, value, _state = self.model(
            fixed["obs"][:, indices],
            fixed["prev_action"][:, indices],
            fixed["prev_reward"][:, indices],
            initial_state,
        )
        new_log_probability = self._logp(fixed["action"][:, indices], mean, log_std)
        old_log_probability = self._logp(
            fixed["action"][:, indices],
            fixed["mu_old"][:, indices],
            fixed["log_std_old"][:, indices],
        )
        ratio = torch.exp(new_log_probability - old_log_probability)
        valid = fixed["valid"][:, indices]

        policy_loss, reward_policy_loss, constraint_policy_loss, clip_fraction = (
            self.split_clipped_policy_losses(
                ratio,
                fixed["reward_advantage"][:, indices],
                fixed["constraint_advantage"][:, indices],
                valid,
                self.ratio_clip,
                self.cost_penalty,
            )
        )

        reward_error = value.r_value - fixed["reward_return"][:, indices]
        reward_value_loss = self.value_loss_coeff * valid_mean(
            0.5 * reward_error.square(), valid)

        # L_C^V already contains 1/2; the total objective multiplies it by the official coefficient
        # 0.5, as in upstream cost_value_loss_coeff=0.5 for this frozen implementation.
        cost_mean_error = value.c_dist.mean(dim=-1) - fixed["cost_return"][:, indices]
        cost_mean_value_loss = valid_mean(0.5 * cost_mean_error.square(), valid)

        # These two helpers deliberately ignore their valid argument.  This counter-intuitive
        # reduction is executable upstream behavior and is protected by dedicated tests.
        quantile_loss = quantile_huber_loss(
            value.c_dist,
            fixed["cost_distribution_target"][:, indices],
            self.model.tau,
            valid,
        )
        tail_loss = weibull_tail_loss(
            value.c_dist,
            value.c_w_location,
            value.c_w_alpha,
            value.c_w_beta,
            self.model.c_tau,
            self.weibull_tail_ind,
            valid,
        )
        entropy = valid_mean(self._gaussian_entropy(log_std), valid)
        total_loss = (
            policy_loss
            + reward_value_loss
            + self.cost_value_loss_coeff * cost_mean_value_loss
            + self.quantile_loss_coeff * quantile_loss
            + self.weibull_loss_coeff * tail_loss
            - self.entropy_loss_coeff * entropy
        )

        support_margin = (
            torch.sort(value.c_dist, dim=-1).values[..., self.weibull_tail_ind:]
            - value.c_w_location.unsqueeze(-1)
        )
        loss_info = {
            "author/pi_total_loss": float(policy_loss.detach().item()),
            "author/pi_reward_loss": float(reward_policy_loss.detach().item()),
            "author/pi_constraint_loss": float(constraint_policy_loss.detach().item()),
            "author/reward_value_loss": float(reward_value_loss.detach().item()),
            "author/cost_mean_value_loss": float(cost_mean_value_loss.detach().item()),
            "author/quantile_loss": float(quantile_loss.detach().item()),
            "author/weibull_tail_loss": float(tail_loss.detach().item()),
            "author/entropy": float(entropy.detach().item()),
            "actor/clip_fraction": float(clip_fraction.detach().item()),
            "author/total_loss": float(total_loss.detach().item()),
            "author/weibull_shape_mean": float(value.c_w_alpha.detach().mean().item()),
            "author/weibull_scale_mean": float(value.c_w_beta.detach().mean().item()),
            "author/weibull_location_mean": float(value.c_w_location.detach().mean().item()),
            "author/weibull_support_margin_min": float(support_margin.detach().min().item()),
        }
        return total_loss, loss_info

    @torch.no_grad()
    def _full_population_behavior_policy_kl(
        self,
        fixed: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return analytic KL(behavior || current) on every recurrent chunk.

        fixed contains the behavior-policy Gaussian parameters and the exact
        recurrent state at every chunk boundary. Replaying the complete transformed
        population measures policy movement without resampling actions and without
        changing any optimizer or RNG state.
        """

        initial_state = RnnState(
            h=fixed["h0"].unsqueeze(0).contiguous(),
            c=fixed["c0"].unsqueeze(0).contiguous(),
        )
        current_mean, current_log_std, _value, _state = self.model(
            fixed["obs"],
            fixed["prev_action"],
            fixed["prev_reward"],
            initial_state,
        )

        # For diagonal Gaussians p=behavior and q=current:
        # KL(p||q)=sum(log sigma_q-log sigma_p
        #              +(sigma_p^2+(mu_p-mu_q)^2)/(2 sigma_q^2)-1/2).
        # Evaluate only this diagnostic in float64 and form variance *ratios*
        # directly.  This avoids float32 overflow from separately materializing
        # exp(2*log_std) while leaving the model, optimizer, and E8 schedule
        # completely unchanged.
        behavior_mean = fixed["mu_old"].to(dtype=torch.float64)
        behavior_log_std = fixed["log_std_old"].to(dtype=torch.float64)
        current_mean = current_mean.to(dtype=torch.float64)
        current_log_std = current_log_std.to(dtype=torch.float64)
        variance_ratio = torch.exp(
            2.0 * (behavior_log_std - current_log_std))
        squared_mean_ratio = (
            (behavior_mean - current_mean).square()
            * torch.exp(-2.0 * current_log_std)
        )
        kl_per_action = (
            current_log_std
            - behavior_log_std
            + 0.5 * (variance_ratio + squared_mean_ratio - 1.0)
        )
        valid = fixed["valid"].to(dtype=torch.float64)
        return valid_mean(kl_per_action.sum(dim=-1), valid)

    def _optimizer_epoch_minibatches(self) -> Tuple[torch.Tensor, ...]:
        """Shuffle recurrent chunk IDs once, then split them into equal minibatches.

        Native B12 freezes minibatches=1 and consequently retains its historical
        one-full-batch update. Registered adapters may opt into more batches only by
        signing and validating a different self.minibatches value.
        """

        # Historical/native agents (and their checkpoint reconstruction tests)
        # predate the registered minibatch field.  Their exact behavior is one
        # full-batch step per epoch, so absence must project to the frozen M=1
        # protocol rather than changing the author's B12 implementation.
        minibatch_count = int(getattr(self, "minibatches", 1))
        if minibatch_count <= 0 or self.new_B % minibatch_count != 0:
            raise RuntimeError(
                "QCPO recurrent chunks must divide evenly across optimizer minibatches")

        # Keep the author's NumPy RNG protocol: exactly one shuffle call per epoch,
        # followed by a deterministic equal split of that single permutation.
        indices_numpy = np.arange(self.new_B)
        np.random.shuffle(indices_numpy)
        minibatches_numpy = np.split(indices_numpy, minibatch_count)
        return tuple(
            torch.as_tensor(indices, dtype=torch.long, device=self.device)
            for indices in minibatches_numpy
        )

    def _optimizer_epoch_diagnostics(
        self,
        fixed: Dict[str, torch.Tensor],
        epoch_index: int,
    ) -> Dict[str, float]:
        """Return optional post-epoch diagnostics; native B12 intentionally adds none."""

        del fixed, epoch_index
        return {}

    def _optimize_fixed_rollout(
        self,
        fixed: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Apply all signed recurrent minibatch steps to one frozen rollout target."""

        loss_info: Dict[str, float] = {}
        epoch_diagnostics: Dict[str, float] = {}
        gradient_norm = None
        for epoch_index in range(self.epochs):
            for indices in self._optimizer_epoch_minibatches():
                self.optimizer.zero_grad(set_to_none=True)
                loss, loss_info = self._loss(fixed, indices)
                self._require_finite("total loss", loss)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip_grad_norm)
                self._require_finite(
                    "gradient norm", torch.as_tensor(gradient_norm))
                self.optimizer.step()
                self.update_counter += 1

            # This hook is diagnostic-only: its output never changes loop control,
            # the learning rate, the optimizer, or the signed fixed targets.
            epoch_diagnostics.update(
                self._optimizer_epoch_diagnostics(fixed, epoch_index))

        if gradient_norm is None:
            raise RuntimeError("QCPO optimizer schedule executed no minibatches")
        loss_info.update(epoch_diagnostics)
        loss_info["author/gradient_norm"] = float(
            torch.as_tensor(gradient_norm).detach().item())
        return loss_info

    @staticmethod
    def _require_finite(name: str, value: torch.Tensor) -> None:
        """Abort literal official numerics on NaN/Inf; never repair them with hidden epsilons."""

        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"QCPO author-aligned non-finite {name}")

    # ------------------------------------------------------------------------------------ training
    def train(self) -> None:
        """Run 836 nominal batches; batch one is dual/RMS-only and later batches take eight steps."""

        print(
            "QCPORefAuthorAlignedGPU: "
            f"identity={self.implementation_identity} env={self.env_name} seed={self.seed} "
            f"q={self.return_threshold} B={self.num_envs} T={self.n} "
            f"new_T={self.new_T} new_B={self.new_B} epochs={self.epochs}",
            flush=True,
        )
        for iteration in self.training_iterations():
            rollout = self._rollout_author_aligned()
            fixed = self._fixed_rollout_targets(rollout)

            # Upstream updates the empirical-quantile integral term before every epoch.  The first
            # 12 trajectories already move lambda; waiting for deque length 100 would be a drift.
            empirical_cost_quantile, scaled_residual = self.update_dual(
                rollout["internal_trajectory_cost"])

            # Upstream updates RMS from ``samples_agent.agent_inputs.observation`` *after*
            # ``transform_sample``. Thus RMS sees the fixed [100,120] augmented observations.
            # Old log-probabilities were computed before this update; batch one is sacrificed so
            # no PPO ratio mixes two normalization states.
            self.model.update_obs_rms(fixed["obs"])
            loss_info: Dict[str, float] = {}
            if int(iteration) == 0 and self.update_counter == 0:
                self.rms_only_batch_count += 1
            else:
                loss_info = self._optimize_fixed_rollout(fixed)

            # Literal density normalization can become NaN when an atom's log-ratio std is zero.
            # Check the fixed target after the optimizer gate so the failure retains all diagnostics.
            self._require_finite("density probability ratio", fixed["probability_ratio"])
            density_log = torch.log(fixed["probability_ratio"])
            extra = {
                **loss_info,
                "lambda/quantile_multiplier": float(self.cost_penalty),
                "lambda/quantile_residual": float(scaled_residual),
                "lambda/dual_window_count": int(len(self.current_ep_costs)),
                "lambda/dual_updates": int(self.dual_update_count),
                "quantile/dual_internal_cost_order_statistic": float(empirical_cost_quantile),
                "quantile/dual_implied_reward": float(-empirical_cost_quantile),
                "author/density_weight_mean": float(fixed["probability_ratio"].mean().item()),
                "author/density_log_weight_mean": float(density_log.mean().item()),
                "author/density_log_weight_min": float(density_log.min().item()),
                "author/density_log_weight_max": float(density_log.max().item()),
                "author/rms_count": float(self.model.obs_rms.count.item()),
                "training/optimizer_steps": int(self.update_counter),
                "training/rms_only_batches": int(self.rms_only_batch_count),
            }
            # Checkpoints retain the complete latest finite diagnostic row, not only scalar losses.
            # This makes an endpoint independently auditable even if W&B is temporarily unavailable.
            self.last_loss_info = dict(extra)
            metrics = self._log_core(
                iteration,
                rollout["disc_return"].detach().cpu().numpy(),
                rollout["diagnostic_disc_cost"].detach().cpu().numpy(),
                rollout["diagnostic_raw_cost"].detach().cpu().numpy(),
                extra=extra,
            )
            self._after_iteration(iteration, metrics)

    # ---------------------------------------------------------------------------------- evaluation
    @torch.no_grad()
    def evaluate_vec(
        self,
        vec_env,
        num_episodes,
        gamma,
        failure_alpha,
        return_threshold=None,
        show_progress=False,
        progress_desc="evaluation",
    ) -> Dict[str, object]:
        """Exact-N stochastic evaluation with the same action/reward/internal-cost histories."""

        from tqdm.auto import tqdm

        requested = int(num_episodes)
        if requested <= 0:
            raise ValueError("num_episodes must be positive")
        batch, horizon = int(vec_env.B), int(vec_env.n)
        rounds = int(math.ceil(requested / batch))
        all_returns, all_discounted_costs, all_raw_costs = [], [], []

        # Match CpuResetCollector: zero histories exist only at sampler construction.  Each
        # terminal transition's action/reward remains the next reset observation's history input.
        raw_observation = vec_env.reset()
        observation_cost = torch.zeros(batch, 1, device=self.device)
        stored_internal_cost = torch.zeros_like(observation_cost)
        previous_action = torch.zeros(batch, self.action_dim, device=self.device)
        previous_reward = torch.zeros(batch, device=self.device)
        recurrent = self._initial_rnn_state(batch)

        progress = None
        if show_progress:
            progress = tqdm(
                total=requested,
                desc=str(progress_desc),
                unit="episode",
                dynamic_ncols=True,
                mininterval=1.0,
                leave=True,
            )
        try:
            for _round in range(rounds):
                active = torch.ones(batch, dtype=torch.bool, device=self.device)
                episode_return = torch.zeros(batch, device=self.device)
                discounted_environment_cost = torch.zeros_like(episode_return)
                raw_environment_cost = torch.zeros_like(episode_return)

                for time_index in range(horizon):
                    augmented = torch.cat((raw_observation, observation_cost), dim=-1)
                    mean, log_std, _value, next_recurrent = self.model(
                        augmented, previous_action, previous_reward, recurrent)
                    action = mean + torch.exp(log_std) * torch.randn_like(mean)
                    step = vec_env.step(action)
                    ended = step.episode_end & active
                    if time_index + 1 < horizon and bool(ended.any()):
                        raise RuntimeError(
                            "QCPO_REF_AUTHOR_ALIGNED fixed-horizon evaluation received "
                            f"episode_end before step {horizon}: t={time_index}")
                    valid = active.to(step.reward.dtype)
                    reward = step.reward * valid
                    episode_return.add_((float(gamma) ** time_index) * reward)
                    discounted_environment_cost.add_(
                        (float(gamma) ** time_index) * step.cost * valid)
                    raw_environment_cost.add_(step.cost * valid)

                    # Apply the identical literal wrapper ordering used by training.
                    ended_column = ended.unsqueeze(-1)
                    next_observation_cost = torch.where(
                        ended_column,
                        torch.zeros_like(stored_internal_cost),
                        stored_internal_cost,
                    )
                    stored_internal_cost = torch.where(
                        ended_column,
                        torch.zeros_like(stored_internal_cost),
                        (step.cost if getattr(self, "raw_cost_constraint", False) else -reward).unsqueeze(-1),
                    )
                    observation_cost = next_observation_cost
                    previous_action = action
                    previous_reward = reward
                    raw_observation = step.observation

                    active = active & ~ended
                    keep = active.to(next_recurrent.h.dtype).view(1, batch, 1)
                    recurrent = RnnState(
                        h=next_recurrent.h * keep,
                        c=next_recurrent.c * keep,
                    )
                    if not bool(active.any()):
                        break

                # Author-mode vector execution deferred the horizon reset.  Reset exactly once
                # after the terminal transition while retaining previous_action/previous_reward.
                raw_observation = vec_env.reset()
                observation_cost.zero_()
                stored_internal_cost.zero_()

                all_returns.append(episode_return)
                all_discounted_costs.append(discounted_environment_cost)
                all_raw_costs.append(raw_environment_cost)
                if progress is not None:
                    progress.update(min(batch, requested - int(progress.n)))
        finally:
            if progress is not None:
                progress.close()

        returns = torch.cat(all_returns)[:requested].cpu().numpy().astype(np.float64)
        diagnostic_discounted = (
            torch.cat(all_discounted_costs)[:requested].cpu().numpy().astype(np.float64))
        diagnostic_raw = (
            torch.cat(all_raw_costs)[:requested].cpu().numpy().astype(np.float64))
        statistics = return_statistics(returns, failure_alpha, return_threshold)
        statistics.update({
            "quantile_q_est": float(statistics["quantile_return"]),
            "quantile_constraint_prob": statistics.get("empirical_prob"),
            "diagnostic_environment_cost_discounted_mean": float(
                diagnostic_discounted.mean()),
            "diagnostic_environment_cost_raw_mean": float(diagnostic_raw.mean()),
            "diagnostic_environment_cost_quantile_linear": linear_empirical_quantile(
                diagnostic_raw, failure_alpha),
            "diagnostic_reward_critic_calibration": None,
            "diagnostic_reward_critic_failure_probability": None,
            "diagnostic_reward_critic_mean": None,
            "diagnostic_reward_critic_std": None,
        })
        return statistics

    # --------------------------------------------------------------------------- checkpoint / resume
    def _checkpoint_identity(self) -> Dict[str, object]:
        """Return fields that transfer mode is not permitted to reinterpret."""

        identity = {
            "implementation_identity": self.implementation_identity,
            "author_upstream_sha": self.author_upstream_sha,
            "target_adapter": self.target_adapter,
            "history_timing": self.history_timing,
            "density_rule": self.density_rule,
            "dual_rule": self.dual_rule,
            "quantile_grid_identity": self.quantile_grid_identity,
            "weibull_identity": self.weibull_identity,
            "policy_clip_rule": self.policy_clip_rule,
            "rms_warmup_identity": self.rms_warmup_identity,
            "layout_identity": self.layout_identity,
            "environment": self.env_name,
            "seed": self.seed,
            "failure_alpha": self.failure_alpha,
            "return_threshold": self.return_threshold,
            "wandb_lineage": self.wandb_lineage,
            "atom_levels": tuple(float(value) for value in self.model.tau.cpu().tolist()),
            "target_atom_index": self.target_outage_ind,
            "tail_start_index": self.weibull_tail_ind,
            "layout": (self.n, self.num_envs, self.new_T, self.new_B),
            # The aligned class treats the full runner signature as non-transferable.  Therefore
            # the generic transfer switch cannot weaken implementation SHA, budget, W&B lineage,
            # optimizer, model, or mathematical-rule fields for a same-class checkpoint either.
            "experiment_signature": dict(getattr(self, "checkpoint_signature", {})),
        }
        identity.update(self._formal_execution_identity())
        return identity

    def _validate_vec_env_rng_state(
        self,
        states,
        envelope_name: str,
    ) -> str:
        """Validate one CPU or MJWarp RNG envelope and return its concrete backend.

        The historical CPU envelope intentionally has no ``backend/seed/slot`` fields; retaining
        its version/action-mode checks preserves MP/SYNC checkpoint compatibility.  MJWarp is not
        allowed through that legacy branch: when configured or declared, every slot must carry the
        exact v1 schema plus this run's seed, stable slot index, and bounded non-negative clocks.
        """

        if not isinstance(states, (list, tuple)) or len(states) != self.num_envs:
            raise ValueError(
                f"checkpoint {envelope_name} must contain exactly "
                f"{self.num_envs} vector slots")

        configured_backend = str(getattr(self, "vec_backend", "") or "").lower()
        declared_backends = {
            str(slot.get("backend"))
            for slot in states
            if isinstance(slot, dict) and "backend" in slot
        }
        if configured_backend:
            backend = configured_backend
        elif declared_backends == {"mjwarp"}:
            # Method-level legacy fixtures predate ``self.vec_backend``; infer only the unambiguous
            # strict GPU schema.  A mixed/unknown declaration is rejected below, never downgraded.
            backend = "mjwarp"
        elif not declared_backends or declared_backends == {"cpu"}:
            backend = "mp"
        else:
            raise ValueError(
                f"checkpoint {envelope_name} has inconsistent backend declarations")
        if backend not in {"mp", "sync", "mjwarp"}:
            raise ValueError(
                f"checkpoint {envelope_name} uses unsupported backend={backend!r}")

        expected_mode = str(getattr(
            self, "action_execution_mode", "author_raw_env_clamp"))
        expected_profile = str(getattr(
            self, "dynamics_profile", "legacy_euler_v1"))
        expected_seed = int(getattr(self, "seed", 0))
        max_counter = (1 << 31) - 1
        for slot_index, slot_state in enumerate(states):
            if not isinstance(slot_state, dict):
                raise ValueError(
                    f"checkpoint {envelope_name} slot {slot_index} is not a mapping")
            version = slot_state.get("version", -1)
            if (
                    isinstance(version, bool)
                    or not isinstance(version, Integral)
                    or int(version) != 1):
                raise ValueError(
                    f"checkpoint {envelope_name} slot {slot_index} "
                    "is not a version-1 mapping")
            if str(slot_state.get("action_mode", "")) != expected_mode:
                raise ValueError(
                    f"checkpoint {envelope_name} slot {slot_index} action mode mismatch")

            if backend != "mjwarp":
                # Historical CPU envelopes omit backend/profile.  The missing profile is Euler;
                # new native envelopes identify their simulator family as CPU, independently of
                # whether the vector orchestrator around those slots is MP or synchronous.
                declared = slot_state.get("backend")
                if declared is not None and str(declared) != "cpu":
                    raise ValueError(
                        f"checkpoint {envelope_name} slot {slot_index} backend mismatch")
                saved_profile = str(slot_state.get(
                    "dynamics_profile", "legacy_euler_v1"))
                if saved_profile != expected_profile:
                    raise ValueError(
                        f"checkpoint {envelope_name} slot {slot_index} dynamics "
                        f"profile mismatch: {saved_profile!r} != {expected_profile!r}")
                continue

            if set(slot_state) != _MJWARP_RNG_FIELDS:
                missing = sorted(_MJWARP_RNG_FIELDS - set(slot_state))
                extra = sorted(set(slot_state) - _MJWARP_RNG_FIELDS)
                raise ValueError(
                    f"checkpoint {envelope_name} slot {slot_index} MJWarp schema "
                    f"mismatch: missing={missing}, extra={extra}")
            if slot_state["backend"] != "mjwarp":
                raise ValueError(
                    f"checkpoint {envelope_name} slot {slot_index} backend mismatch")
            if str(slot_state["dynamics_profile"]) != expected_profile:
                raise ValueError(
                    f"checkpoint {envelope_name} slot {slot_index} dynamics "
                    f"profile mismatch: {slot_state['dynamics_profile']!r} "
                    f"!= {expected_profile!r}")

            # ``bool`` subclasses ``int``; accepting it would silently alias True with seed/slot 1.
            for field, expected in (("seed", expected_seed), ("slot", slot_index)):
                value = slot_state[field]
                if (
                        isinstance(value, bool)
                        or not isinstance(value, Integral)
                        or int(value) != expected):
                    raise ValueError(
                        f"checkpoint {envelope_name} slot {slot_index} {field} mismatch")
            for field in ("reset_count", "goal_count"):
                value = slot_state[field]
                if (
                        isinstance(value, bool)
                        or not isinstance(value, Integral)
                        or not 0 <= int(value) <= max_counter):
                    raise ValueError(
                        f"checkpoint {envelope_name} slot {slot_index} {field} "
                        "is not a non-negative int32 counter")
        return backend

    @staticmethod
    def _validate_mjwarp_sampler_counters(current, pre_reset, sampler_phase) -> None:
        """Prove the serialized MJWarp clocks match the declared sampler boundary."""

        if sampler_phase == QCPO_SAMPLER_PHASE_PRISTINE:
            for slot_index, slot_state in enumerate(current):
                if int(slot_state["reset_count"]) != 0 or int(slot_state["goal_count"]) != 0:
                    raise ValueError(
                        "checkpoint pristine MJWarp sampler has advanced counters at slot "
                        f"{slot_index}")
            return

        # At the carried boundary the collector captured ``pre_reset`` after horizon termination,
        # then called exactly one full reset.  That reset increments reset_count and zeros the goal
        # resampling clock; no other RNG event is legal between it and checkpoint serialization.
        for slot_index, (current_slot, pre_slot) in enumerate(zip(current, pre_reset)):
            if int(current_slot["reset_count"]) != int(pre_slot["reset_count"]) + 1:
                raise ValueError(
                    "checkpoint MJWarp sampler reset_count boundary mismatch at slot "
                    f"{slot_index}")
            if int(current_slot["goal_count"]) != 0:
                raise ValueError(
                    "checkpoint MJWarp carried sampler goal_count must be zero at slot "
                    f"{slot_index}")

    def checkpoint_state(self) -> Dict[str, object]:
        """Serialize shared model/RMS, one Adam, dual deque/PID, clocks, and counters."""

        # A rollout always begins with vec_env.reset().  Global NumPy/Torch RNGs alone cannot
        # reproduce that reset because Safety-Gymnasium keeps the authoritative task-layout RNG
        # inside each native environment.  Capture all twelve stable slots at the same rollout
        # boundary as model/optimizer state; old agents deliberately retain their old payloads.
        checkpoint_rng_state = getattr(self.vec_env, "checkpoint_rng_state", None)
        if not callable(checkpoint_rng_state):
            raise RuntimeError(
                "QCPO author-aligned vec_env lacks checkpoint_rng_state; "
                "a resumable next rollout cannot be guaranteed")
        vec_env_rng_state = copy.deepcopy(checkpoint_rng_state())
        current_sampler_state = getattr(self, "_sampler_state", None)
        pre_reset_rng_state = getattr(self, "_sampler_pre_reset_rng_state", None)
        if current_sampler_state is None and pre_reset_rng_state is None:
            sampler_phase = QCPO_SAMPLER_PHASE_PRISTINE
            sampler_state = None
        elif current_sampler_state is not None and pre_reset_rng_state is not None:
            sampler_phase = QCPO_SAMPLER_PHASE_CARRIED
            sampler_state = {
                name: value.detach().cpu().clone()
                for name, value in current_sampler_state.items()
            }
        else:
            raise RuntimeError(
                "QCPO author-aligned sampler state/pre-reset RNG boundary is inconsistent")
        runtime_phase = getattr(self, "_sampler_phase", sampler_phase)
        if runtime_phase != sampler_phase:
            raise RuntimeError(
                "QCPO author-aligned sampler phase disagrees with runtime state")

        # Validate the outbound envelope too.  A malformed backend response must fail at save time,
        # not produce a checkpoint that can only be diagnosed during a later recovery attempt.
        backend = self._validate_vec_env_rng_state(
            vec_env_rng_state, "vec_env_rng_state")
        if sampler_phase == QCPO_SAMPLER_PHASE_CARRIED:
            pre_reset_rng_state = copy.deepcopy(pre_reset_rng_state)
            pre_backend = self._validate_vec_env_rng_state(
                pre_reset_rng_state, "vec_env_pre_reset_rng_state")
            if pre_backend != backend:
                raise RuntimeError("QCPO author-aligned RNG envelopes use different backends")
        if backend == "mjwarp":
            self._validate_mjwarp_sampler_counters(
                vec_env_rng_state, pre_reset_rng_state, sampler_phase)

        actual_steps = int(self.main_env_steps + self.initialization_env_steps)
        return {
            "training_budget_semantics": self.training_budget_semantics,
            "checkpoint_identity": self._checkpoint_identity(),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lambda_dual": self.lambda_dual.detach().cpu(),
            "pid_i": float(self.pid_i),
            "cost_penalty": float(self.cost_penalty),
            "dual_cost_returns": list(self.current_ep_costs),
            "dual_update_count": int(self.dual_update_count),
            "last_dual_cost_quantile": float(self.last_dual_cost_quantile),
            "last_dual_reward_quantile": float(self.last_dual_reward_quantile),
            "last_dual_residual": float(self.last_dual_residual),
            "update_counter": int(self.update_counter),
            "rms_only_batch_count": int(self.rms_only_batch_count),
            "last_loss_info": dict(self.last_loss_info),
            "main_env_steps": int(self.main_env_steps),
            "initialization_env_steps": int(self.initialization_env_steps),
            "actual_env_steps": actual_steps,
            "total_env_steps": actual_steps,
            "initialization_cursor": int(self.initialization_cursor),
            "vec_env_rng_state": vec_env_rng_state,
            "vec_env_pre_reset_rng_state": pre_reset_rng_state,
            "sampler_phase": sampler_phase,
            "sampler_state": sampler_state,
        }

    def validate_checkpoint_state(self, state: Dict[str, object]) -> None:
        """Reject cross-version, cross-task, cross-seed, or cross-W&B state before tensor mutation."""

        if not isinstance(state, dict):
            raise ValueError("QCPO author-aligned checkpoint is missing agent state")
        if state.get("training_budget_semantics") != self.training_budget_semantics:
            raise ValueError("QCPO author-aligned training budget semantics mismatch")
        saved_identity = state.get("checkpoint_identity")
        expected_identity = self._checkpoint_identity()
        if saved_identity != expected_identity:
            raise ValueError(
                "QCPO author-aligned checkpoint identity mismatch: "
                f"saved={saved_identity!r}, requested={expected_identity!r}")

        main_steps = int(state.get("main_env_steps", -1))
        initialization_steps = int(state.get("initialization_env_steps", -1))
        actual_steps = int(state.get("actual_env_steps", -1))
        if main_steps < 0 or initialization_steps < 0:
            raise ValueError("checkpoint is missing non-negative environment-step clocks")
        if actual_steps != main_steps + initialization_steps:
            raise ValueError("checkpoint environment-step clocks are inconsistent")
        if int(state.get("total_env_steps", -1)) != actual_steps:
            raise ValueError("checkpoint total_env_steps does not equal actual_env_steps")

        restore_rng_state = getattr(self.vec_env, "restore_rng_state", None)
        if not callable(restore_rng_state):
            raise ValueError(
                "QCPO author-aligned vec_env lacks restore_rng_state; "
                "cannot resume the next rollout reproducibly")
        current_rng = state.get("vec_env_rng_state")
        backend = self._validate_vec_env_rng_state(
            current_rng, "vec_env_rng_state")

        # Absence is accepted only for an old carried CPU checkpoint.  All newly written payloads,
        # and every pristine payload, carry the phase explicitly.
        sampler = state.get("sampler_state")
        pre_reset_rng = state.get("vec_env_pre_reset_rng_state")
        sampler_phase = state.get("sampler_phase")
        if sampler_phase is None and sampler is not None and pre_reset_rng is not None:
            sampler_phase = QCPO_SAMPLER_PHASE_CARRIED
        if sampler_phase not in {
                QCPO_SAMPLER_PHASE_PRISTINE, QCPO_SAMPLER_PHASE_CARRIED}:
            raise ValueError("checkpoint sampler_phase is missing or unsupported")
        if sampler_phase == QCPO_SAMPLER_PHASE_PRISTINE:
            if sampler is not None or pre_reset_rng is not None:
                raise ValueError(
                    "checkpoint pristine sampler must have null sampler/pre-reset state")
            pristine_scalars = {
                "main_env_steps": main_steps,
                "initialization_env_steps": initialization_steps,
                "update_counter": int(state.get("update_counter", -1)),
                "rms_only_batch_count": int(state.get("rms_only_batch_count", -1)),
                "dual_update_count": int(state.get("dual_update_count", -1)),
            }
            nonzero = {name: value for name, value in pristine_scalars.items() if value != 0}
            if nonzero or list(state.get("dual_cost_returns", ())) != []:
                raise ValueError(
                    "checkpoint pristine sampler has advanced training state: "
                    f"{nonzero}")
            if backend == "mjwarp":
                self._validate_mjwarp_sampler_counters(
                    current_rng, None, sampler_phase)
            return

        pre_backend = self._validate_vec_env_rng_state(
            pre_reset_rng, "vec_env_pre_reset_rng_state")
        if pre_backend != backend:
            raise ValueError("checkpoint RNG envelopes use different backends")
        sampler = state.get("sampler_state")
        expected_shapes = {
            "raw_observation": (self.num_envs, self.state_dim),
            "observation_cost": (self.num_envs, 1),
            "stored_internal_cost": (self.num_envs, 1),
            "previous_action": (self.num_envs, self.action_dim),
            "previous_reward": (self.num_envs,),
            "recurrent_h": (1, self.num_envs, self.lstm_size),
            "recurrent_c": (1, self.num_envs, self.lstm_size),
        }
        if not isinstance(sampler, dict) or set(sampler) != set(expected_shapes):
            raise ValueError("checkpoint sampler_state fields are incomplete")
        for name, shape in expected_shapes.items():
            value = sampler[name]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(
                    f"checkpoint sampler_state {name} shape mismatch: "
                    f"{getattr(value, 'shape', None)} != {shape}")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"checkpoint sampler_state {name} is non-finite")
        if backend == "mjwarp":
            self._validate_mjwarp_sampler_counters(
                current_rng, pre_reset_rng, sampler_phase)

    def load_checkpoint_state(self, state: Dict[str, object]) -> None:
        """Restore all agent tensors/scalars; outer checkpoint code restores Python/NumPy/Torch RNG."""

        self.validate_checkpoint_state(state)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.lambda_dual.copy_(state["lambda_dual"].to(self.device))
        self.pid_i = float(state["pid_i"])
        self.cost_penalty = float(state["cost_penalty"])
        self.current_ep_costs.clear()
        self.current_ep_costs.extend(float(value) for value in state["dual_cost_returns"])
        self.dual_update_count = int(state["dual_update_count"])
        self.last_dual_cost_quantile = float(state["last_dual_cost_quantile"])
        self.last_dual_reward_quantile = float(state["last_dual_reward_quantile"])
        self.last_dual_residual = float(state["last_dual_residual"])
        self.update_counter = int(state["update_counter"])
        self.rms_only_batch_count = int(state["rms_only_batch_count"])
        self.last_loss_info = dict(state.get("last_loss_info") or {})
        self.main_env_steps = int(state["main_env_steps"])
        self.initialization_env_steps = int(state["initialization_env_steps"])
        self.actual_env_steps = int(state["actual_env_steps"])
        self.total_env_steps = int(state["total_env_steps"])
        self.initialization_cursor = int(state.get("initialization_cursor", 0))

        sampler_phase = state.get("sampler_phase")
        if sampler_phase is None:
            sampler_phase = QCPO_SAMPLER_PHASE_CARRIED  # old CPU carried checkpoints
        restore_rng_state = self.vec_env.restore_rng_state
        if sampler_phase == QCPO_SAMPLER_PHASE_PRISTINE:
            # Do not reset here.  The first call to _rollout_author_aligned must remain responsible
            # for constructing all-zero histories and performing exactly one first layout reset.
            restore_rng_state(copy.deepcopy(state["vec_env_rng_state"]))
            self._sampler_state = None
            self._sampler_pre_reset_rng_state = None
            self._sampler_phase = QCPO_SAMPLER_PHASE_PRISTINE
            return

        # A carried checkpoint is written after CpuResetCollector-equivalent reset. Restoring only its
        # post-reset RNG cannot recreate MuJoCo's already-sampled layout/physics in a fresh process.
        # Rewind each slot to pre-reset RNG, execute that one reset again, then compare the exact
        # native observation before allowing the next rollout to use the carried AgentInputs.
        pre_reset_rng = state["vec_env_pre_reset_rng_state"]
        restore_rng_state(copy.deepcopy(pre_reset_rng))
        reconstructed_observation = self.vec_env.reset()
        saved_sampler = state["sampler_state"]
        expected_observation = saved_sampler["raw_observation"].to(self.device)
        if not torch.equal(reconstructed_observation, expected_observation):
            raise RuntimeError(
                "QCPO author-aligned resume could not reconstruct the next reset observation")
        self._sampler_state = {
            name: value.to(self.device).clone()
            for name, value in saved_sampler.items()
        }
        self._sampler_state["raw_observation"] = reconstructed_observation.clone()
        self._sampler_pre_reset_rng_state = copy.deepcopy(pre_reset_rng)
        self._sampler_phase = QCPO_SAMPLER_PHASE_CARRIED

        # For MJWarp, counter equality is cheap and unambiguous after reconstruction.  Comparing
        # it here proves restore+one-reset reached the exact post-reset clock serialized at save.
        if str(getattr(self, "vec_backend", "")).lower() == "mjwarp":
            reconstructed_rng = self.vec_env.checkpoint_rng_state()
            self._validate_vec_env_rng_state(
                reconstructed_rng, "reconstructed_vec_env_rng_state")
            if reconstructed_rng != state["vec_env_rng_state"]:
                raise RuntimeError(
                    "QCPO author-aligned MJWarp resume reconstructed different RNG clocks")

    def get_training_summary(self) -> Dict[str, object]:
        """Expose the frozen identity and latest dual/training diagnostics to runner JSON."""

        summary = {
            "objective": "mean_reward_subject_to_reward_outage_via_C_equals_negative_G",
            "implementation_identity": self.implementation_identity,
            "author_upstream_sha": self.author_upstream_sha,
            "target_adapter": self.target_adapter,
            "history_timing": self.history_timing,
            "density_rule": self.density_rule,
            "dual_rule": self.dual_rule,
            "quantile_grid_identity": self.quantile_grid_identity,
            "weibull_identity": self.weibull_identity,
            "policy_clip_rule": self.policy_clip_rule,
            "rms_warmup_identity": self.rms_warmup_identity,
            "layout_identity": self.layout_identity,
            "lambda_final": float(self.cost_penalty),
            "pid_i": float(self.pid_i),
            "dual_window_count": int(len(self.current_ep_costs)),
            "dual_update_count": int(self.dual_update_count),
            "last_dual_internal_cost_quantile": float(self.last_dual_cost_quantile),
            "last_dual_implied_reward_quantile": float(self.last_dual_reward_quantile),
            "optimizer_steps": int(self.update_counter),
            "rms_only_batches": int(self.rms_only_batch_count),
            "main_env_steps": int(self.main_env_steps),
            "actual_env_steps": int(self.actual_env_steps),
            "return_threshold": float(self.return_threshold),
            "failure_alpha": float(self.failure_alpha),
            "last_loss_info": dict(self.last_loss_info),
            "sampler_phase": getattr(
                self, "_sampler_phase", QCPO_SAMPLER_PHASE_PRISTINE),
        }
        summary.update(self._formal_execution_identity())
        return summary


__all__ = [
    "QCPORefAuthorAlignedGPU",
    "QCPO_AUTHOR_DENSITY_RULE",
    "QCPO_AUTHOR_DUAL_RULE",
    "QCPO_AUTHOR_HISTORY_TIMING",
    "QCPO_AUTHOR_IMPLEMENTATION_ID",
    "QCPO_AUTHOR_LAYOUT",
    "QCPO_AUTHOR_POLICY_CLIP_RULE",
    "QCPO_AUTHOR_QUANTILE_GRID",
    "QCPO_AUTHOR_RMS_WARMUP",
    "QCPO_AUTHOR_TARGET_ADAPTER",
    "QCPO_AUTHOR_UPSTREAM_SHA",
    "QCPO_AUTHOR_WEIBULL_IDENTITY",
    "QCPO_FORMAL_DYNAMICS_PROFILE",
    "QCPO_FORMAL_TRAIN_WARP_DETERMINISM",
    "QCPO_FORMAL_TRAIN_WARP_DETERMINISTIC_MAX_RECORDS",
    "QCPO_FORMAL_WANDB_PROJECT",
    "QCPO_SAMPLER_PHASE_CARRIED",
    "QCPO_SAMPLER_PHASE_PRISTINE",
]
