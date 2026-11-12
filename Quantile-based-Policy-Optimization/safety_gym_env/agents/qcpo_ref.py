# -*- coding: utf-8 -*-
"""
Paper-oriented QCPO baseline for a reward-outage constraint.

The external problem is ``max E[G]`` subject to ``P(G<q)<=alpha``. Rather than manually mirror
the paper's cost formulas, this agent defines the internal auxiliary cost ``C=-G`` and threshold
``d=-q``. For continuous returns, ``Q^C_{1-alpha}=-Q^G_alpha``, so the paper Lagrangian is

``L = E[G] - lambda*(Q^C_{1-alpha}-d) = E[G] + lambda*(Q^G_alpha-q)``.

Accordingly the code uses the paper signs ``A_reward-lambda*A_cost_quantile`` and
``lambda <- [lambda+eta*(Q^C_{1-alpha}-d)]_+`` while all public metrics remain reward-side.
"""
from collections import deque

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from .common import lr_lambda
from .vec_base import VecAgentBase
from .qcpo_ref_model import RewardQuantileModel
from .dist_rl_utils_ref import (
    masked_gae, quantile_huber_loss, upper_weibull_fit_loss,
    shifted_weibull_pdf, paper_density_weight)
from utils.risk_metrics import linear_empirical_quantile


class QCPORefGPU(VecAgentBase):
    """PPO baseline using a signed ``C=-G`` distribution and the paper's right-tail estimator."""

    def __init__(self, args, env):
        super().__init__(args, env)
        self._require_return_threshold()
        self.gae_lambda = float(getattr(args, "gae_lambda", 0.97))
        self.ratio_clip = float(getattr(args, "ratio_clip", 0.1))
        self.epochs = max(1, int(getattr(args, "epochs", 8)))
        self.entropy_coef = float(getattr(args, "entropy_loss_coeff", 0.0))
        self.value_loss_coef = float(getattr(args, "value_loss_coeff", 1.0))
        self.quantile_loss_coef = float(getattr(args, "quantile_loss_coeff", 1.0))
        self.weibull_loss_coef = float(getattr(args, "weibull_loss_coeff", 1.0))
        self.clip_grad_norm = float(getattr(args, "clip_grad_norm", 10.0))
        self.normalize_combined_advantage = bool(
            getattr(args, "normalize_combined_advantage", True))
        # The existing 19-atom grid contains both the public reward alpha=.20 and the paper-side
        # cost level 1-alpha=.80 exactly. Quantile learning itself is performed only for C=-G.
        levels = [0.05 * i for i in range(1, 20)]
        hidden = getattr(args, "critic_hidden", [512, 512])
        if isinstance(hidden, str):
            hidden = [int(x) for x in hidden.split(",") if x.strip()]

        # Match DQC finite's state/history conditioning without sharing any actor parameters. The
        # actor's rollout-time h0/c0 summarize observation history before consuming s_t; remaining
        # time removes finite-horizon state aliasing. This identity is deliberately orthogonal to
        # the paper actor weighting/formulation fields below.
        expected_critic_input = "finite_obs_remaining_actor_h0c0_v1"
        self.qcpo_ref_critic_input = str(getattr(
            args, "qcpo_ref_critic_input", expected_critic_input))
        self.horizon_mode = str(getattr(args, "horizon_mode", "finite"))
        self.critic_time_feature = str(getattr(
            args, "critic_time_feature", "remaining_fraction"))
        self.critic_actor_context = bool(getattr(args, "critic_actor_context", True))
        critic_identity = {
            "qcpo_ref_critic_input": (
                self.qcpo_ref_critic_input, expected_critic_input),
            "horizon_mode": (self.horizon_mode, "finite"),
            "critic_time_feature": (
                self.critic_time_feature, "remaining_fraction"),
            "critic_actor_context": (self.critic_actor_context, True),
        }
        critic_mismatches = [
            f"{name}={actual!r} (required {expected!r})"
            for name, (actual, expected) in critic_identity.items()
            if actual != expected
        ]
        if critic_mismatches:
            raise ValueError(
                "QCPO_REF finite critic identity mismatch: "
                + ", ".join(critic_mismatches))

        # The MLP width is identical to DQC finite's state side. Action is intentionally absent:
        # QCPO_REF learns state-value/quantile heads, whereas DQC's distributional critic is Q(s,a).
        self.critic_state_dim = int(
            self.state_dim + 1 + 2 * self.lstm_size)
        self.critic = RewardQuantileModel(
            self.critic_state_dim,
            quantile_levels=levels, hidden=hidden).to(self.device)
        self.cost_quantile_level = 1.0 - self.failure_alpha
        self.cost_threshold = -self.return_threshold
        self.constraint_index = self.critic.quantile_index(self.cost_quantile_level)
        self.q75_index = self.critic.quantile_index(0.75)
        self.q85_index = self.critic.quantile_index(0.85)

        # Formulation identity is explicit so an old reward-left-tail checkpoint cannot silently
        # enter this lineage even when network tensor shapes happen to match.
        expected_formulation = "paper_negated_return_eq275_discounted_bh_v2"
        self.qcpo_ref_formulation = str(getattr(
            args, "qcpo_ref_formulation", expected_formulation))
        if self.qcpo_ref_formulation != expected_formulation:
            raise ValueError(
                f"unsupported QCPO_REF formulation {self.qcpo_ref_formulation!r}; "
                f"expected {expected_formulation!r}")
        # These two fields distinguish the missing outer occupancy discount from the old v1 loss,
        # and distinguish paper's fixed B*H estimator from a valid-row conditional mean. Keeping
        # them as explicit state prevents a shape-compatible v1 actor from entering v2 via transfer.
        self.qcpo_ref_actor_weighting = str(getattr(
            args, "qcpo_ref_actor_weighting", "discounted_gamma_t"))
        self.qcpo_ref_actor_reduction = str(getattr(
            args, "qcpo_ref_actor_reduction", "fixed_BH"))
        if self.qcpo_ref_actor_weighting != "discounted_gamma_t":
            raise ValueError(
                "QCPO_REF requires qcpo_ref_actor_weighting='discounted_gamma_t'")
        if self.qcpo_ref_actor_reduction != "fixed_BH":
            raise ValueError("QCPO_REF requires qcpo_ref_actor_reduction='fixed_BH'")
        self.density_log_clip = float(getattr(args, "density_log_clip", 0.5))
        self.weibull_fit_min = float(getattr(args, "weibull_fit_min", 0.75))
        self.dual_window_size = int(getattr(args, "dual_window_size", 100))
        if self.dual_window_size <= 0:
            raise ValueError("dual_window_size must be positive")
        self.dual_cost_returns = deque(maxlen=self.dual_window_size)
        # Keep the user-facing base LR on Adam and let LambdaLR return only the dimensionless
        # batch-index ratio (b/(b+k))^c. Since QPO uses a/(b+k)^c with
        # a=b^c*actor_lr0, both forms yield the same absolute LR at every k.
        self.actor_lr_b = float(args.theta_b)
        self.actor_lr_c = float(args.theta_c)
        self.actor_optimizer = Adam(
            self.actor.parameters(), lr=float(getattr(args, "ref_lr", 2e-4)), eps=1e-5)
        self.actor_scheduler = self._new_actor_scheduler()
        self.critic_optimizer = Adam(
            self.critic.parameters(), lr=float(getattr(args, "critic_lr", 3e-4)), eps=1e-5)
        # Quantile multiplier is dimensionless and has a formulation-specific checkpoint key.
        self.lambda_quantile = torch.tensor(
            float(getattr(args, "penalty_init", 0.0)), dtype=torch.float32, device=self.device)
        self.lambda_dual = self.lambda_quantile            # compatibility for summary consumers
        self.lambda_max = float(getattr(args, "lambda_max", 100.0))
        self.lambda_a = float(args.lambda_a)
        self.lambda_b = float(args.lambda_b)
        self.lambda_c = float(args.lambda_c)
        self.dual_update_count = 0
        self.last_dual_quantile = 0.0
        self.last_dual_cost_quantile = 0.0

    def _actor_lr_ratio(self, batch_index):
        """Return (b/(b+k))^c, the common rollout-batch actor LR multiplier."""
        k = max(0, int(batch_index))
        return (self.actor_lr_b / (self.actor_lr_b + k)) ** self.actor_lr_c

    def _new_actor_scheduler(self, last_epoch=-1):
        """
        Build a scheduler whose index advances once per collected rollout.

        LambdaLR performs its initial step during construction: ``last_epoch=-1`` therefore
        exposes ratio(0), while ``last_epoch=completed-1`` exposes ratio(completed) and upgrades
        a legacy checkpoint for its next rollout batch.
        """
        return LambdaLR(
            self.actor_optimizer,
            lr_lambda=self._actor_lr_ratio,
            last_epoch=int(last_epoch),
        )

    def _set_actor_lr_batch_index(self, batch_index):
        """
        Move LambdaLR to an explicit rollout-batch index without an optimizer step.

        DQC needs this operation during critic-only warm-up, where calling scheduler.step()
        before any actor optimizer step would emit PyTorch's ordering warning. Using the same
        helper here keeps REF and DQC checkpoint state identical in meaning: last_epoch=k and
        optimizer lr=base_lr*(b/(b+k))^c.
        """
        k = max(0, int(batch_index))
        ratio = self._actor_lr_ratio(k)
        learning_rates = [float(base_lr) * ratio for base_lr in self.actor_scheduler.base_lrs]
        for group, learning_rate in zip(
                self.actor_optimizer.param_groups, learning_rates):
            group["lr"] = learning_rate
        self.actor_scheduler.last_epoch = k
        self.actor_scheduler._last_lr = learning_rates
        self.actor_scheduler._step_count = max(
            int(getattr(self.actor_scheduler, "_step_count", 0)), k + 1)

    def _restore_legacy_actor_scheduler(self, completed_batches):
        """Attach the new schedule to an old constant-LR optimizer at its next batch index."""
        completed = max(0, int(completed_batches))
        # Optimizer.load_state_dict restores the old base LR. Freeze it as LambdaLR's base so the
        # upgrade preserves the configured ref_lr instead of silently assuming the default.
        for group in self.actor_optimizer.param_groups:
            group["initial_lr"] = float(group["lr"])
        self.actor_scheduler = self._new_actor_scheduler()
        self._set_actor_lr_batch_index(completed)

    # ============================================================ finite critic inputs ============================================================
    def remaining_fraction(self, time_indices):
        """
        Map zero-based episode indices to `(H-t)/H`, exactly as DQC finite does.

        For H=1000 the current-row feature is 1 at t=0 and .001 at t=999. A boundary next-row query
        uses t+1 and therefore receives 0 at the formal horizon; its output is masked from bootstrap.
        """
        time = torch.as_tensor(
            time_indices, dtype=torch.float32, device=self.device)
        return (float(self.horizon) - time) / float(self.horizon)

    def _critic_features(self, states, time_indices=None, h=None, c=None):
        """
        Concatenate raw observation, remaining time, and detached actor recurrent context.

        Leading dimensions must agree across all inputs. Detaching h/c makes critic backward
        structurally unable to update the actor even though the critic consumes actor memory.
        """
        if time_indices is None or h is None or c is None:
            raise ValueError(
                "QCPO_REF critic queries require time indices and actor h/c context")
        if h.shape[:-1] != states.shape[:-1] or c.shape[:-1] != states.shape[:-1]:
            raise ValueError("QCPO_REF critic state and actor context shapes do not match")
        if h.shape[-1] != self.lstm_size or c.shape[-1] != self.lstm_size:
            raise ValueError("QCPO_REF critic actor context width does not match lstm_size")

        remaining = self.remaining_fraction(time_indices).unsqueeze(-1)
        if remaining.shape[:-1] != states.shape[:-1]:
            remaining = remaining.expand(*states.shape[:-1], 1)
        return torch.cat(
            (states, remaining.to(states.dtype), h.detach(), c.detach()), dim=-1)

    def _critic_rollout_features(self, rollout):
        """Build current features from `S[t], t, h0[t], c0[t]` for all vector rows."""
        states = rollout["S"]
        T, B = states.shape[:2]
        time = torch.arange(
            T, dtype=torch.float32, device=states.device).view(T, 1).expand(T, B)
        return self._critic_features(
            states, time, rollout.get("h0"), rollout.get("c0"))

    def _critic_next_rollout_features(self, rollout):
        """
        Build successor features from `next_states[t], t+1, next_h[t], next_c[t]`.

        next_h/c are captured after the behavior actor consumes s_t and before an ended vector slot
        is reset. For a continuing row they are exactly the pre-action memory for s_{t+1}; boundary
        rows remain available for shape consistency but finite masks remove every bootstrap use.
        """
        next_states = rollout["next_states"]
        T, B = next_states.shape[:2]
        next_time = (
            torch.arange(
                T, dtype=torch.float32, device=next_states.device) + 1.0
        ).view(T, 1).expand(T, B)
        return self._critic_features(
            next_states, next_time,
            rollout.get("next_h"), rollout.get("next_c"))

    def _process_rollout(self, rollout):
        """
        Build finite-return mean GAE and paper Eq. (275)'s ``C=-G`` quantile advantage.

        The common vector collector permits bootstrap at a time-limit truncation for continuing
        algorithms. QCPO_REF's target, however, is exactly the evaluated H-step return, so this
        method deliberately zeroes bootstrap at both termination and truncation.
        """
        # Snapshot the behavior actor's stored context rather than recomputing it after any PPO
        # epoch. The resulting critic inputs therefore stay fixed across all eight regression passes.
        current_features = self._critic_rollout_features(rollout)
        next_features = self._critic_next_rollout_features(rollout)
        with torch.no_grad():
            current = self.critic(current_features)
            next_info = self.critic(next_features)
        valid = rollout["valid"].float()
        episode_end = rollout["terminated"] | rollout["truncated"]
        continuation = (rollout["valid"] & ~episode_end).float()
        finite_bootstrap = continuation
        reward = rollout["R"]

        # The ordinary critic estimates finite reward return and supplies the max-mean advantage.
        reward_advantage, reward_return = masked_gae(
            reward, current.mean_value, next_info.mean_value,
            finite_bootstrap, continuation, self.gamma, self.gae_lambda)

        # Negating only this auxiliary signal turns the reward outage into the paper's upper-cost
        # quantile constraint. The Safety-Gym diagnostic cost channel is intentionally untouched.
        internal_cost = -reward
        q_current = current.quantiles[..., self.constraint_index]
        q_next = next_info.quantiles[..., self.constraint_index]
        density_weight, support_violation, clipped_log_ratio = paper_density_weight(
            internal_cost, q_current,
            (current.location, current.scale, current.shape),
            (next_info.location, next_info.scale, next_info.shape),
            finite_bootstrap, self.gamma, log_clip=self.density_log_clip)

        # First compute the ordinary quantile GAE, matching the reference implementation's
        # gae_quantile_simple(). Then multiply the complete statewise advantage by the bounded
        # Eq. (274) weight; the old code instead inserted raw mu into a theorem-intermediate
        # pseudo-reward, which differs by a Q_current*(mu-1) term.
        cost_quantile_advantage, _ = masked_gae(
            internal_cost, q_current, q_next,
            finite_bootstrap, continuation, self.gamma, self.gae_lambda)
        quantile_advantage = density_weight * cost_quantile_advantage

        # QR-TD learns the complete signed C=-G distribution with the same finite boundary.
        quantile_target = (
            internal_cost.unsqueeze(-1)
            + self.gamma * finite_bootstrap.unsqueeze(-1) * next_info.quantiles
        )
        return {
            "reward_advantage": reward_advantage,
            "reward_return": reward_return,
            "quantile_advantage": quantile_advantage,
            "quantile_target": quantile_target,
            "support_violation": support_violation,
            "density_weight": density_weight,
            "clipped_log_ratio": clipped_log_ratio,
            "finite_bootstrap": finite_bootstrap,
            "valid": valid,
        }

    def _combined_actor_advantage(self, processed):
        """
        Combine and population-normalize the complete paper-sign Lagrangian advantage.

        Normalization uses only valid rows and intentionally does not subtract their mean. The
        outer ``gamma**t`` occupancy measure is applied later to the PPO surrogate, so the scale
        statistic describes ``A_reward-lambda*A_cost_quantile`` rather than a time-reweighted mix.
        """
        combined = (
            processed["reward_advantage"]
            - self.lambda_quantile.detach() * processed["quantile_advantage"]
        )
        valid = processed["valid"]
        if self.normalize_combined_advantage:
            selected = combined[valid.bool()]
            combined = combined / selected.std(unbiased=False).clamp(min=1e-6)
        return combined

    @staticmethod
    def discounted_ppo_loss(
            new_logp, old_logp, combined_advantage, valid, clip_ratio,
            gamma, fixed_denominator):
        """
        Return the discounted-occupancy PPO loss and valid-row diagnostics.

        The optimized estimator is ``-sum(gamma**t * mask * clipped_surrogate)/(B*H)``. Padding
        therefore contributes an exact zero without changing the denominator. Ratio diagnostics
        remain conditional on valid rows because they describe policy displacement, not objective
        mass. ``combined_advantage`` must already contain both reward and quantile terms so the
        outer discount cannot accidentally be applied to only one component.
        """
        denominator = int(fixed_denominator)
        if denominator <= 0:
            raise ValueError("fixed_denominator must be positive")

        ratio = torch.exp(new_logp - old_logp)
        clipped = torch.clamp(
            ratio, 1.0 - float(clip_ratio), 1.0 + float(clip_ratio))
        surrogate = torch.minimum(
            ratio * combined_advantage, clipped * combined_advantage)
        mask = valid.to(surrogate.dtype)

        # Rollout tensors are [T,B]; reshape [T] to [T,1] so the same occupancy coefficient is
        # broadcast across all vector environments at timestep t.
        time_index = torch.arange(
            surrogate.shape[0], dtype=surrogate.dtype, device=surrogate.device)
        occupancy = torch.pow(
            torch.as_tensor(float(gamma), dtype=surrogate.dtype, device=surrogate.device),
            time_index,
        ).unsqueeze(-1)
        loss = -(surrogate * occupancy * mask).sum() / float(denominator)

        clipped_samples = (ratio - 1.0).abs() > float(clip_ratio)
        valid_count = torch.clamp(mask.sum(), min=1.0)
        clip_fraction = (clipped_samples.to(mask.dtype) * mask).sum() / valid_count
        return loss, ratio, clip_fraction

    def _actor_epoch(self, rollout, processed):
        """Apply discounted-occupancy PPO to the complete paper-sign combined advantage."""
        combined = self._combined_actor_advantage(processed)
        valid = processed["valid"]
        mean, log_std, _ = self.actor(rollout["S"])
        new_logp = self._logp_from_params(rollout["A"], mean, log_std)
        loss, ratio, clip_fraction = self.discounted_ppo_loss(
            new_logp, rollout["logp"], combined.detach(), valid, self.ratio_clip,
            self.gamma, self.num_envs * self.horizon)
        entropy = (self._entropy(log_std) * valid).sum() / valid.sum().clamp(min=1.0)
        total = loss - self.entropy_coef * entropy
        self.actor_optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad_norm)
        self.actor_optimizer.step()
        approx_kl = ((rollout["logp"] - new_logp) * valid).sum() / valid.sum().clamp(min=1.0)
        return {
            "actor/loss": float(loss.item()),
            "actor/entropy": float(entropy.item()),
            "actor/approx_kl": float(approx_kl.item()),
            "actor/clip_fraction": float(clip_fraction.item()),
            "actor/ratio_mean": float((ratio * valid).sum().item() / valid.sum().clamp(min=1.0).item()),
        }

    def _critic_epoch(self, rollout, processed):
        """Fit mean reward, signed C=-G atoms, and detached upper-tail Weibull calibration."""
        info = self.critic(self._critic_rollout_features(rollout))
        valid = processed["valid"]
        denom = valid.sum().clamp(min=1.0)
        value_loss = 0.5 * (((info.mean_value - processed["reward_return"]).square()
                             * valid).sum() / denom)
        quantile_loss = quantile_huber_loss(
            info.quantiles, processed["quantile_target"],
            self.critic.quantile_levels, valid)
        weibull_loss = upper_weibull_fit_loss(
            info.quantiles, self.critic.quantile_levels,
            info.location, info.scale, info.shape,
            fit_min=self.weibull_fit_min, valid=valid)
        loss = (
            self.value_loss_coef * value_loss
            + self.quantile_loss_coef * quantile_loss
            + self.weibull_loss_coef * weibull_loss
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_grad_norm)
        self.critic_optimizer.step()
        crossing = (info.quantiles[..., 1:] < info.quantiles[..., :-1]).any(dim=-1)
        return {
            "critic/value_loss": float(value_loss.item()),
            "critic/quantile_loss": float(quantile_loss.item()),
            "critic/upper_weibull_loss": float(weibull_loss.item()),
            "critic/crossing_rate": float((crossing.float() * valid).sum().item() / denom.item()),
        }

    def dual_step_size(self):
        """Current dimensionless quantile-multiplier step size."""
        return lr_lambda(self.dual_update_count, self.lambda_a, self.lambda_b, self.lambda_c)

    @torch.no_grad()
    def update_dual(self, returns):
        """
        Apply the paper dual step on a rolling 100-trajectory internal-cost sample.

        Internally this is ``lambda += eta*(Q^C_0.8-d)``. The returned quantile and residual are
        mirrored back to reward units, giving ``Q^G_0.2`` and ``q-Q^G_0.2`` for public logging.
        The first three B=25 batches fill the paper-sized sample without moving lambda.
        """
        cost_returns = -returns.detach().cpu().reshape(-1)
        self.dual_cost_returns.extend(float(value) for value in cost_returns)
        cost_quantile = linear_empirical_quantile(
            list(self.dual_cost_returns), self.cost_quantile_level)
        reward_quantile = -cost_quantile
        residual = cost_quantile - self.cost_threshold

        # The reference experiment collects at least 100 completed trajectories per update. Our
        # vector batch contains 25, so defer rather than estimating the first multipliers from an
        # artificially smaller empirical distribution.
        step = 0.0
        if len(self.dual_cost_returns) >= self.dual_window_size:
            step = self.dual_step_size()
            self.lambda_quantile.add_(step * residual).clamp_(0.0, self.lambda_max)
            self.dual_update_count += 1
        self.last_dual_quantile = reward_quantile
        self.last_dual_cost_quantile = cost_quantile
        return reward_quantile, residual, float(step)

    @torch.no_grad()
    def _density_diagnostics(self, rollout, processed):
        """Compare Weibull density at cost Q_0.8 with a detached Q_0.75/Q_0.85 estimate."""
        info = self.critic(self._critic_rollout_features(rollout))
        q_constraint = info.quantiles[..., self.constraint_index]
        weibull_density = shifted_weibull_pdf(
            q_constraint, info.location, info.scale, info.shape)
        q75 = info.quantiles[..., self.q75_index]
        q85 = info.quantiles[..., self.q85_index]
        finite_density = 0.10 / (q85 - q75).abs().clamp(min=1e-6)
        valid = processed["valid"]
        denom = valid.sum().clamp(min=1.0)
        return {
            "density/weibull_at_cost_q_1m_alpha": float(
                (weibull_density * valid).sum().item() / denom.item()),
            "density/finite_difference_detached": float((finite_density * valid).sum().item() / denom.item()),
            "density/support_violation_rate": float((processed["support_violation"].float()
                                                       * valid).sum().item() / denom.item()),
            "density/smoothed_weight_mean": float(
                (processed["density_weight"] * valid).sum().item() / denom.item()),
            "density/clipped_log_ratio_mean": float(
                (processed["clipped_log_ratio"] * valid).sum().item() / denom.item()),
        }

    def train(self):
        """Collect rollout, update the dual from returns, then run fixed-data PPO/critic epochs."""
        print(f"QCPORefGPU[paper-C=-G-Eq275]: env={self.env_name} alpha={self.failure_alpha} "
              f"q={self.return_threshold} cost_level={self.cost_quantile_level} "
              f"epochs={self.epochs}")
        for iteration in self.training_iterations():
            rollout = self._rollout_core()
            processed = self._process_rollout(rollout)

            # The original QCPO optimizer updates its multiplier before constructing PPO losses,
            # so every actor epoch in this batch sees the newly projected lambda.
            empirical_q, residual, dual_lr = self.update_dual(rollout["disc_return"])
            actor_info, critic_info = {}, {}
            for _ in range(self.epochs):
                actor_info = self._actor_epoch(rollout, processed)
                critic_info = self._critic_epoch(rollout, processed)
            diagnostics = self._density_diagnostics(rollout, processed)
            info = {
                **actor_info, **critic_info, **diagnostics,
                "lambda/quantile_multiplier": float(self.lambda_quantile.item()),
                "lambda/quantile_residual": residual,
                "lambda/lr": dual_lr,
                "quantile/empirical_linear": empirical_q,
                "quantile/internal_cost_1m_alpha": self.last_dual_cost_quantile,
                "lambda/dual_window_count": len(self.dual_cost_returns),
                "policy/log_std_mean": float(self.actor.log_std.mean().item()),
                # Capture the pre-step scalar actually used by every actor epoch in this batch.
                "training/actor_lr": float(self.actor_scheduler.get_last_lr()[0]),
            }
            iteration_metrics = self._log_core(
                iteration, rollout["disc_return"].detach().cpu().numpy(),
                rollout["diagnostic_disc_cost"].detach().cpu().numpy(),
                rollout["diagnostic_raw_cost"].detach().cpu().numpy(), extra=info)
            self.actor.update_obs_rms(rollout["S"][rollout["valid"]])
            # One rollout may have several PPO epochs, but LR advances exactly once on the common
            # data-collection batch axis. Explicit indexing also makes resume semantics obvious.
            self._set_actor_lr_batch_index(int(iteration) + 1)
            self._after_iteration(iteration, iteration_metrics)

    def checkpoint_state(self):
        """Save the paper formulation identity, critic, multiplier, and rolling dual sample."""
        state = super().checkpoint_state()
        state.update({
            "qcpo_ref_formulation": self.qcpo_ref_formulation,
            "qcpo_ref_actor_weighting": self.qcpo_ref_actor_weighting,
            "qcpo_ref_actor_reduction": self.qcpo_ref_actor_reduction,
            "qcpo_ref_critic_input": self.qcpo_ref_critic_input,
            "horizon_mode": self.horizon_mode,
            "critic_time_feature": self.critic_time_feature,
            "critic_actor_context": self.critic_actor_context,
            "critic_state_dim": self.critic_state_dim,
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_scheduler": self.actor_scheduler.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "lambda_quantile": self.lambda_quantile.detach().cpu(),
            "dual_update_count": self.dual_update_count,
            "dual_cost_returns": list(self.dual_cost_returns),
            "last_dual_quantile": self.last_dual_quantile,
            "last_dual_cost_quantile": self.last_dual_cost_quantile,
        })
        return state

    def validate_checkpoint_state(self, state):
        """
        Reject every pre-v2 actor objective before any model or optimizer tensor is mutated.

        This hook runs before the generic experiment-signature transfer decision. Consequently,
        ``--allow_checkpoint_transfer`` cannot reinterpret a uniform-time/valid-row checkpoint as
        the discounted fixed-BH estimator even if all parameter shapes are otherwise compatible.
        """
        super().validate_checkpoint_state(state)
        if not isinstance(state, dict):
            raise ValueError("QCPO_REF checkpoint is missing agent state")
        required = {
            "qcpo_ref_formulation": self.qcpo_ref_formulation,
            "qcpo_ref_actor_weighting": self.qcpo_ref_actor_weighting,
            "qcpo_ref_actor_reduction": self.qcpo_ref_actor_reduction,
            "qcpo_ref_critic_input": self.qcpo_ref_critic_input,
            "horizon_mode": self.horizon_mode,
            "critic_time_feature": self.critic_time_feature,
            "critic_actor_context": self.critic_actor_context,
            "critic_state_dim": self.critic_state_dim,
        }
        for field, requested in required.items():
            saved = state.get(field)
            if saved != requested:
                raise ValueError(
                    f"QCPO_REF checkpoint {field} mismatch: "
                    f"saved={saved!r}, requested={requested!r}")

    def load_checkpoint_state(self, state):
        """Restore all QCPO-ref state, rejecting pre-fix or cross-formulation checkpoints."""
        self.validate_checkpoint_state(state)
        super().load_checkpoint_state(state)
        self.critic.load_state_dict(state["critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        if "actor_scheduler" in state:
            self.actor_scheduler.load_state_dict(state["actor_scheduler"])
        else:
            # QCPO_REF performs one dual update per completed rollout, so this legacy field is
            # also the exact next batch index for the actor scheduler.
            self._restore_legacy_actor_scheduler(state.get("dual_update_count", 0))
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.lambda_quantile.copy_(state["lambda_quantile"].to(self.device))
        self.dual_update_count = int(state.get("dual_update_count", 0))
        self.dual_cost_returns.clear()
        self.dual_cost_returns.extend(float(value) for value in state["dual_cost_returns"])
        self.last_dual_quantile = float(state.get("last_dual_quantile", 0.0))
        self.last_dual_cost_quantile = float(state.get("last_dual_cost_quantile", 0.0))

    def get_training_summary(self):
        """Report quantile formulation separately from final direct chance evaluation."""
        return {
            "objective": "mean_reward_subject_to_reward_outage_via_negated_return_quantile",
            "qcpo_ref_formulation": self.qcpo_ref_formulation,
            "qcpo_ref_actor_weighting": self.qcpo_ref_actor_weighting,
            "qcpo_ref_actor_reduction": self.qcpo_ref_actor_reduction,
            "qcpo_ref_critic_input": self.qcpo_ref_critic_input,
            "horizon_mode": self.horizon_mode,
            "critic_time_feature": self.critic_time_feature,
            "critic_actor_context": self.critic_actor_context,
            "critic_state_dim": self.critic_state_dim,
            "lambda_quantile_final": float(self.lambda_quantile.item()),
            "empirical_linear_quantile": self.last_dual_quantile,
            "internal_cost_quantile": self.last_dual_cost_quantile,
            "internal_cost_quantile_level": self.cost_quantile_level,
            "internal_cost_threshold": self.cost_threshold,
            "dual_window_size": self.dual_window_size,
            "empirical_quantile_method": "linear",
            "empirical_prob": self.last_empirical_prob,
            "return_threshold": self.return_threshold,
            "failure_alpha": self.failure_alpha,
        }
