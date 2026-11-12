# -*- coding: utf-8 -*-
"""
Pure-online DQC-AC-beta for the direct reward chance constraint.

The recurrent Gaussian actor and the feed-forward distributional critic are independent.
The historical ``online_1step_v1`` identity remains the default.  The separate
``finite_nstep_v1`` identity conditions its critic on time and the detached actor context, uses a
boundary-correct finite-horizon n-step target, and never reinterprets a continuing checkpoint.
Every rollout trains only on fresh transitions.  The historical default performs one critic pass
and one score-function actor update; explicit ``critic_epochs``/``actor_epochs`` ablations may
reuse that one fresh rollout without introducing replay. The optional ``ppo_clip`` actor-reuse
identity adds only behavior likelihood ratios, samplewise clipping, and target-KL early stopping;
The historical mean estimator still has no GAE or normalization. The orthogonal GAE identity
instead assigns expected-return credit to an independent scalar value critic and keeps the n-step
atom critic exclusively on the fixed-q CDF risk path.
"""
import math

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from .common import DistributionalCritic, lr_lambda
from .vec_base import VecAgentBase


class StateValueCritic(nn.Module):
    """
    Independent scalar V(s,t,h,c) used only by the GAE mean estimator.

    The input is the same detached finite-horizon state block consumed by the distributional
    critic before its action is appended: observation, remaining-time, actor h, and actor c.
    No parameter is shared with either the actor or the action-conditioned quantile critic.
    """

    def __init__(self, state_dim, hidden):
        """
        Build a two-hidden-layer Tanh value MLP with one scalar output.

        Args:
            state_dim: Width of observation/time/detached-context input.
            hidden: Exact hidden widths; the v1 identity requires [512, 512].
        """
        super().__init__()
        dims = [int(state_dim)] + [int(width) for width in hidden]
        layers = []
        for input_width, output_width in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(input_width, output_width))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

        # Tanh hidden layers use the standard sqrt(2) orthogonal gain. The scalar output keeps
        # unit gain so initial predictions stay centered without shrinking their useful gradient.
        linear_layers = [module for module in self.modules() if isinstance(module, nn.Linear)]
        for index, module in enumerate(linear_layers):
            gain = math.sqrt(2.0) if index < len(linear_layers) - 1 else 1.0
            nn.init.orthogonal_(module.weight, gain=gain)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, state_features):
        """Map arbitrary leading dimensions plus D features to matching scalar values."""
        return self.net(state_features).squeeze(-1)


class DQCACBetaGPU(VecAgentBase):
    """Fresh-rollout QR critic + configurable recurrent score passes + empirical dual."""

    UPDATE_MODE = "online_1step_v1"
    FINITE_UPDATE_MODE = "finite_nstep_v1"
    UPDATE_MODES = {UPDATE_MODE, FINITE_UPDATE_MODE}
    ACTOR_REUSE_MODES = {"score_function", "ppo_clip"}
    DISTRIBUTIONAL_MEAN = "distributional_action_advantage_v1"
    GAE_MEAN = "state_value_gae_v1"
    MEAN_ESTIMATORS = {DISTRIBUTIONAL_MEAN, GAE_MEAN}

    def __init__(self, args, env):
        """
        Build the deliberately narrow clean baseline.

        Actor path (owned by VecAgentBase): obs RMS -> FC512x2 -> LSTM512 -> action mean.
        Default critic path: concat(observation, action_exec) -> ReLU MLP512x2 -> 32 QR heads.
        Finite critic path: concat(observation, remaining_fraction, detached actor h/c,
        action_exec) -> the same feed-forward MLP. Actor parameters are never shared.
        """
        super().__init__(args, env)
        self._require_return_threshold()

        # -------------------- fixed algorithm identity --------------------
        self.dqc_update_mode = str(
            getattr(args, "dqc_update_mode", self.UPDATE_MODE))
        if self.dqc_update_mode not in self.UPDATE_MODES:
            raise ValueError(
                "DQC requires dqc_update_mode in "
                f"{sorted(self.UPDATE_MODES)!r}, got {self.dqc_update_mode!r}")

        # These values jointly define the critic random variable, input width, and target update.
        # They are one immutable checkpoint identity, not independently mixable tuning knobs.
        finite_mode = self.dqc_update_mode == self.FINITE_UPDATE_MODE
        self.horizon_mode = str(getattr(
            args, "horizon_mode", "finite" if finite_mode else "continuing"))
        self.critic_n_step = int(getattr(
            args, "critic_n_step", 100 if finite_mode else 1))
        self.critic_time_feature = str(getattr(
            args, "critic_time_feature", "remaining_fraction" if finite_mode else "none"))
        self.critic_actor_context = bool(getattr(
            args, "critic_actor_context", finite_mode))
        self.target_update_mode = str(getattr(
            args, "target_update_mode", "hard" if finite_mode else "polyak"))

        self.beta = float(getattr(args, "beta", 0.995))
        self.num_quantiles = int(getattr(args, "num_quantiles", 32))
        self.num_action_samples = max(
            1, int(getattr(args, "num_action_samples", 4)))
        self.warmup_iters = max(0, int(getattr(
            args, "warmup_iters", 10 if finite_mode else 5)))
        self.critic_warmup_mode = str(
            getattr(args, "critic_warmup_mode", "fitted_hard" if finite_mode else "legacy"))
        if self.critic_warmup_mode not in {"legacy", "fitted_hard"}:
            raise ValueError(
                "critic_warmup_mode must be legacy or fitted_hard")
        self.warmup_critic_epochs = int(
            getattr(args, "warmup_critic_epochs", 4 if finite_mode else 1))
        if self.warmup_critic_epochs <= 0:
            raise ValueError("warmup_critic_epochs must be positive")
        if self.critic_warmup_mode == "legacy" and self.warmup_critic_epochs != 1:
            raise ValueError(
                "legacy critic warmup requires warmup_critic_epochs=1")

        self.critic_minibatch_size = max(
            1, int(getattr(args, "critic_minibatch_size", 1024)))
        self.critic_epochs = int(getattr(args, "critic_epochs", 1))
        self.actor_epochs = int(getattr(args, "actor_epochs", 1))
        self.dqc_actor_reuse_mode = str(getattr(
            args, "dqc_actor_reuse_mode", "score_function"))
        self.dqc_ratio_clip = float(getattr(args, "dqc_ratio_clip", 0.1))
        self.dqc_target_kl = float(getattr(args, "dqc_target_kl", 0.0))
        if self.critic_epochs <= 0:
            raise ValueError("critic_epochs must be positive")
        if self.actor_epochs <= 0:
            raise ValueError("actor_epochs must be positive")
        if self.dqc_actor_reuse_mode not in self.ACTOR_REUSE_MODES:
            raise ValueError(
                "dqc_actor_reuse_mode must be score_function or ppo_clip")
        if not math.isfinite(self.dqc_ratio_clip) or not 0.0 < self.dqc_ratio_clip < 1.0:
            raise ValueError("dqc_ratio_clip must be finite and in (0, 1)")
        if not math.isfinite(self.dqc_target_kl) or self.dqc_target_kl < 0.0:
            raise ValueError("dqc_target_kl must be finite and non-negative")
        if self.dqc_actor_reuse_mode == "score_function" and self.dqc_target_kl != 0.0:
            raise ValueError(
                "dqc_target_kl only applies to dqc_actor_reuse_mode=ppo_clip")

        self.huber_kappa = float(getattr(args, "huber_kappa", 0.1))
        self.target_tau = float(getattr(args, "target_tau", 0.005))
        self.critic_target_source = str(
            getattr(args, "critic_target_source", "n_step" if finite_mode else "one_step"))
        if self.critic_target_source not in {"one_step", "n_step", "finite_suffix_mc"}:
            raise ValueError(
                "critic_target_source must be one_step, n_step, or finite_suffix_mc")
        if (self.critic_warmup_mode == "fitted_hard"
                and self.critic_target_source not in {"one_step", "n_step"}):
            raise ValueError(
                "fitted_hard critic warmup requires a Bellman target source")
        self.actor_grad_clip = float(getattr(args, "actor_grad_clip", 10.0))
        self.critic_grad_clip = float(getattr(args, "critic_grad_clip", 10.0))

        # The mean estimator is orthogonal to the n-step distributional risk critic. Historical
        # checkpoints default to the atom-mean identity; GAE is explicit and checkpoint-signed.
        self.dqc_mean_estimator = str(getattr(
            args, "dqc_mean_estimator", self.DISTRIBUTIONAL_MEAN))
        if self.dqc_mean_estimator not in self.MEAN_ESTIMATORS:
            raise ValueError(
                "dqc_mean_estimator must be distributional_action_advantage_v1 "
                "or state_value_gae_v1")
        self.dqc_gae_lambda = float(getattr(args, "dqc_gae_lambda", 0.97))
        value_hidden = getattr(args, "dqc_value_hidden", [512, 512])
        if isinstance(value_hidden, str):
            value_hidden = [
                int(value) for value in value_hidden.split(",") if value.strip()]
        self.dqc_value_hidden = list(value_hidden)
        self.dqc_value_lr = float(getattr(args, "dqc_value_lr", 3e-4))
        self.dqc_value_epochs = int(getattr(args, "dqc_value_epochs", 8))
        self.normalize_combined_advantage = bool(getattr(
            args, "normalize_combined_advantage", False))

        # GAE-Mean v1 is a deliberately fixed formulation, not a menu of loosely compatible
        # knobs. Reject partial overrides before allocating its scalar optimizer state.
        if self.dqc_mean_estimator == self.GAE_MEAN:
            gae_identity = {
                "dqc_update_mode": (self.dqc_update_mode, self.FINITE_UPDATE_MODE),
                "dqc_gae_lambda": (self.dqc_gae_lambda, 0.97),
                "dqc_value_hidden": (self.dqc_value_hidden, [512, 512]),
                "dqc_value_lr": (self.dqc_value_lr, 3e-4),
                "dqc_value_epochs": (self.dqc_value_epochs, 8),
                "normalize_combined_advantage": (
                    self.normalize_combined_advantage, True),
                "critic_epochs": (self.critic_epochs, 8),
                "actor_epochs": (self.actor_epochs, 8),
                "dqc_actor_reuse_mode": (
                    self.dqc_actor_reuse_mode, "ppo_clip"),
                "dqc_ratio_clip": (self.dqc_ratio_clip, 0.1),
                "dqc_target_kl": (self.dqc_target_kl, 0.01),
            }
            gae_mismatches = [
                f"{name}={actual!r} (required {expected!r})"
                for name, (actual, expected) in gae_identity.items()
                if actual != expected
            ]
            if gae_mismatches:
                raise ValueError(
                    "state_value_gae_v1 identity mismatch: "
                    + ", ".join(gae_mismatches))

        # finite_nstep_v1 is deliberately narrow. A partial override changes G_1000, the critic
        # architecture, or the constraint gradient, so reject it before critic state allocation.
        if finite_mode:
            required_identity = {
                "horizon": (self.horizon, 1000),
                "horizon_mode": (self.horizon_mode, "finite"),
                "critic_target_source": (self.critic_target_source, "n_step"),
                "critic_n_step": (self.critic_n_step, 100),
                "critic_time_feature": (
                    self.critic_time_feature, "remaining_fraction"),
                "critic_actor_context": (self.critic_actor_context, True),
                "target_update_mode": (self.target_update_mode, "hard"),
                "beta": (self.beta, 1.0),
                "warmup_iters": (self.warmup_iters, 10),
                "critic_warmup_mode": (self.critic_warmup_mode, "fitted_hard"),
                "warmup_critic_epochs": (self.warmup_critic_epochs, 4),
            }
        else:
            # Historical behavior stays exact. finite_suffix_mc remains the prior target-only
            # diagnostic, but it cannot opt into finite inputs or hard target copies.
            required_identity = {
                "horizon_mode": (self.horizon_mode, "continuing"),
                "critic_n_step": (self.critic_n_step, 1),
                "critic_time_feature": (self.critic_time_feature, "none"),
                "critic_actor_context": (self.critic_actor_context, False),
                "target_update_mode": (self.target_update_mode, "polyak"),
            }
        mismatches = [
            f"{name}={actual!r} (required {expected!r})"
            for name, (actual, expected) in required_identity.items()
            if actual != expected
        ]
        if mismatches:
            raise ValueError(
                f"{self.dqc_update_mode} identity mismatch: " + ", ".join(mismatches))
        # -------------------- independent feed-forward QR critic --------------------
        hidden = getattr(args, "critic_hidden", [512, 512])
        if isinstance(hidden, str):
            hidden = [int(value) for value in hidden.split(",") if value.strip()]
        hidden = list(hidden)
        if hidden != [512, 512] or self.num_quantiles != 32:
            raise ValueError(
                "online_1step_v1 fixes critic_hidden=[512,512] and num_quantiles=32")

        # The finite MLP receives actor context only as detached input data. It shares no actor
        # parameters, and critic backward therefore cannot mutate the recurrent policy.
        critic_state_dim = self.state_dim
        if self.critic_time_feature == "remaining_fraction":
            critic_state_dim += 1
        if self.critic_actor_context:
            critic_state_dim += 2 * self.lstm_size
        self.critic_state_dim = int(critic_state_dim)
        self.reward_critic = DistributionalCritic(
            self.critic_state_dim, self.action_dim,
            self.num_quantiles, hidden).to(self.device)
        self.reward_target_critic = DistributionalCritic(
            self.critic_state_dim, self.action_dim,
            self.num_quantiles, hidden).to(self.device)
        self.reward_target_critic.load_state_dict(self.reward_critic.state_dict())

        # Target parameters are frozen storage. They change only after a complete critic epoch,
        # through configured Polyak/hard copy; autograd can never accumulate their gradients.
        self.reward_target_critic.requires_grad_(False)
        self.reward_target_critic.eval()
        self.critic_optimizer = Adam(
            self.reward_critic.parameters(),
            lr=float(getattr(args, "critic_lr", 3e-4)), eps=1e-5)

        # GAE mean estimation owns a separate scalar network and Adam state. The historical atom
        # mean identity allocates neither, preserving its parameter/RNG lineage exactly.
        self.value_critic = None
        self.value_optimizer = None
        if self.dqc_mean_estimator == self.GAE_MEAN:
            self.value_critic = StateValueCritic(
                self.critic_state_dim, self.dqc_value_hidden).to(self.device)
            self.value_optimizer = Adam(
                self.value_critic.parameters(), lr=self.dqc_value_lr, eps=1e-5)

        # -------------------- recurrent actor optimizer and absolute batch schedule --------------------
        self.actor_lr_b = float(args.theta_b)
        self.actor_lr_c = float(args.theta_c)
        self.actor_optimizer = Adam(
            self.actor.parameters(),
            lr=float(getattr(args, "actor_lr", 2e-4)), eps=1e-5)
        self.actor_scheduler = self._new_actor_scheduler()

        # Head j retains nominal identity tau_j=(j+.5)/N. Atoms are never sorted before QR loss,
        # CDF queries, or crossing diagnostics, because sorting would hide head crossings.
        self.taus = (
            torch.arange(self.num_quantiles, dtype=torch.float32, device=self.device) + 0.5
        ) / float(self.num_quantiles)

        # -------------------- empirical direct-chance multiplier --------------------
        self.lambda_probability = torch.tensor(
            float(getattr(args, "penalty_init", 0.0)),
            dtype=torch.float32, device=self.device)
        self.lambda_dual = self.lambda_probability
        self.lambda_max = float(getattr(args, "lambda_max", 50.0))
        self.lambda_a = float(args.lambda_a)
        self.lambda_b = float(args.lambda_b)
        self.lambda_c = float(args.lambda_c)

        # Explicit counters are checkpoint state. Warmup changes critic_optimizer_steps only;
        # actor_optimizer_steps and dual_update_count remain unchanged for every configured batch.
        self.actor_optimizer_steps = 0
        self.critic_optimizer_steps = 0
        self.value_optimizer_steps = 0
        self.target_update_count = 0
        self.dual_update_count = 0
        self.last_dual_prob = 0.0
        self.actor_kl_early_stop_count = 0

    # ============================================================ actor LR schedule ============================================================
    def _actor_lr_ratio(self, batch_index):
        """Return (b/(b+k))^c so actor_lr is the exact k=0 optimizer LR."""
        k = max(0, int(batch_index))
        return (self.actor_lr_b / (self.actor_lr_b + k)) ** self.actor_lr_c

    def _new_actor_scheduler(self, last_epoch=-1):
        """Construct checkpointable LambdaLR metadata for the absolute rollout index."""
        return LambdaLR(
            self.actor_optimizer,
            lr_lambda=self._actor_lr_ratio,
            last_epoch=int(last_epoch),
        )

    def _set_actor_lr_batch_index(self, batch_index):
        """
        Advance the actor LR by absolute batch index without fabricating an Adam step.

        This explicit setter is required during critic-only warmup: scheduler.step() before the
        first optimizer.step() would both warn and misrepresent actor optimizer state.
        """
        k = max(0, int(batch_index))
        ratio = self._actor_lr_ratio(k)
        learning_rates = [float(base) * ratio for base in self.actor_scheduler.base_lrs]
        for group, learning_rate in zip(
                self.actor_optimizer.param_groups, learning_rates):
            group["lr"] = learning_rate
        self.actor_scheduler.last_epoch = k
        self.actor_scheduler._last_lr = learning_rates
        self.actor_scheduler._step_count = max(
            int(getattr(self.actor_scheduler, "_step_count", 0)), k + 1)

    # ============================================================ finite critic inputs ============================================================
    def _uses_finite_critic(self):
        """Return whether critic/CDF semantics belong to the strict G_1000 branch."""
        return (
            getattr(self, "dqc_update_mode", self.UPDATE_MODE)
            == self.FINITE_UPDATE_MODE
        )

    def remaining_fraction(self, time_indices):
        """
        Map zero-based episode indices to (H-t)/H without changing actor observations.

        The returned tensor adds no feature at the actor boundary.  For H=1000 it is exactly 1 at
        t=0 and .001 at t=999; an n-step endpoint uses its own t+m rather than the start time.
        """
        time = torch.as_tensor(
            time_indices, dtype=torch.float32, device=self.device)
        return (float(self.horizon) - time) / float(self.horizon)

    def _critic_features(self, states, time_indices=None, h=None, c=None):
        """
        Construct the MLP state block while detaching all actor-owned recurrent context.

        Args:
            states: raw observation tensor with arbitrary leading dimensions and final D axis.
            time_indices: matching leading dimensions, containing absolute episode timestep t.
            h/c: matching leading dimensions plus actor LSTM width.

        Returns:
            Default identity: states unchanged.
            Finite identity: concat(states, remaining_fraction[...,None], h.detach(), c.detach()).
        """
        if not self._uses_finite_critic():
            return states
        if time_indices is None or h is None or c is None:
            raise ValueError("finite critic queries require time_indices and actor h/c context")
        remaining = self.remaining_fraction(time_indices).unsqueeze(-1)
        if remaining.shape[:-1] != states.shape[:-1]:
            remaining = remaining.expand(*states.shape[:-1], 1)
        return torch.cat(
            (states, remaining.to(states.dtype), h.detach(), c.detach()), dim=-1)

    def _critic_rollout_features(self, rollout):
        """Build [T,B,Dcritic] current-row inputs using each row's pre-action h0/c0."""
        states = rollout["S"]
        T, B = states.shape[:2]
        time = torch.arange(
            T, dtype=torch.float32, device=states.device).view(T, 1).expand(T, B)
        return self._critic_features(
            states, time, rollout.get("h0"), rollout.get("c0"))

    @torch.no_grad()
    def initial_reward_critic_atoms(self, states, executed_actions):
        """Evaluation adapter for initial t=0, h=c=0 finite critic calibration queries."""
        if not self._uses_finite_critic():
            return self.reward_critic(states, executed_actions)
        count = int(states.shape[0])
        context_shape = (count, self.lstm_size)
        zero_context = torch.zeros(
            context_shape, dtype=states.dtype, device=states.device)
        features = self._critic_features(
            states,
            torch.zeros(count, dtype=torch.float32, device=states.device),
            zero_context,
            zero_context,
        )
        return self.reward_critic(features, executed_actions)

    # ============================================================ reward budget ============================================================
    def _attach_reward_budgets(self, rollout):
        """
        Attach b[t,B] with b0=q and b[t+1]=(b[t]-r[t])/gamma.

        Beta is absent by construction. It is an actor risk-credit discount only and cannot alter
        the critic target, budget, empirical dual, feasibility, or evaluation return definition.
        """
        T, B = rollout["R"].shape
        budgets = torch.empty(T, B, dtype=torch.float32, device=self.device)
        budgets[0].fill_(self.return_threshold)
        for t in range(T - 1):
            budgets[t + 1] = (budgets[t] - rollout["R"][t]) / self.gamma
        rollout["budgets"] = budgets

        # The finite CDF path never divides a late-step budget by gamma**t in float32.  Instead it
        # stores C_t=sum_{k<t} gamma^k r_k in float64 and evaluates C_t+gamma^t*z <= q globally.
        # The float32 budgets tensor remains above solely for compatibility logs/default behavior.
        time = torch.arange(T, dtype=torch.float64, device=self.device)
        discount = torch.pow(
            torch.tensor(self.gamma, dtype=torch.float64, device=self.device),
            time,
        ).unsqueeze(1)
        discounted_reward = discount * rollout["R"].double()
        inclusive_prefix = torch.cumsum(discounted_reward, dim=0)
        reward_prefix = inclusive_prefix - discounted_reward
        rollout["reward_prefix"] = reward_prefix
        return rollout

    @staticmethod
    def finite_global_cdf(atoms, reward_prefix, time_indices, gamma, threshold):
        """
        Evaluate mean_j 1{C_t + gamma^t z_j <= q} in float64.

        This form is algebraically equivalent to z_j <= (q-C_t)/gamma^t in exact arithmetic, but
        avoids the amplified float32 rounding error of dividing a late-timestep remaining budget.
        """
        atom64 = atoms.to(torch.float64)
        prefix64 = torch.as_tensor(
            reward_prefix, dtype=torch.float64, device=atom64.device)
        time64 = torch.as_tensor(
            time_indices, dtype=torch.float64, device=atom64.device)
        discount = torch.pow(
            torch.tensor(float(gamma), dtype=torch.float64, device=atom64.device),
            time64,
        )
        event = (
            prefix64.unsqueeze(-1)
            + discount.unsqueeze(-1) * atom64
            <= float(threshold)
        )
        return event.float().mean(dim=-1)

    def _critic_cdf(self, atoms, rollout):
        """Dispatch continuing budget CDF or strict finite global float64 CDF."""
        if not self._uses_finite_critic():
            return (
                atoms <= rollout["budgets"].unsqueeze(-1)
            ).float().mean(dim=-1)
        T, B = atoms.shape[:2]
        time = torch.arange(
            T, dtype=torch.float64, device=atoms.device).view(T, 1).expand(T, B)
        return self.finite_global_cdf(
            atoms, rollout["reward_prefix"], time,
            self.gamma, self.return_threshold)

    # ============================================================ QR critic targets ============================================================
    @staticmethod
    def finite_suffix_returns(rewards, valid, gamma):
        """
        Compute one sampled finite-horizon suffix return for every valid ``[t,b]`` row.

        ``running`` resets to zero on invalid rows, so an early-ended trajectory cannot consume
        rewards from an adapter auto-reset episode. Both termination and horizon truncation are
        deliberately unbootstrapped: this branch asks whether observed returns calibrate the MLP
        when the one-step Bellman target is removed.
        """
        valid_mask = valid.bool()
        suffix = torch.zeros_like(rewards)
        running = torch.zeros(
            rewards.shape[1], dtype=rewards.dtype, device=rewards.device)
        for timestep in range(int(rewards.shape[0]) - 1, -1, -1):
            candidate = rewards[timestep] + float(gamma) * running
            running = torch.where(
                valid_mask[timestep], candidate, torch.zeros_like(running))
            suffix[timestep] = running
        return suffix

    def quantile_huber_loss(self, predicted, target):
        """
        Return fixed-head QR-DQN Huber loss for predicted [M,N] and target [M,K].

        One-step TD supplies K=N target atoms; finite-suffix MC supplies K=1 sampled return.
        Broadcasting forms error[m,i,j]=target[m,j]-predicted[m,i]. The sign indicator is detached,
        while gradients flow only through predicted head i. No atom sorting occurs.
        """
        error = target.unsqueeze(1) - predicted.unsqueeze(2)       # [M,N,K]
        absolute = error.abs()
        huber = torch.where(
            absolute <= self.huber_kappa,
            0.5 * error.square(),
            self.huber_kappa * (absolute - 0.5 * self.huber_kappa),
        )
        weights = torch.abs(
            self.taus.view(1, -1, 1) - (error.detach() < 0).float())
        return (weights * huber / self.huber_kappa).mean()

    @torch.no_grad()
    def _next_executed_actions(self, rollout):
        """
        Build the target-policy action for every [T,B] transition without replay.

        Ordinary transition:
            Reuse A_exec[t+1,b] already sampled under the exact rollout actor history.
        True termination:
            Action is irrelevant because the bootstrap multiplier is zero.
        Truncation:
            Sample once at final_observation with next_h/next_c captured before slot reset, then
            clip. The adapter reset observation and zero recurrent state never enter this target.
        """
        valid = rollout["valid"].bool()
        terminated = rollout["terminated"].bool()
        truncated = rollout["truncated"].bool()
        continuing = valid & ~terminated & ~truncated
        next_actions = torch.zeros_like(rollout["A_exec"])

        # Every ordinary transition must have a valid successor in the same fresh rollout. A
        # continuing final row would mean the collector silently cut an episode without a target.
        missing_successor = continuing[-1].clone()
        if continuing.shape[0] > 1:
            missing_successor = missing_successor | (
                continuing[:-1] & ~valid[1:]).any(dim=0)
            next_actions[:-1] = torch.where(
                continuing[:-1].unsqueeze(-1),
                rollout["A_exec"][1:],
                next_actions[:-1],
            )
        if bool(missing_successor.any()):
            raise RuntimeError("online one-step target is missing a rollout successor action")

        # Boolean indexing flattens the [T,B] prefix into K truncated rows. Rebuild [1,K,H]
        # recurrent tensors because actor.step expects standard LSTM state layout.
        if bool(truncated.any()):
            final_mask = rollout["final_observation_mask"].bool()
            if bool((truncated & ~final_mask).any()):
                raise RuntimeError("truncated transition is missing final_observation")
            final_states = rollout["final_observation"][truncated]       # [K,D]
            old_context = (
                rollout["next_h"][truncated].unsqueeze(0),              # [1,K,H]
                rollout["next_c"][truncated].unsqueeze(0),              # [1,K,H]
            )
            target_raw, _state, _mean, _log_std = self._sample_actions(
                final_states, old_context, return_state=True)
            next_actions[truncated] = self._executed_actions(target_raw)
        return next_actions

    @torch.no_grad()
    def build_one_step_batch(self, rollout):
        """
        Materialize and detach every current-rollout one-step target before critic optimization.

        Returned leading dimension M is exactly valid.sum(). transition_ids preserve flattened
        [T,B] identity, allowing tests and diagnostics to prove one-pass, no-replacement coverage.
        """
        valid = rollout["valid"].bool()
        next_actions = self._next_executed_actions(rollout)

        # The target critic is called once for the complete valid set while it is frozen. Even
        # terminated rows may have arbitrary next atoms because bootstrap_mask=0 removes them.
        next_atoms = self.reward_target_critic(
            rollout["next_states"][valid], next_actions[valid])          # [M,N]
        bootstrap = (~rollout["terminated"][valid]).float().unsqueeze(1)
        targets = (
            rollout["R"][valid].unsqueeze(1)
            + self.gamma * bootstrap * next_atoms
        ).detach()

        T, B = valid.shape
        transition_ids = torch.arange(T * B, device=self.device).reshape(T, B)[valid]
        return {
            "states": rollout["S"][valid].detach(),
            "executed_actions": rollout["A_exec"][valid].detach(),
            "targets": targets,
            "transition_ids": transition_ids,
        }

    @torch.no_grad()
    def build_n_step_batch(self, rollout):
        """
        Materialize boundary-correct finite-horizon distributional n-step targets.

        For a start row (t,b), each loop offset adds gamma^k r[t+k,b] only while the same finite
        episode is alive.  A terminated or truncated row contributes its own reward, then clears
        the mask before the next offset.  Therefore a real terminal and the H=1000 truncation both
        have exactly zero tail; reset observations/rewards can never enter the old episode.

        Bootstrap occurs only when row t+n is valid and no boundary appeared in t..t+n-1.  Its
        target query uses S[t+n], the endpoint's h0/c0, and A_exec[t+n], so neither the start
        context/time nor an adapter reset observation is substituted at the endpoint.
        """
        valid = rollout["valid"].bool()
        terminated = rollout["terminated"].bool()
        truncated = rollout["truncated"].bool()
        boundary = terminated | truncated
        rewards = rollout["R"]
        T, B = valid.shape
        n_step = min(int(self.critic_n_step), T)
        if n_step <= 0:
            raise ValueError("critic_n_step must be positive")

        # start_alive[t,b] means that the next reward for start t still belongs to its original
        # finite episode.  It is updated after adding each current boundary reward.
        partial_return = torch.zeros_like(rewards)
        start_alive = valid.clone()
        for offset in range(n_step):
            rows = T - offset
            partial_return[:rows] += (
                (self.gamma ** offset)
                * rewards[offset:]
                * start_alive[:rows].to(rewards.dtype)
            )
            if rows > 1:
                start_alive[:rows - 1] = (
                    start_alive[:rows - 1]
                    & ~boundary[offset:T - 1]
                    & valid[offset + 1:]
                )

        # Every terminal/suffix-only target is represented by N identical atoms, matching the
        # historical one-step QR target shape. Interior endpoints replace that zero tail with the
        # frozen target critic's N atoms.
        targets = partial_return.unsqueeze(-1).expand(
            T, B, self.num_quantiles).clone()
        if T > n_step:
            bootstrap_mask = start_alive[:T - n_step]
            if bool(bootstrap_mask.any()):
                endpoint_features = self._critic_rollout_features(rollout)[n_step:]
                endpoint_atoms = self.reward_target_critic(
                    endpoint_features[bootstrap_mask],
                    rollout["A_exec"][n_step:][bootstrap_mask],
                )
                target_prefix = targets[:T - n_step]
                target_prefix[bootstrap_mask] += (
                    (self.gamma ** n_step) * endpoint_atoms)

        critic_features = self._critic_rollout_features(rollout)
        transition_ids = torch.arange(
            T * B, device=self.device).reshape(T, B)[valid]
        return {
            "states": critic_features[valid].detach(),
            "executed_actions": rollout["A_exec"][valid].detach(),
            "targets": targets[valid].detach(),
            "transition_ids": transition_ids,
        }

    @torch.no_grad()
    def build_finite_suffix_batch(self, rollout):
        """
        Build QR supervision from one observed scalar suffix sample per valid transition.

        Target shape ``[M,1]`` is intentional: broadcasting compares each of the N predicted
        quantile heads with the same sampled return. This path never queries the target critic and
        does not alter actor, dual, feasibility, evaluation, or checkpoint-selection semantics.
        """
        valid = rollout["valid"].bool()
        suffix = self.finite_suffix_returns(rollout["R"], valid, self.gamma)
        T, B = valid.shape
        transition_ids = torch.arange(
            T * B, device=self.device).reshape(T, B)[valid]
        return {
            "states": rollout["S"][valid].detach(),
            "executed_actions": rollout["A_exec"][valid].detach(),
            "targets": suffix[valid].unsqueeze(1).detach(),
            "transition_ids": transition_ids,
        }

    @torch.no_grad()
    def build_critic_batch(self, rollout):
        """Dispatch only the explicitly configured critic supervision source."""
        if self.critic_target_source == "one_step":
            return self.build_one_step_batch(rollout)
        if self.critic_target_source == "n_step":
            return self.build_n_step_batch(rollout)
        if self.critic_target_source == "finite_suffix_mc":
            return self.build_finite_suffix_batch(rollout)
        raise ValueError(
            f"unsupported critic_target_source={self.critic_target_source!r}")

    @torch.no_grad()
    def soft_update_target(self):
        """Apply target <- (1-tau) target + tau online exactly once after the epoch."""
        for target, online in zip(
                self.reward_target_critic.parameters(), self.reward_critic.parameters()):
            target.mul_(1.0 - self.target_tau).add_(online, alpha=self.target_tau)
        self.target_update_count += 1

    @torch.no_grad()
    def hard_update_target(self):
        """Copy online parameters into target storage once after a fitted TD generation."""
        self.reward_target_critic.load_state_dict(self.reward_critic.state_dict())
        self.target_update_count += 1

    @staticmethod
    def atom_crossing_statistics(atoms):
        """Return row/adjacent crossing fractions without sorting fixed-identity heads."""
        if atoms.ndim != 2 or atoms.shape[1] <= 1:
            return 0.0, 0.0
        adjacent = atoms[:, 1:] < atoms[:, :-1]
        return (
            float(adjacent.any(dim=1).float().mean().item()),
            float(adjacent.float().mean().item()),
        )

    @torch.no_grad()
    def _post_critic_diagnostics(self, rollout):
        """Query all valid current (observation,A_exec) pairs after the critic epoch."""
        valid = rollout["valid"].bool()
        features = self._critic_rollout_features(rollout)
        all_atoms = self.reward_critic(features, rollout["A_exec"])
        atoms = all_atoms[valid]
        row_fraction, adjacent_fraction = self.atom_crossing_statistics(atoms)
        cdf = self._critic_cdf(all_atoms, rollout)[valid]
        return {
            "diagnostic/reward_critic/atom_crossing_row_fraction": row_fraction,
            "diagnostic/reward_critic/atom_crossing_adjacent_fraction": adjacent_fraction,
            "diagnostic/dqc/critic_cdf_mean": float(cdf.mean().item()),
        }

    def update_critic(
            self, rollout, permutation=None, regression_epochs=1,
            target_update_mode="polyak"):
        """
        Fit one frozen Bellman generation, then update target storage exactly once.

        ``regression_epochs`` repeats full no-replacement passes over the same materialized labels.
        The target critic therefore remains bitwise fixed across every Adam minibatch.  Legacy
        operation is the exact E=1/Polyak case; fitted warmup uses E>=1 followed by a hard copy.
        Finite-suffix MC remains diagnostic-only and never queries or updates target storage.
        """
        regression_epochs = int(regression_epochs)
        if regression_epochs <= 0:
            raise ValueError("regression_epochs must be positive")
        if target_update_mode not in {"polyak", "hard"}:
            raise ValueError(
                "target_update_mode must be polyak or hard")
        if (target_update_mode == "hard"
                and self.critic_target_source not in {"one_step", "n_step"}):
            raise ValueError(
                "hard target updates require critic_target_source=one_step or n_step")

        # Build targets only once. For one-step TD this snapshots the frozen target network before
        # any online optimization, making all E regression passes one fitted Bellman generation.
        batch = self.build_critic_batch(rollout)
        transition_count = int(batch["targets"].shape[0])
        if transition_count <= 0:
            raise RuntimeError("DQC critic received an empty rollout")

        fixed_permutation = None
        if permutation is not None:
            fixed_permutation = torch.as_tensor(
                permutation, dtype=torch.long, device=self.device).reshape(-1)
        expected = torch.arange(transition_count, device=self.device)
        if fixed_permutation is not None and (
                fixed_permutation.numel() != transition_count or not torch.equal(
                    fixed_permutation.sort().values, expected)):
            raise ValueError("critic permutation must cover every transition exactly once")

        num_minibatches = int(math.ceil(
            transition_count / float(self.critic_minibatch_size)))
        weighted_loss = 0.0
        seen_ids = []
        global_minibatch_index = 0

        for _epoch in range(regression_epochs):
            # The caller-provided order is reused for deterministic tests. Production samples a
            # fresh no-replacement permutation per regression epoch, without replay or replacement.
            epoch_permutation = (
                fixed_permutation if fixed_permutation is not None else
                torch.randperm(transition_count, device=self.device))
            minibatches = torch.tensor_split(epoch_permutation, num_minibatches)

            for indices in minibatches:
                predicted = self.reward_critic(
                    batch["states"][indices], batch["executed_actions"][indices])
                loss = self.quantile_huber_loss(
                    predicted, batch["targets"][indices])

                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.reward_critic.parameters(), self.critic_grad_clip)
                self.critic_optimizer.step()
                self.critic_optimizer_steps += 1

                rows = int(indices.numel())
                weighted_loss += float(loss.item()) * rows
                seen_ids.append(batch["transition_ids"][indices])

                # A read-only test hook verifies that target parameters never move inside fitted
                # regression. Production installs no hook and pays no snapshot overhead.
                hook = getattr(self, "_critic_minibatch_hook", None)
                if hook is not None:
                    hook(global_minibatch_index, batch)
                global_minibatch_index += 1

        visited = torch.cat(seen_ids)
        coverage = float(torch.unique(visited).numel()) / float(transition_count)

        # One target mutation closes the fitted generation. The default stays one Polyak update;
        # the explicit warmup branch removes slow/random target lag with a hard copy.
        if self.critic_target_source in {"one_step", "n_step"}:
            if target_update_mode == "hard":
                self.hard_update_target()
            else:
                self.soft_update_target()

        diagnostics = self._post_critic_diagnostics(rollout)
        diagnostics.update({
            "critic/reward_quantile_loss": weighted_loss / float(
                transition_count * regression_epochs),
            "diagnostic/dqc/critic_optimizer_steps": float(
                self.critic_optimizer_steps),
            "diagnostic/dqc/critic_regression_epochs": float(
                regression_epochs),
            "diagnostic/dqc/target_update_hard": float(
                self.critic_target_source in {"one_step", "n_step"}
                and target_update_mode == "hard"),
            "diagnostic/dqc/transition_coverage": coverage,
        })
        return diagnostics

    # ============================================================ scalar GAE mean critic ============================================================
    @staticmethod
    def finite_gae(rewards, values, valid, terminated, truncated, gamma, gae_lambda):
        """
        Build boundary-correct finite-horizon GAE and lambda-return tensors.

        Both true termination and formal horizon truncation set m_t=0. Invalid padding also sets
        m_t=0 and resets the reverse accumulator, so an auto-reset episode can never leak into an
        earlier trajectory. The returned value targets equal stopgrad(A_t + V_t).
        """
        tensors = (rewards, values, valid, terminated, truncated)
        if any(tensor.shape != rewards.shape for tensor in tensors):
            raise ValueError("GAE rewards, values, and masks must have identical [T,B] shape")
        if rewards.ndim != 2:
            raise ValueError("GAE tensors must have shape [T,B]")

        valid_mask = valid.bool()
        boundary = terminated.bool() | truncated.bool()
        continuation = valid_mask & ~boundary

        # V_{t+1} is taken only from the next valid row in the same collected trajectory. The
        # final row has no successor and is therefore unbootstrapped even if a malformed adapter
        # omitted its boundary flag; this is safer than reading an auto-reset observation.
        next_values = torch.zeros_like(values)
        successor_valid = torch.zeros_like(valid_mask)
        if int(rewards.shape[0]) > 1:
            next_values[:-1] = values[1:]
            successor_valid[:-1] = valid_mask[1:]
        continuation = continuation & successor_valid
        continuation_float = continuation.to(rewards.dtype)

        deltas = (
            rewards
            + float(gamma) * continuation_float * next_values
            - values
        )
        advantages = torch.zeros_like(rewards)
        running = torch.zeros(
            rewards.shape[1], dtype=rewards.dtype, device=rewards.device)
        coefficient = float(gamma) * float(gae_lambda)
        for timestep in range(int(rewards.shape[0]) - 1, -1, -1):
            candidate = (
                deltas[timestep]
                + coefficient * continuation_float[timestep] * running
            )
            running = torch.where(
                valid_mask[timestep], candidate, torch.zeros_like(running))
            advantages[timestep] = running

        # Invalid rows are explicitly zeroed. Their arbitrary padded value prediction therefore
        # cannot enter either value regression or actor normalization diagnostics.
        advantages = torch.where(
            valid_mask, advantages, torch.zeros_like(advantages))
        targets = torch.where(
            valid_mask, advantages + values, torch.zeros_like(values))
        return advantages.detach(), targets.detach(), continuation

    @torch.no_grad()
    def prepare_gae_batch(self, rollout):
        """
        Snapshot V, GAE, and lambda-return targets once before any value optimizer step.

        Reusing this materialized dictionary for all value epochs and actor epochs prevents target
        drift and ensures the scalar mean contribution appears exactly once in the actor weight.
        """
        if self.dqc_mean_estimator != self.GAE_MEAN or self.value_critic is None:
            raise RuntimeError("prepare_gae_batch requires state_value_gae_v1")
        features = self._critic_rollout_features(rollout).detach()
        values = self.value_critic(features).detach()
        advantages, targets, continuation = self.finite_gae(
            rollout["R"], values, rollout["valid"],
            rollout["terminated"], rollout["truncated"],
            self.gamma, self.dqc_gae_lambda,
        )
        return {
            "features": features,
            "values": values,
            "advantages": advantages,
            "targets": targets,
            "continuation": continuation,
            "valid": rollout["valid"].bool(),
        }

    def update_value_critic(self, gae_batch, regression_epochs):
        """
        Fit the independent scalar critic to one frozen set of lambda-return targets.

        Production draws one no-replacement order per epoch. A read-only hook receives the exact
        target tensor at each epoch boundary so tests can prove the E passes never rebuild labels.
        """
        if self.value_critic is None or self.value_optimizer is None:
            raise RuntimeError("value critic update requires state_value_gae_v1")
        regression_epochs = int(regression_epochs)
        if regression_epochs <= 0:
            raise ValueError("value regression_epochs must be positive")

        valid = gae_batch["valid"].bool()
        features = gae_batch["features"][valid]
        targets = gae_batch["targets"][valid]
        transition_count = int(targets.numel())
        if transition_count <= 0:
            raise RuntimeError("DQC value critic received an empty rollout")

        with torch.no_grad():
            before_prediction = self.value_critic(features)
            before_rmse = torch.mean(
                (before_prediction - targets).square()).sqrt()

        weighted_loss = 0.0
        num_minibatches = int(math.ceil(
            transition_count / float(self.critic_minibatch_size)))
        for epoch in range(regression_epochs):
            # This callback observes immutable storage; production installs no hook.
            hook = getattr(self, "_value_epoch_hook", None)
            if hook is not None:
                hook(epoch, gae_batch["targets"])
            permutation = torch.randperm(
                transition_count, device=features.device)
            for indices in torch.tensor_split(permutation, num_minibatches):
                prediction = self.value_critic(features[indices])
                loss = 0.5 * torch.mean(
                    (prediction - targets[indices]).square())

                self.value_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.value_critic.parameters(), self.critic_grad_clip,
                    error_if_nonfinite=True)
                self.value_optimizer.step()
                self.value_optimizer_steps += 1
                weighted_loss += float(loss.item()) * int(indices.numel())

        with torch.no_grad():
            after_prediction = self.value_critic(features)
            after_rmse = torch.mean(
                (after_prediction - targets).square()).sqrt()
        return {
            "critic/value_loss": weighted_loss / float(
                transition_count * regression_epochs),
            "diagnostic/dqc/value_rmse_before": float(before_rmse.item()),
            "diagnostic/dqc/value_rmse": float(after_rmse.item()),
            "diagnostic/dqc/value_regression_epochs": float(regression_epochs),
            "diagnostic/dqc/value_optimizer_steps": float(
                self.value_optimizer_steps),
        }

    # ============================================================ score-function actor ============================================================
    @torch.no_grad()
    def actor_advantages(self, rollout, gae_batch=None):
        """
        Compute detached GAE-or-atom mean and action-marginal CDF advantages.

        Actual critic queries use A_exec. Each of M Gaussian A_raw baseline samples is clipped
        before its critic query. Under GAE-Mean the distributional atoms contribute only CDF risk;
        the independent scalar snapshot supplies the mean advantage.
        """
        states = rollout["S"]                                      # [T,B,D]
        mean_estimator = getattr(
            self, "dqc_mean_estimator", self.DISTRIBUTIONAL_MEAN)
        critic_features = self._critic_rollout_features(rollout)   # [T,B,Dcritic]
        atoms_action = self.reward_critic(critic_features, rollout["A_exec"])
        cdf_action = self._critic_cdf(atoms_action, rollout)

        # Replaying the complete sequence from zero state reconstructs the same history for each
        # valid [t,b]. RMS is still frozen at rollout-time moments until this batch finishes.
        means, log_std, _next_state = self.actor(states)
        sampled_q = []
        sampled_cdf = []
        for _ in range(self.num_action_samples):
            sampled_raw = means + torch.exp(log_std) * torch.randn_like(means)
            sampled_exec = self._executed_actions(sampled_raw)
            atoms = self.reward_critic(critic_features, sampled_exec)
            if mean_estimator == self.DISTRIBUTIONAL_MEAN:
                sampled_q.append(atoms.mean(dim=-1))
            sampled_cdf.append(self._critic_cdf(atoms, rollout))

        value_cdf = torch.stack(sampled_cdf, dim=0).mean(dim=0)
        cdf_advantage = (cdf_action - value_cdf).detach()
        if mean_estimator == self.GAE_MEAN:
            if gae_batch is None:
                raise ValueError("state_value_gae_v1 actor requires one frozen GAE batch")
            mean_advantage = gae_batch["advantages"].detach()
            value_mean = gae_batch["values"].detach()
        else:
            q_action = atoms_action.mean(dim=-1)
            value_mean = torch.stack(sampled_q, dim=0).mean(dim=0)
            mean_advantage = (q_action - value_mean).detach()

        T = int(states.shape[0])
        time = torch.arange(T, dtype=torch.float32, device=self.device).unsqueeze(1)
        mean_term = (self.gamma ** time) * mean_advantage
        # QCPO's finite population constraint gradient has no extra Abel discount. The historical
        # identity retains beta**t exactly; finite_nstep_v1 explicitly omits that multiplication.
        risk_discount = 1.0 if self._uses_finite_critic() else (self.beta ** time)
        risk_term = (
            self.lambda_probability.detach() * risk_discount * cdf_advantage)
        combined_raw = (mean_term - risk_term).detach()

        # GAE-Mean divides the complete valid population by one positive scalar and never centers
        # it. Therefore signs, the reward/risk balance, and fixed-q CDF semantics are unchanged.
        normalization_scale = torch.ones(
            (), dtype=combined_raw.dtype, device=combined_raw.device)
        if mean_estimator == self.GAE_MEAN:
            valid_values = combined_raw[rollout["valid"].bool()]
            if valid_values.numel() == 0:
                raise RuntimeError("DQC GAE actor received no valid rollout rows")
            normalization_scale = torch.clamp(
                valid_values.std(unbiased=False), min=1e-8)
        combined = (combined_raw / normalization_scale).detach()
        return {
            "combined": combined,
            "combined_raw": combined_raw,
            "combined_normalization_scale": normalization_scale.detach(),
            "mean_advantage": mean_advantage,
            "cdf_advantage": cdf_advantage,
            "mean_term": mean_term.detach(),
            "risk_term": risk_term.detach(),
            "cdf_action": cdf_action.detach(),
            "value_mean": value_mean.detach(),
            "value_cdf": value_cdf.detach(),
        }

    @staticmethod
    def _masked_diagonal_gaussian_kl(
            old_mean, old_log_std, new_mean, new_log_std, valid):
        """
        Return mean per-decision ``KL(N_old || N_new)`` over valid recurrent rows.

        The result sums independent action dimensions before averaging time/vector rows. It is a
        policy-displacement diagnostic, not a trajectory KL and not part of either actor loss.
        """
        old_variance = torch.exp(2.0 * old_log_std)
        new_variance = torch.exp(2.0 * new_log_std)
        elementwise_kl = (
            new_log_std - old_log_std
            + (old_variance + (old_mean - new_mean).square())
            / (2.0 * new_variance)
            - 0.5
        )
        mask = valid.to(elementwise_kl.dtype)
        row_kl = elementwise_kl.sum(dim=-1)
        return (row_kl * mask).sum() / torch.clamp(mask.sum(), min=1.0)

    def update_actor(self, rollout, combined):
        """
        Perform exactly one score-function Adam step with fixed B*horizon scaling.

        L = -sum(valid * log pi(A_raw|history) * stopgrad(W)) / (B*T_fixed).
        No old log-prob, likelihood ratio, PPO clip, GAE, entropy term, or normalization is used.
        """
        mean, log_std, _next_state = self.actor(rollout["S"])
        # ``log_std`` is an expanded view of the learnable actor parameter. ``detach()`` alone
        # would still share that storage, so Adam's in-place parameter update would silently turn
        # the "old" variance into the new variance and under-report this diagnostic KL. A clone is
        # required for a genuine pre-step policy snapshot; clone the mean as well for symmetry and
        # to keep this invariant true if the actor implementation later returns another view.
        old_mean = mean.detach().clone()
        old_log_std = log_std.detach().clone()
        log_probability = self._logp_from_params(
            rollout["A_raw"], mean, log_std)
        valid = rollout["valid"].to(log_probability.dtype)
        denominator = float(self.num_envs * self.horizon)
        loss = -(log_probability * combined.detach() * valid).sum() / denominator

        self.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        pre_clip_norm = nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.actor_grad_clip)
        self.actor_optimizer.step()
        self.actor_optimizer_steps += 1

        # Exact diagonal-Gaussian KL on the same recurrent histories after the one Adam step.
        # This is diagnostic only and never creates a PPO ratio or another optimization pass.
        with torch.no_grad():
            new_mean, new_log_std, _next_state = self.actor(rollout["S"])
            policy_kl = self._masked_diagonal_gaussian_kl(
                old_mean, old_log_std, new_mean, new_log_std, valid)
        return {
            "actor/loss": float(loss.item()),
            "diagnostic/dqc/actor_grad_norm_pre_clip": float(pre_clip_norm),
            "diagnostic/dqc/policy_kl_after_step": float(policy_kl.item()),
            "diagnostic/dqc/actor_optimizer_steps": float(self.actor_optimizer_steps),
        }

    def update_actor_ppo_epoch(
            self, rollout, combined, behavior_mean, behavior_log_std):
        """
        Apply one fixed-data PPO epoch to the detached DQC mean/CDF weight.

        The behavior likelihood is the rollout-time Gaussian density of ``A_raw``. The critic still
        consumes ``A_exec`` elsewhere; using the projected action in this ratio would be invalid
        because hard Box clipping creates boundary point masses. The denominator intentionally
        remains DQC's fixed ``B*horizon`` instead of QCPO_REF's valid-row normalization.
        """
        mean, log_std, _next_state = self.actor(rollout["S"])
        previous_mean = mean.detach().clone()
        previous_log_std = log_std.detach().clone()
        new_logp = self._logp_from_params(rollout["A_raw"], mean, log_std)
        behavior_logp = rollout.get("logp")
        if behavior_logp is None:
            raise ValueError("DQC PPO reuse requires rollout behavior logp")
        behavior_logp = behavior_logp.detach().to(new_logp.dtype)
        valid_rows = rollout["valid"].bool()
        if not bool(valid_rows.any()):
            raise RuntimeError("DQC PPO actor received no valid rollout rows")
        if (not bool(torch.isfinite(new_logp[valid_rows]).all())
                or not bool(torch.isfinite(behavior_logp[valid_rows]).all())
                or not bool(torch.isfinite(combined.detach()[valid_rows]).all())):
            raise FloatingPointError(
                "non-finite DQC PPO log probability or detached actor weight")

        # Per-decision importance weighting is the practical PPO surrogate. A 1000-step product of
        # prefix ratios would target the full changed state distribution but has unusable variance.
        ratio = torch.exp(new_logp - behavior_logp)
        clipped_ratio = torch.clamp(
            ratio, 1.0 - self.dqc_ratio_clip, 1.0 + self.dqc_ratio_clip)
        weight = combined.detach()
        surrogate = torch.minimum(ratio * weight, clipped_ratio * weight)
        valid = rollout["valid"].to(surrogate.dtype)
        denominator = float(self.num_envs * self.horizon)
        loss = -(surrogate * valid).sum() / denominator
        if not bool(torch.isfinite(ratio[valid_rows]).all()) or not bool(
                torch.isfinite(loss)):
            raise FloatingPointError("non-finite DQC PPO ratio or actor loss")

        self.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # PyTorch's explicit non-finite guard runs before Adam mutates parameters. A finite clipped
        # norm is not a KL bound; it only prevents NaN/Inf gradients from corrupting this lineage.
        pre_clip_norm = nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.actor_grad_clip, error_if_nonfinite=True)
        self.actor_optimizer.step()
        self.actor_optimizer_steps += 1

        # Exact Gaussian KL is evaluated after the optimizer step. The incremental value diagnoses
        # this Adam step; behavior_kl is the target-KL early-stop quantity across all applied epochs.
        with torch.no_grad():
            current_mean, current_log_std, _next_state = self.actor(rollout["S"])
            incremental_kl = self._masked_diagonal_gaussian_kl(
                previous_mean, previous_log_std,
                current_mean, current_log_std, rollout["valid"])
            behavior_kl = self._masked_diagonal_gaussian_kl(
                behavior_mean, behavior_log_std,
                current_mean, current_log_std, rollout["valid"])
            if (not bool(torch.isfinite(current_mean[valid_rows]).all())
                    or not bool(torch.isfinite(current_log_std[valid_rows]).all())
                    or not bool(torch.isfinite(incremental_kl))
                    or not bool(torch.isfinite(behavior_kl))):
                raise FloatingPointError(
                    "non-finite DQC PPO policy parameters or exact KL after actor step")
            valid_count = valid.sum().clamp(min=1.0)
            ratio_mean = (ratio * valid).sum() / valid_count
            ratio_min = ratio.masked_fill(~valid_rows, float("inf")).min()
            ratio_max = ratio.masked_fill(~valid_rows, float("-inf")).max()
            clip_fraction = (
                ((ratio - 1.0).abs() > self.dqc_ratio_clip).to(valid.dtype) * valid
            ).sum() / valid_count
            approx_kl = ((behavior_logp - new_logp) * valid).sum() / valid_count
        return {
            "actor/loss": float(loss.item()),
            "actor/ratio_mean": float(ratio_mean.item()),
            "actor/ratio_min": float(ratio_min.item()),
            "actor/ratio_max": float(ratio_max.item()),
            "actor/clip_fraction": float(clip_fraction.item()),
            "actor/approx_kl": float(approx_kl.item()),
            "diagnostic/dqc/actor_grad_norm_pre_clip": float(pre_clip_norm),
            "diagnostic/dqc/policy_kl_after_step": float(incremental_kl.item()),
            "diagnostic/dqc/policy_kl_behavior_after_epoch": float(behavior_kl.item()),
            "diagnostic/dqc/actor_optimizer_steps": float(self.actor_optimizer_steps),
        }

    def update_actor_epochs(self, rollout, combined):
        """
        Reuse one fixed detached DQC weight for up to E full-rollout actor Adam steps.

        ``actor_advantages()`` is intentionally called by the rollout owner before this method.
        Thus all E passes optimize the same materialized behavior actions and the same detached
        mean/CDF weight, just as an epoch ablation should. ``score_function`` preserves the original
        uncorrected reuse path. ``ppo_clip`` uses rollout likelihood ratios and may stop remaining
        epochs after the exact behavior-to-current KL reaches its configured target.
        """
        epochs = int(getattr(self, "actor_epochs", 1))
        if epochs <= 0:
            raise ValueError("actor_epochs must be positive")

        reuse_mode = str(getattr(
            self, "dqc_actor_reuse_mode", "score_function"))
        target_kl = float(getattr(self, "dqc_target_kl", 0.0))

        # PPO consumes the exact Gaussian parameters saved at action sampling time. The historical
        # score path reconstructs the same deterministic recurrent policy as before, preserving its
        # existing test surface for hand-built rollouts that predate collector policy snapshots.
        if reuse_mode == "ppo_clip":
            if "mean_old" not in rollout or "log_std_old" not in rollout:
                raise ValueError(
                    "DQC PPO reuse requires rollout mean_old and log_std_old")
            reference_mean = rollout["mean_old"].detach().clone()
            reference_log_std = rollout["log_std_old"].detach().clone()
        else:
            # The actor has no persistent hidden state, so this read-only replay cannot mutate
            # training state or consume action-sampling RNG.
            with torch.no_grad():
                reference_mean, reference_log_std, _next_state = self.actor(rollout["S"])
                # log_std is an expanded parameter view; clone owns storage across Adam steps.
                reference_mean = reference_mean.detach().clone()
                reference_log_std = reference_log_std.detach().clone()

        epoch_metrics = []
        behavior_kls = []
        early_stopped = False
        for _epoch in range(epochs):
            # Keeping this loop outside the critic update makes the optimizer count auditable and
            # prevents an accidental critic-E x actor-E nesting. The default takes the byte-for-byte
            # historical score primitive; only the explicit PPO identity consumes behavior logp.
            if reuse_mode == "score_function":
                epoch_info = self.update_actor(rollout, combined)
            elif reuse_mode == "ppo_clip":
                epoch_info = self.update_actor_ppo_epoch(
                    rollout, combined, reference_mean, reference_log_std)
                behavior_kl = float(
                    epoch_info["diagnostic/dqc/policy_kl_behavior_after_epoch"])
                behavior_kls.append(behavior_kl)
            else:
                raise ValueError(f"unsupported DQC actor reuse mode {reuse_mode!r}")
            epoch_metrics.append(epoch_info)

            # PPO clipping is samplewise and is not a hard policy-distance bound. Target KL is a
            # fail-safe checked after each completed Adam step; the triggering step is retained and
            # only the remaining epochs are skipped, matching common PPO early-stop semantics.
            if reuse_mode == "ppo_clip" and target_kl > 0.0 and behavior_kl >= target_kl:
                early_stopped = True
                self.actor_kl_early_stop_count = int(getattr(
                    self, "actor_kl_early_stop_count", 0)) + 1
                break

        # Existing scalar keys retain their historical last-step meaning. Separate mean/max keys
        # expose whether later off-policy score passes become progressively more aggressive.
        applied_epochs = len(epoch_metrics)
        last = dict(epoch_metrics[-1])
        step_kls = [item["diagnostic/dqc/policy_kl_after_step"] for item in epoch_metrics]
        epoch_losses = [item["actor/loss"] for item in epoch_metrics]
        epoch_grad_norms = [
            item["diagnostic/dqc/actor_grad_norm_pre_clip"] for item in epoch_metrics]
        last.update({
            "diagnostic/dqc/actor_epochs_requested": float(epochs),
            "diagnostic/dqc/actor_epochs_applied": float(applied_epochs),
            "diagnostic/dqc/actor_loss_epoch_mean": float(
                sum(epoch_losses) / applied_epochs),
            "diagnostic/dqc/actor_grad_norm_epoch_mean": float(
                sum(epoch_grad_norms) / applied_epochs),
            "diagnostic/dqc/policy_kl_step_mean": float(
                sum(step_kls) / applied_epochs),
            "diagnostic/dqc/policy_kl_step_max": float(max(step_kls)),
            "diagnostic/dqc/actor_kl_early_stop": float(early_stopped),
            "diagnostic/dqc/actor_kl_early_stop_count": float(getattr(
                self, "actor_kl_early_stop_count", 0)),
        })

        # Measure the whole E-step displacement from the rollout behavior policy. This is not a
        # penalty and does not alter gradients; it is the safety diagnostic missing from the
        # original one-step implementation when the same behavior batch is reused.
        with torch.no_grad():
            final_mean, final_log_std, _next_state = self.actor(rollout["S"])
            total_kl = self._masked_diagonal_gaussian_kl(
                reference_mean, reference_log_std,
                final_mean, final_log_std, rollout["valid"])
        last["diagnostic/dqc/policy_kl_after_epochs"] = float(total_kl.item())
        if behavior_kls:
            last["diagnostic/dqc/policy_kl_behavior_epoch_mean"] = float(
                sum(behavior_kls) / len(behavior_kls))
            last["diagnostic/dqc/policy_kl_behavior_epoch_max"] = float(max(behavior_kls))
        return last

    @staticmethod
    def _rms(values):
        """Root mean square with no centering; input is already restricted to valid rows."""
        return float(values.square().mean().sqrt().item()) if values.numel() else 0.0

    def _advantage_diagnostics(self, rollout, terms):
        """Summarize detached mean/risk components without changing their training scale."""
        valid = rollout["valid"].bool()
        mean_advantage = terms["mean_advantage"][valid]
        cdf_advantage = terms["cdf_advantage"][valid]
        mean_rms = self._rms(terms["mean_term"][valid])
        risk_rms = self._rms(terms["risk_term"][valid])

        # Three time buckets expose late-horizon risk-credit collapse without changing the loss.
        T = int(valid.shape[0])
        bucket_rms = {}
        boundaries = (0, T // 3, (2 * T) // 3, T)
        for label, start, stop in zip(
                ("early", "middle", "late"), boundaries[:-1], boundaries[1:]):
            bucket_mask = valid[start:stop]
            selected = terms["risk_term"][start:stop][bucket_mask]
            bucket_rms[
                f"diagnostic/dqc/risk_term_rms_{label}"
            ] = self._rms(selected)

        diagnostics = {
            "diagnostic/dqc/mean_advantage_mean": float(mean_advantage.mean().item()),
            "diagnostic/dqc/mean_advantage_std": float(
                mean_advantage.std(unbiased=False).item()),
            "diagnostic/dqc/cdf_advantage_mean": float(cdf_advantage.mean().item()),
            "diagnostic/dqc/cdf_advantage_std": float(
                cdf_advantage.std(unbiased=False).item()),
            "diagnostic/dqc/mean_term_rms": mean_rms,
            "diagnostic/dqc/risk_term_rms": risk_rms,
            "diagnostic/dqc/risk_to_mean": risk_rms / max(mean_rms, 1e-8),
            "diagnostic/dqc/critic_cdf_mean": float(
                terms["cdf_action"][valid].mean().item()),
        }
        diagnostics.update(bucket_rms)
        return diagnostics

    @staticmethod
    def action_clip_diagnostics(rollout):
        """Measure Box projection frequency/magnitude over valid raw action elements only."""
        valid = rollout["valid"].bool()
        difference = (rollout["A_raw"] - rollout["A_exec"]).abs()
        selected = difference[valid]                                  # [M,A]
        changed = selected > 0.0
        return {
            "diagnostic/action/raw_std": float(
                rollout["A_raw"][valid].std(unbiased=False).item()),
            "diagnostic/action/executed_std": float(
                rollout["A_exec"][valid].std(unbiased=False).item()),
            "diagnostic/action/clip_any_fraction": float(
                changed.any(dim=-1).float().mean().item()),
            "diagnostic/action/clip_element_fraction": float(
                changed.float().mean().item()),
            "diagnostic/action/mean_abs_raw_minus_exec": float(
                selected.mean().item()),
        }

    # ============================================================ empirical probability dual ============================================================
    def dual_step_size(self):
        """Return eta_k=a/(b+k)^c for the next momentum-free scalar update."""
        return lr_lambda(
            self.dual_update_count, self.lambda_a, self.lambda_b, self.lambda_c)

    @torch.no_grad()
    def update_dual(self, returns):
        """
        Update lambda from raw completed-trajectory G only; equality G=q is a failure.

        No critic bootstrap, budget, beta, or Safety cost appears in this empirical estimator.
        """
        probability = float(
            (returns <= self.return_threshold).float().mean().item())
        residual = probability - self.failure_alpha
        step = float(self.dual_step_size())
        self.lambda_probability.add_(step * residual).clamp_(0.0, self.lambda_max)
        self.dual_update_count += 1
        self.last_dual_prob = probability
        return probability, residual, step

    # ============================================================ one complete rollout update ============================================================
    def _update_rollout(self, iteration, rollout, initialization=False):
        """Apply either one critic-only initialization batch or one complete formal update."""
        rollout = self._attach_reward_budgets(rollout)
        iteration = int(iteration)
        in_warmup = bool(initialization)
        fitted_warmup = (
            in_warmup and self.critic_warmup_mode == "fitted_hard")

        # GAE labels are materialized before either critic changes. Initialization uses E4 for
        # both critics; a formal batch uses the signed scalar E8 while distributional QR remains
        # independently configured below. Actor, dual, RMS, and actor scheduler are untouched here.
        gae_batch = None
        info = {}
        mean_estimator = getattr(
            self, "dqc_mean_estimator", self.DISTRIBUTIONAL_MEAN)
        if mean_estimator == self.GAE_MEAN:
            gae_batch = self.prepare_gae_batch(rollout)
            value_epochs = (
                self.warmup_critic_epochs if in_warmup
                else self.dqc_value_epochs
            )
            info.update(self.update_value_critic(gae_batch, value_epochs))

        # Legacy mode remains one regression epoch followed by tau-Polyak. During the explicit
        # fitted branch, labels stay frozen for E full passes and target is hard-copied only after
        # all optimizer steps, so each rollout contributes one well-fitted Bellman generation.
        if fitted_warmup:
            info.update(self.update_critic(
                rollout, regression_epochs=self.warmup_critic_epochs,
                target_update_mode="hard"))
        else:
            info.update(self.update_critic(
                rollout, regression_epochs=int(getattr(self, "critic_epochs", 1)),
                target_update_mode=getattr(self, "target_update_mode", "polyak")))
        info.update(self.action_clip_diagnostics(rollout))
        info["diagnostic/dqc/fitted_warmup_active"] = float(fitted_warmup)

        probability = float(
            (rollout["disc_return"] <= self.return_threshold).float().mean().item())
        residual = probability - self.failure_alpha
        dual_lr = 0.0

        if not in_warmup:
            # The fitted branch leaves scheduler and optimizer param-group LR bitwise fixed during
            # warmup. Synchronize the absolute batch-index LR immediately before its first actor
            # Adam step; this creates neither an optimizer state nor a fake scheduler step.
            if self.critic_warmup_mode == "fitted_hard":
                self._set_actor_lr_batch_index(iteration)

            # Advantage sampling stays outside warmup: no baseline-action RNG is consumed and actor
            # Adam/log_std cannot change while the critic is being evaluated. One detached weight
            # is then reused for actor_epochs score passes; the dual still updates only once.
            if mean_estimator == self.GAE_MEAN:
                terms = self.actor_advantages(rollout, gae_batch)
            else:
                terms = self.actor_advantages(rollout)
            info.update(self._advantage_diagnostics(rollout, terms))
            info.update(self.update_actor_epochs(rollout, terms["combined"]))
            probability, residual, dual_lr = self.update_dual(
                rollout["disc_return"])
        else:
            # Stable logging schema during either warmup mode, without querying M action baselines.
            info.update({
                "diagnostic/dqc/mean_advantage_mean": 0.0,
                "diagnostic/dqc/mean_advantage_std": 0.0,
                "diagnostic/dqc/cdf_advantage_mean": 0.0,
                "diagnostic/dqc/cdf_advantage_std": 0.0,
                "diagnostic/dqc/mean_term_rms": 0.0,
                "diagnostic/dqc/risk_term_rms": 0.0,
                "diagnostic/dqc/risk_to_mean": 0.0,
                "diagnostic/dqc/risk_term_rms_early": 0.0,
                "diagnostic/dqc/risk_term_rms_middle": 0.0,
                "diagnostic/dqc/risk_term_rms_late": 0.0,
                "diagnostic/dqc/policy_kl_after_step": 0.0,
                "diagnostic/dqc/policy_kl_step_mean": 0.0,
                "diagnostic/dqc/policy_kl_step_max": 0.0,
                "diagnostic/dqc/policy_kl_after_epochs": 0.0,
                "diagnostic/dqc/actor_loss_epoch_mean": 0.0,
                "diagnostic/dqc/actor_grad_norm_epoch_mean": 0.0,
                "diagnostic/dqc/actor_epochs_requested": float(
                    getattr(self, "actor_epochs", 1)),
                "diagnostic/dqc/actor_epochs_applied": 0.0,
                "diagnostic/dqc/actor_kl_early_stop": 0.0,
                "diagnostic/dqc/actor_kl_early_stop_count": float(getattr(
                    self, "actor_kl_early_stop_count", 0)),
                "diagnostic/dqc/policy_kl_behavior_after_epoch": 0.0,
                "diagnostic/dqc/policy_kl_behavior_epoch_mean": 0.0,
                "diagnostic/dqc/policy_kl_behavior_epoch_max": 0.0,
                "diagnostic/dqc/actor_optimizer_steps": float(
                    self.actor_optimizer_steps),
            })

        # Public fields are post-update where applicable. q_est/constraint_prob are still computed
        # only from the same raw completed trajectories and never from diagnostic critic metrics.
        info.update({
            "lambda/probability_multiplier": float(self.lambda_probability.item()),
            "lambda/residual": float(residual),
            "lambda/lr": float(dual_lr),
            "constraint/batch_failure_probability": float(probability),
            "dqc/beta": self.beta,
            "diagnostic/dqc/actor_epochs_config": float(
                getattr(self, "actor_epochs", 1)),
            "diagnostic/dqc/critic_epochs_config": float(
                getattr(self, "critic_epochs", 1)),
            "diagnostic/dqc/actor_ppo_clip_config": float(
                getattr(self, "dqc_actor_reuse_mode", "score_function") == "ppo_clip"),
            "diagnostic/dqc/actor_ratio_clip_config": float(getattr(
                self, "dqc_ratio_clip", 0.1)),
            "diagnostic/dqc/actor_target_kl_config": float(getattr(
                self, "dqc_target_kl", 0.0)),
            "training/actor_lr": float(self.actor_optimizer.param_groups[0]["lr"]),
            "diagnostic/dqc/critic_n_step": float(getattr(self, "critic_n_step", 1)),
            "diagnostic/dqc/target_update_hard_config": float(
                getattr(self, "target_update_mode", "polyak") == "hard"),
            "training/critic_lr": float(self.critic_optimizer.param_groups[0]["lr"]),
            "diagnostic/dqc/mean_estimator_gae_config": float(
                mean_estimator == self.GAE_MEAN),
            "diagnostic/dqc/combined_advantage_normalized": float(
                mean_estimator == self.GAE_MEAN),
        })
        if getattr(self, "value_optimizer", None) is not None:
            info.update({
                "training/value_lr": float(
                    self.value_optimizer.param_groups[0]["lr"]),
                "diagnostic/dqc/value_epochs_config": float(
                    self.dqc_value_epochs),
            })

        # Initialization has a save-only boundary: return critic diagnostics without constructing
        # core metrics, touching RMS/scheduler, validating, selecting, or writing W&B history.
        if in_warmup:
            return info

        metrics = self._log_core(
            iteration,
            rollout["disc_return"].detach().cpu().numpy(),
            rollout["diagnostic_disc_cost"].detach().cpu().numpy(),
            rollout["diagnostic_raw_cost"].detach().cpu().numpy(),
            extra=info,
        )

        self.actor.update_obs_rms(rollout["S"][rollout["valid"]])
        self._set_actor_lr_batch_index(iteration + 1)
        return metrics

    def _run_initialization(self):
        """Run each critic-only batch once and atomically save its post-update cursor/RNG."""
        if not 0 <= self.initialization_cursor <= self.warmup_iters:
            raise ValueError("DQC initialization cursor is outside configured warmup batches")
        while self.initialization_cursor < self.warmup_iters:
            rollout = self._rollout_core(phase="initialization")
            self._update_rollout(-1, rollout, initialization=True)
            self.initialization_cursor += 1
            self._after_initialization_batch(self.warmup_iters)

    def train(self):
        """Collect one fresh B-trajectory rollout per iteration and upload one final W&B row."""
        print(
            f"DQCACBetaGPU[{self.dqc_update_mode}]: env={self.env_name} "
            f"target={self.critic_target_source} "
            f"horizon={self.horizon_mode} n={self.critic_n_step} target_update={self.target_update_mode} "
            f"warmup={self.critic_warmup_mode}/E{self.warmup_critic_epochs} "
            f"mean={self.dqc_mean_estimator} "
            f"beta={self.beta} alpha={self.failure_alpha} q={self.return_threshold} "
            f"minibatch={self.critic_minibatch_size} "
            f"critic_epochs={self.critic_epochs} actor_epochs={self.actor_epochs} "
            f"actor_reuse={self.dqc_actor_reuse_mode} "
            f"ratio_clip={self.dqc_ratio_clip} target_kl={self.dqc_target_kl}")
        self._run_initialization()
        for iteration in self.training_iterations():
            metrics = self._update_rollout(iteration, self._rollout_core())
            self._after_iteration(iteration, metrics)

    # ============================================================ checkpoint ============================================================
    def validate_checkpoint_state(self, state):
        """
        Reject replay-era/partial DQC states before any live parameter or optimizer mutation.

        This validation is called by utils.checkpoint immediately after the agent-class check and
        is not bypassed by --allow_checkpoint_transfer. Direct load_checkpoint_state calls repeat it.
        """
        super().validate_checkpoint_state(state)
        if not isinstance(state, dict):
            raise ValueError("DQC checkpoint agent state must be a dictionary")
        mode = state.get("dqc_update_mode")
        if "replay" in state or mode not in self.UPDATE_MODES:
            raise ValueError(
                "replay-era or unknown DQC checkpoint is incompatible with active identities; "
                "start a fresh clean-baseline run")
        if mode != self.dqc_update_mode:
            raise ValueError(
                "DQC dqc_update_mode mismatch: "
                f"checkpoint={mode!r}, requested={self.dqc_update_mode!r}; "
                "continuing and finite critics are never transferable")

        # Missing estimator metadata has exactly one historical interpretation: action-marginal
        # distributional atom mean. A GAE checkpoint must carry its full scalar-network identity.
        saved_mean_estimator = str(state.get(
            "dqc_mean_estimator", self.DISTRIBUTIONAL_MEAN))
        requested_mean_estimator = str(getattr(
            self, "dqc_mean_estimator", self.DISTRIBUTIONAL_MEAN))
        if saved_mean_estimator != requested_mean_estimator:
            raise ValueError(
                "DQC dqc_mean_estimator mismatch: "
                f"checkpoint={saved_mean_estimator!r}, "
                f"requested={requested_mean_estimator!r}")
        if requested_mean_estimator == self.GAE_MEAN:
            saved_value_identity = {
                "dqc_gae_lambda": float(state.get("dqc_gae_lambda", float("nan"))),
                "dqc_value_hidden": list(state.get("dqc_value_hidden", [])),
                "dqc_value_lr": float(state.get("dqc_value_lr", float("nan"))),
                "dqc_value_epochs": int(state.get("dqc_value_epochs", -1)),
                "normalize_combined_advantage": bool(state.get(
                    "normalize_combined_advantage", False)),
            }
            requested_value_identity = {
                "dqc_gae_lambda": self.dqc_gae_lambda,
                "dqc_value_hidden": self.dqc_value_hidden,
                "dqc_value_lr": self.dqc_value_lr,
                "dqc_value_epochs": self.dqc_value_epochs,
                "normalize_combined_advantage": self.normalize_combined_advantage,
            }
            for name, requested in requested_value_identity.items():
                saved = saved_value_identity[name]
                if saved != requested:
                    raise ValueError(
                        f"DQC {name} mismatch: checkpoint={saved!r}, "
                        f"requested={requested!r}")

        # Pre-switch checkpoints mean one_step. This preserves that historical default while a
        # finite-suffix checkpoint can never be resumed under different target semantics.
        saved_target_source = str(state.get("critic_target_source", "one_step"))
        if saved_target_source != self.critic_target_source:
            raise ValueError(
                "DQC critic_target_source mismatch: "
                f"checkpoint={saved_target_source!r}, "
                f"requested={self.critic_target_source!r}")

        # Missing fields identify an old legacy/one-epoch checkpoint. Agent-level validation is
        # deliberately stricter than generic signature transfer: fitted and legacy critic states
        # cannot be silently mixed even when a caller approves unrelated config differences.
        saved_warmup_mode = str(state.get("critic_warmup_mode", "legacy"))
        saved_warmup_epochs = int(state.get("warmup_critic_epochs", 1))
        if saved_warmup_mode != self.critic_warmup_mode:
            raise ValueError(
                "DQC critic_warmup_mode mismatch: "
                f"checkpoint={saved_warmup_mode!r}, "
                f"requested={self.critic_warmup_mode!r}")
        if saved_warmup_epochs != self.warmup_critic_epochs:
            raise ValueError(
                "DQC warmup_critic_epochs mismatch: "
                f"checkpoint={saved_warmup_epochs}, "
                f"requested={self.warmup_critic_epochs}")

        # Epoch counts change both optimizer state and behavior-policy reuse. Missing fields are
        # unambiguously the historical one-pass identity; even an explicit generic signature
        # transfer must not reinterpret that state as an E>1 ablation.
        saved_critic_epochs = int(state.get("critic_epochs", 1))
        saved_actor_epochs = int(state.get("actor_epochs", 1))
        requested_critic_epochs = int(getattr(self, "critic_epochs", 1))
        requested_actor_epochs = int(getattr(self, "actor_epochs", 1))
        if saved_critic_epochs != requested_critic_epochs:
            raise ValueError(
                "DQC critic_epochs mismatch: "
                f"checkpoint={saved_critic_epochs}, requested={requested_critic_epochs}")
        if saved_actor_epochs != requested_actor_epochs:
            raise ValueError(
                "DQC actor_epochs mismatch: "
                f"checkpoint={saved_actor_epochs}, requested={requested_actor_epochs}")

        # Actor-data reuse changes the optimized surrogate. Missing fields mean the historical
        # score-function path; PPO mode, clip width, and target-KL threshold cannot be transferred
        # into one another even when generic signature mismatch transfer was explicitly allowed.
        saved_reuse_mode = str(state.get(
            "dqc_actor_reuse_mode", "score_function"))
        saved_ratio_clip = float(state.get("dqc_ratio_clip", 0.1))
        saved_target_kl = float(state.get("dqc_target_kl", 0.0))
        if saved_reuse_mode != self.dqc_actor_reuse_mode:
            raise ValueError(
                "DQC dqc_actor_reuse_mode mismatch: "
                f"checkpoint={saved_reuse_mode!r}, requested={self.dqc_actor_reuse_mode!r}")
        if not math.isclose(
                saved_ratio_clip, self.dqc_ratio_clip, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "DQC dqc_ratio_clip mismatch: "
                f"checkpoint={saved_ratio_clip}, requested={self.dqc_ratio_clip}")
        if not math.isclose(
                saved_target_kl, self.dqc_target_kl, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "DQC dqc_target_kl mismatch: "
                f"checkpoint={saved_target_kl}, requested={self.dqc_target_kl}")
        # Old online checkpoints predate these fields and have one unambiguous legacy meaning.
        # A finite checkpoint must carry all values explicitly; mode mismatch above prevents an
        # old online state from borrowing finite defaults.
        saved_identity = {
            "horizon": int(state.get("horizon", 1000)),
            "horizon_mode": str(state.get("horizon_mode", "continuing")),
            "critic_n_step": int(state.get("critic_n_step", 1)),
            "critic_time_feature": str(state.get("critic_time_feature", "none")),
            "critic_actor_context": bool(state.get("critic_actor_context", False)),
            "target_update_mode": str(state.get("target_update_mode", "polyak")),
            "beta": float(state.get("beta", 0.995)),
            "warmup_iters": int(state.get("warmup_iters", 5)),
        }
        requested_identity = {
            "horizon": int(getattr(self, "horizon", 1000)),
            "horizon_mode": getattr(self, "horizon_mode", "continuing"),
            "critic_n_step": int(getattr(self, "critic_n_step", 1)),
            "critic_time_feature": getattr(self, "critic_time_feature", "none"),
            "critic_actor_context": bool(getattr(self, "critic_actor_context", False)),
            "target_update_mode": getattr(self, "target_update_mode", "polyak"),
            "beta": float(self.beta),
            "warmup_iters": int(getattr(self, "warmup_iters", 5)),
        }
        for name, requested in requested_identity.items():
            saved = saved_identity[name]
            if saved != requested:
                raise ValueError(
                    f"DQC {name} mismatch: checkpoint={saved!r}, "
                    f"requested={requested!r}")
        cursor = int(state.get("initialization_cursor", -1))
        if not 0 <= cursor <= self.warmup_iters:
            raise ValueError("DQC initialization cursor exceeds configured warmup_iters")
        if int(state["main_env_steps"]) > 0 and cursor != self.warmup_iters:
            raise ValueError("DQC formal checkpoint has incomplete initialization cursor")

        required = {
            "reward_critic", "reward_target_critic", "critic_optimizer",
            "actor_optimizer", "actor_scheduler", "lambda_probability",
            "actor_optimizer_steps", "critic_optimizer_steps",
        }
        if requested_mean_estimator == self.GAE_MEAN:
            required.update({
                "value_critic", "value_optimizer", "value_optimizer_steps"})
        missing = sorted(required - set(state))
        if missing:
            raise ValueError(
                "incomplete online DQC checkpoint state: " + ", ".join(missing))

    def checkpoint_state(self):
        """Save the full online state; no transition data or replay cursor exists."""
        state = super().checkpoint_state()
        state.update({
            "dqc_update_mode": self.dqc_update_mode,
            "critic_target_source": self.critic_target_source,
            "critic_minibatch_size": self.critic_minibatch_size,
            "critic_epochs": self.critic_epochs,
            "actor_epochs": self.actor_epochs,
            "dqc_actor_reuse_mode": self.dqc_actor_reuse_mode,
            "dqc_ratio_clip": self.dqc_ratio_clip,
            "dqc_target_kl": self.dqc_target_kl,
            "dqc_mean_estimator": getattr(
                self, "dqc_mean_estimator", self.DISTRIBUTIONAL_MEAN),
            "dqc_gae_lambda": getattr(self, "dqc_gae_lambda", 0.97),
            "dqc_value_hidden": getattr(self, "dqc_value_hidden", [512, 512]),
            "dqc_value_lr": getattr(self, "dqc_value_lr", 3e-4),
            "dqc_value_epochs": getattr(self, "dqc_value_epochs", 8),
            "normalize_combined_advantage": getattr(
                self, "normalize_combined_advantage", False),
            "horizon": getattr(self, "horizon", 1000),
            "horizon_mode": getattr(self, "horizon_mode", "continuing"),
            "critic_n_step": getattr(self, "critic_n_step", 1),
            "critic_time_feature": getattr(self, "critic_time_feature", "none"),
            "critic_actor_context": getattr(self, "critic_actor_context", False),
            "target_update_mode": getattr(self, "target_update_mode", "polyak"),
            "beta": self.beta,
            "warmup_iters": getattr(self, "warmup_iters", 5),
            "critic_warmup_mode": self.critic_warmup_mode,
            "warmup_critic_epochs": self.warmup_critic_epochs,
            "reward_critic": self.reward_critic.state_dict(),
            "reward_target_critic": self.reward_target_critic.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_scheduler": self.actor_scheduler.state_dict(),
            "lambda_probability": self.lambda_probability.detach().cpu(),
            "dual_update_count": self.dual_update_count,
            "last_dual_prob": self.last_dual_prob,
            "actor_optimizer_steps": self.actor_optimizer_steps,
            "critic_optimizer_steps": self.critic_optimizer_steps,
            "target_update_count": self.target_update_count,
            "actor_kl_early_stop_count": self.actor_kl_early_stop_count,
        })
        if getattr(self, "dqc_mean_estimator", self.DISTRIBUTIONAL_MEAN) == self.GAE_MEAN:
            state.update({
                "value_critic": self.value_critic.state_dict(),
                "value_optimizer": self.value_optimizer.state_dict(),
                "value_optimizer_steps": self.value_optimizer_steps,
            })
        return state

    def load_checkpoint_state(self, state):
        """Restore a fully validated online checkpoint with target/optimizer/counter state."""
        self.validate_checkpoint_state(state)
        super().load_checkpoint_state(state)
        self.reward_critic.load_state_dict(state["reward_critic"])
        self.reward_target_critic.load_state_dict(state["reward_target_critic"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        mean_estimator = getattr(
            self, "dqc_mean_estimator", self.DISTRIBUTIONAL_MEAN)
        if mean_estimator == self.GAE_MEAN:
            self.value_critic.load_state_dict(state["value_critic"])
            self.value_optimizer.load_state_dict(state["value_optimizer"])
            self.value_optimizer_steps = int(state["value_optimizer_steps"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.actor_scheduler.load_state_dict(state["actor_scheduler"])
        self.lambda_probability.copy_(state["lambda_probability"].to(self.device))
        self.dual_update_count = int(state.get("dual_update_count", 0))
        self.last_dual_prob = float(state.get("last_dual_prob", 0.0))
        self.actor_optimizer_steps = int(state["actor_optimizer_steps"])
        self.critic_optimizer_steps = int(state["critic_optimizer_steps"])
        self.target_update_count = int(state.get("target_update_count", 0))
        self.actor_kl_early_stop_count = int(state.get(
            "actor_kl_early_stop_count", 0))

    def get_training_summary(self):
        """Expose empirical feasibility and optimizer counts without replay-era fields."""
        return {
            "objective": "mean_reward_subject_to_direct_failure_probability_dqc_approximation",
            "dqc_update_mode": self.dqc_update_mode,
            "critic_target_source": self.critic_target_source,
            "horizon": self.horizon,
            "horizon_mode": self.horizon_mode,
            "critic_n_step": self.critic_n_step,
            "critic_time_feature": self.critic_time_feature,
            "critic_actor_context": self.critic_actor_context,
            "target_update_mode": self.target_update_mode,
            "critic_warmup_mode": self.critic_warmup_mode,
            "warmup_critic_epochs": self.warmup_critic_epochs,
            "critic_epochs": self.critic_epochs,
            "actor_epochs": self.actor_epochs,
            "dqc_actor_reuse_mode": self.dqc_actor_reuse_mode,
            "dqc_ratio_clip": self.dqc_ratio_clip,
            "dqc_target_kl": self.dqc_target_kl,
            "dqc_mean_estimator": getattr(
                self, "dqc_mean_estimator", self.DISTRIBUTIONAL_MEAN),
            "dqc_gae_lambda": getattr(self, "dqc_gae_lambda", 0.97),
            "dqc_value_epochs": getattr(self, "dqc_value_epochs", 8),
            "normalize_combined_advantage": getattr(
                self, "normalize_combined_advantage", False),
            "lambda_probability_final": float(self.lambda_probability.item()),
            "empirical_prob": self.last_empirical_prob,
            "dual_prob": self.last_dual_prob,
            "return_threshold": self.return_threshold,
            "failure_alpha": self.failure_alpha,
            "beta": self.beta,
            "actor_optimizer_steps": self.actor_optimizer_steps,
            "critic_optimizer_steps": self.critic_optimizer_steps,
            "value_optimizer_steps": getattr(
                self, "value_optimizer_steps", 0),
            "target_update_count": self.target_update_count,
            "actor_kl_early_stop_count": self.actor_kl_early_stop_count,
        }
