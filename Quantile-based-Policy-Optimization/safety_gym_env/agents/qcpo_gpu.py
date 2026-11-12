# -*- coding: utf-8 -*-
"""
Direct trajectory Monte-Carlo reward chance constraint.

max_theta E[G_T] subject to P(G_T <= q) <= alpha
L(theta,lambda)=E[G_T]-lambda(P(G_T<=q)-alpha), lambda>=0.
"""
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from .common import lr_lambda
from .vec_base import VecAgentBase


class QCPOGPU(VecAgentBase):
    """Fresh on-policy score-function estimator with empirical projected dual update."""

    @staticmethod
    def _gradient_l2_norm(gradients):
        """
        Return the global L2 norm of one parameter-gradient tuple.

        ``torch.autograd.grad(..., allow_unused=True)`` may emit ``None`` for parameters that do
        not affect a component loss. Every real gradient tensor keeps its native device; summing
        squared entries before one final ``item()`` avoids a CPU synchronization per parameter.
        """
        squared_norm = None
        for gradient in gradients:
            if gradient is None:
                continue
            contribution = gradient.detach().float().square().sum()
            squared_norm = contribution if squared_norm is None else squared_norm + contribution
        if squared_norm is None:
            return 0.0
        return float(torch.sqrt(squared_norm.clamp_min(0.0)).item())

    def __init__(self, args, env):
        super().__init__(args, env)
        self._require_return_threshold()
        self.actor_grad_clip = float(getattr(args, "actor_grad_clip", 1.0))
        self.normalize_combined_advantage = bool(
            getattr(args, "normalize_combined_advantage", True))
        self.lambda_dual = torch.tensor(
            float(getattr(args, "penalty_init", 0.0)), dtype=torch.float32, device=self.device)
        self.lambda_min = 0.0
        self.lambda_max = float(getattr(args, "lambda_max", 50.0))
        self.dual_update_count = 0
        self.lambda_a = float(args.lambda_a)
        self.lambda_b = float(args.lambda_b)
        self.lambda_c = float(args.lambda_c)
        self.optimizer = Adam(self.actor.parameters(), lr=1.0, eps=1e-5)
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda k: lr_lambda(k, args.theta_a, args.theta_b, args.theta_c))
        self.last_dual_prob = 0.0

    def trajectory_terms(self, returns):
        """
        Build unbiased action-independent-baseline terms per trajectory.

        Raw G and the inclusive event are not normalized. For B>1, each trajectory subtracts
        the other B-1 samples mean, so its baseline is independent of its own action. For B=1,
        no baseline is available and the raw score weight is retained. Optional conditioning
        divides only the already-combined advantage by one common positive scale.
        """
        failures = (returns <= self.return_threshold).to(torch.float32)
        if returns.numel() > 1:
            # Leave-one-out baselines avoid the finite-batch bias from including sample i in
            # its own batch mean: b_i=(sum_j x_j-x_i)/(B-1).
            denominator = float(returns.numel() - 1)
            mean_baseline = (returns.sum() - returns) / denominator
            failure_baseline = (failures.sum() - failures) / denominator
            mean_advantage = returns - mean_baseline
            failure_advantage = failures - failure_baseline
        else:
            mean_advantage = returns.clone()
            failure_advantage = failures.clone()
        combined = mean_advantage - self.lambda_dual.detach() * failure_advantage
        if self.normalize_combined_advantage:
            scale = combined.std(unbiased=False).clamp(min=1e-6)
            combined = combined / scale
        return mean_advantage, failure_advantage, combined, failures

    def update_actor(self, rollout):
        """
        Take one fresh score-function step and expose detached QCPO diagnostics.

        The optimization loss is unchanged from
        ``-log_pi * (A_mean-lambda*A_constraint) / common_scale``. For diagnostics only, one
        additional autograd pass obtains the mean component. The actual lambda-scaled constraint
        gradient is then ``combined_gradient - mean_gradient``. This decomposition neither writes
        into ``parameter.grad`` during the extra pass nor changes the gradient sent to Adam.
        """
        mean_adv, failure_adv, combined, failures = self.trajectory_terms(
            rollout["disc_return"].detach())
        weights = combined.unsqueeze(0).expand_as(rollout["valid"])
        log_probability = self._compute_log_probs(rollout["S"], rollout["A"])
        valid = rollout["valid"].to(log_probability.dtype)

        # Reconstruct the common positive conditioning scale used by trajectory_terms. It is a
        # detached scalar because both trajectory advantages and lambda are detached from actor
        # parameters. The mean component must use the same scale as the combined training loss.
        lambda_used = self.lambda_dual.detach()
        combined_raw = mean_adv - lambda_used * failure_adv
        if self.normalize_combined_advantage:
            common_scale = combined_raw.std(unbiased=False).clamp(min=1e-6)
        else:
            common_scale = torch.ones((), dtype=combined_raw.dtype, device=combined_raw.device)
        mean_weights = (mean_adv / common_scale).unsqueeze(0).expand_as(rollout["valid"])
        denominator = torch.clamp(valid.sum(), min=1.0)

        # Minimize negative mean return plus the positive failure-probability penalty. Their sum
        # is algebraically identical to the original combined actor loss below.
        mean_component_loss = -(
            log_probability * mean_weights.detach() * valid
        ).sum() / denominator
        actor_loss = -(
            log_probability * weights.detach() * valid
        ).sum() / denominator

        # autograd.grad returns tensors without accumulating into parameter.grad. Retaining the
        # graph lets the following actor_loss.backward() produce the exact gradient used by Adam.
        parameters = tuple(parameter for parameter in self.actor.parameters()
                           if parameter.requires_grad)
        self.optimizer.zero_grad(set_to_none=True)
        mean_gradients = torch.autograd.grad(
            mean_component_loss, parameters, retain_graph=True, allow_unused=True)
        actor_loss.backward()

        # Before clipping, parameter.grad is the exact combined gradient. Subtracting the mean
        # component gives the actual lambda-scaled constraint contribution, including its sign.
        combined_gradients = tuple(parameter.grad for parameter in parameters)
        constraint_gradients = tuple(
            None if combined_gradient is None and mean_gradient is None else
            (torch.zeros_like(mean_gradient) if combined_gradient is None else combined_gradient)
            - (torch.zeros_like(combined_gradient) if mean_gradient is None else mean_gradient)
            for combined_gradient, mean_gradient in zip(combined_gradients, mean_gradients)
        )
        mean_gradient_norm = self._gradient_l2_norm(mean_gradients)
        constraint_gradient_norm = self._gradient_l2_norm(constraint_gradients)
        combined_gradient_norm_pre_clip = self._gradient_l2_norm(combined_gradients)

        if self.actor_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip)
        combined_gradient_norm_post_clip = self._gradient_l2_norm(
            tuple(parameter.grad for parameter in parameters))
        self.optimizer.step()

        # LOO advantages have zero batch mean for B>1, so population std is also their RMS and is
        # more informative than logging a nearly constant zero mean. risk_to_mean deliberately
        # uses pre-normalization scales, matching the previously agreed diagnostic definition.
        mean_advantage_std = mean_adv.std(unbiased=False)
        constraint_advantage_std = failure_adv.std(unbiased=False)
        scaled_constraint_advantage_std = lambda_used * constraint_advantage_std
        risk_to_mean = scaled_constraint_advantage_std / mean_advantage_std.clamp(min=1e-8)
        return {
            "actor/loss": float(actor_loss.item()),
            "constraint/batch_failure_probability": float(failures.mean().item()),
            "diagnostic/qcpo_mean_advantage_std": float(mean_advantage_std.item()),
            "diagnostic/qcpo_constraint_advantage_std":
                float(constraint_advantage_std.item()),
            "diagnostic/qcpo_lambda_scaled_constraint_advantage_std":
                float(scaled_constraint_advantage_std.item()),
            "diagnostic/qcpo_risk_to_mean": float(risk_to_mean.item()),
            "diagnostic/qcpo_mean_score_grad_norm": mean_gradient_norm,
            # This is already multiplied by lambda_used: it is the penalty contribution that
            # actually entered the actor update, not the raw gradient of failure probability.
            "diagnostic/qcpo_constraint_score_grad_norm": constraint_gradient_norm,
            "diagnostic/qcpo_combined_grad_norm_pre_clip":
                combined_gradient_norm_pre_clip,
            "diagnostic/qcpo_combined_grad_norm_post_clip":
                combined_gradient_norm_post_clip,
        }

    def dual_step_size(self):
        """eta_lambda(k)=a/(b+k)^c for an interpretable, momentum-free scalar update."""
        return lr_lambda(self.dual_update_count, self.lambda_a, self.lambda_b, self.lambda_c)

    @torch.no_grad()
    def update_dual(self, returns):
        """lambda <- clip(lambda + eta*(p_hat-alpha), 0, lambda_max)."""
        probability = float((returns <= self.return_threshold).float().mean().item())
        residual = probability - self.failure_alpha
        step = self.dual_step_size()
        self.lambda_dual.add_(step * residual)
        self.lambda_dual.clamp_(min=self.lambda_min, max=self.lambda_max)
        self.dual_update_count += 1
        self.last_dual_prob = probability
        return probability, residual, float(step)

    def train(self):
        """Alternate one fresh MC actor update and one empirical probability dual update."""
        print(f"QCPOGPU[reward-chance]: env={self.env_name} alpha={self.failure_alpha} "
              f"q={self.return_threshold} B={self.num_envs} T={self.n}")
        for iteration in self.training_iterations():
            rollout = self._rollout_core()
            actor_info = self.update_actor(rollout)
            self.update_dual(rollout["disc_return"].detach())
            # This is deliberately the sole logged lambda value. The actor used lambda_k above;
            # after the empirical dual step finishes, terminal output and W&B both expose the
            # latest completed-batch state lambda_{k+1}.
            info = {
                **actor_info,
                "lambda/probability_multiplier": float(self.lambda_dual.item()),
                "training/actor_lr": float(self.scheduler.get_last_lr()[0]),
            }
            iteration_metrics = self._log_core(
                iteration,
                rollout["disc_return"].detach().cpu().numpy(),
                rollout["diagnostic_disc_cost"].detach().cpu().numpy(),
                rollout["diagnostic_raw_cost"].detach().cpu().numpy(),
                extra=info)
            self.actor.update_obs_rms(rollout["S"][rollout["valid"]])
            self.scheduler.step()
            self._after_iteration(iteration, iteration_metrics)

    def checkpoint_state(self):
        """Save probability multiplier under a formulation-specific key."""
        state = super().checkpoint_state()
        state.update({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "lambda_probability": self.lambda_dual.detach().cpu(),
            "dual_update_count": self.dual_update_count,
        })
        return state

    def load_checkpoint_state(self, state):
        """Restore actor optimizer and direct-chance multiplier without quantile-key aliasing."""
        super().load_checkpoint_state(state)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.lambda_dual.copy_(state["lambda_probability"].to(self.device))
        self.dual_update_count = int(state.get("dual_update_count", 0))

    def get_training_summary(self):
        """Return direct empirical chance state."""
        return {
            "objective": "mean_reward_subject_to_direct_failure_probability",
            "lambda_probability_final": float(self.lambda_dual.item()),
            "empirical_prob": self.last_empirical_prob,
            "dual_prob": self.last_dual_prob,
            "return_threshold": self.return_threshold,
            "failure_alpha": self.failure_alpha,
        }
