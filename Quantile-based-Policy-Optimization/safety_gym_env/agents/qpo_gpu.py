# -*- coding: utf-8 -*-
"""
Safety Gym QPO: pure lower-quantile maximization.

Objective: max_theta Q_alpha(G_T), G_T=sum_{t=0}^{T-1} gamma^t r_t.
This baseline has no fixed-threshold constraint, mean objective, chance residual, or dual.
An optional return threshold is consumed only by common logging/evaluation after training.
"""
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from .common import lr_lambda, indicator
from .vec_base import VecAgentBase
from utils.risk_metrics import linear_empirical_quantile


class QPOGPU(VecAgentBase):
    """Two-timescale QPO with one fresh on-policy actor update per rollout."""

    def __init__(self, args, env):
        super().__init__(args, env)
        # QPO may train without q; if supplied, q affects only common evaluation metrics.
        self.actor_grad_clip = float(getattr(args, "actor_grad_clip", 1.0))
        self.warmup_q_iters = max(1, int(getattr(args, "warmup_q_iters", 2)))
        self.q_est = None
        # Save pooled initialization samples so a -1 checkpoint resumes without recollection or
        # RNG drift.
        self.initialization_returns = []
        self.q_a = float(args.q_a)
        self.q_b = float(args.q_b)
        self.q_c = float(args.q_c)
        # Actor LR follows the project Robbins-Monro schedule through LambdaLR.
        self.optimizer = Adam(self.actor.parameters(), lr=1.0, eps=1e-5)
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda k: lr_lambda(k, args.theta_a, args.theta_b, args.theta_c))
        self.update_count = 0

    def _q_step_size(self):
        """Current scalar quantile-tracker step eta_Q(k)=a/(b+k)^c."""
        return lr_lambda(self.update_count, self.q_a, self.q_b, self.q_c)

    def _initialize_quantile(self):
        """Estimate Q0 from resumable rollout-only initialization batches."""
        if not 0 <= self.initialization_cursor <= self.warmup_q_iters:
            raise ValueError("QPO initialization cursor is outside configured warmup batches")
        self._write_external_progress(
            completed=self.start_iteration, metrics={}, phase="initialization")
        warmup = range(self.initialization_cursor, self.warmup_q_iters)
        if self.show_progress:
            # The cursor is the number of atomically saved batches, so the bar resumes exactly.
            warmup = tqdm(
                warmup, total=self.warmup_q_iters, initial=self.initialization_cursor,
                desc=f"{self.algo_name}/{self.env_name}/initialization",
                unit="batch", dynamic_ncols=True,
                mininterval=self.progress_mininterval, leave=True)
        for _ in warmup:
            rollout = self._rollout_core(phase="initialization")
            self.initialization_returns.append(rollout["disc_return"].detach().cpu().clone())
            self.initialization_cursor += 1
            self._after_initialization_batch(self.warmup_q_iters)
        initial = linear_empirical_quantile(
            torch.cat(self.initialization_returns).numpy(), self.failure_alpha)
        self.q_est = torch.tensor(float(initial), dtype=torch.float32, device=self.device)
        return initial

    def _update_actor(self, rollout):
        """
        QPO score-function step.

        indicator_i=1{G_i<=Q}; the scalar is broadcast across valid trajectory timesteps.
        Minimizing E[log pi * indicator] decreases mass below the tracked quantile.
        """
        trajectory_indicator = indicator(self.q_est.detach(), rollout["disc_return"]).float()
        weights = trajectory_indicator.unsqueeze(0).expand_as(rollout["valid"])
        log_probability = self._compute_log_probs(rollout["S"], rollout["A"])
        valid = rollout["valid"].to(log_probability.dtype)
        loss = (log_probability * weights * valid).sum() / torch.clamp(valid.sum(), min=1.0)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.actor_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip)
        self.optimizer.step()
        return {"actor/loss": float(loss.item())}

    @torch.no_grad()
    def _update_quantile(self, returns):
        """Update Q from the current B-trajectory probability; ties are included by <=."""
        probability = indicator(self.q_est, returns).float().mean()
        step = self._q_step_size()
        self.q_est.add_(step * (self.failure_alpha - probability))
        return float(probability.item())

    def train(self):
        """Warm up Q, then alternate one fresh actor step and one faster Q step."""
        initial = self._initialize_quantile() if self.q_est is None else float(self.q_est.item())
        print(f"QPOGPU[pure-quantile]: env={self.env_name} alpha={self.failure_alpha} "
              f"Q0={initial:.3f} B={self.num_envs} T={self.n}")
        for iteration in self.training_iterations():
            rollout = self._rollout_core()
            actor_info = self._update_actor(rollout)
            constraint_probability = self._update_quantile(rollout["disc_return"])
            info = {
                **actor_info,
                "quantile/q_est": float(self.q_est.item()),
                "quantile/constraint_prob": constraint_probability,
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
            self.update_count += 1
            self._after_iteration(iteration, iteration_metrics)

    def checkpoint_state(self):
        """Include initialization samples plus formal optimizer and quantile state."""
        state = super().checkpoint_state()
        state.update({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "q_est": None if self.q_est is None else self.q_est.detach().cpu(),
            "update_count": self.update_count,
            "initialization_returns": [
                value.detach().cpu().clone() for value in self.initialization_returns],
        })
        return state

    def validate_checkpoint_state(self, state):
        """Validate base clocks and the atomically resumable quantile sample."""
        super().validate_checkpoint_state(state)
        cursor = int(state.get("initialization_cursor", -1))
        samples = state.get("initialization_returns")
        if not isinstance(samples, list) or len(samples) != cursor:
            raise ValueError("QPO initialization sample count does not match cursor")
        if not 0 <= cursor <= self.warmup_q_iters:
            raise ValueError("QPO initialization cursor exceeds configured warmup_q_iters")
        if int(state["main_env_steps"]) > 0 and cursor != self.warmup_q_iters:
            raise ValueError("QPO formal checkpoint has incomplete initialization cursor")
        if int(state["main_env_steps"]) > 0 and state.get("q_est") is None:
            raise ValueError("QPO formal checkpoint is missing initialized q_est")

    def load_checkpoint_state(self, state):
        """Restore QPO so its next fixed-batch update matches uninterrupted training."""
        self.validate_checkpoint_state(state)
        super().load_checkpoint_state(state)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        value = state.get("q_est")
        self.q_est = None if value is None else value.to(self.device).float()
        self.update_count = int(state.get("update_count", 0))
        self.initialization_returns = [
            torch.as_tensor(item, dtype=torch.float32, device="cpu").clone()
            for item in state["initialization_returns"]]

    def get_training_summary(self):
        """Expose QPO identity and final tracked quantile without chance-solver claims."""
        return {
            "objective": "maximize_reward_quantile",
            "empirical_quantile_method": "linear",
            "q_est_final": None if self.q_est is None else float(self.q_est.item()),
            "failure_alpha": self.failure_alpha,
            "num_envs": self.num_envs,
            "num_iterations": self.num_iterations,
        }
