# -*- coding: utf-8 -*-
"""
Reward-side Safety Gym agent base.

The primary random variable is the raw-reward truncated return
G_T = sum_{t=0}^{T-1} gamma^t r_t. Environment cost is collected only for
``diagnostic/environment_cost_*`` metrics and never enters policy inputs or updates.
"""
import json
import math
import os
import time
import numpy as np
import torch
import wandb
from tqdm.auto import tqdm

from utils.model import RecurrentGaussianActor
from utils.risk_metrics import return_statistics
from envs import make_vec_env


class VecAgentBase:
    """Shared recurrent actor, boundary-correct collector, logging, and checkpoint hooks."""

    def __init__(self, args, env):
        # -------------------- fixed research semantics --------------------
        self.device = args.device
        self.gamma = float(args.gamma)
        self.failure_alpha = float(getattr(args, "failure_alpha", getattr(args, "q_alpha", 0.2)))
        self.q_alpha = self.failure_alpha                 # backward-compatible hyperparameter name
        # Training curves and scheduled events are indexed only by post-initialization interaction.
        # This identity is checkpoint-strict because old DQC checkpoints counted critic warmup
        # inside the advertised budget and cannot be resumed onto the new main-step axis.
        expected_budget_semantics = "main_excludes_initialization_v1"
        self.training_budget_semantics = str(getattr(
            args, "training_budget_semantics", expected_budget_semantics))
        if self.training_budget_semantics != expected_budget_semantics:
            raise ValueError(f"unsupported training_budget_semantics={self.training_budget_semantics!r}")
        threshold = getattr(args, "return_threshold", None)
        self.return_threshold = None if threshold is None else float(threshold)
        self.horizon = int(getattr(args, "horizon", 1000))
        self.num_envs = max(1, int(getattr(args, "num_envs", 8)))
        self.num_iterations = max(1, int(getattr(args, "num_iterations", 1)))
        self.log_interval = max(1, int(getattr(args, "log_interval", 10)))
        self.show_progress = bool(getattr(args, "show_progress", True))
        self.progress_mininterval = max(
            0.1, float(getattr(args, "progress_mininterval", 1.0)))
        self.algo_name = str(getattr(args, "algo_name", self.__class__.__name__))
        self.env_name = str(getattr(args, "env_name", env.env_id))
        self.seed = int(getattr(args, "seed", 0))
        self.eval_env = env

        # -------------------- vector environment --------------------
        self.vec_env = make_vec_env(
            env.env_id, self.num_envs, self.horizon, self.device,
            ref_env=env, seed=int(getattr(args, "seed", 0)),
            backend=getattr(args, "vec_backend", "mp"))
        self.n = self.vec_env.n
        self.state_dim = self.vec_env.obs_dim
        self.action_dim = self.vec_env.act_dim
        self._log_2pi = float(math.log(2.0 * math.pi))
        # The policy distribution lives on an unbounded Gaussian raw action.  Safety Gym,
        # however, advances its physics with Box-clipped actions.  Keep the exact per-dimension
        # bounds on the policy device so DQC can condition its critic on what the environment
        # actually executed while PPO continues to evaluate the latent Gaussian likelihood.
        self._action_low = torch.as_tensor(
            np.asarray(env.action_space.low, dtype=np.float32).reshape(-1),
            dtype=torch.float32, device=self.device)
        self._action_high = torch.as_tensor(
            np.asarray(env.action_space.high, dtype=np.float32).reshape(-1),
            dtype=torch.float32, device=self.device)

        # -------------------- shared policy capacity --------------------
        hidden = getattr(args, "actor_hidden", [512, 512])
        if isinstance(hidden, str):
            hidden = [int(x) for x in hidden.split(",") if x.strip()]
        self.actor = RecurrentGaussianActor(
            self.state_dim, self.action_dim, hidden=hidden,
            lstm_size=int(getattr(args, "lstm_size", 512)),
            init_std=float(getattr(args, "init_std", 1.0)),
            learn_std=bool(getattr(args, "learn_std", True)),
            normalize_observation=bool(getattr(args, "normalize_observation", True)),
        ).to(self.device)
        self.lstm_size = self.actor.lstm_size
        self._eval_rnn_state = None

        # -------------------- W&B, with env-step as shared x-axis --------------------
        tags = [x.strip() for x in str(getattr(args, "wandb_tags", "") or "").split(",") if x.strip()]
        self._wandb_run = wandb.init(
            project=getattr(args, "wandb_project", "safety_gym_reward_risk"),
            name=getattr(args, "wandb_name", None) or f"{self.algo_name}_{args.seed}",
            config={k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v
                    for k, v in vars(args).items()},
            reinit=True, group=getattr(args, "wandb_group", None) or self.algo_name,
            notes=getattr(args, "wandb_notes", None), tags=tags or None,
            dir=getattr(args, "wandb_dir", None))
        if self._wandb_run is not None:
            self._wandb_run.define_metric("progress/env_steps")
            self._wandb_run.define_metric("*", step_metric="progress/env_steps")

        self.last_empirical_prob = None
        self.last_outage_prob = None                  # compatibility: now means reward failure probability
        # Three checkpointed clocks make the advertised main budget and physical interaction cost
        # simultaneously explicit. total_env_steps remains the historical name for actual.
        self.main_env_steps = 0
        self.initialization_env_steps = 0
        self.actual_env_steps = 0
        self.total_env_steps = 0
        self.initialization_cursor = 0
        self.start_iteration = 0
        self.iteration_callback = None
        # The runner installs a save-only callback. It never validates, selects, or logs to W&B.
        self.initialization_callback = None
        self._progress_bar = None
        self._last_progress_metrics = {}
        # Validation/checkpoint text is queued by the callback, then flushed only after this
        # batch's W&B upload, status publication, progress increment, and [train] line complete.
        self._pending_console_messages = []
        # Multi-run launchers set one file per child process. Atomic JSON snapshots let a
        # parent render one aggregate bar without mixing four independent tqdm streams.
        progress_file = getattr(args, "progress_file", None)
        self.progress_file = None if not progress_file else os.path.abspath(str(progress_file))

    def _require_return_threshold(self):
        """Fail fast for chance/quantile constrained algorithms; QPO intentionally skips this."""
        if self.return_threshold is None:
            raise ValueError(
                f"{self.algo_name} requires explicit return_threshold=q; "
                "run QPO first and let the user choose q")
        return self.return_threshold

    # ============================================================ Gaussian helpers ============================================================
    def _logp_from_params(self, actions, means, log_std):
        """Diagonal Gaussian raw-action log likelihood, reduced over action dimension."""
        variance = torch.exp(2.0 * log_std)
        elementwise = -0.5 * (((actions - means).square() / variance)
                              + 2.0 * log_std + self._log_2pi)
        return elementwise.sum(dim=-1)

    def _sample_actions(self, states, rnn_state=None, return_state=False, deterministic=False):
        """Sample raw actions for [B,D], optionally returning the updated recurrent state."""
        means, log_std, next_state = self.actor.step(states, rnn_state)
        actions = means if deterministic else means + torch.exp(log_std) * torch.randn_like(means)
        if return_state:
            return actions, next_state, means, log_std
        return actions

    def _executed_actions(self, raw_actions):
        """
        Map unbounded Gaussian actions to the Box actions executed by Safety Gym.

        Input/output retain the same ``[..., action_dim]`` shape.  ``torch.maximum/minimum``
        broadcasts the per-dimension bounds over time and batch axes.  Lightweight legacy test
        fixtures that bypass ``__init__`` have no bounds; their historical identity behavior is
        retained, while every production agent always owns the real environment bounds.
        """
        action_low = getattr(self, "_action_low", None)
        action_high = getattr(self, "_action_high", None)
        if action_low is None or action_high is None:
            return raw_actions
        return torch.maximum(torch.minimum(raw_actions, action_high), action_low)

    def _compute_log_probs(self, states, actions, initial_state=None):
        """Recompute log pi for a full [T,B] sequence under fixed rollout RMS statistics."""
        if states.ndim == 2:
            means, log_std, _ = self.actor.step(states, initial_state)
        else:
            means, log_std, _ = self.actor(states, initial_state)
        return self._logp_from_params(actions, means, log_std)

    def _entropy(self, log_std):
        """Diagonal Gaussian entropy with the same leading dimensions as log_std."""
        return (log_std + 0.5 * (1.0 + self._log_2pi)).sum(dim=-1)

    @staticmethod
    def ppo_combined_loss(new_logp, old_logp, combined_advantage, valid, clip_ratio):
        """Negative PPO clipped objective for one already-combined Lagrangian advantage."""
        ratio = torch.exp(new_logp - old_logp)
        clipped = torch.clamp(ratio, 1.0 - float(clip_ratio), 1.0 + float(clip_ratio))
        surrogate = torch.minimum(ratio * combined_advantage, clipped * combined_advantage)
        mask = valid.to(surrogate.dtype)
        loss = -(surrogate * mask).sum() / torch.clamp(mask.sum(), min=1.0)
        clipped_samples = (ratio - 1.0).abs() > float(clip_ratio)
        clip_fraction = (
            clipped_samples.to(mask.dtype) * mask
        ).sum() / torch.clamp(mask.sum(), min=1.0)
        return loss, ratio, clip_fraction

    # ============================================================ boundary-correct collector ============================================================
    def _rollout_core(self, keep_logp=True, phase="main"):
        """
        Collect at most one episode per vector slot.

        Early-ended slots may be auto-reset by the adapter, but ``valid=False`` thereafter, so
        reset episodes are never concatenated to the old trajectory. At truncation, TD next_state
        is final_observation and bootstrap_mask=1; at termination bootstrap_mask=0.
        """
        phase = str(phase)
        if phase not in {"main", "initialization"}:
            raise ValueError("rollout phase must be 'main' or 'initialization'")

        B = self.num_envs
        state = self.vec_env.reset()
        recurrent = self.actor.initial_state(B, self.device)
        active = torch.ones(B, dtype=torch.bool, device=self.device)

        fields = {name: [] for name in (
            "S", "A_raw", "A_exec", "R", "C", "next_states",
            "final_observation", "final_observation_mask",
            "terminated", "truncated", "valid", "logp", "mean_old", "log_std_old",
            "h0", "c0", "next_h", "next_c")}
        discounted_return = torch.zeros(B, dtype=torch.float32, device=self.device)
        diagnostic_discounted_cost = torch.zeros_like(discounted_return)
        diagnostic_raw_cost = torch.zeros_like(discounted_return)

        for t in range(self.n):
            with torch.no_grad():
                fields["h0"].append(recurrent[0][0].clone())
                fields["c0"].append(recurrent[1][0].clone())
                raw_action, next_recurrent, mean, log_std = self._sample_actions(
                    state, recurrent, return_state=True)
                # The Gaussian score is evaluated at the unmodified random variable. Hard Box
                # clipping creates boundary point masses, so A_exec cannot be substituted here.
                old_logp = self._logp_from_params(raw_action, mean, log_std)
            step = self.vec_env.step(raw_action)
            valid = active.clone()
            terminated = step.terminated & valid
            truncated = step.truncated & valid
            episode_end = terminated | truncated

            # Production vector adapters return the action reported by SafetyEnv after clipping.
            # The local mapping is a compatibility fallback for old/custom adapters and is exactly
            # the same Box projection, so collector and environment semantics cannot silently drift.
            executed_action = getattr(step, "executed_action", None)
            if executed_action is None:
                executed_action = self._executed_actions(raw_action)

            # For non-boundary rows final_observation is masked out; for boundary rows it is
            # exactly the old episode state and must replace the auto-reset observation in TD.
            next_state_td = torch.where(
                episode_end.unsqueeze(1), step.final_observation, step.observation)
            reward = step.reward * valid.to(step.reward.dtype)
            cost = step.cost * valid.to(step.cost.dtype)

            fields["S"].append(state)
            fields["A_raw"].append(raw_action)
            fields["A_exec"].append(executed_action)
            fields["R"].append(reward)
            fields["C"].append(cost)
            fields["next_states"].append(next_state_td)
            # Preserve the old episode's boundary observation separately from the adapter's reset
            # observation. DQC uses this exact tensor for a truncated one-step target action.
            fields["final_observation"].append(step.final_observation)
            fields["final_observation_mask"].append(
                step.final_observation_mask & episode_end)
            fields["terminated"].append(terminated)
            fields["truncated"].append(truncated)
            fields["valid"].append(valid)
            fields["logp"].append(old_logp)
            fields["mean_old"].append(mean)
            fields["log_std_old"].append(log_std)
            # next_recurrent is the actor context after consuming s_t and before any ended slot is
            # zeroed. At truncation this is the old episode history needed for the target action.
            fields["next_h"].append(next_recurrent[0][0].clone())
            fields["next_c"].append(next_recurrent[1][0].clone())

            discount = self.gamma ** t
            discounted_return += discount * reward
            diagnostic_discounted_cost += discount * cost
            diagnostic_raw_cost += cost
            valid_steps = int(valid.sum().item())
            if phase == "initialization":
                self.initialization_env_steps += valid_steps
            else:
                self.main_env_steps += valid_steps
            # actual is always recomputed from its authoritative components across every save.
            self.actual_env_steps = (
                int(self.main_env_steps) + int(self.initialization_env_steps))
            self.total_env_steps = int(self.actual_env_steps)

            # Ended/inactive slots are forced to zero recurrent state and remain inactive.
            active = active & ~episode_end
            keep = active.to(next_recurrent[0].dtype).view(1, B, 1)
            recurrent = next_recurrent[0] * keep, next_recurrent[1] * keep
            state = step.observation
            if not bool(active.any()):
                break

        rollout = {key: torch.stack(value, dim=0) for key, value in fields.items()}
        # ``A`` remains the cross-algorithm compatibility raw-action alias. DQC uses the explicit
        # A_raw for log-prob and A_exec for every critic query.
        rollout["A"] = rollout["A_raw"]
        rollout.update({
            "disc_return": discounted_return,
            "diagnostic_disc_cost": diagnostic_discounted_cost,
            "diagnostic_raw_cost": diagnostic_raw_cost,
            # terminated alone blocks critic bootstrap; truncated deliberately preserves it.
            "bootstrap_mask": rollout["valid"].float() * (~rollout["terminated"]).float(),
        })
        # RMS is intentionally not updated here: old log-probs and PPO recomputation must use
        # identical moments. Each algorithm updates RMS only after finishing this rollout's actor step.
        return rollout

    # ============================================================ logging ============================================================
    def _env_action_stats(self):
        """Prefix environment-side values so cost cannot be mistaken for a constraint metric."""
        stats = self.vec_env.stats() if hasattr(self.vec_env, "stats") else {}
        out = {}
        for key, value in stats.items():
            prefix = "diagnostic/environment_cost" if "cost" in key else "diagnostic/environment"
            out[f"{prefix}_{key}"] = float(value)
        return out

    def _log_core(self, iteration, returns, diagnostic_disc_cost, diagnostic_raw_cost, extra=None):
        """Log raw G_T risk metrics; Safety cost stays under diagnostic/environment_cost_* only."""
        metrics = return_statistics(returns, self.failure_alpha, self.return_threshold)
        # object.__new__ fixtures may own only the historical counter; production owns all clocks.
        main_steps = int(getattr(self, "main_env_steps", self.total_env_steps))
        initialization_steps = int(getattr(self, "initialization_env_steps", 0))
        actual_steps = int(getattr(
            self, "actual_env_steps", main_steps + initialization_steps))

        log = {
            "disc_reward/aver_reward": metrics["mean"],
            "disc_reward/return_std": metrics["reward_std"],
            "disc_reward/quantile_reward": metrics["quantile_return"],
            # Common cross-algorithm quantile axis. For QCPO/QCPO_REF/DQC this is the
            # linearly-interpolated alpha percentile of the current B-trajectory rollout.
            # QPO passes its persistent Robbins-Monro tracker in ``extra`` below, which
            # deliberately overrides this empirical fallback without branching in the base class.
            "quantile/q_est": metrics["quantile_return"],
            "progress/iteration": int(iteration),
            "progress/trajectories": int((iteration + 1) * self.num_envs),
            "progress/env_steps": main_steps,
            "progress/main_env_steps": main_steps,
            "progress/initialization_env_steps": initialization_steps,
            "progress/actual_env_steps": actual_steps,
            "diagnostic/environment_cost_discounted_mean": float(np.mean(diagnostic_disc_cost)),
            "diagnostic/environment_cost_raw_mean": float(np.mean(diagnostic_raw_cost)),
        }
        if self.return_threshold is not None:
            log.update({
                # This common comparison key always uses the same current B trajectories and the
                # inclusive fixed-threshold event 1{G<=q}. Formulation-specific aliases may remain
                # for backward compatibility, but this is the canonical cross-algorithm curve.
                "quantile/constraint_prob": metrics["empirical_prob"],
                "constraint/empirical_failure_probability": metrics["empirical_prob"],
                "constraint/residual": metrics["constraint_residual"],
                "constraint/feasible": float(metrics["feasible"]),
                "constraint/ties_at_threshold": metrics["ties_at_threshold"],
                "constraint/near_threshold_count": metrics["near_threshold_count"],
            })
            self.last_empirical_prob = metrics["empirical_prob"]
            self.last_outage_prob = metrics["empirical_prob"]
        log.update(self._env_action_stats())
        if extra:
            log.update(extra)
        # Upload is deliberately deferred to _after_iteration. The validation callback can then
        # merge sparse validation/* fields into this same W&B row, preserving exactly one log()
        # call per completed rollout batch and one monotonic iteration step.
        return log

    # ============================================================ terminal progress ============================================================
    def _after_initialization_batch(self, total_batches):
        """
        Publish save-only initialization progress after one complete atomic batch.

        The algorithm increments initialization_cursor before entering this method. The callback
        serializes state and RNG at that cursor. No core logging, validation, or selection runs.
        """
        total_batches = max(0, int(total_batches))
        completed = int(self.initialization_cursor)
        self._write_external_progress(
            completed=int(self.start_iteration), metrics={}, phase="initialization")
        if self.initialization_callback is not None:
            self.initialization_callback(self, completed, total_batches)
        self.console_write(
            f"[initialization] {self.algo_name}/{self.env_name}/s{self.seed} "
            f"batch:{completed:03d}/{total_batches:03d} "
            f"init_env_steps:{self.initialization_env_steps} "
            f"actual_env_steps:{self.actual_env_steps}")

    def training_iterations(self):
        """
        Yield rollout iterations while owning one terminal-safe tqdm progress bar.

        The unit is one completed rollout batch, not one environment transition. The ETA thus
        measures collection, optimizer updates, W&B, and periodic validation together. Resume
        starts at ``start_iteration`` while learning a fresh wall-clock speed estimate.
        """
        iterations = range(self.start_iteration, self.num_iterations)
        self._write_external_progress(
            completed=self.start_iteration, metrics={}, phase="training")
        if not self.show_progress:
            yield from iterations
            return

        # The bar advances only in _after_iteration, after the batch update and W&B log succeed;
        # a failed partial batch is therefore never counted as completed work.
        progress = tqdm(
            total=self.num_iterations,
            initial=self.start_iteration,
            desc=f"{self.algo_name}/{self.env_name}",
            unit="batch",
            dynamic_ncols=True,
            mininterval=self.progress_mininterval,
            leave=True,
        )
        self._progress_bar = progress
        try:
            yield from iterations
        finally:
            progress.close()
            self._progress_bar = None

    def _write_external_progress(self, completed, metrics, phase="training"):
        """
        Atomically publish one child-run snapshot for the launcher-level aggregate bar.

        Shape/semantics: ``completed`` counts finished main rollout batches and ``total`` is
        num_iterations. The core display metrics, plus an optional post-dual lambda, come from
        exactly the same dictionary already sent to W&B. A same-directory os.replace prevents
        the monitor from reading partial JSON.
        """
        if self.progress_file is None:
            return
        snapshot = self._progress_snapshot(metrics)
        payload = {
            "run": f"{self.algo_name}/{self.env_name}",
            "algo": self.algo_name,
            "env": self.env_name,
            "seed": self.seed,
            "phase": str(phase),
            "completed": int(completed),
            "total": int(self.num_iterations),
            "updated_at": float(time.time()),
            "metrics": {
                key: None if value is None else float(value)
                for key, value in snapshot.items()
            },
            # Scheduled validation is carried in the same atomic snapshot as its completed batch.
            # The aggregate monitor can therefore print train and validation on separate lines
            # without parsing interleaved child stdout.
            "validation": self._validation_progress_snapshot(metrics),
        }
        directory = os.path.dirname(self.progress_file) or "."
        os.makedirs(directory, exist_ok=True)
        temporary = f"{self.progress_file}.{os.getpid()}.tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.progress_file)

    @staticmethod
    def _progress_snapshot(metrics):
        """
        Select the requested terminal metrics from one completed rollout batch.

        New runs make both public keys explicit: QPO supplies persistent q_est/tracker probability,
        while other algorithms supply current-batch quantile/direct fixed-threshold probability.
        The nested fallbacks remain only so old checkpoints/log dictionaries still render safely.
        Probability- and quantile-constrained algorithms retain different formulation-specific
        W&B keys, because their multipliers have different units. Both keys are presented under
        the terminal-only label ``lambda``. Each algorithm constructs this dictionary after its
        dual update, so terminal output and W&B refer to the same latest post-batch multiplier.
        """
        mean_reward = metrics.get("disc_reward/aver_reward")
        quantile_reward = metrics.get("disc_reward/quantile_reward")
        q_est = metrics.get(
            "quantile/q_est",
            metrics.get("quantile/empirical_linear", quantile_reward),
        )
        constraint_prob = metrics.get(
            "quantile/constraint_prob",
            metrics.get(
                "constraint/batch_failure_probability",
                metrics.get("constraint/empirical_failure_probability"),
            ),
        )
        snapshot = {
            "mean_reward": mean_reward,
            "quantile_reward": quantile_reward,
            "q_est": q_est,
            "constraint_prob": constraint_prob,
        }
        # Do not add a fake lambda to unconstrained QPO. For constrained methods, select exactly
        # one formulation-specific source without creating a second, ambiguous W&B alias.
        for key in ("lambda/probability_multiplier", "lambda/quantile_multiplier"):
            if key in metrics:
                snapshot["lambda"] = metrics[key]
                break
        return snapshot

    @staticmethod
    def _validation_progress_snapshot(metrics):
        """Extract one compact exact-N validation record, or None on ordinary training batches."""
        mean_key = "validation/disc_reward/aver_reward"
        if mean_key not in metrics:
            return None
        fields = {
            "num_episodes": metrics.get("validation/num_episodes"),
            "mean_reward": metrics.get(mean_key),
            "quantile_reward": metrics.get("validation/disc_reward/quantile_reward"),
            "q_est": metrics.get("validation/quantile/q_est"),
            "constraint_prob": metrics.get("validation/quantile/constraint_prob"),
        }
        return {
            key: None if value is None else float(value)
            for key, value in fields.items()
        }

    @staticmethod
    def _format_progress_value(value):
        """Format one scalar compactly; a probability may be absent for an unconstrained run."""
        return "n/a" if value is None else f"{float(value):.3f}"

    def console_write(self, message):
        """Print without corrupting an active tqdm line and flush for tee/tail consumers."""
        if self._progress_bar is not None:
            self._progress_bar.write(str(message))
        else:
            print(str(message), flush=True)

    def _console_training_line(self, iteration, metrics):
        """Build one durable periodic line that is easy to grep from redirected logs."""
        snapshot = self._progress_snapshot(metrics)
        values = {
            key: self._format_progress_value(value)
            for key, value in snapshot.items()
        }
        line = (
            f"Iter:{iteration + 1:05d}/{self.num_iterations:05d} "
            f"env_steps:{getattr(self, 'main_env_steps', self.total_env_steps)} "
            f"mean_reward:{values['mean_reward']} "
            f"quantile_reward:{values['quantile_reward']} "
            f"q_est:{values['q_est']} "
            f"constraint_prob:{values['constraint_prob']}"
        )
        # ``lambda`` is absent for QPO. When present it is the same post-dual value uploaded in
        # this batch's sole W&B row, so reconnectable text logs and online plots share one timing.
        if "lambda" in values:
            line += f" lambda:{values['lambda']}"
        return line

    def _after_iteration(self, iteration, metrics):
        """Merge optional validation, upload one W&B row, then advance terminal progress."""
        metrics = dict(metrics)

        # Treat validation/checkpointing as part of the iteration's wall time. The callback
        # returns namespaced validation/* values only on scheduled iterations; merging them here
        # creates a sparse unbiased validation curve without a second W&B step/log call.
        if self.iteration_callback is not None:
            validation_metrics = self.iteration_callback(
                self, int(iteration), dict(metrics))
            if validation_metrics:
                metrics.update(validation_metrics)

        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=int(iteration))

        self._last_progress_metrics = dict(metrics)
        self._write_external_progress(
            completed=int(iteration) + 1, metrics=metrics, phase="training")

        if self._progress_bar is not None:
            # Keep the progress/ETA on its own physical line. Key metrics are printed below via
            # tqdm.write/console_write, so narrow tmux panes never wrap them into a piled-up bar.
            self._progress_bar.update(1)

        # Default log_interval=1 gives one durable post-update metric line per batch for all four
        # algorithms. A caller may still explicitly reduce text frequency for very long sweeps.
        completed = int(iteration) + 1
        pending = list(getattr(self, "_pending_console_messages", []))
        should_print_training = (
            int(iteration) == int(self.start_iteration)
            or completed % self.log_interval == 0
            or completed == self.num_iterations
            or bool(pending)
        )
        if should_print_training:
            # Lightweight unit fixtures may construct VecAgentBase without running __init__.
            # Production agents always own these identifiers; the fallbacks keep this logging
            # boundary testable without changing any optimizer or metric semantics.
            display_algo = getattr(self, "algo_name", self.__class__.__name__)
            display_env = getattr(self, "env_name", "unknown_env")
            display_seed = getattr(self, "seed", "?")
            self.console_write(
                f"[train] {display_algo}/{display_env}/s{display_seed} "
                + self._console_training_line(int(iteration), metrics))

        # Validation/checkpoint output follows its matching [train] line and retains intentional
        # blank lines inside the queued message. Clear first so an exception cannot duplicate it.
        self._pending_console_messages = []
        for message in pending:
            self.console_write(message)

    # ============================================================ evaluation/checkpoint compatibility ============================================================
    def reset_eval_state(self, batch_size=1):
        """Reset persistent evaluation hidden state at an episode boundary."""
        self._eval_rnn_state = self.actor.initial_state(batch_size, self.device)

    def choose_action(self, state):
        """Single-environment stochastic action with persistent recurrent state."""
        tensor = torch.as_tensor(np.asarray(state, dtype=np.float32).reshape(1, -1), device=self.device)
        if self._eval_rnn_state is None:
            self.reset_eval_state(1)
        with torch.no_grad():
            action, self._eval_rnn_state, _mean, _log_std = self._sample_actions(
                tensor, self._eval_rnn_state, return_state=True)
        return action[0].cpu().numpy().astype(np.float32)

    select_action = choose_action

    def checkpoint_state(self):
        """Base state; subclasses extend this dictionary with optimizers/critics/duals."""
        main_steps = int(getattr(self, "main_env_steps", self.total_env_steps))
        initialization_steps = int(getattr(self, "initialization_env_steps", 0))
        actual_steps = int(getattr(
            self, "actual_env_steps", main_steps + initialization_steps))
        return {
            "training_budget_semantics": self.training_budget_semantics,
            "actor": self.actor.state_dict(),
            "main_env_steps": main_steps,
            "initialization_env_steps": initialization_steps,
            "actual_env_steps": actual_steps,
            # Compatibility key now has one documented meaning: all physical valid interactions.
            "total_env_steps": actual_steps,
            "initialization_cursor": int(getattr(self, "initialization_cursor", 0)),
        }

    def validate_checkpoint_state(self, state):
        """Reject old mixed-budget checkpoints before loading actor or optimizer tensors."""
        if not isinstance(state, dict):
            raise ValueError("checkpoint agent state must be a dictionary")
        saved_semantics = state.get("training_budget_semantics")
        if saved_semantics != self.training_budget_semantics:
            raise ValueError(
                "training_budget_semantics mismatch: "
                f"checkpoint={saved_semantics!r}, requested={self.training_budget_semantics!r}")

        # Redundant actual/total fields make corrupt or hand-edited counters fail before mutation.
        main_steps = int(state.get("main_env_steps", -1))
        initialization_steps = int(state.get("initialization_env_steps", -1))
        actual_steps = int(state.get("actual_env_steps", -1))
        total_steps = int(state.get("total_env_steps", -1))
        if main_steps < 0 or initialization_steps < 0:
            raise ValueError("checkpoint is missing non-negative main/initialization step clocks")
        if actual_steps != main_steps + initialization_steps or total_steps != actual_steps:
            raise ValueError("checkpoint environment step clocks are inconsistent")

    def load_checkpoint_state(self, state):
        """Restore base actor/RMS and progress state."""
        self.validate_checkpoint_state(state)
        self.actor.load_state_dict(state["actor"])
        self.main_env_steps = int(state["main_env_steps"])
        self.initialization_env_steps = int(state["initialization_env_steps"])
        self.actual_env_steps = int(state["actual_env_steps"])
        self.total_env_steps = int(state["total_env_steps"])
        self.initialization_cursor = int(state.get("initialization_cursor", 0))
