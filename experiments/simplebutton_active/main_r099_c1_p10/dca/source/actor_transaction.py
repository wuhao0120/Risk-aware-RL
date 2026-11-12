"""Full-rollout PPO updates with reversible Adam proposals and cumulative KL budgets.

This mixin only replaces the epoch acceptance loop.  The inherited PPO primitive
continues to own recurrent replay, the split reward/risk surrogate, clipping,
gradients, and Adam.  No critic, multiplier, or observation-statistic update is
performed here.  A rejected proposal never advances a retained optimizer clock.
"""
from __future__ import annotations

import copy
import math
from typing import Any

import torch


class ActorTransactionMixin:
    """Accept at most eight actor steps against the original rollout behavior.

    Epoch k uses cumulative KL budget ``target_kl * k / 8``; it is not a KL
    comparison with epoch k-1.  Each epoch has at most eight candidate attempts.
    Backtracking halves every Adam parameter-group learning rate, keeps accepted
    reductions across this rollout, and restores *both* parameters and all Adam
    state after rejection.  The caller's existing absolute rollout LR schedule
    resets the base LR for the next rollout; no hidden persistent LR multiplier
    is introduced and ordinary rollout-boundary checkpoints remain sufficient.
    """

    actor_transaction_max_attempts = 8
    actor_transaction_lr_factor = 0.5

    def _actor_transaction_epoch(
        self, rollout, combined, behavior_mean, behavior_log_std, budget: float,
    ) -> tuple[dict[str, float] | None, dict[str, Any]]:
        """Propose one full-batch update, restoring exact state on every rejection.

        The snapshot includes optimizer parameter groups and step/moment tensors,
        rather than only model weights.  Existing Adam moments therefore cannot
        leak from a rejected, potentially non-finite candidate into its retry.
        Programmer errors are restored and re-raised; only numerical failures
        identified by the original primitive are eligible for bounded retries.
        """
        actor_state = copy.deepcopy(self.actor.state_dict())
        optimizer_state = copy.deepcopy(self.actor_optimizer.state_dict())
        optimizer_steps = int(self.actor_optimizer_steps)
        entry_lrs = [float(group['lr']) for group in self.actor_optimizer.param_groups]
        details = dict(attempts=0, rejections=0, nonfinite_rejections=0,
                       optimizer_proposals=0, candidate_kls=[], accepted_lr=0.0)

        def restore() -> None:
            self.actor.load_state_dict(actor_state, strict=True)
            # load_state_dict may retain same-device tensor storage: without
            # a new copy here, the next Adam proposal corrupts the snapshot.
            self.actor_optimizer.load_state_dict(copy.deepcopy(optimizer_state))
            self.actor_optimizer_steps = optimizer_steps
            self.actor_optimizer.zero_grad(set_to_none=True)

        for attempt in range(int(self.actor_transaction_max_attempts)):
            multiplier = self.actor_transaction_lr_factor ** attempt
            for group, entry_lr in zip(self.actor_optimizer.param_groups, entry_lrs):
                group['lr'] = entry_lr * multiplier
            details['attempts'] += 1
            try:
                info = self.update_actor_ppo_epoch(
                    rollout, combined, behavior_mean, behavior_log_std)
                details['optimizer_proposals'] += int(self.actor_optimizer_steps) - optimizer_steps
                behavior_kl = float(info['diagnostic/dqc/policy_kl_behavior_after_epoch'])
                # The primitive guards loss/gradient/policy KL; also guard the
                # retained model and optimizer state in case a finite-looking
                # policy output masks numerical corruption in other parameters.
                finite_state = all(
                    not torch.is_floating_point(value) or bool(torch.isfinite(value).all())
                    for value in self.actor.state_dict().values())
                finite_state = finite_state and all(
                    not torch.is_tensor(value) or not torch.is_floating_point(value)
                    or bool(torch.isfinite(value).all())
                    for state in self.actor_optimizer.state.values() for value in state.values())
                finite = math.isfinite(behavior_kl) and finite_state
                if math.isfinite(behavior_kl):
                    details['candidate_kls'].append(behavior_kl)
                # Summed float32 Gaussian KL can be slightly negative from
                # cancellation near identical policies; tolerate roundoff only.
                if finite and -1e-7 <= behavior_kl <= budget:
                    info['diagnostic/dqc/policy_kl_behavior_after_epoch'] = max(0.0, behavior_kl)
                    if int(self.actor_optimizer_steps) != optimizer_steps + 1:
                        raise RuntimeError('actor candidate must perform exactly one Adam step')
                    details['accepted_lr'] = float(self.actor_optimizer.param_groups[0]['lr'])
                    return info, details
                if not finite:
                    details['nonfinite_rejections'] += 1
            except Exception as error:
                details['optimizer_proposals'] += max(
                    0, int(self.actor_optimizer_steps) - optimizer_steps)
                numerical = isinstance(error, FloatingPointError) or (
                    isinstance(error, RuntimeError)
                    and ('non-finite' in str(error).lower()
                         or 'nonfinite' in str(error).lower()))
                restore()
                if not numerical:
                    raise
                details['nonfinite_rejections'] += 1
            details['rejections'] += 1
            restore()
        return None, details

    def update_actor_epochs(self, rollout, combined):
        """Run E8/M1 proposals with cumulative budgets and accepted-step metrics."""
        epochs = int(self.actor_epochs)
        if epochs != 8:
            raise RuntimeError('transactional full-batch actor requires exactly E8')
        if str(self.dqc_actor_reuse_mode) != 'ppo_clip':
            raise RuntimeError('transactional full-batch actor requires PPO clipping')
        target_kl = float(self.dqc_target_kl)
        if not math.isfinite(target_kl) or not math.isclose(target_kl, 0.01, abs_tol=1e-15):
            raise RuntimeError('transactional full-batch actor requires behavior KL 0.01')
        if int(self.actor_transaction_max_attempts) != 8 or self.actor_transaction_lr_factor != 0.5:
            raise RuntimeError('actor transaction requires eight attempts with LR halving')
        self._require_full_actor_population(rollout, combined)
        behavior_mean = rollout['mean_old'].detach().clone()
        behavior_log_std = rollout['log_std_old'].detach().clone()
        initial_kl = self._whole_actor_behavior_kl(rollout, behavior_mean, behavior_log_std)
        if not math.isfinite(initial_kl):
            raise FloatingPointError('non-finite actor behavior KL before proposals')
        entry_lr = float(self.actor_optimizer.param_groups[0]['lr'])
        starting_steps = int(self.actor_optimizer_steps)
        starting_epochs = int(self.formal_b25_actor_epochs_applied_total)
        accepted, attempts = [], []
        try:
            for epoch_index in range(epochs):
                budget = target_kl * (epoch_index + 1) / epochs
                info, detail = self._actor_transaction_epoch(
                    rollout, combined, behavior_mean, behavior_log_std, budget)
                attempts.append(detail)
                if info is None:
                    break
                accepted.append(info)
                self.formal_b25_actor_epochs_applied_total += 1
        finally:
            self._independent_factorial_actor_terms = None

        count = len(accepted)
        stopped = count < epochs
        if stopped:
            self.actor_kl_early_stop_count = int(getattr(self, 'actor_kl_early_stop_count', 0)) + 1
        if (int(self.actor_optimizer_steps) != starting_steps + count
                or int(self.formal_b25_actor_epochs_applied_total) != starting_epochs + count):
            raise RuntimeError('accepted actor optimizer/epoch counters drifted')
        kls = [float(item['diagnostic/dqc/policy_kl_behavior_after_epoch']) for item in accepted]
        step_kls = [float(item['diagnostic/dqc/policy_kl_after_step']) for item in accepted]
        losses = [float(item['actor/loss']) for item in accepted]
        norms = [float(item['diagnostic/dqc/actor_grad_norm_pre_clip']) for item in accepted]
        final_kl = kls[-1] if count else initial_kl
        # Zero accepted epochs is a legitimate outcome.  Empty-aggregate zeros
        # are accompanied by an explicit accepted mask; no rejected loss/KL is
        # presented as an accepted optimization step.
        last = dict(accepted[-1]) if count else {
            'actor/loss': 0.0,
            'diagnostic/dqc/actor_grad_norm_pre_clip': 0.0,
            'diagnostic/dqc/policy_kl_after_step': 0.0,
        }
        valid_rows = float(rollout['valid'].sum().item())
        candidate_kls = [kl for detail in attempts for kl in detail['candidate_kls']]
        cumulative_fields = {
            'actor_candidate_attempts_total': ('attempts', 'actor_backtracking_attempts_total'),
            'actor_candidate_rejections_total': ('rejections', 'actor_backtracking_rejections_total'),
            'actor_candidate_nonfinite_rejections_total': ('nonfinite_rejections', 'actor_backtracking_nonfinite_rejections_total'),
            'actor_candidate_optimizer_steps_total': ('optimizer_proposals', 'actor_optimizer_candidate_steps_total'),
        }
        for field, (detail_name, metric_name) in cumulative_fields.items():
            value = int(getattr(self, field, 0)) + sum(int(d[detail_name]) for d in attempts)
            setattr(self, field, value)
            last[f'diagnostic/dqc/{metric_name}'] = float(value)
        last.update({
            'diagnostic/dqc/actor_optimizer_steps': float(self.actor_optimizer_steps),
            'diagnostic/dqc/actor_epochs_requested': float(epochs),
            'diagnostic/dqc/actor_epochs_applied': float(count),
            'diagnostic/dqc/actor_minibatches_per_epoch': 1.0,
            'diagnostic/dqc/actor_minibatches_applied': float(count),
            'diagnostic/dqc/actor_chunks_per_minibatch': 12.0,
            'diagnostic/dqc/actor_minibatch_size_effective': 12000.0,
            'diagnostic/dqc/actor_minibatch_valid_rows_min': valid_rows,
            'diagnostic/dqc/actor_minibatch_valid_rows_max': valid_rows,
            'diagnostic/dqc/actor_epoch_chunk_coverage_min': 12.0,
            'diagnostic/dqc/actor_epoch_chunk_coverage_max': 12.0,
            'diagnostic/dqc/actor_loss_epoch_mean': sum(losses) / max(count, 1),
            'diagnostic/dqc/actor_grad_norm_epoch_mean': sum(norms) / max(count, 1),
            'diagnostic/dqc/policy_kl_step_mean': sum(step_kls) / max(count, 1),
            'diagnostic/dqc/policy_kl_step_max': max(step_kls, default=0.0),
            'diagnostic/dqc/policy_kl_behavior_after_epoch': final_kl,
            'diagnostic/dqc/policy_kl_behavior_epoch_mean': sum(kls) / max(count, 1),
            'diagnostic/dqc/policy_kl_behavior_epoch_max': max(kls, default=initial_kl),
            'diagnostic/dqc/policy_kl_after_epochs': final_kl,
            'diagnostic/dqc/actor_kl_early_stop': float(stopped),
            'diagnostic/dqc/actor_kl_early_stop_count': float(getattr(self, 'actor_kl_early_stop_count', 0)),
            'diagnostic/dqc/actor_kl_early_stop_epoch': float(count + 1 if stopped else 0),
            'diagnostic/dqc/actor_kl_early_stop_enabled': 1.0,
            'diagnostic/dqc/actor_kl_target': target_kl,
            'diagnostic/dqc/actor_epochs_applied_total': float(self.formal_b25_actor_epochs_applied_total),
            'diagnostic/dqc/actor_backtracking_attempts': float(sum(d['attempts'] for d in attempts)),
            'diagnostic/dqc/actor_backtracking_rejections': float(sum(d['rejections'] for d in attempts)),
            'diagnostic/dqc/actor_backtracking_nonfinite_rejections': float(sum(d['nonfinite_rejections'] for d in attempts)),
            'diagnostic/dqc/actor_optimizer_candidate_steps': float(sum(d['optimizer_proposals'] for d in attempts)),
            'diagnostic/dqc/actor_candidate_kl_max': max(candidate_kls, default=0.0),
            'diagnostic/dqc/actor_accepted_metric_valid': float(count > 0),
            'diagnostic/dqc/actor_base_lr_this_rollout': entry_lr,
            'diagnostic/dqc/actor_accepted_lr_last': float(attempts[count-1]['accepted_lr']) if count else 0.0,
            'diagnostic/dqc/actor_backtracking_lr_multiplier': float(self.actor_optimizer.param_groups[0]['lr']) / max(entry_lr, 1e-30),
            'diagnostic/dqc/actor_backtracking_resets_each_rollout': 1.0,
        })
        for index in range(epochs):
            epoch_kl = kls[index] if index < count else final_kl
            metric = f'policy_kl_behavior_epoch_{index + 1}'
            last[metric] = last[f'diagnostic/dqc/{metric}'] = epoch_kl
            last[f'{metric}_applied'] = last[f'diagnostic/dqc/{metric}_applied'] = float(index < count)
            last[f'diagnostic/dqc/actor_kl_budget_epoch_{index + 1}'] = target_kl * (index + 1) / epochs
            last[f'diagnostic/dqc/actor_attempts_epoch_{index + 1}'] = float(attempts[index]['attempts']) if index < len(attempts) else 0.0
        return last
