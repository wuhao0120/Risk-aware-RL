"""Final SimpleButton DCA: undiscounted objectives, lambda normalization and fresh-batch PI.

The inherited classes supply the tested recurrent collector, independent
optimizers, raw-history replay and transactional E8 PPO. This module replaces
the scientific differences explicitly; historical constructor fields are
provenance, never a claim that Scheme B uses the preceding experiment's labels.
"""
from __future__ import annotations

import copy
import math
from numbers import Integral, Real

import numpy as np
import torch
from objective import COST_GAMMA, REWARD_GAMMA, COST, EVENT, powers
from torch import nn

from actor_transaction import ActorTransactionMixin
from replay_agent import Agent as ReplayAgent, REPLAY_KEYS
from agents.dqc_ac_independent_factorial import DQCACIndependentFactorialGPU
from diagnostics import score_gradient_diagnostics
from schemeb_math import pairwise_pinball_loss, upper_cost_violation_probability, combine_actor_advantages

IDENTITY = 'dynamic_all1_b25_p20_dca_pilot_v1'
ACTOR_NORMALIZATION = 'common_valid_population_std_floor0p01_no_center_no_outer_lambda_v1'
LAMBDA_NORMALIZATION = 'one_plus_lambda_no_std_v1'
PI_KI, PI_EMA, PI_KP, PI_CAP = .2, .8, 5., 50.
ARM_CONFIGS = {
    'schemeb_raw_mean100_lambda_norm': dict(mean_divisor=100.,
        actor_normalization=LAMBDA_NORMALIZATION, pi_ki=PI_KI, pi_ema=PI_EMA, pi_kp=PI_KP),
}
for divisor in (25, 50, 200):
    ARM_CONFIGS[f'schemeb_raw_mean{divisor}_lambda_norm'] = dict(
        ARM_CONFIGS['schemeb_raw_mean100_lambda_norm'], mean_divisor=float(divisor))
ARM_DIVISORS = {arm: config['mean_divisor'] for arm, config in ARM_CONFIGS.items()}
ARMS = tuple(ARM_CONFIGS)
WARMUP_STAGES = (34, 33, 33)
MAIN_STAGES = (134, 133, 133)
ACTOR_COUNTERS = ('actor_candidate_attempts_total', 'actor_candidate_rejections_total',
                  'actor_candidate_nonfinite_rejections_total', 'actor_candidate_optimizer_steps_total')


def pi_step(multiplier, ema, error, *, ki=PI_KI, ema_decay=PI_EMA, kp=PI_KP, cap=PI_CAP):
    """One incremental PI update; project after summing I and smoothed P."""
    updated_ema = ema_decay * float(ema) + (1 - ema_decay) * float(error)
    integral = ki * float(error)
    proportional = kp * (updated_ema - float(ema))
    candidate = float(multiplier) + integral + proportional
    return min(cap, max(0., candidate)), updated_ema, integral, proportional, candidate



def actor_labels_for_arm(reward_credit, risk_advantage, multiplier, valid_mask, arm):
    """Freeze both raw branches and exactly one per-rollout denominator.

    The lambda arm replaces the common population SD, rather than stacking a
    second normalization onto it. Keep raw branches for inherited split PPO,
    which applies this one frozen denominator to each branch exactly once.
    """
    config = ARM_CONFIGS[arm]
    labels = combine_actor_advantages(reward_credit, risk_advantage, multiplier,
        mean_divisor=config['mean_divisor'], valid_mask=valid_mask, std_floor=.01)
    if config['actor_normalization'] == LAMBDA_NORMALIZATION:
        with torch.no_grad():
            lam = torch.as_tensor(multiplier, device=reward_credit.device,
                dtype=reward_credit.dtype).detach().reshape(())
            if not bool(torch.isfinite(lam)) or bool(lam < 0):
                raise ValueError('lambda normalization requires a finite nonnegative multiplier')
            denominator = (1. + lam).clone().detach()
            labels.update(combined_normalization_scale=denominator,
                mean_term=labels['raw_mean_term']/denominator,
                risk_term=labels['raw_risk_term']/denominator,
                combined=labels['raw_combined']/denominator)
    return labels

def complete_reward_suffix(rollout, horizon, gamma=1.):
    """Return detached, undiscounted MC suffixes of complete fresh episodes.

    A collector cut is not a terminal event. Only a true termination or the
    declared finite task boundary can end a valid episode. Accumulation uses
    float64, then labels adopt reward-network float32 units.
    """
    valid = rollout['valid'].bool()
    terminal = rollout['terminated'].bool()
    truncated = rollout['truncated'].bool()
    boundary = terminal | truncated
    if valid.ndim != 2 or rollout['R'].shape != valid.shape:
        raise ValueError('complete MC expects [T,B] rewards and validity')
    lengths = valid.sum(0)
    if bool((lengths == 0).any()):
        raise ValueError('complete MC received an empty episode')
    time = torch.arange(valid.shape[0], device=valid.device)[:, None]
    if not torch.equal(valid, time < lengths[None, :]):
        raise ValueError('complete MC requires contiguous episode validity')
    expected_boundary = time == (lengths - 1)[None, :]
    if not torch.equal(boundary & valid, expected_boundary):
        raise ValueError('complete MC requires a single actual episode boundary')
    if bool((truncated & ~terminal & valid & (time != int(horizon)-1)).any()):
        raise ValueError('arbitrary collection truncation is not a finite task ending')
    reward = torch.where(valid, rollout['R'], 0).double()
    weights = powers(reward, gamma)
    suffix = torch.flip(torch.cumsum(torch.flip(reward * weights, [0]), 0), [0]) / weights
    return torch.where(valid, suffix.to(rollout['R'].dtype), 0).detach()


class Agent(ActorTransactionMixin, ReplayAgent):
    """Resumable Scheme B with a separately declared mean divisor."""

    def __init__(self, args, env):
        self.stability_arm = getattr(args, 'stability_arm', None)
        self.mean_divisor = float(getattr(args, 'mean_divisor', float('nan')))
        if self.stability_arm not in ARMS or self.mean_divisor != ARM_DIVISORS[self.stability_arm]:
            raise ValueError('Scheme B requires a matching arm and explicit mean_divisor')
        if float(args.reward_scale) != 100.:
            raise ValueError('legacy construction unit must remain100; use mean_divisor for the actor')
        if float(args.training_gamma) != REWARD_GAMMA or float(args.report_gamma) != .99:
            raise ValueError('Training gamma must be1; report gamma .99 is logging only')
        config = ARM_CONFIGS[self.stability_arm]
        self.actor_normalization = config['actor_normalization']
        self.actor_std_normalization = self.actor_normalization == ACTOR_NORMALIZATION
        self.pi_ki, self.pi_ema, self.pi_kp = config['pi_ki'], config['pi_ema'], config['pi_kp']
        for name in ('actor_normalization', 'pi_ki', 'pi_ema', 'pi_kp'):
            if getattr(args, 'actual_'+name, None) != getattr(self, name):
                raise ValueError('Scheme B arm/config mismatch: '+name)
        self.dual_controller = 'pi'
        self.risk_advantage_mode = 'transition_independent_next_samples'
        self.dual_lr_multiplier = 1.  # PI gains are not multiplied by historical dual x10.
        self.pi_error_ema = 0.
        self.pi_update_count = 0
        self.pi_last_fresh_probability = 0.
        self._pi_metrics = {}
        self.schemeb_target_cache_generations = 0
        self._critic_transaction_active = False
        for name in ACTOR_COUNTERS:
            setattr(self, name, 0)
        super().__init__(args, env)
        self.dqc_reward_scale = self.mean_divisor
        self.dqc_reward_outer_discount = 'none'
        if self.num_action_samples != 4 or self.lambda_max != PI_CAP or self.independent_factorial_arm != 'C1':
            raise ValueError('Scheme B requires four action samples, cap50, and C1 split PPO with no extra outer division')
        if self.gamma != REWARD_GAMMA or self.num_quantiles != 128:
            raise ValueError('Scheme B requires undiscounted cost/reward gamma1 and128 quantiles')

    def _checkpoint_identity(self):
        identity = super()._checkpoint_identity()
        identity.update(
            experiment_identity=IDENTITY, raw_cost_identity=IDENTITY,
            stability_arm=self.stability_arm, actual_mean_divisor=self.mean_divisor,
            raw_cost_reward_scale=self.mean_divisor,
            raw_cost_actor='raw_reward_MC_minus_Vold_div_c_minus_lambda_raw_cost_transition_probability',
            actual_mean_estimator='complete_training_gamma_MC_suffix_minus_prevalue_fit_V',
            actual_gae_lambda=None, actual_value_target='complete_fresh_undiscounted_reward_MC',
            actual_critic_loss='all_pairs_pure_pinball_mean_NK_then_rows_fixed_midpoint_heads',
            actual_cdf_reader='sort_midpoint_linear_interpolation_right_ties_support0_1_double_budget',
            actual_target_update_clock='three_fixed_target_stages_hard_copy_after_each_stage',
            actual_target_stage_warmup_steps=list(WARMUP_STAGES),
            actual_target_stage_main_steps=list(MAIN_STAGES),
            replay_target_action='one_current_policy_draw_per_available_transition_cached_per_stage',
            actual_risk_advantage=self.risk_advantage_mode,
            actual_transition_value_protocol='independent_current4_and_actual_next4_training_budget_terminal_strict_gt',
            actual_actor_normalization=self.actor_normalization,
            actual_actor_std_normalization=self.actor_std_normalization, actual_actor_outer_lambda_division=False,
            actual_actor_one_plus_lambda_normalization=not self.actor_std_normalization,
            actual_actor_normalization_std_floor=.01 if self.actor_std_normalization else None,
            actual_actor_update_clock='fresh_B25_E8_cumulative_behavior_KL_transaction_v1',
            actual_dual_controller='pi', actual_pi_ki=self.pi_ki, actual_pi_ema=self.pi_ema,
            actual_pi_kp=self.pi_kp, actual_dual_cap=PI_CAP,
            actual_dual_measurement='fresh25_completed_undiscounted_cost_strict_gt15',
            actual_dual_warmup_enabled=True, actual_expected_dual_updates=400,
            actual_gamma_reward=self.gamma, actual_gamma_cost=COST_GAMMA, actual_dual_gamma=1.,
            actual_actor_reward_outer_discount='none',
            raw_cost_primal=COST, raw_cost_dual="undiscounted_environment_cost",
            raw_cost_critic='training_gamma_cost_suffix_128_atoms_td1_current_policy_endpoint',
            actual_actor_risk_event=EVENT,
            actual_terminal_actor_risk_event=EVENT,
            mixed_primal_dual_events=False,
            actual_dual_schedule='pi_once_every_rollout_including_warmup_fresh25_cap50',
            dqc_b25_dual_window_source='latest100_completed_cost_logging_only_not_PI',
            dqc_b25_dual_warmup_protocol='fresh25_PI_update_each_warmup_rollout',
            dqc_b25_dual_update_protocol='fresh25_PI_after_actor_or_warmup_critics',
            dqc_b25_dual_failure_rule='strict_training_gamma_cost_gt15_float64',
            dqc_b25_expected_dual_updates=400,
        )
        return identity

    @staticmethod
    def probability_from_atoms(atoms, budget):
        """Compare double-precision undiscounted remaining-cost budgets before returning network dtype."""
        return upper_cost_violation_probability(atoms.double(), budget).to(atoms.dtype)

    def _critic_cdf(self, atoms, rollout):
        # The inherited name means upper cost probability in this experiment.
        prefix = rollout['reward_prefix'].double()
        return self.probability_from_atoms(atoms, (self.return_threshold-prefix)/powers(prefix,COST_GAMMA))

    def _empirical_failure_probability(self, values):
        array = np.asarray(tuple(values), dtype=np.float64)
        return float(np.mean(array > self.return_threshold)) if array.size else 0.

    def dual_step_size(self):
        return self.pi_ki

    @torch.no_grad()
    def update_dual(self, returns):
        """Use fresh25 only; the separate latest100 deque is reporting state."""
        values = self._validated_completed_returns(returns)
        probability = self._empirical_failure_probability(values)
        self._append_completed_returns(values)  # latest100 is never fed into PI.
        error = probability-self.failure_alpha
        before = self.pi_error_ema
        value, ema, integral, proportional, unclipped = pi_step(float(self.lambda_probability), before, error,
            ki=self.pi_ki, ema_decay=self.pi_ema, kp=self.pi_kp, cap=PI_CAP)
        self.lambda_probability.fill_(value)
        self.pi_error_ema = ema
        self.pi_last_fresh_probability = probability
        self.pi_update_count += 1
        self.dual_update_count += 1
        self._pi_metrics = {'diagnostic/pi/'+key:float(item) for key,item in dict(
            fresh_probability=probability, fresh_episodes=25, error=error,
            ema_before=before, ema_after=ema, integral_increment=integral,
            proportional_increment=proportional, unclipped_lambda=unclipped,
            projection_delta=value-unclipped, update_count=self.pi_update_count,
            rolling100_used_for_update=0., warmup_enabled=1.,
            ki=self.pi_ki, kp=self.pi_kp, ema_decay=self.pi_ema).items()}
        return probability, error, self.pi_ki

    @torch.no_grad()
    def prepare_gae_batch(self, rollout):
        """Keep the parent fitting ABI, but construct MC−Vold, never GAE."""
        targets = complete_reward_suffix(rollout, self.horizon, self.gamma)
        features = self._critic_rollout_features(rollout).detach()
        values = self.value_critic(features).detach()
        valid = rollout['valid'].bool()
        continuation = valid & ~(rollout['terminated'] | rollout['truncated']).bool()
        return dict(features=features, values=values,
                    advantages=torch.where(valid, targets-values, 0).detach(),
                    targets=targets, continuation=continuation, valid=valid)

    @torch.no_grad()
    def _stage_target_cache(self, batch):
        """One detached target/action sample per available row and fixed stage."""
        size = len(batch['partial'])
        targets = torch.empty(size, self.num_quantiles, device=self.device, dtype=batch['partial'].dtype)
        for start in range(0, size, 4096):
            indices = torch.arange(start, min(start+4096, size), device=self.device)
            targets[start:start+len(indices)] = self.td_targets_for_rows(batch, indices)
        self.schemeb_target_cache_generations += 1
        return targets.detach()

    def update_critic(self, rollout, permutation=None, regression_epochs=1, target_update_mode='hard'):
        """Fit100/400 replacement minibatches in three fixed Bellman stages.

        Reencode old trajectories under the one current frozen actor/RMS.
        Refresh the whole replay target cache only after the preceding stage's
        hard copy; repeated rows within a stage see exactly the same labels.
        """
        if permutation is not None or target_update_mode != 'hard':
            raise ValueError('Scheme B requires replacement replay and hard targets')
        complete_reward_suffix(rollout, self.horizon, self.gamma)  # Reject incomplete episode cuts before replay writes.
        self._critic_transaction_active = True
        start_copy = self.target_update_count
        try:
            fresh_count = self._append_replay(rollout)
            batch = self.prepare_replay_rows(fresh_count)
            count = len(batch['partial'])
            size = int(self.critic_minibatch_size)
            stages = WARMUP_STAGES if self._training_initialization else MAIN_STAGES
            fresh_ids = batch['fresh_indices']; n_fresh = size//2
            seen = torch.zeros(count, dtype=torch.bool, device=self.device)
            loss_sum = 0.; step = 0
            for stage, stage_steps in enumerate(stages):
                targets = self._stage_target_cache(batch)
                hook = getattr(self, '_schemeb_stage_hook', None)
                if hook is not None:
                    hook(stage, targets, batch)
                for _ in range(stage_steps):
                    fresh = fresh_ids[torch.randint(len(fresh_ids), (n_fresh,), device=self.device, generator=self.replay_generator)]
                    recent = torch.randint(count, (size-n_fresh,), device=self.device, generator=self.replay_generator)
                    indices = torch.cat((fresh, recent))
                    predicted = self.reward_critic(batch['states'][indices], batch['executed_actions'][indices])
                    loss = pairwise_pinball_loss(predicted, targets[indices])
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError('nonfinite Scheme B pinball loss')
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.reward_critic.parameters(), self.critic_grad_clip, error_if_nonfinite=True)
                    self.critic_optimizer.step()
                    self.critic_optimizer_steps += 1
                    loss_sum += float(loss.detach()); seen[indices] = True
                    check = getattr(self, '_schemeb_minibatch_hook', None)
                    if check is not None:
                        check(stage, step, targets, indices)
                    step += 1
                self.hard_update_target()
            if self.target_update_count-start_copy != 3:
                raise RuntimeError('Scheme B must perform exactly three target copies per rollout')
            result = self._post_critic_diagnostics(rollout)
            result.update({
                'critic/reward_quantile_loss':loss_sum/step,
                'critic/unweighted_quantile_loss':loss_sum/step,
                'diagnostic/critic/pure_pinball':1.,
                'diagnostic/dqc/critic_optimizer_steps':float(self.critic_optimizer_steps),
                'diagnostic/dqc/critic_regression_epochs':float(step*size/len(fresh_ids)),
                'diagnostic/dqc/target_update_hard':1.,
                'diagnostic/dqc/transition_coverage':float(seen.float().mean()),
                'diagnostic/critic/quantiles':128.,
                'diagnostic/critic/minibatches_this_rollout':step,
                'diagnostic/target/minibatch_recomputations':0.,
                'diagnostic/target/cache_generations_this_rollout':3.,
                'diagnostic/target/cache_generations_total':self.schemeb_target_cache_generations,
                'diagnostic/target/copies_this_rollout':3.,
                'diagnostic/target/update_count':self.target_update_count,
                'diagnostic/replay/episodes':len(self.episode_replay),
                'diagnostic/replay/transitions':count,
                'diagnostic/replay/fresh_fraction':n_fresh/size,
                'diagnostic/replay/max_episode_age':float(batch['ages'].max()),
                'diagnostic/replay/importance_sampling_active':0.,
                'diagnostic/replay/is_mean':1., 'diagnostic/replay/is_max':1.,
                'diagnostic/replay/is_p99':1., 'diagnostic/replay/is_ess_fraction':1.,
                'diagnostic/replay/is_clipped_fraction':0.,
            })
            return result
        finally:
            self._critic_transaction_active = False

    @torch.no_grad()
    def _policy_probability(self, features, means, log_stds, budget):
        samples = []
        for _ in range(self.num_action_samples):
            raw = means + log_stds.exp()*torch.randn_like(means)
            atoms = self.reward_critic(features, self._executed_actions(raw))
            samples.append(self.probability_from_atoms(atoms, budget))
        return torch.stack(samples).mean(0).detach()

    @torch.no_grad()
    def actor_advantages(self, rollout, gae_batch=None):
        """Independent current/actual-next action samples; labels frozen through E8."""
        if gae_batch is None:
            raise ValueError('Scheme B actor requires its pre-value-fit MC label snapshot')
        valid = rollout['valid'].bool()
        boundary = (rollout['terminated'] | rollout['truncated']).bool()
        continuing = valid & ~boundary
        states = rollout['S']; T, B = states.shape[:2]
        if bool(continuing[-1].any()) or bool((continuing[:-1] & ~valid[1:]).any()):
            raise ValueError('actual transition risk needs a same-episode successor')
        for next_key, key in (('next_states','S'), ('next_h','h0'), ('next_c','c0')):
            if not torch.equal(rollout[next_key][:-1][continuing[:-1]], rollout[key][1:][continuing[:-1]]):
                raise ValueError('actual transition context mismatch: '+next_key)
        prefix = rollout['reward_prefix'].double()
        weights = powers(prefix,COST_GAMMA)
        next_prefix = prefix + weights * rollout['C'].double()
        if not torch.allclose(next_prefix[:-1][continuing[:-1]], prefix[1:][continuing[:-1]], rtol=0., atol=1e-12):
            raise ValueError('actual transition cost prefix mismatch')
        features = self._critic_rollout_features(rollout)
        means, log_stds, _ = self.actor(states)
        current = self._policy_probability(features, means, log_stds, (self.return_threshold-prefix)/weights)
        successor = (next_prefix > self.return_threshold).to(current.dtype)
        if bool(continuing.any()):
            # Independent draws at actual next observations and pre-next h/c;
            # do not reuse the adjacent row's Monte Carlo action average.
            time = torch.arange(T, device=self.device, dtype=states.dtype)[:,None].expand(T,B)+1
            nh, nc = rollout['next_h'][continuing], rollout['next_c'][continuing]
            ns = rollout['next_states'][continuing]
            next_features = self._critic_features(ns, time[continuing], nh, nc)
            next_mean, next_log_std, _ = self.actor.step(ns, (nh.unsqueeze(0), nc.unsqueeze(0)))
            successor[continuing] = self._policy_probability(next_features, next_mean, next_log_std,
                ((self.return_threshold-next_prefix)/(weights*COST_GAMMA))[continuing])
        advantage = torch.where(valid, successor-current, 0).detach()
        labels = actor_labels_for_arm(gae_batch['advantages'], advantage, self.lambda_probability,
            valid, self.stability_arm)
        # The inherited split PPO primitive divides these RAW branches by the
        # common denominator once. Supplying normalized terms would divide twice.
        terms = dict(combined=labels['combined'], combined_raw=labels['raw_combined'],
            combined_normalization_scale=labels['combined_normalization_scale'],
            mean_term=labels['raw_mean_term'], risk_term=labels['raw_risk_term'],
            mean_advantage=gae_batch['advantages'].detach(), value_mean=gae_batch['values'].detach(),
            cdf_advantage=advantage, cdf_action=successor.detach(), value_cdf=current.detach(),
            transition_successor_probability=successor.detach())
        self._independent_factorial_actor_terms = terms
        return terms

    def _update_rollout(self, iteration, rollout, initialization=False):
        """Freeze MC labels, fit value/cost, update actor, then update fresh PI."""
        rollout = self._attach_reward_budgets(rollout)
        self._training_initialization = bool(initialization)
        before_count = self.dual_update_count
        labels = self.prepare_gae_batch(rollout)
        epochs = self.warmup_critic_epochs if initialization else self.dqc_value_epochs
        info = self.update_value_critic(labels, epochs)
        info.update(self.update_critic(rollout, regression_epochs=epochs, target_update_mode='hard'))
        info.update(self.action_clip_diagnostics(rollout))
        info.update({'diagnostic/dqc/fitted_warmup_active':float(initialization),
                     'diagnostic/dqc/mean_estimator_gae_config':0.,
                     'diagnostic/schemeb/mean_estimator_MC':1.,
                     'diagnostic/schemeb/mean_divisor':self.mean_divisor,
                     'diagnostic/actor_normalization/std_floor':.01 if self.actor_std_normalization else 0.,
                     'diagnostic/actor_normalization/common_std':float(self.actor_std_normalization),
                     'diagnostic/actor_normalization/one_plus_lambda':float(not self.actor_std_normalization)})
        if not initialization:
            self._set_actor_lr_batch_index(int(iteration))
            terms = self.actor_advantages(rollout, labels)
            info.update(self._advantage_diagnostics(rollout, terms))
            valid = rollout['valid'].bool()
            info.update({'diagnostic/actor_normalization/denominator':float(terms['combined_normalization_scale']),
                'diagnostic/risk_signal/state_probability_mean':float(terms['value_cdf'][valid].mean()),
                'diagnostic/risk_signal/state_probability_zero_fraction':float((terms['value_cdf'][valid]==0).float().mean()),
                'diagnostic/risk_signal/independent_next_action_samples':4.,
                'diagnostic/risk_signal/advantage_std':float(terms['cdf_advantage'][valid].std(unbiased=False))})
            if int(iteration)==0 or (int(iteration)+1)%10==0:
                info.update(score_gradient_diagnostics(self, rollout))
                info['diagnostic/fixed_reward_scale/lambda_effective_at_actor_update'] = self.mean_divisor*float(self.lambda_probability)
                info['diagnostic/fixed_reward_scale/lambda_probability_at_actor_update'] = float(self.lambda_probability)
            info.update(self.update_actor_epochs(rollout, terms['combined']))
        probability, residual, dual_lr = self.update_dual(rollout['objective_cost'])
        if self.dual_update_count != before_count+1:
            raise RuntimeError('Scheme B requires exactly one PI update per fresh rollout')
        info.update(self._pi_metrics)
        info.update({'lambda/probability_multiplier':float(self.lambda_probability),
            'lambda/residual':float(residual), 'lambda/lr':float(dual_lr),
            'lambda/dual_updates':self.dual_update_count,
            'constraint/pi_fresh_failure_probability':float(probability),
            'constraint/batch_failure_probability':float(probability),
            'diagnostic/dqc/actor_epochs_config':float(self.actor_epochs),
            'diagnostic/dqc/critic_n_step':1., 'diagnostic/dqc/target_update_hard_config':1.,
            'training/actor_lr':float(self.actor_optimizer.param_groups[0]['lr']),
            'training/critic_lr':float(self.critic_optimizer.param_groups[0]['lr']),
            'training/value_lr':float(self.value_optimizer.param_groups[0]['lr']),
            'diagnostic/dqc/combined_advantage_normalized':1.})
        if initialization:
            return info
        metrics = self._log_core(int(iteration), rollout['objective_return'].cpu().numpy(),
            rollout['report_discounted_cost'].cpu().numpy(), rollout['objective_cost'].cpu().numpy(), extra=info)
        self.actor.update_obs_rms(rollout['S'][rollout['valid']])
        self._set_actor_lr_batch_index(int(iteration)+1)
        metrics['diagnostic/stability/post_rms_behavior_kl'] = self._whole_actor_behavior_kl(
            rollout, rollout['mean_old'], rollout['log_std_old'])
        return metrics

    def _log_core(self, *args, **kwargs):
        metrics = super()._log_core(*args, **kwargs)
        metrics['diagnostic/reward_scale'] = self.mean_divisor
        metrics['constraint/pi_fresh_failure_probability'] = self.pi_last_fresh_probability
        metrics['constraint/pi_measurement_episodes'] = 25.
        return metrics

    def checkpoint_state(self):
        if self._critic_transaction_active:
            raise RuntimeError('Scheme B checkpoints are only valid at complete rollout boundaries')
        state = super().checkpoint_state()
        state.update(self._checkpoint_identity())
        state['schemeb_state'] = dict(pi_error_ema=self.pi_error_ema, pi_updates=self.pi_update_count,
            pi_last_fresh_probability=self.pi_last_fresh_probability,
            target_cache_generations=self.schemeb_target_cache_generations)
        state['stability_actor_counters'] = {key:int(getattr(self,key)) for key in ACTOR_COUNTERS}
        return state

    def _validate_dual_checkpoint_state(self, state):
        """The rolling deque is reporting-only; PI's clock includes warmup."""
        warm = int(state.get('initialization_cursor',-1))
        main_steps = int(state.get('main_env_steps',-1))
        if not 0<=warm<=20 or main_steps<0 or main_steps%25000:
            raise ValueError('invalid Scheme B rollout clock')
        main = main_steps//25000
        if main>self.num_iterations or (main and warm!=20):
            raise ValueError('invalid Scheme B main/warmup clock')
        if int(state.get('initialization_env_steps',-1))!=warm*25000:
            raise ValueError('invalid Scheme B initialization step clock')
        count=warm+main
        window=state.get('dual_return_window')
        if (state.get('dual_window_size')!=100 or not isinstance(window,(list,tuple))
                or len(window)!=min(100,count*25) or state.get('dual_window_ready')!=(count>=4)
                or state.get('dual_total_observations')!=count*25
                or int(state.get('dual_update_count',-1))!=count):
            raise ValueError('invalid Scheme B dual/reporting clock')
        if any(isinstance(x,bool) or not isinstance(x,Real) or not math.isfinite(float(x)) for x in window):
            raise ValueError('invalid Scheme B reporting cost window')
        expected=self._empirical_failure_probability(window)
        if not math.isclose(float(state.get('last_dual_prob',float('nan'))),expected,rel_tol=0.,abs_tol=1e-12):
            raise ValueError('Scheme B reporting probability mismatch')
        extra=state.get('schemeb_state',{})
        if set(extra)!={'pi_error_ema','pi_updates','pi_last_fresh_probability','target_cache_generations'}:
            raise ValueError('incomplete Scheme B controller state')
        if (type(extra['pi_updates']) is not int or extra['pi_updates']!=count
                or type(extra['target_cache_generations']) is not int or extra['target_cache_generations']!=3*count
                or not isinstance(extra['pi_error_ema'],Real) or not math.isfinite(extra['pi_error_ema'])
                or not -self.failure_alpha-1e-12<=extra['pi_error_ema']<=1-self.failure_alpha+1e-12):
            raise ValueError('invalid Scheme B PI/cache state')
        fresh=self._empirical_failure_probability(window[-25:]) if count else 0.
        if not math.isclose(float(extra['pi_last_fresh_probability']),fresh,rel_tol=0.,abs_tol=1e-12):
            raise ValueError('Scheme B fresh PI measurement mismatch')
        if not count and extra['pi_error_ema']!=0.:
            raise ValueError('Scheme B initial PI error must be zero')

    def validate_checkpoint_state(self, state):
        DQCACIndependentFactorialGPU.validate_checkpoint_state(self,state)
        self._validate_dual_checkpoint_state(state)
        warm=int(state['initialization_cursor']); main=int(state['main_env_steps'])//25000
        if int(state['critic_optimizer_steps'])!=warm*100+main*400:
            raise ValueError('Scheme B critic optimizer clock mismatch')
        if int(state['target_update_count'])!=3*(warm+main):
            raise ValueError('Scheme B target-copy clock mismatch')
        if int(state['value_optimizer_steps'])!=warm*100+main*200:
            raise ValueError('Scheme B value optimizer clock mismatch')
        applied=int(state['formal_b25_actor_epochs_applied_total']); early=int(state['actor_kl_early_stop_count'])
        if int(state['actor_optimizer_steps'])!=applied or not 0<=early<=main or not 8*(main-early)<=applied<=8*main-early:
            raise ValueError('Scheme B transactional actor clock mismatch')
        counters=state.get('stability_actor_counters',{})
        if set(counters)!=set(ACTOR_COUNTERS) or any(type(x) is not int or x<0 for x in counters.values()):
            raise ValueError('invalid Scheme B actor transaction counters')
        attempts,rejections,nonfinite,candidates=(counters[key] for key in ACTOR_COUNTERS)
        if attempts!=applied+rejections or not 0<=nonfinite<=rejections or not applied<=candidates<=attempts:
            raise ValueError('inconsistent Scheme B actor candidates')
        if not applied+8*early<=attempts<=main*64:
            raise ValueError('Scheme B actor retry budget exceeded')
        if any(not torch.equal(value,state['reward_target_critic'][key]) for key,value in state['reward_critic'].items()):
            raise ValueError('Scheme B checkpoint must follow final hard copy')
        rng=torch.Generator(device=self.device);rng.set_state(state['replay_rng_state'].cpu())
        shuffle=torch.Generator(device='cpu');shuffle.set_state(state['dqc_formal_b25_actor_shuffle_rng_state'].cpu())
        episodes=state.get('recent_episode_replay');total=int(state.get('replay_total_episodes',-1))
        if total!=(warm+main)*25 or not isinstance(episodes,list) or len(episodes)!=min(total,100):
            raise ValueError('Scheme B replay episode count mismatch')
        for expected_id,ep in zip(range(total-len(episodes),total),episodes):
            if ep.get('episode_id')!=expected_id or any(key not in ep for key in REPLAY_KEYS):
                raise ValueError('Scheme B replay FIFO/schema mismatch')
            n=len(ep['C'])
            if not 1<=n<=self.horizon:
                raise ValueError('Scheme B replay episode length mismatch')
            shapes={'S':(n,self.state_dim),'A_raw':(n,self.action_dim),'A_exec':(n,self.action_dim),
                'mean_old':(n,self.action_dim),'log_std_old':(n,self.action_dim),'R':(n,),'C':(n,),
                'terminated':(n,),'truncated':(n,)}
            for key,shape in shapes.items():
                if not torch.is_tensor(ep[key]) or tuple(ep[key].shape)!=shape or not bool(torch.isfinite(ep[key]).all()):
                    raise ValueError('invalid Scheme B replay tensor '+key)
            boundary=ep['terminated'].bool()|ep['truncated'].bool()
            if not bool(boundary[-1]) or bool(boundary[:-1].any()) or (bool(ep['truncated'][-1]) and not bool(ep['terminated'][-1]) and n!=self.horizon):
                raise ValueError('invalid Scheme B replay boundary')

    def load_checkpoint_state(self, state):
        super().load_checkpoint_state(state)
        extra=state['schemeb_state']
        self.pi_error_ema=float(extra['pi_error_ema']);self.pi_update_count=int(extra['pi_updates'])
        self.pi_last_fresh_probability=float(extra['pi_last_fresh_probability'])
        self.schemeb_target_cache_generations=int(extra['target_cache_generations'])
        for name,value in state['stability_actor_counters'].items():
            setattr(self,name,int(value))

    def get_training_summary(self):
        summary=super().get_training_summary()
        main=int(self.main_env_steps)//25000;warm=int(self.initialization_cursor)
        complete=warm==20 and main==380
        summary.update(experiment_identity=IDENTITY, algorithm_identity=IDENTITY,
            stability_arm=self.stability_arm, mean_divisor=self.mean_divisor,
            actual_mean_estimator='complete_training_gamma_MC_minus_prevalue_fit_V_no_outer_discount',
            actual_critic_loss='pure_pairwise_pinball', actual_cdf_reader='sorted_linear_interpolation',
            dual_controller='pi', actual_dual_measurement='fresh25',
            actual_dual_warmup_enabled=True, expected_dual_updates=400,
            pi_error_ema=self.pi_error_ema, pi_update_count=self.pi_update_count,
            expected_target_update_count=1200, target_cache_generations=self.schemeb_target_cache_generations,
            actor_normalization=self.actor_normalization,
            pi_parameters=dict(ki=self.pi_ki, kp=self.pi_kp, ema=self.pi_ema, cap=PI_CAP),
            full_dual_protocol_complete=complete and self.dual_update_count==400,
            full_actor_target_kl_protocol_complete=complete,
            full_formal_training_protocol_complete=complete and self.target_update_count==1200 and self.critic_optimizer_steps==154000)
        return summary
