"""DCA: finite undiscounted reward/cost, recent episode replay, one-step TD.

Replay changes only the cost critic. Actor/dual retain one transaction per fresh
B25 rollout and the original E8/KL rule. The critic conditions on the recorded
current action. There are no intermediate replay actions in a one-step target,
so no importance ratio is needed. Endpoint actions are sampled from current pi.
"""
from __future__ import annotations

import copy
import numpy as np
import torch
from objective import COST_GAMMA, REWARD_GAMMA, COST, EVENT, episode_total
from torch import nn
from torch.optim import Adam

from raw_cost_agents import RawCostDCA, cost_statistics, EPSILON, LIMIT, LEVEL
from agents.common import DistributionalCritic
from agents.dqc_ac_beta_gpu import DQCACBetaGPU
from agents.dqc_ac_independent_factorial import DQCACIndependentFactorialGPU
from diagnostics import score_gradient_diagnostics

REPLAY_KEYS = ('S', 'A_raw', 'A_exec', 'R', 'C', 'mean_old', 'log_std_old',
               'terminated', 'truncated')
IDENTITY = 'undiscounted_cost_dca_d128_td1_recent100_current_policy_endpoint_v1'


def one_step_components(cost, valid, terminated, truncated, gamma=1.):
    """Return immediate costs, finite continuation masks, and unit row weights.

    The row already conditions on the executed action a_t. The successor action
    a_(t+1) is sampled from the current policy when constructing each minibatch
    target. Thus no behavior/current-policy density correction appears. Finite
    horizon truncation and termination both block bootstrap, as does a missing
    successor from a padded episode. Invalid rows carry zero immediate cost.
    """
    if float(gamma) not in (1., .99):
        raise ValueError('this experiment uses undiscounted finite cost')
    valid = valid.bool()
    boundary = (terminated | truncated).bool()
    partial = torch.where(valid,cost,torch.zeros_like(cost))
    bootstrap = torch.zeros_like(valid)
    if len(cost)>1:
        bootstrap[:-1] = valid[:-1] & ~boundary[:-1] & valid[1:]
    if not bool(torch.isfinite(partial[valid]).all()):
        raise FloatingPointError('nonfinite one-step environment cost')
    return partial,bootstrap,torch.ones_like(cost)


def weighted_quantile_huber(predicted, target, taus, row_weights, kappa):
    """QR loss [M,N,K], nominal unsorted heads; TD1 passes unit row weights."""
    error = target.detach().unsqueeze(1)-predicted.unsqueeze(2)
    absolute = error.abs()
    huber = torch.where(absolute <= kappa, .5*error.square(), kappa*(absolute-.5*kappa))
    asymmetry = (taus.view(1,-1,1)-(error.detach()<0).to(error.dtype)).abs()
    per_row = (asymmetry*huber/kappa).mean((1,2))
    return (per_row*row_weights.detach()).mean(), per_row.detach()


class Agent(RawCostDCA):
    """An isolated experiment identity; no historical source or checkpoint changes."""

    def __init__(self, args, env):
        self.reward_scale = float(getattr(args, 'reward_scale', 10.))
        self.dqc_reward_scale = self.reward_scale
        self.report_gamma = .99
        self.replay_capacity_episodes = int(getattr(args, 'replay_capacity_episodes', 100))
        self.critic_steps_per_batch = int(getattr(args, 'critic_steps_per_batch', 400))
        self.warmup_critic_steps = int(getattr(args, 'warmup_critic_steps', 100))
        if self.reward_scale not in (10., 100.):
            raise ValueError('this experiment compares mean/c=10 and 100')
        if (self.replay_capacity_episodes, self.critic_steps_per_batch, self.warmup_critic_steps) != (100,400,100):
            raise ValueError('expected capacity100, main400, warmup100')
        super().__init__(copy.copy(args), env)
        # Canonical constructors require gamma=.99, but allocate no discount
        # cache. Collector, TD, finite GAE, budgets and global CDF all read the
        # live scalar on every call. Reporting has a separate immutable .99.
        self.gamma = args.training_gamma
        self.beta = 1.
        old = self.reward_critic
        devices = [self.device.index or 0] if self.device.type == 'cuda' else []
        # Replace only the output dimension; actor and existing scalar critic stay
        # on their original seeded RNG stream and the QR backbone is copied.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed+1280003)
            critic = DistributionalCritic(self.critic_state_dim,self.action_dim,128,[512,512]).to(self.device)
            for index in (0,2):
                critic.net[index].load_state_dict(old.net[index].state_dict())
        self.num_quantiles = 128
        self.taus = (torch.arange(128,device=self.device,dtype=torch.float32)+.5)/128.
        self.reward_critic = critic
        self.reward_target_critic = copy.deepcopy(critic).eval().requires_grad_(False)
        self.critic_optimizer = Adam(critic.parameters(),lr=float(args.critic_lr),eps=1e-5)
        self.critic_n_step = 1
        self.target_update_mode = 'hard'
        self.target_tau = 1.
        self.dqc_reward_outer_discount = 'none'
        self.episode_replay = []
        self.replay_total_episodes = 0
        self.replay_generator = torch.Generator(device=self.device)
        self.replay_generator.manual_seed(self.seed+9300007)
        self._training_initialization = False

    def _checkpoint_identity(self):
        identity = super()._checkpoint_identity()
        identity.update(experiment_identity=IDENTITY,
            raw_cost_identity=IDENTITY,
            raw_cost_primal=COST,
            raw_cost_dual="undiscounted_environment_cost",
            raw_cost_violation_event=EVENT,
            raw_cost_critic='undiscounted_cost_suffix_128_atoms_td1_current_policy_endpoint',
            raw_cost_actor='undiscounted_reward_div_c_minus_lambda_upper_cost_probability',
            raw_cost_reward_scale=self.reward_scale,
            replay_capacity_episodes=self.replay_capacity_episodes,
            replay_sampling='half_fresh_half_uniform_recent_including_fresh_with_replacement',
            replay_context='raw_complete_history_current_actor_current_observation_RMS',
            replay_is='none_one_step_conditions_on_recorded_action',
            replay_behavior_parameters='saved_provenance_only_not_used_by_targets',
            replay_target_action='new_current_policy_raw_sample_clipped_at_tplus1_each_minibatch',
            critic_main_steps=self.critic_steps_per_batch,
            critic_warmup_steps=self.warmup_critic_steps,
            actual_num_quantiles=128, actual_n_step=1, actual_gamma_reward=self.gamma,
            actual_gamma_cost=COST_GAMMA, actual_actor_reward_outer_discount='none',
            report_gamma=.99, actual_gae_lambda=self.dqc_gae_lambda, mc_eta=0.,
            actual_actor_update_clock='every_fresh_12000_steps_E8_original_KL',
            actual_target_update_clock='one_hard_copy_after_each_rollout_critic_fit')
        return identity

    @torch.no_grad()
    def _capture_episode_metrics(self, rollout, reward_key):
        """Separate the gamma=1 learning objective from fixed-.99 reporting.

        ``disc_return`` / ``diagnostic_disc_cost`` are inherited collector names:
        their values now use the live training gamma=1, so they are totals.
        Explicit objective aliases prevent report discounting from reaching TD,
        GAE, actor credit, the rolling dual window, or the chance constraint.
        R/C themselves remain unchanged; all reporting sums use valid rows only.
        """
        valid = rollout['valid'].bool()
        reward = torch.where(valid,rollout[reward_key],0).double()
        cost = torch.where(valid,rollout['C'],0).double()
        discount = self.report_gamma**torch.arange(
            reward.shape[0],device=reward.device,dtype=torch.float64)
        raw_reward,raw_cost = reward.sum(0),cost.sum(0)
        report_reward = (discount[:,None]*reward).sum(0)
        report_cost = (discount[:,None]*cost).sum(0)
        target_reward, target_cost = episode_total(reward,self.gamma), episode_total(cost,COST_GAMMA)
        rollout['objective_return'] = target_reward.detach()
        rollout['objective_cost'] = raw_cost.detach()
        rollout['critic_objective_cost'] = target_cost.detach()
        rollout['report_discounted_return'] = report_reward.detach()
        rollout['report_discounted_cost'] = report_cost.detach()
        self._raw_episode_returns = raw_reward.cpu().numpy()
        self._episode_samples = {
            'discounted_return':report_reward.cpu().tolist(),
            'undiscounted_return':raw_reward.cpu().tolist(),
            'discounted_cost':report_cost.cpu().tolist(),
            'undiscounted_cost':raw_cost.cpu().tolist(),
            'objective_return':target_reward.cpu().tolist(),
            'objective_cost':raw_cost.cpu().tolist(),
            'critic_objective_cost':target_cost.cpu().tolist(),
        }
        return rollout

    @torch.no_grad()
    def _append_replay(self, rollout):
        """Retain raw complete episodes on CPU, including behavior Gaussian parameters."""
        valid = rollout['valid'].bool()
        added = []
        for b in range(valid.shape[1]):
            length = int(valid[:,b].sum())
            if not length or not bool(valid[:length,b].all()) or bool(valid[length:,b].any()):
                raise ValueError('replay requires one contiguous episode per collector slot')
            boundary = rollout['terminated'][:length,b] | rollout['truncated'][:length,b]
            if not bool(boundary[-1]) or bool(boundary[:-1].any()):
                raise ValueError('replay requires complete episodes with a single final boundary')
            episode = {key:rollout[key][:length,b].detach().cpu().clone() for key in REPLAY_KEYS}
            episode['episode_id'] = self.replay_total_episodes
            self.replay_total_episodes += 1
            added.append(episode)
        self.episode_replay.extend(added)
        self.episode_replay = self.episode_replay[-self.replay_capacity_episodes:]
        return len(added)

    @torch.no_grad()
    def prepare_replay_rows(self, fresh_episodes):
        """Reconstruct every retained history once per fit under current frozen pi/RMS.

        Actor parameters and RMS remain unchanged throughout this critic transaction.
        No stored actor h/c is reused. All tensors returned here are detached labels
        or input features; the target network itself is queried per minibatch.
        """
        episodes = self.episode_replay
        T, B = max(len(ep['C']) for ep in episodes), len(episodes)
        data = {}
        valid = torch.zeros(T,B,dtype=torch.bool,device=self.device)
        for key in ('S','A_exec','C','terminated','truncated'):
            sample = episodes[0][key]
            values = torch.zeros((T,B)+tuple(sample.shape[1:]),dtype=sample.dtype,device=self.device)
            for b, ep in enumerate(episodes):
                values[:len(ep[key]),b] = ep[key].to(self.device)
            data[key] = values
        for b,ep in enumerate(episodes):
            valid[:len(ep['C']),b] = True
        recurrent = self.actor.initial_state(B,self.device)
        h0 = torch.empty(T,B,self.lstm_size,device=self.device)
        c0 = torch.empty_like(h0)
        means = torch.empty(T,B,self.action_dim,device=self.device)
        log_stds = torch.empty_like(means)
        for t in range(T):
            h0[t],c0[t] = recurrent[0][0],recurrent[1][0]
            mean,log_std,recurrent = self.actor.step(data['S'][t],recurrent)
            means[t],log_stds[t] = mean,log_std
            keep = (valid[t] & ~(data['terminated'][t]|data['truncated'][t])).to(mean.dtype).view(1,B,1)
            recurrent = recurrent[0]*keep,recurrent[1]*keep
        # Behavior Gaussian parameters remain in the CPU episode record only
        # as provenance. One-step targets are independent of those parameters.
        partial,bootstrap,weights = one_step_components(data['C'],valid,
            data['terminated'],data['truncated'],COST_GAMMA)
        time = torch.arange(T,device=self.device,dtype=torch.float32)[:,None].expand(T,B)
        features = self._critic_features(data['S'],time,h0,c0)
        ids = torch.arange(T*B,device=self.device).reshape(T,B)
        endpoint = (ids+B).clamp(max=T*B-1)
        current_valid_ids = ids[valid]
        fresh = (current_valid_ids % B) >= B-fresh_episodes
        ages = (self.replay_total_episodes-1-torch.tensor(
            [ep['episode_id'] for ep in episodes],device=self.device))[None,:].expand(T,B)
        return dict(states=features[valid], executed_actions=data['A_exec'][valid],
            partial=partial[valid],bootstrap=bootstrap[valid],weights=weights[valid],
            endpoint_states=features.reshape(T*B,-1)[endpoint[valid]],
            endpoint_mean=means.reshape(T*B,-1)[endpoint[valid]],
            endpoint_log_std=log_stds.reshape(T*B,-1)[endpoint[valid]],
            fresh_indices=torch.where(fresh)[0], ages=ages[valid])

    @torch.no_grad()
    def td_targets_for_rows(self, batch, indices):
        """Construct fresh [M,128] targets; finite episode endings have zero bootstrap."""
        targets = batch['partial'][indices,None].expand(-1,self.num_quantiles).clone()
        mask = batch['bootstrap'][indices]
        if bool(mask.any()):
            endpoint = indices[mask]
            means = batch['endpoint_mean'][endpoint]
            noise = torch.randn(means.shape,device=self.device,dtype=means.dtype,generator=self.replay_generator)
            raw = means+batch['endpoint_log_std'][endpoint].exp()*noise
            actions = self._executed_actions(raw)
            z = self.reward_target_critic(batch['endpoint_states'][endpoint],actions)
            targets[mask] += COST_GAMMA*z
        return targets.detach()

    def update_critic(self, rollout, permutation=None, regression_epochs=1,target_update_mode='hard'):
        """Exactly 100 warmup or 400 main minibatches, followed by one hard copy."""
        if permutation is not None or target_update_mode != 'hard':
            raise ValueError('recent replay uses signed replacement sampling and hard targets')
        fresh_count = self._append_replay(rollout)
        batch = self.prepare_replay_rows(fresh_count)
        M = len(batch['partial']); size = int(self.critic_minibatch_size)
        steps = self.warmup_critic_steps if self._training_initialization else self.critic_steps_per_batch
        fresh_ids = batch['fresh_indices']; n_fresh = size//2
        loss_sum = 0.; unweighted_sum = 0.; seen = torch.zeros(M,dtype=torch.bool,device=self.device)
        for step in range(steps):
            selected_fresh = fresh_ids[torch.randint(len(fresh_ids),(n_fresh,),device=self.device,generator=self.replay_generator)]
            selected_all = torch.randint(M,(size-n_fresh,),device=self.device,generator=self.replay_generator)
            indices = torch.cat((selected_fresh,selected_all))
            target = self.td_targets_for_rows(batch,indices)
            predicted = self.reward_critic(batch['states'][indices],batch['executed_actions'][indices])
            loss, per_row = weighted_quantile_huber(predicted,target,self.taus,batch['weights'][indices],self.huber_kappa)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('nonfinite weighted critic loss')
            self.critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.reward_critic.parameters(),self.critic_grad_clip,error_if_nonfinite=True)
            self.critic_optimizer.step()
            self.critic_optimizer_steps += 1
            loss_sum += float(loss.detach()); unweighted_sum += float(per_row.mean())
            seen[indices] = True
            hook = getattr(self,'_critic_minibatch_hook',None)
            if hook is not None:
                hook(step,batch)
        self.hard_update_target()
        weight = batch['weights'].double()
        result = self._post_critic_diagnostics(rollout)
        result.update({
            'critic/reward_quantile_loss':loss_sum/steps,
            'critic/unweighted_quantile_loss':unweighted_sum/steps,
            'diagnostic/dqc/critic_optimizer_steps':float(self.critic_optimizer_steps),
            'diagnostic/dqc/critic_regression_epochs':float(steps*size/len(fresh_ids)),
            'diagnostic/dqc/target_update_hard':1.,
            'diagnostic/dqc/transition_coverage':float(seen.float().mean()),
            'diagnostic/critic/quantiles':128.,
            'diagnostic/critic/minibatches_this_rollout':steps,
            'diagnostic/target/minibatch_recomputations':steps,
            'diagnostic/replay/episodes':len(self.episode_replay),
            'diagnostic/replay/transitions':M,
            'diagnostic/replay/fresh_fraction':n_fresh/size,
            'diagnostic/replay/max_episode_age':float(batch['ages'].max()),
            'diagnostic/replay/importance_sampling_active':0.,
            'diagnostic/replay/is_mean':float(weight.mean()),
            'diagnostic/replay/is_max':float(weight.max()),
            'diagnostic/replay/is_p99':float(torch.quantile(weight,.99)),
            'diagnostic/replay/is_ess_fraction':float(weight.sum().square()/(weight.square().sum()*M)),
            'diagnostic/replay/is_clipped_fraction':0.,
        })
        return result

    @torch.no_grad()
    def actor_advantages(self, rollout, gae_batch=None):
        """Scale only undiscounted reward advantage before joint normalization/split PPO."""
        terms = super().actor_advantages(rollout,gae_batch)
        terms['mean_term'] = (terms['mean_term']/self.reward_scale).detach()
        raw = (terms['mean_term']-terms['risk_term']).detach()
        scale = raw[rollout['valid'].bool()].std(unbiased=False).clamp_min(1e-8)
        terms.update(combined_raw=raw,combined_normalization_scale=scale.detach(),combined=(raw/scale).detach())
        self._independent_factorial_actor_terms = terms
        return terms

    def _log_core(self, iteration, returns, diagnostic_disc_cost, diagnostic_raw_cost, extra=None):
        """Four report curves plus explicit raw-objective constraint fields.

        Args retain the parent's positional ABI: returns is the undiscounted
        objective reward; diagnostic_disc_cost is the fixed-.99 report cost;
        diagnostic_raw_cost is the undiscounted objective cost. Discounted
        reward reporting comes from the same saved paired episode samples.
        """
        report_reward = self._episode_samples['discounted_return']
        log = super()._log_core(iteration,report_reward,diagnostic_disc_cost,self._episode_samples['undiscounted_cost'],extra)
        stats = cost_statistics(diagnostic_raw_cost)
        for key in ('quantile/constraint_prob','constraint/empirical_failure_probability','constraint/batch_failure_probability'):
            log[key] = stats['violation_probability']
        log['quantile/q_est'] = stats['quantile']
        log['constraint/residual'] = stats['residual']
        log['constraint/feasible'] = stats['feasible']
        log['objective/reward_mean'] = float(np.mean(returns))
        log['objective/cost_mean'] = stats['mean']
        log['objective/cost_violation_probability'] = stats['violation_probability']
        log['objective/reward_discount'] = self.gamma
        log['objective/cost_discount'] = COST_GAMMA
        log['constraint/dual_cost_discount'] = 1.
        log['report/discount'] = self.report_gamma
        log['diagnostic/reward_scale'] = self.reward_scale
        self.last_empirical_prob = self.last_outage_prob = stats['violation_probability']
        return log

    def _update_rollout(self, iteration, rollout, initialization=False):
        """Explicit primal/dual transaction; fresh undiscounted cost alone enters dual.

        Warmup fits both critics and prefills the rolling100 cost window. Main
        keeps actor E8/KL unchanged; replay never reaches actor or multiplier.
        """
        rollout = self._attach_reward_budgets(rollout)
        self._training_initialization = bool(initialization)
        gae = self.prepare_gae_batch(rollout)
        epochs = self.warmup_critic_epochs if initialization else self.dqc_value_epochs
        info = self.update_value_critic(gae,epochs)
        info.update(self.update_critic(rollout,regression_epochs=epochs,target_update_mode='hard'))
        info.update(self.action_clip_diagnostics(rollout))
        info['diagnostic/dqc/fitted_warmup_active'] = float(initialization)
        samples = rollout['objective_cost']
        if initialization:
            self._append_completed_returns(self._validated_completed_returns(samples))
            return info
        self._set_actor_lr_batch_index(int(iteration))
        terms = self.actor_advantages(rollout,gae)
        info.update(self._advantage_diagnostics(rollout,terms))
        if int(iteration) == 0 or (int(iteration)+1) % 10 == 0:
            info.update(score_gradient_diagnostics(self,rollout))
        info.update(self.update_actor_epochs(rollout,terms['combined']))
        probability,residual,dual_lr = self.update_dual(samples)
        info.update({
            'lambda/probability_multiplier':float(self.lambda_probability),
            'lambda/residual':float(residual),'lambda/lr':float(dual_lr),
            'constraint/batch_failure_probability':float(probability),
            'diagnostic/dqc/actor_epochs_config':float(self.actor_epochs),
            'diagnostic/dqc/critic_n_step':1.,
            'diagnostic/dqc/target_update_hard_config':1.,
            'training/actor_lr':float(self.actor_optimizer.param_groups[0]['lr']),
            'training/critic_lr':float(self.critic_optimizer.param_groups[0]['lr']),
            'training/value_lr':float(self.value_optimizer.param_groups[0]['lr']),
            'diagnostic/dqc/mean_estimator_gae_config':1.,
            'diagnostic/dqc/combined_advantage_normalized':1.,
        })
        metrics = self._log_core(int(iteration),rollout['objective_return'].cpu().numpy(),
            rollout['report_discounted_cost'].cpu().numpy(),samples.cpu().numpy(),extra=info)
        self.actor.update_obs_rms(rollout['S'][rollout['valid']])
        self._set_actor_lr_batch_index(int(iteration)+1)
        return metrics

    def checkpoint_state(self):
        """Save raw replay episodes and its independent device RNG at rollout boundaries."""
        state = super().checkpoint_state()
        state['recent_episode_replay'] = copy.deepcopy(self.episode_replay)
        state['replay_total_episodes'] = self.replay_total_episodes
        state['replay_rng_state'] = self.replay_generator.get_state().cpu().clone()
        return state

    def validate_checkpoint_state(self, state):
        """Keep base tensors/optimizers/environment RNG validation; replace old step budget.

        The historical formal classes hard-code D32, n100 and 200 updates. The
        signed identity here rejects their payloads while the shared loadability
        validator still checks all modules, Adam moments, schedule and worker RNG.
        """
        DQCACIndependentFactorialGPU.validate_checkpoint_state(self,state)
        self._validate_dual_checkpoint_state(state)
        warm = int(state['initialization_cursor'])
        main = int(state['main_env_steps'])//12000
        expected = warm*self.warmup_critic_steps+main*self.critic_steps_per_batch
        if int(state['critic_optimizer_steps']) != expected:
            raise ValueError('replay critic optimizer clock mismatch')
        if int(state['target_update_count']) != warm+main:
            raise ValueError('expected one hard target update per completed rollout')
        if int(state['value_optimizer_steps']) != warm*48+main*96:
            raise ValueError('scalar value optimizer clock mismatch')
        applied = int(state['formal_b25_actor_epochs_applied_total'])
        early = int(state['actor_kl_early_stop_count'])
        if (int(state['actor_optimizer_steps']) != applied or not 0 <= early <= main
                or not 8*main-7*early <= applied <= 8*main-early):
            raise ValueError('actor E8/original KL clock mismatch')
        if any(not torch.equal(value,state['reward_target_critic'][key])
               for key,value in state['reward_critic'].items()):
            raise ValueError('hard-copy critic/target checkpoint mismatch')
        rng = torch.Generator(device=self.device)
        rng.set_state(state['replay_rng_state'].cpu())
        shuffle = torch.Generator(device='cpu')
        shuffle.set_state(state['dqc_formal_b25_actor_shuffle_rng_state'].cpu())
        episodes = state.get('recent_episode_replay')
        total = int(state.get('replay_total_episodes',-1))
        if total != (warm+main)*12 or not isinstance(episodes,list) or len(episodes) != min(total,100):
            raise ValueError('complete episode replay count mismatch')
        for expected_id,ep in zip(range(total-len(episodes),total),episodes):
            if ep.get('episode_id') != expected_id or any(key not in ep for key in REPLAY_KEYS):
                raise ValueError('replay episode FIFO/schema mismatch')
            n = len(ep['C'])
            if not 1 <= n <= self.horizon or any(len(ep[key]) != n for key in REPLAY_KEYS):
                raise ValueError('replay episode lengths mismatch')
            shapes = {'S':(n,self.state_dim),'A_raw':(n,self.action_dim),
                'A_exec':(n,self.action_dim),'mean_old':(n,self.action_dim),
                'log_std_old':(n,self.action_dim),'R':(n,),'C':(n,),
                'terminated':(n,),'truncated':(n,)}
            for key,shape in shapes.items():
                if not torch.is_tensor(ep[key]) or tuple(ep[key].shape) != shape or not bool(torch.isfinite(ep[key]).all()):
                    raise ValueError('invalid replay tensor '+key)
            boundary = ep['terminated'].bool()|ep['truncated'].bool()
            if not bool(boundary[-1]) or bool(boundary[:-1].any()):
                raise ValueError('replay episode has invalid finite boundary')

    def load_checkpoint_state(self, state):
        """Restore parent transaction first, then validated replay tensors/private RNG."""
        self.validate_checkpoint_state(state)
        super().load_checkpoint_state(state)
        self.episode_replay = [{key:(value.detach().cpu().clone() if torch.is_tensor(value) else value)
                                for key,value in ep.items()} for ep in state['recent_episode_replay']]
        self.replay_total_episodes = int(state['replay_total_episodes'])
        self.replay_generator.set_state(state['replay_rng_state'].cpu())

    def get_training_summary(self):
        summary = super().get_training_summary()
        summary.update(experiment_identity=IDENTITY,num_quantiles=128,critic_n_step=1,
            reward_scale=self.reward_scale,constraint_variable=COST,
            gamma_reward=self.gamma,gamma_cost=COST_GAMMA,report_gamma=.99,
            replay_episodes=len(self.episode_replay),
            expected_critic_optimizer_steps=20*self.warmup_critic_steps+814*self.critic_steps_per_batch,
            critic_budget_complete=self.critic_optimizer_steps == 327600,
            expected_actor_maximum_optimizer_steps=6512)
        return summary
