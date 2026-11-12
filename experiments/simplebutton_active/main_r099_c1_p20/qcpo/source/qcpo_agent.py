"""QCPO comparator: only finite-horizon reward/cost discounts change to one.

The original 25 midpoint atoms, two-parameter Weibull, cost_scale=10 network
units, shared optimizer, E8/M1, and first dual/RMS-only batch are unchanged.
No DCA replay, TD3, extra critic fitting, MC loss, or mean/c scaling is applied.
"""
from __future__ import annotations
import copy
import math
import numpy as np
import torch
from objective import COST_GAMMA, REWARD_GAMMA, COST, EVENT, episode_total
from raw_cost_agents import RawCostQCPO, cost_statistics

IDENTITY = 'qcpo_b12_main_r099_c1_p20_v1'


class QCPOUndiscountedAgent(RawCostQCPO):
    """Original QCPO loss/collector with gamma=1 objectives and separate .99 reports."""

    def __init__(self,args,env):
        self.report_gamma = .99
        super().__init__(copy.copy(args),env)
        # Canonical constructor allocates no cached discount powers. Reward GAE,
        # density and quantile GAE read gamma; cost GAE/QR read cost_discount.
        self.gamma = args.training_gamma
        self.cost_discount = self.cost_gamma = COST_GAMMA
        if (self.n_quantile,self.epochs,self.minibatches,self.cost_scale)!=(25,8,1,10.):
            raise ValueError('QCPO comparator must retain original 25-atom E8/M1 cost-scale10 configuration')

    def _checkpoint_identity(self):
        identity = super()._checkpoint_identity()
        identity.update(experiment_identity=IDENTITY,raw_cost_identity=IDENTITY,
            raw_cost_primal=COST,
            raw_cost_dual='undiscounted_environment_cost',
            raw_cost_violation_event=EVENT,
            actual_gamma_reward=self.gamma,actual_gamma_cost=self.cost_discount,actual_dual_gamma=1.,report_gamma=.99,
            actual_num_quantiles=25,replay_enabled=False,reward_scale=1.,
            td3_enabled=False,mc_eta=0.,actual_optimizer_epochs=8,
            actual_minibatches=1,actual_cost_scale=10.,
            actual_reward_actor_outer_discount='none',
            actual_target='official_one_step_25_atoms_finite_done_mask',
            actual_first_batch='dual_and_rms_only',
            dca_enhancements_applied=False)
        return identity

    @torch.no_grad()
    def _capture_episode_metrics(self,rollout,reward_key):
        """Raw totals enter the objective; fixed .99 values are paired report samples."""
        valid = rollout['valid'].bool()
        reward = torch.where(valid,rollout[reward_key],0).double()
        cost = torch.where(valid,rollout['diagnostic_cost'],0).double()
        discount = self.report_gamma**torch.arange(len(reward),device=reward.device,dtype=torch.float64)
        raw_reward,raw_cost = reward.sum(0),cost.sum(0)
        report_reward,report_cost = (discount[:,None]*reward).sum(0),(discount[:,None]*cost).sum(0)
        target_reward, target_cost = episode_total(reward,self.gamma), episode_total(cost,self.cost_discount)
        rollout.update(objective_return=target_reward,objective_cost=raw_cost,
            report_discounted_return=report_reward,report_discounted_cost=report_cost)
        self._raw_episode_returns = raw_reward.cpu().numpy()
        self._episode_samples = dict(discounted_return=report_reward.cpu().tolist(),
            undiscounted_return=raw_reward.cpu().tolist(),discounted_cost=report_cost.cpu().tolist(),
            undiscounted_cost=raw_cost.cpu().tolist(),objective_return=target_reward.cpu().tolist(),
            objective_cost=raw_cost.cpu().tolist())
        return rollout

    def _log_core(self,iteration,returns,diagnostic_disc_cost,diagnostic_raw_cost,extra=None):
        """Preserve all four curves while raw total cost defines every constraint field."""
        report_return = self._episode_samples['discounted_return']
        row = super()._log_core(iteration,report_return,diagnostic_disc_cost,self._episode_samples['undiscounted_cost'],extra)
        stats = cost_statistics(diagnostic_raw_cost)
        row.update({'quantile/q_est':stats['quantile'],
            'objective/reward_mean':float(np.mean(returns)),'objective/cost_mean':stats['mean'],
            'objective/cost_violation_probability':stats['violation_probability'],
            'objective/reward_discount':self.gamma,'objective/cost_discount':self.cost_discount,'report/discount':.99})
        return row

    def update_rollout(self,iteration,rollout):
        """One literal original QCPO transaction, preserving dual/RMS/optimizer order.

        Finite done masks in the original helper block reward/cost bootstrapping
        at H=1000. The original final reset value remains available to the
        official pre-mask Weibull normalization, without changing that rule.
        """
        fixed = self._fixed_rollout_targets(rollout)
        quantile,residual = self.update_dual(rollout['objective_cost'])
        self.model.update_obs_rms(fixed['obs'])
        loss_info = {}
        if int(iteration)==0 and self.update_counter==0:
            self.rms_only_batch_count += 1
        else:
            loss_info = self._optimize_fixed_rollout(fixed)
        self._require_finite('density probability ratio',fixed['probability_ratio'])
        density_log = torch.log(fixed['probability_ratio'])
        extra = dict(loss_info,**{
            'lambda/quantile_multiplier':float(self.cost_penalty),
            'lambda/quantile_residual':float(residual),
            'lambda/dual_window_count':len(self.current_ep_costs),
            'lambda/dual_updates':self.dual_update_count,
            'quantile/dual_internal_cost_order_statistic':float(quantile),
            'author/density_weight_mean':float(fixed['probability_ratio'].mean()),
            'author/density_log_weight_mean':float(density_log.mean()),
            'author/density_log_weight_min':float(density_log.min()),
            'author/density_log_weight_max':float(density_log.max()),
            'author/rms_count':float(self.model.obs_rms.count),
            'training/optimizer_steps':self.update_counter,
            'training/rms_only_batches':self.rms_only_batch_count,
        })
        self.last_loss_info = dict(extra)
        return self._log_core(iteration,rollout['objective_return'].cpu().numpy(),
            rollout['report_discounted_cost'].cpu().numpy(),rollout['objective_cost'].cpu().numpy(),extra)

    def validate_checkpoint_state(self,state):
        """Retain original sampler/RNG validation and prevalidate all optimizer tensors."""
        super().validate_checkpoint_state(state)
        steps = int(state['main_env_steps'])
        if steps%12000 or state['initialization_env_steps']!=0 or not 0<=steps<=10008000:
            raise ValueError('QCPO finite B25 physical clock mismatch')
        batches = steps//12000
        if int(state['update_counter'])!=max(0,batches-1)*8:
            raise ValueError('QCPO original E8/M1 optimizer budget mismatch')
        if int(state['dual_update_count'])!=batches or int(state['rms_only_batch_count'])!=min(1,batches):
            raise ValueError('QCPO dual/RMS clock mismatch')
        if len(state['dual_cost_returns'])!=min(100,batches*12):
            raise ValueError('QCPO original rolling dual window mismatch')
        candidate = copy.deepcopy(self.model)
        candidate.load_state_dict(state['model'])
        optimizer = copy.deepcopy(self.optimizer)
        optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
        for parameter,slots in optimizer.state.items():
            for key,value in slots.items():
                if torch.is_tensor(value) and key!='step' and value.shape!=parameter.shape:
                    raise ValueError('QCPO optimizer moment shape mismatch')
        for key in ('pid_i','cost_penalty','last_dual_cost_quantile','last_dual_residual'):
            if not math.isfinite(float(state[key])):
                raise ValueError('QCPO nonfinite checkpoint scalar '+key)

    def load_checkpoint_state(self,state):
        """Full parent sampler restoration with rollback if exact reset reconstruction fails."""
        self.validate_checkpoint_state(state)
        rollback = copy.deepcopy(self.checkpoint_state())
        try:
            super().load_checkpoint_state(state)
        except BaseException:
            super().load_checkpoint_state(rollback)
            raise
