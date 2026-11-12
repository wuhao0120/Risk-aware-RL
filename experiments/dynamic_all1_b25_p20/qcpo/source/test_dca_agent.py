"""Bounded CPU integration checks; only the simulator creation is substituted."""
import copy
from collections import deque
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import base_runner
base_runner.setup()
import numpy as np
import torch
from torch import nn
import agents.vec_base as vec_base
import agent as module
from agent import Agent, complete_reward_suffix, pi_step
from run import make_args


class VectorBoundary:
    n, obs_dim, act_dim, _seed = 1000, 2, 2, 0
    def __init__(self):
        self.states=[dict(version=1,backend='mjwarp',action_mode='explicit_box_clip',
            dynamics_profile='point_velocity_implicitfast_v1',seed=0,slot=i,reset_count=0,goal_count=0) for i in range(25)]
    def checkpoint_rng_state(self):return copy.deepcopy(self.states)
    def restore_rng_state(self,state):self.states=copy.deepcopy(state)


def construct(arm='schemeb_raw_mean100_lambda_norm'):
    args=make_args(0, device='cpu', arm=arm)
    env=SimpleNamespace(env_id='SafetyPointSimpleButton0-v0',dynamics_profile='point_velocity_implicitfast_v1',
        action_space=SimpleNamespace(low=np.array([-1.,-1.]),high=np.array([1.,1.])))
    torch.manual_seed(2719)
    with patch.object(vec_base,'make_vec_env',return_value=VectorBoundary()):
        return Agent(args,env)


def fixture(agent):
    states=torch.tensor([[[.1,.2],[.2,.3]],[[.3,.4],[.4,.5]],[[.5,.6],[9.,9.]]])
    valid=torch.tensor([[1,1],[1,1],[1,0]],dtype=torch.bool)
    terminal=torch.tensor([[0,0],[0,1],[1,0]],dtype=torch.bool)
    truncated=torch.zeros_like(terminal)
    recurrent=agent.actor.initial_state(2,torch.device('cpu'))
    means=[];stds=[];h=[];c=[];nh=[];nc=[]
    with torch.no_grad():
        for t in range(3):
            h.append(recurrent[0][0].clone());c.append(recurrent[1][0].clone())
            mean,std,recurrent=agent.actor.step(states[t],recurrent)
            means.append(mean);stds.append(std);nh.append(recurrent[0][0].clone());nc.append(recurrent[1][0].clone())
            keep=(valid[t]&~terminal[t]).float().view(1,2,1)
            recurrent=recurrent[0]*keep,recurrent[1]*keep
    raw=torch.stack(means)+.1
    rollout=dict(S=states,R=torch.tensor([[1.,2.],[3.,4.],[5.,0.]]),C=torch.tensor([[4.,5.],[7.,10.],[5.,0.]]),
        valid=valid,terminated=terminal,truncated=truncated,h0=torch.stack(h),c0=torch.stack(c),
        next_h=torch.stack(nh),next_c=torch.stack(nc),A_raw=raw,A_exec=raw.clamp(-1,1),
        mean_old=torch.stack(means),log_std_old=torch.stack(stds),
        next_states=torch.cat((states[1:],torch.full_like(states[:1],99.))))
    return agent._attach_reward_budgets({key:value.detach().clone() for key,value in rollout.items()})


class TinyCritic(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Linear(2,4)
    def forward(self,states,actions):return self.net(torch.cat((states,actions),-1))


class SchemeBAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
        cls.c100=construct()
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)


    def test_complete_mc_snapshot_and_cut_rejection(self):
        rollout=fixture(self.c100)
        batch=self.c100.prepare_gae_batch(rollout)
        expected=torch.tensor([[9.,6.],[8.,4.],[5.,0.]])
        torch.testing.assert_close(batch['targets'],expected)
        torch.testing.assert_close(batch['advantages'][rollout['valid']],(expected-batch['values'])[rollout['valid']])
        old=copy.deepcopy(self.c100.value_critic.state_dict())
        frozen=batch['advantages'].clone()
        try:
            with torch.no_grad():
                for p in self.c100.value_critic.parameters():p.add_(2.)
            torch.testing.assert_close(batch['advantages'],frozen,rtol=0,atol=0)
        finally:self.c100.value_critic.load_state_dict(old)
        cut=copy.deepcopy(rollout);cut['terminated'][2,0]=False;cut['truncated'][2,0]=True
        with self.assertRaisesRegex(ValueError,'collection truncation'):complete_reward_suffix(cut,1000)
        torch.testing.assert_close(complete_reward_suffix(cut,3),expected)

    def test_transition_uses_independent_actual_next_queries_and_strict_terminal(self):
        instance=object.__new__(Agent);instance.__dict__=self.c100.__dict__.copy()
        instance.lambda_probability=torch.tensor(2.)
        rollout=fixture(instance);mc=instance.prepare_gae_batch(rollout)
        current=torch.tensor([[.2,.4],[.7,.3],[.5,.9]])
        # Continuing rows flatten in time-major order: (t0,b0),(t0,b1),(t1,b0).
        next_prob=torch.tensor([.9,.6,.2])
        calls=[]
        def probability(features,means,logstd,budget):
            calls.append((features.clone(),means.clone(),budget.clone()))
            return current.clone() if len(calls)==1 else next_prob.clone()
        with patch.object(instance,'_policy_probability',side_effect=probability):
            terms=instance.actor_advantages(rollout,mc)
        self.assertEqual(len(calls),2)
        torch.testing.assert_close(calls[1][2],torch.tensor([11.,10.,4.],dtype=torch.float64))
        torch.testing.assert_close(calls[1][0][:,:2],torch.tensor([[.3,.4],[.4,.5],[.5,.6]]))
        expected=torch.tensor([[.7,.2],[-.5,-.3],[.5,0.]])
        torch.testing.assert_close(terms['cdf_advantage'],expected)
        # Equal whole-episode cost15 is nonviolating; cost16 is violating.
        self.assertEqual(float(terms['cdf_action'][1,1]),0.)
        self.assertEqual(float(terms['cdf_action'][2,0]),1.)
        self.assertFalse(terms['combined'].requires_grad)
        bad=copy.deepcopy(rollout);bad['next_h'][0,0,0]+=1
        with self.assertRaisesRegex(ValueError,'context mismatch'):instance.actor_advantages(bad,mc)


    def test_arm_specific_normalization_is_detached_frozen_and_not_double_divided(self):
        reward=torch.tensor([20.,40.,90000.],requires_grad=True)
        risk=torch.tensor([.1,-.1,-9000.],requires_grad=True)
        multiplier=torch.tensor(2.,requires_grad=True)
        valid=torch.tensor([True,True,False])
        labels=module.actor_labels_for_arm(reward,risk,multiplier,valid,module.ARMS[0])
        torch.testing.assert_close(labels['combined_normalization_scale'],torch.tensor(3.))
        torch.testing.assert_close(labels['combined'],(reward.detach()/100.-2*risk.detach())/3.)
        self.assertTrue(all(not item.requires_grad for item in labels.values()))
        before=labels['combined'].clone()
        with torch.no_grad():multiplier.fill_(8.)
        torch.testing.assert_close(labels['combined'],before,rtol=0,atol=0)
        self.assertEqual(float(labels['combined_normalization_scale']),3.)
        self.assertNotAlmostEqual(float(labels['raw_combined'][valid].std(unbiased=False)),3.)



    def test_double_budget_preserves_strict_boundary(self):
        atoms=torch.full((2,128),15.,dtype=torch.float32)
        budget=torch.tensor([15.-1e-10,15.],dtype=torch.float64)
        torch.testing.assert_close(self.c100.probability_from_atoms(atoms,budget),torch.tensor([1.,0.]))

    def test_fresh25_pi_updates_during_warmup_and_not_from_window(self):
        instance=object.__new__(Agent);instance.__dict__=self.c100.__dict__.copy()
        instance.lambda_probability=torch.tensor(0.)
        instance.dual_return_window=deque(maxlen=100)
        p,error,lr=instance.update_dual(torch.full((25,),15.))
        self.assertEqual((p,error,lr),(0.,-.2,.2))
        self.assertEqual(instance.pi_update_count,1);self.assertAlmostEqual(instance.pi_error_ema,-.04)
        self.assertEqual(float(instance.lambda_probability),0.)
        p,error,lr=instance.update_dual(torch.full((25,),16.))
        self.assertEqual(p,1.)
        self.assertEqual(instance.last_dual_prob,.5)  # reporting window deliberately differs
        self.assertEqual(instance.pi_update_count,2)
        self.assertAlmostEqual(float(instance.lambda_probability),1.,places=6)

    def test_three_target_caches_fixed_within_stage_and_refreshed_after_copy(self):
        instance=object.__new__(Agent)
        instance.device=torch.device('cpu');instance.horizon=3;instance.num_quantiles=4
        instance._training_initialization=True;instance.critic_minibatch_size=4
        instance.critic_grad_clip=10.;instance.gamma=1.;instance.schemeb_target_cache_generations=0
        instance.target_update_count=0;instance.critic_optimizer_steps=0
        torch.manual_seed(108);instance.reward_critic=TinyCritic();instance.reward_target_critic=copy.deepcopy(instance.reward_critic).requires_grad_(False)
        instance.critic_optimizer=torch.optim.Adam(instance.reward_critic.parameters(),lr=.01)
        instance.replay_generator=torch.Generator().manual_seed(81)
        instance.episode_replay=[{}]
        batch=dict(states=torch.tensor([[1.],[2.],[3.]]),executed_actions=torch.zeros(3,1),
            partial=torch.tensor([1.,2.,3.]),bootstrap=torch.tensor([True,True,False]),
            endpoint_states=torch.tensor([[2.],[3.],[0.]]),endpoint_mean=torch.zeros(3,1),
            endpoint_log_std=torch.zeros(3,1),fresh_indices=torch.tensor([0,1,2]),ages=torch.zeros(3))
        instance._append_replay=lambda rollout:1
        instance.prepare_replay_rows=lambda fresh:batch
        instance._executed_actions=lambda raw:raw.clamp(-1,1)
        instance._post_critic_diagnostics=lambda rollout:{}
        def hardcopy():
            instance.reward_target_critic.load_state_dict(instance.reward_critic.state_dict());instance.target_update_count+=1
        instance.hard_update_target=hardcopy
        caches=[];sizes=[0,0,0]
        def stage_hook(stage,target,rows):
            self.assertEqual(instance.target_update_count,stage)
            caches.append(target.clone())
            torch.testing.assert_close(target[-1],torch.full((4,),3.))
        def minibatch(stage,step,target,indices):
            torch.testing.assert_close(target,caches[stage],rtol=0,atol=0);sizes[stage]+=1
        instance._schemeb_stage_hook=stage_hook;instance._schemeb_minibatch_hook=minibatch
        rollout=dict(R=torch.zeros(3,1),valid=torch.ones(3,1,dtype=torch.bool),
            terminated=torch.tensor([[0],[0],[1]],dtype=torch.bool),truncated=torch.zeros(3,1,dtype=torch.bool))
        metrics=instance.update_critic(rollout)
        self.assertEqual(sizes,[34,33,33]);self.assertEqual(instance.critic_optimizer_steps,100)
        self.assertEqual(instance.target_update_count,3);self.assertEqual(instance.schemeb_target_cache_generations,3)
        self.assertFalse(torch.equal(caches[0],caches[1]));self.assertFalse(torch.equal(caches[1],caches[2]))
        self.assertEqual(metrics['diagnostic/target/minibatch_recomputations'],0)
        for key,value in instance.reward_critic.state_dict().items():
            torch.testing.assert_close(value,instance.reward_target_critic.state_dict()[key],rtol=0,atol=0)

    def test_undiscounted_td_target_and_no_reward_outer_factor(self):
        instance=object.__new__(Agent);instance.__dict__=self.c100.__dict__.copy()
        instance.num_quantiles=4
        class FixedTarget(nn.Module):
            def forward(self,s,a):return s.new_full((len(s),4),10.)
        instance.reward_target_critic=FixedTarget()
        batch=dict(partial=torch.tensor([2.,3.]),bootstrap=torch.tensor([True,False]),
            endpoint_mean=torch.zeros(2,2),endpoint_log_std=torch.zeros(2,2),endpoint_states=torch.zeros(2,3))
        targets=instance.td_targets_for_rows(batch,torch.tensor([0,1]))
        torch.testing.assert_close(targets,torch.tensor([[12.]*4,[3.]*4]))
        rollout=fixture(self.c100);mc=self.c100.prepare_gae_batch(rollout)
        terms=self.c100.actor_advantages(rollout,mc)
        weights=torch.ones(3,1)
        torch.testing.assert_close(terms['mean_term'],weights*mc['advantages']/100.)
        self.assertEqual(self.c100.gamma,1.)

    def test_actor_and_dual_share_raw_event_despite_discounted_report(self):
        instance=object.__new__(Agent);instance.__dict__=self.c100.__dict__.copy()
        instance.lambda_probability=torch.tensor(0.)
        instance.dual_return_window=deque(maxlen=100)
        rollout=fixture(instance)
        rollout['C'][:,0]=torch.tensor([0.,0.,15.1])
        rollout=instance._attach_reward_budgets(rollout)
        mc=instance.prepare_gae_batch(rollout)
        terms=instance.actor_advantages(rollout,mc)
        self.assertEqual(float(terms['cdf_action'][2,0]),1.)
        self.assertGreater(float(rollout['C'][:,0].sum()),15.)
        p,_,_=instance.update_dual(torch.full((25,),15.1,dtype=torch.float64))
        self.assertEqual(p,1.)
        instance._capture_episode_metrics(rollout,'R')
        torch.testing.assert_close(rollout['objective_cost'],torch.tensor([15.1,15.],dtype=torch.float64),atol=1e-6,rtol=0)
        self.assertGreater(float(rollout['critic_objective_cost'][0]),15.)
        self.assertLess(float(rollout['report_discounted_cost'][0]),15.)
        torch.testing.assert_close(rollout['critic_objective_cost'],rollout['objective_cost'])
        torch.testing.assert_close(rollout['objective_return'],rollout['R'].double().sum(0))
        self.assertEqual(instance._episode_samples['objective_return'],instance._episode_samples['undiscounted_return'])
        self.assertNotEqual(instance._episode_samples['objective_return'],instance._episode_samples['discounted_return'])

    def test_initial_checkpoint_roundtrip_scientific_identity_and_clocks(self):
        for instance in (self.c100,):
            state=copy.deepcopy(instance.checkpoint_state())
            instance.validate_checkpoint_state(state)
            before=torch.random.get_rng_state().clone();instance.load_checkpoint_state(state)
            torch.testing.assert_close(before,torch.random.get_rng_state(),rtol=0,atol=0)
            self.assertEqual(instance.pi_update_count,0)
            bad=copy.deepcopy(state);bad['actual_pi_ema']=.3
            with self.assertRaisesRegex(ValueError,'identity mismatch'):instance.validate_checkpoint_state(bad)
            bad=copy.deepcopy(state);bad['actual_actor_normalization']='wrong'
            with self.assertRaisesRegex(ValueError,'identity mismatch'):instance.validate_checkpoint_state(bad)
            bad=copy.deepcopy(state);bad['schemeb_state']['pi_updates']=1
            with self.assertRaisesRegex(ValueError,'PI/cache'):instance.validate_checkpoint_state(bad)


if __name__=='__main__':unittest.main()
