"""QCPO gamma=1 regression checks; DCA-specific enhancements must stay absent.

The original positive 25-atom/Weibull, density normalization, finite chunking,
shared Adam, E8/M1 and first-batch RMS/dual-only contracts are retained.
"""
from collections import deque
import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import base_runner as base
base.setup()
from qcpo_agent import QCPOUndiscountedAgent
from raw_cost_agents import RawCostQCPO
from agents import official_raw_qcpo_utils as official


@pytest.fixture(autouse=True)
def bounded_cpu(monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(7219)
    # The QCPO parent otherwise snapshots every CUDA device even for CPU agents.
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)


def small_agent(T=6,B=2):
    agent=object.__new__(QCPOUndiscountedAgent)
    agent.device=torch.device('cpu')
    agent.gamma=agent.cost_discount=agent.cost_gamma=1.
    agent.report_gamma=.99
    agent.n=agent.new_T=T;agent.num_envs=agent.new_B=B
    agent.n_quantile=25;agent.cost_scale=10.;agent.gae_lambda=.97
    agent.cost_gae_lambda=.97;agent.weibull_tail_ind=17;agent.target_outage_ind=19
    agent.cost_threshold=15.;agent.cost_threshold_scaled=1.5
    agent.target_outage_prob=agent.failure_alpha=.2
    agent.q_alpha=.8;agent.return_threshold=15.
    agent.current_ep_costs=deque(maxlen=100);agent.pid_i=agent.cost_penalty=0.
    agent.pid_Ki=.1;agent.lambda_dual=torch.zeros(1);agent.dual_update_count=0
    return agent


def rollout_targets(T=6,B=2):
    """Nondegenerate raw cost/Weibull data with a finite horizon at the last row."""
    t=torch.arange(T,dtype=torch.float32)[:,None]
    b=torch.arange(B,dtype=torch.float32)[None,:]
    shape=(T,B)
    reward=.7+t*.13+b*.1
    cost=.2+t*.03+b*.08
    distribution=.3+t[...,None]*.11+b[...,None]*.02+torch.arange(25)*.025
    done=torch.zeros(shape);done[-1]=1.
    return dict(reward=reward,diagnostic_cost=cost,done=done,valid=torch.ones(shape),
        r_value=.1+t*.2+b*.04,c_dist=distribution,
        c_w_location=torch.zeros(shape),c_w_alpha=1.2+t*.03+b*.04,
        c_w_beta=.8+t*.02+b*.02,
        r_bootstrap_original=torch.full((B,),99.),c_bootstrap_original=torch.full((B,25),99.),
        location_bootstrap_original=torch.zeros(B),alpha_bootstrap_original=torch.full((B,),1.3),
        beta_bootstrap_original=torch.full((B,),.95),obs=torch.zeros(T,B,3),
        prev_action=torch.zeros(T,B,2),prev_reward=torch.zeros(shape),action=torch.zeros(T,B,2),
        mu_old=torch.zeros(T,B,2),log_std_old=torch.zeros(T,B,2),h0=torch.zeros(T,B,4),c0=torch.zeros(T,B,4),
        disc_return=reward.sum(0),diagnostic_disc_cost=cost.sum(0),diagnostic_raw_cost=cost.sum(0),
        internal_trajectory_cost=cost.sum(0))


def scalar_gae_reference(reward,value,lam):
    advantage=torch.zeros_like(reward);carry=torch.zeros(reward.shape[1])
    for t in range(len(reward)-1,-1,-1):
        successor=value[t+1] if t+1<len(reward) else 0.
        carry=reward[t]+successor-value[t]+lam*carry
        advantage[t]=carry
    return advantage,advantage+value


def test_qcpo_reward_cost_gae_and_one_step_quantiles_use_gamma_one():
    agent=small_agent();rollout=rollout_targets()
    fixed=agent._fixed_rollout_targets(rollout)
    advantage,return_=scalar_gae_reference(rollout['reward'],rollout['r_value'],.97)
    _,cost_return=scalar_gae_reference(rollout['diagnostic_cost']/10.,rollout['c_dist'].mean(-1),.97)
    torch.testing.assert_close(fixed['reward_advantage'],advantage)
    torch.testing.assert_close(fixed['reward_return'],return_)
    torch.testing.assert_close(fixed['cost_return'],cost_return)
    expected=rollout['diagnostic_cost'][...,None].expand(-1,-1,25)/10.
    expected=expected.clone();expected[:-1]+=rollout['c_dist'][1:]
    torch.testing.assert_close(fixed['cost_distribution_target'],expected)
    assert fixed['cost_distribution_target'].shape[-1] == 25
    assert all(not tensor.requires_grad for tensor in fixed.values() if torch.is_tensor(tensor))
    assert bool(torch.isfinite(fixed['probability_ratio']).all())
    assert not torch.allclose(expected[:-1],rollout['diagnostic_cost'][:-1,...,None]/10.+.99*rollout['c_dist'][1:])


def test_qcpo_finite_terminal_cancels_bootstrap_and_chunk_boundary_keeps_unit_bootstrap():
    agent=small_agent();agent.new_T=3;agent.new_B=4
    rollout=rollout_targets();fixed=agent._fixed_rollout_targets(rollout)
    expected=rollout['diagnostic_cost'][...,None].expand(-1,-1,25)/10.
    expected=expected.clone();expected[:-1]+=rollout['c_dist'][1:]
    expected=agent._transform(expected)
    torch.testing.assert_close(fixed['cost_distribution_target'],expected)
    # Last row of first chunk bootstraps; last row of final chunk is finite terminal.
    torch.testing.assert_close(fixed['cost_distribution_target'][-1,0],rollout['diagnostic_cost'][2,0]/10.+rollout['c_dist'][3,0])
    torch.testing.assert_close(fixed['cost_distribution_target'][-1,1],torch.full((25,),float(rollout['diagnostic_cost'][-1,0]/10.)))


def test_qcpo_lambda_one_recovers_raw_reward_sum_without_touching_original_lambda_default():
    agent=small_agent();rollout=rollout_targets()
    assert agent.gae_lambda == agent.cost_gae_lambda == .97
    agent.gae_lambda=agent.cost_gae_lambda=1.
    fixed=agent._fixed_rollout_targets(rollout)
    expected_reward=rollout['reward'].flip(0).cumsum(0).flip(0)
    expected_cost=rollout['diagnostic_cost'].flip(0).cumsum(0).flip(0)/10.
    torch.testing.assert_close(fixed['reward_return'],expected_reward)
    torch.testing.assert_close(fixed['cost_return'],expected_cost)


def test_density_gamma_one_is_finite_without_one_minus_gamma_denominator():
    rollout=rollout_targets()
    cost=rollout['diagnostic_cost']/10.
    tail=rollout['c_dist'][...,17:]
    shape,scale=rollout['c_w_alpha'],rollout['c_w_beta']
    bootstrap_shape=rollout['alpha_bootstrap_original'][None]
    bootstrap_scale=rollout['beta_bootstrap_original'][None]
    result=official.compute_prob_ratio(cost,tail,shape,scale,rollout['done'],bootstrap_shape,bootstrap_scale,1.,log_clip_range=.5,normalize=False)
    next_shape=torch.cat((shape[1:],bootstrap_shape));next_scale=torch.cat((scale[1:],bootstrap_scale))
    def log_density(x,a,b):
        return torch.log(a)[...,None]-a[...,None]*torch.log(b)[...,None]+(a[...,None]-1)*torch.log(x)-(x/b[...,None])**a[...,None]
    difference=log_density(tail-cost[...,None],next_shape,next_scale)-log_density(tail,shape,scale)
    expected=torch.exp(difference.clamp(-.5,.5)*(1-rollout['done'])[...,None])
    torch.testing.assert_close(result,expected)
    assert bool(torch.isfinite(result).all())
    torch.testing.assert_close(result[-1],torch.ones_like(result[-1]))


def test_original_zero_variance_density_is_detected_not_silently_repaired():
    """Retain and expose the official normalization edge case; no hidden epsilon."""
    shape=torch.ones(3,2);tail=torch.ones(3,2,8);cost=torch.zeros(3,2)
    result=official.compute_prob_ratio(cost,tail,shape,shape,torch.zeros(3,2),torch.ones(1,2),torch.ones(1,2),1.,normalize=True)
    assert bool(torch.isnan(result).all())
    with pytest.raises(FloatingPointError):
        small_agent()._require_finite('density probability ratio',result)


def late_cost_rollout(B=25):
    T=1000;reward=torch.zeros(T,B);reward[900]=2.
    cost=torch.zeros(T,B);cost[900]=16.
    return dict(reward=reward,diagnostic_cost=cost,valid=torch.ones(T,B),
        disc_return=reward.sum(0),diagnostic_disc_cost=cost.sum(0),diagnostic_raw_cost=cost.sum(0),
        internal_trajectory_cost=cost.sum(0))


def test_qcpo_late_cost_uses_raw_threshold_and_retains_discounted_reports():
    agent=small_agent(T=1000,B=25);rollout=late_cost_rollout()
    original_reward=rollout['reward'].clone();original_cost=rollout['diagnostic_cost'].clone()
    agent._capture_episode_metrics(rollout,'reward')
    torch.testing.assert_close(rollout['objective_return'],torch.full((25,),2.,dtype=torch.float64))
    torch.testing.assert_close(rollout['objective_cost'],torch.full((25,),16.,dtype=torch.float64))
    torch.testing.assert_close(rollout['report_discounted_cost'],torch.full((25,),16*.99**900,dtype=torch.float64))
    assert bool((rollout['report_discounted_cost']<15.).all())
    empirical_quantile,residual=agent.update_dual(rollout['objective_cost'])
    assert empirical_quantile == 16. and residual == pytest.approx(.1)
    assert agent.cost_penalty == pytest.approx(.01)
    assert agent.cost_threshold == 15. and agent.cost_threshold_scaled == 1.5
    torch.testing.assert_close(rollout['reward'],original_reward,rtol=0,atol=0)
    torch.testing.assert_close(rollout['diagnostic_cost'],original_cost,rtol=0,atol=0)


def test_qcpo_exact_limit_has_zero_dual_residual_and_no_false_violation():
    agent=small_agent()
    quantile,residual=agent.update_dual(torch.full((25,),15.))
    assert quantile == 15. and residual == 0. and agent.cost_penalty == 0.


class DummyVector:
    """Native RNG-schema stub for exercising real model/optimizer checkpoint code."""
    n=1000;obs_dim=2;act_dim=2
    def __init__(self):
        self.state=[dict(version=1,backend='mjwarp',seed=0,slot=b,reset_count=0,goal_count=0,
            action_mode='author_raw_env_clamp',dynamics_profile='point_velocity_implicitfast_v1') for b in range(25)]
    def checkpoint_rng_state(self):return copy.deepcopy(self.state)
    def restore_rng_state(self,state):self.state=copy.deepcopy(state)


@pytest.fixture
def full_agent(monkeypatch,tmp_path):
    import agents.vec_base as vec_base
    from qcpo_run import make_args
    vector=DummyVector()
    monkeypatch.setattr(vec_base,'make_vec_env',lambda *args,**kwargs:vector)
    env=SimpleNamespace(env_id='SafetyPointSimpleButton0-v0',dynamics_profile='point_velocity_implicitfast_v1',
        action_space=SimpleNamespace(low=np.array([-1.,-1.]),high=np.array([1.,1.])))
    args=make_args(0,tmp_path,device='cpu')
    agent=QCPOUndiscountedAgent(args,env)
    return agent,args,env


def test_qcpo_full_constructor_preserves_original_network_optimizer_and_budget(full_agent):
    agent,args,env=full_agent
    assert agent.gamma == agent.cost_discount == agent.cost_gamma == 1.
    assert agent.report_gamma == .99
    assert agent.n_quantile == agent.model.n_quantile == 25
    torch.testing.assert_close(agent.model.tau,(torch.arange(25)+.5)/25.)
    assert agent.target_outage_ind == 19 and agent.weibull_tail_ind == 17
    assert agent.model.raw_cost_support and agent.model.w_location_gap is None
    assert (agent.n,agent.num_envs,agent.new_T,agent.new_B)==(1000,25,100,250)
    assert agent.num_iterations == 400
    assert agent.epochs == 8 and agent.minibatches == 1
    assert (agent.num_iterations-1)*agent.epochs*agent.minibatches == 3192
    assert agent.cost_scale == 10. and agent.pid_Ki == .1
    assert agent.gae_lambda == agent.cost_gae_lambda == .97
    assert agent.learning_rate == agent.optimizer.defaults['lr'] == .0001
    assert len(agent.optimizer.param_groups) == 1
    assert set(id(p) for p in agent.optimizer.param_groups[0]['params']) == set(id(p) for p in agent.model.parameters())
    for enhancement in ('episode_replay','replay_capacity_episodes','critic_steps_per_batch','reward_scale','reward_critic','reward_target_critic'):
        assert not hasattr(agent,enhancement),enhancement
    # Full model tensors match the original RawCostQCPO at the same random seed.
    torch.manual_seed(9017);baseline=RawCostQCPO(args,env)
    torch.manual_seed(9017);new=QCPOUndiscountedAgent(args,env)
    assert baseline.model.state_dict().keys() == new.model.state_dict().keys()
    for key in baseline.model.state_dict():
        torch.testing.assert_close(new.model.state_dict()[key],baseline.model.state_dict()[key],rtol=0,atol=0)
    for name in ('epochs','minibatches','ratio_clip','gae_lambda','cost_gae_lambda','cost_scale','cost_value_loss_coeff','value_loss_coeff','quantile_loss_coeff','weibull_loss_coeff','entropy_loss_coeff','clip_grad_norm','pid_Ki','penalty_init','sum_norm','diff_norm','n_quantile','new_T','new_B'):
        assert getattr(new,name) == getattr(baseline,name),name


def test_qcpo_full_checkpoint_roundtrip_rejects_cross_discount_and_missing_optimizer(full_agent):
    agent,args,env=full_agent
    saved=copy.deepcopy(agent.checkpoint_state())
    agent.validate_checkpoint_state(saved)
    before=next(agent.model.parameters()).detach().clone()
    with torch.no_grad():next(agent.model.parameters()).add_(.2)
    agent.load_checkpoint_state(saved)
    torch.testing.assert_close(next(agent.model.parameters()),before,rtol=0,atol=0)
    assert agent.gamma == agent.cost_discount == 1.
    identity=saved['checkpoint_identity']
    assert identity['actual_gamma_reward'] == identity['actual_gamma_cost'] == 1.
    bad=copy.deepcopy(saved);bad['checkpoint_identity']['actual_gamma_cost']=.99
    with pytest.raises(ValueError,match='identity'):
        agent.validate_checkpoint_state(bad)
    bad=copy.deepcopy(saved);bad.pop('optimizer')
    with pytest.raises((KeyError,ValueError)):
        agent.load_checkpoint_state(bad)


def test_qcpo_optimizer_still_takes_eight_shared_steps_without_dca_budget():
    agent=small_agent();agent.model=torch.nn.Linear(2,1)
    agent.optimizer=torch.optim.Adam(agent.model.parameters(),lr=.0001)
    agent.epochs=8;agent.minibatches=1;agent.update_counter=0;agent.clip_grad_norm=1e4
    calls=[]
    agent._optimizer_epoch_minibatches=lambda:(torch.arange(2),)
    def loss(fixed,indices):
        calls.append(indices.clone())
        return agent.model(torch.ones(2,2)).square().mean(),{}
    agent._loss=loss
    agent._optimizer_epoch_diagnostics=lambda *args:{}
    agent._optimize_fixed_rollout({})
    assert agent.update_counter == len(calls) == 8
    assert len(agent.optimizer.state) == len(list(agent.model.parameters()))


def test_qcpo_transaction_keeps_first_rms_only_batch_and_raw_dual_before_optimizer():
    agent=small_agent(T=1000,B=25);rollout=late_cost_rollout()
    agent._capture_episode_metrics(rollout,'reward')
    calls=[];fixed=dict(obs=torch.zeros(100,250,3),probability_ratio=torch.ones(100,250,8))
    agent.update_counter=agent.rms_only_batch_count=0
    agent._fixed_rollout_targets=lambda data:calls.append('fixed') or fixed
    raw_update_dual=agent.update_dual
    def dual(values):
        calls.append('dual')
        torch.testing.assert_close(values,rollout['objective_cost'])
        assert not torch.equal(values,rollout['report_discounted_cost'])
        return raw_update_dual(values)
    agent.update_dual=dual
    agent.model=SimpleNamespace(obs_rms=SimpleNamespace(count=torch.tensor(0.)))
    def rms(obs):
        calls.append('rms');assert obs is fixed['obs']
        agent.model.obs_rms.count+=25000
    agent.model.update_obs_rms=rms
    def optimize(value):
        calls.append('optimize');assert value is fixed
        agent.update_counter+=8
        return {}
    agent._optimize_fixed_rollout=optimize
    agent._log_core=lambda *args,**kwargs:args[-1]
    agent.update_rollout(0,rollout)
    assert calls==['fixed','dual','rms']
    assert agent.rms_only_batch_count==1 and agent.update_counter==0 and agent.dual_update_count==1
    calls.clear()
    agent.update_rollout(1,rollout)
    assert calls==['fixed','dual','rms','optimize']
    assert agent.rms_only_batch_count==1 and agent.update_counter==8 and agent.dual_update_count==2


def test_qcpo_logs_raw_constraint_with_true_discounted_report_values():
    agent=small_agent(T=1000,B=25);rollout=late_cost_rollout()
    agent.main_env_steps=agent.actual_env_steps=25000;agent.initialization_env_steps=0
    agent.last_dual_cost_quantile=16.
    agent._capture_episode_metrics(rollout,'reward')
    log=agent._log_core(0,rollout['objective_return'].numpy(),rollout['report_discounted_cost'].numpy(),rollout['objective_cost'].numpy())
    assert log['quantile/constraint_prob']==log['constraint/empirical_failure_probability']==1.
    assert log['objective/cost_violation_probability']==log['undisc_cost/violation_probability']==1.
    assert log['disc_cost/violation_probability']==0.
    assert log['objective/reward_mean']==log['undisc_reward/aver_reward']==2.
    assert log['disc_reward/aver_reward']==pytest.approx(2*.99**900)
    assert log['quantile/q_est']==16.
