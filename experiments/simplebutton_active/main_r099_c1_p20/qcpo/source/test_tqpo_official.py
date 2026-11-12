"""把GPU接口与冻结上游可执行公式比较；不以训练收敛作单测。"""
import ast
from collections import deque
from pathlib import Path
from types import SimpleNamespace as NS
import copy
import json
import unittest
import numpy as np
import torch
from tqpo_agent import OfficialActor,TQPO,PROTOCOL,PreviousInputsVec,chunks,gae,order_statistic
from rlpyt.distributions.gaussian import DistInfoStd,Gaussian

ROOT=Path(__file__).resolve().parent

def upstream_class():
    """仅移除未使用的 sklearn/plot imports，保留原类全部可执行代码。"""
    path=ROOT/'official_reference/tqpo/tqpo.py';tree=ast.parse(path.read_text())
    tree.body=[node for node in tree.body if not(isinstance(node,(ast.Import,ast.ImportFrom)) and
        (getattr(node,'module','') or '').startswith(('sklearn','scipy')))]
    # matplotlib is present locally; this does not alter any class method.
    namespace={'__name__':'official_tqpo_numerical_reference'};exec(compile(tree,str(path),'exec'),namespace)
    return namespace['tqpo']

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1);cls.Upstream=upstream_class()
    def ref(self):
        ref=self.Upstream.__new__(self.Upstream)
        for k,v in dict(new_T=100,new_B=120,cost_scale=10.,cost_discount=.99,cost_gae_lambda=.97,
            reward_scale=1.,gae_lambda=.97,discount=.99,mid_batch_reset=True,normalize_advantage=False,
            target_outage_prob=.2,cost_limit=1.5,ep_cost_ema_alpha=0.,ep_cost_eqa_alpha=0.,ep_outage_ema_alpha=0.,
            _ep_cost_ema=1.5,_ep_cost_eqa=1.5,_ep_outage_ema=.2).items():setattr(ref,k,v)
        ref.agent=NS(recurrent=True);ref.current_ep_costs=deque(maxlen=100);ref.current_step_costs=deque(maxlen=100000)
        return ref
    def test_process_returns_matches_original_12000_and_boundary(self):
        torch.manual_seed(81);agent=TQPO(OfficialActor(3));t,b=1000,12
        values=torch.randn(t,b,2);cost=torch.randint(0,2,(t,b)).float();reward=torch.randn(t,b)
        done=torch.zeros(t,b,dtype=torch.bool);done[-1]=True
        rollout=dict(R=reward,C=cost,done=done,values=values,bootstrap_values=torch.randn(b,2),
            S=torch.randn(t,b,8),A=torch.randn(t,b,2),mean=torch.randn(t,b,2),ls=torch.randn(t,b,2),
            h=torch.randn(t,b,512),c=torch.randn(t,b,512))
        processed=agent.process(rollout);ref=self.ref()
        samples=NS(env=NS(reward=reward.clone(),done=done.clone(),env_info=NS(cost=cost.clone(),cum_cost=cost.cumsum(0))),
            agent=NS(agent_info=NS(value=(values[...,0],values[...,1])),bootstrap_value=(rollout['bootstrap_values'][:,0][None],rollout['bootstrap_values'][:,1][None])))
        rr,adv,cr,valid,q=ref.process_returns(0,samples)
        for actual,expected in ((processed['r_return'],rr),(processed['r_adv'],adv),(processed['c_return'],cr),(processed['valid'],valid)):
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertEqual(agent.q_episode,float(q));self.assertEqual(agent.q_step,order_statistic([float(v) for v in ref.current_step_costs],step=True))
        # Terminal GAE is immediate cost minus V even with arbitrary terminal bootstrap.
        torch.testing.assert_close(processed['c_return'][-1,9::10],cost[-1]/10,rtol=0,atol=3e-7)
    def test_model_is_original_and_loss_gradient_matches_upstream(self):
        torch.manual_seed(23);actor=OfficialActor(3);agent=TQPO(actor);agent.lambda_dual=3.2;agent.q_step=.7
        T,B=3,2;S=torch.randn(T,B,8);processed=dict(S=S,A=torch.randn(T,B,2),old_mean=torch.randn(T,B,2),old_ls=torch.randn(T,B,2)*.1,
            r_adv=torch.randn(T,B),r_return=torch.randn(T,B),c_return=torch.tensor([[.2,.8],[1.,.7],[.4,1.2]]),valid=torch.tensor([[1.,1.],[1.,1.],[0.,1.]]),rec=actor.initial_state(B))
        port_loss,_=agent.loss(processed);port_grad=torch.autograd.grad(port_loss,tuple(actor.parameters()))
        ref=self.ref();ref.q_quantile_step=agent.q_step;ref.cost_penalty=agent.lambda_dual
        for key,value in dict(ratio_clip=.1,value_loss_coeff=1.,entropy_loss_coeff=0.,cost_value_loss_coeff=.5,sum_norm=True).items():setattr(ref,key,value)
        class Agent:
            distribution=Gaussian(2)
            def __call__(self,obs,pa,pr,rec):
                mu,ls,value,nrec=actor.model(obs,pa,pr,rec);return DistInfoStd(mu,ls),value,nrec
        ref.agent=Agent()
        from rlpyt.projects.tqpo.tqpo_model import RnnState
        loss,*_=ref.loss((S[...,:4],S[...,4:6],S[...,-1]),processed['A'],processed['r_return'],processed['r_adv'],processed['valid'],
            DistInfoStd(processed['old_mean'],processed['old_ls']),processed['c_return'],RnnState(*(x.transpose(0,1) for x in processed['rec'])))
        ref_grad=torch.autograd.grad(loss,tuple(actor.parameters()))
        torch.testing.assert_close(port_loss,loss,rtol=1e-6,atol=1e-7)
        for a,b in zip(port_grad,ref_grad):torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-7)
    def test_original_model_initialization_and_outputs_exact(self):
        from rlpyt.projects.tqpo.tqpo_model import tqpoModel
        torch.manual_seed(77);port=OfficialActor(3)
        torch.manual_seed(77);ref=tqpoModel((4,),2,25,[512,512],512,lstm_skip=True,constraint=True,normalize_observation=True,var_clip=1e-6)
        for key,x in ref.state_dict().items():torch.testing.assert_close(x,port.model.state_dict()[key],rtol=0,atol=0)
        S=torch.randn(2,3,7);a=port.full(S);b=ref(S[...,:4],S[...,4:6],S[...,-1])
        for x,y in zip(a[:2],b[:2]):torch.testing.assert_close(x,y,rtol=0,atol=0)
    def test_previous_cost_lag_and_train_eval_terminal_inputs(self):
        from dataclasses import dataclass
        @dataclass
        class Step:
            observation: torch.Tensor
            final_observation: torch.Tensor
            reward: torch.Tensor
            cost: torch.Tensor
            episode_end: torch.Tensor
        class Vector:
            act_dim=2;device=torch.device('cpu')
            def __init__(self):self.i=0
            def reset(self):return torch.zeros(1,3)
            def step(self,action):
                self.i+=1
                return Step(torch.ones(1,3)*self.i,torch.ones(1,3)*99,
                    torch.tensor([-8.]),torch.tensor([3.*self.i]),torch.tensor([self.i==3]))
        for training in (False,True):
            env=PreviousInputsVec(Vector(),training=training);env.reset();a=torch.tensor([[.2,-.4]])
            one=env.step(a);two=env.step(a);three=env.step(a)
            self.assertEqual(float(one.observation[0,3]),0.)
            self.assertEqual(float(two.observation[0,3]),3.)
            self.assertEqual(float(three.observation[0,3]),0.)
            self.assertEqual(float(three.final_observation[0,3]),6.)
            reset=env.reset()
            torch.testing.assert_close(reset[0,4:6],a[0] if training else torch.zeros(2))
            self.assertEqual(float(reset[0,-1]),-8. if training else 0.)
    def test_dual_matches_original_first_batch_update(self):
        torch.manual_seed(9);agent=TQPO(OfficialActor(3));t,b=1000,12
        R=torch.randn(t,b);C=torch.zeros(t,b);C[0]=torch.arange(b)*5
        done=torch.zeros(t,b,dtype=torch.bool);done[-1]=True;values=torch.randn(t,b,2)
        rollout=dict(R=R,C=C,done=done,values=values,bootstrap_values=torch.randn(b,2),
            S=torch.randn(t,b,7),A=torch.randn(t,b,2),mean=torch.randn(t,b,2),ls=torch.zeros(t,b,2),
            h=torch.zeros(t,b,512),c=torch.zeros(t,b,512))
        agent.process(rollout);ref=self.ref()
        ref.current_q_quantile_lag=deque(maxlen=100);ref.q_buffer_len=100
        ref.delta=.1;ref.step_cost_limit_itr=None;ref.pid_Ki=.1;ref.pid_i=ref.cost_penalty=.5
        ref.agent=NS(recurrent=True,device='cpu',update_obs_rms=lambda x:None)
        samples=NS(env=NS(observation=rollout['S'][...,:4],prev_reward=rollout['S'][...,-1],reward=R.clone(),done=done.clone(),
            env_info=NS(cost=C.clone(),cum_cost=C.cumsum(0))),agent=NS(prev_action=rollout['S'][...,4:6],action=rollout['A'],
            agent_info=NS(dist_info=DistInfoStd(rollout['mean'],rollout['ls']),value=(values[...,0],values[...,1])),
            bootstrap_value=(rollout['bootstrap_values'][:,0][None],rollout['bootstrap_values'][:,1][None])))
        ref.optimize_agent(0,samples)
        self.assertAlmostEqual(agent.lambda_dual,ref.cost_penalty,places=12)
        self.assertEqual(agent.q_step,ref.q_quantile_step)
        self.assertEqual(agent.last_tilt,ref.tilt_pos)
    def test_guard_and_no_dca_parameters(self):
        history=deque(maxlen=100)
        for k in range(1000):history.append(.1)
        self.assertFalse(len(history)>100)
        self.assertTrue(PROTOCOL['sum_norm']);self.assertFalse(PROTOCOL['normalize_advantage']);self.assertEqual(PROTOCOL['gamma'],.99)
        self.assertEqual(PROTOCOL['batch_size'],12000);self.assertEqual(PROTOCOL['lambda_init'],.5)
    def test_checkpoint_identity_and_untouched_prior_data(self):
        a=TQPO(OfficialActor(3));state=a.state_dict();b=TQPO(OfficialActor(3));b.load_state_dict(state)
        altered=copy.deepcopy(state);altered['tqpo_identity']['gamma']=1.
        with self.assertRaises(ValueError):b.load_state_dict(altered)
        altered=copy.deepcopy(state);altered['actual_env_steps']=1
        with self.assertRaises(ValueError):b.load_state_dict(altered)

if __name__=='__main__':unittest.main()
