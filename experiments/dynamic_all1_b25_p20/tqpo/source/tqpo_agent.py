"""官方 TQPO 的 raw-cost GPU 运行适配，原模型和算法超参数保持冻结。

官方源码在 official_reference；模型直接执行官方 tqpoModel，优化器将
官方公式改为 GPU 张量接口。不得引入 DCA KL/replay/PI 或 mean/100。
"""
from __future__ import annotations
from collections import deque
from dataclasses import replace
import copy
import math
from pathlib import Path
import sys
import numpy as np
import torch
from objective import episode_total
from torch import nn
from objective import COST_GAMMA, REWARD_GAMMA, COST, EVENT
import base_runner as base
base.setup()
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'official_reference'))
from rlpyt.projects.tqpo.tqpo_model import tqpoModel
from rlpyt.distributions.gaussian import Gaussian,DistInfoStd

UPSTREAM='bdcd03fe1144ecf2018ec889ea9aad4eb5a9dc89'
RLPYT_COMMIT='f04f23db1eb7b5915d88401fca67869968a07a37'
IDENTITY = 'dynamic_all1_b25_p20_tqpo_pilot_v1'
PROTOCOL=dict(implementation=IDENTITY,upstream_commit=UPSTREAM,rlpyt_commit=RLPYT_COMMIT,
    environment='Dynamic',dynamics_profile='point_velocity_implicitfast_v1',
    cost_threshold=15.,failure_alpha=.2,gamma=REWARD_GAMMA,cost_gamma=COST_GAMMA,dual_gamma=1.,report_gamma=.99,
    num_envs=25,horizon=1000,batch_size=25000,recurrent_length=100,recurrent_batch=250,
    actor_hidden=[512,512],lstm_size=512,epochs=8,minibatches=1,learning_rate=0.0001,
    adam_eps=1e-8,ratio_clip=.1,gae_lambda=.97,cost_gae_lambda=.97,clip_grad_norm=1e4,
    entropy_loss_coeff=0.,value_loss_coeff=1.,cost_value_loss_coeff=.5,
    normalize_advantage=False,cost_scale=10.,tail_coefficient=.1,sum_norm=True,
    lambda_init=.5,dual_lr=.1,tilt_delta=.1,dual_gap_upper_clip=10.,
    episode_window=100,step_window=100000,q_history_window=100,cdf_guard='len_ge_100_after_history_warmup',
    dual_constraint='undiscounted_environment_cost_order_statistic',
    action_execution_mode='raw_gaussian_environment_box_clip',
    rms_order='after_labels_before_PPO_first_batch_RMS_only',
    sampling_arrangement='25_parallel_1000_step_episodes_to_preserve_25000_cells_and_T100_B250',
    training_previous_inputs='terminal_action_reward_carried_zero_hc_reset_cost',
    evaluation_previous_inputs='zero_action_reward_cost_and_hc_at_episode_start')

class OfficialActor(nn.Module):
    """直接复用原版模型；包装输入为 obs, lagged cost, prev raw action/reward。"""
    def __init__(self,raw_state_dim,action_dim=2):
        super().__init__();self.raw_state_dim=int(raw_state_dim);self.action_dim=int(action_dim);self.lstm_size=512
        self.model=tqpoModel((self.raw_state_dim+1,),self.action_dim,n_quantile=25,
            hidden_sizes=[512,512],lstm_size=512,lstm_skip=True,constraint=True,
            normalize_observation=True,var_clip=1e-6)
    def initial_state(self,batch_size,device=None):
        p=next(self.parameters());device=p.device if device is None else device
        return tuple(torch.zeros(1,batch_size,512,device=device) for _ in range(2))
    def full(self,states,rec=None):
        n=self.raw_state_dim+1
        return self.model(states[...,:n],states[...,n:n+self.action_dim],states[...,-1],rec)
    def forward(self,states,initial_state=None):
        mu,ls,_,rec=self.full(states,initial_state);return mu,ls,tuple(rec)
    def step(self,states,rec=None):return self.forward(states,rec)
    @torch.no_grad()
    def update_obs_rms(self,states):self.model.update_obs_rms(states[...,:self.raw_state_dim+1])

class PreviousInputsVec:
    """官方 cost 输入滞后一拍；训练跨episode保留末步action/reward，评估清零。"""
    def __init__(self,vector,training=False):self.vector=vector;self.training=bool(training)
    def __getattr__(self,key):return getattr(self.vector,key)
    @staticmethod
    def pack(obs,action,reward,cost):return torch.cat((obs,cost.unsqueeze(-1),action,reward.unsqueeze(-1)),dim=-1)
    def reset(self,*args,**kwargs):
        obs=self.vector.reset(*args,**kwargs)
        if not self.training or not hasattr(self,'previous_action'):
            self.previous_action=obs.new_zeros(len(obs),self.vector.act_dim);self.previous_reward=obs.new_zeros(len(obs))
        self.previous_cost=obs.new_zeros(len(obs))
        return self.pack(obs,self.previous_action,self.previous_reward,self.previous_cost)
    def step(self,action):
        step=self.vector.step(action);ended=step.episode_end
        final=self.pack(step.final_observation,action,step.reward,self.previous_cost)
        pa=action.clone();pr=step.reward.clone();lag=self.previous_cost.clone()
        lag[ended]=0
        if not self.training:pa[ended]=0;pr[ended]=0
        observation=self.pack(step.observation,pa,pr,lag)
        self.previous_action=pa.detach().clone();self.previous_reward=pr.detach().clone()
        self.previous_cost=torch.where(ended,torch.zeros_like(step.cost),step.cost).detach().clone()
        return replace(step,observation=observation,final_observation=final)
    def input_state(self):
        return {key:getattr(self,key).detach().cpu().clone() for key in ('previous_action','previous_reward','previous_cost')} if hasattr(self,'previous_action') else None
    def restore_input_state(self,state):
        if state is not None:
            for key,value in state.items():setattr(self,key,value.to(self.vector.device).clone())


def chunks(tensor,length=100):
    """严格复制官方 transform_sample 的 B-major 排列，而非任意分块排列。"""
    t,b=tensor.shape[:2]
    if t%length:raise ValueError('episode length must be divisible by recurrent length')
    return tensor.transpose(0,1).reshape(b*(t//length),length,*tensor.shape[2:]).transpose(0,1).contiguous()


def gae(reward,value,done,bootstrap,gamma=.99,trace=.97):
    """官方 GAE 逐 T100 chunk 回推；块末用采样时旧 value bootstrap。"""
    nd=1-done.to(reward.dtype);adv=torch.zeros_like(reward)
    adv[-1]=reward[-1]+gamma*bootstrap*nd[-1]-value[-1]
    for t in range(len(reward)-2,-1,-1):
        delta=reward[t]+gamma*value[t+1]*nd[t]-value[t]
        adv[t]=delta+gamma*trace*nd[t]*adv[t+1]
    return adv,adv+value


def order_statistic(values,level=.8,step=False):
    values=np.asarray(values,dtype=np.float64)
    index=int(np.floor(len(values)*level))-int(step)
    return float(np.sort(values)[index])


class TQPO:
    """25000条transition/批，首批只更新RMS/dual，其余每批8次共享Adam。"""
    def __init__(self,actor):
        self.actor=actor;self.device=next(actor.parameters()).device
        self.optimizer=torch.optim.Adam(actor.parameters(),lr=0.0001,eps=1e-8)
        self.dist=Gaussian(actor.action_dim)
        self.lambda_dual=.5;self.q_step=0.;self.q_episode=1.5;self.last_cdf=0.;self.last_tilt=.1/1.1
        self.current_ep_costs=deque(maxlen=100);self.current_step_costs=deque(maxlen=100000);self.q_history=deque(maxlen=100)
        self.dual_update_count=0;self.optimizer_epoch_count=0;self.actual_env_steps=0
    @torch.no_grad()
    def process(self,rollout):
        reward,cost,done=chunks(rollout['R']),chunks(rollout['C'])/10.,chunks(rollout['done'])
        oldvalues=rollout['values'];next_values=torch.zeros_like(oldvalues);next_values[:-1]=oldvalues[1:]
        next_values[-1]=rollout['bootstrap_values']
        values=chunks(oldvalues);bv=chunks(next_values)[-1]
        adv,rr=gae(reward,values[...,0],done,bv[...,0],gamma=REWARD_GAMMA)
        _,cr=gae(cost,values[...,1],done,bv[...,1],gamma=COST_GAMMA)
        # CPU np quantile exactly follows original order-statistic, not interpolation.
        self.current_step_costs.extend(cr.detach().double().cpu().reshape(-1).tolist())
        self.q_step=order_statistic(self.current_step_costs,step=True)
        totals=episode_total(rollout['C'], 1.).cpu().tolist();self.current_ep_costs.extend(totals)
        self.q_episode=order_statistic(self.current_ep_costs)/10.;self.q_history.append(self.q_episode)
        self.last_cdf=float(np.mean(np.asarray(self.q_history)<1.5)) if len(self.q_history)>=100 else 0.
        gap=min(10.,self.q_episode-1.5)
        self.last_tilt=((self.last_cdf if gap>0 else 1-self.last_cdf)+.1)/1.1
        self.lambda_dual=max(0.,self.lambda_dual+.1*self.last_tilt*gap);self.dual_update_count+=1
        valid=torch.ones_like(done,dtype=torch.float32)
        valid[1:]=1-torch.cumsum(done[:-1].float(),0).clamp(max=1)
        return dict(S=chunks(rollout['S']),A=chunks(rollout['A']),old_mean=chunks(rollout['mean']),old_ls=chunks(rollout['ls']),
            r_adv=adv.detach(),r_return=rr.detach(),c_return=cr.detach(),valid=valid,
            rec=tuple(chunks(rollout[key])[0].unsqueeze(0) for key in ('h','c')))
    def loss(self,processed):
        mu,ls,value,_=self.actor.full(processed['S'],processed['rec'])
        ratio=self.dist.likelihood_ratio(processed['A'],DistInfoStd(processed['old_mean'],processed['old_ls']),DistInfoStd(mu,ls))
        clipped=ratio.clamp(.9,1.1);valid=processed['valid'];den=valid.sum()
        r_adv=processed['r_adv'];c_adv=-(processed['c_return']>=self.q_step).float()/PROTOCOL["failure_alpha"]*.1
        r_loss=-(torch.minimum(ratio*r_adv,clipped*r_adv)*valid).sum()/den
        c_loss=-(torch.minimum(ratio*c_adv,clipped*c_adv)*valid).sum()/den
        p_loss=(r_loss+self.lambda_dual*c_loss)/(1+self.lambda_dual)
        rv=.5*((value.r_value-processed['r_return']).square()*valid).sum()/den
        cv=.5*((value.c_value-processed['c_return']).square()*valid).sum()/den
        loss=p_loss+rv+.5*cv
        return loss,dict(policy_loss=float(p_loss.detach()),reward_policy_loss=float(r_loss.detach()),cost_policy_loss=float(c_loss.detach()),
            reward_value_loss=float(rv.detach()),cost_value_loss=float(cv.detach()))
    def update(self,rollout):
        processed=self.process(rollout);self.actor.update_obs_rms(processed['S']);info={}
        if self.dual_update_count>1:
            for _ in range(8):
                # Official M1 still shuffles B; preserve the RNG draw and ordering.
                permutation=torch.as_tensor(np.random.permutation(250),device=self.device)
                shuffled={k:(tuple(x[:,permutation].contiguous() for x in v) if k=='rec' else v[:,permutation]) for k,v in processed.items()}
                self.optimizer.zero_grad(set_to_none=True);loss,info=self.loss(shuffled)
                if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite official TQPO loss')
                loss.backward();norm=nn.utils.clip_grad_norm_(self.actor.parameters(),1e4,error_if_nonfinite=True)
                self.optimizer.step();self.optimizer_epoch_count+=1;info['gradient_norm']=float(norm)
        self.actual_env_steps+=25000
        return dict(info,lambda_dual=self.lambda_dual,q_step=self.q_step,q_episode=self.q_episode,
            history_cdf=self.last_cdf,tilt=self.last_tilt,dual_update_count=self.dual_update_count,
            optimizer_epoch_count=self.optimizer_epoch_count,actual_env_steps=self.actual_env_steps)
    def state_dict(self):
        return dict(actor=self.actor.state_dict(),optimizer=self.optimizer.state_dict(),tqpo_identity=copy.deepcopy(PROTOCOL),
            actual_env_steps=self.actual_env_steps,raw_state_dim=self.actor.raw_state_dim,
            **{key:getattr(self,key) for key in ('lambda_dual','q_step','q_episode','last_cdf','last_tilt','dual_update_count','optimizer_epoch_count')},
            **{key:list(getattr(self,key)) for key in ('current_ep_costs','current_step_costs','q_history')})
    def load_state_dict(self,state):
        if state['tqpo_identity']!=PROTOCOL:raise ValueError('official TQPO checkpoint identity mismatch')
        if state['actual_env_steps']!=25000*state['dual_update_count'] or state['optimizer_epoch_count']!=max(0,state['dual_update_count']-1)*8:raise ValueError('TQPO clocks inconsistent')
        self.actor.load_state_dict(state['actor']);self.optimizer.load_state_dict(state['optimizer'])
        for key in ('lambda_dual','q_step','q_episode','last_cdf','last_tilt','dual_update_count','optimizer_epoch_count','actual_env_steps'):setattr(self,key,state[key])
        for key in ('current_ep_costs','current_step_costs','q_history'):getattr(self,key).clear();getattr(self,key).extend(state[key])

@torch.no_grad()
def collect(agent,env):
    """采集25条H1000完整轨迹，保存采样时模型values与分块初始h/c。"""
    actor=agent.actor;obs=env.reset();rec=actor.initial_state(25,agent.device)
    data={key:[] for key in ('S','A','R','C','done','mean','ls','values','h','c')}
    for t in range(1000):
        mu,ls,value,nrec=actor.full(obs,rec);raw=mu+ls.exp()*torch.randn_like(mu)
        step=env.step(raw);done=step.terminated|step.truncated
        # The task uses complete fixed-horizon episodes; do not silently change sampler budget.
        if (t<999 and bool(done.any())) or (t==999 and not bool(done.all())):raise RuntimeError('TQPO fixed-H sampler encountered unexpected episode length')
        values=dict(S=obs,A=raw,R=step.reward,C=step.cost,done=done,mean=mu,ls=ls,
            values=torch.stack((value.r_value,value.c_value),-1),h=rec[0][0],c=rec[1][0])
        for key,value in values.items():data[key].append(value.clone())
        obs=step.observation;rec=tuple(x*(~done)[None,:,None] for x in nrec)
    _,_,value,_=actor.full(obs,rec)
    result={key:torch.stack(value) for key,value in data.items()}
    result['bootstrap_values']=torch.stack((value.r_value,value.c_value),-1)
    return result
