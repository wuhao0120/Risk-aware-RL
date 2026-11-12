"""原版 QCPO 冻结策略验收：无折扣目标，gamma=.99 仅用于配对报告。

保留官方 collector 的延迟 cost 坐标，以及 episode reset 之间的终步 action/reward
输入。独立指与训练分离的 held-out RNG 流；跨 reset 的历史输入遵循原算法。
不查询或比较 QCPO 的分布头，所有性能统计来自实际环境轨迹。
"""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
from objective import COST_GAMMA, REWARD_GAMMA, COST, EVENT
import base_runner as base
base.setup()
import numpy as np
import torch
from agents.qcpo_ref_author_aligned_model import QcpoRefAuthorAlignedModel, RnnState
from envs import SafetyEnv,make_vec_env
from evaluate import sha,four_episode_returns

EXPECTED_IDENTITY={
    'actual_gamma_reward':REWARD_GAMMA,'actual_gamma_cost':COST_GAMMA,'actual_dual_gamma':1.,'report_gamma':.99,
    'actual_num_quantiles':25,'replay_enabled':False,'reward_scale':1.,
    'td3_enabled':False,'mc_eta':0.,'actual_optimizer_epochs':8,'actual_minibatches':1,
    'raw_cost_threshold':15.,'raw_cost_violation_budget':.2,
    'raw_cost_dual':'undiscounted_environment_cost','raw_cost_violation_event':EVENT,
    'actual_cost_scale':10.,'actual_first_batch':'dual_and_rms_only','dca_enhancements_applied':False,
}


def load(run,checkpoint,device='cuda:0'):
    """恢复共享 model 的推断参数和 RMS，严格拒绝 DCA 增强被带入 QCPO。"""
    config=json.loads((run/'config.json').read_text());cfg=config['configuration']
    cp=torch.load(checkpoint,map_location='cpu',weights_only=False);state=cp['agent']
    if cp.get('experiment_signature')!=config['signature']:
        raise ValueError('QCPO checkpoint/config scientific signature mismatch')
    identity=state['checkpoint_identity']
    mismatches={k:(identity.get(k),v) for k,v in EXPECTED_IDENTITY.items() if identity.get(k)!=v}
    if mismatches:raise ValueError(f'QCPO baseline identity mismatch: {mismatches}')
    if float(cfg['gamma'])!=REWARD_GAMMA or float(cfg['report_gamma'])!=.99:
        raise ValueError('QCPO objective gamma must be 1; report gamma must be .99')
    weights=state['model'];obs_dim=weights['obs_rms.mean'].numel()
    model=QcpoRefAuthorAlignedModel(observation_shape=(obs_dim,),action_size=2,n_quantile=25,
        hidden_sizes=(512,512),lstm_size=512,lstm_skip=True,constraint=True,init_log_std=0.,
        normalize_observation=True,var_clip=1e-6,location_epsilon=1e-6,raw_cost_support=True).to(device)
    model.load_state_dict(weights);model.eval().requires_grad_(False)
    return model,cfg,state


def advance_cost_history(stored_cost,current_cost,ended):
    """复刻 wrapper 顺序：先向下一观测暴露旧 stored cost，再保存当前 cost。"""
    mask=ended.unsqueeze(-1)
    observation_cost=torch.where(mask,torch.zeros_like(stored_cost),stored_cost)
    next_stored=torch.where(mask,torch.zeros_like(stored_cost),current_cost.unsqueeze(-1))
    return observation_cost,next_stored


@torch.no_grad()
def policy_rollouts(model,cfg,out,episodes,seed):
    """Exact-N held-out episodes with frozen normalization and official reset history."""
    if (out/'metrics.json').exists():return
    out.mkdir(parents=True,exist_ok=True)
    batch=min(64,episodes);device=next(model.parameters()).device
    profile=cfg['dynamics_profile'];action_mode=cfg.get('action_execution_mode','author_raw_env_clamp')
    if action_mode!='author_raw_env_clamp':raise ValueError('QCPO action execution mode changed')
    ref=SafetyEnv('Dynamic',seed=seed,action_mode=action_mode,dynamics_profile=profile)
    env=make_vec_env('Dynamic',batch,1000,device,ref_env=ref,seed=seed,
        backend='mjwarp',action_mode=action_mode,dynamics_profile=profile)
    values={name:[] for name in ['discounted_reward','undiscounted_reward','discounted_cost','undiscounted_cost']}
    torch.manual_seed(seed+9000001)
    try:
        obs=env.reset();obs_cost=torch.zeros(batch,1,device=device);stored=torch.zeros_like(obs_cost)
        previous_action=torch.zeros(batch,2,device=device);previous_reward=torch.zeros(batch,device=device)
        zero=torch.zeros(1,batch,512,device=device);rec=RnnState(zero,zero.clone())
        for offset in range(0,episodes,batch):
            rewards=torch.zeros(batch,1000,device=device,dtype=torch.float64);costs=torch.zeros_like(rewards)
            active=torch.ones(batch,dtype=torch.bool,device=device)
            for t in range(1000):
                mean,log_std,_,next_rec=model(torch.cat((obs,obs_cost),-1),previous_action,previous_reward,rec)
                action=mean+log_std.exp()*torch.randn_like(mean);step=env.step(action)
                ended=step.episode_end & active
                if t<999 and bool(ended.any()):raise ValueError('unexpected early QCPO episode boundary')
                reward=step.reward*active;cost=step.cost*active
                rewards[:,t]=reward.double();costs[:,t]=cost.double()
                obs_cost,stored=advance_cost_history(stored,step.cost,ended)
                previous_action,previous_reward,obs=action,reward,step.observation
                active &= ~ended;keep=active.to(mean.dtype).view(1,batch,1)
                rec=RnnState(next_rec.h*keep,next_rec.c*keep)
            if bool(active.any()):raise ValueError('unfinished QCPO evaluation episode')
            totals=four_episode_returns(rewards,costs,float(cfg['report_gamma']))
            take=min(batch,episodes-offset)
            for key,value in totals.items():values[key].append(value[:take].cpu().numpy())
            # Official sampler retains terminal action/reward while clearing recurrent state/cost history.
            if offset + batch < episodes:
                obs=env.reset();obs_cost.zero_();stored.zero_();previous_action.zero_();previous_reward.zero_()
    finally:env.close();ref.close()
    data={key:np.concatenate(items) for key,items in values.items()}
    np.savez_compressed(out/'samples.npz',**data)
    metrics={key:dict(mean=float(value.mean()),std=float(value.std())) for key,value in data.items()}
    for key in ('discounted_cost','undiscounted_cost'):
        metrics[key]['violation_probability']=float(np.mean(data[key]>15.))
    base.atomic_json(out/'metrics.json',dict(seed=seed,episodes=episodes,episode_metrics=metrics,
        event='undiscounted_environment_cost > 15',gamma=1.,report_gamma=.99,threshold=15.,
        evaluation='held-out stochastic policy rollouts; no QCPO distribution calibration',
        reset_history='official: terminal action/reward carried; h/c and cost history reset',
        reporting_note='gamma .99 reward/cost are descriptive only; objective and event use raw sums'))


def main(cli):
    torch.set_num_threads(1);torch.cuda.set_device(torch.device(cli.device))
    run=cli.run_dir.resolve();out=(cli.out_dir or run/'evaluation').resolve();out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parent;config=json.loads((run/'config.json').read_text())
    dependencies=[root/name for name in ('qcpo_evaluate.py','evaluate.py','base_runner.py')]+sorted((root/'source').rglob('*.py'))
    source={str(path.relative_to(root)):sha(path) for path in dependencies}
    identity=out/'source_identity.json'
    if identity.exists() and json.loads(identity.read_text())!=source:raise ValueError('QCPO evaluation source drift')
    base.atomic_json(identity,source)
    checkpoints=[run/'latest.pt'] if cli.smoke else [run/f'checkpoint_{step}.pt' for step in ((1000000,5000000,10000000) if cli.all_checkpoints else (10000000,))]
    try:
        for checkpoint in checkpoints:
            digest=sha(checkpoint);model,cfg,state=load(run,checkpoint)
            step=int(state['actual_env_steps']);dest=out/str(step)
            provenance=dict(checkpoint_sha256=digest,actual_env_steps=step,scientific_signature=config['signature'])
            saved=dest/'checkpoint_identity.json'
            if saved.exists() and json.loads(saved.read_text())!=provenance:raise ValueError('mixed QCPO evaluation checkpoint')
            base.atomic_json(saved,provenance)
            base.atomic_json(out/'status.json',dict(status='policy_evaluation',checkpoint=step,pid=os.getpid()))
            policy_rollouts(model,cfg,dest/'marginal',32 if cli.smoke else 512,830001+int(cfg['seed'])*10000)
            if not all(torch.equal(value.cpu(),state['model'][key]) for key,value in model.state_dict().items()):
                raise ValueError('QCPO frozen model or normalization changed during evaluation')
            if sha(checkpoint)!=digest:raise ValueError('QCPO checkpoint changed during evaluation')
            base.atomic_json(dest/'receipt.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=digest,
                model_and_rms_unchanged=True,actual_env_steps=step,backend='mjwarp',stochastic_policy=True,
                action_execution_mode='author_raw_env_clamp',gamma=1.,report_gamma=.99,threshold=15.,
                event='undiscounted_environment_cost > 15',smoke=cli.smoke,evaluation_source=source,
                scientific_signature=config['signature'],baseline_identity=state['checkpoint_identity'],
                distribution_calibration_performed=False))
            del model,state;torch.cuda.empty_cache()
        base.atomic_json(out/'status.json',dict(status='complete',smoke=cli.smoke))
    except Exception:
        base.atomic_json(out/'status.json',dict(status='failed',error=traceback.format_exc()));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--out-dir',type=Path)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--all-checkpoints',action='store_true')
    main(parser.parse_args())
