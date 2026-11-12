"""Independent checkpoint learning curves; training RNG and statistics stay frozen.

Each checkpoint is evaluated in its own newly seeded environment. Five training
seeds, not episodes, are the units used for the final mean +/- sample SD band.
Raw cost >15 defines the common evaluation event; gamma=.99 reports are extra.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from objective import REWARD_GAMMA, COST, EVENT
import base_runner as base
base.setup()
import numpy as np
import torch
from agents.common import DistributionalCritic
from utils.model import RecurrentGaussianActor
from scipy.stats import beta, wasserstein_distance
from envs import SafetyEnv,make_vec_env
from schemeb_math import sorted_quantile_cdf,upper_cost_violation_probability

ROOT=Path(__file__).resolve().parent
TIMES=(0,20,100,300,500,800)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def critic_dimensions(weights):
    """从checkpoint的线性层恢复网络宽度，防止D128被D32默认值误读。"""
    keys=sorted((k for k in weights if k.startswith('net.') and k.endswith('.weight')),
                key=lambda k:int(k.split('.')[1]))
    if not keys:raise ValueError('checkpoint has no critic linear layers')
    widths=[tuple(weights[k].shape) for k in keys]
    if any(len(shape)!=2 for shape in widths):raise ValueError('invalid critic weights')
    if any(widths[i][1]!=widths[i-1][0] for i in range(1,len(widths))):
        raise ValueError('critic hidden dimensions do not chain')
    return widths[0][1]-2,[shape[0] for shape in widths[:-1]],widths[-1][0]


def features(obs,rec,t,prefix=None,q=None):
    """训练输入: raw s、剩余时间、pre-h/c；全局 cost prefix 仅用于 CDF。"""
    return torch.cat((obs,obs.new_full((len(obs),1),(1000-t)/1000),rec[0][0],rec[1][0]),-1)


def make_env(batch,seed,cfg):
    ref=SafetyEnv('SimpleButton',seed=seed,action_mode='explicit_box_clip',dynamics_profile=cfg['dynamics_profile'])
    env=make_vec_env('SimpleButton',batch,1000,torch.device('cuda',torch.cuda.current_device()),ref_env=ref,seed=seed,
                     backend='mjwarp',dynamics_profile=cfg['dynamics_profile'],action_mode='explicit_box_clip')
    return ref,env


def quantile_metrics(z,g):
    """每条独立return与同条件atoms配对；CRPS不是CDF概率RMSE。"""
    z,g=np.asarray(z,float),np.asarray(g,float);n=z.shape[-1]
    tau=(np.arange(n)+.5)/n
    lower=(g[:,None]<z).mean(0);upper=(g[:,None]<=z).mean(0)
    interval_error=np.maximum(lower-tau,0)+np.maximum(tau-upper,0)
    crps=np.abs(z-g[:,None]).mean(-1)-(np.sort(z,axis=-1)*(2*np.arange(1,n+1)-n-1)).sum(-1)/n**2
    return dict(crps=float(crps.mean()),coverage_mae=float(np.abs(upper-tau).mean()),
        coverage_mae_tie_sensitive=float(np.abs(upper-tau).mean()),
        aggregate_quantile_interval_violation=float(interval_error.mean()),
        coverage_note='coverage_mae is tie-sensitive and not a primary quality metric; aggregate interval uses P(G<q)<=tau<=P(G<=q), pooled across distinct histories with one sample each.',
        crossing=float((z[:,1:]<z[:,:-1]).mean()),empirical_mean=float(g.mean()),predicted_mean=float(z.mean()),
        marginal_w1=float(wasserstein_distance(g,z.ravel())))


def four_episode_returns(rewards,costs,report_gamma=.99):
    """[B,T] 轨迹：raw sum 是目标；gamma=.99 的配对统计仅用于报告。"""
    if rewards.shape!=costs.shape or rewards.ndim!=2:
        raise ValueError('paired reward/cost trajectories must have shape [B,T]')
    factors=report_gamma**torch.arange(rewards.shape[1],device=rewards.device,dtype=torch.float64)
    return dict(discounted_reward=(rewards.double()*factors).sum(1),
        undiscounted_reward=rewards.double().sum(1),
        discounted_cost=(costs.double()*factors).sum(1),undiscounted_cost=costs.double().sum(1))




def module_digest(model):
    """Hash parameters plus buffers so accidental observation-RMS updates fail."""
    digest=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        value=value.detach().cpu().contiguous()
        digest.update(key.encode());digest.update(str(value.dtype).encode());digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def evaluation_sources():
    files=[ROOT/'dca_evaluate.py',ROOT/'schemeb_math.py',ROOT/'base_runner.py',ROOT/'agent.py']
    files+=sorted((ROOT/'source').rglob('*.py'))
    return {str(path.relative_to(ROOT)):sha(path) for path in files}


def load_dca(run,checkpoint,device):
    """Strict identity restore of inference networks, without constructing training envs."""
    from agent import IDENTITY,ARMS,ARM_DIVISORS
    meta=json.loads((run/'config.json').read_text());cfg=dict(meta['configuration']);signature=meta['signature']
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);state=saved['agent']
    if saved.get('experiment_signature')!=signature or signature.get('experiment')!=IDENTITY:
        raise ValueError('DCA checkpoint/signature identity mismatch')
    if signature.get('arm') not in ARMS:raise ValueError('unexpected DCA factorial arm')
    canonical=meta['identity']
    observed=json.loads(json.dumps({key:state.get(key) for key in canonical}))
    if canonical!=observed:raise ValueError('DCA canonical checkpoint identity mismatch')
    expected_divisor=ARM_DIVISORS[signature['arm']]
    if cfg['gamma']!=REWARD_GAMMA or cfg['report_gamma']!=.99:
        raise ValueError('Scheme B objectives must be undiscounted; .99 is report-only')
    if float(cfg.get('mean_divisor',signature.get('mean_divisor',0.)))!=expected_divisor:
        raise ValueError('Scheme B actor mean divisor does not match arm')
    a=state['actor']
    actor=RecurrentGaussianActor(a['obs_rms.mean'].numel(),2,hidden=cfg['actor_hidden'],
        lstm_size=cfg['lstm_size'],init_std=cfg['init_std'],learn_std=cfg['learn_std'],
        normalize_observation=cfg['normalize_observation']).to(device)
    actor.load_state_dict(a);actor.eval().requires_grad_(False)
    weights=state['reward_critic'];dim,hidden,n=critic_dimensions(weights)
    if n!=128:raise ValueError('DCA critic must have 128 quantiles')
    critic=DistributionalCritic(dim,2,n,hidden).to(device)
    critic.load_state_dict(weights);critic.eval().requires_grad_(False)
    cfg.update(evaluation_algorithm='dca',evaluation_arm=signature['arm'])
    return actor,critic,cfg,state


def episode_metrics(samples):
    """Episode dispersion is recorded separately from across-training-seed bands."""
    result={key:dict(mean=float(samples[key].mean()),episode_sd=float(samples[key].std(ddof=1)))
        for key in ('undiscounted_reward','discounted_reward','undiscounted_cost','discounted_cost')}
    events=samples['undiscounted_cost']>15.
    result['violation_probability']=float(events.mean())
    result['discounted_violation_probability']=float((samples['discounted_cost']>15.).mean())
    count,n=int(events.sum()),len(events)
    result['violation_count']=count;result['episodes']=n
    result['violation_probability_cp95']=[0. if count==0 else float(beta.ppf(.025,count,n-count+1)),
        1. if count==n else float(beta.ppf(.975,count+1,n-count))]
    return result


def summarize_calibration(data):
    """One realized suffix per history: report bias/Brier, not conditional RMSE."""
    records=[]
    for index,t in enumerate(TIMES):
        valid=data['calibration_valid'][:,index]
        z=data['atoms'][valid,index];g=data['suffix'][valid,index];prefix=data['prefix'][valid,index]
        if not len(g):continue
        probability=upper_cost_violation_probability(torch.as_tensor(z,dtype=torch.float64),
            torch.as_tensor(15.-prefix,dtype=torch.float64)).numpy();event=prefix+g>15.
        records.append(dict(time=t,n=len(g),**quantile_metrics(z,g),
            predicted_probability=float(probability.mean()),empirical_probability=float(event.mean()),
            empirical_undiscounted_probability=float(event.mean()),
            empirical_discounted_probability=float((data['discounted_cost'][valid]>15.).mean()),
            probability_reader='sorted midpoint quantile interpolation, undiscounted upper cost tail',
            probability_bias=float((probability-event).mean()),brier=float(((probability-event)**2).mean()),
            predicted_zero_fraction=float((probability==0).mean()),
            empirical_mass_at_threshold=float((prefix+g==15.).mean())))
    return dict(records=records,predicted_random_variable='undiscounted remaining cost',
        primary_feasibility_event='undiscounted whole-episode cost >15',
        objective_gamma_reward=1.,objective_gamma_cost=1.,dual_gamma=1.,interpretation='Independent marginal diagnostics; one continuation per history, not conditional probability RMSE or action-contrast ground truth')


@torch.no_grad()
def sample_checkpoint(actor,critic,cfg,out,episodes,seed,calibration=True):
    """Complete frozen-policy episodes with separate evaluation RNG and raw-cost labels."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    device=next(actor.parameters()).device;batch=min(64,episodes)
    before=module_digest(actor);critic_before=module_digest(critic) if critic is not None else None
    ref,env=make_env(batch,seed,cfg)
    rng=torch.Generator(device=device).manual_seed(seed+9000001)
    accumulated={key:[] for key in ('undiscounted_reward','discounted_reward','undiscounted_cost','discounted_cost')}
    if calibration:accumulated.update({key:[] for key in ('atoms','suffix','prefix','raw_prefix','raw_suffix','calibration_valid')})
    try:
        for offset in range(0,episodes,batch):
            obs=env.reset();recurrent=actor.initial_state(batch,device)
            active=torch.ones(batch,device=device,dtype=torch.bool)
            rewards=torch.zeros(batch,1000,device=device,dtype=torch.float64);costs=torch.zeros_like(rewards)
            prefix=torch.zeros(batch,device=device,dtype=torch.float64)
            zs,ps,vs=[],[],[]
            for t in range(1000):
                mean,log_std,next_recurrent=actor.step(obs,recurrent)
                raw=mean+log_std.exp()*torch.randn(mean.shape,generator=rng,device=device,dtype=mean.dtype)
                if calibration and t in TIMES:
                    zs.append(critic(features(obs,recurrent,t),raw.clamp(-1,1)))
                    ps.append(prefix.clone());vs.append(active.clone())
                step=env.step(raw)
                rewards[:,t]=torch.where(active,step.reward,0).double()
                costs[:,t]=torch.where(active,step.cost,0).double();prefix+=costs[:,t]
                active &= ~(step.terminated|step.truncated)
                recurrent=(next_recurrent[0]*active[None,:,None],next_recurrent[1]*active[None,:,None])
                obs=step.observation
            if active.any():raise RuntimeError('incomplete finite-horizon evaluation episode')
            values=four_episode_returns(rewards,costs,.99)
            if calibration:
                suffix=torch.flip(torch.cumsum(torch.flip(costs,[1]),dim=1),[1])[:,TIMES]
                values.update(atoms=torch.stack(zs,1),prefix=torch.stack(ps,1),
                    raw_prefix=torch.stack(ps,1),raw_suffix=suffix,calibration_valid=torch.stack(vs,1),suffix=suffix)
                if not torch.allclose(values['prefix']+suffix,prefix[:,None],atol=1e-8,rtol=1e-10):
                    raise ValueError('cost prefix plus suffix must equal whole-episode raw cost')
            take=min(batch,episodes-offset)
            for key,value in values.items():accumulated[key].append(value[:take].cpu().numpy())
            base.atomic_json(out/'progress.json',dict(completed_episodes=offset+take,episodes=episodes))
    finally:env.close();ref.close()
    if before!=module_digest(actor) or (critic is not None and critic_before!=module_digest(critic)):
        raise RuntimeError('evaluation changed policy, critic or observation moments')
    data={key:np.concatenate(values) for key,values in accumulated.items()}
    if any(not np.isfinite(value).all() for value in data.values()):raise ValueError('nonfinite independent evaluation')
    path=out/'samples.npz';temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:np.savez_compressed(stream,**data)
    temporary.replace(path)
    return dict(metrics=episode_metrics(data),calibration=summarize_calibration(data) if calibration else None,
        actor_sha256=before,critic_sha256=critic_before,samples_sha256=sha(path))


def checkpoint_entries(run,smoke=False):
    """Verify every predeclared training checkpoint before selecting independent evaluation."""
    index=json.loads((Path(run)/'checkpoint_index.json').read_text())
    entries=index['checkpoints'] if isinstance(index,dict) else index
    if not entries or entries[0]['actual_env_steps']!=0:raise ValueError('missing pretraining evaluation checkpoint')
    actual=[row['actual_env_steps'] for row in entries];nominal=[row['nominal_env_steps'] for row in entries]
    if actual!=sorted(set(actual)) or nominal!=sorted(set(nominal)):raise ValueError('checkpoint grid must be strictly increasing')
    if not smoke and nominal!=list(range(0,10000001,500000)):raise ValueError('formal curve requires every predeclared checkpoint')
    if not all((Path(run)/row['path']).is_file() for row in entries):raise ValueError('missing checkpoint file')
    return entries


def main(cli):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    run=cli.run_dir.resolve();out=cli.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    entries=checkpoint_entries(run,cli.smoke);config=json.loads((run/'config.json').read_text())
    seed=int(config['signature']['seed'])
    if not cli.all_checkpoints:entries=entries[-1:]
    source=evaluation_sources();started=time.time();rows=[]
    try:
        for index,entry in enumerate(entries):
            checkpoint=run/entry['path'];dest=out/str(entry['actual_env_steps'])
            episodes=64 if cli.smoke else (2048 if index==len(entries)-1 else 1024)
            sampling_seed=240000001+seed*100003+entry['nominal_env_steps']//500000*1009
            signature=dict(checkpoint_sha256=sha(checkpoint),actual_env_steps=entry['actual_env_steps'],
                nominal_env_steps=entry['nominal_env_steps'],seed=sampling_seed,episodes=episodes,
                report_gamma=.99,event='undiscounted_environment_cost > 15',source_sha256=source,
                evaluation_batch=64,smoke=cli.smoke,
                cdf_reader='sorted midpoint interpolation, double budget, strict upper cost',
                objective_gamma_reward=1.,objective_gamma_cost=1.,dual_gamma=1.)
            metric_path=dest/'metrics.json'
            if metric_path.exists():
                result=json.loads(metric_path.read_text())
                if result['signature']!=signature or result['samples_sha256']!=sha(dest/'samples.npz'):
                    raise ValueError('evaluation resume provenance mismatch')
            else:
                actor,critic,cfg,state=load_dca(run,checkpoint,torch.device('cuda:0'))
                if int(state['actual_env_steps'])!=entry['actual_env_steps']:raise ValueError('checkpoint actual-step identity mismatch')
                result=sample_checkpoint(actor,critic,cfg,dest,episodes,sampling_seed,
                    calibration=(index==len(entries)-1))
                result.update(signature=signature,training_seed=seed,arm=config['signature']['arm'])
                base.atomic_json(metric_path,result);del actor,critic,state
            rows.append(dict(actual_env_steps=entry['actual_env_steps'],nominal_env_steps=entry['nominal_env_steps'],
                checkpoint=entry['path'],metrics=result['metrics'],actor_sha256=result['actor_sha256']))
            base.atomic_json(out/'status.json',dict(status='evaluating',completed_checkpoints=len(rows),planned=len(entries),updated_at=time.time()))
            print(f'{run.name} eval steps={entry["actual_env_steps"]} R={result["metrics"]["undiscounted_reward"]["mean"]:.4f} p={result["metrics"]["violation_probability"]:.5f}',flush=True)
        if evaluation_sources()!=source:raise RuntimeError('evaluation source changed during run')
        base.atomic_json(out/'curve.json',dict(arm=config['signature']['arm'],seed=seed,
            checkpoints=rows,smoke=cli.smoke,all_checkpoints=cli.all_checkpoints,source_sha256=source,episodes_per_intermediate_checkpoint=64 if cli.smoke else 1024,
            episodes_final_checkpoint=64 if cli.smoke else 2048,
            constraint='P(undiscounted environment cost > 15) <= 0.2',
            reporting='gamma=.99 returns are also saved; bands are computed across training seeds, not episodes'))
        base.atomic_json(out/'status.json',dict(status='complete',checkpoints=len(rows),elapsed_seconds=time.time()-started,updated_at=time.time()))
    except Exception:
        base.atomic_json(out/'status.json',dict(status='failed',error=traceback.format_exc(),updated_at=time.time()));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--all-checkpoints',action='store_true',help='Optional independent curve; default final checkpoint only')
    main(parser.parse_args())
