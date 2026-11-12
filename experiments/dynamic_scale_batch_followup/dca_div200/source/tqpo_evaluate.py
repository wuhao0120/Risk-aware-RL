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
import base_runner as base
base.setup()
import numpy as np
import torch
from agents.common import DistributionalCritic
from utils.model import RecurrentGaussianActor
from dca_evaluate import features, make_env, quantile_metrics, four_episode_returns, sha

ROOT=Path(__file__).resolve().parent
TIMES=(0,20,100,300,500,800)


def module_digest(model):
    """Hash parameters plus buffers so accidental observation-RMS updates fail."""
    digest=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        value=value.detach().cpu().contiguous()
        digest.update(key.encode());digest.update(str(value.dtype).encode());digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def evaluation_sources():
    files=[ROOT/'dca_evaluate.py',ROOT/'tqpo_evaluate.py',ROOT/'base_runner.py']
    files+=sorted((ROOT/'source').rglob('*.py'))
    for name in ('tqpo_agent.py','run_tqpo.py','official_tqpo_manifest.json'):
        if (ROOT/name).exists():files.append(ROOT/name)
    files+=sorted((ROOT/'official_reference').rglob('*.py'))
    return {str(path.relative_to(ROOT)):sha(path) for path in files}




def episode_metrics(samples):
    """Episode dispersion is recorded separately from across-training-seed bands."""
    result={key:dict(mean=float(samples[key].mean()),episode_sd=float(samples[key].std(ddof=1)))
        for key in ('undiscounted_reward','discounted_reward','undiscounted_cost','discounted_cost')}
    events=samples['undiscounted_cost']>15.
    result['violation_probability']=float(events.mean())
    result['violation_count']=int(events.sum());result['episodes']=len(events)
    return result


def summarize_calibration(data):
    """One realized suffix per history: report bias/Brier, not conditional RMSE."""
    records=[]
    for index,t in enumerate(TIMES):
        valid=data['calibration_valid'][:,index]
        z=data['atoms'][valid,index];g=data['suffix'][valid,index];prefix=data['prefix'][valid,index]
        if not len(g):continue
        probability=(prefix[:,None]+z>15.).mean(-1);event=prefix+g>15.
        records.append(dict(time=t,n=len(g),**quantile_metrics(z,g),
            predicted_probability=float(probability.mean()),empirical_probability=float(event.mean()),
            probability_bias=float((probability-event).mean()),brier=float(((probability-event)**2).mean()),
            predicted_zero_fraction=float((probability==0).mean()),
            empirical_mass_at_threshold=float((np.abs(prefix+g-15.)<1e-8).mean())))
    return dict(records=records,interpretation='Independent marginal diagnostics; one continuation per history, not conditional probability RMSE or action-contrast ground truth')


@torch.no_grad()
def sample_checkpoint(actor,critic,cfg,out,episodes,seed,tqpo=False,calibration=False):
    """Complete stochastic policy episodes; common layout/noise streams across arms.

The TQPO adapter zeros previous inputs at evaluation episode starts, as the
official evaluation collector does. Its training gamma/normalization stay intact.
"""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    device=next(actor.parameters()).device;batch=min(64,episodes)
    before=module_digest(actor);critic_before=module_digest(critic) if critic is not None else None
    if tqpo:
        from run_tqpo import make_eval_env
        ref,env=make_eval_env(batch,seed,cfg)
    else:ref,env=make_env(batch,seed,cfg)
    rng=torch.Generator(device=device).manual_seed(seed+9000001)
    accumulated={key:[] for key in ('undiscounted_reward','discounted_reward','undiscounted_cost','discounted_cost')}
    if calibration:accumulated.update({key:[] for key in ('atoms','suffix','prefix','calibration_valid')})
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
                    calibration_valid=torch.stack(vs,1),suffix=suffix)
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
    """Use recorded actual interaction counts; never relabel TQPO 12k batch boundaries."""
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
    seed=int(config['signature']['seed']);tqpo=True
    from tqpo_agent import IDENTITY
    if config['signature']['experiment'] != IDENTITY: raise ValueError('Expected final TQPO checkpoint')
    if not cli.all_checkpoints: entries=entries[-1:]
    source=evaluation_sources();started=time.time();rows=[]
    try:
        for index,entry in enumerate(entries):
            checkpoint=run/entry['path'];dest=out/str(entry['actual_env_steps'])
            episodes=64 if cli.smoke else (2048 if index==len(entries)-1 else 1024)
            sampling_seed=240000001+seed*100003+entry['nominal_env_steps']//500000*1009
            signature=dict(checkpoint_sha256=sha(checkpoint),actual_env_steps=entry['actual_env_steps'],
                nominal_env_steps=entry['nominal_env_steps'],seed=sampling_seed,episodes=episodes,
                report_gamma=.99,event='undiscounted_environment_cost > 15',source_sha256=source,
                evaluation_batch=64,smoke=cli.smoke,tqpo_official_inputs=tqpo)
            metric_path=dest/'metrics.json'
            if metric_path.exists():
                result=json.loads(metric_path.read_text())
                if result['signature']!=signature or result['samples_sha256']!=sha(dest/'samples.npz'):
                    raise ValueError('evaluation resume provenance mismatch')
            else:
                from run_tqpo import load_evaluation
                actor,cfg,state=load_evaluation(run,checkpoint,torch.device('cuda:0'));critic=None
                if int(state['actual_env_steps'])!=entry['actual_env_steps']:raise ValueError('checkpoint actual-step identity mismatch')
                result=sample_checkpoint(actor,critic,cfg,dest,episodes,sampling_seed,
                    tqpo=tqpo,calibration=(not tqpo and index==len(entries)-1))
                result.update(signature=signature,training_seed=seed,arm='tqpo' if tqpo else config['signature']['arm'])
                base.atomic_json(metric_path,result);del actor,critic,state
            rows.append(dict(actual_env_steps=entry['actual_env_steps'],nominal_env_steps=entry['nominal_env_steps'],
                checkpoint=entry['path'],metrics=result['metrics'],actor_sha256=result['actor_sha256']))
            base.atomic_json(out/'status.json',dict(status='evaluating',completed_checkpoints=len(rows),planned=len(entries),updated_at=time.time()))
            print(f'{run.name} eval steps={entry["actual_env_steps"]} R={result["metrics"]["undiscounted_reward"]["mean"]:.4f} p={result["metrics"]["violation_probability"]:.5f}',flush=True)
        if evaluation_sources()!=source:raise RuntimeError('evaluation source changed during run')
        base.atomic_json(out/'curve.json',dict(arm='tqpo' if tqpo else config['signature']['arm'],seed=seed,
            checkpoints=rows,smoke=cli.smoke,source_sha256=source,episodes_per_intermediate_checkpoint=64 if cli.smoke else 1024,
            episodes_final_checkpoint=64 if cli.smoke else 2048,
            constraint='P(undiscounted environment cost > 15) <= 0.2',
            reporting='gamma=.99 returns are also saved; bands are computed across training seeds, not episodes'))
        base.atomic_json(out/'status.json',dict(status='complete',checkpoints=len(rows),elapsed_seconds=time.time()-started,updated_at=time.time()))
    except Exception:
        base.atomic_json(out/'status.json',dict(status='failed',error=traceback.format_exc(),updated_at=time.time()));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--all-checkpoints',action='store_true',help='Evaluate the full saved checkpoint grid; default final only')
    main(parser.parse_args())
