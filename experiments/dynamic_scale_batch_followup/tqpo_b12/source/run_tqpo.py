"""冻结官方设置的 TQPO：12000/批，834批≈10M，独立checkpoint供共同评估。"""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback
import base_runner as base
base.setup()
os.environ['WANDB_MODE']='disabled'
import numpy as np
import torch
from envs import SafetyEnv,make_vec_env
from utils.checkpoint import _rng_state,_restore_rng_state
from tqpo_agent import OfficialActor,PreviousInputsVec,TQPO,collect,IDENTITY,UPSTREAM,PROTOCOL
ROOT=Path(__file__).resolve().parent
atomic=base.atomic_json


def hashes():
    files=[ROOT/name for name in ('run_tqpo.py','tqpo_agent.py','base_runner.py','official_tqpo_manifest.json')]
    files+=sorted((ROOT/'official_reference').rglob('*.py'))+sorted((ROOT/'source').rglob('*.py'))
    files+=sorted((ROOT/'configs').glob('*.json'))+[ROOT/'objective.py']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def make_eval_env(batch,seed,cfg):
    """独立评估使用原版 CpuEvalCollector 的 episode 开始零previous inputs。"""
    device=torch.device(cfg.get('evaluation_device','cuda:0'))
    ref=SafetyEnv('Dynamic',seed=seed,action_mode='explicit_box_clip',dynamics_profile=PROTOCOL['dynamics_profile'])
    env=make_vec_env('Dynamic',batch,1000,device,ref_env=ref,seed=seed,backend='mjwarp',
        dynamics_profile=PROTOCOL['dynamics_profile'],action_mode='explicit_box_clip')
    return ref,PreviousInputsVec(env,training=False)


def load_evaluation(run_dir,checkpoint,device='cuda:0'):
    """只恢复推断模型；校验完整signature、原始算法身份和实际采样时钟。"""
    run_dir=Path(run_dir);config=json.loads((run_dir/'config.json').read_text())
    cp=torch.load(checkpoint,map_location='cpu',weights_only=False);state=cp['agent']
    if cp.get('experiment_signature')!=config['signature']:raise ValueError('TQPO checkpoint/config signature mismatch')
    if config['signature']['experiment']!=IDENTITY or state['tqpo_identity']!=PROTOCOL:raise ValueError('TQPO raw-cost identity mismatch')
    if state['actual_env_steps']!=state['dual_update_count']*12000:raise ValueError('TQPO actual step mismatch')
    actor=OfficialActor(state['raw_state_dim']).to(device);actor.load_state_dict(state['actor']);actor.eval().requires_grad_(False)
    cfg=dict(config['configuration'],evaluation_device=str(device))
    return actor,cfg,state


def train(cli):
    output=cli.output.resolve();output.mkdir(parents=True,exist_ok=True)
    lock=(output/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    batches=2 if cli.smoke else cli.batches;planned=batches*12000
    if not 2<=batches<=834:raise ValueError("invalid batch budget")
    signature=dict(experiment=IDENTITY,algorithm='TQPO_TILT_GUARD_FIXED',arm='tqpo',seed=cli.seed,
        main_batches=batches,smoke=bool(cli.smoke),nominal_env_steps=planned,
        actual_planned_env_steps=planned,source=hashes(),official_protocol=PROTOCOL)
    if (output/'config.json').exists() and json.loads((output/'config.json').read_text())['signature']!=signature:raise ValueError('TQPO source/configuration drift')
    atomic(output/'config.json',dict(configuration=dict(PROTOCOL,seed=cli.seed,algorithm='TQPO_TILT_GUARD_FIXED'),signature=signature,planned_env_steps=planned))
    started=time.time();agent=env=ref=None;rows=[];episodes=[];index=[];stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    def status(name,**extra):atomic(output/'status.json',dict(status=name,pid=os.getpid(),updated_at=time.time(),elapsed_seconds=time.time()-started,
        actual_env_steps=0 if agent is None else agent.actual_env_steps,**extra))
    def save(path):
        payload=dict(agent=agent.state_dict(),experiment_signature=signature,rng_state=_rng_state(),
            vec_env_rng_state=env.checkpoint_rng_state(),previous_inputs=env.input_state(),rows=rows,episodes=episodes,index=index)
        temp=path.with_name('.'+path.name+'.tmp');torch.save(payload,temp);os.replace(temp,path)
    def save_point(nominal):
        item=dict(path=f'checkpoint_{agent.actual_env_steps}.pt',nominal_env_steps=nominal,actual_env_steps=agent.actual_env_steps)
        if not any(x['path']==item['path'] for x in index):index.append(item)
        save(output/item['path']);atomic(output/'checkpoint_index.json',dict(checkpoints=index))
    try:
        torch.set_num_threads(1);base.seed_all(cli.seed);status('constructing')
        ref=SafetyEnv('Dynamic',seed=cli.seed,action_mode='explicit_box_clip',dynamics_profile=PROTOCOL['dynamics_profile'])
        vector=make_vec_env('Dynamic',12,1000,torch.device('cuda:0'),ref_env=ref,seed=cli.seed,backend='mjwarp',
            dynamics_profile=PROTOCOL['dynamics_profile'],action_mode='explicit_box_clip')
        env=PreviousInputsVec(vector,training=True)
        actor=OfficialActor(int(ref.observation_space.shape[0])).cuda();agent=TQPO(actor)
        if (output/'latest.pt').exists():
            cp=torch.load(output/'latest.pt',map_location='cuda:0',weights_only=False)
            if cp['experiment_signature']!=signature:raise ValueError('TQPO resume source mismatch')
            agent.load_state_dict(cp['agent']);env.restore_rng_state(cp['vec_env_rng_state']);env.restore_input_state(cp['previous_inputs']);_restore_rng_state(cp['rng_state'])
            rows=cp['rows'];episodes=cp['episodes'];index=cp['index']
        for path,values in (('metrics.jsonl',rows),('episodes.jsonl',episodes)):(output/path).write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in values))
        atomic(output/'cuda_ready.json',dict(pid=os.getpid(),time=time.time()))
        if not index:save_point(0)
        for k in range(agent.dual_update_count,batches):
            tick=time.time();rollout=collect(agent,env);metrics=agent.update(rollout)
            discount=.99**torch.arange(1000,device='cuda',dtype=torch.float64)
            R,C=rollout['R'].double(),rollout['C'].double()
            samples=dict(undiscounted_reward=R.sum(0).cpu().tolist(),undiscounted_cost=C.sum(0).cpu().tolist(),
                discounted_reward=(R*discount[:,None]).sum(0).cpu().tolist(),discounted_cost=(C*discount[:,None]).sum(0).cpu().tolist())
            row=dict(iteration=k,phase='rms_only' if k==0 else 'main',actual_env_steps=agent.actual_env_steps,
                wall_time=time.time(),batch_seconds=time.time()-tick,**{name:float(np.mean(v)) for name,v in samples.items()},
                probability=float(np.mean(np.asarray(samples['undiscounted_cost'])>15)),**{name:value for name,value in metrics.items() if name!='actual_env_steps'})
            rows.append(row);episodes.append(dict(actual_env_steps=agent.actual_env_steps,**samples));del rollout
            nominal=(agent.actual_env_steps//500000)*500000
            if cli.smoke:save_point(agent.actual_env_steps)
            elif nominal>index[-1]['nominal_env_steps']:save_point(min(nominal,10000000))
            save(output/'latest.pt')
            for path,value in (('metrics.jsonl',row),('episodes.jsonl',episodes[-1])):
                with (output/path).open('a') as stream:stream.write(json.dumps(value,allow_nan=False)+'\n')
            status('training',main_batches=agent.dual_update_count)
            print(f'TQPO/s{cli.seed} steps={agent.actual_env_steps} R={row["undiscounted_reward"]:.3f} C={row["undiscounted_cost"]:.3f} p={row["probability"]:.3f} lambda={agent.lambda_dual:.3f} seconds={time.time()-tick:.2f}',flush=True)
            if stopped:status('paused');return
        save(output/'final.pt')
        # Concrete model+Adam+dual/history roundtrip with all authoritative clocks.
        cp=torch.load(output/'final.pt',map_location='cuda:0',weights_only=False)
        clone=TQPO(OfficialActor(actor.raw_state_dim).cuda());clone.load_state_dict(cp['agent'])
        for key,tensor in actor.state_dict().items():torch.testing.assert_close(tensor,clone.actor.state_dict()[key],rtol=0,atol=0)
        if clone.state_dict()['current_step_costs']!=agent.state_dict()['current_step_costs']:raise AssertionError('history roundtrip failed')
        if len(clone.optimizer.state)!=len(agent.optimizer.state):raise AssertionError('optimizer roundtrip failed')
        atomic(output/'training_receipt.json',dict(actual_env_steps=agent.actual_env_steps,nominal_env_steps=signature['nominal_env_steps'],
            dual_update_count=agent.dual_update_count,optimizer_epoch_count=agent.optimizer_epoch_count,
            checkpoint_roundtrip=True,history_cdf=agent.last_cdf,cdf_guard_preserved=False,cdf_guard_fixed=True,tqpo_identity=PROTOCOL,
            upstream_commit=UPSTREAM,signature=signature,training_seconds=time.time()-started,
            no_dca_modifications=True,official_source_manifest='official_tqpo_manifest.json'))
        status('trained')
    except Exception:status('failed',error=traceback.format_exc());raise
    finally:
        if env is not None:
            with contextlib.suppress(Exception):env.close()
        if ref is not None:
            with contextlib.suppress(Exception):ref.close()
        lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--batches',type=int,default=834);train(p.parse_args())
