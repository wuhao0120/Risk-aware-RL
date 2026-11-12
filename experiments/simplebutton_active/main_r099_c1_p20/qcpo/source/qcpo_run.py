"""Unmodified-budget QCPO comparator: gamma1 objectives, paired raw/.99 reports."""
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
from objective import COST_GAMMA, REWARD_GAMMA, COST, EVENT
import base_runner as base
base.setup()
os.environ['WANDB_MODE']='disabled'
import numpy as np
import torch
from run_experiment import base_args
from agents.four_env_fullbatch_formal import QREF_FULL_ALGO
from qcpo_agent import QCPOUndiscountedAgent,IDENTITY
from envs import SafetyEnv
from utils.checkpoint import save_checkpoint,load_checkpoint

ROOT=Path(__file__).resolve().parent
atomic=base.atomic_json


def hashes():
    """Only QCPO training dependencies define the scientific source signature."""
    paths=[ROOT/name for name in ('qcpo_run.py','qcpo_agent.py','raw_cost_agents.py','base_runner.py')]
    paths+=sorted((ROOT/'source').rglob('*.py'))
    paths+=sorted((ROOT/'configs').glob('*.json'))+[ROOT/'objective.py']
    return {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def make_args(seed,output=None,device='cuda:0'):
    output=Path(output or ROOT/'runs'/f'qcpo_seed{seed}')
    args=base_args(QREF_FULL_ALGO,int(seed),torch.device(device),'SimpleButton')
    args.algo_name='QCPO_MATCHED_COST';args.return_threshold=None
    args.cost_threshold=15.;args.cost_quantile_level=args.q_alpha=.8
    args.failure_alpha=args.target_outage_prob=.2
    args.training_gamma=REWARD_GAMMA
    args.primal_constraint_variable=COST;args.dual_constraint_variable="undiscounted_environment_cost"
    args.violation_probability_variable=EVENT
    args.base_factor_metadata_is_provenance=True
    args.env_id='SafetyPointSimpleButton0-v0';args.vec_backend='mjwarp'
    args.dynamics_profile='point_velocity_implicitfast_v1'
    args.defer_wandb_init=True;args.show_progress=args.show_eval_progress=False
    args.log_interval=1;args.wandb_mode='disabled';args.wandb_project='SimpleButton_full_gpu'  # Required constructor value; internal W&B is disabled.
    args.wandb_name=args.wandb_id=args.wandb_lineage=output.name
    args.wandb_group='matched-qcpo'
    args.wandb_dir=str(output/'wandb');args.progress_file=str(output/'progress.json')
    return args


def train(cli):
    if not 1<=cli.main_batches<=834:raise ValueError('QCPO main batches must be1..400')
    output=cli.output.resolve();output.mkdir(parents=True,exist_ok=True)
    lock=(output/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    args=make_args(cli.seed,output)
    signature=dict(experiment=IDENTITY,algorithm='QCPO',seed=cli.seed,main_batches=cli.main_batches,
        source=hashes(),gamma_reward=REWARD_GAMMA,gamma_cost=COST_GAMMA,dual_gamma=1.,report_gamma=.99,threshold=15.,epsilon=.2,
        num_quantiles=25,replay_enabled=False,reward_scale=1.,td3_enabled=False,mc_eta=0.,
        optimizer_epochs=8,minibatches=1,cost_scale=10.,first_batch='dual_and_rms_only')
    if (output/'config.json').exists() and json.loads((output/'config.json').read_text())['signature']!=signature:
        raise ValueError('QCPO scientific source/configuration drift')
    agent=env=None;rows=[];episode_rows=[];last=-1;stopped=False;started=time.time()
    def stop(*_):
        nonlocal stopped
        stopped=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    def status(state,**extra):
        atomic(output/'status.json',dict(status=state,pid=os.getpid(),updated_at=time.time(),
            elapsed_seconds=time.time()-started,actual_env_steps=int(agent.actual_env_steps) if agent else 0,**extra))
    def save(path):
        save_checkpoint(path,agent,last,runner_state=dict(rows=rows,episodes=episode_rows))
    try:
        torch.set_num_threads(1);base.seed_all(cli.seed);status('constructing')
        env=SafetyEnv(args.env_id,seed=cli.seed,action_mode=args.action_execution_mode,dynamics_profile=args.dynamics_profile)
        agent=QCPOUndiscountedAgent(args,env);agent.checkpoint_signature=signature
        agent.num_iterations=cli.main_batches
        if agent.gamma!=REWARD_GAMMA or agent.cost_discount!=COST_GAMMA or agent.n_quantile!=25:
            raise ValueError('QCPO objective/architecture protocol mismatch')
        config=base.serial_config(args)
        config.update(wandb_project='ICLR_rebuttal_SimpleButton',wandb_mode='online',wandb_owner='campaign.py')
        config.update(gamma=REWARD_GAMMA,cost_discount=COST_GAMMA,cost_gamma=COST_GAMMA,report_gamma=.99,
            n_quantile=25,num_quantiles=25,reward_scale=1.,replay_enabled=False,td3_enabled=False,
            mc_eta=0.,reward_actor_outer_discount='none',return_threshold=15.,cost_threshold=15.)
        atomic(output/'config.json',dict(configuration=config,signature=signature,
            planned_env_steps=cli.main_batches*12000,identity=agent._checkpoint_identity()))
        if (output/'latest.pt').exists():
            payload=load_checkpoint(output/'latest.pt',agent,map_location=agent.device)
            last=int(payload['iteration']);rows=payload['runner_state']['rows'];episode_rows=payload['runner_state']['episodes']
        for name,values in [('metrics.jsonl',rows),('episodes.jsonl',episode_rows)]:
            (output/name).write_text(''.join(json.dumps(value,allow_nan=False)+'\n' for value in values))
        atomic(output/'cuda_ready.json',dict(pid=os.getpid(),time=time.time()))
        if last<0 and not (output/'latest.pt').exists():save(output/'latest.pt')
        for iteration in range(last+1,cli.main_batches):
            start=time.time();rollout=agent._rollout_author_aligned()
            metrics=agent.update_rollout(iteration,rollout)
            last=iteration
            row=dict(metrics,iteration=iteration,phase='main',actual_env_steps=int(agent.actual_env_steps),
                wall_time=time.time(),batch_seconds=time.time()-start)
            samples=dict(agent._episode_samples)
            episode=dict(phase='main',actual_env_steps=int(agent.actual_env_steps),**samples)
            if not all(np.isfinite(value) for value in row.values() if isinstance(value,(int,float))):
                raise FloatingPointError('nonfinite QCPO training metric')
            rows.append(row);episode_rows.append(episode);del rollout
            save(output/'latest.pt')
            if agent.actual_env_steps in (120000,5004000,10008000):
                save(output/f'checkpoint_{agent.actual_env_steps}.pt')
            for name,value in [('metrics.jsonl',row),('episodes.jsonl',episode)]:
                with (output/name).open('a') as stream:stream.write(json.dumps(value,allow_nan=False)+'\n')
            status('training',main_batches=iteration+1)
            print(f'QCPO/s{cli.seed} steps={agent.actual_env_steps} R={row["objective/reward_mean"]:.3f} '
                  f'C={row["objective/cost_mean"]:.2f} P(C>15)={row["quantile/constraint_prob"]:.3f} '
                  f'updates={agent.update_counter} seconds={time.time()-start:.1f}',flush=True)
            if stopped:status('paused');return
        expected_steps=cli.main_batches*12000
        expected_optimizer=max(0,cli.main_batches-1)*8
        if (agent.actual_env_steps,agent.update_counter,agent.dual_update_count)!=(expected_steps,expected_optimizer,cli.main_batches):
            raise RuntimeError('QCPO final physical/optimizer/dual clocks mismatch')
        load_checkpoint(output/'latest.pt',agent,map_location=agent.device);save(output/'final.pt')
        atomic(output/'training_receipt.json',dict(algorithm='QCPO',actual_env_steps=agent.actual_env_steps,
            optimizer_steps=agent.update_counter,dual_update_count=agent.dual_update_count,
            rms_only_batches=agent.rms_only_batch_count,num_quantiles=25,replay_enabled=False,
            main_batches=cli.main_batches,initialization_env_steps=agent.initialization_env_steps,
            optimizer_epochs=agent.epochs,minibatches=agent.minibatches,identity=agent._checkpoint_identity(),
            reward_scale=1.,td3_enabled=False,mc_eta=0.,gamma_reward=REWARD_GAMMA,gamma_cost=COST_GAMMA,dual_gamma=1.,report_gamma=.99,
            checkpoint_roundtrip=True,training_seconds=time.time()-started,signature=signature))
        status('trained')
    except Exception:
        status('failed',error=traceback.format_exc());raise
    finally:
        if agent is not None:
            with contextlib.suppress(Exception):agent.vec_env.close()
        if env is not None:
            with contextlib.suppress(Exception):env.close()
        lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--main-batches',type=int,default=834)
    train(parser.parse_args())
