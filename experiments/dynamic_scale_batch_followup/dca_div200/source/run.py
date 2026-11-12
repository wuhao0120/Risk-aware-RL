"""Final Dynamic DCA: undiscounted reward/cost with lambda normalization.

The runner preserves complete-episode collection and E8 KL transactions. The
agent owns MC reward labels, stagewise pinball fitting, interpolated upper-cost
probabilities, and one fresh-batch PI step including every warmup rollout.
"""
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
from agents.four_env_fullbatch_formal import DQC_LR2_KL_ALGO
from agent import Agent, IDENTITY, ARMS, ARM_DIVISORS, ARM_CONFIGS, ACTOR_NORMALIZATION, LAMBDA_NORMALIZATION, PI_CAP
from schemeb_math import upper_cost_violation_probability

MEAN_DIVISORS = ARM_DIVISORS
from envs import SafetyEnv
from utils.checkpoint import save_checkpoint, load_checkpoint

ROOT=Path(__file__).resolve().parent
PYTHON=base.PYTHON
atomic=base.atomic_json


def hashes():
    """科学源码冻结；独立评估和调度修复不改变已开始训练的算法身份。"""
    files=[ROOT/p for p in ('run.py','agent.py','replay_agent.py','actor_transaction.py',
        'raw_cost_agents.py','diagnostics.py','base_runner.py','schemeb_math.py')]
    files+=sorted((ROOT/'source').rglob('*.py'))
    files+=sorted((ROOT/'configs').glob('*.json'))+[ROOT/'objective.py']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def make_args(seed, reward_scale=None, output=None, device='cuda:0', arm='schemeb_raw_mean100_lambda_norm'):
    """Build canonical network/environment settings, then declare live semantics.

    reward_scale=100 is only an inherited constructor compatibility setting.
    mean_divisor owns the actual reward-only scaling; the risk branch never
    divides by it. Reward, cost critic, actor event and dual all use gamma=1.
    The separate report_gamma=.99 is used only for additional logged curves.
    """
    if arm not in ARMS:
        raise ValueError('unknown Scheme B arm')
    arm_config=ARM_CONFIGS[arm]
    divisor=MEAN_DIVISORS[arm]
    if reward_scale is not None and float(reward_scale)!=divisor:
        raise ValueError('reward scale does not match registered arm')
    output=Path(output or ROOT/'runs'/f'{arm}_seed{seed}')
    args=base_args(DQC_LR2_KL_ALGO,int(seed),torch.device(device),'Dynamic')
    args.algo_name='DCA_MATCHED_LAMBDA_NORMALIZATION_PI_SCALE'
    args.stability_arm=arm
    args.reward_scale=100.
    args.mean_divisor=divisor
    args.dual_lr_multiplier=1.
    args.dual_base_schedule_is_provenance=True
    args.actual_dual_controller='pi'
    args.actual_risk_advantage='transition_independent_successor_samples'
    args.actual_mean_estimator='complete_training_gamma_mc_minus_prefit_value'
    args.actual_actor_normalization=arm_config['actor_normalization']
    args.actual_pi_ki=arm_config['pi_ki'];args.actual_pi_ema=arm_config['pi_ema'];args.actual_pi_kp=arm_config['pi_kp']
    args.actual_dual_cap=PI_CAP
    args.return_threshold=None
    args.training_gamma=REWARD_GAMMA;args.report_gamma=.99
    args.replay_capacity_episodes=100
    args.critic_steps_per_batch=400;args.warmup_critic_steps=100
    args.cost_threshold=15.;args.cost_quantile_level=.8
    args.failure_alpha=args.target_outage_prob=.2
    args.primal_constraint_variable=COST
    args.dual_constraint_variable="undiscounted_environment_cost"
    args.violation_probability_variable=EVENT
    args.base_factor_metadata_is_provenance=True
    args.env_id='SafetyPointDynamic0-v0';args.vec_backend='mjwarp'
    args.dynamics_profile='point_velocity_implicitfast_v1'
    args.defer_wandb_init=True;args.show_progress=args.show_eval_progress=False
    args.log_interval=1;args.wandb_mode='disabled';args.wandb_project='Dynamic_full_gpu'  # Required constructor value; internal W&B is disabled.
    args.wandb_name=args.wandb_id=args.wandb_lineage=output.name
    args.wandb_group='matched-dca'
    args.wandb_dir=str(output/'wandb');args.progress_file=str(output/'progress.json')
    return args


def configure_budget(agent, main_batches=380, warmup_batches=20):
    """正式和冒烟都执行完整20批warmup；主预算只改变停止位置。"""
    if warmup_batches != 20 or not 1 <= main_batches <= 380:
        raise ValueError('expected full warmup20 and bounded main budget1..380')
    agent.num_iterations=main_batches


def train(cli):
    output=cli.output.resolve();output.mkdir(parents=True,exist_ok=True)
    lock=(output/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    args=make_args(cli.seed,cli.reward_scale,output,arm=cli.arm)
    shared_std=args.actual_actor_normalization==ACTOR_NORMALIZATION
    signature=dict(experiment=IDENTITY,arm=cli.arm,variant=cli.arm,seed=cli.seed,
        mean_divisor=float(args.mean_divisor),reward_scale=float(args.mean_divisor),
        inherited_constructor_reward_scale_is_provenance=100.,
        dual_lr_multiplier=1.,dual_schedule='PI_fresh25_including_warmup',dual_controller='pi',
        risk_advantage='transition_independent_successor_4_samples',actor_normalization=args.actual_actor_normalization,
        pi_parameters=dict(ki=args.actual_pi_ki,ema=args.actual_pi_ema,kp=args.actual_pi_kp,cap=PI_CAP),
        lambda_normalization=not shared_std,shared_std_normalization=shared_std,
        shared_std_floor=.01 if shared_std else None,
        main_batches=cli.main_batches,warmup_batches=20,source=hashes(),
        num_quantiles=128,n_step=1,mc_eta=0.,gamma_reward=REWARD_GAMMA,gamma_cost=COST_GAMMA,dual_gamma=1.,report_gamma=.99,
        mean_estimator='complete_training_gamma_mc_minus_prefit_value',
        threshold=15.,epsilon=.2,critic_steps_per_batch=400,warmup_critic_steps=100,
        critic_loss='all_pairs_pinball_mean_valid_rows_and_128x128_pairs',
        cdf_readout='sorted_midpoint_linear_interpolation',
        replay_capacity_episodes=100,target_mode='hard',target_tau=1.,
        critic_main_stages=[134,133,133],critic_warmup_stages=[34,33,33],
        actor_clock='fresh_B25_E8_KL_transaction_backtracking',
        target_clock='three_hard_copies_per_rollout_after_stage_fixed_target_fit',
        actor_scaling='raw_reward_advantage_div_c_minus_lambda_raw_cost_probability_advantage',
        actor_risk_event=EVENT,dual_risk_event='undiscounted_environment_cost_strict_gt_15',
        mixed_primal_dual_events=False,
        constraint=EVENT)
    if (output/'config.json').exists():
        if json.loads((output/'config.json').read_text())['signature']!=signature:
            raise ValueError('configuration/scientific source drift')
    agent=env=None;rows=[];episode_rows=[];last=-1;stopped=False;started=time.time()
    def stop(*_):
        nonlocal stopped
        stopped=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    def status(state,**extra):
        atomic(output/'status.json',dict(status=state,pid=os.getpid(),updated_at=time.time(),
            elapsed_seconds=time.time()-started,actual_env_steps=int(agent.actual_env_steps) if agent else 0,**extra))
    def write_history():
        for path,values in [('metrics.jsonl',rows),('episodes.jsonl',episode_rows)]:
            (output/path).write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in values))
    def save(path):
        save_checkpoint(path,agent,last,runner_state={'rows':rows,'diagnostics':rows,'episodes':episode_rows})
    def indexed_save():
        steps=int(agent.actual_env_steps)
        path=output/f'checkpoint_{steps}.pt'
        save(path)
        index_path=output/'checkpoint_index.json'
        entries=json.loads(index_path.read_text())['checkpoints'] if index_path.exists() else []
        entry=dict(path=path.name,nominal_env_steps=steps,actual_env_steps=steps)
        entries=[item for item in entries if item['actual_env_steps']!=steps]+[entry]
        atomic(index_path,dict(experiment_identity=IDENTITY,arm=cli.arm,seed=cli.seed,
            checkpoints=sorted(entries,key=lambda item:item['actual_env_steps'])))
    try:
        torch.set_num_threads(1);base.seed_all(cli.seed);status('constructing')
        env=SafetyEnv(args.env_id,seed=cli.seed,action_mode='explicit_box_clip',dynamics_profile=args.dynamics_profile)
        agent=Agent(args,env);agent.checkpoint_signature=signature
        configure_budget(agent,cli.main_batches)
        if agent.gamma != REWARD_GAMMA or agent.return_threshold != 15 or agent.failure_alpha != .2:
            raise ValueError('discount/constraint protocol mismatch')
        config=base.serial_config(args)
        config.update(wandb_project='ICLR_rebuttal_Dynamic',wandb_mode='online',wandb_owner='campaign.py')
        config.update(num_quantiles=128,critic_n_step=1,return_threshold=15.,q_alpha=.8,
            gamma=agent.gamma,cost_gamma=COST_GAMMA,dual_gamma=1.,report_gamma=agent.report_gamma,target_update_mode='hard',target_tau=1.,mc_eta=0.,
            stability_arm=cli.arm,target_update_clock=signature['target_clock'],
            dual_lr_multiplier=1.,mean_divisor=agent.mean_divisor,
            actual_mean_estimator=signature['mean_estimator'],
            actual_dual_initial_lr=agent.dual_step_size(),
            actual_dual_controller=agent.dual_controller,actual_risk_advantage=agent.risk_advantage_mode,
            actual_actor_normalization=agent.actor_normalization,
            actual_actor_std_normalization=agent.actor_std_normalization,actual_actor_outer_lambda_division=False,
            actual_actor_one_plus_lambda_normalization=not agent.actor_std_normalization,
            actual_dual_schedule=signature['dual_schedule'],
            actor_update_clock=signature['actor_clock'],actor_scaling=signature['actor_scaling'],
            canonical_experiment_identity=agent._checkpoint_identity(),
            reward_actor_outer_discount='none',risk_actor_outer_discount='none',
            primal_constraint_variable=COST,dual_constraint_variable=COST)
        atomic(output/'config.json',dict(configuration=config,signature=signature,
            planned_env_steps=(20+cli.main_batches)*25000,identity=agent._checkpoint_identity()))
        if (output/'latest.pt').exists():
            payload=load_checkpoint(output/'latest.pt',agent,map_location=agent.device)
            last=int(payload['iteration']);rows=payload['runner_state']['rows']
            episode_rows=payload['runner_state']['episodes']
        if agent.actual_env_steps==0 and not (output/'checkpoint_0.pt').exists():
            indexed_save()
        write_history()
        atomic(output/'cuda_ready.json',dict(pid=os.getpid(),time=time.time()))
        def batch(iteration,initialization):
            nonlocal last
            start=time.time()
            rollout=agent._rollout_core(phase='initialization' if initialization else 'main')
            with torch.no_grad():
                features=agent._critic_rollout_features(rollout)[0]
                z=agent.reward_critic(features,rollout['A_exec'][0])
                preprob=upper_cost_violation_probability(z.double(),15.).cpu().numpy()
            metrics=agent._update_rollout(iteration,rollout,initialization=initialization)
            if initialization:
                agent.initialization_cursor+=1
                samples=agent._episode_samples
                metrics=agent._log_core(-1,samples['objective_return'],samples['discounted_cost'],samples['objective_cost'],extra=metrics)
            else:last=iteration
            samples=agent._episode_samples
            phase='warmup' if initialization else 'main'
            metrics.update({'diagnostic/batch_seconds':time.time()-start,
                'diagnostic/preupdate_initial_cost_probability_mean':float(preprob.mean()),
                'diagnostic/preupdate_initial_cost_probability_bias':float(preprob.mean()-np.mean(np.asarray(samples['critic_objective_cost'])>15.)),
                'diagnostic/preupdate_actor_dual_probability_bias':float(preprob.mean()-np.mean(np.asarray(samples['objective_cost'])>15.))})
            row=dict(iteration=iteration,phase=phase,actual_env_steps=int(agent.actual_env_steps),wall_time=time.time(),**metrics)
            episode=dict(phase=phase,actual_env_steps=int(agent.actual_env_steps),
                preupdate_cost_probabilities=preprob.tolist(),**samples)
            del rollout
            rows.append(row);episode_rows.append(episode)
            save(output/'latest.pt')
            if agent.actual_env_steps % 500000 == 0:
                indexed_save()
            for path,value in [('metrics.jsonl',row),('episodes.jsonl',episode)]:
                with (output/path).open('a') as stream:stream.write(json.dumps(value,allow_nan=False)+'\n')
            status('training',phase=phase,main_batches=last+1,warmup_batches=agent.initialization_cursor)
            print(f'{cli.arm}/s{cli.seed} steps={agent.actual_env_steps} '
                f'RΣ={row["undisc_reward/aver_reward"]:.4f} CΣ={row["undisc_cost/mean"]:.3f} '
                f'P(CΣ>15)={row["quantile/constraint_prob"]:.3f} '
                f'IS_ESS={row["diagnostic/replay/is_ess_fraction"]:.4f} seconds={time.time()-start:.1f}',flush=True)
        while agent.initialization_cursor<20:
            batch(-1,True)
            if stopped:status('paused');return
        for iteration in range(last+1,cli.main_batches):
            batch(iteration,False)
            if stopped:status('paused');return
        if agent.actual_env_steps!=(20+cli.main_batches)*25000:raise RuntimeError('incomplete physical budget')
        load_checkpoint(output/'latest.pt',agent,map_location=agent.device)
        save(output/'final.pt')
        if not (output/f'checkpoint_{agent.actual_env_steps}.pt').exists():
            indexed_save()
        attempted=int(getattr(agent,'actor_candidate_attempts_total',0))
        accepted=int(agent.actor_optimizer_steps)
        rejected=int(getattr(agent,'actor_candidate_rejections_total',0))
        if attempted!=accepted+rejected:
            raise RuntimeError('Actor candidate accounting must equal accepted plus rejected')
        expected_batches=20+cli.main_batches
        if int(agent.critic_optimizer_steps)!=20*100+cli.main_batches*400:
            raise RuntimeError('critic optimizer budget mismatch')
        if int(agent.target_update_count)!=3*expected_batches or int(agent.dual_update_count)!=expected_batches:
            raise RuntimeError('Scheme B target/PI update clock mismatch')
        atomic(output/'training_receipt.json',dict(actual_env_steps=agent.actual_env_steps,
            critic_optimizer_steps=agent.critic_optimizer_steps,target_update_count=agent.target_update_count,
            actor_optimizer_steps=agent.actor_optimizer_steps,
            actor_attempted_steps=attempted,
            actor_accepted_steps=accepted,
            actor_rejected_steps=rejected,
            actor_nonfinite_rejections=int(getattr(agent,'actor_candidate_nonfinite_rejections_total',0)),
            actor_candidate_optimizer_steps=int(getattr(agent,'actor_candidate_optimizer_steps_total',0)),
            arm=cli.arm,variant=cli.arm,seed=cli.seed,target_update_clock=signature['target_clock'],
            dual_lr_multiplier=agent.dual_lr_multiplier,dual_update_count=agent.dual_update_count,
            dual_controller=agent.dual_controller,risk_advantage=agent.risk_advantage_mode,
            pi_controller_state=dict(error_ema=agent.pi_error_ema,updates=agent.pi_update_count),
            actor_normalization=agent.actor_normalization,
            pi_parameters=signature['pi_parameters'],
            checkpoint_interval_physical_steps=500000,includes_precollection_checkpoint=True,
            num_quantiles=agent.num_quantiles,n_step=agent.critic_n_step,
            target_mode='hard',tau=1.,gamma=agent.gamma,cost_gamma=COST_GAMMA,dual_gamma=1.,report_gamma=agent.report_gamma,replay_episodes=len(agent.episode_replay),checkpoint_roundtrip=True,
            training_seconds=time.time()-started,signature=signature))
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
    parser.add_argument('--reward-scale',type=float,choices=tuple(MEAN_DIVISORS.values()),default=None)
    parser.add_argument('--arm',choices=ARMS,default=ARMS[0])
    parser.add_argument('--run-dir','--output',dest='output',type=Path,required=True)
    parser.add_argument('--main-batches',type=int,default=380)
    train(parser.parse_args())
