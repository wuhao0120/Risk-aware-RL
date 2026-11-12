"""Bounded CPU checks: numerical protocol tests, checkpoint loading, real episodes.

Initialize final networks without the CUDA training vector, restore them through
each method's real loader, then evaluate two complete CPU MuJoCo episodes.
No training convergence or GPU determinism claim follows from this smoke test.
"""
from __future__ import annotations
import argparse
import copy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import base_runner as base
base.setup()
import numpy as np
import torch
from envs import SafetyEnv, make_vec_env
from utils.checkpoint import save_checkpoint
from summarize import episode_metrics

PROFILE = 'point_velocity_implicitfast_v1'


class ConstructionVector:
    """Metadata and RNG interface used only during network initialization."""
    n, act_dim, _seed = 1000, 2, 0
    def __init__(self, obs_dim, action_mode):
        self.obs_dim = obs_dim
        self.states = [dict(version=1, backend='mjwarp', action_mode=action_mode,
                            dynamics_profile=PROFILE, seed=0, slot=i, reset_count=0, goal_count=0)
                       for i in range(25)]
    def checkpoint_rng_state(self):
        return copy.deepcopy(self.states)
    def restore_rng_state(self, states):
        self.states = copy.deepcopy(states)


def cpu_vector(*args, **kwargs):
    """Use the real synchronous simulator for bounded CPU checks."""
    kwargs['backend'] = 'sync'
    return make_vec_env(*args, **kwargs)


def cpu_eval_env(batch, seed, cfg):
    ref = SafetyEnv('SimpleButton', seed=seed, dynamics_profile=PROFILE)
    return ref, make_vec_env('SimpleButton', batch, 1000, torch.device('cpu'),
                            ref_env=ref, seed=seed, backend='sync', dynamics_profile=PROFILE)


def main(output):
    root = Path(__file__).resolve().parent
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Smoke output must be empty; use a fresh directory')
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, '-m', 'pytest', '-q', str(root)], check=True, cwd=root)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    import agents.vec_base as vec_base
    import run, agent, qcpo_run, qcpo_agent, run_tqpo, tqpo_agent
    import dca_evaluate, qcpo_evaluate, tqpo_evaluate
    records = []
    for method in ('dca', 'qcpo', 'tqpo'):
        path = output / f'{method}_seed0'
        path.mkdir()
        mode = 'author_raw_env_clamp' if method == 'qcpo' else 'explicit_box_clip'
        ref = SafetyEnv('SimpleButton', seed=0, action_mode=mode, dynamics_profile=PROFILE)
        dim = int(ref.observation_space.shape[0])
        try:
            if method in ('dca', 'qcpo'):
                args = (run.make_args if method == 'dca' else qcpo_run.make_args)(0, output=path, device='cpu')
                cls = agent.Agent if method == 'dca' else qcpo_agent.QCPOUndiscountedAgent
                with patch.object(vec_base, 'make_vec_env', return_value=ConstructionVector(dim, mode)):
                    trainer = cls(args, ref)
                signature = dict(experiment=agent.IDENTITY if method == 'dca' else qcpo_agent.IDENTITY,
                                 arm=agent.ARMS[0] if method == 'dca' else 'qcpo', seed=0,
                                 mean_divisor=100. if method == 'dca' else 1., smoke_only=True)
                trainer.checkpoint_signature = signature
                cfg = base.serial_config(args)
                cfg.update(gamma=1., cost_gamma=1., cost_discount=1., report_gamma=.99)
                base.atomic_json(path / 'config.json', dict(configuration=cfg, signature=signature,
                                                          identity=trainer._checkpoint_identity()))
                save_checkpoint(path / 'initial.pt', trainer, -1)
                del trainer
                if method == 'dca':
                    actor, critic, cfg, state = dca_evaluate.load_dca(path, path / 'initial.pt', torch.device('cpu'))
                    with patch.object(dca_evaluate, 'make_env', side_effect=cpu_eval_env):
                        result = dca_evaluate.sample_checkpoint(actor, critic, cfg, path / 'evaluation', 2, 777, calibration=True)
                    base.atomic_json(path / 'evaluation/metrics.json', result)
                    del actor, critic, state
                else:
                    model, cfg, state = qcpo_evaluate.load(path, path / 'initial.pt', device='cpu')
                    with patch.object(qcpo_evaluate, 'make_vec_env', side_effect=cpu_vector):
                        qcpo_evaluate.policy_rollouts(model, cfg, path / 'evaluation', 2, 777)
                    del model, state
            else:
                trainer = tqpo_agent.TQPO(tqpo_agent.OfficialActor(dim))
                signature = dict(experiment=tqpo_agent.IDENTITY, seed=0, smoke_only=True)
                base.atomic_json(path / 'config.json', dict(signature=signature, configuration=dict(tqpo_agent.PROTOCOL, seed=0)))
                torch.save(dict(agent=trainer.state_dict(), experiment_signature=signature), path / 'initial.pt')
                del trainer
                actor, cfg, state = run_tqpo.load_evaluation(path, path / 'initial.pt', device='cpu')
                with patch.object(run_tqpo, 'make_vec_env', side_effect=cpu_vector):
                    result = tqpo_evaluate.sample_checkpoint(actor, None, cfg, path / 'evaluation', 2, 777, tqpo=True)
                base.atomic_json(path / 'evaluation/metrics.json', result)
                del actor, state
            with np.load(path / 'evaluation/samples.npz') as data:
                records.append(dict(method=method.upper(), episodes=2,
                                    **episode_metrics(data['undiscounted_reward'], data['undiscounted_cost'])))
            print(f'{method.upper()}: checkpoint restored; two real 1000-step CPU episodes completed', flush=True)
        finally:
            ref.close()
    base.atomic_json(output / 'smoke_receipt.json', dict(training_performed=False, simulator='CPU MuJoCo',
                     tests='passed', checkpoint_loaders='all three final methods', results=records))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
