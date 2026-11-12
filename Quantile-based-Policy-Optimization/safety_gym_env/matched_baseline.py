"""Official QCPO/TQPO sampler, model, update and runner on DCA's SimpleButton physics.

Set PYTHONPATH to the selected upstream rlpyt tree (one per method) and its
dependencies. The adapter preserves their previous-cost observation convention.
"""
import argparse
import copy
import json
import multiprocessing as mp
import os
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from envs.safety_env import SafetyEnv
from dca_simplebutton import BUDGET, HORIZON, summarize
from rlpyt.envs.base import Env, EnvStep
from rlpyt.spaces.float_box import FloatBox
from rlpyt.samplers.collections import TrajInfo
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.utils.logging.context import logger_context
from run_official_simplebutton import drain_qcpo_at_shutdown

StepInfo = namedtuple("StepInfo", ["cost", "cum_cost", "timeout"])


class ButtonEnv(Env):
    def __init__(self):
        self.physics = SafetyEnv("SimpleButton")
        self._observation_space = FloatBox(-1e6, 1e6, (self.physics.obs_dim + 1,))
        self._action_space = FloatBox(-1., 1., (self.physics.act_dim,))
        self.next_seed = None

    def seed(self, seed):
        self.next_seed = seed

    def reset(self):
        obs, _ = self.physics.reset(seed=self.next_seed)
        self.next_seed = None
        self.prev_cost = self.total_cost = 0.
        return np.append(obs, np.float32(0))

    def step(self, action):
        obs, reward, cost, term, trunc, _ = self.physics.step(action)
        # This is the ordering in the authors' SafetyGymEnvWrapper.step.
        augmented = np.append(obs, np.float32(self.prev_cost))
        self.prev_cost = cost
        self.total_cost += cost
        return EnvStep(augmented, np.float32(reward), term or trunc,
                       StepInfo(np.float32(cost), np.float32(self.total_cost), trunc))

    def close(self):
        self.physics.close()


class ButtonTrajInfo(TrajInfo):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Cost = 0.

    def step(self, observation, action, reward, done, agent_info, env_info):
        super().step(observation, action, reward, done, agent_info, env_info)
        self.Cost += env_info.cost


def components(method):
    prefix = 'rlpyt.projects.' + method
    model = getattr(import_module(prefix + '.' + method + '_model'),
                    'QcpoModel' if method == 'qcpo' else 'tqpoModel')
    agent = getattr(import_module(prefix + '.' + method + '_agent'),
                    'QcpoLstmAgent' if method == 'qcpo' else 'tqpoLstmAgent')
    algo = getattr(import_module(prefix + '.' + method), 'QCPO' if method == 'qcpo' else 'tqpo')
    runner = import_module(prefix + '.experiments.minibatch_rl').MinibatchRlConstrained
    config = copy.deepcopy(import_module(prefix + '.experiments.config_' + method).configs['LSTM'])
    return model, agent, algo, runner, config


def train(c):
    model_cls, agent_cls, algo_cls, runner_cls, config = components(c.method)
    config['algo'].update(discount=c.gamma, cost_discount=c.gamma, cost_limit=BUDGET,
                          target_outage_prob=0.20, pid_Ki=0.1)
    # Six batches per log interval divides 834 exactly, so final snapshot is 10.008M.
    batches = c.steps // 12000
    config['runner'].update(n_steps=c.steps, seed=c.seed,
                            log_interval_steps=72000 if batches % 6 == 0 else 12000)
    config['sampler'].update(batch_T=1000, batch_B=12)
    config['env'] = {'id': 'SimpleButton', 'adapter': 'ButtonEnv'}
    cpus = sorted(os.sched_getaffinity(0))
    # Keep all 12 sampler ranks and their seed streams when ranks share CPUs.
    worker_cpus = [(cpu,) for cpu in cpus[:12]] if len(cpus) >= 12 else [tuple(cpus)] * 12
    affinity = dict(cuda_idx=0, master_cpus=cpus, master_torch_threads=1,
                    worker_torch_threads=1, workers_cpus=worker_cpus)
    if c.method == 'qcpo':
        drain_qcpo_at_shutdown()
    sampler = CpuSampler(EnvCls=ButtonEnv, env_kwargs={}, TrajInfoCls=ButtonTrajInfo, **config['sampler'])
    agent = agent_cls(ModelCls=model_cls, model_kwargs=config['model'], **config['agent'])
    runner = runner_cls(algo=algo_cls(**config['algo']), agent=agent, sampler=sampler,
                        affinity=affinity, cost_limit=BUDGET, **config['runner'])
    # Upstream records gradNorm as a device tensor. Its NumPy logger expects CPU.
    store = runner.store_diagnostics
    def store_on_cpu(itr, trajectories, opt_info):
        opt_info.gradNorm[:] = [float(value) for value in opt_info.gradNorm]
        store(itr, trajectories, opt_info)
    runner.store_diagnostics = store_on_cpu
    if c.method == 'tqpo':
        # Upstream recreates a 3-panel plot at each log interval without closing it.
        import matplotlib.pyplot as plt
        plot = runner.plot_traj_info
        def plot_and_close(itr):
            try:
                plot(itr)
            finally:
                plt.close('all')
        runner.plot_traj_info = plot_and_close
    with logger_context(str(c.out), str(c.seed), c.method, config, snapshot_mode='last',
                        override_prefix=True):
        runner.train()
    if runner.n_itr != batches:
        raise RuntimeError('Unexpected final training budget')


def init_eval(method, checkpoint):
    global eval_env, eval_agent
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    model_cls, agent_cls, _, _, config = components(method)
    eval_env = ButtonEnv()
    eval_agent = agent_cls(ModelCls=model_cls, model_kwargs=config['model'], **config['agent'])
    eval_agent.initialize(eval_env.spaces)
    snapshot = torch.load(checkpoint, map_location="cpu", weights_only=False)
    eval_agent.load_state_dict(snapshot['agent_state_dict'])
    eval_agent.sample_mode(snapshot['itr'])


@torch.inference_mode()
def score(seed):
    eval_env.seed(seed)
    x = eval_env.reset()
    torch.manual_seed(seed + 1_000_000_000)
    eval_agent.reset()
    prev_a, prev_r = torch.zeros(1, eval_env.physics.act_dim), torch.zeros(1)
    reward_sum = cost_sum = 0.
    for t in range(1, HORIZON + 1):
        action, _ = eval_agent.step(torch.from_numpy(x)[None], prev_a, prev_r)
        x, r, done, info = eval_env.step(action[0].numpy())
        reward_sum += float(r)
        cost_sum += float(info.cost)
        prev_a, prev_r = action, torch.tensor([r])
        if done:
            return reward_sum, cost_sum, 0, t
    raise RuntimeError("Missing episode boundary")


def main():
    if "MATCHED_CPUS" in os.environ:
        from matched_campaign import honor_campaign_affinity
        honor_campaign_affinity()
    p = argparse.ArgumentParser()
    p.add_argument('method', choices=('qcpo', 'tqpo'))
    p.add_argument('--gamma', type=float, choices=(0.99, 1.0), required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=10008000)
    p.add_argument('--eval-episodes', type=int, default=2048)
    p.add_argument('--eval-seed', type=int, default=92000000)
    p.add_argument('--evaluate', action='store_true')
    c = p.parse_args()
    c.out = c.out.resolve()
    if c.steps % 12000:
        p.error('steps must be whole upstream batches')
    if not c.evaluate:
        train(c)
        return
    checkpoint = c.out / ('run_' + str(c.seed)) / 'params.pkl'
    snapshot = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (snapshot['itr'] + 1) * 12000 != c.steps:
        raise ValueError("Evaluation requires the final checkpoint")
    del snapshot
    with ProcessPoolExecutor(12, mp_context=mp.get_context('fork'),
                             initializer=init_eval, initargs=(c.method, checkpoint)) as pool:
        scores = list(pool.map(score, range(c.eval_seed, c.eval_seed + c.eval_episodes), chunksize=8))
    scores = np.asarray(scores)
    result = summarize(scores)
    result.pop('mean_buttons')
    np.save(c.out / 'final_episodes.npy', scores)
    (c.out / 'final.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
