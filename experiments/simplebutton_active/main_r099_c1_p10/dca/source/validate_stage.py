import sys
from pathlib import Path
import numpy as np
import torch
from base_runner import atomic_json
from dca_evaluate import load_dca, sample_checkpoint, sha
run=Path(sys.argv[1]).resolve()
steps=int(sys.argv[2])
ckpt=run/f'checkpoint_{steps}.pt'
dest=run/f'validation_{steps}'
actor, critic, cfg, state=load_dca(run, ckpt, torch.device('cuda:0'))
assert state['actual_env_steps']==steps and cfg['cost_gamma']==.99 and cfg['dual_gamma']==1 and cfg['mean_divisor']==25
torch.set_num_threads(1)
torch.manual_seed(92000000)
sample_checkpoint(actor,critic,cfg,dest,2048,92000000,calibration=False)
with np.load(dest/'samples.npz') as a:
    cost=a['undiscounted_cost']
    assert cost.shape==(2048,)
    atomic_json(dest/'result.json',dict(steps=steps,episodes=len(cost),checkpoint_sha256=sha(ckpt),return_mean=float(a['undiscounted_reward'].mean()),cost_mean=float(cost.mean()),cost_q90=float(np.quantile(cost,.9)),violation_rate=float((cost>15).mean()),discounted_violation_rate=float((a['discounted_cost']>15).mean())))
