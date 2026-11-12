"""Evaluate an isolated GPU checkpoint on the held-out 2,048 episodes."""
import sys
import os
from pathlib import Path
import numpy as np
import torch
from base_runner import atomic_json
from dca_evaluate import sha

method, path, nsteps = sys.argv[1:4]
steps=int(nsteps)
episodes=int(sys.argv[4]) if len(sys.argv)>4 else 2048
run=Path(path).resolve()
checkpoint=run/f'checkpoint_{steps}.pt'
dest=run/f'validation_{steps}' if episodes==2048 else Path(sys.argv[5])
dest.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1)
evaluation_seed=int(os.environ.get("RISK_EVAL_SEED","92000000"))
torch.manual_seed(evaluation_seed)
if method=='dca':
    from dca_evaluate import load_dca,sample_checkpoint
    actor,critic,cfg,state=load_dca(run,checkpoint,torch.device('cuda:0'))
    sample_checkpoint(actor,critic,cfg,dest,episodes,evaluation_seed,calibration=False)
elif method=='qcpo':
    from qcpo_evaluate import load,policy_rollouts
    model,cfg,state=load(run,checkpoint)
    policy_rollouts(model,cfg,dest,episodes,evaluation_seed)
elif method=='tqpo':
    from run_tqpo import load_evaluation
    from tqpo_evaluate import sample_checkpoint
    actor,cfg,state=load_evaluation(run,checkpoint)
    sample_checkpoint(actor,None,cfg,dest,episodes,evaluation_seed,tqpo=True)
else:
    raise ValueError(method)
assert state['actual_env_steps']==steps
with np.load(dest/'samples.npz') as samples:
    r,c,dc=[samples[k] for k in ('undiscounted_reward','undiscounted_cost','discounted_cost')]
    assert all(a.shape==(episodes,) and np.isfinite(a).all() for a in (r,c,dc))
    atomic_json(dest/'result.json',dict(method=method,steps=steps,episodes=episodes,base_seed=evaluation_seed,
        checkpoint_sha256=sha(checkpoint),return_mean=float(r.mean()),cost_mean=float(c.mean()),
        cost_q90=float(np.quantile(c,.9)),violation_rate=float((c>15).mean()),discounted_violation_rate=float((dc>15).mean())))
