"""Run one snapshot in its own interpreter (snapshots have overlapping modules)."""
import json
from pathlib import Path
import sys
from experiment import ROOT, REGISTRY

key, destination = sys.argv[1:]
spec = REGISTRY[key]
sys.path.insert(0, str(ROOT / spec['source']))
import base_runner
base_runner.setup()
import torch
from envs import SafetyEnv, make_vec_env

torch.set_num_threads(1)
ref = SafetyEnv(spec['environment'], seed=71, action_mode='explicit_box_clip',
                dynamics_profile='point_velocity_implicitfast_v1')
vec = None
try:
    vec = make_vec_env(spec['environment'], 2, 1000, torch.device('cuda:0'), ref_env=ref,
                       seed=71, backend='mjwarp', action_mode='explicit_box_clip',
                       dynamics_profile='point_velocity_implicitfast_v1')
    obs = vec.reset()
    for _ in range(10):
        result = vec.step(torch.zeros((2, 2), device='cuda:0'))
        for x in (result.observation, result.reward, result.cost):
            if not torch.isfinite(x).all():
                raise RuntimeError(f'Nonfinite physics: {key}')
    torch.cuda.synchronize()
    Path(destination).write_text(json.dumps(dict(environment=spec['environment'],
        source=spec['source'], worlds=2, steps=10, observation_shape=list(obs.shape), passed=True), indent=2) + '\n')
finally:
    if vec is not None:
        vec.close()
    ref.close()
