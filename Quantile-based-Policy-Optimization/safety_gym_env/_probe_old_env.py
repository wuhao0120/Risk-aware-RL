# -*- coding: utf-8 -*-
"""
旧环境探针 (跑在 qcpo_ref conda 环境, py3.8 + mujoco-py 2.0 + 旧 safety-gym):
用论文 config0-3 注册 4 个原版环境, 随机策略整 episode 采样, 报
  obs 维度 / horizon / R(折扣 reward 回报) / Σc(未折扣 episode cost) 分布 / d=15 outage,
与新环境 (safety-gymnasium) 的 CALIB 结果对照 → 验证任务语义对齐。
用法: python _probe_old_env.py [episodes_per_env=16]
"""
import sys
import numpy as np

import gym
from gym import register
sys.path.insert(0, '/vepfs-mlp2/c20250510/251204033/dependencies/rlpyt')
from rlpyt.projects.qcpo.safety_gym_envs.config_safety_gym_env import (
    config0, config1, config2, config3)

E = int(sys.argv[1]) if len(sys.argv) > 1 else 16
GAMMA = 0.99

ENVS = [('SimpleButton', 'SimpleButtonEnv-v0', config0),
        ('Dynamic', 'DynamicEnv-v0', config1),
        ('Gremlin', 'GremlinEnv-v0', config2),
        ('DynamicButton', 'DynamicButtonEnv-v0', config3)]
for _, eid, cfg in ENVS:
    register(id=eid, entry_point='safety_gym.envs.mujoco:Engine', kwargs={'config': cfg})

for name, eid, _ in ENVS:
    env = gym.make(eid)
    rng = np.random.RandomState(0)
    env.seed(0)
    Rs, Cs, cost_steps = [], [], 0
    horizon = None
    for ep in range(E):
        obs = env.reset()
        od = int(np.asarray(obs).size)
        R, C, t, dr = 0.0, 0.0, 0, 1.0
        done = False
        while not done:
            a = rng.uniform(-1, 1, env.action_space.shape).astype(np.float32)
            obs, r, done, info = env.step(a)
            c = float(info.get('cost', 0.0))
            R += dr * r; dr *= GAMMA
            C += c; cost_steps += int(c > 0); t += 1
        horizon = t
        Rs.append(R); Cs.append(C)
    R = np.asarray(Rs); C = np.asarray(Cs)
    print(f"[{name:14s}] obs={od} horizon={horizon} E={E}")
    print(f"   R:  mean={R.mean():.3f} std={R.std():.3f} [{R.min():.2f},{R.max():.2f}]")
    print(f"   Σc: mean={C.mean():.2f} std={C.std():.2f} "
          f"pct[50,80,90,95]={np.round(np.percentile(C, [50, 80, 90, 95]), 1)}")
    print(f"   outage: P(Σc>=15)={np.mean(C >= 15):.3f}  P(Σc>=25)={np.mean(C >= 25):.3f}  "
          f"avg_step_cost={C.sum() / (E * horizon):.4f}")
    env.close()
print("OLD ENV PROBE DONE")
