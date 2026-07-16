# -*- coding: utf-8 -*-
"""
新环境对照探针 (zprl): 与 _probe_old_env.py 完全同口径 —— 均匀随机动作 U(-1,1),
整 episode (1000 步), 报 R(折扣)/Σc 分布与 d=15 outage。用 mp VecEnv 加速。
用法: python _probe_new_env_uniform.py [episodes_per_env=64]
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import sys
sys.path.insert(0, '.')
import numpy as np
import torch


def main():
    from envs import SafetyEnv, SafetyVecEnvMP, PAPER_ENV_IDS

    E = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    GAMMA, B, T = 0.99, 16, 1000
    dev = torch.device('cpu')

    for name in PAPER_ENV_IDS:
        ref = SafetyEnv(name)
        vec = SafetyVecEnvMP(name, num_envs=B, horizon=T, device=dev, ref_env=ref, seed=0)
        rng = np.random.RandomState(0)
        Rs, Cs = [], []
        for _ in range(max(1, E // B)):
            s = vec.reset()
            R = torch.zeros(B); C = torch.zeros(B)
            dr = 1.0
            for t in range(T):
                a = torch.as_tensor(rng.uniform(-1, 1, (B, vec.act_dim)).astype(np.float32))
                s, r, c, done = vec.step(a)
                R += dr * r; dr *= GAMMA
                C += c
            Rs.append(R); Cs.append(C)
        R = torch.cat(Rs).numpy(); C = torch.cat(Cs).numpy()
        print(f"[{name:14s}] obs={vec.obs_dim} E={R.shape[0]}")
        print(f"   R:  mean={R.mean():.3f} std={R.std():.3f} [{R.min():.2f},{R.max():.2f}]")
        print(f"   Σc: mean={C.mean():.2f} std={C.std():.2f} "
              f"pct[50,80,90,95]={np.round(np.percentile(C, [50, 80, 90, 95]), 1)}")
        print(f"   outage: P(Σc>=15)={np.mean(C >= 15):.3f}  P(Σc>=25)={np.mean(C >= 25):.3f}  "
              f"avg_step_cost={C.mean() / T:.4f}")
        vec.close(); ref.close()
    print("NEW ENV UNIFORM PROBE DONE")


if __name__ == '__main__':
    main()
