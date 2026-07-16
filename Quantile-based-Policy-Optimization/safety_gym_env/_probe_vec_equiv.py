# -*- coding: utf-8 -*-
"""
临时探针 3: SafetyVecEnv (串行) vs SafetyVecEnvMP (多进程) 等价性对拍 + 吞吐对比。
同 seed + 同一段固定动作序列 → 两后端的 obs/reward/cost 应【逐元素一致】
(mp 只是把同样的 env 摊进多核, 不动任何动力学)。
注意: mp 后端用 spawn, 主模块必须有 __main__ 保护。
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import sys, time
sys.path.insert(0, '.')
import numpy as np
import torch


def main():
    from envs import SafetyEnv, SafetyVecEnv, SafetyVecEnvMP

    ENV, B, T, SEED = 'Dynamic', 4, 100, 0
    dev = torch.device('cpu')
    rng = np.random.default_rng(123)
    ACTIONS = torch.as_tensor(rng.uniform(-1, 1, (T, B, 2)).astype(np.float32))

    ref = SafetyEnv(ENV)
    out = {}
    for name, cls in [('sync', SafetyVecEnv), ('mp', SafetyVecEnvMP)]:
        vec = cls(ENV, num_envs=B, horizon=T, device=dev, ref_env=ref, seed=SEED)
        t0 = time.time()
        s = vec.reset()
        S, R, C = [s.numpy()], [], []
        for t in range(T):
            s, r, c, done = vec.step(ACTIONS[t])
            S.append(s.numpy()); R.append(r.numpy()); C.append(c.numpy())
        dt = time.time() - t0
        out[name] = (np.stack(S), np.stack(R), np.stack(C), dt)
        vec.close()
        print(f"[{name:4s}] {B}x{T} steps in {dt:.2f}s  ({B*T/dt:.0f} env-steps/s)")

    dS = np.abs(out['sync'][0] - out['mp'][0]).max()
    dR = np.abs(out['sync'][1] - out['mp'][1]).max()
    dC = np.abs(out['sync'][2] - out['mp'][2]).max()
    print(f"max|Δobs|={dS:.2e}  max|Δreward|={dR:.2e}  max|Δcost|={dC:.2e}")
    assert dS == 0.0 and dR == 0.0 and dC == 0.0, "sync/mp 轨迹不一致!"
    print("EQUIV OK")

    # ---- 大 B 吞吐: mp 后端扩展性 ----
    for B2 in [16, 32]:
        vec = SafetyVecEnvMP(ENV, num_envs=B2, horizon=200, device=dev, ref_env=ref, seed=SEED)
        s = vec.reset()
        t0 = time.time()
        for t in range(200):
            a = torch.as_tensor(rng.uniform(-1, 1, (B2, 2)).astype(np.float32))
            s, r, c, done = vec.step(a)
        dt = time.time() - t0
        print(f"[mp B={B2:3d}] 200 steps in {dt:.2f}s  ({B2*200/dt:.0f} env-steps/s)")
        vec.close()
    ref.close()
    print("PROBE VEC DONE")


if __name__ == '__main__':
    main()
