# -*- coding: utf-8 -*-
"""
临时探针: 验证 4 个论文环境的 cost 真的会触发、reward 非退化、gremlin 会动。
方法: 恒定前进+缓转弯动作让点机器人扫场 2000 步 (比随机动作走得远), 统计 cost/reward;
     并直接检查 task 内部的 hazards/buttons/gremlins 是否真实挂载。
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import sys
sys.path.insert(0, '.')
import numpy as np
from envs.paper_envs import make_paper_env, PAPER_ENV_IDS

for key in PAPER_ENV_IDS:
    env = make_paper_env(key)
    obs, info = env.reset(seed=0)
    task = env.unwrapped.task
    # ---- 世界内容检查: 各障碍是否真实挂载 ----
    geoms = {g.name: getattr(g, 'num', 1) for g in task._geoms.values()} if hasattr(task, '_geoms') else {}
    mocaps = {m.name: getattr(m, 'num', 1) for m in task._mocaps.values()} if hasattr(task, '_mocaps') else {}
    free = {o.name: getattr(o, 'num', 1) for o in task._free_geoms.values()} if hasattr(task, '_free_geoms') else {}

    tot_cost, tot_rew, n_cost_steps = 0.0, 0.0, 0
    cost_src = {}
    grem_pos0 = None
    grem_moved = 0.0
    ep = 0
    rng = np.random.default_rng(0)
    for t in range(2000):
        # 扫场动作: 前进 + 慢转 + 少量噪声 (点机器人 act=[推力, 转向])
        a = np.clip(np.array([1.0, 0.35 * np.sin(t / 60.0)]) +
                    rng.normal(0, 0.2, 2), -1, 1).astype(np.float32)
        obs, r, c, term, trunc, info = env.step(a)
        tot_rew += r
        tot_cost += c
        n_cost_steps += int(c > 0)
        for k, v in info.items():
            if k.startswith('cost_') and np.any(np.asarray(v) > 0):
                cost_src[k] = cost_src.get(k, 0) + 1
        # gremlin 运动检查
        if 'Gremlin' in key and hasattr(task, 'gremlins'):
            gp = np.asarray(task.gremlins.pos)[:, :2] if hasattr(task.gremlins, 'pos') else None
            if gp is not None:
                if grem_pos0 is None:
                    grem_pos0 = gp.copy()
                grem_moved = max(grem_moved, float(np.abs(gp - grem_pos0).max()))
        if term or trunc:
            ep += 1
            obs, info = env.reset()
    print(f"[{key:14s}] geoms={geoms} mocaps={mocaps} free={free}")
    print(f"   2000步扫场: tot_cost={tot_cost:.1f} cost_steps={n_cost_steps} tot_rew={tot_rew:.1f} "
          f"episodes_done={ep} cost_src={sorted(cost_src)} grem_moved={grem_moved:.3f}")
    env.close()
print("PROBE DONE")
