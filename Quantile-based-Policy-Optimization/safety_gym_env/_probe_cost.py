# -*- coding: utf-8 -*-
"""
临时探针 2: 定向驾驶验证 cost 布线 —— 用简单比例控制器把点机器人开向【最近的障碍物】
(hazard / 非目标 button / gremlin), 确认每个论文环境的 cost 源真的会触发。
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import sys
sys.path.insert(0, '.')
import numpy as np
from envs.paper_envs import make_paper_env, PAPER_ENV_IDS


def drive_towards(task, target_xy):
    """点机器人比例控制: 返回 [推力, 转向] 朝 target_xy 开。"""
    pos = np.asarray(task.agent.pos)[:2]
    mat = np.asarray(task.agent.mat)                          # 3x3 旋转矩阵
    heading = np.arctan2(mat[1, 0], mat[0, 0])               # 机体 x 轴朝向
    vec = target_xy - pos
    ang = np.arctan2(vec[1], vec[0]) - heading
    ang = (ang + np.pi) % (2 * np.pi) - np.pi                # wrap 到 [-π,π]
    steer = np.clip(2.0 * ang, -1, 1)
    force = 1.0 if abs(ang) < np.pi / 2 else 0.3             # 背对时先转弯
    return np.array([force, steer], dtype=np.float32)


def obstacle_targets(task, key):
    """返回该任务所有'碰了要吃 cost'的目标坐标列表 [(x,y), ...]。"""
    ts = []
    if hasattr(task, 'hazards'):
        ts += [np.asarray(p)[:2] for p in task.hazards.pos]
    if hasattr(task, 'gremlins'):
        ts += [np.asarray(p)[:2] for p in task.gremlins.pos]
    if hasattr(task, 'buttons'):
        # 非目标 button 才有 cost; goal_button 是索引
        gb = getattr(task, '_goal_button', None) if hasattr(task, '_goal_button') else None
        if gb is None:
            gb = getattr(task, 'goal_button', None)
        for i, p in enumerate(task.buttons.pos):
            if gb is None or i != int(gb):
                ts.append(np.asarray(p)[:2])
    return ts


for key in PAPER_ENV_IDS:
    env = make_paper_env(key)
    obs, info = env.reset(seed=1)
    task = env.unwrapped.task
    tot_cost, n_cost_steps = 0.0, 0
    cost_src = {}
    for t in range(1000):
        targets = obstacle_targets(task, key)
        pos = np.asarray(task.agent.pos)[:2]
        tgt = min(targets, key=lambda p: np.linalg.norm(p - pos)) if targets else pos
        a = drive_towards(task, tgt)
        obs, r, c, term, trunc, info = env.step(a)
        tot_cost += c
        n_cost_steps += int(c > 0)
        for k, v in info.items():
            if k.startswith('cost_') and k != 'cost_sum' and np.any(np.asarray(v) > 0):
                cost_src[k] = cost_src.get(k, 0) + 1
        if term or trunc:
            obs, info = env.reset()
    print(f"[{key:14s}] 1000步定向驾驶: tot_cost={tot_cost:.1f} cost_steps={n_cost_steps} "
          f"cost_src={dict(sorted(cost_src.items()))}")
    env.close()
print("COST PROBE DONE")
