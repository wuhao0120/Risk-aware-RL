# -*- coding: utf-8 -*-
"""
safety-gymnasium 冒烟测试 (P0 可行性门): 确认可用 env id + reset/step/obs/action/cost 结构,
且 headless (无显示) 能跑物理步进。用 zprl python 运行:
    /vepfs-mlp2/c20250510/251204033/.conda/envs/zprl/bin/python _smoke.py
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')        # headless: 优先 EGL (无需 X); 失败可换 'osmesa'
import numpy as np
import gymnasium
import safety_gymnasium                            # import 即向 gymnasium registry 注册 Safety* 环境

# ---- 1. 列出相关 env id (点机器人 Goal/Button 任务, 对应论文 button/goal/gremlin) ----
ids = sorted(k for k in gymnasium.envs.registry.keys() if 'Safety' in k)
point_ids = [k for k in ids if 'Point' in k]
print(f"=== Safety* 环境总数: {len(ids)} ; Point 环境: {point_ids}")

# ---- 2. 逐个冒烟: reset + 数步 step, 打印 obs/action/cost 结构 + episode 长度 ----
for env_id in ['SafetyPointGoal1-v0', 'SafetyPointButton1-v0',
               'SafetyPointGoal2-v0', 'SafetyPointButton2-v0']:
    try:
        env = safety_gymnasium.make(env_id)
        obs, info = env.reset(seed=0)
        print(f"\n--- {env_id} ---")
        print(f" obs_space={env.observation_space.shape} act_space={env.action_space.shape} "
              f"act_range=[{float(env.action_space.low.min()):.2f},{float(env.action_space.high.max()):.2f}] "
              f"num_steps={getattr(env.unwrapped,'num_steps',None)}")
        print(f" reset obs shape={np.asarray(obs).shape} info_keys={list(info.keys())[:8]}")
        tot_c = 0.0
        for t in range(20):                        # 只跑 20 步验证接口 (mujoco CPU 步进)
            a = env.action_space.sample()
            obs, reward, cost, term, trunc, info = env.step(a)   # safety-gymnasium: 6 元组含 cost
            tot_c += float(cost)
        print(f" 20步: last_r={reward:.3f} last_cost={cost} cum_cost={tot_c:.2f} "
              f"term={term} trunc={trunc} info_keys={list(info.keys())[:8]}")
        env.close()
    except Exception as e:
        import traceback
        print(f"[FAIL] {env_id}: {e}")
        traceback.print_exc()
