# -*- coding: utf-8 -*-
"""
paper_envs.py —— 复现 NeurIPS'22 QCPO 论文 config0-3 的 4 个 safety-gymnasium 环境。

用户硬性要求「环境和论文对齐」。论文 (wyjung0625/QCPO) 用 OpenAI safety-gym 的 Engine +
原始 config dict 定义 4 个点机器人环境; 本文件用 safety-gymnasium (新 mujoco) 的【任务子类】
逐项复刻这些 config 的任务语义 (机器人=point, 任务=goal/button, hazard/button/gremlin 的
数量/尺寸/keepout/固定坐标, lidar bins/max_dist, placements 范围)。

对齐口径 (见 DESIGN.md): 复刻的是【任务语义】—— 机器人/障碍布局/cost 来源/约束结构与论文
config0-3 逐项一致; obs 向量的【编码】无法与原 safety-gym(mujoco-py) 逐字节相同 (两库传感器
/lidar 实现不同), 这是 safety-gym→safety-gymnasium 迁移的固有残差, 已记录。

论文 config → 本文件类 (config_safety_gym_env.py):
    config0 SimpleButtonEnv  → SimpleButtonLevel0   : button, 2 buttons@[(-1,-1),(1,1)],
                                                       3 hazards@[(0,0),(-1,1),(0.5,-0.5)], lidar 16/3
    config1 DynamicEnv       → DynamicLevel0         : goal, 3 hazards(随机), lidar 16/3
    config2 GremlinEnv       → GremlinLevel0         : goal, 5 hazards + 3 gremlins, lidar 16/5, ext±2
    config3 DynamicButtonEnv → DynamicButtonLevel0   : button, 6 buttons(随机), lidar 16/3

机制 (builder.py 实证): Builder(task_id, config) 靠 get_task_class_name(task_id) 得到类名,
再 getattr(safety_gymnasium.tasks, 类名) 取类 → 故只需 (1) 把自定义类注入 tasks 命名空间,
(2) 注册一个 task_id 使其类名解析到该类。step() 已返回 6 元组 (obs,reward,cost,term,trunc,info),
cost=info['cost_sum'] (hazards/buttons/gremlins 各源之和)。
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')                    # headless (须在建 env 前)

import safety_gymnasium
from safety_gymnasium import tasks as _sg_tasks              # Builder 从这里 getattr 取任务类
from safety_gymnasium.assets.geoms import Hazards            # 危险区 (num/size/keepout/locations/cost)
from safety_gymnasium.assets.mocaps import Gremlins          # 移动障碍 (num/travel/keepout)
from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0
from safety_gymnasium.tasks.safe_navigation.button.button_level0 import ButtonLevel0


# ============================================================ 4 个论文任务类 ============================================================
class SimpleButtonLevel0(ButtonLevel0):
    """config0 SimpleButtonEnv: 2 个固定 button + 3 个固定 hazard 的 button 任务。"""

    def __init__(self, config) -> None:
        super().__init__(config=config)                      # ButtonLevel0: 已建 Point + Buttons(num=4) + Goal
        self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]  # 论文 placements_extents
        self.lidar_conf.max_dist = 3                         # 论文 lidar_max_dist
        self.lidar_conf.num_bins = 16                        # 论文 lidar_num_bins
        # 覆盖 buttons: 2 个固定坐标, 受约束 (原 ButtonLevel0 是 num=4/不约束)
        self.buttons.num = 2                                 # 论文 buttons_num
        self.buttons.size = 0.1                              # 论文 buttons_size
        self.buttons.keepout = 0.2                           # 论文 buttons_keepout
        self.buttons.locations = [(-1, -1), (1, 1)]          # 论文 buttons_locations (固定)
        self.buttons.is_constrained = True                   # 论文 constrain_buttons
        # 加 3 个固定 hazard (论文 hazards_num/size/keepout/locations)
        self._add_geoms(Hazards(num=3, size=0.3, keepout=0.305,
                                locations=[(0, 0), (-1, 1), (0.5, -0.5)]))


class DynamicLevel0(GoalLevel0):
    """config1 DynamicEnv: 3 个随机 hazard 的 goal 任务。"""

    def __init__(self, config) -> None:
        super().__init__(config=config)                      # GoalLevel0: 已建 Point + Goal
        self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]
        self.lidar_conf.max_dist = 3
        self.lidar_conf.num_bins = 16
        self.goal.size = 0.3                                 # 论文 goal_size
        self.goal.keepout = 0.305                            # 论文 goal_keepout
        self._add_geoms(Hazards(num=3, size=0.3, keepout=0.305))   # 随机位置


class GremlinLevel0(GoalLevel0):
    """config2 GremlinEnv: 5 hazard + 3 gremlin 的 goal 任务 (更大场地)。"""

    def __init__(self, config) -> None:
        super().__init__(config=config)
        self.placements_conf.extents = [-2, -2, 2, 2]        # 论文 placements_extents=±2
        self.lidar_conf.max_dist = 5                         # 论文 lidar_max_dist=5
        self.lidar_conf.num_bins = 16
        self.goal.size = 0.3
        self.goal.keepout = 0.305
        self._add_geoms(Hazards(num=5, size=0.3, keepout=0.305))
        self._add_mocaps(Gremlins(num=3, travel=0.35, keepout=0.4))  # 论文 gremlins_*


class DynamicButtonLevel0(ButtonLevel0):
    """config3 DynamicButtonEnv: 6 个随机 button 的 button 任务 (无 hazard/gremlin)。"""

    def __init__(self, config) -> None:
        super().__init__(config=config)
        self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]
        self.lidar_conf.max_dist = 3
        self.lidar_conf.num_bins = 16
        self.buttons.num = 6                                 # 论文 buttons_num=6
        self.buttons.size = 0.1
        self.buttons.keepout = 0.2
        self.buttons.is_constrained = True                   # 论文 constrain_buttons


# ============================================================ 注入 + 注册 ============================================================
# 把 4 个类注入 safety_gymnasium.tasks 命名空间 (Builder._get_task 靠 getattr(tasks, 类名) 取类)。
_PAPER_TASKS = {
    'SimpleButtonLevel0': SimpleButtonLevel0,
    'DynamicLevel0': DynamicLevel0,
    'GremlinLevel0': GremlinLevel0,
    'DynamicButtonLevel0': DynamicButtonLevel0,
}
for _cname, _cls in _PAPER_TASKS.items():
    setattr(_sg_tasks, _cname, _cls)

# 论文短名 → 注册的 safety_gymnasium env id。task_id 经 get_task_class_name 解析到上面类名:
#   'SafetyPointSimpleButton0-v0'  → 'SimpleButtonLevel0'   (findall[2:]=['Simple','Button0']→'SimpleButton0')
#   'SafetyPointDynamic0-v0'       → 'DynamicLevel0'
#   'SafetyPointGremlin0-v0'       → 'GremlinLevel0'
#   'SafetyPointDynamicButton0-v0' → 'DynamicButtonLevel0'
PAPER_ENV_IDS = {
    'SimpleButton':  'SafetyPointSimpleButton0-v0',
    'Dynamic':       'SafetyPointDynamic0-v0',
    'Gremlin':       'SafetyPointGremlin0-v0',
    'DynamicButton': 'SafetyPointDynamicButton0-v0',
}

_REGISTERED = False


def _register_all(max_episode_steps=1000):
    """把 4 个论文 env id 注册到 safety_gymnasium 注册表 (幂等, 重复注册忽略)。"""
    global _REGISTERED
    if _REGISTERED:
        return
    for _eid in PAPER_ENV_IDS.values():
        try:
            safety_gymnasium.register(
                id=_eid,
                entry_point='safety_gymnasium.builder:Builder',
                kwargs={'config': {'agent_name': 'Point'}, 'task_id': _eid},
                max_episode_steps=max_episode_steps)
        except Exception:                                    # 已注册 → 忽略
            pass
    _REGISTERED = True


_register_all()                                              # import 即注册


def make_paper_env(name, **kwargs):
    """name ∈ {SimpleButton,Dynamic,Gremlin,DynamicButton} (或直接 env id) → safety_gymnasium env。"""
    _register_all()
    eid = PAPER_ENV_IDS.get(name, name)
    return safety_gymnasium.make(eid, **kwargs)


# ============================================================ 自测 (python envs/paper_envs.py) ============================================================
if __name__ == '__main__':
    import numpy as np
    for _key in PAPER_ENV_IDS:
        try:
            env = make_paper_env(_key)
            obs, info = env.reset(seed=0)
            od = int(np.asarray(obs).size)
            ad = int(np.prod(env.action_space.shape))
            n = getattr(env.unwrapped, 'num_steps', None)
            max_c = 0.0; cost_src = set()
            for t in range(400):                             # 400 步随机: 尽量触发 cost
                a = env.action_space.sample()
                obs, r, c, term, trunc, info = env.step(a)
                max_c = max(max_c, float(c))
                cost_src |= {k for k in info if k.startswith('cost_') and info.get(k, 0)}
                if term or trunc:
                    obs, info = env.reset()
            print(f"[OK] {_key:14s} id={PAPER_ENV_IDS[_key]:32s} obs={od} act={ad} "
                  f"horizon={n} max_cost_400={max_c:.2f} cost_src={sorted(cost_src)}")
            env.close()
        except Exception as e:
            import traceback
            print(f"[FAIL] {_key}: {e}")
            traceback.print_exc()
