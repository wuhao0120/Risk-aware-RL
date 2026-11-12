"""SimpleButton: the frozen two-button, three-hazard paper task."""
import os
os.environ.setdefault('MUJOCO_GL', 'disable')
import safety_gymnasium
from safety_gymnasium import tasks as _sg_tasks
from safety_gymnasium.assets.geoms import Hazards
from safety_gymnasium.tasks.safe_navigation.button.button_level0 import ButtonLevel0
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

_PAPER_TASKS = {'SimpleButtonLevel0': SimpleButtonLevel0}
for _cname, _cls in _PAPER_TASKS.items():
    setattr(_sg_tasks, _cname, _cls)
PAPER_ENV_IDS = {'SimpleButton': 'SafetyPointSimpleButton0-v0'}
_REGISTERED = False


def _register_all(max_episode_steps=1000):
    """把 SimpleButton env id 注册到 safety_gymnasium 注册表 (幂等, 重复注册忽略)。"""
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
    """name = SimpleButton (或直接 env id) → safety_gymnasium env。"""
    _register_all()
    eid = PAPER_ENV_IDS.get(name, name)
    return safety_gymnasium.make(eid, **kwargs)


