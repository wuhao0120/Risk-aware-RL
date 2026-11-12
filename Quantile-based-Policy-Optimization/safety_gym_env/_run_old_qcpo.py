# -*- coding: utf-8 -*-
"""
原版 QCPO 轻量训练驱动 (qcpo_ref conda 环境: py3.8 + torch1.5.1cpu + mujoco-py2.0 + 旧 safety-gym + rlpyt)。

完全复用原仓库 train_qcpo.build_and_train (算法/采样/日志一行不改), 仅替代 launch_qcpo.py 的
子进程 launcher: 本脚本自己写 variant_config.json + 算 affinity code, 然后同进程调用。
超参 = launch_qcpo.py 缺省 (cost_limit=15, target_prob=0.2, pid_Ki=0.1, LSTM config), 只缩预算。

用法: python _run_old_qcpo.py <EnvName: SimpleButton|Dynamic|Gremlin|DynamicButton>
      [n_steps=1e6] [n_cpu=16] [log_root=_ref_runs] [log_interval_steps=5e4]
输出: <log_root>/<EnvName>/run_0/{progress.csv, traj_summary.csv, params.json}
"""
import os
import sys

sys.path.insert(0, '/vepfs-mlp2/c20250510/251204033/dependencies/rlpyt')

ENV_IDS = {
    'SimpleButton': 'SimpleButtonEnv-v0',
    'Dynamic': 'DynamicEnv-v0',
    'Gremlin': 'GremlinEnv-v0',
    'DynamicButton': 'DynamicButtonEnv-v0',
}

env_key = sys.argv[1] if len(sys.argv) > 1 else 'SimpleButton'
n_steps = float(sys.argv[2]) if len(sys.argv) > 2 else 1e6
n_cpu = int(sys.argv[3]) if len(sys.argv) > 3 else 16
log_root = sys.argv[4] if len(sys.argv) > 4 else '_ref_runs'
log_interval = float(sys.argv[5]) if len(sys.argv) > 5 else 5e4

from rlpyt.utils.launching.variant import save_variant
from rlpyt.projects.qcpo.experiments.train_qcpo import build_and_train

# ---- variant (对应 launch_qcpo.py 的 variant_levels; 其余走 config_qcpo.py 缺省) ----
variant = {
    'env': {'id': ENV_IDS[env_key]},
    'algo': {'cost_limit': 15, 'target_outage_prob': 0.2, 'pid_Ki': 0.1},
    'runner': {'n_steps': n_steps, 'log_interval_steps': log_interval},
}
log_dir = os.path.abspath(os.path.join(log_root, env_key))
os.makedirs(log_dir, exist_ok=True)
save_variant(variant, log_dir)

# ---- affinity: launcher 同款 code (n_cpu 核, 0 gpu, 1 core/worker), run_slot=0 ----
slot_affinity_code = f'0slt_{n_cpu}cpu_0gpu_1cpr_1cpw_0hto'
print(f'[old-qcpo] env={env_key} n_steps={n_steps:.0f} affinity={slot_affinity_code} log={log_dir}')

build_and_train(slot_affinity_code=slot_affinity_code, log_dir=log_dir,
                run_ID='0', config_key='LSTM')
