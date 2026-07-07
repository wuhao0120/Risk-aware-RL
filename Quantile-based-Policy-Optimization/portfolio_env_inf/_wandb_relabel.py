# -*- coding: utf-8 -*-
"""
_wandb_relabel.py —— 一次性整理 wandb 项目 portfolio_inf 的实验记录 (2026-06-13)。

目标 (用户要求"实验记录一目了然: 什么算法对比什么 + 最终评估 mean/Q/P"):
    1. 统一命名: {实验号}-{算法}-B{批量}x{迭代}[-变体]-s{seed}
    2. 统一分组: E1_same_budget_B512x1500 (同预算图) / E2_full_convergence (全量图)
    3. notes 回填: 中文实验说明 + 最终评估一句话 (来自 _runs/*.json 的统一评估器结果)
    4. summary 回填: eval/constraint_ok (P≤α+0.015) + budget/total_env_steps
    5. 2 个断链垃圾 run 改名 zz-trash- 前缀沉底 (不删除, 由用户在网页上自行决定)

幂等: 重复运行只是再写一遍同样的字段。
"""
import json
import os

import wandb

BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = '1206052611-ecnu/portfolio_inf'

# 与 run_experiment.py EXP_META 保持一致 (新 run 自动带, 旧 run 由本脚本补)
E1_NOTE = ('实验一·同预算对比: 四算法统一 B=512 × 1500迭代 × n=100 = 7680万env步, '
           '同探索噪声σ=0.35、同q=1.0。对比: DQCAC vs QPO/QCPO/QPPO。预期: '
           'trajectory-level基线在小批量下信噪比不足、难收敛 —— 这正是 DQCAC '
           'per-transition TD 样本效率优势的直接证据。')
E2_NOTE = ('实验二·全量收敛对比: 各算法用各自调优配置跑到(近)收敛, 比渐近性能与所需样本量。'
           '对比: DQCAC(B512×1500=76.8M步) vs QPO(B4096×3000) / QPPO(B2048×3000) / '
           'QCPO调优(B4096×6000/12000, theta_lr0=0.12)。')
ALPHA, Q = 0.25, 1.0                                       # 统一口径 (α / 约束阈值 q)


def eval_line(j):
    """从结果 JSON 拼"最终评估"一句话 (与 run_experiment.py 训练后追加的格式一致)。"""
    e = j['eval']
    s = (f"最终评估(N={e['num_episodes']}): mean={e['mean']:.3f}, "
         f"Q{ALPHA:g}={e['quantile']:.3f}, P(Z<={Q:g})={e['empirical_prob']:.3f}, "
         f"std={e['std']:.3f}")
    if e.get('cdf_initial') is not None:                   # DQCAC: critic 校准偏差
        s += f", critic偏差={e['cdf_initial'] - e['empirical_prob']:+.3f}"
    return s


# (run_id, 新名称, 分组, 标签, 实验说明, 结果JSON文件, 总env步数)
RELABEL = [
    ('pzczfvtc', 'E1-QPO-B512x1500-s0', 'E1_same_budget_B512x1500',
     ['E1', 'QPO', 'B512', 'iters1500', 's0'], E1_NOTE,
     'QPO_uni_s0.json', 512 * 1500 * 100),
    ('8jwnch50', 'E1-QCPO-B512x1500-s0', 'E1_same_budget_B512x1500',
     ['E1', 'QCPO', 'B512', 'iters1500', 's0'], E1_NOTE,
     'QCPO_uni_s0.json', 512 * 1500 * 100),
    # DQCAC seed0 同时是 E1 的锚点和 E2 的代表 → 名带 E1E2, 双标签, 归 E1 组
    ('tjdsxwxs', 'E1E2-DQCAC-B512x1500-s0', 'E1_same_budget_B512x1500',
     ['E1', 'E2', 'DQCAC', 'B512', 'iters1500', 's0'],
     E1_NOTE + ' (本run同时作为实验二E2的DQCAC代表, 见标签)',
     'DQCAC_v4_s0.json', 512 * 1500 * 100),
    ('l4hegty4', 'E2-QPO-B4096x3000-s0', 'E2_full_convergence',
     ['E2', 'QPO', 'B4096', 'iters3000', 's0'], E2_NOTE,
     'QPO_v4_s0.json', 4096 * 3000 * 100),
    ('nsv7g71z', 'E2-QPPO-B2048x3000-s0', 'E2_full_convergence',
     ['E2', 'QPPO', 'B2048', 'iters3000', 's0'], E2_NOTE,
     'QPPO_v4_s0.json', 2048 * 3000 * 100),
    ('sa9x5z0x', 'E2-QCPO-B4096x6000-lr0.12-s0', 'E2_full_convergence',
     ['E2', 'QCPO', 'B4096', 'iters6000', 's0'], E2_NOTE,
     'QCPO_v4b_s0.json', 4096 * 6000 * 100),
    ('2nnipyy3', 'E2-DQCAC-B512x1500-s1', 'E2_full_convergence',
     ['E2', 'DQCAC', 'B512', 'iters1500', 's1'], E2_NOTE + ' (DQCAC多种子复现 seed1)',
     'DQCAC_v4_s1.json', 512 * 1500 * 100),
    ('skpn7iuv', 'E2-DQCAC-B512x1500-s2', 'E2_full_convergence',
     ['E2', 'DQCAC', 'B512', 'iters1500', 's2'], E2_NOTE + ' (DQCAC多种子复现 seed2)',
     'DQCAC_v4_s2.json', 512 * 1500 * 100),
    # 欠收敛存档: 不进对比图, zz 前缀沉底, 但保留 (记录"QCPO默认lr不够"这一调参证据)
    ('ef9x2a8v', 'zz-archive-QCPO-B4096x3000-lr0.06-s0', None,
     ['archive', 'QCPO'],
     '存档·不进对比图: QCPO 默认 lr0=0.06×3000迭代 欠收敛 (mean=3.10, 单调爬升未到顶), '
     '证明 QCPO 需要 lr0=0.12 + 更长迭代才公平 → 调优版见 E2-QCPO-B4096x6000-lr0.12-s0',
     'QCPO_v4_s0.json', 4096 * 3000 * 100),
]

# 断链垃圾 run: 只改名沉底+标注原因, 不删除 (是否删除由用户在 wandb 网页上决定)
TRASH = [
    ('ymp5s70u', 'zz-trash-QPO-uni-断链于iter250',
     '垃圾run·可删: 6/10深夜会话结束时后台链被杀, 只跑到 iter 250/1500, 无分析价值'),
    ('kr25xqwr', 'zz-trash-QPPO-uni-半截被停',
     '垃圾run·可删: 6/13 重启链后为改实验记录代码被中途停止的半截 QPPO, '
     '完整版见 E1-QPPO-B512x1500-s0'),
]


def main():
    api = wandb.Api()

    for rid, name, reason in TRASH:
        try:
            r = api.run(f'{PROJ}/{rid}')
            r.name = name
            r.notes = reason
            r.tags = ['trash']
            r.update()
            print(f'[垃圾沉底] {rid} -> {name}')
        except Exception as ex:
            print(f'[垃圾沉底失败] {rid}: {ex}')

    for rid, name, group, tags, note, jf, steps in RELABEL:
        r = api.run(f'{PROJ}/{rid}')
        old = r.name
        with open(os.path.join(BASE, '_runs', jf), encoding='utf-8') as f:
            j = json.load(f)
        e = j['eval']
        r.name = name
        r.notes = note + ' ‖ ' + eval_line(j)
        r.tags = tags
        if group is not None:
            try:
                r.group = group                            # 公开 API 支持改组则改
            except Exception as ex:
                print(f'      (group 不可改: {ex})')
        # summary 回填: 评估指标 + 约束达标 + 总预算 (runs 表可直接排序/筛选)
        r.summary.update({
            'eval/mean': e['mean'], 'eval/quantile': e['quantile'],
            'eval/std': e['std'], 'eval/empirical_prob': e['empirical_prob'],
            'eval/num_episodes': e['num_episodes'],
            'eval/constraint_ok': bool(e['empirical_prob'] <= ALPHA + 0.015),
            'budget/total_env_steps': steps,
        })
        r.update()
        print(f'[edit] {rid}  {old}  ->  {name}')

    print('done.')


if __name__ == '__main__':
    main()
