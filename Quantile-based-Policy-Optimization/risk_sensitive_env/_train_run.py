# -*- coding: utf-8 -*-
"""
_train_run.py —— DQCACBetaGPU / QCPO 调参【临时】runner (用完即删)

完全复刻 notebook 的 DQCACBetaGPUArgs / QCPO 配置, 支持命令行 key=value 覆盖,
进程内捕获每迭代 wandb.log 指标, 跑完做一次"新鲜"评估并对照目标判据给诊断。
不连 wandb 云 (WANDB_MODE=disabled)。

用法:
    python _train_run.py DQCACBetaGPU tag=base
    python _train_run.py DQCACBetaGPU lambda_min=0.0 tag=nofloor num_iterations=400 num_envs=256
    python _train_run.py QCPO tag=ref num_episodes=3000
"""
import os, sys, json, time
os.environ['WANDB_MODE'] = 'disabled'          # 不连云, 不需登录

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import numpy as np
import torch
import random
import wandb

# ---------------- 进程内捕获 wandb.log ----------------
# 注意: wandb.init() 会把 wandb.log 重新绑定到 run 的方法, 覆盖这里的 patch。
# 因此真正生效的 patch 必须在 agent 构造(其内部 wandb.init)之后再打一次(见 main)。
_LOG = []
def _cap(d, *a, **k):
    try:
        _LOG.append({kk: (float(vv) if isinstance(vv, (int, float)) else vv) for kk, vv in d.items()})
    except Exception:
        _LOG.append(dict(d))
wandb.log = _cap

from agents import QCPO, QPO, DQCACBetaGPU, DQCACBeta, QCPOGPU
from envs import RiskSensitiveEnv


# ============================================================ 参数 ============================================================
def base_args(algo):
    """返回与 notebook 对应 cell 完全一致的 args 命名空间。"""
    a = argparse_ns()
    a.env_name = 'RiskSensitiveEnv'
    a.seed = 0
    a.algo_name = algo
    a.q_alpha = 0.25
    a.gamma = 0.99
    a.log_interval = 50
    a.est_interval = 100
    a.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a.wandb_dir = None

    if algo == 'DQCACBetaGPU':
        a.num_envs = 512
        a.num_iterations = 1500        # critic 校准 + mean 收敛需 ~1200+
        a.beta = 0.90                  # 配合校准 critic 推到 QCPO 边界 (mean≈12)
        a.quantile_threshold = 5.0
        a.init_std = float(np.sqrt(1e-1))
        a.theta_a = (10000 ** 0.9) * 2e-4
        a.theta_b = 10000
        a.theta_c = 0.9
        a.q_a = (10000 ** 0.6) * 1e-2
        a.q_b = 10000
        a.q_c = 0.6
        a.lambda_a = 0.3
        a.lambda_b = 5000
        a.lambda_c = 0.1
        a.lambda_max = 50.0
        a.lambda_min = 0.0             # critic 校准好时均衡 λ 健康, 不需 floor
        a.outer_interval = 1
        a.num_quantiles = 32
        a.huber_kappa = 0.1            # 【critic-dual 关键】κ=0.1 近纯分位数回归→critic 无偏 (κ=1 低估方差→低估尾部 P)
        a.critic_hidden = [64, 64]
        a.critic_lr = 1e-3
        a.target_tau = 0.05            # 加快10x: critic-dual 下消除 critic 滞后→避免 λ 极限环
        a.target_update_interval = 1
        a.n_step = 1                   # 原生 per-transition; ≥2 会失控
        a.num_action_samples = 4
        a.updates_per_episode = 10
        a.advantage_norm = 'qcpo'
        a.norm_ema_decay = 0.1
        a.warmup_iters = 30            # 先把 critic 练到校准再开 actor/λ, 跳过初期 Ghat≈1 假信号
        a.entropy_coef = 0.0
        a.actor_grad_clip = 100.0
        a.critic_grad_clip = 10.0
    elif algo == 'QCPO':
        a.max_episode = 100000
        a.init_std = float(np.sqrt(1e-1))
        a.theta_a = (10000 ** 0.9) * 1e-3
        a.theta_b = 10000
        a.theta_c = 0.9
        a.q_a = (10000 ** 0.6) * 1e-2
        a.q_b = 10000
        a.q_c = 0.6
        a.lambda_a = 0.3
        a.lambda_b = 5000
        a.lambda_c = 0.1
        a.outer_interval = 10
        a.quantile_threshold = 5.0
    elif algo == 'QCPOGPU':
        # 全 GPU 向量化 QCPO 参考: 与 QCPO 算法对齐 (theta 用 QCPO 的 1e-3 系数, decay=0.01, 不裁梯度),
        # 数据流/更新结构与 DQCACBetaGPU 对齐 (B 路并行 + updates_per_iteration 复用 batch)。
        a.num_envs = 512
        a.num_iterations = 800
        a.quantile_threshold = 5.0
        a.init_std = float(np.sqrt(1e-1))
        a.theta_a = (10000 ** 0.9) * 1e-3      # QCPO 原始 actor 学习率系数
        a.theta_b = 10000
        a.theta_c = 0.9
        a.lambda_a = 0.3
        a.lambda_b = 5000
        a.lambda_c = 0.1
        a.outer_interval = 1
        a.updates_per_episode = 10             # 每迭代复用 batch 更新次数 (= DQCACBetaGPU 同款)
        a.norm_ema_decay = 0.01                # return_rms decay (= QCPO 默认)
        a.actor_grad_clip = 0.0                # QCPO 不裁 actor 梯度
        a.warmup_rms_iters = 2                 # return_rms 预热迭代数
    return a


class argparse_ns:
    pass


def cast(v):
    """把命令行字符串转成合适类型。"""
    if v.lower() in ('none', 'null'):
        return None
    if v.lower() == 'true':
        return True
    if v.lower() == 'false':
        return False
    try:
        if '.' not in v and 'e' not in v.lower():
            return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v                                # 保留字符串 (如 qcpo / separate)


# ============================================================ 评估 (fresh empirical_prob + critic cdf_initial) ============================================================
def eval_full(agent, env, num_episodes, gamma, q_alpha, q):
    """
    跑 num_episodes 条评估轨迹, 在【同一批 eval 轨迹】上算:
      mean / Q_alpha / std        : 来自回报 U(τ)
      empirical_prob              : 这批轨迹里 U(τ)<=q 的比例 (真值)
      cdf_initial                 : 对每条轨迹的 (s0,a0), 让 critic 估 P(Z<=q|s0,a0)=mean_i 1{ψ_i<=q},
                                    再对 num_episodes 条求平均 (critic 预测; 无 critic 的算法返回 None)
    二者在同一批 eval 轨迹上 → 干净的 critic 校准对比 (empirical=真值, cdf_initial=critic预测)。
    """
    rets, s0s, a0s = [], [], []
    with torch.no_grad():
        for _ in range(num_episodes):
            s = env.reset()
            if isinstance(s, tuple):
                s = s[0]
            s0_flat = np.asarray(s, dtype=np.float32).flatten().copy()  # 本条轨迹初始状态 s0
            G, disc, first = 0.0, 1.0, True
            while True:
                if isinstance(s, torch.Tensor):
                    s = s.cpu().numpy()
                a = agent.select_action(s.flatten())
                if first:                                          # 记录本条轨迹实际采取的首个动作 a0
                    s0s.append(s0_flat)
                    a0s.append(np.asarray(a, dtype=np.float32).flatten())
                    first = False
                s2, r, done, _ = env.step(a)
                G += disc * r
                disc *= gamma
                if done:
                    break
                s = s2
            rets.append(G)
    rets = np.asarray(rets, dtype=np.float64)
    mean_e = float(rets.mean())
    q_e = float(np.percentile(rets, q_alpha * 100))
    P_e = float(np.mean(rets <= q))                                # eval 经验违约概率 (真值)
    std_e = float(rets.std())

    # critic 在【这批 eval 轨迹的 (s0,a0)】上估计违约概率/回报分布矩 → eval 校准诊断 (仅 critic-based 算法)
    #  用同一批 eval (s0,a0) 让 critic 给出: ① P(Z<=q|s0,a0) (cdf_init) ② E[Z|s0,a0] ③ std(Z|s0,a0)
    #  与 eval 真值对比: cdf_init↔P_e(真违约率), pred_mean↔mean_e, pred_std↔std_e
    #  → 严谨判断 critic 学得好不好 (不依赖训练期含噪的 empirical_prob)
    cdf_init = pred_mean = pred_std = None
    if hasattr(agent, 'critic'):
        dev = agent.device
        S0 = torch.as_tensor(np.stack(s0s), dtype=torch.float32, device=dev)   # [N_ep, sd]
        A0 = torch.as_tensor(np.stack(a0s), dtype=torch.float32, device=dev)   # [N_ep, ad]
        if getattr(agent, 'critic_step_feature', False):           # step 增广: s0 是第 0 步 → 拼 t/n=0
            S0 = torch.cat([S0, torch.zeros(S0.shape[0], 1, device=dev)], dim=1)
        with torch.no_grad():
            psi = agent.critic(S0, A0)                              # [N_ep, N_q] 各轨迹回报分布分位数
            # 每条轨迹: P(Z<=q|s0,a0)=mean_i 1{ψ_i<=q}; 再对 N_ep 条平均
            cdf_init = float((psi <= q).float().mean(dim=1).mean().item())
            pred_mean = float(psi.mean().item())                   # critic 估 E[Z|s0,a0] (对 N_q 取均值再对 N_ep 平均)
            pred_std = float(psi.std(dim=1).mean().item())         # critic 估 std(Z|s0,a0) (每条轨迹分位数 std, 再对 N_ep 平均)
    return mean_e, q_e, P_e, std_e, cdf_init, pred_mean, pred_std


# ============================================================ 诊断打印 ============================================================
def seg(series, n=8):
    a = np.asarray([x for x in series if x is not None and np.isfinite(x)], dtype=float)
    if len(a) == 0:
        return 'no-data'
    L = max(1, len(a) // n)
    return '  '.join(f'{np.mean(a[i*L:(i+1)*L]):8.3f}' for i in range(n))


def last_seg(series, frac=8):
    a = np.asarray([x for x in series if x is not None and np.isfinite(x)], dtype=float)
    if len(a) == 0:
        return float('nan')
    return float(np.mean(a[-max(1, len(a)//frac):]))


def col(key):
    return [d.get(key) for d in _LOG if key in d]


def main():
    algo = sys.argv[1] if len(sys.argv) > 1 else 'DQCACBetaGPU'
    overrides = {}
    tag = 'run'
    num_eval = 3000
    for arg in sys.argv[2:]:
        if '=' not in arg:
            continue
        k, v = arg.split('=', 1)
        if k == 'tag':
            tag = v
        elif k == 'num_episodes':
            num_eval = int(v)
        else:
            overrides[k] = cast(v)

    args = base_args(algo)
    for k, v in overrides.items():
        setattr(args, k, v)

    # 随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"\n{'='*78}\n[{tag}] algo={algo}  device={args.device}  overrides={overrides}")
    if algo == 'DQCACBetaGPU':
        print(f"  B={args.num_envs} iters={args.num_iterations} n_step={args.n_step} "
              f"tau={args.target_tau} lambda_a={args.lambda_a} lambda_min={args.lambda_min} "
              f"adv_norm={args.advantage_norm} K={args.num_action_samples} "
              f"upd/iter={args.updates_per_episode} actor_clip={args.actor_grad_clip}")

    env = RiskSensitiveEnv(n=10)
    AgentCls = {'DQCACBetaGPU': DQCACBetaGPU, 'QCPOGPU': QCPOGPU,
                'QCPO': QCPO, 'QPO': QPO, 'DQCACBeta': DQCACBeta}[algo]

    t0 = time.time()
    agent = AgentCls(args, env)
    wandb.log = _cap                            # 关键: agent.__init__ 内的 wandb.init 把 wandb.log 重绑了, 这里再夺回
    agent.train()
    train_t = time.time() - t0

    q = float(getattr(args, 'quantile_threshold', 5.0))
    mean_e, q_e, P_e, std_e, cdf_init_eval, pred_mean_eval, pred_std_eval = \
        eval_full(agent, env, num_eval, args.gamma, args.q_alpha, q)
    print(f"\n  train_time {train_t:.1f}s  ({len(_LOG)} logs captured)")

    # ---- 趋势 (DQCACBetaGPU 全有; QCPOGPU 缺的键显示 no-data) ----
    if algo in ('DQCACBetaGPU', 'QCPOGPU'):
        rows = [
            ('mean',        'disc_reward/aver_reward'),
            ('q_est',       'disc_reward/quantile_reward'),
            ('P(Z<=q)',     'constraint/empirical_prob'),
            ('cdf_init Ghat','constraint/cdf_estimate_initial'),
            ('calib_err',   'constraint/cdf_calibration_error'),
            ('lambda',      'lambda/value'),
            ('return_std',  'disc_reward/return_std'),
            ('pred_mean',   'critic/pred_return_mean'),
            ('pred_std',    'critic/pred_return_std'),
            ('risk r',      'action/avg_risk_episode'),
            ('w_std',       'actor/w_std'),
            ('norm_loss',   'critic/quantile_huber_loss_normalized'),
            ('sig_ret',     'norm/return_sigma_ema'),
            ('sig_c',       'norm/constraint_sigma_ema'),
        ]
        print("\n  ----- 8-seg trend (start -> end) -----")
        for label, key in rows:
            print(f"  {label:14s}: {seg(col(key))}")

    # ---- eval + verdict ----
    print(f"\n  ----- eval ({num_eval} episodes) -----")
    print(f"  mean={mean_e:.3f}  Q_{args.q_alpha}={q_e:.3f}  P(Z<=q={q})={P_e:.3f}  std={std_e:.3f}")
    # critic 校准报告 (后训练大样本评估; 不依赖训练期含噪 empirical_prob):
    #   严谨判据 = critic 在 eval (s0,a0) 的预测 vs 同批 eval 真值, 三项都对上才算 critic 学好了
    if cdf_init_eval is not None:
        binom_se = float(np.sqrt(max(P_e, 1e-6) * (1 - P_e) / max(num_eval, 1)))  # 真值 P 的二项标准误 (噪声基线)
        print(f"  [critic calibration vs eval ground-truth ({num_eval} ep, P真值SE=+-{binom_se:.4f})]")
        print(f"    P(Z<=q):  critic={cdf_init_eval:.3f}  truth={P_e:.3f}  bias={cdf_init_eval-P_e:+.3f}")
        print(f"    E[Z|s0]:  critic={pred_mean_eval:.3f}  truth={mean_e:.3f}  bias={pred_mean_eval-mean_e:+.3f}")
        print(f"    std[Z]:   critic={pred_std_eval:.3f}  truth={std_e:.3f}  bias={pred_std_eval-std_e:+.3f}")
    if algo == 'DQCACBetaGPU':
        cdf_last = cdf_init_eval if cdf_init_eval is not None else last_seg(col('constraint/cdf_estimate_initial'))
        lam_last = last_seg(col('lambda/value'))
        lam_series = np.asarray([x for x in col('lambda/value') if x is not None], float)
        # 只看后半程的 λ 最小值: 前半程从 0 爬升是正常的, 不算"塌缩"
        lam_2nd = lam_series[len(lam_series)//2:] if len(lam_series) >= 4 else lam_series
        lam_min_2nd = float(np.min(lam_2nd)) if len(lam_2nd) else float('nan')
        nl_last = last_seg(col('critic/quantile_huber_loss_normalized'))
        rstd_last = last_seg(col('disc_reward/return_std'))
        print(f"\n  ----- goal verdict -----")
        def verdict(ok, name, detail):
            print(f"  [{'OK ' if ok else 'BAD'}] {name:24s} {detail}")
        verdict(abs(cdf_last - 0.25) <= 0.05, 'critic Ghat~=0.25', f'Ghat={cdf_last:.3f} (target 0.25)')
        verdict(P_e <= 0.25 + 0.015, 'eval P(Z<=q)<=a', f'{P_e:.3f} (a=0.25)')
        verdict(q_e >= q * 0.98, 'eval quantile>=q', f'Q={q_e:.3f} (q={q})')
        verdict(11.0 <= mean_e <= 13.0, 'mean~=QCPO(~12)', f'mean={mean_e:.3f}')
        # λ 健康度: critic-dual 下校准好的 critic 让均衡 λ 很小(~0.2)且小幅振荡(偶尔触0是投影边界, 不是塌缩),
        # 故看【后半程均值】是否落在 (0.02, 0.98*max) 而非旧经验-dual régime 的 >0.5 阈值; 真正的不健康=runaway/pegged。
        lam_mean_2nd = float(np.mean(lam_2nd)) if len(lam_2nd) else float('nan')
        verdict(0.02 < lam_mean_2nd < 0.98 * args.lambda_max, 'lambda healthy(2nd half)',
                f'lam_mean_2nd={lam_mean_2nd:.3f} lam_min_2nd={lam_min_2nd:.3f} lam_last={lam_last:.2f} '
                f'(small equilibrium ok; bad=runaway/pegged {args.lambda_max})')
        verdict(np.isfinite(nl_last) and nl_last < 50, 'loss not exploded', f'norm_loss={nl_last:.3f}')
        verdict(np.isfinite(rstd_last) and rstd_last < 20, 'variance stable', f'return_std={rstd_last:.3f}')

    # ---- 存盘 ----
    out = {
        'tag': tag, 'algo': algo, 'overrides': {k: (str(v)) for k, v in overrides.items()},
        'train_seconds': train_t,
        'eval': {'mean': mean_e, 'quantile': q_e, 'P_Zleq_q': P_e, 'std': std_e, 'q': q,
                 'critic_cdf_init': cdf_init_eval, 'critic_pred_mean': pred_mean_eval,
                 'critic_pred_std': pred_std_eval},
        'history': _LOG,
    }
    os.makedirs(os.path.join(BASE_DIR, '_runs'), exist_ok=True)
    fp = os.path.join(BASE_DIR, '_runs', f'{algo}_{tag}.json')
    with open(fp, 'w', encoding='utf-8') as f:
        json.dump(out, f)
    print(f"\n  saved: {fp}\n{'='*78}")


if __name__ == '__main__':
    main()
