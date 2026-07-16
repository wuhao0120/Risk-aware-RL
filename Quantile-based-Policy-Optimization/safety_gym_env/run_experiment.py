# -*- coding: utf-8 -*-
"""
run_experiment.py —— safety_gym_env 统一实验入口 (QCPO / DQCAC / CALIB)。

环境对齐 NeurIPS'22 QCPO 论文 config0-3 (SimpleButton/Dynamic/Gremlin/DynamicButton,
由 envs/paper_envs.py 复现; 详见 DESIGN.md)。统一对比标准 (硬约定):
    1. 同一环境 (论文 config) + 同 γ=0.99 / cost_gamma / ω / d;
    2. 同一评估协议: 训练后 evaluate_policy_vec (整段=一 episode) 报 mean(R)/
       empirical_prob=P(Z≤q)/quantile_return (+DQCAC cost-critic 校准); 同探索 σ;
    3. 同一 wandb 方案 (project='safety_gym_qcrl', x 轴=progress/env_steps;
       约束键与 risk_sensitive 下尾口径对齐: Z=-C, q=-d)。

约束口径 (见 DESIGN.md; 双口径支持, 由 cost_gamma 切换):
    优化: max E[R]  s.t.  P(C ≥ d) ≤ ω,  C = Σ cost_gamma^t c_t
          ≡ max E[R]  s.t.  P(Z ≤ q) ≤ α,  Z=-C, q=-d, α=ω。
    - 默认【未折扣】论文口径: cost_gamma=1.0, d=15, ω=0.2 (对齐 launch_qcpo.py 的
      cost_limit=15/target_prob=0.2)。此时 C=Σc, QCPO 指示函数直接落在论文约束变量上;
      DQCAC 自动切 episodic 配方 (episode 末不 bootstrap + critic_step_feature=True)。
    - 折扣口径 (开关): --set cost_gamma=0.99 cost_limit=<CALIB 校准值>, DQCAC 回到
      continuing 截断恒 bootstrap 配方 (γc^1000≈4e-5, 残差可忽略)。

用法:
    MUJOCO_GL=egl python run_experiment.py --algo DQCAC --env SimpleButton
    MUJOCO_GL=egl python run_experiment.py --algo CALIB --env SimpleButton     # 校准 d
    MUJOCO_GL=egl python run_experiment.py --algo QCPO  --env Gremlin --set num_iterations=50 --wandb_mode disabled
    MUJOCO_GL=egl python run_experiment.py --algo QCPO_REF --env SimpleButton  # NIPS'22 基线移植版
算法名: QCPO / DQCAC / QCPO_REF / CALIB    环境名: SimpleButton / Dynamic / Gremlin / DynamicButton
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')                     # headless (须在 import safety_gymnasium 前)
import sys
import json
import time
import random
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import numpy as np
import torch


# ============================================================ 参数体系 ============================================================
class argparse_ns:
    """轻量命名空间 (与 portfolio_env_inf/run_experiment.py 同款)。"""
    pass


def cast(v):
    """命令行字符串 → 合适类型 (int/float/bool/None/str)。"""
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
        return v


# ============================================================ 论文环境表 ============================================================
# name → (SafetyEnv env_id, 默认约束)。默认【未折扣】论文口径: d=15 / ω=0.2
# (launch_qcpo.py: cost_limit=15, target_prob=0.2), C=Σc (cost_gamma=1)。
# 折扣口径按需覆盖: --set cost_gamma=0.99 cost_limit=<CALIB 值>。
PAPER_ENVS = {
    'SimpleButton':  dict(env_id='SimpleButton',  omega=0.2, cost_limit=15.0),
    'Dynamic':       dict(env_id='Dynamic',       omega=0.2, cost_limit=15.0),
    'Gremlin':       dict(env_id='Gremlin',       omega=0.2, cost_limit=15.0),
    'DynamicButton': dict(env_id='DynamicButton', omega=0.2, cost_limit=15.0),
}


def base_args(algo, seed, device, env_key):
    """两算法默认超参 (移植自 portfolio_env_inf 已验证配置; safety-gym CMDP 适配)。"""
    a = argparse_ns()
    ev = PAPER_ENVS[env_key]
    # -------------------- 统一字段 (两算法一致) --------------------
    a.env_name = env_key
    a.env_id = ev['env_id']
    a.seed = seed
    a.algo_name = algo
    a.q_alpha = ev['omega']          # ω 目标 outage 概率 P(C≥d)≤ω
    a.cost_limit = ev['cost_limit']  # d 约束阈值 (默认未折扣口径 d=15, 论文对齐)
    a.gamma = 0.99                   # 奖励折扣 γ (论文一致)
    a.cost_gamma = 1.0               # cost 折扣 γc: 1.0=未折扣论文口径 (默认); 0.99=折扣口径开关
    a.horizon = 1000                 # 一段 rollout 步长 T (论文 safety-gym episode 长度)
    a.num_envs = 16                  # 并行 env 数 B (mp 后端: B 个 worker 进程, 128 核可加大)
    a.vec_backend = 'mp'             # 'mp' 多进程并行 (默认) / 'sync' 单进程串行 (调试)
    a.num_iterations = 300           # 迭代数 (最小 pipeline 用; 全量另调)
    a.updates_per_episode = 10       # 每迭代内层更新次数
    a.actor_updates_per_episode = 1  # 非 PPO actor 每个 rollout 只做一次严格 on-policy 更新
    a.init_std = 0.5                 # 策略探索 σ (动作空间 [-1,1])
    a.actor_hidden = [256, 256]      # MLP 策略 (观测 60-76 维)
    a.log_interval = 10
    a.device = device
    a.wandb_dir = BASE_DIR
    a.wandb_project = 'safety_gym_qcrl'

    if algo == 'QCPO':
        # MC 轨迹级约束版。safety-gym 观测信息量足 → θ 默认 adam (非 portfolio 噪声特征情形)。
        a.theta_optimizer = 'adam'
        a.theta_b = 1000
        a.theta_c = 0.9
        a.theta_a = (a.theta_b ** a.theta_c) * 3e-4    # adam lr(0)=3e-4 (MLP 典型)
        a.lambda_a = 1.0                                # 快 λ
        a.lambda_b = 2000
        a.lambda_c = 0.1
        a.lambda_max = 50.0
        a.outer_interval = 1
        a.norm_ema_decay = 0.01
        a.actor_grad_clip = 1.0
        a.warmup_rms_iters = 2
        a.updates_per_episode = 5                       # 轨迹级复用 (新鲜度 vs 效率折中)
    elif algo == 'DQCAC':
        # per-transition 双 critic 版 (继承 portfolio DQCACBetaGPU 已验证配方)。
        a.beta = 0.95
        a.critic_step_feature = None                    # None → 跟随口径: 未折扣 episodic=True, 折扣=False
        a.theta_b = 10000
        a.theta_c = 0.9
        a.theta_a = (a.theta_b ** a.theta_c) * 2e-4     # actor lr≈近常数 2e-4 (已验证)
        a.lambda_a = 0.3
        a.lambda_b = 5000
        a.lambda_c = 0.1
        a.lambda_max = 50.0
        a.lambda_min = 0.0
        a.outer_interval = 1
        a.num_quantiles = 32
        a.huber_kappa = 0.1                             # 近纯分位回归 → critic 无偏 (已验证)
        a.critic_hidden = [256, 256]
        a.critic_lr = 1e-3
        a.target_tau = 0.05                             # 消除 critic 滞后 (已验证)
        a.target_update_interval = 1
        # n-step TD: T=1000 未折扣口径下 1-step 传播太慢 (600 updates 传不满 1000 步链,
        # 实测 cost-critic 在 s0 恒 0 → λ 不动)。n_step=100 → bootstrap 链长 10, 数百
        # updates 即可覆盖; on-policy 每迭代重采, n-step 和 + 截断 mask 均合法。
        a.n_step = 100
        a.num_action_samples = 4
        a.advantage_norm = 'qcpo'                       # EMA 归一化 (反 λ 卷绕, 已验证)
        a.norm_ema_decay = 0.1
        a.warmup_iters = 30                             # 先校准 cost-critic 再开 actor/λ
        a.entropy_coef = 0.0
        a.actor_grad_clip = 100.0
        a.critic_grad_clip = 10.0
        # reward actor 主干消融：默认保持旧 distributional；实验按 gae → gae_ppo 逐项打开。
        a.reward_actor_mode = 'distributional'
        a.gae_lambda = 0.97
        a.reward_advantage_norm = False                 # QCPO_refs 默认不标准化 reward advantage
        a.ppo_ratio_clip = 0.1
        a.reward_value_lr = 3e-4
        a.reward_value_grad_clip = 10.0
    elif algo == 'QCPO_REF':
        # NIPS'22 QCPO 移植版 (config_qcpo.py / launch_qcpo.py 论文缺省超参)。
        a.ref_lr = 1e-4
        a.value_loss_coeff = 1.0
        a.entropy_loss_coeff = 0.0
        a.clip_grad_norm = 1e4
        a.gae_lambda = 0.97
        a.minibatches = 1
        a.epochs = 8
        a.ratio_clip = 0.1
        a.cost_value_loss_coeff = 0.5
        a.ep_cost_ema_alpha = 0.0                       # 0 = 硬更新
        a.ep_outage_ema_alpha = 0.0
        a.ep_cost_eqa_alpha = 0.0
        a.cost_scale = 10.0                             # "yes 10."
        a.weibull_tail_prob = 0.3
        a.n_quantile = 25
        a.pid_Ki = 0.1
        a.sum_norm = True                               # L=(J_r+λJ_c)/(1+λ)
        a.diff_norm = False
        a.penalty_init = 0.0
        a.reward_scale = 1.0
        a.new_T = 100                                   # LSTM BPTT 段长 (论文 new_T)
        a.ref_hidden = [512, 512]
        a.lstm_size = 512
        a.lstm_skip = True
        a.ref_init_log_std = 0.0                        # 可学习 log_std, 初始 σ=1
        a.normalize_observation = True
        a.var_clip = 1e-6
    return a


# ============================================================ CALIB: cost 分布校准 ============================================================
def calibrate(args, num_episodes=64):
    """
    随机(未训练)策略在整段 horizon 上测 R / C(折扣) / Σc(未折扣) 分布 + VecEnv 耗时 →
    选约束阈值 d 的依据 (对应 risk/portfolio 的 _noisy_calib.py 方法论)。
    """
    from envs import SafetyEnv, make_vec_env
    from agents.vec_base import VecAgentBase

    env = SafetyEnv(args.env_id)
    base = VecAgentBase(args, env)                     # 未训练随机策略
    vec = make_vec_env(args.env_id, num_envs=args.num_envs, horizon=args.horizon,
                       device=args.device, ref_env=env, seed=args.seed + 100,
                       backend=getattr(args, 'vec_backend', 'mp'))
    B, n = vec.B, vec.n
    rounds = max(1, int(np.ceil(num_episodes / B)))
    Rs, Cs, Cus = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for _ in range(rounds):
            s = vec.reset()
            R = torch.zeros(B, device=args.device)
            C = torch.zeros(B, device=args.device)
            Cu = torch.zeros(B, device=args.device)
            dr = dc = 1.0
            for t in range(n):
                a = base._sample_actions(s)
                s, r, c, done = vec.step(a)
                R += dr * r; dr *= args.gamma
                C += dc * c; dc *= args.cost_gamma
                Cu += c
            Rs.append(R); Cs.append(C); Cus.append(Cu)
    dt = time.time() - t0
    R = torch.cat(Rs).cpu().numpy(); C = torch.cat(Cs).cpu().numpy(); Cu = torch.cat(Cus).cpu().numpy()

    print(f"\n===== CALIB {args.env_id} (random policy, E={R.shape[0]}) =====")
    print(f"timing: {dt:.1f}s, {dt/rounds:.2f}s/(B={B},T={n}), {B*n*rounds/dt:.0f} env-steps/s")
    print(f"R:        mean={R.mean():.3f} std={R.std():.3f} [{R.min():.2f},{R.max():.2f}]")
    print(f"C_disc:   mean={C.mean():.3f} std={C.std():.3f} pct[50,80,90,95]={np.round(np.percentile(C,[50,80,90,95]),2)}")
    print(f"C_undisc: mean={Cu.mean():.3f} std={Cu.std():.3f} pct[50,80,90,95]={np.round(np.percentile(Cu,[50,80,90,95]),2)}")
    print("outage 网格 (选 d):")
    for d in [1, 2, 3, 5, 8, 12, 20, 25, 30]:
        print(f"  d={d:>4g}: P(C_disc>=d)={np.mean(C>=d):.3f}  P(C_undisc>=d)={np.mean(Cu>=d):.3f}")
    return {'R_mean': float(R.mean()), 'C_disc': C.tolist()[:0], 'timing_s': dt}


# ============================================================ 主流程 ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='DQCAC',
                        choices=['QCPO', 'DQCAC', 'QCPO_REF', 'CALIB'])
    parser.add_argument('--env', type=str, default='SimpleButton', choices=list(PAPER_ENVS.keys()))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='0', help="cuda 序号或 'cpu'")
    parser.add_argument('--num_eval', type=int, default=64, help="训练后评估轨迹数 (CPU 慢, 别太大)")
    parser.add_argument('--wandb_mode', type=str, default=None, help="online/offline/disabled")
    parser.add_argument('--tag', type=str, default='run')
    parser.add_argument('--set', nargs='*', default=[], metavar='K=V', help="覆盖任意超参")
    cli = parser.parse_args()

    if cli.wandb_mode:
        os.environ['WANDB_MODE'] = cli.wandb_mode

    device = torch.device('cpu') if cli.device == 'cpu' else \
        torch.device(f"cuda:{cli.device}" if torch.cuda.is_available() else "cpu")

    algo_for_args = 'DQCAC' if cli.algo == 'CALIB' else cli.algo
    args = base_args(algo_for_args, cli.seed, device, cli.env)
    overrides = {}
    for kv in cli.set:
        k, v = kv.split('=', 1)
        overrides[k] = cast(v)
        setattr(args, k, overrides[k])
    # theta_lr0 便捷覆盖 (按最终 theta_b/c 重算 theta_a)
    if 'theta_lr0' in overrides:
        args.theta_a = (args.theta_b ** args.theta_c) * float(overrides['theta_lr0'])

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- CALIB 模式 ----
    if cli.algo == 'CALIB':
        os.environ.setdefault('WANDB_MODE', 'disabled')
        calibrate(args, num_episodes=cli.num_eval)
        return

    # ---- 训练 ----
    from envs import SafetyEnv, make_vec_env
    from utils import evaluate_policy_vec
    from agents import QCPOGPU, DQCACBetaGPU, QCPORefGPU

    AgentCls = {'QCPO': QCPOGPU, 'DQCAC': DQCACBetaGPU, 'QCPO_REF': QCPORefGPU}[cli.algo]
    env = SafetyEnv(args.env_id)

    print(f"\n{'='*78}\n[{cli.tag}] algo={cli.algo} env={cli.env}({args.env_id}) seed={args.seed} "
          f"device={device} overrides={overrides}")
    t0 = time.time()
    agent = AgentCls(args, env)
    agent.train()
    train_t = time.time() - t0
    print(f"\ntrain_time {train_t:.1f}s")

    # ---- 统一评估 (同协议; QCPO_REF 为 LSTM 策略, 用其自管状态的同口径评估器) ----
    eval_vec = make_vec_env(args.env_id, num_envs=args.num_envs, horizon=args.horizon,
                            device=device, ref_env=env, seed=args.seed + 777,
                            backend=getattr(args, 'vec_backend', 'mp'))
    if cli.algo == 'QCPO_REF':
        res = agent.evaluate_vec(eval_vec, cli.num_eval, args.gamma, args.cost_gamma,
                                 args.q_alpha, args.cost_limit)
    else:
        res = evaluate_policy_vec(agent, eval_vec, cli.num_eval, args.gamma, args.cost_gamma,
                                  args.q_alpha, args.cost_limit)
    d, w = args.cost_limit, args.q_alpha
    emp = res['empirical_prob']
    print(f"\n----- eval ({res['num_episodes']} episodes) -----")
    print(f"R(mean)={res['mean']:.3f}  P(Z<=q):{emp:.03f} (alpha={w}, q=-d={-d:.1f})  "
          f"Q_alpha(Z)={res['quantile_return']:.03f}  margin={res['constraint_margin']:.03f}  "
          f"[debug C_disc={res['cost_disc_mean']:.3f} C_undisc={res['cost_undisc_mean']:.3f}]")
    if res['cost_cdf_initial'] is not None:                    # DQCAC: cost-critic 校准
        print(f"[cost-critic calibration] P(Z<=q): critic={res['cost_cdf_initial']:.3f} "
              f"truth={emp:.3f} bias={res['cost_cdf_initial']-emp:+.3f}")
    ok = emp <= args.q_alpha + 0.02
    print(f"[{'OK ' if ok else 'BAD'}] P(Z<=q)<=alpha: {emp:.3f} (alpha={args.q_alpha})")
    eval_vec.close(); agent.vec_env.close()               # 回收 mp worker 进程

    # ---- 写 wandb + 存盘 (eval 主键用下尾命名; cost_* 仍写入作 debug) ----
    run = getattr(agent, '_wandb_run', None)
    if run is not None:
        eval_log = {
            'eval/mean': res['mean'],
            'eval/reward_std': res['reward_std'],
            'eval/empirical_prob': emp,
            'eval/quantile_return': res['quantile_return'],
            'eval/quantile_margin_to_threshold': res['quantile_margin_to_threshold'],
            'eval/constraint_margin': res['constraint_margin'],
            'eval/num_episodes': res['num_episodes'],
            'eval/debug_cost_disc_mean': res['cost_disc_mean'],
            'eval/debug_cost_undisc_mean': res['cost_undisc_mean'],
            'eval/debug_cost_quantile': res['cost_quantile'],
        }
        if res.get('cost_cdf_initial') is not None:
            eval_log['eval/cost_cdf_initial'] = res['cost_cdf_initial']
        if res.get('pred_cost_mean') is not None:
            eval_log['eval/pred_cost_mean'] = res['pred_cost_mean']
        if res.get('pred_cost_std') is not None:
            eval_log['eval/pred_cost_std'] = res['pred_cost_std']
        run.log(eval_log)
        try:
            run.summary['eval/constraint_ok'] = bool(ok)
            run.summary['budget/total_env_steps'] = int(args.num_envs) * int(args.num_iterations) * int(args.horizon)
        except Exception as e:
            print(f"[warn] wandb summary 写入失败: {e}")
        run.finish()
    summary = getattr(agent, 'get_training_summary', lambda: {})()
    out = {'tag': cli.tag, 'algo': cli.algo, 'env': cli.env, 'seed': args.seed,
           'overrides': {k: str(v) for k, v in overrides.items()},
           'train_seconds': train_t, 'eval': res, 'summary': summary}
    os.makedirs(os.path.join(BASE_DIR, '_runs'), exist_ok=True)
    fp = os.path.join(BASE_DIR, '_runs', f"{cli.algo}_{cli.env}_{cli.tag}_s{args.seed}.json")
    with open(fp, 'w', encoding='utf-8') as f:
        json.dump(out, f)
    print(f"saved: {fp}\n{'='*78}")


if __name__ == '__main__':
    main()
