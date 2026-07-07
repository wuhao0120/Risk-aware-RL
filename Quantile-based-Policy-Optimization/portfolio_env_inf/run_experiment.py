# -*- coding: utf-8 -*-
"""
run_experiment.py —— portfolio_env_inf 统一实验入口 (四算法同口径对比)。

统一对比标准 (硬约定, 改一处即全体生效):
    1. 同一环境参数 (PortfolioEnv 默认) + 同一 γ=0.9 / α=0.25 / q (全员持有同一 q 用于报告);
    2. 同一评估协议: 训练后 evaluate_policy_vec (全 GPU, 默认 3000 条新鲜轨迹) 统计
       mean / Q_α / std / P(Z≤q) (+DQCAC 的 critic 校准三件套); 同一探索 σ (init_std);
    3. 同一 wandb 方案: project='portfolio_inf', x 轴=progress/env_steps, 指标键名与
       risk_sensitive_inf 完全一致 (见 agents/vec_base.py)。
    注: 采样规模/更新结构/学习率属【各算法调优自由度】(原项目同此惯例) —— trajectory-level
    的 QPO/QCPO 信噪比低, 需要 B=4096 大批量 (这正是 DQC-AC per-transition TD 的卖点);
    各算法的 env-step 预算差异在 wandb 的 progress/env_steps x 轴上直接可见, 不隐藏。

用法:
    python run_experiment.py --algo DQCAC --seed 0
    python run_experiment.py --algo QPO  --set num_iterations=200 --wandb_mode disabled
    python run_experiment.py --algo CALIB                 # 环境校准: 参考策略表 (选 q 用)
算法名: QPO / QCPO / QPPO / DQCAC / CALIB
"""
import os
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
    """轻量命名空间 (与 risk_sensitive_env_inf/_train_run.py 同款)。"""
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


# ============================================================ 环境变体 ============================================================
# 实验A (2资产) 把 5 资产砍到仅 股4/股5 反相关对 —— 5 资产里另外 3 只是干扰项 (股(2,3)
# 被 (4,5) 严格支配, 股1 均值可忽略), 真正的风险决策只活在 (4,5) 的倾斜上, 故 2 资产
# 完整保留"风险中性最优(全仓股5)↔约束最优(倾斜)"的全部故事, 但动作 1 维、obs 7 维、
# q 校准解析可解、噪声特征类失稳消失。mu/sigma=None → PortfolioEnv 用 5 资产默认值。
# q: 由 _calib2.py 带 σ=0.35 噪声策略类校准 (见 DESIGN.md §二同款方法论)。
VARIANTS = {
    '5asset': dict(mu=None, sigma=None, q=1.0),                # 默认 (原 5 资产)
    '2asset': dict(mu=[0.20, 0.60],                            # 实验A: 股4/股5 反相关对
                   sigma=[[0.16, -0.19], [-0.19, 0.25]],       # ρ≈-0.95 (与 5 资产逐项一致)
                   q=1.0),                                     # _calib2.py 校准: q=1.0/α=0.25 (与5资产同口径)
    #   σ=0.35 噪声表: 全仓股5 P=0.340✗(违约), 约束最优 a*≈0.34 mean≈4.49 P=0.243✓(binding),
    #   QPO 保守解 a≈0.45 mean 4.10 → 阶梯 DQCAC/QCPO≈4.49 > QPO/QPPO≈4.10, gap +0.39 干净。
}


def base_args(algo, seed, device):
    """
    四算法默认超参 (移植自 risk_sensitive_env_inf/_train_run.py 的已验证配置, 标注差异)。
    统一字段在最上面 —— 这是"对比标准统一"的实现位置。
    """
    a = argparse_ns()
    # -------------------- 统一字段 (四算法一致, 勿单独改动) --------------------
    a.env_name = 'PortfolioEnvInf'
    a.seed = seed
    a.algo_name = algo
    a.q_alpha = 0.25                 # 约束/分位水平 α (统一)
    a.gamma = 0.9                    # 折扣 γ: 有效视界 10 步 ≪ n=100 → 截断≈∞ (γ^100≈3e-5)
    a.quantile_threshold = 1.0       # 约束阈值 q (统一持有)。⚠ 必须由【带探索噪声 σ=0.35 的
                                     # 策略类】校准 (_noisy_calib.py), 不能用确定性 CALIB 表:
                                     # σ=0.35 把 hedge45 的 Q_α 从 2.49 压到 1.66 → 旧 q=2.0
                                     # 对全策略类不可行 (v3 全员违约+λ 顶满的根因)。q=1.0 下:
                                     # binding 点 tilt a≈0.33 (约束最优 mean≈4.5), 全仓股5
                                     # P=0.333 违约, hedge 区可行 (P≈0.19) → 可行且 binding。
    a.env_n = 100                    # 一段 rollout 截断长度 T
    a.num_envs = 512                 # 并行 env 数 B (默认; QPO/QCPO 因 SNR 需要在下面覆盖为 4096)
    a.num_iterations = 1500          # 迭代数 (默认; trajectory-level 算法在下面覆盖)
    a.updates_per_episode = 10       # 每迭代内层更新次数 (默认)
    a.init_std = 0.35                # 策略探索 σ (logits 空间, 统一; 0.5 时探索噪声本身就把
                                     # 对冲组合的 P(Z≤q) 抬过 α → 约束不可达, 0.35 实测可达)
    a.actor_hidden = None            # 线性+bias 策略 (统一; 'l64,64' 可开 MLP)
    a.log_interval = 50
    a.est_interval = 100
    a.device = device
    a.wandb_dir = BASE_DIR              # wandb 本地文件固定落在本目录 (防 cwd 漂移污染他处)
    a.wandb_project = 'portfolio_inf'   # 独立 wandb project (方案与 risk_sensitive_inf 统一)

    if algo == 'QPO':
        # 纯分位数最大化 (保守基线)。
        # 稳定性修复 (见 DESIGN.md): θ 用 SGD (Adam 在噪声特征坐标恒速随机游走 → W 爆炸);
        # q_est 用批分位数 EMA 追踪 (SA 在迭代尺度下滞后失效)。theta_b 用迭代尺度。
        a.num_envs = 4096                                      # 大批量 (trajectory-level SNR ∝ √B)
        a.num_iterations = 3000                                # 信号弱 → 长跑 (GPU ~8 min)
        a.theta_optimizer = 'sgd'
        a.theta_b = 3000
        a.theta_c = 0.9
        a.theta_a = (a.theta_b ** a.theta_c) * 0.12            # SGD lr(0)=0.12 (探针实测有效值:
                                                               # 稳定且 ~2500 迭代可达对冲区; 0.5+ 会冲进饱和角点)
        a.q_track_mode = 'ema'
        a.q_track_rho = 0.3
        a.q_b = 500                                            # (sa 对照模式才用)
        a.q_c = 0.6
        a.q_a = (a.q_b ** a.q_c) * 1e-2
        a.warmup_q_iters = 2
        a.actor_grad_clip = 1.0                                # 范数失控保护 (探针实证必要)
    elif algo == 'QCPO':
        # 约束版 (经验 dual)。算法与 _train_run.py QCPOGPU 一致; 稳定性修复同 QPO (SGD),
        # 另加 λ 上界 (长暂态防 λ runaway → 纯指示函数退化域)。
        a.num_envs = 4096                                      # 大批量 (trajectory-level SNR ∝ √B)
        a.num_iterations = 3000
        a.updates_per_episode = 1                              # 每迭代 1 次更新 (新鲜噪声; 复用会把
                                                               # 同批噪声方向放大 10x → 漂移失控, 探针实证)
        a.theta_optimizer = 'sgd'
        a.theta_b = 3000
        a.theta_c = 0.9
        a.theta_a = (a.theta_b ** a.theta_c) * 0.06            # SGD lr(0)=0.06 (探针实测有效值: 稳定)
        a.lambda_a = 1.0                                       # λ lr≈0.43/迭代 (快 λ: 策略到达约束边界时 λ 已就位)
        a.lambda_b = 5000
        a.lambda_c = 0.1
        a.lambda_max = 50.0
        a.outer_interval = 1
        a.norm_ema_decay = 0.01                                # return_rms decay (QCPO 默认)
        a.actor_grad_clip = 1.0                                # 范数失控保护 (探针实证必要)
        a.warmup_rms_iters = 2
    elif algo == 'QPPO':
        # PPO 化分位数最大化 (保守基线 2)。PPO 惯例常数 lr; q 追踪同 QPO (ema)。
        a.num_envs = 2048                                      # 中批量 (PPO 裁剪自带稳定性)
        a.num_iterations = 3000
        a.q_track_mode = 'ema'
        a.q_track_rho = 0.3
        a.ppo_lr = 1e-3                                        # 3e-4 实测 1500 迭代欠收敛 → 提速
        a.clip_eps = 0.2
        a.vf_coef = 0.5
        a.ent_coef = 0.0
        a.grad_clip = 0.5
        a.critic_hidden = [64, 64]
        a.q_b = 500
        a.q_c = 0.6
        a.q_a = (a.q_b ** a.q_c) * 1e-2
        a.warmup_q_iters = 2
    elif algo == 'DQCAC':
        # DQC-AC-β (critic-dual)。逐项继承 _train_run.py DQCACBetaGPU 的已验证配置。
        a.beta = 0.90
        a.critic_step_feature = False                          # 平稳环境: step-blind 即正确
        a.theta_b = 10000                                      # 已验证: ≈常数 2e-4
        a.theta_c = 0.9
        a.theta_a = (a.theta_b ** a.theta_c) * 2e-4
        a.lambda_a = 0.3
        a.lambda_b = 5000
        a.lambda_c = 0.1
        a.lambda_max = 50.0
        a.lambda_min = 0.0
        a.outer_interval = 1
        a.num_quantiles = 32
        a.huber_kappa = 0.1                                    # 近纯分位回归 → critic 无偏 (已验证)
        a.critic_hidden = [64, 64]
        a.critic_lr = 1e-3
        a.target_tau = 0.05                                    # 消除 critic 滞后 (已验证)
        a.target_update_interval = 1
        a.n_step = 1
        a.num_action_samples = 4
        a.advantage_norm = 'qcpo'                              # EMA 归一化 (反 λ 卷绕, 已验证)
        a.norm_ema_decay = 0.1
        a.warmup_iters = 30                                    # 先校准 critic 再开 actor/λ
        a.entropy_coef = 0.0
        a.actor_grad_clip = 100.0
        a.critic_grad_clip = 10.0
    return a


# ============================================================ CALIB: 参考策略校准 ============================================================
class _ConstPolicy:
    """常量目标权重'伪智能体': _sample_actions 恒返回 log(w_ref) logits (无探索噪声)。"""

    def __init__(self, w_ref, device):
        w = np.asarray(w_ref, dtype=np.float64)
        w = np.maximum(w, 1e-8)                                # 零权重 → log 前加下限
        self._logits = torch.as_tensor(np.log(w / w.sum()), dtype=torch.float32,
                                       device=device)          # softmax(log w)=w
        self.device = device

    def _sample_actions(self, states):
        """[B, sd] → 常量 logits [B, K] (确定性参考策略)。"""
        return self._logits.unsqueeze(0).expand(states.shape[0], -1)


def calibrate(args, num_episodes=2000, q_grid=(0.0, 1.0, 1.5, 2.0, 2.5, 3.0)):
    """
    环境校准: 对一组参考常量权重策略, 用统一评估器算 mean/std/Q_α/P(Z≤q) 表 →
    选 q 的依据 (q 应使: 对冲参考可行、全仓股5 明显违约、约束在均值前沿上有意义地 binding)。
    """
    from envs import PortfolioEnv, PortfolioVecTorch
    from utils import evaluate_policy_vec

    env = PortfolioEnv(n=args.env_n)
    refs = {
        'uniform': np.full(5, 0.2),
        'stock0': np.eye(5)[0], 'stock1': np.eye(5)[1], 'stock2': np.eye(5)[2],
        'stock3': np.eye(5)[3], 'stock4(maxmu)': np.eye(5)[4],
        'hedge23': np.array([0, .5, .5, 0, 0]),
        'hedge45': np.array([0, 0, 0, .5, .5]),
        'tilt45_a0.45': np.array([0, 0, 0, .45, .55]),
        'tilt45_a0.40': np.array([0, 0, 0, .40, .60]),
        'tilt45_a0.35': np.array([0, 0, 0, .35, .65]),
    }
    print(f"\n===== CALIB (E={num_episodes}, gamma={args.gamma}, alpha={args.q_alpha}) =====")
    header = f"{'policy':16s} {'mean':>7s} {'std':>7s} {'Q_a':>7s} " + \
             ' '.join(f"P<={q:<4g}" for q in q_grid)
    print(header)
    out = {}
    for name, w in refs.items():
        vec = PortfolioVecTorch(num_envs=args.num_envs, device=args.device, ref_env=env)
        agent = _ConstPolicy(w, args.device)
        res = evaluate_policy_vec(agent, vec, num_episodes, args.gamma, args.q_alpha,
                                  args.quantile_threshold)
        # 额外: 对 q 网格逐点算违约率 (重用同一批回报需重跑; 这里直接再算 — 简洁起见
        # 用结果字典的 mean/std 正态近似补充, 主列仍是经验值)
        zs = _collect_returns(agent, vec, num_episodes, args.gamma)
        probs = [float(np.mean(zs <= q)) for q in q_grid]
        print(f"{name:16s} {res['mean']:7.3f} {res['std']:7.3f} {res['quantile']:7.3f} " +
              ' '.join(f"{p:7.3f}" for p in probs))
        out[name] = {'mean': res['mean'], 'std': res['std'], 'quantile': res['quantile'],
                     'probs': dict(zip(map(str, q_grid), probs))}
    return out


def _collect_returns(agent, vec_env, num_episodes, gamma):
    """跑 ceil(E/B) 轮 vec rollout, 返回全部截断折扣回报 numpy [E'] (CALIB 网格用)。"""
    import math
    B, n = vec_env.B, vec_env.n
    Z_all = []
    with torch.no_grad():
        for _ in range(max(1, math.ceil(num_episodes / B))):
            s = vec_env.reset()
            z = torch.zeros(B, device=s.device)
            disc = 1.0
            for _t in range(n):
                a = agent._sample_actions(s)
                s, r, _ = vec_env.step(a)
                z += disc * r
                disc *= gamma
            Z_all.append(z)
    return torch.cat(Z_all).cpu().numpy()


# ============================================================ 主流程 ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='DQCAC',
                        choices=['QPO', 'QCPO', 'QPPO', 'DQCAC', 'CALIB'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='0', help="cuda 序号或 'cpu'")
    parser.add_argument('--num_eval', type=int, default=3000, help="训练后评估轨迹数")
    parser.add_argument('--wandb_mode', type=str, default=None,
                        help="online/offline/disabled (默认不改环境变量)")
    parser.add_argument('--tag', type=str, default='run')
    parser.add_argument('--variant', type=str, default='5asset',
                        choices=list(VARIANTS.keys()),
                        help="环境变体: 5asset(默认) / 2asset(实验A, 仅股4/股5反相关对)")
    parser.add_argument('--set', nargs='*', default=[], metavar='K=V',
                        help="覆盖任意超参, 如 --set num_iterations=200 quantile_threshold=1.5")
    cli = parser.parse_args()

    if cli.wandb_mode:                                          # 须在 import wandb 前生效
        os.environ['WANDB_MODE'] = cli.wandb_mode

    device = torch.device('cpu') if cli.device == 'cpu' else \
        torch.device(f"cuda:{cli.device}" if torch.cuda.is_available() else "cpu")

    args = base_args(cli.algo, cli.seed, device)
    # 环境变体: 在 --set 之前应用 (mu/sigma 定环境, q 给默认值 → --set quantile_threshold 仍可覆盖)
    vdef = VARIANTS[cli.variant]
    args.env_variant = cli.variant
    args.env_mu = vdef['mu']
    args.env_sigma = vdef['sigma']
    args.quantile_threshold = vdef['q']
    overrides = {}
    for kv in cli.set:
        k, v = kv.split('=', 1)
        overrides[k] = cast(v)
        setattr(args, k, overrides[k])
    # 派生参数陷阱: theta_a 在 base_args 内由默认 theta_b 算出, 单独覆盖 theta_b 不会重算。
    # 提供 theta_lr0 便捷覆盖: 在所有 set 之后按最终 theta_b/c 重算 theta_a, 保证 lr(0)=theta_lr0。
    if 'theta_lr0' in overrides:
        args.theta_a = (args.theta_b ** args.theta_c) * float(overrides['theta_lr0'])
    if hasattr(args, 'theta_a'):
        print(f"effective theta lr(0) = {args.theta_a / (args.theta_b ** args.theta_c):.4f}")

    # ---- wandb 实验记录自动填充: 按 wandb_name 前缀 (E1-/E2-) 归入对比实验 ----
    # 命名规范: {实验号}-{算法}-B{批量}x{迭代}[-变体]-s{seed}, 如 E1-QCPO-B512x1500-s0。
    # group = 同一张对比图的 run 同组 (wandb 界面按组折叠);
    # notes = 中文实验说明 (对比什么/预期什么; 训练结束后下方再追加"最终评估: mean/Q/P");
    # tags  = 筛选标签 (实验号/算法/批量/迭代/种子)。均可被 --set wandb_group=... 显式覆盖。
    EXP_META = {
        'E1': ('E1_same_budget_B512x1500',
               '实验一·同预算对比: 四算法统一 B=512 × 1500迭代 × n=100 = 7680万env步, '
               '同探索噪声σ=0.35、同q=1.0。对比: DQCAC vs QPO/QCPO/QPPO。预期: '
               'trajectory-level基线在小批量下信噪比不足、难收敛 —— 这正是 DQCAC '
               'per-transition TD 样本效率优势的直接证据。'),
        'E2': ('E2_full_convergence',
               '实验二·全量收敛对比: 各算法用各自调优配置跑到(近)收敛, 比渐近性能与所需样本量。'
               '对比: DQCAC(B512×1500=76.8M步) vs QPO(B4096×3000) / QPPO(B2048×3000) / '
               'QCPO调优(B4096×6000/12000, theta_lr0=0.12)。'),
        'A': ('A_2asset_verify',
              '实验A·2资产验证: 仅股4/股5反相关对(μ=[.2,.6], ρ≈-.95), 动作1维/obs7维, '
              '环境大幅简化。验证简化后风险约束阶梯是否照样成立 (DQCAC vs QPO/QCPO/QPPO)。'
              '缩量验证, 非全量调参 (沿用各算法已验证配置)。'),
    }
    _name = str(getattr(args, 'wandb_name', '') or '')
    _exp = _name.split('-', 1)[0]
    if _exp in EXP_META and 'wandb_group' not in overrides:
        args.wandb_group, args.wandb_notes = EXP_META[_exp]
        args.wandb_tags = (f"{_exp},{cli.algo},B{args.num_envs},"
                           f"iters{args.num_iterations},s{args.seed}")

    # 随机种子 (统一)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- CALIB 模式: 只做环境校准表 ----
    if cli.algo == 'CALIB':
        out = calibrate(args, num_episodes=cli.num_eval)
        os.makedirs(os.path.join(BASE_DIR, '_runs'), exist_ok=True)
        with open(os.path.join(BASE_DIR, '_runs', 'calibration.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(out, f, indent=2)
        print("saved: _runs/calibration.json")
        return

    # ---- 训练 ----
    from envs import PortfolioEnv, PortfolioVecTorch
    from utils import evaluate_policy_vec
    from agents import QPOGPU, QCPOGPU, QPPOGPU, DQCACBetaGPU
    import wandb

    AgentCls = {'QPO': QPOGPU, 'QCPO': QCPOGPU,
                'QPPO': QPPOGPU, 'DQCAC': DQCACBetaGPU}[cli.algo]
    env = PortfolioEnv(n=int(args.env_n),                      # numpy 参考 env (参数来源)
                       mu=args.env_mu, sigma=args.env_sigma)   # None→5资产默认; 实验A→2资产

    print(f"\n{'=' * 78}\n[{cli.tag}] algo={cli.algo} seed={args.seed} device={device} "
          f"overrides={overrides}")
    t0 = time.time()
    agent = AgentCls(args, env)
    agent.train()
    train_t = time.time() - t0
    print(f"\ntrain_time {train_t:.1f}s")

    # ---- 统一评估 (新鲜 vec env, 全员同协议) ----
    eval_vec = PortfolioVecTorch(num_envs=args.num_envs, device=device, ref_env=env)
    res = evaluate_policy_vec(agent, eval_vec, cli.num_eval,
                              args.gamma, args.q_alpha, args.quantile_threshold)
    q = args.quantile_threshold
    print(f"\n----- eval ({res['num_episodes']} episodes) -----")
    print(f"mean={res['mean']:.3f}  Q_{args.q_alpha}={res['quantile']:.3f}  "
          f"P(Z<=q={q})={res['empirical_prob']:.3f}  std={res['std']:.3f}")
    if res['cdf_initial'] is not None:                         # DQCAC: critic 校准三件套
        print(f"[critic calibration]  P: critic={res['cdf_initial']:.3f} "
              f"truth={res['empirical_prob']:.3f} bias={res['cdf_initial'] - res['empirical_prob']:+.3f}")
        print(f"                      E[Z]: critic={res['pred_mean']:.3f} truth={res['mean']:.3f}")
        print(f"                      std:  critic={res['pred_std']:.3f} truth={res['std']:.3f}")

    # ---- 约束判据 (constrained 算法看 P≤α; 无约束基线只报告) ----
    if cli.algo in ('QCPO', 'DQCAC'):
        ok = res['empirical_prob'] <= args.q_alpha + 0.015
        print(f"[{'OK ' if ok else 'BAD'}] P(Z<=q)<=alpha: {res['empirical_prob']:.3f} "
              f"(alpha={args.q_alpha})")

    # ---- eval 指标写入 wandb + 存盘 (走 agent 持有的 run 对象, 见 vec_base 注释) ----
    # 三处落点, 保证"打开 wandb 一目了然":
    #   1. run.log(eval/*)        → history 末尾 + runs 表 summary 列 (mean/Q_α/P/std...)
    #   2. run.summary[...]       → 约束达标布尔 + 总 env-step 预算 (排序/筛选用)
    #   3. run.notes 追加一句人读结论 → run 页顶部直接看到 "最终评估: mean=… Q=… P=…"
    run = getattr(agent, '_wandb_run', None)
    if run is not None:
        run.log({f"eval/{k}": v for k, v in res.items() if v is not None})
        try:
            run.summary['eval/constraint_ok'] = bool(
                res['empirical_prob'] <= args.q_alpha + 0.015)     # P(Z≤q)≤α (+1.5pt 容差)
            run.summary['budget/total_env_steps'] = \
                int(args.num_envs) * int(args.num_iterations) * int(args.env_n)
            eval_str = (f"最终评估(N={res['num_episodes']}): mean={res['mean']:.3f}, "
                        f"Q{args.q_alpha:g}={res['quantile']:.3f}, "
                        f"P(Z<={q:g})={res['empirical_prob']:.3f}, std={res['std']:.3f}")
            if res['cdf_initial'] is not None:                     # DQCAC: critic 校准偏差
                eval_str += f", critic偏差={res['cdf_initial'] - res['empirical_prob']:+.3f}"
            run.notes = ((run.notes + ' ‖ ') if run.notes else '') + eval_str
        except Exception as e:                                     # disabled 模式等降级路径
            print(f"[warn] wandb summary/notes 写入失败 (不影响训练结果): {e}")
        run.finish()
    summary = getattr(agent, 'get_training_summary', lambda: {})()
    out = {'tag': cli.tag, 'algo': cli.algo, 'seed': args.seed,
           'overrides': {k: str(v) for k, v in overrides.items()},
           'train_seconds': train_t, 'eval': res, 'summary': summary}
    os.makedirs(os.path.join(BASE_DIR, '_runs'), exist_ok=True)
    fp = os.path.join(BASE_DIR, '_runs', f"{cli.algo}_{cli.tag}_s{args.seed}.json")
    with open(fp, 'w', encoding='utf-8') as f:
        json.dump(out, f)
    print(f"saved: {fp}\n{'=' * 78}")


if __name__ == '__main__':
    main()
