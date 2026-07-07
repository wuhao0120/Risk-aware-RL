# -*- coding: utf-8 -*-
"""
DQC-AC-β —— 全 GPU、B 路并行【向量化】版 (DQCACBetaGPU)

本文件是 agents/dqc_ac_beta.py (DQCACBeta) 的【全 GPU 向量化重实现】:
    ┌──────────────┬─────────────────────────────────────────────────────────────┐
    │ baseline     │ DQCACBeta: CPU rollout (numpy env + 逐步 python 循环 + 每步    │
    │ (不改动)     │           numpy↔torch) + GPU update (critic/actor/dual)。     │
    ├──────────────┼─────────────────────────────────────────────────────────────┤
    │ 本版         │ DQCACBetaGPU: rollout 与 update 全程在 GPU。用                 │
    │ (新增)       │           RiskSensitiveVecTorch 一次并行采 B 条轨迹, 把        │
    │              │           baseline 的"逐 transition 更新数学"原样搬到 [n,B]   │
    │              │           批量上 —— 损失/约束/dual/budget/β 全部一致。        │
    └──────────────┴─────────────────────────────────────────────────────────────┘

【不改动 baseline】: 直接 import 复用 DistributionalCritic / lr_lambda / indicator,
                    不拷贝、不修改 dqc_ac_beta.py。

训练规模语义 (按 env-step 对齐, 便于把曲线叠加到 baseline 上对比):
    每次迭代 = 用【冻结策略】并行采 B 条轨迹
             → 内层 updates_per_episode 次 (critic + actor)
             → 每 outer_interval 迭代更新一次 λ (critic 在初始态估计的 P(Z≤q))。
    wandb 同时按 迭代 / 累计轨迹数 / 累计 env-step 记录 (define_metric 默认 x 轴 = env-step)。
    新超参: num_envs (并行 env 数 B)、num_iterations (迭代次数)。
    关系: 总 env-step = num_iterations × num_envs × n。

与 baseline 不可避免的差异 (向量化固有, 非 bug, 已与用户确认):
    1. 采集方式: B 条轨迹用同一【冻结策略】并行采集 (baseline 串行、轨迹间策略已漂移)
       → 非逐位一致, 这是批量 vs 串行的根本差别。
    2. 优势归一化范围: advantage_norm='separate' 在 n·B 行上做 (baseline 在单条 n 行上做)
       → 同一操作、更大批, 统计更稳。
    3. 经验统计窗口: P(Z≤q)/各回报统计直接用"本次迭代的 B 条轨迹" (B≥256 已比 baseline
       的 100 条滚动窗口更紧), 不再维护跨迭代滚动 deque。
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam                            # 与 baseline 一致: base lr=1 + LambdaLR
from torch.optim.lr_scheduler import LambdaLR           # 学习率调度器 lr=a/(b+k)^c
import wandb                                            # 统一日志系统

from utils import Actor, RunningMeanStd                 # Actor: 线性高斯策略 μ(s)=W·s; RunningMeanStd: EMA 在线均值/方差 (qcpo 归一化器)
from envs import RiskSensitiveVecTorch                  # 全 GPU、B 路并行的向量化环境
from .dqc_ac_beta import DistributionalCritic, lr_lambda, indicator  # 复用 baseline 的可复用件


class DQCACBetaGPU(object):
    """
    DQC-AC-β 全 GPU 向量化版主体。算法与 dqc_ac_beta.DQCACBeta 完全对齐,
    仅把"逐条轨迹 [T] 的更新"改为"一次 B 条轨迹 [n,B] 摊平成 [n·B] 的批量更新"。
    """

    # ============================================================ 初始化 ============================================================
    def __init__(self, args, env):
        """
        初始化: actor + 单分布式 critic(+target) + λ + 优化器/调度器 + 向量化 env + wandb。
        与 DQCACBeta.__init__ 逐项对应, 仅多出 num_envs / num_iterations 与内部 vec_env。

        Args:
            args: 超参命名空间 (与 baseline 同名字段 + num_envs / num_iterations)
            env:  numpy 版 RiskSensitiveEnv 实例。双重用途:
                  (1) 作 RiskSensitiveVecTorch(ref_env=env) 的奖励参数来源 (保证同分布);
                  (2) 作评估用环境 (monte_carlo_evaluate 走 select_action, 与 baseline 同口径)。
        """
        # -------------------- 基础参数 (与 DQCACBeta 同名) --------------------
        self.device = args.device                          # 计算设备 (本版训练全程在此 device)
        self.gamma = args.gamma                            # 折扣 γ (budget/critic/dual 用)
        self.beta = getattr(args, 'beta', 0.95)            # Abel 风险折扣 β (仅 actor 约束项用)
        self.q_alpha = args.q_alpha                        # 约束水平 α
        self.quantile_threshold = args.quantile_threshold  # 阈值 q (= budget 初值 b_0)
        self.log_interval = args.log_interval              # 控制台打印间隔 (按迭代计)
        self.outer_interval = max(1, int(getattr(args, 'outer_interval', 1)))  # λ 更新间隔 (按迭代计)

        # -------------------- 向量化训练专属超参 (本版新增) --------------------
        self.num_envs = max(1, int(getattr(args, 'num_envs', 256)))            # 并行 env 数 B
        self.num_iterations = max(1, int(getattr(args, 'num_iterations', 400)))  # 迭代次数

        # -------------------- critic / TD 超参 (用 getattr 给默认值, 与 baseline 一致) --------------------
        self.num_quantiles = getattr(args, 'num_quantiles', 32)            # 分位数个数 N
        self.huber_kappa = getattr(args, 'huber_kappa', 1.0)               # quantile Huber 的 κ
        self.target_tau = getattr(args, 'target_tau', 0.005)               # target 软更新系数
        self.target_update_interval = getattr(args, 'target_update_interval', 1)  # target 更新间隔
        self.n_step = max(1, int(getattr(args, 'n_step', 1)))              # critic TD 目标步数 N (1=1-step bootstrap; ≥n 即 MC)
        # critic step 增广: 本环境状态是洗牌 one-hot[order[t]], order 每 episode 重洗 → 前馈 critic 无法从状态判断步数 t,
        # 而 return-to-go Z_t 强依赖 t (t=0 方差大、t=9 方差小)。step-blind critic 只能学"步数混合/几何视界"不动点 →
        # 分布过散 → P(Z≤q)(Ghat) 被抬高(实测 0.34 vs 真值 0.245)。给 critic 输入追加 step 特征 t/n 即可校准。
        # 只动 critic 输入: actor 仍 step-blind (最优策略每步同 r, 与 QCPO 对齐); TD/loss/budget/约束/对偶/per-transition 均不变。
        self.critic_step_feature = bool(getattr(args, 'critic_step_feature', True))  # True=增广(默认,修复); False=原 step-blind
        self.num_action_samples = max(1, int(getattr(args, 'num_action_samples', 1)))  # baseline 的 K
        self.updates_per_episode = max(1, int(getattr(args, 'updates_per_episode', 10)))  # 每迭代内层更新次数
        self.advantage_norm = getattr(args, 'advantage_norm', 'separate')  # separate / none / qcpo
        self.entropy_coef = getattr(args, 'entropy_coef', 0.0)             # 熵正则系数 (默认 0)
        self.lambda_max = getattr(args, 'lambda_max', 50.0)                # λ 上界
        self.lambda_min = getattr(args, 'lambda_min', 0.0)                 # λ 下限(>0 防 dual 塌缩到0/约束完全释放; 默认0=原行为)
        self.critic_grad_clip = getattr(args, 'critic_grad_clip', 10.0)    # critic 梯度裁剪
        self.actor_grad_clip = getattr(args, 'actor_grad_clip', 10.0)      # actor 梯度裁剪 (qcpo 配方建议 ~100, 近乎关闭, 由 args 设)

        # -------------------- qcpo 归一化配方专属超参 (advantage_norm=='qcpo' 时生效) --------------------
        # 把 QCPO (qcpo.py) 的"反应式 EMA 回报归一化"移植进来, 取代 'separate' 的逐 batch std 归一化:
        #   - 'separate': 每 batch 现算 std → 非平稳/含噪 → λ 卷绕(winds up) 不收敛 (已诊断, 见 memory)。
        #   - 'qcpo':     用跨迭代 EMA(decay) 的稳定 σ 归一化 → λ 能 settle 在 O(1~10)。
        self.norm_ema_decay = float(getattr(args, 'norm_ema_decay', 0.1))  # EMA 衰减 (≈ 1/decay 迭代窗宽); 决策=0.1
        # qcpo 模式默认 warmup 几个迭代(只更新 critic + EMA, 不更新 actor/λ), 让归一化器与 critic 先稳;
        # 非 qcpo 模式默认 0 (行为与原版逐位一致, 作 A/B 对照基线)。args.warmup_iters 显式给 int 则覆盖;
        # 缺省或 None (如 run_experiment 的默认 None) → 走上述按模式的默认值。
        _wi = getattr(args, 'warmup_iters', None)
        self.warmup_iters = int(_wi) if _wi is not None else (5 if self.advantage_norm == 'qcpo' else 0)
        # 两个 EMA 归一化器 (RunningMeanStd, 复用 utils 件): 仅在 qcpo 模式被更新/使用; 实例化无副作用。
        #   return_rms:     跟踪【回报分布】尺度 σ_ret(≈11~12) → 归一化均值优势 Â_m=(q_m-v_m)/σ_ret (对齐 QCPO 的 (U-μ)/σ)
        #   constraint_rms: 跟踪【约束优势自身】std σ_c → 归一化 Â_c=(ψ_c-v_c)/σ_c → ~O(1), 使 λ 落在 O(1~10) (决策 2a)
        self.return_rms = RunningMeanStd(decay=self.norm_ema_decay)         # 均值优势归一化器 (从 batch 回报更新)
        self.constraint_rms = RunningMeanStd(decay=self.norm_ema_decay)     # 约束优势归一化器 (从 batch 约束优势更新)

        # -------------------- 环境与维度 --------------------
        self.env_name = args.env_name
        self.eval_env = env                                                # numpy env, 仅供评估
        # 内部自建全 GPU 向量化环境 (奖励参数从 ref_env 复制 → 与 baseline 完全同分布)
        self.vec_env = RiskSensitiveVecTorch(num_envs=self.num_envs,
                                             device=self.device, ref_env=env)
        self.n = self.vec_env.n                                            # episode 步长 (= env.n)
        self.state_dim = int(np.prod(env.observation_space.shape))         # 状态维度 (one-hot, =n)
        self.action_dim = int(np.prod(env.action_space.shape))             # 动作维度 (=1)
        self._log_2pi = float(np.log(2.0 * np.pi))                         # 高斯 logπ 常量项

        # 分位数水平 τ_i = (i+0.5)/N, i=0..N-1, 形状 [N] (与 baseline 一致)
        self.taus = torch.tensor(
            [(i + 0.5) / self.num_quantiles for i in range(self.num_quantiles)],
            dtype=torch.float32, device=self.device
        )

        # -------------------- Actor (与 baseline 同一线性高斯策略) + 两时间尺度 LR --------------------
        self.actor = Actor(self.state_dim, self.action_dim, args.init_std).to(self.device)  # 线性高斯策略 μ(s)=W·s
        self.actor_optimizer = Adam(self.actor.parameters(), 1.0, eps=1e-5)  # base lr=1, 真实 lr 由 scheduler 给
        self.actor_scheduler = LambdaLR(                       # θ 的两时间尺度 lr=a/(b+k)^c
            self.actor_optimizer,
            lr_lambda=lambda k: lr_lambda(k, args.theta_a, args.theta_b, args.theta_c)
        )

        # -------------------- 单分布式 critic + target critic (复用 baseline 的网络类) --------------------
        hidden = getattr(args, 'critic_hidden', [64, 64])
        if isinstance(hidden, str):                                        # 兼容 "64,64" 字符串
            hidden = [int(x) for x in hidden.split(',') if x.strip()]
        # critic 输入维度: step 增广时 = state_dim + 1 (追加 step 特征 t/n); 否则 = state_dim (原 step-blind)
        cdim = self.state_dim + (1 if self.critic_step_feature else 0)
        self.critic = DistributionalCritic(cdim, self.action_dim,
                                           self.num_quantiles, list(hidden)).to(self.device)
        self.target_critic = DistributionalCritic(cdim, self.action_dim,
                                                  self.num_quantiles, list(hidden)).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())       # target ← online (硬拷贝初始化)
        self.critic_optimizer = Adam(self.critic.parameters(),
                                     getattr(args, 'critic_lr', 1e-3), eps=1e-5)

        # -------------------- 拉格朗日乘子 λ (与 baseline 完全一致) --------------------
        self.lambda_dual = torch.tensor([0.0], dtype=torch.float32,    # λ≥0 标量, 需要梯度 (由 dual 优化)
                                        device=self.device, requires_grad=True)
        self.lambda_optimizer = Adam([self.lambda_dual], 1.0, eps=1e-5)  # base lr=1, 真实 lr 由 scheduler 给
        self.lambda_scheduler = LambdaLR(                      # λ 的 lr=a/(b+k)^c
            self.lambda_optimizer,
            lr_lambda=lambda k: lr_lambda(k, args.lambda_a, args.lambda_b, args.lambda_c)
        )

        # -------------------- 运行时统计 (计数 + 诊断量, 不再用滚动 deque) --------------------
        self.learning_steps = 0                                            # critic/actor 累计更新步数
        self.last_dual_prob = 0.0                                          # 上次 dual 用的违约概率 (现为 critic 在初始态的估计)
        self.last_cdf_initial = 0.0                                        # critic 估计的 P(Z≤q) (诊断)
        self.last_empirical_prob = 0.0                                     # 上次记录的经验 P(Z≤q)
        self.last_pred_return_mean = 0.0                                   # critic 在 s_0 估计的回报均值 E[Z|s_0]
        self.last_pred_return_std = 0.0                                    # critic 在 s_0 估计的回报标准差 std(Z|s_0)

        # -------------------- wandb (与 baseline 一致, 额外把 env-step 设为默认 x 轴) --------------------
        wandb.init(project=args.env_name, name=f"{args.algo_name}_{args.seed}",
                   config=vars(args), reinit=True, group=args.algo_name,
                   dir=getattr(args, 'wandb_dir', None))
        # 把所有指标的默认 x 轴设为累计 env-step, 这样与 baseline (按 episode×n 换算) 可叠加对比
        wandb.define_metric("progress/env_steps")
        wandb.define_metric("*", step_metric="progress/env_steps")

    # ============================================================ 主训练循环 ============================================================
    def train(self):
        """
        主循环 (对应 DQCACBeta.train, 把"逐 episode"换成"逐迭代"), 每次迭代:
          1. 并行采 B 条轨迹 (内部重置/递推 budget b、权重 d=γ^t、e=β^t)
          2. 内层: 逐 transition 更新 critic + actor (在 n·B 行批量上), 重复 updates_per_episode 次
          3. 外层: 每 outer_interval 迭代更新一次 λ (critic 在 B 个初始态 (s0,a0) 估计的 P(Z≤q), 不乘 β)
          4. 日志 (按迭代/累计轨迹/累计 env-step) + 学习率调度器步进
        """
        print(f"DQCACBetaGPU: beta={self.beta}, alpha={self.q_alpha}, q={self.quantile_threshold}, "
              f"N={self.num_quantiles}, B={self.num_envs}, iters={self.num_iterations}, "
              f"updates/iter={self.updates_per_episode}, device={self.device}")

        for it in range(self.num_iterations):
            # ===== 1. 并行采样 B 条轨迹 (全 GPU) =====
            batch = self._rollout_vec()

            # ===== 1b. qcpo: 用本次迭代的 B 条轨迹刷新两个 EMA 归一化器 (σ_ret / σ_c), 跨迭代平滑 =====
            if self.advantage_norm == 'qcpo':
                self._update_norm_stats(batch)

            # warmup 期 (仅 qcpo, 前 warmup_iters 迭代): 只练 critic + 刷 EMA, 暂不动 actor/λ,
            # 让归一化器与 critic 先稳定, 避免首批 actor 更新被噪声尺度污染。非 qcpo: warmup_iters=0 → 恒 False。
            in_warmup = it < self.warmup_iters

            # ===== 2. 内层更新 (critic + actor, 在 [n·B] 批量上逐 transition) =====
            critic_info, actor_info = {}, {}
            for _ in range(self.updates_per_episode):
                critic_info = self.update_critic(batch)            # QRTD: 学回报分布
                if not in_warmup:                                  # warmup 期跳过 actor 更新
                    actor_info = self.update_actor(batch)          # PG: w=γ^t Â_m - λ β^t Â_c
                if self.learning_steps % self.target_update_interval == 0:
                    self._soft_update_target()                     # Polyak 软更新 target critic
                self.learning_steps += 1

            # ===== 3. 外层更新 (λ), 每 outer_interval 迭代一次 (warmup 期不动 λ, 策略尚未优化) =====
            if not in_warmup and it % self.outer_interval == 0:
                self.update_dual(batch)                            # critic 估计 P(Z≤q|s0,a0), 与 actor 的 critic 信号一致

            # ===== 4. 日志 + 调度器步进 =====
            self._log(it, batch, critic_info, actor_info)
            self.actor_scheduler.step()                            # actor 学习率衰减
            self.lambda_scheduler.step()                           # λ 学习率衰减

    # ============================================================ 向量化采样 B 条轨迹 ============================================================
    def _rollout_vec(self):
        """
        用冻结策略一次并行采 B 条完整轨迹 (固定 n 步、B 路锁步), 全程 GPU、无 numpy↔torch。

        与 baseline._rollout_one_episode 的对应:
          - baseline: 单条轨迹 → 序列长度 T, 各量形如 [T] / [T, sd]
          - 本版:     B 条轨迹 → [n, B, *], 末尾摊平成 [n·B, *] 当作一个大 batch
        budget b 与权重 d=γ^t, e=β^t 的重置/递推同 baseline:
            b_0=q, b_{t+1}=(b_t-r_t)/γ (逐 env 不同 → [B]);  d_t=γ^t, e_t=β^t (只随 t 变)
        返回: 把各序列摊平好的 dict (键名与 baseline 对齐, 另含 [B] 的每条轨迹回报 disc_returns)。
        """
        n, B = self.n, self.num_envs
        s = self.vec_env.reset()                               # 初始状态 [B, sd] (GPU)
        b = torch.full((B,), float(self.quantile_threshold),
                       dtype=torch.float32, device=self.device)  # budget b_0=q, [B]
        S, A, R, S2, Bud = [], [], [], [], []                  # 逐步收集 (元素分别为 [B,*]/[B])
        disc_return = torch.zeros(B, dtype=torch.float32, device=self.device)  # 每条轨迹 U(τ), [B]
        disc = 1.0                                             # 当前折扣因子 γ^t (标量)

        for t in range(n):
            with torch.no_grad():                              # 采样不建图 (冻结策略采集)
                a = self._sample_actions(s)                    # a~π(·|s), [B, ad]
            s2, r, done = self.vec_env.step(a)                 # 全 GPU step: s2 [B,sd], r [B], done bool

            # 记录本步 transition 及 budget b_t (注意 b 是当前步的预算, 在递推前 clone)
            S.append(s); A.append(a); R.append(r); S2.append(s2); Bud.append(b.clone())

            disc_return = disc_return + disc * r               # 累计 U(τ)=Σγ^t r_t, 逐 env [B]
            disc *= self.gamma                                 # γ^{t+1}
            b = (b - r) / self.gamma                           # budget 递推 b_{t+1}=(b_t-r_t)/γ, [B]
            s = s2

        # ---- 堆叠成 [n, B, *] 再摊平成 [n·B, *] (行序: 先 t 后 env, 即 index = t*B + j) ----
        Smat = torch.stack(S, dim=0)                           # [n, B, sd] 保留 [n,B] 形以算 n-step
        Rmat = torch.stack(R, dim=0)                           # [n, B]
        states = Smat.reshape(n * B, -1)                       # [n·B, sd]
        actions = torch.stack(A, dim=0).reshape(n * B, -1)     # [n·B, ad]
        rewards = Rmat.reshape(n * B)                          # [n·B]
        next_states = torch.stack(S2, dim=0).reshape(n * B, -1)  # [n·B, sd]
        budgets = torch.stack(Bud, dim=0).reshape(n * B)       # [n·B]  对应各 transition 的 b_t

        # ---- n-step TD 目标预计算 (N=self.n_step; N=1 退化为原 1-step) ----
        # 对 transition (t, env): n步累计回报 = Σ_{k=0}^{m-1} γ^k r_{t+k}, m=min(N, n-t) (episode 末自动截断);
        #   若 t+N ≤ n-1 (bootstrap 态 s_{t+N} 仍是非终止决策态) → 末端 bootstrap γ^N·ψ̄(s_{t+N},·);
        #   否则 (t+N≥n, 已到终止) 不 bootstrap, 只用累计奖励。bootstrap 用的 a''~π 与 target critic 在 update 时算。
        Ns = self.n_step
        nstep_rew = torch.zeros(n, B, dtype=torch.float32, device=self.device)
        disc = 1.0
        for k in range(Ns):                                    # 累加 γ^k r_{t+k}: Rmat[k:] 右移对齐, [:n-k] 在末尾截断
            nstep_rew[:n - k] += disc * Rmat[k:]
            disc *= self.gamma
        t_ar = torch.arange(n, device=self.device)             # [n]: 0..n-1
        boot_mask_t = (t_ar + Ns <= n - 1).float()             # [n] 是否仍可 bootstrap (1=可, 末端几步=0)
        boot_idx = torch.clamp(t_ar + Ns, max=n - 1)           # [n] bootstrap 态 s_{t+N} 的索引 (越界 clamp, 被 mask 关掉)
        boot_states = Smat[boot_idx].reshape(n * B, -1)        # [n·B, sd]  s_{t+N}
        boot_mask = boot_mask_t.unsqueeze(1).expand(n, B).reshape(n * B)  # [n·B]
        nstep_reward = nstep_rew.reshape(n * B)                # [n·B]  Σ_{k<m} γ^k r_{t+k}
        # critic step 增广用: 当前 transition 步数 t、bootstrap 态 s_{t+N} 的步数 (= boot_idx); actor 不用
        steps = t_ar.unsqueeze(1).expand(n, B).reshape(n * B).float()       # [n·B]  当前步 t
        boot_steps = boot_idx.unsqueeze(1).expand(n, B).reshape(n * B).float()  # [n·B]  bootstrap 态步数

        # done: 固定 n 步锁步 ⇒ 仅最后一步=1, 其余=0; [n,B] → [n·B] (与上面同序)
        dones = torch.zeros(n, B, dtype=torch.float32, device=self.device)
        dones[n - 1] = 1.0
        dones = dones.reshape(n * B)

        # d=γ^t, e=β^t: 只随 t 变、所有 env 相同 → [n] 广播到 [n,B] 再摊平 [n·B]
        t_idx = torch.arange(n, dtype=torch.float32, device=self.device)  # [n]: 0,1,...,n-1
        d = (self.gamma ** t_idx).unsqueeze(1).expand(n, B).reshape(n * B)  # γ^t, [n·B]
        e = (self.beta ** t_idx).unsqueeze(1).expand(n, B).reshape(n * B)   # β^t, [n·B]

        return {
            'states': states,            # [n·B, sd]
            'actions': actions,          # [n·B, ad]
            'rewards': rewards,          # [n·B]
            'next_states': next_states,  # [n·B, sd]
            'dones': dones,              # [n·B]
            'nstep_reward': nstep_reward,  # [n·B]  Σ_{k<m} γ^k r_{t+k}  (n-step 累计回报)
            'boot_states': boot_states,  # [n·B, sd]  bootstrap 态 s_{t+N}
            'boot_mask': boot_mask,      # [n·B]  是否 bootstrap (末端=0)
            'steps': steps,              # [n·B]  当前 transition 步数 t (critic step 增广用)
            'boot_steps': boot_steps,    # [n·B]  bootstrap 态步数 (critic step 增广用)
            'budgets': budgets,          # [n·B]  b_t
            'd': d,                      # [n·B]  γ^t
            'e': e,                      # [n·B]  β^t
            's0': S[0],                  # [B, sd] 初始状态 s_0 (供 critic-based Ĝ 诊断)
            'disc_returns': disc_return, # [B] 每条轨迹的标量回报 U(τ) (供 dual / 日志)
        }

    # ============================================================ Critic 更新 (QRTD) ============================================================
    def update_critic(self, batch):
        """
        一次分布式 n-step TD 更新 (Quantile Regression TD + quantile Huber loss)。N=self.n_step:
            y_j = Σ_{k=0}^{m-1} γ^k r_{t+k} + γ^N·1{可bootstrap}·ψ̄_j(s_{t+N}, a'')
            a''~π(·|s_{t+N}); ψ̄ 为 target critic。N=1 即原 1-step (r + γ(1-done)ψ̄);
            到达 episode 终止 (t+N≥n) 时不 bootstrap, 只用累计奖励 (终止态正确退化)。
        n-step 减少 bootstrap 偏差 → critic 回报分布更准 → Ψ̂/Q̂ 更可信 (代价: target 方差变大)。
        batch 维 = n·B; ψ_i/ψ̄_j 的下标 j 是 num_quantiles 维, 与步数 N 无关。
        """
        s, a = batch['states'], batch['actions']
        nstep_r = batch['nstep_reward']                            # [n·B]  Σ_{k<m} γ^k r_{t+k}
        boot_s, boot_mask = batch['boot_states'], batch['boot_mask']  # [n·B, sd], [n·B]
        steps, boot_steps = batch['steps'], batch['boot_steps']    # [n·B] 各自步数 (critic 增广用)

        with torch.no_grad():                                      # target 不建图
            a_boot = self._sample_actions(boot_s)                  # a''~π(·|s_{t+N}), [n·B, ad] (actor step-blind, 用原始 boot_s)
            psi_next = self.target_critic(self._aug(boot_s, boot_steps), a_boot)  # ψ̄_j(s_{t+N},a''), critic 增广 s_{t+N} 的步数
            # y_j = Σγ^k r + γ^N·1{可bootstrap}·ψ̄_j; 用 unsqueeze(1) 广播到 [n·B, N_q]
            y = nstep_r.unsqueeze(1) + (self.gamma ** self.n_step) * boot_mask.unsqueeze(1) * psi_next

        psi = self.critic(self._aug(s, steps), a)                  # ψ_i(s,a), critic 增广当前步 t, [n·B, N_q] (建图)
        loss = self._quantile_huber_loss(psi, y)                   # 标量

        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.critic_grad_clip and self.critic_grad_clip > 0:    # 梯度裁剪 (稳定性)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_grad_clip)
        self.critic_optimizer.step()
        # loss 含对 N 个 target 分位数的 .sum(dim=2)(未除 N) → loss ∝ N, 跨 N 不可比;
        # 额外记一个 /N 的归一化版本(纯日志, backward 已完成, 零训练影响)便于跨 N 对比。
        return {'critic/quantile_huber_loss': float(loss.item()),
                'critic/quantile_huber_loss_normalized': float(loss.item()) / self.num_quantiles}

    def _quantile_huber_loss(self, psi, y):
        """
        Quantile Huber Loss (QR-DQN 核心), ρ^κ_τ(u)=|τ-1{u<0}|·L_κ(u)/κ。与 baseline 完全一致。
        对所有 (当前分位数 i, 目标分位数 j) 配对计算 TD error u=y_j-ψ_i, 再加权求和。

        输入: psi [n·B, N] (当前 ψ_i), y [n·B, N] (目标 y_j)
        输出: 标量 loss
        """
        # u[b,i,j] = y[b,j] - psi[b,i]; y.unsqueeze(1)=[*,1,N], psi.unsqueeze(2)=[*,N,1] → [*,N,N]
        u = y.unsqueeze(1) - psi.unsqueeze(2)
        abs_u = u.abs()
        # Huber: L_κ(u)=0.5u² (|u|≤κ) 否则 κ(|u|-0.5κ)
        huber = torch.where(abs_u <= self.huber_kappa,
                            0.5 * u.pow(2),
                            self.huber_kappa * (abs_u - 0.5 * self.huber_kappa))
        # 分位数权重 |τ_i - 1{u<0}|; taus 按 i 维 (dim=1) 排布
        taus = self.taus.view(1, -1, 1)                            # [1, N, 1]
        weight = (taus - (u.detach() < 0).float()).abs()           # [*, N, N]
        # ρ = weight·huber/κ; 对目标 j 求和(dim=2), 对分位数 i 取均值(dim=1), 再对 batch 取均值
        return (weight * huber / self.huber_kappa).sum(dim=2).mean(dim=1).mean()

    # ============================================================ Actor 更新 (per-transition) ============================================================
    def update_actor(self, batch):
        """
        一次 per-transition 策略梯度更新。与 baseline 逐式一致 (batch 维 T → n·B)。
        梯度权重: w = d·Â_m - λ·e·Â_c = γ^t Â_m - λ β^t Â_c
            目标信号 Q̂_m = critic 均值;  约束信号 Ψ̂ = critic 局部 CDF (查询 budget b)
            优势 Â_m = Q̂_m - V̂_m,  Â_c = Ψ̂ - V̂_c  (baseline 降方差)
        loss = -E[logπ(a|s)·w] - entropy_coef·E[H];  w 已 detach, 梯度只经 logπ 流向 θ。
        """
        s, a, b = batch['states'], batch['actions'], batch['budgets']
        d, e = batch['d'], batch['e']                              # γ^t, β^t (均 [n·B])
        steps = batch['steps']                                     # [n·B] 当前步 t (critic 增广用; actor logπ 不用)

        with torch.no_grad():                                      # 优势对 actor 不建图
            psi = self.critic(self._aug(s, steps), a)              # [n·B, N]  critic 增广当前步 t
            q_m = psi.mean(dim=1)                                  # Q̂_m(s,a)=(1/N)Σ ψ_i, [n·B]
            psi_c = (psi <= b.unsqueeze(1)).float().mean(dim=1)    # Ψ̂(s,a,b)=(1/N)Σ1{ψ_i≤b}, [n·B]
            v_m, v_c = self._estimate_baselines(s, b, steps)       # V̂_m(s), V̂_c(s,b), 各 [n·B]
            if self.advantage_norm == 'qcpo':
                # QCPO 配方: 用跨迭代 EMA 的【稳定】尺度归一化 (而非逐 batch 现算 std):
                #   Â_m = (q_m - v_m) / σ_ret   —— σ_ret=回报分布 std(≈12), 对齐 QCPO 的 (U-μ)/σ
                #   Â_c = (ψ_c - v_c) / σ_c     —— σ_c=约束优势自身 std, 拉到 ~O(1) 使 λ 落 O(1~10) (决策 2a)
                # σ_ret/σ_c 由 _update_norm_stats 每迭代刷新一次, 内层 updates_per_episode 次更新内冻结 → 稳定。
                a_m = (q_m - v_m) / self.return_rms.std            # 均值优势 / EMA 回报 std
                a_c = (psi_c - v_c) / self.constraint_rms.std      # 约束优势 / EMA 约束优势 std
            else:
                a_m = self._maybe_norm(q_m - v_m)                  # 均值优势 Â_m ('separate' 逐 batch / 'none' 原样)
                a_c = self._maybe_norm(psi_c - v_c)                # 约束优势 Â_c
            # 复合权重: 先算优势再乘 d/e; 约束项带 λ (用 detach, λ 仅由 dual 优化)
            w = d * a_m - self.lambda_dual.detach() * e * a_c      # [n·B]

        log_probs = self._compute_log_probs(s, a)                  # logπ(a|s), [n·B] (对 θ 可导)
        actor_loss = -(log_probs * w).mean() \
            - self.entropy_coef * self._entropy(s).mean()          # -E[logπ·w] - τ·E[H]

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.actor_grad_clip and self.actor_grad_clip > 0:      # 梯度裁剪
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip)
        self.actor_optimizer.step()                                # scheduler.step() 在 train() 末尾
        # 诊断量: a_m/a_c 已被 _maybe_norm 归一化为均值0 → 它们的 .mean() 恒≈0 无意义。
        # 改记有信息量的: 归一化前优势的"标准差"(信号强度) + critic 原始量 + 实际权重 w 的统计。
        raw_adv_m = q_m - v_m                                      # 归一化前的均值优势 Â_m (raw)
        raw_adv_c = psi_c - v_c                                    # 归一化前的约束优势 Â_c (raw)
        return {'actor/loss': float(actor_loss.item()),
                'advantage/mean_adv_std': float(raw_adv_m.std(unbiased=False).item()),  # Â_m 信号强度
                'advantage/risk_adv_std': float(raw_adv_c.std(unbiased=False).item()),  # Â_c 信号强度
                'critic/q_mean': float(q_m.mean().item()),         # critic 估计的平均回报 Q̂_m(s,a)
                'constraint/psi_c_mean': float(psi_c.mean().item()),  # 平均 per-transition 违约估计 Ψ̂(s,a,b)
                'actor/w_mean': float(w.mean().item()),            # 实际梯度权重 w 的均值 (含 d/e/λ)
                'actor/w_std': float(w.std(unbiased=False).item())}  # w 的标准差 (actor 梯度方差来源)

    # ============================================================ Dual 更新 (λ) ============================================================
    def update_dual(self, batch):
        """
        拉格朗日乘子 λ 的投影梯度上升 (与 qcpo/DQCACBeta 一致, 不乘 β):
            P̂(Z≤q) = critic 在本次迭代 B 个初始态 (s0,a0) 上估计的违约概率;  λ ← [λ + ε_k(P̂-α)]_+
        设计 loss=-λ(P̂-α) ⇒ ∂loss/∂λ=-(P̂-α) ⇒ optimizer 下降一步即 λ←λ+lr(P̂-α)。

        【本版改动】原 dual 用【经验】违反率 (B 条轨迹回报 U(τ)≤q 的比例); 现改用【critic】估计:
            每条轨迹 P(Z≤q|s0,a0) = (1/N)Σ_i 1{ψ_i(s0,a0) ≤ q}, 再对 B 条初始态平均。
        动机: 原 actor 约束项已用 critic 的局部 CDF Ψ̂ 作信号, 而 dual 却用经验违反率 →
              actor 与 dual 两个更新对"违约"的口径不一致 (critic 信号 vs 真值信号),
              critic 估偏时二者拉扯。改成同源 critic 估计后, λ 与 actor 由同一分布估计驱动,
              口径一致 (前提: critic 已校准, 即 critic_step_feature=True, 否则 dual 会被 critic 偏差带偏)。
        注: 不改 dual 的数学形式/投影/不乘 β, 只换"概率从哪来"; TD/budget/actor 逻辑均不动。

        输入: batch —— 含 's0' [B, sd] (本迭代各轨迹初始态) 与 'disc_returns' [B] (经验值, 仅作诊断对比)。
        """
        with torch.no_grad():                                      # dual 概率估计不建图 (λ 梯度只来自下面 -λ·gap)
            s0 = batch['s0']                                       # [B, sd] 本迭代各轨迹初始状态 s_0
            a0 = self._sample_actions(s0)                          # a_0~π(·|s_0), [B, ad] (actor step-blind)
            psi0 = self.critic(self._aug(s0, 0), a0)              # ψ_i(s_0,a_0), [B, N]; critic 增广 step=0 (s0 是第 0 步)
            # critic 估计违约概率: 每条轨迹 (1/N)Σ1{ψ_i≤q} → [B], 再对 B 条初始态平均 → 标量
            p = float((psi0 <= self.quantile_threshold).float().mean(dim=1).mean().item())
        self.last_dual_prob = p                                    # dual 实际使用的违约概率 (现为 critic 估计)

        gap = torch.tensor([p - self.q_alpha], dtype=torch.float32, device=self.device)
        self.lambda_optimizer.zero_grad(set_to_none=True)
        (-self.lambda_dual * gap).backward()
        self.lambda_optimizer.step()
        with torch.no_grad():                                      # 投影到 [λ_min, λ_max] (λ_min>0 时约束不完全释放)
            self.lambda_dual.clamp_(min=self.lambda_min, max=self.lambda_max)

    # ============================================================ 辅助方法 (与 baseline 同语义) ============================================================
    def _sample_actions(self, states):
        """
        从当前高斯策略采样动作 (重参数化), 支持 [*, sd] 批量输入。
        返回: [*, ad]。调用方负责是否包在 no_grad 里 (采集/target/baseline 用 no_grad)。
        """
        means = self.actor(states)                                 # μ(s), [*, ad]
        std = torch.exp(self.actor.log_std).view(1, -1).expand_as(means)  # σ 广播, [*, ad]
        return means + torch.randn_like(means) * std               # a = μ + σ·ε

    def _aug(self, states, steps):
        """
        critic 专用: 给状态追加 step 特征 t/n → [*, sd+1]。actor 调用【不】用此函数 (保持 step-blind)。
        critic_step_feature=False 时原样返回 (退化为 step-blind, 作 A/B 对照)。

        Args:
            states: [*, sd]
            steps:  [*] 标量步数索引 (float/int), 或 python 标量 (会广播)
        """
        if not self.critic_step_feature:
            return states
        if not torch.is_tensor(steps):                         # 标量 → 广播到 [*]
            steps = torch.full((states.shape[0],), float(steps), device=states.device)
        sf = (steps.float() / self.n).reshape(-1, 1)           # [*,1] 归一化步数 t/n
        return torch.cat([states, sf], dim=1)                  # [*, sd+1]

    def _compute_log_probs(self, states, actions):
        """
        对角高斯 logπ(a|s) = Σ_dim [-0.5((a-μ)²/σ² + 2logσ + log2π)]。
        对 μ (actor 权重) 可导 → 提供策略梯度; actions 来自 rollout (无梯度)。
        返回: [*]
        """
        means = self.actor(states)                                 # [*, ad], 对 θ 可导
        std = torch.exp(self.actor.log_std).view(1, -1).expand_as(means)
        var = std.pow(2)
        log_probs = -0.5 * (((actions - means) ** 2) / var
                            + 2.0 * torch.log(std) + self._log_2pi)
        return log_probs.sum(dim=-1)                               # 对动作维求和 → [*]

    def _entropy(self, states):
        """高斯熵 H = Σ_dim [0.5(1+log2π) + logσ]; log_std 固定 ⇒ 常数, 仅为接口完整。"""
        ent = (0.5 * (1.0 + self._log_2pi) + self.actor.log_std).sum()
        return ent.expand(states.shape[0])                         # [*]

    def _estimate_baselines(self, states, budgets, steps=None):
        """
        用 K 个动作样本近似两个 baseline:
            V̂_m(s) = E_{a~π}[Q̂_m(s,a)],   V̂_c(s,b) = E_{a~π}[Ψ̂(s,a,b)]
        steps: critic step 增广用 (与 states 对齐的步数); 动作采样仍用 step-blind states。
        返回: v_m [*], v_c [*] (在 no_grad 上下文中调用)
        """
        b_col = budgets.unsqueeze(1)                               # [*, 1], 供 CDF 查询广播
        s_aug = self._aug(states, steps) if steps is not None else states  # critic 用增广态
        q_list, c_list = [], []
        for _ in range(self.num_action_samples):
            a = self._sample_actions(states)                       # 重新采样动作 [*, ad] (actor step-blind)
            psi = self.critic(s_aug, a)                            # [*, N]  critic 用增广态
            q_list.append(psi.mean(dim=1))                         # Q̂_m, [*]
            c_list.append((psi <= b_col).float().mean(dim=1))      # Ψ̂(·,b), [*]
        v_m = torch.stack(q_list, dim=0).mean(dim=0)               # 对 K 取平均 → [*]
        v_c = torch.stack(c_list, dim=0).mean(dim=0)
        return v_m, v_c

    def _maybe_norm(self, x, eps=1e-8):
        """
        优势归一化 (advantage_norm=='separate' 时): (x-mean)/std, 在整个 [n·B] 批上做。
        单样本 (numel<=1) 跳过; std 过小时只中心化。'none' 时原样返回。
        """
        if self.advantage_norm != 'separate' or x.numel() <= 1:    # 'none' 或单样本: 不归一化
            return x
        std = x.std(unbiased=False)                                # 批标准差 (有偏, 与 baseline 一致)
        if std.item() < eps:                                       # std 过小 → 只中心化, 避免除零放大
            return x - x.mean()
        return (x - x.mean()) / (std + eps)                        # 标准化 (x-mean)/std

    def _ema_update(self, rms, batch_mean, batch_var):
        """
        对一个 RunningMeanStd 做【批量】EMA 更新 (qcpo 专用)。

        RunningMeanStd 内置的 .update(x) 只接受【标量】(QCPO 每 episode 一条回报喂一次);
        本版每迭代一次产生 B / n·B 个样本, 必须按整批的一阶/二阶【矩】混入, 才能让 .std 平滑
        跟踪【样本分布】的尺度 —— 若误用 .update(batch.mean()) 则 var 退化成"批均值的方差"≈0。
        本方法直接改写 rms 的 .mean/.var/._initialized, 复用其 .std 的 sqrt(max(var,1e-8)) 下限保护。

        Args:
            rms:        RunningMeanStd 实例 (self.return_rms 或 self.constraint_rms)
            batch_mean: 本批样本均值 (python float)
            batch_var:  本批样本方差 (有偏 unbiased=False, python float)
        更新公式 (α=rms.decay): mean ← (1-α)·mean + α·batch_mean;  var ← (1-α)·var + α·batch_var
        """
        if not rms._initialized:                                   # 首批: 直接用本批统计量初始化, 避免 mean=0 偏差
            rms.mean = batch_mean
            rms.var = max(batch_var, 1e-8)
            rms._initialized = True
            return
        a = rms.decay                                              # EMA 衰减系数 α
        rms.mean = (1.0 - a) * rms.mean + a * batch_mean           # EMA 均值
        rms.var = (1.0 - a) * rms.var + a * batch_var              # EMA 方差 (→ .std 平滑跟踪分布尺度)

    def _update_norm_stats(self, batch):
        """
        qcpo 模式: 每迭代用本次 B 条轨迹刷新两个 EMA 归一化器 (整批矩 → EMA, 跨迭代平滑)。

          return_rms     ← 本批【轨迹回报】disc_returns [B] 的 (mean,var) → σ_ret 跟踪回报分布尺度(≈12)
          constraint_rms ← 本批【约束优势】(ψ_c - v_c) [n·B] 的 (mean,var) → σ_c 跟踪约束优势尺度(~0.x)

        约束优势用【当前 critic】在 rollout 批上现算一次 (与 update_actor 同口径), 不建图。
        刷新发生在内层 updates_per_episode 循环【之前】 → 该尺度在内层更新中冻结 → 稳定 (反卷绕 λ 的关键)。
        """
        # --- 回报尺度 σ_ret: 直接来自本批 B 条轨迹的标量回报 (与 QCPO 的 return_rms 同义) ---
        z = batch['disc_returns'].detach()                         # [B] 本迭代各轨迹回报 U(τ)
        self._ema_update(self.return_rms,
                         float(z.mean().item()),
                         float(z.var(unbiased=False).item()))

        # --- 约束优势尺度 σ_c: 用当前 critic 在 rollout 批上算一次 Â_c=(ψ_c - v_c) 的批方差 ---
        with torch.no_grad():                                      # 仅取尺度, 不建图
            s, a, b = batch['states'], batch['actions'], batch['budgets']
            steps = batch['steps']                                 # critic step 增广用
            psi = self.critic(self._aug(s, steps), a)              # ψ_i(s,a), [n·B, N] (critic 增广当前步)
            psi_c = (psi <= b.unsqueeze(1)).float().mean(dim=1)    # Ψ̂(s,a,b)=(1/N)Σ1{ψ_i≤b}, [n·B]
            _, v_c = self._estimate_baselines(s, b, steps)         # V̂_c(s,b) baseline, [n·B]
            adv_c = psi_c - v_c                                    # 约束优势 (raw, 未归一化), [n·B]
        self._ema_update(self.constraint_rms,
                         float(adv_c.mean().item()),
                         float(adv_c.var(unbiased=False).item()))

    def _soft_update_target(self):
        """Polyak 软更新: target ← (1-τ)·target + τ·online。"""
        with torch.no_grad():
            for tp, op in zip(self.target_critic.parameters(), self.critic.parameters()):
                tp.data.lerp_(op.data, self.target_tau)            # lerp_: tp = tp + τ(op - tp)

    def _initial_cdf_estimate(self, s0):
        """
        critic 估计的 P(Z≤q): 在本次迭代 B 个初始状态 s_0 上求 mean Ψ̂(s_0,a_0,q)。
        (仅作校准诊断记录, 不驱动 λ —— λ 用经验违反率, 更鲁棒。)

        输入: s0 [B, sd]
        """
        with torch.no_grad():
            a0 = self._sample_actions(s0)                          # [B, ad] (actor step-blind)
            psi = self.critic(self._aug(s0, 0), a0)                # [B, N]  critic 增广 step=0 (s0 是第 0 步)
            q_b = torch.full((s0.shape[0],), self.quantile_threshold,
                             dtype=torch.float32, device=self.device)
            self.last_cdf_initial = float((psi <= q_b.unsqueeze(1)).float().mean(dim=1).mean())
            # critic 对回报分布 Z|s_0 的形状估计: 用 N 个分位数近似其均值/标准差 (诊断 critic 是否在变宽)
            self.last_pred_return_mean = float(psi.mean().item())             # E[Z|s_0] ≈ mean_i ψ_i
            self.last_pred_return_std = float(psi.std(dim=1).mean().item())   # std(Z|s_0) ≈ 分位数的标准差, 再对 B 平均
        return self.last_cdf_initial

    def _compute_episode_avg_risk(self):
        """读取 vec_env.render() 返回的本迭代每步 batch 平均风险序列, 取均值 (与 baseline 同义)。"""
        if hasattr(self.vec_env, 'render'):
            risk_series = self.vec_env.render()                    # [T] numpy, 每步 batch 平均 risk
            if risk_series is not None:
                try:
                    if len(risk_series) > 0:                       # 防空序列
                        return float(np.mean(risk_series))         # 整条 episode 的平均风险等级
                except TypeError:
                    pass                                           # render 返回不可迭代时静默跳过
        return 0.0                                                 # 无 render / 空 → 0.0

    def _log(self, it, batch, critic_info, actor_info):
        """
        wandb 日志 (与 baseline 同款指标 + budget 统计 + critic 校准诊断 + 进度三轴)。
        各回报统计直接用本次迭代 B 条轨迹的 disc_returns (不再用滚动窗口)。
        """
        z = batch['disc_returns'].detach().cpu().numpy()           # [B] 本迭代各轨迹回报
        empirical_prob = float(np.mean(z <= self.quantile_threshold))  # P̂(Z≤q)
        avg_return = float(np.mean(z))
        return_std = float(np.std(z))                              # 真实回报标准差 (直接显示"分布在变宽")
        quantile_return = float(np.percentile(z, self.q_alpha * 100))
        budgets_np = batch['budgets'].detach().cpu().numpy()
        self.last_empirical_prob = empirical_prob

        # 先算 critic 的初始态估计 (Ĝ + 预测均值/std), 之后用来算"校准误差"
        ghat = self._initial_cdf_estimate(batch['s0'])             # critic 估计的 P(Z≤q)

        # 进度三轴: 迭代 / 累计轨迹数 / 累计 env-step (env-step 已设为默认 x 轴)
        env_steps = (it + 1) * self.num_envs * self.n
        trajectories = (it + 1) * self.num_envs

        log_dict = {
            'disc_reward/discounted_reward': avg_return,            # 本迭代平均回报 (无单条概念, 用均值)
            'disc_reward/aver_reward': avg_return,
            'disc_reward/quantile_reward': quantile_return,
            'disc_reward/return_std': return_std,                  # 真实回报 std (Q2/Q4: 风险↑→方差↑)
            'quantile/q_est': quantile_return,                     # 经验 α-分位数 (监控)
            'quantile/margin_to_threshold': quantile_return - self.quantile_threshold,
            'constraint/empirical_prob': empirical_prob,           # P̂(Z≤q)
            'constraint/margin': self.q_alpha - empirical_prob,    # α - P̂, 正值=约束满足
            'constraint/cdf_estimate_initial': ghat,               # critic 估计 P(Z≤q) (诊断)
            'constraint/cdf_calibration_error': abs(ghat - empirical_prob),  # |Ĝ - 经验|: critic 校准误差
            'critic/pred_return_mean': self.last_pred_return_mean, # critic 估计 E[Z|s0] (对比 avg_return 看偏差)
            'critic/pred_return_std': self.last_pred_return_std,   # critic 估计 std(Z|s0) (对比 return_std 看是否跟得上)
            'constraint/dual_prob': self.last_dual_prob,
            'lambda/value': float(self.lambda_dual.detach().item()),
            'lambda/lr': float(self.lambda_scheduler.get_last_lr()[0]),
            'action/avg_risk_episode': self._compute_episode_avg_risk(),
            'budget/min': float(np.min(budgets_np)),
            'budget/max': float(np.max(budgets_np)),
            'budget/mean': float(np.mean(budgets_np)),
            'training/learning_steps': self.learning_steps,
            'training/actor_lr': float(self.actor_scheduler.get_last_lr()[0]),
            'progress/iteration': it,
            'progress/trajectories': trajectories,
            'progress/env_steps': env_steps,
        }
        log_dict.update(critic_info)
        log_dict.update(actor_info)
        if self.advantage_norm == 'qcpo':                          # qcpo: 记录两个 EMA 归一化尺度 (诊断 λ 量级是否合理)
            log_dict['norm/return_sigma_ema'] = float(self.return_rms.std)        # σ_ret (≈ 回报 std)
            log_dict['norm/constraint_sigma_ema'] = float(self.constraint_rms.std)  # σ_c (约束优势 std)
        wandb.log(log_dict, step=it)

        # 周期性控制台打印 (与 baseline 风格一致)
        if it % self.log_interval == 0 and it != 0:               # 每 log_interval 迭代打印一次
            print(f'Iter:{it:05d} (env_step:{env_steps}) || disc_a_r:{avg_return:.03f} '
                  f'disc_q_r:{quantile_return:.03f} lambda:{self.lambda_dual.item():.04f}')  # 第1行: 回报 + λ
            print(f'Iter:{it:05d} || P(Z<=q):{empirical_prob:.03f} '
                  f'alpha:{self.q_alpha:.03f} Ghat_critic:{self.last_cdf_initial:.03f} '
                  f'critic_loss:{critic_info.get("critic/quantile_huber_loss", 0.0):.04f}\n')  # 第2行: 约束诊断

    # ============================================================ 评估接口 (与 baseline 一致, 走 numpy eval_env) ============================================================
    def choose_action(self, state):
        """训练/评估时采样动作: 输入扁平 numpy 状态, 输出 (ad,) float32 numpy 动作。"""
        s = torch.as_tensor(np.asarray(state, dtype=np.float32).reshape(1, -1), device=self.device)
        with torch.no_grad():
            a = self._sample_actions(s)                            # [1, ad]
        return a.squeeze(0).cpu().numpy().astype(np.float32)       # (ad,) float32, 满足 env 动作空间

    def select_action(self, state):
        """评估时采样动作 (与 choose_action 同, 随机策略; 供 monte_carlo_evaluate 调用)。"""
        return self.choose_action(state)

    def get_training_summary(self):
        """暴露最终约束指标给 run_experiment.py (hasattr 调用, 可选)。"""
        return {
            'lambda_final': float(self.lambda_dual.detach().item()),
            'empirical_violation_prob': self.last_empirical_prob,
            'cdf_estimate_initial': self.last_cdf_initial,
            'dual_prob': self.last_dual_prob,
            'beta': self.beta,
            'num_envs': self.num_envs,
            'num_iterations': self.num_iterations,
        }
