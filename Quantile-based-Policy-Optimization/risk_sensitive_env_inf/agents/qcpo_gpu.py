# -*- coding: utf-8 -*-
"""
QCPO_GPU —— QCPO 的【全 GPU、B 路并行向量化】参考实现 (QCPOGPU)

目的: 给 DQCACBetaGPU 提供一个【同机、快速收敛、行为对齐原始 QCPO】的对照基准。
     原始 `agents/qcpo.py` (串行单 env, 逐 episode) 收敛慢; 本文件【不改动】它,
     新建一个全 GPU 向量化版, 算法逐式对齐, 仅把"逐 episode 串行"换成"一次 B 条并行 + 批量更新"。

【与原始 QCPO 的算法对齐】(agents/qcpo.py):
    优化问题:  max_θ E[U(τ)]  s.t.  P_θ(U(τ) ≤ q) ≤ α      (U(τ)=Σ γ^t r_t)
    策略梯度:  每个 timestep 的梯度权重 = (U(τ)-μ)/σ  -  λ·𝟙{U(τ) ≤ q}
               · 均值项 (U-μ)/σ: 用 RunningMeanStd(EMA, decay=0.01) 归一化 (与 qcpo.py 同)
               · 约束项 λ·𝟙{U≤q}: 用【原始尺度】经验指示, 权重恒为 1, 广播到整条轨迹所有 timestep
               · 注意: 两项都【没有】γ^t / β^t 折扣, 也【没有】baseline (与 qcpo.py 逐式一致)
    Dual:      λ ← [λ + ε_k·(P̂(U≤q) - α)]_+   (经验违反概率驱动, 与 qcpo.py 同)

【向量化(与 DQCACBetaGPU 同源)固有差异, 非 bug】:
    1. B 条轨迹用同一【冻结策略】并行采集 (原版串行、轨迹间策略已漂移) → 非逐位一致。
    2. 每迭代内层 updates_per_iteration 次复用同一 batch 做策略更新 (原版每 episode 1 次, 但有 max_episode 次);
       取 updates_per_iteration=10 与 DQCACBetaGPU 对齐 → 同样的"更新结构", 公平对比。
    3. return_rms 每迭代用本批 B 条轨迹回报【逐条】更新 (与原版"每 episode 一条"语义一致, 同 decay=0.01)。
    4. 经验 P̂(U≤q) 直接用本迭代 B 条轨迹 (B 已比原版 outer_interval 窗口紧)。
"""

import numpy as np
import torch
from torch.optim import Adam                              # 与原版一致: base lr=1 + LambdaLR
from torch.optim.lr_scheduler import LambdaLR             # lr=a/(b+k)^c
import wandb                                              # 统一日志

from utils import Actor, RunningMeanStd                   # Actor: 线性高斯策略; RunningMeanStd: EMA 回报归一化器
from envs import RiskSensitiveVecTorch                    # 全 GPU、B 路并行【无状态 continuing】环境 (与 DQCACBetaGPU 共用)
from .common import lr_lambda                              # ∞-horizon setting 内共享件 (见 common.py)


class QCPOGPU(object):
    """QCPO 全 GPU 向量化版。算法逐式对齐 qcpo.QCPO, 仅数据流改为 B 路并行批量。"""

    # ============================================================ 初始化 ============================================================
    def __init__(self, args, env):
        # -------------------- 基础参数 (与 qcpo.py 同名) --------------------
        self.device = args.device                          # 计算设备 (全程在此)
        self.gamma = args.gamma                            # 折扣 γ (算 U(τ) 用)
        self.q_alpha = args.q_alpha                        # 约束水平 α
        self.quantile_threshold = args.quantile_threshold  # 约束阈值 q
        self.log_interval = args.log_interval              # 控制台打印间隔 (按迭代)
        self.outer_interval = max(1, int(getattr(args, 'outer_interval', 1)))  # λ 更新间隔 (按迭代)

        # -------------------- 向量化训练专属超参 (与 DQCACBetaGPU 对齐) --------------------
        self.num_envs = max(1, int(getattr(args, 'num_envs', 256)))            # 并行 env 数 B
        self.num_iterations = max(1, int(getattr(args, 'num_iterations', 400)))  # 迭代次数
        self.updates_per_iteration = max(1, int(getattr(args, 'updates_per_episode', 10)))  # 每迭代内层策略更新次数
        self.warmup_rms_iters = max(1, int(getattr(args, 'warmup_rms_iters', 2)))  # return_rms 预热迭代数 (只采集不更新策略)
        self.actor_grad_clip = float(getattr(args, 'actor_grad_clip', 0.0))    # actor 梯度裁剪 (0=不裁, 原版 QCPO 不裁)

        # -------------------- 环境与维度 --------------------
        self.env_name = args.env_name
        self.eval_env = env                                                    # numpy env, 仅评估用
        self.vec_env = RiskSensitiveVecTorch(num_envs=self.num_envs,
                                             device=self.device, ref_env=env)  # 与 baseline 同分布
        self.n = self.vec_env.n                                                # episode 步长
        self.state_dim = int(np.prod(env.observation_space.shape))             # 状态维 (=n)
        self.action_dim = int(np.prod(env.action_space.shape))                 # 动作维 (=1)
        self._log_2pi = float(np.log(2.0 * np.pi))                             # 高斯 logπ 常量

        # -------------------- Actor (与 qcpo.py 同一线性高斯策略) + return 归一化器 --------------------
        self.actor = Actor(self.state_dim, self.action_dim, args.init_std).to(self.device)  # μ(s)=W·s
        # return_rms: EMA 回报归一化 (与 qcpo.py 完全一致, 默认 decay=0.01 ≈ 最近 100 条轨迹)
        self.return_rms = RunningMeanStd(decay=float(getattr(args, 'norm_ema_decay', 0.01)))

        # 策略优化器 + 两时间尺度 LR (与 qcpo.py 同: base lr=1, 真实 lr 由 LambdaLR 给 lr=a/(b+k)^c)
        self.optimizer = Adam(self.actor.parameters(), 1.0, eps=1e-5)
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda k: lr_lambda(k, args.theta_a, args.theta_b, args.theta_c)
        )

        # -------------------- 拉格朗日乘子 λ (与 qcpo.py 一致) --------------------
        self.lambda_dual = torch.tensor([0.0], dtype=torch.float32,
                                        device=self.device, requires_grad=True)
        self.lambda_optimizer = Adam([self.lambda_dual], 1.0, eps=1e-5)
        self.lambda_scheduler = LambdaLR(
            self.lambda_optimizer,
            lr_lambda=lambda k: lr_lambda(k, args.lambda_a, args.lambda_b, args.lambda_c)
        )

        # -------------------- 运行时统计 --------------------
        self.last_empirical_prob = 0.0                     # 上次经验 P(U≤q)
        self.last_dual_prob = 0.0                          # 上次 dual 用的经验违反率

        # -------------------- wandb (∞-horizon: 独立 project 'risk_sensitive_inf', 与 DQCACBetaGPU 同) --------------------
        wandb.init(project=getattr(args, 'wandb_project', 'risk_sensitive_inf'),
                   name=getattr(args, 'wandb_name', None) or f"{args.algo_name}_{args.seed}",
                   config=vars(args), reinit=True, group=args.algo_name,
                   dir=getattr(args, 'wandb_dir', None))
        wandb.define_metric("progress/env_steps")
        wandb.define_metric("*", step_metric="progress/env_steps")

    # ============================================================ 主训练循环 ============================================================
    def train(self):
        """
        主循环。每迭代: 并行采 B 条轨迹 → 刷新 return_rms → 内层 updates_per_iteration 次策略更新
                        → 每 outer_interval 迭代更新一次 λ → 日志 + LR 调度。
        """
        print(f"QCPOGPU: alpha={self.q_alpha}, q={self.quantile_threshold}, "
              f"B={self.num_envs}, iters={self.num_iterations}, "
              f"updates/iter={self.updates_per_iteration}, device={self.device}")

        # ===== return_rms 预热 (只采集冻结轨迹刷新归一化器, 不更新策略; 对齐 qcpo.warm_up) =====
        for _ in range(self.warmup_rms_iters):
            warm = self._rollout_vec()
            self._update_rms(warm['disc_returns'])
        print(f"QCPOGPU warm up || rms_mean:{self.return_rms.mean:.3f} rms_std:{self.return_rms.std:.3f}")

        for it in range(self.num_iterations):
            # ===== 1. 并行采 B 条轨迹 (冻结策略) =====
            batch = self._rollout_vec()

            # ===== 2. 刷新 return_rms (逐条轨迹回报, 与原版每 episode 更新语义一致) =====
            self._update_rms(batch['disc_returns'])

            # ===== 3. 内层策略更新 (复用本批, updates_per_iteration 次) =====
            actor_info = {}
            for _ in range(self.updates_per_iteration):
                actor_info = self.update_actor(batch)

            # ===== 4. 外层 λ 更新 (每 outer_interval 迭代) =====
            if it % self.outer_interval == 0:
                self.update_dual(batch['disc_returns'])

            # ===== 5. 日志 + LR 调度 =====
            self._log(it, batch, actor_info)
            self.scheduler.step()
            self.lambda_scheduler.step()

    # ============================================================ 向量化采样 B 条轨迹 ============================================================
    def _rollout_vec(self):
        """
        用冻结策略并行采 B 条完整轨迹 (固定 n 步、B 路锁步), 全程 GPU。
        返回: states [n·B, sd], actions [n·B, ad], disc_returns [B] (每条轨迹 U(τ)=Σγ^t r_t)。
        """
        n, B = self.n, self.num_envs
        s = self.vec_env.reset()                               # [B, sd]
        S, A = [], []                                          # 逐步收集 states/actions
        disc_return = torch.zeros(B, dtype=torch.float32, device=self.device)  # U(τ), [B]
        disc = 1.0                                             # γ^t

        for t in range(n):
            with torch.no_grad():                              # 冻结策略采集, 不建图
                a = self._sample_actions(s)                    # a~π(·|s), [B, ad]
            s2, r, done = self.vec_env.step(a)                 # s2 [B,sd], r [B]
            S.append(s); A.append(a)
            disc_return = disc_return + disc * r               # 累计 U(τ)
            disc *= self.gamma
            s = s2

        states = torch.stack(S, dim=0).reshape(n * B, -1)      # [n·B, sd] (行序: t*B+j)
        actions = torch.stack(A, dim=0).reshape(n * B, -1)     # [n·B, ad]
        return {
            'states': states,            # [n·B, sd]
            'actions': actions,          # [n·B, ad]
            'disc_returns': disc_return, # [B]  每条轨迹 U(τ)
            's0': S[0],                  # [B, sd] 初始状态 (日志用)
        }

    # ============================================================ Actor 更新 (与 qcpo.update_inner 对齐) ============================================================
    def update_actor(self, batch):
        """
        一次策略梯度更新 (REINFORCE + Reward Normalization, 与 qcpo.py 逐式一致):
            weight(τ) = (U(τ)-μ)/σ  -  λ·𝟙{U(τ) ≤ q}       (每条轨迹一个标量)
            loss = -E_{(t,env)}[ logπ(a_t|s_t) · weight(env) ]   (weight 广播到该轨迹所有 timestep)
        · 均值项归一化、约束项原始尺度、权重恒 1、无 γ^t、无 baseline —— 与原版完全相同。
        """
        n, B = self.n, self.num_envs
        U = batch['disc_returns'].detach()                     # [B] 每条轨迹 U(τ)
        # 约束指示 (原始尺度, 不归一化): 𝟙{U(τ) ≤ q}
        ind = (U <= self.quantile_threshold).float()           # [B]
        # 均值项归一化: (U-μ)/σ, μ/σ 来自 EMA return_rms (本迭代已冻结)
        U_norm = (U - self.return_rms.mean) / self.return_rms.std  # [B]
        # 梯度权重 (每条轨迹一个标量): (U-μ)/σ - λ·𝟙{U≤q}
        weight = U_norm - self.lambda_dual.detach() * ind      # [B]
        # 广播到该轨迹所有 n 个 timestep → [n, B] → [n·B] (行序与 states 对齐: t*B+j)
        weight_full = weight.unsqueeze(0).expand(n, B).reshape(n * B)  # [n·B]

        # logπ(a_t|s_t), 对 θ 可导
        log_probs = self._compute_log_probs(batch['states'], batch['actions'])  # [n·B]
        # loss = -E[logπ · weight]; optimizer 下降 = 梯度上升 (与 qcpo.py 同)
        actor_loss = -(log_probs * weight_full.detach()).mean()

        self.optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.actor_grad_clip and self.actor_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip)
        self.optimizer.step()

        return {'actor/loss': float(actor_loss.item()),
                'actor/weight_mean': float(weight.mean().item()),
                'actor/weight_std': float(weight.std(unbiased=False).item())}

    # ============================================================ Dual 更新 (与 qcpo.update_dual 对齐) ============================================================
    def update_dual(self, disc_returns):
        """
        λ ← [λ + ε_k·(P̂(U≤q) - α)]_+, 用本迭代 B 条轨迹的经验违反概率 (与 qcpo.py 同)。
        loss=-λ(P̂-α) ⇒ optimizer 下降一步 = λ←λ+lr(P̂-α); 再投影到 [0,∞)。
        """
        z = disc_returns.detach()                              # [B]
        p = float((z <= self.quantile_threshold).float().mean().item())  # P̂(U≤q)
        self.last_dual_prob = p
        gap = torch.tensor([p - self.q_alpha], dtype=torch.float32, device=self.device)
        self.lambda_optimizer.zero_grad(set_to_none=True)
        (-self.lambda_dual * gap).backward()
        self.lambda_optimizer.step()
        with torch.no_grad():
            self.lambda_dual.clamp_(min=0.0)                   # 投影 λ≥0

    # ============================================================ 辅助 (与 DQCACBetaGPU 同语义) ============================================================
    def _update_rms(self, disc_returns):
        """用本批 B 条轨迹回报【逐条】更新 EMA return_rms (与原版'每 episode 一条'语义一致)。"""
        for z in disc_returns.detach().cpu().tolist():
            self.return_rms.update(z)

    def _sample_actions(self, states):
        """从当前高斯策略重参数化采样: a=μ(s)+σ·ε, 支持 [*,sd] 批量。返回 [*,ad]。"""
        means = self.actor(states)                             # μ(s)
        std = torch.exp(self.actor.log_std).view(1, -1).expand_as(means)
        return means + torch.randn_like(means) * std

    def _compute_log_probs(self, states, actions):
        """对角高斯 logπ(a|s)=Σ_dim[-0.5((a-μ)²/σ²+2logσ+log2π)], 对 θ 可导。返回 [*]。"""
        means = self.actor(states)
        std = torch.exp(self.actor.log_std).view(1, -1).expand_as(means)
        var = std.pow(2)
        log_probs = -0.5 * (((actions - means) ** 2) / var
                            + 2.0 * torch.log(std) + self._log_2pi)
        return log_probs.sum(dim=-1)

    def _compute_episode_avg_risk(self):
        """读取 vec_env 本迭代每步 batch 平均风险序列, 取均值 (日志用)。"""
        if hasattr(self.vec_env, 'render'):
            risk_series = self.vec_env.render()
            if risk_series is not None:
                try:
                    if len(risk_series) > 0:
                        return float(np.mean(risk_series))
                except TypeError:
                    pass
        return 0.0

    def _log(self, it, batch, actor_info):
        """wandb 日志 (与 DQCACBetaGPU 同款 + return 归一化统计 + 进度三轴)。"""
        z = batch['disc_returns'].detach().cpu().numpy()       # [B] 本迭代回报
        empirical_prob = float(np.mean(z <= self.quantile_threshold))  # P̂(U≤q)
        avg_return = float(np.mean(z))
        return_std = float(np.std(z))
        quantile_return = float(np.percentile(z, self.q_alpha * 100))
        self.last_empirical_prob = empirical_prob

        env_steps = (it + 1) * self.num_envs * self.n
        trajectories = (it + 1) * self.num_envs
        log_dict = {
            'disc_reward/discounted_reward': avg_return,
            'disc_reward/aver_reward': avg_return,
            'disc_reward/quantile_reward': quantile_return,
            'disc_reward/return_std': return_std,
            'quantile/q_est': quantile_return,
            'quantile/margin_to_threshold': quantile_return - self.quantile_threshold,
            'constraint/empirical_prob': empirical_prob,
            'constraint/margin': self.q_alpha - empirical_prob,
            'constraint/dual_prob': self.last_dual_prob,
            'lambda/value': float(self.lambda_dual.detach().item()),
            'lambda/lr': float(self.lambda_scheduler.get_last_lr()[0]),
            'normalize/return_mean': float(self.return_rms.mean),
            'normalize/return_std': float(self.return_rms.std),
            'action/avg_risk_episode': self._compute_episode_avg_risk(),
            'training/actor_lr': float(self.scheduler.get_last_lr()[0]),
            'progress/iteration': it,
            'progress/trajectories': trajectories,
            'progress/env_steps': env_steps,
        }
        log_dict.update(actor_info)
        wandb.log(log_dict, step=it)

        if it % self.log_interval == 0 and it != 0:
            print(f'Iter:{it:05d} (env_step:{env_steps}) || disc_a_r:{avg_return:.03f} '
                  f'disc_q_r:{quantile_return:.03f} lambda:{self.lambda_dual.item():.04f}')
            print(f'Iter:{it:05d} || P(U<=q):{empirical_prob:.03f} '
                  f'alpha:{self.q_alpha:.03f} margin:{self.q_alpha - empirical_prob:.03f}\n')

    # ============================================================ 评估接口 (与 DQCACBetaGPU 一致) ============================================================
    def choose_action(self, state):
        """采样动作: 扁平 numpy state → (ad,) float32 numpy。"""
        s = torch.as_tensor(np.asarray(state, dtype=np.float32).reshape(1, -1), device=self.device)
        with torch.no_grad():
            a = self._sample_actions(s)
        return a.squeeze(0).cpu().numpy().astype(np.float32)

    def select_action(self, state):
        """评估采样动作 (随机策略, 供 monte_carlo_evaluate)。"""
        return self.choose_action(state)

    def get_training_summary(self):
        """暴露最终约束指标 (run_experiment.py 可选调用)。"""
        return {
            'lambda_final': float(self.lambda_dual.detach().item()),
            'empirical_violation_prob': self.last_empirical_prob,
            'dual_prob': self.last_dual_prob,
            'num_envs': self.num_envs,
            'num_iterations': self.num_iterations,
        }
