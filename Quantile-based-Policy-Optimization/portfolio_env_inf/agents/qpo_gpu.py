# -*- coding: utf-8 -*-
"""
QPOGPU —— QPO (Quantile Policy Optimization) 的全 GPU、B 路并行向量化版 (portfolio_env_inf)。

算法对齐 risk_sensitive_env/agents/qpo.py (两时间尺度 QPO):
    目标:        max_θ Q_α(Z),  Z = Σ γ^t r_t   (无约束, 纯分位数最大化 → 天然保守基线)
    策略梯度:    ∇θ Q_α ∝ -E[∇θ logπ · 𝟙{Z(τ) ≤ Q}]  (分位数梯度的示性函数形式)
                 → loss = +mean(logπ(a_t|s_t)·𝟙{Z(τ)≤Q})  (optimizer 下降 = 压低 P(Z≤Q) = 抬高分位数)
                 𝟙 按轨迹广播到所有 timestep (轨迹级全有/全无, 与 qpo.py 一致)
    分位数追踪:  Q 的梯度 = -(α - E[𝟙{Z≤Q}])  →  Q ← Q + lr_q·(α - P̂(Z≤Q))  (两时间尺度)
    无归一化 / 无 baseline / 无 λ —— 与 qpo.py 逐式一致。

向量化结构与 QCPOGPU 完全对齐 (B 条并行 + 每迭代 updates_per_episode 次复用 batch),
保证与 QCPO/QPPO/DQCAC 的"更新结构"一致 → 公平对比。
"""
import numpy as np
import torch
from torch.optim import Adam                                  # base lr=1 + LambdaLR (模板配方)
from torch.optim.lr_scheduler import LambdaLR                 # lr = a/(b+k)^c

from .vec_base import VecAgentBase                            # 共享: env/策略/rollout/日志
from .common import lr_lambda, indicator                      # 共享件 (与 risk_sensitive_inf 同源)


class QPOGPU(VecAgentBase):
    """QPO 全 GPU 向量化版: 纯分位数最大化 (两时间尺度), 四算法对比中的保守基线。"""

    # ============================================================ 初始化 ============================================================
    def __init__(self, args, env):
        super().__init__(args, env)                            # 基类: env/actor/wandb/维度

        # -------------------- 算法专属超参 --------------------
        self.updates_per_iteration = max(1, int(getattr(args, 'updates_per_episode', 10)))  # 每迭代复用 batch 次数
        self.warmup_q_iters = max(1, int(getattr(args, 'warmup_q_iters', 2)))  # Q 预热迭代数 (估初始分位数)
        self.actor_grad_clip = float(getattr(args, 'actor_grad_clip', 0.0))    # 0=不裁 (qpo.py 不裁)

        # -------------------- 两时间尺度优化器 --------------------
        # θ 优化器: 默认 SGD (本环境关键修复, 见 DESIGN.md "稳定性修复"):
        #   原版 qpo.py 用 Adam, 但 Adam 把每坐标步长归一化到 ~lr —— 本环境 obs 含 20 维
        #   纯噪声 est 特征, 对应权重坐标的梯度是纯噪声, Adam 下变成恒速随机游走 →
        #   W 范数爆炸 → 状态噪声放大成 logits 方差 → 回报方差爆炸 (探针实证)。
        #   SGD 步长 ∝ 真实梯度幅度, 噪声坐标自然得到小步 (也更忠实原论文的 Robbins-Monro SA)。
        self.theta_optimizer_name = str(getattr(args, 'theta_optimizer', 'sgd')).lower()
        if self.theta_optimizer_name == 'adam':
            self.optimizer = Adam(self.actor.parameters(), 1.0, eps=1e-5)
        else:
            self.optimizer = torch.optim.SGD(self.actor.parameters(), 1.0)
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda k: lr_lambda(k, args.theta_a, args.theta_b, args.theta_c))

        # Q (分位数估计) 追踪模式 (本环境关键修复 2):
        #   'ema' (默认): q_est ← (1-ρ)q_est + ρ·批经验分位数。向量化下 B=512 的批分位数是
        #          低噪声估计, EMA 几迭代内收敛 → 恢复"q 快、θ 慢"两时间尺度。
        #          (原 SA 路线在按迭代步进的 GPU 版里每迭代只挪 ~lr_q, 滞后数百迭代 → 失效, 探针实证)
        #   'sa':  原版随机近似更新 (Adam + α-P̂ 梯度), 作 A/B 对照。
        self.q_track_mode = str(getattr(args, 'q_track_mode', 'ema')).lower()
        self.q_track_rho = float(getattr(args, 'q_track_rho', 0.3))   # EMA 步长 ρ
        self.q_est = None                                      # torch 标量 (sa 模式 requires_grad)
        self._q_args = (args.q_a, args.q_b, args.q_c)          # Q 的 LR 调度参数 (sa 模式用)
        self.q_optimizer = None
        self.q_scheduler = None

    # ============================================================ 主训练循环 ============================================================
    def train(self):
        """
        预热 (估初始 Q) → 每迭代: 并行采 B 条轨迹 → 内层 updates_per_iteration 次策略更新
        → 一次 Q 更新 (两时间尺度由 lr 调度差异保证) → 统一日志 + LR 调度。
        """
        print(f"QPOGPU: alpha={self.q_alpha}, B={self.num_envs}, iters={self.num_iterations}, "
              f"updates/iter={self.updates_per_iteration}, device={self.device}")

        # ===== Q 预热: 冻结初始策略采样, 用经验 α-分位数初始化 Q (对齐 qpo.warm_up) =====
        warm_z = []
        for _ in range(self.warmup_q_iters):
            warm_z.append(self._rollout_core()['disc_returns'])
        q0 = float(np.percentile(torch.cat(warm_z).cpu().numpy(), self.q_alpha * 100))
        print(f"QPOGPU warm up || init q_est:{q0:.3f} (q_track_mode={self.q_track_mode})")
        self.q_est = torch.tensor([q0], dtype=torch.float32, device=self.device,
                                  requires_grad=(self.q_track_mode == 'sa'))
        if self.q_track_mode == 'sa':                          # 仅 SA 模式需要优化器/调度器
            self.q_optimizer = Adam([self.q_est], 1.0, eps=1e-5)
            self.q_scheduler = LambdaLR(
                self.q_optimizer,
                lr_lambda=lambda k: lr_lambda(k, *self._q_args))

        for it in range(self.num_iterations):
            # ===== 1. 并行采 B 条轨迹 (冻结策略) =====
            roll = self._rollout_core()
            n, B = self.n, self.num_envs
            states = roll['S'].reshape(n * B, -1)              # [n·B, sd]
            actions = roll['A'].reshape(n * B, -1)             # [n·B, ad]
            U = roll['disc_returns'].detach()                  # [B] 每条轨迹 Z(τ)

            # ===== 2. 内层策略更新 (复用本批 updates_per_iteration 次, 与 QCPOGPU 同结构) =====
            actor_info = {}
            for _ in range(self.updates_per_iteration):
                actor_info = self._update_actor(states, actions, U)

            # ===== 3. Q 更新 (每迭代一次, 两时间尺度由 lr 调度差异保证) =====
            self._update_q(U)

            # ===== 4. 统一日志 + LR 调度 =====
            extra = {'training/actor_lr': float(self.scheduler.get_last_lr()[0])}
            extra.update(actor_info)
            self._log_core(it, U.cpu().numpy(), q_est_value=self.q_est.item(), extra=extra)
            self.scheduler.step()
            if self.q_scheduler is not None:                   # 仅 SA 模式有 q 调度器
                self.q_scheduler.step()

    # ============================================================ 策略更新 (与 qpo.update 对齐) ============================================================
    def _update_actor(self, states, actions, U):
        """
        一次 QPO 策略梯度更新:
            ind(τ) = 𝟙{Z(τ) ≤ Q}  (轨迹级, [B] → 广播 [n·B])
            loss = +mean(logπ(a_t|s_t) · ind)   —— 下降 loss = 压低 P(Z≤Q) = 抬高 Q_α
        与 qpo.py 一致: 不归一化、无 baseline (示性函数本身有界 [0,1])。
        """
        n, B = self.n, self.num_envs
        ind = indicator(self.q_est.detach(), U)                # [B] 𝟙{Z≤Q}
        ind_full = ind.unsqueeze(0).expand(n, B).reshape(n * B)  # 广播到所有 timestep

        log_probs = self._compute_log_probs(states, actions)   # [n·B], 对 θ 可导
        loss = torch.mean(log_probs * ind_full)                # +号: 下降即分位数上升 (见上)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.actor_grad_clip and self.actor_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip)
        self.optimizer.step()
        return {'actor/loss': float(loss.item()),
                'actor/ind_mean': float(ind.mean().item())}    # P̂(Z≤Q) (应追踪 α)

    # ============================================================ 分位数更新 (与 qpo.update 对齐) ============================================================
    def _update_q(self, U):
        """
        Q 追踪更新 (两种模式, 见 __init__ 注释):
          'ema': Q ← (1-ρ)Q + ρ·percentile(本批 Z, α) —— 低噪声批估计, 几迭代内贴住真分位数
          'sa' : grad = -(α - mean 𝟙{Z≤Q}) → Adam 一步 ⇒ Q ← Q + lr_q·(α - P̂(Z≤Q)) (原版)
        """
        if self.q_track_mode == 'ema':
            batch_q = float(np.percentile(U.cpu().numpy(), self.q_alpha * 100))  # 批经验分位数
            with torch.no_grad():                              # EMA 追踪 (无梯度)
                self.q_est.mul_(1.0 - self.q_track_rho).add_(self.q_track_rho * batch_q)
        else:
            self.q_optimizer.zero_grad(set_to_none=True)
            self.q_est.grad = -torch.mean(
                self.q_alpha - indicator(self.q_est, U), dim=0, keepdim=True)
            self.q_optimizer.step()

    # ============================================================ 总结接口 ============================================================
    def get_training_summary(self):
        """暴露最终指标 (run_experiment.py 可选调用)。"""
        return {
            'q_est_final': float(self.q_est.detach().item()) if self.q_est is not None else None,
            'empirical_violation_prob': getattr(self, 'last_empirical_prob', None),
            'num_envs': self.num_envs,
            'num_iterations': self.num_iterations,
        }
