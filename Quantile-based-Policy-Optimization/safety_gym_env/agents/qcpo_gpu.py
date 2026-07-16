# -*- coding: utf-8 -*-
"""
QCPOGPU (safety_gym_env · CMDP 版) —— 用户 QCPO 迁移到 safety-gym outage 约束环境。

算法逐式对齐 portfolio_env_inf/agents/qcpo_gpu.py (已验证的 MC 轨迹级 QCPO), 唯一改动 =
【目标与约束换到两条不同回报流上】(CMDP), 且约束方向翻为【上尾】:
    优化问题:  max_θ E[R]  s.t.  P_θ(C ≥ d) ≤ ω        (R=Σγ^t r_t 奖励回报; C=Σγc^t c_t cost 回报)
    策略梯度:  每 timestep 权重 = (R-μ_R)/σ_R  -  λ·𝟙{C ≥ d}    (轨迹级标量, 广播到全轨迹)
               · 目标项用 reward 回报 R 的 EMA 归一化 (与原版对 Z 归一化同构)
               · 约束项用 cost 回报 C 的【上尾】示性 𝟙{C≥d} (原版是下尾 𝟙{Z≤q})
               · 无 γ^t/β^t 折扣、无 baseline (与 qcpo.py 逐式一致)
    Dual:      λ ← [λ + ε_k·(P̂(C≥d) - ω)]_+           (本批经验 outage 违反率驱动)

与原版仅有的差异: 一条 rollout 现在算【两个】回报 (R 目标 / C 约束); indicator 用 indicator_ge。
"""
import numpy as np
import torch
from torch.optim import Adam                                  # base lr=1 + LambdaLR
from torch.optim.lr_scheduler import LambdaLR                 # lr = a/(b+k)^c

from utils import RunningMeanStd                              # EMA 回报归一化器 (QCPO 配方)
from .vec_base import VecAgentBase                            # 共享: env/策略/rollout(含cost)/日志
from .common import lr_lambda                                 # lr 衰减 (共享件)


class QCPOGPU(VecAgentBase):
    """QCPO CMDP 版 (MC 轨迹级; 经验 outage 驱动 dual; 算法与 portfolio 版逐式一致)。"""

    # ============================================================ 初始化 ============================================================
    def __init__(self, args, env):
        super().__init__(args, env)                           # 基类: env/actor/wandb/维度

        # -------------------- 算法专属超参 (与 portfolio 模板同名同义) --------------------
        self.outer_interval = max(1, int(getattr(args, 'outer_interval', 1)))       # λ 更新间隔
        self.updates_per_iteration = max(1, int(getattr(args, 'updates_per_episode', 10)))
        self.warmup_rms_iters = max(1, int(getattr(args, 'warmup_rms_iters', 2)))   # rms 预热
        self.actor_grad_clip = float(getattr(args, 'actor_grad_clip', 1.0))         # 范数裁剪 (MLP 保护)

        # return_rms: EMA 奖励回报归一化 (decay=0.01 ≈ 最近 100 条轨迹, QCPO 配方)
        self.return_rms = RunningMeanStd(decay=float(getattr(args, 'norm_ema_decay', 0.01)))

        # -------------------- 优化器: θ 两时间尺度 + λ --------------------
        # θ 优化器: 默认 SGD (对齐 QCPOGPU 模板 + Robbins-Monro SA); safety-gym 观测信息量足,
        # 也可切 adam (theta_optimizer='adam') 加速收敛。LR = a/(b+k)^c 按迭代步进。
        self.theta_optimizer_name = str(getattr(args, 'theta_optimizer', 'sgd')).lower()
        if self.theta_optimizer_name == 'adam':
            self.optimizer = Adam(self.actor.parameters(), 1.0, eps=1e-5)
        else:
            self.optimizer = torch.optim.SGD(self.actor.parameters(), 1.0)
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda k: lr_lambda(k, args.theta_a, args.theta_b, args.theta_c))

        # λ 上界 (防长暂态 λ runaway → 纯指示函数退化域)
        self.lambda_max = float(getattr(args, 'lambda_max', 50.0))
        self.lambda_dual = torch.tensor([0.0], dtype=torch.float32,
                                        device=self.device, requires_grad=True)     # λ≥0
        self.lambda_optimizer = Adam([self.lambda_dual], 1.0, eps=1e-5)
        self.lambda_scheduler = LambdaLR(
            self.lambda_optimizer,
            lr_lambda=lambda k: lr_lambda(k, args.lambda_a, args.lambda_b, args.lambda_c))

        self.last_dual_prob = 0.0                             # dual 用的经验 outage

    # ============================================================ 主训练循环 ============================================================
    def train(self):
        """rms 预热 → 每迭代: 采样(含cost) → 刷 rms → 内层策略更新 → (外层)λ 更新 → 日志+调度。"""
        print(f"QCPOGPU[CMDP]: env={self.env_name}, omega={self.q_alpha}, d(cost_limit)={self.cost_limit}, "
              f"B={self.num_envs}, T={self.n}, iters={self.num_iterations}, "
              f"updates/iter={self.updates_per_iteration}, opt={self.theta_optimizer_name}, device={self.device}")

        # ===== return_rms 预热 (只采集刷 reward 回报归一化器, 不更新策略) =====
        for _ in range(self.warmup_rms_iters):
            self._update_rms(self._rollout_core()['disc_return'])
        print(f"QCPOGPU warm up || rms_mean:{self.return_rms.mean:.3f} rms_std:{self.return_rms.std:.3f}")

        for it in range(self.num_iterations):
            # ===== 1. 并行采 B 条轨迹 (冻结策略, 含 cost 流) =====
            roll = self._rollout_core()
            n, B = self.n, self.num_envs
            states = roll['S'].reshape(n * B, -1)             # [n·B, sd]
            actions = roll['A'].reshape(n * B, -1)            # [n·B, ad]
            R = roll['disc_return']                           # [B] 奖励回报 (目标)
            C = roll['disc_cost']                             # [B] cost 回报 (约束变量)

            # ===== 2. 刷新 return_rms (奖励回报, 逐条) =====
            self._update_rms(R)

            # ===== 3. 内层策略更新 (复用本批) =====
            actor_info = {}
            for _ in range(self.updates_per_iteration):
                actor_info = self._update_actor(states, actions, R.detach(), C.detach())

            # ===== 4. 外层 λ 更新 (经验 outage 驱动) =====
            if it % self.outer_interval == 0:
                self._update_dual(C.detach())

            # ===== 5. 统一日志 + LR 调度 =====
            R_np = R.detach().cpu().numpy()
            Zc_np = C.detach().cpu().numpy()
            Cu_np = roll['undisc_cost'].detach().cpu().numpy()
            extra = {
                'lambda/value': float(self.lambda_dual.detach().item()),
                'lambda/lr': float(self.lambda_scheduler.get_last_lr()[0]),
                'constraint/dual_prob': self.last_dual_prob,
                'normalize/return_mean': float(self.return_rms.mean),
                'normalize/return_std': float(self.return_rms.std),
                'training/actor_lr': float(self.scheduler.get_last_lr()[0]),
            }
            extra.update(actor_info)
            self._log_core(it, R_np, Zc_np, Cu_np, extra=extra)
            self.scheduler.step()
            self.lambda_scheduler.step()

    # ============================================================ 策略更新 (与模板逐式一致, 约束换 cost 上尾) ============================================================
    def _update_actor(self, states, actions, R, C):
        """
        weight(τ) = (R-μ_R)/σ_R  -  λ·𝟙{C≥d}   (每条轨迹标量, 广播到全部 n 个 timestep)
        loss = -E[logπ·weight]
        """
        n, B = self.n, self.num_envs
        ind = (C >= self.cost_limit).float()                  # [B] 𝟙{C≥d} 上尾 (outage)
        R_norm = (R - self.return_rms.mean) / self.return_rms.std   # [B] EMA 归一化奖励回报
        weight = R_norm - self.lambda_dual.detach() * ind     # [B] 复合权重
        weight_full = weight.unsqueeze(0).expand(n, B).reshape(n * B)   # [n·B]

        log_probs = self._compute_log_probs(states, actions)  # [n·B]
        actor_loss = -(log_probs * weight_full.detach()).mean()

        self.optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.actor_grad_clip and self.actor_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip)
        self.optimizer.step()
        return {'actor/loss': float(actor_loss.item()),
                'actor/weight_mean': float(weight.mean().item()),
                'actor/weight_std': float(weight.std(unbiased=False).item())}

    # ============================================================ Dual 更新 (经验 outage 驱动) ============================================================
    def _update_dual(self, C):
        """λ ← [λ + ε_k·(P̂(C≥d) - ω)]_+, 本批经验 outage 违反率驱动。"""
        p = float((C >= self.cost_limit).float().mean().item())    # P̂(C≥d)
        self.last_dual_prob = p
        gap = torch.tensor([p - self.q_alpha], dtype=torch.float32, device=self.device)  # P̂-ω
        self.lambda_optimizer.zero_grad(set_to_none=True)
        (-self.lambda_dual * gap).backward()                  # ∂/∂λ=-(P̂-ω) → 下降即上升 λ
        self.lambda_optimizer.step()
        with torch.no_grad():
            self.lambda_dual.clamp_(min=0.0, max=self.lambda_max)

    # ============================================================ 辅助 ============================================================
    def _update_rms(self, disc_returns):
        """用本批 B 条轨迹【奖励回报】逐条更新 EMA return_rms。"""
        for z in disc_returns.detach().cpu().tolist():
            self.return_rms.update(z)

    def get_training_summary(self):
        """暴露最终约束指标。"""
        return {
            'lambda_final': float(self.lambda_dual.detach().item()),
            'empirical_prob': self.last_empirical_prob,
            'empirical_outage_prob': self.last_outage_prob,   # 兼容 (=empirical_prob)
            'dual_prob': self.last_dual_prob,
            'num_envs': self.num_envs,
            'num_iterations': self.num_iterations,
        }
