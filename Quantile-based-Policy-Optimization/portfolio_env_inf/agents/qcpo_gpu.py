# -*- coding: utf-8 -*-
"""
QCPOGPU —— QCPO 的全 GPU、B 路并行向量化版 (portfolio_env_inf)。

算法逐式对齐 risk_sensitive_env_inf/agents/qcpo_gpu.py (那是已验证的 ∞-horizon 参考实现),
本文件只把环境从 RiskSensitiveVecTorch 换成 PortfolioVecTorch (经 VecAgentBase),
更新数学不动:
    优化问题:  max_θ E[Z(τ)]  s.t.  P_θ(Z(τ) ≤ q) ≤ α      (Z=Σγ^t r_t)
    策略梯度:  每 timestep 梯度权重 = (Z-μ)/σ - λ·𝟙{Z ≤ q}   (轨迹级标量, 广播到全轨迹)
               · 均值项用 RunningMeanStd (EMA) 归一化;  约束项原始尺度
               · 无 γ^t/β^t 折扣、无 baseline (与 qcpo.py 逐式一致)
    Dual:      λ ← [λ + ε_k·(P̂(Z≤q) - α)]_+   (本批经验违反率驱动)
"""
import numpy as np
import torch
from torch.optim import Adam                                  # base lr=1 + LambdaLR
from torch.optim.lr_scheduler import LambdaLR                 # lr = a/(b+k)^c

from utils import RunningMeanStd                              # EMA 回报归一化器 (QCPO 配方)
from .vec_base import VecAgentBase                            # 共享: env/策略/rollout/日志
from .common import lr_lambda                                 # lr 衰减 (共享件)


class QCPOGPU(VecAgentBase):
    """QCPO 全 GPU 向量化版 (约束: 经验违反率驱动 dual; 算法与 risk_sensitive_inf 版一致)。"""

    # ============================================================ 初始化 ============================================================
    def __init__(self, args, env):
        super().__init__(args, env)                            # 基类: env/actor/wandb/维度

        # -------------------- 算法专属超参 (与 risk_sensitive_inf 模板同名同义) --------------------
        self.outer_interval = max(1, int(getattr(args, 'outer_interval', 1)))  # λ 更新间隔 (迭代)
        self.updates_per_iteration = max(1, int(getattr(args, 'updates_per_episode', 10)))  # 内层复用次数
        self.warmup_rms_iters = max(1, int(getattr(args, 'warmup_rms_iters', 2)))  # rms 预热迭代
        self.actor_grad_clip = float(getattr(args, 'actor_grad_clip', 0.0))    # 0=不裁 (QCPO 默认)

        # return_rms: EMA 回报归一化 (decay=0.01 ≈ 最近 100 条轨迹, QCPO 配方)
        self.return_rms = RunningMeanStd(decay=float(getattr(args, 'norm_ema_decay', 0.01)))

        # -------------------- 优化器: θ 两时间尺度 + λ --------------------
        # θ 默认 SGD (与 QPOGPU 同一关键修复, 见 DESIGN.md "稳定性修复"):
        # Adam 在 20 维纯噪声 est 特征坐标上做恒速随机游走 → W 爆炸 → 回报方差爆炸 (探针实证);
        # SGD 步长 ∝ 真实梯度幅度, 且更忠实原论文的 Robbins-Monro SA 形式。
        self.theta_optimizer_name = str(getattr(args, 'theta_optimizer', 'sgd')).lower()
        if self.theta_optimizer_name == 'adam':
            self.optimizer = Adam(self.actor.parameters(), 1.0, eps=1e-5)
        else:
            self.optimizer = torch.optim.SGD(self.actor.parameters(), 1.0)
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda k: lr_lambda(k, args.theta_a, args.theta_b, args.theta_c))

        # λ 上界 (新增保护): P>α 的长暂态里 λ 以 ~lr_λ/迭代线性攀升, 无界时会进入
        # "纯指示函数 unlearning"退化域 (与 QPO 同病, 探针实证); 上界与 DQCAC 的 lambda_max 同义。
        self.lambda_max = float(getattr(args, 'lambda_max', 50.0))

        self.lambda_dual = torch.tensor([0.0], dtype=torch.float32,
                                        device=self.device, requires_grad=True)  # λ≥0
        self.lambda_optimizer = Adam([self.lambda_dual], 1.0, eps=1e-5)
        self.lambda_scheduler = LambdaLR(
            self.lambda_optimizer,
            lr_lambda=lambda k: lr_lambda(k, args.lambda_a, args.lambda_b, args.lambda_c))

        # -------------------- 运行时统计 --------------------
        self.last_empirical_prob = 0.0
        self.last_dual_prob = 0.0

    # ============================================================ 主训练循环 ============================================================
    def train(self):
        """rms 预热 → 每迭代: 采样 → 刷 rms → 内层策略更新 → (外层) λ 更新 → 日志 + 调度。"""
        print(f"QCPOGPU: alpha={self.q_alpha}, q={self.quantile_threshold}, "
              f"B={self.num_envs}, iters={self.num_iterations}, "
              f"updates/iter={self.updates_per_iteration}, device={self.device}")

        # ===== return_rms 预热 (只采集刷新归一化器, 不更新策略; 对齐 qcpo.warm_up) =====
        for _ in range(self.warmup_rms_iters):
            self._update_rms(self._rollout_core()['disc_returns'])
        print(f"QCPOGPU warm up || rms_mean:{self.return_rms.mean:.3f} rms_std:{self.return_rms.std:.3f}")

        for it in range(self.num_iterations):
            # ===== 1. 并行采 B 条轨迹 (冻结策略) =====
            roll = self._rollout_core()
            n, B = self.n, self.num_envs
            states = roll['S'].reshape(n * B, -1)              # [n·B, sd]
            actions = roll['A'].reshape(n * B, -1)             # [n·B, ad]
            U = roll['disc_returns']                           # [B]

            # ===== 2. 刷新 return_rms (逐条, 与原版每 episode 一条语义一致) =====
            self._update_rms(U)

            # ===== 3. 内层策略更新 (复用本批) =====
            actor_info = {}
            for _ in range(self.updates_per_iteration):
                actor_info = self._update_actor(states, actions, U.detach())

            # ===== 4. 外层 λ 更新 =====
            if it % self.outer_interval == 0:
                self._update_dual(U.detach())

            # ===== 5. 统一日志 + LR 调度 =====
            z = U.detach().cpu().numpy()
            extra = {
                'lambda/value': float(self.lambda_dual.detach().item()),
                'lambda/lr': float(self.lambda_scheduler.get_last_lr()[0]),
                'constraint/dual_prob': self.last_dual_prob,
                'normalize/return_mean': float(self.return_rms.mean),
                'normalize/return_std': float(self.return_rms.std),
                'training/actor_lr': float(self.scheduler.get_last_lr()[0]),
            }
            extra.update(actor_info)
            # q_est 键: QCPO 无学习的 Q → 报本批经验 α-分位数 (与 risk_sensitive_inf 模板一致)
            self._log_core(it, z, q_est_value=float(np.percentile(z, self.q_alpha * 100)),
                           extra=extra)
            self.scheduler.step()
            self.lambda_scheduler.step()

    # ============================================================ 策略更新 (与模板 update_actor 逐式一致) ============================================================
    def _update_actor(self, states, actions, U):
        """
        weight(τ) = (Z-μ)/σ - λ·𝟙{Z≤q}  (每条轨迹标量, 广播到全部 n 个 timestep)
        loss = -E[logπ·weight]  (optimizer 下降 = 梯度上升)
        """
        n, B = self.n, self.num_envs
        ind = (U <= self.quantile_threshold).float()           # [B] 𝟙{Z≤q} (原始尺度)
        U_norm = (U - self.return_rms.mean) / self.return_rms.std  # [B] EMA 归一化均值项
        weight = U_norm - self.lambda_dual.detach() * ind      # [B] 复合权重
        weight_full = weight.unsqueeze(0).expand(n, B).reshape(n * B)  # [n·B]

        log_probs = self._compute_log_probs(states, actions)   # [n·B]
        actor_loss = -(log_probs * weight_full.detach()).mean()

        self.optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.actor_grad_clip and self.actor_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip)
        self.optimizer.step()
        return {'actor/loss': float(actor_loss.item()),
                'actor/weight_mean': float(weight.mean().item()),
                'actor/weight_std': float(weight.std(unbiased=False).item())}

    # ============================================================ Dual 更新 (与模板一致) ============================================================
    def _update_dual(self, U):
        """λ ← [λ + ε_k·(P̂(Z≤q) - α)]_+, 本批经验违反率驱动。"""
        p = float((U <= self.quantile_threshold).float().mean().item())
        self.last_dual_prob = p
        gap = torch.tensor([p - self.q_alpha], dtype=torch.float32, device=self.device)
        self.lambda_optimizer.zero_grad(set_to_none=True)
        (-self.lambda_dual * gap).backward()                   # ∂/∂λ = -(P̂-α) → 下降即上升 λ
        self.lambda_optimizer.step()
        with torch.no_grad():
            self.lambda_dual.clamp_(min=0.0, max=self.lambda_max)  # 投影 [0, λ_max]

    # ============================================================ 辅助 ============================================================
    def _update_rms(self, disc_returns):
        """用本批 B 条轨迹回报逐条更新 EMA return_rms (与原版语义一致)。"""
        for z in disc_returns.detach().cpu().tolist():
            self.return_rms.update(z)

    def get_training_summary(self):
        """暴露最终约束指标 (run_experiment.py 可选调用)。"""
        return {
            'lambda_final': float(self.lambda_dual.detach().item()),
            'empirical_violation_prob': self.last_empirical_prob,
            'dual_prob': self.last_dual_prob,
            'num_envs': self.num_envs,
            'num_iterations': self.num_iterations,
        }
