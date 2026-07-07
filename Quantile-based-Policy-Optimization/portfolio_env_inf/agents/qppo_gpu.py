# -*- coding: utf-8 -*-
"""
QPPOGPU —— QPPO (Quantile-based PPO) 的全 GPU、B 路并行向量化版 (portfolio_env_inf)。

算法来源 portfolio_management/agents/qppo.py, 目标与 QPO 相同 (max Q_α(Z), 无约束),
但用 PPO 机制 (重要性比率 + 裁剪 + critic baseline + 多 epoch 复用) 替代朴素 REINFORCE。

【与原版 qppo.py 的三处适配】(∞-horizon 平稳设定 + 向量化, 已在设计文档记录):
  1. 分位数估计: 原版维护按时间窗 [T0,T) 的分位数【向量】(有限步、各截止时刻的部分回报
     分位数) → 本版只有单一标量 q_est (完整截断回报 Z 的 α-分位数)。平稳 continuing 设定下
     部分回报窗口失去意义, 统一口径也要求四算法优化同一个 Q_α(Z)。
  2. 重要性比率: 原版用"轨迹前 l 步 logπ 比率之和"的轨迹级乘积比率 → 本版用【逐 timestep
     比率】r_t=exp(logπ_new-logπ_old) + 轨迹级优势广播 (标准 PPO 代理目标; n=100 的轨迹级
     乘积比率数值爆炸, 且与统一的"轨迹权重广播"结构一致)。
  3. q_est 更新: 原版用 ratio 修正的 𝟙 (复用旧批) → 本版每迭代在【新鲜 on-policy 批】上
     更新一次 (无需 IS 修正, 无偏), 两时间尺度由独立 lr 调度保证。
核心保留: 优势 = -𝟙{Z≤Q} - V(s) (示性函数作"奖励", critic 回归 -𝟙), PPO 裁剪,
          actor/critic 联合优化器, 梯度裁剪 0.5。
"""
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR                 # q_est 两时间尺度调度

from utils import Critic                                      # V(s) baseline 网络
from .vec_base import VecAgentBase                            # 共享: env/策略/rollout/日志
from .common import lr_lambda, indicator                      # 共享件


class QPPOGPU(VecAgentBase):
    """QPPO 全 GPU 向量化版: PPO 机制的分位数最大化 (QPO 的 PPO 化, 无约束基线)。"""

    # ============================================================ 初始化 ============================================================
    def __init__(self, args, env):
        super().__init__(args, env)                            # 基类: env/actor/wandb/维度

        # -------------------- PPO 超参 (与原版 qppo.py 同名同默认) --------------------
        self.clip_eps = float(getattr(args, 'clip_eps', 0.2))  # PPO 裁剪半径 ε
        self.vf_coef = float(getattr(args, 'vf_coef', 0.5))    # critic loss 权重
        self.ent_coef = float(getattr(args, 'ent_coef', 0.0))  # 熵正则 (σ固定→熵常数, 默认0)
        self.ppo_epochs = max(1, int(getattr(args, 'updates_per_episode', 10)))  # epoch 数 (统一=10)
        self.grad_clip = float(getattr(args, 'grad_clip', 0.5))  # 梯度裁剪 (原版 0.5)
        self.warmup_q_iters = max(1, int(getattr(args, 'warmup_q_iters', 2)))  # Q 预热迭代

        # -------------------- critic (V(s) baseline) + 联合优化器 --------------------
        hidden = getattr(args, 'critic_hidden', [64, 64])
        if isinstance(hidden, str):
            hidden = [int(x) for x in hidden.split(',') if x.strip()]
        self.critic_v = Critic(self.state_dim, list(hidden)).to(self.device)  # V(s) (回归 -𝟙)
        # actor+critic 联合 Adam, 常数 lr (PPO 惯例; 原版 StepLR 衰减, 本版预算固定不需要)
        self.ppo_lr = float(getattr(args, 'ppo_lr', 3e-4))
        self.optimizer = Adam(list(self.actor.parameters()) + list(self.critic_v.parameters()),
                              self.ppo_lr, eps=1e-5)
        self.MSELoss = torch.nn.MSELoss()

        # -------------------- q_est 追踪 (预热后初始化; 模式说明见 QPOGPU.__init__) --------------------
        # 'ema' (默认): q_est ← (1-ρ)q_est + ρ·批经验分位数 (恢复"q 快"时间尺度, 关键修复)
        # 'sa':         原版随机近似 (Adam), 作 A/B 对照
        self.q_track_mode = str(getattr(args, 'q_track_mode', 'ema')).lower()
        self.q_track_rho = float(getattr(args, 'q_track_rho', 0.3))
        self.q_est = None
        self._q_args = (args.q_a, args.q_b, args.q_c)
        self.q_optimizer = None
        self.q_scheduler = None

    # ============================================================ 主训练循环 ============================================================
    def train(self):
        """预热 Q → 每迭代: 采样(带 logπ_old) → 更新 Q → PPO epochs → 统一日志 + 调度。"""
        print(f"QPPOGPU: alpha={self.q_alpha}, B={self.num_envs}, iters={self.num_iterations}, "
              f"epochs/iter={self.ppo_epochs}, lr={self.ppo_lr}, device={self.device}")

        # ===== Q 预热: 经验 α-分位数初始化 (对齐 qppo.warm_up 语义, 单标量版) =====
        warm_z = []
        for _ in range(self.warmup_q_iters):
            warm_z.append(self._rollout_core()['disc_returns'])
        q0 = float(np.percentile(torch.cat(warm_z).cpu().numpy(), self.q_alpha * 100))
        print(f"QPPOGPU warm up || init q_est:{q0:.3f} (q_track_mode={self.q_track_mode})")
        self.q_est = torch.tensor([q0], dtype=torch.float32, device=self.device,
                                  requires_grad=(self.q_track_mode == 'sa'))
        if self.q_track_mode == 'sa':                          # 仅 SA 模式需要优化器/调度器
            self.q_optimizer = Adam([self.q_est], 1.0, eps=1e-5)
            self.q_scheduler = LambdaLR(
                self.q_optimizer, lr_lambda=lambda k: lr_lambda(k, *self._q_args))

        for it in range(self.num_iterations):
            # ===== 1. 并行采样 (额外保留采集时刻 logπ_old → PPO 比率分母) =====
            roll = self._rollout_core(keep_logp=True)
            n, B = self.n, self.num_envs
            states = roll['S'].reshape(n * B, -1)              # [n·B, sd]
            actions = roll['A'].reshape(n * B, -1)             # [n·B, ad]
            logp_old = roll['logp'].reshape(n * B).detach()    # [n·B] 采集策略 logπ_old
            U = roll['disc_returns'].detach()                  # [B] Z(τ)

            # ===== 2. Q 更新 (新鲜 on-policy 批; 模式说明见 QPOGPU._update_q) =====
            if self.q_track_mode == 'ema':
                batch_q = float(np.percentile(U.cpu().numpy(), self.q_alpha * 100))
                with torch.no_grad():                          # EMA 追踪批经验分位数
                    self.q_est.mul_(1.0 - self.q_track_rho).add_(self.q_track_rho * batch_q)
            else:
                self.q_optimizer.zero_grad(set_to_none=True)
                self.q_est.grad = -torch.mean(
                    self.q_alpha - indicator(self.q_est, U), dim=0, keepdim=True)
                self.q_optimizer.step()                        # Q ← Q + lr_q(α - P̂(Z≤Q))

            # ===== 3. 轨迹级优势 (计算一次, epochs 内冻结; 对齐原版 old_state_values 语义) =====
            ind = indicator(self.q_est.detach(), U)            # [B] 𝟙{Z≤Q}
            ind_full = ind.unsqueeze(0).expand(n, B).reshape(n * B)  # [n·B] 广播
            with torch.no_grad():
                v_old = self.critic_v(states).squeeze(-1)      # [n·B] 采集时刻 V(s)
            advantages = (-ind_full - v_old).detach()          # [n·B] A = -𝟙 - V(s) (原版公式)

            # ===== 4. PPO epochs (全批, 复用 ppo_epochs 次) =====
            info = {}
            for _ in range(self.ppo_epochs):
                info = self._ppo_step(states, actions, logp_old, ind_full, advantages)

            # ===== 5. 统一日志 + 调度 =====
            extra = {'training/actor_lr': self.ppo_lr}
            extra.update(info)
            self._log_core(it, U.cpu().numpy(), q_est_value=self.q_est.item(), extra=extra)
            if self.q_scheduler is not None:                   # 仅 SA 模式有 q 调度器
                self.q_scheduler.step()

    # ============================================================ PPO 单步 ============================================================
    def _ppo_step(self, states, actions, logp_old, ind_full, advantages):
        """
        一次裁剪代理目标更新 (与原版 update 内核对齐, 比率改逐 timestep):
            r_t = exp(logπ_new - logπ_old)  (clamp ±10 防溢出, 原版同)
            actor_loss  = -E[min(r·A, clip(r,1±ε)·A)]
            critic_loss = MSE(V(s), -𝟙)            (critic 学的是 -P(Z≤Q|s) 的回归)
            loss = actor + vf_coef·critic - ent_coef·H
        """
        logp = self._compute_log_probs(states, actions)        # [n·B] 当前策略
        ratios = torch.exp(torch.clamp(logp - logp_old, -10., 10.))  # [n·B] 比率

        surr1 = ratios * advantages                            # 未裁剪项
        surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()           # PPO 悲观下界

        v = self.critic_v(states).squeeze(-1)                  # [n·B] V(s) (建图)
        critic_loss = self.MSELoss(v, -ind_full)               # 回归 -𝟙

        entropy = self._entropy(states).mean()                 # 常数 (σ固定), 接口完整
        loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)   # 原版 0.5
        torch.nn.utils.clip_grad_norm_(self.critic_v.parameters(), self.grad_clip)
        self.optimizer.step()
        return {'actor/loss': float(actor_loss.item()),
                'critic/v_loss': float(critic_loss.item()),
                'actor/ratio_mean': float(ratios.mean().item()),
                'actor/ind_mean': float(ind_full.mean().item())}  # P̂(Z≤Q) (应追踪 α)

    # ============================================================ 总结接口 ============================================================
    def get_training_summary(self):
        """暴露最终指标 (run_experiment.py 可选调用)。"""
        return {
            'q_est_final': float(self.q_est.detach().item()) if self.q_est is not None else None,
            'empirical_violation_prob': getattr(self, 'last_empirical_prob', None),
            'num_envs': self.num_envs,
            'num_iterations': self.num_iterations,
        }
