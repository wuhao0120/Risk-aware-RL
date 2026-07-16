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

默认 qcpo_reward_mode=mc 时，与原版仅有的任务适配是一条 rollout 同时计算 reward/cost
两种回报并使用上尾 indicator；可选 gae 模式是明确标注的 reward-credit hybrid 消融。
"""
import numpy as np
import torch
from torch.optim import Adam                                  # base lr=1 + LambdaLR
from torch.optim.lr_scheduler import LambdaLR                 # lr = a/(b+k)^c

from utils import RunningMeanStd                              # EMA 回报归一化器 (QCPO 配方)
from .vec_base import VecAgentBase                            # 共享: env/策略/rollout(含cost)/日志
from .common import ScalarValueCritic, lr_lambda              # 可选 V_r(s)+GAE / lr 衰减


class QCPOGPU(VecAgentBase):
    """QCPO CMDP：默认 MC 轨迹级；可选 reward V+GAE hybrid；constraint 始终轨迹级。"""

    # ============================================================ 初始化 ============================================================
    def __init__(self, args, env):
        super().__init__(args, env)                           # 基类: env/actor/wandb/维度

        # -------------------- 算法专属超参 (与 portfolio 模板同名同义) --------------------
        self.outer_interval = max(1, int(getattr(args, 'outer_interval', 1)))       # λ 更新间隔
        self.updates_per_iteration = max(1, int(getattr(args, 'updates_per_episode', 10)))
        self.warmup_rms_iters = max(1, int(getattr(args, 'warmup_rms_iters', 2)))   # rms 预热
        self.actor_grad_clip = float(getattr(args, 'actor_grad_clip', 1.0))         # 范数裁剪 (MLP 保护)

        # 同一 rollout 复用多次时，必须用采样时冻结的行为策略概率作 IS 分母。
        # on_policy 只允许一步 actor update；ppo 才允许多 epoch，并始终保留同一 old_log_prob。
        self.actor_update_mode = str(getattr(args, 'qcpo_actor_update_mode', 'on_policy')).lower()
        if self.actor_update_mode not in {'on_policy', 'ppo'}:
            raise ValueError("qcpo_actor_update_mode must be 'on_policy' or 'ppo'")
        if self.actor_update_mode == 'on_policy' and self.updates_per_iteration != 1:
            raise ValueError(
                "QCPO reuses one rollout without importance correction: set "
                "updates_per_episode=1, or qcpo_actor_update_mode=ppo")
        self.ppo_ratio_clip = float(getattr(args, 'ppo_ratio_clip', 0.1))
        self.log_std_min = float(getattr(args, 'log_std_min', -5.0))
        self.log_std_max = float(getattr(args, 'log_std_max', 2.0))

        # return_rms: EMA 奖励回报归一化 (decay=0.01 ≈ 最近 100 条轨迹, QCPO 配方)
        self.return_rms = RunningMeanStd(decay=float(getattr(args, 'norm_ema_decay', 0.01)))

        # reward credit assignment 做成显式消融：mc 保持原始轨迹级 QCPO；gae 是
        # scalar V_r(s,t)+GAE hybrid，constraint 仍保持轨迹级 indicator，不改变其语义。
        self.reward_mode = str(getattr(args, 'qcpo_reward_mode', 'mc')).lower()
        if self.reward_mode not in {'mc', 'gae'}:
            raise ValueError("qcpo_reward_mode must be 'mc' or 'gae'")
        self.gae_lambda = float(getattr(args, 'gae_lambda', 0.97))
        self.reward_advantage_norm = bool(getattr(args, 'reward_advantage_norm', False))
        self.value_step_feature = bool(getattr(args, 'reward_value_step_feature', True))
        self.reward_value_lr = float(getattr(args, 'reward_value_lr', 3e-4))
        self.reward_value_grad_clip = float(getattr(args, 'reward_value_grad_clip', 10.0))

        # GAE 模式维护独立 V_r(s,t)。有限期界 DynamicButton 的价值依赖剩余时间，
        # 因而默认追加 t/T；网络宽度与 actor MLP 相同，减少容量这个混杂变量。
        self.reward_value = None
        self.reward_value_optimizer = None
        if self.reward_mode == 'gae':
            value_hidden = getattr(args, 'actor_hidden', [256, 256])
            if isinstance(value_hidden, str):
                value_hidden = [int(x) for x in value_hidden.split(',') if x.strip()]
            value_dim = self.state_dim + (1 if self.value_step_feature else 0)
            self.reward_value = ScalarValueCritic(value_dim, hidden=value_hidden).to(self.device)
            self.reward_value_optimizer = Adam(
                self.reward_value.parameters(), self.reward_value_lr, eps=1e-5)

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
              f"updates/iter={self.updates_per_iteration}, actor_mode={self.actor_update_mode}, "
              f"reward_mode={self.reward_mode}, obs_norm={self.normalize_observation}, "
              f"opt={self.theta_optimizer_name}, device={self.device}")

        # ===== return/observation RMS 预热 (只采样和刷统计，不更新策略) =====
        for _ in range(self.warmup_rms_iters):
            warmup_roll = self._rollout_core()
            self._update_rms(warmup_roll['disc_return'])
            self._update_obs_rms(warmup_roll['S'])
        print(f"QCPOGPU warm up || rms_mean:{self.return_rms.mean:.3f} rms_std:{self.return_rms.std:.3f}")

        for it in range(self.num_iterations):
            # ===== 1. 并行采 B 条轨迹 (冻结策略, 含 cost 流) =====
            keep_logp = self.actor_update_mode == 'ppo'
            roll = self._rollout_core(keep_logp=keep_logp)
            n, B = self.n, self.num_envs
            states = roll['S'].reshape(n * B, -1)             # [n·B, sd]
            actions = roll['A'].reshape(n * B, -1)            # [n·B, ad]
            R = roll['disc_return']                           # [B] 奖励回报 (目标)
            C = roll['disc_cost']                             # [B] cost 回报 (约束变量)
            old_log_probs = roll.get('logp', None)
            if old_log_probs is not None:
                old_log_probs = old_log_probs.reshape(n * B).detach()  # 全 epoch 固定行为分母

            # ===== 2. 刷新 return_rms；GAE target 每个 rollout 只计算并冻结一次 =====
            self._update_rms(R)
            gae_batch = self._prepare_reward_gae(roll) if self.reward_mode == 'gae' else None

            # ===== 3. 内层更新：actor/PPO 与可选 reward value 复用同一批固定 target =====
            actor_info, value_info = {}, {}
            for _ in range(self.updates_per_iteration):
                if gae_batch is not None:
                    value_info = self._update_reward_value(gae_batch)
                actor_info = self._update_actor(
                    states, actions, R.detach(), C.detach(),
                    old_log_probs=old_log_probs,
                    reward_advantages=None if gae_batch is None else gae_batch['advantages'])

            # ===== 4. 外层 λ 更新 (经验 outage 驱动) =====
            if it % self.outer_interval == 0:
                self._update_dual(C.detach())

            # actor 更新期间 observation moments 必须冻结，确保本 rollout 的输入变换与
            # 采样时完全相同；本批统计只供下一次 rollout 使用，避免制造隐式 off-policy。
            self._update_obs_rms(roll['S'])

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
            if self.normalize_observation:
                obs_std = self.obs_normalizer.var.detach().clamp_min(0.0).sqrt()
                extra.update({
                    'obs_norm/count': float(self.obs_normalizer.count.item()),
                    'obs_norm/mean_abs': float(self.obs_normalizer.mean.detach().abs().mean().item()),
                    'obs_norm/std_min': float(obs_std.min().item()),
                    'obs_norm/std_median': float(obs_std.median().item()),
                    'obs_norm/std_max': float(obs_std.max().item()),
                })
            extra.update(value_info)
            extra.update(actor_info)
            self._log_core(it, R_np, Zc_np, Cu_np, extra=extra)
            self.scheduler.step()
            self.lambda_scheduler.step()

    # ============================================================ Actor 更新 (MC/GAE reward + 轨迹级 cost) ============================================================
    def _update_actor(self, states, actions, R, C, old_log_probs=None,
                      reward_advantages=None):
        """
        reward 权重可选轨迹 MC 或 per-step GAE；risk 始终是轨迹 outage indicator。
        PPO 时分别对 reward 用 pessimistic min、risk 用 conservative max，避免先把
        两者相减再 clip 导致约束项在混合符号 advantage 下失去保守性。
        """
        n, B = self.n, self.num_envs
        ind = (C >= self.cost_limit).float()                  # [B] 𝟙{C≥d} 上尾 (outage)
        risk_weight = ind.unsqueeze(0).expand(n, B).reshape(n * B)

        if self.reward_mode == 'mc':
            R_norm = (R - self.return_rms.mean) / self.return_rms.std
            reward_weight = R_norm.unsqueeze(0).expand(n, B).reshape(n * B)
        else:
            if reward_advantages is None:
                raise RuntimeError("GAE actor update requires frozen reward_advantages")
            reward_weight = reward_advantages
        lagrange = self.lambda_dual.detach()
        combined_weight = reward_weight - lagrange * risk_weight

        log_probs = self._compute_log_probs(states, actions)  # [n·B]
        ppo_info = {}
        if self.actor_update_mode == 'ppo':
            if old_log_probs is None:
                raise RuntimeError("PPO update requires rollout behavior old_log_probs")
            # ratio 分母在整个 rollout 的所有 epoch 中保持不变；optimizer.step() 后只
            # 重算分子 logπ_current，绝不能把 old_log_probs 覆盖成当前策略概率。
            log_ratio = log_probs - old_log_probs
            ratio = torch.exp(log_ratio)
            clipped_ratio = torch.clamp(
                ratio, 1.0 - self.ppo_ratio_clip, 1.0 + self.ppo_ratio_clip)
            reward_surr = torch.minimum(
                ratio * reward_weight.detach(), clipped_ratio * reward_weight.detach())
            risk_surr = torch.maximum(
                ratio * risk_weight.detach(), clipped_ratio * risk_weight.detach())
            actor_loss = -reward_surr.mean() + lagrange * risk_surr.mean()
            ppo_info = {
                'ppo/ratio_mean': float(ratio.detach().mean().item()),
                'ppo/ratio_std': float(ratio.detach().std(unbiased=False).item()),
                'ppo/clip_fraction': float(
                    ((ratio.detach() - 1.0).abs() > self.ppo_ratio_clip).float().mean().item()),
                'ppo/approx_kl': float(
                    ((ratio.detach() - 1.0) - log_ratio.detach()).mean().item()),
            }
        else:
            actor_loss = -(log_probs * combined_weight.detach()).mean()

        self.optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.actor_grad_clip and self.actor_grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.actor_grad_clip)
        else:
            grad_norm = torch.tensor(0.0, device=self.device)
        self.optimizer.step()

        # 可学习探索方差必须保留在有限区间，避免少数 MC 轨迹把高斯熵瞬间推爆/压没。
        with torch.no_grad():
            self.actor.log_std.clamp_(min=self.log_std_min, max=self.log_std_max)
        info = {
            'actor/loss': float(actor_loss.item()),
            'actor/weight_mean': float(combined_weight.mean().item()),
            'actor/weight_std': float(combined_weight.std(unbiased=False).item()),
            'actor/reward_weight_std': float(reward_weight.std(unbiased=False).item()),
            'actor/risk_weight_std': float(risk_weight.std(unbiased=False).item()),
            'actor/grad_norm': float(grad_norm.item()),
            'actor/log_std_mean': float(self.actor.log_std.detach().mean().item()),
            'debug/reward_mode_is_gae': float(self.reward_mode == 'gae'),
        }
        info.update(ppo_info)
        return info

    # ============================================================ 可选 scalar V_r(s,t)+GAE ============================================================
    def _value_inputs(self, states, steps):
        """共享 observation RMS，并按需追加 t/T，输出 [T·B,state_dim(+1)]。"""
        value_states = self._normalize_states(states)
        if not self.value_step_feature:
            return value_states
        step_feature = (steps.float() / self.n).reshape(-1, 1)
        return torch.cat([value_states, step_feature], dim=1)

    def _prepare_reward_gae(self, roll):
        """
        用 rollout 时的 V_r 计算冻结 GAE advantage 与 λ-return。

        每段 T=1000 是完整 episode，最后一步 next_value=0；中间位置取同轨迹下一时刻
        value。advantages/targets 在所有 PPO/value epochs 间不重算，避免移动监督目标。
        """
        if self.reward_value is None:
            raise RuntimeError("reward GAE requested while qcpo_reward_mode=mc")

        n, B = self.n, self.num_envs
        states = roll['S'].reshape(n * B, -1)
        rewards = roll['R']                                    # [T,B]
        steps = torch.arange(n, device=self.device).unsqueeze(1).expand(n, B).reshape(n * B)
        value_inputs = self._value_inputs(states, steps)

        with torch.no_grad():
            values_old = self.reward_value(value_inputs).reshape(n, B)
            next_values = torch.cat(
                [values_old[1:], torch.zeros_like(values_old[:1])], dim=0)
            deltas = rewards + self.gamma * next_values - values_old

            gae = torch.zeros(B, dtype=torch.float32, device=self.device)
            advantages = torch.empty_like(deltas)
            for t in range(n - 1, -1, -1):
                gae = deltas[t] + self.gamma * self.gae_lambda * gae
                advantages[t] = gae
            targets = advantages + values_old

            raw_advantages = advantages.reshape(n * B)
            actor_advantages = raw_advantages
            if self.reward_advantage_norm:
                adv_mean = raw_advantages.mean()
                adv_std = raw_advantages.std(unbiased=False).clamp_min(1e-8)
                actor_advantages = (raw_advantages - adv_mean) / adv_std

        return {
            'value_inputs': value_inputs.detach(),
            'targets': targets.reshape(n * B).detach(),
            'advantages': actor_advantages.detach(),
            'raw_advantages': raw_advantages.detach(),
        }

    def _update_reward_value(self, gae_batch):
        """拟合本 rollout 冻结的 GAE λ-return，并报告 value 拟合健康度。"""
        value_pred = self.reward_value(gae_batch['value_inputs'])
        value_targets = gae_batch['targets']
        value_error = value_pred - value_targets
        value_loss = 0.5 * value_error.pow(2).mean()

        self.reward_value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        if self.reward_value_grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.reward_value.parameters(), self.reward_value_grad_clip)
        else:
            grad_norm = torch.zeros((), device=self.device)
        self.reward_value_optimizer.step()

        target_var = value_targets.var(unbiased=False)
        explained_var = torch.where(
            target_var > 1e-8,
            1.0 - value_error.detach().var(unbiased=False) / target_var,
            torch.zeros_like(target_var))
        return {
            'reward_value/loss': float(value_loss.item()),
            'reward_value/explained_variance': float(explained_var.item()),
            'reward_value/grad_norm': float(grad_norm.item()),
            'advantage/mean_adv_std': float(
                gae_batch['raw_advantages'].std(unbiased=False).item()),
        }

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

    def _update_obs_rms(self, states):
        '''在 rollout 边界批量合并 observation moments；关闭开关时保持旧行为。'''
        if self.normalize_observation:
            self.obs_normalizer.update(states)

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
