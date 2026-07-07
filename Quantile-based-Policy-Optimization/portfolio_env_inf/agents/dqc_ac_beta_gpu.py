# -*- coding: utf-8 -*-
"""
DQCACBetaGPU —— DQC-AC-β 全 GPU、B 路并行向量化版 (portfolio_env_inf 移植)。

算法逐式对齐 risk_sensitive_env_inf/agents/dqc_ac_beta_gpu.py (已验证的 ∞-horizon 实现):
    优化问题:  max_θ E[Z]  s.t.  P_θ(Z ≤ q) ≤ α
    Critic:    QR-TD 分布式 critic ψ_i(s,a) (quantile Huber, target 软更新, 1-step bootstrap;
               continuing: 截断点恒 bootstrap, dones≡0)
    Actor:     per-transition 梯度权重 w = γ^t·Â_m - λ·β^t·Â_c
               Â_m = Q̂_m - V̂_m (critic 均值优势),  Â_c = Ψ̂(s,a,b_t) - V̂_c (局部 CDF 优势)
               budget 递推 b_0=q, b_{t+1}=(b_t-r_t)/γ
    Dual:      λ ← [λ + ε_k·(Ĝ - α)]_+, Ĝ = critic 在初始态 (s0,a0) 估计的 P(Z≤q)
               (与 actor 的 critic 信号同源, 见模板 update_dual 注释)
    归一化:    advantage_norm='qcpo' → 跨迭代 EMA σ_ret/σ_c 归一化 (反 λ 卷绕, 已验证配方)

与模板的仅有差异 (环境适配, 数学不变):
    1. 环境/策略/日志骨架走 VecAgentBase (PortfolioVecTorch + 25 维观测 + 5 维动作);
    2. budget b_t 改为 rollout 后从奖励矩阵【后处理递推】(b 只依赖过去奖励, 数学等价);
    3. 本环境严格平稳 → critic_step_feature 恒默认 False (机制保留作 A/B)。
"""
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam                                  # base lr=1 + LambdaLR (模板配方)
from torch.optim.lr_scheduler import LambdaLR                 # lr = a/(b+k)^c
import wandb

from utils import RunningMeanStd                              # EMA 归一化器 (qcpo 配方)
from .vec_base import VecAgentBase                            # 共享: env/策略/rollout/日志
from .common import DistributionalCritic, lr_lambda           # QR critic + lr 衰减 (共享件)


class DQCACBetaGPU(VecAgentBase):
    """DQC-AC-β 全 GPU 向量化版 (per-transition TD + critic-dual; 与模板算法一致)。"""

    # ============================================================ 初始化 ============================================================
    def __init__(self, args, env):
        super().__init__(args, env)                            # 基类: env/actor/wandb/维度

        # -------------------- 算法专属超参 (与模板同名同义) --------------------
        self.beta = getattr(args, 'beta', 0.95)                # Abel 风险折扣 β (仅约束项)
        self.outer_interval = max(1, int(getattr(args, 'outer_interval', 1)))  # λ 更新间隔
        self.num_quantiles = getattr(args, 'num_quantiles', 32)            # 分位数个数 N
        self.huber_kappa = getattr(args, 'huber_kappa', 0.1)               # κ (0.1=近纯分位回归, 已验证)
        self.target_tau = getattr(args, 'target_tau', 0.05)                # target 软更新系数
        self.target_update_interval = getattr(args, 'target_update_interval', 1)
        self.n_step = max(1, int(getattr(args, 'n_step', 1)))              # TD 步数 (保持 1)
        self.critic_step_feature = bool(getattr(args, 'critic_step_feature', False))  # 平稳环境: False
        self.num_action_samples = max(1, int(getattr(args, 'num_action_samples', 4)))  # baseline K
        self.updates_per_episode = max(1, int(getattr(args, 'updates_per_episode', 10)))
        self.advantage_norm = getattr(args, 'advantage_norm', 'qcpo')      # qcpo (已验证) / separate / none
        self.entropy_coef = getattr(args, 'entropy_coef', 0.0)
        self.lambda_max = getattr(args, 'lambda_max', 50.0)
        self.lambda_min = getattr(args, 'lambda_min', 0.0)
        self.critic_grad_clip = getattr(args, 'critic_grad_clip', 10.0)
        self.actor_grad_clip = getattr(args, 'actor_grad_clip', 100.0)

        # qcpo 归一化配方 (与模板一致: EMA σ_ret / σ_c + warmup)
        self.norm_ema_decay = float(getattr(args, 'norm_ema_decay', 0.1))
        _wi = getattr(args, 'warmup_iters', None)
        self.warmup_iters = int(_wi) if _wi is not None else (5 if self.advantage_norm == 'qcpo' else 0)
        self.return_rms = RunningMeanStd(decay=self.norm_ema_decay)       # σ_ret (回报分布尺度)
        self.constraint_rms = RunningMeanStd(decay=self.norm_ema_decay)   # σ_c (约束优势尺度)

        # 分位数水平 τ_i = (i+0.5)/N
        self.taus = torch.tensor(
            [(i + 0.5) / self.num_quantiles for i in range(self.num_quantiles)],
            dtype=torch.float32, device=self.device)

        # -------------------- actor 两时间尺度优化器 --------------------
        self.actor_optimizer = Adam(self.actor.parameters(), 1.0, eps=1e-5)
        self.actor_scheduler = LambdaLR(
            self.actor_optimizer,
            lr_lambda=lambda k: lr_lambda(k, args.theta_a, args.theta_b, args.theta_c))

        # -------------------- 分布式 critic + target --------------------
        hidden = getattr(args, 'critic_hidden', [64, 64])
        if isinstance(hidden, str):
            hidden = [int(x) for x in hidden.split(',') if x.strip()]
        cdim = self.state_dim + (1 if self.critic_step_feature else 0)    # step 增广维度
        self.critic = DistributionalCritic(cdim, self.action_dim,
                                           self.num_quantiles, list(hidden)).to(self.device)
        self.target_critic = DistributionalCritic(cdim, self.action_dim,
                                                  self.num_quantiles, list(hidden)).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())      # 硬拷贝初始化
        self.critic_optimizer = Adam(self.critic.parameters(),
                                     getattr(args, 'critic_lr', 1e-3), eps=1e-5)

        # -------------------- 拉格朗日乘子 λ --------------------
        self.lambda_dual = torch.tensor([0.0], dtype=torch.float32,
                                        device=self.device, requires_grad=True)
        self.lambda_optimizer = Adam([self.lambda_dual], 1.0, eps=1e-5)
        self.lambda_scheduler = LambdaLR(
            self.lambda_optimizer,
            lr_lambda=lambda k: lr_lambda(k, args.lambda_a, args.lambda_b, args.lambda_c))

        # -------------------- 运行时统计 (诊断量, 与模板一致) --------------------
        self.learning_steps = 0
        self.last_dual_prob = 0.0                              # dual 用的 critic P(Z≤q)
        self.last_cdf_initial = 0.0                            # critic 估计 P(Z≤q) (诊断)
        self.last_empirical_prob = 0.0
        self.last_pred_return_mean = 0.0                       # critic 估 E[Z|s0]
        self.last_pred_return_std = 0.0                        # critic 估 std(Z|s0)

    # ============================================================ 主训练循环 (与模板 train 一致) ============================================================
    def train(self):
        """每迭代: 采样 → (qcpo)刷 EMA → 内层 critic+actor → (外层)λ → 日志+调度。"""
        print(f"DQCACBetaGPU: beta={self.beta}, alpha={self.q_alpha}, q={self.quantile_threshold}, "
              f"N={self.num_quantiles}, B={self.num_envs}, iters={self.num_iterations}, "
              f"updates/iter={self.updates_per_episode}, device={self.device}")

        for it in range(self.num_iterations):
            # ===== 1. 并行采样 + DQCAC 专属后处理 (budget / n-step / d,e 权重) =====
            batch = self._rollout_vec()

            # ===== 1b. qcpo: 刷新 EMA 归一化器 (σ_ret/σ_c, 内层冻结 → 稳定) =====
            if self.advantage_norm == 'qcpo':
                self._update_norm_stats(batch)
            in_warmup = it < self.warmup_iters                 # warmup: 只练 critic + EMA

            # ===== 2. 内层更新 (critic + actor) =====
            critic_info, actor_info = {}, {}
            for _ in range(self.updates_per_episode):
                critic_info = self.update_critic(batch)
                if not in_warmup:
                    actor_info = self.update_actor(batch)
                if self.learning_steps % self.target_update_interval == 0:
                    self._soft_update_target()
                self.learning_steps += 1

            # ===== 3. 外层 λ 更新 (critic 在 (s0,a0) 估计的 P(Z≤q) 驱动) =====
            if not in_warmup and it % self.outer_interval == 0:
                self.update_dual(batch)

            # ===== 4. 日志 + 调度 =====
            self._log(it, batch, critic_info, actor_info)
            self.actor_scheduler.step()
            self.lambda_scheduler.step()

    # ============================================================ 采样 + 后处理 (对齐模板 _rollout_vec) ============================================================
    def _rollout_vec(self):
        """
        基类 _rollout_core 采 B 条轨迹后, 补齐 DQCAC 需要的批量字段 (键名与模板一致):
            budgets: b_0=q, b_{t+1}=(b_t-r_t)/γ  —— b 只依赖过去奖励 → 从 R 矩阵后处理递推
                     (与模板 rollout 内逐步递推【数学等价】, 实现更简)
            n-step TD 件: nstep_reward / boot_states / boot_mask / steps / boot_steps
                     continuing: 恒 bootstrap (boot_mask≡1), dones≡0, 截断次态 s_T 可达
            d=γ^t, e=β^t: 只随 t 变, 广播 [n·B]
        """
        n, B = self.n, self.num_envs
        roll = self._rollout_core()                            # S [n,B,sd], A, R [n,B], S2_last
        Smat, Amat, Rmat = roll['S'], roll['A'], roll['R']

        # ---- budget 后处理递推: b[t+1] = (b[t] - r[t]) / γ  (b[0]=q) ----
        Bud = torch.empty(n, B, dtype=torch.float32, device=self.device)
        Bud[0] = float(self.quantile_threshold)                # b_0 = q
        for t in range(n - 1):
            Bud[t + 1] = (Bud[t] - Rmat[t]) / self.gamma       # 递推 (逐式同模板)

        # ---- 摊平 [n·B, *] (行序: 先 t 后 env, index = t*B + j, 同模板) ----
        states = Smat.reshape(n * B, -1)
        actions = Amat.reshape(n * B, -1)
        rewards = Rmat.reshape(n * B)
        budgets = Bud.reshape(n * B)
        # next_states: S 序列右移一格 + 末位截断次态 s_T
        S_ext = torch.cat([Smat, roll['S2_last'].unsqueeze(0)], dim=0)    # [n+1,B,sd]
        next_states = S_ext[1:].reshape(n * B, -1)

        # ---- n-step TD 目标预计算 (continuing: 恒 bootstrap; N=1 即精确 1-step) ----
        Ns = self.n_step
        nstep_rew = torch.zeros(n, B, dtype=torch.float32, device=self.device)
        disc = 1.0
        for k in range(Ns):                                    # Σ_{k<N} γ^k r_{t+k} (末端自动截断)
            nstep_rew[:n - k] += disc * Rmat[k:]
            disc *= self.gamma
        t_ar = torch.arange(n, device=self.device)
        boot_idx = torch.clamp(t_ar + Ns, max=n)               # bootstrap 态索引 (可达 s_T=n)
        boot_states = S_ext[boot_idx].reshape(n * B, -1)
        boot_mask = torch.ones(n * B, dtype=torch.float32, device=self.device)  # 恒 bootstrap
        steps = t_ar.unsqueeze(1).expand(n, B).reshape(n * B).float()           # step 特征 (默认未用)
        boot_steps = boot_idx.unsqueeze(1).expand(n, B).reshape(n * B).float()

        # ---- d=γ^t, e=β^t ----
        t_idx = torch.arange(n, dtype=torch.float32, device=self.device)
        d = (self.gamma ** t_idx).unsqueeze(1).expand(n, B).reshape(n * B)
        e = (self.beta ** t_idx).unsqueeze(1).expand(n, B).reshape(n * B)

        return {
            'states': states, 'actions': actions, 'rewards': rewards,
            'next_states': next_states,
            'dones': torch.zeros(n * B, dtype=torch.float32, device=self.device),
            'nstep_reward': nstep_rew.reshape(n * B),
            'boot_states': boot_states, 'boot_mask': boot_mask,
            'steps': steps, 'boot_steps': boot_steps,
            'budgets': budgets, 'd': d, 'e': e,
            's0': Smat[0],                                     # [B,sd] 初始状态
            'disc_returns': roll['disc_returns'],              # [B] Z(τ)
        }

    # ============================================================ Critic 更新 (QRTD, 与模板逐式一致) ============================================================
    def update_critic(self, batch):
        """y_j = Σγ^k r + γ^N·ψ̄_j(s_{t+N},a''), a''~π; quantile Huber 回归 (建图侧 ψ_i(s,a))。"""
        s, a = batch['states'], batch['actions']
        nstep_r = batch['nstep_reward']
        boot_s, boot_mask = batch['boot_states'], batch['boot_mask']
        steps, boot_steps = batch['steps'], batch['boot_steps']

        with torch.no_grad():                                  # target 不建图
            a_boot = self._sample_actions(boot_s)              # a''~π(·|s_{t+N})
            psi_next = self.target_critic(self._aug(boot_s, boot_steps), a_boot)
            y = nstep_r.unsqueeze(1) + (self.gamma ** self.n_step) * boot_mask.unsqueeze(1) * psi_next

        psi = self.critic(self._aug(s, steps), a)              # [n·B, N] (建图)
        loss = self._quantile_huber_loss(psi, y)

        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.critic_grad_clip and self.critic_grad_clip > 0:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_grad_clip)
        self.critic_optimizer.step()
        return {'critic/quantile_huber_loss': float(loss.item()),
                'critic/quantile_huber_loss_normalized': float(loss.item()) / self.num_quantiles}

    def _quantile_huber_loss(self, psi, y):
        """Quantile Huber Loss ρ^κ_τ(u)=|τ-1{u<0}|·L_κ(u)/κ (QR-DQN 核心, 与模板一致)。"""
        u = y.unsqueeze(1) - psi.unsqueeze(2)                  # [*,N,N] 配对 TD error
        abs_u = u.abs()
        huber = torch.where(abs_u <= self.huber_kappa,
                            0.5 * u.pow(2),
                            self.huber_kappa * (abs_u - 0.5 * self.huber_kappa))
        taus = self.taus.view(1, -1, 1)                        # [1,N,1] (i 维)
        weight = (taus - (u.detach() < 0).float()).abs()
        return (weight * huber / self.huber_kappa).sum(dim=2).mean(dim=1).mean()

    # ============================================================ Actor 更新 (与模板逐式一致) ============================================================
    def update_actor(self, batch):
        """w = γ^t·Â_m - λ·β^t·Â_c;  loss = -E[logπ·w] - τ_H·E[H]。"""
        s, a, b = batch['states'], batch['actions'], batch['budgets']
        d, e = batch['d'], batch['e']
        steps = batch['steps']

        with torch.no_grad():                                  # 优势不建图
            psi = self.critic(self._aug(s, steps), a)          # [n·B, N]
            q_m = psi.mean(dim=1)                              # Q̂_m(s,a)
            psi_c = (psi <= b.unsqueeze(1)).float().mean(dim=1)  # Ψ̂(s,a,b)
            v_m, v_c = self._estimate_baselines(s, b, steps)   # V̂_m, V̂_c
            if self.advantage_norm == 'qcpo':
                a_m = (q_m - v_m) / self.return_rms.std        # EMA σ_ret 归一化 (已验证配方)
                a_c = (psi_c - v_c) / self.constraint_rms.std  # EMA σ_c 归一化
            else:
                a_m = self._maybe_norm(q_m - v_m)
                a_c = self._maybe_norm(psi_c - v_c)
            w = d * a_m - self.lambda_dual.detach() * e * a_c  # [n·B] 复合权重

        log_probs = self._compute_log_probs(s, a)              # 对 θ 可导
        actor_loss = -(log_probs * w).mean() \
            - self.entropy_coef * self._entropy(s).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.actor_grad_clip and self.actor_grad_clip > 0:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip)
        self.actor_optimizer.step()

        raw_adv_m = q_m - v_m
        raw_adv_c = psi_c - v_c
        return {'actor/loss': float(actor_loss.item()),
                'advantage/mean_adv_std': float(raw_adv_m.std(unbiased=False).item()),
                'advantage/risk_adv_std': float(raw_adv_c.std(unbiased=False).item()),
                'critic/q_mean': float(q_m.mean().item()),
                'constraint/psi_c_mean': float(psi_c.mean().item()),
                'actor/w_mean': float(w.mean().item()),
                'actor/w_std': float(w.std(unbiased=False).item())}

    # ============================================================ Dual 更新 (critic-prob 驱动, 与模板一致) ============================================================
    def update_dual(self, batch):
        """Ĝ = critic 在 B 个 (s0, a0~π) 上估计的 P(Z≤q); λ ← [λ + ε_k(Ĝ-α)]_+ (同源信号)。"""
        with torch.no_grad():
            s0 = batch['s0']
            a0 = self._sample_actions(s0)
            psi0 = self.critic(self._aug(s0, 0), a0)           # [B, N]
            p = float((psi0 <= self.quantile_threshold).float().mean(dim=1).mean().item())
        self.last_dual_prob = p

        gap = torch.tensor([p - self.q_alpha], dtype=torch.float32, device=self.device)
        self.lambda_optimizer.zero_grad(set_to_none=True)
        (-self.lambda_dual * gap).backward()
        self.lambda_optimizer.step()
        with torch.no_grad():
            self.lambda_dual.clamp_(min=self.lambda_min, max=self.lambda_max)

    # ============================================================ 辅助 (与模板同语义) ============================================================
    def _aug(self, states, steps):
        """critic step 增广 (critic_step_feature=False 时原样返回; 本环境平稳, 默认关闭)。"""
        if not self.critic_step_feature:
            return states
        if not torch.is_tensor(steps):
            steps = torch.full((states.shape[0],), float(steps), device=states.device)
        sf = (steps.float() / self.n).reshape(-1, 1)
        return torch.cat([states, sf], dim=1)

    def _estimate_baselines(self, states, budgets, steps=None):
        """V̂_m(s)=E_a[Q̂_m], V̂_c(s,b)=E_a[Ψ̂] —— K 个动作样本近似 (no_grad 中调用)。"""
        b_col = budgets.unsqueeze(1)
        s_aug = self._aug(states, steps) if steps is not None else states
        q_list, c_list = [], []
        for _ in range(self.num_action_samples):
            a = self._sample_actions(states)
            psi = self.critic(s_aug, a)
            q_list.append(psi.mean(dim=1))
            c_list.append((psi <= b_col).float().mean(dim=1))
        return (torch.stack(q_list, dim=0).mean(dim=0),
                torch.stack(c_list, dim=0).mean(dim=0))

    def _maybe_norm(self, x, eps=1e-8):
        """'separate': 批内 (x-mean)/std; 'none': 原样 (与模板一致)。"""
        if self.advantage_norm != 'separate' or x.numel() <= 1:
            return x
        std = x.std(unbiased=False)
        if std.item() < eps:
            return x - x.mean()
        return (x - x.mean()) / (std + eps)

    def _ema_update(self, rms, batch_mean, batch_var):
        """整批矩 EMA 混入 RunningMeanStd (与模板 _ema_update 一致)。"""
        if not rms._initialized:
            rms.mean = batch_mean
            rms.var = max(batch_var, 1e-8)
            rms._initialized = True
            return
        a = rms.decay
        rms.mean = (1.0 - a) * rms.mean + a * batch_mean
        rms.var = (1.0 - a) * rms.var + a * batch_var

    def _update_norm_stats(self, batch):
        """qcpo 模式: 每迭代刷新 σ_ret (轨迹回报) 与 σ_c (约束优势) 两个 EMA (内层冻结)。"""
        z = batch['disc_returns'].detach()                     # [B]
        self._ema_update(self.return_rms,
                         float(z.mean().item()), float(z.var(unbiased=False).item()))
        with torch.no_grad():
            s, a, b = batch['states'], batch['actions'], batch['budgets']
            steps = batch['steps']
            psi = self.critic(self._aug(s, steps), a)
            psi_c = (psi <= b.unsqueeze(1)).float().mean(dim=1)
            _, v_c = self._estimate_baselines(s, b, steps)
            adv_c = psi_c - v_c
        self._ema_update(self.constraint_rms,
                         float(adv_c.mean().item()), float(adv_c.var(unbiased=False).item()))

    def _soft_update_target(self):
        """Polyak: target ← (1-τ)·target + τ·online。"""
        with torch.no_grad():
            for tp, op in zip(self.target_critic.parameters(), self.critic.parameters()):
                tp.data.lerp_(op.data, self.target_tau)

    def _initial_cdf_estimate(self, s0):
        """critic 在 s0 上的 P(Z≤q) / E[Z] / std(Z) 估计 (校准诊断, 与模板一致)。"""
        with torch.no_grad():
            a0 = self._sample_actions(s0)
            psi = self.critic(self._aug(s0, 0), a0)            # [B, N]
            self.last_cdf_initial = float(
                (psi <= self.quantile_threshold).float().mean(dim=1).mean())
            self.last_pred_return_mean = float(psi.mean().item())
            self.last_pred_return_std = float(psi.std(dim=1).mean().item())
        return self.last_cdf_initial

    # ============================================================ 日志 (统一键 + 模板专属键) ============================================================
    def _log(self, it, batch, critic_info, actor_info):
        """基类统一键 (_log_core) + DQCAC 专属诊断键 (budget/critic 校准/λ/norm)。"""
        z = batch['disc_returns'].detach().cpu().numpy()
        ghat = self._initial_cdf_estimate(batch['s0'])         # critic 估计 P(Z≤q)
        budgets_np = batch['budgets'].detach().cpu().numpy()
        empirical_prob = float(np.mean(z <= self.quantile_threshold))

        extra = {
            'constraint/cdf_estimate_initial': ghat,
            'constraint/cdf_calibration_error': abs(ghat - empirical_prob),
            'critic/pred_return_mean': self.last_pred_return_mean,
            'critic/pred_return_std': self.last_pred_return_std,
            'constraint/dual_prob': self.last_dual_prob,
            'lambda/value': float(self.lambda_dual.detach().item()),
            'lambda/lr': float(self.lambda_scheduler.get_last_lr()[0]),
            'budget/min': float(np.min(budgets_np)),
            'budget/max': float(np.max(budgets_np)),
            'budget/mean': float(np.mean(budgets_np)),
            'training/learning_steps': self.learning_steps,
            'training/actor_lr': float(self.actor_scheduler.get_last_lr()[0]),
        }
        extra.update(critic_info)
        extra.update(actor_info)
        if self.advantage_norm == 'qcpo':
            extra['norm/return_sigma_ema'] = float(self.return_rms.std)
            extra['norm/constraint_sigma_ema'] = float(self.constraint_rms.std)

        # q_est 键: 报本批经验 α-分位数 (与模板一致)
        self._log_core(it, z, q_est_value=float(np.percentile(z, self.q_alpha * 100)),
                       extra=extra)

        if it % self.log_interval == 0 and it != 0:            # 模板风格第 2 行诊断
            print(f'Iter:{it:05d} || P(Z<=q):{empirical_prob:.03f} '
                  f'alpha:{self.q_alpha:.03f} Ghat_critic:{self.last_cdf_initial:.03f} '
                  f'lambda:{self.lambda_dual.item():.04f}\n')

    # ============================================================ 总结接口 ============================================================
    def get_training_summary(self):
        """暴露最终约束指标 (run_experiment.py 可选调用)。"""
        return {
            'lambda_final': float(self.lambda_dual.detach().item()),
            'empirical_violation_prob': self.last_empirical_prob,
            'cdf_estimate_initial': self.last_cdf_initial,
            'dual_prob': self.last_dual_prob,
            'beta': self.beta,
            'num_envs': self.num_envs,
            'num_iterations': self.num_iterations,
        }
