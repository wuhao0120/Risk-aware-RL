# -*- coding: utf-8 -*-
"""
DQCACBetaGPU (safety_gym_env · CMDP 版) —— 用户 DQC-AC-β 迁移到 safety-gym outage 约束环境。

算法逐式对齐 portfolio_env_inf/agents/dqc_ac_beta_gpu.py (已验证的 per-transition DQCAC),
核心三机制不变, 仅按 CMDP【双回报流】扩展 + 约束翻为【上尾 cost】:
    优化问题:  max_θ E[R]  s.t.  P_θ(C ≥ d) ≤ ω
    Critic:    【两个】QR 分布式 critic (都 QR-TD, target 软更新, 1-step bootstrap, 截断恒 bootstrap):
                 reward_critic ψ^r(s,a) → 目标均值优势 Q̂_m=mean(ψ^r)
                 cost_critic   ψ^c(s,a) → 约束【上尾】CDF Ψ̂(s,a,b)=(1/N)Σ𝟙{ψ^c_i ≥ b}
    Actor:     per-transition 权重 w = γ^t·Â_m - λ·β^t·Â_c
                 Â_m = Q̂_m - V̂_m         (reward 均值优势)
                 Â_c = Ψ̂(s,a,b) - V̂_c    (cost 局部上尾 CDF 优势)
               budget 在【cost】上递推: b_0=d, b_{t+1}=(b_t - c_t)/γc  (对应论文 remain_discounted_cost)
    Dual:      λ ← [λ + ε_k·(Ĝ - ω)]_+,  Ĝ = cost_critic 在 (s0,a0) 估计的 P(C≥d) (上尾, 同源信号)
    归一化:    advantage_norm='qcpo' → 跨迭代 EMA σ_R (reward 回报) / σ_c (cost 约束优势)

与 portfolio 单-critic 版的差异 (数学同构):
    1. 目标与约束分属 reward/cost 两条流 → 两个分布式 critic (原版一个 critic 兼顾均值+CDF);
    2. budget 与约束在 cost 上, 且 indicator 为【上尾】indicator_ge (原版下尾);
    3. 环境 = SafetyVecEnv (CPU); critic_step_feature 默认 False (截断 bootstrap = continuing 处理)。

约束口径双支持 (episodic 开关, 默认由 cost_gamma 推断):
    - 未折扣口径 (论文对齐, cost_gamma=1): episodic=True → episode 末【不 bootstrap】
      (无穷期未折扣目标发散) + critic_step_feature=True (剩余 cost 分布依赖剩余步数,
      用户在 risk_sensitive_env 已验证的有限期界配方); budget 递推自动退化 b←b−c。
    - 折扣口径 (cost_gamma=0.99): episodic=False → 截断恒 bootstrap (continuing 处理,
      γc^1000≈4e-5, episode 末残差可忽略), critic_step_feature=False。
"""
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from utils import RunningMeanStd
from .vec_base import VecAgentBase
from .common import DistributionalCritic, lr_lambda, indicator_ge


class DQCACBetaGPU(VecAgentBase):
    """DQC-AC-β CMDP 版 (reward+cost 双分布式 critic; budget/critic-dual 在 cost 上; 上尾约束)。"""

    # ============================================================ 初始化 ============================================================
    def __init__(self, args, env):
        super().__init__(args, env)                           # 基类: env/actor/wandb/维度

        # -------------------- 算法专属超参 (与 portfolio 模板同名同义) --------------------
        self.beta = getattr(args, 'beta', 0.95)               # Abel 风险折扣 β (仅 cost 约束项)
        self.outer_interval = max(1, int(getattr(args, 'outer_interval', 1)))
        self.num_quantiles = getattr(args, 'num_quantiles', 32)
        self.huber_kappa = getattr(args, 'huber_kappa', 0.1)  # κ (0.1=近纯分位回归, 无偏)
        self.target_tau = getattr(args, 'target_tau', 0.05)   # target 软更新系数
        self.target_update_interval = getattr(args, 'target_update_interval', 1)
        self.n_step = max(1, int(getattr(args, 'n_step', 1)))
        # episodic (未折扣论文口径) 默认由 cost_gamma 推断: γc=1 → episode 末不 bootstrap
        self.episodic = bool(getattr(args, 'episodic', self.cost_gamma >= 1.0 - 1e-9))
        # step feature 默认跟随 episodic: 有限期界下剩余 cost 分布依赖剩余步数 (已验证配方)
        _csf = getattr(args, 'critic_step_feature', None)
        self.critic_step_feature = bool(_csf) if _csf is not None else self.episodic
        self.num_action_samples = max(1, int(getattr(args, 'num_action_samples', 4)))
        self.updates_per_episode = max(1, int(getattr(args, 'updates_per_episode', 10)))
        self.advantage_norm = getattr(args, 'advantage_norm', 'qcpo')
        self.entropy_coef = getattr(args, 'entropy_coef', 0.0)
        self.lambda_max = getattr(args, 'lambda_max', 50.0)
        self.lambda_min = getattr(args, 'lambda_min', 0.0)
        self.critic_grad_clip = getattr(args, 'critic_grad_clip', 10.0)
        self.actor_grad_clip = getattr(args, 'actor_grad_clip', 100.0)

        # qcpo 归一化配方: σ_R (reward 回报尺度) / σ_c (cost 约束优势尺度) + warmup
        self.norm_ema_decay = float(getattr(args, 'norm_ema_decay', 0.1))
        _wi = getattr(args, 'warmup_iters', None)
        self.warmup_iters = int(_wi) if _wi is not None else (5 if self.advantage_norm == 'qcpo' else 0)
        self.return_rms = RunningMeanStd(decay=self.norm_ema_decay)       # σ_R (reward 回报)
        self.constraint_rms = RunningMeanStd(decay=self.norm_ema_decay)   # σ_c (cost 约束优势)

        # 分位数水平 τ_i=(i+0.5)/N
        self.taus = torch.tensor(
            [(i + 0.5) / self.num_quantiles for i in range(self.num_quantiles)],
            dtype=torch.float32, device=self.device)

        # -------------------- actor 两时间尺度优化器 (Adam, 与模板一致) --------------------
        self.actor_optimizer = Adam(self.actor.parameters(), 1.0, eps=1e-5)
        self.actor_scheduler = LambdaLR(
            self.actor_optimizer,
            lr_lambda=lambda k: lr_lambda(k, args.theta_a, args.theta_b, args.theta_c))

        # -------------------- 两个分布式 critic + target (reward / cost) --------------------
        hidden = getattr(args, 'critic_hidden', [256, 256])   # safety-gym 观测高维 → MLP 大一点
        if isinstance(hidden, str):
            hidden = [int(x) for x in hidden.split(',') if x.strip()]
        cdim = self.state_dim + (1 if self.critic_step_feature else 0)
        # reward critic: ψ^r(s,a), 只用其均值 mean(ψ^r) 作 Q̂_m (目标优势)
        self.reward_critic = DistributionalCritic(cdim, self.action_dim,
                                                  self.num_quantiles, list(hidden)).to(self.device)
        self.reward_target_critic = DistributionalCritic(cdim, self.action_dim,
                                                         self.num_quantiles, list(hidden)).to(self.device)
        self.reward_target_critic.load_state_dict(self.reward_critic.state_dict())
        # cost critic: ψ^c(s,a), 用【上尾】CDF Ψ̂(s,a,b) 作约束优势 + budget 查询
        self.cost_critic = DistributionalCritic(cdim, self.action_dim,
                                                self.num_quantiles, list(hidden)).to(self.device)
        self.cost_target_critic = DistributionalCritic(cdim, self.action_dim,
                                                       self.num_quantiles, list(hidden)).to(self.device)
        self.cost_target_critic.load_state_dict(self.cost_critic.state_dict())
        # 一个优化器统管两 critic (loss = reward_qr + cost_qr, 一次 backward)
        self.critic_optimizer = Adam(
            list(self.reward_critic.parameters()) + list(self.cost_critic.parameters()),
            getattr(args, 'critic_lr', 1e-3), eps=1e-5)

        # -------------------- 拉格朗日乘子 λ --------------------
        self.lambda_dual = torch.tensor([0.0], dtype=torch.float32,
                                        device=self.device, requires_grad=True)
        self.lambda_optimizer = Adam([self.lambda_dual], 1.0, eps=1e-5)
        self.lambda_scheduler = LambdaLR(
            self.lambda_optimizer,
            lr_lambda=lambda k: lr_lambda(k, args.lambda_a, args.lambda_b, args.lambda_c))

        # -------------------- 运行时统计 (诊断量) --------------------
        self.learning_steps = 0
        self.last_dual_prob = 0.0                             # dual 用的 cost-critic P(C≥d)
        self.last_cdf_initial = 0.0                           # cost-critic 估计 P(C≥d) (校准诊断)
        self.last_pred_cost_mean = 0.0                        # cost-critic 估 E[C|s0]
        self.last_pred_cost_std = 0.0                         # cost-critic 估 std(C|s0)

    # ============================================================ 主训练循环 (与模板一致) ============================================================
    def train(self):
        """每迭代: 采样(含cost) → (qcpo)刷 EMA → 内层双critic+actor → (外层)λ → 日志+调度。"""
        print(f"DQCACBetaGPU[CMDP]: env={self.env_name}, beta={self.beta}, omega={self.q_alpha}, "
              f"d(cost_limit)={self.cost_limit}, cost_gamma={self.cost_gamma}, episodic={self.episodic}, "
              f"step_feature={self.critic_step_feature}, N={self.num_quantiles}, B={self.num_envs}, T={self.n}, "
              f"iters={self.num_iterations}, updates/iter={self.updates_per_episode}, device={self.device}")

        for it in range(self.num_iterations):
            # ===== 1. 采样 + DQCAC 专属后处理 (cost budget / n-step / d,e) =====
            batch = self._rollout_vec()

            # ===== 1b. qcpo: 刷新 EMA 归一化器 (σ_R / σ_c) =====
            if self.advantage_norm == 'qcpo':
                self._update_norm_stats(batch)
            in_warmup = it < self.warmup_iters

            # ===== 2. 内层更新 (双 critic + actor) =====
            critic_info, actor_info = {}, {}
            for _ in range(self.updates_per_episode):
                critic_info = self.update_critic(batch)
                if not in_warmup:
                    actor_info = self.update_actor(batch)
                if self.learning_steps % self.target_update_interval == 0:
                    self._soft_update_target()
                self.learning_steps += 1

            # ===== 3. 外层 λ 更新 (cost-critic 在 (s0,a0) 估 P(C≥d) 驱动) =====
            if not in_warmup and it % self.outer_interval == 0:
                self.update_dual(batch)

            # ===== 4. 日志 + 调度 =====
            self._log(it, batch, critic_info, actor_info)
            self.actor_scheduler.step()
            self.lambda_scheduler.step()

    # ============================================================ 采样 + 后处理 (对齐模板, budget 换 cost) ============================================================
    def _rollout_vec(self):
        """
        基类 _rollout_core 采 B 条轨迹 (含 cost 流) 后, 补齐 DQCAC 批量字段:
            cost budget: b_0=d, b_{t+1}=(b_t - c_t)/γc  —— 从 cost 矩阵 C 后处理递推 (对应论文 remain cost)
            reward/cost 各自的 n-step TD 件 (continuing: 截断恒 bootstrap, dones≡0)
            d=γ^t (reward 折扣), e=β^t (Abel 风险折扣)
        """
        n, B = self.n, self.num_envs
        roll = self._rollout_core()                           # S,A,R(reward),C(cost),S2_last
        Smat, Amat, Rmat, Cmat = roll['S'], roll['A'], roll['R'], roll['C']

        # ---- cost budget 后处理递推: b[t+1]=(b[t]-c[t])/γc, b[0]=d ----
        Bud = torch.empty(n, B, dtype=torch.float32, device=self.device)
        Bud[0] = float(self.cost_limit)                       # b_0 = d
        for t in range(n - 1):
            Bud[t + 1] = (Bud[t] - Cmat[t]) / self.cost_gamma # 递推 (只依赖过去 cost)

        # ---- 摊平 [n·B, *] (行序: 先 t 后 env, index=t·B+j) ----
        states = Smat.reshape(n * B, -1)
        actions = Amat.reshape(n * B, -1)
        rewards = Rmat.reshape(n * B)
        costs = Cmat.reshape(n * B)
        budgets = Bud.reshape(n * B)
        # next_states: 右移一格 + 末位截断次态 s_T
        S_ext = torch.cat([Smat, roll['S2_last'].unsqueeze(0)], dim=0)    # [n+1,B,sd]
        next_states = S_ext[1:].reshape(n * B, -1)

        # ---- n-step TD 目标预计算 (reward 用 γ; cost 用 γc) ----
        # bootstrap 口径: episodic=False (折扣/continuing) → 截断恒 bootstrap;
        #                episodic=True  (未折扣/论文口径) → episode 末不 bootstrap
        #                (t+Ns 越过 episode 末端的行 mask=0, 其 n-step 和已按可得奖励截断)。
        Ns = self.n_step
        nstep_rew = torch.zeros(n, B, dtype=torch.float32, device=self.device)
        nstep_cost = torch.zeros(n, B, dtype=torch.float32, device=self.device)
        dr, dc = 1.0, 1.0
        for k in range(Ns):
            nstep_rew[:n - k] += dr * Rmat[k:]; dr *= self.gamma
            nstep_cost[:n - k] += dc * Cmat[k:]; dc *= self.cost_gamma
        t_ar = torch.arange(n, device=self.device)
        boot_idx = torch.clamp(t_ar + Ns, max=n)              # bootstrap 态索引 (可达 s_T=n)
        boot_states = S_ext[boot_idx].reshape(n * B, -1)
        if self.episodic:                                     # episode 末不 bootstrap (有限期界)
            boot_mask = (t_ar + Ns < n).float().unsqueeze(1).expand(n, B).reshape(n * B)
        else:                                                 # continuing: 恒 bootstrap
            boot_mask = torch.ones(n * B, dtype=torch.float32, device=self.device)
        steps = t_ar.unsqueeze(1).expand(n, B).reshape(n * B).float()
        boot_steps = boot_idx.unsqueeze(1).expand(n, B).reshape(n * B).float()

        # ---- d=γ^t, e=β^t ----
        t_idx = torch.arange(n, dtype=torch.float32, device=self.device)
        d = (self.gamma ** t_idx).unsqueeze(1).expand(n, B).reshape(n * B)
        e = (self.beta ** t_idx).unsqueeze(1).expand(n, B).reshape(n * B)

        return {
            'states': states, 'actions': actions, 'rewards': rewards, 'costs': costs,
            'next_states': next_states,
            'dones': torch.zeros(n * B, dtype=torch.float32, device=self.device),
            'nstep_reward': nstep_rew.reshape(n * B),
            'nstep_cost': nstep_cost.reshape(n * B),
            'boot_states': boot_states, 'boot_mask': boot_mask,
            'steps': steps, 'boot_steps': boot_steps,
            'budgets': budgets, 'd': d, 'e': e,
            's0': Smat[0],                                    # [B,sd]
            'disc_return': roll['disc_return'],               # [B] R
            'disc_cost': roll['disc_cost'],                   # [B] C
            'undisc_cost': roll['undisc_cost'],               # [B] Σc
        }

    # ============================================================ Critic 更新 (双 QR-TD) ============================================================
    def update_critic(self, batch):
        """reward critic (QR-TD on reward) + cost critic (QR-TD on cost), 合并一次 backward。"""
        s, a = batch['states'], batch['actions']
        boot_s, boot_mask = batch['boot_states'], batch['boot_mask']
        steps, boot_steps = batch['steps'], batch['boot_steps']

        with torch.no_grad():                                 # target 不建图
            a_boot = self._sample_actions(boot_s)             # a''~π(·|s_{t+N})
            # reward target
            psi_r_next = self.reward_target_critic(self._aug(boot_s, boot_steps), a_boot)
            y_r = batch['nstep_reward'].unsqueeze(1) + \
                (self.gamma ** self.n_step) * boot_mask.unsqueeze(1) * psi_r_next
            # cost target
            psi_c_next = self.cost_target_critic(self._aug(boot_s, boot_steps), a_boot)
            y_c = batch['nstep_cost'].unsqueeze(1) + \
                (self.cost_gamma ** self.n_step) * boot_mask.unsqueeze(1) * psi_c_next

        psi_r = self.reward_critic(self._aug(s, steps), a)    # [n·B, N] (建图)
        psi_c = self.cost_critic(self._aug(s, steps), a)      # [n·B, N] (建图)
        loss_r = self._quantile_huber_loss(psi_r, y_r)
        loss_c = self._quantile_huber_loss(psi_c, y_c)
        loss = loss_r + loss_c

        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.critic_grad_clip and self.critic_grad_clip > 0:
            nn.utils.clip_grad_norm_(
                list(self.reward_critic.parameters()) + list(self.cost_critic.parameters()),
                self.critic_grad_clip)
        self.critic_optimizer.step()
        return {'critic/reward_qr_loss': float(loss_r.item()),
                'critic/cost_qr_loss': float(loss_c.item())}

    def _quantile_huber_loss(self, psi, y):
        """Quantile Huber Loss ρ^κ_τ(u) (QR-DQN 核心, 与模板一致)。"""
        u = y.unsqueeze(1) - psi.unsqueeze(2)                 # [*,N,N] 配对 TD error
        abs_u = u.abs()
        huber = torch.where(abs_u <= self.huber_kappa,
                            0.5 * u.pow(2),
                            self.huber_kappa * (abs_u - 0.5 * self.huber_kappa))
        taus = self.taus.view(1, -1, 1)
        weight = (taus - (u.detach() < 0).float()).abs()
        return (weight * huber / self.huber_kappa).sum(dim=2).mean(dim=1).mean()

    # ============================================================ Actor 更新 (与模板逐式一致, 约束换 cost 上尾) ============================================================
    def update_actor(self, batch):
        """w = γ^t·Â_m - λ·β^t·Â_c;  Â_m 来自 reward critic 均值, Â_c 来自 cost critic 上尾 CDF。"""
        s, a, b = batch['states'], batch['actions'], batch['budgets']
        d, e = batch['d'], batch['e']
        steps = batch['steps']

        with torch.no_grad():                                 # 优势不建图
            psi_r = self.reward_critic(self._aug(s, steps), a)          # [n·B, N]
            q_m = psi_r.mean(dim=1)                                     # Q̂_m (reward 均值)
            psi_c = self.cost_critic(self._aug(s, steps), a)           # [n·B, N]
            psi_cdf = (psi_c >= b.unsqueeze(1)).float().mean(dim=1)    # Ψ̂(s,a,b) 上尾 CDF
            v_m, v_c = self._estimate_baselines(s, b, steps)          # V̂_m(reward), V̂_c(cost)
            if self.advantage_norm == 'qcpo':
                a_m = (q_m - v_m) / self.return_rms.std               # EMA σ_R 归一化
                a_c = (psi_cdf - v_c) / self.constraint_rms.std       # EMA σ_c 归一化
            else:
                a_m = self._maybe_norm(q_m - v_m)
                a_c = self._maybe_norm(psi_cdf - v_c)
            w = d * a_m - self.lambda_dual.detach() * e * a_c         # [n·B] 复合权重

        log_probs = self._compute_log_probs(s, a)             # 对 θ 可导
        actor_loss = -(log_probs * w).mean() \
            - self.entropy_coef * self._entropy(s).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.actor_grad_clip and self.actor_grad_clip > 0:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip)
        self.actor_optimizer.step()

        raw_adv_m = q_m - v_m
        raw_adv_c = psi_cdf - v_c
        return {'actor/loss': float(actor_loss.item()),
                'advantage/mean_adv_std': float(raw_adv_m.std(unbiased=False).item()),
                'advantage/risk_adv_std': float(raw_adv_c.std(unbiased=False).item()),
                'critic/q_mean': float(q_m.mean().item()),
                'constraint/psi_c_mean': float(psi_cdf.mean().item()),
                'actor/w_mean': float(w.mean().item()),
                'actor/w_std': float(w.std(unbiased=False).item())}

    # ============================================================ Dual 更新 (cost-critic P(C≥d) 驱动) ============================================================
    def update_dual(self, batch):
        """Ĝ = cost_critic 在 B 个 (s0, a0~π) 上估计的 P(C≥d) (上尾); λ ← [λ + ε_k(Ĝ-ω)]_+。"""
        with torch.no_grad():
            s0 = batch['s0']
            a0 = self._sample_actions(s0)
            psi0 = self.cost_critic(self._aug(s0, 0), a0)     # [B, N]
            p = float((psi0 >= self.cost_limit).float().mean(dim=1).mean().item())  # P(C≥d) 上尾
        self.last_dual_prob = p

        gap = torch.tensor([p - self.q_alpha], dtype=torch.float32, device=self.device)  # Ĝ-ω
        self.lambda_optimizer.zero_grad(set_to_none=True)
        (-self.lambda_dual * gap).backward()
        self.lambda_optimizer.step()
        with torch.no_grad():
            self.lambda_dual.clamp_(min=self.lambda_min, max=self.lambda_max)

    # ============================================================ 辅助 (与模板同语义) ============================================================
    def _aug(self, states, steps):
        """critic step 增广 (critic_step_feature=False 时原样; 截断 bootstrap → 默认关闭)。"""
        if not self.critic_step_feature:
            return states
        if not torch.is_tensor(steps):
            steps = torch.full((states.shape[0],), float(steps), device=states.device)
        sf = (steps.float() / self.n).reshape(-1, 1)
        return torch.cat([states, sf], dim=1)

    def _estimate_baselines(self, states, budgets, steps=None):
        """V̂_m(s)=E_a[mean ψ^r], V̂_c(s,b)=E_a[Ψ̂ 上尾] —— K 个动作样本近似。"""
        b_col = budgets.unsqueeze(1)
        s_aug = self._aug(states, steps) if steps is not None else states
        q_list, c_list = [], []
        for _ in range(self.num_action_samples):
            a = self._sample_actions(states)
            q_list.append(self.reward_critic(s_aug, a).mean(dim=1))            # reward 均值
            c_list.append((self.cost_critic(s_aug, a) >= b_col).float().mean(dim=1))  # cost 上尾 CDF
        return (torch.stack(q_list, dim=0).mean(dim=0),
                torch.stack(c_list, dim=0).mean(dim=0))

    def _maybe_norm(self, x, eps=1e-8):
        """'separate': 批内标准化; 'none': 原样。"""
        if self.advantage_norm != 'separate' or x.numel() <= 1:
            return x
        std = x.std(unbiased=False)
        if std.item() < eps:
            return x - x.mean()
        return (x - x.mean()) / (std + eps)

    def _ema_update(self, rms, batch_mean, batch_var):
        """整批矩 EMA 混入 RunningMeanStd。"""
        if not rms._initialized:
            rms.mean = batch_mean
            rms.var = max(batch_var, 1e-8)
            rms._initialized = True
            return
        a = rms.decay
        rms.mean = (1.0 - a) * rms.mean + a * batch_mean
        rms.var = (1.0 - a) * rms.var + a * batch_var

    def _update_norm_stats(self, batch):
        """qcpo 模式: 刷新 σ_R (reward 回报) 与 σ_c (cost 约束优势) 两个 EMA。"""
        z = batch['disc_return'].detach()                     # [B] reward 回报
        self._ema_update(self.return_rms,
                         float(z.mean().item()), float(z.var(unbiased=False).item()))
        with torch.no_grad():
            s, b = batch['states'], batch['budgets']
            a = batch['actions']
            steps = batch['steps']
            psi_cdf = (self.cost_critic(self._aug(s, steps), a) >= b.unsqueeze(1)).float().mean(dim=1)
            _, v_c = self._estimate_baselines(s, b, steps)
            adv_c = psi_cdf - v_c
        self._ema_update(self.constraint_rms,
                         float(adv_c.mean().item()), float(adv_c.var(unbiased=False).item()))

    def _soft_update_target(self):
        """Polyak: target ← (1-τ)·target + τ·online (两 critic 同步)。"""
        with torch.no_grad():
            for tp, op in zip(self.reward_target_critic.parameters(), self.reward_critic.parameters()):
                tp.data.lerp_(op.data, self.target_tau)
            for tp, op in zip(self.cost_target_critic.parameters(), self.cost_critic.parameters()):
                tp.data.lerp_(op.data, self.target_tau)

    def _initial_cdf_estimate(self, s0):
        """cost-critic 在 s0 上的 P(C≥d)/E[C]/std(C) 估计 (校准诊断)。"""
        with torch.no_grad():
            a0 = self._sample_actions(s0)
            psi = self.cost_critic(self._aug(s0, 0), a0)      # [B, N]
            self.last_cdf_initial = float(
                (psi >= self.cost_limit).float().mean(dim=1).mean())      # 上尾
            self.last_pred_cost_mean = float(psi.mean().item())
            self.last_pred_cost_std = float(psi.std(dim=1).mean().item())
        return self.last_cdf_initial

    # ============================================================ 日志 ============================================================
    def _log(self, it, batch, critic_info, actor_info):
        """基类统一键 (_log_core 下尾口径) + DQCAC 专属诊断键 (budget/cost-critic 校准/λ/norm)。"""
        R_np = batch['disc_return'].detach().cpu().numpy()
        Zc_np = batch['disc_cost'].detach().cpu().numpy()
        Cu_np = batch['undisc_cost'].detach().cpu().numpy()
        ghat = self._initial_cdf_estimate(batch['s0'])        # cost-critic 估 P(C≥d)=P(Z≤q)
        budgets_np = batch['budgets'].detach().cpu().numpy()
        empirical_prob = float(np.mean(Zc_np >= self.cost_limit))  # = P(Z≤q)

        extra = {
            'constraint/cdf_estimate_initial': ghat,          # Ĝ ≈ P(Z≤q|s0) (与 empirical 同口径)
            'constraint/cdf_calibration_error': abs(ghat - empirical_prob),
            'critic/pred_cost_mean': self.last_pred_cost_mean,
            'critic/pred_cost_std': self.last_pred_cost_std,
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

        self._log_core(it, R_np, Zc_np, Cu_np, extra=extra)  # 控制台两行由 _log_core 统一打

    # ============================================================ 总结接口 ============================================================
    def get_training_summary(self):
        """暴露最终约束指标。"""
        return {
            'lambda_final': float(self.lambda_dual.detach().item()),
            'empirical_prob': self.last_empirical_prob,
            'empirical_outage_prob': self.last_outage_prob,   # 兼容旧字段 (=empirical_prob)
            'cdf_estimate_initial': self.last_cdf_initial,
            'dual_prob': self.last_dual_prob,
            'beta': self.beta,
            'num_envs': self.num_envs,
            'num_iterations': self.num_iterations,
        }
