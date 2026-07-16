# -*- coding: utf-8 -*-
"""
DQCACBetaGPU (safety_gym_env · CMDP 版) —— 用户 DQC-AC-β 迁移到 safety-gym outage 约束环境。

算法逐式对齐 portfolio_env_inf/agents/dqc_ac_beta_gpu.py (已验证的 per-transition DQCAC),
核心三机制不变, 仅按 CMDP【双回报流】扩展 + 约束翻为【上尾 cost】:
    优化问题:  max_θ E[R]  s.t.  P_θ(C ≥ d) ≤ ω
    Critic:    reward/cost 两个 QR 分布式 critic (QR N-step TD + target 软更新):
                 reward_critic ψ^r(s,a) → 默认 reward 均值优势 Q̂_m=mean(ψ^r)
                 cost_critic   ψ^c(s,a) → 约束上尾 CDF Ψ̂(s,a,b)=(1/N)Σ𝟙{ψ^c_i ≥ b}
    Actor:     默认 distributional: w = γ^t·(Q̂_m-V̂_m) - λ·β^t·(Ψ̂-V̂_c)
               可选 gae: scalar V_r(s,t)+冻结 GAE λ-return，替换低信噪比 reward Q 优势
               可选 gae_ppo: 在 GAE 上再使用 old-logπ ratio 与 PPO clip 多 epoch 更新
               cost budget 始终递推 b_0=d, b_{t+1}=(b_t-c_t)/γc
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
from .common import DistributionalCritic, ScalarValueCritic, lr_lambda, indicator_ge


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
        # critic/value 可复用同一 rollout 多次；非 PPO actor 必须保持 1 次严格 on-policy 更新。
        self.actor_updates_per_episode = max(
            1, int(getattr(args, 'actor_updates_per_episode', 1)))
        if self.actor_updates_per_episode > self.updates_per_episode:
            raise ValueError("actor_updates_per_episode cannot exceed updates_per_episode")
        self.advantage_norm = getattr(args, 'advantage_norm', 'qcpo')
        self.entropy_coef = getattr(args, 'entropy_coef', 0.0)
        self.lambda_max = getattr(args, 'lambda_max', 50.0)
        self.lambda_min = getattr(args, 'lambda_min', 0.0)
        self.critic_grad_clip = getattr(args, 'critic_grad_clip', 10.0)
        self.actor_grad_clip = getattr(args, 'actor_grad_clip', 100.0)

        # reward actor 主干做成显式消融开关，默认 distributional 完全复现旧实现。
        # - distributional: Q_r(s,a)-E_a Q_r(s,a)，即排查前 DQCACBeta；
        # - gae:            标量 V_r(s)+GAE，但仍用单次 logπ policy-gradient；
        # - gae_ppo:        同一 GAE，再加 old-logπ ratio 与 PPO clip 多 epoch 更新。
        self.reward_actor_mode = str(getattr(args, 'reward_actor_mode', 'distributional')).lower()
        valid_reward_modes = {'distributional', 'gae', 'gae_ppo'}
        if self.reward_actor_mode != 'gae_ppo' and self.actor_updates_per_episode > 1:
            raise ValueError("multiple actor updates on one rollout require gae_ppo importance ratio + clip")
        if self.normalize_observation and self.reward_actor_mode != 'gae_ppo':
            raise ValueError(
                "post-rollout observation normalization requires gae_ppo importance correction")
        if self.reward_actor_mode not in valid_reward_modes:
            raise ValueError(f"reward_actor_mode must be one of {sorted(valid_reward_modes)}, "
                             f"got {self.reward_actor_mode!r}")
        self.gae_lambda = float(getattr(args, 'gae_lambda', 0.97))
        self.reward_advantage_norm = bool(getattr(args, 'reward_advantage_norm', False))
        self.ppo_ratio_clip = float(getattr(args, 'ppo_ratio_clip', 0.1))
        self.reward_value_lr = float(getattr(args, 'reward_value_lr', 3e-4))
        self.reward_value_grad_clip = float(getattr(args, 'reward_value_grad_clip', 10.0))

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

        # GAE/PPO 模式额外维护 V_r(s)。cost critic、budget 与 dual 均保持原实现，
        # 因此短实验只替换 reward advantage 来源，不偷偷改风险约束链路。
        self.reward_value = None
        self.reward_value_optimizer = None
        if self.reward_actor_mode != 'distributional':
            self.reward_value = ScalarValueCritic(cdim, hidden=list(hidden)).to(self.device)
            self.reward_value_optimizer = Adam(
                self.reward_value.parameters(), self.reward_value_lr, eps=1e-5)

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
              f"iters={self.num_iterations}, critic_updates/iter={self.updates_per_episode}, "
              f"actor_updates/iter={self.actor_updates_per_episode}, reward_actor={self.reward_actor_mode}, "
              f"device={self.device}")

        for it in range(self.num_iterations):
            # ===== 1. 采样 + DQCAC 专属后处理 (cost budget / n-step / d,e) =====
            batch = self._rollout_vec()

            # ===== 1b. qcpo: 刷新 EMA 归一化器 (σ_R / σ_c) =====
            if self.advantage_norm == 'qcpo':
                self._update_norm_stats(batch)
            # GAE 与 value target 每个 rollout 固定一次，供后续多个 epoch 共同使用。
            if self.reward_actor_mode != 'distributional':
                self._prepare_reward_gae(batch)
            # obs norm 首批只建立 moments；随后 PPO ratio 才比较同一批的 old/current policy。
            norm_warmup = self.obs_norm_warmup_iters if self.normalize_observation else 0
            in_warmup = it < max(self.warmup_iters, norm_warmup)

            # ===== 2. 内层更新 (双 critic + actor) =====
            critic_info, actor_info = {}, {}
            actor_updated = False
            for update_idx in range(self.updates_per_episode):
                critic_info = self.update_critic(batch)
                # 标量 reward value 在 warmup 中也训练；actor 仍由下面的 in_warmup 控制。
                # value 每个 epoch 拟合同一批冻结 λ-return，actor 复用冻结 advantage。
                if self.reward_actor_mode != 'distributional':
                    critic_info.update(self.update_reward_value(batch))
                if not in_warmup and update_idx < self.actor_updates_per_episode:
                    actor_info = self.update_actor(batch)
                    actor_updated = True                       # scheduler 只能跟随真实 optimizer.step()
                if self.learning_steps % self.target_update_interval == 0:
                    self._soft_update_target()
                self.learning_steps += 1

            # ===== 3. 外层 λ 更新 (cost-critic 在 (s0,a0) 估 P(C≥d) 驱动) =====
            dual_updated = False
            if not in_warmup and it % self.outer_interval == 0:
                self.update_dual(batch)
                dual_updated = True                            # warmup 时不推进 λ 的学习率时间轴

            # ===== 4. 日志 + 调度 =====
            self._log(it, batch, critic_info, actor_info)
            if actor_updated:
                self.actor_scheduler.step()                    # 每个有 actor update 的 rollout 推进一步
            if dual_updated:
                self.lambda_scheduler.step()                   # 严格位于 lambda_optimizer.step() 之后

    # ============================================================ 采样 + 后处理 (对齐模板, budget 换 cost) ============================================================
    def _rollout_vec(self):
        """
        基类 _rollout_core 采 B 条轨迹 (含 cost 流) 后, 补齐 DQCAC 批量字段:
            cost budget: b_0=d, b_{t+1}=(b_t - c_t)/γc  —— 从 cost 矩阵 C 后处理递推 (对应论文 remain cost)
            reward/cost 各自的 n-step TD 件 (continuing: 截断恒 bootstrap, dones≡0)
            d=γ^t (reward 折扣), e=β^t (Abel 风险折扣)
        """
        n, B = self.n, self.num_envs
        # PPO 必须保存采样策略的 logπ_old；其它模式不保留，节省一个 [T,B] 张量。
        keep_logp = self.reward_actor_mode == 'gae_ppo'
        roll = self._rollout_core(keep_logp=keep_logp)        # S,A,R,C,S2_last,(可选 logp)
        Smat, Amat, Rmat, Cmat = roll['S'], roll['A'], roll['R'], roll['C']
        if self.normalize_observation:
            # 采样期间 moments 冻结；整段完成后一次合并，随后 actor/value/critic 共用新统计。
            self.obs_normalizer.update(Smat)

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

        batch = {
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
        if keep_logp:
            batch['old_log_probs'] = roll['logp'].reshape(n * B).detach()  # rollout 固定行为策略
        return batch

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

    # ============================================================ 可选 reward V+GAE ============================================================
    def _prepare_reward_gae(self, batch):
        """
        用 rollout 时的 V_r(s) 计算一次固定 GAE advantage/value target。

        当前论文环境每段 T=1000 就是完整 episode，QCPO_refs 在最后一步设置 done=1，
        所以这里末步 next_value=0；中间步仍用 V(s_{t+1})。target 在 PPO/GAE 的
        多个 epoch 之间保持冻结，避免 value 网络每更新一次就移动一次监督目标。
        """
        if self.reward_value is None:
            raise RuntimeError("reward GAE requested in distributional mode")

        n, B = self.n, self.num_envs
        states = batch['states']                              # [T·B, state_dim]，时间主序
        value_states = self._aug(states, batch['steps'])         # episodic 模式追加 t/T
        rewards = batch['rewards'].reshape(n, B)              # [T,B] 原始每步 reward

        with torch.no_grad():
            values_old = self.reward_value(value_states).reshape(n, B)
            # episode 最后一步不 bootstrap；其它位置的 next value 来自同一轨迹下一状态。
            next_values = torch.cat([values_old[1:], torch.zeros_like(values_old[:1])], dim=0)
            deltas = rewards + self.gamma * next_values - values_old

            gae = torch.zeros(B, dtype=torch.float32, device=self.device)
            advantages = torch.empty_like(deltas)
            for t in range(n - 1, -1, -1):
                gae = deltas[t] + self.gamma * self.gae_lambda * gae
                advantages[t] = gae
            returns = advantages + values_old

            raw_advantages = advantages.reshape(n * B)
            actor_advantages = raw_advantages
            if self.reward_advantage_norm:
                adv_mean = raw_advantages.mean()
                adv_std = raw_advantages.std(unbiased=False).clamp_min(1e-8)
                actor_advantages = (raw_advantages - adv_mean) / adv_std

            batch['_reward_advantage_raw'] = raw_advantages.detach()
            batch['_reward_advantage'] = actor_advantages.detach()
            batch['_reward_value_targets'] = returns.reshape(n * B).detach()

    def update_reward_value(self, batch):
        """对当前 V_r(s) 拟合本 rollout 冻结的 GAE λ-return target。"""
        if self.reward_value is None or self.reward_value_optimizer is None:
            raise RuntimeError("reward value update requested in distributional mode")
        if '_reward_value_targets' not in batch:
            raise RuntimeError("call _prepare_reward_gae(batch) before value epochs")

        states = batch['states']
        value_states = self._aug(states, batch['steps'])
        value_targets = batch['_reward_value_targets']
        value_pred = self.reward_value(value_states)
        value_error = value_pred - value_targets
        value_loss = 0.5 * value_error.pow(2).mean()

        self.reward_value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        if self.reward_value_grad_clip > 0:
            value_grad_norm = nn.utils.clip_grad_norm_(
                self.reward_value.parameters(), self.reward_value_grad_clip)
        else:
            value_grad_norm = torch.zeros((), device=self.device)
        self.reward_value_optimizer.step()

        # explained variance=1-Var(target-pred)/Var(target)；target 近常数时定义为 0。
        target_var = value_targets.var(unbiased=False)
        explained_var = torch.where(
            target_var > 1e-8,
            1.0 - value_error.detach().var(unbiased=False) / target_var,
            torch.zeros_like(target_var))
        return {
            'reward_value/loss': float(value_loss.item()),
            'reward_value/explained_variance': float(explained_var.item()),
            'reward_value/grad_norm': float(value_grad_norm.item()),
            'reward_value/pred_mean': float(value_pred.detach().mean().item()),
            'reward_value/target_mean': float(value_targets.mean().item()),
        }

    # ============================================================ Actor 更新 (reward 主干可消融) ============================================================
    def update_actor(self, batch):
        """按 reward_actor_mode 选择 distributional / GAE / GAE+PPO，cost 风险优势保持一致。"""
        s, a, b = batch['states'], batch['actions'], batch['budgets']
        d, e = batch['d'], batch['e']
        steps = batch['steps']

        with torch.no_grad():
            # cost 优势在三种 reward 模式下完全相同，仍查询 distributional cost critic。
            psi_c = self.cost_critic(self._aug(s, steps), a)           # [T·B,N]
            psi_cdf = (psi_c >= b.unsqueeze(1)).float().mean(dim=1)    # Ψ̂(s,a,b) 上尾 CDF
            v_m, v_c = self._estimate_baselines(
                s, b, steps, need_reward=self.reward_actor_mode == 'distributional')
            raw_adv_c = psi_cdf - v_c
            if self.advantage_norm == 'qcpo':
                a_c = raw_adv_c / self.constraint_rms.std              # EMA σ_c 归一化
            else:
                a_c = self._maybe_norm(raw_adv_c)
            risk_weight = e * a_c

            if self.reward_actor_mode == 'distributional':
                psi_r = self.reward_critic(self._aug(s, steps), a)     # [T·B,N]
                q_m = psi_r.mean(dim=1)                                # Q̂_m (reward 均值)
                raw_adv_m = q_m - v_m
                if self.advantage_norm == 'qcpo':
                    reward_weight = d * raw_adv_m / self.return_rms.std
                else:
                    reward_weight = d * self._maybe_norm(raw_adv_m)
            else:
                # 标准 episodic GAE 已通过递推包含 γ/λ，不再额外乘 γ^t；与 QCPO_refs 一致。
                raw_adv_m = batch['_reward_advantage_raw']
                reward_weight = batch['_reward_advantage']

            lagrange = self.lambda_dual.detach()
            combined_weight = reward_weight - lagrange * risk_weight

        log_probs = self._compute_log_probs(s, a)             # 当前策略 logπθ(a|s)，对 θ 可导
        entropy = self._entropy(s).mean()
        ppo_info = {}
        if self.reward_actor_mode == 'gae_ppo':
            old_log_probs = batch['old_log_probs']            # rollout 时冻结的行为策略 logπ_old
            log_ratio = log_probs - old_log_probs
            ratio = torch.exp(log_ratio)
            clipped_ratio = torch.clamp(
                ratio, 1.0 - self.ppo_ratio_clip, 1.0 + self.ppo_ratio_clip)

            # reward 用 PPO pessimistic min；cost 是要最小化的坏事件，故用 conservative max。
            reward_surr = torch.minimum(ratio * reward_weight, clipped_ratio * reward_weight)
            risk_surr = torch.maximum(ratio * risk_weight, clipped_ratio * risk_weight)
            actor_loss = -reward_surr.mean() + lagrange * risk_surr.mean() \
                - self.entropy_coef * entropy

            # approx_kl 采用 Schulman 常用近似 (ratio-1)-log_ratio。
            ppo_info = {
                'ppo/ratio_mean': float(ratio.detach().mean().item()),
                'ppo/ratio_std': float(ratio.detach().std(unbiased=False).item()),
                'ppo/clip_fraction': float(
                    ((ratio.detach() - 1.0).abs() > self.ppo_ratio_clip).float().mean().item()),
                'ppo/approx_kl': float(((ratio.detach() - 1.0) - log_ratio.detach()).mean().item()),
            }
        else:
            actor_loss = -(log_probs * combined_weight).mean() - self.entropy_coef * entropy

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.actor_grad_clip and self.actor_grad_clip > 0:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip)
        else:
            actor_grad_norm = torch.zeros((), device=self.device)
        self.actor_optimizer.step()

        actor_info = {
            'actor/loss': float(actor_loss.item()),
            'actor/grad_norm': float(actor_grad_norm.item()),
            'actor/entropy': float(entropy.item()),
            'advantage/mean_adv_std': float(raw_adv_m.std(unbiased=False).item()),
            'advantage/risk_adv_std': float(raw_adv_c.std(unbiased=False).item()),
            'constraint/psi_c_mean': float(psi_cdf.mean().item()),
            'actor/w_mean': float(combined_weight.mean().item()),
            'actor/w_std': float(combined_weight.std(unbiased=False).item()),
            'debug/reward_actor_is_gae': float(self.reward_actor_mode != 'distributional'),
            'debug/reward_actor_is_ppo': float(self.reward_actor_mode == 'gae_ppo'),
        }
        if self.reward_actor_mode == 'distributional':
            actor_info['critic/q_mean'] = float(q_m.mean().item())
        actor_info.update(ppo_info)
        return actor_info

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
        """先共享观测归一化，再按需追加 critic/value 的 t/T step feature。"""
        states = self._normalize_states(states)
        if not self.critic_step_feature:
            return states
        if not torch.is_tensor(steps):
            steps = torch.full((states.shape[0],), float(steps), device=states.device)
        sf = (steps.float() / self.n).reshape(-1, 1)
        return torch.cat([states, sf], dim=1)

    def _estimate_baselines(self, states, budgets, steps=None, need_reward=True):
        """
        用 K 个策略动作近似 V_c(s,b)，仅在 distributional reward 模式下同时近似 V_m(s)。

        GAE actor 已由独立 V_r(s) 提供 reward baseline；跳过无用的 reward critic K 次前向
        不改变 cost advantage 数值，可显著减少长 rollout 的 GPU 计算。
        """
        b_col = budgets.unsqueeze(1)
        # 所有 critic 前向都必须与 actor 共享 observation 统计；steps=None 仅表示
        # 不追加 t/T，不能跳过归一化，否则该兼容分支会混用两套输入尺度。
        s_aug = self._aug(states, steps) if steps is not None else self._normalize_states(states)
        q_list, c_list = [], []
        for _ in range(self.num_action_samples):
            a = self._sample_actions(states)
            if need_reward:
                q_list.append(self.reward_critic(s_aug, a).mean(dim=1))          # reward 均值
            c_list.append((self.cost_critic(s_aug, a) >= b_col).float().mean(dim=1))
        reward_baseline = torch.stack(q_list, dim=0).mean(dim=0) if need_reward else None
        cost_baseline = torch.stack(c_list, dim=0).mean(dim=0)
        return reward_baseline, cost_baseline

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
            _, v_c = self._estimate_baselines(s, b, steps, need_reward=False)
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
            'training/actor_updates_per_iteration': self.actor_updates_per_episode,
        }
        extra.update(critic_info)
        extra.update(actor_info)
        if self.advantage_norm == 'qcpo':
            extra['norm/return_sigma_ema'] = float(self.return_rms.std)
            extra['norm/constraint_sigma_ema'] = float(self.constraint_rms.std)
        if self.normalize_observation:
            obs_std = self.obs_normalizer.var.detach().clamp_min(0.0).sqrt()
            extra['obs_norm/count'] = float(self.obs_normalizer.count.item())
            extra['obs_norm/mean_abs'] = float(self.obs_normalizer.mean.detach().abs().mean().item())
            extra['obs_norm/std_min'] = float(obs_std.min().item())
            extra['obs_norm/std_median'] = float(obs_std.median().item())
            extra['obs_norm/std_max'] = float(obs_std.max().item())

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
            'reward_actor_mode': self.reward_actor_mode,
            'actor_updates_per_episode': self.actor_updates_per_episode,
            'num_envs': self.num_envs,
            'num_iterations': self.num_iterations,
        }
