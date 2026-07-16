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
    Dual:      默认保留 cost-critic CDF + Adam；可选 empirical_pid 由最近完整轨迹的
               outage 或 Q_(1-ω)(C)-d 驱动积分项，并可用 sum normalization 防止 λ 吞没 reward。
    PPO 一致性: old log-prob 与 reward/cost advantage 在同一 rollout 的多个 actor epoch 中固定。
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
from collections import deque
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from utils import RecurrentActorValue, RunningMeanStd
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
        # nstep 是历史默认；mc 用完整 finite-horizon cost return 直接监督每个 quantile。
        # 后者只作为传播偏差消融，不会把 reward actor 或 reward critic 偷换成 MC。
        self.cost_target_mode = str(getattr(args, 'cost_target_mode', 'nstep')).lower()
        if self.cost_target_mode not in {'nstep', 'mc'}:
            raise ValueError("cost_target_mode must be 'nstep' or 'mc'")
        if self.cost_target_mode == 'mc' and not self.episodic:
            raise ValueError(
                "cost_target_mode='mc' requires episodic=True because a truncated "
                "continuing rollout is not a complete cost return")
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
        # 0 表示历史整批 QR loss；正数按 transition 分块累计等权梯度，只降低
        # [batch,N,N] pairwise TD-error 峰值显存，不改变 optimizer step 次数。
        self.critic_minibatch_size = max(
            0, int(getattr(args, 'critic_minibatch_size', 0)))

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
        self.log_std_min = float(getattr(args, 'log_std_min', -5.0))
        self.log_std_max = float(getattr(args, 'log_std_max', 2.0))

        # policy_arch=mlp_lstm 时，actor 与 reward V 共享一个已经和 QCPO_refs
        # 数值逐项对拍的 MLP+LSTM 骨干。分布 critic 仍保持 DQCAC 所需的
        # action-conditioned Z(s,a)，不能误换成 QCPO_refs 的 state-value cost head。
        self.policy_arch = str(getattr(args, 'policy_arch', 'mlp')).lower()
        if self.policy_arch not in {'mlp', 'mlp_lstm'}:
            raise ValueError("policy_arch must be 'mlp' or 'mlp_lstm'")
        self.recurrent_policy = self.policy_arch == 'mlp_lstm'
        self.recurrent_seq_len = max(1, int(getattr(args, 'recurrent_seq_len', 100)))
        self.recurrent_value_loss_coef = float(
            getattr(args, 'recurrent_value_loss_coef', 1.0))
        if self.recurrent_policy:
            if self.reward_actor_mode != 'gae_ppo':
                raise ValueError("DQCAC mlp_lstm currently requires reward_actor_mode=gae_ppo")
            if self.n % self.recurrent_seq_len != 0:
                raise ValueError("horizon must be divisible by recurrent_seq_len")
            recurrent_hidden = getattr(args, 'recurrent_hidden', [512, 512])
            if isinstance(recurrent_hidden, str):
                recurrent_hidden = [
                    int(x) for x in recurrent_hidden.split(',') if x.strip()]
            self.actor = RecurrentActorValue(
                observation_dim=self.state_dim + 1,
                action_dim=self.action_dim,
                hidden=recurrent_hidden,
                lstm_size=int(getattr(args, 'lstm_size', 512)),
                lstm_skip=bool(getattr(args, 'lstm_skip', True)),
                init_std=float(getattr(args, 'init_std', 1.0)),
                learn_std=bool(getattr(args, 'learn_std', True)),
                normalize_observation=self.normalize_observation,
                var_clip=self.obs_norm_var_clip).to(self.device)

        # dual 消融：旧 critic_adam 保持可复现；empirical_pid 用真实完成轨迹控制 λ。
        self.dual_update_mode = str(getattr(args, 'dual_update_mode', 'critic_adam')).lower()
        if self.dual_update_mode not in {'critic_adam', 'empirical_pid'}:
            raise ValueError("dual_update_mode must be 'critic_adam' or 'empirical_pid'")
        self.dual_pid_signal = str(getattr(args, 'dual_pid_signal', 'outage')).lower()
        if self.dual_pid_signal not in {'outage', 'cost_quantile'}:
            raise ValueError("dual_pid_signal must be 'outage' or 'cost_quantile'")
        self.pid_Ki = float(getattr(args, 'pid_Ki', 0.1))
        self.pid_Kp = float(getattr(args, 'pid_Kp', 0.0))
        if self.pid_Kp < 0.0:
            raise ValueError("pid_Kp must be non-negative")
        self.pid_window_episodes = max(1, int(getattr(args, 'pid_window_episodes', 100)))
        self.pid_cost_scale = float(getattr(args, 'pid_cost_scale', 10.0))
        # 默认 rho=1/deadband=0/delta_max=inf/reference=0 逐式复现旧 I 控制器。
        # 新实验显式打开 leak/deadband，并按新增 episode 数缩放控制器时间轴。
        self.pid_integral_leak = float(getattr(args, 'pid_integral_leak', 1.0))
        self.pid_deadband = float(getattr(args, 'pid_deadband', 0.0))
        self.pid_delta_max = float(getattr(args, 'pid_delta_max', float('inf')))
        self.pid_reference_episodes = float(getattr(args, 'pid_reference_episodes', 0.0))
        if not 0.0 < self.pid_integral_leak <= 1.0:
            raise ValueError("pid_integral_leak must be in (0, 1]")
        if self.pid_deadband < 0.0:
            raise ValueError("pid_deadband must be non-negative")
        if self.pid_delta_max <= 0.0:
            raise ValueError("pid_delta_max must be positive")
        if self.pid_reference_episodes < 0.0:
            raise ValueError("pid_reference_episodes must be non-negative")
        self.sum_norm = bool(getattr(args, 'sum_norm', False))

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
        if self.reward_actor_mode != 'distributional' and not self.recurrent_policy:
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
        self.empirical_cost_window = deque(maxlen=self.pid_window_episodes)
        self.pid_i = float(self.lambda_min)                    # QCPO_refs 默认 Kp=Kd=0，仅积分项
        self.last_dual_raw_prob = 0.0                          # 当前 rollout 的经验 outage
        self.last_dual_window_prob = 0.0                       # 最近窗口经验 outage
        self.last_dual_cost_quantile = float(self.cost_limit)  # 窗口 Q_(1-ω)(C)
        self.last_dual_prob_gap = 0.0                          # window outage - ω
        self.last_dual_quantile_gap = 0.0                      # Q_(1-ω)(C) - d
        self.last_dual_control_error = 0.0                     # deadband 前的原始控制误差
        self.last_dual_filtered_error = 0.0                    # deadband 后实际积分误差
        self.last_pid_episode_scale = 1.0                      # 本次新增 episode / reference
        self.last_pid_effective_leak = 1.0                     # rho ** episode_scale
        self.last_pid_delta = 0.0                              # clip 后积分增量（不含 leak）
        self.last_pid_actual_delta = 0.0                       # 最终 I_state_new-I_state_old
        self.last_pid_proportional = 0.0                       # Kp * filtered_error
        self.last_pid_output = 0.0                             # clip(I_state + P)

    # ============================================================ 主训练循环 (与模板一致) ============================================================
    def train(self):
        """每迭代: 采样(含cost) → (qcpo)刷 EMA → 内层双critic+actor → (外层)λ → 日志+调度。"""
        print(f"DQCACBetaGPU[CMDP]: env={self.env_name}, beta={self.beta}, omega={self.q_alpha}, "
              f"d(cost_limit)={self.cost_limit}, cost_gamma={self.cost_gamma}, episodic={self.episodic}, "
              f"step_feature={self.critic_step_feature}, N={self.num_quantiles}, B={self.num_envs}, T={self.n}, "
              f"cost_target={self.cost_target_mode}, "
              f"iters={self.num_iterations}, critic_updates/iter={self.updates_per_episode}, "
              f"actor_updates/iter={self.actor_updates_per_episode}, reward_actor={self.reward_actor_mode}, "
              f"arch={self.policy_arch}, dual={self.dual_update_mode}/{self.dual_pid_signal}, "
              f"sum_norm={self.sum_norm}, "
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

            # 经验 PID 不依赖尚未校准的 critic，并在 actor epochs 前更新，逐位对齐 QCPO_refs 时序。
            dual_updated = False
            if (not in_warmup and self.dual_update_mode == 'empirical_pid'
                    and it % self.outer_interval == 0):
                self.update_dual(batch)
                dual_updated = True

            # ===== 2. 内层更新 (双 critic + actor) =====
            critic_info, actor_info = {}, {}
            actor_updated = False
            for update_idx in range(self.updates_per_episode):
                critic_info = self.update_critic(batch)
                # 标量 reward value 在 warmup 中也训练；actor 仍由下面的 in_warmup 控制。
                # value 每个 epoch 拟合同一批冻结 λ-return，actor 复用冻结 advantage。
                if self.reward_actor_mode != 'distributional' and not self.recurrent_policy:
                    critic_info.update(self.update_reward_value(batch))
                if not in_warmup and update_idx < self.actor_updates_per_episode:
                    actor_info = self.update_actor(batch)
                    actor_updated = True                       # scheduler 只能跟随真实 optimizer.step()
                if self.learning_steps % self.target_update_interval == 0:
                    self._soft_update_target()
                self.learning_steps += 1

            # 旧 critic_adam 路径保留 actor 后更新时序，保证历史实验可复现。
            if (not in_warmup and self.dual_update_mode == 'critic_adam'
                    and it % self.outer_interval == 0):
                self.update_dual(batch)
                dual_updated = True                            # warmup 时不推进 λ 的学习率时间轴

            # recurrent policy 的 augmented-observation moments 必须在所有 PPO epoch
            # 完成后再合并；否则固定 old_logπ 与 current logπ 会使用不同输入变换。
            if self.recurrent_policy and self.normalize_observation:
                self.actor.update_obs_rms(batch['actor_obs'])

            # ===== 4. 日志 + 调度 =====
            self._log(it, batch, critic_info, actor_info)
            if actor_updated:
                self.actor_scheduler.step()                    # 每个有 actor update 的 rollout 推进一步
            if dual_updated and self.dual_update_mode == 'critic_adam':
                self.lambda_scheduler.step()                   # 严格位于 lambda_optimizer.step() 之后

    # ============================================================ 循环策略 rollout / BPTT 工具 ============================================================
    @staticmethod
    def _logp_from_params(actions, means, log_stds):
        """由循环策略输出计算对角高斯 logπ；支持任意 leading 维。"""
        std = torch.exp(log_stds)
        standardized = (actions - means) / (std + 1e-8)
        action_dim = actions.shape[-1]
        return -((log_stds + 0.5 * standardized.pow(2)).sum(dim=-1)
                 + 0.5 * action_dim * np.log(2.0 * np.pi))

    @staticmethod
    def _sample_from_params(means, log_stds):
        """从已带历史条件的高斯参数采样，避免错误地用零 hidden 重算动作。"""
        return means + torch.exp(log_stds) * torch.randn_like(means)

    def _transform_recurrent(self, tensor):
        """把时间主序 [T,B,*] 切成 QCPO_refs 相同的 [seq_len,new_B,*]。"""
        T, B = tensor.shape[:2]
        rest = tuple(tensor.shape[2:])
        new_B = T * B // self.recurrent_seq_len
        return (tensor.transpose(0, 1).reshape(
            new_B, self.recurrent_seq_len, *rest).transpose(0, 1).contiguous())

    def _sample_initial_actions(self, states):
        """在 episode 初始零历史处采样；仅供 s0 dual/CDF 查询，不能用于任意中间状态。"""
        if not self.recurrent_policy:
            return self._sample_actions(states)
        B = states.shape[0]
        prev_cost = torch.zeros(B, 1, device=states.device)
        prev_action = torch.zeros(B, self.action_dim, device=states.device)
        prev_reward = torch.zeros(B, device=states.device)
        h0, c0 = self.actor.initial_state(B, states.device)
        actor_obs = torch.cat([states, prev_cost], dim=1)
        means, log_stds, _value, _state = self.actor(
            actor_obs.unsqueeze(0), prev_action.unsqueeze(0),
            prev_reward.unsqueeze(0), (h0, c0))
        return self._sample_from_params(means[0], log_stds[0])

    def _rollout_core(self, keep_logp=False):
        """
        MLP 沿用共享基类；MLP+LSTM 显式保存每步历史输入与进入前 hidden。

        额外计算 terminal_action：DQCAC 的 N-step target 在 t+N=T 时仍可能
        bootstrap。该动作必须来自同一行为策略和完整历史，不能用零 hidden 近似。
        """
        if not self.recurrent_policy:
            return super()._rollout_core(keep_logp=keep_logp)

        n, B = self.n, self.num_envs
        state = self.vec_env.reset()
        prev_cost = torch.zeros(B, 1, device=self.device)
        prev_action = torch.zeros(B, self.action_dim, device=self.device)
        prev_reward = torch.zeros(B, device=self.device)
        hidden, cell = self.actor.initial_state(B, self.device)

        states, actions, rewards, costs = [], [], [], []
        actor_obs, prev_actions, prev_rewards = [], [], []
        hidden_in, cell_in, values, means_all, log_stds_all, log_probs = (
            [], [], [], [], [], [])
        disc_return = torch.zeros(B, device=self.device)
        disc_cost = torch.zeros(B, device=self.device)
        undisc_cost = torch.zeros(B, device=self.device)
        reward_discount, cost_discount = 1.0, 1.0
        next_state = state

        with torch.no_grad():
            for _t in range(n):
                augmented_obs = torch.cat([state, prev_cost], dim=1)
                hidden_in.append(hidden[0].clone())
                cell_in.append(cell[0].clone())
                means, log_stds, value, (next_hidden, next_cell) = self.actor(
                    augmented_obs.unsqueeze(0), prev_action.unsqueeze(0),
                    prev_reward.unsqueeze(0), (hidden, cell))
                means, log_stds, value = means[0], log_stds[0], value[0]
                action = self._sample_from_params(means, log_stds)
                if keep_logp:
                    log_probs.append(self._logp_from_params(action, means, log_stds))

                next_state, reward, cost, _done = self.vec_env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                costs.append(cost)
                actor_obs.append(augmented_obs)
                prev_actions.append(prev_action.clone())
                prev_rewards.append(prev_reward.clone())
                values.append(value)
                means_all.append(means)
                log_stds_all.append(log_stds)

                disc_return += reward_discount * reward
                disc_cost += cost_discount * cost
                undisc_cost += cost
                reward_discount *= self.gamma
                cost_discount *= self.cost_gamma

                prev_cost = cost.unsqueeze(1)
                prev_action, prev_reward = action, reward
                hidden, cell = next_hidden, next_cell
                state = next_state

            # s_T 的动作仅作为 continuing N-step bootstrap 使用，不与环境交互。
            terminal_obs = torch.cat([next_state, prev_cost], dim=1)
            terminal_mean, terminal_log_std, _terminal_value, _terminal_state = self.actor(
                terminal_obs.unsqueeze(0), prev_action.unsqueeze(0),
                prev_reward.unsqueeze(0), (hidden, cell))
            terminal_action = self._sample_from_params(
                terminal_mean[0], terminal_log_std[0])

        rollout = {
            'S': torch.stack(states),
            'A': torch.stack(actions),
            'R': torch.stack(rewards),
            'C': torch.stack(costs),
            'S2_last': next_state,
            'disc_return': disc_return,
            'disc_cost': disc_cost,
            'undisc_cost': undisc_cost,
            'actor_obs': torch.stack(actor_obs),
            'prev_action': torch.stack(prev_actions),
            'prev_reward': torch.stack(prev_rewards),
            'h0': torch.stack(hidden_in),
            'c0': torch.stack(cell_in),
            'actor_value': torch.stack(values),
            'actor_mean': torch.stack(means_all),
            'actor_log_std': torch.stack(log_stds_all),
            'terminal_action': terminal_action,
        }
        if keep_logp:
            rollout['logp'] = torch.stack(log_probs)
        return rollout

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

        # 完整 episode 的 cost return-to-go：G^c_t=c_t+gamma_c*G^c_{t+1}。
        # 只在 mc 消融中构造，避免默认 nstep 路径额外占用一个 [T,B] 张量。
        mc_cost = None
        if self.cost_target_mode == 'mc':
            mc_cost = torch.empty_like(Cmat)
            running_cost = torch.zeros(B, dtype=torch.float32, device=self.device)
            for t in range(n - 1, -1, -1):
                running_cost = Cmat[t] + self.cost_gamma * running_cost
                mc_cost[t] = running_cost
        t_ar = torch.arange(n, device=self.device)
        boot_idx = torch.clamp(t_ar + Ns, max=n)              # bootstrap 态索引 (可达 s_T=n)
        boot_states = S_ext[boot_idx].reshape(n * B, -1)
        boot_actions = None
        if self.recurrent_policy:
            # recurrent target 使用同一 on-policy rollout 在完整历史下采到的 a_{t+N}。
            # 末尾索引 n 对应 _rollout_core 单独生成但未执行的 terminal_action。
            action_ext = torch.cat([Amat, roll['terminal_action'].unsqueeze(0)], dim=0)
            boot_actions = action_ext[boot_idx].reshape(n * B, -1)
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
        if mc_cost is not None:
            batch['mc_cost'] = mc_cost.reshape(n * B).detach()
        if self.recurrent_policy:
            # 保留时间主序张量；actor update 再按 recurrent_seq_len 切块，保证每个
            # chunk 的初始 (h,c) 正是行为策略采样时进入该位置的状态。
            batch.update({
                'actor_obs': roll['actor_obs'],
                'prev_action': roll['prev_action'],
                'prev_reward': roll['prev_reward'],
                'h0': roll['h0'],
                'c0': roll['c0'],
                'actor_value': roll['actor_value'],
                'actor_mean': roll['actor_mean'],
                'actor_log_std': roll['actor_log_std'],
                'boot_actions': boot_actions.detach(),
            })
        return batch

    # ============================================================ Critic 更新 (双 QR-TD) ============================================================
    def update_critic(self, batch):
        """
        更新 reward/cost 两个 QR-TD critic；可选 transition chunking 降低 N² 峰值显存。

        chunk 路径对每块 mean loss 乘 `chunk_size/total_size` 后 backward，所有块
        共享一次 zero_grad/clip/optimizer.step。因此它与整批 mean loss 的梯度定义
        相同，不会因为块数增加而偷偷放大学习率。
        """
        states, actions = batch['states'], batch['actions']
        boot_states, boot_mask = batch['boot_states'], batch['boot_mask']
        steps, boot_steps = batch['steps'], batch['boot_steps']

        with torch.no_grad():                                 # target 分支不建立 autograd 图
            # MLP 可直接在 boot state 重采；recurrent 必须使用完整历史下保存的
            # on-policy boot action，否则这里会隐式把所有中间状态当作 episode 起点。
            boot_actions = (
                batch['boot_actions'] if self.recurrent_policy
                else self._sample_actions(boot_states))
            boot_inputs = self._aug(boot_states, boot_steps)

            # 两条 target 都完整保存为 [T*B,N]；真正的 N² 张量只在下面 chunk 内产生。
            reward_next = self.reward_target_critic(boot_inputs, boot_actions)
            reward_target = (
                batch['nstep_reward'].unsqueeze(1)
                + (self.gamma ** self.n_step)
                * boot_mask.unsqueeze(1) * reward_next)
            if self.cost_target_mode == 'nstep':
                cost_next = self.cost_target_critic(boot_inputs, boot_actions)
                cost_target = (
                    batch['nstep_cost'].unsqueeze(1)
                    + (self.cost_gamma ** self.n_step)
                    * boot_mask.unsqueeze(1) * cost_next)
            else:
                # 每个 transition 只有一个真实 MC realization；重复到 N 列保持
                # _quantile_huber_loss 对 target-sample 求和的历史 loss/梯度尺度不变。
                # 重复列不制造新信息，只让本消融无需同时重调 critic_lr/grad clip。
                cost_target = batch['mc_cost'].unsqueeze(1).expand(
                    -1, self.num_quantiles)

        state_inputs = self._aug(states, steps)
        total_size = int(states.shape[0])
        chunk_size = self.critic_minibatch_size
        use_chunks = 0 < chunk_size < total_size
        self.critic_optimizer.zero_grad(set_to_none=True)

        if not use_chunks:
            # 默认历史路径：构造完整 [T*B,N,N] pairwise error，一次 backward。
            reward_pred = self.reward_critic(state_inputs, actions)
            cost_pred = self.cost_critic(state_inputs, actions)
            reward_loss = self._quantile_huber_loss(reward_pred, reward_target)
            cost_loss = self._quantile_huber_loss(cost_pred, cost_target)
            (reward_loss + cost_loss).backward()
            reward_loss_value = float(reward_loss.item())
            cost_loss_value = float(cost_loss.item())
        else:
            # 顺序分块避免额外 permutation 张量并保持可复现；每块图在 backward 后释放。
            reward_loss_value, cost_loss_value = 0.0, 0.0
            for begin in range(0, total_size, chunk_size):
                finish = min(begin + chunk_size, total_size)
                weight = float(finish - begin) / float(total_size)
                reward_pred = self.reward_critic(
                    state_inputs[begin:finish], actions[begin:finish])
                cost_pred = self.cost_critic(
                    state_inputs[begin:finish], actions[begin:finish])
                reward_loss_chunk = self._quantile_huber_loss(
                    reward_pred, reward_target[begin:finish])
                cost_loss_chunk = self._quantile_huber_loss(
                    cost_pred, cost_target[begin:finish])
                (weight * (reward_loss_chunk + cost_loss_chunk)).backward()
                reward_loss_value += weight * float(reward_loss_chunk.item())
                cost_loss_value += weight * float(cost_loss_chunk.item())

        # 分开记录两个 critic 的裁剪前梯度范数，诊断 MC 大 target 是否让
        # joint clip 长期由 cost 分支主导。这里只读取 .grad，不建立二阶计算图。
        reward_parameters = list(self.reward_critic.parameters())
        cost_parameters = list(self.cost_critic.parameters())

        def gradient_norm(parameters):
            squared_norm = torch.zeros((), dtype=torch.float32, device=self.device)
            for parameter in parameters:
                if parameter.grad is not None:
                    squared_norm += parameter.grad.detach().float().pow(2).sum()
            return squared_norm.sqrt()

        reward_grad_norm = gradient_norm(reward_parameters)
        cost_grad_norm = gradient_norm(cost_parameters)
        joint_parameters = reward_parameters + cost_parameters
        joint_grad_norm = gradient_norm(joint_parameters)
        if self.critic_grad_clip and self.critic_grad_clip > 0:
            nn.utils.clip_grad_norm_(joint_parameters, self.critic_grad_clip)
        self.critic_optimizer.step()
        return {
            'critic/reward_qr_loss': reward_loss_value,
            'critic/cost_qr_loss': cost_loss_value,
            'critic/chunked_update': float(use_chunks),
            'critic/cost_target_mean': float(cost_target.mean().item()),
            'critic/cost_target_is_mc': float(self.cost_target_mode == 'mc'),
            'critic/reward_grad_norm': float(reward_grad_norm.item()),
            'critic/cost_grad_norm': float(cost_grad_norm.item()),
            'critic/joint_grad_norm': float(joint_grad_norm.item()),
            'critic/grad_clip_fraction': float(
                joint_grad_norm.item() > self.critic_grad_clip
                if self.critic_grad_clip and self.critic_grad_clip > 0 else 0.0),
        }

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
        if self.reward_value is None and not self.recurrent_policy:
            raise RuntimeError("reward GAE requested in distributional mode")

        n, B = self.n, self.num_envs
        states = batch['states']                              # [T·B, state_dim]，时间主序
        value_states = None if self.recurrent_policy else self._aug(
            states, batch['steps'])                           # MLP episodic 模式追加 t/T
        rewards = batch['rewards'].reshape(n, B)              # [T,B] 原始每步 reward

        with torch.no_grad():
            # recurrent value 是 rollout 时共享 actor/V 骨干的冻结输出；所有 PPO
            # epoch 共用它构造的 GAE target，不能在 value 更新后重新计算移动目标。
            values_old = (
                batch['actor_value'] if self.recurrent_policy
                else self.reward_value(value_states).reshape(n, B))
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

    # ============================================================ 循环 Actor + reward-V 联合更新 ============================================================
    def _update_recurrent_actor_value(self, batch):
        """
        用固定 behavior hidden/logπ/GAE target 执行一次 recurrent PPO 更新。

        reward-V 与 actor 共享 QCPO_refs 同形骨干，因此只做一次联合 backward；cost
        advantage 仍来自 DQCAC 的 action-conditioned distributional critic，并在首个
        actor epoch 后缓存，防止后续 critic epoch 移动 PPO 的监督目标。
        """
        transform = self._transform_recurrent
        n, B = self.n, self.num_envs

        # [T,B,*] 按每条轨迹切成 seq_len 块；h0/c0 取每块进入前行为状态。
        observations = transform(batch['actor_obs'])
        prev_actions = transform(batch['prev_action'])
        prev_rewards = transform(batch['prev_reward'])
        actions = transform(batch['actions'].reshape(n, B, self.action_dim))
        old_log_probs = transform(batch['old_log_probs'].reshape(n, B))
        reward_weight = transform(batch['_reward_advantage'].reshape(n, B))
        value_targets = transform(batch['_reward_value_targets'].reshape(n, B))
        hidden0 = transform(batch['h0'])[0].unsqueeze(0).contiguous()
        cell0 = transform(batch['c0'])[0].unsqueeze(0).contiguous()

        means, log_stds, value_pred, _final_state = self.actor(
            observations, prev_actions, prev_rewards, (hidden0, cell0))

        # cost advantage 固定在 behavior policy：实际动作和 K 个 baseline 动作均具有
        # rollout 的完整历史条件。不能对中间 state 调 _sample_initial_actions。
        with torch.no_grad():
            states = batch['states']
            budgets = batch['budgets']
            steps = batch['steps']
            if '_risk_weight' not in batch:
                state_inputs = self._aug(states, steps)
                psi_c = self.cost_critic(state_inputs, batch['actions'])
                psi_cdf = (psi_c >= budgets.unsqueeze(1)).float().mean(dim=1)

                behavior_mean = batch['actor_mean'].reshape(n * B, self.action_dim)
                behavior_log_std = batch['actor_log_std'].reshape(n * B, self.action_dim)
                baseline_cdfs = []
                for _sample_idx in range(self.num_action_samples):
                    baseline_action = self._sample_from_params(
                        behavior_mean, behavior_log_std)
                    baseline_psi = self.cost_critic(state_inputs, baseline_action)
                    baseline_cdfs.append(
                        (baseline_psi >= budgets.unsqueeze(1)).float().mean(dim=1))
                baseline_cdf = torch.stack(baseline_cdfs, dim=0).mean(dim=0)
                raw_adv_c = psi_cdf - baseline_cdf

                if self.advantage_norm == 'qcpo':
                    normalized_adv_c = raw_adv_c / self.constraint_rms.std
                else:
                    normalized_adv_c = self._maybe_norm(raw_adv_c)
                risk_weight_flat = batch['e'] * normalized_adv_c
                batch['_risk_weight'] = risk_weight_flat.detach()
                batch['_risk_advantage_raw'] = raw_adv_c.detach()
                batch['_risk_cdf'] = psi_cdf.detach()
            else:
                risk_weight_flat = batch['_risk_weight']
                raw_adv_c = batch['_risk_advantage_raw']
                psi_cdf = batch['_risk_cdf']
            risk_weight = transform(risk_weight_flat.reshape(n, B))

        # PPO 分母始终是 rollout 时保存的 logπ_old；每个 epoch 只重算分子。
        log_probs = self._logp_from_params(actions, means, log_stds)
        log_ratio = log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        if '_first_epoch_ratio_max_error' not in batch:
            # 首个 actor epoch 前 actor/RMS 均未改变；若 history、chunk h0 或行为
            # probability 接错，这个误差会立刻非零，是比最终 KL 更敏感的自检。
            batch['_first_epoch_ratio_max_error'] = float(
                (ratio.detach() - 1.0).abs().max().item())
        clipped_ratio = torch.clamp(
            ratio, 1.0 - self.ppo_ratio_clip, 1.0 + self.ppo_ratio_clip)

        # reward 最大化采用 pessimistic min；cost 风险要最小化，采用 conservative max。
        reward_surr = torch.minimum(
            ratio * reward_weight, clipped_ratio * reward_weight)
        risk_surr = torch.maximum(
            ratio * risk_weight, clipped_ratio * risk_weight)
        lagrange = self.lambda_dual.detach()
        normalizer = 1.0 + lagrange if self.sum_norm else torch.ones_like(lagrange)
        reward_coef = normalizer.reciprocal()
        risk_coef = lagrange / normalizer

        entropy = (0.5 * (1.0 + self._log_2pi) + log_stds).sum(dim=-1).mean()
        policy_loss = (
            -reward_coef * reward_surr.mean()
            + risk_coef * risk_surr.mean()
            - self.entropy_coef * entropy)
        value_error = value_pred - value_targets
        value_loss = 0.5 * value_error.pow(2).mean()
        total_loss = policy_loss + self.recurrent_value_loss_coef * value_loss

        self.actor_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if self.actor_grad_clip and self.actor_grad_clip > 0:
            grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.actor_grad_clip)
        else:
            grad_norm = torch.zeros((), device=self.device)
        self.actor_optimizer.step()
        with torch.no_grad():
            self.actor.log_std.clamp_(self.log_std_min, self.log_std_max)

        # value explained variance 与 MLP 路径同定义；这里的 grad_norm 是共享骨干
        # policy+value 联合梯度，另用 joint 键明确标注，避免误读为纯 value 梯度。
        target_var = value_targets.var(unbiased=False)
        explained_var = torch.where(
            target_var > 1e-8,
            1.0 - value_error.detach().var(unbiased=False) / target_var,
            torch.zeros_like(target_var))
        combined_weight = reward_coef * reward_weight - risk_coef * risk_weight
        return {
            'actor/loss': float(policy_loss.item()),
            'actor/grad_norm': float(grad_norm.item()),
            'actor/entropy': float(entropy.item()),
            'actor/reward_coefficient': float(reward_coef.item()),
            'actor/risk_coefficient': float(risk_coef.item()),
            'actor/w_mean': float(combined_weight.mean().item()),
            'actor/w_std': float(combined_weight.std(unbiased=False).item()),
            'actor/log_std_mean': float(self.actor.log_std.detach().mean().item()),
            'advantage/mean_adv_std': float(
                batch['_reward_advantage_raw'].std(unbiased=False).item()),
            'advantage/risk_adv_std': float(raw_adv_c.std(unbiased=False).item()),
            'constraint/psi_c_mean': float(psi_cdf.mean().item()),
            'reward_value/loss': float(value_loss.item()),
            'reward_value/explained_variance': float(explained_var.item()),
            'reward_value/joint_grad_norm': float(grad_norm.item()),
            'ppo/ratio_mean': float(ratio.detach().mean().item()),
            'ppo/ratio_std': float(ratio.detach().std(unbiased=False).item()),
            'ppo/first_epoch_ratio_max_error': float(
                batch['_first_epoch_ratio_max_error']),
            'ppo/clip_fraction': float(
                ((ratio.detach() - 1.0).abs() > self.ppo_ratio_clip).float().mean().item()),
            'ppo/approx_kl': float(
                ((ratio.detach() - 1.0) - log_ratio.detach()).mean().item()),
            'debug/reward_actor_is_gae': 1.0,
            'debug/reward_actor_is_ppo': 1.0,
            'debug/policy_is_recurrent': 1.0,
        }

    # ============================================================ Actor 更新 (reward 主干可消融) ============================================================
    def update_actor(self, batch):
        """按 reward_actor_mode 选择 distributional / GAE / GAE+PPO，cost 风险优势保持一致。"""
        if self.recurrent_policy:
            return self._update_recurrent_actor_value(batch)
        s, a, b = batch['states'], batch['actions'], batch['budgets']
        d, e = batch['d'], batch['e']
        steps = batch['steps']

        with torch.no_grad():
            # PPO 的行为分母与 advantage 都必须相对同一 rollout 固定；首次 actor epoch
            # 查询 cost critic 后缓存，后续 epochs 不再因 critic 更新而移动 risk target。
            if '_risk_weight' not in batch:
                psi_c = self.cost_critic(self._aug(s, steps), a)       # [T·B,N]
                psi_cdf = (psi_c >= b.unsqueeze(1)).float().mean(dim=1)
                v_m, v_c = self._estimate_baselines(
                    s, b, steps, need_reward=self.reward_actor_mode == 'distributional')
                raw_adv_c = psi_cdf - v_c
                if self.advantage_norm == 'qcpo':
                    a_c = raw_adv_c / self.constraint_rms.std          # EMA σ_c 归一化
                else:
                    a_c = self._maybe_norm(raw_adv_c)
                risk_weight = e * a_c
                batch['_risk_weight'] = risk_weight.detach()
                batch['_risk_advantage_raw'] = raw_adv_c.detach()
                batch['_risk_cdf'] = psi_cdf.detach()
            else:
                risk_weight = batch['_risk_weight']
                raw_adv_c = batch['_risk_advantage_raw']
                psi_cdf = batch['_risk_cdf']
                v_m = None

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
            normalizer = 1.0 + lagrange if self.sum_norm else torch.ones_like(lagrange)
            reward_coef = normalizer.reciprocal()
            risk_coef = lagrange / normalizer
            combined_weight = reward_coef * reward_weight - risk_coef * risk_weight

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
            actor_loss = -reward_coef * reward_surr.mean() + risk_coef * risk_surr.mean() \
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
            'actor/reward_coefficient': float(reward_coef.item()),
            'actor/risk_coefficient': float(risk_coef.item()),
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
    def _update_pid_integral(self, control_error, new_episodes):
        """执行默认兼容的 bounded leaky-I，并把 num_envs 映射到 episode 时间轴。

        reference_episodes=0 时 episode_scale 固定为 1，配合默认 leak=1、
        deadband=0、delta_max=inf，结果严格退化为旧式
        `pid_i <- clip(pid_i + Ki * error)`。正 reference 则令同样数量的新轨迹
        产生相同累计 leak/积分量，避免 B 改变后每 env-step 的 dual 增益漂移。
        """
        old_pid = float(self.pid_i)
        if self.pid_reference_episodes > 0.0:
            episode_scale = float(new_episodes) / self.pid_reference_episodes
        else:
            episode_scale = 1.0

        # 连续 deadband：阈值内为 0，阈值外减去边界，避免刚越界时控制量跳变。
        error = float(control_error)
        error_magnitude = max(0.0, abs(error) - self.pid_deadband)
        filtered_error = float(np.copysign(error_magnitude, error)) if error_magnitude else 0.0
        effective_leak = self.pid_integral_leak ** episode_scale
        # 对 leak<1 使用几何和，使一次 B=20 update 与两次 B=10 update 在
        # 常值误差下严格等价；rho=1 时连续极限就是 episode_scale。
        if self.pid_integral_leak < 1.0:
            integral_scale = (1.0 - effective_leak) / (1.0 - self.pid_integral_leak)
        else:
            integral_scale = episode_scale
        raw_delta = self.pid_Ki * filtered_error * integral_scale
        effective_delta_max = self.pid_delta_max * episode_scale
        bounded_delta = min(effective_delta_max, max(-effective_delta_max, raw_delta))

        proposal = effective_leak * old_pid + bounded_delta
        self.pid_i = min(self.lambda_max, max(self.lambda_min, proposal))
        self.last_dual_filtered_error = filtered_error
        self.last_pid_episode_scale = episode_scale
        self.last_pid_effective_leak = effective_leak
        self.last_pid_delta = bounded_delta
        self.last_pid_actual_delta = self.pid_i - old_pid

    def _pid_output_value(self):
        """返回 clip(I_state + Kp*filtered_error)，Kp=0 时严格等于旧 bounded-I。"""
        proportional = self.pid_Kp * self.last_dual_filtered_error
        output = min(
            self.lambda_max, max(self.lambda_min, self.pid_i + proportional))
        self.last_pid_proportional = proportional
        self.last_pid_output = output
        return output

    def update_dual(self, batch):
        """
        按 dual_update_mode 更新 λ：旧 critic Adam，或 QCPO_refs 风格的经验积分控制器。

        empirical_pid 默认使用最近窗口 outage gap；也可选择 cost_quantile 信号，后者按
        pid_cost_scale 缩放，逐式对应 QCPO_refs 的 Q_(1-ω)(C)/scale-d/scale。
        """
        if self.dual_update_mode == 'critic_adam':
            with torch.no_grad():
                s0 = batch['s0']
                a0 = self._sample_initial_actions(s0)
                psi0 = self.cost_critic(self._aug(s0, 0), a0)  # [B,N]
                p = float((psi0 >= self.cost_limit).float().mean(dim=1).mean().item())
            self.last_dual_prob = p
            self.last_dual_prob_gap = p - self.q_alpha
            self.last_dual_control_error = self.last_dual_prob_gap

            gap = torch.tensor(
                [self.last_dual_prob_gap], dtype=torch.float32, device=self.device)
            self.lambda_optimizer.zero_grad(set_to_none=True)
            (-self.lambda_dual * gap).backward()
            self.lambda_optimizer.step()
            with torch.no_grad():
                self.lambda_dual.clamp_(min=self.lambda_min, max=self.lambda_max)
            return

        # 当前 rollout 的真实 C 加入固定长度窗口；完全绕开尚未校准的 cost critic CDF。
        costs = batch['disc_cost'].detach().cpu().numpy().astype(np.float64)
        raw_prob = float(np.mean(costs >= self.cost_limit))
        self.empirical_cost_window.extend(costs.tolist())
        window = np.asarray(self.empirical_cost_window, dtype=np.float64)
        window_prob = float(np.mean(window >= self.cost_limit))
        sorted_costs = np.sort(window)
        q_ind = min(int(np.floor(len(sorted_costs) * (1.0 - self.q_alpha))),
                    len(sorted_costs) - 1)
        cost_quantile = float(sorted_costs[q_ind])

        self.last_dual_raw_prob = raw_prob
        self.last_dual_window_prob = window_prob
        self.last_dual_cost_quantile = cost_quantile
        self.last_dual_prob_gap = window_prob - self.q_alpha
        self.last_dual_quantile_gap = cost_quantile - self.cost_limit

        if self.dual_pid_signal == 'outage':
            control_error = self.last_dual_prob_gap
        else:
            if self.pid_cost_scale <= 0:
                raise ValueError("pid_cost_scale must be positive for cost_quantile PID")
            control_error = self.last_dual_quantile_gap / self.pid_cost_scale
        self.last_dual_control_error = float(control_error)

        # QCPO_refs 默认 Kp=Kd=0；这里的默认参数精确复现其 bounded I，
        # 显式参数可打开 leaky/deadband/episode-scaled 版本抑制窗口滞后。
        self._update_pid_integral(control_error, new_episodes=len(costs))
        pid_output = self._pid_output_value()
        with torch.no_grad():
            self.lambda_dual.fill_(pid_output)
        self.last_dual_prob = window_prob

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

    def _estimate_baselines(self, states, budgets, steps=None, need_reward=True,
                            policy_mean=None, policy_log_std=None):
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
            if self.recurrent_policy:
                # 中间时刻的策略分布必须由 rollout 的完整历史给出；这里使用固定
                # behavior 参数，使 cost advantage 在所有 PPO epoch 中保持一致。
                if policy_mean is None or policy_log_std is None:
                    raise RuntimeError("recurrent baseline requires history-conditioned policy params")
                a = self._sample_from_params(policy_mean, policy_log_std)
            else:
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
            policy_mean = (
                batch['actor_mean'].reshape(-1, self.action_dim)
                if self.recurrent_policy else None)
            policy_log_std = (
                batch['actor_log_std'].reshape(-1, self.action_dim)
                if self.recurrent_policy else None)
            _, v_c = self._estimate_baselines(
                s, b, steps, need_reward=False,
                policy_mean=policy_mean, policy_log_std=policy_log_std)
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
            a0 = self._sample_initial_actions(s0)
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
            'lambda/lr': float(self.pid_Ki if self.dual_update_mode == 'empirical_pid'
                               else self.lambda_scheduler.get_last_lr()[0]),
            'dual/mode_empirical_pid': float(self.dual_update_mode == 'empirical_pid'),
            'dual/pid_signal_is_quantile': float(self.dual_pid_signal == 'cost_quantile'),
            'dual/raw_empirical_prob': self.last_dual_raw_prob,
            'dual/window_empirical_prob': self.last_dual_window_prob,
            'dual/window_cost_quantile': self.last_dual_cost_quantile,
            'dual/prob_gap': self.last_dual_prob_gap,
            'dual/quantile_gap': self.last_dual_quantile_gap,
            'dual/control_error': self.last_dual_control_error,
            'dual/filtered_error': self.last_dual_filtered_error,
            'dual/pid_episode_scale': self.last_pid_episode_scale,
            'dual/pid_effective_leak': self.last_pid_effective_leak,
            'dual/pid_delta': self.last_pid_delta,
            'dual/pid_actual_delta': self.last_pid_actual_delta,
            'dual/pid_proportional': self.last_pid_proportional,
            'dual/pid_output': self.last_pid_output,
            'dual/pid_i': self.pid_i,
            'dual/window_size': float(len(self.empirical_cost_window)),
            'dual/sum_norm_enabled': float(self.sum_norm),
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
            # recurrent actor 的 RMS 覆盖 [state, previous_cost]；MLP 使用共享 raw-state RMS。
            obs_rms = self.actor.obs_rms if self.recurrent_policy else self.obs_normalizer
            obs_std = obs_rms.var.detach().clamp_min(0.0).sqrt()
            extra['obs_norm/count'] = float(obs_rms.count.item())
            extra['obs_norm/mean_abs'] = float(obs_rms.mean.detach().abs().mean().item())
            extra['obs_norm/std_min'] = float(obs_std.min().item())
            extra['obs_norm/std_median'] = float(obs_std.median().item())
            extra['obs_norm/std_max'] = float(obs_std.max().item())

        self._log_core(it, R_np, Zc_np, Cu_np, extra=extra)  # 控制台两行由 _log_core 统一打

    # ============================================================ 循环策略独立评估 ============================================================
    def evaluate_vec(self, vec_env, num_episodes, gamma, cost_gamma,
                     omega, cost_limit):
        """用完整 previous cost/action/reward 与 hidden 评估循环 DQCAC，并校准 cost critic。"""
        if not self.recurrent_policy:
            raise RuntimeError("evaluate_vec is only needed for mlp_lstm DQCAC")

        rounds = max(1, int(np.ceil(num_episodes / vec_env.B)))
        rewards_all, costs_all, undisc_costs_all = [], [], []
        initial_states, initial_actions = [], []
        with torch.no_grad():
            for _round_idx in range(rounds):
                B = vec_env.B
                state = vec_env.reset()
                prev_cost = torch.zeros(B, 1, device=self.device)
                prev_action = torch.zeros(B, self.action_dim, device=self.device)
                prev_reward = torch.zeros(B, device=self.device)
                hidden, cell = self.actor.initial_state(B, self.device)
                reward_return = torch.zeros(B, device=self.device)
                cost_return = torch.zeros(B, device=self.device)
                undisc_cost = torch.zeros(B, device=self.device)
                reward_discount, cost_discount = 1.0, 1.0

                for timestep in range(vec_env.n):
                    actor_obs = torch.cat([state, prev_cost], dim=1)
                    means, log_stds, _value, (hidden, cell) = self.actor(
                        actor_obs.unsqueeze(0), prev_action.unsqueeze(0),
                        prev_reward.unsqueeze(0), (hidden, cell))
                    action = self._sample_from_params(means[0], log_stds[0])
                    if timestep == 0:
                        initial_states.append(state.clone())
                        initial_actions.append(action.clone())
                    state, reward, cost, _done = vec_env.step(action)
                    reward_return += reward_discount * reward
                    cost_return += cost_discount * cost
                    undisc_cost += cost
                    reward_discount *= gamma
                    cost_discount *= cost_gamma
                    prev_cost = cost.unsqueeze(1)
                    prev_action, prev_reward = action, reward

                rewards_all.append(reward_return)
                costs_all.append(cost_return)
                undisc_costs_all.append(undisc_cost)

            # cost critic 校准使用同一评估批真实 s0 与循环策略零历史下的 a0。
            s0 = torch.cat(initial_states, dim=0)
            a0 = torch.cat(initial_actions, dim=0)
            psi0 = self.cost_critic(self._aug(s0, 0), a0)
            cost_cdf_initial = float(
                (psi0 >= float(cost_limit)).float().mean(dim=1).mean().item())
            pred_cost_mean = float(psi0.mean().item())
            pred_cost_std = float(psi0.std(dim=1).mean().item())

        reward_np = torch.cat(rewards_all).cpu().numpy().astype(np.float64)
        cost_np = torch.cat(costs_all).cpu().numpy().astype(np.float64)
        undisc_np = torch.cat(undisc_costs_all).cpu().numpy().astype(np.float64)
        transformed = -cost_np
        threshold = -float(cost_limit)
        empirical = float(np.mean(transformed <= threshold))
        quantile = float(np.percentile(transformed, omega * 100))
        return {
            'mean': float(reward_np.mean()),
            'reward_std': float(reward_np.std()),
            'empirical_prob': empirical,
            'quantile_return': quantile,
            'quantile_margin_to_threshold': quantile - threshold,
            'constraint_margin': omega - empirical,
            'cost_disc_mean': float(cost_np.mean()),
            'cost_undisc_mean': float(undisc_np.mean()),
            'outage_prob': empirical,
            'cost_quantile': float(np.percentile(cost_np, (1.0 - omega) * 100)),
            'num_episodes': int(reward_np.shape[0]),
            'cost_cdf_initial': cost_cdf_initial,
            'pred_cost_mean': pred_cost_mean,
            'pred_cost_std': pred_cost_std,
        }

    # ============================================================ 总结接口 ============================================================
    def get_training_summary(self):
        """暴露最终约束指标。"""
        return {
            'lambda_final': float(self.lambda_dual.detach().item()),
            'empirical_prob': self.last_empirical_prob,
            'empirical_outage_prob': self.last_outage_prob,   # 兼容旧字段 (=empirical_prob)
            'cdf_estimate_initial': self.last_cdf_initial,
            'dual_prob': self.last_dual_prob,
            'dual_update_mode': self.dual_update_mode,
            'dual_pid_signal': self.dual_pid_signal,
            'pid_i': self.pid_i,
            'dual_cost_quantile': self.last_dual_cost_quantile,
            'sum_norm': self.sum_norm,
            'beta': self.beta,
            'reward_actor_mode': self.reward_actor_mode,
            'policy_arch': self.policy_arch,
            'actor_updates_per_episode': self.actor_updates_per_episode,
            'num_envs': self.num_envs,
            'num_iterations': self.num_iterations,
        }
