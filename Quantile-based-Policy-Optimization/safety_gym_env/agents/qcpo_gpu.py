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

from utils import RecurrentActorValue, RunningMeanStd         # recurrent policy/V + EMA 回报统计
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

        # policy_arch=mlp_lstm 时，用与 QCPO_refs 数值对拍过的 policy/reward-V 骨干替换
        # 基类 MLP。替换发生在 optimizer 创建前，因此旧 MLP 不会残留在参数组中。
        self.policy_arch = str(getattr(args, 'policy_arch', 'mlp')).lower()
        if self.policy_arch not in {'mlp', 'mlp_lstm'}:
            raise ValueError("policy_arch must be 'mlp' or 'mlp_lstm'")
        self.recurrent_policy = self.policy_arch == 'mlp_lstm'
        self.recurrent_seq_len = max(1, int(getattr(args, 'recurrent_seq_len', 100)))
        if self.recurrent_policy:
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
        if self.reward_mode == 'gae' and not self.recurrent_policy:
            value_hidden = getattr(args, 'actor_hidden', [256, 256])
            if isinstance(value_hidden, str):
                value_hidden = [int(x) for x in value_hidden.split(',') if x.strip()]
            value_dim = self.state_dim + (1 if self.value_step_feature else 0)
            self.reward_value = ScalarValueCritic(value_dim, hidden=value_hidden).to(self.device)
            self.reward_value_optimizer = Adam(
                self.reward_value.parameters(), self.reward_value_lr, eps=1e-5)
        if self.recurrent_policy and self.reward_mode != 'gae':
            raise ValueError("mlp_lstm validation currently requires qcpo_reward_mode=gae")
        self.recurrent_value_loss_coef = float(
            getattr(args, 'recurrent_value_loss_coef', 1.0))

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
              f"reward_mode={self.reward_mode}, arch={self.policy_arch}, "
              f"obs_norm={self.normalize_observation}, opt={self.theta_optimizer_name}, "
              f"device={self.device}")

        # ===== return/observation RMS 预热 (只采样和刷统计，不更新策略) =====
        for _ in range(self.warmup_rms_iters):
            warmup_roll = self._rollout_core()
            self._update_rms(warmup_roll['disc_return'])
            self._update_obs_rms(warmup_roll)
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

            # ===== 3. 内层更新：MLP 分离 V/actor；recurrent 使用联合 policy/value BPTT =====
            actor_info, value_info = {}, {}
            for _ in range(self.updates_per_iteration):
                if self.recurrent_policy:
                    actor_info, value_info = self._update_recurrent_actor_value(
                        roll, C.detach(), gae_batch)
                else:
                    if gae_batch is not None:
                        value_info = self._update_reward_value(gae_batch)
                    actor_info = self._update_actor(
                        states, actions, R.detach(), C.detach(),
                        old_log_probs=old_log_probs,
                        reward_advantages=None if gae_batch is None
                        else gae_batch['advantages'])

            # ===== 4. 外层 λ 更新 (经验 outage 驱动) =====
            if it % self.outer_interval == 0:
                self._update_dual(C.detach())

            # actor 更新期间 observation moments 必须冻结，确保本 rollout 的输入变换与
            # 采样时完全相同；本批统计只供下一次 rollout 使用，避免制造隐式 off-policy。
            self._update_obs_rms(roll)

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
                obs_rms = self.actor.obs_rms if self.recurrent_policy else self.obs_normalizer
                obs_std = obs_rms.var.detach().clamp_min(0.0).sqrt()
                extra.update({
                    'obs_norm/count': float(obs_rms.count.item()),
                    'obs_norm/mean_abs': float(obs_rms.mean.detach().abs().mean().item()),
                    'obs_norm/std_min': float(obs_std.min().item()),
                    'obs_norm/std_median': float(obs_std.median().item()),
                    'obs_norm/std_max': float(obs_std.max().item()),
                })
            extra.update(value_info)
            extra.update(actor_info)
            self._log_core(it, R_np, Zc_np, Cu_np, extra=extra)
            self.scheduler.step()
            self.lambda_scheduler.step()

    # ============================================================ Recurrent rollout / BPTT ============================================================
    @staticmethod
    def _logp_from_params(actions, means, log_stds):
        """对角高斯 log likelihood；任意 leading 维，输出去掉 action 维。"""
        std = torch.exp(log_stds)
        z = (actions - means) / (std + 1e-8)
        return -((log_stds + 0.5 * z.pow(2)).sum(dim=-1)
                 + 0.5 * actions.shape[-1] * np.log(2.0 * np.pi))

    def _rollout_core(self, keep_logp=False):
        """MLP 沿用基类；MLP+LSTM 采集完整历史与每步进入前 hidden state。"""
        if not self.recurrent_policy:
            return super()._rollout_core(keep_logp=keep_logp)

        n, B = self.n, self.num_envs
        s = self.vec_env.reset()
        prev_cost = torch.zeros(B, 1, device=self.device)
        prev_action = torch.zeros(B, self.action_dim, device=self.device)
        prev_reward = torch.zeros(B, device=self.device)
        h, c = self.actor.initial_state(B, self.device)

        S, A, R, C = [], [], [], []
        OBS, PA, PR, H0, C0, VAL, LP = [], [], [], [], [], [], []
        disc_return = torch.zeros(B, device=self.device)
        disc_cost = torch.zeros(B, device=self.device)
        undisc_cost = torch.zeros(B, device=self.device)
        dr, dc = 1.0, 1.0
        s2 = s

        with torch.no_grad():
            for _t in range(n):
                actor_obs = torch.cat([s, prev_cost], dim=1)
                H0.append(h[0].clone())
                C0.append(c[0].clone())
                mean, log_std, value, (h_next, c_next) = self.actor(
                    actor_obs.unsqueeze(0), prev_action.unsqueeze(0),
                    prev_reward.unsqueeze(0), (h, c))
                mean, log_std, value = mean[0], log_std[0], value[0]
                action = mean + torch.exp(log_std) * torch.randn_like(mean)
                if keep_logp:
                    LP.append(self._logp_from_params(action, mean, log_std))

                s2, reward, cost, _done = self.vec_env.step(action)
                S.append(s); A.append(action); R.append(reward); C.append(cost)
                OBS.append(actor_obs); PA.append(prev_action.clone())
                PR.append(prev_reward.clone()); VAL.append(value)
                disc_return = disc_return + dr * reward
                disc_cost = disc_cost + dc * cost
                undisc_cost = undisc_cost + cost
                dr *= self.gamma
                dc *= self.cost_gamma

                prev_cost = cost.unsqueeze(1)
                prev_action, prev_reward = action, reward
                h, c = h_next, c_next
                s = s2

        out = {
            'S': torch.stack(S), 'A': torch.stack(A),
            'R': torch.stack(R), 'C': torch.stack(C),
            'S2_last': s2, 'disc_return': disc_return,
            'disc_cost': disc_cost, 'undisc_cost': undisc_cost,
            'actor_obs': torch.stack(OBS),
            'prev_action': torch.stack(PA), 'prev_reward': torch.stack(PR),
            'h0': torch.stack(H0), 'c0': torch.stack(C0),
            'actor_value': torch.stack(VAL),
        }
        if keep_logp:
            out['logp'] = torch.stack(LP)
        return out

    def _transform_recurrent(self, tensor):
        """[T,B,*] -> [seq_len,new_B,*]，逐轨迹按时间切块。"""
        T, B = tensor.shape[:2]
        rest = tuple(tensor.shape[2:])
        new_B = T * B // self.recurrent_seq_len
        return (tensor.transpose(0, 1).reshape(
            new_B, self.recurrent_seq_len, *rest).transpose(0, 1).contiguous())

    def _update_recurrent_actor_value(self, roll, trajectory_cost, gae_batch):
        """对一个 rollout 执行一次 recurrent PPO+value 联合更新。"""
        if gae_batch is None:
            raise RuntimeError("recurrent update requires frozen GAE data")
        tf = self._transform_recurrent
        obs = tf(roll['actor_obs'])
        prev_action = tf(roll['prev_action'])
        prev_reward = tf(roll['prev_reward'])
        actions = tf(roll['A'])
        old_log_probs = tf(roll['logp'])
        reward_weight = tf(gae_batch['advantages'].reshape(self.n, self.num_envs))
        value_targets = tf(gae_batch['targets'].reshape(self.n, self.num_envs))
        risk_episode = (trajectory_cost >= self.cost_limit).float()
        risk_weight = tf(risk_episode.unsqueeze(0).expand(self.n, self.num_envs))
        h0 = tf(roll['h0'])[0].unsqueeze(0).contiguous()
        c0 = tf(roll['c0'])[0].unsqueeze(0).contiguous()

        means, log_stds, value_pred, _ = self.actor(
            obs, prev_action, prev_reward, (h0, c0))
        log_probs = self._logp_from_params(actions, means, log_stds)
        log_ratio = log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(
            ratio, 1.0 - self.ppo_ratio_clip, 1.0 + self.ppo_ratio_clip)
        reward_surr = torch.minimum(
            ratio * reward_weight, clipped_ratio * reward_weight)
        risk_surr = torch.maximum(
            ratio * risk_weight, clipped_ratio * risk_weight)
        lagrange = self.lambda_dual.detach()
        policy_loss = -reward_surr.mean() + lagrange * risk_surr.mean()

        value_error = value_pred - value_targets
        value_loss = 0.5 * value_error.pow(2).mean()
        total_loss = policy_loss + self.recurrent_value_loss_coef * value_loss
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if self.actor_grad_clip and self.actor_grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.actor_grad_clip)
        else:
            grad_norm = torch.zeros((), device=self.device)
        self.optimizer.step()
        with torch.no_grad():
            self.actor.log_std.clamp_(self.log_std_min, self.log_std_max)

        target_var = value_targets.var(unbiased=False)
        explained_var = torch.where(
            target_var > 1e-8,
            1.0 - value_error.detach().var(unbiased=False) / target_var,
            torch.zeros_like(target_var))
        combined_weight = reward_weight - lagrange * risk_weight
        actor_info = {
            'actor/loss': float(policy_loss.item()),
            'actor/weight_mean': float(combined_weight.mean().item()),
            'actor/weight_std': float(combined_weight.std(unbiased=False).item()),
            'actor/reward_weight_std': float(reward_weight.std(unbiased=False).item()),
            'actor/risk_weight_std': float(risk_weight.std(unbiased=False).item()),
            'actor/grad_norm': float(grad_norm.item()),
            'actor/log_std_mean': float(self.actor.log_std.detach().mean().item()),
            'ppo/ratio_mean': float(ratio.detach().mean().item()),
            'ppo/ratio_std': float(ratio.detach().std(unbiased=False).item()),
            'ppo/clip_fraction': float(
                ((ratio.detach() - 1.0).abs() > self.ppo_ratio_clip).float().mean().item()),
            'ppo/approx_kl': float(
                ((ratio.detach() - 1.0) - log_ratio.detach()).mean().item()),
            'debug/reward_mode_is_gae': 1.0,
            'debug/policy_is_recurrent': 1.0,
        }
        value_info = {
            'reward_value/loss': float(value_loss.item()),
            'reward_value/explained_variance': float(explained_var.item()),
            'reward_value/grad_norm': float(grad_norm.item()),
            'advantage/mean_adv_std': float(
                gae_batch['raw_advantages'].std(unbiased=False).item()),
        }
        return actor_info, value_info

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
        if self.reward_value is None and not self.recurrent_policy:
            raise RuntimeError("reward GAE requested while qcpo_reward_mode=mc")

        n, B = self.n, self.num_envs
        states = roll['S'].reshape(n * B, -1)
        rewards = roll['R']                                    # [T,B]
        steps = torch.arange(n, device=self.device).unsqueeze(1).expand(n, B).reshape(n * B)
        value_inputs = None if self.recurrent_policy else self._value_inputs(states, steps)

        with torch.no_grad():
            values_old = roll['actor_value'] if self.recurrent_policy else                 self.reward_value(value_inputs).reshape(n, B)
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
            'value_inputs': None if value_inputs is None else value_inputs.detach(),
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

    def _update_obs_rms(self, roll):
        '''在 rollout 边界更新对应 policy 输入 RMS；训练 epochs 内保持冻结。'''
        if not self.normalize_observation:
            return
        if self.recurrent_policy:
            self.actor.update_obs_rms(roll['actor_obs'])
        else:
            self.obs_normalizer.update(roll['S'])

    def evaluate_vec(self, vec_env, num_episodes, gamma, cost_gamma,
                     omega, cost_limit):
        """recurrent policy 的独立 vector Monte-Carlo 评估；每轮 reset hidden/history。"""
        if not self.recurrent_policy:
            raise RuntimeError("evaluate_vec is only needed for mlp_lstm QCPO")

        episode_count = int(num_episodes)
        if episode_count <= 0:
            raise ValueError("num_episodes must be a positive integer")
        rounds = int(np.ceil(episode_count / vec_env.B))
        rewards_all, costs_all, undisc_costs_all = [], [], []
        with torch.no_grad():
            for _ in range(rounds):
                B = vec_env.B
                state = vec_env.reset()
                prev_cost = torch.zeros(B, 1, device=self.device)
                prev_action = torch.zeros(B, self.action_dim, device=self.device)
                prev_reward = torch.zeros(B, device=self.device)
                h, c = self.actor.initial_state(B, self.device)
                reward_return = torch.zeros(B, device=self.device)
                cost_return = torch.zeros(B, device=self.device)
                undisc_cost = torch.zeros(B, device=self.device)
                dr, dc = 1.0, 1.0

                for _t in range(vec_env.n):
                    actor_obs = torch.cat([state, prev_cost], dim=1)
                    mean, log_std, _value, (h, c) = self.actor(
                        actor_obs.unsqueeze(0), prev_action.unsqueeze(0),
                        prev_reward.unsqueeze(0), (h, c))
                    action = mean[0] + torch.exp(log_std[0]) * torch.randn_like(mean[0])
                    state, reward, cost, _done = vec_env.step(action)
                    reward_return += dr * reward
                    cost_return += dc * cost
                    undisc_cost += cost
                    dr *= gamma
                    dc *= cost_gamma
                    prev_cost = cost.unsqueeze(1)
                    prev_action, prev_reward = action, reward

                rewards_all.append(reward_return)
                costs_all.append(cost_return)
                undisc_costs_all.append(undisc_cost)

        # 保留完整episode动力学，只截断ceil(B)带来的尾部样本，保证精确评估数量。
        reward_np = torch.cat(rewards_all)[:episode_count].cpu().numpy().astype(np.float64)
        cost_np = torch.cat(costs_all)[:episode_count].cpu().numpy().astype(np.float64)
        undisc_np = torch.cat(undisc_costs_all)[:episode_count].cpu().numpy().astype(np.float64)
        transformed = -cost_np
        q = -float(cost_limit)
        empirical = float(np.mean(transformed <= q))
        q_est = float(np.percentile(transformed, omega * 100))
        return {
            'mean': float(reward_np.mean()),
            'reward_std': float(reward_np.std()),
            'empirical_prob': empirical,
            'quantile_return': q_est,
            'quantile_margin_to_threshold': q_est - q,
            'constraint_margin': omega - empirical,
            'cost_disc_mean': float(cost_np.mean()),
            'cost_undisc_mean': float(undisc_np.mean()),
            'outage_prob': empirical,
            'cost_quantile': float(np.percentile(cost_np, (1.0 - omega) * 100)),
            'num_episodes': int(reward_np.shape[0]),
            'cost_cdf_initial': None,
            'pred_cost_mean': None,
            'pred_cost_std': None,
        }

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
