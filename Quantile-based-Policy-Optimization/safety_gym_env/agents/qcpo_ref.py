# -*- coding: utf-8 -*-
"""
QCPORefGPU —— NIPS'22 QCPO 参考算法【核心移植版】(基线, 与用户算法同环境对比)。

来源: QCPO_nips_ref/qcpo/qcpo.py + qcpo_agent.py (wyjung0625/QCPO, rlpyt-based)。
移植原则 (用户决策 2026-07-14): 算法数学【逐式保留】——
    - PPO ratio-clip 策略优化 (epochs×minibatch, recurrent 按轨迹段切 B);
    - PID-Lagrangian: pid_i ← [pid_i + Ki·(Q̂_{1-ω}(ep_cost) − d)]_+, sum_norm 组合
      L = (J_r + λ·J_c)/(1+λ);
    - cost 分位数 critic c_dist=exp(linear) (n_quantile=25) + quantile huber loss;
    - Weibull 尾部模型 (α,β 头) + weibull_tail_loss + LDP 概率比 compute_prob_ratio
      加权的分位数 GAE cost 优势 (取 target_outage 分位那一列);
    - LSTM 策略 (skip 连接, 输入含 prev_action/prev_reward), obs 归一化 (running m/v);
    - obs 追加 prev_cost (论文 SafetyGymEnvWrapper obs_prev_cost=True 行为)。
只替换 rlpyt 管线: CpuSampler/runner → 本目录 SafetyVecEnv(MP) 时间主序 [T,B] rollout;
minibatch 的 new_T 切段 (transform_sample) 逐式复刻, 段首 LSTM 状态取自采样时记录。

约束口径 (与论文一致): P(未折扣 episode cost Σc ≥ d) ≤ ω, d=cost_limit (raw),
算法内部 cost/d 均 ÷cost_scale(=10) (论文 cost_scale 技巧, 不改变约束语义)。
"""
from collections import deque

import numpy as np
import torch
from torch.optim import Adam

from .vec_base import VecAgentBase
from .qcpo_ref_model import QcpoRefModel
from .dist_rl_utils_ref import (
    valid_mean, valid_from_done, generalized_advantage_estimation,
    normalize, quantile_huber_loss, weibull_tail_loss,
    quantile_target_estimation, gae_quantile_simple, compute_prob_ratio)


class QCPORefGPU(VecAgentBase):
    """NIPS'22 QCPO 移植版 (PPO+PID + 分位数 cost critic + Weibull 尾 + LSTM)。"""

    # ============================================================ 初始化 ============================================================
    def __init__(self, args, env):
        super().__init__(args, env)                           # 基类: vec_env/wandb/维度 (基类 actor 不用)

        # -------------------- 论文超参 (config_qcpo.py / launch_qcpo.py 缺省) --------------------
        self.learning_rate = float(getattr(args, 'ref_lr', 1e-4))
        self.value_loss_coeff = float(getattr(args, 'value_loss_coeff', 1.0))
        self.entropy_loss_coeff = float(getattr(args, 'entropy_loss_coeff', 0.0))
        self.clip_grad_norm = float(getattr(args, 'clip_grad_norm', 1e4))
        self.gae_lambda = float(getattr(args, 'gae_lambda', 0.97))
        self.minibatches = int(getattr(args, 'minibatches', 1))
        self.epochs = int(getattr(args, 'epochs', 8))
        self.ratio_clip = float(getattr(args, 'ratio_clip', 0.1))
        self.cost_discount = float(getattr(args, 'cost_discount', self.gamma))   # 论文 None→discount
        self.cost_gae_lambda = float(getattr(args, 'cost_gae_lambda', self.gae_lambda))
        self.cost_value_loss_coeff = float(getattr(args, 'cost_value_loss_coeff', 0.5))
        self.ep_cost_ema_alpha = float(getattr(args, 'ep_cost_ema_alpha', 0.0))  # 0=硬更新
        self.ep_outage_ema_alpha = float(getattr(args, 'ep_outage_ema_alpha', 0.0))
        self.ep_cost_eqa_alpha = float(getattr(args, 'ep_cost_eqa_alpha', 0.0))
        self.cost_scale = float(getattr(args, 'cost_scale', 10.0))               # 论文 "yes 10."
        self.target_outage_prob = float(self.q_alpha)                            # ω
        self.weibull_tail_prob = float(getattr(args, 'weibull_tail_prob', 0.3))
        self.n_quantile = int(getattr(args, 'n_quantile', 25))
        self.normalize_advantage = bool(getattr(args, 'normalize_advantage', False))
        self.normalize_cost_advantage = bool(getattr(args, 'normalize_cost_advantage', False))
        self.pid_Ki = float(getattr(args, 'pid_Ki', 0.1))
        self.sum_norm = bool(getattr(args, 'sum_norm', True))     # L=(J_r+λJ_c)/(1+λ)
        self.diff_norm = bool(getattr(args, 'diff_norm', False))
        self.penalty_max = float(getattr(args, 'penalty_max', 100.0))
        self.reward_scale = float(getattr(args, 'reward_scale', 1.0))
        self.penalty_init = float(getattr(args, 'penalty_init', 0.0))
        self.new_T = int(getattr(args, 'new_T', 100))             # LSTM BPTT 段长
        assert not (self.sum_norm and self.diff_norm)
        assert (self.n * self.num_envs) % self.new_T == 0, "T·B 必须被 new_T 整除"
        self.new_B = (self.n * self.num_envs) // self.new_T
        assert self.n % self.new_T == 0, "horizon 必须被 new_T 整除 (整段切块)"

        # 约束阈值: 论文内部尺度 = raw / cost_scale (语义不变)
        self.cost_limit_scaled = float(self.cost_limit) / self.cost_scale

        # 分位索引: target_outage_ind=⌊N(1-ω)-0.5⌋, weibull_tail_ind=⌊N(1-tail)-0.5⌋ (原式)
        self.target_outage_ind = int(np.floor(self.n_quantile * (1 - self.target_outage_prob) - 0.5))
        self.weibull_tail_ind = int(np.floor(self.n_quantile * (1 - self.weibull_tail_prob) - 0.5))
        assert self.target_outage_ind >= self.weibull_tail_ind, "需 ω ≤ weibull_tail_prob"

        # -------------------- 模型 (obs 追加 prev_cost → 维度+1) --------------------
        hidden = getattr(args, 'ref_hidden', [512, 512])
        if isinstance(hidden, str):
            hidden = [int(x) for x in hidden.split(',') if x.strip()]
        self.obs_dim_aug = self.state_dim + 1                 # + prev_cost (论文 wrapper)
        self.model = QcpoRefModel(
            observation_shape=(self.obs_dim_aug,),
            action_size=self.action_dim,
            n_quantile=self.n_quantile,
            hidden_sizes=list(hidden),
            lstm_size=int(getattr(args, 'lstm_size', 512)),
            lstm_skip=bool(getattr(args, 'lstm_skip', True)),
            constraint=True,
            init_log_std=float(getattr(args, 'ref_init_log_std', 0.0)),
            normalize_observation=bool(getattr(args, 'normalize_observation', True)),
            var_clip=float(getattr(args, 'var_clip', 1e-6)),
        ).to(self.device)
        self.lstm_size = int(getattr(args, 'lstm_size', 512))
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

        # -------------------- PID-Lagrangian 状态 (原式) --------------------
        self._ep_cost_ema = self.cost_limit_scaled            # 初始无导数
        self._ep_outage_ema = self.target_outage_prob
        self._ep_cost_eqa = self.cost_limit_scaled
        self.pid_i = self.cost_penalty = self.penalty_init
        self.current_ep_costs = deque(maxlen=100)             # 最近 100 条 episode 的 raw Σc
        self.update_counter = 0
        self.last_dual_prob = 0.0                             # 经验 outage (兼容统一日志)
        # 兼容 run_experiment 对 lambda_dual 的读取 (λ ≙ cost_penalty)
        self.lambda_dual = torch.tensor([0.0], dtype=torch.float32, device=self.device)

    # ============================================================ 高斯策略工具 (对角高斯, 逐位对齐 rlpyt Gaussian) ============================================================
    def _logp(self, action, mean, log_std):
        """对角高斯 logπ(a|s), 任意 leading 维。
        公式逐位对齐 rlpyt.distributions.gaussian.Gaussian.log_likelihood
        (z 分母带 EPS=1e-8; 已用 _verify_ref_port.py 对拍)。"""
        std = torch.exp(log_std)
        z = (action - mean) / (std + 1e-8)
        return -((log_std + 0.5 * z ** 2).sum(dim=-1)
                 + 0.5 * action.shape[-1] * self._log_2pi)

    def _gauss_entropy(self, log_std):
        """对角高斯熵 Σ(logσ + 0.5·log(2πe))。"""
        return (log_std + 0.5 * (1.0 + self._log_2pi)).sum(dim=-1)

    # ============================================================ rollout ([T,B] 时间主序, 含 LSTM 状态/aug obs) ============================================================
    def _rollout_ref(self):
        """
        冻结策略采一段 [T,B] (T=horizon=episode 长度)。返回 dict:
            obs [T,B,obs+1] (含 prev_cost), prev_action [T,B,ad], prev_reward [T,B],
            action [T,B,ad], mu/log_std [T,B,ad] (old dist), done [T,B] (仅末步=1),
            reward [T,B], cost_raw [T,B], r_value [T,B], c_dist [T,B,N], c_w_alpha/beta [T,B],
            h0/c0 [T,B,H] (各步进入前的 LSTM 状态), r_bv [B], c_bdist [B,N], c_bw_a/b [1,B],
            ep_cost [B] (raw Σc), disc_return [B]
        """
        T, B, H = self.n, self.num_envs, self.lstm_size
        dev = self.device
        s_raw = self.vec_env.reset()                          # [B, obs]
        prev_cost = torch.zeros(B, 1, device=dev)
        prev_action = torch.zeros(B, self.action_dim, device=dev)
        prev_reward = torch.zeros(B, device=dev)
        h = torch.zeros(1, B, H, device=dev)
        c = torch.zeros(1, B, H, device=dev)

        OBS, PA, PR, ACT, MU, LSTD = [], [], [], [], [], []
        RV, CD, CWA, CWB, H0, C0 = [], [], [], [], [], []
        REW, COST = [], []
        disc_return = torch.zeros(B, device=dev)
        ep_cost = torch.zeros(B, device=dev)
        dr = 1.0

        with torch.no_grad():
            for t in range(T):
                obs = torch.cat([s_raw, prev_cost], dim=1)    # [B, obs+1] (论文 obs_prev_cost)
                H0.append(h[0].clone()); C0.append(c[0].clone())     # 进入本步前的状态
                mu, log_std, value, rnn = self.model(
                    obs.unsqueeze(0), prev_action.unsqueeze(0), prev_reward.unsqueeze(0), (h, c))
                mu, log_std = mu[0], log_std[0]               # [B, ad]
                a = mu + torch.exp(log_std) * torch.randn_like(mu)   # 采样 (env clip 兜底)
                s2_raw, r, cost, done = self.vec_env.step(a)

                OBS.append(obs); PA.append(prev_action.clone()); PR.append(prev_reward.clone())
                ACT.append(a); MU.append(mu); LSTD.append(log_std)
                RV.append(value.r_value[0]); CD.append(value.c_dist[0])
                CWA.append(value.c_w_alpha[0]); CWB.append(value.c_w_beta[0])
                REW.append(r); COST.append(cost)
                disc_return = disc_return + dr * r; dr *= self.gamma
                ep_cost = ep_cost + cost

                prev_cost = cost.unsqueeze(1)                 # raw cost 进 obs (论文 wrapper 同)
                prev_action, prev_reward = a, r
                h, c = rnn.h, rnn.c
                s_raw = s2_raw

            # ---- bootstrap 值 (episode 末 done=1 → GAE 中被 mask, 仅补全 API) ----
            obs_T = torch.cat([s_raw, prev_cost], dim=1)
            mu_b, _lsb, value_b, _ = self.model(
                obs_T.unsqueeze(0), prev_action.unsqueeze(0), prev_reward.unsqueeze(0), (h, c))

        done_mat = torch.zeros(T, B, device=dev)
        done_mat[-1] = 1.0                                    # 段末=episode 末 (timeout=done, 论文同)

        return {
            'obs': torch.stack(OBS), 'prev_action': torch.stack(PA), 'prev_reward': torch.stack(PR),
            'action': torch.stack(ACT), 'mu_old': torch.stack(MU), 'log_std_old': torch.stack(LSTD),
            'reward': torch.stack(REW), 'cost_raw': torch.stack(COST), 'done': done_mat,
            'r_value': torch.stack(RV), 'c_dist': torch.stack(CD),
            'c_w_alpha': torch.stack(CWA), 'c_w_beta': torch.stack(CWB),
            'h0': torch.stack(H0), 'c0': torch.stack(C0),
            'r_bv': value_b.r_value[0], 'c_bdist': value_b.c_dist[0],
            'c_bw_alpha': value_b.c_w_alpha, 'c_bw_beta': value_b.c_w_beta,    # [1,B]
            'ep_cost': ep_cost, 'disc_return': disc_return,
        }

    # ============================================================ transform_sample (原式: 整段切 new_T 块) ============================================================
    def _transform(self, x):
        """[T,B,*] → [new_T,new_B,*]: 每条轨迹按时间切成 T/new_T 块, 块作为新 B 维 (原式)。"""
        T0, B0 = x.shape[0], x.shape[1]
        rest = tuple(x.shape[2:])
        if T0 == self.new_T and B0 == self.new_B:
            return x
        return (x.transpose(0, 1).reshape(self.new_B, self.new_T, *rest)
                .transpose(0, 1).contiguous())

    # ============================================================ returns / 优势 (process_returns 原式) ============================================================
    def _process_returns(self, roll):
        """[T,B] 原始布局上按原式计算 (GAE / 分位 GAE / LDP 概率比 / 分位 TD 目标 / PID 统计)。"""
        reward = roll['reward']
        cost = roll['cost_raw'] / self.cost_scale             # 论文: cost ÷ cost_scale
        done = roll['done']
        r_value, c_dist = roll['r_value'], roll['c_dist']
        c_w_alpha, c_w_beta = roll['c_w_alpha'], roll['c_w_beta']
        r_bv, c_bdist = roll['r_bv'], roll['c_bdist']

        c_value, c_bv = torch.mean(c_dist, dim=-1), torch.mean(c_bdist, dim=-1)

        # ---- reward GAE ----
        r_advantage, r_return = generalized_advantage_estimation(
            reward, r_value, done, r_bv, self.gamma, self.gae_lambda)

        # ---- cost: 标量 GAE 回报 + 分位 GAE 优势 × LDP 概率比 (原式) ----
        _, c_return = generalized_advantage_estimation(
            cost, c_value, done, c_bv, self.cost_discount, self.cost_gae_lambda)
        tail_ind = self.weibull_tail_ind
        with torch.no_grad():
            c_tail = torch.sort(c_dist)[0][:, :, tail_ind:]
        prob_ratio_rev = compute_prob_ratio(
            cost, c_tail, c_w_alpha, c_w_beta, done,
            roll['c_bw_alpha'], roll['c_bw_beta'], self.gamma, log_clip_range=0.5)
        c_adv_q, _ = gae_quantile_simple(
            cost, c_dist, done, c_bdist, self.gamma, self.cost_gae_lambda)
        c_adv_q_addcost = (1. + torch.log(prob_ratio_rev)) * c_adv_q[:, :, tail_ind:]

        # ---- 分位数 TD 目标 (原式 1-step) ----
        c_dist_target = quantile_target_estimation(cost, c_dist, done, c_bdist, self.cost_discount)

        # ---- valid (recurrent: 首个 done 及之前有效; 本布局 done 仅末步 → valid≡1) ----
        valid = valid_from_done(done)

        # ---- episode 统计 → PID 输入 (原式: 最近 100 条 ep_cost 的 (1-ω) 分位) ----
        ep_costs = roll['ep_cost'].detach().cpu().numpy()     # raw Σc, [B]
        ep_outages = (ep_costs / self.cost_scale > self.cost_limit_scaled).astype(np.float64)
        self.current_ep_costs.extend(ep_costs.tolist())

        ep_cost_avg = float(np.mean(ep_costs)) / self.cost_scale
        self._ep_cost_ema = self.ep_cost_ema_alpha * self._ep_cost_ema \
            + (1 - self.ep_cost_ema_alpha) * ep_cost_avg

        arr = np.sort(np.asarray(self.current_ep_costs))
        q_ind = min(int(np.floor(len(arr) * (1 - self.target_outage_prob))), len(arr) - 1)
        ep_cost_quantile = float(arr[q_ind]) / self.cost_scale
        self._ep_cost_eqa = self.ep_cost_eqa_alpha * self._ep_cost_eqa \
            + (1 - self.ep_cost_eqa_alpha) * ep_cost_quantile

        self._ep_outage_ema = self.ep_outage_ema_alpha * self._ep_outage_ema \
            + (1 - self.ep_outage_ema_alpha) * float(np.mean(ep_outages))
        self.last_dual_prob = float(np.mean(ep_outages))

        # ---- 优势归一化开关 (原式默认关) ----
        if self.normalize_advantage:
            r_advantage = normalize(r_advantage, valid)
        if self.normalize_cost_advantage:
            c_adv_q_addcost = normalize(c_adv_q_addcost, valid)

        # c_advantage: 取 target_outage 分位那一列 (原式索引)
        c_advantage = c_adv_q_addcost[:, :, self.target_outage_ind - self.weibull_tail_ind]

        return (r_return, r_advantage, c_return, c_advantage, c_dist_target, valid)

    # ============================================================ PPO loss (原式) ============================================================
    def _loss(self, obs, prev_action, prev_reward, action, r_return, r_advantage,
              valid, mu_old, log_std_old, c_return, c_advantage, c_dist_target,
              init_rnn_state):
        """一个 minibatch (recurrent: [new_T, mb_B]) 的联合损失 (逐式对齐 qcpo.py loss)。"""
        mu, log_std, value, _rnn = self.model(obs, prev_action, prev_reward, init_rnn_state)

        ratio = torch.exp(self._logp(action, mu, log_std)
                          - self._logp(action, mu_old, log_std_old))
        surr_1 = ratio * r_advantage
        clipped_ratio = torch.clamp(ratio, 1. - self.ratio_clip, 1. + self.ratio_clip)
        surr_2 = clipped_ratio * r_advantage
        pi_loss = -valid_mean(torch.min(surr_1, surr_2), valid)
        pi_r_loss = pi_loss

        value_error = value.r_value - r_return
        r_value_loss = self.value_loss_coeff * valid_mean(0.5 * value_error ** 2, valid)

        entropy = valid_mean(self._gauss_entropy(log_std), valid)
        entropy_loss = -self.entropy_loss_coeff * entropy

        # 约束项 (objective_penalized, 原式 max(违反侧) + sum_norm 组合)
        c_surr_1 = ratio * c_advantage
        c_surr_2 = clipped_ratio * c_advantage
        pi_c_loss = valid_mean(torch.max(c_surr_1, c_surr_2), valid)
        if self.diff_norm:                                    # (1-λ)J_r + λJ_c
            pi_loss = (1 - self.cost_penalty) * pi_loss + self.cost_penalty * pi_c_loss
        elif self.sum_norm:                                   # (J_r + λJ_c)/(1+λ)
            pi_loss = (pi_loss + self.cost_penalty * pi_c_loss) / (1 + self.cost_penalty)
        else:
            pi_loss = pi_loss + self.cost_penalty * pi_c_loss

        loss = pi_loss + entropy_loss + r_value_loss

        # cost critic: 标量均值 MSE + 分位 huber + Weibull 尾 (原式)
        c_value_error = torch.mean(value.c_dist, dim=-1) - c_return
        c_value_loss = valid_mean(0.5 * c_value_error ** 2, valid)
        c_quantile_loss = quantile_huber_loss(value.c_dist, c_dist_target, self.model.tau, valid)
        c_weibull_loss = weibull_tail_loss(value.c_dist, value.c_w_alpha, value.c_w_beta,
                                           self.model.c_tau, self.weibull_tail_ind, valid)
        loss = loss + self.cost_value_loss_coeff * c_value_loss + c_quantile_loss + c_weibull_loss

        info = {
            'ref/pi_r_loss': float(pi_r_loss.item()),
            'ref/pi_c_loss': float(pi_c_loss.item()),
            'ref/r_value_loss': float(r_value_loss.item()),
            'ref/c_value_loss': float(c_value_loss.item()),
            'ref/c_quantile_loss': float(c_quantile_loss.item()),
            'ref/c_weibull_loss': float(c_weibull_loss.item()),
            'ref/entropy': float(entropy.item()),
        }
        return loss, info

    # ============================================================ 主训练循环 ============================================================
    def train(self):
        """每迭代: rollout → returns/优势 → PID λ → obs_rms → PPO epochs → 日志。"""
        print(f"QCPORefGPU[NIPS22]: env={self.env_name}, omega={self.target_outage_prob}, "
              f"d(cost_limit)={self.cost_limit} (scaled {self.cost_limit_scaled}), "
              f"cost_scale={self.cost_scale}, N={self.n_quantile}, "
              f"tail_prob={self.weibull_tail_prob}, pid_Ki={self.pid_Ki}, "
              f"B={self.num_envs}, T={self.n}, new_T={self.new_T}, new_B={self.new_B}, "
              f"iters={self.num_iterations}, epochs={self.epochs}, device={self.device}")

        for it in range(self.num_iterations):
            # ===== 1. 采样 =====
            roll = self._rollout_ref()

            # ===== 2. returns / 优势 / PID 输入统计 =====
            (r_return, r_advantage, c_return, c_advantage,
             c_dist_target, valid) = self._process_returns(roll)

            # ===== 3. PID-Lagrangian 更新 (原式: 先 λ 后 epochs) =====
            delta = float(self._ep_cost_eqa - self.cost_limit_scaled)
            self.pid_i = max(0., self.pid_i + delta * self.pid_Ki)
            if self.diff_norm:
                self.pid_i = max(0., min(1., self.pid_i))
            self.cost_penalty = max(0., self.pid_i)
            if self.diff_norm:
                self.cost_penalty = min(1., self.cost_penalty)
            if not (self.diff_norm or self.sum_norm):
                self.cost_penalty = min(self.cost_penalty, self.penalty_max)
            with torch.no_grad():
                self.lambda_dual.fill_(self.cost_penalty)     # 兼容统一读取

            # ===== 4. obs 归一化统计 (原式: 首迭代牺牲, 只刷统计) =====
            self.model.update_obs_rms(roll['obs'])
            loss_info = {}
            if it > 0:
                # ===== 5. PPO epochs (recurrent: 只 shuffle 块 B, 全 new_T 展开) =====
                tf = self._transform
                obs = tf(roll['obs']); pa = tf(roll['prev_action']); pr = tf(roll['prev_reward'])
                act = tf(roll['action']); mu_old = tf(roll['mu_old']); lstd_old = tf(roll['log_std_old'])
                r_ret = tf(r_return); r_adv = tf(r_advantage)
                c_ret = tf(c_return); c_adv = tf(c_advantage); c_tgt = tf(c_dist_target)
                vld = tf(valid)
                h0 = tf(roll['h0'])[0]; c0 = tf(roll['c0'])[0]   # 各块首步 LSTM 状态 [new_B,H]

                mb_size = self.new_B // self.minibatches
                for _ in range(self.epochs):
                    perm = torch.randperm(self.new_B, device=self.device)
                    for k in range(self.minibatches):
                        idx = perm[k * mb_size:(k + 1) * mb_size]
                        init_rnn = (h0[idx].unsqueeze(0).contiguous(),
                                    c0[idx].unsqueeze(0).contiguous())
                        self.optimizer.zero_grad(set_to_none=True)
                        loss, loss_info = self._loss(
                            obs[:, idx], pa[:, idx], pr[:, idx], act[:, idx],
                            r_ret[:, idx], r_adv[:, idx], vld[:, idx],
                            mu_old[:, idx], lstd_old[:, idx],
                            c_ret[:, idx], c_adv[:, idx], c_tgt[:, idx], init_rnn)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                        self.optimizer.step()
                        self.update_counter += 1

            # ===== 6. 统一日志 (约束变量 = 未折扣 raw Σc, 论文口径) =====
            R_np = roll['disc_return'].detach().cpu().numpy()
            ep_np = roll['ep_cost'].detach().cpu().numpy()
            # 未折扣 episode return (= rlpyt TrajInfo.Return 同口径, 供与原版栈对照)
            undisc_R = roll['reward'].sum(dim=0).detach().cpu().numpy()
            extra = {
                'reward/undisc_return': float(undisc_R.mean()),
                'lambda/value': float(self.cost_penalty),     # λ ≙ cost_penalty (PID)
                'ref/pid_i': float(self.pid_i),
                'ref/ep_cost_quantile': float(self._ep_cost_eqa * self.cost_scale),
                'ref/ep_cost_ema': float(self._ep_cost_ema * self.cost_scale),
                'ref/ep_outage_ema': float(self._ep_outage_ema),
                'constraint/dual_prob': self.last_dual_prob,
                'training/updates': self.update_counter,
            }
            extra.update(loss_info)
            self._log_core(it, R_np, ep_np, ep_np, extra=extra)

    # ============================================================ 评估 (LSTM 状态自管, 与统一评估同口径) ============================================================
    def evaluate_vec(self, vec_env, num_episodes, gamma, cost_gamma, omega, cost_limit):
        """
        与 utils.evaluate_policy_vec 同口径的批量评估 (每段=一 episode), 但自管 LSTM 状态
        与 prev_cost/action/reward 输入。附带 cost_cdf_initial: c_dist(s0) 的 P(C≥d) (×cost_scale 还原)。
        """
        import math
        B, n, H = vec_env.B, vec_env.n, self.lstm_size
        dev = self.device
        rounds = max(1, math.ceil(num_episodes / B))
        R_all, Zc_all, Cu_all, cdf0_all = [], [], [], []

        with torch.no_grad():
            for _ in range(rounds):
                s_raw = vec_env.reset()
                prev_cost = torch.zeros(B, 1, device=dev)
                prev_action = torch.zeros(B, self.action_dim, device=dev)
                prev_reward = torch.zeros(B, device=dev)
                h = torch.zeros(1, B, H, device=dev); c = torch.zeros(1, B, H, device=dev)
                R = torch.zeros(B, device=dev); Zc = torch.zeros(B, device=dev)
                Cu = torch.zeros(B, device=dev)
                dr, dc = 1.0, 1.0
                for t in range(n):
                    obs = torch.cat([s_raw, prev_cost], dim=1)
                    mu, log_std, value, rnn = self.model(
                        obs.unsqueeze(0), prev_action.unsqueeze(0), prev_reward.unsqueeze(0), (h, c))
                    if t == 0:                                # c_dist(s0) → P(C≥d) 校准诊断
                        cdf0 = (value.c_dist[0] * self.cost_scale >= float(cost_limit)).float().mean(dim=1)
                        cdf0_all.append(cdf0)
                    a = mu[0] + torch.exp(log_std[0]) * torch.randn_like(mu[0])
                    s_raw, r, cost, done = vec_env.step(a)
                    R += dr * r; dr *= gamma
                    Zc += dc * cost; dc *= cost_gamma
                    Cu += cost
                    prev_cost = cost.unsqueeze(1); prev_action = a; prev_reward = r
                    h, c = rnn.h, rnn.c
                R_all.append(R); Zc_all.append(Zc); Cu_all.append(Cu)

        R = torch.cat(R_all).cpu().numpy().astype(np.float64)
        Zc = torch.cat(Zc_all).cpu().numpy().astype(np.float64)
        Cu = torch.cat(Cu_all).cpu().numpy().astype(np.float64)
        d = float(cost_limit)
        Z = -Zc                                               # 下尾口径: Z=-C, q=-d
        q = -d
        emp = float(np.mean(Z <= q))                          # = P(C≥d)
        q_est = float(np.percentile(Z, omega * 100))          # Q_α(Z)
        return {
            'mean': float(R.mean()), 'reward_std': float(R.std()),
            'empirical_prob': emp,
            'quantile_return': q_est,
            'quantile_margin_to_threshold': q_est - q,
            'constraint_margin': omega - emp,
            # debug / 论文对照
            'cost_disc_mean': float(Zc.mean()), 'cost_undisc_mean': float(Cu.mean()),
            'outage_prob': emp,                               # 兼容旧键 (=empirical_prob)
            'cost_quantile': float(np.percentile(Zc, (1.0 - omega) * 100)),
            'num_episodes': int(R.shape[0]),
            'cost_cdf_initial': float(torch.cat(cdf0_all).mean().item()),
            'pred_cost_mean': None, 'pred_cost_std': None,
        }

    # ============================================================ 总结接口 ============================================================
    def get_training_summary(self):
        return {
            'lambda_final': float(self.cost_penalty),
            'pid_i': float(self.pid_i),
            'empirical_prob': self.last_empirical_prob,
            'empirical_outage_prob': self.last_outage_prob,   # 兼容 (=empirical_prob)
            'ep_cost_quantile': float(self._ep_cost_eqa * self.cost_scale),
            'dual_prob': self.last_dual_prob,
            'num_envs': self.num_envs,
            'num_iterations': self.num_iterations,
        }
