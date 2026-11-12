"""原始 cost 实验：只把既有 reward-outage 约束转换为 cost 上界。

记 Rγ=Σγ^t r_t，Cγ=Σγ^t c_t，C1=Σc_t，ε=.2，u=1-ε=.8，d=15。
Q_u(Cγ)≤d 等价于 F_Cγ(d)≥u，即 P(Cγ>d)≤ε=.2。

QCPO: L_Q=E[Rγ]-λ_Q(Q_u(Cγ)-d)。官方实现的两个 PPO 项分别
最大化 reward surrogate、最小化 cost-quantile surrogate。权重、GAE、
25-atom midpoint grid、floor 索引、split clipping、sum_norm、Ki 均保持原版。
论文理论梯度包含 discounted occupancy；官方 PPO 不显式乘 γ^t，本实验
遵照用户追加要求保留这个近似，不额外修正其更新算法。

DCA: L_p=E[Rγ]+λ_p(F_Cγ(d)-u)。给定轨迹前缀 P_t=Σ_{k<t}γ^k c_k，
F_t(s,a)=E[1{P_t+γ^t Z_c(s,a)≤d}]，A_F=F_t(s,a)-E_a F_t(s,a)。
则 actor 上升权重为 γ^t A_r + λ_p A_F。实现保留已有 split cost 接口，
将 cost-risk advantage 写为 -A_F，所以最终仍为 reward - cost-risk。
CDF 查询、target atoms、优势权重均 detach；QR 点各用自己的 tau；
只改变 cost critic 输入，独立 reward GAE 仍使用 r_t。

按用户明确指定采用混合控制器：QCPO 的 dual 用 Q_u(C1)-d（内部除以
官方 cost_scale=10）；DCA dual 用 P(C1>d)-(1-u)。这不是同一个 discounted
Lagrangian 的精确原始-对偶梯度，不对这个混合方案声称论文收敛保证。

核对来源：Jung et al. NeurIPS 2022 Eq.(1),(272)-(277), App.E.4，
https://arxiv.org/html/2211.15034；官方 SHA 9e633849154e7f2471bf98709bcc8ba83514708e。
DCA 推导对应本仓库 paper/ICLR_alldone_20260913_revised.tex 的 Lagrangian。
"""
from __future__ import annotations

import copy
import numpy as np
import torch

from agents.four_env_fullbatch_formal import (
    FOUR_ENV_CLASSES, DQC_LR2_KL_ALGO, QREF_FULL_ALGO, task_spec)
from agents.dqc_ac_beta_gpu import DQCACBetaGPU
from agents.qcpo_ref_author_aligned import RnnState
from agents.dist_rl_utils_author_aligned import (
    valid_from_done, generalized_advantage_estimation, valid_mean)
from agents import official_raw_qcpo_utils as official

IDENTITY = 'environment_cost_epsilon0p2_quantile0p8_d15_discounted_primal_undiscounted_dual_v2'
EPSILON = .1
LEVEL = 1.0 - EPSILON
LIMIT = 15.0


def cost_statistics(costs, epsilon=EPSILON, limit=LIMIT):
    """成本超标为严格 >；等于 15 不超标。经验 Q 同时报告逆CDF与线性估计。"""
    values = np.asarray(costs, dtype=np.float64).reshape(-1)
    if not values.size or not np.isfinite(values).all():
        raise ValueError('cost statistics require finite completed episodes')
    level = 1.0 - float(epsilon)
    p = float(np.mean(values > limit))
    return dict(mean=float(values.mean()), std=float(values.std()),
                quantile=float(np.quantile(values, level, method='inverted_cdf')),
                quantile_linear=float(np.quantile(values, level, method='linear')),
                violation_probability=p, residual=p-epsilon,
                feasible=float(p <= epsilon), ties=float(np.sum(values == limit)))


class RawCostMetrics:
    """公共日志与 checkpoint 身份；legacy 因子仅标记来源，raw_cost_* 字段为现行语义。"""

    def _install_cost_identity(self, args):
        self.raw_cost_constraint = True
        self.cost_quantile_level = LEVEL
        self.cost_threshold = LIMIT
        self.failure_alpha = EPSILON
        self.q_alpha = LEVEL
        # 复用父类预算/CDF threshold 存储槽，含义由新 checkpoint identity 明确覆盖。
        self.return_threshold = LIMIT
        self.algo_name = args.algo_name

    def _checkpoint_identity(self):
        identity = super()._checkpoint_identity()
        identity.update(raw_cost_identity=IDENTITY, raw_cost_threshold=LIMIT,
                        raw_cost_quantile_level=LEVEL, raw_cost_violation_budget=EPSILON,
                        raw_cost_primal='discounted_environment_cost',
                        raw_cost_dual='undiscounted_environment_cost',
                        raw_cost_violation_event='C_undiscounted > 15',
                        base_identity_fields_are_provenance=True)
        return identity

    def _capture_episode_metrics(self, rollout, reward_key):
        reward = rollout[reward_key]
        valid = rollout['valid'].bool()
        self._raw_episode_returns = torch.where(valid, reward, 0).double().sum(0).cpu().numpy()
        self._episode_samples = {
            'discounted_return': rollout['disc_return'].detach().cpu().tolist(),
            'undiscounted_return': self._raw_episode_returns.tolist(),
            'discounted_cost': rollout['diagnostic_disc_cost'].detach().cpu().tolist(),
            'undiscounted_cost': rollout['diagnostic_raw_cost'].detach().cpu().tolist()}
        return rollout

    def _log_core(self, iteration, returns, diagnostic_disc_cost, diagnostic_raw_cost, extra=None):
        """两种 return/cost 均保存；公共违约字段一律引用 undiscounted cost。"""
        discounted = cost_statistics(diagnostic_disc_cost)
        raw = cost_statistics(diagnostic_raw_cost)
        log = dict(extra or {})
        # 删除旧 reward-outage 名称，避免含义已改变而仍输出“implied reward”。
        for key in list(log):
            if key == 'quantile/dual_implied_reward':
                del log[key]
        log.update({
            'progress/iteration': int(iteration),
            'progress/env_steps': int(self.main_env_steps),
            'progress/main_env_steps': int(self.main_env_steps),
            'progress/initialization_env_steps': int(self.initialization_env_steps),
            'progress/actual_env_steps': int(self.actual_env_steps),
            'disc_reward/aver_reward': float(np.mean(returns)),
            'disc_reward/return_std': float(np.std(returns)),
            'disc_reward/quantile_reward': float(np.quantile(returns, EPSILON)),
            'undisc_reward/aver_reward': float(np.mean(self._raw_episode_returns)),
            'undisc_reward/return_std': float(np.std(self._raw_episode_returns)),
            'undisc_reward/quantile_reward': float(np.quantile(self._raw_episode_returns, EPSILON)),
            'quantile/q_est': discounted['quantile'],
            'quantile/constraint_prob': raw['violation_probability'],
            'constraint/empirical_failure_probability': raw['violation_probability'],
            'constraint/batch_failure_probability': raw['violation_probability'],
            'constraint/residual': raw['residual'],
            'constraint/feasible': raw['feasible'],
            'constraint/quantile_level': LEVEL,
            'constraint/allowed_violation_probability': EPSILON,
            'constraint/cost_threshold': LIMIT,
        })
        for prefix, values in [('disc_cost', discounted), ('undisc_cost', raw)]:
            log.update({f'{prefix}/{key}': value for key,value in values.items()})
            log[f'{prefix}/quantile_constraint_satisfied'] = float(values['quantile'] <= LIMIT)
        if hasattr(self, 'dual_return_window'):
            log.update({'constraint/dual_window_failure_probability': float(self.last_dual_prob),
                        'constraint/dual_window_residual': self.last_dual_prob-EPSILON,
                        'constraint/dual_window_count': len(self.dual_return_window)})
        if hasattr(self, 'last_dual_cost_quantile'):
            log['lambda/undiscounted_cost_quantile_official_order_statistic'] = self.last_dual_cost_quantile
        self.last_empirical_prob = raw['violation_probability']
        self.last_outage_prob = self.last_empirical_prob
        return log


class RawCostDCA(RawCostMetrics, FOUR_ENV_CLASSES[DQC_LR2_KL_ALGO]):
    """原 DCA：reward GAE 独立保留，QR/CDF 改为环境 discounted cost。"""

    def __init__(self, args, env):
        projected = copy.copy(args)
        projected.algo_name = DQC_LR2_KL_ALGO
        projected.failure_alpha = projected.q_alpha = .2
        projected.return_threshold = task_spec(args.env_name).return_threshold
        super().__init__(projected, env)
        self._install_cost_identity(args)
        # 固定初始分布 discounted-return 的 reward PG 要乘 γ^t；risk 事件不乘。
        self.dqc_reward_outer_discount = 'discounted_gamma_t'

    def _rollout_core(self, *args, **kwargs):
        return self._capture_episode_metrics(super()._rollout_core(*args, **kwargs), 'R')

    def _attach_reward_budgets(self, rollout):
        """仅预算使用 c_t；不覆盖交给独立 reward value/GAE 的 R 张量。"""
        cost = torch.where(rollout['valid'].bool(), rollout['C'], 0).double()
        from objective import COST_GAMMA
        weights = COST_GAMMA ** torch.arange(len(cost), device=cost.device, dtype=torch.float64)[:, None]
        weighted_cost = weights * cost
        prefix = weighted_cost.cumsum(0) - weighted_cost
        rollout['reward_prefix'] = prefix
        rollout['budgets'] = ((self.return_threshold - prefix) / weights).float()
        return rollout

    def build_critic_batch(self, rollout):
        """n-step QR Bellman targets: Σγ^k c_{t+k}+γ^n Z_target，边界掩码保持。"""
        return super().build_critic_batch(dict(rollout, R=rollout['C']))

    def _critic_cdf(self, atoms, rollout):
        """复用 cost 接口的正惩罚方向：返回 1-F_cost，而不是下侧 F_cost。"""
        return 1.0 - super()._critic_cdf(atoms, rollout)

    def _empirical_failure_probability(self, values):
        values = np.asarray(tuple(values), dtype=np.float32)
        return float(np.mean(values > LIMIT)) if values.size else 0.0

    def _checkpoint_identity(self):
        identity = super()._checkpoint_identity()
        identity.update(raw_cost_actor='gamma_t_reward_minus_lambda_upper_cost_probability',
                        raw_cost_critic='discounted_cost_suffix_32_atoms_n100',
                        raw_cost_reward_scale=1.0)
        return identity


def compute_prob_ratio(cost, quantiles, location, shape, scale, done,
                       bootstrap_location, bootstrap_shape, bootstrap_scale, discount,
                       log_clip_range=.5, normalize=True):
    """调用原版两参数 Weibull helper；location 仅为现有模型接口兼容占位。"""
    return official.compute_prob_ratio(cost, quantiles, shape, scale, done,
        bootstrap_shape.unsqueeze(0), bootstrap_scale.unsqueeze(0), discount,
        log_clip_range=log_clip_range, normalize=normalize)


def quantile_huber_loss(quantiles, targets, tau, valid):
    return official.quantile_huber_loss(quantiles, targets, tau, valid)


def weibull_tail_loss(quantiles, location, shape, scale, c_tau, tail_ind, valid):
    # 原版是 (0.5 * error)**2；不是旧 signed adapter 的 0.5 * error**2。
    return official.weibull_tail_loss(quantiles, shape, scale, c_tau, tail_ind, valid)


def gae_quantile_simple(cost, distribution, done, bootstrap, gamma, lam):
    return official.gae_quantile_simple(cost, distribution, done, bootstrap, gamma, lam)


def quantile_target_estimation(cost, distribution, done, bootstrap, gamma):
    return official.quantile_target_estimation(cost, distribution, done, bootstrap.unsqueeze(0), gamma)


from agents.qcpo_ref_author_aligned import QCPORefAuthorAlignedGPU

class RawCostQCPO(RawCostMetrics, QCPORefAuthorAlignedGPU):
    """官方 QCPO 原始非负 cost 模型与优化，目标/统计口径为本次指定配置。"""

    def __init__(self, args, env):
        # 提前标识使父 collector/model 在第一次 rollout 之前选择原始 cost。
        self.raw_cost_constraint = True
        projected = copy.copy(args)
        projected.failure_alpha = projected.q_alpha = EPSILON
        projected.return_threshold = LIMIT
        super().__init__(projected, env)
        self._install_cost_identity(args)
        self.cost_threshold_scaled = LIMIT / self.cost_scale
        self.target_outage_prob = EPSILON
        # 原版 floor 选择离散 midpoint；保留其近似，不擅自更换25点网格。
        self.target_outage_ind = int(np.floor(self.n_quantile*(1-self.target_outage_prob)-.5))
        self.last_dual_cost_quantile = LIMIT
        self.last_dual_reward_quantile = 0.0

    def _rollout_author_aligned(self):
        return self._capture_episode_metrics(super()._rollout_author_aligned(), 'reward')

    def _checkpoint_identity(self):
        identity = super()._checkpoint_identity()
        identity.update(raw_cost_qcpo_target_index=self.target_outage_ind,
                        raw_cost_qcpo_grid='official_midpoint25_unchanged',
                        raw_cost_qcpo_support='official_exp_atoms_zero_location_weibull',
                        raw_cost_qcpo_density='official_center_divide5std_clip_done',
                        raw_cost_qcpo_actor='official_no_extra_outer_discount',
                        raw_cost_qcpo_dual='official_floor_order_statistic_on_training_episode_cost')
        return identity

    @torch.no_grad()
    def _fixed_rollout_targets(self, rollout: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Build every target/advantage once; all eight PPO epochs reuse these detached tensors."""

        transform = self._transform
        reward = transform(rollout["reward"])
        internal_cost = transform(rollout["diagnostic_cost"]) / self.cost_scale
        done = transform(rollout["done"])
        valid = valid_from_done(done)

        reward_value = transform(rollout["r_value"])
        cost_distribution = transform(rollout["c_dist"])
        location = transform(rollout["c_w_location"])
        shape = transform(rollout["c_w_alpha"])
        scale = transform(rollout["c_w_beta"])

        reward_bootstrap = self._chunk_bootstrap(
            reward_value, rollout["r_bootstrap_original"])
        cost_bootstrap = self._chunk_bootstrap(
            cost_distribution, rollout["c_bootstrap_original"])
        location_bootstrap = self._chunk_bootstrap(
            location, rollout["location_bootstrap_original"])
        shape_bootstrap = self._chunk_bootstrap(
            shape, rollout["alpha_bootstrap_original"])
        scale_bootstrap = self._chunk_bootstrap(
            scale, rollout["beta_bootstrap_original"])

        # Ordinary reward GAE uses no normalization and no extra gamma^t actor weighting.
        reward_advantage, reward_return = generalized_advantage_estimation(
            reward,
            reward_value,
            done,
            reward_bootstrap,
            self.gamma,
            self.gae_lambda,
        )

        # The auxiliary mean-cost target is the scalar GAE return of the mean of 25 positive cost atoms.
        cost_value = cost_distribution.mean(dim=-1)
        cost_bootstrap_value = cost_bootstrap.mean(dim=-1)
        _cost_advantage, cost_return = generalized_advantage_estimation(
            internal_cost,
            cost_value,
            done,
            cost_bootstrap_value,
            self.cost_discount,
            self.cost_gae_lambda,
        )

        # Density ratios use the sorted actor-range current atoms.  The helper performs per-tail-atom
        # T/B centering, division by exactly 5*std (no epsilon), clipping, then the done mask.
        sorted_cost = torch.sort(cost_distribution, dim=-1).values.detach()
        # ε=.2 -> Q_.8；直接使用官方 top-8 slice 与官方索引，无需扩展。
        tail = sorted_cost[..., self.weibull_tail_ind:]
        probability_ratio = compute_prob_ratio(
            internal_cost,
            tail,
            location,
            shape,
            scale,
            done,
            location_bootstrap,
            shape_bootstrap,
            scale_bootstrap,
            self.cost_discount,
            log_clip_range=0.5,
            normalize=True,
        )

        # First compute ordinary quantile GAE, then apply the current-time density log-ratio weight.
        quantile_advantage, _quantile_return = gae_quantile_simple(
            internal_cost,
            cost_distribution,
            done,
            cost_bootstrap,
            self.cost_discount,
            self.cost_gae_lambda,
        )
        weighted_tail_advantage = (
            1.0 + torch.log(probability_ratio)
        ) * quantile_advantage[..., self.weibull_tail_ind:]
        constraint_advantage = weighted_tail_advantage[
            ..., self.target_outage_ind - self.weibull_tail_ind]

        # The helper freezes the one-step target and sorts its 25 atoms before QR regression.  QR's
        # pairwise reduction would be permutation-invariant, but the explicit sort is signature-tested.
        cost_distribution_target = quantile_target_estimation(
            internal_cost,
            cost_distribution,
            done,
            cost_bootstrap,
            self.cost_discount,
        )

        return {
            "obs": transform(rollout["obs"]),
            "prev_action": transform(rollout["prev_action"]),
            "prev_reward": transform(rollout["prev_reward"]),
            "action": transform(rollout["action"]),
            "mu_old": transform(rollout["mu_old"]),
            "log_std_old": transform(rollout["log_std_old"]),
            "h0": transform(rollout["h0"])[0],
            "c0": transform(rollout["c0"])[0],
            "reward_return": reward_return.detach(),
            "reward_advantage": reward_advantage.detach(),
            "cost_return": cost_return.detach(),
            "constraint_advantage": constraint_advantage.detach(),
            "cost_distribution_target": cost_distribution_target.detach(),
            "valid": valid.detach(),
            "probability_ratio": probability_ratio.detach(),
            "sorted_cost": sorted_cost,
        }

    def _loss(
        self,
        fixed: Dict[str, torch.Tensor],
        indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Evaluate one full recurrent minibatch and assemble the exact joint objective."""

        initial_state = RnnState(
            h=fixed["h0"][indices].unsqueeze(0).contiguous(),
            c=fixed["c0"][indices].unsqueeze(0).contiguous(),
        )
        mean, log_std, value, _state = self.model(
            fixed["obs"][:, indices],
            fixed["prev_action"][:, indices],
            fixed["prev_reward"][:, indices],
            initial_state,
        )
        new_log_probability = self._logp(fixed["action"][:, indices], mean, log_std)
        old_log_probability = self._logp(
            fixed["action"][:, indices],
            fixed["mu_old"][:, indices],
            fixed["log_std_old"][:, indices],
        )
        ratio = torch.exp(new_log_probability - old_log_probability)
        valid = fixed["valid"][:, indices]

        policy_loss, reward_policy_loss, constraint_policy_loss, clip_fraction = (
            self.split_clipped_policy_losses(
                ratio,
                fixed["reward_advantage"][:, indices],
                fixed["constraint_advantage"][:, indices],
                valid,
                self.ratio_clip,
                self.cost_penalty,
            )
        )

        reward_error = value.r_value - fixed["reward_return"][:, indices]
        reward_value_loss = self.value_loss_coeff * valid_mean(
            0.5 * reward_error.square(), valid)

        # L_C^V already contains 1/2; the total objective multiplies it by the official coefficient
        # 0.5, as in upstream cost_value_loss_coeff=0.5 for this frozen implementation.
        cost_mean_error = value.c_dist.mean(dim=-1) - fixed["cost_return"][:, indices]
        cost_mean_value_loss = valid_mean(0.5 * cost_mean_error.square(), valid)

        # These two helpers deliberately ignore their valid argument.  This counter-intuitive
        # reduction is executable upstream behavior and is protected by dedicated tests.
        quantile_loss = quantile_huber_loss(
            value.c_dist,
            fixed["cost_distribution_target"][:, indices],
            self.model.tau,
            valid,
        )
        tail_loss = weibull_tail_loss(
            value.c_dist,
            value.c_w_location,
            value.c_w_alpha,
            value.c_w_beta,
            self.model.c_tau,
            self.weibull_tail_ind,
            valid,
        )
        entropy = valid_mean(self._gaussian_entropy(log_std), valid)
        total_loss = (
            policy_loss
            + reward_value_loss
            + self.cost_value_loss_coeff * cost_mean_value_loss
            + self.quantile_loss_coeff * quantile_loss
            + self.weibull_loss_coeff * tail_loss
            - self.entropy_loss_coeff * entropy
        )

        support_margin = (
            torch.sort(value.c_dist, dim=-1).values[..., self.weibull_tail_ind:]
            - value.c_w_location.unsqueeze(-1)
        )
        loss_info = {
            "author/pi_total_loss": float(policy_loss.detach().item()),
            "author/pi_reward_loss": float(reward_policy_loss.detach().item()),
            "author/pi_constraint_loss": float(constraint_policy_loss.detach().item()),
            "author/reward_value_loss": float(reward_value_loss.detach().item()),
            "author/cost_mean_value_loss": float(cost_mean_value_loss.detach().item()),
            "author/quantile_loss": float(quantile_loss.detach().item()),
            "author/weibull_tail_loss": float(tail_loss.detach().item()),
            "author/entropy": float(entropy.detach().item()),
            "actor/clip_fraction": float(clip_fraction.detach().item()),
            "author/total_loss": float(total_loss.detach().item()),
            "author/weibull_shape_mean": float(value.c_w_alpha.detach().mean().item()),
            "author/weibull_scale_mean": float(value.c_w_beta.detach().mean().item()),
            "author/weibull_location_mean": float(value.c_w_location.detach().mean().item()),
            "author/weibull_support_margin_min": float(support_margin.detach().min().item()),
        }
        return total_loss, loss_info
