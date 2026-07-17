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
import copy
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from utils import RecurrentActorValue, RecurrentCostEncoder, RunningMeanStd
from .vec_base import VecAgentBase
from .common import (DistributionalCritic, ExceedanceProbabilityCritic,
                     ImplicitQuantileCritic, ScalarValueCritic, lr_lambda,
                     indicator_ge)


def _build_cost_quantile_grid(num_quantiles, mode, query_tau, half_width,
                              local_fraction, device):
    """
    构造 cost-only quantile grid 与 uniform-τ 积分的 importance 权重。

    uniform 返回标准 midpoint grid 和 1/N 权重。query_mixture 把 τ 看作来自
    (1-r)Uniform(0,1)+r Uniform(low,high)：先对 mixture CDF 的等距 midpoint
    做解析反演，再用 1/g(τ) 归一化权重近似 uniform-τ 积分。这样局部输出头更密，
    但 CDF/分布矩仍保持概率质量之和为 1，不会把重复查询点误当成更多概率。
    """
    count = int(num_quantiles)
    if count <= 0:
        raise ValueError("num_quantiles must be positive")
    selected_mode = str(mode).lower()
    if selected_mode not in {'uniform', 'query_mixture'}:
        raise ValueError("cost_quantile_grid_mode must be 'uniform' or 'query_mixture'")

    # u_i 是 sampling distribution 下每个等质量分层的 midpoint；无随机采样，
    # 所以相同配置跨 run 完全可复现，也不会扰动 policy action 的 RNG 流。
    u = (torch.arange(count, dtype=torch.float64, device=device) + 0.5) / count
    if selected_mode == 'uniform':
        density = torch.ones_like(u)
        weights = torch.full_like(u, 1.0 / count)
        return u.float(), weights.float(), density.float()

    center = float(query_tau)
    radius = float(half_width)
    mixture = float(local_fraction)
    low, high = center - radius, center + radius
    if not 0.0 < low < high < 1.0:
        raise ValueError("query_tau +/- cost_quantile_local_half_width must lie in (0,1)")
    if not 0.0 < mixture < 1.0:
        raise ValueError("cost_quantile_local_fraction must lie in (0,1)")

    # mixture CDF 的三个线性区间：
    # outside density=(1-r)，inside density=(1-r)+r/(high-low)。
    base_density = 1.0 - mixture
    local_density = base_density + mixture / (high - low)
    cdf_low = base_density * low
    cdf_high = base_density * high + mixture
    taus = torch.empty_like(u)
    below = u < cdf_low
    inside = (u >= cdf_low) & (u <= cdf_high)
    above = u > cdf_high
    taus[below] = u[below] / base_density
    taus[inside] = (
        u[inside] + mixture * low / (high - low)) / local_density
    taus[above] = (u[above] - mixture) / base_density

    # Deterministic importance sampling：E_uniform[f(τ)] =
    # E_mixture[f(τ)/g(τ)]。有限分层下重新归一化，保证权重严格和为 1。
    density = torch.where(inside, torch.full_like(u, local_density),
                          torch.full_like(u, base_density))
    inverse_density = density.reciprocal()
    weights = inverse_density / inverse_density.sum()
    return taus.float(), weights.float(), density.float()


def _binary_probability_metrics(probabilities, outcomes):
    """
    计算Bernoulli概率预测的proper score、skill与排序分辨率。

    输入是逐轨迹预测概率和0/1 outage标签；返回值只用于独立评估日志，不参与
    critic loss、actor advantage或PID。ROC-AUC使用带平均tie rank的Mann--Whitney
    公式，因此hard QR的1/N离散概率不会因任意排序tie而得到虚假的分辨率。
    """
    predicted = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    labels = np.asarray(outcomes, dtype=np.float64).reshape(-1)
    if predicted.shape != labels.shape or predicted.size == 0:
        raise ValueError("probabilities and outcomes must be non-empty aligned vectors")
    if not np.isfinite(predicted).all() or not np.isfinite(labels).all():
        raise ValueError("probabilities and outcomes must be finite")
    if np.any((labels != 0.0) & (labels != 1.0)):
        raise ValueError("outcomes must contain only binary 0/1 labels")
    if np.any(predicted < -1e-7) or np.any(predicted > 1.0 + 1e-7):
        raise ValueError("probabilities must lie in [0, 1]")

    # Brier Skill以当前独立评估集的经验基率常数预测为climatology；BSS>0才表示
    # 逐状态概率预测优于永远输出同一个outage rate。单一类别时skill未定义。
    brier = float(np.mean((predicted - labels) ** 2))
    prevalence = float(labels.mean())
    climatology_brier = prevalence * (1.0 - prevalence)
    brier_skill = (
        1.0 - brier / climatology_brier
        if climatology_brier > 0.0 else float('nan'))

    positive = labels == 1.0
    negative = ~positive
    positive_mean = (
        float(predicted[positive].mean()) if bool(positive.any()) else float('nan'))
    negative_mean = (
        float(predicted[negative].mean()) if bool(negative.any()) else float('nan'))
    discrimination_gap = positive_mean - negative_mean

    # 先稳定排序，再为每个完全相同的score分配平均1-based rank。这样全常数预测
    # 精确得到AUC=.5，而不是由episode原始顺序决定0或1。
    positive_count = int(positive.sum())
    negative_count = int(negative.sum())
    roc_auc = float('nan')
    if positive_count > 0 and negative_count > 0:
        order = np.argsort(predicted, kind='mergesort')
        sorted_scores = predicted[order]
        ranks = np.empty(predicted.size, dtype=np.float64)
        begin = 0
        while begin < predicted.size:
            finish = begin + 1
            while (finish < predicted.size
                   and sorted_scores[finish] == sorted_scores[begin]):
                finish += 1
            average_rank = 0.5 * ((begin + 1) + finish)
            ranks[order[begin:finish]] = average_rank
            begin = finish
        positive_rank_sum = float(ranks[positive].sum())
        roc_auc = (
            positive_rank_sum - positive_count * (positive_count + 1) / 2.0
        ) / (positive_count * negative_count)

    return {
        'brier': brier,
        'brier_skill': float(brier_skill),
        'roc_auc': float(roc_auc),
        'positive_mean': positive_mean,
        'negative_mean': negative_mean,
        'discrimination_gap': float(discrimination_gap),
        'prediction_std': float(predicted.std()),
    }


class DQCACBetaGPU(VecAgentBase):
    """DQC-AC-β CMDP 版 (reward+cost 双分布式 critic; budget/critic-dual 在 cost 上; 上尾约束)。"""

    # ============================================================ 初始化 ============================================================
    def __init__(self, args, env):
        super().__init__(args, env)                           # 基类: env/actor/wandb/维度

        # -------------------- 算法专属超参 (与 portfolio 模板同名同义) --------------------
        self.beta = getattr(args, 'beta', 0.95)               # Abel 风险折扣 β (仅 cost 约束项)
        self.outer_interval = max(1, int(getattr(args, 'outer_interval', 1)))
        self.num_quantiles = getattr(args, 'num_quantiles', 32)

        # cost-only distribution family：qr 是所有历史实验的固定输出头；iqn 用
        # 连续 τ cosine embedding。reward critic 仍保持固定 QR，确保实验只改变
        # 约束分布估计，不把 reward GAE/PPO 主干一起换掉。
        self.cost_distribution_model = str(
            getattr(args, 'cost_distribution_model', 'qr')).lower()
        if self.cost_distribution_model not in {'qr', 'iqn'}:
            raise ValueError("cost_distribution_model must be 'qr' or 'iqn'")
        self.cost_iqn_train_quantiles = int(
            getattr(args, 'cost_iqn_train_quantiles', self.num_quantiles))
        self.cost_iqn_query_quantiles = int(
            getattr(args, 'cost_iqn_query_quantiles', 128))
        self.cost_iqn_cosines = int(getattr(args, 'cost_iqn_cosines', 64))
        self.cost_iqn_seed = int(
            getattr(args, 'cost_iqn_seed', self.seed + 104729))
        if min(self.cost_iqn_train_quantiles,
               self.cost_iqn_query_quantiles,
               self.cost_iqn_cosines) <= 0:
            raise ValueError("all IQN sample/count parameters must be positive")

        # C-DCF1 把 actor 查询来源与分布表示解耦。quantile 完全复现历史：
        # QR/IQN -> threshold count/integration；direct 则额外训练 Bernoulli head，
        # 直接估计 P(C_remaining >= budget | s,a,budget)。QR 仍保留作mean/分位诊断。
        self.cost_cdf_estimator = str(
            getattr(args, 'cost_cdf_estimator', 'quantile')).lower()
        if self.cost_cdf_estimator not in {'quantile', 'direct'}:
            raise ValueError("cost_cdf_estimator must be 'quantile' or 'direct'")
        self.cost_direct_cdf_lr = float(
            getattr(args, 'cost_direct_cdf_lr',
                    getattr(args, 'critic_lr', 1e-3)))
        self.cost_direct_cdf_grad_clip = float(
            getattr(args, 'cost_direct_cdf_grad_clip', 10.0))
        # C-DCF2 只改变direct head的查询参数版本：online保持C-DCF1 exact；
        # ema对每次online Adam step做Polyak低通，抑制每批20条轨迹的比例噪声。
        self.cost_direct_cdf_query_mode = str(
            getattr(args, 'cost_direct_cdf_query_mode', 'online')).lower()
        self.cost_direct_cdf_ema_tau = float(
            getattr(args, 'cost_direct_cdf_ema_tau', 0.005))
        raw_direct_budget_scale = getattr(
            args, 'cost_direct_cdf_budget_scale', None)
        self.cost_direct_cdf_budget_scale = (
            max(abs(float(self.cost_limit)), 1.0)
            if raw_direct_budget_scale is None
            else float(raw_direct_budget_scale))
        if self.cost_direct_cdf_lr <= 0.0:
            raise ValueError("cost_direct_cdf_lr must be positive")
        if self.cost_direct_cdf_grad_clip < 0.0:
            raise ValueError("cost_direct_cdf_grad_clip must be non-negative")
        if self.cost_direct_cdf_query_mode not in {'online', 'ema'}:
            raise ValueError(
                "cost_direct_cdf_query_mode must be 'online' or 'ema'")
        if not 0.0 < self.cost_direct_cdf_ema_tau <= 1.0:
            raise ValueError("cost_direct_cdf_ema_tau must be in (0, 1]")
        if self.cost_direct_cdf_budget_scale <= 0.0:
            raise ValueError("cost_direct_cdf_budget_scale must be positive")
        if (self.cost_cdf_estimator != 'direct'
                and self.cost_direct_cdf_query_mode != 'online'):
            raise ValueError(
                "EMA direct query requires cost_cdf_estimator='direct'")

        # C-Q4 只改变 cost critic 的 τ grid；reward critic 始终保留 uniform midpoint。
        # query_mixture 默认中心为 τ*=1-alpha，即上尾机会约束的决策边界。
        self.cost_quantile_grid_mode = str(
            getattr(args, 'cost_quantile_grid_mode', 'uniform')).lower()
        _query_tau = getattr(args, 'cost_quantile_query_tau', None)
        self.cost_quantile_query_tau = (
            1.0 - float(self.q_alpha) if _query_tau is None else float(_query_tau))
        self.cost_quantile_local_half_width = float(
            getattr(args, 'cost_quantile_local_half_width', 0.1))
        self.cost_quantile_local_fraction = float(
            getattr(args, 'cost_quantile_local_fraction', 0.5))
        # query_focused 对每个非均匀 prediction head 等权，主动加强查询区训练；
        # importance 则按 1/g(τ) 加权，近似保留全局 uniform-τ W1 目标。
        self.cost_quantile_prediction_weighting = str(
            getattr(args, 'cost_quantile_prediction_weighting', 'query_focused')).lower()
        if self.cost_quantile_prediction_weighting not in {'query_focused', 'importance'}:
            raise ValueError(
                "cost_quantile_prediction_weighting must be 'query_focused' or 'importance'")

        # hard 完全复现历史 quantile count；sigmoid 只平滑 actor 查询点的
        # indicator，不改变 QR target/loss、经验 outage 或 hard-CDF 校准指标。
        self.cost_cdf_mode = str(getattr(args, 'cost_cdf_mode', 'hard')).lower()
        if self.cost_cdf_mode not in {'hard', 'sigmoid'}:
            raise ValueError("cost_cdf_mode must be 'hard' or 'sigmoid'")
        self.cost_cdf_temperature = float(
            getattr(args, 'cost_cdf_temperature', 1.0))
        if self.cost_cdf_temperature <= 0.0:
            raise ValueError("cost_cdf_temperature must be positive")
        self.huber_kappa = getattr(args, 'huber_kappa', 0.1)  # κ (0.1=近纯分位回归, 无偏)
        self.target_tau = getattr(args, 'target_tau', 0.05)   # target 软更新系数
        self.target_update_interval = getattr(args, 'target_update_interval', 1)
        # C-X1 默认仍由 online cost critic 生成风险优势，逐式兼容全部历史实验。
        # target 模式让 actor/constraint-RMS 只查询上一个 rollout 已形成的 Polyak
        # critic，从而隔离“本批标签先训练 online critic、再立刻驱动本批 actor”的泄漏。
        self.cost_actor_query_mode = str(
            getattr(args, 'cost_actor_query_mode', 'online')).lower()
        if self.cost_actor_query_mode not in {
                'online', 'target', 'crossfit', 'preupdate'}:
            raise ValueError(
                "cost_actor_query_mode must be 'online', 'target', 'crossfit', "
                "or 'preupdate'")
        self.n_step = max(1, int(getattr(args, 'n_step', 1)))
        if self.n_step > self.n:
            raise ValueError(
                f'n_step={self.n_step} cannot exceed horizon={self.n}')
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

        # C-W1 对齐 cost critic 与风险 actor 的有效时间分布。uniform 是逐式
        # 兼容默认；risk_discount 用 discount^t 并归一化到 batch mean=1，
        # 因而只改变 transition 相对权重，不隐式改变 critic learning-rate 尺度。
        self.cost_critic_time_weighting = str(
            getattr(args, 'cost_critic_time_weighting', 'uniform')).lower()
        if self.cost_critic_time_weighting not in {'uniform', 'risk_discount'}:
            raise ValueError(
                "cost_critic_time_weighting must be 'uniform' or 'risk_discount'")
        _cost_weight_discount = getattr(
            args, 'cost_critic_weight_discount', None)
        self.cost_critic_weight_discount = (
            float(self.beta) if _cost_weight_discount is None
            else float(_cost_weight_discount))
        self.cost_critic_weight_floor = float(
            getattr(args, 'cost_critic_weight_floor', 0.0))
        if not 0.0 < self.cost_critic_weight_discount <= 1.0:
            raise ValueError("cost_critic_weight_discount must be in (0,1]")
        if self.cost_critic_weight_floor < 0.0:
            raise ValueError("cost_critic_weight_floor must be non-negative")

        # C-S0 只重排 cost critic 的监督测度：recent replay 保存每条完整轨迹的
        # (s0, a0, MC cost)，并对初始风险分布增加独立 QR 辅助项。coef 表示辅助
        # loss 与原 transition loss 的比值；更新时再除以 (1+coef)，保持 cost
        # objective 总尺度，避免把“重视 s0”混淆成“提高 critic learning rate”。
        self.cost_s0_aux_coef = float(
            getattr(args, 'cost_s0_aux_coef', 0.0))
        # QCPO_refs 同时拟合 cost distribution 的均值与 quantiles。这里的系数
        # 与 reference 的 cost_value_loss_coeff 同义；默认0保证历史路径逐式不变。
        self.cost_mean_anchor_coef = float(
            getattr(args, 'cost_mean_anchor_coef', 0.0))
        # QCPO_refs 先把cost除以10再计算MSE；raw-cost DQCAC必须显式补偿
        # 这个单位差，否则二次MSE相对线性QR会被额外放大一个cost_scale。
        self.cost_mean_anchor_cost_scale = float(
            getattr(args, 'cost_mean_anchor_cost_scale', 10.0))
        # QCPO_refs 的 constraint head 输出 exp(logit)，但其监督和budget都先除以10。
        # 本实现继续使用raw cost，所以正值映射必须乘回scale；linear默认不增加
        # 任何tensor操作，保证历史checkpoint和逐张量回归完全兼容。
        self.cost_quantile_output = str(
            getattr(args, 'cost_quantile_output', 'linear')).lower()
        self.cost_quantile_output_scale = float(
            getattr(args, 'cost_quantile_output_scale', 10.0))
        # C-H6默认关闭：用QCPO_refs单位/归约的cost辅助目标训练policy/reward共享
        # MLP+LSTM。cost head本身仍只属于critic optimizer，避免两个Adam拥有同一参数。
        self.cost_shared_backbone_coef = float(
            getattr(args, 'cost_shared_backbone_coef', 0.0))
        self.cost_shared_backbone_cost_scale = float(
            getattr(args, 'cost_shared_backbone_cost_scale', 10.0))
        self.cost_shared_backbone_huber_kappa = float(
            getattr(args, 'cost_shared_backbone_huber_kappa', 1.0))
        self.cost_s0_replay_batches = int(
            getattr(args, 'cost_s0_replay_batches', 4))
        if self.cost_s0_aux_coef < 0.0:
            raise ValueError("cost_s0_aux_coef must be non-negative")
        if self.cost_mean_anchor_coef < 0.0:
            raise ValueError("cost_mean_anchor_coef must be non-negative")
        if self.cost_mean_anchor_cost_scale <= 0.0:
            raise ValueError("cost_mean_anchor_cost_scale must be positive")
        if self.cost_quantile_output not in {'linear', 'exp', 'softplus'}:
            raise ValueError(
                "cost_quantile_output must be 'linear', 'exp', or 'softplus'")
        if self.cost_quantile_output_scale <= 0.0:
            raise ValueError("cost_quantile_output_scale must be positive")
        if self.cost_shared_backbone_coef < 0.0:
            raise ValueError("cost_shared_backbone_coef must be non-negative")
        if self.cost_shared_backbone_cost_scale <= 0.0:
            raise ValueError("cost_shared_backbone_cost_scale must be positive")
        if self.cost_shared_backbone_huber_kappa <= 0.0:
            raise ValueError("cost_shared_backbone_huber_kappa must be positive")
        if self.cost_s0_replay_batches <= 0:
            raise ValueError("cost_s0_replay_batches must be positive")
        # step feature 默认跟随 episodic: 有限期界下剩余 cost 分布依赖剩余步数 (已验证配方)
        _csf = getattr(args, 'critic_step_feature', None)
        self.critic_step_feature = bool(_csf) if _csf is not None else self.episodic
        self.num_action_samples = max(1, int(getattr(args, 'num_action_samples', 4)))
        self.updates_per_episode = max(1, int(getattr(args, 'updates_per_episode', 10)))
        # critic/value 可复用同一 rollout 多次；非 PPO actor 必须保持 1 次严格 on-policy 更新。
        # checkpoint critic-calibration 显式允许0次policy update；默认路径仍把用户
        # 输入下限夹到1，保持历史训练语义。独立freeze flag还会同时禁止dual更新。
        self.freeze_policy_updates = bool(
            getattr(args, 'freeze_policy_updates', False))
        requested_actor_updates = int(getattr(args, 'actor_updates_per_episode', 1))
        self.actor_updates_per_episode = (
            0 if self.freeze_policy_updates else max(1, requested_actor_updates))
        if self.actor_updates_per_episode > self.updates_per_episode:
            raise ValueError("actor_updates_per_episode cannot exceed updates_per_episode")
        # P-M7 默认关闭。interval>1 时 critic/PID 仍逐 rollout 更新，但 actor 权重和
        # actor 自带 observation RMS 等收满 interval 个同策略 rollout 后才一起更新。
        # 这与“把同一 rollout 做更多 epoch”不同：新增的是独立轨迹，不是重复标签。
        self.actor_update_interval = max(
            1, int(getattr(args, 'actor_update_interval', 1)))
        self.actor_update_events = 0
        self.actor_rollouts_since_update = 0
        # 保留最近一次真正到期的 PPO 事件诊断，使 disabled/offline W&B 的短测试也能
        # 从 JSON/checkpoint 证明首 epoch 的 π_new/π_old 是否严格从 1 开始。
        self.last_actor_first_epoch_ratio_max_error = 0.0
        self.last_actor_update_batch_trajectories = 0.0
        self.last_actor_updates_completed = 0.0
        self.last_actor_approx_kl = 0.0
        self.last_actor_clip_fraction = 0.0
        self.last_shared_cost_qr_loss = 0.0
        self.last_shared_cost_mean_loss = 0.0
        self.last_shared_cost_weighted_loss = 0.0
        self.last_shared_cost_feature_refresh_abs_mean = 0.0
        self.last_shared_cost_body_grad_norm = 0.0
        self.last_shared_cost_lstm_grad_norm = 0.0
        self.last_shared_cost_policy_head_grad_norm = 0.0
        self.last_shared_cost_value_head_grad_norm = 0.0
        self.last_shared_cost_head_grad_present = 0.0
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

        # QR-DQN 原实现对 target sample 维求和，因此 target quantile 数 N 翻倍时，
        # loss 与裁剪前梯度也约翻倍。legacy_sum 是逐式兼容的默认路径；
        # reference_mean 乘 reference/N_target，使不同 N 共用 N=32 的优化尺度。
        self.quantile_target_reduction = str(
            getattr(args, 'quantile_target_reduction', 'legacy_sum')).lower()
        valid_target_reductions = {'legacy_sum', 'reference_mean'}
        if self.quantile_target_reduction not in valid_target_reductions:
            raise ValueError(
                "quantile_target_reduction must be 'legacy_sum' or 'reference_mean'")
        self.quantile_loss_reference_samples = int(
            getattr(args, 'quantile_loss_reference_samples', 32))
        if self.quantile_loss_reference_samples <= 0:
            raise ValueError("quantile_loss_reference_samples must be positive")

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
        # target-KL 是默认关闭的PPO安全阀。clip限制单样本ratio，但无法保证整批
        # policy displacement；正阈值在越界epoch反传前停止，避免再把policy推远。
        self.ppo_target_kl = float(getattr(args, 'ppo_target_kl', 0.0))
        if self.ppo_target_kl < 0.0:
            raise ValueError("ppo_target_kl must be non-negative; 0 disables early stop")
        if self.ppo_target_kl > 0.0 and self.reward_actor_mode != 'gae_ppo':
            raise ValueError("ppo_target_kl requires reward_actor_mode=gae_ppo")
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
            self.recurrent_hidden = getattr(
                args, 'recurrent_hidden', [512, 512])
            if isinstance(self.recurrent_hidden, str):
                self.recurrent_hidden = [
                    int(x) for x in self.recurrent_hidden.split(',') if x.strip()]
            self.actor = RecurrentActorValue(
                observation_dim=self.state_dim + 1,
                action_dim=self.action_dim,
                hidden=self.recurrent_hidden,
                lstm_size=int(getattr(args, 'lstm_size', 512)),
                lstm_skip=bool(getattr(args, 'lstm_skip', True)),
                init_std=float(getattr(args, 'init_std', 1.0)),
                learn_std=bool(getattr(args, 'learn_std', True)),
                normalize_observation=self.normalize_observation,
                var_clip=self.obs_norm_var_clip).to(self.device)

        # cost history 三路线：
        # - raw：历史默认，只看当前 Markov observation；
        # - actor_feature：C-H0.5，共享但 detach PPO recurrent feature；
        # - cost_lstm：C-H1，用 cost QR loss 独立训练同输入协议的 MLP+LSTM。
        self.cost_history_mode = str(
            getattr(args, 'cost_history_mode', 'raw')).lower()
        valid_cost_history_modes = {'raw', 'actor_feature', 'cost_lstm'}
        if self.cost_history_mode not in valid_cost_history_modes:
            raise ValueError(
                f"cost_history_mode must be one of {sorted(valid_cost_history_modes)}")
        if self.cost_history_mode != 'raw' and not self.recurrent_policy:
            raise ValueError(
                f"cost_history_mode={self.cost_history_mode!r} requires policy_arch='mlp_lstm'")
        # C-H1 首轮只验证完整 finite-horizon MC calibration。n-step 需要为
        # t+N 构造 target cost history，是另一项独立算法改动，不能混入本消融。
        if self.cost_history_mode == 'cost_lstm' and self.cost_target_mode != 'mc':
            raise ValueError("cost_history_mode='cost_lstm' currently requires cost_target_mode='mc'")
        if self.cost_s0_aux_coef > 0.0:
            if self.cost_target_mode != 'mc':
                raise ValueError("cost_s0_aux_coef>0 currently requires cost_target_mode='mc'")
            if self.cost_history_mode != 'raw':
                raise ValueError("cost_s0_aux_coef>0 currently requires cost_history_mode='raw'")

        # C-H6首轮只改变QCPO_refs式共享表示梯度，不同时混入IQN、局部网格、
        # target/crossfit查询、独立cost-LSTM或actor cadence。严格限制组合使中长跑
        # 的差异可以归因于“cost监督是否进入policy/reward MLP+LSTM”这一项。
        if self.cost_shared_backbone_coef > 0.0:
            if not self.recurrent_policy or self.reward_actor_mode != 'gae_ppo':
                raise ValueError(
                    "cost_shared_backbone_coef>0 requires recurrent gae_ppo")
            if self.cost_history_mode != 'actor_feature':
                raise ValueError(
                    "cost_shared_backbone_coef>0 requires cost_history_mode='actor_feature'")
            if self.cost_target_mode != 'mc':
                raise ValueError(
                    "cost_shared_backbone_coef>0 currently requires cost_target_mode='mc'")
            if self.cost_distribution_model != 'qr':
                raise ValueError(
                    "cost_shared_backbone_coef>0 currently requires QR cost distribution")
            if self.cost_quantile_grid_mode != 'uniform':
                raise ValueError(
                    "cost_shared_backbone_coef>0 currently requires uniform quantiles")
            if self.cost_cdf_estimator != 'quantile':
                raise ValueError(
                    "cost_shared_backbone_coef>0 currently requires quantile CDF")
            if self.cost_actor_query_mode != 'online':
                raise ValueError(
                    "cost_shared_backbone_coef>0 currently requires online cost query")
            if self.cost_quantile_output != 'linear':
                raise ValueError(
                    "cost_shared_backbone_coef>0 first ablation requires linear cost output")
            if self.cost_s0_aux_coef > 0.0:
                raise ValueError(
                    "cost_shared_backbone_coef>0 currently requires cost_s0_aux_coef=0")
            if self.actor_update_interval != 1:
                raise ValueError(
                    "cost_shared_backbone_coef>0 currently requires actor_update_interval=1")
            if self.freeze_policy_updates or self.actor_updates_per_episode <= 0:
                raise ValueError(
                    "cost_shared_backbone_coef>0 requires live actor updates")
        # 主cost critic继续使用已验证的huber_kappa=0.1；共享辅助项单独使用
        # cost_shared_backbone_huber_kappa=1，精确对应QCPO_refs而不暗改head优化目标。

        # C-DCF1 首轮只在P-M3已经验证的QR/raw/MC/online链路隔离比较查询表示。
        # 这些限制不是永久API边界；它们防止首次实验同时混入IQN、history encoder、
        # current-batch时序或recent-s0 replay，使CDF收益能够被唯一归因。
        if self.cost_cdf_estimator == 'direct':
            if self.cost_distribution_model != 'qr':
                raise ValueError(
                    "direct cost CDF currently requires cost_distribution_model='qr'")
            if self.cost_quantile_grid_mode != 'uniform':
                raise ValueError(
                    "direct cost CDF currently requires uniform QR diagnostics")
            if self.cost_target_mode != 'mc' or self.cost_history_mode != 'raw':
                raise ValueError(
                    "direct cost CDF currently requires MC/raw cost supervision")
            if self.cost_actor_query_mode != 'online':
                raise ValueError(
                    "direct cost CDF currently requires cost_actor_query_mode='online'")
            if self.cost_s0_aux_coef > 0.0:
                raise ValueError("direct cost CDF currently requires cost_s0_aux_coef=0")

        # C-IQN1 首轮只替换 P-M3 已验证的 raw/MC cost critic。固定策略校准门先
        # 判断连续 τ 表示是否改善泛化；在通过前不把 n-step target、cost-LSTM、
        # cross-fit 或 recent-s0 replay 混进同一实验。
        if self.cost_distribution_model == 'iqn':
            if self.cost_quantile_grid_mode != 'uniform':
                raise ValueError(
                    "cost IQN currently requires cost_quantile_grid_mode='uniform'")
            if self.cost_target_mode != 'mc':
                raise ValueError("cost IQN currently requires cost_target_mode='mc'")
            if self.cost_history_mode != 'raw':
                raise ValueError("cost IQN currently requires cost_history_mode='raw'")
            if self.cost_s0_aux_coef > 0.0:
                raise ValueError("cost IQN currently requires cost_s0_aux_coef=0")
            if self.cost_actor_query_mode == 'crossfit':
                raise ValueError("cost IQN does not yet support crossfit critics")

        # C-X2 首轮只检验严格的两折样本隔离。按完整环境轨迹拆 fold，要求偶数 B；
        # MC/raw/full-batch 限制把 n-step target、历史 encoder、recent replay 与
        # transition chunking 都排除，避免一次实验混入四种尚未对拍的分支语义。
        if self.cost_actor_query_mode == 'crossfit':
            if self.num_envs < 2 or self.num_envs % 2 != 0:
                raise ValueError(
                    "cost_actor_query_mode='crossfit' requires an even num_envs >= 2")
            if self.cost_target_mode != 'mc':
                raise ValueError(
                    "cost_actor_query_mode='crossfit' currently requires cost_target_mode='mc'")
            if self.cost_history_mode != 'raw':
                raise ValueError(
                    "cost_actor_query_mode='crossfit' currently requires cost_history_mode='raw'")
            if self.cost_s0_aux_coef > 0.0:
                raise ValueError(
                    "cost_actor_query_mode='crossfit' currently requires cost_s0_aux_coef=0")
            if self.critic_minibatch_size != 0:
                raise ValueError(
                    "cost_actor_query_mode='crossfit' currently requires critic_minibatch_size=0")

        # C-X3首轮只服务已经验证的P-M3 recurrent/MC/raw/full-batch主路径。
        # MLP critic update可能在内部采样bootstrap action；把actor提前会交换RNG
        # 顺序并混入第二个变量，所以在专门对拍前显式拒绝这些未验证组合。
        if self.cost_actor_query_mode == 'preupdate':
            if not self.recurrent_policy or self.reward_actor_mode != 'gae_ppo':
                raise ValueError(
                    "cost_actor_query_mode='preupdate' currently requires recurrent gae_ppo")
            if self.cost_target_mode != 'mc' or self.cost_history_mode != 'raw':
                raise ValueError(
                    "cost_actor_query_mode='preupdate' currently requires MC/raw cost critic")
            if self.cost_s0_aux_coef > 0.0 or self.critic_minibatch_size != 0:
                raise ValueError(
                    "cost_actor_query_mode='preupdate' currently requires s0_aux=0 and full-batch")

        # 首个 actor-interval 消融只开放已经完整验证的 P-M3 recurrent/GAE-PPO/
        # MC/raw/online 路径。两个 rollout 的 behavior actor、log-prob 分母和 obs RMS
        # 必须完全相同，才能把合并 batch 仍称为 on-policy；其它组合先显式拒绝。
        if self.actor_update_interval > 1 and not self.freeze_policy_updates:
            if not self.recurrent_policy or self.reward_actor_mode != 'gae_ppo':
                raise ValueError(
                    "actor_update_interval>1 currently requires recurrent gae_ppo")
            if self.cost_target_mode != 'mc' or self.cost_history_mode != 'raw':
                raise ValueError(
                    "actor_update_interval>1 currently requires MC/raw cost supervision")
            if self.cost_actor_query_mode != 'online' or self.cost_s0_aux_coef > 0.0:
                raise ValueError(
                    "actor_update_interval>1 currently requires online query and s0_aux=0")
            # 该校验位于 BaseVecAgent.__init__ 之前，因此 warmup_iters 尚未写入
            # self；直接按下方正式初始化所用的同一规则从 args 解析，避免初始化
            # 顺序依赖，同时保持未显式配置时 QCPO advantage 的默认 5 轮语义。
            raw_warmup = getattr(args, 'warmup_iters', None)
            configured_warmup = (
                int(raw_warmup) if raw_warmup is not None
                else (5 if self.advantage_norm == 'qcpo' else 0))
            effective_warmup = max(
                configured_warmup,
                self.obs_norm_warmup_iters if self.normalize_observation else 0)
            actor_rollouts = self.num_iterations - effective_warmup
            if actor_rollouts <= 0 or actor_rollouts % self.actor_update_interval != 0:
                raise ValueError(
                    "post-warmup num_iterations must be positive and divisible by "
                    "actor_update_interval")

        # dual 消融：旧 critic_adam 保持可复现；empirical_pid 用真实完成轨迹控制 λ。
        self.dual_update_mode = str(getattr(args, 'dual_update_mode', 'critic_adam')).lower()
        if self.dual_update_mode not in {'critic_adam', 'empirical_pid'}:
            raise ValueError("dual_update_mode must be 'critic_adam' or 'empirical_pid'")
        # P-M8默认关闭。正interval让经验PID累计多个同策略rollout后只更新一次，
        # 避免Actor冻结期间controller先响应多次；critic_adam仍保留历史逐批时序。
        self.pid_update_interval = max(
            1, int(getattr(args, 'pid_update_interval', 1)))
        if self.pid_update_interval > 1 and not self.freeze_policy_updates:
            if self.dual_update_mode != 'empirical_pid':
                raise ValueError(
                    "pid_update_interval>1 requires dual_update_mode='empirical_pid'")
            if self.pid_update_interval != self.actor_update_interval:
                raise ValueError(
                    "pid_update_interval>1 must equal actor_update_interval so controller "
                    "and actor respond on the same rollout boundary")
            if self.outer_interval != 1:
                raise ValueError(
                    "pid_update_interval>1 currently requires outer_interval=1")
        self.dual_pid_signal = str(getattr(args, 'dual_pid_signal', 'outage')).lower()
        if self.dual_pid_signal not in {'outage', 'cost_quantile'}:
            raise ValueError("dual_pid_signal must be 'outage' or 'cost_quantile'")
        # 控制器可瞄准比真实 alpha 更保守的概率，吸收 window 估计误差和一次
        # PPO update 的闭环相位滞后。None 精确退化为旧 window_prob-q_alpha。
        raw_pid_target = getattr(args, 'pid_target_prob', None)
        self.pid_target_prob = (
            self.q_alpha if raw_pid_target is None else float(raw_pid_target))
        if not 0.0 <= self.pid_target_prob <= self.q_alpha:
            raise ValueError("pid_target_prob must lie in [0, q_alpha]")
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

        # reward critic 始终使用 τ_i=(i+0.5)/N；self.taus 名称保留给旧代码/测试。
        self.taus = torch.tensor(
            [(i + 0.5) / self.num_quantiles for i in range(self.num_quantiles)],
            dtype=torch.float32, device=self.device)
        # cost critic 可用同一 uniform grid，或对 τ≈1-alpha 做 deterministic mixture 加密。
        # cost_cdf_weights 同时用于 CDF、分布矩与 target distribution 积分。
        self.cost_taus, self.cost_cdf_weights, self.cost_tau_density = (
            _build_cost_quantile_grid(
                self.num_quantiles,
                self.cost_quantile_grid_mode,
                self.cost_quantile_query_tau,
                self.cost_quantile_local_half_width,
                self.cost_quantile_local_fraction,
                self.device))
        local_low = self.cost_quantile_query_tau - self.cost_quantile_local_half_width
        local_high = self.cost_quantile_query_tau + self.cost_quantile_local_half_width
        self.cost_quantile_local_count = int(
            ((self.cost_taus >= local_low) & (self.cost_taus <= local_high)).sum().item())
        if self.cost_distribution_model == 'iqn':
            # IQN 的 CDF 实际用 dense default query grid；局部计数也必须报告该网格，
            # 不能继续显示训练用的旧 N=32 固定 head 数。
            iqn_query_taus = (
                torch.arange(
                    self.cost_iqn_query_quantiles,
                    dtype=torch.float32, device=self.device) + 0.5
            ) / self.cost_iqn_query_quantiles
            self.cost_quantile_local_count = int(
                ((iqn_query_taus >= local_low)
                 & (iqn_query_taus <= local_high)).sum().item())

        # -------------------- actor 两时间尺度优化器 (Adam, 与模板一致) --------------------
        self.actor_optimizer = Adam(self.actor.parameters(), 1.0, eps=1e-5)
        self.actor_scheduler = LambdaLR(
            self.actor_optimizer,
            lr_lambda=lambda k: lr_lambda(k, args.theta_a, args.theta_b, args.theta_c))

        # -------------------- 两个分布式 critic + target (reward / cost) --------------------
        hidden = getattr(args, 'critic_hidden', [256, 256])   # safety-gym 观测高维 → MLP 大一点
        if isinstance(hidden, str):
            hidden = [int(x) for x in hidden.split(',') if x.strip()]
        reward_cdim = self.state_dim + (1 if self.critic_step_feature else 0)
        cost_base_dim = (
            self.actor.lstm_size
            if self.cost_history_mode in {'actor_feature', 'cost_lstm'}
            else self.state_dim)
        cost_cdim = cost_base_dim + (1 if self.critic_step_feature else 0)
        # reward critic: ψ^r(s,a), 只用其均值 mean(ψ^r) 作 Q̂_m (目标优势)
        self.reward_critic = DistributionalCritic(
            reward_cdim, self.action_dim, self.num_quantiles,
            list(hidden)).to(self.device)
        self.reward_target_critic = DistributionalCritic(
            reward_cdim, self.action_dim, self.num_quantiles,
            list(hidden)).to(self.device)
        self.reward_target_critic.load_state_dict(self.reward_critic.state_dict())
        # cost critic 保持 action conditioning。qr 分支逐式保留历史构造/RNG；
        # iqn 分支把 τ 作为输入，训练采样和 CDF 查询点数可以彼此独立。
        if self.cost_distribution_model == 'qr':
            self.cost_critic = DistributionalCritic(
                cost_cdim, self.action_dim, self.num_quantiles,
                list(hidden)).to(self.device)
            self.cost_target_critic = DistributionalCritic(
                cost_cdim, self.action_dim, self.num_quantiles,
                list(hidden)).to(self.device)
        else:
            self.cost_critic = ImplicitQuantileCritic(
                cost_cdim, self.action_dim, list(hidden),
                num_cosines=self.cost_iqn_cosines,
                default_query_quantiles=self.cost_iqn_query_quantiles).to(self.device)
            self.cost_target_critic = ImplicitQuantileCritic(
                cost_cdim, self.action_dim, list(hidden),
                num_cosines=self.cost_iqn_cosines,
                default_query_quantiles=self.cost_iqn_query_quantiles).to(self.device)
        self.cost_target_critic.load_state_dict(self.cost_critic.state_dict())

        # IQN τ 使用独立 generator，绝不推进策略动作的全局 torch RNG。QR 默认
        # 分支不创建 generator，也不增加任何随机调用，支持旧 checkpoint exact 回归。
        self.cost_iqn_generator = None
        if self.cost_distribution_model == 'iqn':
            self.cost_iqn_generator = torch.Generator(device=self.device)
            self.cost_iqn_generator.manual_seed(self.cost_iqn_seed)

        # C-X2 仅在显式 crossfit 模式构造独立 peer critic。默认 None 不消费任何
        # RNG，保证 online/target 的网络初始化与首批动作逐位兼容。peer 使用独立
        # 随机初始化并只接收 fold-1 标签；主 cost_critic 只接收 fold-0 标签。
        self.cost_crossfit_critic = None
        if self.cost_actor_query_mode == 'crossfit':
            self.cost_crossfit_critic = DistributionalCritic(
                cost_cdim, self.action_dim, self.num_quantiles,
                list(hidden)).to(self.device)

        # C-H1 编码器放在 critic heads 初始化之后，保证 actor/reward critic 与
        # C-H0.5 在相同 seed 下不因额外网络提前消费 RNG 而改变初始参数。
        self.cost_history_encoder = None
        self.cost_target_history_encoder = None
        if self.cost_history_mode == 'cost_lstm':
            self.cost_history_encoder = RecurrentCostEncoder(
                observation_dim=self.state_dim + 1,
                action_dim=self.action_dim,
                hidden=self.recurrent_hidden,
                lstm_size=self.actor.lstm_size,
                lstm_skip=bool(getattr(args, 'lstm_skip', True))).to(self.device)
            self.cost_target_history_encoder = RecurrentCostEncoder(
                observation_dim=self.state_dim + 1,
                action_dim=self.action_dim,
                hidden=self.recurrent_hidden,
                lstm_size=self.actor.lstm_size,
                lstm_skip=bool(getattr(args, 'lstm_skip', True))).to(self.device)
            self.cost_target_history_encoder.load_state_dict(
                self.cost_history_encoder.state_dict())

        # 一个优化器统管双 critic 与可选 cost encoder；reward/cost loss 各自
        # backward 后累积梯度，最后统一 clip 和 optimizer.step。
        critic_parameters = (
            list(self.reward_critic.parameters())
            + list(self.cost_critic.parameters()))
        if self.cost_crossfit_critic is not None:
            # 两个 fold critic 共用一个 Adam step；下面的 loss 按 fold 样本比例
            # 聚合，保持总体 cost objective 仍是全批 transition mean。
            critic_parameters += list(self.cost_crossfit_critic.parameters())
        if self.cost_history_encoder is not None:
            critic_parameters += list(self.cost_history_encoder.parameters())
        self.critic_optimizer = Adam(
            critic_parameters, getattr(args, 'critic_lr', 1e-3), eps=1e-5)

        # GAE/PPO 模式额外维护 V_r(s)。cost critic、budget 与 dual 均保持原实现，
        # 因此短实验只替换 reward advantage 来源，不偷偷改风险约束链路。
        self.reward_value = None
        self.reward_value_optimizer = None
        if self.reward_actor_mode != 'distributional' and not self.recurrent_policy:
            self.reward_value = ScalarValueCritic(
                reward_cdim, hidden=list(hidden)).to(self.device)
            self.reward_value_optimizer = Adam(
                self.reward_value.parameters(), self.reward_value_lr, eps=1e-5)

        # direct CDF 是完全独立的Bernoulli critic/optimizer。只在显式开启时构造；
        # 默认None不消费RNG、不改变旧optimizer参数组，支持历史QR逐位回归。
        self.cost_exceedance_critic = None
        self.cost_exceedance_ema_critic = None
        self.cost_exceedance_optimizer = None
        if self.cost_cdf_estimator == 'direct':
            # 多一个网络通常会推进全局torch RNG，继而改变首批Gaussian action noise。
            # 保存并恢复host/CUDA状态，使候选和QR baseline的policy/environment轨迹
            # 在相同seed下仍可common-random-number严格配对。
            host_rng_state = torch.get_rng_state()
            direct_device = torch.device(self.device)
            cuda_rng_state = (
                torch.cuda.get_rng_state(direct_device)
                if direct_device.type == 'cuda' and torch.cuda.is_available()
                else None)
            try:
                self.cost_exceedance_critic = ExceedanceProbabilityCritic(
                    cost_cdim, self.action_dim, list(hidden),
                    budget_scale=self.cost_direct_cdf_budget_scale).to(self.device)
            finally:
                torch.set_rng_state(host_rng_state)
                if cuda_rng_state is not None:
                    torch.cuda.set_rng_state(cuda_rng_state, direct_device)
            self.cost_exceedance_optimizer = Adam(
                self.cost_exceedance_critic.parameters(),
                self.cost_direct_cdf_lr, eps=1e-5)
            if self.cost_direct_cdf_query_mode == 'ema':
                # deepcopy不执行随机初始化；EMA与online在step 0逐位一致，也不会推进
                # Gaussian action RNG。requires_grad=False保证它只由显式Polyak更新。
                self.cost_exceedance_ema_critic = copy.deepcopy(
                    self.cost_exceedance_critic).to(self.device)
                self.cost_exceedance_ema_critic.requires_grad_(False)
                self.cost_exceedance_ema_critic.eval()

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
        self.last_cdf_initial = 0.0                           # actor实际使用的cost-CDF
        self.last_cdf_smooth_initial = 0.0                    # quantile平滑或direct同值
        self.last_qr_cdf_initial = 0.0                        # 始终保留QR hard-CDF作对照
        self.last_qr_cdf_smooth_initial = 0.0                 # 始终保留QR sigmoid-CDF
        self.last_direct_online_cdf_initial = 0.0             # EMA实验的未平滑head对照
        self.last_pred_cost_mean = 0.0                        # cost-critic 估 E[C|s0]
        self.last_pred_cost_std = 0.0                         # cost-critic 估 std(C|s0)
        self.last_cost_quantile_crossing_fraction = 0.0       # 相邻 τ 输出违反单调性的比例
        self.last_risk_query_target_online_abs_mean = 0.0     # actor查询网络间风险CDF差
        self.last_risk_query_preupdate_postupdate_abs_mean = 0.0  # 同批QR前后CDF漂移
        self.last_cost_crossfit_peer_abs_mean = 0.0            # 两折critic同输入CDF分歧
        # replay 只保存少量 GPU tensor；原始 state 在使用时重新走当前 RMS，不能
        # 缓存旧归一化结果，否则 observation statistics 漂移会污染监督输入。
        self.cost_s0_replay = deque(maxlen=self.cost_s0_replay_batches)
        self.last_s0_holdout_pre = {}                          # 最新新批训练前校准
        self.last_s0_holdout_post = {}                         # 同一批critic更新后校准
        self.empirical_cost_window = deque(maxlen=self.pid_window_episodes)
        self.pid_i = float(self.lambda_min)                    # QCPO_refs 默认 Kp=Kd=0，仅积分项
        self.last_dual_raw_prob = 0.0                          # 当前 rollout 的经验 outage
        self.last_dual_window_prob = 0.0                       # 最近窗口经验 outage
        self.last_dual_cost_quantile = float(self.cost_limit)  # 窗口 Q_(1-ω)(C)
        self.last_dual_prob_gap = 0.0                          # window outage - 真实 ω
        self.last_dual_control_prob_gap = 0.0                  # window outage - PID safety setpoint
        self.last_dual_quantile_gap = 0.0                      # Q_(1-ω)(C) - d
        self.last_dual_control_error = 0.0                     # deadband 前的原始控制误差
        self.last_dual_filtered_error = 0.0                    # deadband 后实际积分误差
        self.last_pid_episode_scale = 1.0                      # 本次新增 episode / reference
        self.last_pid_effective_leak = 1.0                     # rho ** episode_scale
        self.last_pid_delta = 0.0                              # clip 后积分增量（不含 leak）
        self.last_pid_actual_delta = 0.0                       # 最终 I_state_new-I_state_old
        self.last_pid_proportional = 0.0                       # Kp * filtered_error
        self.last_pid_output = 0.0                             # clip(I_state + P)
        self.pid_update_events = 0                             # 实际执行经验PID的cadence事件数
        self.pid_rollouts_since_update = 0                     # 当前PID窗口已累计rollout数
        self.last_pid_update_due = 0.0                         # 本轮是否执行了PID响应
        self.last_pid_rollouts_accumulated = 0.0               # 本轮PID事件累计rollout数
        self.last_pid_update_batch_episodes = 0.0              # 最近事件合并的完整轨迹数

    # ============================================================ 主训练循环 (与模板一致) ============================================================
    def train(self):
        """每迭代: 采样(含cost) → (qcpo)刷 EMA → 内层双critic+actor → (外层)λ → 日志+调度。"""
        print(f"DQCACBetaGPU[CMDP]: env={self.env_name}, beta={self.beta}, omega={self.q_alpha}, "
              f"d(cost_limit)={self.cost_limit}, cost_gamma={self.cost_gamma}, episodic={self.episodic}, "
              f"step_feature={self.critic_step_feature}, N={self.num_quantiles}, B={self.num_envs}, T={self.n}, "
              f"cost_target={self.cost_target_mode}, cost_history={self.cost_history_mode}, "
              f"cost_model={self.cost_distribution_model}"
              f"/train{self.cost_iqn_train_quantiles}"
              f"/query{self.cost_iqn_query_quantiles}"
              f"/cos{self.cost_iqn_cosines}, "
              f"cost_time_weight={self.cost_critic_time_weighting}"
              f"/d{self.cost_critic_weight_discount:g}"
              f"/floor{self.cost_critic_weight_floor:g}, "
              f"s0_aux={self.cost_s0_aux_coef:g}"
              f"/replay{self.cost_s0_replay_batches}, "
              f"cost_mean_anchor={self.cost_mean_anchor_coef:g}"
              f"/scale{self.cost_mean_anchor_cost_scale:g}, "
              f"cost_output={self.cost_quantile_output}"
              f"/scale{self.cost_quantile_output_scale:g}, "
              f"shared_cost={self.cost_shared_backbone_coef:g}"
              f"/scale{self.cost_shared_backbone_cost_scale:g}"
              f"/k{self.cost_shared_backbone_huber_kappa:g}, "
              f"cost_cdf={self.cost_cdf_estimator}/{self.cost_cdf_mode}"
              f"/T{self.cost_cdf_temperature:g}"
              f"/lr{self.cost_direct_cdf_lr:g}"
              f"/query{self.cost_direct_cdf_query_mode}"
              f"/ema{self.cost_direct_cdf_ema_tau:g}"
              f"/bscale{self.cost_direct_cdf_budget_scale:g}, "
              f"cost_grid={self.cost_quantile_grid_mode}/{self.cost_quantile_prediction_weighting}"
              f"/local{self.cost_quantile_local_count}, "
              f"qr_target={self.quantile_target_reduction}/ref{self.quantile_loss_reference_samples}, "
              f"iters={self.num_iterations}, critic_updates/iter={self.updates_per_episode}, "
              f"actor_updates/iter={self.actor_updates_per_episode}"
              f"/interval{self.actor_update_interval}, reward_actor={self.reward_actor_mode}, "
              f"ppo_target_kl={self.ppo_target_kl:g}, "
              f"actor_cost_query={self.cost_actor_query_mode}, "
              f"arch={self.policy_arch}, dual={self.dual_update_mode}/{self.dual_pid_signal}"
              f"/interval{self.pid_update_interval}"
              f"/target{self.pid_target_prob:g}, "
              f"sum_norm={self.sum_norm}, policy_frozen={self.freeze_policy_updates}, "
              f"obs_stats_frozen={self.freeze_observation_stats}, "
              f"device={self.device}")

        # interval=1 时该 cache 始终为空，历史路径不增加任何 tensor 拼接或 RNG 调用。
        # interval>1 时只暂存 actor 所需的 on-policy 字段；critic 仍当轮立即训练。
        actor_rollout_cache = []
        # PID cache只保存每条完整轨迹的MC cost。interval=1继续走原update_dual(batch)
        # 分支；interval>1不会缓存state/action，因此显存开销只有O(B*interval)标量。
        pid_cost_cache = []
        for it in range(self.num_iterations):
            # ===== 1. 采样 + DQCAC 专属后处理 (cost budget / n-step / d,e) =====
            batch = self._rollout_vec()
            if self.cost_history_mode == 'cost_lstm':
                # EMA constraint baseline 在 critic update 前就会查询 cost feature；
                # 先用当前 encoder 刷新整段历史，后续每次 encoder step 后再刷新。
                self._refresh_cost_history_features(batch)

            # 新 rollout 在进入任何本轮 optimizer step 前就是 prequential holdout。
            # 先记录泛化，再把它加入可选 recent replay；诊断本身不采样、不会改 RNG。
            self.last_s0_holdout_pre = self._s0_prequential_calibration(batch)
            batch['_s0_holdout_pre'] = self.last_s0_holdout_pre
            self._append_cost_s0_replay(batch)

            # ===== 1b. qcpo: 刷新 EMA 归一化器 (σ_R / σ_c) =====
            if self.advantage_norm == 'qcpo':
                self._update_norm_stats(batch)
            # GAE 与 value target 每个 rollout 固定一次，供后续多个 epoch 共同使用。
            if self.reward_actor_mode != 'distributional':
                self._prepare_reward_gae(batch)
            # obs norm 首批只建立 moments；随后 PPO ratio 才比较同一批的 old/current policy。
            norm_warmup = self.obs_norm_warmup_iters if self.normalize_observation else 0
            in_warmup = it < max(self.warmup_iters, norm_warmup)

            # 此时 actor/critic/obs RMS 都仍是产生本批 rollout 的版本。若启用 checkpoint，
            # 必须在 dual、critic、PPO 任一步之前保存，保证快照与本批 reward/outage 对应。
            self._maybe_save_rollout_checkpoint(it, batch)

            # 经验 PID 不依赖尚未校准的 critic，并在 actor epochs 前更新，逐位对齐 QCPO_refs 时序。
            # P-M8把同一behavior actor的多批cost合并后只响应一次；第一批只累计，
            # lambda保持不动，第二批B40更新后的lambda立即供同边界Actor PPO使用。
            dual_updated = False
            self.last_pid_update_due = 0.0
            self.last_pid_rollouts_accumulated = 0.0
            self.last_pid_update_batch_episodes = 0.0
            if (not self.freeze_policy_updates and not in_warmup
                    and self.dual_update_mode == 'empirical_pid'
                    and it % self.outer_interval == 0):
                dual_batch = batch
                pid_update_due = True
                if self.pid_update_interval > 1:
                    pid_cost_cache.append(batch['disc_cost'].detach())
                    self.pid_rollouts_since_update = len(pid_cost_cache)
                    pid_update_due = (
                        len(pid_cost_cache) == self.pid_update_interval)
                    if pid_update_due:
                        # update_dual(empirical_pid)只读取disc_cost；沿episode维拼接得到
                        # B*interval个真实MC标签，并让episode-scaled leak/I保持样本时间轴。
                        dual_batch = {
                            'disc_cost': torch.cat(pid_cost_cache, dim=0),
                        }
                else:
                    self.pid_rollouts_since_update = 1

                self.last_pid_rollouts_accumulated = float(
                    self.pid_rollouts_since_update)

                if pid_update_due:
                    self.update_dual(dual_batch)
                    dual_updated = True
                    self.pid_update_events += 1
                    self.last_pid_update_due = 1.0
                    self.last_pid_update_batch_episodes = float(
                        dual_batch['disc_cost'].numel())
                    if self.pid_update_interval > 1:
                        pid_cost_cache.clear()
                    self.pid_rollouts_since_update = 0

            # interval>1 的每条 rollout 都预先抽取 K 组 risk-baseline action。抽样位置
            # 与历史 actor 首 epoch 使用相同全局 RNG 序列；即使本轮暂不更新 actor，
            # 也推进相同数量的随机数，使后续环境 action noise 可与 interval=1 配对。
            actor_update_due = (
                not self.freeze_policy_updates and not in_warmup)
            actor_batch = batch
            if actor_update_due and self.actor_update_interval > 1:
                self._precompute_actor_baseline_actions(batch)
                actor_rollout_cache.append(batch)
                self.actor_rollouts_since_update = len(actor_rollout_cache)
                actor_update_due = (
                    len(actor_rollout_cache) == self.actor_update_interval)
                actor_batch = (
                    self._merge_actor_rollout_batches(actor_rollout_cache)
                    if actor_update_due else None)
            elif actor_update_due:
                self.actor_rollouts_since_update = 1

            # ===== 2. 内层更新 (双 critic + actor) =====
            critic_info, actor_info = {}, {}
            # update_critic返回的是“本次”optimizer step的统计；过去这里每轮覆盖，
            # 因而日志中的grad_clip_fraction只代表最后一次更新，而不是全部更新的比例。
            # 保存Python标量不会保留计算图、消耗随机数或改变任一optimizer状态。
            critic_update_diagnostics = []
            actor_updated = False
            actor_updates_completed = 0
            actor_early_stopped = False

            def apply_actor_epoch():
                """执行一次允许的actor epoch，并统一维护early-stop/scheduler计数。"""
                nonlocal actor_info, actor_updated
                nonlocal actor_updates_completed, actor_early_stopped
                if actor_batch is None:
                    raise RuntimeError("actor epoch requested without an accumulated batch")
                actor_info = self.update_actor(actor_batch)
                # target-KL越界的probe epoch不执行optimizer.step；它和剩余epoch
                # 都不能推进actor scheduler，但critic更新预算保持不变。
                update_applied = bool(
                    actor_info.get('ppo/update_applied', 1.0))
                if update_applied:
                    actor_updated = True
                    actor_updates_completed += 1
                if bool(actor_info.get('ppo/early_stop', 0.0)):
                    actor_early_stopped = True

            for update_idx in range(self.updates_per_episode):
                actor_epoch_allowed = (
                    not self.freeze_policy_updates and not in_warmup
                    and actor_update_due
                    and update_idx < self.actor_updates_per_episode
                    and not actor_early_stopped)
                # C-X3只把首个actor epoch提前到任何current-batch QR step之前。
                # update_actor会在这一刻缓存risk weight；后续epoch即使位于critic
                # 更新后也复用缓存，因此当前批cost标签不会即时回灌同批actor。
                actor_before_critic = (
                    self.cost_actor_query_mode == 'preupdate'
                    and update_idx == 0 and actor_epoch_allowed)
                if actor_before_critic:
                    apply_actor_epoch()

                critic_info = self.update_critic(batch)
                # 标量 reward value 在warmup中也训练；actor仍由下面的in_warmup控制。
                # value每个epoch拟合同一批冻结λ-return，actor复用冻结advantage。
                if self.reward_actor_mode != 'distributional' and not self.recurrent_policy:
                    critic_info.update(self.update_reward_value(batch))
                critic_update_diagnostics.append({
                    'cost_grad_norm': float(critic_info['critic/cost_grad_norm']),
                    'cost_head_grad_norm': float(
                        critic_info['critic/cost_head_grad_norm']),
                    'cost_history_grad_norm': float(
                        critic_info['critic/cost_history_grad_norm']),
                    'joint_grad_norm': float(critic_info['critic/joint_grad_norm']),
                    'reward_grad_norm': float(critic_info['critic/reward_grad_norm']),
                    'grad_clipped': float(critic_info['critic/grad_clip_fraction']),
                    'cost_qr_loss': float(critic_info['critic/cost_qr_loss']),
                    'cost_mean_anchor_loss': float(
                        critic_info['critic/cost_mean_anchor_loss']),
                    'cost_objective_loss': float(
                        critic_info['critic/cost_objective_loss']),
                })
                if actor_epoch_allowed and not actor_before_critic:
                    apply_actor_epoch()
                if self.learning_steps % self.target_update_interval == 0:
                    self._soft_update_target()
                self.learning_steps += 1

            # 同一rollout被重复拟合时，first→last轨迹比单独记录最后一次更能识别
            # “当前batch loss下降，但跨rollout泛化变差”的重复更新过拟合。保留原有
            # critic/grad_clip_fraction的末次语义，新增字段才是全部update的真比例。
            if critic_update_diagnostics:
                diagnostic_sequences = {
                    key: np.asarray(
                        [item[key] for item in critic_update_diagnostics],
                        dtype=np.float64)
                    for key in ('cost_grad_norm', 'joint_grad_norm',
                                'reward_grad_norm', 'cost_head_grad_norm',
                                'cost_history_grad_norm', 'grad_clipped',
                                'cost_qr_loss', 'cost_mean_anchor_loss',
                                'cost_objective_loss')
                }
                cost_grad = diagnostic_sequences['cost_grad_norm']
                cost_head_grad = diagnostic_sequences['cost_head_grad_norm']
                cost_history_grad = diagnostic_sequences['cost_history_grad_norm']
                joint_grad = diagnostic_sequences['joint_grad_norm']
                reward_grad = diagnostic_sequences['reward_grad_norm']
                cost_loss = diagnostic_sequences['cost_qr_loss']
                cost_mean_loss = diagnostic_sequences['cost_mean_anchor_loss']
                cost_objective_loss = diagnostic_sequences['cost_objective_loss']
                critic_info.update({
                    'critic/update_count': float(len(critic_update_diagnostics)),
                    'critic/update_grad_clip_fraction': float(
                        diagnostic_sequences['grad_clipped'].mean()),
                    'critic/cost_grad_norm_first': float(cost_grad[0]),
                    'critic/cost_grad_norm_mean': float(cost_grad.mean()),
                    'critic/cost_grad_norm_max': float(cost_grad.max()),
                    'critic/cost_grad_norm_last': float(cost_grad[-1]),
                    'critic/cost_head_grad_norm_first': float(cost_head_grad[0]),
                    'critic/cost_head_grad_norm_mean': float(
                        cost_head_grad.mean()),
                    'critic/cost_head_grad_norm_max': float(
                        cost_head_grad.max()),
                    'critic/cost_head_grad_norm_last': float(cost_head_grad[-1]),
                    'critic/cost_history_grad_norm_first': float(
                        cost_history_grad[0]),
                    'critic/cost_history_grad_norm_mean': float(
                        cost_history_grad.mean()),
                    'critic/cost_history_grad_norm_max': float(
                        cost_history_grad.max()),
                    'critic/cost_history_grad_norm_last': float(
                        cost_history_grad[-1]),
                    'critic/joint_grad_norm_first': float(joint_grad[0]),
                    'critic/joint_grad_norm_mean': float(joint_grad.mean()),
                    'critic/joint_grad_norm_max': float(joint_grad.max()),
                    'critic/joint_grad_norm_last': float(joint_grad[-1]),
                    'critic/reward_grad_norm_mean': float(reward_grad.mean()),
                    'critic/cost_qr_loss_first': float(cost_loss[0]),
                    'critic/cost_qr_loss_mean': float(cost_loss.mean()),
                    'critic/cost_qr_loss_min': float(cost_loss.min()),
                    'critic/cost_qr_loss_last': float(cost_loss[-1]),
                    'critic/cost_mean_anchor_loss_first': float(
                        cost_mean_loss[0]),
                    'critic/cost_mean_anchor_loss_mean': float(
                        cost_mean_loss.mean()),
                    'critic/cost_mean_anchor_loss_min': float(
                        cost_mean_loss.min()),
                    'critic/cost_mean_anchor_loss_last': float(
                        cost_mean_loss[-1]),
                    'critic/cost_objective_loss_first': float(
                        cost_objective_loss[0]),
                    'critic/cost_objective_loss_mean': float(
                        cost_objective_loss.mean()),
                    'critic/cost_objective_loss_min': float(
                        cost_objective_loss.min()),
                    'critic/cost_objective_loss_last': float(
                        cost_objective_loss[-1]),
                })

            if self.cost_actor_query_mode == 'preupdate':
                # actor缓存发生在首个critic step之前；这里用同一实际动作和budget
                # 查询完成全部更新后的online critic，量化本来会即时灌回actor的漂移。
                preupdate_drift = self._measure_preupdate_query_drift(batch)
                actor_info[
                    'advantage/risk_query_preupdate_postupdate_abs_mean'] = preupdate_drift

            # configured epochs与实际optimizer steps必须同时记录；否则target-KL只看
            # 最终KL会误以为仍执行了固定8次更新。
            actor_info['training/actor_updates_completed'] = float(
                actor_updates_completed)
            actor_info['ppo/early_stop'] = float(actor_early_stopped)
            actor_info['ppo/target_kl'] = self.ppo_target_kl
            if actor_updated:
                self.actor_update_events += 1
            accumulated_rollouts = (
                len(actor_rollout_cache)
                if self.actor_update_interval > 1 and not in_warmup
                else int(actor_update_due))
            actor_info['training/actor_update_due'] = float(actor_update_due)
            actor_info['training/actor_rollouts_accumulated'] = float(
                accumulated_rollouts)
            actor_info['training/actor_batch_trajectories'] = float(
                accumulated_rollouts * self.num_envs if actor_update_due else 0)
            actor_info['training/actor_update_events'] = float(
                self.actor_update_events)
            if actor_update_due:
                # actor_info 在多个 epoch 间会被最新一次覆盖；first_epoch 字段缓存在
                # actor_batch 中，因此这里仍是更新前 ratio 的自检，而 KL/clip 是末 epoch。
                self.last_actor_first_epoch_ratio_max_error = float(
                    actor_info.get('ppo/first_epoch_ratio_max_error', 0.0))
                self.last_actor_update_batch_trajectories = float(
                    actor_info['training/actor_batch_trajectories'])
                self.last_actor_updates_completed = float(
                    actor_info['training/actor_updates_completed'])
                self.last_actor_approx_kl = float(
                    actor_info.get('ppo/approx_kl', 0.0))
                self.last_actor_clip_fraction = float(
                    actor_info.get('ppo/clip_fraction', 0.0))
                self.last_shared_cost_qr_loss = float(
                    actor_info.get('shared_cost/qr_loss', 0.0))
                self.last_shared_cost_mean_loss = float(
                    actor_info.get('shared_cost/mean_loss', 0.0))
                self.last_shared_cost_weighted_loss = float(
                    actor_info.get('shared_cost/weighted_loss', 0.0))
                self.last_shared_cost_feature_refresh_abs_mean = float(
                    actor_info.get('shared_cost/feature_refresh_abs_mean', 0.0))
                self.last_shared_cost_body_grad_norm = float(
                    actor_info.get('shared_cost/joint_body_grad_norm', 0.0))
                self.last_shared_cost_lstm_grad_norm = float(
                    actor_info.get('shared_cost/joint_lstm_grad_norm', 0.0))
                self.last_shared_cost_policy_head_grad_norm = float(
                    actor_info.get(
                        'shared_cost/joint_policy_head_grad_norm', 0.0))
                self.last_shared_cost_value_head_grad_norm = float(
                    actor_info.get(
                        'shared_cost/joint_value_head_grad_norm', 0.0))
                self.last_shared_cost_head_grad_present = float(
                    actor_info.get('shared_cost/cost_head_grad_present', 0.0))

            # 旧 critic_adam 路径保留 actor 后更新时序，保证历史实验可复现。
            if (not self.freeze_policy_updates and not in_warmup
                    and self.dual_update_mode == 'critic_adam'
                    and it % self.outer_interval == 0):
                self.update_dual(batch)
                dual_updated = True                            # warmup 时不推进 λ 的学习率时间轴

            # 最后一个actor epoch发生在最后一次critic刷新之后；post holdout必须查询
            # 当前共享表示，而不是最后一次head update前的陈旧feature。RMS此时仍冻结。
            if self.cost_shared_backbone_coef > 0.0:
                self._refresh_shared_actor_cost_features(batch)

            # 与 pre 指标使用完全相同的 s0/a0/真实回报；这里只改变 critic 版本。
            # 放在 actor.obs_rms 合并前，避免把输入统计变化混入 pre/post 差值。
            self.last_s0_holdout_post = self._s0_prequential_calibration(batch)
            batch['_s0_holdout_post'] = self.last_s0_holdout_post

            # recurrent policy 的 augmented-observation moments 必须在所有 PPO epoch
            # 完成后再合并；否则固定 old_logπ 与 current logπ 会使用不同输入变换。
            if (self.recurrent_policy and self.normalize_observation
                    and not self.freeze_observation_stats):
                # warmup 仍逐批建立初始 moments；正式 interval 窗口内则必须冻结 RMS，
                # 否则即使 actor 参数没 step，第二条 rollout 也不再来自同一 behavior policy。
                rms_batch = None
                if in_warmup or self.actor_update_interval == 1:
                    rms_batch = batch
                elif actor_update_due:
                    rms_batch = actor_batch
                if rms_batch is not None:
                    self.actor.update_obs_rms(rms_batch['actor_obs'])

            # due rollout 已把全部缓存轨迹用于同一次 PPO batch；在日志写入前保留
            # accumulated 数值，但立即释放 GPU tensor，避免进入下一 cadence 窗口。
            if actor_update_due:
                if self.actor_update_interval > 1:
                    actor_rollout_cache.clear()
                self.actor_rollouts_since_update = 0

            # ===== 4. 日志 + 调度 =====
            self._log(it, batch, critic_info, actor_info)
            if actor_updated:
                self.actor_scheduler.step()                    # 每个有 actor update 的 rollout 推进一步
            if dual_updated and self.dual_update_mode == 'critic_adam':
                self.lambda_scheduler.step()                   # 严格位于 lambda_optimizer.step() 之后

        if actor_rollout_cache:
            raise RuntimeError(
                "training ended with unused actor rollouts; check interval divisibility")
        if pid_cost_cache:
            raise RuntimeError(
                "training ended with unused PID costs; check interval divisibility")

    def _precompute_actor_baseline_actions(self, batch):
        """
        为 cadence cache 预抽 K 组 behavior-policy action，保持全局 RNG 消耗配对。

        这些 action 只在真正的 actor epoch 中送入当时最新的 cost critic；提前抽样
        不读取 critic、不改变梯度。每个元素形状为 [T*B,A]，最终堆成 [K,T*B,A]。
        """
        with torch.no_grad():
            means = batch['actor_mean'].reshape(-1, self.action_dim)
            log_stds = batch['actor_log_std'].reshape(-1, self.action_dim)
            batch['_actor_baseline_actions'] = torch.stack([
                self._sample_from_params(means, log_stds)
                for _ in range(self.num_action_samples)
            ], dim=0).detach()

    def _merge_actor_rollout_batches(self, batches):
        """
        沿环境维合并多个同策略 recurrent rollout，返回一次严格 on-policy PPO batch。

        rollout 的 flatten 顺序是 time-major (t*B+j)，不能直接在 dim=0 拼接；
        必须先恢复 [T,B,*]、沿 B 拼接，再重新 flatten。这样 h0/c0、old_logπ、
        GAE target、实际动作和预抽 baseline action 的 episode 对齐保持不变。
        """
        if len(batches) != self.actor_update_interval:
            raise ValueError("actor rollout cache length does not match configured interval")
        T, B = self.n, self.num_envs
        merged = {'_actor_num_envs': B * len(batches)}

        time_major_keys = (
            'actor_obs', 'prev_action', 'prev_reward', 'h0', 'c0',
            'actor_mean', 'actor_log_std')
        for key in time_major_keys:
            merged[key] = torch.cat([entry[key] for entry in batches], dim=1)

        flat_time_keys = (
            'states', 'actions', 'budgets', 'steps', 'e', 'old_log_probs',
            '_reward_advantage_raw', '_reward_advantage',
            '_reward_value_targets')
        for key in flat_time_keys:
            tensors = []
            for entry in batches:
                value = entry[key]
                tensors.append(value.reshape(T, B, *value.shape[1:]))
            combined = torch.cat(tensors, dim=1)
            merged[key] = combined.reshape(
                T * merged['_actor_num_envs'], *combined.shape[2:])

        baseline_actions = [
            entry['_actor_baseline_actions'].reshape(
                self.num_action_samples, T, B, self.action_dim)
            for entry in batches]
        merged['_actor_baseline_actions'] = torch.cat(
            baseline_actions, dim=2).reshape(
                self.num_action_samples,
                T * merged['_actor_num_envs'],
                self.action_dim)
        return merged

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

    def _inverse_transform_recurrent(self, tensor, batch_size):
        """
        把chunk主序 [seq_len,new_B,*] 精确还原为时间主序 [T,B,*]。

        _transform_recurrent先把每条episode切成连续chunk，再把chunk放到batch维；
        这里必须先恢复[B,T,*]再转置。直接reshape成[T,B,*]会把不同环境和
        时间块交错，cost标签虽不报shape错误，却会监督到错误的history feature。
        """
        sequence_length, chunk_batch = tensor.shape[:2]
        rest = tuple(tensor.shape[2:])
        total_steps = int(sequence_length * chunk_batch)
        B = int(batch_size)
        if B <= 0 or total_steps % B != 0:
            raise ValueError(
                "recurrent inverse transform requires total elements divisible by batch_size")
        horizon = total_steps // B
        return (tensor.transpose(0, 1).reshape(
            B, horizon, *rest).transpose(0, 1).contiguous())

    def _recompute_recurrent_actor_features(self, batch):
        """
        用当前actor参数和behavior chunk初始状态重算整条rollout的共享表示。

        返回值按原始time-major顺序展平为[T*B,H]。h0/c0仍取采样时保存的
        chunk入口状态，与PPO当前策略前向完全相同；这避免把每个chunk错误地当作
        零历史episode，也让critic head始终看到和当前policy backbone一致的特征。
        """
        B = int(batch.get('_actor_num_envs', self.num_envs))
        observations = self._transform_recurrent(batch['actor_obs'])
        prev_actions = self._transform_recurrent(batch['prev_action'])
        prev_rewards = self._transform_recurrent(batch['prev_reward'])
        hidden0 = self._transform_recurrent(
            batch['h0'])[0].unsqueeze(0).contiguous()
        cell0 = self._transform_recurrent(
            batch['c0'])[0].unsqueeze(0).contiguous()
        _means, _log_stds, _values, _final_state, features = self.actor(
            observations, prev_actions, prev_rewards, (hidden0, cell0),
            return_features=True)
        time_major = self._inverse_transform_recurrent(features, B)
        expected_shape = (self.n, B, self.actor.lstm_size)
        if tuple(time_major.shape) != expected_shape:
            raise RuntimeError(
                f"recomputed actor feature shape {tuple(time_major.shape)} "
                f"does not match {expected_shape}")
        return time_major.reshape(self.n * B, self.actor.lstm_size)

    @torch.no_grad()
    def _refresh_shared_actor_cost_features(self, batch):
        """
        刷新共享cost head使用的detach actor feature，并记录表示漂移。

        critic optimizer只训练action-conditioned head；因此每次head更新前必须用
        当前actor重算输入。若继续使用rollout时缓存的旧feature，actor更新后head会
        在陈旧表示上训练，下一轮共享backbone更新又在新表示上查询，制造额外off-policy
        feature drift。detach边界确保本函数本身不把critic梯度写入actor。
        """
        if self.cost_shared_backbone_coef <= 0.0:
            return
        if self.cost_history_mode != 'actor_feature':
            raise RuntimeError(
                "shared actor feature refresh requires cost_history_mode='actor_feature'")
        # critic-only steps不会改变actor；dirty=False时复用刚按当前actor刷新的feature，
        # 避免在剩余critic epochs中重复做整条LSTM前向。默认缺失视为dirty，保证
        # 每个新rollout第一次head更新前仍执行一次数值对齐刷新。
        if not bool(batch.get('_shared_cost_feature_dirty', True)):
            batch['_shared_cost_feature_refresh_abs_mean'] = 0.0
            batch['_shared_cost_feature_refresh_applied'] = 0.0
            return
        previous = batch.get('cost_feature')
        current = self._recompute_recurrent_actor_features(batch).detach()
        refresh_gap = 0.0
        if previous is not None:
            if tuple(previous.shape) != tuple(current.shape):
                raise RuntimeError("shared actor feature refresh shape mismatch")
            refresh_gap = float((current - previous).abs().mean().item())
        batch['cost_feature'] = current
        batch['_shared_cost_feature_refresh_abs_mean'] = refresh_gap
        batch['_shared_cost_feature_refresh_applied'] = 1.0
        batch['_shared_cost_feature_refresh_count'] = int(
            batch.get('_shared_cost_feature_refresh_count', 0)) + 1
        batch['_shared_cost_feature_dirty'] = False

    def _normalize_cost_history_observation(self, actor_observation):
        """
        用 actor 的 augmented-observation RMS 标准化独立 cost encoder 输入。

        RMS buffer 不含可训练参数；共享它只保证 actor/cost encoder 看到相同尺度，
        不会让 cost QR loss 写入 actor MLP/LSTM。normalize_observation=False 时原样返回。
        """
        if not self.normalize_observation:
            return actor_observation
        return self.actor.obs_rms(actor_observation)

    @torch.no_grad()
    def _refresh_cost_history_features(self, batch):
        """
        用当前独立 encoder 重算 rollout 每个历史位置的 detach cost feature。

        按 recurrent_seq_len 顺序扫过 episode，并在 chunk 间携带 hidden/cell；
        no_grad 路径不需要截断反向图，但采用同一分块可与训练数值逐段对拍。
        结果按时间主序展平到 [T*B,H]，供 actor risk、EMA 与日志查询复用。
        """
        if self.cost_history_mode != 'cost_lstm':
            return
        if self.cost_history_encoder is None:
            raise RuntimeError("cost_lstm mode requires cost_history_encoder")

        observations = batch['actor_obs']
        prev_actions = batch['prev_action']
        prev_rewards = batch['prev_reward']
        T, B = observations.shape[:2]
        hidden, cell = self.cost_history_encoder.initial_state(B, self.device)
        feature_chunks = []

        for begin in range(0, T, self.recurrent_seq_len):
            finish = min(begin + self.recurrent_seq_len, T)
            normalized_obs = self._normalize_cost_history_observation(
                observations[begin:finish])
            features, (hidden, cell) = self.cost_history_encoder(
                normalized_obs,
                prev_actions[begin:finish],
                prev_rewards[begin:finish],
                (hidden, cell))
            feature_chunks.append(features)

        batch['cost_feature'] = torch.cat(
            feature_chunks, dim=0).reshape(T * B, -1).detach()

    @torch.no_grad()
    def _initial_cost_history_feature(self, states, actor_features=None):
        """
        返回 s0 对应的 cost 条件特征。

        raw 不需要 feature；actor_feature 直接复用策略的零历史 feature；cost_lstm
        则用独立编码器在 previous cost/action/reward 全零的协议下计算一次。
        """
        if self.cost_history_mode == 'raw':
            return None
        if self.cost_history_mode == 'actor_feature':
            if actor_features is None:
                raise RuntimeError("actor_feature mode requires policy feature at s0")
            return actor_features
        if self.cost_history_encoder is None:
            raise RuntimeError("cost_lstm mode requires cost_history_encoder")

        B = states.shape[0]
        previous_cost = torch.zeros(B, 1, device=states.device)
        previous_action = torch.zeros(
            B, self.action_dim, device=states.device)
        previous_reward = torch.zeros(B, device=states.device)
        augmented_obs = torch.cat([states, previous_cost], dim=1)
        normalized_obs = self._normalize_cost_history_observation(
            augmented_obs.unsqueeze(0))
        initial_state = self.cost_history_encoder.initial_state(
            B, states.device)
        features, _final_state = self.cost_history_encoder(
            normalized_obs,
            previous_action.unsqueeze(0),
            previous_reward.unsqueeze(0),
            initial_state)
        return features[0]

    def _sample_initial_actions(self, states, return_features=False):
        """
        在 episode 初始零历史处采样，可选返回与动作严格对齐的 recurrent feature。

        该 helper 只供 s0 dual/CDF 查询；任意中间状态仍必须使用 rollout 保存的
        历史特征，不能用零 hidden 重新近似。
        """
        if not self.recurrent_policy:
            action = self._sample_actions(states)
            return (action, None) if return_features else action
        B = states.shape[0]
        prev_cost = torch.zeros(B, 1, device=states.device)
        prev_action = torch.zeros(B, self.action_dim, device=states.device)
        prev_reward = torch.zeros(B, device=states.device)
        h0, c0 = self.actor.initial_state(B, states.device)
        actor_obs = torch.cat([states, prev_cost], dim=1)
        means, log_stds, _value, _state, features = self.actor(
            actor_obs.unsqueeze(0), prev_action.unsqueeze(0),
            prev_reward.unsqueeze(0), (h0, c0), return_features=True)
        action = self._sample_from_params(means[0], log_stds[0])
        return (action, features[0]) if return_features else action

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
        hidden_in, cell_in, values, features, means_all, log_stds_all, log_probs = (
            [], [], [], [], [], [], [])
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
                means, log_stds, value, (next_hidden, next_cell), feature = self.actor(
                    augmented_obs.unsqueeze(0), prev_action.unsqueeze(0),
                    prev_reward.unsqueeze(0), (hidden, cell), return_features=True)
                means, log_stds, value, feature = (
                    means[0], log_stds[0], value[0], feature[0])
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
                features.append(feature)
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
            (terminal_mean, terminal_log_std, _terminal_value, _terminal_state,
             terminal_feature) = self.actor(
                terminal_obs.unsqueeze(0), prev_action.unsqueeze(0),
                prev_reward.unsqueeze(0), (hidden, cell), return_features=True)
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
            'actor_feature': torch.stack(features),
            'actor_mean': torch.stack(means_all),
            'actor_log_std': torch.stack(log_stds_all),
            'terminal_action': terminal_action,
            'terminal_feature': terminal_feature[0],
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
        if self.normalize_observation and not self.freeze_observation_stats:
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
        boot_actions, boot_cost_features = None, None
        if self.recurrent_policy:
            # recurrent target 使用同一 on-policy rollout 在完整历史下采到的 a_{t+N}。
            # 末尾索引 n 对应 _rollout_core 单独生成但未执行的 terminal action/feature。
            action_ext = torch.cat([Amat, roll['terminal_action'].unsqueeze(0)], dim=0)
            feature_ext = torch.cat([
                roll['actor_feature'], roll['terminal_feature'].unsqueeze(0)], dim=0)
            boot_actions = action_ext[boot_idx].reshape(n * B, -1)
            boot_cost_features = feature_ext[boot_idx].reshape(n * B, -1)
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
        if self.cost_actor_query_mode == 'crossfit':
            # index=t*B+j，因此 env j 的全部 T 个 transition 必须共享同一 fold。
            # fold 0 标签只训练主 critic，fold 1 标签只训练 peer；actor 查询相反
            # critic。不能逐 transition 随机拆分，否则同一轨迹标签会泄漏到两边。
            s0_fold = torch.arange(B, device=self.device, dtype=torch.long) % 2
            batch['_cost_crossfit_s0_fold'] = s0_fold
            batch['_cost_crossfit_fold'] = (
                s0_fold.unsqueeze(0).expand(n, B).reshape(n * B))
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
                # rollout feature 已由 no_grad 产生；显式 detach 表明 cost critic
                # 不会借共享输入把梯度写回 behavior actor。
                'actor_feature': roll['actor_feature'].reshape(n * B, -1).detach(),
                'actor_mean': roll['actor_mean'],
                'actor_log_std': roll['actor_log_std'],
                'boot_actions': boot_actions.detach(),
                'boot_cost_features': boot_cost_features.detach(),
            })
            if self.cost_history_mode == 'actor_feature':
                # 统一 key 让后续调用不必猜测 feature 来源；仍保留 actor_feature
                # 供行为策略诊断，且两者都已 detach。
                batch['cost_feature'] = batch['actor_feature']
        return batch

    # ============================================================ 初始状态校准 / recent replay ============================================================
    @torch.no_grad()
    def _s0_prequential_calibration(self, batch):
        """
        在当前 rollout 尚未用于本轮 critic 更新时，评估真实 a0 上的初始风险校准。

        这是 prequential holdout：样本来自刚完成的新 rollout，因此没有参与当前
        batch 的任何 optimizer step。与训练后的同批结果配对，可以区分跨 rollout
        泛化和“在同一批 s0 上重复拟合 20 次”造成的表面校准。整个函数复用实际
        a0，不采新动作、不消耗 RNG，也不改变默认训练轨迹。
        """
        sample_count = int(batch['disc_cost'].numel())           # 每条完整轨迹提供一个 s0
        initial_states = batch['s0']                             # [B,state_dim] 原始 observation
        initial_actions = batch['actions'][:sample_count]        # time-major 前 B 行就是 a0

        # 非 raw 历史路径使用 rollout 已保存并 detach 的零历史 feature；当前主实验
        # 是 raw，但诊断本身可覆盖 actor_feature/cost_lstm，便于发现各分支的泛化差异。
        initial_features = None
        if self.cost_history_mode != 'raw':
            if 'cost_feature' not in batch:
                raise RuntimeError("s0 calibration requires refreshed cost_feature")
            initial_features = batch['cost_feature'][:sample_count]
        initial_inputs = self._cost_inputs(
            initial_states, 0, initial_features)

        if self.cost_actor_query_mode == 'crossfit':
            # 当前s0属于哪个标签fold，就查询没有见过该fold标签的另一critic。
            quantiles = self._cost_actor_query_quantiles(
                initial_inputs, initial_actions,
                label_folds=batch['_cost_crossfit_s0_fold'])
        else:
            # online/target实验的holdout历史定义始终评估online critic，保持可比。
            quantiles = self._cost_quantiles(
                self.cost_critic, initial_inputs, initial_actions)
        qr_probability = self._cost_tail_probability(
            quantiles, float(self.cost_limit), mode='hard')
        predicted_probability = (
            self._direct_cost_tail_probability(
                initial_inputs, initial_actions, float(self.cost_limit))
            if self.cost_cdf_estimator == 'direct' else qr_probability)
        observed_event = (
            batch['disc_cost'] >= float(self.cost_limit)).to(torch.float32)
        predicted_mean, _predicted_std = self._cost_quantile_moments(quantiles)

        cdf_bias = predicted_probability.mean() - observed_event.mean()
        cost_mean_bias = predicted_mean.mean() - batch['disc_cost'].mean()
        metrics = {
            'cdf': float(predicted_probability.mean().item()),
            'truth': float(observed_event.mean().item()),
            'cdf_bias': float(cdf_bias.item()),
            'cdf_abs_error': float(cdf_bias.abs().item()),
            # Brier score 是逐状态 proper score；仅比较两个总体均值会掩盖错误排序。
            'brier': float(
                (predicted_probability - observed_event).pow(2).mean().item()),
            'pred_cost_mean': float(predicted_mean.mean().item()),
            'truth_cost_mean': float(batch['disc_cost'].mean().item()),
            'cost_mean_bias': float(cost_mean_bias.item()),
            'samples': float(sample_count),
        }
        if self.cost_cdf_estimator == 'direct':
            # 现有键始终代表驱动actor的selected estimator；qr_*在完全相同的
            # 新轨迹/实际a0上给出内部对照，不需要另跑一个环境样本。
            qr_bias = qr_probability.mean() - observed_event.mean()
            metrics.update({
                'qr_cdf': float(qr_probability.mean().item()),
                'qr_cdf_bias': float(qr_bias.item()),
                'qr_cdf_abs_error': float(qr_bias.abs().item()),
                'qr_brier': float(
                    (qr_probability - observed_event).pow(2).mean().item()),
            })
            if self.cost_direct_cdf_query_mode == 'ema':
                # selected键是EMA；显式保留online对照，验证EMA是否减少跨批遗忘。
                online_probability = self._direct_cost_tail_probability(
                    initial_inputs, initial_actions, float(self.cost_limit),
                    source='online')
                online_bias = online_probability.mean() - observed_event.mean()
                metrics.update({
                    'direct_online_cdf': float(
                        online_probability.mean().item()),
                    'direct_online_cdf_bias': float(online_bias.item()),
                    'direct_online_cdf_abs_error': float(
                        online_bias.abs().item()),
                    'direct_online_brier': float(
                        (online_probability - observed_event).pow(2).mean().item()),
                })
        return metrics

    @torch.no_grad()
    def _append_cost_s0_replay(self, batch):
        """
        把当前 rollout 的 (raw s0, behavior a0, 完整 MC cost) 加入有限 recent replay。

        只在辅助系数大于 0 时保存。state 保持原始尺度，后续每次 auxiliary update
        都通过当前 observation RMS 重新变换；这避免缓存旧归一化值造成伪分布漂移。
        """
        if self.cost_s0_aux_coef <= 0.0:
            return
        sample_count = int(batch['disc_cost'].numel())
        self.cost_s0_replay.append({
            'states': batch['s0'].detach().clone(),
            'actions': batch['actions'][:sample_count].detach().clone(),
            'targets': batch['disc_cost'].detach().clone(),
        })

    def _cost_s0_auxiliary_loss(self):
        """
        在 recent 初始样本上计算 action-conditioned cost quantile regression loss。

        返回未缩放 loss；update_critic 使用 ratio/(1+ratio) 与基础 cost loss 做
        凸组合，保持 cost objective 总权重为 1。MC scalar 重复到 N_target 列只为
        延续项目既有 QR target 尺度，不会把一个轨迹 realization 冒充 N 个样本。
        """
        if self.cost_s0_aux_coef <= 0.0 or not self.cost_s0_replay:
            return None, {
                'critic/cost_s0_aux_loss': 0.0,
                'critic/cost_s0_aux_scaled_loss': 0.0,
                'critic/cost_s0_replay_samples': 0.0,
                'critic/cost_s0_replay_batches': 0.0,
                'critic/cost_s0_target_mean': 0.0,
            }

        states = torch.cat(
            [entry['states'] for entry in self.cost_s0_replay], dim=0)
        actions = torch.cat(
            [entry['actions'] for entry in self.cost_s0_replay], dim=0)
        scalar_targets = torch.cat(
            [entry['targets'] for entry in self.cost_s0_replay], dim=0)

        # 参数校验已限制 auxiliary 为 raw+MC；输入在这里使用当前 RMS 和 t=0 feature。
        inputs = self._cost_inputs(states, 0, None)
        prediction = self._cost_quantiles(
            self.cost_critic, inputs, actions)
        targets = scalar_targets.unsqueeze(1).expand(
            -1, self.num_quantiles)
        loss = self._cost_quantile_huber_loss(prediction, targets)

        auxiliary_scale = (
            self.cost_s0_aux_coef / (1.0 + self.cost_s0_aux_coef))
        return loss, {
            'critic/cost_s0_aux_loss': float(loss.detach().item()),
            'critic/cost_s0_aux_scaled_loss': float(
                auxiliary_scale * loss.detach().item()),
            'critic/cost_s0_replay_samples': float(states.shape[0]),
            'critic/cost_s0_replay_batches': float(len(self.cost_s0_replay)),
            'critic/cost_s0_target_mean': float(scalar_targets.mean().item()),
        }

    def _cost_mean_anchor_scale(self, num_target_samples):
        '''
        把 QCPO_refs 的 mean-cost 系数换算到本实现 QR 的 target-sum 尺度。

        reference 对 pairwise QR 的全部维度取 mean，且cost先除以S=10；
        本实现QR在N_target维求和并使用raw cost。大残差区QR随cost线性缩放、
        MSE则二次缩放，故有效系数为coef*N_target*target_scale/S。N=32、
        legacy_sum、coef=.5、S=10时得到1.6，保持mean/QR相对权重。
        '''
        return (
            self.cost_mean_anchor_coef
            * float(num_target_samples)
            * self._quantile_target_scale(num_target_samples)
            / self.cost_mean_anchor_cost_scale)

    def _cost_mean_anchor_loss(
            self, prediction, targets, sample_weights=None):
        '''
        返回 QCPO_refs 式 0.5*(predicted_mean-target_mean)^2。

        QR uniform grid 逐行普通平均；query-mixture 使用相同 quadrature 权重，
        IQN 的训练 taus 来自 uniform sampling，因此普通样本平均仍是无偏积分。
        可选 transition 权重与 QR 的 risk-discount 测度完全相同。
        '''
        predicted_mean = self._cost_quantile_moments(prediction)[0]
        target_mean = self._cost_quantile_moments(targets)[0]
        per_transition = 0.5 * (predicted_mean - target_mean).pow(2)
        if sample_weights is None:
            return per_transition.mean()
        if sample_weights.numel() != prediction.shape[0]:
            raise ValueError(
                "sample_weights length must match mean-anchor batch size")
        return (
            per_transition * sample_weights.reshape(-1)).mean()

    def _shared_backbone_reference_cost_loss(
            self, actor_features, batch):
        """
        返回只训练共享actor MLP+LSTM的QCPO_refs尺度cost辅助目标。

        参考实现先把cost和quantiles都除以10，再对batch、prediction quantile和
        target quantile三维全部取mean。当前MC标签每个transition只有一个realization；
        把它重复N次后再取mean与只保留一列严格等价，因此这里直接计算[M,N]，
        避免无信息的[M,N,N]张量。可选risk-discount权重与主cost critic一致。

        前向期间临时关闭cost head参数的requires_grad。这样固定head仍对输入feature
        提供Jacobian，但autograd不会为head创建参数梯度；actor Adam只拥有共享骨干，
        critic Adam只拥有head，彻底避免同一参数被两个优化器交替写入。
        """
        if self.cost_shared_backbone_coef <= 0.0:
            zero = actor_features.new_zeros(())
            return zero, {
                'shared_cost/qr_loss': 0.0,
                'shared_cost/mean_loss': 0.0,
                'shared_cost/objective_loss': 0.0,
                'shared_cost/weighted_loss': 0.0,
            }
        expected_rows = int(batch['actions'].shape[0])
        if actor_features.ndim != 2 or actor_features.shape[0] != expected_rows:
            raise ValueError(
                "shared actor features must have shape [T*B, hidden_size]")
        if 'mc_cost' not in batch:
            raise RuntimeError("shared backbone cost loss requires MC cost targets")

        # actor_feature模式通常还拼接t/T；这里不能调用_cost_inputs，因为该函数
        # 对actor_feature有意detach。共享辅助路径必须保留feature到actor的计算图。
        cost_inputs = actor_features
        if self.critic_step_feature:
            step_feature = (
                batch['steps'].float() / self.n).reshape(-1, 1)
            cost_inputs = torch.cat([actor_features, step_feature], dim=1)

        # 清掉上一个critic step遗留的.grad后再冻结head。requires_grad状态在forward
        # 建图时决定；forward完成后恢复原状态不会让随后backward重新创建head梯度。
        cost_head_parameters = list(self.cost_critic.parameters())
        original_requires_grad = [
            parameter.requires_grad for parameter in cost_head_parameters]
        for parameter in cost_head_parameters:
            parameter.grad = None
            parameter.requires_grad_(False)
        try:
            prediction = self._cost_quantiles(
                self.cost_critic, cost_inputs, batch['actions'])
        finally:
            for parameter, requires_grad in zip(
                    cost_head_parameters, original_requires_grad):
                parameter.requires_grad_(requires_grad)

        scale = self.cost_shared_backbone_cost_scale
        scaled_prediction = prediction / scale
        scaled_target = batch['mc_cost'].detach() / scale
        pairwise_delta = scaled_target.unsqueeze(1) - scaled_prediction
        absolute_delta = pairwise_delta.abs()
        kappa = self.cost_shared_backbone_huber_kappa
        huber = torch.where(
            absolute_delta > kappa,
            kappa * (absolute_delta - 0.5 * kappa),
            0.5 * pairwise_delta.pow(2))
        tau = self.cost_taus.to(
            device=prediction.device, dtype=prediction.dtype).view(1, -1)
        quantile_weight = (
            tau - (pairwise_delta.detach() < 0).to(prediction.dtype)).abs()
        qr_per_transition = (
            quantile_weight * huber / kappa).mean(dim=1)

        # QCPO_refs中c_value_se本身含0.5，再乘cost_value_loss_coeff（正式值0.5）。
        mean_error = scaled_prediction.mean(dim=1) - scaled_target
        mean_per_transition = 0.5 * mean_error.pow(2)
        sample_weights = self._cost_critic_sample_weights(batch['steps'])
        if sample_weights is None:
            qr_loss = qr_per_transition.mean()
            mean_loss = mean_per_transition.mean()
        else:
            weights = sample_weights.reshape(-1)
            qr_loss = (qr_per_transition * weights).mean()
            mean_loss = (mean_per_transition * weights).mean()

        objective = qr_loss + self.cost_mean_anchor_coef * mean_loss
        weighted_loss = self.cost_shared_backbone_coef * objective
        return weighted_loss, {
            'shared_cost/qr_loss': float(qr_loss.detach().item()),
            'shared_cost/mean_loss': float(mean_loss.detach().item()),
            'shared_cost/objective_loss': float(objective.detach().item()),
            'shared_cost/weighted_loss': float(weighted_loss.detach().item()),
        }

    # ============================================================ Critic 更新 (双 QR-TD) ============================================================
    def _backward_recurrent_cost_loss(
            self, batch, cost_target, cost_sample_weights=None):
        """
        对独立 cost MLP+LSTM 做一次按时间顺序的 truncated-BPTT QR 更新。

        每个 chunk 包含 recurrent_seq_len 个连续 timestep 和全部 B 条 episode。
        chunk loss 乘元素占比后立即 backward；hidden/cell 只向后携带数值并 detach，
        因而优化目标仍是整批 transition mean，而反向图最多跨一个 chunk。
        """
        if self.cost_history_encoder is None:
            raise RuntimeError("cost_lstm update requires cost_history_encoder")

        observations = batch['actor_obs']
        prev_actions = batch['prev_action']
        prev_rewards = batch['prev_reward']
        actions = batch['actions'].reshape(
            self.n, self.num_envs, self.action_dim)
        states = batch['states'].reshape(
            self.n, self.num_envs, self.state_dim)
        steps = batch['steps'].reshape(self.n, self.num_envs)
        targets = cost_target.reshape(
            self.n, self.num_envs, self.num_quantiles)
        sample_weights = (
            None if cost_sample_weights is None else
            cost_sample_weights.reshape(self.n, self.num_envs))
        hidden, cell = self.cost_history_encoder.initial_state(
            self.num_envs, self.device)
        total_size = float(self.n * self.num_envs)
        loss_value = 0.0
        mean_loss_value = 0.0
        mean_anchor_scale = self._cost_mean_anchor_scale(
            cost_target.shape[-1])

        for begin in range(0, self.n, self.recurrent_seq_len):
            finish = min(begin + self.recurrent_seq_len, self.n)
            normalized_obs = self._normalize_cost_history_observation(
                observations[begin:finish])
            features, (next_hidden, next_cell) = self.cost_history_encoder(
                normalized_obs,
                prev_actions[begin:finish],
                prev_rewards[begin:finish],
                (hidden, cell))

            # 展平顺序仍是 time-major 的 t*B+j，与 target/action 原批索引一致。
            chunk_states = states[begin:finish].reshape(-1, self.state_dim)
            chunk_steps = steps[begin:finish].reshape(-1)
            chunk_features = features.reshape(-1, features.shape[-1])
            chunk_inputs = self._cost_inputs(
                chunk_states, chunk_steps, chunk_features)
            chunk_actions = actions[begin:finish].reshape(-1, self.action_dim)
            chunk_targets = targets[begin:finish].reshape(
                -1, self.num_quantiles)

            prediction = self._cost_quantiles(
                self.cost_critic, chunk_inputs, chunk_actions)
            chunk_sample_weights = (
                None if sample_weights is None else
                sample_weights[begin:finish].reshape(-1))
            chunk_loss = self._cost_quantile_huber_loss(
                prediction, chunk_targets, chunk_sample_weights)
            chunk_weight = float((finish - begin) * self.num_envs) / total_size
            if self.cost_mean_anchor_coef <= 0.0:
                # 默认关闭分支保留历史 backward 表达式和浮点运算顺序。
                (chunk_weight * chunk_loss).backward()
            else:
                chunk_mean_loss = self._cost_mean_anchor_loss(
                    prediction, chunk_targets, chunk_sample_weights)
                (chunk_weight * (
                    chunk_loss
                    + mean_anchor_scale * chunk_mean_loss)).backward()
                mean_loss_value += (
                    chunk_weight * float(chunk_mean_loss.item()))
            loss_value += chunk_weight * float(chunk_loss.item())

            # 先保存数值再释放当前 chunk 图；下个 chunk 继承历史但不跨边界反传。
            hidden = next_hidden.detach()
            cell = next_cell.detach()

        return loss_value, mean_loss_value

    def update_critic(self, batch):
        """
        更新 reward/cost 两个 QR-TD critic；可选 transition chunking 降低 N² 峰值显存。

        chunk 路径对每块 mean loss 乘 `chunk_size/total_size` 后 backward，所有块
        共享一次 zero_grad/clip/optimizer.step。因此它与整批 mean loss 的梯度定义
        相同，不会因为块数增加而偷偷放大学习率。
        """
        # 共享backbone每个actor epoch后都会变化；head更新前必须在no_grad下刷新
        # 当前表示。默认coef=0时函数立即返回，不增加一次actor forward。
        if self.cost_shared_backbone_coef > 0.0:
            self._refresh_shared_actor_cost_features(batch)
        states, actions = batch['states'], batch['actions']
        boot_states, boot_mask = batch['boot_states'], batch['boot_mask']
        steps, boot_steps = batch['steps'], batch['boot_steps']

        with torch.no_grad():                                 # target 分支不建立 autograd 图
            # MLP 可直接在 boot state 重采；recurrent 必须使用完整历史下保存的
            # on-policy boot action，否则这里会隐式把所有中间状态当作 episode 起点。
            boot_actions = (
                batch['boot_actions'] if self.recurrent_policy
                else self._sample_actions(boot_states))
            reward_boot_inputs = self._aug(boot_states, boot_steps)

            # 两条 target 都完整保存为 [T*B,N]；真正的 N² 张量只在下面 chunk 内产生。
            reward_next = self.reward_target_critic(reward_boot_inputs, boot_actions)
            reward_target = (
                batch['nstep_reward'].unsqueeze(1)
                + (self.gamma ** self.n_step)
                * boot_mask.unsqueeze(1) * reward_next)
            if self.cost_target_mode == 'nstep':
                # cost_lstm 首轮被参数校验限制为 MC；这里的 boot feature 仍只服务
                # raw/actor_feature 两条历史兼容路线。
                cost_boot_inputs = self._cost_inputs(
                    boot_states, boot_steps, batch.get('boot_cost_features'))
                cost_next = self._cost_quantiles(
                    self.cost_target_critic, cost_boot_inputs, boot_actions)
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

        reward_state_inputs = self._aug(states, steps)
        cost_sample_weights = self._cost_critic_sample_weights(steps)
        recurrent_cost = self.cost_history_mode == 'cost_lstm'
        cost_state_inputs = (
            None if recurrent_cost else self._cost_inputs(
                states, steps, batch.get('cost_feature')))
        total_size = int(states.shape[0])
        # 每个 critic update 为整批 transition 一次性采样 IQN τ；chunk 路径只切片，
        # 因而改变显存块大小不会改变某个 transition 使用的 τ 或训练目标。
        cost_prediction_taus = self._sample_cost_iqn_taus(total_size)
        chunk_size = self.critic_minibatch_size
        use_chunks = 0 < chunk_size < total_size
        self.critic_optimizer.zero_grad(set_to_none=True)
        # 默认值只服务日志；crossfit 分支会分别填入两个独立 critic 的 fold mean loss。
        crossfit_fold0_loss_value = 0.0
        crossfit_fold1_loss_value = 0.0
        crossfit_fold0_fraction = 0.0
        crossfit_fold1_fraction = 0.0
        cost_mean_loss_value = 0.0
        cost_mean_anchor_scale = self._cost_mean_anchor_scale(
            cost_target.shape[-1])

        # 辅助目标只改变 cost 监督在 transition 与 recent-s0 间的质量分配。
        # coef=0 时 s0_aux_loss=None，下面保留历史 backward 表达式逐位不变。
        s0_aux_loss, s0_aux_info = self._cost_s0_auxiliary_loss()
        if s0_aux_loss is None:
            cost_base_scale, cost_aux_scale = 1.0, 0.0
        else:
            cost_base_scale = 1.0 / (1.0 + self.cost_s0_aux_coef)
            cost_aux_scale = self.cost_s0_aux_coef * cost_base_scale

        if recurrent_cost:
            # reward critic 仍按 transition chunk 配置更新；cost loss 另按连续时间
            # chunk 做 TBPTT，二者梯度累积后只执行一次 joint optimizer step。
            reward_loss_value = 0.0
            if not use_chunks:
                reward_pred = self.reward_critic(
                    reward_state_inputs, actions)
                reward_loss = self._quantile_huber_loss(
                    reward_pred, reward_target)
                reward_loss.backward()
                reward_loss_value = float(reward_loss.item())
            else:
                for begin in range(0, total_size, chunk_size):
                    finish = min(begin + chunk_size, total_size)
                    weight = float(finish - begin) / float(total_size)
                    reward_pred = self.reward_critic(
                        reward_state_inputs[begin:finish], actions[begin:finish])
                    reward_loss_chunk = self._quantile_huber_loss(
                        reward_pred, reward_target[begin:finish])
                    (weight * reward_loss_chunk).backward()
                    reward_loss_value += (
                        weight * float(reward_loss_chunk.item()))
            cost_loss_value, cost_mean_loss_value = (
                self._backward_recurrent_cost_loss(
                    batch, cost_target, cost_sample_weights))
        elif not use_chunks:
            # 默认历史路径：构造完整 [T*B,N,N] pairwise error，一次 backward。
            reward_pred = self.reward_critic(reward_state_inputs, actions)
            reward_loss = self._quantile_huber_loss(reward_pred, reward_target)
            if self.cost_actor_query_mode == 'crossfit':
                # fold 0/1 都按完整环境轨迹划分。每个 critic 的 QR loss 只看自己
                # fold 的标签；按样本占比加权后，二者之和仍等于全批 transition mean，
                # 不会因为多一个网络把 cost objective 或 joint clip 尺度直接翻倍。
                folds = batch['_cost_crossfit_fold']
                fold0_mask = folds == 0
                fold1_mask = folds == 1
                if not bool(fold0_mask.any()) or not bool(fold1_mask.any()):
                    raise RuntimeError("crossfit critic update requires two non-empty folds")
                fold0_pred = self._cost_quantiles(
                    self.cost_critic,
                    cost_state_inputs[fold0_mask], actions[fold0_mask])
                fold1_pred = self._cost_quantiles(
                    self.cost_crossfit_critic,
                    cost_state_inputs[fold1_mask], actions[fold1_mask])
                fold0_loss = self._cost_quantile_huber_loss(
                    fold0_pred, cost_target[fold0_mask],
                    None if cost_sample_weights is None
                    else cost_sample_weights[fold0_mask])
                fold1_loss = self._cost_quantile_huber_loss(
                    fold1_pred, cost_target[fold1_mask],
                    None if cost_sample_weights is None
                    else cost_sample_weights[fold1_mask])
                crossfit_fold0_fraction = float(
                    fold0_mask.float().mean().item())
                crossfit_fold1_fraction = float(
                    fold1_mask.float().mean().item())
                cost_loss = (
                    crossfit_fold0_fraction * fold0_loss
                    + crossfit_fold1_fraction * fold1_loss)
                cost_mean_loss = None
                if self.cost_mean_anchor_coef > 0.0:
                    fold0_mean_loss = self._cost_mean_anchor_loss(
                        fold0_pred, cost_target[fold0_mask],
                        None if cost_sample_weights is None
                        else cost_sample_weights[fold0_mask])
                    fold1_mean_loss = self._cost_mean_anchor_loss(
                        fold1_pred, cost_target[fold1_mask],
                        None if cost_sample_weights is None
                        else cost_sample_weights[fold1_mask])
                    cost_mean_loss = (
                        crossfit_fold0_fraction * fold0_mean_loss
                        + crossfit_fold1_fraction * fold1_mean_loss)
                crossfit_fold0_loss_value = float(fold0_loss.item())
                crossfit_fold1_loss_value = float(fold1_loss.item())
            else:
                # 非crossfit分支保持单一cost critic、调用顺序与浮点表达式不变。
                cost_pred = self._cost_training_quantiles(
                    cost_state_inputs, actions, cost_prediction_taus)
                cost_loss = self._cost_quantile_huber_loss(
                    cost_pred, cost_target, cost_sample_weights,
                    prediction_taus=cost_prediction_taus)
                cost_mean_loss = (
                    self._cost_mean_anchor_loss(
                        cost_pred, cost_target, cost_sample_weights)
                    if self.cost_mean_anchor_coef > 0.0 else None)
            if s0_aux_loss is None and cost_mean_loss is None:
                # 两个可选目标都关闭时保持历史图和浮点运算顺序。
                (reward_loss + cost_loss).backward()
            elif s0_aux_loss is None:
                (reward_loss + cost_loss
                 + cost_mean_anchor_scale * cost_mean_loss).backward()
            elif cost_mean_loss is None:
                (reward_loss + cost_base_scale * cost_loss).backward()
                (cost_aux_scale * s0_aux_loss).backward()
            else:
                (reward_loss + cost_base_scale * (
                    cost_loss
                    + cost_mean_anchor_scale * cost_mean_loss)).backward()
                (cost_aux_scale * s0_aux_loss).backward()
            reward_loss_value = float(reward_loss.item())
            cost_loss_value = float(cost_loss.item())
            cost_mean_loss_value = float(
                0.0 if cost_mean_loss is None else cost_mean_loss.item())
        else:
            # 顺序分块避免额外 permutation 张量并保持可复现；每块图在 backward 后释放。
            reward_loss_value, cost_loss_value = 0.0, 0.0
            for begin in range(0, total_size, chunk_size):
                finish = min(begin + chunk_size, total_size)
                weight = float(finish - begin) / float(total_size)
                reward_pred = self.reward_critic(
                    reward_state_inputs[begin:finish], actions[begin:finish])
                chunk_prediction_taus = (
                    None if cost_prediction_taus is None
                    else cost_prediction_taus[begin:finish])
                cost_pred = self._cost_training_quantiles(
                    cost_state_inputs[begin:finish], actions[begin:finish],
                    chunk_prediction_taus)
                reward_loss_chunk = self._quantile_huber_loss(
                    reward_pred, reward_target[begin:finish])
                cost_loss_chunk = self._cost_quantile_huber_loss(
                    cost_pred, cost_target[begin:finish],
                    None if cost_sample_weights is None
                    else cost_sample_weights[begin:finish],
                    prediction_taus=chunk_prediction_taus)
                cost_mean_loss_chunk = (
                    self._cost_mean_anchor_loss(
                        cost_pred, cost_target[begin:finish],
                        None if cost_sample_weights is None
                        else cost_sample_weights[begin:finish])
                    if self.cost_mean_anchor_coef > 0.0 else None)
                if s0_aux_loss is None and cost_mean_loss_chunk is None:
                    # 两个可选目标都关闭时保持原 chunk backward 顺序。
                    (weight * (reward_loss_chunk + cost_loss_chunk)).backward()
                elif s0_aux_loss is None:
                    (weight * (
                        reward_loss_chunk + cost_loss_chunk
                        + cost_mean_anchor_scale
                        * cost_mean_loss_chunk)).backward()
                elif cost_mean_loss_chunk is None:
                    (weight * (
                        reward_loss_chunk
                        + cost_base_scale * cost_loss_chunk)).backward()
                else:
                    (weight * (
                        reward_loss_chunk
                        + cost_base_scale * (
                            cost_loss_chunk
                            + cost_mean_anchor_scale
                            * cost_mean_loss_chunk))).backward()
                reward_loss_value += weight * float(reward_loss_chunk.item())
                cost_loss_value += weight * float(cost_loss_chunk.item())
                if cost_mean_loss_chunk is not None:
                    cost_mean_loss_value += (
                        weight * float(cost_mean_loss_chunk.item()))
            if s0_aux_loss is not None:
                # recent-s0 图很小，只在全部 transition chunk 释放后反传一次。
                (cost_aux_scale * s0_aux_loss).backward()

        # 分开记录两个critic及cost head/history encoder的裁剪前梯度范数，
        # 诊断joint clip究竟由reward、quantile head还是循环表示主导。
        reward_parameters = list(self.reward_critic.parameters())
        cost_head_parameters = list(self.cost_critic.parameters())
        if self.cost_crossfit_critic is not None:
            # peer 参数也参与同一次joint norm/clip；否则日志会低报真实optimizer输入。
            cost_head_parameters += list(self.cost_crossfit_critic.parameters())
        cost_history_parameters = []
        if self.cost_history_encoder is not None:
            cost_history_parameters = list(self.cost_history_encoder.parameters())
        cost_parameters = cost_head_parameters + cost_history_parameters

        def gradient_norm(parameters):
            squared_norm = torch.zeros((), dtype=torch.float32, device=self.device)
            for parameter in parameters:
                if parameter.grad is not None:
                    squared_norm += parameter.grad.detach().float().pow(2).sum()
            return squared_norm.sqrt()

        reward_grad_norm = gradient_norm(reward_parameters)
        cost_head_grad_norm = gradient_norm(cost_head_parameters)
        cost_history_grad_norm = gradient_norm(cost_history_parameters)
        cost_grad_norm = gradient_norm(cost_parameters)
        joint_parameters = reward_parameters + cost_parameters
        joint_grad_norm = gradient_norm(joint_parameters)
        if self.critic_grad_clip and self.critic_grad_clip > 0:
            nn.utils.clip_grad_norm_(joint_parameters, self.critic_grad_clip)
        self.critic_optimizer.step()

        # Bernoulli查询head使用独立optimizer/clip；默认关闭时不执行forward/backward，
        # 因而不会改变QR/reward joint optimizer的参数、梯度或Adam时间轴。
        direct_cdf_info = {}
        if self.cost_exceedance_critic is not None:
            direct_cdf_info = self._update_direct_cost_cdf(
                batch, cost_state_inputs, cost_sample_weights)
        if recurrent_cost:
            # actor risk 与 rollout 日志必须使用 optimizer.step 后的当前 encoder。
            self._refresh_cost_history_features(batch)
        critic_info = {
            'critic/reward_qr_loss': reward_loss_value,
            'critic/cost_qr_loss': cost_loss_value,
            'critic/cost_mean_anchor_enabled': float(
                self.cost_mean_anchor_coef > 0.0),
            'critic/cost_mean_anchor_coef': self.cost_mean_anchor_coef,
            'critic/cost_mean_anchor_cost_scale': (
                self.cost_mean_anchor_cost_scale),
            'critic/cost_mean_anchor_scale': float(cost_mean_anchor_scale),
            'critic/cost_mean_anchor_loss': cost_mean_loss_value,
            'critic/cost_mean_anchor_scaled_loss': float(
                cost_mean_anchor_scale * cost_mean_loss_value),
            'critic/chunked_update': float(use_chunks),
            'critic/cost_recurrent_tbptt': float(recurrent_cost),
            'critic/shared_actor_feature_refresh_abs_mean': float(
                batch.get('_shared_cost_feature_refresh_abs_mean', 0.0)),
            'critic/shared_actor_feature_refresh_applied': float(
                batch.get('_shared_cost_feature_refresh_applied', 0.0)),
            'critic/shared_actor_feature_refresh_count': float(
                batch.get('_shared_cost_feature_refresh_count', 0)),
            'critic/cost_time_weighted': float(
                cost_sample_weights is not None),
            'critic/cost_time_weight_min': float(
                1.0 if cost_sample_weights is None
                else cost_sample_weights.min().item()),
            'critic/cost_time_weight_max': float(
                1.0 if cost_sample_weights is None
                else cost_sample_weights.max().item()),
            'critic/cost_time_weight_ess_fraction': float(
                1.0 if cost_sample_weights is None
                else cost_sample_weights.sum().pow(2).div(
                    cost_sample_weights.numel()
                    * cost_sample_weights.pow(2).sum()).item()),
            'critic/cost_target_mean': float(
                self._cost_quantile_moments(cost_target)[0].mean().item()),
            'critic/cost_target_is_mc': float(self.cost_target_mode == 'mc'),
            # 当前固定-N critic 的 target count 等于 num_quantiles；显式记录有效
            # loss scale，后续 profile 能区分 N 本身与优化尺度变化。
            'critic/quantile_target_scale': self._quantile_target_scale(
                self.num_quantiles),
            'critic/quantile_target_reference': float(
                self.quantile_loss_reference_samples),
            'critic/cost_iqn_tau_mean': float(
                0.0 if cost_prediction_taus is None
                else cost_prediction_taus.mean().item()),
            'critic/cost_iqn_tau_std': float(
                0.0 if cost_prediction_taus is None
                else cost_prediction_taus.std(unbiased=False).item()),
            'critic/cost_iqn_tau_min': float(
                0.0 if cost_prediction_taus is None
                else cost_prediction_taus.min().item()),
            'critic/cost_iqn_tau_max': float(
                0.0 if cost_prediction_taus is None
                else cost_prediction_taus.max().item()),
            'critic/reward_grad_norm': float(reward_grad_norm.item()),
            'critic/cost_head_grad_norm': float(cost_head_grad_norm.item()),
            'critic/cost_history_grad_norm': float(
                cost_history_grad_norm.item()),
            'critic/cost_grad_norm': float(cost_grad_norm.item()),
            'critic/joint_grad_norm': float(joint_grad_norm.item()),
            'critic/grad_clip_fraction': float(
                joint_grad_norm.item() > self.critic_grad_clip
                if self.critic_grad_clip and self.critic_grad_clip > 0 else 0.0),
            'critic/cost_s0_aux_enabled': float(s0_aux_loss is not None),
            'critic/cost_s0_base_scale': float(cost_base_scale),
            'critic/cost_s0_aux_scale': float(cost_aux_scale),
            'critic/cost_objective_loss': float(
                cost_base_scale * (
                    cost_loss_value
                    + cost_mean_anchor_scale * cost_mean_loss_value)
                + s0_aux_info['critic/cost_s0_aux_scaled_loss']),
        }
        if self.cost_actor_query_mode == 'crossfit':
            critic_info.update({
                'critic/cost_crossfit_enabled': 1.0,
                'critic/cost_crossfit_fold0_loss': crossfit_fold0_loss_value,
                'critic/cost_crossfit_fold1_loss': crossfit_fold1_loss_value,
                'critic/cost_crossfit_fold0_fraction': crossfit_fold0_fraction,
                'critic/cost_crossfit_fold1_fraction': crossfit_fold1_fraction,
            })
        critic_info.update(s0_aux_info)
        critic_info.update(direct_cdf_info)
        return critic_info

    def _update_direct_cost_cdf(
            self, batch, cost_state_inputs, cost_sample_weights=None):
        """
        用MC remaining-cost二元事件更新直接查询点概率critic。

        对每个transition，当前budget满足递推 b[t+1]=(b[t]-c[t])/gamma_c，
        因此标签 I{MC_cost_to_go[t] >= b[t]} 与该轨迹从初始状态是否超限等价。
        虽然同一trajectory的Bernoulli outcome相同，(state, action, budget, step)
        条件输入随时间变化；使用全部transition正是为了覆盖actor实际查询的区域。

        该函数与QR共享时间权重和可选transition chunk边界，但只共享输入数据，不
        共享参数或optimizer。每次调用始终只有一个Adam step，chunking不会放大学习率。
        """
        if self.cost_exceedance_critic is None:
            return {}
        if 'mc_cost' not in batch:
            raise RuntimeError("direct cost CDF requires per-transition mc_cost labels")
        if cost_state_inputs is None:
            raise RuntimeError("direct cost CDF requires materialized raw cost inputs")

        actions = batch['actions']
        budgets = batch['budgets']
        labels = (
            batch['mc_cost'] >= budgets).to(dtype=torch.float32).detach()
        total_size = int(labels.numel())
        chunk_size = self.critic_minibatch_size
        use_chunks = 0 < chunk_size < total_size

        # 每个chunk的mean乘chunk_size/total_size，等价于完整batch mean BCE。
        # BCEWithLogitsLoss避免先sigmoid再log在极端logit处产生数值下溢。
        self.cost_exceedance_optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        probability_sum = torch.zeros((), device=self.device)
        brier_sum = torch.zeros((), device=self.device)
        weighted_brier_sum = torch.zeros((), device=self.device)
        for begin in range(0, total_size, chunk_size if use_chunks else total_size):
            finish = min(
                begin + (chunk_size if use_chunks else total_size), total_size)
            logits = self.cost_exceedance_critic(
                cost_state_inputs[begin:finish],
                actions[begin:finish],
                budgets[begin:finish])
            label_chunk = labels[begin:finish]
            element_loss = F.binary_cross_entropy_with_logits(
                logits, label_chunk, reduction='none')
            weight_chunk = (
                None if cost_sample_weights is None
                else cost_sample_weights[begin:finish])
            chunk_loss = (
                element_loss.mean() if weight_chunk is None
                else (element_loss * weight_chunk).mean())
            objective_weight = float(finish - begin) / float(total_size)
            (objective_weight * chunk_loss).backward()
            loss_value += objective_weight * float(chunk_loss.detach().item())

            # 校准诊断使用optimizer.step前的pre-update预测，与上面的BCE严格同版本。
            with torch.no_grad():
                probability = torch.sigmoid(logits.detach())
                squared_error = (probability - label_chunk).pow(2)
                probability_sum += probability.sum()
                brier_sum += squared_error.sum()
                weighted_brier_sum += (
                    squared_error.sum() if weight_chunk is None
                    else (squared_error * weight_chunk).sum())

        direct_parameters = list(self.cost_exceedance_critic.parameters())
        squared_grad_norm = torch.zeros(
            (), dtype=torch.float32, device=self.device)
        for parameter in direct_parameters:
            if parameter.grad is not None:
                squared_grad_norm += (
                    parameter.grad.detach().float().pow(2).sum())
        direct_grad_norm = squared_grad_norm.sqrt()
        if self.cost_direct_cdf_grad_clip > 0.0:
            nn.utils.clip_grad_norm_(
                direct_parameters, self.cost_direct_cdf_grad_clip)
        self.cost_exceedance_optimizer.step()
        ema_parameter_gap = self._soft_update_direct_cost_cdf_ema()

        # 理论上每条trajectory的T个标签完全相同；非零说明budget或MC回报的
        # 时间索引/折扣定义接错，比只看最终BCE更能发现静默监督bug。
        label_matrix = labels.reshape(self.n, self.num_envs)
        label_inconsistency = (
            label_matrix != label_matrix[0].unsqueeze(0)).float().mean()
        return {
            'critic/cost_direct_cdf_enabled': 1.0,
            'critic/cost_direct_cdf_loss': loss_value,
            'critic/cost_direct_cdf_brier': float(
                (brier_sum / total_size).item()),
            'critic/cost_direct_cdf_weighted_brier': float(
                (weighted_brier_sum / total_size).item()),
            'critic/cost_direct_cdf_probability_mean': float(
                (probability_sum / total_size).item()),
            'critic/cost_direct_cdf_truth_mean': float(labels.mean().item()),
            'critic/cost_direct_cdf_bias': float(
                (probability_sum / total_size - labels.mean()).item()),
            'critic/cost_direct_cdf_grad_norm': float(
                direct_grad_norm.item()),
            'critic/cost_direct_cdf_grad_clip_fraction': float(
                direct_grad_norm.item() > self.cost_direct_cdf_grad_clip
                if self.cost_direct_cdf_grad_clip > 0.0 else 0.0),
            'critic/cost_direct_cdf_chunked_update': float(use_chunks),
            'critic/cost_direct_cdf_label_inconsistency_fraction': float(
                label_inconsistency.item()),
            'critic/cost_direct_cdf_query_is_ema': float(
                self.cost_direct_cdf_query_mode == 'ema'),
            'critic/cost_direct_cdf_ema_tau': self.cost_direct_cdf_ema_tau,
            'critic/cost_direct_cdf_ema_online_parameter_abs_mean': (
                ema_parameter_gap),
        }

    @torch.no_grad()
    def _soft_update_direct_cost_cdf_ema(self):
        """
        每个direct Adam step后对查询head做一次Polyak更新并返回参数平均差。

        tau=0.005、每rollout 20次更新时，有效新权重约为
        1-(1-0.005)^20=0.0954；query参数约跨10个rollout低通，而online仍可
        快速拟合监督标签。online模式直接返回0，保持C-DCF1运算路径不变。
        """
        if self.cost_exceedance_ema_critic is None:
            return 0.0

        # EMA参数不在optimizer中；lerp_执行 θema←(1-tau)θema+tau·θonline。
        absolute_gap = torch.zeros((), device=self.device)
        parameter_count = 0
        for ema_parameter, online_parameter in zip(
                self.cost_exceedance_ema_critic.parameters(),
                self.cost_exceedance_critic.parameters()):
            ema_parameter.lerp_(
                online_parameter, self.cost_direct_cdf_ema_tau)
            absolute_gap += (
                ema_parameter - online_parameter).abs().sum()
            parameter_count += int(ema_parameter.numel())
        return float((absolute_gap / max(parameter_count, 1)).item())

    def _sample_cost_iqn_taus(self, batch_size):
        """
        为一次 cost update 采样 [M,N_train] uniform τ，使用独立 RNG。

        torch.rand 理论区间含 0；clamp 到机器 epsilon 只排除 IQN 不定义的端点，
        对连续 Uniform 积分没有可测偏差。QR 返回 None，保持历史调用与 RNG exact。
        """
        if self.cost_distribution_model != 'iqn':
            return None
        values = torch.rand(
            int(batch_size), self.cost_iqn_train_quantiles,
            dtype=torch.float32, device=self.device,
            generator=self.cost_iqn_generator)
        epsilon = torch.finfo(values.dtype).eps
        return values.clamp_(epsilon, 1.0 - epsilon)

    def _cost_training_quantiles(self, inputs, actions, prediction_taus):
        """
        统一 cost training forward；查询路径经同一适配器使用 critic 默认 dense grid。

        QR 没有 τ 输入，逐式保留旧 forward。IQN 必须显式接收本 update 已冻结的
        per-transition τ；若遗漏就会误用 query grid 并把训练样本数改成 128。
        """
        if self.cost_distribution_model == 'iqn':
            if prediction_taus is None:
                raise RuntimeError("IQN cost training requires sampled prediction_taus")
            return self._cost_quantiles(
                self.cost_critic, inputs, actions, prediction_taus)
        return self._cost_quantiles(self.cost_critic, inputs, actions)

    def _transform_cost_quantiles(self, raw_quantiles):
        """
        把cost head的无约束输出映射到当前raw-cost物理单位。

        linear直接返回同一Tensor对象，不改变默认计算图或浮点运算。exp精确对应
        QCPO_refs的正值参数化，并乘回其训练前除掉的cost scale。softplus用log(2)
        归一化，使零logit同样映射到scale，便于只消融尾部梯度/数值稳定性。
        """
        if self.cost_quantile_output == 'linear':
            return raw_quantiles
        if self.cost_quantile_output == 'exp':
            return self.cost_quantile_output_scale * torch.exp(raw_quantiles)
        return (
            self.cost_quantile_output_scale
            * F.softplus(raw_quantiles) / float(np.log(2.0)))

    def _cost_quantiles(
            self, critic, inputs, actions, prediction_taus=None):
        """
        统一执行QR/IQN cost critic前向并应用输出单位适配。

        prediction_taus=None表示固定QR heads或IQN确定性查询网格；非None只用于IQN
        训练时已经冻结的每transition采样点。所有训练、target、actor和评估路径
        都经过本函数，避免只修online critic却漏掉target或独立评估。
        """
        raw_quantiles = (
            critic(inputs, actions)
            if prediction_taus is None else
            critic(inputs, actions, prediction_taus))
        return self._transform_cost_quantiles(raw_quantiles)

    def _quantile_target_scale(self, num_target_samples):
        """
        返回 QR target-sample 求和后的兼容缩放系数。

        输入 num_target_samples 是 y 的最后一维 N_target。legacy_sum 返回 1，
        因而默认路径与历史代码逐式相同；reference_mean 返回 N_ref/N_target，
        使 target Monte-Carlo 样本数只改变积分分辨率，不隐式改变 critic 梯度尺度。
        """
        if self.quantile_target_reduction == 'legacy_sum':
            return 1.0
        return float(self.quantile_loss_reference_samples) / float(num_target_samples)

    def _cost_critic_sample_weights(self, steps):
        """
        返回 mean=1 的 cost transition 权重；uniform 返回 None 保留旧运算路径。

        risk_discount 的 raw weight=max(discount^t, floor)。归一化只执行一次，
        chunk/TBPTT 路径随后切片并按 chunk size 聚合，数学上仍是整批加权均值。
        """
        if self.cost_critic_time_weighting == 'uniform':
            return None
        step_values = steps.to(dtype=torch.float32)
        discount = torch.tensor(
            self.cost_critic_weight_discount,
            dtype=step_values.dtype, device=step_values.device)
        raw_weights = torch.pow(discount, step_values)
        if self.cost_critic_weight_floor > 0.0:
            raw_weights = raw_weights.clamp_min(
                self.cost_critic_weight_floor)
        return raw_weights / raw_weights.mean().clamp_min(1e-12)

    def _cost_quantile_huber_loss(
            self, psi, y, sample_weights=None, prediction_taus=None):
        """
        用 cost-only τ/importance 配置计算 QR loss。

        target quantiles 表示随机变量分布积分，所以 query grid 时始终用 1/g 权重；
        prediction heads 是否加权由 query_focused/importance 消融决定。MC target
        虽然每列相同，也走同一接口，确保切回 n-step 时不会悄悄改变概率语义。
        """
        prediction_weights = (
            self.cost_cdf_weights
            if self.cost_quantile_prediction_weighting == 'importance' else None)
        target_weights = (
            self.cost_cdf_weights
            if self.cost_quantile_grid_mode == 'query_mixture' else None)
        selected_taus = (
            prediction_taus
            if self.cost_distribution_model == 'iqn' else self.cost_taus)
        if self.cost_distribution_model == 'iqn':
            # uniform random τ 已直接按目标测度采样，不再附加固定-grid importance。
            prediction_weights = None
            target_weights = None
        return self._quantile_huber_loss(
            psi, y, taus=selected_taus,
            prediction_weights=prediction_weights,
            target_weights=target_weights,
            sample_weights=sample_weights)

    def _quantile_huber_loss(self, psi, y, taus=None,
                              prediction_weights=None, target_weights=None,
                              sample_weights=None):
        """
        计算 pairwise Quantile Huber Loss ρ^κ_τ(u)。

        psi 为 [M,N_pred]，y 为 [M,N_target]。默认参数严格复现历史
        sum(target)→mean(prediction)→mean(batch)；非均匀 cost grid 可分别对
        target distribution 和 prediction objective 提供归一化到和为 1 的权重。
        """
        selected_taus = self.taus if taus is None else taus
        if selected_taus.ndim == 1:
            if selected_taus.numel() != psi.shape[-1]:
                raise ValueError("taus length must match prediction quantile count")
        elif selected_taus.ndim == 2:
            if selected_taus.shape != psi.shape:
                raise ValueError(
                    "per-transition taus shape must exactly match prediction quantiles")
        else:
            raise ValueError("taus must have rank 1 or 2")
        u = y.unsqueeze(1) - psi.unsqueeze(2)                 # [M,N_pred,N_target]
        abs_u = u.abs()                                       # Huber 分段判断使用 |u|
        huber = torch.where(
            abs_u <= self.huber_kappa,
            0.5 * u.pow(2),
            self.huber_kappa * (abs_u - 0.5 * self.huber_kappa))
        tau_view = (
            selected_taus.view(1, -1, 1)
            if selected_taus.ndim == 1 else selected_taus.unsqueeze(-1))
        weight = (tau_view - (u.detach() < 0).float()).abs()  # 不对 indicator 求导
        pairwise_loss = weight * huber / self.huber_kappa    # [M,N_pred,N_target]

        # target_weights 估计 uniform-τ 下的 target distribution expectation；
        # 乘 N_target 后继续使用既有 N_ref/N_target 尺度开关。
        if target_weights is None:
            target_sum = pairwise_loss.sum(dim=2)             # 历史 exact 分支
        else:
            if target_weights.numel() != y.shape[-1]:
                raise ValueError("target_weights length must match target quantile count")
            target_sum = (
                pairwise_loss * target_weights.view(1, 1, -1)).sum(dim=2)
            target_sum = target_sum * float(y.shape[-1])
        if self.quantile_target_reduction == 'reference_mean':
            target_sum = target_sum * self._quantile_target_scale(y.shape[-1])

        # query_focused 继续对局部加密 prediction heads 等权；importance route
        # 则恢复 uniform-τ 积分。双 None 分支保留旧 mean 的运算顺序。
        if prediction_weights is None and sample_weights is None:
            return target_sum.mean(dim=1).mean()
        if prediction_weights is None:
            per_transition = target_sum.mean(dim=1)
        else:
            if prediction_weights.numel() != psi.shape[-1]:
                raise ValueError(
                    "prediction_weights length must match prediction quantile count")
            per_transition = (
                target_sum * prediction_weights.view(1, -1)).sum(dim=1)
        if sample_weights is None:
            return per_transition.mean()
        if sample_weights.numel() != psi.shape[0]:
            raise ValueError(
                "sample_weights length must match transition batch size")
        return (per_transition * sample_weights.reshape(-1)).mean()

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
        n = self.n
        B = int(batch.get('_actor_num_envs', self.num_envs))

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

        shared_cost_enabled = self.cost_shared_backbone_coef > 0.0
        if shared_cost_enabled:
            # 同一次actor forward同时产生policy/value和可微history feature；这样cost
            # 辅助梯度与PPO/value看到完全相同的当前MLP+LSTM表示，不重复构建recurrent图。
            means, log_stds, value_pred, _final_state, actor_features = self.actor(
                observations, prev_actions, prev_rewards, (hidden0, cell0),
                return_features=True)
            actor_features = self._inverse_transform_recurrent(
                actor_features, B).reshape(n * B, self.actor.lstm_size)
            shared_cost_loss, shared_cost_info = (
                self._shared_backbone_reference_cost_loss(
                    actor_features, batch))
        else:
            # 默认关闭分支保留历史四元actor forward，保证旧checkpoint数值路径不变。
            means, log_stds, value_pred, _final_state = self.actor(
                observations, prev_actions, prev_rewards, (hidden0, cell0))
            shared_cost_loss = observations.new_zeros(())
            shared_cost_info = {
                'shared_cost/qr_loss': 0.0,
                'shared_cost/mean_loss': 0.0,
                'shared_cost/objective_loss': 0.0,
                'shared_cost/weighted_loss': 0.0,
            }

        # cost advantage 固定在 behavior policy：实际动作和 K 个 baseline 动作均具有
        # rollout 的完整历史条件。不能对中间 state 调 _sample_initial_actions。
        with torch.no_grad():
            states = batch['states']
            budgets = batch['budgets']
            steps = batch['steps']
            if '_risk_weight' not in batch:
                cost_inputs = self._cost_inputs(
                    states, steps, batch.get('cost_feature'))
                query_folds = batch.get('_cost_crossfit_fold')
                psi_cdf = self._cost_actor_query_probability(
                    cost_inputs, batch['actions'], budgets,
                    label_folds=query_folds)
                # target模式额外查询同刻online网络只作诊断，不采样、不参与梯度；
                # online默认直接返回0，保证历史路径不增加任何critic forward。
                batch['_risk_query_target_online_abs_mean'] = (
                    self._cost_actor_query_disagreement(
                        cost_inputs, batch['actions'], budgets, psi_cdf).detach())

                behavior_mean = batch['actor_mean'].reshape(n * B, self.action_dim)
                behavior_log_std = batch['actor_log_std'].reshape(n * B, self.action_dim)
                baseline_cdfs = []
                precomputed_actions = batch.get('_actor_baseline_actions')
                if precomputed_actions is not None:
                    expected_shape = (
                        self.num_action_samples, n * B, self.action_dim)
                    if tuple(precomputed_actions.shape) != expected_shape:
                        raise ValueError(
                            "precomputed actor baseline action shape mismatch: "
                            f"{tuple(precomputed_actions.shape)} != {expected_shape}")
                for sample_idx in range(self.num_action_samples):
                    baseline_action = (
                        precomputed_actions[sample_idx]
                        if precomputed_actions is not None else
                        self._sample_from_params(behavior_mean, behavior_log_std))
                    baseline_cdfs.append(
                        self._cost_actor_query_probability(
                            cost_inputs, baseline_action, budgets,
                            label_folds=query_folds))
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
        if shared_cost_enabled:
            total_loss = (
                policy_loss
                + self.recurrent_value_loss_coef * value_loss
                + shared_cost_loss)
        else:
            # coef=0保持原表达式与加法顺序，避免默认回归因无效+0改变浮点图。
            total_loss = policy_loss + self.recurrent_value_loss_coef * value_loss

        self.actor_optimizer.zero_grad(set_to_none=True)
        approx_kl = ((ratio.detach() - 1.0) - log_ratio.detach()).mean()
        target_kl_hit = False
        if self.ppo_target_kl > 0.0:
            target_kl_hit = float(approx_kl.item()) > self.ppo_target_kl

        # 分组件范数只在共享消融中计算；它们是policy+value+cost的联合梯度，
        # 用于判断辅助目标是否淹没mu/value，而不额外执行autograd.grad或第二次backward。
        body_grad_norm = torch.zeros((), device=self.device)
        lstm_grad_norm = torch.zeros((), device=self.device)
        policy_head_grad_norm = torch.zeros((), device=self.device)
        value_head_grad_norm = torch.zeros((), device=self.device)
        cost_head_grad_present = 0.0

        def component_grad_norm(parameters):
            squared_norm = torch.zeros(
                (), dtype=torch.float32, device=self.device)
            for parameter in parameters:
                if parameter.grad is not None:
                    squared_norm += parameter.grad.detach().float().pow(2).sum()
            return squared_norm.sqrt()

        if target_kl_hit:
            # 当前forward已经测得behavior→current KL越界；本epoch不能再反传一步。
            # zero_grad已清除上个epoch残留梯度，后续critic optimizer与actor独立。
            grad_norm = torch.zeros((), device=self.device)
        else:
            total_loss.backward()
            if shared_cost_enabled:
                body_grad_norm = component_grad_norm(
                    self.actor.body.parameters())
                lstm_grad_norm = component_grad_norm(
                    self.actor.lstm.parameters())
                policy_head_grad_norm = component_grad_norm(
                    self.actor.mu.parameters())
                value_head_grad_norm = component_grad_norm(
                    self.actor.value.parameters())
                cost_head_grad_present = float(any(
                    parameter.grad is not None
                    for parameter in self.cost_critic.parameters()))
                if cost_head_grad_present:
                    raise RuntimeError(
                        "shared cost actor backward unexpectedly wrote cost-head gradients")
            if self.actor_grad_clip and self.actor_grad_clip > 0:
                grad_norm = nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.actor_grad_clip)
            else:
                grad_norm = torch.zeros((), device=self.device)
            self.actor_optimizer.step()
            if shared_cost_enabled:
                # 下一次critic head update必须先重算actor step后的共享表示。
                batch['_shared_cost_feature_dirty'] = True
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
            'shared_cost/enabled': float(shared_cost_enabled),
            'shared_cost/coefficient': self.cost_shared_backbone_coef,
            'shared_cost/cost_scale': self.cost_shared_backbone_cost_scale,
            'shared_cost/huber_kappa': self.cost_shared_backbone_huber_kappa,
            'shared_cost/feature_refresh_abs_mean': float(
                batch.get('_shared_cost_feature_refresh_abs_mean', 0.0)),
            'shared_cost/joint_body_grad_norm': float(body_grad_norm.item()),
            'shared_cost/joint_lstm_grad_norm': float(lstm_grad_norm.item()),
            'shared_cost/joint_policy_head_grad_norm': float(
                policy_head_grad_norm.item()),
            'shared_cost/joint_value_head_grad_norm': float(
                value_head_grad_norm.item()),
            'shared_cost/cost_head_grad_present': cost_head_grad_present,
            **shared_cost_info,
            'advantage/mean_adv_std': float(
                batch['_reward_advantage_raw'].std(unbiased=False).item()),
            'advantage/risk_adv_std': float(raw_adv_c.std(unbiased=False).item()),
            'advantage/risk_adv_abs_mean': float(raw_adv_c.abs().mean().item()),
            'advantage/risk_adv_nonzero_fraction': float(
                (raw_adv_c.abs() > 1e-6).float().mean().item()),
            'advantage/risk_query_target_online_abs_mean': float(
                batch['_risk_query_target_online_abs_mean'].item()),
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
            'ppo/approx_kl': float(approx_kl.item()),
            'ppo/target_kl': self.ppo_target_kl,
            'ppo/early_stop': float(target_kl_hit),
            'ppo/update_applied': float(not target_kl_hit),
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
                cost_inputs = self._aug(s, steps)
                query_folds = batch.get('_cost_crossfit_fold')
                psi_cdf = self._cost_actor_query_probability(
                    cost_inputs, a, b, label_folds=query_folds)
                batch['_risk_query_target_online_abs_mean'] = (
                    self._cost_actor_query_disagreement(
                        cost_inputs, a, b, psi_cdf).detach())
                v_m, v_c = self._estimate_baselines(
                    s, b, steps,
                    need_reward=self.reward_actor_mode == 'distributional',
                    cost_query_folds=query_folds)
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
        target_kl_hit = False
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

            # approx_kl 采用 Schulman 常用近似 (ratio-1)-log_ratio。正target
            # 在当前epoch backward前检查，因此越界probe不会再把policy推远一步。
            approx_kl = ((ratio.detach() - 1.0) - log_ratio.detach()).mean()
            if self.ppo_target_kl > 0.0:
                target_kl_hit = float(approx_kl.item()) > self.ppo_target_kl
            ppo_info = {
                'ppo/ratio_mean': float(ratio.detach().mean().item()),
                'ppo/ratio_std': float(ratio.detach().std(unbiased=False).item()),
                'ppo/clip_fraction': float(
                    ((ratio.detach() - 1.0).abs() > self.ppo_ratio_clip).float().mean().item()),
                'ppo/approx_kl': float(approx_kl.item()),
                'ppo/target_kl': self.ppo_target_kl,
                'ppo/early_stop': float(target_kl_hit),
                'ppo/update_applied': float(not target_kl_hit),
            }
        else:
            actor_loss = -(log_probs * combined_weight).mean() - self.entropy_coef * entropy

        self.actor_optimizer.zero_grad(set_to_none=True)
        if target_kl_hit:
            actor_grad_norm = torch.zeros((), device=self.device)
        else:
            actor_loss.backward()
            if self.actor_grad_clip and self.actor_grad_clip > 0:
                actor_grad_norm = nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.actor_grad_clip)
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
            'advantage/risk_adv_abs_mean': float(raw_adv_c.abs().mean().item()),
            'advantage/risk_adv_nonzero_fraction': float(
                (raw_adv_c.abs() > 1e-6).float().mean().item()),
            'advantage/risk_query_target_online_abs_mean': float(
                batch['_risk_query_target_online_abs_mean'].item()),
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
                a0, actor_feature0 = self._sample_initial_actions(
                    s0, return_features=True)
                cost_feature0 = self._initial_cost_history_feature(
                    s0, actor_feature0)
                cost_input0 = self._cost_inputs(s0, 0, cost_feature0)
                if self.cost_cdf_estimator == 'direct':
                    direct_probability = self._direct_cost_tail_probability(
                        cost_input0, a0, self.cost_limit)
                    p = float(direct_probability.mean().item())
                else:
                    psi0 = self._cost_quantiles(
                        self.cost_critic, cost_input0, a0)   # [B,N]
                    p = float(
                        self._cost_tail_probability(
                            psi0, self.cost_limit, mode='hard').mean().item())
            self.last_dual_prob = p
            self.last_dual_prob_gap = p - self.q_alpha
            # critic_adam 不是 empirical PID，继续严格优化真实 alpha，不应用 safety setpoint。
            self.last_dual_control_prob_gap = self.last_dual_prob_gap
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
        # prob_gap 始终报告真实约束违反；control_prob_gap 才使用保守 setpoint。
        # 默认 target=q_alpha 时两者逐位相同，所有历史配置保持不变。
        self.last_dual_prob_gap = window_prob - self.q_alpha
        self.last_dual_control_prob_gap = window_prob - self.pid_target_prob
        self.last_dual_quantile_gap = cost_quantile - self.cost_limit

        if self.dual_pid_signal == 'outage':
            control_error = self.last_dual_control_prob_gap
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
    def _cost_tail_probability(self, quantiles, budgets, mode=None):
        """
        在查询 budget 处把 cost quantiles 转成上尾概率 surrogate。

        hard 是历史 `(1/N) sum I{z_i>=b}`，分辨率受 N 限制；sigmoid 用
        `(1/N) sum sigmoid((z_i-b)/temperature)` 给查询点邻域连续权重。平滑只在
        actor risk advantage/其 baseline/尺度统计中启用；日志与 empirical PID
        仍显式调用 hard，避免把 surrogate bias 误报为真实约束满足。
        """
        selected_mode = self.cost_cdf_mode if mode is None else str(mode).lower()
        if torch.is_tensor(budgets):
            budget_values = budgets.to(
                device=quantiles.device, dtype=quantiles.dtype)
            if budget_values.ndim == quantiles.ndim - 1:
                budget_values = budget_values.unsqueeze(-1)
        else:
            budget_values = float(budgets)
        if selected_mode == 'hard':
            tail_values = (quantiles >= budget_values).float()
        elif selected_mode == 'sigmoid':
            scaled_margin = (
                (quantiles - budget_values) / self.cost_cdf_temperature)
            tail_values = torch.sigmoid(scaled_margin)
        else:
            raise ValueError(f"unknown cost CDF mode: {selected_mode!r}")

        # uniform 默认路径继续直接 mean，保持历史 floating-point 运算顺序。
        if self.cost_quantile_grid_mode == 'uniform':
            return tail_values.mean(dim=-1)
        weights = self.cost_cdf_weights.to(
            device=quantiles.device, dtype=quantiles.dtype)
        return (tail_values * weights).sum(dim=-1)

    def _direct_cost_tail_probability(
            self, inputs, actions, budgets, source=None):
        """
        查询direct Bernoulli critic并把logit映射到[0,1]上尾概率。

        source=None读取配置选择的online/EMA；source='online'用于EMA消融的内部
        对照。接口不主动detach：online模式单元测试仍可检查head梯度；正式actor、
        dual和评估均在no_grad中，不会通过critic对action做确定性策略梯度。
        """
        if self.cost_exceedance_critic is None:
            raise RuntimeError("direct cost CDF critic is not initialized")

        # selected_source只路由网络，不改变输入、budget或概率定义。
        selected_source = (
            self.cost_direct_cdf_query_mode if source is None else str(source))
        if selected_source == 'online':
            selected_critic = self.cost_exceedance_critic
        elif selected_source == 'ema':
            selected_critic = self.cost_exceedance_ema_critic
            if selected_critic is None:
                raise RuntimeError(
                    "direct cost CDF EMA critic is not initialized")
        else:
            raise ValueError(
                f"unknown direct CDF source: {selected_source!r}")
        logits = selected_critic(inputs, actions, budgets)
        return torch.sigmoid(logits)

    def _cost_actor_query_probability(
            self, inputs, actions, budgets, mode=None, label_folds=None):
        """
        返回真正驱动actor的P(C_remaining>=budget|s,a,budget)。

        quantile默认路径保持原调用顺序：先按online/target/crossfit路由quantiles，
        再用hard/sigmoid积分。direct路径直接输出概率，mode对它没有数值意义；
        保留参数只是让日志/评估可共用接口。
        """
        if self.cost_cdf_estimator == 'direct':
            return self._direct_cost_tail_probability(
                inputs, actions, budgets)
        quantiles = self._cost_actor_query_quantiles(
            inputs, actions, label_folds=label_folds)
        return self._cost_tail_probability(
            quantiles, budgets, mode=mode)

    def _cost_quantile_moments(self, quantiles):
        """
        返回每行 cost distribution 的加权 mean/std。

        uniform 分支保留旧 mean 与 torch.std(unbiased=True)；query_mixture 用
        CDF quadrature 权重计算概率矩，避免局部密集 heads 把 mean/std 拉向查询区。
        """
        if self.cost_quantile_grid_mode == 'uniform':
            return quantiles.mean(dim=-1), quantiles.std(dim=-1)
        weights = self.cost_cdf_weights.to(
            device=quantiles.device, dtype=quantiles.dtype)
        mean = (quantiles * weights).sum(dim=-1)
        variance = (
            (quantiles - mean.unsqueeze(-1)).pow(2) * weights).sum(dim=-1)
        return mean, variance.clamp_min(0.0).sqrt()

    def _cost_inputs(self, states, steps, history_features=None):
        """
        构造 cost distribution critic 的条件输入。

        raw 路径逐式调用历史 `_aug(state,t)`；actor_feature 路径要求与每个 action
        同一历史位置的 φ_t，并在这里再次 detach，形成严格的半梯度边界。φ_t 已由
        actor 自己的 observation RMS/MLP/LSTM 产生，不能再经过 raw-state normalizer。
        """
        if self.cost_history_mode == 'raw':
            return self._aug(states, steps)
        if history_features is None:
            raise RuntimeError(
                f"cost_history_mode={self.cost_history_mode!r} requires aligned history features")
        # actor_feature 必须截断到策略的梯度；cost_lstm 则保留独立 encoder
        # 的计算图，使 cost QR loss 能通过 quantile head 反传到历史表示。
        features = (
            history_features.detach()
            if self.cost_history_mode == 'actor_feature' else history_features)
        if not self.critic_step_feature:
            return features
        if not torch.is_tensor(steps):
            steps = torch.full(
                (features.shape[0],), float(steps), device=features.device)
        step_feature = (steps.float() / self.n).reshape(-1, 1)
        return torch.cat([features, step_feature], dim=1)

    def _aug(self, states, steps):
        """先共享观测归一化，再按需追加 critic/value 的 t/T step feature。"""
        states = self._normalize_states(states)
        if not self.critic_step_feature:
            return states
        if not torch.is_tensor(steps):
            steps = torch.full((states.shape[0],), float(steps), device=states.device)
        sf = (steps.float() / self.n).reshape(-1, 1)
        return torch.cat([states, sf], dim=1)

    def _cost_actor_query_critic(self):
        """
        返回单网络actor-query critic；crossfit必须改用逐样本路由helper。

        online/preupdate都查询online网络，区别只在外层调用时序；target使用
        Polyak网络。crossfit的每行样本可能需要不同网络，若误调用本函数会把
        整批送入一个critic并破坏out-of-fold语义，
        因此显式报错而不静默选择主网络。
        """
        if self.cost_actor_query_mode == 'target':
            return self.cost_target_critic
        if self.cost_actor_query_mode == 'crossfit':
            raise RuntimeError(
                "crossfit actor query requires _cost_actor_query_quantiles")
        return self.cost_critic

    @torch.no_grad()
    def _measure_preupdate_query_drift(self, batch):
        """
        比较缓存的pre-update风险CDF与本批全部critic step后的online CDF。

        两次查询复用相同states/actions/budgets，不采样也不改变训练；差值只来自
        current-batch QR监督更新。若actor因warmup/冻结没有创建缓存，则返回0。
        """
        if self.cost_actor_query_mode != 'preupdate' or '_risk_cdf' not in batch:
            self.last_risk_query_preupdate_postupdate_abs_mean = 0.0
            return 0.0
        cost_inputs = self._cost_inputs(
            batch['states'], batch['steps'], batch.get('cost_feature'))
        updated_quantiles = self._cost_quantiles(
            self.cost_critic, cost_inputs, batch['actions'])
        updated_cdf = self._cost_tail_probability(
            updated_quantiles, batch['budgets'])
        drift = (batch['_risk_cdf'] - updated_cdf).abs().mean()
        self.last_risk_query_preupdate_postupdate_abs_mean = float(drift.item())
        return self.last_risk_query_preupdate_postupdate_abs_mean

    def _cost_actor_query_quantiles(self, inputs, actions, label_folds=None):
        """
        返回actor使用的cost quantiles；crossfit按标签fold查询相反critic。

        label_folds=0表示该行真实标签训练主critic，所以查询peer；fold=1反之。
        online/target忽略fold并保持原来的一次完整forward。函数只用于无梯度的
        actor/normalizer/校准查询，不参与critic参数反向传播。
        """
        if self.cost_actor_query_mode != 'crossfit':
            return self._cost_quantiles(
                self._cost_actor_query_critic(), inputs, actions)
        if self.cost_crossfit_critic is None:
            raise RuntimeError("crossfit peer critic is not initialized")
        if label_folds is None:
            raise RuntimeError("crossfit actor query requires aligned label_folds")
        folds = label_folds.reshape(-1).to(device=inputs.device)
        if folds.numel() != inputs.shape[0]:
            raise ValueError("crossfit label_folds must align with query rows")
        fold0_mask = folds == 0
        fold1_mask = folds == 1
        if not bool(fold0_mask.any()) or not bool(fold1_mask.any()):
            raise RuntimeError("crossfit actor query requires two non-empty folds")

        # 先分配最终[T*B,N]，再把两个半批forward写回原time-major位置。
        quantiles = torch.empty(
            inputs.shape[0], self.num_quantiles,
            dtype=inputs.dtype, device=inputs.device)
        quantiles[fold0_mask] = self._cost_quantiles(
            self.cost_crossfit_critic,
            inputs[fold0_mask], actions[fold0_mask])
        quantiles[fold1_mask] = self._cost_quantiles(
            self.cost_critic, inputs[fold1_mask], actions[fold1_mask])
        return quantiles

    @torch.no_grad()
    def _cost_actor_query_disagreement(self, inputs, actions, budgets, queried_cdf):
        """
        测量target与online对同一(s,a,b)的风险概率差，不改变优化目标。

        online模式返回同device的精确0且不做额外前向。target模式才用online critic
        复算CDF，并记录平均绝对差；它回答Polyak查询是否真的形成了非零时间隔离，
        避免把数值上几乎相同的网络误报成有效cross-batch机制。
        """
        if self.cost_actor_query_mode == 'crossfit':
            # 同一输入同时过两个模型，只用于诊断模型不确定性；优化仍严格使用
            # 上面按fold选择的out-of-fold queried_cdf，不取均值也不取最大值。
            primary_cdf = self._cost_tail_probability(
                self._cost_quantiles(
                    self.cost_critic, inputs, actions), budgets)
            peer_cdf = self._cost_tail_probability(
                self._cost_quantiles(
                    self.cost_crossfit_critic, inputs, actions), budgets)
            disagreement = (primary_cdf - peer_cdf).abs().mean()
            self.last_cost_crossfit_peer_abs_mean = float(disagreement.item())
            self.last_risk_query_target_online_abs_mean = float(disagreement.item())
            return disagreement
        if self.cost_actor_query_mode != 'target':
            self.last_risk_query_target_online_abs_mean = 0.0
            return torch.zeros((), device=queried_cdf.device)
        online_quantiles = self._cost_quantiles(
            self.cost_critic, inputs, actions)
        online_cdf = self._cost_tail_probability(online_quantiles, budgets)
        disagreement = (queried_cdf - online_cdf).abs().mean()
        self.last_risk_query_target_online_abs_mean = float(disagreement.item())
        return disagreement

    def _estimate_baselines(self, states, budgets, steps=None, need_reward=True,
                            policy_mean=None, policy_log_std=None,
                            cost_features=None, cost_query_folds=None):
        """
        用 K 个策略动作近似 V_c(s,b)，仅在 distributional reward 模式下同时近似 V_m(s)。

        GAE actor 已由独立 V_r(s) 提供 reward baseline；跳过无用的 reward critic K 次前向
        不改变 cost advantage 数值，可显著减少长 rollout 的 GPU 计算。
        """
        # 所有 critic 前向都必须与 actor 共享 observation 统计；steps=None 仅表示
        # 不追加 t/T，不能跳过归一化，否则该兼容分支会混用两套输入尺度。
        s_aug = self._aug(states, steps) if steps is not None else self._normalize_states(states)
        # raw 路径沿用完全相同的 s_aug；actor_feature 路径只为 cost head 换成
        # 对齐历史特征，reward baseline 仍保持 state-conditioned 定义。
        cost_aug = (
            s_aug if self.cost_history_mode == 'raw'
            else self._cost_inputs(states, 0 if steps is None else steps, cost_features))
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
            c_list.append(self._cost_actor_query_probability(
                cost_aug, a, budgets, label_folds=cost_query_folds))
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
            cost_features = batch.get('cost_feature')
            cost_inputs = self._cost_inputs(s, steps, cost_features)
            query_folds = batch.get('_cost_crossfit_fold')
            psi_cdf = self._cost_actor_query_probability(
                cost_inputs, a, b, label_folds=query_folds)
            policy_mean = (
                batch['actor_mean'].reshape(-1, self.action_dim)
                if self.recurrent_policy else None)
            policy_log_std = (
                batch['actor_log_std'].reshape(-1, self.action_dim)
                if self.recurrent_policy else None)
            _, v_c = self._estimate_baselines(
                s, b, steps, need_reward=False,
                policy_mean=policy_mean, policy_log_std=policy_log_std,
                cost_features=cost_features, cost_query_folds=query_folds)
            adv_c = psi_cdf - v_c
        self._ema_update(self.constraint_rms,
                         float(adv_c.mean().item()), float(adv_c.var(unbiased=False).item()))

    def _soft_update_target(self):
        """Polyak 同步 reward/cost critic 与可选的独立 cost history encoder。"""
        with torch.no_grad():
            for tp, op in zip(self.reward_target_critic.parameters(), self.reward_critic.parameters()):
                tp.data.lerp_(op.data, self.target_tau)
            for tp, op in zip(self.cost_target_critic.parameters(), self.cost_critic.parameters()):
                tp.data.lerp_(op.data, self.target_tau)
            if self.cost_target_history_encoder is not None:
                for target_parameter, online_parameter in zip(
                        self.cost_target_history_encoder.parameters(),
                        self.cost_history_encoder.parameters()):
                    target_parameter.data.lerp_(
                        online_parameter.data, self.target_tau)

    def _initial_cdf_estimate(self, s0, label_folds=None):
        """cost-critic 在 s0 上的 P(C≥d)/E[C]/std(C) 估计 (校准诊断)。"""
        with torch.no_grad():
            a0, actor_feature0 = self._sample_initial_actions(
                s0, return_features=True)
            cost_feature0 = self._initial_cost_history_feature(
                s0, actor_feature0)
            cost_input0 = self._cost_inputs(s0, 0, cost_feature0)
            if self.cost_actor_query_mode == 'crossfit':
                # 训练日志必须报告真正驱动该fold actor的out-of-fold critic，而不是
                # 总用主critic。另在同一(s0,a0)上记录两模型hard-CDF分歧。
                psi = self._cost_actor_query_quantiles(
                    cost_input0, a0, label_folds=label_folds)
                primary_cdf = self._cost_tail_probability(
                    self._cost_quantiles(
                        self.cost_critic, cost_input0, a0),
                    self.cost_limit, mode='hard')
                peer_cdf = self._cost_tail_probability(
                    self._cost_quantiles(
                        self.cost_crossfit_critic, cost_input0, a0),
                    self.cost_limit, mode='hard')
                self.last_cost_crossfit_peer_abs_mean = float(
                    (primary_cdf - peer_cdf).abs().mean().item())
            else:
                psi = self._cost_quantiles(
                    self.cost_critic, cost_input0, a0)       # [B,N] 历史exact路径
            qr_hard_probability = self._cost_tail_probability(
                psi, self.cost_limit, mode='hard')
            qr_smooth_probability = self._cost_tail_probability(
                psi, self.cost_limit, mode='sigmoid')
            self.last_qr_cdf_initial = float(
                qr_hard_probability.mean().item())
            self.last_qr_cdf_smooth_initial = float(
                qr_smooth_probability.mean().item())
            if self.cost_cdf_estimator == 'direct':
                selected_probability = self._direct_cost_tail_probability(
                    cost_input0, a0, self.cost_limit)
                self.last_cdf_initial = float(
                    selected_probability.mean().item())
                # direct head已经输出连续概率，不存在hard/sigmoid两个表示。
                self.last_cdf_smooth_initial = self.last_cdf_initial
                if self.cost_direct_cdf_query_mode == 'ema':
                    self.last_direct_online_cdf_initial = float(
                        self._direct_cost_tail_probability(
                            cost_input0, a0, self.cost_limit,
                            source='online').mean().item())
            else:
                self.last_cdf_initial = self.last_qr_cdf_initial
                self.last_cdf_smooth_initial = (
                    self.last_qr_cdf_smooth_initial)
            pred_mean, pred_std = self._cost_quantile_moments(psi)
            self.last_pred_cost_mean = float(pred_mean.mean().item())
            self.last_pred_cost_std = float(pred_std.mean().item())
            self.last_cost_quantile_crossing_fraction = float(
                0.0 if psi.shape[-1] < 2 else
                (psi[:, 1:] < psi[:, :-1]).float().mean().item())
        return self.last_cdf_initial

    # ============================================================ 日志 ============================================================
    def _log(self, it, batch, critic_info, actor_info):
        """基类统一键 (_log_core 下尾口径) + DQCAC 专属诊断键 (budget/cost-critic 校准/λ/norm)。"""
        R_np = batch['disc_return'].detach().cpu().numpy()
        Zc_np = batch['disc_cost'].detach().cpu().numpy()
        Cu_np = batch['undisc_cost'].detach().cpu().numpy()
        ghat = self._initial_cdf_estimate(
            batch['s0'], batch.get('_cost_crossfit_s0_fold'))  # OOF或online P(C≥d)
        budgets_np = batch['budgets'].detach().cpu().numpy()
        empirical_prob = float(np.mean(Zc_np >= self.cost_limit))  # = P(Z≤q)

        extra = {
            'constraint/cdf_estimate_initial': ghat,          # Ĝ ≈ P(Z≤q|s0) (与 empirical 同口径)
            'constraint/cdf_calibration_error': abs(ghat - empirical_prob),
            'constraint/cdf_estimate_smooth_initial': self.last_cdf_smooth_initial,
            'critic/pred_cost_mean': self.last_pred_cost_mean,
            'critic/pred_cost_std': self.last_pred_cost_std,
            'critic/cost_quantile_crossing_fraction': (
                self.last_cost_quantile_crossing_fraction),
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
            'dual/control_prob_gap': self.last_dual_control_prob_gap,
            'dual/pid_target_prob': self.pid_target_prob,
            'dual/pid_safety_margin': self.q_alpha - self.pid_target_prob,
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
            'dual/pid_update_interval': float(self.pid_update_interval),
            'dual/pid_update_due': self.last_pid_update_due,
            'dual/pid_rollouts_accumulated': (
                self.last_pid_rollouts_accumulated),
            'dual/pid_update_batch_episodes': (
                self.last_pid_update_batch_episodes),
            'dual/pid_update_events': float(self.pid_update_events),
            'dual/sum_norm_enabled': float(self.sum_norm),
            'debug/cost_history_actor_feature': float(
                self.cost_history_mode == 'actor_feature'),
            'debug/cost_history_cost_lstm': float(
                self.cost_history_mode == 'cost_lstm'),
            'debug/cost_critic_time_weighted': float(
                self.cost_critic_time_weighting != 'uniform'),
            'debug/cost_critic_weight_discount': self.cost_critic_weight_discount,
            'debug/cost_critic_weight_floor': self.cost_critic_weight_floor,
            'debug/cost_actor_query_is_target': float(
                self.cost_actor_query_mode == 'target'),
            'debug/cost_actor_query_is_preupdate': float(
                self.cost_actor_query_mode == 'preupdate'),
            'debug/cost_cdf_is_sigmoid': float(
                self.cost_cdf_mode == 'sigmoid'),
            'debug/cost_cdf_temperature': self.cost_cdf_temperature,
            'debug/cost_cdf_estimator_is_direct': float(
                self.cost_cdf_estimator == 'direct'),
            'debug/cost_direct_cdf_query_is_ema': float(
                self.cost_direct_cdf_query_mode == 'ema'),
            'debug/cost_distribution_is_iqn': float(
                self.cost_distribution_model == 'iqn'),
            'debug/cost_quantile_output_is_exp': float(
                self.cost_quantile_output == 'exp'),
            'debug/cost_quantile_output_is_softplus': float(
                self.cost_quantile_output == 'softplus'),
            'debug/cost_quantile_output_scale': self.cost_quantile_output_scale,
            'debug/cost_shared_backbone_enabled': float(
                self.cost_shared_backbone_coef > 0.0),
            'debug/cost_shared_backbone_coef': self.cost_shared_backbone_coef,
            'debug/cost_shared_backbone_cost_scale': (
                self.cost_shared_backbone_cost_scale),
            'debug/cost_shared_backbone_huber_kappa': (
                self.cost_shared_backbone_huber_kappa),
            'debug/cost_iqn_train_quantiles': float(
                self.cost_iqn_train_quantiles),
            'debug/cost_iqn_query_quantiles': float(
                self.cost_iqn_query_quantiles),
            'debug/cost_iqn_cosines': float(self.cost_iqn_cosines),
            'debug/cost_quantile_grid_is_query': float(
                self.cost_quantile_grid_mode == 'query_mixture'),
            'debug/cost_quantile_prediction_is_importance': float(
                self.cost_quantile_prediction_weighting == 'importance'),
            'debug/cost_quantile_local_count': float(self.cost_quantile_local_count),
            'debug/cost_quantile_weight_min': float(
                self.cost_cdf_weights.min().item()),
            'debug/cost_quantile_weight_max': float(
                self.cost_cdf_weights.max().item()),
            'budget/min': float(np.min(budgets_np)),
            'budget/max': float(np.max(budgets_np)),
            'budget/mean': float(np.mean(budgets_np)),
            'training/learning_steps': self.learning_steps,
            'training/actor_lr': float(self.actor_scheduler.get_last_lr()[0]),
            'training/actor_updates_per_iteration': self.actor_updates_per_episode,
            'training/actor_update_interval': self.actor_update_interval,
        }
        # pre 是本轮训练前的新样本泛化，post 是同一批被重复更新后的拟合；
        # 两者键完全对齐，profile 可直接计算 post-pre 而不混入动作/RNG差异。
        for phase in ('pre', 'post'):
            metrics = batch.get(f'_s0_holdout_{phase}')
            if metrics is not None:
                for metric_name, metric_value in metrics.items():
                    extra[
                        f'critic/s0_holdout_{phase}_{metric_name}'] = metric_value
        if self.cost_cdf_estimator == 'direct':
            # selected CDF继续占用历史主键；QR诊断键使profile能在同一run内量化
            # “直接监督”相对“完整分布再反演”的增益。
            extra.update({
                'constraint/qr_cdf_estimate_initial': (
                    self.last_qr_cdf_initial),
                'constraint/qr_cdf_estimate_smooth_initial': (
                    self.last_qr_cdf_smooth_initial),
                'constraint/qr_cdf_calibration_error': abs(
                    self.last_qr_cdf_initial - empirical_prob),
            })
            if self.cost_direct_cdf_query_mode == 'ema':
                extra.update({
                    'constraint/direct_online_cdf_estimate_initial': (
                        self.last_direct_online_cdf_initial),
                    'constraint/direct_online_cdf_calibration_error': abs(
                        self.last_direct_online_cdf_initial - empirical_prob),
                })
        extra.update(critic_info)
        extra.update(actor_info)
        if self.cost_actor_query_mode == 'crossfit':
            # 旧target-online键为兼容既有profile继续保留；该别名明确当前比较的是
            # 两个独立fold critic，避免把peer disagreement误读成Polyak lag。
            extra['debug/cost_actor_query_is_crossfit'] = 1.0
            extra['advantage/risk_query_crossfit_peer_abs_mean'] = float(
                actor_info.get(
                    'advantage/risk_query_target_online_abs_mean',
                    self.last_risk_query_target_online_abs_mean))
            extra['critic/cost_crossfit_peer_abs_mean'] = (
                self.last_cost_crossfit_peer_abs_mean)
        if self.cost_actor_query_mode == 'preupdate':
            extra['advantage/risk_query_preupdate_postupdate_abs_mean'] = float(
                actor_info.get(
                    'advantage/risk_query_preupdate_postupdate_abs_mean',
                    self.last_risk_query_preupdate_postupdate_abs_mean))
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

        episode_count = int(num_episodes)
        if episode_count <= 0:
            raise ValueError("num_episodes must be a positive integer")
        rounds = int(np.ceil(episode_count / vec_env.B))
        rewards_all, costs_all, undisc_costs_all = [], [], []
        initial_states, initial_actions, initial_features = [], [], []
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
                    means, log_stds, _value, (hidden, cell), features = self.actor(
                        actor_obs.unsqueeze(0), prev_action.unsqueeze(0),
                        prev_reward.unsqueeze(0), (hidden, cell), return_features=True)
                    action = self._sample_from_params(means[0], log_stds[0])
                    if timestep == 0:
                        initial_states.append(state.clone())
                        initial_actions.append(action.clone())
                        initial_features.append(features[0].clone())
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
            # critic概率和真实outage必须使用完全相同的前E条轨迹。
            s0 = torch.cat(initial_states, dim=0)[:episode_count]
            a0 = torch.cat(initial_actions, dim=0)[:episode_count]
            actor_feature0 = torch.cat(initial_features, dim=0)[:episode_count]
            cost_feature0 = self._initial_cost_history_feature(
                s0, actor_feature0)
            cost_input0 = self._cost_inputs(s0, 0, cost_feature0)
            crossfit_eval = None
            qr_cdf_eval = None
            selected_probability_eval = None
            selected_smooth_probability_eval = None
            qr_probability_eval = None
            direct_online_probability_eval = None
            if self.cost_actor_query_mode == 'crossfit':
                # 新评估状态没有训练fold身份，因此用两个独立估计的等权ensemble。
                # CDF先各自查询再平均，不能先平均quantiles后count，否则二者不等价。
                psi_primary = self._cost_quantiles(
                    self.cost_critic, cost_input0, a0)
                psi_peer = self._cost_quantiles(
                    self.cost_crossfit_critic, cost_input0, a0)
                hard_primary = self._cost_tail_probability(
                    psi_primary, float(cost_limit), mode='hard')
                hard_peer = self._cost_tail_probability(
                    psi_peer, float(cost_limit), mode='hard')
                smooth_primary = self._cost_tail_probability(
                    psi_primary, float(cost_limit), mode='sigmoid')
                smooth_peer = self._cost_tail_probability(
                    psi_peer, float(cost_limit), mode='sigmoid')
                selected_probability_eval = 0.5 * (
                    hard_primary + hard_peer)
                selected_smooth_probability_eval = 0.5 * (
                    smooth_primary + smooth_peer)
                cost_cdf_initial = float(
                    selected_probability_eval.mean().item())
                cost_cdf_smooth_initial = float(
                    selected_smooth_probability_eval.mean().item())

                # 把两critic视为等权分布mixture：Var=E[var+mean²]-E[mean]²。
                mean_primary, std_primary = self._cost_quantile_moments(
                    psi_primary)
                mean_peer, std_peer = self._cost_quantile_moments(psi_peer)
                mixture_mean = 0.5 * (mean_primary + mean_peer)
                mixture_second = 0.5 * (
                    std_primary.pow(2) + mean_primary.pow(2)
                    + std_peer.pow(2) + mean_peer.pow(2))
                mixture_std = (
                    mixture_second - mixture_mean.pow(2)).clamp_min(0.0).sqrt()
                pred_cost_mean = float(mixture_mean.mean().item())
                pred_cost_std = float(mixture_std.mean().item())
                crossing_primary = (
                    psi_primary[:, 1:] < psi_primary[:, :-1]).float().mean()
                crossing_peer = (
                    psi_peer[:, 1:] < psi_peer[:, :-1]).float().mean()
                cost_quantile_crossing_fraction = float(
                    (0.5 * (crossing_primary + crossing_peer)).item())
                crossfit_eval = {
                    'cost_cdf_primary_initial': float(
                        hard_primary.mean().item()),
                    'cost_cdf_peer_initial': float(hard_peer.mean().item()),
                    'cost_cdf_crossfit_peer_abs_mean': float(
                        (hard_primary - hard_peer).abs().mean().item()),
                }
            else:
                # 单critic分支逐式保留历史运算顺序，支持默认checkpoint exact回归。
                psi0 = self._cost_quantiles(
                    self.cost_critic, cost_input0, a0)
                qr_hard = self._cost_tail_probability(
                    psi0, float(cost_limit), mode='hard')
                qr_smooth = self._cost_tail_probability(
                    psi0, float(cost_limit), mode='sigmoid')
                if self.cost_cdf_estimator == 'direct':
                    direct_probability = self._direct_cost_tail_probability(
                        cost_input0, a0, float(cost_limit))
                    selected_probability_eval = direct_probability
                    selected_smooth_probability_eval = direct_probability
                    qr_probability_eval = qr_hard
                    cost_cdf_initial = float(
                        direct_probability.mean().item())
                    cost_cdf_smooth_initial = cost_cdf_initial
                    qr_cdf_eval = {
                        'cost_cdf_qr_initial': float(
                            qr_hard.mean().item()),
                        'cost_cdf_qr_smooth_initial': float(
                            qr_smooth.mean().item()),
                    }
                    if self.cost_direct_cdf_query_mode == 'ema':
                        direct_online_probability_eval = (
                            self._direct_cost_tail_probability(
                                cost_input0, a0, float(cost_limit),
                                source='online'))
                else:
                    selected_probability_eval = qr_hard
                    selected_smooth_probability_eval = qr_smooth
                    cost_cdf_initial = float(qr_hard.mean().item())
                    cost_cdf_smooth_initial = float(
                        qr_smooth.mean().item())
                pred_mean0, pred_std0 = self._cost_quantile_moments(psi0)
                pred_cost_mean = float(pred_mean0.mean().item())
                pred_cost_std = float(pred_std0.mean().item())
                cost_quantile_crossing_fraction = float(
                    0.0 if psi0.shape[-1] < 2 else
                    (psi0[:, 1:] < psi0[:, :-1]).float().mean().item())

        # 最后一轮环境完整运行，统计仅保留请求的episode_count条。
        reward_np = torch.cat(rewards_all)[:episode_count].cpu().numpy().astype(np.float64)
        cost_np = torch.cat(costs_all)[:episode_count].cpu().numpy().astype(np.float64)
        undisc_np = torch.cat(undisc_costs_all)[:episode_count].cpu().numpy().astype(np.float64)
        transformed = -cost_np
        threshold = -float(cost_limit)
        empirical = float(np.mean(transformed <= threshold))
        quantile = float(np.percentile(transformed, omega * 100))
        observed_outage = (cost_np >= float(cost_limit)).astype(np.float64)
        selected_probability_np = (
            selected_probability_eval.detach().cpu().numpy().astype(np.float64))
        selected_smooth_probability_np = (
            selected_smooth_probability_eval.detach().cpu().numpy().astype(np.float64))
        hard_probability_metrics = _binary_probability_metrics(
            selected_probability_np, observed_outage)
        smooth_probability_metrics = _binary_probability_metrics(
            selected_smooth_probability_np, observed_outage)
        result = {
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
            'cost_cdf_smooth_initial': cost_cdf_smooth_initial,
            # 历史Brier字段保持hard estimator语义；新增smooth字段才对应T1 actor查询。
            'cost_cdf_brier_initial': hard_probability_metrics['brier'],
            'cost_cdf_brier_skill_initial': hard_probability_metrics['brier_skill'],
            'cost_cdf_roc_auc_initial': hard_probability_metrics['roc_auc'],
            'cost_cdf_discrimination_gap_initial': (
                hard_probability_metrics['discrimination_gap']),
            'cost_cdf_prediction_std_initial': hard_probability_metrics['prediction_std'],
            'cost_cdf_outage_mean_initial': hard_probability_metrics['positive_mean'],
            'cost_cdf_safe_mean_initial': hard_probability_metrics['negative_mean'],
            'cost_cdf_smooth_brier_initial': smooth_probability_metrics['brier'],
            'cost_cdf_smooth_brier_skill_initial': (
                smooth_probability_metrics['brier_skill']),
            'cost_cdf_smooth_roc_auc_initial': smooth_probability_metrics['roc_auc'],
            'cost_cdf_smooth_discrimination_gap_initial': (
                smooth_probability_metrics['discrimination_gap']),
            'cost_cdf_smooth_prediction_std_initial': (
                smooth_probability_metrics['prediction_std']),
            'cost_cdf_smooth_outage_mean_initial': (
                smooth_probability_metrics['positive_mean']),
            'cost_cdf_smooth_safe_mean_initial': (
                smooth_probability_metrics['negative_mean']),
            'pred_cost_mean': pred_cost_mean,
            'pred_cost_std': pred_cost_std,
            'cost_quantile_crossing_fraction': (
                cost_quantile_crossing_fraction),
        }
        if crossfit_eval is not None:
            result.update(crossfit_eval)
        if qr_cdf_eval is not None:
            result.update(qr_cdf_eval)
            qr_probability_np = (
                qr_probability_eval.detach().cpu().numpy().astype(np.float64))
            result['cost_cdf_qr_brier_initial'] = float(np.mean(
                (qr_probability_np - observed_outage) ** 2))
        if direct_online_probability_eval is not None:
            direct_online_probability_np = (
                direct_online_probability_eval.detach().cpu().numpy().astype(
                    np.float64))
            result['cost_cdf_direct_online_initial'] = float(
                direct_online_probability_np.mean())
            result['cost_cdf_direct_online_brier_initial'] = float(np.mean(
                (direct_online_probability_np - observed_outage) ** 2))
        return result

    # ============================================================ 总结接口 ============================================================
    def get_training_summary(self):
        """暴露最终约束指标。"""
        summary = {
            'lambda_final': float(self.lambda_dual.detach().item()),
            'empirical_prob': self.last_empirical_prob,
            'empirical_outage_prob': self.last_outage_prob,   # 兼容旧字段 (=empirical_prob)
            'cdf_estimate_initial': self.last_cdf_initial,
            'cost_quantile_crossing_fraction': (
                self.last_cost_quantile_crossing_fraction),
            'dual_prob': self.last_dual_prob,
            'dual_update_mode': self.dual_update_mode,
            'dual_pid_signal': self.dual_pid_signal,
            'pid_target_prob': self.pid_target_prob,
            'pid_safety_margin': self.q_alpha - self.pid_target_prob,
            'pid_i': self.pid_i,
            'pid_update_interval': self.pid_update_interval,
            'pid_update_events': self.pid_update_events,
            'pid_rollouts_since_update': self.pid_rollouts_since_update,
            'pid_rollouts_accumulated_last_iteration': (
                self.last_pid_rollouts_accumulated),
            'pid_update_batch_episodes_last_event': (
                self.last_pid_update_batch_episodes),
            'dual_cost_quantile': self.last_dual_cost_quantile,
            'sum_norm': self.sum_norm,
            'beta': self.beta,
            'reward_actor_mode': self.reward_actor_mode,
            'policy_arch': self.policy_arch,
            'cost_history_mode': self.cost_history_mode,
            'cost_critic_time_weighting': self.cost_critic_time_weighting,
            'cost_critic_weight_discount': self.cost_critic_weight_discount,
            'cost_critic_weight_floor': self.cost_critic_weight_floor,
            'cost_actor_query_mode': self.cost_actor_query_mode,
            'risk_query_target_online_abs_mean': (
                self.last_risk_query_target_online_abs_mean),
            'cost_s0_aux_coef': self.cost_s0_aux_coef,
            'cost_s0_replay_batches': self.cost_s0_replay_batches,
            'cost_mean_anchor_coef': self.cost_mean_anchor_coef,
            'cost_mean_anchor_cost_scale': self.cost_mean_anchor_cost_scale,
            'cost_mean_anchor_scale': self._cost_mean_anchor_scale(
                self.num_quantiles),
            'cost_quantile_output': self.cost_quantile_output,
            'cost_quantile_output_scale': self.cost_quantile_output_scale,
            'cost_shared_backbone_coef': self.cost_shared_backbone_coef,
            'cost_shared_backbone_cost_scale': (
                self.cost_shared_backbone_cost_scale),
            'cost_shared_backbone_huber_kappa': (
                self.cost_shared_backbone_huber_kappa),
            's0_holdout_pre': dict(self.last_s0_holdout_pre),
            's0_holdout_post': dict(self.last_s0_holdout_post),
            'cost_cdf_mode': self.cost_cdf_mode,
            'cost_cdf_temperature': self.cost_cdf_temperature,
            'cost_cdf_estimator': self.cost_cdf_estimator,
            'cost_direct_cdf_lr': self.cost_direct_cdf_lr,
            'cost_direct_cdf_grad_clip': self.cost_direct_cdf_grad_clip,
            'cost_direct_cdf_budget_scale': (
                self.cost_direct_cdf_budget_scale),
            'cost_direct_cdf_query_mode': self.cost_direct_cdf_query_mode,
            'cost_direct_cdf_ema_tau': self.cost_direct_cdf_ema_tau,
            'direct_online_cdf_estimate_initial': (
                self.last_direct_online_cdf_initial),
            'qr_cdf_estimate_initial': self.last_qr_cdf_initial,
            'cost_distribution_model': self.cost_distribution_model,
            'cost_iqn_train_quantiles': self.cost_iqn_train_quantiles,
            'cost_iqn_query_quantiles': self.cost_iqn_query_quantiles,
            'cost_iqn_cosines': self.cost_iqn_cosines,
            'cost_iqn_seed': self.cost_iqn_seed,
            'cost_quantile_grid_mode': self.cost_quantile_grid_mode,
            'cost_quantile_query_tau': self.cost_quantile_query_tau,
            'cost_quantile_local_half_width': self.cost_quantile_local_half_width,
            'cost_quantile_local_fraction': self.cost_quantile_local_fraction,
            'cost_quantile_prediction_weighting': self.cost_quantile_prediction_weighting,
            'cost_quantile_local_count': self.cost_quantile_local_count,
            'quantile_target_reduction': self.quantile_target_reduction,
            'quantile_loss_reference_samples': self.quantile_loss_reference_samples,
            'actor_updates_per_episode': self.actor_updates_per_episode,
            'actor_update_interval': self.actor_update_interval,
            'actor_update_events': self.actor_update_events,
            'actor_first_epoch_ratio_max_error': (
                self.last_actor_first_epoch_ratio_max_error),
            'actor_update_batch_trajectories': (
                self.last_actor_update_batch_trajectories),
            'actor_updates_completed_last_event': (
                self.last_actor_updates_completed),
            'actor_last_approx_kl': self.last_actor_approx_kl,
            'actor_last_clip_fraction': self.last_actor_clip_fraction,
            'shared_cost_last_qr_loss': self.last_shared_cost_qr_loss,
            'shared_cost_last_mean_loss': self.last_shared_cost_mean_loss,
            'shared_cost_last_weighted_loss': (
                self.last_shared_cost_weighted_loss),
            'shared_cost_last_feature_refresh_abs_mean': (
                self.last_shared_cost_feature_refresh_abs_mean),
            'shared_cost_last_body_grad_norm': (
                self.last_shared_cost_body_grad_norm),
            'shared_cost_last_lstm_grad_norm': (
                self.last_shared_cost_lstm_grad_norm),
            'shared_cost_last_policy_head_grad_norm': (
                self.last_shared_cost_policy_head_grad_norm),
            'shared_cost_last_value_head_grad_norm': (
                self.last_shared_cost_value_head_grad_norm),
            'shared_cost_last_head_grad_present': (
                self.last_shared_cost_head_grad_present),
            'ppo_target_kl': self.ppo_target_kl,
            'freeze_policy_updates': self.freeze_policy_updates,
            'freeze_observation_stats': self.freeze_observation_stats,
            'num_envs': self.num_envs,
            'num_iterations': self.num_iterations,
        }
        if self.cost_actor_query_mode == 'crossfit':
            summary.update({
                'cost_crossfit_enabled': True,
                'risk_query_crossfit_peer_abs_mean': (
                    self.last_risk_query_target_online_abs_mean),
                'cost_crossfit_peer_abs_mean': (
                    self.last_cost_crossfit_peer_abs_mean),
            })
        if self.cost_actor_query_mode == 'preupdate':
            summary['risk_query_preupdate_postupdate_abs_mean'] = (
                self.last_risk_query_preupdate_postupdate_abs_mean)
        return summary
