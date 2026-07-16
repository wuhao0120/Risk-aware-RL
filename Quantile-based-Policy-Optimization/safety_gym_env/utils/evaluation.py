# -*- coding: utf-8 -*-
"""
统一评估器 (safety_gym_env · CMDP 版) —— QCPO / DQCAC 共用同一评估口径。

    日志/报告用与 risk_sensitive 对齐的下尾口径 (Z=-C, q=-d, α=ω):
      mean / reward_std          : 任务回报 R=Σγ^t r_t 的均值 / 标准差 (目标, 越大越好)
      empirical_prob             : P̂(Z ≤ q) = P̂(C ≥ d) (可行 ⟺ ≤ ω)
      quantile_return            : Q_α(Z) = -Q_{1-α}(C)
      quantile_margin_to_threshold : Q_α(Z) − q  (可行 ⟺ ≥ 0 当经验分位约束视角)
      constraint_margin          : α − P̂

    原始 cost 诊断 (debug / 对论文数字):
      cost_disc_mean / cost_undisc_mean / cost_quantile(=Q_{1-ω}(C)) / outage_prob(=empirical_prob)
      cost_cdf_initial / pred_cost_mean/std : cost critic 校准 (仅 DQCAC)

    主路径 evaluate_policy_vec: 用 SafetyVecEnv 批量评估 (每段=一 episode)。
"""
import math
import numpy as np
import torch


def _cost_critic_initial_stats(agent, S0, A0, d):
    """
    用 agent 的【cost 分布式 critic】在 eval 批 (s0,a0) 上估计 P(C≥d)/E[C]/std(C)。
    仅当 agent 有 .cost_critic 和 .num_quantiles (DQCAC) 时计算；否则四项均为 None。
    数值上 P(C≥d)=P(Z≤q), 与 empirical_prob 同口径可比。
    """
    if not (hasattr(agent, 'cost_critic') and hasattr(agent, 'num_quantiles')):
        return None, None, None, None
    with torch.no_grad():
        # episodic 配方下 critic 带 step 特征 → 用 agent._aug 做 s0(step=0) 增广
        S0_in = agent._aug(S0, 0) if hasattr(agent, '_aug') else S0
        psi = agent.cost_critic(S0_in, A0)                    # [N, N_q] cost 回报分位数
        if hasattr(agent, '_cost_tail_probability'):
            # query-mixture QR 需要 quadrature 权重；IQN 的 default grid 是 uniform。
            cdf = float(agent._cost_tail_probability(
                psi, d, mode='hard').mean().item())
        else:
            cdf = float((psi >= d).float().mean(dim=1).mean().item())
        if hasattr(agent, '_cost_quantile_moments'):
            means, stds = agent._cost_quantile_moments(psi)
            pmean = float(means.mean().item())
            pstd = float(stds.mean().item())
        else:
            pmean = float(psi.mean().item())
            pstd = float(psi.std(dim=1).mean().item())
        crossing = float(
            0.0 if psi.shape[-1] < 2 else
            (psi[:, 1:] < psi[:, :-1]).float().mean().item())
    return cdf, pmean, pstd, crossing


def evaluate_policy_vec(agent, vec_env, num_episodes, gamma, cost_gamma, omega, cost_limit):
    """
    【主评估路径】CPU-VecEnv 批量 Monte-Carlo 评估 (冻结随机策略, 每段=一 episode)。

    Args:
        agent:      具有 _sample_actions(states)->actions 的智能体 (两算法都有)
        vec_env:    独立的 SafetyVecEnv 实例 (勿与训练 env 共用)
        num_episodes: 目标评估轨迹数 E
        gamma/cost_gamma: 奖励/cost 折扣
        omega:      α=ω 目标约束概率
        cost_limit: d 约束阈值 (q=-d)

    Returns:
        dict: 下尾主键 + cost debug 兼容键, 见模块 docstring
    """
    B, n = vec_env.B, vec_env.n
    rounds = max(1, math.ceil(num_episodes / B))
    R_all, Zc_all, Cu_all, S0_all, A0_all = [], [], [], [], []

    with torch.no_grad():
        for _ in range(rounds):
            s = vec_env.reset()                               # [B, sd]
            R = torch.zeros(B, device=s.device)               # 奖励回报
            Zc = torch.zeros(B, device=s.device)              # 折扣 cost 回报
            Cu = torch.zeros(B, device=s.device)              # 未折扣累计 cost
            dr, dc = 1.0, 1.0
            for t in range(n):
                a = agent._sample_actions(s)
                if t == 0:                                    # 记录 (s0,a0) 供 critic 校准
                    S0_all.append(s.clone()); A0_all.append(a.clone())
                s, r, c, done = vec_env.step(a)
                R += dr * r; dr *= gamma
                Zc += dc * c; dc *= cost_gamma
                Cu += c
            R_all.append(R); Zc_all.append(Zc); Cu_all.append(Cu)

    R = torch.cat(R_all).cpu().numpy().astype(np.float64)
    Zc = torch.cat(Zc_all).cpu().numpy().astype(np.float64)
    Cu = torch.cat(Cu_all).cpu().numpy().astype(np.float64)
    d = float(cost_limit)
    Z = -Zc                                                   # Z = -C
    q = -d
    emp = float(np.mean(Z <= q))                              # = P(C≥d)
    q_est = float(np.percentile(Z, omega * 100))              # Q_α(Z)
    result = {
        'mean': float(R.mean()),                              # 目标: E[R]
        'reward_std': float(R.std()),
        'empirical_prob': emp,
        'quantile_return': q_est,
        'quantile_margin_to_threshold': q_est - q,
        'constraint_margin': omega - emp,
        # debug / 论文对照
        'cost_disc_mean': float(Zc.mean()),
        'cost_undisc_mean': float(Cu.mean()),
        'outage_prob': emp,                                   # 兼容旧键
        'cost_quantile': float(np.percentile(Zc, (1.0 - omega) * 100)),
        'num_episodes': int(R.shape[0]),
    }
    # cost critic 校准 (仅 DQCAC); 数值 = P(Z≤q|s0)
    S0 = torch.cat(S0_all, dim=0)
    A0 = torch.cat(A0_all, dim=0)
    cdf, pmean, pstd, crossing = _cost_critic_initial_stats(
        agent, S0, A0, d)
    result['cost_cdf_initial'] = cdf
    result['pred_cost_mean'] = pmean
    result['pred_cost_std'] = pstd
    result['cost_quantile_crossing_fraction'] = crossing
    return result
