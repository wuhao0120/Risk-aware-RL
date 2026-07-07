# -*- coding: utf-8 -*-
"""
统一评估器 (portfolio_env_inf 专用) —— 四算法 (QPO/QCPO/QPPO/DQCAC) 共用同一评估口径:

    指标 (与 risk_sensitive_env_inf 的 monte_carlo_evaluate_constraint / _train_run.eval_full 同名同义):
      mean / quantile / std   : 截断折扣回报 Z=Σ_{t<n} γ^t r_t 的均值 / α-分位数 / 标准差
      empirical_prob          : eval 轨迹里 Z ≤ q 的比例 (真值)
      cdf_initial             : critic 在同批 (s0,a0) 上估计的 P(Z≤q) (仅分布式 critic 算法,
                                即 DQCAC; 其余返回 None) → 干净的 critic 校准对比
      pred_mean / pred_std    : critic 估计的 E[Z|s0,a0] / std (同上, 仅 DQCAC)

    两条路径:
      evaluate_policy_vec : 【主路径】用 PortfolioVecTorch 全 GPU 批量评估 (3000 条轨迹秒级);
                            所有算法统一走这里 → 对比标准统一。
      monte_carlo_evaluate_constraint : numpy 单 env 串行版 (与 risk_sensitive_env_inf 同款,
                            备用/交叉校验 vec 路径)。
"""
import math
import numpy as np
import torch


def _critic_initial_stats(agent, S0, A0, q):
    """
    用 agent 的【分布式】critic 在 eval 批 (s0,a0) 上估计 P(Z≤q)/E[Z]/std(Z)。

    仅当 agent 同时有 .critic 和 .num_quantiles (即 DQCAC 的 QR critic) 时计算;
    QPPO 的标量 V(s) critic 不是回报分布 → 返回 (None, None, None)。

    Args:
        S0: [N, sd] torch; A0: [N, ad] torch; q: float 阈值
    Returns:
        (cdf_initial, pred_mean, pred_std) —— float 或 None
    """
    if not (hasattr(agent, 'critic') and hasattr(agent, 'num_quantiles')):
        return None, None, None
    with torch.no_grad():
        S0c = S0
        if getattr(agent, 'critic_step_feature', False):       # step 增广 (本环境恒 False, 兼容保留)
            S0c = torch.cat([S0, torch.zeros(S0.shape[0], 1, device=S0.device)], dim=1)
        psi = agent.critic(S0c, A0)                            # [N, N_q] 回报分布分位数
        cdf = float((psi <= q).float().mean(dim=1).mean().item())   # P(Z≤q|s0,a0) 平均
        pmean = float(psi.mean().item())                       # E[Z|s0,a0] 平均
        pstd = float(psi.std(dim=1).mean().item())             # std(Z|s0,a0) 平均
    return cdf, pmean, pstd


def evaluate_policy_vec(agent, vec_env, num_episodes, gamma, q_alpha, quantile_threshold):
    """
    【主评估路径】全 GPU 向量化 Monte-Carlo 评估 (冻结策略)。

    流程: ceil(E/B) 次 vec rollout, 每次 B 条轨迹 × n 步, 用 agent._sample_actions
    (随机策略, 与训练同口径) 采样; 累计 Z=Σγ^t r_t; 取前 num_episodes 条算统计。

    Args:
        agent:    具有 _sample_actions(states [B,sd])→[B,ad] 的 GPU 智能体 (四算法都有)
        vec_env:  独立的 PortfolioVecTorch 实例 (勿与训练 env 共用, 避免打断训练 rollout 状态)
        num_episodes: 目标评估轨迹数 E
        gamma / q_alpha / quantile_threshold: 折扣 γ / 分位水平 α / 约束阈值 q

    Returns:
        dict: mean / quantile / std / empirical_prob / cdf_initial / pred_mean / pred_std
              / num_episodes (实际条数, = ceil(E/B)·B)
    """
    B, n = vec_env.B, vec_env.n
    rounds = max(1, math.ceil(num_episodes / B))               # rollout 轮数
    Z_all, S0_all, A0_all = [], [], []

    with torch.no_grad():
        for _ in range(rounds):
            s = vec_env.reset()                                # [B, sd]
            disc_return = torch.zeros(B, device=s.device)      # [B] 累计 Z
            disc = 1.0                                         # γ^t
            for t in range(n):
                a = agent._sample_actions(s)                   # [B, ad] 随机策略采样
                if t == 0:                                     # 记录 (s0, a0) 供 critic 校准
                    S0_all.append(s.clone()); A0_all.append(a.clone())
                s, r, done = vec_env.step(a)
                disc_return += disc * r
                disc *= gamma
            Z_all.append(disc_return)

    Z = torch.cat(Z_all).cpu().numpy().astype(np.float64)      # [rounds·B] 全部回报
    q = float(quantile_threshold)
    result = {
        'mean': float(Z.mean()),
        'quantile': float(np.percentile(Z, q_alpha * 100)),
        'std': float(Z.std()),
        'empirical_prob': float(np.mean(Z <= q)),              # 经验违约概率 (真值)
        'num_episodes': int(Z.shape[0]),
    }
    # critic 校准 (仅 DQCAC): 同批 eval (s0,a0) 上的 critic 预测 vs 上面的真值
    S0 = torch.cat(S0_all, dim=0)
    A0 = torch.cat(A0_all, dim=0)
    cdf, pmean, pstd = _critic_initial_stats(agent, S0, A0, q)
    result['cdf_initial'] = cdf
    result['pred_mean'] = pmean
    result['pred_std'] = pstd
    return result


def monte_carlo_evaluate_constraint(agent, env, num_episodes, quantile_threshold,
                                    gamma=None, q_alpha=None):
    """
    numpy 单 env 串行评估 (与 risk_sensitive_env_inf/utils/evaluation.py 同款语义,
    适配 gymnasium 5 值 API)。备用路径: 交叉校验 evaluate_policy_vec 的 numpy/torch 一致性。
    """
    gamma = gamma if gamma is not None else getattr(agent, 'gamma', 0.99)
    q_alpha = q_alpha if q_alpha is not None else getattr(agent, 'q_alpha', 0.25)
    q = float(quantile_threshold)

    rets, s0s, a0s = [], [], []
    with torch.no_grad():
        for _ in range(num_episodes):
            state = env.reset()
            if isinstance(state, tuple):                       # gymnasium: (obs, info)
                state = state[0]
            s0_flat = np.asarray(state, dtype=np.float32).flatten().copy()
            disc_r, disc, first = 0.0, 1.0, True
            while True:
                state = np.asarray(state, dtype=np.float32).flatten()
                action = agent.select_action(state)            # 随机策略采样 (训练同口径)
                if first:
                    s0s.append(s0_flat)
                    a0s.append(np.asarray(action, dtype=np.float32).flatten())
                    first = False
                obs, reward, terminated, truncated, _ = env.step(action)
                disc_r += disc * reward
                disc *= gamma
                if terminated or truncated:
                    break
                state = obs
            rets.append(disc_r)

    rets = np.asarray(rets, dtype=np.float64)
    result = {
        'mean': float(rets.mean()),
        'quantile': float(np.percentile(rets, q_alpha * 100)),
        'std': float(rets.std()),
        'empirical_prob': float(np.mean(rets <= q)),
        'num_episodes': int(rets.shape[0]),
    }
    dev = getattr(agent, 'device', torch.device('cpu'))
    S0 = torch.as_tensor(np.stack(s0s), dtype=torch.float32, device=dev)
    A0 = torch.as_tensor(np.stack(a0s), dtype=torch.float32, device=dev)
    cdf, pmean, pstd = _critic_initial_stats(agent, S0, A0, q)
    result['cdf_initial'] = cdf
    result['pred_mean'] = pmean
    result['pred_std'] = pstd
    return result
