"""正式 T1000×B25 actor 的分支梯度诊断；不更新模型、RMS、RNG 或 dual。"""
from __future__ import annotations

import math

import torch


def score_gradient_diagnostics(agent, rollout):
    """测量首次 PPO epoch 在行为策略处的 reward/risk loss 梯度。

    每次重放五条完整轨迹，按正式 B*H 分母累加，与全量首次 epoch 的
    score gradient 相同（仅浮点累加次序不同）。两个分支共用同一个已冻结
    normalization_scale，risk 权重已经含 μ，不能再次乘入。autograd.grad
    返回独立张量，不写 parameter.grad，也不消耗动作采样 RNG。
    """
    terms = agent._independent_factorial_actor_terms
    if not isinstance(terms, dict):
        raise RuntimeError('gradient diagnostic requires the current frozen actor terms')
    parameters = tuple(agent.actor.parameters())
    reward = [torch.zeros_like(p) for p in parameters]
    risk = [torch.zeros_like(p) for p in parameters]
    scale = terms['combined_normalization_scale'].detach()
    denominator = float(agent.num_envs * agent.horizon)
    for start in range(0, rollout['S'].shape[1], 5):
        stop = start + 5
        mean, log_std, _ = agent.actor(rollout['S'][:, start:stop])
        logp = agent._logp_from_params(rollout['A_raw'][:, start:stop], mean, log_std)
        valid = rollout['valid'][:, start:stop]
        reward_loss = -(logp * terms['mean_term'][:, start:stop].detach() / scale * valid).sum() / denominator
        risk_loss = (logp * terms['risk_term'][:, start:stop].detach() / scale * valid).sum() / denominator
        reward_grads = torch.autograd.grad(reward_loss, parameters, retain_graph=True)
        risk_grads = torch.autograd.grad(risk_loss, parameters)
        for destination, value in zip(reward, reward_grads):
            destination.add_(value.detach())
        for destination, value in zip(risk, risk_grads):
            destination.add_(value.detach())
    reward_norm = sum(g.double().square().sum() for g in reward).sqrt().item()
    risk_norm = sum(g.double().square().sum() for g in risk).sqrt().item()
    dot = sum(a.double().mul(b.double()).sum() for a, b in zip(reward, risk)).item()
    if not all(math.isfinite(value) for value in (reward_norm, risk_norm, dot)):
        raise FloatingPointError('nonfinite reward/risk score gradient')
    c = agent.dqc_reward_scale
    # raw 反事实仅撤销固定除数，保留本批共同分母；不是另一条训练轨迹。
    raw_reward_norm = reward_norm * c
    metrics = {
        'reward_score_gradient_norm_scaled': reward_norm,
        'reward_score_gradient_norm_raw_common_scale': raw_reward_norm,
        'risk_score_gradient_norm': risk_norm,
        'risk_reward_gradient_ratio_scaled': risk_norm / reward_norm if reward_norm else 0.0,
        'risk_reward_gradient_ratio_raw': risk_norm / raw_reward_norm if raw_reward_norm else 0.0,
        'gradient_ratio_defined': float(reward_norm > 0),
        'reward_risk_loss_gradient_cosine': dot / (reward_norm * risk_norm) if reward_norm * risk_norm else 0.0,
        'gradient_cosine_defined': float(reward_norm * risk_norm > 0),
        'mu_at_actor_update': float(agent.lambda_probability.detach()),
        'lambda_effective_at_actor_update': float(getattr(getattr(agent, 'gradient_scaler', None), 'scale', c)) * float(agent.lambda_probability.detach()),
    }
    return {'diagnostic/fixed_reward_scale/' + name: value for name, value in metrics.items()}
