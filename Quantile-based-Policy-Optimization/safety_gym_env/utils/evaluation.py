# -*- coding: utf-8 -*-
"""
Exact-N reward-side Monte Carlo evaluation for recurrent Safety Gym policies.

Evaluation invariants:
    - stochastic latent actions，与训练策略分布一致；
    - 每个 vector round 从零 recurrent state 开始；
    - early-ended slot 后续 auto-reset transitions 全部由 active mask 排除；
    - raw G_T 不添加 critic bootstrap，即使 episode 是 truncated；
    - critic calibration 和 Safety cost 都仅输出 diagnostic keys；
    - ceil(N/B)*B 个内部样本在所有统计前裁成精确 N；
    - failure 用 inclusive G_T<=q 直接计数；
    - linear percentile/Wilson/ties 由统一 risk_metrics 实现；
    - test seeds 由 runner 与 validation seeds 分离。
"""
import math
import numpy as np
import torch
from tqdm.auto import tqdm
from .risk_metrics import return_statistics, linear_empirical_quantile


def reward_critic_calibration_statistics(atoms, returns, threshold, alpha):
    """
    Compare initial-state critic atoms against independent completed-trajectory outcomes.

    Args:
        atoms:
            ``[N,K]`` detached reward-critic outputs. Head ``j`` has nominal quantile level
            ``tau_j=(j+0.5)/K``; the network is not assumed to enforce monotonic head ordering.
        returns:
            ``[N]`` raw discounted truncated returns from the same validation trajectories.
        threshold:
            Fixed reward-side chance threshold ``q``. Failure is inclusive, ``G_1000 <= q``.
        alpha:
            Requested lower-tail probability. It selects the nearest critic atom for the focused
            alpha-coverage diagnostic; it never changes the primary empirical chance estimate.

    Returns:
        A detached scalar dictionary. All fields describe critic calibration only: callers must
        keep them under ``diagnostic`` namespaces and out of feasibility/checkpoint selection.

    Why one outcome per row is sufficient:
        Each row supplies a probabilistic prediction for its own initial ``(state, action)`` and
        one Bernoulli outcome. Brier score is a proper scoring rule for exactly this setting.
        Quantile coverage is aggregated marginally across the heterogeneous initial conditions.

    The critic bootstraps at the 1000-step sampling truncation whereas empirical validation reports
    unbootstrapped ``G_1000``. With gamma=0.99 their tail discrepancy is small but not identically
    zero, so these values remain diagnostics rather than a replacement for direct event counting.
    """
    atom_tensor = torch.as_tensor(atoms, dtype=torch.float32)
    # Keep empirical outcomes in float64, matching return_statistics(). A threshold such as 0.1 is
    # not exactly representable in float32; downcasting G before an inclusive comparison could turn
    # a near-threshold non-failure into an artificial tie and disagree with primary feasibility.
    return_tensor = torch.as_tensor(
        returns, dtype=torch.float64, device=atom_tensor.device).reshape(-1)

    # Shape checks fail loudly: silently broadcasting one return across several critic rows would
    # produce plausible-looking but mathematically meaningless calibration scores.
    if atom_tensor.ndim != 2 or atom_tensor.shape[0] != return_tensor.numel():
        raise ValueError(
            "calibration expects atoms [N,K] and one return per row; "
            f"got atoms={tuple(atom_tensor.shape)}, returns={tuple(return_tensor.shape)}")
    if atom_tensor.shape[1] <= 0:
        raise ValueError("calibration needs at least one critic quantile atom")
    if not bool(torch.isfinite(atom_tensor).all()) or not bool(torch.isfinite(return_tensor).all()):
        raise ValueError("calibration inputs must be finite")

    q = float(threshold)
    num_atoms = int(atom_tensor.shape[1])
    # ``cdf_probability[i]`` is the critic's conditional estimate Psi(s0_i,a0_i,q).
    # ``failure_label[i]`` is the independent Monte Carlo realization for that same trajectory.
    cdf_probability = (atom_tensor <= q).float().mean(dim=1)
    failure_label = (return_tensor <= q).float()
    empirical_failure = failure_label.mean()
    initial_cdf_mean = cdf_probability.mean()
    initial_cdf_gap = initial_cdf_mean - empirical_failure
    # The atom mean is the distributional critic's conditional expected-return approximation.
    # Pairing each row with its realized trajectory return exposes signed bias and RMSE without
    # allowing either diagnostic to influence validation feasibility or checkpoint selection.
    predicted_row_mean = atom_tensor.mean(dim=1).to(torch.float64)
    return_error = predicted_row_mean - return_tensor

    # Ten equal-width bins make the reliability error stable and interpretable across K=32 atom
    # critics. Empty bins contribute zero; p=1 belongs to the last bin rather than index ten.
    bin_index = torch.clamp((cdf_probability * 10.0).long(), max=9)
    cdf_ece = torch.zeros((), dtype=atom_tensor.dtype, device=atom_tensor.device)
    for index in range(10):
        members = bin_index == index
        if bool(members.any()):
            bin_gap = torch.abs(
                cdf_probability[members].mean() - failure_label[members].mean())
            cdf_ece += members.float().mean() * bin_gap

    # QR-style heads have fixed nominal levels even when their values cross. Keeping head identity
    # intact lets coverage expose miscalibration; sorting here would hide the crossing defect.
    taus = (
        torch.arange(num_atoms, dtype=atom_tensor.dtype, device=atom_tensor.device) + 0.5
    ) / float(num_atoms)
    empirical_coverage = (
        return_tensor.unsqueeze(1) <= atom_tensor.to(torch.float64)
    ).float().mean(dim=0)
    coverage_error = empirical_coverage - taus
    alpha_index = int(torch.argmin(torch.abs(taus - float(alpha))).item())

    # A row crossing means at least one nominally higher quantile is below its predecessor.
    # For K=1 there is no adjacent pair, so both crossing diagnostics are defined as zero.
    adjacent_crossing = atom_tensor[:, 1:] < atom_tensor[:, :-1]
    if num_atoms > 1:
        row_crossing_fraction = adjacent_crossing.any(dim=1).float().mean()
        adjacent_crossing_fraction = adjacent_crossing.float().mean()
    else:
        row_crossing_fraction = torch.zeros((), device=atom_tensor.device)
        adjacent_crossing_fraction = torch.zeros((), device=atom_tensor.device)

    return {
        "sample_count": int(return_tensor.numel()),
        "initial_cdf_mean": float(initial_cdf_mean.item()),
        "empirical_failure_probability": float(empirical_failure.item()),
        # Signed gap > 0 means the critic is pessimistic relative to observed validation failures.
        "initial_cdf_gap": float(initial_cdf_gap.item()),
        "initial_cdf_abs_gap": float(initial_cdf_gap.abs().item()),
        "brier_score": float(
            (cdf_probability - failure_label).square().mean().item()),
        # Variation across initial (state, executed-action) predictions; this is a diagnostic for a
        # collapsed CDF head, not the Monte Carlo uncertainty of the empirical probability.
        "action_cdf_std": float(cdf_probability.std(unbiased=False).item()),
        "cdf_ece_10bin": float(cdf_ece.item()),
        # Exact 0/1 checks are appropriate because the empirical atom CDF takes values j/K exactly.
        "cdf_zero_fraction": float((cdf_probability == 0.0).float().mean().item()),
        "cdf_one_fraction": float((cdf_probability == 1.0).float().mean().item()),
        "cdf_saturation_fraction": float(
            ((cdf_probability == 0.0) | (cdf_probability == 1.0)).float().mean().item()),
        "atom_crossing_row_fraction": float(row_crossing_fraction.item()),
        "atom_crossing_adjacent_fraction": float(adjacent_crossing_fraction.item()),
        "quantile_coverage_mae": float(coverage_error.abs().mean().item()),
        "alpha_atom_level": float(taus[alpha_index].item()),
        "alpha_atom_coverage": float(empirical_coverage[alpha_index].item()),
        "alpha_atom_coverage_error": float(coverage_error[alpha_index].item()),
        "predicted_return_mean": float(atom_tensor.mean().item()),
        "return_mean_bias": float(return_error.mean().item()),
        "return_rmse": float(return_error.square().mean().sqrt().item()),
        # Average conditional atom spread, not the cross-state standard deviation of predictions.
        "predicted_return_std": float(
            atom_tensor.std(dim=1, unbiased=False).mean().item()),
    }


def _reward_critic_initial_stats(agent, states, latent_actions, returns, threshold, alpha):
    """
    Run DQC's initial critic on executed actions and return detached calibration diagnostics.

    PPO likelihood ratios remain defined on ``latent_actions``. The critic instead conditions on
    the action actually seen by Safety Gym after clipping. New agents expose this conversion through
    ``_executed_actions``; the fallback keeps lightweight evaluator test doubles compatible.
    """
    if threshold is None or not hasattr(agent, "reward_critic"):
        return None
    with torch.no_grad():
        critic_actions = (
            agent._executed_actions(latent_actions)
            if hasattr(agent, "_executed_actions")
            else latent_actions
        )
        # finite_nstep_v1 adds t=0 and zero actor h/c without changing this public evaluator.
        atoms = (
            agent.initial_reward_critic_atoms(states, critic_actions)
            if hasattr(agent, "initial_reward_critic_atoms")
            else agent.reward_critic(states, critic_actions)
        )
    return reward_critic_calibration_statistics(
        atoms, returns, threshold=threshold, alpha=alpha)


def _common_quantile_statistics(agent, returns, statistics):
    """
    Build the two public quantile metrics shared by every algorithm during evaluation.

    QPO owns a persistent Robbins-Monro tracker, so its q_est is policy state and its
    constraint probability calibrates that tracker on this independent exact-N return sample.
    Other algorithms have no persistent tracker: their q_est is this sample's linear empirical
    percentile, while constraint_prob is the inclusive probability relative to their fixed q.
    """
    tracker = getattr(agent, "q_est", None)
    if tracker is not None:
        # Detach a tensor tracker because evaluation diagnostics must not create actor gradients.
        q_est = float(tracker.detach().item() if torch.is_tensor(tracker) else tracker)
        constraint_probability = float(np.mean(returns <= q_est))
    else:
        q_est = float(statistics["quantile_return"])
        # All constrained algorithms provide a fixed q. None remains explicit only for an
        # unfinished/unconfigured baseline; inventing a probability would hide that condition.
        probability = statistics.get("empirical_prob")
        constraint_probability = None if probability is None else float(probability)
    return {
        "quantile_q_est": q_est,
        "quantile_constraint_prob": constraint_probability,
    }


def evaluate_policy_vec(agent, vec_env, num_episodes, gamma, failure_alpha,
                        return_threshold=None, show_progress=False,
                        progress_desc="evaluation"):
    """
    Evaluate exactly ``num_episodes`` stochastic policies on raw G_T.

    Rounds may generate ceil(N/B)*B episodes internally, but arrays are sliced to N before every
    statistic. Early reset episodes are masked and never spliced into the evaluated trajectory.
    """
    requested = int(num_episodes)
    if requested <= 0:
        raise ValueError("num_episodes must be positive")
    B, horizon = vec_env.B, vec_env.n
    rounds = int(math.ceil(requested / B))
    returns, costs_discounted, costs_raw = [], [], []
    initial_states, initial_actions = [], []

    # Validation keeps this disabled to avoid nested bars during training. Final test enables
    # it explicitly, so the post-training 1000-episode stage also reports a live wall-clock ETA.
    progress = None
    if show_progress:
        progress = tqdm(
            total=requested, desc=str(progress_desc), unit="episode",
            dynamic_ncols=True, mininterval=1.0, leave=True)

    try:
        with torch.no_grad():
            for _ in range(rounds):
                state = vec_env.reset()
                recurrent = agent.actor.initial_state(B, state.device)
                active = torch.ones(B, dtype=torch.bool, device=state.device)
                episode_return = torch.zeros(B, device=state.device)
                cost_discounted = torch.zeros(B, device=state.device)
                cost_raw = torch.zeros(B, device=state.device)
                for t in range(horizon):
                    action, next_recurrent, _mean, _log_std = agent._sample_actions(
                        state, recurrent, return_state=True)
                    if t == 0:
                        initial_states.append(state.clone())
                        initial_actions.append(action.clone())
                    step = vec_env.step(action)
                    valid = active.to(step.reward.dtype)
                    episode_return += (float(gamma) ** t) * step.reward * valid
                    cost_discounted += (float(gamma) ** t) * step.cost * valid
                    cost_raw += step.cost * valid
                    ended = step.episode_end & active
                    active = active & ~ended
                    keep = active.to(next_recurrent[0].dtype).view(1, B, 1)
                    recurrent = next_recurrent[0] * keep, next_recurrent[1] * keep
                    state = step.observation
                    if not bool(active.any()):
                        break
                returns.append(episode_return)
                costs_discounted.append(cost_discounted)
                costs_raw.append(cost_raw)
                if progress is not None:
                    progress.update(min(B, requested - int(progress.n)))
    finally:
        if progress is not None:
            progress.close()

    # Exact-N slicing happens before all metrics, including critic calibration arrays.
    G = torch.cat(returns)[:requested].cpu().numpy().astype(np.float64)
    C_disc = torch.cat(costs_discounted)[:requested].cpu().numpy().astype(np.float64)
    C_raw = torch.cat(costs_raw)[:requested].cpu().numpy().astype(np.float64)
    result = return_statistics(G, failure_alpha, return_threshold)
    # Compute the common axes only after exact-N slicing. Validation therefore uses all requested
    # 100 trajectories and final test all requested 1000, never ceil(N/B)*B excess samples.
    result.update(_common_quantile_statistics(agent, G, result))
    result.update({
        "diagnostic_environment_cost_discounted_mean": float(C_disc.mean()),
        "diagnostic_environment_cost_raw_mean": float(C_raw.mean()),
        "diagnostic_environment_cost_quantile_linear": linear_empirical_quantile(
            C_raw, failure_alpha),
    })
    S0 = torch.cat(initial_states)[:requested]
    A0 = torch.cat(initial_actions)[:requested]
    calibration = _reward_critic_initial_stats(
        agent, S0, A0, G, return_threshold, failure_alpha)
    result["diagnostic_reward_critic_calibration"] = calibration
    # Preserve the three pre-calibration result fields for old JSON readers. They are aliases of
    # the richer nested diagnostics and remain None for algorithms without a reward critic.
    result["diagnostic_reward_critic_failure_probability"] = (
        None if calibration is None else calibration["initial_cdf_mean"])
    result["diagnostic_reward_critic_mean"] = (
        None if calibration is None else calibration["predicted_return_mean"])
    result["diagnostic_reward_critic_std"] = (
        None if calibration is None else calibration["predicted_return_std"])
    return result
