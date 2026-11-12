"""Small, independently testable numerical rules for the Scheme B alignment.

These primitives do not select an optimizer, collect data, or launch training.
Distribution heads keep their own fixed midpoint quantile levels during fitting;
sorting is used only when reading a CDF.  Actor labels are detached before PPO.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor


def midpoint_quantile_levels(
    num_quantiles: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return fixed levels ``tau_i = (i + 1/2) / N`` for heads ``i=0,...,N-1``."""
    if num_quantiles < 1:
        raise ValueError("num_quantiles must be positive")
    dtype = torch.get_default_dtype() if dtype is None else dtype
    if not dtype.is_floating_point:
        raise TypeError("quantile levels require a floating point dtype")
    return (torch.arange(num_quantiles, device=device, dtype=dtype) + 0.5) / num_quantiles


def pairwise_pinball_loss(
    predictions: Tensor,
    targets: Tensor,
    *,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    """Fit every fixed quantile head against every detached target sample.

    Inputs have shapes ``[..., N]`` and ``[..., K]`` with identical batch axes.
    The pairwise residual is ``delta[..., i, j] = y[..., j] - q[..., i]``;
    its loss is ``(tau_i - 1{delta < 0}) * delta``.  This is pure pinball,
    without a Huber quadratic region.  Predictions are deliberately NOT sorted:
    head i is trained at its own fixed midpoint level even when heads cross.

    Both target samples and quantile levels are stop-gradient constants.  With
    ``reduction='none'``, the result has shape ``[...]`` after averaging N*K
    pairs per batch element; ``mean``/``sum`` further reduce those batch axes.
    The straightforward pair tensor makes the exact all-pairs objective clear.
    """
    if predictions.ndim < 1 or targets.ndim < 1:
        raise ValueError("predictions and targets must include a quantile/sample axis")
    if predictions.shape[:-1] != targets.shape[:-1]:
        raise ValueError("prediction and target batch axes must match exactly")
    if predictions.shape[-1] < 1 or targets.shape[-1] < 1:
        raise ValueError("prediction and target sample axes must be nonempty")
    if not predictions.is_floating_point() or not targets.is_floating_point():
        raise TypeError("pinball predictions and targets must be floating point")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("reduction must be mean, sum, or none")
    y = targets.detach().to(device=predictions.device, dtype=predictions.dtype)
    delta = y.unsqueeze(-2) - predictions.unsqueeze(-1)
    tau = midpoint_quantile_levels(
        predictions.shape[-1], device=predictions.device, dtype=predictions.dtype
    ).unsqueeze(-1)
    pair_loss = (tau - (delta.detach() < 0).to(predictions.dtype)) * delta
    per_element = pair_loss.mean(dim=(-2, -1))
    if reduction == "none":
        return per_element
    return per_element.mean() if reduction == "mean" else per_element.sum()


def sorted_quantile_cdf(
    quantiles: Tensor,
    threshold: Tensor | float,
    *,
    min_gap: float = 1e-8,
) -> Tensor:
    """Read ``P(Z <= threshold)`` by sorting then midpoint-level interpolation.

    ``quantiles`` has shape ``[..., N]``.  A scalar threshold, a threshold
    broadcastable to the batch shape ``[...]``, or one of shape ``[..., 1]``
    is accepted.  Below the minimum the CDF is 0, and at or above the maximum
    it is 1.  Between distinct quantiles, interpolate their midpoint levels.

    ``searchsorted(..., right=True)`` chooses the last equal quantile as the
    lower endpoint.  Thus exact ties have an explicit right-continuous rule;
    no arbitrary division by a zero-width duplicate interval occurs.  Gaps
    are additionally floored at ``min_gap``.  This interpolation rule cannot
    recover a missing tail: a threshold above every head still gives CDF 1.

    This is an inference/label-building primitive: it returns detached values.
    Sorting here does not reorder the trainable heads in the pinball loss.
    """
    if quantiles.ndim < 1 or quantiles.shape[-1] < 1:
        raise ValueError("quantiles must have a nonempty final axis")
    if not quantiles.is_floating_point():
        raise TypeError("quantiles must be floating point")
    if not math.isfinite(min_gap) or min_gap <= 0:
        raise ValueError("min_gap must be positive and finite")
    with torch.no_grad():
        q = quantiles.detach().sort(dim=-1).values.contiguous()
        b = torch.as_tensor(threshold, dtype=q.dtype, device=q.device).detach()
        if b.shape == q.shape[:-1] + (1,):
            b = b.squeeze(-1)
        try:
            b = torch.broadcast_to(b, q.shape[:-1])
        except RuntimeError as exc:
            raise ValueError("threshold must broadcast to the quantile batch axes") from exc
        if q.shape[-1] == 1:
            return (b >= q[..., 0]).to(q.dtype)
        upper_index = torch.searchsorted(q, b.unsqueeze(-1).contiguous(), right=True).squeeze(-1)
        lower_index = (upper_index - 1).clamp(0, q.shape[-1] - 1)
        upper_index_clamped = upper_index.clamp(0, q.shape[-1] - 1)
        lower_q = q.gather(-1, lower_index.unsqueeze(-1)).squeeze(-1)
        upper_q = q.gather(-1, upper_index_clamped.unsqueeze(-1)).squeeze(-1)
        lower_tau = (lower_index.to(q.dtype) + 0.5) / q.shape[-1]
        upper_tau = (upper_index_clamped.to(q.dtype) + 0.5) / q.shape[-1]
        fraction = ((b - lower_q) / (upper_q - lower_q).clamp_min(min_gap)).clamp(0, 1)
        cdf = lower_tau + fraction * (upper_tau - lower_tau)
        cdf = torch.where(b < q[..., 0], torch.zeros_like(cdf), cdf)
        cdf = torch.where(b >= q[..., -1], torch.ones_like(cdf), cdf)
        return cdf.clamp(0, 1)


def upper_cost_violation_probability(
    cost_quantiles: Tensor,
    remaining_budget: Tensor | float,
    *,
    min_gap: float = 1e-8,
) -> Tensor:
    """Return ``P(remaining cost > remaining budget) = 1 - F(budget)``.

    The budget is ``15 - accumulated undiscounted cost`` in SimpleButton.
    Equality belongs to the nonviolating event.  This does not convert the
    distribution into a lower-reward-tail risk or discount the cost budget.
    """
    return 1.0 - sorted_quantile_cdf(cost_quantiles, remaining_budget, min_gap=min_gap)


def strict_terminal_cost_violation(total_cost: Tensor, threshold: float = 15.0) -> Tensor:
    """Create the detached terminal event ``1{whole-episode raw cost > 15}``."""
    if not total_cost.is_floating_point():
        raise TypeError("total_cost must be floating point")
    return (total_cost.detach() > threshold).to(total_cost.dtype)


def combine_actor_advantages(
    raw_reward_advantage: Tensor,
    risk_advantage: Tensor,
    multiplier: Tensor | float,
    *,
    mean_divisor: float,
    valid_mask: Tensor | None = None,
    std_floor: float = 0.01,
) -> dict[str, Tensor]:
    """Freeze separate PPO labels with one shared population standard deviation.

    First form ``mean = A_reward / c`` and ``risk = lambda * A_probability``.
    The common denominator is ``max(std_valid(mean - risk, ddof=0), .01)``.
    Divide BOTH branches by this denominator; do not center either branch,
    divide by ``1 + lambda``, or secretly divide the risk branch by c.
    Consequently changing c from 100 to 20 increases reward relative to risk
    by five before and after the shared normalization, even though the common
    denominator itself can change.  This is a reward/risk tradeoff change.

    All returned labels, including the denominator, are detached from critic
    and dual graphs.  ``valid_mask`` affects the denominator only; padded rows
    must still be excluded from the downstream PPO objective.
    """
    if raw_reward_advantage.shape != risk_advantage.shape:
        raise ValueError("reward and risk advantage shapes must match exactly")
    if not raw_reward_advantage.is_floating_point() or not risk_advantage.is_floating_point():
        raise TypeError("actor advantages must be floating point")
    if not math.isfinite(mean_divisor) or mean_divisor <= 0:
        raise ValueError("mean_divisor must be positive and finite")
    if not math.isfinite(std_floor) or std_floor <= 0:
        raise ValueError("std_floor must be positive and finite")
    with torch.no_grad():
        reward = raw_reward_advantage.detach()
        probability = risk_advantage.detach().to(device=reward.device, dtype=reward.dtype)
        lam = torch.as_tensor(multiplier, dtype=reward.dtype, device=reward.device).detach()
        if lam.numel() != 1:
            raise ValueError("multiplier must be scalar")
        if valid_mask is None:
            valid = torch.ones_like(reward, dtype=torch.bool)
        else:
            if valid_mask.shape != reward.shape:
                raise ValueError("valid_mask must have the advantage shape")
            valid = valid_mask.to(device=reward.device, dtype=torch.bool)
        raw_mean = reward / mean_divisor
        raw_risk = lam.reshape(()) * probability
        raw_combined = raw_mean - raw_risk
        valid_combined = raw_combined[valid]
        if valid_combined.numel() == 0:
            raise ValueError("at least one valid advantage is required")
        denominator = valid_combined.std(unbiased=False).clamp_min(std_floor)
        return {
            "raw_mean_term": raw_mean,
            "raw_risk_term": raw_risk,
            "raw_combined": raw_combined,
            "mean_term": raw_mean / denominator,
            "risk_term": raw_risk / denominator,
            "combined": raw_combined / denominator,
            "combined_normalization_scale": denominator,
        }
