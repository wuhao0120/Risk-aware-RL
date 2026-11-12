# -*- coding: utf-8 -*-
"""
QCPO-reference distributional helpers for the internal cost ``C=-G``.

The user-facing task constrains the reward outage ``P(G<q)``. Internally we negate rewards and
reuse the paper's cost-side derivation without changing signs by hand:

    C=-G, d=-q, P(G<q)=P(C>d), and Q^C_{1-alpha}=-Q^G_alpha.

Tensor and boundary conventions:
    - rollout tensors are time-major ``[T,B,...]``;
    - both termination and the formal H-step truncation stop bootstrap and GAE recursion;
    - quantile loss compares every predicted atom with every target atom;
    - the shifted Weibull support is ``x>location`` and its right tail models high costs;
    - Eq. (274)'s clipped log-density ratio is detached before actor optimization;
    - an out-of-support numerator receives the neutral weight one and a visible diagnostic;
    - no Safety-Gym cost channel enters these helpers: ``cost`` always means ``-reward``.
"""
import math

import torch


def masked_gae(reward, value, next_value, bootstrap_mask, continuation_mask, gamma, gae_lambda):
    """
    Compute boundary-correct GAE for scalar ``[T,B]`` signals.

    ``bootstrap_mask`` decides whether the one-step target contains ``next_value``;
    ``continuation_mask`` independently prevents the recursive GAE term from crossing an episode
    boundary. QCPO_REF passes zero for both masks at either termination or H-step truncation,
    because its formal objective is the finite-horizon return used by final evaluation.
    """
    delta = reward + float(gamma) * bootstrap_mask * next_value - value
    advantage = torch.zeros_like(delta)
    running = torch.zeros_like(delta[0])
    for t in reversed(range(delta.shape[0])):
        running = delta[t] + float(gamma) * float(gae_lambda) * continuation_mask[t] * running
        advantage[t] = running
    return advantage, advantage + value


def quantile_huber_loss(predicted, target, levels, valid=None, kappa=1.0):
    """Return QR-DQN's pairwise quantile-Huber loss for ``[T,B,N]`` atoms."""
    pairwise = target.unsqueeze(-2) - predicted.unsqueeze(-1)
    absolute = pairwise.abs()
    huber = torch.where(
        absolute <= float(kappa),
        0.5 * pairwise.square(),
        float(kappa) * (absolute - 0.5 * float(kappa)),
    )
    weights = torch.abs(levels.view(1, 1, -1, 1) - (pairwise.detach() < 0).float())
    per_transition = (weights * huber / float(kappa)).mean(dim=(-1, -2))
    if valid is None:
        return per_transition.mean()
    mask = valid.to(per_transition.dtype)
    return (per_transition * mask).sum() / torch.clamp(mask.sum(), min=1.0)


def shifted_weibull_log_pdf(x, location, scale, shape, epsilon=1e-8):
    """
    Evaluate the log PDF of a three-parameter Weibull on ``x>location``.

    The PDF is ``k/beta*z**(k-1)*exp(-z**k)``, where
    ``z=(x-location)/beta``. Returning the support mask separately lets the caller apply the
    paper implementation's neutral ratio at impossible transformed targets instead of creating a
    huge ratio by clamping an invalid point onto the boundary.
    """
    safe_scale = scale.clamp(min=epsilon)
    normalized = (x - location) / safe_scale
    inside = normalized > 0
    safe_normalized = normalized.clamp(min=epsilon)

    # Compute z**k in log space. The exponent cap avoids inf-inf when both current and next PDFs
    # are extremely small; values above exp(80) are already indistinguishable for the clipped
    # log-density ratio used by Eq. (274).
    log_z = torch.log(safe_normalized)
    powered = torch.exp((shape * log_z).clamp(max=80.0))
    log_density = (
        torch.log(shape.clamp(min=epsilon))
        - torch.log(safe_scale)
        + (shape - 1.0) * log_z
        - powered
    )
    return log_density, inside


def shifted_weibull_pdf(x, location, scale, shape, epsilon=1e-8):
    """Return the shifted-Weibull PDF, with exact zero outside ``x>location``."""
    log_density, inside = shifted_weibull_log_pdf(
        x, location, scale, shape, epsilon=epsilon)
    density = torch.exp(log_density)
    return torch.where(inside, density, torch.zeros_like(density))


def shifted_weibull_quantile(levels, location, scale, shape, epsilon=1e-8):
    """Evaluate ``location + scale*(-log(1-tau))**(1/shape)``."""
    levels = levels.clamp(min=epsilon, max=1.0 - epsilon)
    base = -torch.log1p(-levels)
    return location.unsqueeze(-1) + scale.unsqueeze(-1) * base.pow(
        1.0 / shape.unsqueeze(-1))


def upper_weibull_fit_loss(
        quantiles, levels, location, scale, shape, fit_min=0.75, valid=None,
        epsilon=1e-8):
    """
    Fit the shifted Weibull to detached high-cost quantiles.

    The paper fits the rightmost eight of 25 cost atoms. This implementation uses 19 atoms at
    levels 0.05--0.95 and therefore selects 0.75--0.95 (five right-tail atoms), preserving the
    same roughly 30-percent tail fraction without changing the existing critic output width.
    """
    selected = levels >= float(fit_min) - 1e-7
    selected_levels = levels[selected].clamp(min=epsilon, max=1.0 - epsilon)
    target = quantiles[..., selected].detach()

    # Paper Eq. (257) regresses Weibull quantiles in log scale. With a learned shift, the positive
    # quantity is q-location rather than q itself:
    # log(q_tau-location) = log(scale) + log(-log(1-tau))/shape.
    analytic_log_offset = (
        torch.log(scale.clamp(min=epsilon)).unsqueeze(-1)
        + torch.log(-torch.log1p(-selected_levels)).view(
            *((1,) * location.ndim), -1) / shape.clamp(min=epsilon).unsqueeze(-1)
    )
    target_log_offset = torch.log(
        (target - location.unsqueeze(-1)).clamp(min=epsilon))
    per_transition = 0.5 * (analytic_log_offset - target_log_offset).square().mean(dim=-1)
    if valid is None:
        return per_transition.mean()
    mask = valid.to(per_transition.dtype)
    return (per_transition * mask).sum() / torch.clamp(mask.sum(), min=1.0)


@torch.no_grad()
def paper_density_weight(
        cost, q_current, current_params, next_params, bootstrap_mask, gamma,
        log_clip=0.5, epsilon=1e-8):
    """
    Compute the smoothed quantile-advantage weight from paper Eq. (274)--(275).

    For internal discounted cost ``C=-G``, the theorem's density ratio is

    ``mu = f_next((Q_current-cost)/gamma) / (gamma*f_current(Q_current))``.

    The final estimator does *not* multiply the TD target by raw ``mu``. It uses
    ``1+clip(log(mu),-c_clip,c_clip)`` and then multiplies the whole quantile TD/GAE advantage.
    With the paper value ``c_clip=0.5`` the returned weight is bounded in ``[0.5,1.5]``.
    """
    location, scale, shape = current_params
    next_location, next_scale, next_shape = next_params
    transformed = (q_current - cost) / float(gamma)

    current_log_pdf, current_inside = shifted_weibull_log_pdf(
        q_current, location, scale, shape, epsilon=epsilon)
    next_log_pdf, next_inside = shifted_weibull_log_pdf(
        transformed, next_location, next_scale, next_shape, epsilon=epsilon)
    log_ratio = next_log_pdf - math.log(float(gamma)) - current_log_pdf

    # A terminal/truncated transition has no next-return distribution. The reference code sets
    # its log ratio to zero; we do the same. Invalid/non-finite density evaluations are also made
    # neutral, while support_violation exposes only genuine live-transition support misses.
    live = bootstrap_mask > 0
    usable = live & current_inside & next_inside & torch.isfinite(log_ratio)
    clipped_log_ratio = torch.where(
        usable,
        log_ratio.clamp(min=-float(log_clip), max=float(log_clip)),
        torch.zeros_like(log_ratio),
    )
    weight = 1.0 + clipped_log_ratio
    support_violation = live & (~current_inside | ~next_inside)
    return weight, support_violation, clipped_log_ratio
