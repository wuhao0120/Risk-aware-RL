# -*- coding: utf-8 -*-
"""Author-code-compatible distributional helpers for shifted-Weibull QCPO.

The control flow and reductions in this module intentionally follow Jung et
al.'s released ``dist_rl_utils.py`` and its rlpyt dependencies.  Inputs are
time-major, normally ``[T,B,...]``.  The only mathematical extension is the
three-parameter Weibull support ``x > location`` required by signed internal
cost returns.

Two historical details are preserved deliberately:

* quantile-Huber and Weibull fitting accept ``valid`` for call compatibility
  but reduce over every element, exactly as the released helper functions do;
* density log-ratios are centered across ``T,B`` and divided by ``5*std``
  without an epsilon before clipping, and the done mask is applied last.

The frozen author-aligned variant additionally sorts and detaches both
distributional TD targets and Weibull tail targets before they enter a loss.
"""

from typing import Optional, Tuple

import torch


def valid_mean(
    tensor: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
    dim=None,
) -> torch.Tensor:
    """Return rlpyt's exact optional-mask mean, with no denominator clamp."""

    reduction_dim = () if dim is None else dim
    if valid is None:
        return tensor.mean(dim=reduction_dim)
    valid = valid.to(dtype=tensor.dtype)
    return (
        (tensor * valid).sum(dim=reduction_dim)
        / valid.sum(dim=reduction_dim)
    )


def normalize(
    value: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Standardize the valid population using the author's sample std rule."""

    if valid is None:
        mean = value.mean()
        std = value.std()
    else:
        valid_mask = valid > 0
        mean = value[valid_mask].mean()
        std = value[valid_mask].std()

    # This clamp belongs to the released advantage-normalization helper.  It is
    # intentionally *not* reused by compute_prob_ratio(), whose standardization
    # has no epsilon/floor in the official code.
    return (value - mean) / torch.clamp(std, min=1e-6)


def valid_from_done(done: torch.Tensor) -> torch.Tensor:
    """Mark the first terminal row valid and every later row in that lane invalid."""

    done_float = done.to(dtype=torch.float32)
    valid = torch.ones_like(done_float)
    if done_float.shape[0] > 1:
        valid[1:] = 1.0 - torch.clamp(
            torch.cumsum(done_float[:-1], dim=0), max=1.0
        )
    return valid


@torch.no_grad()
def discount_return(
    reward: torch.Tensor,
    done: torch.Tensor,
    bootstrap_value: torch.Tensor,
    discount: float,
    return_dest: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the released rlpyt backward discounted-return recursion."""

    return_ = (
        return_dest
        if return_dest is not None
        else torch.zeros_like(reward)
    )
    not_done = (1 - done).to(dtype=reward.dtype)
    return_[-1] = (
        reward[-1]
        + float(discount) * bootstrap_value * not_done[-1]
    )
    for time in reversed(range(len(reward) - 1)):
        return_[time] = (
            reward[time]
            + float(discount) * return_[time + 1] * not_done[time]
        )
    return return_


@torch.no_grad()
def generalized_advantage_estimation(
    reward: torch.Tensor,
    value: torch.Tensor,
    done: torch.Tensor,
    bootstrap_value: torch.Tensor,
    discount: float,
    gae_lambda: float,
    advantage_dest: Optional[torch.Tensor] = None,
    return_dest: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Port rlpyt GAE exactly for time-major scalar values."""

    advantage = (
        advantage_dest
        if advantage_dest is not None
        else torch.zeros_like(reward)
    )
    return_ = (
        return_dest
        if return_dest is not None
        else torch.zeros_like(reward)
    )
    not_done = (1 - done).to(dtype=reward.dtype)

    advantage[-1] = (
        reward[-1]
        + float(discount) * bootstrap_value * not_done[-1]
        - value[-1]
    )
    for time in reversed(range(len(reward) - 1)):
        delta = (
            reward[time]
            + float(discount) * value[time + 1] * not_done[time]
            - value[time]
        )
        advantage[time] = (
            delta
            + float(discount)
            * float(gae_lambda)
            * not_done[time]
            * advantage[time + 1]
        )
    return_.copy_(advantage + value)
    return advantage, return_


def quantile_huber_loss(
    quantiles: torch.Tensor,
    target_quantiles: torch.Tensor,
    tau: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the official all-pairs unit-threshold quantile-Huber mean.

    ``valid`` is intentionally ignored.  Target atoms are sorted and detached
    here as a second safety boundary even though quantile_target_estimation()
    already returns a sorted no-grad tensor in the intended training path.
    """

    del valid
    target_sorted = torch.sort(
        target_quantiles, dim=-1
    ).values.detach()
    pairwise_delta = (
        target_sorted[..., None, :] - quantiles[..., :, None]
    )
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(
        abs_pairwise_delta > 1.0,
        abs_pairwise_delta - 0.5,
        0.5 * pairwise_delta.square(),
    )

    # Prediction atom i receives the midpoint weight tau_i against every
    # detached target atom j.  One global mean matches the released code.
    leading_ones = (1,) * (pairwise_delta.ndim - 2)
    tau_view = tau.reshape(*leading_ones, -1, 1)
    weight = torch.abs(
        tau_view - (pairwise_delta < 0).to(pairwise_delta.dtype)
    )
    return (weight * huber_loss).mean()


def weibull_tail_loss(
    quantiles: torch.Tensor,
    c_w_location: torch.Tensor,
    c_w_alpha: torch.Tensor,
    c_w_beta: torch.Tensor,
    c_tau: torch.Tensor,
    tail_ind: int,
    valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fit a shifted Weibull to sorted, detached right-tail quantile atoms.

    This is the official log-quantile regression with ``log(q)`` replaced by
    ``log(q-location)``.  The model guarantees location is below its minimum
    atom, so no numerical clamp is inserted into this author-aligned formula.
    ``valid`` remains intentionally unused, as in the released implementation.
    """

    del valid
    quantiles_sorted = torch.sort(
        quantiles, dim=-1
    ).values.detach()
    tail_quantiles = quantiles_sorted[..., int(tail_ind):]
    log_tail_c_tau = torch.log(c_tau[int(tail_ind):])

    # Broadcast tau constants across all time/batch leading dimensions.
    tau_shape = (1,) * c_w_location.ndim + (log_tail_c_tau.numel(),)
    fitted_log_offset = (
        torch.log(c_w_beta).unsqueeze(-1)
        + c_w_alpha.reciprocal().unsqueeze(-1)
        * log_tail_c_tau.reshape(tau_shape)
    )
    observed_log_offset = torch.log(
        tail_quantiles - c_w_location.unsqueeze(-1)
    )
    return (0.5 * (fitted_log_offset - observed_log_offset).square()).mean()


# The explicit name documents the only mathematical extension while the short
# name above preserves the official call surface.
shifted_weibull_tail_loss = weibull_tail_loss


@torch.no_grad()
def quantile_target_estimation(
    reward: torch.Tensor,
    r_dist: torch.Tensor,
    done: torch.Tensor,
    bootstrap_r_dist: torch.Tensor,
    discount: float,
) -> torch.Tensor:
    """Build sorted one-step distributional TD targets for every time row."""

    # The frozen variant requires all targets to be ordered before entering the
    # pairwise loss.  Detach is explicit despite no_grad so this contract stays
    # visible if the decorator is ever refactored.
    r_dist_sorted = torch.sort(r_dist, dim=-1).values.detach()
    bootstrap_sorted = torch.sort(
        bootstrap_r_dist, dim=-1
    ).values.detach()
    if bootstrap_sorted.ndim == r_dist_sorted.ndim:
        if bootstrap_sorted.shape[0] != 1:
            raise ValueError(
                "bootstrap distribution with a time axis must have length one"
            )
        bootstrap_sorted = bootstrap_sorted.squeeze(0)

    quantile_target = torch.zeros_like(r_dist_sorted)
    not_done = (1 - done).to(dtype=reward.dtype)
    quantile_target[-1] = (
        reward[-1].unsqueeze(-1)
        + float(discount)
        * bootstrap_sorted
        * not_done[-1].unsqueeze(-1)
    )
    for time in reversed(range(len(reward) - 1)):
        quantile_target[time] = (
            reward[time].unsqueeze(-1)
            + float(discount)
            * r_dist_sorted[time + 1]
            * not_done[time].unsqueeze(-1)
        )
    return quantile_target


@torch.no_grad()
def gae_quantile_simple(
    reward: torch.Tensor,
    r_dist: torch.Tensor,
    done: torch.Tensor,
    r_bdist: torch.Tensor,
    discount: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the author's atom-wise GAE after sorting distributions by value."""

    r_dist_sorted = torch.sort(r_dist, dim=-1).values.detach()
    r_bdist_sorted = torch.sort(r_bdist, dim=-1).values.detach()
    if r_bdist_sorted.ndim == r_dist_sorted.ndim:
        if r_bdist_sorted.shape[0] != 1:
            raise ValueError(
                "bootstrap distribution with a time axis must have length one"
            )
        r_bdist_sorted = r_bdist_sorted.squeeze(0)

    advantage_quantile = torch.zeros_like(r_dist_sorted)
    return_quantile = torch.zeros_like(r_dist_sorted)
    not_done = (1 - done).to(dtype=reward.dtype)

    advantage_quantile[-1] = (
        reward[-1].unsqueeze(-1)
        + not_done[-1].unsqueeze(-1)
        * float(discount)
        * r_bdist_sorted
        - r_dist_sorted[-1]
    )
    for time in reversed(range(len(reward) - 1)):
        delta = (
            reward[time].unsqueeze(-1)
            + not_done[time].unsqueeze(-1)
            * float(discount)
            * r_dist_sorted[time + 1]
            - r_dist_sorted[time]
        )
        advantage_quantile[time] = (
            delta
            + float(discount)
            * float(gae_lambda)
            * not_done[time].unsqueeze(-1)
            * advantage_quantile[time + 1]
        )
    return_quantile.copy_(advantage_quantile + r_dist_sorted)
    return advantage_quantile, return_quantile


def _expand_parameter_trailing(
    parameter: torch.Tensor,
    target_ndim: int,
) -> torch.Tensor:
    """Append singleton atom axes until ``parameter`` can broadcast to a target."""

    while parameter.ndim < int(target_ndim):
        parameter = parameter.unsqueeze(-1)
    return parameter


def shifted_weibull_log_pdf(
    value: torch.Tensor,
    location: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the shifted-Weibull log density and return ``-inf`` outside support."""

    # Time/batch parameters are normally ``[T,B]`` while atom values are
    # ``[T,B,K]``.  PyTorch aligns dimensions from the right, so append the
    # missing atom axes explicitly instead of relying on accidental shapes.
    location = _expand_parameter_trailing(location, value.ndim)
    alpha = _expand_parameter_trailing(alpha, value.ndim)
    beta = _expand_parameter_trailing(beta, value.ndim)
    offset = value - location
    supported = offset > 0
    # Replace unsupported offsets only while evaluating the branch expression; torch.where then
    # restores exact -inf, avoiding NaNs from eagerly computing log(non-positive).
    safe_offset = torch.where(supported, offset, torch.ones_like(offset))
    inside = (
        torch.log(alpha)
        - alpha * torch.log(beta)
        + (alpha - 1.0) * torch.log(safe_offset)
        - torch.pow(safe_offset / beta, alpha)
    )
    return torch.where(
        supported, inside, torch.full_like(inside, -torch.inf))


def shifted_weibull_pdf(
    value: torch.Tensor,
    location: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Return the exact three-parameter density, including zero outside ``value>location``."""

    return torch.exp(shifted_weibull_log_pdf(value, location, alpha, beta))


def shifted_weibull_quantile(
    tau: torch.Tensor,
    location: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Evaluate ``location + beta*(-log(1-tau))**(1/alpha)`` with broadcast semantics."""

    tau = torch.as_tensor(tau, dtype=beta.dtype, device=beta.device)
    if bool(((tau < 0) | (tau >= 1)).any()):
        raise ValueError("shifted Weibull quantile levels must lie in [0,1)")

    # A one-dimensional tau grid denotes a trailing atom axis for non-scalar
    # time/batch parameters.  A fully shaped tau tensor keeps elementwise
    # semantics.  Both forms are used by tests and diagnostic callers.
    if tau.ndim == 1 and location.ndim > 0:
        tau = tau.reshape((1,) * location.ndim + (tau.shape[0],))
    target_ndim = max(tau.ndim, location.ndim)
    location = _expand_parameter_trailing(location, target_ndim)
    alpha = _expand_parameter_trailing(alpha, target_ndim)
    beta = _expand_parameter_trailing(beta, target_ndim)
    return location + beta * torch.pow(-torch.log1p(-tau), alpha.reciprocal())


def _bootstrap_row(
    value: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Accept official ``[1,B]`` or convenient ``[B]`` bootstrap parameters."""

    if value.ndim == reference.ndim - 1:
        return value.unsqueeze(0)
    if value.ndim == reference.ndim and value.shape[0] == 1:
        return value
    raise ValueError(
        "bootstrap Weibull parameter must have shape [B] or [1,B]; "
        f"received {tuple(value.shape)}"
    )


@torch.no_grad()
def compute_prob_ratio(
    cost: torch.Tensor,
    c_weibull_tail: torch.Tensor,
    c_w_location: torch.Tensor,
    c_w_alpha: torch.Tensor,
    c_w_beta: torch.Tensor,
    done: torch.Tensor,
    c_bw_location: torch.Tensor,
    c_bw_alpha: torch.Tensor,
    c_bw_beta: torch.Tensor,
    discount: float,
    log_clip_range: float = 0.2,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute the released reverse-density ratio on shifted-Weibull support.

    For tail atom ``q_t``, the numerator is the next-state density at
    ``(q_t-cost_t)/discount`` and the denominator is
    ``discount * current_density(q_t)``.  Impossible next-support points first
    receive raw log-ratio zero, matching the official positive-support code.
    Standardization includes those zeros and terminal rows; done masking occurs
    only after center/(5*std) and clipping.  Impossible support is then reset
    to neutral log-weight zero without changing those normalization statistics.
    """

    support_epsilon = 1e-3
    not_done = (1 - done).to(dtype=cost.dtype)

    bootstrap_location = _bootstrap_row(c_bw_location, c_w_location)
    bootstrap_alpha = _bootstrap_row(c_bw_alpha, c_w_alpha)
    bootstrap_beta = _bootstrap_row(c_bw_beta, c_w_beta)
    next_location = torch.cat(
        (c_w_location[1:], bootstrap_location), dim=0
    )
    next_alpha = torch.cat((c_w_alpha[1:], bootstrap_alpha), dim=0)
    next_beta = torch.cat((c_w_beta[1:], bootstrap_beta), dim=0)

    transformed_value = (
        c_weibull_tail - cost.unsqueeze(-1)
    ) / float(discount)
    transformed_offset = transformed_value - next_location.unsqueeze(-1)
    possible = transformed_offset > support_epsilon
    safe_transformed_offset = torch.clamp(
        transformed_offset, min=support_epsilon
    )

    current_offset = (
        c_weibull_tail - c_w_location.unsqueeze(-1)
    )
    current_alpha = c_w_alpha.unsqueeze(-1)
    current_beta = c_w_beta.unsqueeze(-1)
    next_alpha_expanded = next_alpha.unsqueeze(-1)
    next_beta_expanded = next_beta.unsqueeze(-1)

    # Expand the official two-parameter log-PDF by substituting shifted
    # offsets.  Model-side location construction guarantees current_offset>0.
    current_log_density = (
        torch.log(current_alpha)
        - current_alpha * torch.log(current_beta)
        + (current_alpha - 1.0) * torch.log(current_offset)
        - torch.pow(current_offset / current_beta, current_alpha)
    )
    next_log_density = (
        torch.log(next_alpha_expanded)
        - next_alpha_expanded * torch.log(next_beta_expanded)
        + (next_alpha_expanded - 1.0)
        * torch.log(safe_transformed_offset)
        - torch.pow(
            safe_transformed_offset / next_beta_expanded,
            next_alpha_expanded,
        )
    )

    log_prob_ratio = torch.zeros_like(c_weibull_tail)
    discount_tensor = torch.as_tensor(
        float(discount),
        dtype=c_weibull_tail.dtype,
        device=c_weibull_tail.device,
    )
    candidate = (
        next_log_density - torch.log(discount_tensor) - current_log_density
    )
    log_prob_ratio[possible] = candidate[possible]

    if normalize:
        # Exact official ordering: center over both T and B independently for
        # each tail atom, divide by 5*sample_std with NO epsilon, then clip.
        average_log = torch.mean(log_prob_ratio, dim=(0, 1))
        std_log = torch.std(log_prob_ratio, dim=(0, 1))
        log_prob_ratio = (
            log_prob_ratio - average_log.unsqueeze(0).unsqueeze(0)
        ) / (5.0 * std_log.unsqueeze(0).unsqueeze(0))

    # Applying done after normalization/clipping is observable because terminal
    # raw ratios still influence the batch statistics.  Keep this order exact.
    log_prob_ratio = not_done.unsqueeze(-1) * torch.clamp(
        log_prob_ratio,
        min=-float(log_clip_range),
        max=float(log_clip_range),
    )

    # The shifted support adds one required safety rule: impossible transformed
    # targets are finally restored to log-weight zero.  Their raw zeros still
    # influenced center/std above, but their actor multiplier is exactly one.
    log_prob_ratio = torch.where(
        possible, log_prob_ratio, torch.zeros_like(log_prob_ratio)
    )
    return torch.exp(log_prob_ratio)


__all__ = [
    "compute_prob_ratio",
    "discount_return",
    "gae_quantile_simple",
    "generalized_advantage_estimation",
    "normalize",
    "quantile_huber_loss",
    "quantile_target_estimation",
    "shifted_weibull_log_pdf",
    "shifted_weibull_pdf",
    "shifted_weibull_quantile",
    "shifted_weibull_tail_loss",
    "valid_from_done",
    "valid_mean",
    "weibull_tail_loss",
]
