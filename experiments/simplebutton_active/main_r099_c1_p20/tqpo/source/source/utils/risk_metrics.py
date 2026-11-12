# -*- coding: utf-8 -*-
"""Reward-side chance-constraint statistics with explicit empirical conventions."""
import math
import numpy as np


def linear_empirical_quantile(values, alpha):
    """Return NumPy's linearly interpolated empirical alpha-percentile."""
    sample = np.asarray(values, dtype=np.float64).reshape(-1)
    if sample.size == 0:
        raise ValueError("linear empirical quantile needs at least one sample")
    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError(f"alpha must be in (0,1], got {alpha}")
    # Explicit method="linear" freezes the NumPy convention across environments and versions.
    return float(np.percentile(sample, 100.0 * float(alpha), method="linear"))


def wilson_interval(successes, total, confidence=0.95):
    """Wilson score interval for a binomial probability (95% uses z=1.95996)."""
    if total <= 0:
        raise ValueError("Wilson interval needs total > 0")
    if not 0 <= successes <= total:
        raise ValueError("successes must lie in [0,total]")
    # The project uses 95% reporting; reject silent approximations for other confidence levels.
    if not math.isclose(float(confidence), 0.95):
        raise ValueError("only confidence=0.95 is supported")
    z = 1.959963984540054
    p_hat = float(successes) / float(total)
    z2_over_n = z * z / total
    denominator = 1.0 + z2_over_n
    center = (p_hat + 0.5 * z2_over_n) / denominator
    radius = z * math.sqrt((p_hat * (1.0 - p_hat) / total)
                           + (z * z / (4.0 * total * total))) / denominator
    return float(max(0.0, center - radius)), float(min(1.0, center + radius))


def return_statistics(returns, alpha, threshold=None, near_radius=None):
    """
    Compute primary raw-return metrics.

    Failure is inclusive: 1{G_1000 <= q}. Quantile uses NumPy linear interpolation and is
    diagnostic only; feasibility always uses direct event counting.
    """
    sample = np.asarray(returns, dtype=np.float64).reshape(-1)
    if sample.size == 0:
        raise ValueError("return statistics need at least one sample")
    alpha = float(alpha)
    ordered = np.sort(sample)
    result = {
        "num_episodes": int(sample.size),
        "mean": float(sample.mean()),
        "reward_std": float(sample.std()),
        "quantile_return": linear_empirical_quantile(sample, alpha),
        "worst_5pct": float(ordered[:max(1, int(math.ceil(0.05 * sample.size)))].mean()),
        "minimum_return": float(ordered[0]),
        "maximum_return": float(ordered[-1]),
    }
    if threshold is None:
        return result
    q = float(threshold)
    failures = sample <= q
    failure_count = int(failures.sum())
    probability = float(failure_count / sample.size)
    lower, upper = wilson_interval(failure_count, sample.size)
    # Default band is 1% of threshold scale, floored to avoid a zero-width diagnostic.
    radius = float(near_radius) if near_radius is not None else 0.01 * max(1.0, abs(q))
    result.update({
        "return_threshold": q,
        "failure_count": failure_count,
        "empirical_prob": probability,
        "constraint_residual": probability - alpha,
        "constraint_margin": alpha - probability,
        "feasible": bool(probability <= alpha),
        "wilson95_low": lower,
        "wilson95_high": upper,
        "ties_at_threshold": int(np.count_nonzero(sample == q)),
        "near_threshold_radius": radius,
        "near_threshold_count": int(np.count_nonzero(np.abs(sample - q) <= radius)),
        "quantile_margin_to_threshold": result["quantile_return"] - q,
    })
    return result


def select_validation_candidate(candidates, alpha, qpo=False):
    """Apply the frozen checkpoint-selection rule to metric dictionaries."""
    if not candidates:
        raise ValueError("at least one candidate is required")
    if qpo:
        # QPO has a different objective: maximize validation linear percentile, then mean.
        return max(candidates, key=lambda x: (x["quantile_return"], x["mean"]))
    feasible = [x for x in candidates if x["empirical_prob"] <= float(alpha)]
    if feasible:
        return max(feasible, key=lambda x: x["mean"])
    # No feasible model: minimize residual/probability, then maximize mean.
    return min(candidates, key=lambda x: (x["constraint_residual"], -x["mean"]))
