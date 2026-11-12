# -*- coding: utf-8 -*-
"""Focused regression tests for DQC validation-only critic calibration diagnostics."""
import math

import torch

from utils.evaluation import (
    _reward_critic_initial_stats,
    reward_critic_calibration_statistics,
)


def test_reward_critic_calibration_matches_hand_computed_scores():
    """
    Verify CDF/Brier/ECE/crossing/coverage conventions on one transparent four-row example.

    The third row intentionally reverses all four quantile heads. Calibration must preserve nominal
    head identity rather than sort outputs, otherwise both crossing and coverage defects disappear.
    """
    atoms = torch.tensor([
        [-2.0, -1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0, 2.0],
        [1.0, 0.0, -1.0, -2.0],
        [1.0, 2.0, 3.0, 4.0],
    ])
    returns = torch.tensor([-1.0, 1.0, -0.5, 2.0])

    metrics = reward_critic_calibration_statistics(
        atoms, returns, threshold=0.0, alpha=0.2)

    # Per-row CDFs are [.75,.50,.75,0], labels are [1,0,1,0].
    assert metrics["sample_count"] == 4
    assert metrics["initial_cdf_mean"] == 0.5
    assert metrics["empirical_failure_probability"] == 0.5
    assert metrics["initial_cdf_gap"] == 0.0
    assert metrics["initial_cdf_abs_gap"] == 0.0
    assert metrics["brier_score"] == 0.09375
    assert math.isclose(metrics["action_cdf_std"], math.sqrt(0.09375), rel_tol=1e-6)
    assert metrics["cdf_ece_10bin"] == 0.25
    assert metrics["cdf_zero_fraction"] == 0.25
    assert metrics["cdf_one_fraction"] == 0.0
    assert metrics["cdf_saturation_fraction"] == 0.25

    # One of four rows crosses, and its three reversed pairs are 3 of 12 total adjacent pairs.
    assert metrics["atom_crossing_row_fraction"] == 0.25
    assert metrics["atom_crossing_adjacent_fraction"] == 0.25

    # Marginal coverages [.25,.75,.75,.75] compare with taus [.125,.375,.625,.875].
    assert metrics["quantile_coverage_mae"] == 0.1875
    assert metrics["alpha_atom_level"] == 0.125
    assert metrics["alpha_atom_coverage"] == 0.25
    assert metrics["alpha_atom_coverage_error"] == 0.125
    assert metrics["predicted_return_mean"] == 0.5
    assert metrics["return_mean_bias"] == 0.125
    assert math.isclose(metrics["return_rmse"], math.sqrt(0.1875), rel_tol=1e-6)


class _RecordingCritic:
    """Capture the action supplied by evaluation while returning two ordered atoms per row."""
    def __init__(self):
        self.actions = None

    def __call__(self, states, actions):
        self.actions = actions.detach().clone()
        return torch.stack((actions[:, 0] - 1.0, actions[:, 0] + 1.0), dim=1)


class _ExecutedActionAgent:
    """Minimal agent contract proving that critic diagnostics use executed, not latent, actions."""
    def __init__(self):
        self.reward_critic = _RecordingCritic()

    @staticmethod
    def _executed_actions(latent_actions):
        return torch.clamp(latent_actions, -1.0, 1.0)


def test_initial_critic_calibration_uses_executed_actions():
    agent = _ExecutedActionAgent()
    states = torch.zeros(2, 3)
    latent_actions = torch.tensor([[2.5], [-3.0]])
    returns = torch.tensor([0.0, 0.0])

    metrics = _reward_critic_initial_stats(
        agent, states, latent_actions, returns, threshold=0.0, alpha=0.2)

    # Safety Gym would execute [+1,-1]; passing latent [+2.5,-3] would fail this exact assertion.
    assert torch.equal(agent.reward_critic.actions, torch.tensor([[1.0], [-1.0]]))
    assert metrics["sample_count"] == 2
    assert math.isfinite(metrics["brier_score"])


def test_calibration_failure_label_preserves_primary_float64_threshold_semantics():
    """A promoted float32 return just above q=0.1 must not become an artificial tie."""
    observed_return = float(torch.tensor(0.1, dtype=torch.float32).item())
    assert observed_return > 0.1
    metrics = reward_critic_calibration_statistics(
        torch.tensor([[0.0]]), [observed_return], threshold=0.1, alpha=0.2)
    assert metrics["empirical_failure_probability"] == 0.0


def test_single_atom_calibration_defines_crossing_as_zero():
    """K=1 has no adjacent quantile pair and must not produce NaN from an empty mean."""
    metrics = reward_critic_calibration_statistics(
        torch.tensor([[-1.0], [1.0]]),
        torch.tensor([-2.0, 2.0]),
        threshold=0.0,
        alpha=0.2,
    )
    assert metrics["atom_crossing_row_fraction"] == 0.0
    assert metrics["atom_crossing_adjacent_fraction"] == 0.0
    assert metrics["alpha_atom_level"] == 0.5
