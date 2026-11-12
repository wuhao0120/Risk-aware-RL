# -*- coding: utf-8 -*-
"""Exact-N evaluation coverage retained after deleting replay-specific tests."""
import math

import torch

from envs.safety_env_vec import VecStepResult
from utils.evaluation import evaluate_policy_vec


class EvalActor:
    """Only zero recurrent-state construction is required by this evaluator fixture."""

    @staticmethod
    def initial_state(batch_size, device):
        return (
            torch.zeros(1, batch_size, 1, device=device),
            torch.zeros(1, batch_size, 1, device=device),
        )


class EvalAgent:
    actor = EvalActor()

    @staticmethod
    def _sample_actions(states, recurrent, return_state=False):
        del return_state
        actions = torch.zeros(states.shape[0], 1)
        return actions, recurrent, actions.clone(), actions.clone()


class EvalVector:
    B, n = 3, 1

    def __init__(self):
        self.round = -1

    def reset(self):
        self.round += 1
        return torch.zeros(self.B, 1)

    def step(self, _actions):
        rewards = torch.arange(1, 4, dtype=torch.float32) + 3 * self.round
        return VecStepResult(
            observation=torch.zeros(self.B, 1),
            reward=rewards,
            cost=torch.zeros(self.B),
            terminated=torch.zeros(self.B, dtype=torch.bool),
            truncated=torch.ones(self.B, dtype=torch.bool),
            final_observation=torch.ones(self.B, 1),
            final_observation_mask=torch.ones(self.B, dtype=torch.bool),
        )


def test_evaluation_slices_to_exact_requested_episode_count_before_statistics():
    result = evaluate_policy_vec(
        EvalAgent(), EvalVector(), num_episodes=5, gamma=0.99,
        failure_alpha=0.2, return_threshold=3.0)
    assert result["num_episodes"] == 5
    assert result["mean"] == 3.0
    assert math.isclose(result["quantile_return"], 1.8, abs_tol=1e-7)
    assert result["failure_count"] == 3
    assert math.isclose(result["quantile_q_est"], 1.8, abs_tol=1e-7)
    assert result["quantile_constraint_prob"] == 0.6


def test_evaluation_calibrates_qpo_tracker_without_changing_fixed_q_failure_count():
    agent = EvalAgent()
    agent.q_est = torch.tensor(2.0)
    result = evaluate_policy_vec(
        agent, EvalVector(), num_episodes=5, gamma=0.99,
        failure_alpha=0.2, return_threshold=3.0)
    assert result["empirical_prob"] == 0.6
    assert result["quantile_q_est"] == 2.0
    assert result["quantile_constraint_prob"] == 0.4
