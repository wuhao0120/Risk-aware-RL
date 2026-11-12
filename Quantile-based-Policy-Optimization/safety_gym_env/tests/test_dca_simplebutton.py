"""Exact regression cases for cost-side port of the synthetic transition estimator."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dca_simplebutton import pinball_loss, training_targets, transition_credit, upper_tail


class TestCostTransition(unittest.TestCase):
    def test_strict_budget_and_terminal_replacement(self):
        current = torch.tensor([0.3, 0.4, 0.6])
        successor = torch.tensor([0.7, 0.99, 0.01])
        result = transition_credit(
            current,
            successor,
            torch.tensor([False, True, True]),
            torch.tensor([0.5, 0.0, -1.0]),
        )
        torch.testing.assert_close(result, torch.tensor([0.4, -0.4, 0.4]))

    def test_cdf_interpolation_support_and_ties(self):
        z = torch.tensor([[0.0, 0.0, 1.0, 2.0]]).expand(5, -1)
        tail = upper_tail(z, torch.tensor([-0.1, 0.0, 0.5, 1.0, 2.0]))
        torch.testing.assert_close(tail, torch.tensor([1.0, 0.625, 0.5, 0.375, 0.0]))

    def test_discounted_threshold_and_reward_suffix(self):
        costs = torch.tensor([[1., 2., 14.]])
        reward = torch.tensor([[1., -2., 3.]])
        x = torch.zeros(1, 3, 2)
        x[0, :, -2] = torch.tensor([15., 14., 12.]) / 15
        for gamma in (1., .99):
            returns, b = training_targets(x, reward, costs, gamma)
            torch.testing.assert_close(returns[0], torch.tensor(
                [1 - 2*gamma + 3*gamma**2, -2 + 3*gamma, 3.]))
            torch.testing.assert_close(gamma*b[:, 1:], b[:, :-1] - costs[:, :-1]/15)
            self.assertEqual(bool(b[0, -1] < costs[0, -1]/15),
                             bool((costs[0] * gamma**torch.arange(3)).sum() > 15))

    def test_pinball_matches_full_pairwise_values_and_gradients(self):
        torch.manual_seed(7)
        z = torch.randn(8, 16, requires_grad=True)
        targets = torch.randn_like(z)
        tau = (torch.arange(16) + 0.5) / 16
        delta = targets[:, None, :] - z[:, :, None]
        literal = ((tau[:, None] - (delta < 0).float()) * delta).mean()
        fast = pinball_loss(z, targets, tau)
        torch.testing.assert_close(fast, literal)
        torch.testing.assert_close(
            torch.autograd.grad(fast, z)[0], torch.autograd.grad(literal, z)[0]
        )


if __name__ == "__main__":
    unittest.main()
