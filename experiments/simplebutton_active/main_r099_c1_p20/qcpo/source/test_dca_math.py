"""CPU checks for the algorithm semantics of the Scheme B numerical kernels."""

import unittest

import torch

from schemeb_math import (
    combine_actor_advantages,
    midpoint_quantile_levels,
    pairwise_pinball_loss,
    sorted_quantile_cdf,
    strict_terminal_cost_violation,
    upper_cost_violation_probability,
)


class PinballTests(unittest.TestCase):
    def test_exact_all_pairs_reference_and_target_stop_gradient(self):
        prediction = torch.tensor([[2., -1., 3.], [-2., 2., 4.]], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[0., 5.], [-3., 1.]], dtype=torch.float64, requires_grad=True)
        actual = pairwise_pinball_loss(prediction, target, reduction="none")
        expected = []
        for batch in range(2):
            terms = []
            for head in range(3):
                tau = (head + 0.5) / 3
                for sample in range(2):
                    delta = float(target[batch, sample].detach() - prediction[batch, head].detach())
                    terms.append(max(tau * delta, (tau - 1) * delta))
            expected.append(sum(terms) / len(terms))
        torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64))
        actual.mean().backward()
        self.assertIsNone(target.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())
        torch.testing.assert_close(prediction.grad, torch.tensor([[1/18, -1/12, -1/18], [1/18, 1/12, 1/36]], dtype=torch.float64))

    def test_crossed_heads_keep_their_assigned_levels(self):
        prediction = torch.tensor([[3., -1.]], requires_grad=True)
        target = torch.tensor([[0.]])
        loss = pairwise_pinball_loss(prediction, target)
        self.assertAlmostEqual(float(loss), 1.5)
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([[.375, -.375]]))
        # Sorting the predictions during fitting changes this objective.
        self.assertNotEqual(float(loss), float(pairwise_pinball_loss(prediction.detach().sort().values, target)))

    def test_pure_pinball_has_no_huber_quadratic_region(self):
        q = torch.tensor([[0.]], requires_grad=True)
        loss = pairwise_pinball_loss(q, torch.tensor([[1e-4]]))
        loss.backward()
        self.assertAlmostEqual(float(loss), 5e-5, places=10)
        self.assertEqual(float(q.grad), -.5)
        torch.testing.assert_close(midpoint_quantile_levels(4), torch.tensor([.125, .375, .625, .875]))

    def test_shapes_and_reductions_are_explicit(self):
        q = torch.zeros(2, 3, 4)
        y = torch.ones(2, 3, 7)
        per_sample = pairwise_pinball_loss(q, y, reduction="none")
        self.assertEqual(per_sample.shape, (2, 3))
        self.assertEqual(float(pairwise_pinball_loss(q, y)), .5)
        self.assertEqual(float(pairwise_pinball_loss(q, y, reduction="sum")), 3.)
        with self.assertRaises(ValueError):
            pairwise_pinball_loss(q, torch.ones(3, 7))


class CdfTests(unittest.TestCase):
    def test_sorting_interpolation_and_upper_tail(self):
        q = torch.tensor([[4., 0., 2., 1.]]).repeat(7, 1).requires_grad_()
        b = torch.tensor([-1., 0., .5, 1.5, 3., 4., 5.])
        cdf = sorted_quantile_cdf(q, b)
        torch.testing.assert_close(cdf, torch.tensor([0., .125, .25, .5, .75, 1., 1.]))
        torch.testing.assert_close(upper_cost_violation_probability(q, b), 1 - cdf)
        self.assertFalse(cdf.requires_grad)
        # Sorting for reading does not mutate the trainable head order.
        torch.testing.assert_close(q[0].detach(), torch.tensor([4., 0., 2., 1.]))

    def test_duplicate_quantiles_have_explicit_right_continuity(self):
        q = torch.tensor([[0., 1., 1., 3.]], dtype=torch.float64).repeat(6, 1)
        b = torch.tensor([1.-1e-9, 1., 1.+1e-9, 2., 3., 4.], dtype=torch.float64)
        f = sorted_quantile_cdf(q, b)
        self.assertLess(float(f[0]), .375)
        self.assertEqual(float(f[1]), .625)
        self.assertAlmostEqual(float(f[2]), .625, places=8)
        self.assertEqual(float(f[3]), .75)
        self.assertEqual(float(f[4]), 1.)
        self.assertEqual(float(f[5]), 1.)
        self.assertTrue(torch.isfinite(f).all())

    def test_point_mass_strict_event_and_missing_tail_saturation(self):
        q = torch.full((3, 128), 15.)
        b = torch.tensor([14., 15., 16.])
        torch.testing.assert_close(upper_cost_violation_probability(q, b), torch.tensor([1., 0., 0.]))
        torch.testing.assert_close(strict_terminal_cost_violation(b), torch.tensor([0., 0., 1.]))
        # Better CDF reading alone cannot recover a tail that no quantile learned.
        underfit_q = torch.linspace(0., 5., 128).unsqueeze(0)
        self.assertEqual(float(upper_cost_violation_probability(underfit_q, 15.)), 0.)

    def test_broadcast_single_head_and_tiny_gap_are_finite(self):
        q = torch.tensor([[[0., 1., 2.], [2., 0., 1.]]])
        torch.testing.assert_close(sorted_quantile_cdf(q, 1.), torch.tensor([[.5, .5]]))
        torch.testing.assert_close(sorted_quantile_cdf(q, torch.tensor([[[1.], [1.]]])), torch.tensor([[.5, .5]]))
        torch.testing.assert_close(sorted_quantile_cdf(torch.tensor([[2.], [2.]]), torch.tensor([1., 2.])), torch.tensor([0., 1.]))
        tiny = sorted_quantile_cdf(torch.tensor([0., 1e-12], dtype=torch.float64), 5e-13)
        self.assertTrue(torch.isfinite(tiny))
        self.assertGreaterEqual(float(tiny), 0.)
        self.assertLessEqual(float(tiny), 1.)


class ActorLabelTests(unittest.TestCase):
    def test_mean_divisor_changes_relative_weight_even_with_common_std(self):
        reward = torch.tensor([20., 40., -20., 60.], requires_grad=True)
        risk = torch.tensor([.1, -.2, .3, -.1], requires_grad=True)
        lam = torch.tensor(2., requires_grad=True)
        c100 = combine_actor_advantages(reward, risk, lam, mean_divisor=100)
        c20 = combine_actor_advantages(reward, risk, lam, mean_divisor=20)
        torch.testing.assert_close(c20["raw_mean_term"], 5 * c100["raw_mean_term"])
        torch.testing.assert_close(c20["raw_risk_term"], c100["raw_risk_term"])
        ratio100 = c100["mean_term"].norm() / c100["risk_term"].norm()
        ratio20 = c20["mean_term"].norm() / c20["risk_term"].norm()
        torch.testing.assert_close(ratio20, 5 * ratio100)
        for labels in (c100, c20):
            self.assertTrue(all(not t.requires_grad for t in labels.values()))
            torch.testing.assert_close(labels["combined"], labels["mean_term"] - labels["risk_term"])

    def test_valid_population_std_without_centering_or_outer_lambda(self):
        reward = torch.tensor([20., 40., 90000.])
        risk = torch.tensor([.1, -.1, -9000.])
        labels = combine_actor_advantages(reward, risk, 2., mean_divisor=20., valid_mask=torch.tensor([True, True, False]))
        # Raw combined [.8, 2.2] has population std .7 and nonzero mean 1.5.
        torch.testing.assert_close(labels["combined_normalization_scale"], torch.tensor(.7))
        torch.testing.assert_close(labels["combined"][:2], torch.tensor([.8, 2.2]) / .7)
        torch.testing.assert_close(labels["mean_term"][:2], torch.tensor([1., 2.]) / .7)
        torch.testing.assert_close(labels["risk_term"][:2], torch.tensor([.2, -.2]) / .7)

    def test_floor_and_single_valid_sample(self):
        labels = combine_actor_advantages(torch.tensor([1.]), torch.tensor([.0]), 2., mean_divisor=20.)
        self.assertEqual(float(labels["combined_normalization_scale"]), float(torch.tensor(.01)))
        torch.testing.assert_close(labels["combined"], torch.tensor([5.]))
        with self.assertRaises(ValueError):
            combine_actor_advantages(torch.ones(2), torch.zeros(2), 1., mean_divisor=20., valid_mask=torch.zeros(2, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
