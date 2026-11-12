# -*- coding: utf-8 -*-
from collections import deque
import math
import numpy as np
import torch

from utils.risk_metrics import (linear_empirical_quantile, return_statistics,
                                select_validation_candidate)
from agents.qcpo_gpu import QCPOGPU
from agents.qcpo_ref import QCPORefGPU
from agents.vec_base import VecAgentBase
from progress_monitor import _postfix, _training_line, _validation_line
from run_experiment import base_args, checkpoint_signature, scheduled_event_due


def test_checkpoint_signature_contains_the_effective_robbins_monro_actor_scale():
    # QPO/QCPO LambdaLR closes over theta_a, not merely the human-facing actor_lr0 alias.
    args = base_args("QPO", seed=3, device=torch.device("cpu"), env_key="SimpleButton")
    first = checkpoint_signature(args, "QPO")
    assert first["theta_a"] == args.theta_a
    args.theta_a *= 2.0
    second = checkpoint_signature(args, "QPO")
    assert second["theta_a"] == 2.0 * first["theta_a"]
    assert first != second


def test_qcpo_ref_checkpoint_signature_freezes_paper_negated_return_identity():
    """A pre-fix reward-left-tail checkpoint must not be shape-compatible with the new lineage."""
    args = base_args("QCPO_REF", seed=0, device=torch.device("cpu"), env_key="SimpleButton")
    signature = checkpoint_signature(args, "QCPO_REF")
    assert signature["training_budget_semantics"] == "main_excludes_initialization_v1"
    assert signature["qcpo_ref_formulation"] == "paper_negated_return_eq275_discounted_bh_v2"
    assert signature["qcpo_ref_actor_weighting"] == "discounted_gamma_t"
    assert signature["qcpo_ref_actor_reduction"] == "fixed_BH"
    assert signature["qcpo_ref_critic_input"] == "finite_obs_remaining_actor_h0c0_v1"
    assert signature["horizon_mode"] == "finite"
    assert signature["critic_time_feature"] == "remaining_fraction"
    assert signature["critic_actor_context"] is True
    assert signature["density_log_clip"] == 0.5
    assert signature["weibull_fit_min"] == 0.75
    assert signature["dual_window_size"] == 100


def test_validation_and_checkpoint_schedules_are_independent_and_zero_stays_disabled():
    # A checkpoint at batch 25 must not imply a validation when validation_every=10.
    assert scheduled_event_due(20, interval=10, last_iteration=9)
    assert not scheduled_event_due(25, interval=10, last_iteration=19)
    assert scheduled_event_due(25, interval=25, last_iteration=-1)
    # Final-force fills a missing enabled event but never resurrects an interval explicitly set 0.
    assert scheduled_event_due(23, interval=10, last_iteration=19, force=True)
    assert not scheduled_event_due(23, interval=0, last_iteration=-1, force=True)


def test_linear_percentile_and_inclusive_ties_are_not_quantile_feasibility():
    # Atom example: linear Q_.2=0 meets q=0, but ties still make failure probability .3.
    returns = np.array([0.0] * 3 + [1.0] * 7)
    stats = return_statistics(returns, alpha=0.2, threshold=0.0)
    assert linear_empirical_quantile(returns, 0.2) == 0.0
    assert stats["ties_at_threshold"] == 3
    assert stats["failure_count"] == 3
    assert stats["empirical_prob"] == 0.3
    assert stats["constraint_residual"] == pytest_approx(0.1)
    assert not stats["feasible"]


def test_linear_percentile_matches_numpy_interpolation():
    values = np.array([8.0, 1.0, 4.0, 2.0, 9.0])
    # NumPy linear positions are h=(N-1)*alpha: .8 and 1.6 for this sample.
    assert math.isclose(linear_empirical_quantile(values, 0.2), 1.8)
    assert math.isclose(linear_empirical_quantile(values, 0.4), 3.2)
    assert linear_empirical_quantile(values, 0.2) == np.percentile(
        values, 20.0, method="linear")



def test_training_metric_schema_uses_disc_reward_namespace():
    # Lock the QPO-compatible W&B names and reject removed return/tracker aliases.
    agent = object.__new__(VecAgentBase)
    agent.failure_alpha = 0.2
    agent.return_threshold = None
    agent.num_envs = 5
    agent.total_env_steps = 25
    agent.log_interval = 10
    agent.last_empirical_prob = None
    agent.last_outage_prob = None
    agent._wandb_run = None
    agent.vec_env = type("Stats", (), {"stats": lambda self: {}})()

    metrics = agent._log_core(
        iteration=0,
        returns=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        diagnostic_disc_cost=np.zeros(5),
        diagnostic_raw_cost=np.zeros(5),
        extra={"quantile/q_est": 2.5, "quantile/constraint_prob": 0.2},
    )
    assert metrics["disc_reward/aver_reward"] == 3.0
    assert metrics["disc_reward/return_std"] == np.std([1, 2, 3, 4, 5])
    assert math.isclose(metrics["disc_reward/quantile_reward"], 1.8)
    assert metrics["quantile/q_est"] == 2.5
    assert metrics["quantile/constraint_prob"] == 0.2
    removed = {
        "return/truncated_return_1000_mean", "return/truncated_return_1000_std",
        "return/left_quantile", "return/quantile_linear", "return/lower_cvar",
        "quantile/tracker_probability", "quantile/tracker_lr",
        "actor/failure_below_q_est",
    }
    assert removed.isdisjoint(metrics)


def test_main_initialization_and_actual_clocks_match_batches_200_and_400():
    """QPO/DQC initialization is visible but cannot shift the formal 5M/10M W&B axis."""
    for initialization_batches in (2, 10):
        for completed_batches in (200, 400):
            agent = object.__new__(VecAgentBase)
            agent.failure_alpha = 0.2
            agent.return_threshold = -0.03
            agent.num_envs = 25
            agent.main_env_steps = completed_batches * 25 * 1000
            agent.initialization_env_steps = initialization_batches * 25 * 1000
            agent.actual_env_steps = (
                agent.main_env_steps + agent.initialization_env_steps)
            agent.total_env_steps = agent.actual_env_steps
            agent.last_empirical_prob = None
            agent.last_outage_prob = None
            agent.vec_env = type("Stats", (), {"stats": lambda self: {}})()
            metrics = agent._log_core(
                iteration=completed_batches - 1,
                returns=np.zeros(25),
                diagnostic_disc_cost=np.zeros(25),
                diagnostic_raw_cost=np.zeros(25),
            )
            assert metrics["progress/iteration"] == completed_batches - 1
            assert metrics["progress/env_steps"] == completed_batches * 25000
            assert metrics["progress/main_env_steps"] == completed_batches * 25000
            assert metrics["progress/initialization_env_steps"] == (
                initialization_batches * 25000)
            assert metrics["progress/actual_env_steps"] == (
                (completed_batches + initialization_batches) * 25000)


def test_initialization_boundary_calls_only_save_callback_and_external_progress():
    """No W&B or validation/best callback is reachable from an initialization batch."""
    agent = object.__new__(VecAgentBase)
    agent.initialization_cursor = 3
    agent.initialization_env_steps = 75000
    agent.actual_env_steps = 75000
    agent.start_iteration = 0
    agent.algo_name, agent.env_name, agent.seed = "DQCAC", "SimpleButton", 0
    external, saved, console = [], [], []
    agent._write_external_progress = lambda **kwargs: external.append(kwargs)
    agent.initialization_callback = (
        lambda _agent, completed, total: saved.append((completed, total)))
    agent.console_write = console.append
    agent.iteration_callback = lambda *_args: (_ for _ in ()).throw(
        AssertionError("validation callback must not run during initialization"))
    agent._wandb_run = type(
        "NoWandb", (), {"log": lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("W&B must not log during initialization"))})()

    agent._after_initialization_batch(total_batches=10)
    assert saved == [(3, 10)]
    assert external == [{
        "completed": 0, "metrics": {}, "phase": "initialization"}]
    assert len(console) == 1 and "batch:003/010" in console[0]


def test_constrained_training_schema_adds_common_batch_quantile_metrics():
    # QCPO/QCPO_REF/DQC all call this common logger with the same B trajectory returns. The public
    # q_est is the linear batch percentile and constraint_prob directly counts inclusive G<=q.
    agent = object.__new__(VecAgentBase)
    agent.failure_alpha = 0.2
    agent.return_threshold = 2.0
    agent.num_envs = 5
    agent.total_env_steps = 25
    agent.last_empirical_prob = None
    agent.last_outage_prob = None
    agent.vec_env = type("Stats", (), {"stats": lambda self: {}})()
    metrics = agent._log_core(
        iteration=0,
        returns=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        diagnostic_disc_cost=np.zeros(5),
        diagnostic_raw_cost=np.zeros(5),
    )
    assert math.isclose(metrics["quantile/q_est"], 1.8)
    assert metrics["quantile/constraint_prob"] == 0.4
    assert metrics["constraint/empirical_failure_probability"] == 0.4


def test_wandb_upload_occurs_once_for_each_completed_rollout_batch():
    # Validation fields merge into the same post-update row; no duplicate W&B step is emitted.
    class FakeRun:
        def __init__(self):
            self.calls = []

        def log(self, metrics, step):
            self.calls.append((dict(metrics), int(step)))

    agent = object.__new__(VecAgentBase)
    agent.failure_alpha = 0.2
    agent.return_threshold = 1.5
    agent.num_envs = 2
    agent.num_iterations = 2
    agent.start_iteration = 0
    agent.total_env_steps = 4
    agent.log_interval = 100
    agent.last_empirical_prob = None
    agent.last_outage_prob = None
    agent.vec_env = type("Stats", (), {"stats": lambda self: {}})()
    agent._wandb_run = FakeRun()
    agent._progress_bar = None
    agent.progress_file = None
    agent._last_progress_metrics = {}
    agent.console_write = lambda _message: None
    agent.iteration_callback = lambda _agent, iteration, _metrics: (
        {"validation/disc_reward/quantile_reward": 7.0}
        if iteration == 1 else None)

    for iteration in range(2):
        metrics = agent._log_core(
            iteration=iteration,
            returns=np.array([1.0, 2.0]),
            diagnostic_disc_cost=np.zeros(2),
            diagnostic_raw_cost=np.zeros(2),
        )
        # _log_core only builds the schema; _after_iteration owns the single upload.
        assert len(agent._wandb_run.calls) == iteration
        agent._after_iteration(iteration, metrics)

    assert [step for _metrics, step in agent._wandb_run.calls] == [0, 1]
    # The canonical non-QPO fields must reach the actual run.log payload, not merely the terminal
    # fallback. For returns [1,2], linear Q_.2=1.2 and one of two trajectories fails at q=1.5.
    for uploaded, _step in agent._wandb_run.calls:
        assert math.isclose(uploaded["quantile/q_est"], 1.2)
        assert uploaded["quantile/constraint_prob"] == 0.5
    assert "validation/disc_reward/quantile_reward" not in agent._wandb_run.calls[0][0]
    assert agent._wandb_run.calls[1][0]["validation/disc_reward/quantile_reward"] == 7.0


def test_terminal_progress_uses_persistent_qpo_tracker_and_batch_probability():
    # The terminal bar must not confuse QPO's persistent q_est with the batch percentile.
    metrics = {
        "disc_reward/aver_reward": 3.0,
        "disc_reward/quantile_reward": 1.8,
        "quantile/q_est": 2.5,
        "quantile/constraint_prob": 0.2,
    }
    snapshot = VecAgentBase._progress_snapshot(metrics)
    assert snapshot == {
        "mean_reward": 3.0,
        "quantile_reward": 1.8,
        "q_est": 2.5,
        "constraint_prob": 0.2,
    }


def test_terminal_progress_reads_common_non_qpo_wandb_schema():
    # Direct-chance algorithms now upload the same public keys as QPO. Their q_est/probability are
    # current-batch empirical values rather than QPO's persistent tracker state.
    metrics = {
        "disc_reward/aver_reward": 4.0,
        "disc_reward/quantile_reward": 2.0,
        "quantile/q_est": 2.0,
        "quantile/constraint_prob": 0.3,
        "constraint/empirical_failure_probability": 0.3,
    }
    snapshot = VecAgentBase._progress_snapshot(metrics)
    assert snapshot["q_est"] == 2.0
    assert snapshot["constraint_prob"] == 0.3
    assert metrics["quantile/q_est"] == metrics["disc_reward/quantile_reward"]


def test_terminal_progress_uses_latest_formulation_specific_lambda_without_alias():
    # QCPO uploads one probability-multiplier key after its dual step. The terminal presents that
    # exact scalar as ``lambda`` without adding a misleading generic key to the W&B dictionary.
    metrics = {
        "disc_reward/aver_reward": 4.0,
        "disc_reward/quantile_reward": 2.0,
        "constraint/empirical_failure_probability": 0.3,
        "lambda/probability_multiplier": 0.125,
    }
    snapshot = VecAgentBase._progress_snapshot(metrics)
    assert snapshot["lambda"] == 0.125
    agent = object.__new__(VecAgentBase)
    agent.num_iterations = 10
    agent.total_env_steps = 25
    assert "lambda:0.125" in agent._console_training_line(0, metrics)
    assert "lambda/value" not in metrics
    assert "lambda" not in metrics
    assert "lambda/residual" not in metrics
    assert "lambda/lr" not in metrics


def test_aggregate_monitor_separates_progress_from_train_and_validation_metrics():
    # One atomic child snapshot drives three independently testable physical-line renderings:
    # compact bar postfix, durable train line, and scheduled validation line.
    payload = {
        "algo": "QCPO", "env": "SimpleButton", "seed": 2,
        "phase": "training", "completed": 10, "total": 200,
        "metrics": {
            "mean_reward": 0.4, "quantile_reward": 0.1,
            "q_est": 0.1, "constraint_prob": 0.08, "lambda": 0.02,
        },
        "validation": {
            "num_episodes": 100, "mean_reward": 0.35,
            "quantile_reward": 0.05, "q_est": 0.05,
            "constraint_prob": 0.10,
        },
    }
    bar = _postfix(payload, discovered=3, expected=3)
    train = _training_line(payload)
    validation = _validation_line(payload)
    assert "mean" not in bar and "q_est" not in bar and "lambda" not in bar
    assert train.startswith("[train] QCPO/SimpleButton/s2 batch=010/200")
    assert "mean=0.400" in train and "lambda=0.020" in train
    assert validation.startswith("[validation] QCPO/SimpleButton/s2 batch=010/200")
    assert "N=100" in validation and "p=0.100" in validation


def test_checkpoint_selection_uses_direct_probability_then_mean():
    candidates = [
        {"empirical_prob": 0.2, "constraint_residual": 0.0, "mean": 10.0},
        {"empirical_prob": 0.1, "constraint_residual": -0.1, "mean": 8.0},
        {"empirical_prob": 0.3, "constraint_residual": 0.1, "mean": 20.0},
    ]
    assert select_validation_candidate(candidates, 0.2) is candidates[0]
    qpo = [{"quantile_return": 1.0, "mean": 100.0},
           {"quantile_return": 2.0, "mean": 0.0}]
    assert select_validation_candidate(qpo, 0.2, qpo=True) is qpo[1]


def test_probability_dual_direction_and_projection():
    agent = object.__new__(QCPOGPU)
    agent.return_threshold = 0.0
    agent.failure_alpha = 0.2
    agent.lambda_dual = torch.tensor(0.5)
    agent.lambda_min, agent.lambda_max = 0.0, 10.0
    agent.lambda_a, agent.lambda_b, agent.lambda_c = 1.0, 1.0, 0.0
    agent.dual_update_count = 0
    agent.last_dual_prob = 0.0
    # 3/4 failures > .2: lambda increases.
    before = float(agent.lambda_dual)
    probability, residual, _ = agent.update_dual(torch.tensor([-1.0, 0.0, 1.0, -2.0]))
    assert probability == 0.75 and residual > 0
    assert float(agent.lambda_dual) > before
    # Feasible repeated updates decrease and project at zero.
    agent.lambda_dual.fill_(0.01)
    agent.update_dual(torch.ones(4))
    assert float(agent.lambda_dual) == 0.0


def test_combined_ppo_clip_matches_manual_values():
    new = torch.log(torch.tensor([[1.3, 0.8]]))
    old = torch.zeros_like(new)
    advantage = torch.tensor([[2.0, -3.0]])
    valid = torch.ones_like(new, dtype=torch.bool)
    loss, ratio, clip_fraction = VecAgentBase.ppo_combined_loss(
        new, old, advantage, valid, clip_ratio=0.1)
    # Positive A clips upper ratio to 1.1; negative A chooses min(.8*A,.9*A)=.9*A.
    manual = -((1.1 * 2.0) + (0.9 * -3.0)) / 2.0
    assert math.isclose(float(loss), manual, rel_tol=1e-6, abs_tol=1e-6)
    assert torch.allclose(ratio, torch.tensor([[1.3, 0.8]]))
    assert float(clip_fraction) == 1.0


def test_qcpo_ref_discounted_ppo_matches_manual_clip_mask_and_fixed_bh():
    """Positive/negative clipping and padding use gamma^t while reduction stays full B*H."""
    new = torch.log(torch.tensor([[1.3, 0.8], [1.0, 50.0]]))
    old = torch.zeros_like(new)
    advantage = torch.tensor([[2.0, -3.0], [4.0, 1000.0]])
    valid = torch.tensor([[True, True], [True, False]])
    loss, ratio, clip_fraction = QCPORefGPU.discounted_ppo_loss(
        new, old, advantage, valid, clip_ratio=0.1,
        gamma=0.5, fixed_denominator=4)

    # t=0 uses +A upper clipping and -A lower clipping; t=1 contributes .5*4. Padding is zero,
    # while the denominator remains the physical B*H=4 rather than the three valid decisions.
    expected = -((1.1 * 2.0) + (0.9 * -3.0) + (0.5 * 4.0)) / 4.0
    assert math.isclose(float(loss), expected, rel_tol=1e-6, abs_tol=1e-6)
    assert torch.allclose(ratio, torch.tensor([[1.3, 0.8], [1.0, 50.0]]))
    assert math.isclose(float(clip_fraction), 2.0 / 3.0, abs_tol=1e-7)


def test_qcpo_ref_early_late_actor_gradient_ratio_is_exact_gamma_power():
    """With identical valid rows, the t=0 versus t=3 score gradient is gamma**(-3)."""
    gamma = 0.8
    new = torch.full((4, 1), 0.01, requires_grad=True)
    old = torch.zeros_like(new)
    advantage = torch.ones_like(new)
    valid = torch.ones_like(new, dtype=torch.bool)
    loss, _ratio, _clip_fraction = QCPORefGPU.discounted_ppo_loss(
        new, old, advantage, valid, clip_ratio=0.1,
        gamma=gamma, fixed_denominator=4)
    loss.backward()
    gradient_ratio = float(new.grad[0, 0] / new.grad[3, 0])
    assert math.isclose(
        gradient_ratio, gamma ** -3, rel_tol=1e-6, abs_tol=1e-6)


def test_qcpo_ref_normalizes_complete_advantage_before_occupancy_weighting():
    """Reward and lambda-risk terms share one valid-row population scale before gamma^t."""
    agent = object.__new__(QCPORefGPU)
    agent.lambda_quantile = torch.tensor(2.0)
    agent.normalize_combined_advantage = True
    processed = {
        "reward_advantage": torch.tensor([[4.0, 8.0], [10.0, 99.0]]),
        "quantile_advantage": torch.tensor([[1.0, -1.0], [2.0, -99.0]]),
        "valid": torch.tensor([[1.0, 1.0], [1.0, 0.0]]),
    }
    raw = torch.tensor([[2.0, 10.0], [6.0, 297.0]])
    expected = raw / raw[processed["valid"].bool()].std(unbiased=False)
    assert torch.allclose(agent._combined_actor_advantage(processed), expected)


def test_qcpo_ref_gamma_one_all_valid_reduces_to_old_valid_mean():
    """At gamma=1 without padding, fixed B*H equals the historical valid-row mean."""
    new = torch.log(torch.tensor([[1.05, 0.95], [1.02, 0.98]]))
    old = torch.zeros_like(new)
    advantage = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
    valid = torch.ones_like(new, dtype=torch.bool)
    old_loss, old_ratio, old_clip = VecAgentBase.ppo_combined_loss(
        new, old, advantage, valid, clip_ratio=0.1)
    new_loss, new_ratio, new_clip = QCPORefGPU.discounted_ppo_loss(
        new, old, advantage, valid, clip_ratio=0.1,
        gamma=1.0, fixed_denominator=4)
    assert torch.equal(new_ratio, old_ratio)
    assert torch.equal(new_clip, old_clip)
    assert torch.allclose(new_loss, old_loss)


def test_cost_is_not_an_input_to_qcpo_trajectory_terms():
    agent = object.__new__(QCPOGPU)
    agent.return_threshold = 0.0
    agent.lambda_dual = torch.tensor(2.0)
    agent.normalize_combined_advantage = False
    returns = torch.tensor([-1.0, 2.0, 3.0])
    first = agent.trajectory_terms(returns)
    # Leave-one-out baselines exclude each trajectory from its own baseline. For returns
    # [-1,2,3], the corresponding advantages are [-3.5,1.0,2.5]. Inclusive failures are
    # [1,0,0], hence their leave-one-out advantages are [1,-0.5,-0.5].
    assert torch.allclose(first[0], torch.tensor([-3.5, 1.0, 2.5]))
    assert torch.allclose(first[1], torch.tensor([1.0, -0.5, -0.5]))
    # No cost argument exists; arbitrary diagnostic cost changes cannot affect returned tensors.
    arbitrary_cost_a = torch.tensor([0.0, 1000.0, -5.0])
    arbitrary_cost_b = torch.tensor([999.0, -1.0, 42.0])
    assert not torch.equal(arbitrary_cost_a, arbitrary_cost_b)
    second = agent.trajectory_terms(returns)
    for left, right in zip(first, second):
        assert torch.equal(left, right)


def test_qcpo_diagnostic_gradient_decomposition_preserves_actor_update():
    # A one-parameter score model makes the expected combined gradient independently calculable.
    # This locks the diagnostic decomposition without constructing a Safety Gym environment.
    agent = object.__new__(QCPOGPU)
    agent.return_threshold = 0.0
    agent.lambda_dual = torch.tensor(0.5)
    agent.normalize_combined_advantage = True
    agent.actor_grad_clip = 0.05
    agent.actor = torch.nn.Linear(1, 1, bias=False)
    agent.actor.weight.data.fill_(0.2)
    agent.optimizer = torch.optim.SGD(agent.actor.parameters(), lr=0.1)
    agent._compute_log_probs = lambda states, _actions: agent.actor(states).squeeze(-1)

    states = torch.tensor([[[-1.0], [0.5], [2.0]]])
    returns = torch.tensor([-1.0, 2.0, 3.0])
    rollout = {
        "S": states,
        "A": torch.zeros_like(states),
        "valid": torch.ones(1, 3, dtype=torch.bool),
        "disc_return": returns,
    }

    # Compute the original combined-loss gradient on an independent scalar parameter. Clipping a
    # one-dimensional gradient to .05 then SGD(lr=.1) must move the weight by exactly .005.
    mean_adv, failure_adv, combined, _failures = agent.trajectory_terms(returns)
    reference_weight = torch.tensor(0.2, requires_grad=True)
    reference_logp = states.squeeze(-1) * reference_weight
    reference_loss = -(reference_logp * combined.unsqueeze(0)).mean()
    reference_gradient = torch.autograd.grad(reference_loss, reference_weight)[0]
    expected_clipped_gradient = reference_gradient.clamp(-0.05, 0.05)
    expected_weight = reference_weight.detach() - 0.1 * expected_clipped_gradient

    rng_before = torch.get_rng_state().clone()
    info = agent.update_actor(rollout)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert torch.allclose(agent.actor.weight.detach().reshape(()), expected_weight, atol=1e-7)
    assert info["diagnostic/qcpo_mean_score_grad_norm"] >= 0.0
    assert info["diagnostic/qcpo_constraint_score_grad_norm"] >= 0.0
    assert info["diagnostic/qcpo_combined_grad_norm_pre_clip"] >= 0.05
    assert info["diagnostic/qcpo_combined_grad_norm_post_clip"] <= 0.0500001
    expected_ratio = (
        0.5 * failure_adv.std(unbiased=False)
        / mean_adv.std(unbiased=False).clamp(min=1e-8)
    )
    assert math.isclose(
        info["diagnostic/qcpo_risk_to_mean"], float(expected_ratio), rel_tol=1e-6)


def pytest_approx(value, tol=1e-7):
    """Tiny local approximation helper keeps this file runnable without pytest namespace tricks."""
    class Approx:
        def __eq__(self, other):
            return math.isclose(float(other), float(value), rel_tol=tol, abs_tol=tol)
    return Approx()


def test_negated_return_quantile_and_actor_sign_are_exact_mirrors():
    """Linear quantiles satisfy Q_.8(-G)=-Q_.2(G), and both actor spellings agree."""
    returns = np.asarray([-3.0, -1.0, 0.5, 4.0, 8.0], dtype=np.float64)
    reward_q20 = linear_empirical_quantile(returns, 0.2)
    cost_q80 = linear_empirical_quantile(-returns, 0.8)
    assert math.isclose(cost_q80, -reward_q20, abs_tol=1e-12)

    reward_advantage = torch.tensor([2.0, -1.0])
    cost_quantile_advantage = torch.tensor([0.3, -0.7])
    multiplier = 1.4
    paper_side = reward_advantage - multiplier * cost_quantile_advantage
    reward_side = reward_advantage + multiplier * (-cost_quantile_advantage)
    assert torch.equal(paper_side, reward_side)


def test_quantile_dual_uses_mirrored_linear_percentile_and_correct_direction():
    agent = object.__new__(QCPORefGPU)
    agent.failure_alpha, agent.return_threshold = 0.2, 0.0
    agent.cost_quantile_level, agent.cost_threshold = 0.8, 0.0
    agent.dual_window_size = 5
    agent.dual_cost_returns = deque(maxlen=agent.dual_window_size)
    agent.lambda_quantile = torch.tensor(0.5)
    agent.lambda_max = 10.0
    agent.lambda_a, agent.lambda_b, agent.lambda_c = 1.0, 1.0, 0.0
    agent.dual_update_count = 0
    agent.last_dual_quantile = 0.0
    agent.last_dual_cost_quantile = 0.0
    before = float(agent.lambda_quantile)
    empirical, residual, _ = agent.update_dual(torch.tensor([-2.0, -1.0, 2.0, 3.0, 4.0]))
    assert math.isclose(empirical, -1.2, abs_tol=1e-7)
    assert math.isclose(residual, 1.2, abs_tol=1e-7)
    assert float(agent.lambda_quantile) > before
    agent.return_threshold = -10.0
    agent.cost_threshold = 10.0
    agent.lambda_quantile.fill_(0.01)
    agent.update_dual(torch.tensor([-2.0, -1.0, 2.0, 3.0, 4.0]))
    assert float(agent.lambda_quantile) == 0.0


def test_quantile_dual_waits_for_full_paper_sized_trajectory_window():
    """B=25 needs four batches before a 100-trajectory empirical dual update is allowed."""
    agent = object.__new__(QCPORefGPU)
    agent.failure_alpha, agent.return_threshold = 0.2, 0.0
    agent.cost_quantile_level, agent.cost_threshold = 0.8, 0.0
    agent.dual_window_size = 100
    agent.dual_cost_returns = deque(maxlen=agent.dual_window_size)
    agent.lambda_quantile = torch.tensor(0.5)
    agent.lambda_max = 10.0
    agent.lambda_a, agent.lambda_b, agent.lambda_c = 1.0, 1.0, 0.0
    agent.dual_update_count = 0
    agent.last_dual_quantile = 0.0
    agent.last_dual_cost_quantile = 0.0
    returns = torch.linspace(-2.0, 2.0, 25)

    for _ in range(3):
        before = float(agent.lambda_quantile)
        _quantile, _residual, step = agent.update_dual(returns)
        assert step == 0.0
        assert float(agent.lambda_quantile) == before
        assert agent.dual_update_count == 0
    _quantile, _residual, step = agent.update_dual(returns)
    assert step == 1.0
    assert len(agent.dual_cost_returns) == 100
    assert agent.dual_update_count == 1
