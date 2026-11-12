# -*- coding: utf-8 -*-
"""Analytical tests for the independent scalar-value DQC GAE-Mean identity."""
import copy

import torch

from agents.dqc_ac_beta_gpu import DQCACBetaGPU, StateValueCritic
from run_experiment import base_args, checkpoint_signature, configure_dqc_identity
from test_dqc_online import _assert_nested_equal, _checkpoint_agent
from utils.checkpoint import load_checkpoint, save_checkpoint


def test_finite_gae_handles_horizon_termination_and_padding_exactly():
    """
    Check finite GAE against hand recursion for two differently ended vector slots.

    Slot 0 reaches a formal truncation at t=3. Slot 1 terminates at t=1 and has deliberately
    arbitrary rewards/values in invalid padding; those rows must remain exactly zero.
    """
    rewards = torch.tensor([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 999.0],
        [4.0, 999.0],
    ])
    values = torch.tensor([
        [0.5, 3.0],
        [1.0, 4.0],
        [1.5, 777.0],
        [2.0, 777.0],
    ])
    valid = torch.tensor([
        [True, True],
        [True, True],
        [True, False],
        [True, False],
    ])
    terminated = torch.zeros_like(valid)
    terminated[1, 1] = True
    truncated = torch.zeros_like(valid)
    truncated[3, 0] = True

    advantages, targets, continuation = DQCACBetaGPU.finite_gae(
        rewards, values, valid, terminated, truncated,
        gamma=0.9, gae_lambda=0.8,
    )

    # For slot 0, delta=[1.4,2.35,3.3,2] and gamma*lambda=.72.
    expected_slot0 = torch.tensor([
        1.4 + 0.72 * (2.35 + 0.72 * (3.3 + 0.72 * 2.0)),
        2.35 + 0.72 * (3.3 + 0.72 * 2.0),
        3.3 + 0.72 * 2.0,
        2.0,
    ])
    # For slot 1, terminal t=1 has delta=20-4=16 and no V_2 bootstrap.
    expected_slot1 = torch.tensor([
        (10.0 + 0.9 * 4.0 - 3.0) + 0.72 * 16.0,
        16.0,
        0.0,
        0.0,
    ])
    expected_advantages = torch.stack(
        (expected_slot0, expected_slot1), dim=1)
    expected_targets = torch.where(
        valid, expected_advantages + values, torch.zeros_like(values))

    assert torch.allclose(advantages, expected_advantages, atol=1e-6)
    assert torch.allclose(targets, expected_targets, atol=1e-6)
    assert torch.equal(
        continuation,
        torch.tensor([
            [True, True],
            [True, False],
            [True, False],
            [False, False],
        ]),
    )


class _SeparatedRiskCritic:
    """
    Return very different atom means while preserving a transparent fixed-q CDF.

    Negative executed actions have two negative atoms and CDF one at q=0. Positive baseline
    actions have positive atoms and CDF zero. The huge means make accidental atom-mean reuse easy
    to detect in the GAE actor term.
    """

    def __call__(self, _features, actions):
        negative = actions[..., 0] < 0.0
        low = torch.where(negative, -200.0, 100.0)
        high = torch.where(negative, -100.0, 200.0)
        return torch.stack((low, high), dim=-1)


class _PositiveBaselineActor:
    """Produce effectively deterministic positive Gaussian baseline samples."""

    def __call__(self, states):
        shape = (*states.shape[:-1], 1)
        mean = torch.ones(shape, dtype=states.dtype)
        log_std = torch.full(shape, -30.0, dtype=states.dtype)
        return mean, log_std, None


def _gae_actor_agent():
    """Build only fields used by actor_advantages under the finite GAE identity."""
    agent = object.__new__(DQCACBetaGPU)
    agent.device = torch.device("cpu")
    agent.dqc_update_mode = agent.FINITE_UPDATE_MODE
    agent.dqc_mean_estimator = agent.GAE_MEAN
    agent.gamma = 0.5
    agent.beta = 0.01
    agent.horizon = 1000
    agent.lstm_size = 1
    agent.return_threshold = 0.0
    agent.lambda_probability = torch.tensor(2.0)
    agent.num_action_samples = 2
    agent._action_low = torch.tensor([-1.0])
    agent._action_high = torch.tensor([1.0])
    agent.reward_critic = _SeparatedRiskCritic()
    agent.actor = _PositiveBaselineActor()
    return agent


def test_gae_actor_uses_one_outer_gamma_no_risk_discount_and_positive_scale_only():
    """
    Verify W_t=gamma^t A_GAE-lambda A_CDF before population-std division.

    This simultaneously proves that distributional atom means do not enter the new actor mean
    term, finite CDF risk receives neither gamma nor beta, and normalization never centers signs.
    """
    agent = _gae_actor_agent()
    T = 3
    rollout = {
        "S": torch.zeros(T, 1, 1),
        "A_exec": -torch.ones(T, 1, 1),
        "valid": torch.ones(T, 1, dtype=torch.bool),
        "h0": torch.zeros(T, 1, 1),
        "c0": torch.zeros(T, 1, 1),
        "reward_prefix": torch.zeros(T, 1, dtype=torch.float64),
    }
    gae_batch = {
        "advantages": torch.tensor([[2.0], [3.0], [4.0]]),
        "values": torch.tensor([[7.0], [8.0], [9.0]]),
    }
    terms = agent.actor_advantages(rollout, gae_batch)

    expected_mean = torch.tensor([[2.0], [1.5], [1.0]])
    expected_risk = torch.full((T, 1), 2.0)
    expected_raw = expected_mean - expected_risk
    expected_scale = expected_raw.std(unbiased=False)

    assert torch.equal(terms["mean_advantage"], gae_batch["advantages"])
    assert torch.equal(terms["mean_term"], expected_mean)
    assert torch.equal(terms["cdf_advantage"], torch.ones(T, 1))
    assert torch.equal(terms["risk_term"], expected_risk)
    assert torch.equal(terms["combined_raw"], expected_raw)
    assert torch.equal(terms["combined_normalization_scale"], expected_scale)
    assert torch.allclose(terms["combined"], expected_raw / expected_scale)
    assert torch.equal(
        torch.sign(terms["combined"]), torch.sign(expected_raw))


def test_value_regression_reuses_one_frozen_target_for_all_eight_epochs():
    """The E8 scalar fit must never recompute lambda-return labels after an optimizer step."""
    torch.manual_seed(17)
    agent = object.__new__(DQCACBetaGPU)
    agent.value_critic = StateValueCritic(2, [4, 4])
    agent.value_optimizer = torch.optim.Adam(
        agent.value_critic.parameters(), lr=1e-2)
    agent.value_optimizer_steps = 0
    agent.critic_minibatch_size = 2
    agent.critic_grad_clip = 10.0

    valid = torch.tensor([
        [True, True],
        [True, True],
        [True, False],
    ])
    original_targets = torch.tensor([
        [1.0, -1.0],
        [2.0, -2.0],
        [3.0, 99.0],
    ])
    gae_batch = {
        "features": torch.arange(12, dtype=torch.float32).reshape(3, 2, 2),
        "targets": original_targets.clone(),
        "valid": valid,
    }
    observed_targets = []
    target_storage = []

    def observe_epoch(_epoch, targets):
        """Record values and storage identity without mutating production tensors."""
        observed_targets.append(targets.detach().clone())
        target_storage.append(targets.data_ptr())

    agent._value_epoch_hook = observe_epoch
    info = agent.update_value_critic(gae_batch, regression_epochs=8)

    assert len(observed_targets) == 8
    assert len(set(target_storage)) == 1
    assert all(torch.equal(value, original_targets) for value in observed_targets)
    assert torch.equal(gae_batch["targets"], original_targets)
    # Five valid rows form ceil(5/2)=3 minibatches per epoch.
    assert agent.value_optimizer_steps == 8 * 3
    assert info["diagnostic/dqc/value_regression_epochs"] == 8.0
    assert info["diagnostic/dqc/value_optimizer_steps"] == 24.0


def _gae_checkpoint_agent():
    """Adapt the existing complete checkpoint fixture to the strict finite GAE identity."""
    agent = _checkpoint_agent(
        target_source="n_step",
        warmup_mode="fitted_hard",
        warmup_epochs=4,
        critic_epochs=8,
        actor_epochs=8,
        actor_reuse_mode="ppo_clip",
        ratio_clip=0.1,
        target_kl=0.01,
    )
    agent.dqc_update_mode = agent.FINITE_UPDATE_MODE
    agent.horizon = 1000
    agent.horizon_mode = "finite"
    agent.critic_n_step = 100
    agent.critic_time_feature = "remaining_fraction"
    agent.critic_actor_context = True
    agent.target_update_mode = "hard"
    agent.beta = 1.0
    agent.warmup_iters = 10
    agent.initialization_cursor = 10

    # Step clocks remain internally consistent while explicitly separating 10 warmup batches.
    agent.main_env_steps = 123
    agent.initialization_env_steps = 250000
    agent.actual_env_steps = 250123
    agent.total_env_steps = 250123

    agent.dqc_mean_estimator = agent.GAE_MEAN
    agent.dqc_gae_lambda = 0.97
    agent.dqc_value_hidden = [512, 512]
    agent.dqc_value_lr = 3e-4
    agent.dqc_value_epochs = 8
    agent.normalize_combined_advantage = True
    agent.value_critic = StateValueCritic(1, agent.dqc_value_hidden)
    agent.value_optimizer = torch.optim.Adam(
        agent.value_critic.parameters(), lr=agent.dqc_value_lr)
    agent.value_optimizer_steps = 7

    # Create real Adam first/second moments so round-trip checks cover more than empty metadata.
    optimizer_probe = sum(
        parameter.square().sum() for parameter in agent.value_critic.parameters())
    agent.value_optimizer.zero_grad(set_to_none=True)
    optimizer_probe.backward()
    agent.value_optimizer.step()
    agent.value_optimizer.zero_grad(set_to_none=True)

    # The generic payload signature and agent-level validator independently carry the identity.
    agent.checkpoint_signature = {
        "algorithm": "DQCACBeta",
        "training_budget_semantics": agent.training_budget_semantics,
        "dqc_update_mode": agent.dqc_update_mode,
        "dqc_mean_estimator": agent.dqc_mean_estimator,
        "dqc_gae_lambda": agent.dqc_gae_lambda,
        "dqc_value_hidden": agent.dqc_value_hidden,
        "dqc_value_lr": agent.dqc_value_lr,
        "dqc_value_epochs": agent.dqc_value_epochs,
        "normalize_combined_advantage": True,
    }
    return agent


def test_gae_checkpoint_restores_value_state_and_rejects_old_mean_before_actor_mutation(
        tmp_path):
    """Round-trip scalar state; reject same-finite-critic atom-mean lineage before loading actor."""
    torch.manual_seed(23)
    source = _gae_checkpoint_agent()
    checkpoint = tmp_path / "gae.pt"
    save_checkpoint(checkpoint, source, iteration=9)

    restored = _gae_checkpoint_agent()
    load_checkpoint(checkpoint, restored, restore_rng=False)
    for name, tensor in source.value_critic.state_dict().items():
        assert torch.equal(restored.value_critic.state_dict()[name], tensor)
    assert restored.value_optimizer_steps == source.value_optimizer_steps
    _assert_nested_equal(
        restored.value_optimizer.state_dict(),
        source.value_optimizer.state_dict())

    # Keep every finite QR/PPO field identical and change only the old mean estimator identity.
    old_mean = _gae_checkpoint_agent()
    old_mean.dqc_mean_estimator = old_mean.DISTRIBUTIONAL_MEAN
    old_mean_path = tmp_path / "old_mean.pt"
    save_checkpoint(old_mean_path, old_mean, iteration=9)

    requested = _gae_checkpoint_agent()
    actor_before = copy.deepcopy(requested.actor.state_dict())
    try:
        load_checkpoint(
            old_mean_path, requested, restore_rng=False,
            allow_signature_mismatch=True)
    except ValueError as error:
        assert "dqc_mean_estimator mismatch" in str(error)
    else:
        raise AssertionError("old distributional mean identity must not resume as GAE-Mean")
    for name, tensor in actor_before.items():
        assert torch.equal(requested.actor.state_dict()[name], tensor)


def test_gae_preset_and_checkpoint_signature_fix_all_v1_fields():
    """One explicit mean selector expands to the pre-registered E8/PPO/value configuration."""
    args = base_args("DQCACBeta", 0, torch.device("cpu"), "SimpleButton")
    args.dqc_update_mode = "finite_nstep_v1"
    args.dqc_mean_estimator = "state_value_gae_v1"
    configure_dqc_identity(args, {
        "dqc_update_mode": "finite_nstep_v1",
        "dqc_mean_estimator": "state_value_gae_v1",
    })
    signature = checkpoint_signature(args, "DQCACBeta")

    expected = {
        "dqc_mean_estimator": "state_value_gae_v1",
        "dqc_gae_lambda": 0.97,
        "dqc_value_hidden": [512, 512],
        "dqc_value_lr": 3e-4,
        "dqc_value_epochs": 8,
        "normalize_combined_advantage": True,
        "critic_epochs": 8,
        "actor_epochs": 8,
        "dqc_actor_reuse_mode": "ppo_clip",
        "dqc_ratio_clip": 0.1,
        "dqc_target_kl": 0.01,
    }
    for name, value in expected.items():
        assert getattr(args, name) == value
        assert signature[name] == value
    assert args.wandb_group == "dqc_gae_mean_simplebutton_v1"
    assert args.experiment_variant == "n100_gae_mean_s0"
