# -*- coding: utf-8 -*-
"""Focused mathematical, boundary, optimizer, and checkpoint tests for online_1step_v1."""
import copy
import math
import warnings

import torch

from agents.common import DistributionalCritic
from agents.dqc_ac_beta_gpu import DQCACBetaGPU
from agents.vec_base import VecAgentBase
from envs.safety_env_vec import VecStepResult
from run_experiment import base_args, checkpoint_signature
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.model import RecurrentGaussianActor


class _BoundaryVector:
    """Slot 0 terminates at t=0; slot 1 reaches a true truncation at t=1."""

    B, n = 2, 2

    def __init__(self):
        self.calls = 0

    def reset(self):
        self.calls = 0
        return torch.zeros(2, 1)

    def step(self, _raw_actions):
        self.calls += 1
        if self.calls == 1:
            return VecStepResult(
                observation=torch.tensor([[100.0], [1.0]]),
                reward=torch.tensor([1.0, 2.0]),
                cost=torch.tensor([7.0, 8.0]),
                terminated=torch.tensor([True, False]),
                truncated=torch.tensor([False, False]),
                final_observation=torch.tensor([[10.0], [1.0]]),
                final_observation_mask=torch.tensor([True, False]),
            )
        return VecStepResult(
            observation=torch.tensor([[101.0], [200.0]]),
            reward=torch.tensor([999.0, 3.0]),
            cost=torch.tensor([999.0, 9.0]),
            terminated=torch.tensor([False, False]),
            truncated=torch.tensor([False, True]),
            final_observation=torch.tensor([[11.0], [20.0]]),
            final_observation_mask=torch.tensor([False, True]),
        )


def _collector_fixture():
    """Construct only VecAgentBase fields used by the shared collector."""
    agent = object.__new__(VecAgentBase)
    agent.num_envs = agent.n = 2
    agent.device = torch.device("cpu")
    agent.gamma = 0.9
    agent.actor = RecurrentGaussianActor(
        1, 1, hidden=(4, 4), lstm_size=3,
        init_std=0.1, normalize_observation=False)
    agent.vec_env = _BoundaryVector()
    agent._log_2pi = math.log(2.0 * math.pi)
    agent._action_low = torch.tensor([-1.0])
    agent._action_high = torch.tensor([1.0])
    agent.main_env_steps = 0
    agent.initialization_env_steps = 0
    agent.actual_env_steps = 0
    agent.total_env_steps = 0
    return agent


def test_collector_separates_raw_and_executed_actions_and_preserves_boundary_context():
    """A_raw is the likelihood variable; A_exec/final context are critic-side data."""
    agent = _collector_fixture()

    def fixed_sample(states, rnn_state=None, return_state=False, deterministic=False):
        del deterministic
        raw = torch.tensor([[2.0], [-3.0]], dtype=states.dtype)
        mean = torch.zeros_like(raw)
        log_std = torch.zeros_like(raw)
        next_state = (rnn_state[0] + 1.0, rnn_state[1] + 2.0)
        return (raw, next_state, mean, log_std) if return_state else raw

    agent._sample_actions = fixed_sample
    rollout = agent._rollout_core()

    assert rollout["A"] is rollout["A_raw"]
    assert "A_latent" not in rollout
    assert torch.equal(rollout["A_raw"][0, :, 0], torch.tensor([2.0, -3.0]))
    assert torch.equal(rollout["A_exec"][0, :, 0], torch.tensor([1.0, -1.0]))
    assert bool(torch.isfinite(rollout["logp"]).all())

    # The terminated slot never splices reset transitions. Truncation keeps final=20 rather than
    # reset=200, plus the old episode state captured before recurrent zeroing.
    assert rollout["valid"][:, 0].tolist() == [True, False]
    assert float(rollout["next_states"][1, 1, 0]) == 20.0
    assert float(rollout["final_observation"][1, 1, 0]) == 20.0
    assert bool(rollout["final_observation_mask"][1, 1])
    assert torch.equal(rollout["next_h"][1, 1], torch.full((3,), 2.0))
    assert torch.equal(rollout["next_c"][1, 1], torch.full((3,), 4.0))
    assert float(rollout["bootstrap_mask"][0, 0]) == 0.0
    assert float(rollout["bootstrap_mask"][1, 1]) == 1.0


def test_reward_budget_hand_recurrence_and_beta_independence():
    """Changing beta cannot enter b[t+1]=(b[t]-r[t])/gamma."""
    first = object.__new__(DQCACBetaGPU)
    first.return_threshold, first.gamma, first.beta = 5.0, 0.5, 0.1
    first.device = torch.device("cpu")
    second = object.__new__(DQCACBetaGPU)
    second.return_threshold, second.gamma, second.beta = 5.0, 0.5, 0.999
    second.device = torch.device("cpu")
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    left = first._attach_reward_budgets({"R": rewards.clone()})["budgets"][:, 0]
    right = second._attach_reward_budgets({"R": rewards.clone()})["budgets"][:, 0]
    assert torch.equal(left, torch.tensor([5.0, 8.0, 12.0]))
    assert torch.equal(left, right)


class _RecordingTarget:
    """Return transparent atoms z=[next_state+action, next_state+action+2]."""

    def __init__(self):
        self.states = None
        self.actions = None

    def __call__(self, states, actions):
        self.states = states.detach().clone()
        self.actions = actions.detach().clone()
        base = states[:, 0] + actions[:, 0]
        return torch.stack((base, base + 2.0), dim=1)


def _boundary_target_rollout():
    """Three valid rows: ordinary t0/b0, truncated t0/b1, terminated t1/b0."""
    return {
        "S": torch.tensor([[[0.0], [10.0]], [[1.0], [999.0]]]),
        "A_raw": torch.tensor([[[0.1], [2.0]], [[0.8], [0.0]]]),
        "A_exec": torch.tensor([[[0.1], [1.0]], [[0.7], [0.0]]]),
        "R": torch.tensor([[1.0, 2.0], [3.0, 0.0]]),
        "next_states": torch.tensor([[[1.0], [20.0]], [[30.0], [999.0]]]),
        "final_observation": torch.tensor([[[0.0], [20.0]], [[30.0], [0.0]]]),
        "final_observation_mask": torch.tensor([[False, True], [True, False]]),
        "terminated": torch.tensor([[False, False], [True, False]]),
        "truncated": torch.tensor([[False, True], [False, False]]),
        "valid": torch.tensor([[True, True], [True, False]]),
        "next_h": torch.tensor([[[11.0], [22.0]], [[33.0], [0.0]]]),
        "next_c": torch.tensor([[[111.0], [222.0]], [[333.0], [0.0]]]),
        "budgets": torch.zeros(2, 2),
    }


def test_one_step_targets_distinguish_ordinary_terminated_and_truncated_history():
    """Truncation samples at final observation/old h,c; ordinary target reuses next A_exec."""
    agent = object.__new__(DQCACBetaGPU)
    agent.device = torch.device("cpu")
    agent.gamma = 0.5
    agent._action_low = torch.tensor([-1.0])
    agent._action_high = torch.tensor([1.0])
    agent.reward_target_critic = _RecordingTarget()
    recorded = {}

    def sample(states, rnn_state=None, return_state=False, deterministic=False):
        del deterministic
        recorded["states"] = states.detach().clone()
        recorded["h"] = rnn_state[0].detach().clone()
        recorded["c"] = rnn_state[1].detach().clone()
        raw = torch.full((states.shape[0], 1), 4.0)
        return (raw, rnn_state, raw, torch.zeros_like(raw)) if return_state else raw

    agent._sample_actions = sample
    batch = agent.build_one_step_batch(_boundary_target_rollout())

    # Only the truncated row triggers an actor draw: reset observation 200/999 is never supplied.
    assert torch.equal(recorded["states"], torch.tensor([[20.0]]))
    assert torch.equal(recorded["h"], torch.tensor([[[22.0]]]))
    assert torch.equal(recorded["c"], torch.tensor([[[222.0]]]))
    # Target critic row order is flattened [t0b0,t0b1,t1b0]. Ordinary uses next rollout .7;
    # truncated uses clipped sampled action 1; terminated action remains irrelevant zero.
    assert torch.allclose(agent.reward_target_critic.actions[:, 0], torch.tensor([0.7, 1.0, 0.0]))
    expected = torch.tensor([
        [1.0 + 0.5 * 1.7, 1.0 + 0.5 * 3.7],
        [2.0 + 0.5 * 21.0, 2.0 + 0.5 * 23.0],
        [3.0, 3.0],
    ])
    assert torch.allclose(batch["targets"], expected)
    assert not batch["targets"].requires_grad


def test_two_step_deterministic_target_and_fixed_head_qr_loss_match_manual_values():
    """One-step target and pairwise QR Huber arithmetic remain independently transparent."""
    agent = object.__new__(DQCACBetaGPU)
    agent.device = torch.device("cpu")
    agent.gamma = 0.5
    agent.huber_kappa = 1.0
    agent.taus = torch.tensor([0.25, 0.75])
    agent._action_low = torch.tensor([-1.0])
    agent._action_high = torch.tensor([1.0])
    agent.reward_target_critic = _RecordingTarget()
    def fixed_target_action(states, rnn_state=None, return_state=False, deterministic=False):
        del deterministic
        raw = torch.full((states.shape[0], 1), 4.0)
        result = (raw, rnn_state, raw, torch.zeros_like(raw))
        return result if return_state else raw

    agent._sample_actions = fixed_target_action
    rollout = _boundary_target_rollout()
    target = agent.build_one_step_batch(rollout)["targets"]
    predicted = torch.tensor([[1.0, 2.0]])
    selected_target = target[:1]
    actual = agent.quantile_huber_loss(predicted, selected_target)

    error = selected_target.unsqueeze(1) - predicted.unsqueeze(2)
    absolute = error.abs()
    huber = torch.where(absolute <= 1.0, 0.5 * error.square(), absolute - 0.5)
    weight = torch.abs(agent.taus.view(1, 2, 1) - (error < 0).float())
    manual = (weight * huber).mean()
    assert torch.allclose(actual, manual)


def _optimizer_agent(num_quantiles=2):
    """Small real-network fixture for one-pass critic/checkpoint determinism tests."""
    agent = object.__new__(DQCACBetaGPU)
    agent.device = torch.device("cpu")
    agent.gamma = 0.9
    agent.huber_kappa = 0.1
    agent.taus = (
        torch.arange(num_quantiles, dtype=torch.float32) + 0.5
    ) / float(num_quantiles)
    agent.critic_minibatch_size = 2
    agent.critic_grad_clip = 10.0
    agent.target_tau = 0.2
    agent.critic_target_source = "one_step"
    agent.critic_warmup_mode = "legacy"
    agent.warmup_critic_epochs = 1
    agent.reward_critic = DistributionalCritic(1, 1, num_quantiles, [4])
    agent.reward_target_critic = DistributionalCritic(1, 1, num_quantiles, [4])
    agent.reward_target_critic.load_state_dict(agent.reward_critic.state_dict())
    agent.reward_target_critic.requires_grad_(False)
    agent.critic_optimizer = torch.optim.Adam(agent.reward_critic.parameters(), lr=1e-3)
    agent.critic_optimizer_steps = 0
    agent.target_update_count = 0
    agent._action_low = torch.tensor([-1.0])
    agent._action_high = torch.tensor([1.0])
    return agent


def _five_transition_rollout(cost_value=0.0):
    """Lengths 3 and 2 exercise early termination plus a final true termination."""
    valid = torch.tensor([[True, True], [True, True], [True, False]])
    terminated = torch.tensor([[False, False], [False, True], [True, False]])
    states = torch.tensor([[[0.0], [10.0]], [[1.0], [11.0]], [[2.0], [0.0]]])
    actions = torch.tensor([[[0.1], [0.2]], [[0.3], [0.4]], [[0.5], [0.0]]])
    rewards = torch.tensor([[1.0, 2.0], [1.5, 2.5], [3.0, 0.0]])
    return {
        "S": states,
        "A_raw": actions.clone(),
        "A_exec": actions.clone(),
        "R": rewards,
        "C": torch.full_like(rewards, float(cost_value)),
        "next_states": states + 1.0,
        "final_observation": states + 1.0,
        "final_observation_mask": terminated.clone(),
        "terminated": terminated,
        "truncated": torch.zeros_like(terminated),
        "valid": valid,
        "next_h": torch.zeros(3, 2, 1),
        "next_c": torch.zeros(3, 2, 1),
        "budgets": torch.zeros(3, 2),
    }


def test_finite_suffix_returns_reset_at_boundaries_and_match_manual_values():
    """Invalid post-boundary rewards cannot leak into a trajectory's sampled suffix return."""
    rewards = torch.tensor([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 999.0],
    ])
    valid = torch.tensor([
        [True, True],
        [True, True],
        [True, False],
    ])
    suffix = DQCACBetaGPU.finite_suffix_returns(rewards, valid, gamma=0.5)
    expected = torch.tensor([
        [2.75, 20.0],       # 1 + .5*2 + .25*3; 10 + .5*20
        [3.50, 20.0],       # 2 + .5*3; the second trajectory ends here
        [3.00, 0.0],        # invalid adapter-reset reward 999 is excluded
    ])
    assert torch.equal(suffix, expected)


def test_finite_suffix_batch_is_scalar_qr_sample_and_never_queries_bootstrap_path():
    """MC supervision is [M,1] and does not call the target actor or target critic."""
    agent = object.__new__(DQCACBetaGPU)
    agent.device = torch.device("cpu")
    agent.gamma = 0.5
    agent.huber_kappa = 1.0
    agent.taus = torch.tensor([0.25, 0.75])
    agent.critic_target_source = "finite_suffix_mc"

    def forbidden(*_args, **_kwargs):
        raise AssertionError("finite_suffix_mc must not query a bootstrap path")

    agent._sample_actions = forbidden
    agent.reward_target_critic = forbidden
    batch = agent.build_critic_batch(_boundary_target_rollout())

    # Flatten order is [t0b0,t0b1,t1b0]. Truncation b1 and termination b0 both stop recursion.
    assert batch["targets"].shape == (3, 1)
    assert torch.equal(batch["targets"], torch.tensor([[2.5], [2.0], [3.0]]))
    assert torch.equal(batch["transition_ids"], torch.tensor([0, 1, 2]))

    # A scalar target K=1 broadcasts against both fixed heads; the hand-computed loss is .375.
    predicted = torch.tensor([[1.0, 2.0]])
    loss = agent.quantile_huber_loss(predicted, torch.tensor([[3.0]]))
    assert torch.allclose(loss, torch.tensor(0.375))


def test_finite_suffix_epoch_updates_online_critic_but_leaves_target_bitwise_fixed():
    """The causal intervention replaces supervision without querying or moving target storage."""
    torch.manual_seed(17)
    agent = _optimizer_agent()
    agent.critic_target_source = "finite_suffix_mc"
    initial_online = {
        name: value.detach().clone()
        for name, value in agent.reward_critic.state_dict().items()
    }
    initial_target = {
        name: value.detach().clone()
        for name, value in agent.reward_target_critic.state_dict().items()
    }

    info = agent.update_critic(
        _five_transition_rollout(), permutation=[4, 1, 3, 0, 2])
    final_online = agent.reward_critic.state_dict()
    final_target = agent.reward_target_critic.state_dict()

    assert info["diagnostic/dqc/transition_coverage"] == 1.0
    assert agent.critic_optimizer_steps == 3
    assert agent.target_update_count == 0
    assert any(
        not torch.equal(final_online[name], initial_online[name])
        for name in initial_online)
    for name in initial_target:
        assert torch.equal(final_target[name], initial_target[name])


def test_online_epoch_covers_every_transition_once_and_polyak_updates_only_at_end():
    """M=5/minibatch=2 gives balanced [2,2,1], three Adam steps, then one target update."""
    torch.manual_seed(7)
    agent = _optimizer_agent()
    initial_target = {
        name: value.detach().clone()
        for name, value in agent.reward_target_critic.state_dict().items()
    }
    snapshots = []

    def hook(_index, _batch):
        snapshots.append({
            name: value.detach().clone()
            for name, value in agent.reward_target_critic.state_dict().items()
        })

    agent._critic_minibatch_hook = hook
    info = agent.update_critic(_five_transition_rollout(), permutation=[4, 1, 3, 0, 2])
    assert info["diagnostic/dqc/transition_coverage"] == 1.0
    assert agent.critic_optimizer_steps == 3
    assert agent.target_update_count == 1
    assert len(snapshots) == 3
    for snapshot in snapshots:
        for name in initial_target:
            assert torch.equal(snapshot[name], initial_target[name])

    # The final target is exactly one Polyak interpolation from the epoch-start target.
    online = agent.reward_critic.state_dict()
    final_target = agent.reward_target_critic.state_dict()
    for name in initial_target:
        expected = initial_target[name] * 0.8 + online[name] * 0.2
        assert torch.allclose(final_target[name], expected, atol=1e-7)


def test_fitted_hard_builds_one_target_then_regresses_repeatedly_and_hard_copies_once():
    """E fitted epochs reuse one frozen Bellman target; only their final boundary moves target."""
    torch.manual_seed(23)
    agent = _optimizer_agent()
    initial_target = {
        name: value.detach().clone()
        for name, value in agent.reward_target_critic.state_dict().items()
    }
    target_snapshots = []
    build_calls = 0
    original_builder = agent.build_critic_batch

    def counted_builder(rollout):
        nonlocal build_calls
        build_calls += 1
        return original_builder(rollout)

    def hook(_index, _batch):
        target_snapshots.append({
            name: value.detach().clone()
            for name, value in agent.reward_target_critic.state_dict().items()
        })

    agent.build_critic_batch = counted_builder
    agent._critic_minibatch_hook = hook
    info = agent.update_critic(
        _five_transition_rollout(),
        permutation=[4, 1, 3, 0, 2],
        regression_epochs=3,
        target_update_mode="hard",
    )

    # Five rows split [2,2,1], so three fitted epochs perform exactly nine Adam steps. The
    # Bellman target is materialized once before all nine and remains bitwise frozen throughout.
    assert build_calls == 1
    assert agent.critic_optimizer_steps == 9
    assert len(target_snapshots) == 9
    assert info["diagnostic/dqc/transition_coverage"] == 1.0
    for snapshot in target_snapshots:
        for name in initial_target:
            assert torch.equal(snapshot[name], initial_target[name])

    # A fitted generation ends in one exact target <- online copy, not E Polyak updates.
    assert agent.target_update_count == 1
    online = agent.reward_critic.state_dict()
    final_target = agent.reward_target_critic.state_dict()
    for name in online:
        assert torch.equal(final_target[name], online[name])


def test_hard_target_update_rejects_finite_suffix_supervision():
    """The fitted-hard warmup is a one-step Bellman propagation intervention, never MC mode."""
    agent = _optimizer_agent()
    agent.critic_target_source = "finite_suffix_mc"
    initial_target = copy.deepcopy(agent.reward_target_critic.state_dict())
    try:
        agent.update_critic(
            _five_transition_rollout(), regression_epochs=2, target_update_mode="hard")
    except ValueError as error:
        assert "one_step" in str(error) and "hard" in str(error)
    else:
        raise AssertionError("hard fitted warmup must reject finite_suffix_mc supervision")
    for name, value in initial_target.items():
        assert torch.equal(agent.reward_target_critic.state_dict()[name], value)


class _ScalarScoreActor(torch.nn.Module):
    """Minimal sequence actor whose mean has one directly testable parameter."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.2))

    def forward(self, states):
        mean = self.weight * states
        return mean, torch.zeros_like(mean), None


def test_actor_uses_one_score_step_fixed_bt_scaling_without_ppo_or_normalization():
    agent = object.__new__(DQCACBetaGPU)
    agent.actor = _ScalarScoreActor()
    agent.actor_optimizer = torch.optim.SGD(agent.actor.parameters(), lr=0.1)
    agent.actor_grad_clip = 10.0
    agent.actor_optimizer_steps = 0
    agent.num_envs, agent.horizon = 2, 5
    agent._log_2pi = math.log(2.0 * math.pi)
    rollout = {
        "S": torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]),
        "A_raw": torch.tensor([[[0.0], [1.0]], [[2.0], [3.0]]]),
        "valid": torch.tensor([[True, True], [True, False]]),
    }
    combined = torch.tensor([[1.0, -2.0], [3.0, 999.0]])

    mean, log_std, _ = agent.actor(rollout["S"])
    logp = agent._logp_from_params(rollout["A_raw"], mean, log_std)
    manual = -(
        logp * combined * rollout["valid"].float()
    ).sum() / float(2 * 5)
    info = agent.update_actor(rollout, combined)
    assert math.isclose(info["actor/loss"], float(manual), rel_tol=1e-6)
    assert agent.actor_optimizer_steps == 1
    assert not any("ratio" in key or "clip_fraction" in key for key in info)


def _scalar_actor_rollout_with_behavior_logp(agent):
    """Build one padded rollout and freeze its raw-action likelihood under the current actor."""
    rollout = {
        "S": torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]),
        "A_raw": torch.tensor([[[0.0], [1.0]], [[2.0], [3.0]]]),
        # Deliberately unrelated executed actions catch accidental critic/action-space reuse in PPO.
        "A_exec": torch.tensor([[[9.0], [8.0]], [[7.0], [6.0]]]),
        "valid": torch.tensor([[True, True], [True, False]]),
    }
    with torch.no_grad():
        behavior_mean, behavior_log_std, _ = agent.actor(rollout["S"])
        rollout["logp"] = agent._logp_from_params(
            rollout["A_raw"], behavior_mean, behavior_log_std).detach().clone()
        rollout["mean_old"] = behavior_mean.detach().clone()
        rollout["log_std_old"] = behavior_log_std.detach().clone()
    return rollout, behavior_mean.detach().clone(), behavior_log_std.detach().clone()


def test_dqc_ppo_epoch_matches_manual_ratio_clip_and_fixed_bt_scaling():
    """The PPO branch uses old raw-action logp, signed clipping, and DQC's B*T denominator."""
    agent = object.__new__(DQCACBetaGPU)
    agent.actor = _ScalarScoreActor()
    agent.actor_optimizer = torch.optim.SGD(agent.actor.parameters(), lr=0.01)
    agent.actor_grad_clip = 10.0
    agent.actor_optimizer_steps = 0
    agent.dqc_ratio_clip = 0.1
    agent.num_envs, agent.horizon = 2, 5
    agent._log_2pi = math.log(2.0 * math.pi)
    rollout, behavior_mean, behavior_log_std = (
        _scalar_actor_rollout_with_behavior_logp(agent))
    combined = torch.tensor([[1.0, -2.0], [3.0, 999.0]])

    # Move only the current policy after materializing behavior likelihoods, creating ratios on
    # both sides of one while leaving the stored actions and fixed DQC weight unchanged.
    with torch.no_grad():
        agent.actor.weight.add_(0.15)
    mean, log_std, _ = agent.actor(rollout["S"])
    new_logp = agent._logp_from_params(rollout["A_raw"], mean, log_std)
    ratio = torch.exp(new_logp - rollout["logp"])
    clipped = ratio.clamp(0.9, 1.1)
    surrogate = torch.minimum(ratio * combined, clipped * combined)
    expected = -(
        surrogate * rollout["valid"].float()
    ).sum() / float(agent.num_envs * agent.horizon)

    info = agent.update_actor_ppo_epoch(
        rollout, combined, behavior_mean, behavior_log_std)
    assert math.isclose(info["actor/loss"], float(expected), rel_tol=1e-6)
    assert agent.actor_optimizer_steps == 1
    assert info["actor/clip_fraction"] > 0.0
    assert info["actor/ratio_min"] < info["actor/ratio_max"]
    assert math.isfinite(info["diagnostic/dqc/policy_kl_behavior_after_epoch"])


def test_dqc_ppo_nonfinite_behavior_logp_fails_before_optimizer_mutation():
    """A corrupt behavior snapshot must pause the lineage without changing actor state."""
    agent = object.__new__(DQCACBetaGPU)
    agent.actor = _ScalarScoreActor()
    agent.actor_optimizer = torch.optim.SGD(agent.actor.parameters(), lr=0.01)
    agent.actor_grad_clip = 10.0
    agent.actor_optimizer_steps = 0
    agent.dqc_ratio_clip = 0.1
    agent.num_envs, agent.horizon = 2, 5
    agent._log_2pi = math.log(2.0 * math.pi)
    rollout, behavior_mean, behavior_log_std = (
        _scalar_actor_rollout_with_behavior_logp(agent))
    combined = torch.tensor([[1.0, -2.0], [3.0, 999.0]])

    # Corrupt a valid row rather than padding: the implementation must reject it before calling
    # backward/Adam. Invalid padded values are intentionally ignored throughout the rollout API.
    rollout["logp"][0, 0] = float("nan")
    weight_before = agent.actor.weight.detach().clone()
    try:
        agent.update_actor_ppo_epoch(
            rollout, combined, behavior_mean, behavior_log_std)
    except FloatingPointError as error:
        assert "non-finite DQC PPO log probability" in str(error)
    else:
        raise AssertionError("non-finite behavior logp did not fail fast")

    assert agent.actor_optimizer_steps == 0
    assert torch.equal(agent.actor.weight.detach(), weight_before)


def test_dqc_ppo_first_epoch_has_same_score_gradient_as_fresh_on_policy_step():
    """At ratio=1, the PPO surrogate and original score estimator update theta identically."""
    score = object.__new__(DQCACBetaGPU)
    score.actor = _ScalarScoreActor()
    score.actor_optimizer = torch.optim.SGD(score.actor.parameters(), lr=0.01)
    score.actor_grad_clip = 10.0
    score.actor_optimizer_steps = 0
    score.num_envs, score.horizon = 2, 5
    score._log_2pi = math.log(2.0 * math.pi)
    rollout, behavior_mean, behavior_log_std = _scalar_actor_rollout_with_behavior_logp(score)
    combined = torch.tensor([[1.0, -2.0], [3.0, 999.0]])

    ppo = object.__new__(DQCACBetaGPU)
    ppo.actor = copy.deepcopy(score.actor)
    ppo.actor_optimizer = torch.optim.SGD(ppo.actor.parameters(), lr=0.01)
    ppo.actor_grad_clip = 10.0
    ppo.actor_optimizer_steps = 0
    ppo.dqc_ratio_clip = 0.1
    ppo.num_envs, ppo.horizon = score.num_envs, score.horizon
    ppo._log_2pi = score._log_2pi

    score.update_actor(copy.deepcopy(rollout), combined)
    ppo.update_actor_ppo_epoch(
        copy.deepcopy(rollout), combined, behavior_mean, behavior_log_std)
    assert torch.allclose(score.actor.weight, ppo.actor.weight, rtol=0.0, atol=1e-7)


def test_dqc_ppo_target_kl_stops_remaining_actor_epochs_after_triggering_step():
    """Exact behavior KL retains the triggering Adam step and skips epochs 2 through 8."""
    agent = object.__new__(DQCACBetaGPU)
    agent.actor = _ScalarScoreActor()
    agent.actor_optimizer = torch.optim.SGD(agent.actor.parameters(), lr=0.05)
    agent.actor_grad_clip = 10.0
    agent.actor_optimizer_steps = 0
    agent.actor_epochs = 8
    agent.dqc_actor_reuse_mode = "ppo_clip"
    agent.dqc_ratio_clip = 0.1
    agent.dqc_target_kl = 1e-12
    agent.actor_kl_early_stop_count = 0
    agent.num_envs, agent.horizon = 2, 5
    agent._log_2pi = math.log(2.0 * math.pi)
    rollout, _behavior_mean, _behavior_log_std = (
        _scalar_actor_rollout_with_behavior_logp(agent))
    combined = torch.tensor([[1.0, -2.0], [3.0, 999.0]])

    info = agent.update_actor_epochs(rollout, combined)
    assert agent.actor_optimizer_steps == 1
    assert info["diagnostic/dqc/actor_epochs_requested"] == 8.0
    assert info["diagnostic/dqc/actor_epochs_applied"] == 1.0
    assert info["diagnostic/dqc/actor_kl_early_stop"] == 1.0
    assert info["diagnostic/dqc/actor_kl_early_stop_count"] == 1.0
    assert info["diagnostic/dqc/policy_kl_after_epochs"] >= agent.dqc_target_kl


def test_dqc_ppo_large_target_kl_applies_all_eight_actor_epochs():
    """A non-binding KL guard preserves the configured eight PPO optimizer steps."""
    agent = object.__new__(DQCACBetaGPU)
    agent.actor = _ScalarScoreActor()
    agent.actor_optimizer = torch.optim.SGD(agent.actor.parameters(), lr=0.01)
    agent.actor_grad_clip = 10.0
    agent.actor_optimizer_steps = 0
    agent.actor_epochs = 8
    agent.dqc_actor_reuse_mode = "ppo_clip"
    agent.dqc_ratio_clip = 0.1
    agent.dqc_target_kl = 100.0
    agent.actor_kl_early_stop_count = 0
    agent.num_envs, agent.horizon = 2, 5
    agent._log_2pi = math.log(2.0 * math.pi)
    rollout, _behavior_mean, _behavior_log_std = (
        _scalar_actor_rollout_with_behavior_logp(agent))
    combined = torch.tensor([[1.0, -2.0], [3.0, 999.0]])

    info = agent.update_actor_epochs(rollout, combined)
    assert agent.actor_optimizer_steps == 8
    assert info["diagnostic/dqc/actor_epochs_applied"] == 8.0
    assert info["diagnostic/dqc/actor_kl_early_stop"] == 0.0
    assert info["diagnostic/dqc/actor_kl_early_stop_count"] == 0.0
    assert info["diagnostic/dqc/policy_kl_after_epochs"] < agent.dqc_target_kl


def test_actor_epochs_apply_eight_full_rollout_steps_and_report_total_kl():
    """The epoch wrapper calls the one-step primitive eight times without fabricating PPO."""
    agent = object.__new__(DQCACBetaGPU)
    agent.actor = _ScalarScoreActor()
    agent.actor_optimizer = torch.optim.SGD(agent.actor.parameters(), lr=0.01)
    agent.actor_grad_clip = 10.0
    agent.actor_optimizer_steps = 0
    agent.actor_epochs = 8
    agent.num_envs, agent.horizon = 2, 5
    agent._log_2pi = math.log(2.0 * math.pi)
    rollout = {
        "S": torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]),
        "A_raw": torch.tensor([[[0.0], [1.0]], [[2.0], [3.0]]]),
        "valid": torch.tensor([[True, True], [True, False]]),
    }
    combined = torch.tensor([[1.0, -2.0], [3.0, 999.0]])

    initial_weight = agent.actor.weight.detach().clone()
    info = agent.update_actor_epochs(rollout, combined)
    assert agent.actor_optimizer_steps == 8
    assert info["diagnostic/dqc/actor_epochs_applied"] == 8.0
    assert info["diagnostic/dqc/actor_optimizer_steps"] == 8.0
    assert info["diagnostic/dqc/policy_kl_after_epochs"] >= 0.0
    assert info["diagnostic/dqc/policy_kl_step_max"] >= (
        info["diagnostic/dqc/policy_kl_step_mean"])
    assert not torch.equal(agent.actor.weight.detach(), initial_weight)
    assert not any("ratio" in key or "clip_fraction" in key for key in info)


def test_actor_epoch_kl_snapshots_expanded_learnable_log_std_storage():
    """Reported KL includes variance-only movement from the real recurrent actor parameter."""
    agent = object.__new__(DQCACBetaGPU)
    agent.actor = RecurrentGaussianActor(
        1, 1, hidden=(2, 2), lstm_size=2,
        init_std=1.0, normalize_observation=False)

    # Zero and freeze every mean-path parameter so the eight updates can change only log_std.
    # RecurrentGaussianActor.forward() returns an expanded view of this scalar parameter, which
    # makes this fixture exercise the storage-aliasing case that a standalone tensor actor misses.
    with torch.no_grad():
        for name, parameter in agent.actor.named_parameters():
            if name != "log_std":
                parameter.zero_()
    for name, parameter in agent.actor.named_parameters():
        parameter.requires_grad_(name == "log_std")

    agent.actor_optimizer = torch.optim.SGD([agent.actor.log_std], lr=0.05)
    agent.actor_grad_clip = 10.0
    agent.actor_optimizer_steps = 0
    agent.actor_epochs = 8
    agent.num_envs, agent.horizon = 2, 2
    agent._log_2pi = math.log(2.0 * math.pi)
    rollout = {
        "S": torch.zeros(2, 2, 1),
        "A_raw": torch.full((2, 2, 1), 2.0),
        "valid": torch.tensor([[True, True], [True, False]]),
    }
    combined = torch.ones(2, 2)

    with torch.no_grad():
        initial_mean, initial_log_std, _ = agent.actor(rollout["S"])
        initial_mean = initial_mean.clone()
        initial_log_std = initial_log_std.clone()
    info = agent.update_actor_epochs(rollout, combined)
    with torch.no_grad():
        final_mean, final_log_std, _ = agent.actor(rollout["S"])

    # Hand-compute KL(N_old || N_final) for each valid recurrent history. Because the entire mean
    # path is fixed at zero, every positive term must come from the learnable standard deviation.
    initial_variance = torch.exp(2.0 * initial_log_std)
    final_variance = torch.exp(2.0 * final_log_std)
    elementwise_kl = (
        final_log_std - initial_log_std
        + (initial_variance + (initial_mean - final_mean).square())
        / (2.0 * final_variance)
        - 0.5
    )
    valid = rollout["valid"].to(elementwise_kl.dtype)
    expected_kl = (
        elementwise_kl.sum(dim=-1) * valid
    ).sum() / valid.sum()

    assert torch.equal(initial_mean, final_mean)
    assert not torch.equal(initial_log_std, final_log_std)
    assert expected_kl.item() > 0.0
    assert info["diagnostic/dqc/policy_kl_step_mean"] > 0.0
    assert math.isclose(
        info["diagnostic/dqc/policy_kl_after_epochs"],
        float(expected_kl.item()), rel_tol=1e-6, abs_tol=1e-8)


class _RiskCritic:
    """Negative executed actions have high lower CDF; positive actions have zero lower CDF."""

    def __init__(self):
        self.actions = []

    def __call__(self, states, actions):
        del states
        self.actions.append(actions.detach().clone())
        low = torch.where(actions[..., 0] < 0, -2.0, 1.0)
        high = torch.where(actions[..., 0] < 0, -1.0, 2.0)
        return torch.stack((low, high), dim=-1)


class _PositiveBaselineActor:
    def __call__(self, states):
        shape = (*states.shape[:-1], 1)
        return torch.full(shape, 5.0), torch.full(shape, -20.0), None


def test_mean_cdf_baselines_and_advantages_are_detached_with_negative_high_cdf_risk_term():
    agent = object.__new__(DQCACBetaGPU)
    agent.device = torch.device("cpu")
    agent.gamma, agent.beta = 0.99, 0.995
    agent.lambda_probability = torch.tensor(2.0)
    agent.num_action_samples = 2
    agent._action_low, agent._action_high = torch.tensor([-1.0]), torch.tensor([1.0])
    agent.reward_critic = _RiskCritic()
    agent.actor = _PositiveBaselineActor()
    rollout = {
        "S": torch.zeros(1, 1, 1),
        "A_raw": torch.full((1, 1, 1), -3.0),
        "A_exec": torch.full((1, 1, 1), -1.0),
        "budgets": torch.zeros(1, 1),
        "valid": torch.ones(1, 1, dtype=torch.bool),
    }
    terms = agent.actor_advantages(rollout)
    for name in ("combined", "mean_advantage", "cdf_advantage", "value_mean", "value_cdf"):
        assert not terms[name].requires_grad
    assert float(terms["cdf_advantage"]) == 1.0
    assert float(terms["risk_term"]) > 0.0
    assert float(terms["combined"]) < float(terms["mean_term"])
    assert torch.equal(agent.reward_critic.actions[0], torch.tensor([[[-1.0]]]))
    assert all(float(action.abs().max()) <= 1.0 for action in agent.reward_critic.actions)


def test_empirical_dual_is_inclusive_and_moves_both_directions_without_critic_bootstrap():
    agent = object.__new__(DQCACBetaGPU)
    agent.return_threshold, agent.failure_alpha = 0.0, 0.2
    agent.lambda_probability = torch.tensor(0.5)
    agent.lambda_max = 50.0
    agent.lambda_a, agent.lambda_b, agent.lambda_c = 1.0, 1.0, 0.0
    agent.dual_update_count = 0
    agent.last_dual_prob = 0.0
    probability, residual, _ = agent.update_dual(torch.tensor([-1.0, 0.0, 1.0, 2.0]))
    assert probability == 0.5 and residual > 0.0
    assert float(agent.lambda_probability) > 0.5
    agent.lambda_probability.fill_(0.1)
    probability, residual, _ = agent.update_dual(torch.ones(4))
    assert probability == 0.0 and residual < 0.0
    assert float(agent.lambda_probability) == 0.0


def _assert_nested_equal(actual, expected):
    """Recursively compare optimizer/scheduler state without tensor truth-value ambiguity."""
    assert type(actual) is type(expected)
    if torch.is_tensor(actual):
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_nested_equal(left, right)
    else:
        assert actual == expected


class _RMSRecorder:
    """Warmup fixture: count observation updates; any actor baseline call fails the test."""

    def __init__(self):
        self.calls = 0

    def update_obs_rms(self, observations):
        assert observations.numel() > 0
        self.calls += 1


def test_first_five_initialization_batches_freeze_actor_dual_rms_and_scheduler():
    agent = object.__new__(DQCACBetaGPU)
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    agent.actor_optimizer = torch.optim.Adam([parameter], lr=2e-4)
    critic_parameter = torch.nn.Parameter(torch.tensor(0.0))
    agent.critic_optimizer = torch.optim.Adam([critic_parameter], lr=3e-4)
    agent.actor_lr_b, agent.actor_lr_c = 10000.0, 0.9
    agent.actor_scheduler = agent._new_actor_scheduler()
    agent.actor = _RMSRecorder()
    agent.device = torch.device("cpu")
    agent.gamma = 0.99
    agent.beta = 0.995
    agent.return_threshold = -0.03
    agent.failure_alpha = 0.2
    agent.warmup_iters = 5
    agent.critic_warmup_mode = "legacy"
    agent.warmup_critic_epochs = 1
    agent.actor_optimizer_steps = 0
    agent.dual_update_count = 0
    agent.lambda_probability = torch.tensor(0.0)
    agent.num_envs, agent.horizon = 1, 2
    agent.update_critic = lambda _rollout, **_kwargs: {
        "critic/reward_quantile_loss": 0.0,
        "diagnostic/dqc/critic_cdf_mean": 0.0,
        "diagnostic/dqc/critic_optimizer_steps": 5.0,
        "diagnostic/dqc/transition_coverage": 1.0,
        "diagnostic/reward_critic/atom_crossing_row_fraction": 0.0,
        "diagnostic/reward_critic/atom_crossing_adjacent_fraction": 0.0,
    }
    agent.actor_advantages = lambda _rollout: (_ for _ in ()).throw(
        AssertionError("warmup must not sample actor baselines"))
    agent._log_core = lambda iteration, returns, diagnostic_disc_cost, diagnostic_raw_cost, extra: extra

    rollout = {
        "S": torch.zeros(2, 1, 1),
        "A_raw": torch.zeros(2, 1, 1),
        "A_exec": torch.zeros(2, 1, 1),
        "R": torch.zeros(2, 1),
        "C": torch.zeros(2, 1),
        "valid": torch.ones(2, 1, dtype=torch.bool),
        "disc_return": torch.zeros(1),
        "diagnostic_disc_cost": torch.zeros(1),
        "diagnostic_raw_cost": torch.zeros(1),
    }
    for iteration in range(5):
        metrics = agent._update_rollout(
            -1, copy.deepcopy(rollout), initialization=True)
    assert agent.actor_optimizer_steps == 0
    assert agent.dual_update_count == 0
    assert len(agent.actor_optimizer.state) == 0
    assert agent.actor.calls == 0
    assert agent.actor_scheduler.last_epoch == 0
    expected = {
        "critic/reward_quantile_loss",
        "diagnostic/dqc/mean_advantage_mean",
        "diagnostic/dqc/mean_advantage_std",
        "diagnostic/dqc/cdf_advantage_mean",
        "diagnostic/dqc/cdf_advantage_std",
        "diagnostic/dqc/mean_term_rms",
        "diagnostic/dqc/risk_term_rms",
        "diagnostic/dqc/risk_to_mean",
        "diagnostic/dqc/critic_cdf_mean",
        "diagnostic/dqc/critic_optimizer_steps",
        "diagnostic/dqc/transition_coverage",
        "diagnostic/reward_critic/atom_crossing_row_fraction",
        "diagnostic/reward_critic/atom_crossing_adjacent_fraction",
        "diagnostic/action/clip_any_fraction",
        "diagnostic/action/clip_element_fraction",
        "diagnostic/action/mean_abs_raw_minus_exec",
    }
    assert expected <= set(metrics)
    removed = {"dqc/replay_size", "actor/approx_kl", "actor/clip_fraction", "actor/ratio_mean"}
    assert removed.isdisjoint(metrics)


def test_fitted_hard_warmup_freezes_actor_rms_adam_scheduler_and_dual_bitwise():
    """Only critic state may move during fitted-hard warmup; every policy-side state is fixed."""
    torch.manual_seed(29)
    agent = object.__new__(DQCACBetaGPU)
    # Use a real normalized recurrent actor so this checks parameters and RMS buffers together.
    agent.actor = RecurrentGaussianActor(
        1, 1, hidden=(2, 2), lstm_size=2,
        init_std=0.5, normalize_observation=True)
    agent.actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=2e-4)

    # Seed Adam moments before the warmup call. An empty optimizer state could only prove that no
    # state was initialized; populated exp_avg/exp_avg_sq tensors prove bitwise preservation.
    agent.actor_optimizer.zero_grad(set_to_none=True)
    sum(parameter.sum() for parameter in agent.actor.parameters()).backward()
    agent.actor_optimizer.step()
    agent.actor_lr_b, agent.actor_lr_c = 10000.0, 0.9
    agent.actor_scheduler = agent._new_actor_scheduler()
    critic_parameter = torch.nn.Parameter(torch.tensor(0.0))
    agent.critic_optimizer = torch.optim.Adam([critic_parameter], lr=3e-4)
    agent.device = torch.device("cpu")
    agent.gamma, agent.beta = 0.99, 0.995
    agent.return_threshold, agent.failure_alpha = -0.03, 0.2
    agent.warmup_iters = 5
    agent.critic_warmup_mode = "fitted_hard"
    agent.warmup_critic_epochs = 3
    agent.actor_optimizer_steps = 1
    agent.dual_update_count = 4
    agent.last_dual_prob = 0.6
    agent.lambda_probability = torch.tensor(0.75)
    agent.lambda_dual = agent.lambda_probability
    agent.num_envs, agent.horizon = 1, 2
    update_arguments = {}

    def critic_update(_rollout, **kwargs):
        update_arguments.update(kwargs)
        return {
            "critic/reward_quantile_loss": 0.0,
            "diagnostic/dqc/critic_cdf_mean": 0.0,
            "diagnostic/dqc/critic_optimizer_steps": 3.0,
            "diagnostic/dqc/transition_coverage": 1.0,
            "diagnostic/reward_critic/atom_crossing_row_fraction": 0.0,
            "diagnostic/reward_critic/atom_crossing_adjacent_fraction": 0.0,
        }

    agent.update_critic = critic_update
    agent.actor_advantages = lambda _rollout: (_ for _ in ()).throw(
        AssertionError("fitted-hard warmup must not query actor baselines"))
    agent._log_core = lambda iteration, returns, diagnostic_disc_cost, diagnostic_raw_cost, extra: extra
    rollout = {
        "S": torch.tensor([[[3.0]], [[5.0]]]),
        "A_raw": torch.zeros(2, 1, 1),
        "A_exec": torch.zeros(2, 1, 1),
        "R": torch.zeros(2, 1),
        "C": torch.zeros(2, 1),
        "valid": torch.ones(2, 1, dtype=torch.bool),
        "disc_return": torch.zeros(1),
        "diagnostic_disc_cost": torch.zeros(1),
        "diagnostic_raw_cost": torch.zeros(1),
    }
    actor_before = copy.deepcopy(agent.actor.state_dict())
    adam_before = copy.deepcopy(agent.actor_optimizer.state_dict())
    scheduler_before = copy.deepcopy(agent.actor_scheduler.state_dict())
    lambda_before = agent.lambda_probability.detach().clone()

    agent._update_rollout(-1, rollout, initialization=True)

    assert update_arguments == {
        "regression_epochs": 3,
        "target_update_mode": "hard",
    }
    _assert_nested_equal(agent.actor.state_dict(), actor_before)
    _assert_nested_equal(agent.actor_optimizer.state_dict(), adam_before)
    _assert_nested_equal(agent.actor_scheduler.state_dict(), scheduler_before)
    assert torch.equal(agent.lambda_probability, lambda_before)
    assert agent.actor_optimizer_steps == 1
    assert agent.dual_update_count == 4
    assert agent.last_dual_prob == 0.6


def test_crossing_and_action_clip_diagnostics_match_hand_counts():
    ordered = torch.tensor([[0.0, 1.0, 2.0], [-1.0, 0.0, 1.0]])
    crossed = torch.tensor([[0.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    assert DQCACBetaGPU.atom_crossing_statistics(ordered) == (0.0, 0.0)
    row, adjacent = DQCACBetaGPU.atom_crossing_statistics(crossed)
    assert row == 1.0
    assert adjacent == 0.75

    rollout = {
        "A_raw": torch.tensor([[[2.0, 0.5], [0.0, -3.0]]]),
        "A_exec": torch.tensor([[[1.0, 0.5], [0.0, -1.0]]]),
        "valid": torch.tensor([[True, True]]),
    }
    metrics = DQCACBetaGPU.action_clip_diagnostics(rollout)
    assert metrics["diagnostic/action/clip_any_fraction"] == 1.0
    assert metrics["diagnostic/action/clip_element_fraction"] == 0.5
    assert metrics["diagnostic/action/mean_abs_raw_minus_exec"] == 0.75


def test_safety_cost_changes_cannot_change_critic_targets_actor_terms_or_dual():
    """C is diagnostic-only: core DQC methods have identical output under arbitrary cost values."""
    torch.manual_seed(13)
    agent = _optimizer_agent()
    first = _five_transition_rollout(cost_value=0.0)
    second = _five_transition_rollout(cost_value=999.0)
    first_batch = agent.build_one_step_batch(first)
    second_batch = agent.build_one_step_batch(second)
    assert torch.equal(first_batch["targets"], second_batch["targets"])
    assert torch.equal(first_batch["states"], second_batch["states"])

    dual = object.__new__(DQCACBetaGPU)
    dual.return_threshold, dual.failure_alpha = 0.0, 0.2
    dual.lambda_probability = torch.tensor(0.0)
    dual.lambda_max = 50.0
    dual.lambda_a, dual.lambda_b, dual.lambda_c = 1.0, 1.0, 0.0
    dual.dual_update_count, dual.last_dual_prob = 0, 0.0
    returns = torch.tensor([-1.0, 1.0])
    result = dual.update_dual(returns)
    assert result[0] == 0.5


def _checkpoint_agent(
        target_source="one_step", warmup_mode="legacy", warmup_epochs=1,
        critic_epochs=1, actor_epochs=1, actor_reuse_mode="score_function",
        ratio_clip=0.1, target_kl=0.0):
    """Build all DQC checkpoint fields without constructing Safety Gym or W&B."""
    agent = _optimizer_agent()
    agent.dqc_update_mode = agent.UPDATE_MODE
    agent.critic_target_source = str(target_source)
    agent.critic_warmup_mode = str(warmup_mode)
    agent.warmup_critic_epochs = int(warmup_epochs)
    agent.critic_epochs = int(critic_epochs)
    agent.actor_epochs = int(actor_epochs)
    agent.dqc_actor_reuse_mode = str(actor_reuse_mode)
    agent.dqc_ratio_clip = float(ratio_clip)
    agent.dqc_target_kl = float(target_kl)
    agent.actor = RecurrentGaussianActor(
        1, 1, hidden=(2, 2), lstm_size=2, normalize_observation=False)
    agent.actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=2e-4)
    agent.actor_lr_b, agent.actor_lr_c = 10000.0, 0.9
    agent.actor_scheduler = agent._new_actor_scheduler()
    agent.lambda_probability = torch.tensor(0.25)
    agent.lambda_dual = agent.lambda_probability
    agent.dual_update_count = 2
    agent.last_dual_prob = 0.3
    agent.actor_optimizer_steps = 1
    agent.actor_kl_early_stop_count = 0
    agent.training_budget_semantics = "main_excludes_initialization_v1"
    agent.main_env_steps = 123
    agent.initialization_env_steps = 10
    agent.actual_env_steps = 133
    agent.total_env_steps = 133
    agent.initialization_cursor = 5
    agent.warmup_iters = 5
    agent.last_empirical_prob = 0.4
    agent.failure_alpha = 0.2
    agent.return_threshold = -0.03
    agent.beta = 0.995
    agent.checkpoint_signature = {
        "training_budget_semantics": agent.training_budget_semantics,
        "algorithm": "DQCACBeta",
        "dqc_update_mode": agent.UPDATE_MODE,
        "critic_target_source": agent.critic_target_source,
        "critic_warmup_mode": agent.critic_warmup_mode,
        "warmup_critic_epochs": agent.warmup_critic_epochs,
        "critic_minibatch_size": 2,
        "critic_epochs": agent.critic_epochs,
        "actor_epochs": agent.actor_epochs,
        "dqc_actor_reuse_mode": agent.dqc_actor_reuse_mode,
        "dqc_ratio_clip": agent.dqc_ratio_clip,
        "dqc_target_kl": agent.dqc_target_kl,
    }
    return agent


def test_online_checkpoint_roundtrip_reproduces_next_fixed_rollout_update_and_rejects_old(tmp_path):
    torch.manual_seed(19)
    source = _checkpoint_agent()
    checkpoint = tmp_path / "online.pt"
    save_checkpoint(checkpoint, source, iteration=4)

    expected_info = source.update_critic(_five_transition_rollout())
    expected_online = copy.deepcopy(source.reward_critic.state_dict())
    expected_target = copy.deepcopy(source.reward_target_critic.state_dict())

    restored = _checkpoint_agent()
    load_checkpoint(checkpoint, restored)
    actual_info = restored.update_critic(_five_transition_rollout())
    assert actual_info["critic/reward_quantile_loss"] == expected_info["critic/reward_quantile_loss"]
    for name, value in expected_online.items():
        assert torch.equal(restored.reward_critic.state_dict()[name], value)
    for name, value in expected_target.items():
        assert torch.equal(restored.reward_target_critic.state_dict()[name], value)
    restored_state = restored.checkpoint_state()
    assert "replay" not in restored_state
    assert restored_state["critic_warmup_mode"] == "legacy"
    assert restored_state["warmup_critic_epochs"] == 1

    # Simulate a signed replay-era payload. The specific state hook is not bypassed by transfer,
    # and rejection occurs before the target actor tensor is overwritten.
    legacy_payload = torch.load(checkpoint, weights_only=False)
    legacy_payload["agent"].pop("dqc_update_mode")
    legacy_payload["agent"]["replay"] = {"storage": []}
    legacy_path = tmp_path / "legacy.pt"
    torch.save(legacy_payload, legacy_path)
    strict = _checkpoint_agent()
    actor_before = copy.deepcopy(strict.actor.state_dict())
    try:
        load_checkpoint(
            legacy_path, strict, restore_rng=False, allow_signature_mismatch=True)
    except ValueError as error:
        assert "replay-era" in str(error)
    else:
        raise AssertionError("replay-era DQC checkpoint must fail fast")
    for name, value in actor_before.items():
        assert torch.equal(strict.actor.state_dict()[name], value)


def test_checkpoint_missing_warmup_fields_means_legacy_one_epoch(tmp_path):
    """Pre-feature state has explicit legacy/1 meaning instead of becoming ambiguous."""
    source = _checkpoint_agent(warmup_mode="legacy", warmup_epochs=1)
    checkpoint = tmp_path / "with_warmup_identity.pt"
    save_checkpoint(checkpoint, source, iteration=4)
    payload = torch.load(checkpoint, weights_only=False)

    # Model an older checkpoint at both identity layers. Signature transfer is needed solely
    # because the generic checker cannot infer defaults; DQC state validation still must infer
    # legacy/1 and would reject a semantically incompatible requested warmup below.
    payload["agent"].pop("critic_warmup_mode")
    payload["agent"].pop("warmup_critic_epochs")
    payload["experiment_signature"].pop("critic_warmup_mode")
    payload["experiment_signature"].pop("warmup_critic_epochs")
    old_path = tmp_path / "before_warmup_identity.pt"
    torch.save(payload, old_path)

    restored = _checkpoint_agent(warmup_mode="legacy", warmup_epochs=1)
    load_checkpoint(
        old_path, restored, restore_rng=False, allow_signature_mismatch=True)
    assert restored.critic_warmup_mode == "legacy"
    assert restored.warmup_critic_epochs == 1
    for name, value in source.actor.state_dict().items():
        assert torch.equal(restored.actor.state_dict()[name], value)


def test_checkpoint_warmup_identity_mismatch_fails_before_mutation(tmp_path):
    """Transfer approval cannot reinterpret either fitted-hard mode or its regression depth."""
    cases = [
        ("fitted_hard", 3, "legacy", 1, "critic_warmup_mode mismatch"),
        ("fitted_hard", 3, "fitted_hard", 2, "warmup_critic_epochs mismatch"),
    ]
    for index, (saved_mode, saved_epochs, mode, epochs, expected) in enumerate(cases):
        source = _checkpoint_agent(
            warmup_mode=saved_mode, warmup_epochs=saved_epochs)
        checkpoint = tmp_path / f"warmup_mismatch_{index}.pt"
        save_checkpoint(checkpoint, source, iteration=4)
        requested = _checkpoint_agent(warmup_mode=mode, warmup_epochs=epochs)
        actor_before = copy.deepcopy(requested.actor.state_dict())
        try:
            load_checkpoint(
                checkpoint, requested, restore_rng=False,
                allow_signature_mismatch=True)
        except ValueError as error:
            assert expected in str(error)
        else:
            raise AssertionError("checkpoint must reject a changed critic warmup identity")
        for name, value in actor_before.items():
            assert torch.equal(requested.actor.state_dict()[name], value)


def test_checkpoint_epoch_identity_roundtrips_and_mismatch_fails_before_mutation(tmp_path):
    """c8a8 resumes itself, but neither epoch count can be reinterpreted under transfer."""
    source = _checkpoint_agent(critic_epochs=8, actor_epochs=8)
    checkpoint = tmp_path / "c8a8.pt"
    save_checkpoint(checkpoint, source, iteration=4)

    restored = _checkpoint_agent(critic_epochs=8, actor_epochs=8)
    load_checkpoint(checkpoint, restored, restore_rng=False)
    assert restored.critic_epochs == 8
    assert restored.actor_epochs == 8

    cases = [
        (1, 8, "critic_epochs mismatch"),
        (8, 1, "actor_epochs mismatch"),
    ]
    for critic_epochs, actor_epochs, expected in cases:
        requested = _checkpoint_agent(
            critic_epochs=critic_epochs, actor_epochs=actor_epochs)
        actor_before = copy.deepcopy(requested.actor.state_dict())
        critic_before = copy.deepcopy(requested.reward_critic.state_dict())
        try:
            load_checkpoint(
                checkpoint, requested, restore_rng=False,
                allow_signature_mismatch=True)
        except ValueError as error:
            assert expected in str(error)
        else:
            raise AssertionError("checkpoint must reject a changed DQC epoch identity")
        for name, value in actor_before.items():
            assert torch.equal(requested.actor.state_dict()[name], value)
        for name, value in critic_before.items():
            assert torch.equal(requested.reward_critic.state_dict()[name], value)


def test_checkpoint_ppo_reuse_identity_roundtrips_and_mismatch_fails_before_mutation(tmp_path):
    """PPO mode, ratio width, and target KL are immutable even under generic transfer."""
    source = _checkpoint_agent(
        critic_epochs=8, actor_epochs=8, actor_reuse_mode="ppo_clip",
        ratio_clip=0.1, target_kl=0.01)
    checkpoint = tmp_path / "ppo_c8a8.pt"
    save_checkpoint(checkpoint, source, iteration=4)

    restored = _checkpoint_agent(
        critic_epochs=8, actor_epochs=8, actor_reuse_mode="ppo_clip",
        ratio_clip=0.1, target_kl=0.01)
    load_checkpoint(checkpoint, restored, restore_rng=False)
    assert restored.dqc_actor_reuse_mode == "ppo_clip"
    assert restored.dqc_ratio_clip == 0.1
    assert restored.dqc_target_kl == 0.01

    cases = [
        ("score_function", 0.1, 0.0, "dqc_actor_reuse_mode mismatch"),
        ("ppo_clip", 0.2, 0.01, "dqc_ratio_clip mismatch"),
        ("ppo_clip", 0.1, 0.02, "dqc_target_kl mismatch"),
    ]
    for mode, clip, target, expected in cases:
        requested = _checkpoint_agent(
            critic_epochs=8, actor_epochs=8, actor_reuse_mode=mode,
            ratio_clip=clip, target_kl=target)
        actor_before = copy.deepcopy(requested.actor.state_dict())
        try:
            load_checkpoint(
                checkpoint, requested, restore_rng=False,
                allow_signature_mismatch=True)
        except ValueError as error:
            assert expected in str(error)
        else:
            raise AssertionError("checkpoint must reject a changed DQC PPO reuse identity")
        for name, value in actor_before.items():
            assert torch.equal(requested.actor.state_dict()[name], value)


def test_checkpoint_target_source_mismatch_fails_before_mutation_even_with_transfer(tmp_path):
    """Agent-level source identity cannot be bypassed by ordinary signature-transfer approval."""
    source = _checkpoint_agent(target_source="finite_suffix_mc")
    checkpoint = tmp_path / "finite_suffix.pt"
    save_checkpoint(checkpoint, source, iteration=4)

    requested = _checkpoint_agent(target_source="one_step")
    actor_before = copy.deepcopy(requested.actor.state_dict())
    critic_before = copy.deepcopy(requested.reward_critic.state_dict())
    try:
        load_checkpoint(
            checkpoint, requested, restore_rng=False, allow_signature_mismatch=True)
    except ValueError as error:
        assert "critic_target_source mismatch" in str(error)
    else:
        raise AssertionError("cross-source DQC resume must fail fast")

    # validate_checkpoint_state runs before actor, critic, optimizer, or RNG mutation.
    for name, value in actor_before.items():
        assert torch.equal(requested.actor.state_dict()[name], value)
    for name, value in critic_before.items():
        assert torch.equal(requested.reward_critic.state_dict()[name], value)


def test_old_mixed_budget_checkpoint_fails_before_actor_mutation_even_with_transfer(tmp_path):
    """Generic transfer cannot reinterpret a checkpoint that counted warmup inside main steps."""
    source = _checkpoint_agent()
    checkpoint = tmp_path / "new_budget.pt"
    save_checkpoint(checkpoint, source, iteration=4)
    payload = torch.load(checkpoint, weights_only=False)
    payload["agent"].pop("training_budget_semantics")
    old_path = tmp_path / "old_mixed_budget.pt"
    torch.save(payload, old_path)

    requested = _checkpoint_agent()
    actor_before = copy.deepcopy(requested.actor.state_dict())
    try:
        load_checkpoint(
            old_path, requested, restore_rng=False,
            allow_signature_mismatch=True)
    except ValueError as error:
        assert "training_budget_semantics mismatch" in str(error)
    else:
        raise AssertionError("old mixed-budget DQC checkpoint must fail fast")
    for name, value in actor_before.items():
        assert torch.equal(requested.actor.state_dict()[name], value)


def test_formal_dqc_resume_with_complete_cursor_never_repeats_initialization():
    """A latest.pt state with cursor=warmup_iters enters main training directly."""
    agent = object.__new__(DQCACBetaGPU)
    agent.initialization_cursor = 10
    agent.warmup_iters = 10
    agent._rollout_core = lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError("formal resume must not recollect initialization"))
    agent._update_rollout = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("formal resume must not repeat critic warmup"))
    agent._after_initialization_batch = lambda *_args: (_ for _ in ()).throw(
        AssertionError("formal resume must not write initialization checkpoint"))
    agent._run_initialization()


def test_dqc_default_config_and_signature_have_only_online_update_knobs():
    args = base_args("DQCACBeta", 0, torch.device("cpu"), "SimpleButton")
    signature = checkpoint_signature(args, "DQCACBeta")
    assert args.return_threshold == -0.03
    assert args.warmup_iters == 5
    assert args.critic_minibatch_size == 1024
    assert args.critic_epochs == 1
    assert args.actor_epochs == 1
    assert args.dqc_actor_reuse_mode == "score_function"
    assert args.dqc_ratio_clip == 0.1
    assert args.dqc_target_kl == 0.0
    assert args.dqc_update_mode == "online_1step_v1"
    assert args.critic_target_source == "one_step"
    assert args.critic_warmup_mode == "legacy"
    assert args.warmup_critic_epochs == 1
    assert args.normalize_combined_advantage is False
    assert "dqc_online_1step_mlp_v1" in args.wandb_tags
    for removed in ("n_step", "critic_updates", "replay_capacity", "replay_batch_size"):
        assert not hasattr(args, removed)
        assert removed not in signature
    assert signature["dqc_update_mode"] == "online_1step_v1"
    assert signature["training_budget_semantics"] == "main_excludes_initialization_v1"
    assert signature["critic_target_source"] == "one_step"
    assert signature["critic_warmup_mode"] == "legacy"
    assert signature["warmup_critic_epochs"] == 1
    assert signature["critic_epochs"] == 1
    assert signature["actor_epochs"] == 1
    assert signature["dqc_actor_reuse_mode"] == "score_function"
    assert signature["dqc_ratio_clip"] == 0.1
    assert signature["dqc_target_kl"] == 0.0


def test_warmup_absolute_lr_advancement_emits_no_scheduler_order_warning():
    agent = object.__new__(DQCACBetaGPU)
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    agent.actor_lr_b, agent.actor_lr_c = 10000.0, 0.9
    agent.actor_optimizer = torch.optim.Adam([parameter], lr=2e-4)
    agent.actor_scheduler = agent._new_actor_scheduler()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for completed in range(1, 6):
            agent._set_actor_lr_batch_index(completed)
    assert len(agent.actor_optimizer.state) == 0
    assert agent.actor_scheduler.last_epoch == 5
    assert not any("lr_scheduler.step" in str(item.message) for item in caught)
