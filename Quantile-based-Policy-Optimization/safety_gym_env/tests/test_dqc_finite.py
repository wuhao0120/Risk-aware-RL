# -*- coding: utf-8 -*-
"""Finite-horizon n-step target, context, CDF, warmup, and identity tests."""
import copy
import math

import torch

from agents.dqc_ac_beta_gpu import DQCACBetaGPU
from run_experiment import (
    base_args,
    checkpoint_signature,
    configure_dqc_identity,
)
from utils.model import RecurrentGaussianActor


class _FixedEndpointTarget:
    """Record finite endpoint features and return transparent two-atom tails."""

    def __init__(self):
        self.features = None
        self.actions = None

    def __call__(self, features, actions):
        self.features = features.detach().clone()
        self.actions = actions.detach().clone()
        atoms = torch.tensor([10.0, 20.0], dtype=features.dtype)
        return atoms.view(1, 2).expand(features.shape[0], 2)


def _finite_target_agent():
    """Build only fields needed by the finite target/input helpers."""
    agent = object.__new__(DQCACBetaGPU)
    agent.dqc_update_mode = agent.FINITE_UPDATE_MODE
    agent.device = torch.device("cpu")
    agent.gamma = 0.9
    agent.return_threshold = 0.0
    agent.horizon = 1000
    agent.critic_n_step = 100
    agent.num_quantiles = 2
    agent.lstm_size = 2
    agent.reward_target_critic = _FixedEndpointTarget()
    return agent


def _finite_boundary_rollout():
    """
    Build one H-truncated slot and one early-terminal slot.

    Slot 0 has 103 valid rows and truncates at t=102. Slot 1 terminates at t=2; the later reward
    value 999 is deliberately invalid and must never enter any old-episode target.
    """
    T, B, hidden = 103, 2, 2
    time = torch.arange(T, dtype=torch.float32).view(T, 1)
    states = time.view(T, 1, 1).expand(T, B, 1).clone()
    actions = torch.zeros(T, B, 1)

    # Actor h0/c0 are unique at every endpoint, making a wrong start-context substitution visible.
    h0 = torch.empty(T, B, hidden)
    c0 = torch.empty(T, B, hidden)
    for timestep in range(T):
        for slot in range(B):
            base = float(timestep + 10 * slot)
            h0[timestep, slot] = torch.tensor([base, base + 0.5])
            c0[timestep, slot] = torch.tensor([base + 1000.0, base + 1000.5])

    valid = torch.zeros(T, B, dtype=torch.bool)
    valid[:, 0] = True
    valid[:3, 1] = True
    terminated = torch.zeros_like(valid)
    terminated[2, 1] = True
    truncated = torch.zeros_like(valid)
    truncated[102, 0] = True

    rewards = torch.ones(T, B)
    rewards[:3, 1] = torch.tensor([1.0, 2.0, 3.0])
    rewards[3:, 1] = 999.0
    return {
        "S": states,
        "A_exec": actions,
        "R": rewards,
        "valid": valid,
        "terminated": terminated,
        "truncated": truncated,
        "h0": h0,
        "c0": c0,
    }


def _target_by_transition_id(batch, transition_id):
    """Select one flattened [t,b] target without depending on compact valid-row offsets."""
    selected = batch["transition_ids"] == int(transition_id)
    assert int(selected.sum()) == 1
    return batch["targets"][selected][0]


def test_finite_100step_target_and_both_boundaries_match_manual_values():
    """Interior t=0 bootstraps at 100; terminal and short horizon suffixes have zero tail."""
    agent = _finite_target_agent()
    rollout = _finite_boundary_rollout()
    batch = agent.build_n_step_batch(rollout)
    gamma = agent.gamma

    # Slot 0, t=0: exactly 100 rewards followed by the two frozen endpoint atoms at t=100.
    reward_100 = sum(gamma ** k for k in range(100))
    expected_t0 = torch.tensor([
        reward_100 + (gamma ** 100) * 10.0,
        reward_100 + (gamma ** 100) * 20.0,
    ])
    assert torch.allclose(
        _target_by_transition_id(batch, 0), expected_t0, atol=2e-6)

    # Slot 0, t=3 has exactly 100 observed rewards through the t=102 horizon truncation.  Its
    # target is a repeated scalar and cannot include any target atom beyond H.
    expected_suffix = torch.full((2,), reward_100)
    assert torch.allclose(
        _target_by_transition_id(batch, 3 * 2), expected_suffix, atol=2e-6)

    # Slot 1 terminates at t=2. The terminal reward is included once, while invalid 999 rewards
    # and all target atoms are excluded. Equality of both atoms proves the tail is exactly zero.
    terminal_return = 1.0 + gamma * 2.0 + (gamma ** 2) * 3.0
    assert torch.allclose(
        _target_by_transition_id(batch, 1),
        torch.full((2,), terminal_return),
    )
    assert torch.equal(
        _target_by_transition_id(batch, 2 * 2 + 1),
        torch.tensor([3.0, 3.0]),
    )

    # The final horizon row itself has only r_102. A reset observation/action is never queried.
    assert torch.equal(
        _target_by_transition_id(batch, 102 * 2),
        torch.tensor([1.0, 1.0]),
    )


def test_remaining_fraction_and_endpoint_context_use_t_plus_n_not_start():
    """Time is 1 at t=0/.001 at t=999, and bootstrap rows carry endpoint h0/c0."""
    agent = _finite_target_agent()
    assert torch.equal(
        agent.remaining_fraction(torch.tensor([0.0, 999.0])),
        torch.tensor([1.0, 0.001]),
    )

    rollout = _finite_boundary_rollout()
    batch = agent.build_n_step_batch(rollout)
    recorded = agent.reward_target_critic.features

    # Only slot 0 starts t=0,1,2 have valid endpoints t=100,101,102. The first recorded feature is
    # [s_100, remaining=.9, h0_100(two), c0_100(two)] in that exact concatenation order.
    assert recorded.shape == (3, 6)
    expected = torch.tensor([100.0, 0.9, 100.0, 100.5, 1100.0, 1100.5])
    assert torch.allclose(recorded[0], expected)

    # The online-regression input for transition t=0 still uses t=0 and its own zero-time context.
    first_feature = batch["states"][batch["transition_ids"] == 0][0]
    assert torch.allclose(
        first_feature,
        torch.tensor([0.0, 1.0, 0.0, 0.5, 1000.0, 1000.5]),
    )


def test_same_observation_action_different_actor_context_forms_different_input():
    """Actor h/c are detached data: they distinguish critic rows but receive no gradient."""
    agent = _finite_target_agent()
    states = torch.zeros(2, 1, requires_grad=False)
    time = torch.tensor([5.0, 5.0])
    h = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    c = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    features = agent._critic_features(states, time, h, c)

    assert not torch.equal(features[0], features[1])
    assert torch.equal(features[:, 0], torch.zeros(2))
    assert not features.requires_grad


def test_float64_global_cdf_matches_high_precision_direct_budget():
    """Global C_t+gamma^t*z<=q equals the algebraic direct-budget event in float64."""
    gamma, threshold = 0.99, -0.03
    atoms = torch.tensor([
        [-0.3, -0.1, 0.0, 0.2],
        [-0.2, 0.0, 0.1, 0.4],
        [-1.0, -0.5, 0.5, 1.0],
    ], dtype=torch.float64)
    prefix = torch.tensor([-0.02, -0.031, 0.2], dtype=torch.float64)
    time = torch.tensor([0.0, 500.0, 999.0], dtype=torch.float64)

    actual = DQCACBetaGPU.finite_global_cdf(
        atoms, prefix, time, gamma, threshold)
    discount = torch.pow(torch.tensor(gamma, dtype=torch.float64), time)
    budget = (threshold - prefix) / discount
    expected = (atoms <= budget.unsqueeze(1)).float().mean(dim=1)
    assert torch.equal(actual, expected)


class _RiskCritic:
    """Negative executed action has failure CDF one; positive baseline action has CDF zero."""

    def __call__(self, _features, actions):
        low = torch.where(actions[..., 0] < 0, -2.0, 1.0)
        high = torch.where(actions[..., 0] < 0, -1.0, 2.0)
        return torch.stack((low, high), dim=-1)


class _PositiveActor:
    """Return a deterministic positive action-marginal sample at every history."""

    def __call__(self, states):
        shape = (*states.shape[:-1], 1)
        return torch.ones(shape), torch.full(shape, -30.0), None


def test_finite_risk_term_has_no_beta_time_discount_and_matches_qcpo_population_gradient():
    """Finite risk credit stays constant over t; oracle CDF decomposition equals trajectory MC."""
    agent = _finite_target_agent()
    agent.beta = 0.1                         # stale value must be ignored by finite actor semantics
    agent.lambda_probability = torch.tensor(2.0)
    agent.num_action_samples = 2
    agent._action_low = torch.tensor([-1.0])
    agent._action_high = torch.tensor([1.0])
    agent.reward_critic = _RiskCritic()
    agent.actor = _PositiveActor()
    T = 3
    rollout = {
        "S": torch.zeros(T, 1, 1),
        "A_exec": -torch.ones(T, 1, 1),
        "h0": torch.zeros(T, 1, 2),
        "c0": torch.zeros(T, 1, 2),
        "reward_prefix": torch.zeros(T, 1, dtype=torch.float64),
        "budgets": torch.zeros(T, 1),
    }
    terms = agent.actor_advantages(rollout)
    assert torch.equal(terms["cdf_advantage"], torch.ones(T, 1))
    assert torch.equal(terms["risk_term"], torch.full((T, 1), 2.0))

    # Two-step Bernoulli finite MDP: failure iff both actions are zero. The trajectory QCPO score
    # gradient E[I_failure * sum_t score_t] must equal the sum of exact conditional-CDF advantages.
    probability_one = 0.4
    actions = (0, 1)
    action_probability = {
        0: 1.0 - probability_one,
        1: probability_one,
    }
    score = {
        0: -probability_one,
        1: 1.0 - probability_one,
    }
    qcpo_gradient = 0.0
    dqc_gradient = 0.0
    failure_baseline = (1.0 - probability_one) ** 2
    for first in actions:
        first_probability = action_probability[first]
        first_cdf = (1.0 - probability_one) if first == 0 else 0.0
        dqc_gradient += (
            first_probability
            * (first_cdf - failure_baseline)
            * score[first]
        )
        for second in actions:
            trajectory_probability = first_probability * action_probability[second]
            failure = float(first == 0 and second == 0)
            qcpo_gradient += (
                trajectory_probability * failure * (score[first] + score[second]))
            second_baseline = (1.0 - probability_one) if first == 0 else 0.0
            second_cdf = failure
            dqc_gradient += (
                trajectory_probability
                * (second_cdf - second_baseline)
                * score[second]
            )
    assert math.isclose(dqc_gradient, qcpo_gradient, rel_tol=0.0, abs_tol=1e-12)


def _nested_equal(actual, expected):
    """Recursively compare module/optimizer/scheduler state without tensor truth ambiguity."""
    assert type(actual) is type(expected)
    if torch.is_tensor(actual):
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _nested_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _nested_equal(left, right)
    else:
        assert actual == expected


def test_ten_finite_initialization_batches_freeze_policy_then_main_batch1_updates():
    """Warmup stays E4/actor-frozen; the active c8a8 batch reuses data without nested loops."""
    torch.manual_seed(37)
    agent = object.__new__(DQCACBetaGPU)
    agent.dqc_update_mode = agent.FINITE_UPDATE_MODE
    agent.actor = RecurrentGaussianActor(
        1, 1, hidden=(2, 2), lstm_size=2,
        init_std=0.5, normalize_observation=True)
    agent.actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=2e-4)
    critic_parameter = torch.nn.Parameter(torch.tensor(0.0))
    agent.critic_optimizer = torch.optim.Adam([critic_parameter], lr=3e-4)
    agent.actor_lr_b, agent.actor_lr_c = 10000.0, 0.9
    agent.actor_scheduler = agent._new_actor_scheduler()
    agent.actor_grad_clip = 10.0
    agent.actor_optimizer_steps = 0
    agent.device = torch.device("cpu")
    agent._log_2pi = math.log(2.0 * math.pi)
    agent.gamma, agent.beta = 0.99, 1.0
    agent.return_threshold, agent.failure_alpha = -0.03, 0.2
    agent.warmup_iters = 10
    agent.critic_warmup_mode = "fitted_hard"
    agent.warmup_critic_epochs = 4
    agent.critic_epochs = 8
    agent.actor_epochs = 8
    agent.target_update_mode = "hard"
    agent.critic_n_step = 100
    agent.lambda_probability = torch.tensor(0.0)
    agent.lambda_max = 50.0
    agent.lambda_a, agent.lambda_b, agent.lambda_c = 1.0, 1.0, 0.0
    agent.dual_update_count = 0
    agent.last_dual_prob = 0.0
    agent.num_envs, agent.horizon = 1, 2

    critic_calls = []
    agent.target_update_count = 0

    def critic_update(_rollout, **kwargs):
        critic_calls.append(dict(kwargs))
        agent.target_update_count += 1
        return {
            "critic/reward_quantile_loss": 0.0,
            "diagnostic/dqc/critic_cdf_mean": 0.0,
            "diagnostic/dqc/critic_optimizer_steps": 0.0,
            "diagnostic/dqc/transition_coverage": 1.0,
            "diagnostic/reward_critic/atom_crossing_row_fraction": 0.0,
            "diagnostic/reward_critic/atom_crossing_adjacent_fraction": 0.0,
        }

    agent.update_critic = critic_update
    agent.actor_advantages = lambda _rollout: (_ for _ in ()).throw(
        AssertionError("finite warmup must not query actor baselines"))
    agent._log_core = (
        lambda iteration, returns, diagnostic_disc_cost, diagnostic_raw_cost, extra: extra)
    rollout = {
        "S": torch.tensor([[[1.0]], [[2.0]]]),
        "A_raw": torch.tensor([[[0.8]], [[-0.6]]]),
        "A_exec": torch.tensor([[[0.8]], [[-0.6]]]),
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
    for iteration in range(10):
        agent._update_rollout(
            -1, copy.deepcopy(rollout), initialization=True)

    # Ten batches each fit four epochs but request one hard Bellman-generation close.
    assert critic_calls == [
        {"regression_epochs": 4, "target_update_mode": "hard"}
        for _ in range(10)
    ]
    assert agent.target_update_count == 10
    _nested_equal(agent.actor.state_dict(), actor_before)
    _nested_equal(agent.actor_optimizer.state_dict(), adam_before)
    _nested_equal(agent.actor_scheduler.state_dict(), scheduler_before)
    assert torch.equal(agent.lambda_probability, lambda_before)
    assert agent.actor_optimizer_steps == 0
    assert agent.dual_update_count == 0

    # Formal batch index 0 now follows ten excluded initialization batches. It fits E8, applies
    # eight actor Adam steps, and advances the main scheduler exactly once.
    agent.actor_advantages = lambda _rollout: {
        "combined": torch.ones(2, 1),
    }
    agent._advantage_diagnostics = lambda _rollout, _terms: {}
    actor_lrs = []
    one_actor_step = agent.update_actor

    def tracked_actor_step(_rollout, _combined):
        actor_lrs.append(float(agent.actor_optimizer.param_groups[0]["lr"]))
        return one_actor_step(_rollout, _combined)

    agent.update_actor = tracked_actor_step
    agent._update_rollout(0, copy.deepcopy(rollout))
    assert critic_calls[-1] == {
        "regression_epochs": 8,
        "target_update_mode": "hard",
    }
    assert agent.target_update_count == 11
    assert agent.actor_optimizer_steps == 8
    assert len(actor_lrs) == 8
    assert len(set(actor_lrs)) == 1
    assert actor_lrs[0] == 2e-4
    assert agent.dual_update_count == 1
    assert len(agent.actor_optimizer.state) > 0
    assert agent.actor_scheduler.last_epoch == 1
    assert float(agent.actor.obs_rms.count) > float(actor_before["obs_rms.count"])


def test_finite_preset_signature_config_and_cross_mode_resume_fail_fast():
    """All new fields enter config/signature and an online state cannot load as finite."""
    args = base_args("DQCACBeta", 0, torch.device("cpu"), "SimpleButton")
    args.dqc_update_mode = "finite_nstep_v1"
    args.critic_epochs = 8
    args.actor_epochs = 8
    args.dqc_actor_reuse_mode = "ppo_clip"
    args.dqc_ratio_clip = 0.1
    args.dqc_target_kl = 0.01
    args.wandb_group = "dqc_c8a8"
    args.wandb_name = "dqc_c8a8_s0"
    configure_dqc_identity(args, {
        "dqc_update_mode": "finite_nstep_v1",
        "critic_epochs": 8,
        "actor_epochs": 8,
        "dqc_actor_reuse_mode": "ppo_clip",
        "dqc_ratio_clip": 0.1,
        "dqc_target_kl": 0.01,
        "wandb_group": "dqc_c8a8",
        "wandb_name": "dqc_c8a8_s0",
    })
    signature = checkpoint_signature(args, "DQCACBeta")
    expected = {
        "horizon_mode": "finite",
        "critic_target_source": "n_step",
        "critic_n_step": 100,
        "critic_time_feature": "remaining_fraction",
        "critic_actor_context": True,
        "target_update_mode": "hard",
        "beta": 1.0,
    }
    for name, value in expected.items():
        assert getattr(args, name) == value
        assert signature[name] == value
    assert signature["critic_epochs"] == 8
    assert signature["actor_epochs"] == 8
    assert signature["dqc_actor_reuse_mode"] == "ppo_clip"
    assert signature["dqc_ratio_clip"] == 0.1
    assert signature["dqc_target_kl"] == 0.01
    assert args.wandb_group == "dqc_c8a8"
    assert args.wandb_name == "dqc_c8a8_s0"

    finite = object.__new__(DQCACBetaGPU)
    finite.dqc_update_mode = finite.FINITE_UPDATE_MODE
    finite.critic_target_source = "n_step"
    finite.critic_warmup_mode = "fitted_hard"
    finite.warmup_critic_epochs = 4
    finite.training_budget_semantics = "main_excludes_initialization_v1"
    online_state = {
        "training_budget_semantics": finite.training_budget_semantics,
        "main_env_steps": 0,
        "initialization_env_steps": 0,
        "actual_env_steps": 0,
        "total_env_steps": 0,
        "initialization_cursor": 0,
        "dqc_update_mode": finite.UPDATE_MODE,
    }
    try:
        finite.validate_checkpoint_state(online_state)
    except ValueError as error:
        assert "dqc_update_mode mismatch" in str(error)
    else:
        raise AssertionError("continuing checkpoint must not load into finite_nstep_v1")
