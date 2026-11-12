# -*- coding: utf-8 -*-
import math
import random
from types import SimpleNamespace
import numpy as np
import torch

from agents.qcpo_ref import QCPORefGPU
from agents.dqc_ac_beta_gpu import DQCACBetaGPU
from agents.qpo_gpu import QPOGPU
from agents.qcpo_ref_model import RewardQuantileModel
from agents.dist_rl_utils_ref import (
    paper_density_weight, shifted_weibull_pdf, shifted_weibull_quantile,
    upper_weibull_fit_loss)
from utils.checkpoint import (
    save_checkpoint, load_checkpoint, _rng_state, _restore_rng_state)


def test_shifted_weibull_pdf_and_inverse_match_analytic_formula():
    location = torch.tensor([[-2.0]])
    scale = torch.tensor([[3.0]])
    shape = torch.tensor([[2.0]])
    x = torch.tensor([[1.0]])
    density = shifted_weibull_pdf(x, location, scale, shape)
    expected = (2.0 / 3.0) * math.exp(-1.0)
    assert math.isclose(float(density), expected, rel_tol=1e-6)
    assert float(shifted_weibull_pdf(
        torch.tensor([[-2.0]]), location, scale, shape)) == 0.0
    level = torch.tensor([0.2])
    quantile = shifted_weibull_quantile(
        level, location.squeeze(0), scale.squeeze(0), shape.squeeze(0))
    expected_q = -2.0 + 3.0 * (-math.log(0.8)) ** 0.5
    assert math.isclose(float(quantile), expected_q, rel_tol=1e-6)


def test_reward_quantiles_are_signed_monotone_and_include_alpha_point():
    model = RewardQuantileModel(3, hidden=(4, 4))
    info = model(torch.randn(7, 3))
    assert info.quantiles.shape == (7, 19)
    assert torch.all(info.quantiles[:, 1:] > info.quantiles[:, :-1])
    assert model.quantile_index(0.2) == 3
    assert model.quantile_index(0.8) == 15
    # The architecture does not clamp the first quantile to positive support.
    with torch.no_grad():
        model.base_quantile_head.bias.fill_(-5.0)
    assert torch.all(model(torch.zeros(2, 3)).quantiles[:, 0] < 0)


def test_upper_weibull_calibration_cannot_backprop_into_quantile_atoms():
    """Detached targets and location anchor keep Weibull fitting out of the QR-TD head."""
    torch.manual_seed(4)
    model = RewardQuantileModel(3, hidden=(4, 4))
    info = model(torch.randn(2, 3, 3))
    loss = upper_weibull_fit_loss(
        info.quantiles, model.quantile_levels,
        info.location, info.scale, info.shape, fit_min=0.75)
    model.zero_grad(set_to_none=True)
    loss.backward()

    # Weibull heads and their detached location gap receive calibration gradients, while the base
    # and monotone-increment heads remain controlled solely by quantile TD regression.
    assert model.location_gap_head.weight.grad is not None
    assert model.scale_head.weight.grad is not None
    assert model.shape_head.weight.grad is not None
    assert model.base_quantile_head.weight.grad is None
    assert model.increment_head.weight.grad is None


def test_paper_density_weight_is_bounded_and_invalid_support_is_neutral():
    """Eq.274 returns [0.5,1.5], and an impossible live target is visible but not explosive."""
    cost = torch.tensor([[0.0]])
    q_current = torch.tensor([[1.0]])
    bootstrap = torch.ones_like(cost)
    current = (torch.tensor([[0.0]]), torch.tensor([[2.0]]), torch.tensor([[1.5]]))
    next_params = (torch.tensor([[-1.0]]), torch.tensor([[0.1]]), torch.tensor([[3.0]]))
    weight, violation, clipped_log = paper_density_weight(
        cost, q_current, current, next_params, bootstrap, gamma=0.99, log_clip=0.5)
    assert 0.5 <= float(weight) <= 1.5
    assert -0.5 <= float(clipped_log) <= 0.5
    assert not bool(violation)

    # (Q-cost)/gamma lies below next location, so the paper code's log ratio falls back to zero.
    impossible_cost = torch.tensor([[10.0]])
    weight, violation, clipped_log = paper_density_weight(
        impossible_cost, q_current, current, next_params, bootstrap,
        gamma=0.99, log_clip=0.5)
    assert float(weight) == 1.0
    assert float(clipped_log) == 0.0
    assert bool(violation)


def test_qcpo_ref_uses_finite_boundary_and_negated_reward_quantile_target():
    """A time-limit truncation must target C_t=-r_t without a continuing-state bootstrap."""
    class ConstantCritic:
        def __init__(self):
            self.inputs = []

        def __call__(self, states):
            # Retain both current and successor blocks so this boundary test also locks the exact
            # finite-time/context alignment used to build the critic's one-step targets.
            self.inputs.append(states.detach().clone())
            leading = states.shape[:-1]
            zeros = torch.zeros(*leading)
            quantiles = torch.zeros(*leading, 19)
            return SimpleNamespace(
                mean_value=zeros,
                quantiles=quantiles,
                location=torch.full(leading, -1.0),
                scale=torch.ones(*leading),
                shape=torch.ones(*leading),
            )

    agent = object.__new__(QCPORefGPU)
    agent.critic = ConstantCritic()
    agent.device = torch.device("cpu")
    agent.horizon = 2
    agent.lstm_size = 2
    agent.constraint_index = 15
    agent.gamma = 0.99
    agent.gae_lambda = 0.97
    agent.density_log_clip = 0.5
    rollout = {
        "S": torch.zeros(2, 1, 1),
        "next_states": torch.zeros(2, 1, 1),
        "R": torch.tensor([[1.0], [2.0]]),
        "valid": torch.ones(2, 1, dtype=torch.bool),
        "terminated": torch.zeros(2, 1, dtype=torch.bool),
        "truncated": torch.tensor([[False], [True]]),
        # h0/c0 are the behavior actor's memory before consuming s_t. next_h/c are the old
        # episode's memory immediately after s_t, even on the time-limit boundary.
        "h0": torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]),
        "c0": torch.tensor([[[5.0, 6.0]], [[7.0, 8.0]]]),
        "next_h": torch.tensor([[[3.0, 4.0]], [[9.0, 10.0]]]),
        "next_c": torch.tensor([[[7.0, 8.0]], [[11.0, 12.0]]]),
        # The shared continuing collector offers this bootstrap; QCPO_REF must override it.
        "bootstrap_mask": torch.ones(2, 1),
    }
    processed = agent._process_rollout(rollout)
    assert torch.equal(processed["finite_bootstrap"], torch.tensor([[1.0], [0.0]]))
    assert torch.allclose(processed["quantile_target"][-1, 0], torch.full((19,), -2.0))

    # Current t=0 receives remaining=1 and h0/c0. Its continuing successor receives remaining=.5
    # and next_h/c, exactly matching the next current row. The truncated t=1 successor receives
    # remaining=0, but finite_bootstrap=0 makes its critic output irrelevant to every target.
    current_features, next_features = agent.critic.inputs
    assert torch.equal(
        current_features[0, 0],
        torch.tensor([0.0, 1.0, 1.0, 2.0, 5.0, 6.0]),
    )
    assert torch.equal(next_features[0, 0], current_features[1, 0])
    assert torch.equal(
        next_features[1, 0],
        torch.tensor([0.0, 0.0, 9.0, 10.0, 11.0, 12.0]),
    )


def test_qcpo_ref_uses_same_finite_state_history_block_as_dqc():
    """Both critics consume obs, remaining time, detached h, detached c in the same order."""
    qref = object.__new__(QCPORefGPU)
    dqc = object.__new__(DQCACBetaGPU)
    for agent in (qref, dqc):
        agent.device = torch.device("cpu")
        agent.horizon = 1000
        agent.lstm_size = 2
    dqc.dqc_update_mode = dqc.FINITE_UPDATE_MODE

    states = torch.tensor([[2.0], [3.0]], requires_grad=True)
    time = torch.tensor([0.0, 999.0])
    h = torch.tensor([[4.0, 5.0], [6.0, 7.0]], requires_grad=True)
    c = torch.tensor([[8.0, 9.0], [10.0, 11.0]], requires_grad=True)
    qref_features = qref._critic_features(states, time, h, c)
    dqc_features = dqc._critic_features(states, time, h, c)

    assert torch.equal(qref_features, dqc_features)
    assert torch.allclose(
        qref_features,
        torch.tensor([
            [2.0, 1.0, 4.0, 5.0, 8.0, 9.0],
            [3.0, 0.001, 6.0, 7.0, 10.0, 11.0],
        ]),
    )

    # Critic regression may differentiate through its observation input tensor, but actor memory is
    # rollout data: detach prevents a critic backward pass from mutating the recurrent actor.
    qref_features.sum().backward()
    assert torch.equal(states.grad, torch.ones_like(states))
    assert h.grad is None
    assert c.grad is None


class DummyAgent:
    def __init__(self):
        self.weight = torch.tensor([1.0])
    def checkpoint_state(self):
        return {"weight": self.weight.clone()}
    def load_checkpoint_state(self, state):
        self.weight.copy_(state["weight"])


def test_rng_restore_accepts_cuda_map_location_tensors():
    """A CUDA map_location must not turn the saved CPU RNG into an invalid CUDA tensor."""
    if not torch.cuda.is_available():
        return
    state = _rng_state()
    mapped = dict(state)
    mapped["torch_cpu"] = state["torch_cpu"].to("cuda")
    mapped["torch_cuda"] = [value.to("cuda") for value in state["torch_cuda"]]
    _restore_rng_state(mapped)
    assert torch.get_rng_state().device.type == "cpu"
    assert torch.get_rng_state().dtype == torch.uint8


def test_checkpoint_restores_agent_and_all_rng_streams(tmp_path):
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    agent = DummyAgent()
    path = tmp_path / "state.pt"
    save_checkpoint(path, agent, iteration=4, metrics={"mean": 1.0})
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
    agent.weight.fill_(99.0)
    # Consume additional draws, then load must rewind to the checkpoint RNG state.
    random.random(); np.random.rand(); torch.rand(())
    payload = load_checkpoint(path, agent)
    actual = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert payload["iteration"] == 4
    assert float(agent.weight) == 1.0
    assert actual == expected
    assert not (tmp_path / "state.pt.tmp").exists()


class TrainAgent:
    """Tiny Adam fixture for next-update equivalence after reload."""
    def __init__(self):
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.05)
    def update(self):
        x, target = torch.tensor([[2.0]]), torch.tensor([[1.0]])
        loss = (self.model(x) - target).square().mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss)
    def checkpoint_state(self):
        return {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}
    def load_checkpoint_state(self, state):
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])


def test_checkpoint_signature_rejects_wrong_or_legacy_experiment_before_state_mutation(tmp_path):
    """Strict runner signatures prevent silent cross-q/env resume; transfer remains explicit."""
    source = DummyAgent()
    source.checkpoint_signature = {
        "algorithm": "DQCACBeta", "env_id": "SimpleButton", "return_threshold": -0.03}
    signed_path = tmp_path / "signed.pt"
    save_checkpoint(signed_path, source, iteration=0)

    wrong = DummyAgent()
    wrong.weight.fill_(7.0)
    wrong.checkpoint_signature = {
        "algorithm": "DQCACBeta", "env_id": "SimpleButton", "return_threshold": 0.0}
    try:
        load_checkpoint(signed_path, wrong, restore_rng=False)
    except ValueError as error:
        assert "return_threshold" in str(error)
    else:
        raise AssertionError("mismatched q must fail without explicit transfer")
    # Compatibility is checked before load_checkpoint_state, so failure leaves live state intact.
    assert float(wrong.weight) == 7.0
    load_checkpoint(
        signed_path, wrong, restore_rng=False, allow_signature_mismatch=True)
    assert float(wrong.weight) == 1.0

    legacy_source = DummyAgent()
    legacy_path = tmp_path / "legacy.pt"
    save_checkpoint(legacy_path, legacy_source, iteration=0)
    strict_target = DummyAgent()
    strict_target.checkpoint_signature = dict(source.checkpoint_signature)
    try:
        load_checkpoint(legacy_path, strict_target, restore_rng=False)
    except ValueError as error:
        assert "legacy" in str(error)
    else:
        raise AssertionError("pre-signature checkpoint must require an explicit override")
    load_checkpoint(
        legacy_path, strict_target, restore_rng=False, allow_signature_mismatch=True)


def test_qcpo_ref_v1_checkpoint_rejected_before_actor_mutation_even_with_transfer(tmp_path):
    """The algorithm-level v2 identity cannot be bypassed by generic signature transfer."""
    target = object.__new__(QCPORefGPU)
    target.training_budget_semantics = "main_excludes_initialization_v1"
    target.qcpo_ref_formulation = "paper_negated_return_eq275_discounted_bh_v2"
    target.qcpo_ref_actor_weighting = "discounted_gamma_t"
    target.qcpo_ref_actor_reduction = "fixed_BH"
    target.qcpo_ref_critic_input = "finite_obs_remaining_actor_h0c0_v1"
    target.horizon_mode = "finite"
    target.critic_time_feature = "remaining_fraction"
    target.critic_actor_context = True
    target.critic_state_dim = 6
    target.actor = torch.nn.Linear(1, 1, bias=False)
    actor_before = target.actor.weight.detach().clone()

    payload = {
        "format_version": 1,
        "iteration": 7,
        "agent_class": "QCPORefGPU",
        "agent": {
            "training_budget_semantics": target.training_budget_semantics,
            "main_env_steps": 0,
            "initialization_env_steps": 0,
            "actual_env_steps": 0,
            "total_env_steps": 0,
            "initialization_cursor": 0,
            "qcpo_ref_formulation": "paper_negated_return_eq275_v1",
            "actor": {"weight": torch.full_like(actor_before, 99.0)},
        },
        "rng": _rng_state(),
    }
    checkpoint = tmp_path / "qcpo_ref_v1.pt"
    torch.save(payload, checkpoint)
    try:
        load_checkpoint(
            checkpoint, target, restore_rng=False,
            allow_signature_mismatch=True)
    except ValueError as error:
        assert "qcpo_ref_formulation mismatch" in str(error)
    else:
        raise AssertionError("uniform-time QCPO_REF v1 checkpoint must be rejected")
    assert torch.equal(target.actor.weight, actor_before)


def test_qcpo_ref_observation_only_v2_checkpoint_rejected_before_actor_mutation(tmp_path):
    """The prior discounted actor lineage cannot enter the new history-conditioned critic."""
    target = object.__new__(QCPORefGPU)
    target.training_budget_semantics = "main_excludes_initialization_v1"
    target.qcpo_ref_formulation = "paper_negated_return_eq275_discounted_bh_v2"
    target.qcpo_ref_actor_weighting = "discounted_gamma_t"
    target.qcpo_ref_actor_reduction = "fixed_BH"
    target.qcpo_ref_critic_input = "finite_obs_remaining_actor_h0c0_v1"
    target.horizon_mode = "finite"
    target.critic_time_feature = "remaining_fraction"
    target.critic_actor_context = True
    target.critic_state_dim = 6
    target.actor = torch.nn.Linear(1, 1, bias=False)
    actor_before = target.actor.weight.detach().clone()

    # This payload models the completed q=2.3 QCPO_REF: its actor objective is already v2, while
    # the critic saw observation alone. Generic checkpoint-transfer permission must not bypass this
    # algorithm-level identity gate or load the deliberately poisoned actor tensor below.
    payload = {
        "format_version": 1,
        "iteration": 399,
        "agent_class": "QCPORefGPU",
        "agent": {
            "training_budget_semantics": target.training_budget_semantics,
            "main_env_steps": 10_000_000,
            "initialization_env_steps": 0,
            "actual_env_steps": 10_000_000,
            "total_env_steps": 10_000_000,
            "initialization_cursor": 0,
            "qcpo_ref_formulation": target.qcpo_ref_formulation,
            "qcpo_ref_actor_weighting": target.qcpo_ref_actor_weighting,
            "qcpo_ref_actor_reduction": target.qcpo_ref_actor_reduction,
            "qcpo_ref_critic_input": "observation_only_v1",
            "actor": {"weight": torch.full_like(actor_before, 99.0)},
        },
        "rng": _rng_state(),
    }
    checkpoint = tmp_path / "qcpo_ref_observation_only_v2.pt"
    torch.save(payload, checkpoint)
    try:
        load_checkpoint(
            checkpoint, target, restore_rng=False,
            allow_signature_mismatch=True)
    except ValueError as error:
        assert "qcpo_ref_critic_input mismatch" in str(error)
    else:
        raise AssertionError("observation-only QCPO_REF v2 checkpoint must be rejected")
    assert torch.equal(target.actor.weight, actor_before)


def test_qpo_interrupted_initialization_resume_matches_uninterrupted_rng_and_q(tmp_path):
    """A cursor-1 checkpoint reuses its saved samples and resumes from the next RNG draw."""
    def build_agent():
        agent = object.__new__(QPOGPU)
        agent.actor = torch.nn.Linear(1, 1, bias=False)
        agent.optimizer = torch.optim.Adam(agent.actor.parameters(), lr=0.01)
        agent.scheduler = torch.optim.lr_scheduler.LambdaLR(
            agent.optimizer, lr_lambda=lambda _k: 1.0)
        agent.training_budget_semantics = "main_excludes_initialization_v1"
        agent.main_env_steps = 0
        agent.initialization_env_steps = 0
        agent.actual_env_steps = 0
        agent.total_env_steps = 0
        agent.initialization_cursor = 0
        agent.initialization_returns = []
        agent.warmup_q_iters = 2
        agent.failure_alpha = 0.5
        agent.q_est = None
        agent.q_a = agent.q_b = agent.q_c = 1.0
        agent.update_count = 0
        agent.device = torch.device("cpu")
        agent.algo_name, agent.env_name, agent.seed = "QPO", "SimpleButton", 0
        agent.start_iteration = 0
        agent.show_progress = False
        agent.initialization_callback = None
        agent._write_external_progress = lambda **_kwargs: None
        agent.console_write = lambda _message: None

        def rollout(phase="main", **_kwargs):
            assert phase == "initialization"
            values = torch.rand(2)
            agent.initialization_env_steps += 2
            agent.actual_env_steps = (
                agent.main_env_steps + agent.initialization_env_steps)
            agent.total_env_steps = agent.actual_env_steps
            return {"disc_return": values}

        agent._rollout_core = rollout
        return agent

    torch.manual_seed(37)
    uninterrupted = build_agent()
    expected_q = uninterrupted._initialize_quantile()
    expected_next_rng = torch.rand(4)

    torch.manual_seed(37)
    interrupted = build_agent()
    initialization_checkpoint = tmp_path / "initialization_latest.pt"

    def save_then_interrupt(current_agent, completed, _total):
        save_checkpoint(
            initialization_checkpoint, current_agent, iteration=-1, metrics=None)
        if completed == 1:
            raise RuntimeError("simulated interruption")

    interrupted.initialization_callback = save_then_interrupt
    try:
        interrupted._initialize_quantile()
    except RuntimeError as error:
        assert "simulated interruption" in str(error)
    else:
        raise AssertionError("fixture must interrupt after the first initialization batch")

    resumed = build_agent()
    payload = load_checkpoint(initialization_checkpoint, resumed)
    assert payload["iteration"] == -1
    assert resumed.initialization_cursor == 1
    assert len(resumed.initialization_returns) == 1
    actual_q = resumed._initialize_quantile()
    actual_next_rng = torch.rand(4)

    assert actual_q == expected_q
    assert torch.equal(actual_next_rng, expected_next_rng)
    assert resumed.main_env_steps == 0
    assert resumed.initialization_env_steps == 4
    assert resumed.actual_env_steps == 4


def test_checkpoint_reload_produces_identical_next_optimizer_update(tmp_path):
    torch.manual_seed(5)
    uninterrupted = TrainAgent()
    uninterrupted.update()
    path = tmp_path / "adam.pt"
    save_checkpoint(path, uninterrupted, iteration=0)
    expected_loss = uninterrupted.update()
    expected_weight = uninterrupted.model.weight.detach().clone()
    restored = TrainAgent()
    load_checkpoint(path, restored, restore_rng=False)
    actual_loss = restored.update()
    assert math.isclose(actual_loss, expected_loss, rel_tol=1e-7, abs_tol=1e-7)
    assert torch.allclose(restored.model.weight, expected_weight, atol=1e-7)
