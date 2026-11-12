"""Public final configuration, risk boundaries and detached actor scaling."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import base_runner
base_runner.setup()
from agent import ARMS, actor_labels_for_arm
from summarize import episode_metrics


def test_only_final_dca_arm_and_one_lambda_denominator():
    assert ARMS == ('schemeb_raw_mean100_lambda_norm',)
    reward = torch.tensor([[100., 200.]], requires_grad=True)
    risk = torch.tensor([[.2, -.1]], requires_grad=True)
    multiplier = torch.tensor(2., requires_grad=True)
    labels = actor_labels_for_arm(reward, risk, multiplier, torch.ones_like(reward, dtype=torch.bool), ARMS[0])
    torch.testing.assert_close(labels['combined'], torch.tensor([[.2, 2.2/3]]))
    assert labels['combined_normalization_scale'].item() == 3.
    assert not labels['combined'].requires_grad
    assert not labels['combined_normalization_scale'].requires_grad


def test_strict_cost_boundary_and_linear_report_quantile():
    result = episode_metrics(np.arange(5.), np.array([0., 15., 15., 15., 16.]))
    assert result['violation_percent'] == 20.
    assert result['feasible']
    assert result['cost_quantile_80'] == pytest.approx(15.2)
    with pytest.raises(ValueError):
        episode_metrics([1.], [np.nan])


def test_public_budget_matches_complete_episodes_and_paper_seeds():
    config = json.loads((Path(__file__).parent / 'configs/protocol.json').read_text())
    for method, spec in config['methods'].items():
        batches = spec['main_batches'] + spec.get('warmup_batches', 0)
        assert batches * spec['batch_size'] == spec['training_steps']
        assert spec['batch_size'] == 1000 * spec['num_envs']
        assert spec['seeds'] == list(range(4 if method == 'qcpo' else 5))
    assert config['methods']['tqpo']['gamma_cost'] == .99
