# -*- coding: utf-8 -*-
"""Cross-algorithm rollout-batch actor learning-rate schedule regressions."""
import math
import warnings

import torch

from agents.common import lr_lambda
from agents.dqc_ac_beta_gpu import DQCACBetaGPU
from agents.qcpo_ref import QCPORefGPU


BASE_LR = 2e-4
B = 10000.0
C = 0.9


def _schedule_fixture(agent_class):
    """
    Build only the optimizer/scheduler fields needed by the unbound agent helpers.

    Avoiding full Safety Gym construction keeps this a deterministic mathematical unit test:
    one scalar parameter is sufficient because LambdaLR changes optimizer param-group state,
    independent of model architecture.
    """
    agent = object.__new__(agent_class)
    agent.actor_lr_b = B
    agent.actor_lr_c = C
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    agent.actor_optimizer = torch.optim.Adam([parameter], lr=BASE_LR)
    agent.actor_scheduler = agent._new_actor_scheduler()
    return agent, parameter


def _qpo_absolute_lr(batch_index):
    """QPO's a/(b+k)^c form with a=b^c*actor_lr0."""
    return lr_lambda(batch_index, (B ** C) * BASE_LR, B, C)


def test_ref_and_dqc_batch_schedules_match_qpo_absolute_lr():
    """Both PPO-style agents must match QPO at k=0 and every later batch index."""
    for agent_class in (QCPORefGPU, DQCACBetaGPU):
        agent, _parameter = _schedule_fixture(agent_class)
        for k in range(200):
            actual = float(agent.actor_optimizer.param_groups[0]["lr"])
            assert math.isclose(actual, _qpo_absolute_lr(k), rel_tol=1e-12)
            # Advancing the rollout index must not create or mutate any Adam parameter state.
            assert len(agent.actor_optimizer.state) == 0
            agent._set_actor_lr_batch_index(k + 1)
            assert len(agent.actor_optimizer.state) == 0


def test_scheduler_state_resume_preserves_identical_next_batch_lr():
    """Saving optimizer+scheduler state must reproduce the exact next LR sequence."""
    for agent_class in (QCPORefGPU, DQCACBetaGPU):
        source, _source_parameter = _schedule_fixture(agent_class)
        source._set_actor_lr_batch_index(37)
        optimizer_state = source.actor_optimizer.state_dict()
        scheduler_state = source.actor_scheduler.state_dict()

        restored, _restored_parameter = _schedule_fixture(agent_class)
        restored.actor_optimizer.load_state_dict(optimizer_state)
        restored.actor_scheduler.load_state_dict(scheduler_state)
        assert restored.actor_scheduler.last_epoch == source.actor_scheduler.last_epoch
        assert restored.actor_scheduler.get_last_lr() == source.actor_scheduler.get_last_lr()

        source._set_actor_lr_batch_index(38)
        restored._set_actor_lr_batch_index(38)
        assert restored.actor_scheduler.get_last_lr() == source.actor_scheduler.get_last_lr()


def test_legacy_restore_positions_schedule_at_next_completed_batch():
    """QCPO_REF keeps its legacy upgrade; replay-era DQC checkpoints are rejected."""
    agent, _parameter = _schedule_fixture(QCPORefGPU)
    completed = 50
    agent._restore_legacy_actor_scheduler(completed)
    actual = float(agent.actor_optimizer.param_groups[0]["lr"])
    assert math.isclose(actual, _qpo_absolute_lr(completed), rel_tol=1e-12)
    assert agent.actor_scheduler.last_epoch == completed


def test_dqc_warmup_batch_advancement_has_no_scheduler_order_warning():
    """Five critic-only batches advance LR without faking an Adam update or warning."""
    agent, _parameter = _schedule_fixture(DQCACBetaGPU)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for completed_batch in range(1, 6):
            agent._set_actor_lr_batch_index(completed_batch)
    assert len(agent.actor_optimizer.state) == 0
    assert not any("lr_scheduler.step" in str(item.message) for item in caught)
    assert math.isclose(
        float(agent.actor_optimizer.param_groups[0]["lr"]),
        _qpo_absolute_lr(5),
        rel_tol=1e-12,
    )
