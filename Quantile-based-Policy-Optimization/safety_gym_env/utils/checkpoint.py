# -*- coding: utf-8 -*-
"""Format-versioned, atomic training checkpoint helpers."""
import os
import random
from pathlib import Path
import numpy as np
import torch

CHECKPOINT_FORMAT_VERSION = 1


def _rng_state():
    """Capture Python/NumPy/Torch RNG state, including all CUDA generators when present."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _cpu_byte_rng_state(value):
    """Normalize a torch.load-mapped RNG tensor to the CPU uint8 type generators require."""
    # ``torch.load(..., map_location=cuda)`` relocates every tensor, including the CPU RNG
    # payload. torch.set_rng_state accepts only a CPU ByteTensor, so blindly restoring the
    # relocated value fails before training resumes. The same normalization is safe for old
    # checkpoints and keeps the serialized format unchanged.
    return torch.as_tensor(value, dtype=torch.uint8, device="cpu").contiguous()


def _restore_rng_state(state):
    """Restore Python/NumPy/Torch RNG state regardless of torch.load map_location."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_cpu_byte_rng_state(state["torch_cpu"]))
    if torch.cuda.is_available() and "torch_cuda" in state:
        cuda_states = [_cpu_byte_rng_state(value) for value in state["torch_cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)


def _signature_mismatches(expected, actual):
    """Return stable, human-readable differences between two flat experiment signatures."""
    differences = []
    for key in sorted(set(expected) | set(actual)):
        expected_value = expected.get(key, "<missing>")
        actual_value = actual.get(key, "<missing>")
        if expected_value != actual_value:
            differences.append(
                f"{key}: checkpoint={actual_value!r}, requested={expected_value!r}")
    return differences


def validate_checkpoint_payload(payload, agent, allow_signature_mismatch=False):
    """
    Reject a checkpoint from a different agent or experiment before mutating live state.

    New runner checkpoints carry a flat immutable signature.  Pre-signature checkpoints require
    the explicit transfer/legacy override; this avoids silently pairing old validation metrics or
    stale algorithm state with a different environment, q, alpha, horizon, or network definition.
    ``allow_signature_mismatch`` never bypasses the Python agent-class check because loading an
    unrelated state dictionary is not a meaningful transfer operation.
    """
    saved_class = payload.get("agent_class")
    requested_class = agent.__class__.__name__
    if saved_class != requested_class:
        raise ValueError(
            f"checkpoint agent_class={saved_class!r} cannot load into {requested_class!r}")

    # Algorithm-specific state validation is stricter than an experiment-signature transfer.
    # DQC uses this hook to reject replay-era or partial state before actor/optimizer/RNG mutation,
    # even when a caller explicitly allows ordinary hyperparameter transfer.
    state_validator = getattr(agent, "validate_checkpoint_state", None)
    if state_validator is not None:
        state_validator(payload.get("agent"))

    expected = getattr(agent, "checkpoint_signature", None)
    if expected is None:
        return
    actual = payload.get("experiment_signature")
    if actual is None:
        if allow_signature_mismatch:
            return
        raise ValueError(
            "checkpoint has no experiment_signature (legacy format); pass the runner's "
            "--allow_checkpoint_transfer flag only after manually verifying env/q/alpha/horizon")
    differences = _signature_mismatches(dict(expected), dict(actual))
    if differences and not allow_signature_mismatch:
        raise ValueError(
            "checkpoint experiment signature mismatch:\n  " + "\n  ".join(differences))


def save_checkpoint(path, agent, iteration, metrics=None):
    """Atomically save only at a caller-controlled rollout boundary."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "iteration": int(iteration),
        "agent_class": agent.__class__.__name__,
        "agent": agent.checkpoint_state(),
        "metrics": dict(metrics or {}),
        "rng": _rng_state(),
    }
    signature = getattr(agent, "checkpoint_signature", None)
    if signature is not None:
        payload["experiment_signature"] = dict(signature)
    torch.save(payload, temporary)
    # os.replace is atomic on one filesystem and never exposes a partially written checkpoint.
    os.replace(temporary, destination)
    return str(destination)


def load_checkpoint(path, agent, map_location=None, restore_rng=True,
                    allow_signature_mismatch=False):
    """Load a compatible checkpoint and restore complete agent/RNG states."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    version = int(payload.get("format_version", -1))
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format {version}")
    # Compatibility is checked before actor/optimizer/RNG mutation, so a wrong resume path cannot
    # leave a partially overwritten live agent after raising an error.
    validate_checkpoint_payload(
        payload, agent, allow_signature_mismatch=allow_signature_mismatch)
    agent.load_checkpoint_state(payload["agent"])
    if restore_rng:
        _restore_rng_state(payload["rng"])
    return payload
