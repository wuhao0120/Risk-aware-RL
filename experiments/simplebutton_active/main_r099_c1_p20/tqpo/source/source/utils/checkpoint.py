# -*- coding: utf-8 -*-
"""Format-versioned, atomic training checkpoint helpers."""
import os
import random
import uuid
from pathlib import Path
import numpy as np
import torch

CHECKPOINT_FORMAT_VERSION = 1


def _sync_directory(directory):
    """Flush directory metadata after an atomic rename or rollback."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(str(directory), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unique_sibling(destination, role):
    """Return a process- and launch-unique hidden sibling on the destination filesystem."""
    token = uuid.uuid4().hex
    return destination.with_name(
        f".{destination.name}.{os.getpid()}.{token}.{role}")


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


def validate_rng_state_payload(state):
    """Validate a serialized global RNG envelope without changing global RNGs.

    A resume is transactional only when malformed Python/NumPy/Torch generator
    state is rejected before any model, optimizer, worker, or process-global
    state is loaded. Temporary generator objects exercise the same public
    ``set_state`` APIs while leaving the generators used by training untouched.
    """

    if not isinstance(state, dict):
        raise ValueError("checkpoint rng state must be a dictionary")
    required = {"python", "numpy", "torch_cpu"}
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(
            "checkpoint is missing global RNG state: " + ", ".join(missing))

    try:
        random.Random().setstate(state["python"])
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint Python RNG state is invalid") from error
    try:
        np.random.RandomState().set_state(state["numpy"])
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError("checkpoint NumPy RNG state is invalid") from error
    try:
        cpu_state = _cpu_byte_rng_state(state["torch_cpu"])
        torch.Generator(device="cpu").set_state(cpu_state)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError("checkpoint Torch CPU RNG state is invalid") from error

    cuda_state = state.get("torch_cuda")
    if torch.cuda.is_available():
        if not isinstance(cuda_state, (list, tuple)):
            raise ValueError("checkpoint is missing Torch CUDA RNG states")
        expected = int(torch.cuda.device_count())
        if len(cuda_state) != expected:
            raise ValueError(
                "checkpoint Torch CUDA RNG state count mismatch: "
                f"checkpoint={len(cuda_state)}, visible_devices={expected}")
        for index, value in enumerate(cuda_state):
            try:
                generator = torch.Generator(device=f"cuda:{index}")
                generator.set_state(_cpu_byte_rng_state(value))
            except (RuntimeError, TypeError, ValueError) as error:
                raise ValueError(
                    f"checkpoint Torch CUDA RNG state {index} is invalid") from error
    elif cuda_state is not None:
        if not isinstance(cuda_state, (list, tuple)):
            raise ValueError("checkpoint Torch CUDA RNG states must be a sequence")
        for index, value in enumerate(cuda_state):
            tensor = _cpu_byte_rng_state(value)
            if tensor.ndim != 1 or tensor.numel() == 0:
                raise ValueError(
                    f"checkpoint Torch CUDA RNG state {index} is invalid")


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


def save_checkpoint(path, agent, iteration, metrics=None, runner_state=None):
    """Durably replace one rollout-boundary checkpoint without exposing partial bytes."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _unique_sibling(destination, "tmp")
    rollback = _unique_sibling(destination, "rollback")
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "iteration": int(iteration),
        "agent_class": agent.__class__.__name__,
        "agent": agent.checkpoint_state(),
        "metrics": dict(metrics or {}),
        "rng": _rng_state(),
    }
    # Runner-owned histories must not masquerade as validation metrics.  Keep
    # this optional envelope separate from algorithm state and omit it for all
    # historical call sites so their checkpoint schema remains unchanged.
    if runner_state is not None:
        if not isinstance(runner_state, dict):
            raise TypeError("checkpoint runner_state must be a dictionary")
        payload["runner_state"] = dict(runner_state)
    signature = getattr(agent, "checkpoint_signature", None)
    if signature is not None:
        payload["experiment_signature"] = dict(signature)

    # Supplying an already-open exclusive file prevents a coincidental PID/name collision from
    # truncating another writer's staging artifact. PyTorch does not fsync a caller-owned stream;
    # flush both Python buffering and the kernel page cache before the atomic namespace change.
    replaced = False
    had_destination = destination.is_file()
    try:
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())

        # A hard link costs no second checkpoint-sized quota allocation. It exists only so a rare
        # post-replace directory-fsync error can restore the previously published checkpoint before
        # the exception escapes; ordinary quota/write/replace failures never touch destination.
        if had_destination:
            os.link(destination, rollback)
        os.replace(temporary, destination)
        replaced = True
        try:
            _sync_directory(destination.parent)
        except BaseException:
            if had_destination:
                os.replace(rollback, destination)
            else:
                destination.unlink(missing_ok=True)
            replaced = False
            # Restoration is best-effort durable even when the injected/original directory sync
            # failed. Preserve the first exception because it identifies the failed commit.
            try:
                _sync_directory(destination.parent)
            except BaseException:
                pass
            raise
    finally:
        # Only this call's UUID-scoped artifacts are eligible for cleanup. Never glob or remove a
        # shared ``*.tmp`` name: another process may legitimately be serializing beside us.
        temporary.unlink(missing_ok=True)
        rollback.unlink(missing_ok=True)

    # The replacement is already durable. Syncing removal of the rollback link prevents stale
    # recovery names after a clean save; failure here cannot invalidate the committed checkpoint.
    if replaced and had_destination:
        try:
            _sync_directory(destination.parent)
        except OSError:
            pass
    return str(destination)



def load_checkpoint_payload(payload, agent, restore_rng=True,
                            allow_signature_mismatch=False):
    """Restore one already-deserialized payload after complete prevalidation."""

    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    version = int(payload.get("format_version", -1))
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format {version}")
    # Compatibility is checked before actor/optimizer/RNG mutation, so a wrong resume path cannot
    # leave a partially overwritten live agent after raising an error.
    validate_checkpoint_payload(
        payload, agent, allow_signature_mismatch=allow_signature_mismatch)
    if restore_rng:
        validate_rng_state_payload(payload.get("rng"))
    agent.load_checkpoint_state(payload["agent"])
    if restore_rng:
        _restore_rng_state(payload["rng"])
    return payload


def load_checkpoint(path, agent, map_location=None, restore_rng=True,
                    allow_signature_mismatch=False):
    """Load a compatible checkpoint and restore complete agent/RNG states."""

    payload = torch.load(path, map_location=map_location, weights_only=False)
    return load_checkpoint_payload(
        payload,
        agent,
        restore_rng=restore_rng,
        allow_signature_mismatch=allow_signature_mismatch,
    )
