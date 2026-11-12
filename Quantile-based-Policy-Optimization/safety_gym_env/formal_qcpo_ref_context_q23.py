#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fresh q=2.30 QCPO_REF rerun with DQC finite's time/history critic context.

Only the critic input changes relative to the completed q=2.30 QCPO_REF
baseline. Training, batch-200 freezing, exact-N=1000 CRN evaluation, and
checkpoint validation remain delegated to formal_qcpo_ref_dqc_gae.py.
"""
import argparse  # Expose explicit fresh-run, recovery, evaluation, and verification actions.
from contextlib import contextmanager  # Restore the reusable harness globals after every action.
import json  # Validate the pinned baseline and read atomically written result artifacts.
import math  # Reject NaN/Inf exact-evaluation metrics before tracked publication.
import os  # Force the post-training W&B synchronization command into online mode.
from pathlib import Path  # Anchor every artifact path independently of the caller's cwd.
import subprocess  # Run the explicit W&B sync gate after the online training process exits.
import sys  # Reuse the active conda interpreter for both training and W&B synchronization.

import torch  # Select the exact device used for checkpoint reconstruction and evaluation.

import formal_qcpo_ref_dqc_gae as core  # Reuse the tested formal lifecycle without copying it.


# ============================================================================ frozen lineage identity
BASE_DIR = Path(__file__).resolve().parent  # Keep all paths inside safety_gym_env.
CAMPAIGN = "binding_q2p3_qref_dqc_context_seed0_v1"
RAW_ROOT = BASE_DIR / "_runs" / CAMPAIGN  # Store large checkpoints/logs outside tracked results.
TRACKED_MANIFEST = BASE_DIR / "campaigns" / CAMPAIGN / "manifest.json"
RETURN_THRESHOLD = 2.30  # Preserve the already selected binding threshold exactly.

# The old three-way result is immutable comparison evidence. Pinning its bytes prevents a later
# edit from silently changing the baseline after this new policy has already trained.
BASELINE_MANIFEST = (
    BASE_DIR / "campaigns" / "binding_q2p3_simplebutton_seed0_v1" / "manifest.json")
BASELINE_MANIFEST_SHA256 = (
    "d412889140d6ee8b01c6244818ca09fe049fac8856c95b7135c670d1d7ecc672")
BASELINE_CAMPAIGN = "binding_q2p3_simplebutton_seed0_v1"

WANDB_ID = "qrefctx23s0"  # Stable service identity; never reuse the observation-only run ID.
WANDB_GROUP = BASELINE_CAMPAIGN  # Put the new curve beside the completed q=2.3 comparison.
WANDB_SYNC_MARKER = RAW_ROOT / "wandb_sync.json"  # Publish only after explicit sync succeeds.

# QCPO_REF keeps its paper-negated-return critic and actor math. The identity suffix means only
# that each state-value/distribution query now sees remaining time and detached behavior h0/c0.
SPEC = core.FormalSpec(
    key=WANDB_ID,
    algorithm="QCPO_REF",
    wandb_id=WANDB_ID,
    wandb_group=WANDB_GROUP,
    wandb_tag="qcpo-ref-dqc-context-v1-q2p3-lr2e4-main10m-s0",
    tmux_name="binding_q23_qref_dqc_context_s0",
    parent_policy_identity="paper_negated_return_eq275_discounted_bh_v2",
    policy_identity=(
        "paper_negated_return_eq275_discounted_bh_v2_dqc_finite_context_v1"),
    variant_changes=(
        "critic_input_obs_to_obs_remaining_fraction_detached_actor_h0c0;"
        "fresh_seed0_training;"
        "all_actor_gae_quantile_weibull_dual_budget_fields_unchanged"),
    initialization_batches=0,
    wandb_name="QCPO_REF-DQC-Context-Critic-q2.30-lr2e4-s0",
)
SPECS = {"qref_context": SPEC}  # Install the one lineage into the generic evaluation harness.


@contextmanager
def installed_core_identity():
    """Install this lineage for one core operation, then restore every prior global."""
    names = (
        "CAMPAIGN", "RETURN_THRESHOLD", "RAW_ROOT", "TRACKED_MANIFEST", "SPECS")
    previous = tuple(getattr(core, name) for name in names)
    replacements = (
        CAMPAIGN, RETURN_THRESHOLD, RAW_ROOT, TRACKED_MANIFEST, SPECS)
    for name, value in zip(names, replacements):
        setattr(core, name, value)
    try:
        yield
    finally:
        for name, value in zip(names, previous):
            setattr(core, name, value)


def _validate_registered_identity():
    """Reject an accidental collision or service-invalid human-readable identity."""
    if SPEC.wandb_id in {"qref23s0", "dqcg23s0", "qcpo23s0"}:
        raise RuntimeError("context rerun must not reuse a completed q=2.3 W&B ID")
    if SPEC.wandb_group != BASELINE_CAMPAIGN:
        raise RuntimeError("context rerun must remain in the completed comparison group")
    if not SPEC.wandb_name:
        raise RuntimeError("context rerun requires a readable W&B display name")
    if len(SPEC.wandb_tag) > 64:
        raise RuntimeError("context rerun W&B tag exceeds the service limit")


def local_wandb_run_dirs():
    """Return every local online/resume directory owned by the stable new W&B ID."""
    wandb_root = BASE_DIR / "wandb"
    if not wandb_root.exists():
        return []
    suffix = "-" + SPEC.wandb_id
    return sorted(
        path for path in wandb_root.iterdir()
        if path.is_dir()
        and path.name.endswith(suffix)
        and (
            path.name.startswith("run-")
            or path.name.startswith("offline-run-")
        )
    )


def sync_wandb_online():
    """
    Explicitly synchronize all local segments after online training completes.

    Online mode normally streams each row immediately, but this additional command verifies and
    uploads any files retained during a transient network outage. A durable marker is written only
    after the CLI returns zero; exact evaluation and manifest publication are gated on that marker.
    """
    run_dirs = local_wandb_run_dirs()
    if not run_dirs:
        raise RuntimeError(
            f"{SPEC.wandb_id}: no local W&B run directory exists to synchronize")
    command = [
        sys.executable, "-m", "wandb", "sync", "--include-online",
        *(str(path) for path in run_dirs),
    ]
    environment = os.environ.copy()
    environment["WANDB_MODE"] = "online"
    completed = subprocess.run(
        command, cwd=BASE_DIR, env=environment, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{SPEC.wandb_id}: explicit W&B sync failed with "
            f"exit code {completed.returncode}")

    marker = {
        "synced": True,
        "wandb_id": SPEC.wandb_id,
        "wandb_mode": "online",
        "git_sha": core.git_sha(),
        "local_run_dirs": [str(path) for path in run_dirs],
        "command": command,
    }
    core.write_json_atomic(WANDB_SYNC_MARKER, marker)
    return marker


def read_valid_sync_marker():
    """Return the same-code online sync marker, or None when the publication gate is not met."""
    try:
        marker = json.loads(WANDB_SYNC_MARKER.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return None
    required = (
        marker.get("synced") is True
        and marker.get("wandb_id") == SPEC.wandb_id
        and marker.get("wandb_mode") == "online"
        and marker.get("git_sha") == core.git_sha()
    )
    return marker if required else None


def load_pinned_baseline_manifest():
    """Hash-check and parse the completed three-way q=2.3 comparison."""
    if not BASELINE_MANIFEST.exists():
        raise RuntimeError(f"pinned baseline manifest is missing: {BASELINE_MANIFEST}")
    actual_hash = core.sha256_file(BASELINE_MANIFEST)
    if actual_hash != BASELINE_MANIFEST_SHA256:
        raise RuntimeError(
            "pinned baseline manifest hash mismatch: "
            f"expected={BASELINE_MANIFEST_SHA256}, actual={actual_hash}")
    baseline = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    if baseline.get("campaign") != BASELINE_CAMPAIGN:
        raise RuntimeError("pinned baseline manifest has the wrong campaign identity")
    if set((baseline.get("results") or {})) != {"dqc", "qref", "qcpo"}:
        raise RuntimeError("pinned baseline manifest is missing a comparison lineage")
    return baseline


def _existing_lineage_artifacts():
    """Resolve only new-lineage artifacts whose presence makes a launch non-fresh."""
    candidates = (
        SPEC.checkpoint_root,
        SPEC.progress_path,
        SPEC.log_path,
        SPEC.raw_result_path,
        SPEC.runner_result_path,
        WANDB_SYNC_MARKER,
        TRACKED_MANIFEST,
    )
    return [
        path for path in (*candidates, *local_wandb_run_dirs())
        if Path(path).exists()
    ]


def assert_fresh_lineage():
    """Fail before process creation if any prior segment owns the new stable identity."""
    existing = _existing_lineage_artifacts()
    if existing:
        formatted = "\n".join(f"  - {path}" for path in existing)
        raise RuntimeError(
            "QCPO_REF context lineage already owns artifacts; use "
            "--allow_resume only for same-signature recovery:\n" + formatted)


def validate_exact_result(result):
    """
    Validate all four exact-N=1000 records against their immutable checkpoints.

    This gate deliberately re-reads iteration metadata and SHA256 from disk. A truncated, manually
    edited, or stale raw JSON therefore cannot become the tracked comparison manifest merely because
    its top-level W&B ID and policy identity look plausible.
    """
    expected_identity = {
        "campaign": CAMPAIGN,
        "environment": core.ENVIRONMENT,
        "algorithm": SPEC.algorithm,
        "seed": core.SEED,
        "wandb_id": SPEC.wandb_id,
        "wandb_group": SPEC.wandb_group,
        "wandb_tag": SPEC.wandb_tag,
        "parent_policy_identity": SPEC.parent_policy_identity,
        "policy_identity": SPEC.policy_identity,
        "variant_changes": SPEC.variant_changes,
        "training_budget_semantics": "main_excludes_initialization_v1",
    }
    for field, expected in expected_identity.items():
        if result.get(field) != expected:
            raise RuntimeError(
                f"context result {field} mismatch: "
                f"saved={result.get(field)!r}, expected={expected!r}")

    protocol = result.get("evaluation_protocol") or {}
    expected_protocol = {
        "name": "exact_n1000_common_random_numbers_v1",
        "num_episodes": core.FINAL_EVALUATION_EPISODES,
        "environment_seed": core.FINAL_EVALUATION_SEED,
        "action_rng_seed": core.FINAL_EVALUATION_SEED,
        "checkpoint_rng_restored": False,
    }
    if protocol != expected_protocol:
        raise RuntimeError("context result has the wrong exact-CRN evaluation protocol")

    evaluations = result.get("evaluations") or {}
    expected_checkpoints = core.policy_checkpoints(SPEC)
    if set(evaluations) != set(expected_checkpoints):
        raise RuntimeError("context result must contain main5m and main10m evaluations")
    for setting, checkpoint_group in expected_checkpoints.items():
        records = evaluations.get(setting) or {}
        if set(records) != {"endpoint", "selected"}:
            raise RuntimeError(
                f"context result {setting} must contain endpoint and selected")
        budget_batch = int(checkpoint_group["batch"])
        for role in ("endpoint", "selected"):
            record = records[role]
            checkpoint = Path(checkpoint_group[role])
            if record.get("checkpoint") != str(checkpoint):
                raise RuntimeError(
                    f"context result {setting}/{role} has the wrong checkpoint path")
            saved_iteration = int(record.get("checkpoint_iteration", -1))
            disk_iteration = core.checkpoint_iteration(checkpoint)
            if saved_iteration != disk_iteration:
                raise RuntimeError(
                    f"context result {setting}/{role} iteration differs from checkpoint")
            if role == "endpoint" and saved_iteration != budget_batch - 1:
                raise RuntimeError(
                    f"context result {setting} endpoint is not batch {budget_batch}")
            if role == "selected" and not 0 <= saved_iteration < budget_batch:
                raise RuntimeError(
                    f"context result {setting} selected iteration is outside its budget")
            actual_hash = core.sha256_file(checkpoint)
            if record.get("sha256") != actual_hash:
                raise RuntimeError(
                    f"context result {setting}/{role} checkpoint hash mismatch")

            # Each checkpoint is saved after its own completed rollout. Selected policies may be
            # earlier than the budget boundary, so their clock follows iteration+1, not 5M/10M.
            expected_main_steps = (
                (saved_iteration + 1) * core.NUM_ENVS * core.HORIZON)
            clocks = record.get("step_clocks") or {}
            expected_clocks = {
                "main_env_steps": expected_main_steps,
                "initialization_env_steps": 0,
                "actual_env_steps": expected_main_steps,
            }
            if clocks != expected_clocks:
                raise RuntimeError(
                    f"context result {setting}/{role} has inconsistent step clocks")

            metrics = record.get("metrics") or {}
            if int(metrics.get("num_episodes", -1)) != core.FINAL_EVALUATION_EPISODES:
                raise RuntimeError(
                    f"context result {setting}/{role} is not exact-N=1000")
            if float(metrics.get("return_threshold", math.nan)) != RETURN_THRESHOLD:
                raise RuntimeError(
                    f"context result {setting}/{role} has the wrong threshold")
            for name in (
                    "mean", "reward_std", "quantile_return",
                    "empirical_prob", "return_variance"):
                value = metrics.get(name)
                if value is None or not math.isfinite(float(value)):
                    raise RuntimeError(
                        f"context result {setting}/{role} has non-finite {name}")
            probability = float(metrics["empirical_prob"])
            failure_count = int(metrics.get("failure_count", -1))
            if not 0.0 <= probability <= 1.0:
                raise RuntimeError(
                    f"context result {setting}/{role} has invalid outage probability")
            if failure_count < 0 or not math.isclose(
                    probability,
                    failure_count / float(core.FINAL_EVALUATION_EPISODES),
                    rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(
                    f"context result {setting}/{role} has inconsistent failure count")


def write_tracked_manifest_if_complete():
    """Publish one compact result only after exact evaluation, sync, and baseline hash checks."""
    if not SPEC.raw_result_path.exists():
        return False
    sync_marker = read_valid_sync_marker()
    if sync_marker is None:
        return False
    baseline = load_pinned_baseline_manifest()
    result = json.loads(SPEC.raw_result_path.read_text(encoding="utf-8"))
    if not core.runner_complete(SPEC):
        raise RuntimeError("context runner endpoint artifacts are incomplete")
    if not core.verify_milestone(SPEC):
        raise RuntimeError("context exact batch-200 milestone is incomplete")
    validate_exact_result(result)
    if result.get("git_sha") != core.git_sha():
        raise RuntimeError("context result git SHA differs from the live committed code")
    if result.get("wandb_id") != SPEC.wandb_id:
        raise RuntimeError("context result has the wrong W&B identity")
    if result.get("policy_identity") != SPEC.policy_identity:
        raise RuntimeError("context result has the wrong policy identity")

    baseline_results = baseline["results"]
    payload = {
        "campaign": CAMPAIGN,
        "git_sha": core.git_sha(),
        "fresh_training": True,
        "seed0_only_per_user": True,
        "change_contract": {
            "parent_policy_identity": SPEC.parent_policy_identity,
            "policy_identity": SPEC.policy_identity,
            "qcpo_ref_critic_input": "finite_obs_remaining_actor_h0c0_v1",
            "horizon_mode": "finite",
            "critic_time_feature": "remaining_fraction",
            "critic_actor_context": True,
            "unchanged": (
                "actor_lstm;discounted_fixed_BH_PPO;reward_GAE;"
                "negated_return_quantiles;Weibull;dual;training_budget"),
        },
        "baseline_reference": {
            "campaign": baseline["campaign"],
            "manifest": str(BASELINE_MANIFEST.relative_to(BASE_DIR)),
            "sha256": BASELINE_MANIFEST_SHA256,
            "git_sha": baseline["git_sha"],
            "old_qref_critic_input": "observation_only_v1",
            "wandb_ids": {
                alias: baseline_results[alias]["wandb_id"]
                for alias in ("dqc", "qref", "qcpo")
            },
        },
        "common_setting": {
            "seed": core.SEED,
            "environment": core.ENVIRONMENT,
            "actor_lr0": core.ACTOR_LR,
            "return_threshold": RETURN_THRESHOLD,
            "failure_alpha": core.FAILURE_ALPHA,
            "gamma": core.GAMMA,
            "num_envs": core.NUM_ENVS,
            "horizon": core.HORIZON,
            "main_batches": core.FINAL_BATCH,
            "main_env_steps": core.FINAL_BATCH * core.NUM_ENVS * core.HORIZON,
            "initialization_env_steps": 0,
            "actual_env_steps": core.FINAL_BATCH * core.NUM_ENVS * core.HORIZON,
            "training_budget_semantics": "main_excludes_initialization_v1",
            "validation": {
                "exact_n": core.VALIDATION_EPISODES,
                "crn_seed": core.VALIDATION_SEED,
                "every_main_batches": 10,
            },
            "posthoc": {
                "exact_n": core.FINAL_EVALUATION_EPISODES,
                "crn_seed": core.FINAL_EVALUATION_SEED,
                "roles": ["endpoint", "selected"],
                "budgets": ["main5m", "main10m"],
            },
        },
        "wandb_sync": sync_marker,
        "results": {"qref_context": result},
    }
    core.write_json_atomic(TRACKED_MANIFEST, payload)
    return True


def run_lineage(device_text, allow_resume=False):
    """Train from scratch/recover, explicitly sync online W&B, evaluate, then publish."""
    if not allow_resume:
        assert_fresh_lineage()
    core.run_training(SPEC, device_text, "online")
    sync_wandb_online()
    device = torch.device(
        "cpu" if device_text == "cpu"
        else f"cuda:{device_text}" if torch.cuda.is_available()
        else "cpu")
    core.evaluate_lineage(SPEC, device)
    if not write_tracked_manifest_if_complete():
        raise RuntimeError("context rerun publication gate remained incomplete")


def verify_complete():
    """Require endpoint/milestone integrity, exact result, explicit sync, and tracked manifest."""
    if not core.runner_complete(SPEC):
        raise RuntimeError(f"{SPEC.key}: formal endpoint artifacts are incomplete")
    if not core.verify_milestone(SPEC):
        raise RuntimeError(f"{SPEC.key}: exact batch-200 milestone is incomplete")
    if read_valid_sync_marker() is None:
        raise RuntimeError(f"{SPEC.key}: explicit online W&B sync is incomplete")
    if not SPEC.raw_result_path.exists() or not TRACKED_MANIFEST.exists():
        raise RuntimeError(f"{SPEC.key}: exact evaluation/manifest is incomplete")
    load_pinned_baseline_manifest()


def main():
    """Dispatch one immutable QCPO_REF context lineage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "dry-run", "evaluate", "verify"))
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--wandb_mode", default="online", choices=("online",),
        help="formal context rerun is intentionally online-only")
    parser.add_argument(
        "--allow_resume", action="store_true",
        help="recover only this identity's signature-matched latest checkpoint")
    cli = parser.parse_args()
    _validate_registered_identity()

    with installed_core_identity():
        if cli.action == "dry-run":
            core.run_training(SPEC, cli.device, cli.wandb_mode, dry_run=True)
            return
        if cli.action == "run":
            run_lineage(cli.device, allow_resume=cli.allow_resume)
            print(f"[formal complete] result={SPEC.raw_result_path}", flush=True)
            return
        if cli.action == "verify":
            verify_complete()
            print(f"{SPEC.key}: artifacts verified", flush=True)
            return

        # Evaluate is an explicit recovery tool after training and sync. It still applies strict
        # signature checks and cannot publish until the pinned baseline and sync marker validate.
        device = torch.device(
            "cpu" if cli.device == "cpu"
            else f"cuda:{cli.device}" if torch.cuda.is_available()
            else "cpu")
        core.evaluate_lineage(SPEC, device)
        if not write_tracked_manifest_if_complete():
            raise RuntimeError("context rerun publication gate remained incomplete")


if __name__ == "__main__":
    main()
