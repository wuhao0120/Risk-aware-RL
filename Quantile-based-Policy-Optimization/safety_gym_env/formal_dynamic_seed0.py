#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic seed-0 two-threshold formal campaign.

The wrapper extends the existing formal lifecycle only. It never changes an agent, loss, network,
environment, or training equation. QPO first establishes an empirical reward scale without a fixed
q; the initial and strict constrained stages require user-supplied thresholds and fresh parameters.
"""
import argparse  # Expose explicit stage/action gates instead of deriving thresholds automatically.
from contextlib import contextmanager  # Restore the reusable harness after every stage operation.
from dataclasses import dataclass  # Freeze one fully resolved stage before constructing commands.
import hashlib  # Sign thresholds, checkpoint signatures, and compact result references.
import json  # Validate atomic lineage contracts and exact-evaluation artifacts.
import math  # Reject non-finite user thresholds and result metrics.
import os  # Force explicit online W&B synchronization for formal runs.
from pathlib import Path  # Keep paths stable regardless of the caller's working directory.
import subprocess  # Launch parallel phase-one workers and query the committed Git identity.
import sys  # Reuse the active interpreter for workers and W&B synchronization.
import time  # Retry the W&B history audit during short service-consistency delays.

import torch  # Load signed checkpoints and select the exact post-hoc evaluation device.

import formal_qcpo_ref_dqc_gae as core  # Reuse checkpoint watching, freezing, and exact CRN tests.


# ============================================================================ campaign invariants
BASE_DIR = Path(__file__).resolve().parent  # Anchor every generated artifact under safety_gym_env.
CAMPAIGN = "dynamic_seed0_v1"  # One tracked identity contains the QPO, q0, and q1 decisions.
ENVIRONMENT = "Dynamic"  # First clean difficulty increase after SimpleButton.
RAW_CAMPAIGN_ROOT = BASE_DIR / "_runs" / CAMPAIGN  # Large artifacts remain untracked.
TRACKED_MANIFEST = BASE_DIR / "campaigns" / CAMPAIGN / "manifest.json"
FROZEN_SHA_PATH = RAW_CAMPAIGN_ROOT / "frozen_executable_sha.json"

# Formal budgets are repeated here as assertions, not tunable alternatives to the reusable core.
SEED = 0
NUM_ENVS = 25
HORIZON = 1000
FINAL_BATCH = 400
MILESTONE_BATCH = 200
CHECKPOINT_EVERY = 50
MAIN_ENV_STEPS = FINAL_BATCH * NUM_ENVS * HORIZON
VALIDATION_EVERY = 10
WANDB_PROJECT = "safety_gym_reward_risk"

# Only Python sources that can affect a formal result participate in the cleanliness gate. Existing
# user edits under paper/, talk/, and PDFs are deliberately outside this path list.
EXECUTABLE_PATHS = (
    "Quantile-based-Policy-Optimization/safety_gym_env/agents",
    "Quantile-based-Policy-Optimization/safety_gym_env/envs",
    "Quantile-based-Policy-Optimization/safety_gym_env/utils",
    "Quantile-based-Policy-Optimization/safety_gym_env/tests",
    "Quantile-based-Policy-Optimization/safety_gym_env/run_experiment.py",
    "Quantile-based-Policy-Optimization/safety_gym_env/formal_qcpo_ref_dqc_gae.py",
    "Quantile-based-Policy-Optimization/safety_gym_env/formal_dynamic_seed0.py",
)

# This records why the stable parent is used without overstating the state-CDF ablation evidence.
STATE_CDF_DECISION = {
    "decision": "not_promoted_to_default",
    "evidence_scope": "SimpleButton seed 0 engineering evidence",
    "evidence": (
        "state-CDF had a positive 10M endpoint signal only on seed 0, while early and selected "
        "results were unstable; that evidence is insufficient to promote it to the default"),
    "complexity_delta_percent": 20.6,
    "checkpoint_size_delta_percent": 25.7,
    "critic_update_work_delta_percent": 50.0,
    "prohibited_claims": [
        "state-CDF had no improvement",
        "measured wall-clock must be slower",
    ],
}


@dataclass(frozen=True)
class StageConfig:
    """
    Fully resolve one CLI stage before any artifact or external run can be created.

    The threshold is None only for QPO. Strict stages also carry the previously confirmed q0 so the
    transition q0 -> q1 is explicit in paths, metadata, and human-readable output.
    """

    stage: str
    threshold: object
    q0: object
    token: str
    campaign: str
    raw_root: Path
    stage_result_path: Path
    specs: dict
    phases: tuple


# ============================================================================ stage construction
def _finite_threshold(value, name):
    """Return a finite float and preserve a precise error label for conditional CLI validation."""
    if value is None:
        raise ValueError(f"{name} is required")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def threshold_token(value):
    """Map the exact 17-digit float spelling to a short deterministic artifact/W&B token."""
    canonical = format(float(value), ".17g")
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
    return f"q{digest}"


def _constrained_specs(stage, threshold, q0, token):
    """Build fresh DQC, QCPO_REF-context, and trajectory-LOO QCPO identities for one q."""
    prefix = "dyi" if stage == "initial" else "dys"
    group = f"{CAMPAIGN}_{stage}_{token}"
    transition = (
        f"dynamic_fresh_seed0_q0_{format(threshold, '.17g')}"
        if stage == "initial"
        else (
            f"dynamic_fresh_seed0_q0_{format(q0, '.17g')}_"
            f"to_q1_{format(threshold, '.17g')}")
    )
    common = {
        "wandb_group": group,
        "checkpoint_every": CHECKPOINT_EVERY,
        "uses_return_threshold": True,
    }
    return {
        "dqc": core.FormalSpec(
            key=f"{prefix}dq{token}s0",
            algorithm="DQCACBeta",
            wandb_id=f"{prefix}dq{token}s0",
            wandb_tag=f"dynamic-dqc-{stage}-{token}-main10m-s0",
            tmux_name=f"dynamic_{stage}_dqc_{token}_s0",
            parent_policy_identity="finite_nstep_v1_state_value_gae_v1_stable_parent",
            policy_identity="finite_nstep_v1_state_value_gae_v1",
            variant_changes=(
                f"{transition};sampled_action_cdf_m4;no_learned_state_distribution"),
            initialization_batches=10,
            wandb_name=f"DQC-GAE-Dynamic-{stage}-{token}-s0",
            # M=4 is explicit even though it is also the stable default. This keeps command audits
            # and checkpoint signatures truthful if a future generic default changes.
            extra_overrides=("num_action_samples=4",),
            **common,
        ),
        "qref": core.FormalSpec(
            key=f"{prefix}qr{token}s0",
            algorithm="QCPO_REF",
            wandb_id=f"{prefix}qr{token}s0",
            wandb_tag=f"dynamic-qref-context-{stage}-{token}-main10m-s0",
            tmux_name=f"dynamic_{stage}_qref_{token}_s0",
            parent_policy_identity=(
                "paper_negated_return_eq275_discounted_bh_v2_stable_parent"),
            policy_identity=(
                "paper_negated_return_eq275_discounted_bh_v2_dqc_finite_context_v1"),
            variant_changes=(
                f"{transition};finite_remaining_time_detached_actor_h0c0_critic"),
            initialization_batches=0,
            wandb_name=f"QCPO-REF-Context-Dynamic-{stage}-{token}-s0",
            **common,
        ),
        "qcpo": core.FormalSpec(
            key=f"{prefix}qc{token}s0",
            algorithm="QCPO",
            wandb_id=f"{prefix}qc{token}s0",
            wandb_tag=f"dynamic-qcpo-loo-{stage}-{token}-main10m-s0",
            tmux_name=f"dynamic_{stage}_qcpo_{token}_s0",
            parent_policy_identity="qcpo_reward_chance_loo_v1_stable_parent",
            policy_identity="qcpo_reward_chance_loo_v1",
            variant_changes=f"{transition};trajectory_level_leave_one_out",
            initialization_batches=0,
            wandb_name=f"QCPO-LOO-Dynamic-{stage}-{token}-s0",
            **common,
        ),
    }


def make_stage_config(stage, return_threshold=None, q0=None):
    """
    Validate stage-specific threshold rules and return immutable commands/artifact identities.

    QPO rejects q entirely. Initial interprets the explicit return threshold as q0. Strict requires
    both the recorded q0 and a strictly larger q1; equality and reversal fail before filesystem use.
    """
    stage = str(stage)
    if stage == "qpo":
        if return_threshold is not None or q0 is not None:
            raise ValueError("QPO stage must not receive a fixed return threshold or q0")
        token = "anchor"
        raw_root = RAW_CAMPAIGN_ROOT / "qpo"
        group = f"{CAMPAIGN}_qpo_anchor"
        specs = {
            "qpo": core.FormalSpec(
                key="dynqpos0v1",
                algorithm="QPO",
                wandb_id="dynqpos0v1",
                wandb_group=group,
                wandb_tag="dynamic-qpo-anchor-lr2e4-main10m-s0",
                tmux_name="dynamic_qpo_anchor_s0",
                parent_policy_identity="safety_gym_stable_q2p3_context_v1",
                policy_identity="pure_quantile_two_timescale_v1",
                variant_changes="environment_simplebutton_to_dynamic;fresh_seed0_training",
                initialization_batches=2,
                wandb_name="QPO-Anchor-Dynamic-lr2e4-s0",
                checkpoint_every=CHECKPOINT_EVERY,
                uses_return_threshold=False,
            ),
        }
        phases = (("qpo",),)
        threshold = None
        prior = None
    elif stage == "initial":
        threshold = _finite_threshold(return_threshold, "--return-threshold")
        if q0 is not None:
            raise ValueError("initial stage uses --return-threshold as q0; do not pass --q0")
        prior = None
        token = threshold_token(threshold)
        raw_root = RAW_CAMPAIGN_ROOT / f"initial_{token}"
        specs = _constrained_specs(stage, threshold, prior, token)
        phases = (("dqc", "qref"), ("qcpo",))
    elif stage == "strict":
        threshold = _finite_threshold(return_threshold, "--return-threshold q1")
        prior = _finite_threshold(q0, "--q0")
        if not threshold > prior:
            raise ValueError(
                f"strict stage requires q1 > q0; received q1={threshold}, q0={prior}")
        token = f"{threshold_token(prior)}_{threshold_token(threshold)}"
        raw_root = RAW_CAMPAIGN_ROOT / f"strict_{token}"
        specs = _constrained_specs(stage, threshold, prior, token)
        phases = (("dqc", "qref"), ("qcpo",))
    else:
        raise ValueError(f"unsupported stage {stage!r}")

    # A stage-specific core campaign makes every raw evaluation self-identifying.
    stage_campaign = f"{CAMPAIGN}_{stage}_{token}"
    return StageConfig(
        stage=stage,
        threshold=threshold,
        q0=prior,
        token=token,
        campaign=stage_campaign,
        raw_root=raw_root,
        stage_result_path=raw_root / "stage_manifest.json",
        specs=specs,
        phases=phases,
    )


@contextmanager
def installed_core_identity(config):
    """Install Dynamic/stage paths in the reusable module and restore every global afterward."""
    names = (
        "CAMPAIGN", "ENVIRONMENT", "RETURN_THRESHOLD",
        "RAW_ROOT", "TRACKED_MANIFEST", "SPECS",
    )
    previous = tuple(getattr(core, name) for name in names)
    replacements = (
        config.campaign, ENVIRONMENT, config.threshold,
        config.raw_root, config.stage_result_path, config.specs,
    )
    for name, value in zip(names, replacements):
        setattr(core, name, value)
    try:
        yield
    finally:
        for name, value in zip(names, previous):
            setattr(core, name, value)


def validate_registered_identity(config):
    """Reject identity collisions, overlong service tags, or any accidental learned state CDF."""
    ids = [spec.wandb_id for spec in config.specs.values()]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{config.stage}: W&B IDs are not unique")
    for alias, spec in config.specs.items():
        if not spec.wandb_name:
            raise RuntimeError(f"{alias}: a readable W&B name is required")
        if len(spec.wandb_tag) > 64:
            raise RuntimeError(f"{alias}: W&B tag exceeds 64 characters")
        if int(spec.checkpoint_every) != CHECKPOINT_EVERY:
            raise RuntimeError(f"{alias}: rolling checkpoint interval must be 50 batches")

        # Inspect the true runner argv, not merely the declarative spec, so the guard covers future
        # reusable-harness changes as well as today's explicit overrides.
        with installed_core_identity(config):
            command_text = " ".join(
                core.build_training_command(spec, "0", "online")).lower()
        if alias == "dqc":
            required = (
                "dqc_update_mode=finite_nstep_v1",
                "dqc_mean_estimator=state_value_gae_v1",
                "num_action_samples=4",
            )
            if any(item not in command_text for item in required):
                raise RuntimeError("DQC stable finite-nstep/GAE/M=4 identity is incomplete")
            forbidden = (
                "learned_state_distribution_v1",
                "state_cdf",
                "state-cdf",
            )
            if any(item in command_text for item in forbidden):
                raise RuntimeError("learned state-CDF components are forbidden in this campaign")


# ============================================================================ executable and resume contracts
def _repo_root():
    """Resolve the repository root through Git without relying on a caller-specific cwd."""
    return Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=BASE_DIR, text=True).strip())


def assert_executable_tree_clean():
    """
    Require every executable Python path to match HEAD while ignoring unrelated user work.

    Both staged and unstaged differences are reported by porcelain status. Generated run artifacts
    and campaign JSON are outside the listed paths, so completed formal artifacts do not block verify.
    """
    command = [
        "git", "status", "--porcelain", "--untracked-files=all", "--",
        *EXECUTABLE_PATHS,
    ]
    dirty = subprocess.check_output(
        command, cwd=_repo_root(), text=True).strip()
    if dirty:
        raise RuntimeError(
            "formal executable tree differs from HEAD; commit/freeze it before launch:\n" + dirty)


def frozen_executable_sha(create=False):
    """
    Return one campaign-wide committed SHA, creating the atomic marker only for a formal launch.

    Every later stage and resume must use this exact SHA. This turns a code repair into an explicit
    new executable lineage instead of mixing affected comparisons under one campaign.
    """
    assert_executable_tree_clean()
    current = core.git_sha()
    if FROZEN_SHA_PATH.exists():
        marker = json.loads(FROZEN_SHA_PATH.read_text(encoding="utf-8"))
        if marker.get("git_sha") != current:
            raise RuntimeError(
                "Dynamic campaign executable SHA is frozen at "
                f"{marker.get('git_sha')}; current HEAD is {current}")
        return current
    if not create:
        raise RuntimeError("campaign executable SHA has not been frozen by the QPO launch")
    core.write_json_atomic(FROZEN_SHA_PATH, {
        "campaign": CAMPAIGN,
        "git_sha": current,
        "frozen_before_first_formal_run": True,
    })
    return current


def _signature_digest(signature):
    """Hash a checkpoint signature with canonical ordering and no representation ambiguity."""
    encoded = json.dumps(
        signature, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def requested_signature(spec):
    """Reconstruct the exact runner signature, including Dynamic, q, and DQC M=4."""
    from run_experiment import checkpoint_signature

    args = core.evaluation_args(spec, torch.device("cpu"))
    return checkpoint_signature(args, spec.algorithm)


def lineage_contract(config, alias, spec, executable_sha):
    """Build the deterministic same-env/algo/q/SHA/signature contract for one lineage."""
    signature = requested_signature(spec)
    return {
        "schema": "dynamic_formal_lineage_v1",
        "campaign": CAMPAIGN,
        "stage_campaign": config.campaign,
        "stage": config.stage,
        "environment": ENVIRONMENT,
        "seed": SEED,
        "alias": alias,
        "algorithm": spec.algorithm,
        "return_threshold": config.threshold,
        "prior_q0": config.q0,
        "executable_git_sha": executable_sha,
        "wandb_id": spec.wandb_id,
        "wandb_group": spec.wandb_group,
        "policy_identity": spec.policy_identity,
        "checkpoint_signature": signature,
        "checkpoint_signature_sha256": _signature_digest(signature),
    }


def lineage_contract_path(spec):
    """Store the contract beside only this lineage's signed rolling checkpoints."""
    return spec.checkpoint_root / "lineage.json"


def _agent_class_name(algorithm):
    """Map the public runner algorithm to the checkpoint's concrete Python class."""
    return {
        "QPO": "QPOGPU",
        "QCPO": "QCPOGPU",
        "QCPO_REF": "QCPORefGPU",
        "DQCACBeta": "DQCACBetaGPU",
    }[algorithm]


def validate_resume_lineage(config, alias, spec, executable_sha):
    """
    Validate lineage JSON and checkpoint bytes before core.run_training can create an agent.

    The reusable loader repeats the complete signature check before mutating parameters. This outer
    check additionally signs Git SHA and artifact identity, neither of which is a model parameter.
    """
    expected = lineage_contract(config, alias, spec, executable_sha)
    contract_path = lineage_contract_path(spec)
    try:
        saved = json.loads(contract_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError) as error:
        raise RuntimeError(
            f"{alias}: --allow-resume requires an intact lineage contract") from error
    if saved != expected:
        raise RuntimeError(
            f"{alias}: resume lineage differs in environment/algorithm/q/SHA/signature")

    checkpoint = core.resume_checkpoint(spec)
    if checkpoint is None and core.runner_complete(spec):
        checkpoint = spec.live_checkpoint_dir / "latest.pt"
    if checkpoint is None:
        raise RuntimeError(
            f"{alias}: --allow-resume requires a signed rolling or initialization checkpoint")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("agent_class") != _agent_class_name(spec.algorithm):
        raise RuntimeError(f"{alias}: resume checkpoint has the wrong agent class")
    actual_signature = payload.get("experiment_signature")
    if actual_signature != expected["checkpoint_signature"]:
        raise RuntimeError(
            f"{alias}: resume checkpoint signature differs from the lineage contract")
    return checkpoint


def _local_wandb_dirs(spec):
    """Find all local segments belonging to one globally unique stable W&B ID."""
    wandb_root = BASE_DIR / "wandb"
    if not wandb_root.exists():
        return []
    suffix = "-" + spec.wandb_id
    return sorted(
        path for path in wandb_root.iterdir()
        if path.is_dir()
        and path.name.endswith(suffix)
        and (
            path.name.startswith("run-")
            or path.name.startswith("offline-run-")
        )
    )


def sync_marker_path(spec):
    """Keep one explicit sync/history audit marker per lineage."""
    return spec.checkpoint_root / "wandb_sync.json"


def _existing_spec_artifacts(spec):
    """Return every campaign-owned artifact that makes a lineage non-fresh."""
    candidates = (
        spec.checkpoint_root,
        spec.progress_path,
        spec.log_path,
        spec.raw_result_path,
        spec.runner_result_path,
        sync_marker_path(spec),
    )
    return [
        path for path in (*candidates, *_local_wandb_dirs(spec))
        if Path(path).exists()
    ]


def assert_fresh_stage(config):
    """Reject any prior artifact across a stage before starting its first subprocess."""
    existing = []
    for spec in config.specs.values():
        existing.extend(_existing_spec_artifacts(spec))
    if config.stage_result_path.exists():
        existing.append(config.stage_result_path)
    if existing:
        formatted = "\n".join(f"  - {path}" for path in existing)
        raise RuntimeError(
            f"{config.stage}: formal artifacts already exist; use --allow-resume only for "
            "same-lineage recovery:\n" + formatted)


# ============================================================================ W&B synchronization and 400-row audit
def _wandb_run_path(spec):
    """Build the API suffix; the authenticated default entity is resolved by the caller."""
    return f"{WANDB_PROJECT}/{spec.wandb_id}"


def query_wandb_main_history(spec):
    """
    Require a finished W&B run with exactly iterations 0..399 and reconstruct early AUC.

    The network audit is performed after explicit sync. It is stronger than counting a local
    runner JSON because an interrupted/resumed runner JSON contains only its newest process segment.
    """
    import wandb

    last_error = None
    for attempt in range(6):
        try:
            api = wandb.Api(timeout=60)
            # WANDB_ENTITY wins when a team is explicitly selected. Otherwise use the same
            # authenticated default entity that wandb.init chose for this formal run.
            entity = str(
                os.environ.get("WANDB_ENTITY")
                or getattr(api, "default_entity", "")
            ).strip()
            if not entity:
                raise RuntimeError("W&B API did not expose a default entity")
            run_path = f"{entity}/{_wandb_run_path(spec)}"
            run = api.run(run_path)
            rows = list(run.scan_history(page_size=1000))
            main_rows = [
                row for row in rows
                if row.get("progress/iteration") is not None
            ]
            iterations = [int(row["progress/iteration"]) for row in main_rows]
            if iterations != list(range(FINAL_BATCH)):
                raise RuntimeError(
                    f"{spec.key}: W&B main history must be exactly 0..399; "
                    f"received {len(iterations)} rows")

            # Rebuild the 0--2M diagnostics from the service history so a valid resume retains the
            # same early-AUC definition as an uninterrupted process.
            validations = []
            diagnostics = []
            for row in main_rows:
                iteration = int(row["progress/iteration"])
                if iteration >= 80:
                    continue
                main_steps = (iteration + 1) * NUM_ENVS * HORIZON
                diagnostic = {
                    "iteration": iteration,
                    "main_env_steps": main_steps,
                }
                for name in (
                    "actor/clip_fraction",
                    "diagnostic/dqc/policy_kl_after_epochs",
                    "diagnostic/dqc/value_rmse",
                    "diagnostic/dqc/critic_cdf_mean",
                ):
                    if row.get(name) is not None:
                        diagnostic[name] = float(row[name])
                diagnostics.append(diagnostic)

                mean = row.get("validation/disc_reward/aver_reward")
                if mean is not None:
                    validations.append({
                        "iteration": iteration,
                        "mean": float(mean),
                    })
            early = core.summarize_early_history({
                "validation_history": validations,
                "training_diagnostics_history": diagnostics,
            })
            state = str(getattr(run, "state", "")).lower()
            if state != "finished":
                raise RuntimeError(
                    f"{spec.key}: W&B state must be finished, received {state!r}")
            return {
                "run_path": run_path,
                "state": state,
                "main_history_rows": len(iterations),
                "main_iterations": [iterations[0], iterations[-1]],
                "early_0_2m": early,
            }
        except Exception as error:
            last_error = error
            if attempt < 5:
                time.sleep(5.0)
    raise RuntimeError(
        f"{spec.key}: W&B history audit failed after explicit sync") from last_error


def read_valid_sync_marker(spec, executable_sha):
    """Return only a same-SHA, finished, exact-400-row sync marker."""
    try:
        marker = json.loads(
            sync_marker_path(spec).read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return None
    valid = (
        marker.get("synced") is True
        and marker.get("wandb_id") == spec.wandb_id
        and marker.get("executable_git_sha") == executable_sha
        and marker.get("state") == "finished"
        and int(marker.get("main_history_rows", -1)) == FINAL_BATCH
        and marker.get("main_iterations") == [0, FINAL_BATCH - 1]
    )
    return marker if valid else None


def sync_wandb_online(spec, executable_sha):
    """Explicitly sync every local segment, then publish a marker only after the API row audit."""
    existing = read_valid_sync_marker(spec, executable_sha)
    if existing is not None:
        return existing
    run_dirs = _local_wandb_dirs(spec)
    if not run_dirs:
        raise RuntimeError(f"{spec.key}: no local W&B run directory exists")
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
            f"{spec.key}: explicit W&B sync exited {completed.returncode}")

    audit = query_wandb_main_history(spec)
    marker = {
        "synced": True,
        "wandb_id": spec.wandb_id,
        "executable_git_sha": executable_sha,
        "local_run_dirs": [str(path) for path in run_dirs],
        "command": command,
        **audit,
    }
    core.write_json_atomic(sync_marker_path(spec), marker)
    return marker


# ============================================================================ training and exact-result validation
def _audit_runner_result(spec):
    """Check runner schedule metadata while W&B remains authoritative for cross-resume row count."""
    result = json.loads(spec.runner_result_path.read_text(encoding="utf-8"))
    if result.get("algo") != spec.algorithm or result.get("env") != ENVIRONMENT:
        raise RuntimeError(f"{spec.key}: runner result has the wrong algorithm/environment")
    if int(result.get("seed", -1)) != SEED:
        raise RuntimeError(f"{spec.key}: runner result has the wrong seed")
    if int(result.get("checkpoint_every", -1)) != CHECKPOINT_EVERY:
        raise RuntimeError(f"{spec.key}: runner did not use 50-batch rolling checkpoints")
    if int(result.get("validation_every", -1)) != VALIDATION_EVERY:
        raise RuntimeError(f"{spec.key}: runner did not use 10-batch validation")
    saved_threshold = result.get("return_threshold")
    if saved_threshold != core.RETURN_THRESHOLD:
        raise RuntimeError(f"{spec.key}: runner threshold differs from the stage")
    return result


def _augment_raw_result(spec, executable_sha, sync_marker):
    """Attach service-audited early history and signature metadata to the atomic raw result."""
    result = json.loads(spec.raw_result_path.read_text(encoding="utf-8"))
    signature = requested_signature(spec)
    result["executable_git_sha"] = executable_sha
    result["checkpoint_signature_sha256"] = _signature_digest(signature)
    result["wandb_main_history_rows"] = sync_marker["main_history_rows"]
    result["early_0_2m"] = sync_marker["early_0_2m"]

    # QPO's persistent tracker is diagnostic only. Independent empirical linear q20 remains the
    # threshold-selection quantity presented to the user.
    if spec.algorithm == "QPO":
        for checkpoint_group in result.get("evaluations", {}).values():
            for record in checkpoint_group.values():
                metrics = record.get("metrics") or {}
                metrics["diagnostic_q_tracker_deviation"] = (
                    float(metrics["quantile_q_est"])
                    - float(metrics["quantile_return"])
                )
    core.write_json_atomic(spec.raw_result_path, result)
    return result


def validate_exact_result(config, alias, spec, executable_sha):
    """Validate identity, hashes, checkpoint signatures, clocks, and every exact-N=1000 record."""
    result = json.loads(spec.raw_result_path.read_text(encoding="utf-8"))
    expected_top = {
        "campaign": config.campaign,
        "git_sha": executable_sha,
        "executable_git_sha": executable_sha,
        "environment": ENVIRONMENT,
        "algorithm": spec.algorithm,
        "seed": SEED,
        "wandb_id": spec.wandb_id,
        "wandb_group": spec.wandb_group,
        "policy_identity": spec.policy_identity,
        "training_budget_semantics": "main_excludes_initialization_v1",
        "wandb_main_history_rows": FINAL_BATCH,
    }
    for field, expected in expected_top.items():
        if result.get(field) != expected:
            raise RuntimeError(
                f"{alias}: exact result {field}={result.get(field)!r}, expected {expected!r}")

    protocol = result.get("evaluation_protocol") or {}
    expected_protocol = {
        "name": "exact_n1000_common_random_numbers_v1",
        "num_episodes": core.FINAL_EVALUATION_EPISODES,
        "environment_seed": core.FINAL_EVALUATION_SEED,
        "action_rng_seed": core.FINAL_EVALUATION_SEED,
        "checkpoint_rng_restored": False,
    }
    if protocol != expected_protocol:
        raise RuntimeError(f"{alias}: exact CRN protocol mismatch")

    signature = requested_signature(spec)
    if result.get("checkpoint_signature_sha256") != _signature_digest(signature):
        raise RuntimeError(f"{alias}: exact result checkpoint signature hash mismatch")
    evaluations = result.get("evaluations") or {}
    checkpoints = core.policy_checkpoints(spec)
    if set(evaluations) != {"main5m", "main10m"}:
        raise RuntimeError(f"{alias}: exact result must contain 5M and 10M policies")

    for budget, checkpoint_group in checkpoints.items():
        records = evaluations.get(budget) or {}
        if set(records) != {"endpoint", "selected"}:
            raise RuntimeError(f"{alias}/{budget}: endpoint and selected are both required")
        batch = int(checkpoint_group["batch"])
        for role, checkpoint in (
                ("endpoint", checkpoint_group["endpoint"]),
                ("selected", checkpoint_group["selected"])):
            record = records[role]
            iteration = core.checkpoint_iteration(checkpoint)
            if int(record.get("checkpoint_iteration", -2)) != iteration:
                raise RuntimeError(f"{alias}/{budget}/{role}: checkpoint iteration mismatch")
            if role == "endpoint" and iteration != batch - 1:
                raise RuntimeError(f"{alias}/{budget}: endpoint is not the exact budget boundary")
            if role == "selected" and not 0 <= iteration < batch:
                raise RuntimeError(f"{alias}/{budget}: selected checkpoint lies outside the budget")
            if record.get("sha256") != core.sha256_file(checkpoint):
                raise RuntimeError(f"{alias}/{budget}/{role}: checkpoint hash mismatch")

            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if payload.get("experiment_signature") != signature:
                raise RuntimeError(f"{alias}/{budget}/{role}: checkpoint signature mismatch")
            expected_main = (iteration + 1) * NUM_ENVS * HORIZON
            expected_clocks = {
                "main_env_steps": expected_main,
                "initialization_env_steps": spec.initialization_env_steps,
                "actual_env_steps": expected_main + spec.initialization_env_steps,
            }
            if record.get("step_clocks") != expected_clocks:
                raise RuntimeError(f"{alias}/{budget}/{role}: step clocks are inconsistent")

            metrics = record.get("metrics") or {}
            if int(metrics.get("num_episodes", -1)) != core.FINAL_EVALUATION_EPISODES:
                raise RuntimeError(f"{alias}/{budget}/{role}: evaluation is not exact-N=1000")
            for field in ("mean", "reward_std", "quantile_return", "return_variance"):
                if not math.isfinite(float(metrics.get(field, math.nan))):
                    raise RuntimeError(f"{alias}/{budget}/{role}: non-finite {field}")
            if spec.uses_return_threshold:
                if float(metrics.get("return_threshold", math.nan)) != config.threshold:
                    raise RuntimeError(f"{alias}/{budget}/{role}: threshold mismatch")
                failure_count = int(metrics.get("failure_count", -1))
                probability = float(metrics.get("empirical_prob", math.nan))
                if failure_count < 0 or not math.isclose(
                        probability,
                        failure_count / core.FINAL_EVALUATION_EPISODES,
                        rel_tol=0.0, abs_tol=1e-12):
                    raise RuntimeError(
                        f"{alias}/{budget}/{role}: failure count/probability mismatch")
                for field in ("wilson95_low", "wilson95_high"):
                    if not math.isfinite(float(metrics.get(field, math.nan))):
                        raise RuntimeError(f"{alias}/{budget}/{role}: missing {field}")
            else:
                if "return_threshold" in metrics or "empirical_prob" in metrics:
                    raise RuntimeError("QPO exact result must remain free of a fixed-q constraint")
                expected_deviation = (
                    float(metrics["quantile_q_est"])
                    - float(metrics["quantile_return"]))
                if not math.isclose(
                        float(metrics.get("diagnostic_q_tracker_deviation", math.nan)),
                        expected_deviation, rel_tol=0.0, abs_tol=1e-12):
                    raise RuntimeError("QPO tracker deviation is inconsistent")

    return result


def prepare_lineage(config, alias, spec, executable_sha, allow_resume):
    """Create a fresh contract or validate the only allowed same-lineage recovery path."""
    existing = _existing_spec_artifacts(spec)
    contract = lineage_contract(config, alias, spec, executable_sha)
    if existing:
        if not allow_resume:
            formatted = "\n".join(f"  - {path}" for path in existing)
            raise RuntimeError(
                f"{alias}: lineage is not fresh; pass --allow-resume only after review:\n"
                + formatted)
        validate_resume_lineage(config, alias, spec, executable_sha)
        return
    core.write_json_atomic(lineage_contract_path(spec), contract)


def run_one(config, alias, device_text, allow_resume=False):
    """Train/recover one signed lineage, sync online W&B, and evaluate four exact policies."""
    spec = config.specs[alias]
    executable_sha = frozen_executable_sha(create=(config.stage == "qpo"))
    prepare_lineage(config, alias, spec, executable_sha, allow_resume)
    core.run_training(spec, device_text, "online")
    _audit_runner_result(spec)
    sync_marker = sync_wandb_online(spec, executable_sha)
    device = torch.device(
        "cpu" if device_text == "cpu"
        else f"cuda:{device_text}" if torch.cuda.is_available()
        else "cpu")
    core.evaluate_lineage(spec, device)
    _augment_raw_result(spec, executable_sha, sync_marker)
    validate_exact_result(config, alias, spec, executable_sha)
    print(f"[lineage complete] {alias} result={spec.raw_result_path}", flush=True)


def evaluate_one(config, alias, device_text):
    """Re-run exact CRN evaluation only after training, signature, and sync gates pass."""
    spec = config.specs[alias]
    executable_sha = frozen_executable_sha(create=False)
    validate_resume_lineage(config, alias, spec, executable_sha)
    if not core.runner_complete(spec) or not core.verify_milestone(spec):
        raise RuntimeError(f"{alias}: formal checkpoint artifacts are incomplete")
    sync_marker = read_valid_sync_marker(spec, executable_sha)
    if sync_marker is None:
        raise RuntimeError(f"{alias}: explicit W&B sync/history audit is incomplete")
    device = torch.device(
        "cpu" if device_text == "cpu"
        else f"cuda:{device_text}" if torch.cuda.is_available()
        else "cpu")
    core.evaluate_lineage(spec, device)
    _augment_raw_result(spec, executable_sha, sync_marker)
    validate_exact_result(config, alias, spec, executable_sha)


# ============================================================================ summaries, prerequisites, and atomic manifests
def _metric_summary(record, qpo=False):
    """Select only decision metrics; raw trajectories/checkpoints remain under ignored run storage."""
    metrics = record["metrics"]
    summary = {
        "checkpoint_iteration": int(record["checkpoint_iteration"]),
        "mean": float(metrics["mean"]),
        "q20_linear": float(metrics["quantile_return"]),
        "std": float(metrics["reward_std"]),
        "variance": float(metrics["return_variance"]),
    }
    if qpo:
        summary.update({
            "q_tracker": float(metrics["quantile_q_est"]),
            "q_tracker_deviation": float(
                metrics["diagnostic_q_tracker_deviation"]),
            "q_tracker_empirical_probability": float(
                metrics["quantile_constraint_prob"]),
        })
    else:
        summary.update({
            "failure_count": int(metrics["failure_count"]),
            "failure_probability": float(metrics["empirical_prob"]),
            "wilson95": [
                float(metrics["wilson95_low"]),
                float(metrics["wilson95_high"]),
            ],
        })
    return summary


def stage_summary(config, results):
    """Build the QPO gate report or the constrained endpoint-primary comparison."""
    if config.stage == "qpo":
        result = results["qpo"]
        policies = result["evaluations"]["main10m"]
        return {
            "threshold_selection_basis": "independent_exact_n1000_empirical_linear_q20",
            "q_tracker_role": "diagnostic_only",
            "endpoint": _metric_summary(policies["endpoint"], qpo=True),
            "selected": _metric_summary(policies["selected"], qpo=True),
        }

    comparisons = {}
    for alias, result in results.items():
        main10m = result["evaluations"]["main10m"]
        comparisons[alias] = {
            "primary_compute_matched_10m_endpoint": _metric_summary(
                main10m["endpoint"]),
            "secondary_validation_selected": _metric_summary(
                main10m["selected"]),
            "early_0_2m": result["early_0_2m"],
        }
    return {
        "return_threshold": config.threshold,
        "prior_q0": config.q0,
        "primary_rule": "compute_matched_10m_endpoint",
        "secondary_rule": "validation_selected",
        "comparisons": comparisons,
    }


def _result_reference(spec):
    """Hash one raw exact JSON so the tracked campaign file stays compact and tamper-evident."""
    return {
        "path": str(spec.raw_result_path.relative_to(BASE_DIR)),
        "sha256": core.sha256_file(spec.raw_result_path),
    }


def write_stage_manifest(config):
    """Atomically publish one small stage result only after every exact/sync gate passes."""
    executable_sha = frozen_executable_sha(create=False)
    results = {}
    references = {}
    for alias, spec in config.specs.items():
        if not core.runner_complete(spec) or not core.verify_milestone(spec):
            return False
        if read_valid_sync_marker(spec, executable_sha) is None:
            return False
        if not spec.raw_result_path.exists():
            return False
        results[alias] = validate_exact_result(
            config, alias, spec, executable_sha)
        references[alias] = _result_reference(spec)

    status = {
        "qpo": "awaiting_user_q0",
        "initial": "awaiting_user_q1",
        "strict": "complete",
    }[config.stage]
    payload = {
        "schema": "dynamic_stage_manifest_v1",
        "campaign": CAMPAIGN,
        "stage_campaign": config.campaign,
        "stage": config.stage,
        "status": status,
        "executable_git_sha": executable_sha,
        "environment": ENVIRONMENT,
        "seed": SEED,
        "return_threshold": config.threshold,
        "prior_q0": config.q0,
        "common_setting": {
            "num_envs": NUM_ENVS,
            "horizon": HORIZON,
            "gamma": core.GAMMA,
            "failure_alpha": core.FAILURE_ALPHA,
            "actor_lr0": core.ACTOR_LR,
            "main_batches": FINAL_BATCH,
            "main_env_steps": MAIN_ENV_STEPS,
            "rolling_checkpoint_every_batches": CHECKPOINT_EVERY,
            "milestone_batch": MILESTONE_BATCH,
            "validation": {
                "exact_n": core.VALIDATION_EPISODES,
                "seed": core.VALIDATION_SEED,
                "every_batches": VALIDATION_EVERY,
            },
            "posthoc": {
                "exact_n": core.FINAL_EVALUATION_EPISODES,
                "seed": core.FINAL_EVALUATION_SEED,
                "budgets": ["main5m", "main10m"],
                "roles": ["endpoint", "selected"],
            },
        },
        "physical_initialization_steps": {
            alias: spec.initialization_env_steps
            for alias, spec in config.specs.items()
        },
        "summary": stage_summary(config, results),
        "raw_results": references,
    }
    core.write_json_atomic(config.stage_result_path, payload)
    write_tracked_manifest(config, payload)
    return True


def _read_stage_manifest(config):
    """Load and hash-check one completed stage's compact result."""
    try:
        payload = json.loads(
            config.stage_result_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError) as error:
        raise RuntimeError(
            f"{config.stage}: completed stage manifest is required") from error
    if payload.get("campaign") != CAMPAIGN or payload.get("stage") != config.stage:
        raise RuntimeError(f"{config.stage}: stage manifest identity mismatch")
    if payload.get("return_threshold") != config.threshold:
        raise RuntimeError(f"{config.stage}: recorded threshold differs from the request")
    if payload.get("prior_q0") != config.q0:
        raise RuntimeError(f"{config.stage}: recorded q0 differs from the request")
    executable_sha = frozen_executable_sha(create=False)
    if payload.get("executable_git_sha") != executable_sha:
        raise RuntimeError(f"{config.stage}: stage manifest executable SHA mismatch")
    for alias, reference in (payload.get("raw_results") or {}).items():
        if alias not in config.specs:
            raise RuntimeError(f"{config.stage}: unexpected raw result alias {alias}")
        path = BASE_DIR / reference["path"]
        if core.sha256_file(path) != reference["sha256"]:
            raise RuntimeError(f"{config.stage}/{alias}: raw exact result hash mismatch")
    return payload


def validate_prerequisites(config, require_completed=True):
    """Enforce QPO -> user q0 -> initial -> user q1 -> strict without auto-selecting q."""
    if not require_completed or config.stage == "qpo":
        return
    qpo_config = make_stage_config("qpo")
    qpo_manifest = _read_stage_manifest(qpo_config)
    if qpo_manifest.get("status") != "awaiting_user_q0":
        raise RuntimeError("QPO threshold-selection gate is not complete")
    if config.stage == "strict":
        initial_config = make_stage_config(
            "initial", return_threshold=config.q0)
        initial_manifest = _read_stage_manifest(initial_config)
        if initial_manifest.get("status") != "awaiting_user_q1":
            raise RuntimeError("initial constrained comparison gate is not complete")
        if float(initial_manifest["return_threshold"]) != float(config.q0):
            raise RuntimeError("strict --q0 does not match the recorded initial threshold")


def write_tracked_manifest(config, current_stage_payload):
    """Atomically update one small tracked JSON with summaries and raw-result hash references."""
    qpo_config = make_stage_config("qpo")
    stages = {}
    if qpo_config.stage_result_path.exists():
        stages["qpo"] = _read_stage_manifest(qpo_config)
    if config.stage in ("initial", "strict"):
        initial_threshold = config.threshold if config.stage == "initial" else config.q0
        initial_config = make_stage_config(
            "initial", return_threshold=initial_threshold)
        if initial_config.stage_result_path.exists():
            stages["initial"] = _read_stage_manifest(initial_config)
    if config.stage == "strict":
        stages["strict"] = current_stage_payload

    # Keep only small summaries/references rather than embedding the full exact-N metric documents.
    compact_stages = {
        name: {
            "stage": payload["stage"],
            "status": payload["status"],
            "return_threshold": payload["return_threshold"],
            "prior_q0": payload["prior_q0"],
            "summary": payload["summary"],
            "raw_results": payload["raw_results"],
        }
        for name, payload in stages.items()
    }
    tracked = {
        "schema": "dynamic_seed0_campaign_v1",
        "campaign": CAMPAIGN,
        "executable_git_sha": frozen_executable_sha(create=False),
        "environment": ENVIRONMENT,
        "seed": SEED,
        "state_cdf_parent_decision": STATE_CDF_DECISION,
        "experiment_claim_limit": (
            "Dynamic seed 0 engineering evidence only; no cross-seed significance or universal "
            "algorithm ranking"),
        "reward_constraint": "P(G_1000 <= q) <= 0.20",
        "cost_threshold_note": (
            "paper d_th=15 is a cost threshold and is not used as reward-side q"),
        "stages": compact_stages,
    }
    core.write_json_atomic(TRACKED_MANIFEST, tracked)


def verify_stage(config):
    """Require endpoint=399, milestone=199, sync=400 rows, exact hashes, and stage manifest."""
    executable_sha = frozen_executable_sha(create=False)
    for alias, spec in config.specs.items():
        validate_resume_lineage(config, alias, spec, executable_sha)
        if not core.runner_complete(spec):
            raise RuntimeError(f"{alias}: endpoint iteration 399/result is incomplete")
        if not core.verify_milestone(spec):
            raise RuntimeError(f"{alias}: 5M milestone iteration 199 is incomplete")
        _audit_runner_result(spec)
        if read_valid_sync_marker(spec, executable_sha) is None:
            raise RuntimeError(f"{alias}: W&B sync/400-row audit is incomplete")
        validate_exact_result(config, alias, spec, executable_sha)
    payload = _read_stage_manifest(config)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


# ============================================================================ phase orchestration and CLI
def _child_command(config, action, alias, device, allow_resume=False):
    """Re-enter this wrapper with one variant so phase-one DQC/QREF can run concurrently."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        action,
        "--stage", config.stage,
        "--variant", alias,
        "--device", str(device),
        "--wandb-mode", "online",
        "--child",
    ]
    if config.threshold is not None:
        command.extend(("--return-threshold", format(config.threshold, ".17g")))
    if config.q0 is not None:
        command.extend(("--q0", format(config.q0, ".17g")))
    if allow_resume:
        command.append("--allow-resume")
    return command


def run_phase(config, aliases, device, allow_resume):
    """Run every alias in a phase concurrently and preserve valid peers when one fails."""
    processes = {}
    for alias in aliases:
        command = _child_command(
            config, "run", alias, device, allow_resume=allow_resume)
        process = subprocess.Popen(command, cwd=BASE_DIR)
        processes[alias] = process
        print(f"[phase started] alias={alias} pid={process.pid}", flush=True)
    failures = {}
    for alias, process in processes.items():
        return_code = process.wait()
        if return_code != 0:
            failures[alias] = return_code
        else:
            print(f"[phase complete] alias={alias}", flush=True)
    if failures:
        raise RuntimeError(f"{config.stage}: phase failed: {failures}")


def run_stage(config, device, allow_resume=False):
    """Execute QPO alone or DQC/QREF then QCPO, preserving all human threshold gates."""
    validate_prerequisites(config, require_completed=True)
    if not allow_resume:
        # Freshness is checked before the QPO launch creates the campaign-wide SHA marker.
        assert_fresh_stage(config)
    executable_sha = frozen_executable_sha(create=(config.stage == "qpo"))
    if allow_resume:
        # The SHA check occurs before launching any child. Individual children then validate their
        # contract/signature or create a still-absent later-phase lineage from fresh parameters.
        if executable_sha != core.git_sha():
            raise RuntimeError("current executable SHA differs from the frozen campaign")

    for phase in config.phases:
        run_phase(config, phase, device, allow_resume)
        for alias in phase:
            if not config.specs[alias].raw_result_path.exists():
                raise RuntimeError(f"{alias}: exact result missing after successful child")
    if not write_stage_manifest(config):
        raise RuntimeError(f"{config.stage}: atomic stage manifest gate remained incomplete")
    payload = _read_stage_manifest(config)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)


def dry_run_stage(config, device, wandb_mode):
    """Print the true phase order and exact runner commands without creating any artifact."""
    for phase in config.phases:
        print(f"[dry phase] {phase}")
        for alias in phase:
            spec = config.specs[alias]
            command = core.build_training_command(
                spec, device, wandb_mode)
            print(" ".join(command))


def build_parser():
    """Construct the CLI separately so conditional stage failures are unit-testable."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("run", "dry-run", "verify", "evaluate"))
    parser.add_argument(
        "--stage", required=True, choices=("qpo", "initial", "strict"))
    parser.add_argument(
        "--return-threshold", "--return_threshold", type=float, default=None,
        dest="return_threshold",
        help="required q0 for initial and q1 for strict; forbidden for QPO")
    parser.add_argument(
        "--q0", type=float, default=None,
        help="strict only: previously confirmed initial threshold")
    parser.add_argument("--variant", choices=("qpo", "dqc", "qref", "qcpo"))
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--wandb-mode", "--wandb_mode",
        default="online", choices=("online", "offline", "disabled"),
        dest="wandb_mode")
    parser.add_argument(
        "--allow-resume", "--allow_resume", action="store_true",
        dest="allow_resume",
        help="recover only same env/algo/q/SHA/checkpoint-signature lineage")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return parser


def parse_cli(argv=None):
    """Parse and fail fast on threshold/variant/action combinations before any mutation."""
    parser = build_parser()
    cli = parser.parse_args(argv)
    try:
        config = make_stage_config(
            cli.stage,
            return_threshold=cli.return_threshold,
            q0=cli.q0)
    except ValueError as error:
        parser.error(str(error))
    if cli.allow_resume and cli.action != "run":
        parser.error("--allow-resume is valid only with action=run")
    if cli.variant is not None and cli.variant not in config.specs:
        parser.error(
            f"--variant {cli.variant} does not belong to stage {cli.stage}")
    if cli.child and cli.variant is None:
        parser.error("--child requires --variant")
    if cli.action == "run" and cli.wandb_mode != "online":
        parser.error("formal run is online-only so explicit W&B sync can be verified")
    return cli, config


def main(argv=None):
    """Dispatch a mutation-free dry-run, formal stage, exact re-evaluation, or verification."""
    cli, config = parse_cli(argv)
    validate_registered_identity(config)
    with installed_core_identity(config):
        if cli.action == "dry-run":
            dry_run_stage(config, cli.device, cli.wandb_mode)
            return

        if cli.child:
            # Parent workers never publish the shared stage manifest concurrently.
            run_one(
                config, cli.variant, cli.device,
                allow_resume=cli.allow_resume)
            return

        if cli.action == "run":
            run_stage(
                config, cli.device,
                allow_resume=cli.allow_resume)
            return

        validate_prerequisites(config, require_completed=True)
        aliases = (
            (cli.variant,) if cli.variant is not None
            else tuple(config.specs)
        )
        if cli.action == "evaluate":
            for alias in aliases:
                evaluate_one(config, alias, cli.device)
            if not write_stage_manifest(config):
                raise RuntimeError(
                    f"{config.stage}: stage manifest gate remained incomplete")
            return

        verify_stage(config)


if __name__ == "__main__":
    main()
