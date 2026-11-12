#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sequential seed-0 QCPO_REF discounted-occupancy and DQC GAE-Mean campaign.

Each invocation owns one training subprocess, watches the atomic batch-200 checkpoint, evaluates
5M/10M endpoint and selected policies under identical exact-N=1000 CRN, and writes a compact raw
result. The caller starts the DQC invocation only after the QCPO_REF invocation returns success.
"""
import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


# ============================================================================ frozen campaign identity
BASE_DIR = Path(__file__).resolve().parent
RUNNER = BASE_DIR / "run_experiment.py"
CAMPAIGN = "qcpo_ref_discounted_dqc_gae_seed0_v1"
ENVIRONMENT = "SimpleButton"
SEED = 0
ACTOR_LR = 2e-4
RETURN_THRESHOLD = -0.03
FAILURE_ALPHA = 0.2
GAMMA = 0.99
NUM_ENVS = 25
HORIZON = 1000
MILESTONE_BATCH = 200
FINAL_BATCH = 400
VALIDATION_EPISODES = 100
FINAL_EVALUATION_EPISODES = 1000
VALIDATION_SEED = 555
FINAL_EVALUATION_SEED = 777

RAW_ROOT = BASE_DIR / "_runs" / CAMPAIGN
TRACKED_MANIFEST = BASE_DIR / "campaigns" / CAMPAIGN / "manifest.json"


@dataclass(frozen=True)
class FormalSpec:
    """All immutable identifiers for one of the two sequential formal lineages."""

    key: str
    algorithm: str
    wandb_id: str
    wandb_group: str
    wandb_tag: str
    tmux_name: str
    parent_policy_identity: str
    policy_identity: str
    variant_changes: str
    initialization_batches: int
    wandb_name: str = ""
    # New campaigns may keep the reusable 5M watcher while retaining more frequent rolling
    # recovery points. Existing campaigns default to their historical batch-200 schedule.
    checkpoint_every: int = MILESTONE_BATCH
    # QPO deliberately trains without a fixed return threshold; constrained algorithms retain the
    # old required --return_threshold command-line contract by default.
    uses_return_threshold: bool = True
    # Variant-specific signed overrides keep formal wrappers out of agent/loss implementation.
    # Tuples make the frozen dataclass identity immutable and straightforward to audit in tests.
    extra_overrides: tuple = ()

    @property
    def checkpoint_root(self):
        """Runner-owned live files plus watcher-owned immutable 5M files."""
        return RAW_ROOT / "checkpoints" / self.key

    @property
    def live_checkpoint_dir(self):
        """Rolling latest/best files written at formal batches 200 and 400."""
        return self.checkpoint_root / "live"

    @property
    def milestone_checkpoint_dir(self):
        """Immutable batch-200 endpoint and then-current selected policy."""
        return self.checkpoint_root / "main5m"

    @property
    def progress_path(self):
        """Atomic status row emitted only by formal training batches."""
        return RAW_ROOT / "progress" / f"{self.key}.json"

    @property
    def log_path(self):
        """Combined stdout/stderr for the child runner."""
        return RAW_ROOT / "logs" / f"{self.key}.log"

    @property
    def raw_result_path(self):
        """Four-policy post-hoc result used to build the tracked manifest."""
        return RAW_ROOT / "evaluations" / f"{self.key}.json"

    @property
    def runner_tag(self):
        """Stable short runner tag; W&B names and IDs remain separately explicit."""
        return self.wandb_id

    @property
    def runner_result_path(self):
        """Result location fixed by run_experiment output naming."""
        filename = (
            f"{self.algorithm}_{ENVIRONMENT}_{self.runner_tag}_s{SEED}.json")
        return BASE_DIR / "_runs" / filename

    @property
    def initialization_env_steps(self):
        """Physical interactions excluded from the advertised formal main budget."""
        return self.initialization_batches * NUM_ENVS * HORIZON


SPECS = {
    "qref": FormalSpec(
        key="qrefd2a",
        algorithm="QCPO_REF",
        wandb_id="qrefd2a",
        wandb_group="qcpo_ref_return_quantile_discounted_v2",
        wandb_tag="qcpo_ref_paper_negret_eq275_discounted_bh_v2_lr2e4_main10m",
        tmux_name="qref_discocc_lr2e4_10m_s0",
        parent_policy_identity="uniform_time_v1",
        policy_identity="paper_negated_return_eq275_discounted_bh_v2",
        variant_changes=(
            "outer_gamma_t_on_complete_combined_advantage;"
            "fixed_BH_actor_reduction;main_steps_exclude_initialization"),
        initialization_batches=0,
    ),
    "dqc": FormalSpec(
        key="dqcg2m1a",
        algorithm="DQCACBeta",
        wandb_id="dqcg2m1a",
        wandb_group="dqc_gae_mean_simplebutton_v1",
        wandb_tag=(
            "dqc_finite_n100_gae_mean_v1_ppo_clip0p1_"
            "tkl0p01_c8a8_lr2e4_main10m"),
        tmux_name="dqc_gae_mean_lr2e4_10m_s0",
        parent_policy_identity="finite_nstep_v1_distributional_action_advantage_v1",
        policy_identity="finite_nstep_v1_state_value_gae_v1",
        variant_changes=(
            "independent_scalar_state_value_gae_lambda0p97;"
            "distributional_critic_cdf_only;combined_population_std;"
            "main_steps_exclude_critic_initialization"),
        initialization_batches=10,
    ),
}


# ============================================================================ atomic artifact helpers
def git_sha():
    """Read the committed code identity without touching the user's unrelated changes."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=BASE_DIR, text=True).strip()


def write_json_atomic(path, payload):
    """Replace a small JSON document atomically on its destination filesystem."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        destination.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    os.replace(temporary, destination)


def sha256_file(path):
    """Stream one checkpoint into SHA256 without holding its large tensor payload twice."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def checkpoint_iteration(path):
    """Read checkpoint iteration metadata, returning None while the atomic file is absent."""
    checkpoint = Path(path)
    if not checkpoint.exists():
        return None
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return int(payload["iteration"])


def atomic_copy(source, destination):
    """Copy an already-atomic checkpoint through a destination-side temporary file."""
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        destination.name + f".{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def verify_milestone(spec):
    """Validate batch numbers and hashes for an existing READY milestone."""
    ready_path = spec.milestone_checkpoint_dir / "READY"
    metadata_path = spec.milestone_checkpoint_dir / "hashes.json"
    if not ready_path.exists() or not metadata_path.exists():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("formal_batch", -1)) != MILESTONE_BATCH:
        raise RuntimeError(f"{spec.key}: READY metadata has the wrong formal batch")
    if int(metadata.get("endpoint_iteration", -1)) != MILESTONE_BATCH - 1:
        raise RuntimeError(f"{spec.key}: READY metadata has the wrong endpoint iteration")
    if set(metadata.get("sha256") or {}) != {"latest.pt", "best.pt"}:
        raise RuntimeError(f"{spec.key}: READY metadata has an incomplete checkpoint hash set")
    for name, expected_hash in metadata["sha256"].items():
        checkpoint = spec.milestone_checkpoint_dir / name
        if sha256_file(checkpoint) != expected_hash:
            raise RuntimeError(f"{spec.key}: 5M hash mismatch for {name}")
    if checkpoint_iteration(
            spec.milestone_checkpoint_dir / "latest.pt") != MILESTONE_BATCH - 1:
        raise RuntimeError(f"{spec.key}: READY endpoint is not formal batch 200")
    selected_iteration = checkpoint_iteration(
        spec.milestone_checkpoint_dir / "best.pt")
    if selected_iteration != int(metadata.get("selected_iteration", -1)):
        raise RuntimeError(
            f"{spec.key}: READY selected iteration differs from hashes metadata")
    if selected_iteration is None or not 0 <= selected_iteration < MILESTONE_BATCH:
        raise RuntimeError(f"{spec.key}: READY selected checkpoint is outside batch 200")
    return True


def capture_five_million(spec):
    """
    Freeze exact formal batch 200, write both hashes, then publish READY last.

    If live latest has already advanced to batch 400 without READY, fail rather than mislabeling a
    10M policy as 5M. Repeated calls are idempotent and re-verify every durable hash.
    """
    if verify_milestone(spec):
        return True
    live_latest = spec.live_checkpoint_dir / "latest.pt"
    live_iteration = checkpoint_iteration(live_latest)
    if live_iteration is None or live_iteration < MILESTONE_BATCH - 1:
        return False
    if live_iteration != MILESTONE_BATCH - 1:
        raise RuntimeError(
            f"{spec.key}: lost exact 5M checkpoint before watcher capture")

    live_best = spec.live_checkpoint_dir / "best.pt"
    selected_iteration = checkpoint_iteration(live_best)
    if selected_iteration is None or selected_iteration > MILESTONE_BATCH - 1:
        raise RuntimeError(f"{spec.key}: invalid batch-200 selected checkpoint")

    frozen_latest = spec.milestone_checkpoint_dir / "latest.pt"
    frozen_best = spec.milestone_checkpoint_dir / "best.pt"
    atomic_copy(live_latest, frozen_latest)
    atomic_copy(live_best, frozen_best)
    metadata = {
        "formal_batch": MILESTONE_BATCH,
        "endpoint_iteration": MILESTONE_BATCH - 1,
        "selected_iteration": selected_iteration,
        "sha256": {
            "latest.pt": sha256_file(frozen_latest),
            "best.pt": sha256_file(frozen_best),
        },
    }
    write_json_atomic(
        spec.milestone_checkpoint_dir / "hashes.json", metadata)
    write_json_atomic(
        spec.milestone_checkpoint_dir / "READY",
        {"ready": True, "hashes_file": "hashes.json"})
    return verify_milestone(spec)


def read_progress(path):
    """Return one atomic progress snapshot; absence is normal before formal batch one."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return None


def assert_finite_progress(spec, progress):
    """Pause only on non-finite metrics; slow but finite learning remains allowed."""
    if not progress:
        return
    for section in (
            progress.get("metrics") or {},
            progress.get("validation") or {}):
        for name, value in section.items():
            if value is not None and not math.isfinite(float(value)):
                raise RuntimeError(
                    f"{spec.key}: non-finite progress metric {name}={value!r}")


# ============================================================================ one formal training command
def build_training_command(spec, device, wandb_mode, resume=None):
    """Build the exact 400-main-batch runner invocation for one registered identity."""
    main_steps = FINAL_BATCH * NUM_ENVS * HORIZON
    actual_steps = main_steps + spec.initialization_env_steps
    tags = ",".join((
        "reward-side", "G1000", "alpha-0.2", spec.wandb_tag,
    ))
    overrides = [
        f"actor_lr0={ACTOR_LR:.17g}",
        f"num_iterations={FINAL_BATCH}",
        f"git_sha={git_sha()}",
        f"parent_policy_identity={spec.parent_policy_identity}",
        f"policy_identity={spec.policy_identity}",
        f"variant_changes={spec.variant_changes}",
        "training_budget_semantics=main_excludes_initialization_v1",
        f"main_rollout_batches={FINAL_BATCH}",
        f"main_env_steps={main_steps}",
        f"initialization_env_steps={spec.initialization_env_steps}",
        f"actual_env_steps={actual_steps}",
        "evaluation_protocol=exact_n100_validation_seed555_test_seed777_crn_v1",
        f"wandb_group={spec.wandb_group}",
        # Stable ID controls resume; the optional display name is only human-facing.
        f"wandb_name={spec.wandb_name or spec.wandb_id}",
        f"wandb_tags={tags}",
        f"progress_file={spec.progress_path}",
        "show_progress=false",
        "show_eval_progress=false",
    ]
    if spec.algorithm == "DQCACBeta":
        overrides.extend((
            "dqc_update_mode=finite_nstep_v1",
            "dqc_mean_estimator=state_value_gae_v1",
        ))
    overrides.extend(str(item) for item in spec.extra_overrides)

    command = [
        sys.executable, str(RUNNER),
        "--algo", spec.algorithm,
        "--env", ENVIRONMENT,
        "--seed", str(SEED),
        "--device", str(device),
        "--num_eval", str(FINAL_EVALUATION_EPISODES),
        "--validation_episodes", str(VALIDATION_EPISODES),
        "--validation_every", "10",
        "--checkpoint_every", str(spec.checkpoint_every),
        "--checkpoint_dir", str(spec.live_checkpoint_dir),
        "--wandb_mode", str(wandb_mode),
        "--tag", spec.runner_tag,
    ]
    if spec.uses_return_threshold:
        if RETURN_THRESHOLD is None:
            raise ValueError(
                f"{spec.key}: constrained formal lineage requires a return threshold")
        # Insert the threshold before evaluation options so QPO can omit both tokens entirely.
        threshold_index = command.index("--num_eval")
        command[threshold_index:threshold_index] = [
            "--return_threshold", str(RETURN_THRESHOLD)]
    if resume is not None:
        command.extend(("--resume", str(resume)))
    command.append("--set")
    command.extend(overrides)
    return command


def runner_complete(spec):
    """Require endpoint iteration 399 plus the runner's post-training JSON."""
    return (
        checkpoint_iteration(spec.live_checkpoint_dir / "latest.pt")
        == FINAL_BATCH - 1
        and spec.runner_result_path.exists()
    )


def resume_checkpoint(spec):
    """Resume only the same lineage's signature-validated rolling latest checkpoint."""
    latest = spec.live_checkpoint_dir / "latest.pt"
    iteration = checkpoint_iteration(latest)
    if iteration is not None:
        # An interruption may occur after endpoint checkpoint 399 but before the runner JSON,
        # final selected-policy test, or W&B finish. Re-entering from endpoint state performs no
        # additional main update and completes those trailing artifacts without fresh retraining.
        if iteration >= FINAL_BATCH - 1:
            return latest if not spec.runner_result_path.exists() else None
        return latest

    # QPO and DQC both have save-only initialization batches. A machine interruption before the
    # first main checkpoint must resume that signed cursor instead of silently recollecting warmup.
    initialization = spec.live_checkpoint_dir / "initialization_latest.pt"
    return initialization if checkpoint_iteration(initialization) == -1 else None


def run_training(spec, device, wandb_mode, dry_run=False):
    """Train one lineage while synchronously watching NaNs, exit status, and the 5M boundary."""
    captured = capture_five_million(spec)
    resume = resume_checkpoint(spec)
    command = build_training_command(
        spec, device, wandb_mode, resume=resume)
    if dry_run:
        print(" ".join(command))
        return
    if runner_complete(spec):
        if not captured:
            capture_five_million(spec)
        print(f"[skip complete] {spec.key}", flush=True)
        return

    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    spec.live_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({
        "WANDB_RUN_ID": spec.wandb_id,
        "WANDB_RESUME": "allow",
        "PYTHONUNBUFFERED": "1",
    })
    with spec.log_path.open("a", encoding="utf-8") as stream:
        stream.write("\n[formal command] " + " ".join(command) + "\n")
        stream.flush()
        process = subprocess.Popen(
            command, cwd=BASE_DIR, stdout=stream,
            stderr=subprocess.STDOUT, env=environment)
        print(f"[started] pid={process.pid} {spec.key}", flush=True)
        while True:
            progress = read_progress(spec.progress_path)
            try:
                assert_finite_progress(spec, progress)
                completed = int((progress or {}).get("completed", 0))
                if not captured and completed >= MILESTONE_BATCH:
                    captured = capture_five_million(spec)
            except Exception:
                process.terminate()
                process.wait(timeout=30)
                raise

            return_code = process.poll()
            if return_code is not None:
                if return_code != 0:
                    raise RuntimeError(
                        f"{spec.key} exited with {return_code}; see {spec.log_path}")
                break
            time.sleep(1.0)

    if not captured:
        captured = capture_five_million(spec)
    if not captured or not runner_complete(spec):
        raise RuntimeError(f"{spec.key}: incomplete formal artifacts after runner exit")


# ============================================================================ exact common-random-number post-hoc evaluation
def evaluation_args(spec, device):
    """Reconstruct the exact signed training configuration without creating a new lineage."""
    from run_experiment import base_args, cast, configure_dqc_identity

    args = base_args(spec.algorithm, SEED, device, ENVIRONMENT)
    args.return_threshold = RETURN_THRESHOLD
    args.actor_lr0 = ACTOR_LR
    args.theta_a = (args.theta_b ** args.theta_c) * ACTOR_LR
    args.ref_lr = ACTOR_LR
    args.actor_lr = ACTOR_LR
    args.num_iterations = FINAL_BATCH
    args.show_progress = False
    args.show_eval_progress = False
    args.wandb_group = CAMPAIGN + "_posthoc_disabled"
    args.wandb_name = "posthoc_" + spec.key
    explicit = {}
    if spec.algorithm == "DQCACBeta":
        explicit.update({
            "dqc_update_mode": "finite_nstep_v1",
            "dqc_mean_estimator": "state_value_gae_v1",
        })
    for assignment in spec.extra_overrides:
        if "=" not in str(assignment):
            raise ValueError(
                f"{spec.key}: invalid extra override {assignment!r}; expected key=value")
        name, raw_value = str(assignment).split("=", 1)
        explicit[name] = cast(raw_value)
    for name, value in explicit.items():
        setattr(args, name, value)
    if spec.algorithm == "DQCACBeta":
        configure_dqc_identity(args, explicit)
    return args


def reset_evaluation_rng(seed):
    """Reset every action-sampling RNG before each policy role for strict CRN."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def policy_checkpoints(spec):
    """Enumerate 5M/10M endpoint and selected checkpoints in fixed order."""
    return {
        "main5m": {
            "batch": MILESTONE_BATCH,
            "endpoint": spec.milestone_checkpoint_dir / "latest.pt",
            "selected": spec.milestone_checkpoint_dir / "best.pt",
        },
        "main10m": {
            "batch": FINAL_BATCH,
            "endpoint": spec.live_checkpoint_dir / "latest.pt",
            "selected": spec.live_checkpoint_dir / "best.pt",
        },
    }


def evaluate_lineage(spec, device):
    """Strictly load and evaluate four policies, then atomically write the raw result."""
    from agents import DQCACBetaGPU, QCPOGPU, QCPORefGPU, QPOGPU
    from envs import SafetyEnv, make_vec_env
    from run_experiment import checkpoint_signature
    from utils import evaluate_policy_vec, load_checkpoint

    os.environ["WANDB_MODE"] = "disabled"
    classes = {
        "QPO": QPOGPU,
        "QCPO": QCPOGPU,
        "QCPO_REF": QCPORefGPU,
        "DQCACBeta": DQCACBetaGPU,
    }
    args = evaluation_args(spec, device)
    env = SafetyEnv(args.env_id)
    agent = classes[spec.algorithm](args, env)
    agent.checkpoint_signature = checkpoint_signature(args, spec.algorithm)
    evaluations = {}
    try:
        for setting, checkpoint_group in policy_checkpoints(spec).items():
            batch = int(checkpoint_group["batch"])
            setting_result = {}
            for role in ("endpoint", "selected"):
                checkpoint = checkpoint_group[role]
                payload = load_checkpoint(
                    checkpoint, agent, map_location=device,
                    restore_rng=False, allow_signature_mismatch=False)
                iteration = int(payload["iteration"])
                if role == "endpoint" and iteration != batch - 1:
                    raise RuntimeError(
                        f"{spec.key}/{setting}: wrong endpoint iteration {iteration}")
                if role == "selected" and not 0 <= iteration < batch:
                    raise RuntimeError(
                        f"{spec.key}/{setting}: selected iteration {iteration} out of range")

                reset_evaluation_rng(FINAL_EVALUATION_SEED)
                evaluation_env = make_vec_env(
                    args.env_id, NUM_ENVS, HORIZON, device,
                    ref_env=env, seed=FINAL_EVALUATION_SEED,
                    backend=args.vec_backend)
                try:
                    metrics = evaluate_policy_vec(
                        agent, evaluation_env,
                        FINAL_EVALUATION_EPISODES,
                        GAMMA, FAILURE_ALPHA, RETURN_THRESHOLD,
                        show_progress=False,
                        progress_desc=f"{spec.key}/{setting}/{role}")
                finally:
                    evaluation_env.close()
                metrics["return_variance"] = float(
                    metrics["reward_std"] ** 2)
                setting_result[role] = {
                    "checkpoint": str(checkpoint),
                    "checkpoint_iteration": iteration,
                    "sha256": sha256_file(checkpoint),
                    "metrics": metrics,
                    "step_clocks": {
                        "main_env_steps": int(agent.main_env_steps),
                        "initialization_env_steps": int(
                            agent.initialization_env_steps),
                        "actual_env_steps": int(agent.actual_env_steps),
                    },
                }
            evaluations[setting] = setting_result
    finally:
        agent.vec_env.close()
        env.close()
        wandb_run = getattr(agent, "_wandb_run", None)
        if wandb_run is not None:
            wandb_run.finish()

    runner_result = json.loads(
        spec.runner_result_path.read_text(encoding="utf-8"))
    output = {
        "campaign": CAMPAIGN,
        "git_sha": git_sha(),
        "environment": ENVIRONMENT,
        "algorithm": spec.algorithm,
        "seed": SEED,
        "wandb_id": spec.wandb_id,
        "wandb_group": spec.wandb_group,
        "wandb_tag": spec.wandb_tag,
        "parent_policy_identity": spec.parent_policy_identity,
        "policy_identity": spec.policy_identity,
        "variant_changes": spec.variant_changes,
        "training_budget_semantics": "main_excludes_initialization_v1",
        "evaluation_protocol": {
            "name": "exact_n1000_common_random_numbers_v1",
            "num_episodes": FINAL_EVALUATION_EPISODES,
            "environment_seed": FINAL_EVALUATION_SEED,
            "action_rng_seed": FINAL_EVALUATION_SEED,
            "checkpoint_rng_restored": False,
        },
        "early_0_2m": summarize_early_history(runner_result),
        "evaluations": evaluations,
    }
    write_json_atomic(spec.raw_result_path, output)
    return output


def summarize_early_history(runner_result):
    """Compute pre-registered 0-2M validation AUC and DQC diagnostic means."""
    max_steps = 2_000_000
    validations = [
        row for row in runner_result.get("validation_history", [])
        if (int(row.get("iteration", -1)) + 1) * NUM_ENVS * HORIZON <= max_steps
    ]
    validation_steps = np.asarray([
        (int(row["iteration"]) + 1) * NUM_ENVS * HORIZON
        for row in validations
    ], dtype=np.float64)
    validation_means = np.asarray([
        float(row["mean"]) for row in validations
    ], dtype=np.float64)
    if validation_steps.size >= 2:
        auc = float(np.trapz(validation_means, validation_steps))
        normalized_auc = auc / float(validation_steps[-1] - validation_steps[0])
    else:
        auc = None
        normalized_auc = None

    diagnostics = [
        row for row in runner_result.get("training_diagnostics_history", [])
        if int(row.get("main_env_steps", 0)) <= max_steps
    ]

    def mean_field(name):
        """Average only rows that own the formulation-specific diagnostic."""
        values = [float(row[name]) for row in diagnostics if name in row]
        return None if not values else float(np.mean(values))

    calibrations = [
        row["diagnostic_reward_critic_calibration"]
        for row in validations
        if row.get("diagnostic_reward_critic_calibration") is not None
    ]
    return {
        "validation_points": int(validation_steps.size),
        "validation_mean_auc_env_steps": auc,
        "validation_mean_auc_normalized": normalized_auc,
        "policy_kl_after_epochs_mean": mean_field(
            "diagnostic/dqc/policy_kl_after_epochs"),
        "clip_fraction_mean": mean_field("actor/clip_fraction"),
        "value_rmse_mean": mean_field("diagnostic/dqc/value_rmse"),
        "critic_cdf_mean": mean_field("diagnostic/dqc/critic_cdf_mean"),
        "cdf_initial_abs_gap_mean": (
            None if not calibrations else float(np.mean([
                item["initial_cdf_abs_gap"] for item in calibrations]))),
        "cdf_brier_mean": (
            None if not calibrations else float(np.mean([
                item["brier_score"] for item in calibrations]))),
    }


def write_tracked_manifest_if_complete():
    """Publish one small JSON only after both sequential lineages pass post-hoc evaluation."""
    missing = [
        spec.raw_result_path for spec in SPECS.values()
        if not spec.raw_result_path.exists()
    ]
    if missing:
        return False
    results = {
        alias: json.loads(spec.raw_result_path.read_text(encoding="utf-8"))
        for alias, spec in SPECS.items()
    }
    payload = {
        "campaign": CAMPAIGN,
        "git_sha": git_sha(),
        "formal_order": ["qref", "dqc"],
        "common_setting": {
            "seed": SEED,
            "environment": ENVIRONMENT,
            "actor_lr0": ACTOR_LR,
            "return_threshold": RETURN_THRESHOLD,
            "failure_alpha": FAILURE_ALPHA,
            "gamma": GAMMA,
            "num_envs": NUM_ENVS,
            "horizon": HORIZON,
            "main_batches": FINAL_BATCH,
            "main_env_steps": FINAL_BATCH * NUM_ENVS * HORIZON,
            "validation": {
                "exact_n": VALIDATION_EPISODES,
                "crn_seed": VALIDATION_SEED,
                "every_main_batches": 10,
            },
            "posthoc": {
                "exact_n": FINAL_EVALUATION_EPISODES,
                "crn_seed": FINAL_EVALUATION_SEED,
                "roles": ["endpoint", "selected"],
                "budgets": ["main5m", "main10m"],
            },
        },
        "legacy_runs": {
            "qref2e4a": {
                "policy_identity": "uniform_time_v1",
            },
            "dqcp8k1a": {
                "policy_identity": "distributional_action_advantage_v1",
                "budget_note": (
                    "400 physical batches included 10 warmup batches; "
                    "390 batches and 9.75M steps updated the actor"),
            },
        },
        "seed0_engineering_gate_only": True,
        "claim_rule": (
            "Only add seeds 1 and 2 if DQC endpoint outage is not worse than 0.2 "
            "and its endpoint mean and q20 exceed QCPO_REF"),
        "results": results,
    }
    write_json_atomic(TRACKED_MANIFEST, payload)
    return True


def main():
    """Run, dry-run, or post-hoc evaluate one registered formal identity."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "dry-run", "evaluate", "verify"))
    parser.add_argument("--variant", required=True, choices=tuple(SPECS))
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--wandb_mode", default="online",
        choices=("online", "offline", "disabled"))
    cli = parser.parse_args()
    spec = SPECS[cli.variant]
    device = torch.device(
        "cpu" if cli.device == "cpu"
        else f"cuda:{cli.device}" if torch.cuda.is_available()
        else "cpu")

    if cli.action == "dry-run":
        run_training(spec, cli.device, cli.wandb_mode, dry_run=True)
        return
    if cli.action == "verify":
        if not runner_complete(spec) or not verify_milestone(spec):
            raise RuntimeError(f"{spec.key}: formal artifacts are incomplete")
        print(f"{spec.key}: artifacts verified")
        return
    if cli.action == "run":
        run_training(spec, cli.device, cli.wandb_mode)

    # Both run and evaluate modes enforce all checkpoint/signature checks before writing results.
    evaluate_lineage(spec, device)
    write_tracked_manifest_if_complete()
    print(f"[formal complete] {spec.key} result={spec.raw_result_path}")


if __name__ == "__main__":
    main()
