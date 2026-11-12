#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter-only launcher and post-hoc evaluator for the SimpleButton joint sweep.

The harness never changes an agent, runner, checkpoint, or evaluation implementation.  Training is
delegated to ``run_experiment.py`` with explicit ``--set`` overrides.  Each lineage runs fresh to
10M main environment steps; an external watcher freezes the atomic batch-200 ``latest.pt`` and
``best.pt`` before the live files can be replaced at batch 400.  Only after every long run finishes
does post-hoc evaluation compare 5M/10M endpoint and validation-selected policies under one exact
N=1000 common-random-number stream.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import statistics
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
CAMPAIGN = "joint_sweep_simplebutton_v1"
ENVIRONMENT = "SimpleButton"
WANDB_GROUP = CAMPAIGN

# The user reduced the original grid to two actor learning rates.  Order is operationally fixed so
# reconnecting to the launcher always resumes the same next algorithm wave.
LEARNING_RATES = (("2e-4", 2e-4), ("4e-4", 4e-4))
ALGORITHMS = ("QPO", "QCPO", "DQCACBeta")
SEEDS = (0, 1, 2)

# One rollout contains B*H=25,000 main interactions.  Batch 200/400 therefore denotes 5M/10M.
NUM_ENVS = 25
HORIZON = 1000
MILESTONE_BATCH = 200
FINAL_BATCH = 400
RETURN_THRESHOLD = -0.03
FAILURE_ALPHA = 0.2
GAMMA = 0.99

# All post-hoc policy roles use the same environment reset seed and the same action-sampling RNG.
# It is independent of every training seed and of the runner's validation seed stream.
FINAL_EVALUATION_SEED = 2026080177
FINAL_EVALUATION_EPISODES = 1000
VALIDATION_EPISODES = 100

# Raw checkpoints/logs/evaluations stay ignored.  The compact manifest and CSV tables live outside
# ``results/`` because the repository intentionally ignores that generic directory name.
RAW_ROOT = BASE_DIR / "_runs" / CAMPAIGN
TRACKED_ROOT = BASE_DIR / "campaigns" / CAMPAIGN


@dataclass(frozen=True)
class RunSpec:
    """One 10M training lineage; the 5M result is a frozen checkpoint inside this same run."""

    lr_label: str
    actor_lr0: float
    algorithm: str
    seed: int

    @property
    def lineage_id(self):
        """Stable key shared by logs, checkpoints, progress, post-hoc JSON, and W&B tags."""
        return f"{ENVIRONMENT}_{self.lr_label}_{self.algorithm}_s{self.seed}"

    @property
    def training_setting(self):
        """W&B setting for the only actual training run in this lineage."""
        return f"lr{self.lr_label}_main10m"

    def result_setting(self, batch):
        """Map the frozen checkpoint batch to the setting name used by comparison tables."""
        main_steps = int(batch) * NUM_ENVS * HORIZON
        return f"lr{self.lr_label}_main{_step_label(main_steps)}"

    @property
    def live_checkpoint_dir(self):
        """Runner-owned rolling files; latest.pt is batch 200 until batch 400 replaces it."""
        return RAW_ROOT / "checkpoints" / self.lr_label / self.algorithm / f"s{self.seed}" / "live"

    @property
    def milestone_checkpoint_dir(self):
        """Watcher-owned immutable copies of the exact 5M endpoint and selected policies."""
        return RAW_ROOT / "checkpoints" / self.lr_label / self.algorithm / f"s{self.seed}" / "main5m"

    @property
    def log_path(self):
        """Combined child stdout/stderr, kept in the ignored campaign tree."""
        return RAW_ROOT / "logs" / f"{self.lineage_id}.log"

    @property
    def progress_path(self):
        """Atomic JSON status emitted by the unchanged VecAgentBase implementation."""
        return RAW_ROOT / "progress" / f"{self.lineage_id}.json"

    @property
    def evaluation_path(self):
        """Machine-readable four-policy post-hoc result produced only after long training."""
        return RAW_ROOT / "evaluations" / f"{self.lineage_id}.json"

    @property
    def runner_tag(self):
        """Unique runner tag; the runner additionally includes algorithm and seed in its JSON name."""
        return f"{CAMPAIGN}_{self.lr_label}_main10m_{self.algorithm}_s{self.seed}"

    @property
    def runner_result_path(self):
        """Location fixed by the unchanged run_experiment.py output naming convention."""
        filename = f"{self.algorithm}_{ENVIRONMENT}_{self.runner_tag}_s{self.seed}.json"
        return BASE_DIR / "_runs" / filename

    @property
    def wandb_id(self):
        """Deterministic eight-character ID permits W&B resume after a recoverable process failure."""
        digest = hashlib.sha1(f"{CAMPAIGN}:{self.lineage_id}".encode("utf-8")).hexdigest()
        return digest[:8]

    @property
    def policy_identity(self):
        """Human-facing identity; DQC's finite preset remains the algorithm's own strict signature."""
        if self.algorithm == "DQCACBeta":
            return "finite_nstep_v1"
        return self.algorithm.lower()


def _step_label(steps):
    """Render exact million-step budgets with the lowercase labels used by setting names."""
    steps = int(steps)
    if steps % 1_000_000 != 0:
        raise ValueError(f"main step count must be a whole million, got {steps}")
    return f"{steps // 1_000_000}m"


def enumerate_runs():
    """Return 18 lineages in LR -> algorithm wave -> seed order."""
    runs = []
    for lr_label, actor_lr0 in LEARNING_RATES:
        for algorithm in ALGORITHMS:
            for seed in SEEDS:
                runs.append(RunSpec(lr_label, actor_lr0, algorithm, seed))
    return runs


def git_provenance():
    """Resolve the current commit and dirty flag without mutating or staging the worktree."""
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=BASE_DIR, text=True).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=BASE_DIR, text=True).strip())
    return sha, dirty


def build_training_command(spec, device, wandb_mode, git_sha, git_dirty, resume=None):
    """
    Build one unchanged-runner invocation using only CLI and ``--set`` parameter overrides.

    Tracker/dual/critic rates are intentionally absent.  ``actor_lr0`` is the sole optimized
    variable; run_experiment.py already propagates it to theta_a/ref_lr/actor_lr as appropriate.
    """
    initialization_steps = 2 * NUM_ENVS * HORIZON if spec.algorithm == "QPO" else 0
    actual_steps = FINAL_BATCH * NUM_ENVS * HORIZON + initialization_steps

    # The runner's finite preset currently owns the DQC-only default W&B group/name.  We still pass
    # campaign values as explicit config/tags for every algorithm without changing preset logic.
    wandb_name = f"{ENVIRONMENT}_{spec.training_setting}_{spec.algorithm}_s{spec.seed}"
    tags = ",".join((
        "reward-side", "G1000", "alpha-0.2", CAMPAIGN,
        spec.training_setting, spec.algorithm, f"seed-{spec.seed}", spec.policy_identity,
    ))
    overrides = [
        f"actor_lr0={spec.actor_lr0:.17g}",
        f"num_iterations={FINAL_BATCH}",
        f"campaign={CAMPAIGN}",
        f"setting={spec.training_setting}",
        f"main_rollout_batches={FINAL_BATCH}",
        f"main_env_steps={FINAL_BATCH * NUM_ENVS * HORIZON}",
        f"initialization_env_steps={initialization_steps}",
        f"actual_env_steps={actual_steps}",
        f"policy_identity={spec.policy_identity}",
        "evaluation_protocol=posthoc_exact_n1000_crn_v1",
        f"git_sha={git_sha}",
        f"git_dirty={str(bool(git_dirty)).lower()}",
        f"wandb_group={WANDB_GROUP}",
        f"wandb_name={wandb_name}",
        f"wandb_tags={tags}",
        f"experiment_label={CAMPAIGN}",
        f"progress_file={spec.progress_path}",
        "show_progress=false",
        "show_eval_progress=false",
    ]
    if spec.algorithm == "DQCACBeta":
        overrides.append("dqc_update_mode=finite_nstep_v1")

    command = [
        sys.executable, str(RUNNER),
        "--algo", spec.algorithm,
        "--env", ENVIRONMENT,
        "--seed", str(spec.seed),
        "--device", str(device),
        "--return_threshold", str(RETURN_THRESHOLD),
        "--num_eval", str(FINAL_EVALUATION_EPISODES),
        "--validation_episodes", str(VALIDATION_EPISODES),
        "--validation_every", "10",
        # batch 200 is the only rolling checkpoint until batch 400, giving the watcher 200 full
        # rollout batches to freeze the 5M files before the next replacement.
        "--checkpoint_every", str(MILESTONE_BATCH),
        "--checkpoint_dir", str(spec.live_checkpoint_dir),
        "--wandb_mode", str(wandb_mode),
        "--tag", spec.runner_tag,
    ]
    if resume is not None:
        command.extend(("--resume", str(resume)))
    command.append("--set")
    command.extend(overrides)
    return command


# ============================================================================ durable checkpoint watcher
def _checkpoint_iteration(path):
    """Read only checkpoint metadata needed to distinguish batch 200 from overwritten batch 400."""
    checkpoint = Path(path)
    if not checkpoint.exists():
        return None
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return int(payload["iteration"])


def _atomic_copy(source, destination):
    """Copy one complete atomic runner checkpoint through a same-directory temporary file."""
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def capture_five_million_checkpoint(spec):
    """
    Freeze batch-200 endpoint/best files, or return False while that checkpoint is not yet available.

    A live batch-400 file with no milestone copy is a hard error: silently calling it 5M would
    corrupt the experiment.  Existing complete milestone files make this operation idempotent.
    """
    milestone_latest = spec.milestone_checkpoint_dir / "latest.pt"
    milestone_best = spec.milestone_checkpoint_dir / "best.pt"
    if (_checkpoint_iteration(milestone_latest) == MILESTONE_BATCH - 1
            and _checkpoint_iteration(milestone_best) is not None):
        return True

    live_latest = spec.live_checkpoint_dir / "latest.pt"
    live_iteration = _checkpoint_iteration(live_latest)
    if live_iteration is None or live_iteration < MILESTONE_BATCH - 1:
        return False
    if live_iteration != MILESTONE_BATCH - 1:
        raise RuntimeError(
            f"{spec.lineage_id}: live latest.pt is batch {live_iteration + 1}, "
            "but the 5M checkpoint was not captured")

    live_best = spec.live_checkpoint_dir / "best.pt"
    best_iteration = _checkpoint_iteration(live_best)
    if best_iteration is None or best_iteration > MILESTONE_BATCH - 1:
        raise RuntimeError(
            f"{spec.lineage_id}: invalid 5M selected checkpoint iteration {best_iteration}")

    _atomic_copy(live_latest, milestone_latest)
    _atomic_copy(live_best, milestone_best)
    return True


def _read_progress(path):
    """Read one complete atomic status snapshot; transient absence is normal before first batch."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return None


def _assert_finite_progress(spec, progress):
    """Stop a wave on NaN/Inf while allowing absent validation/lambda metrics during warmup."""
    if not progress:
        return
    sections = (progress.get("metrics") or {}, progress.get("validation") or {})
    for section in sections:
        for key, value in section.items():
            if value is not None and not math.isfinite(float(value)):
                raise RuntimeError(
                    f"{spec.lineage_id}: non-finite progress metric {key}={value!r}")


def _runner_is_complete(spec):
    """A run is complete only when the final durable endpoint and runner JSON both exist."""
    latest = spec.live_checkpoint_dir / "latest.pt"
    return _checkpoint_iteration(latest) == FINAL_BATCH - 1 and spec.runner_result_path.exists()


def _resume_checkpoint(spec):
    """Resume only the same lineage's signature-checked latest.pt; never transfer across LR/seed."""
    latest = spec.live_checkpoint_dir / "latest.pt"
    iteration = _checkpoint_iteration(latest)
    if iteration is None or iteration >= FINAL_BATCH - 1:
        return None
    return latest


def run_algorithm_wave(specs, device, wandb_mode, git_sha, git_dirty, poll_seconds=1.0,
                       dry_run=False):
    """Run exactly three training seeds concurrently and wait before the next algorithm wave."""
    commands = []
    captured = set()
    for spec in specs:
        # A recovered batch-200 live file must be copied before a resumed child can eventually
        # replace it at batch 400.
        if capture_five_million_checkpoint(spec):
            captured.add(spec)
        resume = _resume_checkpoint(spec)
        commands.append((spec, build_training_command(
            spec, device, wandb_mode, git_sha, git_dirty, resume=resume)))

    if dry_run:
        for spec, command in commands:
            print(spec.lineage_id + "\n  " + " ".join(command))
        return

    processes = {}
    streams = {}
    try:
        for spec, command in commands:
            if _runner_is_complete(spec):
                if spec not in captured and capture_five_million_checkpoint(spec):
                    captured.add(spec)
                print(f"[skip complete] {spec.lineage_id}", flush=True)
                continue

            spec.log_path.parent.mkdir(parents=True, exist_ok=True)
            spec.live_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            stream = open(spec.log_path, "a", encoding="utf-8")
            env = os.environ.copy()
            env.update({
                "WANDB_RUN_ID": spec.wandb_id,
                "WANDB_RESUME": "allow",
                "PYTHONUNBUFFERED": "1",
            })
            stream.write("\n[launcher command] " + " ".join(command) + "\n")
            stream.flush()
            processes[spec] = subprocess.Popen(
                command, cwd=BASE_DIR, stdout=stream, stderr=subprocess.STDOUT, env=env)
            streams[spec] = stream
            print(f"[started] pid={processes[spec].pid} {spec.lineage_id}", flush=True)

        while processes:
            failure = None
            for spec, process in list(processes.items()):
                progress = _read_progress(spec.progress_path)
                try:
                    _assert_finite_progress(spec, progress)
                    # Progress reaches 200 only after latest.pt/best.pt are atomically complete.
                    # Copy each large DQC checkpoint pair once instead of deserializing every poll.
                    completed = int((progress or {}).get("completed", 0))
                    if spec not in captured and completed >= MILESTONE_BATCH:
                        if capture_five_million_checkpoint(spec):
                            captured.add(spec)
                except Exception as error:
                    failure = error
                    break

                code = process.poll()
                if code is None:
                    continue
                streams[spec].close()
                del streams[spec]
                del processes[spec]
                if code != 0:
                    failure = RuntimeError(
                        f"{spec.lineage_id} exited with code {code}; see {spec.log_path}")
                    break
                if spec not in captured and capture_five_million_checkpoint(spec):
                    captured.add(spec)
                if not _runner_is_complete(spec):
                    failure = RuntimeError(
                        f"{spec.lineage_id} exited without a complete batch-{FINAL_BATCH} result")
                    break
                print(f"[finished] {spec.lineage_id}", flush=True)

            if failure is not None:
                for process in processes.values():
                    process.terminate()
                raise failure
            if processes:
                time.sleep(max(0.1, float(poll_seconds)))
    finally:
        for stream in streams.values():
            stream.close()


def run_training_campaign(device, wandb_mode, poll_seconds=1.0, dry_run=False):
    """Execute LR waves serially; within each algorithm wave seeds 0/1/2 are concurrent."""
    git_sha, git_dirty = git_provenance()
    runs = enumerate_runs()
    for lr_label, _actor_lr0 in LEARNING_RATES:
        for algorithm in ALGORITHMS:
            wave = [
                spec for spec in runs
                if spec.lr_label == lr_label and spec.algorithm == algorithm
            ]
            print(f"\n[wave] lr={lr_label} algorithm={algorithm} seeds={SEEDS}\n", flush=True)
            run_algorithm_wave(
                wave, device, wandb_mode, git_sha, git_dirty,
                poll_seconds=poll_seconds, dry_run=dry_run)


# ============================================================================ post-hoc common-RNG evaluation
def reset_evaluation_rng(seed=FINAL_EVALUATION_SEED):
    """Restart environment/action random streams identically before every checkpoint evaluation."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _evaluation_args(spec, device):
    """Rebuild the exact signature using the same parameter overrides as the training command."""
    from run_experiment import base_args, configure_dqc_identity

    args = base_args(spec.algorithm, spec.seed, device, ENVIRONMENT)
    args.return_threshold = RETURN_THRESHOLD
    args.actor_lr0 = float(spec.actor_lr0)
    args.theta_a = (args.theta_b ** args.theta_c) * float(spec.actor_lr0)
    args.ref_lr = float(spec.actor_lr0)
    args.actor_lr = float(spec.actor_lr0)
    args.num_iterations = FINAL_BATCH
    args.show_progress = False
    args.show_eval_progress = False
    args.wandb_group = CAMPAIGN + "_posthoc_disabled"
    args.wandb_name = "posthoc_" + spec.lineage_id
    if spec.algorithm == "DQCACBeta":
        args.dqc_update_mode = "finite_nstep_v1"
        configure_dqc_identity(args, {"dqc_update_mode": "finite_nstep_v1"})
    return args


def _policy_checkpoints(spec):
    """Return both roles at the frozen 5M milestone and durable 10M endpoint."""
    return {
        MILESTONE_BATCH: {
            "endpoint": spec.milestone_checkpoint_dir / "latest.pt",
            "selected": spec.milestone_checkpoint_dir / "best.pt",
        },
        FINAL_BATCH: {
            "endpoint": spec.live_checkpoint_dir / "latest.pt",
            "selected": spec.live_checkpoint_dir / "best.pt",
        },
    }


def evaluate_lineage(spec, device):
    """Evaluate four policy checkpoints with exact N and write one ignored raw JSON document."""
    from agents import QPOGPU, QCPOGPU, DQCACBetaGPU
    from envs import SafetyEnv, make_vec_env
    from run_experiment import checkpoint_signature
    from utils import evaluate_policy_vec, load_checkpoint

    classes = {"QPO": QPOGPU, "QCPO": QCPOGPU, "DQCACBeta": DQCACBetaGPU}
    args = _evaluation_args(spec, device)
    env = SafetyEnv(args.env_id)
    agent = classes[spec.algorithm](args, env)
    agent.checkpoint_signature = checkpoint_signature(args, spec.algorithm)
    evaluations = {}

    try:
        for batch, policies in _policy_checkpoints(spec).items():
            setting = spec.result_setting(batch)
            budget_results = {}
            endpoint_observed_steps = None
            for role, checkpoint in policies.items():
                payload = load_checkpoint(
                    checkpoint, agent, map_location=device, restore_rng=False,
                    allow_signature_mismatch=False)
                checkpoint_iteration = int(payload["iteration"])
                if role == "endpoint" and checkpoint_iteration != int(batch) - 1:
                    raise ValueError(
                        f"{spec.lineage_id}/{setting}: endpoint batch is "
                        f"{checkpoint_iteration + 1}, expected {batch}")
                if role == "selected" and checkpoint_iteration >= int(batch):
                    raise ValueError(
                        f"{spec.lineage_id}/{setting}: selected checkpoint is after its budget")

                if role == "endpoint":
                    endpoint_observed_steps = int(agent.total_env_steps)
                reset_evaluation_rng(FINAL_EVALUATION_SEED)
                evaluation_env = make_vec_env(
                    args.env_id, NUM_ENVS, HORIZON, device, ref_env=env,
                    seed=FINAL_EVALUATION_SEED, backend=args.vec_backend)
                try:
                    metrics = evaluate_policy_vec(
                        agent, evaluation_env, FINAL_EVALUATION_EPISODES,
                        GAMMA, FAILURE_ALPHA, RETURN_THRESHOLD,
                        show_progress=False,
                        progress_desc=f"posthoc/{setting}/{spec.algorithm}/s{spec.seed}/{role}")
                finally:
                    evaluation_env.close()
                budget_results[f"eval_{role}"] = metrics
                budget_results[f"{role}_checkpoint"] = str(checkpoint)
                budget_results[f"{role}_iteration"] = checkpoint_iteration

            initialization_steps = 2 * NUM_ENVS * HORIZON if spec.algorithm == "QPO" else 0
            main_steps = int(batch) * NUM_ENVS * HORIZON
            budget_results["eval"] = budget_results["eval_selected"]
            budget_results["budget"] = {
                "main_rollout_batches": int(batch),
                "main_env_steps": main_steps,
                "initialization_env_steps": initialization_steps,
                "actual_env_steps": main_steps + initialization_steps,
                "observed_actual_env_steps": endpoint_observed_steps,
            }
            evaluations[setting] = budget_results
    finally:
        agent.vec_env.close()
        env.close()
        wandb_run = getattr(agent, "_wandb_run", None)
        if wandb_run is not None:
            wandb_run.finish()

    git_sha, git_dirty = git_provenance()
    output = {
        "campaign": CAMPAIGN,
        "environment": ENVIRONMENT,
        "algorithm": spec.algorithm,
        "seed": spec.seed,
        "actor_lr0": spec.actor_lr0,
        "return_threshold": RETURN_THRESHOLD,
        "failure_alpha": FAILURE_ALPHA,
        "gamma": GAMMA,
        "num_envs": NUM_ENVS,
        "horizon": HORIZON,
        "policy_identity": spec.policy_identity,
        "wandb_id": spec.wandb_id,
        "wandb_group_requested": WANDB_GROUP,
        "training_result": str(spec.runner_result_path),
        "evaluation_protocol": {
            "name": "posthoc_exact_n1000_common_random_numbers_v1",
            "num_episodes": FINAL_EVALUATION_EPISODES,
            "reset_seed": FINAL_EVALUATION_SEED,
            "action_rng_seed": FINAL_EVALUATION_SEED,
            "checkpoint_rng_restored": False,
        },
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "evaluations": evaluations,
    }
    spec.evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(spec.evaluation_path, output)
    return output


def evaluate_campaign(device):
    """Run post-hoc evaluation only after all 18 long-training endpoints are complete."""
    incomplete = [spec.lineage_id for spec in enumerate_runs() if not _runner_is_complete(spec)]
    if incomplete:
        raise RuntimeError("long training is incomplete: " + ", ".join(incomplete))
    for spec in enumerate_runs():
        capture_five_million_checkpoint(spec)
        print(f"[posthoc evaluation] {spec.lineage_id}", flush=True)
        evaluate_lineage(spec, device)


def _write_json_atomic(path, payload):
    """Write a small JSON artifact without exposing a partial document to aggregators."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
