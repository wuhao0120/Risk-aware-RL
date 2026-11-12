#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fresh seed-0 binding-threshold stress test for DQC, QCPO_REF, and QCPO.

The campaign changes only the fixed reward-return threshold from -0.03 to 2.30.
DQC GAE-Mean and discounted-occupancy QCPO_REF run concurrently in phase one;
QCPO starts only after both phase-one lineages finish training, artifact checks,
and exact-N=1000 CRN evaluation. All training remains delegated to the tested
formal harness and run_experiment.py.
"""
import argparse  # Parse explicit campaign and single-lineage CLI actions.
from contextlib import contextmanager  # Guarantee restoration of the reusable core module globals.
import json  # Load atomically written exact-evaluation JSON artifacts.
import subprocess  # Launch phase-one lineages as independent child processes.
import sys  # Reuse the active Python interpreter for every child process.
from pathlib import Path  # Resolve stable paths independently of the caller working directory.

import torch  # Select the device used by strict post-hoc checkpoint evaluation.

import formal_qcpo_ref_dqc_gae as core  # Reuse the tested training, freezing, and evaluation harness.


# ============================================================================ frozen campaign identity
BASE_DIR = Path(__file__).resolve().parent  # Anchor every output path under safety_gym_env.
CAMPAIGN = "binding_q2p3_simplebutton_seed0_v1"  # Identify this threshold experiment independently of prior runs.
TRACKED_MANIFEST = BASE_DIR / "campaigns" / CAMPAIGN / "manifest.json"  # Publish only the final small, tracked result artifact here.
RAW_ROOT = BASE_DIR / "_runs" / CAMPAIGN  # Keep checkpoints, logs, and raw evaluations outside tracked results.

# This threshold was selected once from the completed non-binding baseline:
# exact-N=1000 q20 was about 2.386 for both new algorithms. A fixed 2.30
# leaves a small known-feasible margin while making most of training binding.
RETURN_THRESHOLD = 2.30  # Use the same fixed q=2.30 constraint in all three algorithms.
FIRST_PHASE = ("dqc", "qref")  # Run DQC and corrected QCPO_REF together in phase one.
SECOND_PHASE = ("qcpo",)  # Delay QCPO until every phase-one gate has passed.
FORMAL_ORDER = (FIRST_PHASE, SECOND_PHASE)  # Persist the user-requested phase ordering in the manifest.
WANDB_GROUP = CAMPAIGN  # Group all three online W&B runs in one comparison view.

# Stable W&B IDs are deliberately distinct from prior q=-0.03 lineages.
# Human-facing names carry the threshold and formulation; tags stay <=64 chars.
SPECS = {  # Freeze FormalSpec field 'SPECS' as part of this lineage identity.
    "dqc": core.FormalSpec(
        key="dqcg23s0",  # Freeze FormalSpec field 'key' as part of this lineage identity.
        algorithm="DQCACBeta",  # Freeze FormalSpec field 'algorithm' as part of this lineage identity.
        wandb_id="dqcg23s0",  # Freeze FormalSpec field 'wandb_id' as part of this lineage identity.
        wandb_group=WANDB_GROUP,  # Freeze FormalSpec field 'wandb_group' as part of this lineage identity.
        wandb_tag="dqc-gae-mean-q2p3-lr2e4-main10m-s0",  # Freeze FormalSpec field 'wandb_tag' as part of this lineage identity.
        tmux_name="binding_q23_dqc_gae_s0",  # Freeze FormalSpec field 'tmux_name' as part of this lineage identity.
        parent_policy_identity="finite_nstep_v1_state_value_gae_v1_qminus0p03",  # Freeze FormalSpec field 'parent_policy_identity' as part of this lineage identity.
        policy_identity="finite_nstep_v1_state_value_gae_v1",  # Freeze FormalSpec field 'policy_identity' as part of this lineage identity.
        variant_changes=(  # Freeze FormalSpec field 'variant_changes' as part of this lineage identity.
            "return_threshold_minus0p03_to_2p30;fresh_seed0_training;"
            "all_other_algorithm_and_budget_fields_unchanged"),
        initialization_batches=10,  # Freeze FormalSpec field 'initialization_batches' as part of this lineage identity.
        wandb_name="DQC-GAE-Mean-q2.30-lr2e4-s0",  # Freeze FormalSpec field 'wandb_name' as part of this lineage identity.
    ),
    "qref": core.FormalSpec(
        key="qref23s0",  # Freeze FormalSpec field 'key' as part of this lineage identity.
        algorithm="QCPO_REF",  # Freeze FormalSpec field 'algorithm' as part of this lineage identity.
        wandb_id="qref23s0",  # Freeze FormalSpec field 'wandb_id' as part of this lineage identity.
        wandb_group=WANDB_GROUP,  # Freeze FormalSpec field 'wandb_group' as part of this lineage identity.
        wandb_tag="qcpo-ref-discocc-v2-q2p3-lr2e4-main10m-s0",  # Freeze FormalSpec field 'wandb_tag' as part of this lineage identity.
        tmux_name="binding_q23_qcpo_ref_s0",  # Freeze FormalSpec field 'tmux_name' as part of this lineage identity.
        parent_policy_identity=(  # Freeze FormalSpec field 'parent_policy_identity' as part of this lineage identity.
            "paper_negated_return_eq275_discounted_bh_v2_qminus0p03"),
        policy_identity="paper_negated_return_eq275_discounted_bh_v2",  # Freeze FormalSpec field 'policy_identity' as part of this lineage identity.
        variant_changes=(  # Freeze FormalSpec field 'variant_changes' as part of this lineage identity.
            "return_threshold_minus0p03_to_2p30;fresh_seed0_training;"
            "all_other_algorithm_and_budget_fields_unchanged"),
        initialization_batches=0,  # Freeze FormalSpec field 'initialization_batches' as part of this lineage identity.
        wandb_name="QCPO_REF-DiscOcc-v2-q2.30-lr2e4-s0",  # Freeze FormalSpec field 'wandb_name' as part of this lineage identity.
    ),
    "qcpo": core.FormalSpec(
        key="qcpo23s0",  # Freeze FormalSpec field 'key' as part of this lineage identity.
        algorithm="QCPO",  # Freeze FormalSpec field 'algorithm' as part of this lineage identity.
        wandb_id="qcpo23s0",  # Freeze FormalSpec field 'wandb_id' as part of this lineage identity.
        wandb_group=WANDB_GROUP,  # Freeze FormalSpec field 'wandb_group' as part of this lineage identity.
        wandb_tag="qcpo-loo-q2p3-lr2e4-main10m-s0",  # Freeze FormalSpec field 'wandb_tag' as part of this lineage identity.
        tmux_name="binding_q23_qcpo_s0",  # Freeze FormalSpec field 'tmux_name' as part of this lineage identity.
        parent_policy_identity="qcpo_reward_chance_loo_v1_qminus0p03",  # Freeze FormalSpec field 'parent_policy_identity' as part of this lineage identity.
        policy_identity="qcpo_reward_chance_loo_v1",  # Freeze FormalSpec field 'policy_identity' as part of this lineage identity.
        variant_changes=(  # Freeze FormalSpec field 'variant_changes' as part of this lineage identity.
            "return_threshold_minus0p03_to_2p30;fresh_seed0_training;"
            "all_other_algorithm_and_budget_fields_unchanged"),
        initialization_batches=0,  # Freeze FormalSpec field 'initialization_batches' as part of this lineage identity.
        wandb_name="QCPO-q2.30-lr2e4-s0",  # Freeze FormalSpec field 'wandb_name' as part of this lineage identity.
    ),
}


@contextmanager
def installed_core_identity():
    """Install this campaign for one operation, then restore old globals."""
    names = ("CAMPAIGN", "RETURN_THRESHOLD", "RAW_ROOT",  # These core globals jointly determine paths, threshold, and registered specs.
             "TRACKED_MANIFEST", "SPECS")
    previous = tuple(getattr(core, name) for name in names)  # Capture object references before installing this campaign.
    replacements = (CAMPAIGN, RETURN_THRESHOLD, RAW_ROOT,  # Prepare this campaign's replacement values in the same field order.
                    TRACKED_MANIFEST, SPECS)
    for name, value in zip(names, replacements):  # Install each replacement only for the context-managed operation.
        setattr(core, name, value)
    try:
        yield  # Yield control after every required core field is installed.
    finally:
        for name, value in zip(names, previous):  # Restore the prior core identity even when the operation raises.
            setattr(core, name, value)


def _validate_registered_identity():
    """Fail before launch if IDs collide or a W&B tag violates server limits."""
    ids = [spec.wandb_id for spec in SPECS.values()]  # Collect stable online W&B IDs for a local uniqueness check.
    if len(ids) != len(set(ids)):  # A duplicate ID could silently merge two external lineages.
        raise RuntimeError("binding campaign W&B IDs must be unique")
    for spec in SPECS.values():  # Validate every registered lineage before launching any process.
        if not spec.wandb_name:  # Require a readable display name independently of the stable ID.
            raise RuntimeError(f"{spec.key}: readable W&B display name is required")
        if len(spec.wandb_tag) > 64:  # Reject tags that the W&B service would refuse after launch.
            raise RuntimeError(
                f"{spec.key}: W&B tag exceeds the 64-character service limit")


def write_tracked_manifest_if_complete():
    """
    Publish one small JSON only after all three raw evaluations exist.

    Parallel phase-one processes may call this function concurrently. Missing
    QCPO output makes both calls no-ops; the phase-two process publishes the
    complete manifest atomically after its own exact evaluation succeeds.
    """
    missing = [  # Collect lineages whose exact-N=1000 raw result is still absent.
        spec.raw_result_path for spec in SPECS.values()
        if not spec.raw_result_path.exists()
    ]
    if missing:  # An incomplete campaign must not publish a three-way manifest.
        return False

    results = {  # Read each completed raw artifact without recomputing metrics.
        alias: json.loads(spec.raw_result_path.read_text(encoding="utf-8"))
        for alias, spec in SPECS.items()
    }
    payload = {  # Build one self-contained payload after all three results exist.
        "campaign": CAMPAIGN,  # Record manifest field 'campaign' for reproducibility and audit.
        "git_sha": core.git_sha(),  # Record manifest field 'git_sha' for reproducibility and audit.
        "formal_order": [list(phase) for phase in FORMAL_ORDER],  # Record manifest field 'formal_order' for reproducibility and audit.
        "fresh_training": True,  # Record manifest field 'fresh_training' for reproducibility and audit.
        "seed0_only_per_user": True,  # Record manifest field 'seed0_only_per_user' for reproducibility and audit.
        "threshold_preregistration": {  # Record manifest field 'threshold_preregistration' for reproducibility and audit.
            "return_threshold": RETURN_THRESHOLD,  # Record manifest field 'return_threshold' for reproducibility and audit.
            "failure_alpha": core.FAILURE_ALPHA,  # Record manifest field 'failure_alpha' for reproducibility and audit.
            "constraint": "P(G_1000 <= 2.30) <= 0.20",  # Record manifest field 'constraint' for reproducibility and audit.
            "selection_basis": {  # Record manifest field 'selection_basis' for reproducibility and audit.
                "qref_qminus0p03_exact_n1000_q20": 2.3865774631500245,  # Record manifest field 'qref_qminus0p03_exact_n1000_q20' for reproducibility and audit.
                "dqc_qminus0p03_exact_n1000_q20": 2.38616828918457,  # Record manifest field 'dqc_qminus0p03_exact_n1000_q20' for reproducibility and audit.
                "reason": (  # Record manifest field 'reason' for reproducibility and audit.
                    "near-binding with about 0.086 baseline q20 margin; "
                    "2.35 was rejected as too close to the observed frontier"),
            },
        },
        "common_setting": {  # Record manifest field 'common_setting' for reproducibility and audit.
            "seed": core.SEED,  # Record manifest field 'seed' for reproducibility and audit.
            "environment": core.ENVIRONMENT,  # Record manifest field 'environment' for reproducibility and audit.
            "actor_lr0": core.ACTOR_LR,  # Record manifest field 'actor_lr0' for reproducibility and audit.
            "gamma": core.GAMMA,  # Record manifest field 'gamma' for reproducibility and audit.
            "num_envs": core.NUM_ENVS,  # Record manifest field 'num_envs' for reproducibility and audit.
            "horizon": core.HORIZON,  # Record manifest field 'horizon' for reproducibility and audit.
            "main_batches": core.FINAL_BATCH,  # Record manifest field 'main_batches' for reproducibility and audit.
            "main_env_steps": core.FINAL_BATCH * core.NUM_ENVS * core.HORIZON,  # Record manifest field 'main_env_steps' for reproducibility and audit.
            "training_budget_semantics": "main_excludes_initialization_v1",  # Record manifest field 'training_budget_semantics' for reproducibility and audit.
            "validation": {  # Record manifest field 'validation' for reproducibility and audit.
                "exact_n": core.VALIDATION_EPISODES,  # Record manifest field 'exact_n' for reproducibility and audit.
                "crn_seed": core.VALIDATION_SEED,  # Record manifest field 'crn_seed' for reproducibility and audit.
                "every_main_batches": 10,  # Record manifest field 'every_main_batches' for reproducibility and audit.
            },
            "posthoc": {  # Record manifest field 'posthoc' for reproducibility and audit.
                "exact_n": core.FINAL_EVALUATION_EPISODES,  # Record manifest field 'exact_n' for reproducibility and audit.
                "crn_seed": core.FINAL_EVALUATION_SEED,  # Record manifest field 'crn_seed' for reproducibility and audit.
                "roles": ["endpoint", "selected"],  # Record manifest field 'roles' for reproducibility and audit.
                "budgets": ["main5m", "main10m"],  # Record manifest field 'budgets' for reproducibility and audit.
            },
        },
        "physical_initialization_steps": {  # Record manifest field 'physical_initialization_steps' for reproducibility and audit.
            alias: spec.initialization_env_steps
            for alias, spec in SPECS.items()
        },
        "results": results,  # Record manifest field 'results' for reproducibility and audit.
    }
    core.write_json_atomic(TRACKED_MANIFEST, payload)  # Write through a temporary file and atomically replace the manifest.
    return True  # Signal that the three-lineage publication gate passed.


def _existing_lineage_artifacts():
    """Resolve every path whose presence would make an initial launch non-fresh."""
    existing = []  # Only campaign-owned paths participate in the freshness check.
    for spec in SPECS.values():
        candidates = (  # Check every persistent artifact owned by this registered lineage.
            spec.checkpoint_root,
            spec.progress_path,
            spec.log_path,
            spec.raw_result_path,
            spec.runner_result_path,
        )
        existing.extend(path for path in candidates if path.exists())  # Any existing candidate means this is no longer a from-scratch launch.
    if TRACKED_MANIFEST.exists():  # Treat a previously published manifest as an existing lineage artifact.
        existing.append(TRACKED_MANIFEST)
    return existing  # Return exact conflicting paths for an actionable error message.


def assert_fresh_campaign():
    """
    Enforce the user's from-scratch request before phase one starts.

    Recovery is a separate explicit mode: after a machine/process interruption,
    --allow_resume permits the core harness to load only the same signature and
    same lineage's latest checkpoint.
    """
    existing = _existing_lineage_artifacts()  # Resolve conflicts before any W&B run or training process is created.
    if existing:
        formatted = "\n".join(f"  - {path}" for path in existing)  # Format all conflicts so recovery intent can be reviewed explicitly.
        raise RuntimeError(
            "fresh binding campaign already owns artifacts; use "
            "--allow_resume only for same-lineage recovery:\n" + formatted)


def _variant_command(variant, device, wandb_mode):
    """Build one child wrapper command; algorithm details stay in SPECS/core."""
    return [  # Use an argument array so no training value is interpreted by a shell.
        sys.executable,  # Use the same conda interpreter as the parent campaign process.
        str(Path(__file__).resolve()),  # Re-enter this wrapper so campaign identity is installed in the child.
        "run",  # The run action includes training, verification, and exact evaluation.
        "--variant", variant,  # Select exactly one registered lineage in each child process.
        "--device", str(device),  # Forward the requested CUDA index unchanged.
        "--wandb_mode", str(wandb_mode),  # Forward online/offline/disabled W&B mode unchanged.
    ]


def run_phase(variants, device, wandb_mode):
    """
    Run all variants in one phase concurrently and require every exit code zero.

    A failed first child does not kill its peer: preserving the peer's valid
    progress is safer and still prevents phase two because the aggregate phase
    result remains nonzero.
    """
    processes = {}  # Retain one process handle and exit code per phase variant.
    for variant in variants:
        command = _variant_command(variant, device, wandb_mode)  # Build the immutable command from the selected FormalSpec.
        process = subprocess.Popen(command, cwd=BASE_DIR)  # Launch without a shell from the stable safety_gym_env directory.
        processes[variant] = process  # Store the handle before waiting so both phase-one children coexist.
        print(
            f"[phase started] variant={variant} pid={process.pid}",  # Print the PID immediately for external monitoring and diagnosis.
            flush=True,
        )

    failures = {}  # Aggregate every failure instead of hiding a peer's exit status.
    for variant, process in processes.items():
        code = process.wait()  # Wait for completion without terminating a still-valid peer.
        if code != 0:  # A nonzero child exit code makes the whole phase fail.
            failures[variant] = code
        else:
            print(f"[phase complete] variant={variant}", flush=True)  # Report completion only after that child's full lifecycle succeeds.
    if failures:  # Block phase two whenever any phase-one child failed.
        raise RuntimeError(f"campaign phase failed: {failures}")


def run_campaign(device, wandb_mode, allow_resume=False, dry_run=False):
    """Execute DQC/QCPO_REF first, then gate QCPO on both complete results."""
    if not allow_resume:  # Fresh mode is the default; resume requires explicit recovery intent.
        assert_fresh_campaign()

    if dry_run:  # Dry-run prints the true phase order and exact training commands.
        for phase in FORMAL_ORDER:
            print(f"[dry phase] {phase}")
            for variant in phase:
                print(" ".join(core.build_training_command(  # Build commands through the same core function used for formal runs.
                    SPECS[variant], device, wandb_mode)))
        return

    run_phase(FIRST_PHASE, device, wandb_mode)  # Start DQC and QCPO_REF concurrently on the requested device.
    for variant in FIRST_PHASE:  # Require both phase-one exact results, not only process completion.
        if not SPECS[variant].raw_result_path.exists():
            raise RuntimeError(
                f"{variant}: phase-one exact evaluation result is missing")

    # QCPO starts only after both phase-one result files have been atomically
    # written, which is stronger than waiting for training processes alone.
    run_phase(SECOND_PHASE, device, wandb_mode)  # Start QCPO only after both phase-one raw results were published.
    if not write_tracked_manifest_if_complete():  # Require all three exact results before considering the campaign complete.
        raise RuntimeError("three-lineage manifest gate remained incomplete")


def run_one(action, variant, device_text, wandb_mode):
    """Dispatch one registered lineage through the reusable formal lifecycle."""
    spec = SPECS[variant]  # Resolve all lineage configuration from the immutable registry.
    device = torch.device(  # Use CUDA for formal evaluation when available and requested.
        "cpu" if device_text == "cpu"
        else f"cuda:{device_text}" if torch.cuda.is_available()
        else "cpu"
    )
    if action == "dry-run":  # Print the exact training command without creating run artifacts.
        core.run_training(spec, device_text, wandb_mode, dry_run=True)
        return
    if action == "verify":
        if not core.runner_complete(spec) or not core.verify_milestone(spec):  # Verify both the runner endpoint and frozen 5M milestone.
            raise RuntimeError(f"{spec.key}: formal artifacts are incomplete")
        print(f"{spec.key}: artifacts verified")
        return
    if action == "run":  # Execute formal training and checkpoint freezing for this lineage.
        core.run_training(spec, device_text, wandb_mode)

    # Both run and evaluate enforce strict checkpoint signatures before result
    # publication. Post-hoc agent construction forces WANDB_MODE=disabled.
    core.evaluate_lineage(spec, device)  # Strictly load four policies and run exact-N=1000 CRN evaluation.
    write_tracked_manifest_if_complete()  # Parallel phase-one calls remain no-ops until QCPO also completes.
    print(f"[binding complete] {spec.key} result={spec.raw_result_path}")


def main():
    """Parse campaign-level or one-lineage operations."""
    parser = argparse.ArgumentParser()  # Expose only explicit campaign lifecycle operations.
    parser.add_argument(
        "action",
        choices=("campaign", "run", "dry-run", "evaluate", "verify"),
    )
    parser.add_argument("--variant", choices=tuple(SPECS))  # Limit single-lineage actions to the three registered aliases.
    parser.add_argument("--device", default="0")  # Default formal execution to the only available A100, CUDA device 0.
    parser.add_argument(  # Expose W&B mode while retaining online as the requested default.
        "--wandb_mode",
        default="online",  # Make online upload the default for all child training commands.
        choices=("online", "offline", "disabled"),
    )
    parser.add_argument(
        "--allow_resume",  # Allow resume only as an explicit same-lineage recovery operation.
        action="store_true",
        help="recover only this campaign's signature-matched checkpoints",
    )
    cli = parser.parse_args()  # Parsing failure occurs before any external or filesystem mutation.

    _validate_registered_identity()  # Validate IDs, names, and tag limits before installing campaign state.
    with installed_core_identity():  # Install q=2.30 paths and specs only around the selected operation.
        if cli.action == "campaign":  # Campaign mode owns the complete ordered three-lineage lifecycle.
            run_campaign(
                cli.device,
                cli.wandb_mode,
                allow_resume=cli.allow_resume,  # Forward the explicit recovery flag to the freshness gate.
            )
            return
        if cli.variant is None:  # Single-lineage actions require an unambiguous registered variant.
            parser.error(f"--variant is required for action={cli.action}")
        run_one(cli.action, cli.variant, cli.device, cli.wandb_mode)  # Dispatch the chosen lineage with the explicit W&B mode.


if __name__ == "__main__":  # Do not run the campaign merely because the module was imported.
    main()
