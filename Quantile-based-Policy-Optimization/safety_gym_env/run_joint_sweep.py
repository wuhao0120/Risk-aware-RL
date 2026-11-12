#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI for the parameter-only SimpleButton 5M-checkpoint/10M-training campaign."""
import argparse
import os

import torch

from joint_sweep_results import aggregate_campaign
from joint_sweep_simplebutton import (
    ALGORITHMS,
    CAMPAIGN,
    ENVIRONMENT,
    FAILURE_ALPHA,
    FINAL_BATCH,
    FINAL_EVALUATION_EPISODES,
    FINAL_EVALUATION_SEED,
    GAMMA,
    HORIZON,
    LEARNING_RATES,
    MILESTONE_BATCH,
    NUM_ENVS,
    RETURN_THRESHOLD,
    SEEDS,
    TRACKED_ROOT,
    _resume_checkpoint,
    _write_json_atomic,
    build_training_command,
    enumerate_runs,
    evaluate_campaign,
    git_provenance,
    run_training_campaign,
)


def manifest_payload():
    """Describe the reduced grid, execution topology, frozen budgets, and all 18 lineages."""
    runs = enumerate_runs()
    return {
        "campaign": CAMPAIGN,
        "environment": ENVIRONMENT,
        "training_run_count": len(runs),
        "lineage_count": len(runs),
        "result_setting_count": len(LEARNING_RATES) * 2,
        "posthoc_policy_evaluation_count": len(runs) * 2 * 2,
        "execution_order": {
            "learning_rates": [label for label, _value in LEARNING_RATES],
            "algorithm_waves": list(ALGORITHMS),
            "concurrent_training_seeds": list(SEEDS),
            "maximum_concurrency": len(SEEDS),
        },
        "grid": {
            "actor_lr0": [value for _label, value in LEARNING_RATES],
            "fresh_training_main_env_steps": FINAL_BATCH * NUM_ENVS * HORIZON,
            "frozen_checkpoint_main_env_steps": MILESTONE_BATCH * NUM_ENVS * HORIZON,
            "return_threshold": RETURN_THRESHOLD,
            "failure_alpha": FAILURE_ALPHA,
            "gamma": GAMMA,
            "num_envs": NUM_ENVS,
            "horizon": HORIZON,
        },
        "evaluation": {
            "num_episodes": FINAL_EVALUATION_EPISODES,
            "common_reset_seed": FINAL_EVALUATION_SEED,
            "common_action_rng_seed": FINAL_EVALUATION_SEED,
            "policy_roles": ["endpoint", "selected"],
            "budget_batches": [MILESTONE_BATCH, FINAL_BATCH],
        },
        "runs": [{
            "lineage_id": spec.lineage_id,
            "algorithm": spec.algorithm,
            "seed": spec.seed,
            "actor_lr0": spec.actor_lr0,
            "policy_identity": spec.policy_identity,
            "wandb_id": spec.wandb_id,
            "training_setting": spec.training_setting,
            "result_settings": [
                spec.result_setting(MILESTONE_BATCH),
                spec.result_setting(FINAL_BATCH),
            ],
        } for spec in runs],
    }


def write_manifest(path=None):
    """Write the compact tracked manifest; raw checkpoints and metrics remain under ignored _runs."""
    destination = TRACKED_ROOT / "manifest.json" if path is None else path
    _write_json_atomic(destination, manifest_payload())
    return destination


def _device(value):
    """Use the requested CUDA index when present and otherwise preserve CPU testability."""
    if str(value) == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{value}")


def parse_args(argv=None):
    """Expose independent recovery entry points for every expensive campaign phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("manifest", help="write the tracked campaign manifest")

    commands = subparsers.add_parser("commands", help="print all parameter-only runner commands")
    commands.add_argument("--device", default="0")
    commands.add_argument(
        "--wandb-mode", default="online", choices=("online", "offline", "disabled"))

    run = subparsers.add_parser(
        "run", help="run all long lineages, then post-hoc evaluate and aggregate")
    run.add_argument("--device", default="0")
    run.add_argument(
        "--wandb-mode", default="online", choices=("online", "offline", "disabled"))
    run.add_argument("--poll", type=float, default=1.0)
    run.add_argument("--dry-run", action="store_true")

    evaluate = subparsers.add_parser(
        "evaluate", help="post-hoc evaluate all completed 5M/10M checkpoints")
    evaluate.add_argument("--device", default="0")
    subparsers.add_parser(
        "aggregate", help="validate raw post-hoc JSON and regenerate all CSV tables")
    return parser.parse_args(argv)


def main(argv=None):
    """Dispatch a resumable parameter-only campaign action."""
    cli = parse_args(argv)
    if cli.action == "manifest":
        print(write_manifest())
        return

    if cli.action == "commands":
        git_sha, git_dirty = git_provenance()
        for spec in enumerate_runs():
            command = build_training_command(
                spec, cli.device, cli.wandb_mode, git_sha, git_dirty,
                resume=_resume_checkpoint(spec))
            print(spec.lineage_id + "\n  " + " ".join(command))
        return

    if cli.action == "run":
        write_manifest()
        run_training_campaign(
            cli.device, cli.wandb_mode, poll_seconds=cli.poll, dry_run=cli.dry_run)
        if not cli.dry_run:
            # The requested order is strict: 5M comparison begins only after every 10M run exits.
            os.environ["WANDB_MODE"] = "disabled"
            evaluate_campaign(_device(cli.device))
            aggregate_campaign()
        return

    if cli.action == "evaluate":
        os.environ["WANDB_MODE"] = "disabled"
        evaluate_campaign(_device(cli.device))
        return
    if cli.action == "aggregate":
        aggregate_campaign()
        return
    raise AssertionError(f"unhandled action {cli.action!r}")


if __name__ == "__main__":
    main()
