# -*- coding: utf-8 -*-
"""Parameter-only joint sweep enumeration, checkpoint watcher, CRN, and table tests."""
import copy
import math
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from joint_sweep_results import (
    aggregate_campaign,
    expected_setting_names,
    metric_values,
    per_seed_rows,
    sample_summary,
    setting_table_rows,
    validate_evaluation_records,
)
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
    MILESTONE_BATCH,
    NUM_ENVS,
    RETURN_THRESHOLD,
    SEEDS,
    build_training_command,
    capture_five_million_checkpoint,
    enumerate_runs,
    reset_evaluation_rng,
)
from run_experiment import base_args, checkpoint_signature, configure_dqc_identity
from run_joint_sweep import manifest_payload
from utils.checkpoint import validate_checkpoint_payload


def test_reduced_grid_has_18_unique_lineages_and_four_result_settings():
    """Two LRs x three algorithms x three seeds gives 18 long runs, not separate 5M stages."""
    runs = enumerate_runs()
    assert len(runs) == 18
    assert len({spec.lineage_id for spec in runs}) == 18
    assert len({spec.runner_tag for spec in runs}) == 18
    assert [spec.lr_label for spec in runs[:9]] == ["2e-4"] * 9
    assert [spec.algorithm for spec in runs[:9:3]] == list(ALGORITHMS)
    assert [spec.seed for spec in runs[:3]] == list(SEEDS)
    assert expected_setting_names() == [
        "lr2e-4_main5m", "lr2e-4_main10m",
        "lr4e-4_main5m", "lr4e-4_main10m",
    ]


def test_training_commands_use_parameter_overrides_and_one_10m_run_per_lineage():
    """Commands must not expose tracker/dual/critic LR tuning or a second 5M training process."""
    for spec in enumerate_runs():
        command = build_training_command(
            spec, device="0", wandb_mode="disabled",
            git_sha="deadbeef", git_dirty=False)
        joined = " ".join(command)
        assert "--checkpoint_every 200" in joined
        assert "num_iterations=400" in command
        assert f"actor_lr0={spec.actor_lr0:.17g}" in command
        assert "main_env_steps=10000000" in command
        assert "return_threshold" not in " ".join(
            item for item in command[command.index("--set") + 1:]
            if item.startswith(("q_a=", "lambda_a=", "critic_lr=")))
        assert not any(item == "num_iterations=200" for item in command)
        if spec.algorithm == "DQCACBeta":
            assert "dqc_update_mode=finite_nstep_v1" in command
        else:
            assert "dqc_update_mode=finite_nstep_v1" not in command


def test_actor_lr_override_matches_all_three_algorithm_schedule_inputs():
    """The unchanged runner's canonical override yields one common absolute LR schedule."""
    for actor_lr0 in (2e-4, 4e-4):
        for algorithm in ALGORITHMS:
            args = base_args(algorithm, 0, torch.device("cpu"), ENVIRONMENT)
            args.actor_lr0 = actor_lr0
            args.theta_a = (args.theta_b ** args.theta_c) * actor_lr0
            args.ref_lr = actor_lr0
            args.actor_lr = actor_lr0
            if algorithm == "DQCACBeta":
                args.dqc_update_mode = "finite_nstep_v1"
                configure_dqc_identity(args, {"dqc_update_mode": "finite_nstep_v1"})

            for batch in (0, 1, 199, 399):
                direct = args.theta_a / ((args.theta_b + batch) ** args.theta_c)
                ratio = actor_lr0 * (
                    args.theta_b / (args.theta_b + batch)) ** args.theta_c
                assert math.isclose(direct, ratio, rel_tol=1e-14, abs_tol=0.0)
            assert args.actor_lr0 == actor_lr0
            assert args.actor_lr == actor_lr0


def test_checkpoint_signature_rejects_cross_lr_resume_but_accepts_same_lineage():
    """Existing signature validation is the fail-fast barrier for accidental 2e-4 -> 4e-4 reuse."""
    source_args = base_args("QPO", 0, torch.device("cpu"), ENVIRONMENT)
    source_args.return_threshold = RETURN_THRESHOLD
    source_args.actor_lr0 = 2e-4
    source_args.theta_a = (source_args.theta_b ** source_args.theta_c) * 2e-4
    source_signature = checkpoint_signature(source_args, "QPO")

    class FakeAgent:
        """Only class identity and expected signature are needed before state mutation."""

    same = FakeAgent()
    same.checkpoint_signature = dict(source_signature)
    payload = {
        "agent_class": "FakeAgent",
        "agent": {},
        "experiment_signature": dict(source_signature),
    }
    validate_checkpoint_payload(payload, same)

    different = FakeAgent()
    different.checkpoint_signature = dict(source_signature)
    different.checkpoint_signature["actor_lr0"] = 4e-4
    different.checkpoint_signature["theta_a"] = (
        source_args.theta_b ** source_args.theta_c) * 4e-4
    try:
        validate_checkpoint_payload(payload, different)
    except ValueError as error:
        assert "actor_lr0" in str(error)
        assert "theta_a" in str(error)
    else:
        raise AssertionError("cross-LR resume must fail before state mutation")


def test_watcher_copies_only_exact_batch200_checkpoint(tmp_path):
    """The external watcher freezes endpoint/best atomically and rejects a lost batch-200 file."""
    live = tmp_path / "live"
    milestone = tmp_path / "main5m"
    live.mkdir()
    torch.save({"iteration": MILESTONE_BATCH - 1}, live / "latest.pt")
    torch.save({"iteration": 179}, live / "best.pt")
    spec = SimpleNamespace(
        live_checkpoint_dir=live,
        milestone_checkpoint_dir=milestone,
        lineage_id="fixture",
    )
    assert capture_five_million_checkpoint(spec)
    assert torch.load(milestone / "latest.pt", weights_only=False)["iteration"] == 199
    assert torch.load(milestone / "best.pt", weights_only=False)["iteration"] == 179

    lost = SimpleNamespace(
        live_checkpoint_dir=tmp_path / "lost-live",
        milestone_checkpoint_dir=tmp_path / "lost-main5m",
        lineage_id="lost-fixture",
    )
    lost.live_checkpoint_dir.mkdir()
    torch.save({"iteration": FINAL_BATCH - 1}, lost.live_checkpoint_dir / "latest.pt")
    try:
        capture_five_million_checkpoint(lost)
    except RuntimeError as error:
        assert "not captured" in str(error)
    else:
        raise AssertionError("batch-400 latest.pt cannot be mislabeled as the 5M endpoint")


def test_posthoc_roles_restart_identical_python_numpy_and_torch_crn():
    """Reloading a policy cannot affect the explicit common action/reset stream."""
    reset_evaluation_rng(FINAL_EVALUATION_SEED)
    first = (random.random(), np.random.random(3), torch.rand(3))
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    reset_evaluation_rng(FINAL_EVALUATION_SEED)
    second = (random.random(), np.random.random(3), torch.rand(3))
    assert first[0] == second[0]
    assert np.array_equal(first[1], second[1])
    assert torch.equal(first[2], second[2])


def _synthetic_metrics(seed, role, tracker_probability=0.91):
    """Return exact-N finite metrics with a deliberately unrelated QPO tracker probability."""
    failure_count = 100 + 10 * int(seed) + (1 if role == "selected" else 0)
    probability = failure_count / FINAL_EVALUATION_EPISODES
    return {
        "num_episodes": FINAL_EVALUATION_EPISODES,
        "failure_count": failure_count,
        "empirical_prob": probability,
        "quantile_constraint_prob": tracker_probability,
        "wilson95_low": max(0.0, probability - 0.02),
        "wilson95_high": min(1.0, probability + 0.02),
        "quantile_return": -0.2 + 0.01 * seed,
        "mean": 1.0 + seed,
        "reward_std": 2.0 + seed,
    }


def synthetic_records():
    """Construct a complete 18-lineage/four-setting result set for table unit tests."""
    records = []
    for spec in enumerate_runs():
        evaluations = {}
        for batch in (MILESTONE_BATCH, FINAL_BATCH):
            setting = spec.result_setting(batch)
            endpoint = _synthetic_metrics(spec.seed, "endpoint")
            selected = _synthetic_metrics(spec.seed, "selected")
            initialization = 2 * NUM_ENVS * HORIZON if spec.algorithm == "QPO" else 0
            main_steps = batch * NUM_ENVS * HORIZON
            evaluations[setting] = {
                "eval_endpoint": endpoint,
                "eval_selected": selected,
                "eval": copy.deepcopy(selected),
                "endpoint_checkpoint": f"/{spec.lineage_id}/{setting}/latest.pt",
                "selected_checkpoint": f"/{spec.lineage_id}/{setting}/best.pt",
                "endpoint_iteration": batch - 1,
                "selected_iteration": batch - 11,
                "budget": {
                    "main_rollout_batches": batch,
                    "main_env_steps": main_steps,
                    "initialization_env_steps": initialization,
                    "actual_env_steps": main_steps + initialization,
                    "observed_actual_env_steps": main_steps + initialization,
                },
            }
        records.append({
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
            "evaluation_protocol": {
                "num_episodes": FINAL_EVALUATION_EPISODES,
                "reset_seed": FINAL_EVALUATION_SEED,
                "action_rng_seed": FINAL_EVALUATION_SEED,
            },
            "evaluations": evaluations,
        })
    return records


def test_qpo_horizontal_outage_uses_fixed_q_empirical_probability():
    """A .91 tracker-relative probability must never leak into the cross-algorithm outage row."""
    records = synthetic_records()
    validate_evaluation_records(records)
    rows = per_seed_rows(records, "lr2e-4_main5m")
    qpo_endpoint = next(
        row for row in rows
        if row["algorithm"] == "QPO" and row["seed"] == 0
        and row["policy_role"] == "endpoint")
    assert qpo_endpoint["outage_probability"] == 0.1
    assert qpo_endpoint["outage_probability"] != 0.91


def test_population_episode_variance_and_three_seed_sample_variance_are_distinct():
    """reward_std squared is per-policy ddof=0 variance; cross-seed variance uses ddof=1."""
    values = metric_values(_synthetic_metrics(1, "endpoint"), selected_iteration=199)
    assert values["episode_return_variance"] == 9.0
    mean, variance = sample_summary([1.0, 2.0, 3.0])
    assert mean == 2.0
    assert variance == 1.0


def test_horizontal_and_longitudinal_csv_layout(tmp_path):
    """Four settings generate 18-row per-seed files, 9-row tables, 24 long and 4 wide rows."""
    records = synthetic_records()
    generated = aggregate_campaign(output_root=tmp_path, records=records)
    assert len(generated["per_seed"]) == 4 * 18
    assert len(generated["long"]) == 4 * len(ALGORITHMS) * 2
    assert len(generated["wide"]) == 4
    for setting in expected_setting_names():
        assert (tmp_path / setting / "per_seed.csv").exists()
        assert (tmp_path / setting / "setting_table.csv").exists()
        rows = per_seed_rows(records, setting)
        table = setting_table_rows(rows)
        assert len(table) == 9
        assert "QPO_seed0" in table[0]
        assert "DQCACBeta_sample_variance" in table[0]
    assert (tmp_path / "aggregate_long.csv").exists()
    assert (tmp_path / "aggregate_wide.csv").exists()


def test_aggregation_fail_fast_on_missing_duplicate_n_or_config_mismatch():
    """Every common corruption mode must fail before a CSV is emitted."""
    records = synthetic_records()
    mutations = []
    mutations.append(records[:-1])
    duplicate = copy.deepcopy(records)
    duplicate[-1] = copy.deepcopy(duplicate[0])
    mutations.append(duplicate)
    wrong_n = copy.deepcopy(records)
    first_setting = next(iter(wrong_n[0]["evaluations"]))
    wrong_n[0]["evaluations"][first_setting]["eval_endpoint"]["num_episodes"] = 999
    mutations.append(wrong_n)
    wrong_q = copy.deepcopy(records)
    wrong_q[0]["return_threshold"] = -0.04
    mutations.append(wrong_q)

    for mutation in mutations:
        try:
            validate_evaluation_records(mutation)
        except ValueError:
            pass
        else:
            raise AssertionError("corrupt aggregate input must fail fast")


def test_manifest_records_reduced_run_and_evaluation_counts():
    """Manifest distinguishes 18 expensive training runs from 72 post-hoc policy evaluations."""
    manifest = manifest_payload()
    assert manifest["training_run_count"] == 18
    assert manifest["lineage_count"] == 18
    assert manifest["result_setting_count"] == 4
    assert manifest["posthoc_policy_evaluation_count"] == 72
    assert len(manifest["runs"]) == 18
