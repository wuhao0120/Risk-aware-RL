# -*- coding: utf-8 -*-
"""QCPO_REF finite-context rerun identity, isolation, sync, and publication tests."""
import json
from types import SimpleNamespace

import torch

import formal_qcpo_ref_context_q23 as campaign
import formal_qcpo_ref_dqc_gae as core
from run_experiment import checkpoint_signature


def _override_suffix(command):
    """Return the immutable key=value configuration passed after --set."""
    return command[command.index("--set") + 1:]


def _save_iteration(path, iteration):
    """Write the minimal atomic-checkpoint metadata consumed by the formal validators."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"iteration": int(iteration)}, path)


def _synthetic_exact_result():
    """Build four finite exact-N=1000 records backed by the fixture's checkpoint files."""
    evaluations = {}
    for setting, checkpoint_group in core.policy_checkpoints(
            campaign.SPEC).items():
        setting_records = {}
        for role in ("endpoint", "selected"):
            checkpoint = checkpoint_group[role]
            iteration = core.checkpoint_iteration(checkpoint)
            main_steps = (iteration + 1) * core.NUM_ENVS * core.HORIZON
            setting_records[role] = {
                "checkpoint": str(checkpoint),
                "checkpoint_iteration": iteration,
                "sha256": core.sha256_file(checkpoint),
                "metrics": {
                    "num_episodes": 1000,
                    "mean": 2.5,
                    "reward_std": 0.5,
                    "quantile_return": 2.3,
                    "return_threshold": 2.3,
                    "failure_count": 200,
                    "empirical_prob": 0.2,
                    "return_variance": 0.25,
                },
                "step_clocks": {
                    "main_env_steps": main_steps,
                    "initialization_env_steps": 0,
                    "actual_env_steps": main_steps,
                },
            }
        evaluations[setting] = setting_records
    return {
        "campaign": campaign.CAMPAIGN,
        "git_sha": core.git_sha(),
        "environment": core.ENVIRONMENT,
        "algorithm": campaign.SPEC.algorithm,
        "seed": core.SEED,
        "wandb_id": campaign.SPEC.wandb_id,
        "wandb_group": campaign.SPEC.wandb_group,
        "wandb_tag": campaign.SPEC.wandb_tag,
        "parent_policy_identity": campaign.SPEC.parent_policy_identity,
        "policy_identity": campaign.SPEC.policy_identity,
        "variant_changes": campaign.SPEC.variant_changes,
        "training_budget_semantics": "main_excludes_initialization_v1",
        "evaluation_protocol": {
            "name": "exact_n1000_common_random_numbers_v1",
            "num_episodes": 1000,
            "environment_seed": 777,
            "action_rng_seed": 777,
            "checkpoint_rng_restored": False,
        },
        "evaluations": evaluations,
    }


def test_context_rerun_command_and_checkpoint_identity_are_frozen():
    """The new run changes critic context only and retains the q=2.3 10M protocol."""
    original = (
        core.CAMPAIGN, core.RETURN_THRESHOLD, core.RAW_ROOT,
        core.TRACKED_MANIFEST, core.SPECS)
    with campaign.installed_core_identity():
        command = core.build_training_command(
            campaign.SPEC, device="0", wandb_mode="online")
        suffix = _override_suffix(command)
        threshold_index = command.index("--return_threshold")
        assert command[threshold_index + 1] == "2.3"
        assert "--resume" not in command
        assert "num_iterations=400" in suffix
        assert "main_env_steps=10000000" in suffix
        assert "initialization_env_steps=0" in suffix
        assert "actual_env_steps=10000000" in suffix
        assert f"wandb_group={campaign.BASELINE_CAMPAIGN}" in suffix
        assert f"wandb_name={campaign.SPEC.wandb_name}" in suffix
        assert f"policy_identity={campaign.SPEC.policy_identity}" in suffix

        # evaluation_args reconstructs the same base_args used by training. Its strict checkpoint
        # signature proves the context cannot silently load the completed observation-only critic.
        args = core.evaluation_args(campaign.SPEC, torch.device("cpu"))
        signature = checkpoint_signature(args, campaign.SPEC.algorithm)
        assert signature["qcpo_ref_critic_input"] == (
            "finite_obs_remaining_actor_h0c0_v1")
        assert signature["horizon_mode"] == "finite"
        assert signature["critic_time_feature"] == "remaining_fraction"
        assert signature["critic_actor_context"] is True

    restored = (
        core.CAMPAIGN, core.RETURN_THRESHOLD, core.RAW_ROOT,
        core.TRACKED_MANIFEST, core.SPECS)
    assert restored == original
    assert campaign.SPEC.wandb_id == "qrefctx23s0"
    assert campaign.SPEC.wandb_group == campaign.BASELINE_CAMPAIGN
    assert len(campaign.SPEC.wandb_tag) <= 64


def test_context_paths_are_new_and_fresh_guard_is_workspace_independent(tmp_path):
    """Old q=2.3 artifacts are ignored, while any new-lineage artifact blocks fresh launch."""
    saved = (
        campaign.BASE_DIR, campaign.RAW_ROOT, campaign.TRACKED_MANIFEST,
        campaign.WANDB_SYNC_MARKER, core.BASE_DIR)
    campaign.BASE_DIR = tmp_path / "base"
    campaign.RAW_ROOT = tmp_path / "raw"
    campaign.TRACKED_MANIFEST = tmp_path / "tracked" / "manifest.json"
    campaign.WANDB_SYNC_MARKER = campaign.RAW_ROOT / "wandb_sync.json"
    core.BASE_DIR = tmp_path / "runner"
    try:
        with campaign.installed_core_identity():
            assert campaign.SPEC.checkpoint_root.parts[-2:] == (
                "checkpoints", campaign.WANDB_ID)
            campaign.assert_fresh_lineage()
            progress = campaign.SPEC.progress_path
            progress.parent.mkdir(parents=True, exist_ok=True)
            progress.write_text("{}\n", encoding="utf-8")
            try:
                campaign.assert_fresh_lineage()
            except RuntimeError as error:
                assert "--allow_resume" in str(error)
                assert str(progress) in str(error)
            else:
                raise AssertionError("new-lineage progress must block a fresh launch")
    finally:
        (
            campaign.BASE_DIR, campaign.RAW_ROOT, campaign.TRACKED_MANIFEST,
            campaign.WANDB_SYNC_MARKER, core.BASE_DIR,
        ) = saved


def test_manifest_requires_exact_result_sync_and_pinned_baseline(tmp_path):
    """Publication preserves separate old/new SHAs and rejects baseline tampering."""
    saved = (
        campaign.BASE_DIR, campaign.RAW_ROOT, campaign.TRACKED_MANIFEST,
        campaign.WANDB_SYNC_MARKER, campaign.BASELINE_MANIFEST,
        campaign.BASELINE_MANIFEST_SHA256, core.BASE_DIR, core.git_sha)
    campaign.BASE_DIR = tmp_path
    campaign.RAW_ROOT = tmp_path / "raw"
    campaign.TRACKED_MANIFEST = tmp_path / "tracked" / "manifest.json"
    campaign.WANDB_SYNC_MARKER = campaign.RAW_ROOT / "wandb_sync.json"
    campaign.BASELINE_MANIFEST = tmp_path / "baseline" / "manifest.json"
    core.BASE_DIR = tmp_path / "runner"
    core.git_sha = lambda: "context-test-git-sha"
    try:
        baseline = {
            "campaign": campaign.BASELINE_CAMPAIGN,
            "git_sha": "baseline-git-sha",
            "results": {
                "dqc": {"wandb_id": "dqcg23s0"},
                "qref": {"wandb_id": "qref23s0"},
                "qcpo": {"wandb_id": "qcpo23s0"},
            },
        }
        core.write_json_atomic(campaign.BASELINE_MANIFEST, baseline)
        campaign.BASELINE_MANIFEST_SHA256 = core.sha256_file(
            campaign.BASELINE_MANIFEST)

        with campaign.installed_core_identity():
            assert not campaign.write_tracked_manifest_if_complete()
            core.write_json_atomic(campaign.WANDB_SYNC_MARKER, {
                "synced": True,
                "wandb_id": campaign.WANDB_ID,
                "wandb_mode": "online",
                "git_sha": core.git_sha(),
                "local_run_dirs": ["wandb/run-test-qrefctx23s0"],
                "command": ["python", "-m", "wandb", "sync"],
            })
            assert not campaign.write_tracked_manifest_if_complete()

            # Reconstruct all immutable artifacts that the production evaluator would own.
            checkpoints = core.policy_checkpoints(campaign.SPEC)
            _save_iteration(checkpoints["main5m"]["endpoint"], 199)
            _save_iteration(checkpoints["main5m"]["selected"], 179)
            _save_iteration(checkpoints["main10m"]["endpoint"], 399)
            _save_iteration(checkpoints["main10m"]["selected"], 329)
            core.write_json_atomic(
                campaign.SPEC.milestone_checkpoint_dir / "hashes.json", {
                    "formal_batch": 200,
                    "endpoint_iteration": 199,
                    "selected_iteration": 179,
                    "sha256": {
                        "latest.pt": core.sha256_file(
                            checkpoints["main5m"]["endpoint"]),
                        "best.pt": core.sha256_file(
                            checkpoints["main5m"]["selected"]),
                    },
                })
            core.write_json_atomic(
                campaign.SPEC.milestone_checkpoint_dir / "READY",
                {"ready": True, "hashes_file": "hashes.json"})
            core.write_json_atomic(campaign.SPEC.runner_result_path, {
                "algorithm": campaign.SPEC.algorithm,
                "completed": 400,
            })

            result = _synthetic_exact_result()
            incomplete = dict(result)
            incomplete["evaluations"] = {}
            try:
                campaign.validate_exact_result(incomplete)
            except RuntimeError as error:
                assert "main5m and main10m" in str(error)
            else:
                raise AssertionError("empty exact evaluations must not pass publication")

            core.write_json_atomic(campaign.SPEC.raw_result_path, result)
            assert campaign.write_tracked_manifest_if_complete()
            manifest = json.loads(
                campaign.TRACKED_MANIFEST.read_text(encoding="utf-8"))
            assert manifest["git_sha"] == core.git_sha()
            assert manifest["baseline_reference"]["git_sha"] == "baseline-git-sha"
            assert manifest["baseline_reference"]["old_qref_critic_input"] == (
                "observation_only_v1")
            assert manifest["baseline_reference"]["wandb_ids"] == {
                "dqc": "dqcg23s0",
                "qref": "qref23s0",
                "qcpo": "qcpo23s0",
            }
            assert set(manifest["results"]) == {"qref_context"}

            # Once the result exists, changed baseline bytes must fail rather than silently
            # publishing a comparison against evidence different from the preregistered hash.
            campaign.TRACKED_MANIFEST.unlink()
            campaign.BASELINE_MANIFEST.write_text(
                "{\"campaign\": \"tampered\"}\n", encoding="utf-8")
            try:
                campaign.write_tracked_manifest_if_complete()
            except RuntimeError as error:
                assert "hash mismatch" in str(error)
            else:
                raise AssertionError("tampered baseline manifest must be rejected")
    finally:
        (
            campaign.BASE_DIR, campaign.RAW_ROOT, campaign.TRACKED_MANIFEST,
            campaign.WANDB_SYNC_MARKER, campaign.BASELINE_MANIFEST,
            campaign.BASELINE_MANIFEST_SHA256, core.BASE_DIR, core.git_sha,
        ) = saved


def test_explicit_online_sync_covers_all_local_segments_and_gates_failure(tmp_path):
    """The wrapper invokes wandb sync --include-online and writes no marker on failure."""
    saved = (
        campaign.BASE_DIR, campaign.RAW_ROOT,
        campaign.WANDB_SYNC_MARKER, campaign.subprocess.run, core.git_sha)
    campaign.BASE_DIR = tmp_path
    campaign.RAW_ROOT = tmp_path / "raw"
    campaign.WANDB_SYNC_MARKER = campaign.RAW_ROOT / "wandb_sync.json"
    wandb_root = tmp_path / "wandb"
    first = wandb_root / "run-20260804-000000-qrefctx23s0"
    second = wandb_root / "run-20260804-000001-qrefctx23s0"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    calls = []

    def successful_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    try:
        # subprocess.check_output internally calls subprocess.run. Freeze the already-tested git
        # helper so this test can replace only the external W&B process boundary.
        core.git_sha = lambda: "context-test-git-sha"
        campaign.subprocess.run = successful_run
        marker = campaign.sync_wandb_online()
        command, kwargs = calls[0]
        assert command[:5] == [
            campaign.sys.executable, "-m", "wandb", "sync", "--include-online"]
        assert command[5:] == [str(first), str(second)]
        assert kwargs["env"]["WANDB_MODE"] == "online"
        assert marker["synced"] is True
        assert campaign.WANDB_SYNC_MARKER.exists()

        # A failed re-sync must not manufacture a new success marker. Remove the old marker first
        # to model an interrupted initial run rather than a later optional verification attempt.
        campaign.WANDB_SYNC_MARKER.unlink()
        campaign.subprocess.run = lambda *_args, **_kwargs: SimpleNamespace(returncode=7)
        try:
            campaign.sync_wandb_online()
        except RuntimeError as error:
            assert "exit code 7" in str(error)
        else:
            raise AssertionError("nonzero W&B sync must fail the formal lifecycle")
        assert not campaign.WANDB_SYNC_MARKER.exists()
    finally:
        (
            campaign.BASE_DIR, campaign.RAW_ROOT,
            campaign.WANDB_SYNC_MARKER, campaign.subprocess.run, core.git_sha,
        ) = saved
