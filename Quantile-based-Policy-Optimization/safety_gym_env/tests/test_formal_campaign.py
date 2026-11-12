# -*- coding: utf-8 -*-
"""Formal two-run command identity, milestone hash, and early-summary tests."""
import json

import torch

import formal_qcpo_ref_dqc_gae as campaign


def _save_iteration(path, iteration):
    """Write the minimal torch payload consumed by checkpoint_iteration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"iteration": int(iteration)}, path)


def test_formal_commands_fix_ordered_identities_budgets_and_wandb_names():
    """QCPO_REF has 10M actual steps; DQC advertises 10M main plus 250k initialization."""
    qref = campaign.SPECS["qref"]
    dqc = campaign.SPECS["dqc"]
    qref_command = campaign.build_training_command(
        qref, device="0", wandb_mode="online")
    dqc_command = campaign.build_training_command(
        dqc, device="0", wandb_mode="online")

    # Parse only the parameter override suffix; ordinary argparse values are asserted separately.
    qref_set = qref_command[qref_command.index("--set") + 1:]
    dqc_set = dqc_command[dqc_command.index("--set") + 1:]
    assert "num_iterations=400" in qref_set
    assert "initialization_env_steps=0" in qref_set
    assert "actual_env_steps=10000000" in qref_set
    assert "policy_identity=paper_negated_return_eq275_discounted_bh_v2" in qref_set
    assert "wandb_group=qcpo_ref_return_quantile_discounted_v2" in qref_set

    assert "num_iterations=400" in dqc_set
    assert "initialization_env_steps=250000" in dqc_set
    assert "actual_env_steps=10250000" in dqc_set
    assert "dqc_update_mode=finite_nstep_v1" in dqc_set
    assert "dqc_mean_estimator=state_value_gae_v1" in dqc_set
    assert "policy_identity=finite_nstep_v1_state_value_gae_v1" in dqc_set
    assert "wandb_group=dqc_gae_mean_simplebutton_v1" in dqc_set
    assert qref.wandb_id == "qrefd2a"
    assert dqc.wandb_id == "dqcg2m1a"
    assert qref.tmux_name == "qref_discocc_lr2e4_10m_s0"
    assert dqc.tmux_name == "dqc_gae_mean_lr2e4_10m_s0"


def test_milestone_capture_hashes_exact_batch200_and_publishes_ready_last(tmp_path):
    """Watcher freezes endpoint/selected, verifies hashes, and rejects a missed batch-200 window."""
    previous_root = campaign.RAW_ROOT
    campaign.RAW_ROOT = tmp_path
    try:
        spec = campaign.SPECS["qref"]
        _save_iteration(spec.live_checkpoint_dir / "latest.pt", 199)
        _save_iteration(spec.live_checkpoint_dir / "best.pt", 179)
        assert campaign.capture_five_million(spec)
        assert (spec.milestone_checkpoint_dir / "READY").exists()

        metadata = json.loads(
            (spec.milestone_checkpoint_dir / "hashes.json").read_text(
                encoding="utf-8"))
        assert metadata["endpoint_iteration"] == 199
        assert metadata["selected_iteration"] == 179
        for name, digest in metadata["sha256"].items():
            assert campaign.sha256_file(
                spec.milestone_checkpoint_dir / name) == digest
        assert campaign.verify_milestone(spec)

        # READY metadata is part of the immutable label, not merely a convenient hash container.
        # A selected checkpoint changed after capture must fail even if an attacker refreshes its
        # hash but forgets/tries to preserve the original selected iteration label.
        _save_iteration(spec.milestone_checkpoint_dir / "best.pt", 189)
        metadata["sha256"]["best.pt"] = campaign.sha256_file(
            spec.milestone_checkpoint_dir / "best.pt")
        campaign.write_json_atomic(
            spec.milestone_checkpoint_dir / "hashes.json", metadata)
        try:
            campaign.verify_milestone(spec)
        except RuntimeError as error:
            assert "selected iteration differs" in str(error)
        else:
            raise AssertionError("READY selected iteration metadata must be enforced")

        # A separate lineage that first appears at batch 400 cannot be silently named main5m.
        missed = campaign.FormalSpec(
            key="missed",
            algorithm="QCPO_REF",
            wandb_id="missed",
            wandb_group="group",
            wandb_tag="tag",
            tmux_name="tmux",
            parent_policy_identity="parent",
            policy_identity="policy",
            variant_changes="change",
            initialization_batches=0,
        )
        _save_iteration(missed.live_checkpoint_dir / "latest.pt", 399)
        try:
            campaign.capture_five_million(missed)
        except RuntimeError as error:
            assert "lost exact 5M checkpoint" in str(error)
        else:
            raise AssertionError("batch 400 must not be mislabeled as batch 200")
    finally:
        campaign.RAW_ROOT = previous_root


def test_early_summary_uses_only_formal_steps_through_two_million():
    """AUC and diagnostic means exclude warmup and any formal row after main step 2M."""
    validations = []
    diagnostics = []
    for iteration in (9, 19, 39, 79, 89):
        validations.append({
            "iteration": iteration,
            "mean": float(iteration),
            "diagnostic_reward_critic_calibration": {
                "initial_cdf_abs_gap": 0.1 * iteration,
                "brier_score": 0.01 * iteration,
            },
        })
    for iteration in range(90):
        diagnostics.append({
            "iteration": iteration,
            "main_env_steps": (iteration + 1) * 25000,
            "actor/clip_fraction": float(iteration),
            "diagnostic/dqc/policy_kl_after_epochs": 0.1 * iteration,
            "diagnostic/dqc/value_rmse": 2.0 * iteration,
            "diagnostic/dqc/critic_cdf_mean": 3.0 * iteration,
        })
    summary = campaign.summarize_early_history({
        "validation_history": validations,
        "training_diagnostics_history": diagnostics,
    })

    # iteration 89 is 2.25M and excluded; formal iterations 0..79 remain in diagnostics.
    assert summary["validation_points"] == 4
    assert summary["clip_fraction_mean"] == 39.5
    assert summary["policy_kl_after_epochs_mean"] == 3.95
    assert summary["value_rmse_mean"] == 79.0
    assert summary["critic_cdf_mean"] == 118.5
    assert summary["cdf_initial_abs_gap_mean"] == (
        0.1 * (9 + 19 + 39 + 79) / 4.0)
    assert summary["validation_mean_auc_env_steps"] is not None
    assert summary["validation_mean_auc_normalized"] is not None


def test_manifest_records_legacy_step_semantics_only_after_both_results(tmp_path):
    """Tracked manifest is gated on two valid raw results and labels both legacy identities."""
    previous_root = campaign.RAW_ROOT
    previous_manifest = campaign.TRACKED_MANIFEST
    campaign.RAW_ROOT = tmp_path / "raw"
    campaign.TRACKED_MANIFEST = tmp_path / "tracked" / "manifest.json"
    try:
        assert not campaign.write_tracked_manifest_if_complete()
        for alias, spec in campaign.SPECS.items():
            campaign.write_json_atomic(
                spec.raw_result_path, {"alias": alias, "evaluations": {}})
        assert campaign.write_tracked_manifest_if_complete()
        manifest = json.loads(
            campaign.TRACKED_MANIFEST.read_text(encoding="utf-8"))
        assert manifest["formal_order"] == ["qref", "dqc"]
        assert manifest["legacy_runs"]["qref2e4a"]["policy_identity"] == "uniform_time_v1"
        assert "9.75M" in manifest["legacy_runs"]["dqcp8k1a"]["budget_note"]
        assert manifest["seed0_engineering_gate_only"] is True
    finally:
        campaign.RAW_ROOT = previous_root
        campaign.TRACKED_MANIFEST = previous_manifest
