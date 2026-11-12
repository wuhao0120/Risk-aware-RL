# -*- coding: utf-8 -*-
"""Binding q=2.30 campaign identity, isolation, and publication-gate tests."""
import json

import torch

import formal_binding_q23 as campaign
import formal_qcpo_ref_dqc_gae as reusable
from run_experiment import base_args, checkpoint_signature


def _override_suffix(command):
    """Return only key=value items after the runner's explicit --set marker."""
    return command[command.index("--set") + 1:]


def test_registered_threshold_order_ids_names_and_budgets_are_frozen():
    """Every fresh lineage uses q=2.30 and the user-requested two-phase order."""
    assert campaign.RETURN_THRESHOLD == 2.30
    assert campaign.FORMAL_ORDER == (("dqc", "qref"), ("qcpo",))
    assert len({spec.wandb_id for spec in campaign.SPECS.values()}) == 3

    # Importing the wrapper must not mutate the prior q=-0.03 module.  The
    # context installs q=2.30 only while its reusable helpers are being called.
    assert reusable.RETURN_THRESHOLD == -0.03
    with campaign.installed_core_identity():
        assert reusable.RETURN_THRESHOLD == 2.30
        for alias, spec in campaign.SPECS.items():
            command = reusable.build_training_command(
                spec, device="0", wandb_mode="online")
            suffix = _override_suffix(command)
            threshold_index = command.index("--return_threshold")
            assert command[threshold_index + 1] == "2.3"
            assert "num_iterations=400" in suffix
            assert "main_env_steps=10000000" in suffix
            assert "training_budget_semantics=main_excludes_initialization_v1" in suffix
            assert f"wandb_group={campaign.WANDB_GROUP}" in suffix
            assert f"wandb_name={spec.wandb_name}" in suffix
            assert len(spec.wandb_tag) <= 64
            assert "--resume" not in command

            if alias == "dqc":
                assert "initialization_env_steps=250000" in suffix
                assert "actual_env_steps=10250000" in suffix
                assert "dqc_update_mode=finite_nstep_v1" in suffix
                assert "dqc_mean_estimator=state_value_gae_v1" in suffix
            else:
                assert "initialization_env_steps=0" in suffix
                assert "actual_env_steps=10000000" in suffix
                assert "dqc_update_mode=finite_nstep_v1" not in suffix
    assert reusable.RETURN_THRESHOLD == -0.03


def test_q23_checkpoint_signature_differs_from_nonbinding_lineage():
    """Threshold is checkpoint-signed, so an old q=-0.03 state cannot resume."""
    old_args = base_args("QCPO", 0, torch.device("cpu"), "SimpleButton")
    old_args.return_threshold = -0.03
    old_signature = checkpoint_signature(old_args, "QCPO")

    new_args = base_args("QCPO", 0, torch.device("cpu"), "SimpleButton")
    new_args.return_threshold = campaign.RETURN_THRESHOLD
    new_args.policy_identity = campaign.SPECS["qcpo"].policy_identity
    new_signature = checkpoint_signature(new_args, "QCPO")

    assert old_signature["return_threshold"] == -0.03
    assert new_signature["return_threshold"] == 2.30
    assert old_signature != new_signature


def test_manifest_waits_for_all_three_and_records_parallel_then_qcpo(tmp_path):
    """No tracked result is published until DQC, QCPO_REF, and QCPO all exist."""
    old_raw = campaign.RAW_ROOT
    old_manifest = campaign.TRACKED_MANIFEST
    campaign.RAW_ROOT = tmp_path / "raw"
    campaign.TRACKED_MANIFEST = tmp_path / "tracked" / "manifest.json"
    try:
        with campaign.installed_core_identity():
            assert not campaign.write_tracked_manifest_if_complete()

            # Phase one alone is insufficient even when both parallel results
            # are valid JSON; QCPO is a hard publication gate.
            for alias in campaign.FIRST_PHASE:
                spec = campaign.SPECS[alias]
                reusable.write_json_atomic(
                    spec.raw_result_path,
                    {"alias": alias, "evaluations": {}})
            assert not campaign.write_tracked_manifest_if_complete()

            qcpo = campaign.SPECS["qcpo"]
            reusable.write_json_atomic(
                qcpo.raw_result_path,
                {"alias": "qcpo", "evaluations": {}})
            assert campaign.write_tracked_manifest_if_complete()

        manifest = json.loads(
            campaign.TRACKED_MANIFEST.read_text(encoding="utf-8"))
        assert manifest["formal_order"] == [["dqc", "qref"], ["qcpo"]]
        assert manifest["fresh_training"] is True
        assert manifest["seed0_only_per_user"] is True
        assert manifest["threshold_preregistration"]["return_threshold"] == 2.30
        assert manifest["threshold_preregistration"]["failure_alpha"] == 0.20
        assert manifest["physical_initialization_steps"] == {
            "dqc": 250000,
            "qref": 0,
            "qcpo": 0,
        }
        assert set(manifest["results"]) == {"dqc", "qref", "qcpo"}
    finally:
        campaign.RAW_ROOT = old_raw
        campaign.TRACKED_MANIFEST = old_manifest


def test_fresh_guard_rejects_any_existing_lineage_artifact(tmp_path):
    """An initial launch cannot silently adopt a checkpoint or partial progress."""
    old_raw = campaign.RAW_ROOT
    old_manifest = campaign.TRACKED_MANIFEST
    old_base = reusable.BASE_DIR
    campaign.RAW_ROOT = tmp_path / "raw"
    campaign.TRACKED_MANIFEST = tmp_path / "tracked" / "manifest.json"
    # FormalSpec.runner_result_path is anchored by the reusable module's BASE_DIR rather than
    # RAW_ROOT. Isolate that legacy runner output too, so completed real campaign artifacts cannot
    # make this unit fixture depend on which experiments happen to exist in the workspace.
    reusable.BASE_DIR = tmp_path / "base"
    try:
        with campaign.installed_core_identity():
            campaign.assert_fresh_campaign()
            progress = campaign.SPECS["dqc"].progress_path
            progress.parent.mkdir(parents=True, exist_ok=True)
            progress.write_text("{}\n", encoding="utf-8")
            try:
                campaign.assert_fresh_campaign()
            except RuntimeError as error:
                assert "--allow_resume" in str(error)
                assert str(progress) in str(error)
            else:
                raise AssertionError("fresh launch must reject an existing progress file")
    finally:
        campaign.RAW_ROOT = old_raw
        campaign.TRACKED_MANIFEST = old_manifest
        reusable.BASE_DIR = old_base
