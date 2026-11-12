# -*- coding: utf-8 -*-
"""Dynamic seed-0 stage gates, command identities, resume contracts, and exact schema tests."""
import json  # Build deterministic lineage and result fixtures.
import os  # Select headless MuJoCo before constructing the real Dynamic environment.
from pathlib import Path  # Redirect every generated test artifact into tmp_path.
import sys  # Install one in-memory W&B API double without importing external test frameworks.
from types import SimpleNamespace  # Provide the minimal W&B module surface used by the harness.

import torch  # Serialize realistic checkpoint metadata and signatures.

import formal_dynamic_seed0 as campaign
import formal_qcpo_ref_dqc_gae as core


def _override_suffix(command):
    """Return only signed key=value values after the runner's explicit marker."""
    return command[command.index("--set") + 1:]


def _expect_cli_failure(arguments, message):
    """Require argparse to reject an invalid stage request before any artifact is created."""
    try:
        campaign.parse_cli(arguments)
    except SystemExit as error:
        assert int(error.code) != 0
    else:
        raise AssertionError(message)


def test_dynamic_environment_is_registered_with_obs44_action2():
    """Construct the actual registered environment instead of trusting a static config table."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    from envs import SafetyEnv

    env = SafetyEnv("Dynamic")
    try:
        observation, _info = env.reset(seed=0)
        assert env.obs_dim == 44
        assert env.act_dim == 2
        assert tuple(observation.shape) == (44,)
        assert tuple(env.action_space.shape) == (2,)
    finally:
        env.close()


def test_commands_freeze_qpo_and_three_constrained_recipes():
    """QPO has no q; constrained commands lock budgets, threshold, context, LOO, and DQC M=4."""
    qpo_config = campaign.make_stage_config("qpo")
    initial = campaign.make_stage_config("initial", return_threshold=1.25)
    strict = campaign.make_stage_config(
        "strict", return_threshold=1.50, q0=1.25)

    # Identities and artifact roots must differ across algorithms, stages, and threshold values.
    all_specs = [
        *qpo_config.specs.values(),
        *initial.specs.values(),
        *strict.specs.values(),
    ]
    assert len({spec.wandb_id for spec in all_specs}) == len(all_specs)
    assert qpo_config.raw_root != initial.raw_root != strict.raw_root

    with campaign.installed_core_identity(qpo_config):
        qpo = core.build_training_command(
            qpo_config.specs["qpo"], "0", "online")
        suffix = _override_suffix(qpo)
        assert "--return_threshold" not in qpo
        assert qpo[qpo.index("--env") + 1] == "Dynamic"
        assert qpo[qpo.index("--checkpoint_every") + 1] == "50"
        assert "num_iterations=400" in suffix
        assert "main_env_steps=10000000" in suffix
        assert "initialization_env_steps=50000" in suffix
        assert "actual_env_steps=10050000" in suffix

    with campaign.installed_core_identity(initial):
        commands = {
            alias: core.build_training_command(spec, "0", "online")
            for alias, spec in initial.specs.items()
        }
        for alias, command in commands.items():
            suffix = _override_suffix(command)
            assert command[command.index("--return_threshold") + 1] == "1.25"
            assert command[command.index("--checkpoint_every") + 1] == "50"
            assert "num_iterations=400" in suffix
            assert "main_env_steps=10000000" in suffix
            if alias == "dqc":
                assert "initialization_env_steps=250000" in suffix
                assert "actual_env_steps=10250000" in suffix
                assert "dqc_update_mode=finite_nstep_v1" in suffix
                assert "dqc_mean_estimator=state_value_gae_v1" in suffix
                assert "num_action_samples=4" in suffix
                text = " ".join(command).lower()
                assert "learned_state_distribution_v1" not in text
                assert "state_cdf" not in text
            else:
                assert "initialization_env_steps=0" in suffix
                assert "actual_env_steps=10000000" in suffix
        assert "dqc_finite_context_v1" in initial.specs["qref"].policy_identity
        assert initial.specs["qcpo"].policy_identity == "qcpo_reward_chance_loo_v1"


def test_stage_cli_thresholds_and_resume_flags_fail_fast():
    """Missing q, missing q0, q1<=q0, QPO q, and non-run resume all fail during parsing."""
    _expect_cli_failure(
        ["dry-run", "--stage", "initial"],
        "initial stage must require q0")
    _expect_cli_failure(
        ["dry-run", "--stage", "strict", "--return-threshold", "2.0"],
        "strict stage must require prior q0")
    _expect_cli_failure(
        [
            "dry-run", "--stage", "strict",
            "--return-threshold", "1.25", "--q0", "1.25",
        ],
        "strict q1 must exceed q0")
    _expect_cli_failure(
        [
            "dry-run", "--stage", "strict",
            "--return-threshold", "1.0", "--q0", "1.25",
        ],
        "strict q1 reversal must fail")
    _expect_cli_failure(
        ["dry-run", "--stage", "qpo", "--return-threshold", "1.0"],
        "QPO must not receive a fixed q")
    _expect_cli_failure(
        ["verify", "--stage", "qpo", "--allow-resume"],
        "resume is meaningful only for run")
    _expect_cli_failure(
        ["run", "--stage", "qpo", "--wandb-mode", "disabled"],
        "formal runs must retain the online sync gate")


def test_resume_contract_rejects_cross_threshold_and_sha(tmp_path):
    """Only one exact environment/algorithm/q/SHA/checkpoint-signature lineage can resume."""
    old_root = campaign.RAW_CAMPAIGN_ROOT
    campaign.RAW_CAMPAIGN_ROOT = tmp_path / "raw"
    executable_sha = core.git_sha()
    try:
        initial = campaign.make_stage_config(
            "initial", return_threshold=1.25)
        with campaign.installed_core_identity(initial):
            spec = initial.specs["qcpo"]
            contract = campaign.lineage_contract(
                initial, "qcpo", spec, executable_sha)
            core.write_json_atomic(
                campaign.lineage_contract_path(spec), contract)
            signature = campaign.requested_signature(spec)
            spec.live_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "iteration": 49,
                "agent_class": "QCPOGPU",
                "experiment_signature": signature,
            }, spec.live_checkpoint_dir / "latest.pt")
            checkpoint = campaign.validate_resume_lineage(
                initial, "qcpo", spec, executable_sha)
            assert checkpoint.name == "latest.pt"

            try:
                campaign.validate_resume_lineage(
                    initial, "qcpo", spec, "different-sha")
            except RuntimeError as error:
                assert "differs" in str(error)
            else:
                raise AssertionError("cross-SHA resume must fail")

        # Simulate an accidental copy of q0 files into a distinct strict-q1 artifact root.
        strict = campaign.make_stage_config(
            "strict", return_threshold=1.50, q0=1.25)
        with campaign.installed_core_identity(strict):
            strict_spec = strict.specs["qcpo"]
            core.write_json_atomic(
                campaign.lineage_contract_path(strict_spec), contract)
            strict_spec.live_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "iteration": 49,
                "agent_class": "QCPOGPU",
                "experiment_signature": signature,
            }, strict_spec.live_checkpoint_dir / "latest.pt")
            try:
                campaign.validate_resume_lineage(
                    strict, "qcpo", strict_spec, executable_sha)
            except RuntimeError as error:
                assert "differs" in str(error)
            else:
                raise AssertionError("q0 checkpoint must not resume strict q1")
    finally:
        campaign.RAW_CAMPAIGN_ROOT = old_root


def test_reusable_resume_accepts_signed_initialization_cursor(tmp_path):
    """QPO/DQC interruptions before main batch one resume initialization_latest rather than restart."""
    old_root = core.RAW_ROOT
    old_base = core.BASE_DIR
    core.RAW_ROOT = tmp_path / "raw"
    core.BASE_DIR = tmp_path / "base"
    try:
        spec = core.FormalSpec(
            key="warmup",
            algorithm="QPO",
            wandb_id="warmup",
            wandb_group="group",
            wandb_tag="tag",
            tmux_name="tmux",
            parent_policy_identity="parent",
            policy_identity="policy",
            variant_changes="test",
            initialization_batches=2,
            uses_return_threshold=False,
        )
        path = spec.live_checkpoint_dir / "initialization_latest.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"iteration": -1}, path)
        assert core.resume_checkpoint(spec) == path

        # Endpoint state is also resumable when the runner JSON/W&B finish was interrupted.
        latest = spec.live_checkpoint_dir / "latest.pt"
        torch.save({"iteration": 399}, latest)
        assert core.resume_checkpoint(spec) == latest
        spec.runner_result_path.parent.mkdir(parents=True, exist_ok=True)
        spec.runner_result_path.write_text("{}\n", encoding="utf-8")
        assert core.resume_checkpoint(spec) is None
    finally:
        core.RAW_ROOT = old_root
        core.BASE_DIR = old_base


def test_wandb_history_audit_requires_exact_400_finished_rows():
    """The service audit counts uploaded main rows and rebuilds eight 0--2M validation points."""
    rows = []
    for iteration in range(400):
        row = {
            "progress/iteration": iteration,
            "disc_reward/aver_reward": float(iteration),
        }
        if (iteration + 1) % 10 == 0:
            row["validation/disc_reward/aver_reward"] = float(iteration)
        rows.append(row)

    class FakeRun:
        """Expose exactly the scan/state fields read by the campaign."""

        state = "finished"

        def scan_history(self, page_size):
            assert page_size == 1000
            return iter(rows)

    class FakeApi:
        """Record path resolution through the authenticated default entity."""

        default_entity = "entity"

        def __init__(self, timeout):
            assert timeout == 60

        def run(self, path):
            assert path == "entity/safety_gym_reward_risk/dynqpos0v1"
            return FakeRun()

    old_wandb = sys.modules.get("wandb")
    sys.modules["wandb"] = SimpleNamespace(Api=FakeApi)
    try:
        spec = campaign.make_stage_config("qpo").specs["qpo"]
        audit = campaign.query_wandb_main_history(spec)
        assert audit["state"] == "finished"
        assert audit["main_history_rows"] == 400
        assert audit["main_iterations"] == [0, 399]
        assert audit["early_0_2m"]["validation_points"] == 8

        # Missing the last main row is a hard failure; suppress retries in this unit fixture.
        rows.pop()
        old_sleep = campaign.time.sleep
        campaign.time.sleep = lambda _seconds: None
        try:
            campaign.query_wandb_main_history(spec)
        except RuntimeError as error:
            assert "history audit failed" in str(error)
        else:
            raise AssertionError("399 uploaded main rows must fail verification")
        finally:
            campaign.time.sleep = old_sleep
    finally:
        if old_wandb is None:
            del sys.modules["wandb"]
        else:
            sys.modules["wandb"] = old_wandb


def _exact_metrics(threshold):
    """Return a finite exact-N constrained metric fixture with internally consistent failures."""
    return {
        "num_episodes": 1000,
        "mean": 2.0,
        "reward_std": 0.5,
        "quantile_return": 1.5,
        "return_variance": 0.25,
        "return_threshold": threshold,
        "failure_count": 100,
        "empirical_prob": 0.1,
        "wilson95_low": 0.08,
        "wilson95_high": 0.12,
    }


def test_exact_result_schema_checks_hash_signature_iterations_and_clocks(tmp_path):
    """Validate all four exact policies against immutable bytes and stage-specific signatures."""
    old_root = campaign.RAW_CAMPAIGN_ROOT
    campaign.RAW_CAMPAIGN_ROOT = tmp_path / "raw"
    executable_sha = core.git_sha()
    try:
        config = campaign.make_stage_config(
            "initial", return_threshold=1.25)
        with campaign.installed_core_identity(config):
            spec = config.specs["qcpo"]
            signature = campaign.requested_signature(spec)
            evaluations = {}
            iterations = {
                "main5m": {"endpoint": 199, "selected": 149},
                "main10m": {"endpoint": 399, "selected": 349},
            }
            for budget, group in core.policy_checkpoints(spec).items():
                records = {}
                for role in ("endpoint", "selected"):
                    checkpoint = Path(group[role])
                    iteration = iterations[budget][role]
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "iteration": iteration,
                        "agent_class": "QCPOGPU",
                        "experiment_signature": signature,
                    }, checkpoint)
                    main_steps = (
                        (iteration + 1)
                        * campaign.NUM_ENVS
                        * campaign.HORIZON
                    )
                    records[role] = {
                        "checkpoint": str(checkpoint),
                        "checkpoint_iteration": iteration,
                        "sha256": core.sha256_file(checkpoint),
                        "metrics": _exact_metrics(config.threshold),
                        "step_clocks": {
                            "main_env_steps": main_steps,
                            "initialization_env_steps": 0,
                            "actual_env_steps": main_steps,
                        },
                    }
                evaluations[budget] = records

            raw = {
                "campaign": config.campaign,
                "git_sha": executable_sha,
                "executable_git_sha": executable_sha,
                "environment": "Dynamic",
                "algorithm": "QCPO",
                "seed": 0,
                "wandb_id": spec.wandb_id,
                "wandb_group": spec.wandb_group,
                "policy_identity": spec.policy_identity,
                "training_budget_semantics": "main_excludes_initialization_v1",
                "wandb_main_history_rows": 400,
                "checkpoint_signature_sha256":
                    campaign._signature_digest(signature),
                "evaluation_protocol": {
                    "name": "exact_n1000_common_random_numbers_v1",
                    "num_episodes": 1000,
                    "environment_seed": 777,
                    "action_rng_seed": 777,
                    "checkpoint_rng_restored": False,
                },
                "early_0_2m": {},
                "evaluations": evaluations,
            }
            core.write_json_atomic(spec.raw_result_path, raw)
            result = campaign.validate_exact_result(
                config, "qcpo", spec, executable_sha)
            assert result["evaluations"]["main10m"]["endpoint"][
                "checkpoint_iteration"] == 399

            # Any post-evaluation checkpoint mutation invalidates the raw result hash immediately.
            endpoint = Path(
                core.policy_checkpoints(spec)["main10m"]["endpoint"])
            torch.save({
                "iteration": 399,
                "agent_class": "QCPOGPU",
                "experiment_signature": signature,
                "mutated": True,
            }, endpoint)
            try:
                campaign.validate_exact_result(
                    config, "qcpo", spec, executable_sha)
            except RuntimeError as error:
                assert "hash mismatch" in str(error)
            else:
                raise AssertionError("mutated checkpoint must invalidate exact result")
    finally:
        campaign.RAW_CAMPAIGN_ROOT = old_root


def test_state_cdf_decision_and_claim_limit_are_narrow():
    """Keep the rollback evidence accurate without claiming no gain or measured wall-clock harm."""
    decision = campaign.STATE_CDF_DECISION
    assert decision["decision"] == "not_promoted_to_default"
    assert decision["complexity_delta_percent"] == 20.6
    assert decision["checkpoint_size_delta_percent"] == 25.7
    assert decision["critic_update_work_delta_percent"] == 50.0
    assert "positive 10M endpoint signal" in decision["evidence"]
    assert "early and selected" in decision["evidence"]
    assert "state-CDF had no improvement" in decision["prohibited_claims"]
