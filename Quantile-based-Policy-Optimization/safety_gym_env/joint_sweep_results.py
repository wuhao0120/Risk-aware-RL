#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict validation and CSV rendering for the SimpleButton joint sweep harness."""
import csv
import json
import math
import os
import statistics
from pathlib import Path

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
    RunSpec,
    enumerate_runs,
)


# These four measurements appear for both endpoint and validation-selected policies.  The fifth
# horizontal row, selected_batch, is meaningful only for the selected policy.
PRIMARY_METRICS = (
    "outage_probability",
    "reward_q20",
    "reward_mean",
    "episode_return_variance",
)


def expected_setting_names():
    """Return four LR x checkpoint-budget settings in deterministic comparison order."""
    names = []
    for lr_label, _actor_lr0 in LEARNING_RATES:
        for batch in (MILESTONE_BATCH, FINAL_BATCH):
            template = RunSpec(lr_label, 0.0, ALGORITHMS[0], 0)
            names.append(template.result_setting(batch))
    return names


def load_evaluation_records(specs=None):
    """Load one ignored raw post-hoc JSON per lineage and fail on any missing file."""
    specs = list(enumerate_runs() if specs is None else specs)
    records = []
    missing = []
    for spec in specs:
        if not spec.evaluation_path.exists():
            missing.append(str(spec.evaluation_path))
            continue
        records.append(json.loads(spec.evaluation_path.read_text(encoding="utf-8")))
    if missing:
        raise ValueError("missing post-hoc result(s):\n  " + "\n  ".join(missing))
    validate_evaluation_records(records, specs)
    return records


def _record_key(record):
    """Canonical lineage key used to reject duplicate or unexpected results."""
    return (
        str(record.get("algorithm")),
        int(record.get("seed", -1)),
        float(record.get("actor_lr0", float("nan"))),
    )


def validate_evaluation_records(records, specs=None):
    """
    Enforce exact algorithm/seed coverage, configuration identity, budgets, CRN, and N=1000.

    Both policy roles are checked independently.  QPO outage is always read from ``empirical_prob``
    relative to fixed q=-.03; its tracker calibration field cannot satisfy this contract.
    """
    specs = list(enumerate_runs() if specs is None else specs)
    expected = {
        (spec.algorithm, spec.seed, float(spec.actor_lr0)): spec
        for spec in specs
    }
    actual_keys = [_record_key(record) for record in records]
    if len(actual_keys) != len(set(actual_keys)):
        raise ValueError("duplicate algorithm/seed/actor_lr0 lineage in evaluation records")
    if set(actual_keys) != set(expected):
        missing = sorted(set(expected) - set(actual_keys))
        unexpected = sorted(set(actual_keys) - set(expected))
        raise ValueError(f"lineage coverage mismatch: missing={missing}, unexpected={unexpected}")

    for record in records:
        spec = expected[_record_key(record)]
        common_expected = {
            "campaign": CAMPAIGN,
            "environment": ENVIRONMENT,
            "return_threshold": RETURN_THRESHOLD,
            "failure_alpha": FAILURE_ALPHA,
            "gamma": GAMMA,
            "num_envs": NUM_ENVS,
            "horizon": HORIZON,
            "policy_identity": spec.policy_identity,
        }
        for field, value in common_expected.items():
            if record.get(field) != value:
                raise ValueError(
                    f"{spec.lineage_id}: {field}={record.get(field)!r}, expected {value!r}")

        protocol = record.get("evaluation_protocol") or {}
        if int(protocol.get("num_episodes", -1)) != FINAL_EVALUATION_EPISODES:
            raise ValueError(f"{spec.lineage_id}: evaluation protocol is not exact N=1000")
        if protocol.get("reset_seed") != protocol.get("action_rng_seed"):
            raise ValueError(f"{spec.lineage_id}: reset/action CRN seeds differ")
        if int(protocol.get("reset_seed", -1)) != FINAL_EVALUATION_SEED:
            raise ValueError(f"{spec.lineage_id}: unexpected common evaluation seed")

        expected_settings = {
            spec.result_setting(MILESTONE_BATCH): MILESTONE_BATCH,
            spec.result_setting(FINAL_BATCH): FINAL_BATCH,
        }
        evaluations = record.get("evaluations") or {}
        if set(evaluations) != set(expected_settings):
            raise ValueError(
                f"{spec.lineage_id}: budget settings are {sorted(evaluations)}, "
                f"expected {sorted(expected_settings)}")

        for setting, batch in expected_settings.items():
            result = evaluations[setting]
            budget = result.get("budget") or {}
            initialization = 2 * NUM_ENVS * HORIZON if spec.algorithm == "QPO" else 0
            expected_budget = {
                "main_rollout_batches": batch,
                "main_env_steps": batch * NUM_ENVS * HORIZON,
                "initialization_env_steps": initialization,
                "actual_env_steps": batch * NUM_ENVS * HORIZON + initialization,
                "observed_actual_env_steps": batch * NUM_ENVS * HORIZON + initialization,
            }
            for field, value in expected_budget.items():
                if int(budget.get(field, -1)) != int(value):
                    raise ValueError(
                        f"{spec.lineage_id}/{setting}: budget {field}="
                        f"{budget.get(field)!r}, expected {value}")

            endpoint_iteration = int(result.get("endpoint_iteration", -1))
            selected_iteration = int(result.get("selected_iteration", -1))
            if endpoint_iteration != batch - 1:
                raise ValueError(f"{spec.lineage_id}/{setting}: wrong endpoint checkpoint")
            if not 0 <= selected_iteration <= endpoint_iteration:
                raise ValueError(f"{spec.lineage_id}/{setting}: selected checkpoint is out of range")

            for role in ("endpoint", "selected"):
                metrics = result.get(f"eval_{role}") or {}
                if int(metrics.get("num_episodes", -1)) != FINAL_EVALUATION_EPISODES:
                    raise ValueError(
                        f"{spec.lineage_id}/{setting}/{role}: evaluation is not exact N=1000")
                required = (
                    "empirical_prob", "failure_count", "wilson95_low", "wilson95_high",
                    "quantile_return", "mean", "reward_std",
                )
                for field in required:
                    value = metrics.get(field)
                    if value is None or not math.isfinite(float(value)):
                        raise ValueError(
                            f"{spec.lineage_id}/{setting}/{role}: invalid {field}={value!r}")

                # Inclusive failures must be internally consistent with the exact 1000 episodes.
                failure_count = int(metrics["failure_count"])
                probability = float(metrics["empirical_prob"])
                if not math.isclose(
                        probability, failure_count / FINAL_EVALUATION_EPISODES,
                        rel_tol=0.0, abs_tol=1e-15):
                    raise ValueError(
                        f"{spec.lineage_id}/{setting}/{role}: count/probability mismatch")
            if result.get("eval") != result.get("eval_selected"):
                raise ValueError(f"{spec.lineage_id}/{setting}: eval alias is not selected")
    return True


def metric_values(metrics, selected_iteration):
    """Map evaluator output to fixed-q, q20, mean, population episode variance, and batch."""
    return {
        "outage_probability": float(metrics["empirical_prob"]),
        "reward_q20": float(metrics["quantile_return"]),
        "reward_mean": float(metrics["mean"]),
        # reward_std is NumPy's ddof=0 statistic on the 1000 evaluation episodes.
        "episode_return_variance": float(metrics["reward_std"]) ** 2,
        "selected_batch": int(selected_iteration) + 1,
    }


def per_seed_rows(records, setting):
    """Build 18 machine-readable rows (3 algorithms x 3 seeds x 2 policy roles)."""
    rows = []
    for record in records:
        result = (record.get("evaluations") or {}).get(setting)
        if result is None:
            continue
        budget = result["budget"]
        for role in ("endpoint", "selected"):
            metrics = result[f"eval_{role}"]
            checkpoint_iteration = int(result[f"{role}_iteration"])
            values = metric_values(metrics, checkpoint_iteration)
            rows.append({
                "campaign": CAMPAIGN,
                "setting": setting,
                "environment": ENVIRONMENT,
                "algorithm": record["algorithm"],
                "seed": int(record["seed"]),
                "policy_role": role,
                "checkpoint": result[f"{role}_checkpoint"],
                "checkpoint_iteration": checkpoint_iteration,
                "wandb_id": record["wandb_id"],
                "policy_identity": record["policy_identity"],
                "actor_lr0": float(record["actor_lr0"]),
                "q": float(record["return_threshold"]),
                "alpha": float(record["failure_alpha"]),
                "gamma": float(record["gamma"]),
                "num_envs": int(record["num_envs"]),
                "horizon": int(record["horizon"]),
                "main_rollout_batches": int(budget["main_rollout_batches"]),
                "main_env_steps": int(budget["main_env_steps"]),
                "initialization_env_steps": int(budget["initialization_env_steps"]),
                "actual_env_steps": int(budget["actual_env_steps"]),
                "observed_actual_env_steps": int(budget["observed_actual_env_steps"]),
                "evaluation_n": int(metrics["num_episodes"]),
                "evaluation_seed": FINAL_EVALUATION_SEED,
                "failure_count": int(metrics["failure_count"]),
                "wilson95_low": float(metrics["wilson95_low"]),
                "wilson95_high": float(metrics["wilson95_high"]),
                **values,
            })
    expected_count = len(ALGORITHMS) * len(SEEDS) * 2
    if len(rows) != expected_count:
        raise ValueError(f"{setting}: expected {expected_count} per-seed rows, got {len(rows)}")
    return sorted(rows, key=lambda row: (
        ALGORITHMS.index(row["algorithm"]), row["seed"], row["policy_role"]))


def sample_summary(values):
    """Return mean and ddof=1 sample variance across exactly three training seeds."""
    values = [float(value) for value in values]
    if len(values) != len(SEEDS):
        raise ValueError(f"sample summary needs exactly three seeds, got {len(values)}")
    return statistics.fmean(values), statistics.variance(values)


def setting_table_rows(seed_rows):
    """Create metric rows with each algorithm's seed0/1/2, mean, and seed sample variance."""
    table = []
    row_specs = []
    for role in ("endpoint", "selected"):
        for metric in PRIMARY_METRICS:
            row_specs.append((role, metric))
    row_specs.append(("selected", "selected_batch"))

    for role, metric in row_specs:
        output = {"metric": f"{role}_{metric}"}
        for algorithm in ALGORITHMS:
            selected = [
                row for row in seed_rows
                if row["algorithm"] == algorithm and row["policy_role"] == role
            ]
            selected.sort(key=lambda row: int(row["seed"]))
            values = [row[metric] for row in selected]
            mean, sample_variance = sample_summary(values)
            for seed, value in zip(SEEDS, values):
                output[f"{algorithm}_seed{seed}"] = value
            output[f"{algorithm}_mean"] = mean
            output[f"{algorithm}_sample_variance"] = sample_variance
        table.append(output)
    return table


def aggregate_long_rows(all_seed_rows):
    """Collapse setting x algorithm x role into mean and ddof=1 variance columns."""
    output = []
    for setting in expected_setting_names():
        setting_rows = [row for row in all_seed_rows if row["setting"] == setting]
        for algorithm in ALGORITHMS:
            for role in ("endpoint", "selected"):
                group = [
                    row for row in setting_rows
                    if row["algorithm"] == algorithm and row["policy_role"] == role
                ]
                group.sort(key=lambda row: row["seed"])
                if len(group) != len(SEEDS):
                    raise ValueError(f"{setting}/{algorithm}/{role}: incomplete three-seed group")
                row = {
                    "campaign": CAMPAIGN,
                    "setting": setting,
                    "environment": ENVIRONMENT,
                    "algorithm": algorithm,
                    "policy_role": role,
                    "actor_lr0": group[0]["actor_lr0"],
                    "main_env_steps": group[0]["main_env_steps"],
                    "initialization_env_steps": group[0]["initialization_env_steps"],
                    "actual_env_steps": group[0]["actual_env_steps"],
                    "evaluation_n_per_seed": FINAL_EVALUATION_EPISODES,
                    "training_seed_count": len(SEEDS),
                }
                metrics = PRIMARY_METRICS + (("selected_batch",) if role == "selected" else ())
                for metric in metrics:
                    mean, variance = sample_summary([item[metric] for item in group])
                    row[f"{metric}_mean"] = mean
                    row[f"{metric}_sample_variance"] = variance
                output.append(row)
    return output


def aggregate_wide_rows(long_rows):
    """Pivot summaries to one row per setting for LR and 5M-versus-10M comparisons."""
    output = []
    for setting in expected_setting_names():
        groups = [row for row in long_rows if row["setting"] == setting]
        if len(groups) != len(ALGORITHMS) * 2:
            raise ValueError(f"{setting}: wide pivot received {len(groups)} groups")
        first = groups[0]
        row = {
            "campaign": CAMPAIGN,
            "setting": setting,
            "environment": ENVIRONMENT,
            "actor_lr0": first["actor_lr0"],
            "main_env_steps": first["main_env_steps"],
        }
        for group in groups:
            prefix = f"{group['algorithm']}_{group['policy_role']}"
            for key, value in group.items():
                if key.endswith("_mean") or key.endswith("_sample_variance"):
                    row[f"{prefix}_{key}"] = value
        output.append(row)
    return output


def _write_csv_atomic(path, rows):
    """Write a deterministic small CSV through an atomic same-directory replacement."""
    rows = list(rows)
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with open(temporary, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, destination)


def aggregate_campaign(output_root=TRACKED_ROOT, records=None):
    """Validate first, then emit four horizontal settings and both longitudinal tables."""
    records = load_evaluation_records() if records is None else list(records)
    validate_evaluation_records(records)
    output_root = Path(output_root)
    all_seed_rows = []
    for setting in expected_setting_names():
        rows = per_seed_rows(records, setting)
        all_seed_rows.extend(rows)
        setting_dir = output_root / setting
        _write_csv_atomic(setting_dir / "per_seed.csv", rows)
        _write_csv_atomic(setting_dir / "setting_table.csv", setting_table_rows(rows))

    long_rows = aggregate_long_rows(all_seed_rows)
    wide_rows = aggregate_wide_rows(long_rows)
    _write_csv_atomic(output_root / "aggregate_long.csv", long_rows)
    _write_csv_atomic(output_root / "aggregate_wide.csv", wide_rows)
    return {"per_seed": all_seed_rows, "long": long_rows, "wide": wide_rows}
