#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export complete histories from local W&B offline run files without networking.

The existing export_wandb_metrics.py intentionally uses the public W&B API.
This companion handles runs created with wandb_mode=offline and writes the
same core artifacts: per-run config, summary, history, and combined CSV files.
profile_wandb_metrics.py can therefore consume either source unchanged.

Example:
    python export_wandb_offline.py \
        --run wandb/offline-run-20260717_072646-jhm20t8t \
        --run wandb/offline-run-20260717_072720-bodb903c \
        --out _runs/wandb_export/dqc_pm8_multiseed
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Reuse the online exporter's serializers and stable CSV column order. Keeping
# one writer prevents offline and online artifacts from drifting subtly.
from export_wandb_metrics import (
    jsonable,
    safe_name,
    utc_now,
    write_csv,
    write_json,
)


@dataclass
class OfflineRun:
    """
    Fully decoded content of one local W&B record stream.

    History initially contains only logged values. Identity fields are attached
    after the stream is scanned, so protobuf record ordering is irrelevant.
    """

    source: Path
    run_id: str
    name: str
    entity: str
    project: str
    group: str
    tags: List[str]
    created_at: str
    git_commit: str
    config: Dict[str, Any]
    summary: Dict[str, Any]
    history: List[Dict[str, Any]]
    exit_code: Optional[int]


def _decode_json(value_json: str) -> Any:
    """
    Decode the JSON representation stored in one W&B protobuf value.

    A malformed legacy value is kept as raw text rather than aborting an
    otherwise readable experiment export.
    """

    try:
        return json.loads(value_json)
    except (TypeError, json.JSONDecodeError):
        return value_json


def _item_key(item: Any) -> str:
    """
    Recover a key from Config, History, or Summary update items.

    W&B 0.18 stores ordinary metric names in nested_key. Multiple components
    occur for actual nested objects and are flattened with a dot.
    """

    nested = list(getattr(item, "nested_key", ()))
    if nested:
        return nested[0] if len(nested) == 1 else ".".join(nested)
    return str(getattr(item, "key", ""))


def _apply_updates(container: Any, target: Dict[str, Any]) -> None:
    """
    Replay one incremental ConfigRecord or SummaryRecord into target.

    Summary records are deltas rather than full snapshots. Both updates and
    removals must therefore be replayed to reproduce the final summary.
    """

    for item in getattr(container, "update", ()):
        key = _item_key(item)
        if key:
            target[key] = _decode_json(item.value_json)

    # getattr keeps this compatible with protobuf versions that omit removals.
    for item in getattr(container, "remove", ()):
        key = _item_key(item)
        if key:
            target.pop(key, None)


def _resolve_record(path: Path) -> Path:
    """
    Resolve an offline-run directory or an explicit run-*.wandb file.

    Exactly one stream is required to avoid joining a stale run with a restart
    that happens to share a parent directory.
    """

    resolved = path.expanduser().resolve()
    if resolved.is_file():
        if resolved.suffix != ".wandb":
            raise ValueError(f"Not a .wandb record file: {resolved}")
        return resolved

    if not resolved.is_dir():
        raise FileNotFoundError(resolved)

    candidates = sorted(resolved.glob("run-*.wandb"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one run-*.wandb in {resolved}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _timestamp_text(timestamp: Any) -> str:
    """
    Convert a protobuf Timestamp to an ISO UTC string.

    An empty timestamp remains empty rather than being confused with epoch zero.
    """

    seconds = int(getattr(timestamp, "seconds", 0))
    nanos = int(getattr(timestamp, "nanos", 0))
    if seconds == 0 and nanos == 0:
        return ""
    value = seconds + nanos / 1_000_000_000.0
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def read_offline_run(path: Path, allow_active: bool = False) -> OfflineRun:
    """
    Decode one W&B DataStore stream entirely from local disk.

    Imports are lazy so --help remains usable without W&B. Completed runs must
    contain an ExitRecord; partial snapshots require explicit --allow-active.
    """

    try:
        from wandb.proto import wandb_internal_pb2 as pb
        from wandb.sdk.internal.datastore import DataStore
    except Exception as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(f"Cannot import local W&B parser: {exc}") from exc

    record_file = _resolve_record(path)
    store = DataStore()
    store.open_for_scan(str(record_file))

    # RunRecord supplies identity/config once. Summary and History then arrive
    # incrementally throughout training.
    identity: Dict[str, Any] = {}
    config: Dict[str, Any] = {}
    summary: Dict[str, Any] = {}
    history: List[Dict[str, Any]] = []
    exit_code: Optional[int] = None

    while True:
        payload = store.scan_data()
        if payload is None:
            break

        record = pb.Record()
        record.ParseFromString(payload)
        kind = record.WhichOneof("record_type")

        if kind == "run":
            run = record.run
            identity = {
                "run_id": str(run.run_id),
                "name": str(run.display_name or run.run_id),
                "entity": str(getattr(run, "entity", "")),
                "project": str(run.project),
                "group": str(run.run_group),
                "tags": list(run.tags),
                "created_at": _timestamp_text(run.start_time),
                "git_commit": str(getattr(run.git, "commit", "")),
            }
            _apply_updates(run.config, config)
        elif kind == "history":
            row: Dict[str, Any] = {}
            for item in record.history.item:
                key = _item_key(item)
                if key:
                    row[key] = _decode_json(item.value_json)
            if row:
                history.append(row)
        elif kind == "summary":
            _apply_updates(record.summary, summary)
        elif kind == "exit":
            exit_code = int(record.exit.exit_code)

    if not identity:
        raise ValueError(f"No RunRecord found in {record_file}")
    if exit_code is None and not allow_active:
        raise ValueError(
            f"Run has no ExitRecord and may still be active: {record_file}; "
            "pass --allow-active only for an intentional partial snapshot"
        )

    return OfflineRun(
        source=record_file,
        config=config,
        summary=summary,
        history=history,
        exit_code=exit_code,
        **identity,
    )


def export_run(
    run: OfflineRun,
    out_dir: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Write one decoded run using the online exporter's directory schema.

    Metadata is attached to history rows after scanning, making unusual record
    ordering safe for downstream analysis.
    """

    state = (
        "finished"
        if run.exit_code == 0
        else "failed"
        if run.exit_code is not None
        else "running-offline"
    )
    target = out_dir / "runs" / safe_name(f"{run.name}__{run.run_id}")
    target.mkdir(parents=True, exist_ok=True)

    metadata = {
        "exported_at": utc_now(),
        "id": run.run_id,
        "name": run.name,
        "path": f"offline/{run.project}/{run.run_id}",
        "url": None,
        "state": state,
        "group": run.group,
        "tags": run.tags,
        "created_at": run.created_at,
        "updated_at": datetime.fromtimestamp(
            run.source.stat().st_mtime,
            tz=timezone.utc,
        ).isoformat(),
        "source_wandb_file": str(run.source),
        "git_commit": run.git_commit,
        "exit_code": run.exit_code,
        "offline": True,
    }
    write_json(target / "metadata.json", metadata)
    write_json(target / "config.json", run.config)
    write_json(target / "summary.json", run.summary)

    rows: List[Dict[str, Any]] = []
    jsonl_path = target / "history.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for original in run.history:
            row = {
                **original,
                "run_id": run.run_id,
                "run_name": run.name,
                "run_state": state,
                "entity": run.entity or "offline",
                "project": run.project,
            }
            clean = {str(key): jsonable(value) for key, value in row.items()}
            rows.append(clean)
            handle.write(json.dumps(clean, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    write_csv(target / "history.csv", rows)

    index = {
        **metadata,
        "history_rows": len(rows),
        "config_algo": run.config.get("algo_name"),
        "config_env": run.config.get("env_name"),
        "config_seed": run.config.get("seed"),
        "summary_empirical_prob": run.summary.get(
            "constraint/empirical_prob"
        ),
        "summary_margin": run.summary.get("constraint/margin"),
        "summary_reward": run.summary.get("disc_reward/aver_reward"),
        "summary_lambda": run.summary.get("lambda/value"),
    }
    return index, rows


def build_parser() -> argparse.ArgumentParser:
    """Build a deliberately local-only command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help=(
            "offline-run directory or explicit run-*.wandb file; "
            "repeat as needed"
        ),
    )
    parser.add_argument(
        "--out",
        required=True,
        help="output directory for index and combined history artifacts",
    )
    parser.add_argument(
        "--allow-active",
        action="store_true",
        help="allow export without ExitRecord; off by default",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Decode selected local runs, deduplicate, and write combined files."""

    args = build_parser().parse_args(argv)
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve before deduplication so a directory and its explicit run file do
    # not add the same stream twice.
    paths: List[Path] = []
    seen = set()
    for value in args.run:
        record = _resolve_record(Path(value))
        if record not in seen:
            paths.append(record)
            seen.add(record)

    indexes: List[Dict[str, Any]] = []
    combined: List[Dict[str, Any]] = []
    for path in paths:
        run = read_offline_run(path, allow_active=args.allow_active)
        print(
            f"[offline-export] id={run.run_id} name={run.name!r} "
            f"rows={len(run.history)} exit={run.exit_code}"
        )
        index, rows = export_run(run, out_dir)
        indexes.append(index)
        combined.extend(rows)

    write_csv(out_dir / "runs_index.csv", indexes)
    write_json(out_dir / "runs_index.json", indexes)
    write_csv(out_dir / "combined_history.csv", combined)
    write_json(
        out_dir / "export_metadata.json",
        {
            "exported_at": utc_now(),
            "offline": True,
            "runs": len(indexes),
            "history_rows": len(combined),
            "sources": [str(path) for path in paths],
        },
    )
    print(
        f"[done] exported {len(indexes)} offline runs and "
        f"{len(combined)} history rows -> {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
