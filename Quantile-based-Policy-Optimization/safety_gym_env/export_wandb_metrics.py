#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export full W&B metric histories for safety_gym_env experiments.

Typical use while training is still running:

    python export_wandb_metrics.py

The script first parses _runs/logs/*.log to discover the exact W&B run ids that
were printed by wandb.init(). It then uses wandb.Api().run(...).scan_history()
to export every logged metric row, not just the sparse console prints.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


RUN_URL_RE = re.compile(
    r"https://wandb\.ai/(?P<entity>[^/\s]+)/(?P<project>[^/\s]+)/runs/(?P<run_id>[A-Za-z0-9_-]+)"
)
SYNCING_RE = re.compile(r"wandb:\s+.*?Syncing run (?P<name>.+?)\s*$")
LOCAL_DIR_RE = re.compile(r"wandb:\s+Run data is saved locally in (?P<path>.+?)\s*$")


@dataclass(frozen=True)
class RunRef:
    """A run reference discovered from a console log or provided by the user."""

    entity: str
    project: str
    run_id: str
    name_hint: str = ""
    log_file: str = ""
    local_dir: str = ""

    @property
    def path(self) -> str:
        return f"{self.entity}/{self.project}/{self.run_id}"


def utc_now() -> str:
    """Return a stable ISO timestamp for export metadata."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_name(value: str) -> str:
    """Make a string safe for a local directory name."""

    cleaned = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value.strip())
    return cleaned.strip("_") or "unnamed"


def jsonable(value: Any) -> Any:
    """Convert W&B objects and non-finite floats into JSON-safe values."""

    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, data: Any) -> None:
    """Write readable JSON with UTF-8 encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(data), f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def priority_columns(columns: Iterable[str]) -> List[str]:
    """Keep analysis-critical columns first, then append the rest alphabetically."""

    priority = [
        "run_id",
        "run_name",
        "run_state",
        "entity",
        "project",
        "_step",
        "_timestamp",
        "_runtime",
        "progress/env_steps",
        "progress/iteration",
        "progress/trajectories",
        "disc_reward/aver_reward",
        "disc_reward/discounted_reward",
        "disc_reward/quantile_reward",
        "disc_reward/return_std",
        "quantile/q_est",
        "quantile/margin_to_threshold",
        "constraint/empirical_prob",
        "constraint/margin",
        "constraint/cdf_estimate_initial",
        "lambda/value",
        "debug/cost_mean",
        "debug/undisc_cost_mean",
        "debug/cost_limit",
    ]
    seen = set(columns)
    out = [c for c in priority if c in seen]
    out.extend(sorted(c for c in seen if c not in set(out)))
    return out


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Write a list of dict rows as CSV, preserving sparse metric columns."""

    path.parent.mkdir(parents=True, exist_ok=True)
    all_columns = set()
    for row in rows:
        all_columns.update(row.keys())
    fieldnames = priority_columns(all_columns)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: jsonable(row.get(k)) for k in fieldnames})


def parse_log_file(path: Path) -> List[RunRef]:
    """Extract W&B run ids, names, and local dirs from one console log."""

    text = path.read_text(encoding="utf-8", errors="ignore")
    names = [m.group("name").strip() for m in SYNCING_RE.finditer(text)]
    local_dirs = [m.group("path").strip() for m in LOCAL_DIR_RE.finditer(text)]
    refs: List[RunRef] = []
    for i, match in enumerate(RUN_URL_RE.finditer(text)):
        refs.append(
            RunRef(
                entity=match.group("entity"),
                project=match.group("project"),
                run_id=match.group("run_id"),
                name_hint=names[min(i, len(names) - 1)] if names else "",
                log_file=str(path),
                local_dir=local_dirs[min(i, len(local_dirs) - 1)] if local_dirs else "",
            )
        )
    return refs


def discover_runs_from_logs(log_dir: Path) -> List[RunRef]:
    """Parse all log files and deduplicate by entity/project/run_id."""

    refs: List[RunRef] = []
    if log_dir.exists():
        for path in sorted(log_dir.glob("*.log")):
            refs.extend(parse_log_file(path))

    deduped: Dict[Tuple[str, str, str], RunRef] = {}
    for ref in refs:
        deduped[(ref.entity, ref.project, ref.run_id)] = ref
    return list(deduped.values())


def parse_explicit_run(value: str, entity: Optional[str], project: str) -> RunRef:
    """Parse --run as id, entity/project/id, or W&B run URL."""

    match = RUN_URL_RE.search(value)
    if match:
        return RunRef(match.group("entity"), match.group("project"), match.group("run_id"))

    parts = [p for p in value.strip().split("/") if p]
    if len(parts) == 3:
        return RunRef(parts[0], parts[1], parts[2])
    if len(parts) == 1 and entity:
        return RunRef(entity, project, parts[0])

    raise ValueError(
        f"Cannot parse run '{value}'. Use a run id with --entity, a W&B URL, "
        "or entity/project/run_id."
    )


def load_wandb_api() -> Any:
    """Import wandb lazily so --help works even if wandb is unavailable."""

    try:
        import wandb  # type: ignore
    except Exception as exc:  # pragma: no cover - import error path
        raise SystemExit(f"Failed to import wandb: {exc}") from exc
    return wandb.Api()


def run_matches(run: Any, name_globs: Sequence[str], groups: Sequence[str], tags: Sequence[str]) -> bool:
    """Client-side filters for project run listing."""

    if name_globs and not any(fnmatch.fnmatch(run.name or "", pat) for pat in name_globs):
        return False
    if groups and (run.group or "") not in set(groups):
        return False
    if tags and not set(tags).issubset(set(run.tags or [])):
        return False
    return True


def list_project_runs(
    api: Any,
    entity: str,
    project: str,
    name_globs: Sequence[str],
    groups: Sequence[str],
    tags: Sequence[str],
    max_runs: Optional[int],
) -> List[Any]:
    """List and filter project runs when --all-project-runs or filters are used."""

    selected: List[Any] = []
    for run in api.runs(f"{entity}/{project}"):
        if run_matches(run, name_globs, groups, tags):
            selected.append(run)
            if max_runs and len(selected) >= max_runs:
                break
    return selected


def export_one_run(api: Any, run: Any, out_dir: Path, page_size: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Export one run's config, summary, metadata, and full scan_history()."""

    run_dir = out_dir / "runs" / safe_name(f"{run.name or 'run'}__{run.id}")
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "exported_at": utc_now(),
        "id": run.id,
        "name": run.name,
        "path": "/".join(run.path),
        "url": run.url,
        "state": run.state,
        "group": run.group,
        "job_type": run.job_type,
        "tags": list(run.tags or []),
        "created_at": str(getattr(run, "created_at", "")),
        "updated_at": str(getattr(run, "updated_at", "")),
    }
    config = dict(run.config or {})
    summary = dict(run.summary or {})

    write_json(run_dir / "metadata.json", meta)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "summary.json", summary)

    history_rows: List[Dict[str, Any]] = []
    jsonl_path = run_dir / "history.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in run.scan_history(page_size=page_size):
            clean = {str(k): jsonable(v) for k, v in dict(row).items()}
            clean.update(
                {
                    "run_id": run.id,
                    "run_name": run.name,
                    "run_state": run.state,
                    "entity": run.entity,
                    "project": run.project,
                }
            )
            history_rows.append(clean)
            f.write(json.dumps(clean, ensure_ascii=False, sort_keys=True))
            f.write("\n")

    write_csv(run_dir / "history.csv", history_rows)

    index_row = {
        **meta,
        "history_rows": len(history_rows),
        "config_algo": config.get("algo_name"),
        "config_env": config.get("env_name"),
        "config_seed": config.get("seed"),
        "summary_empirical_prob": summary.get("constraint/empirical_prob"),
        "summary_margin": summary.get("constraint/margin"),
        "summary_reward": summary.get("disc_reward/aver_reward"),
        "summary_lambda": summary.get("lambda/value"),
    }
    return index_row, history_rows


def resolve_runs(args: argparse.Namespace, api: Any, log_refs: List[RunRef]) -> List[Any]:
    """Resolve log-discovered, explicit, or project-listed run selections."""

    runs_by_path: Dict[str, Any] = {}

    explicit_refs = [parse_explicit_run(v, args.entity, args.project) for v in args.run]
    for ref in explicit_refs or log_refs:
        if args.entity and ref.entity != args.entity:
            continue
        if args.project and ref.project != args.project:
            continue
        runs_by_path[ref.path] = api.run(ref.path)

    if args.all_project_runs or args.name_glob or args.group or args.tag:
        entity = args.entity or (log_refs[0].entity if log_refs else None)
        if not entity:
            raise SystemExit("Provide --entity, or keep W&B run URLs in _runs/logs/*.log.")
        for run in list_project_runs(
            api=api,
            entity=entity,
            project=args.project,
            name_globs=args.name_glob,
            groups=args.group,
            tags=args.tag,
            max_runs=args.max_runs,
        ):
            runs_by_path["/".join(run.path)] = run

    return list(runs_by_path.values())


def export_once(args: argparse.Namespace) -> Path:
    """Run one export pass and return the output directory."""

    root = Path(args.root).resolve()
    log_dir = (root / args.logs).resolve()
    out_dir = (root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log_refs = discover_runs_from_logs(log_dir)
    api = load_wandb_api()
    runs = resolve_runs(args, api, log_refs)
    if not runs:
        raise SystemExit(
            "No W&B runs selected. Keep the console logs with W&B URLs, or pass "
            "--entity ENTITY --project PROJECT --all-project-runs / --run RUN_ID."
        )

    index_rows: List[Dict[str, Any]] = []
    combined_rows: List[Dict[str, Any]] = []
    for run in runs:
        print(f"[export] {run.entity}/{run.project}/{run.id}  name={run.name!r} state={run.state}")
        index_row, history_rows = export_one_run(api, run, out_dir, args.page_size)
        index_rows.append(index_row)
        combined_rows.extend(history_rows)

    write_csv(out_dir / "runs_index.csv", index_rows)
    write_json(out_dir / "runs_index.json", index_rows)
    write_csv(out_dir / "combined_history.csv", combined_rows)
    write_json(
        out_dir / "export_metadata.json",
        {
            "exported_at": utc_now(),
            "root": str(root),
            "logs": str(log_dir),
            "out": str(out_dir),
            "runs": len(index_rows),
            "history_rows": len(combined_rows),
        },
    )
    print(f"[done] exported {len(index_rows)} runs, {len(combined_rows)} history rows -> {out_dir}")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="safety_gym_env root directory")
    parser.add_argument("--logs", default="_runs/logs", help="relative directory containing console .log files")
    parser.add_argument("--out", default="_runs/wandb_export/latest", help="relative output directory")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"), help="W&B entity/team")
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "safety_gym_qcrl"), help="W&B project")
    parser.add_argument("--run", action="append", default=[], help="run id, W&B URL, or entity/project/run_id")
    parser.add_argument("--name-glob", action="append", default=[], help="select project runs by shell-style name pattern")
    parser.add_argument("--group", action="append", default=[], help="select project runs by W&B group")
    parser.add_argument("--tag", action="append", default=[], help="select project runs containing this tag")
    parser.add_argument("--all-project-runs", action="store_true", help="export every run in entity/project")
    parser.add_argument("--max-runs", type=int, default=None, help="cap project-listed runs after filtering")
    parser.add_argument("--page-size", type=int, default=1000, help="W&B scan_history page size")
    parser.add_argument("--watch-sec", type=int, default=0, help="repeat export every N seconds until interrupted")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    while True:
        export_once(args)
        if args.watch_sec <= 0:
            return 0
        print(f"[watch] sleeping {args.watch_sec}s; press Ctrl-C to stop")
        time.sleep(args.watch_sec)


if __name__ == "__main__":
    raise SystemExit(main())
