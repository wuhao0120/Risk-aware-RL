#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render one aggregate progress bar for a launcher that owns many training processes.

Each child atomically publishes a small JSON snapshot after a completed rollout batch.
The monitor sums those counters, so queued, running, and finished jobs share one denominator.
Every newly observed child update is printed on its own durable line because returns from
different Safety Gym tasks and seeds must remain identifiable rather than being averaged together.
The single tqdm line contains only aggregate completion, throughput, and ETA.
"""
import argparse
import glob
import json
import os
import re
import time

from tqdm.auto import tqdm


# Durable ``Iter`` lines survive terminal reconnection and are also usable for jobs that were
# launched before JSON status publication was added.
_ITER_RE = re.compile(r"Iter:(\d+)/(\d+)")
_BAR_RE = re.compile(r"(\d+)/(\d+)\s*\[")
_METRIC_RE = re.compile(
    r"mean_reward[:=]([^\s]+)\s+"
    r"quantile_reward[:=]([^\s]+)\s+"
    r"q_est[:=]([^\s]+)\s+"
    r"constraint_prob[:=]([^\s]+)"
    r"(?:\s+lambda[:=]([^\s]+))?"
)


def _finite_number(value):
    """Convert one console token to float; ``n/a`` and malformed values stay absent."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_status_records(status_dir, total_per_run):
    """Return complete atomic child snapshots keyed by their unique status filename."""
    records = {}
    for path in sorted(glob.glob(os.path.join(status_dir, "*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, ValueError, TypeError):
            continue
        child_total = max(1, int(payload.get("total", total_per_run)))
        completed = max(0, int(payload.get("completed", 0)))
        payload["completed"] = min(completed, child_total, total_per_run)
        records[os.path.basename(path)] = payload
    return records


def read_status_files(status_dir, total_per_run):
    """Return aggregate progress plus the most recently updated child snapshot."""
    records = read_status_records(status_dir, total_per_run)
    statuses = list(records.values())
    aggregate_completed = sum(item["completed"] for item in statuses)
    newest = max(statuses, key=lambda item: float(item.get("updated_at", 0.0)), default=None)
    return aggregate_completed, newest


def _tail_text(path, limit_bytes=2 * 1024 * 1024):
    """Read only a log tail and normalize tqdm carriage returns into parseable lines."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as stream:
            stream.seek(max(0, size - limit_bytes))
            data = stream.read()
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace").replace("\r", "\n")


def read_existing_logs(paths, total_per_run):
    """
    Recover aggregate counters from logs produced before the JSON status interface existed.

    The maximum matching counter in each file is used. This is a compatibility monitor only;
    future launcher runs always use atomic status files and therefore avoid parsing text.
    """
    completed_sum = 0
    newest = None
    newest_time = -1.0
    for path in paths:
        text = _tail_text(path)
        counters = [
            int(done)
            for done, total in _ITER_RE.findall(text) + _BAR_RE.findall(text)
            if int(total) == total_per_run
        ]
        completed_sum += min(max(counters, default=0), total_per_run)

        metric_matches = list(_METRIC_RE.finditer(text))
        if metric_matches:
            match = metric_matches[-1]
            modified = os.path.getmtime(path)
            if modified >= newest_time:
                newest_time = modified
                newest = {
                    "env": os.path.basename(path),
                    "phase": "training",
                    "metrics": {
                        "mean_reward": _finite_number(match.group(1)),
                        "quantile_reward": _finite_number(match.group(2)),
                        "q_est": _finite_number(match.group(3)),
                        "constraint_prob": _finite_number(match.group(4)),
                        "lambda": _finite_number(match.group(5)),
                    },
                }
    return completed_sum, newest


def _format_value(value):
    """Keep the aggregate bar compact while making unavailable metrics explicit."""
    return "n/a" if value is None else f"{float(value):.3f}"


def _postfix(newest, discovered, expected):
    """Keep only compact job/phase state on the physical tqdm progress line."""
    if newest is None:
        return f"jobs={discovered}/{expected} waiting_for_workers"
    algo = str(newest.get("algo", "unknown"))
    env = str(newest.get("env", "unknown"))
    phase = str(newest.get("phase", "training"))
    return f"jobs {discovered}/{expected} | {algo}/{env}/{phase}"


def _training_line(payload, delta=1):
    """Render one completed child update on a line physically separate from tqdm."""
    metrics = payload.get("metrics") or {}
    algo = str(payload.get("algo", "unknown"))
    env = str(payload.get("env", "unknown"))
    seed = payload.get("seed", "?")
    completed = int(payload.get("completed", 0))
    total = int(payload.get("total", 0))
    jump = "" if int(delta) == 1 else f" (+{int(delta)}, latest)"
    text = (
        f"[train] {algo}/{env}/s{seed} batch={completed:03d}/{total:03d}{jump} "
        f"mean={_format_value(metrics.get('mean_reward'))} "
        f"q_batch={_format_value(metrics.get('quantile_reward'))} "
        f"q_est={_format_value(metrics.get('q_est'))} "
        f"p={_format_value(metrics.get('constraint_prob'))}"
    )
    if metrics.get("lambda") is not None:
        text += f" lambda={_format_value(metrics.get('lambda'))}"
    return text


def _validation_line(payload):
    """Render scheduled exact-N validation immediately after its matching training line."""
    validation = payload.get("validation")
    if not validation:
        return None
    algo = str(payload.get("algo", "unknown"))
    env = str(payload.get("env", "unknown"))
    seed = payload.get("seed", "?")
    completed = int(payload.get("completed", 0))
    total = int(payload.get("total", 0))
    episodes = validation.get("num_episodes")
    episodes_text = "n/a" if episodes is None else str(int(episodes))
    return (
        f"[validation] {algo}/{env}/s{seed} batch={completed:03d}/{total:03d} "
        f"N={episodes_text} mean={_format_value(validation.get('mean_reward'))} "
        f"q_batch={_format_value(validation.get('quantile_reward'))} "
        f"q_est={_format_value(validation.get('q_est'))} "
        f"p={_format_value(validation.get('constraint_prob'))}"
    )


def parse_args():
    """Parse either atomic-status mode or compatibility log mode."""
    parser = argparse.ArgumentParser(description="One aggregate tqdm bar for parallel training")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--status-dir", help="directory containing one atomic JSON per child")
    source.add_argument("--logs", nargs="+", help="existing child logs to monitor")
    parser.add_argument("--expected-runs", type=int, required=True)
    parser.add_argument("--total-per-run", type=int, required=True)
    parser.add_argument("--label", default="training total")
    parser.add_argument("--stop-file", default=None)
    parser.add_argument("--poll", type=float, default=1.0)
    return parser.parse_args()


def main():
    """Poll child state until all batches finish or the launcher signals early termination."""
    args = parse_args()
    expected = max(1, int(args.expected_runs))
    total_per_run = max(1, int(args.total_per_run))
    grand_total = expected * total_per_run
    poll = max(0.1, float(args.poll))

    print(
        f"\n{args.label}\n"
        f"  workload : {expected} runs x {total_per_run} batches = {grand_total} total\n"
        "  live     : mean=mean_reward, q_batch=batch quantile, "
        "q_est=QPO tracker or non-QPO batch quantile, p=constraint probability, "
        "lambda=latest post-batch multiplier when applicable\n",
        flush=True,
    )
    progress = tqdm(
        total=grand_total,
        desc=args.label,
        unit="batch",
        dynamic_ncols=True,
        mininterval=poll,
        leave=True,
    )
    previous = 0
    known_status = {}
    try:
        while True:
            if args.status_dir:
                records = read_status_records(args.status_dir, total_per_run)
                statuses = list(records.values())
                completed = sum(item["completed"] for item in statuses)
                newest = max(
                    statuses,
                    key=lambda item: float(item.get("updated_at", 0.0)),
                    default=None)
                discovered = len(records)

                # A resumed child first appears with non-zero completed work. Add that baseline
                # directly to n/last_print_n so tqdm does not mistake old work for instantaneous
                # throughput. Only increments observed after first sight affect the ETA rate.
                baseline = work_delta = 0
                completed_updates = []
                first_records = not known_status and bool(records)
                for key, payload in records.items():
                    current = int(payload["completed"])
                    if key not in known_status:
                        baseline += current
                        # Fresh runs normally appear first at zero. If polling first sees non-zero
                        # work, print exactly the latest snapshot and disclose the skipped delta.
                        if current > 0:
                            completed_updates.append((payload, current))
                    elif current > known_status[key]:
                        delta = current - known_status[key]
                        work_delta += delta
                        completed_updates.append((payload, delta))
                    known_status[key] = current
                if first_records:
                    now = progress._time()
                    progress.start_t = now
                    progress.last_print_t = now
                if baseline:
                    progress.n += baseline
                    progress.last_print_n += baseline
                    progress.refresh()
                if work_delta:
                    progress.update(work_delta)
                # tqdm.write temporarily clears and redraws the bar, guaranteeing metrics never
                # share or wrap its physical line. Sort concurrent seed updates deterministically.
                for payload, delta in sorted(
                        completed_updates,
                        key=lambda item: (float(item[0].get("updated_at", 0.0)),
                                          str(item[0].get("algo", "")),
                                          int(item[0].get("seed", -1)))):
                    progress.write(_training_line(payload, delta))
                    validation_line = _validation_line(payload)
                    if validation_line is not None:
                        progress.write("")
                        progress.write(validation_line)
                        progress.write("")
            else:
                completed, newest = read_existing_logs(args.logs, total_per_run)
                discovered = len(args.logs)
                completed = min(max(0, completed), grand_total)
                if completed > previous:
                    progress.update(completed - previous)
                    previous = completed

            completed = min(max(0, completed), grand_total)
            progress.set_postfix_str(_postfix(newest, discovered, expected), refresh=True)

            # In launcher mode the bar stays alive at 100% while final evaluation and artifact
            # writes finish; the parent creates stop_file only after every child process exits.
            # Compatibility log mode has no parent signal and therefore exits at 100% itself.
            if completed >= grand_total and not (args.status_dir and args.stop_file):
                break
            if args.stop_file and os.path.exists(args.stop_file):
                if completed < grand_total:
                    progress.write(
                        f"{args.label}: workers stopped at {completed}/{grand_total} "
                        "completed batches")
                break
            time.sleep(poll)
    except KeyboardInterrupt:
        progress.write(f"{args.label}: monitor detached; training processes were not stopped")
    finally:
        progress.close()


if __name__ == "__main__":
    main()
