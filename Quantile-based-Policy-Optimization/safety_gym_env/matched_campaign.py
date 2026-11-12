"""Four-GPU SimpleButton campaign. Run one guard and four lanes under supervisord."""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE / "runs/matched_simplebutton"
WORKSPACE = HERE.parents[2]
PYTHON = WORKSPACE / ".conda/envs/zprl/bin/python"
UPSTREAM = Path("/root/Documents/Codex/2026-11-07/files-pasted-by-the-user-official/work/upstream")
HOLD = Path("/root/Documents/Codex/2026-11-03/gpu/outputs/gpu_hold.py")
HOLD_STATE = HOLD.parent.parent / "work/gpu_hold"
HOLD_PY = "/root/miniconda3/bin/python"
DELAY = 40 * 60
GPU_COUNT = 4
LANE_COUNT = 16
CPUS_PER_LANE = 8
CELLS = (("dca", 1.), ("dca", .99), ("qcpo", 1.), ("tqpo", .99), ("qcpo", .99), ("tqpo", 1.))
JOBS = [(m, g, s) for s in range(10) for m, g in CELLS]


def now():
    return time.time()


def utc(ts):
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(ts))


def atomic(path, data):
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text())


def directory(job):
    m, g, s = job
    return ROOT / "jobs" / f"{m}-{'discount' if g < 1 else 'undiscount'}-s{s}"


def run_hold(action):
    cmd = [HOLD_PY, str(HOLD), action, "--gpus", "0,1,2,3", "--fraction", "0.8"]
    return subprocess.check_output(cmd, text=True, timeout=40).strip()


def hold_workers():
    status = json.loads(run_hold("status"))
    if status.get("gpus") != list(range(4)) or status.get("memory_fraction") != 0.8:
        return {}
    children = Path(f'/proc/{status["pid"]}/task/{status["pid"]}/children').read_text().split()
    found = {}
    for pid in children:
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if str(HOLD).encode() not in args or b"worker" not in args:
            continue
        gpu = int(args[args.index(b"--gpu") + 1])
        log = HOLD_STATE / f"gpu-{gpu}.log"
        if f"READY pid={pid} gpu={gpu} " in log.read_text():
            found[gpu] = int(pid)
    return found


def guard():
    released = ROOT / "released.json"
    if not released.exists():
        print("Armed. Release original four-card holder.", flush=True)
        print(run_hold("stop"), flush=True)
        if json.loads(run_hold("status")).get("pid"):
            raise RuntimeError("Holder still running")
        atomic(released, dict(time=now(), delay_seconds=DELAY))
    due_file = ROOT / "hold-due.json"
    while not due_file.exists():
        terminal = [directory(j) / "train.json" for j in JOBS]
        if all(p.exists() for p in terminal):
            finished = max(read(p)["ended"] for p in terminal)
            atomic(due_file, dict(last_training_end=finished, due=finished + DELAY,
                                  due_utc=utc(finished + DELAY)))
            break
        time.sleep(3)
    due = read(due_file)["due"]
    while now() < due:
        time.sleep(min(10, due - now()))
    while True:
        try:
            print(run_hold("start"), flush=True)
            workers = hold_workers()
            if len(workers) == 4:
                atomic(ROOT / "hold-restored.json",
                       dict(verified=now(), due=due, workers=workers, fraction=0.8))
                print("Confirmed four live READY workers", workers, flush=True)
                return
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            print("Restore retry", repr(e), flush=True)
        time.sleep(5)


def launch(command, env, log_path, lock_fd, timeout, training):
    cmd = ["taskset", "-c", env['MATCHED_CPUS'], *map(str, command)]
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, env=env, cwd=HERE, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True,
                                pass_fds=(lock_fd,))
        try:
            while proc.poll() is None:
                time.sleep(5)
                # Both training entry points flush at least once every ten batches.
                if training and now() - log_path.stat().st_mtime > 1800:
                    raise TimeoutError(f"No training log for 30 minutes: {log_path}")
                if now() > timeout:
                    raise TimeoutError(str(log_path))
            if proc.returncode:
                raise RuntimeError(f"exit {proc.returncode}: {log_path}")
        finally:
            # Sampler children can outlive a failed upstream parent.
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                time.sleep(1)
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()


def placement(slot):
    if not 0 <= slot < LANE_COUNT:
        raise ValueError(f"Invalid lane {slot}")
    gpu = slot % GPU_COUNT
    # Keep the original two groups in their initial CPU halves during expansion.
    offset = (0, 16, 8, 24)[slot // GPU_COUNT]
    first = gpu * 32 + offset
    return gpu, f"{first}-{first + CPUS_PER_LANE - 1}"


def honor_campaign_affinity():
    # A manager started before expansion may still hand its child 16 or 32 CPUs.
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, allowed[:CPUS_PER_LANE])


def step(job, slot, fd, phase):
    gpu, cpus = placement(slot)
    m, gamma, seed = job
    d = directory(job)
    tries = sorted(d.glob(phase + "-attempt-*"))
    attempt = d / f"{phase}-attempt-{len(tries) + 1}"
    attempt.mkdir()
    start = now()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu) if phase == "train" else "",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONUNBUFFERED="1", MUJOCO_GL="disable", MPLBACKEND="Agg",
               MATCHED_CPUS=cpus, PYTHON=str(PYTHON), WORKERS="16",
               GAMMA=str(gamma), EVAL_EPISODES="2048", EVAL_SEED="92000000")
    atomic(attempt / "start.json", dict(start=start, gpu=gpu, lane=slot, cpus=cpus, pid=os.getpid()))
    phase_result = d / (phase + ".json")
    try:
        if phase == "eval":
            trained = Path(read(d / "train.json")["data"])
        data = attempt / "data"
        if m == "dca":
            if phase == "train":
                cmd = ["bash", HERE / "run_simplebutton_dca.sh", data, seed, 625, "--train-only"]
            else:
                cmd = [PYTHON, HERE / "dca_simplebutton.py", "--eval-checkpoint", trained / "policy.pt",
                       "--out", data, "--device", "cpu", "--workers", 16, "--eval-episodes", 2048,
                       "--eval-seed", 92000000]
        else:
            env["PYTHONPATH"] = os.pathsep.join(map(str, (UPSTREAM / (m+"-rlpyt"), ROOT / "deps")))
            cmd = [PYTHON, HERE / "matched_baseline.py", m, "--seed", seed, "--gamma", gamma,
                   "--out", data if phase == "train" else trained]
            if phase == "eval":
                cmd.append("--evaluate")
                data = trained
        launch(cmd, env, attempt / "output.log", fd, start + (24 if phase == "train" else 5)*3600,
               phase == "train")
        if phase == "eval":
            result = read(data / "final.json")
            if result["episodes"] != 2048 or result["mean_episode_length"] != 1000 or not all(
                math.isfinite(result[key]) for key in ("reward", "cost", "q80", "violation")):
                raise ValueError("Invalid final evaluation")
        elif m == "dca":
            last = json.loads((data / "train.jsonl").read_text().splitlines()[-1])
            if last["interactions"] != 10_000_000 or not (data / "policy.pt").is_file():
                raise ValueError("Invalid DCA endpoint")
        elif not (data / f"run_{seed}/params.pkl").is_file():
            raise ValueError("Missing upstream snapshot")
        atomic(phase_result, dict(ok=True, start=start, ended=now(), data=str(data)))
    except Exception as e:
        error = dict(ok=False, start=start, ended=now(), error=str(e))
        atomic(attempt / "error.json", error)
        print(job, phase, error, flush=True)
        if sum((old/"error.json").exists() for old in tries) >= 2:
            atomic(phase_result, error)


def lane(slot):
    placement(slot)
    while not (ROOT / "released.json").exists():
        time.sleep(1)
    while True:
        for job in JOBS:
            d = directory(job)
            d.mkdir(parents=True, exist_ok=True)
            train = d / "train.json"
            if (d / "eval.json").exists() or (train.exists() and not read(train)["ok"]):
                continue
            with (d / "claim").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if not train.exists():
                    step(job, slot, lock.fileno(), "train")
                if train.exists() and read(train)["ok"] and not (d / "eval.json").exists():
                    step(job, slot, lock.fileno(), "eval")
                break
        else:
            if all((directory(j) / "eval.json").exists() or
                   ((directory(j) / "train.json").exists() and not read(directory(j) / "train.json")["ok"])
                   for j in JOBS):
                return
            time.sleep(10)


def report():
    rows = []
    elapsed = {}
    active = []
    failed = []
    train_count = 0
    rates = {}
    for j in JOBS:
        m, gamma, seed = j
        d = directory(j)
        tr, ev = d / "train.json", d / "eval.json"
        if tr.exists():
            info = read(tr)
            if info["ok"]:
                train_count += 1
                elapsed.setdefault(m, []).append(info["ended"] - info["start"])
            else:
                failed.append(d.name + " train")
        if ev.exists():
            record = read(ev)
            if not record["ok"]:
                failed.append(d.name + " eval")
                continue
            res = read(Path(record["data"]) / "final.json")
            rows.append(dict(method=m, gamma=gamma, seed=seed, **{k: res[k] for k in
                        ("reward", "cost", "q80", "violation", "violations", "episodes")}))
        elif not tr.exists():
            attempts = sorted(d.glob("train-attempt-*"))
            if not attempts:
                continue
            a = attempts[-1]
            started = read(a / "start.json") if (a / "start.json").exists() else {}
            steps = 0
            if m == "dca" and (a/"data/train.jsonl").exists():
                lines = (a/"data/train.jsonl").read_text().splitlines()
                if lines:
                    steps = json.loads(lines[-1])["interactions"]
            elif (p := a/f"data/run_{seed}/progress.csv").exists():
                with p.open() as f:
                    records = list(csv.DictReader(f))
                if records:
                    steps = int(float(records[-1]["Diagnostics/CumSteps"]))
            if steps:
                # Exclude initial startup when successive timestamped samples exist.
                progress = d / "rate-sample.json"
                past = read(progress) if progress.exists() else None
                if past and steps > past["steps"] and started.get("start") == past["start"]:
                    rate = (steps - past["steps"]) / (now() - past["time"])
                else:
                    rate = steps / (now() - started["start"])
                if not past or steps != past["steps"]:
                    atomic(progress, dict(start=started["start"], steps=steps, time=now(), rate=rate))
                elif past:
                    rate = past["rate"]
                rates.setdefault(m, []).append(rate)
            active.append(dict(job=d.name, gpu=started.get("gpu"), steps=steps))
    if rows:
        csv_tmp = ROOT / f"per-seed.{os.getpid()}.tmp"
        with csv_tmp.open("w") as f:
            w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
        csv_tmp.replace(ROOT/"per-seed.csv")
    summary = []
    for m, g in CELLS:
        group = [r for r in rows if r["method"] == m and r["gamma"] == g]
        if group:
            summary.append(dict(method=m, gamma=g, n=len(group), feasible=sum(r["violation"] <= .20 for r in group),
                                **{k: [statistics.mean(r[k] for r in group),
                                       statistics.stdev(r[k] for r in group) if len(group)>1 else None]
                                   for k in ("reward","cost","q80","violation")}))
    return dict(at=utc(now()), train_complete=train_count, eval_complete=len(rows), failures=failed,
                active=active, steps_per_second={m:round(statistics.median(v)) for m,v in rates.items()},
                summary=summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("role", choices=("guard", "lane", "status", "check-holder"))
    p.add_argument("slot", nargs="?", type=int)
    args = p.parse_args()
    if args.role == "guard":
        guard()
    elif args.role == "lane":
        lane(args.slot)
    elif args.role == "check-holder":
        print(hold_workers())
    else:
        print(json.dumps(report(), indent=2))
