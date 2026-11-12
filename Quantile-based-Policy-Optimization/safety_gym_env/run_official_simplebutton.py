"""Run the upstream QCPO or TQPO SimpleButton pipeline from its own rlpyt installation.

Prepare separate rlpyt trees as instructed in each upstream README (TQPO supplies
its own samplers/parallel/base.py). Use that installation's legacy Python and
MuJoCo runtime. Only experiment settings are supplied through rlpyt's variant
interface; sampling, losses, networks and multiplier updates come from upstream.
"""

import argparse
import json
import os
import queue
import runpy
import sys
import threading
from pathlib import Path


def drain_qcpo_at_shutdown():
    """Unblock upstream worker QueueFeederThreads after the final optimization."""
    from rlpyt.samplers.parallel.base import ParallelSamplerBase

    shutdown = ParallelSamplerBase.shutdown

    def finish(sampler):
        stop = threading.Event()

        def drain():
            while not stop.is_set():
                try:
                    sampler.traj_infos_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

        thread = threading.Thread(target=drain, daemon=True)
        thread.start()
        try:
            return shutdown(sampler)
        finally:
            stop.set()
            thread.join()

    ParallelSamplerBase.shutdown = finish


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("method", choices=("qcpo", "tqpo"))
    p.add_argument("--rlpyt-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--steps", default=10000000, type=int)
    p.add_argument("--cpus", default=4, type=int)
    p.add_argument("--run-id", default=0, type=int)
    args = p.parse_args()
    root = args.rlpyt_root.resolve()
    task = (
        root
        / "rlpyt"
        / "projects"
        / args.method
        / "experiments"
        / ("train_" + args.method + ".py")
    )
    if not task.is_file():
        p.error("prepare the official project in the specified rlpyt tree first")
    if min(args.steps, args.cpus) <= 0 or 12000 % args.cpus:
        p.error(
            "steps must be positive; cpus must divide the official 12000-transition batch"
        )
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    variant = {
        "env": {
            "id": "SimpleButtonEnv-v0" if args.method == "qcpo" else "SimpleEnv-v0"
        },
        "algo": {"cost_limit": 15, "target_outage_prob": 0.2, "pid_Ki": 0.1},
        "runner": {"n_steps": args.steps, "log_interval_steps": min(50000, args.steps)},
    }
    (out / "variant_config.json").write_text(json.dumps(variant, indent=2))
    sys.path.insert(0, str(root))
    os.environ.setdefault("MPLBACKEND", "Agg")
    if args.method == "qcpo":
        drain_qcpo_at_shutdown()
    # The upstream launcher treats cpr as CPUs per run/slot.
    affinity = f"0slt_{args.cpus}cpu_0gpu_{args.cpus}cpr_1cpw_0hto"
    sys.argv = [str(task), affinity, str(out), str(args.run_id), "LSTM"]
    os.chdir(out)
    runpy.run_path(str(task), run_name="__main__")


if __name__ == "__main__":
    main()
