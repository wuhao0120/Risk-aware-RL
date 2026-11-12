"""Refresh campaign ETA and fill the rebuttal with complete ten-seed results."""
import csv
import json
import fcntl
import subprocess
import statistics
import time
from pathlib import Path

import matched_campaign as c

DRAFT = c.HERE.parent / 'paper/rebuttal/simplebutton_response.md'


def eta():
    sizes = dict(dca=10000000, qcpo=10008000, tqpo=10008000)
    measured = {m: [] for m in sizes}
    recent = {m: [] for m in sizes}
    changed = c.ROOT/'parallel-start.json'
    since = c.read(changed)['start'] if changed.exists() else float('inf')
    eval_seconds = {m: [] for m in sizes}
    done = {}
    for j in c.JOBS:
        m, _, s = j
        d = c.directory(j)
        if (d/'train.json').exists():
            tr = c.read(d/'train.json')
            if tr['ok']:
                measured[m].append(sizes[m]/(tr['ended']-tr['start']))
                done[j] = sizes[m]
                if tr['start'] >= since:
                    recent[m].append(sizes[m]/(tr['ended']-tr['start']))
            if (d/'eval.json').exists():
                ev = c.read(d/'eval.json')
                if ev['ok']:
                    eval_seconds[m].append(ev['ended']-ev['start'])
            continue
        attempts = sorted(d.glob('train-attempt-*'))
        if not attempts:
            continue
        a = attempts[-1]
        if m == 'dca':
            log = a/'data/train.jsonl'
            if log.exists():
                lines = log.read_text().splitlines()
                if len(lines) >= 10:
                    steps = json.loads(lines[-1])['interactions']
                    done[j] = steps
                    started = c.read(a/'start.json')['start']
                    speed = steps/(log.stat().st_mtime-started)
                    measured[m].append(speed)
                    if started >= since and steps >= 160000:
                        recent[m].append(speed)
        else:
            log = a/f'data/run_{s}/progress.csv'
            if log.exists():
                with log.open() as f:
                    rows = list(csv.DictReader(f))
                if rows:
                    steps = int(rows[-1]['Diagnostics/CumSteps'])
                    done[j] = steps
                    measured[m].append(steps/float(rows[-1]['Diagnostics/CumTime (s)']))
                    if len(rows) >= 2:
                        interval = float(rows[-1]['Diagnostics/CumTime (s)'])-float(rows[-2]['Diagnostics/CumTime (s)'])
                        if interval > 0 and log.stat().st_mtime-interval >= since:
                            recent[m].append(float(rows[-1]['Diagnostics/StepsPerSecond']))
    if not all(measured.values()):
        return {}
    if changed.exists() and not all(recent.values()):
        # Re-measure each method under the new CPU sharing before projecting an ETA.
        missing = [m.upper() for m in sizes if not recent[m]]
        return dict(note="Collecting post-expansion throughput for " + ", ".join(missing) +
                    ". Restore uses actual last training end + 2400 s.")
    rates = {m:statistics.median(recent[m] or v) for m,v in measured.items()}
    train_seconds = sum((sizes[m]-done.get((m,g,s),0))/rates[m] for m,g,s in c.JOBS)
    eval_low = eval_high = 0.
    for j in c.JOBS:
        m = j[0]
        d = c.directory(j)
        if (d / 'eval.json').exists():
            continue
        if eval_seconds[m]:
            sec = statistics.median(eval_seconds[m])
            lo, hi = sec*.9, sec*1.15
        else:
            lo, hi = sizes[m]/rates[m]*.10, sizes[m]/rates[m]*.35
        starts = sorted(d.glob('eval-attempt-*/start.json'))
        spent = c.now()-c.read(starts[-1])['start'] if starts else 0.
        eval_low += max(0, lo-spent)
        eval_high += max(0, hi-spent)
    running = sum(1 for j in c.JOBS if not (c.directory(j)/'eval.json').exists())
    lanes = max(1, min(c.LANE_COUNT, running))
    low = c.now()+(train_seconds+eval_low)/lanes
    high = c.now()+(train_seconds+eval_high)/lanes
    if running > lanes:
        high += max(sizes[m]/r for m,r in rates.items())*.6
    return dict(throughput_steps_s={k:round(v) for k,v in rates.items()},
                full_queue_eta_utc=[c.utc(low),c.utc(high)],
                note="Projected training and CPU evaluation finish at four concurrent jobs per GPU. Restore uses actual last training end + 2400 s.")


def endpoint_text(status):
    n = status['eval_complete']
    groups = [row for row in status['summary'] if row['n'] == 10]
    if not groups:
        return (f"{n} of 60 final checkpoints have completed independent evaluation. "
                "The matched comparison awaits ten training seeds in each condition.")
    lines = [f"{n} of 60 final checkpoints have completed evaluation. Each row below contains all ten seeds for its condition. Values are means ± sample standard deviations across training seeds.",
             "", "| Method | Training γ | Return | Cost | Cost Q80 | Violation (%) | Feasible seeds |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in groups:
        cells = []
        for k in ('reward','cost','q80','violation'):
            mean, sd = row[k]
            scale = 100 if k=='violation' else 1
            cells.append(f"{mean*scale:.2f} ± {sd*scale:.2f}")
        lines.append(f"| {row['method'].upper()} | {row['gamma']:.2f} | " +
                     " | ".join(cells) + f" | {row['feasible']}/10 |")
    with (c.ROOT/'per-seed.csv').open() as f:
        rows = list(csv.DictReader(f))
    for method in ('tqpo','qcpo','dca'):
        arms = [{int(r['seed']):r for r in rows if r['method']==method and float(r['gamma'])==g}
                for g in (1., .99)]
        if not all(len(a) == 10 for a in arms):
            continue
        deltas = []
        for metric in ('reward','violation'):
            scale = 100 if metric=='violation' else 1
            differences = [scale*(float(arms[0][s][metric])-float(arms[1][s][metric])) for s in range(10)]
            mean = statistics.mean(differences)
            half = 2.262157*statistics.stdev(differences)/(10**.5)
            deltas.append(f"{mean:+.2f} (95% CI {mean-half:+.2f} to {mean+half:+.2f})")
        feasible = [sum(float(r['violation'])<=.2 for r in arm.values()) for arm in arms]
        lines += ["", f"For {method.upper()}, changing the training target from γ = 0.99 to γ = 1.00 changed mean return by {deltas[0]} and violation probability by {deltas[1]} percentage points. The empirical feasible-seed count changed from {feasible[1]}/10 to {feasible[0]}/10."]
    by_cell = {(row['method'],row['gamma']):row for row in groups}
    for gamma in (1., .99):
        if not all((method,gamma) in by_cell for method in ('dca','qcpo','tqpo')):
            continue
        dca = by_cell['dca',gamma]
        baselines = [by_cell[m,gamma] for m in ('qcpo','tqpo')]
        gaps = [dca['reward'][0] - row['reward'][0] for row in baselines]
        lines += ["", f"At the matched training discount γ = {gamma:.2f}, DCA's mean return differed from QCPO's by {gaps[0]:+.2f} and from TQPO's by {gaps[1]:+.2f}. DCA, QCPO and TQPO met the empirical violation limit in {dca['feasible']}, {baselines[0]['feasible']} and {baselines[1]['feasible']} of ten seeds, respectively."]
        caps, batches = [], 0
        for seed in range(10):
            result = c.read(c.directory(('dca',gamma,seed))/'train.json')
            records = (Path(result['data'])/'train.jsonl').read_text().splitlines()
            caps.append(sum(json.loads(line)['multiplier'] >= 50 for line in records))
            batches += len(records)
        lines += ["", f"DCA's PI multiplier reached its cap of 50 in {sum(n>0 for n in caps)}/10 seeds at γ = {gamma:.2f}, spending {sum(caps)}/{batches} rollout updates at the cap. This measures saturation during learning. The monotonicity assumption concerns the equilibrium risk under separate fixed multipliers."]
    lines += ["", "Intervals use paired differences across the ten training seeds and a Student t critical value with nine degrees of freedom."]
    return "\n".join(lines)


def backup_restore():
    """If the primary guard misses its deadline, independently retry the same holder."""
    restored = c.ROOT/'hold-restored.json'
    if restored.exists() or not (c.ROOT/'released.json').exists():
        return
    records = [c.directory(j)/'train.json' for j in c.JOBS]
    if not all(p.exists() for p in records):
        return
    ended = max(c.read(p)['ended'] for p in records)
    due = ended + c.DELAY
    if c.now() < due + 15:
        return
    # The primary normally completes within this grace interval. The original
    # holder's own flock makes concurrent starts idempotent.
    print(c.utc(c.now()), "backup restore check", c.run_hold("start"), flush=True)
    workers = c.hold_workers()
    if len(workers) == 4:
        c.atomic(restored, dict(verified=c.now(), due=due, workers=workers, fraction=.8,
                                verified_by="independent reporter"))


def refresh():
    backup_restore()
    status = c.report()
    status.update(eta())
    if (c.ROOT/'hold-due.json').exists():
        status['hold_timer'] = c.read(c.ROOT/'hold-due.json')
    if (c.ROOT/'hold-restored.json').exists():
        status['hold_restored'] = c.read(c.ROOT/'hold-restored.json')
    c.atomic(c.ROOT/'status.json',status)
    final_table = c.HERE.parent/'paper/rebuttal/simplebutton_discount/60_of_60/summary.csv'
    if status['eval_complete'] == 60 and not final_table.exists():
        subprocess.run([c.PYTHON, c.HERE/'matched_plot.py'], check=True, timeout=120)
    original = DRAFT.read_text()
    before, tail = original.split('<!-- MATCHED-RESULTS -->')
    _, after = tail.split('<!-- /MATCHED-RESULTS -->')
    if status['eval_complete'] == 60:
        before = before.replace('We are repeating the comparison', 'We repeated the comparison')
    new = before+'<!-- MATCHED-RESULTS -->\n'+endpoint_text(status)+'\n<!-- /MATCHED-RESULTS -->'+after
    if new != original:
        tmp = DRAFT.with_suffix('.md.tmp')
        tmp.write_text(new)
        tmp.replace(DRAFT)
    return status


if __name__ == '__main__':
    lease = (c.ROOT/'report.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX)
    while True:
        try:
            status = refresh()
            if 'hold_restored' in status and status['eval_complete']+len(status['failures'])==60:
                break
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as e:
            print(c.utc(c.now()), repr(e), flush=True)
        time.sleep(60)
