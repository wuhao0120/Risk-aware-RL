"""Audited held-out results and four-seed curves for both consistency campaigns."""
import csv
import hashlib
import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
RAW = BASE / 'baselines_rawrisk_r1_c1_p10'
DISC = BASE / 'all_discount_r099_c099_dual099_p10_div25'
OLD = BASE / 'gpu_mixed_r1_c099_p10'
DCA = BASE / 'dca_rawrisk_r1_c1_p10_div25'
NEW = [('undiscounted', RAW, m, 1.) for m in ('qcpo', 'tqpo')]
NEW += [('discounted', DISC, m, .99) for m in ('dca', 'qcpo', 'tqpo')]
SERIES = [('undiscounted', DCA, 'dca', 1.)] + NEW
SERIES += [('mixed', OLD, m, 1.) for m in ('dca', 'qcpo', 'tqpo')]
CACHE, HASH_CACHE = {}, {}
GRID = 50_000


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def emit_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_rows(path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break
    if len(rows) >= len(CACHE.get(path, [])):
        CACHE[path] = rows
    return CACHE.get(path, [])


def digest(path):
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in HASH_CACHE:
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
        HASH_CACHE[key] = h.hexdigest()
    return HASH_CACHE[key]


def steps_for(method, stage):
    return stage if method == 'dca' else ((stage + 11999) // 12000) * 12000


def performance_curve(run):
    windows = [deque(maxlen=100) for _ in range(4)]
    smooth, points = deque(maxlen=10), []
    rows = read_rows(run / 'episodes.jsonl')
    for row in rows:
        values = (row.get('undiscounted_return', row.get('undiscounted_reward')),
                  row.get('discounted_return', row.get('discounted_reward')),
                  row['undiscounted_cost'], row['discounted_cost'])
        for window, value in zip(windows, values):
            window.extend(value)
        if row['actual_env_steps'] >= (len(points) + 1) * GRID:
            r, dr, c, dc = map(np.asarray, windows)
            smooth.append((r.mean(), dr.mean(), 100 * (c > 15).mean(),
                           100 * (dc > 15).mean(), np.quantile(c, .9), np.quantile(dc, .9)))
            points.append(np.mean(smooth, axis=0))
    return points, rows[-1]['actual_env_steps'] if rows else 0


def curves():
    progress, all_rows = {}, []
    metrics = ('raw_return', 'discounted_return', 'raw_violation_pct',
               'discounted_violation_pct', 'raw_cost_q90', 'discounted_cost_q90')
    titles = ('Undiscounted return', 'Discounted return', 'Raw-cost violation (%)',
              'Discounted-cost violation (%)', 'Raw-cost 0.9 quantile', 'Discounted-cost 0.9 quantile')
    colors = dict(dca='#246698', qcpo='#ae8725', tqpo='#39805b')
    for setting in ('undiscounted', 'discounted', 'mixed'):
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
        axes = axes.ravel()
        for group, root, method, gamma in SERIES:
            if group != setting:
                continue
            series = []
            for seed in range(4):
                points, steps = performance_curve(root / f'{method}-s{seed}')
                series.append(points)
                progress[f'{group}/{method}/s{seed}'] = steps
            n = min(map(len, series))
            if not n:
                continue
            data = np.asarray([s[:n] for s in series])
            mean, sd = data.mean(0), data.std(0, ddof=1)
            x = GRID * np.arange(1, n + 1)
            for k, ax in enumerate(axes):
                ax.plot(x / 1e6, mean[:, k], color=colors[method], label=method.upper())
                ax.fill_between(x / 1e6, mean[:, k] - sd[:, k], mean[:, k] + sd[:, k], color=colors[method], alpha=.16)
                for i, step in enumerate(x):
                    all_rows.append(dict(setting=setting, method=method, metric=metrics[k], nominal_steps=int(step),
                        **{f'seed{s}': float(data[s, i, k]) for s in range(4)}, mean=float(mean[i,k]), sample_sd=float(sd[i,k])))
        for i, ax in enumerate(axes):
            ax.set_title(titles[i]); ax.grid(axis='y', alpha=.2)
            ax.spines[['top', 'right']].set_visible(False)
            if i in (2, 3): ax.axhline(10, color='#555', ls=':', lw=1)
            if i in (4, 5):
                ax.axhline(15, color='#555', ls=':', lw=1)
                ax.set_yscale('symlog', linthresh=20)
        if axes[0].lines: axes[0].legend(frameon=False)
        fig.suptitle(f'SimpleButton | {setting} training | Mean +/- 1 sample SD, 4 seeds')
        fig.supxlabel('Training interactions (millions)')
        out = ROOT / 'curves'; out.mkdir(exist_ok=True)
        for fmt in ('png', 'pdf'):
            path = out / f'{setting}.{fmt}'; tmp = path.with_suffix('.tmp')
            fig.savefig(tmp, format=fmt, dpi=180); tmp.replace(path)
        plt.close(fig)
    emit_csv(ROOT / 'curves' / 'training_curves.csv', all_rows)
    return progress


def validation():
    rows, groups = [], []
    for setting, root, method, gamma in SERIES:
        for stage in (5_000_000, 10_000_000):
            steps, group = steps_for(method, stage), []
            for seed in range(4):
                run = root / f'{method}-s{seed}'; loc = run / f'validation_{steps}'
                if not (loc / 'result.json').exists(): continue
                result = json.loads((loc / 'result.json').read_text())
                assert result['steps'] == steps and result['episodes'] == 2048
                assert result['base_seed'] == 92_000_000
                assert digest(run / f'checkpoint_{steps}.pt') == result['checkpoint_sha256']
                with np.load(loc / 'samples.npz') as samples:
                    r, dr, c, dc = [samples[k] for k in ('undiscounted_reward','discounted_reward','undiscounted_cost','discounted_cost')]
                    assert all(v.shape == (2048,) and np.isfinite(v).all() for v in (r,dr,c,dc))
                    values = dict(return_mean=float(r.mean()), discounted_return_mean=float(dr.mean()),
                        cost_mean=float(c.mean()), discounted_cost_mean=float(dc.mean()),
                        cost_q90=float(np.quantile(c,.9)), discounted_cost_q90=float(np.quantile(dc,.9)),
                        violation_pct=float(100*(c>15).mean()), discounted_violation_pct=float(100*(dc>15).mean()))
                for k in ('return_mean','cost_mean','cost_q90'): assert abs(values[k]-result[k]) < 1e-8
                assert abs(values['violation_pct']-100*result['violation_rate']) < 1e-8
                assert abs(values['discounted_violation_pct']-100*result['discounted_violation_rate']) < 1e-8
                row = dict(setting=setting, method=method, nominal_steps=stage, actual_steps=steps, seed=seed, **values)
                rows.append(row); group.append(row)
            if len(group) != 4: continue
            aggregate = dict(setting=setting, method=method, nominal_steps=stage, actual_steps=steps, seeds=4)
            for key in values:
                vals = [r[key] for r in group]
                aggregate[key] = float(np.mean(vals)); aggregate[key+'_sd'] = float(np.std(vals, ddof=1))
            aggregate['raw_feasible_seeds'] = sum(r['violation_pct'] <= 10 for r in group)
            aggregate['discounted_feasible_seeds'] = sum(r['discounted_violation_pct'] <= 10 for r in group)
            aggregate['primary_constraint_gamma'] = gamma
            groups.append(aggregate)
    emit_csv(ROOT / 'per_seed_results.csv', rows); emit_csv(ROOT / 'results.csv', groups)
    for setting, root in (('undiscounted',RAW), ('discounted',DISC)):
        emit_csv(root / 'results.csv', [r for r in groups if r['setting']==setting])
        emit_csv(root / 'per_seed_results.csv', [r for r in rows if r['setting']==setting])
    return rows, groups


def update():
    steps = curves(); rows, groups = validation()
    keys = {(s,m) for s,_,m,_ in NEW}
    new_rows = [r for r in rows if (r['setting'],r['method']) in keys]
    receipts = 0
    for setting,root,method,gamma in NEW:
        for seed in range(4):
            run = root/f'{method}-s{seed}'
            path = run/'training_receipt.json'
            if not path.exists(): continue
            receipt = json.loads(path.read_text())
            config = json.loads((run/'config.json').read_text())
            assert receipt['checkpoint_roundtrip'] is True
            assert receipt['actual_env_steps'] == steps_for(method,10_000_000)
            assert receipt['signature'] == config['signature']
            assert config['signature']['seed'] == seed
            if method=='tqpo':
                p=receipt['tqpo_identity']
                assert (p['gamma'],p['cost_gamma'],p['dual_gamma'])==(gamma,gamma,gamma)
                assert p['failure_alpha']==.1 and p['cost_threshold']==15.
                assert receipt['optimizer_epoch_count']==6664 and receipt['dual_update_count']==834
            else:
                s=config['signature']
                assert (s['gamma_reward'],s['gamma_cost'],s['dual_gamma'])==(gamma,gamma,gamma)
                assert receipt['dual_update_count']==(400 if method=='dca' else 834)
                if method=='qcpo': assert receipt['optimizer_steps']==6664
            receipts+=1
    value = dict(updated=datetime.now(timezone.utc).isoformat(), steps=steps,
        new_validation_completed=len(new_rows), new_validation_expected=40,
        training_receipts=receipts, aggregate=groups, complete=len(new_rows)==40 and receipts==20)
    atomic_json(ROOT/'progress.json', value)
    return value


if __name__ == '__main__':
    print(json.dumps(update(), indent=2))
