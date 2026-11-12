"""Recompute appendix tables and learning curves from generated episode samples.

No episode pooling across seeds. Bands and table uncertainties use sample SD
(ddof=1). The event is strictly cost > 15; quantiles use linear interpolation.
Training curves use latest 100 episodes, a 50K transition grid, then a trailing
0.5M window inside each seed before averaging across seeds.
"""
from __future__ import annotations

from collections import deque
import csv
import json
from pathlib import Path

import numpy as np

METRICS = ('mean_return', 'mean_cost', 'cost_quantile_80', 'violation_percent')


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def episode_metrics(reward, cost):
    """Compute one seed's statistics; reject empty or nonfinite samples."""
    reward, cost = np.asarray(reward), np.asarray(cost)
    if reward.ndim != 1 or reward.shape != cost.shape or not reward.size:
        raise ValueError('Expected nonempty paired one-dimensional episode arrays')
    if not np.isfinite(reward).all() or not np.isfinite(cost).all():
        raise ValueError('Nonfinite episode data')
    p = float(np.mean(cost > 15.0))
    return dict(mean_return=float(reward.mean()), mean_cost=float(cost.mean()),
                cost_quantile_80=float(np.quantile(cost, .8, method='linear')),
                violation_percent=100 * p, feasible=p <= .2)


def summarize(root, selected, output):
    """Require every requested seed and its final endpoint before reporting."""
    protocol = json.loads((Path(__file__).parent / 'configs/protocol.json').read_text())
    output = Path(output)
    records, curves = [], []
    grid = np.arange(50000, 10000001, 50000)
    for method, seeds in selected.items():
        spec = protocol['methods'][method]
        for seed in seeds:
            run = Path(root) / f'{method}_seed{seed}'
            sample = run / 'evaluation' / str(spec['training_steps'])
            if method == 'qcpo':
                sample /= 'marginal'
            with np.load(sample / 'samples.npz', allow_pickle=False) as data:
                reward, cost = data['undiscounted_reward'], data['undiscounted_cost']
                if len(reward) != spec['evaluation_episodes']:
                    raise ValueError(f'{run.name}: wrong final evaluation episode count')
                records.append(dict(method=method.upper(), seed=seed,
                                    training_steps=spec['training_steps'],
                                    test_episodes=len(reward), **episode_metrics(reward, cost)))
            rewards, costs = deque(maxlen=100), deque(maxlen=100)
            endpoints, rows = [], []
            previous = 0
            for line in (run / 'episodes.jsonl').read_text().splitlines():
                row = json.loads(line)
                step = int(row['actual_env_steps'])
                rr = row.get('undiscounted_return', row.get('undiscounted_reward'))
                cc = row['undiscounted_cost']
                if step - previous != spec['batch_size'] or len(rr) != spec['num_envs'] or len(cc) != len(rr):
                    raise ValueError(f'{run.name}: incomplete or duplicated training episode stream')
                previous = step
                rewards.extend(rr); costs.extend(cc)
                stats = episode_metrics(rewards, costs)
                endpoints.append(step)
                rows.append([stats['mean_return'], stats['violation_percent'], stats['mean_cost']])
            if not endpoints or endpoints[-1] != spec['training_steps']:
                raise ValueError(f'{run.name}: incomplete training endpoint')
            indices = np.searchsorted(endpoints, grid)
            values = np.asarray(rows)[indices]
            steps = np.asarray(endpoints)[indices]
            starts = np.searchsorted(steps, steps - 500000, side='right')
            smoothed = np.asarray([values[j:i+1].mean(0) for i, j in enumerate(starts)])
            for index, step in enumerate(steps):
                curves.append(dict(method=method.upper(), seed=seed, grid_steps=int(grid[index]),
                                   actual_env_steps=int(step), mean_return=values[index, 0],
                                   violation_percent=values[index, 1], mean_cost=values[index, 2],
                                   smoothed_return=smoothed[index, 0],
                                   smoothed_violation_percent=smoothed[index, 1], smoothed_cost=smoothed[index, 2]))
    summary = []
    for method, seeds in selected.items():
        group = [row for row in records if row['method'] == method.upper()]
        row = dict(method=method.upper(), seeds=len(group),
                   test_episodes_per_seed=protocol['methods'][method]['evaluation_episodes'],
                   training_steps=protocol['methods'][method]['training_steps'],
                   feasible_seeds=sum(x['feasible'] for x in group))
        for key in METRICS:
            values = np.asarray([x[key] for x in group])
            row[key+'_mean'] = float(values.mean())
            row[key+'_sd'] = float(values.std(ddof=1)) if len(values) > 1 else None
        summary.append(row)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'final_per_seed.csv', records)
    write_csv(output / 'final_mean_std.csv', summary)
    write_csv(output / 'training_curves_per_seed.csv', curves)
    tex = []
    for row in summary:
        cells = [row['method']]
        for key in METRICS:
            mean, sd = row[key+'_mean'], row[key+'_sd']
            cells.append(f'${mean:.2f} \\pm {sd:.2f}$' if sd is not None else f'${mean:.2f}$')
        cells.append(f"{row['feasible_seeds']}/{row['seeds']}")
        tex.append(' & '.join(cells) + r' \\')
    (output / 'table_rows.tex').write_text('\n'.join(tex) + '\n')
    plot_curves(curves, output)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def plot_curves(curves, output):
    """Export unsmoothed and smoothed PDF/PNG figures with the full data range."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'DCA': '#2878b5', 'QCPO': '#ed7d31', 'TQPO': '#6c50a3'}
    for smooth in (False, True):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
        keys = ('smoothed_return', 'smoothed_violation_percent', 'smoothed_cost') if smooth else ('mean_return', 'violation_percent', 'mean_cost')
        for method in dict.fromkeys(row['method'] for row in curves):
            rows = [row for row in curves if row['method'] == method]
            seeds = sorted({row['seed'] for row in rows})
            steps = sorted({row['grid_steps'] for row in rows})
            for ax, key in zip(axes, keys):
                values = np.asarray([[row[key] for row in rows if row['seed'] == seed] for seed in seeds])
                mean = values.mean(0); sd = values.std(0, ddof=1) if len(seeds)>1 else np.zeros_like(mean)
                x = np.asarray(steps)/1e6
                ax.plot(x, mean, color=colors[method], label=method)
                ax.fill_between(x, mean-sd, mean+sd, color=colors[method], alpha=.16)
        for ax, label in zip(axes, ['Episode return', 'Violation probability (%)', 'Episode cost']):
            ax.set(xlabel='Training transitions (millions)', ylabel=label)
            ax.grid(alpha=.2)
        axes[1].axhline(20, color='black', linestyle='--', linewidth=1)
        axes[0].legend(frameon=False)
        name = 'learning_smoothed_0p5m' if smooth else 'learning_unsmoothed'
        for ext in ('pdf', 'png'):
            fig.savefig(output / f'{name}.{ext}', dpi=200)
        plt.close(fig)
