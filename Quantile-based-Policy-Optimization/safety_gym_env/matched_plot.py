"""Plot the fixed final checkpoints with paired training seeds and export the tables."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import matched_campaign as c


def collect():
    rows = []
    for method, gamma, seed in c.JOBS:
        d = c.directory((method, gamma, seed))
        marker = d / 'eval.json'
        if not marker.exists() or not c.read(marker)['ok']:
            continue
        data = Path(c.read(marker)['data'])
        trained = Path(c.read(d/'train.json')['data'])
        if method == 'dca':
            cfg = c.read(trained/'config.json')
            assert (cfg['seed'], cfg['gamma']) == (seed, gamma)
            assert c.read(data/'config.json')['eval_seed'] == 92000000
        else:
            cfg = c.read(trained/f'run_{seed}/params.json')
            assert (cfg['runner']['seed'], cfg['algo']['discount'], cfg['algo']['cost_discount']) == (seed, gamma, gamma)
        scores = np.load(data/'final_episodes.npy')
        assert scores.shape == (2048, 4) and np.isfinite(scores).all() and np.all(scores[:,3] == 1000)
        costs = scores[:,1]
        rate = np.mean(costs > 15)
        endpoint = c.read(data/'final.json')
        metrics = [scores[:,0].mean(), costs.mean(), np.quantile(costs, .8), rate]
        np.testing.assert_allclose(metrics, [endpoint[k] for k in ('reward','cost','q80','violation')])
        rows.append(dict(method=method.upper(), gamma=gamma, seed=seed, episodes=len(costs),
                         reward=metrics[0], cost=metrics[1], q80=metrics[2], violation_pct=100*rate,
                         violations=int(np.sum(costs > 15)), feasible=int(rate <= .20)))
    return rows


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(rows, folder):
    plt.rcParams.update({'font.family':['DejaVu Sans','Droid Sans Fallback'],
                         'axes.unicode_minus':False, 'pdf.fonttype':42, 'font.size':11})
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.8), sharey='row')
    fig.subplots_adjust(left=.09, right=.977, bottom=.19, top=.79, wspace=.15, hspace=.28)
    fig.text(.09, .95, 'SimpleButton 上的折扣训练对照', fontsize=20, color='#152733')
    fig.text(.09, .897, f"最终策略 {len(rows)}/60   ·   每个策略 2,048 局   ·   统一未折扣评测",
             fontsize=11.5, color='#455962')
    palette = ['#a75d22','#087977']
    for j, method in enumerate(('DCA','QCPO','TQPO')):
        arms = [{r['seed']:r for r in rows if r['method']==method and r['gamma']==g} for g in (.99,1.)]
        paired = sorted(arms[0].keys() & arms[1].keys())
        axes[0,j].set_title(f'{method}  ·  {len(paired)} 对 seed', fontsize=13, loc='left', pad=9)
        for i, key in enumerate(('reward','violation_pct')):
            ax = axes[i,j]
            for seed in paired:
                y = [arm[seed][key] for arm in arms]
                ax.plot([0,1], y, color='#aabac1', lw=1.1, zorder=2)
                ax.scatter([0,1],y,s=23, color=palette, edgecolor='white', linewidth=.4,zorder=3)
            if paired:
                means = [np.mean([arm[s][key] for s in paired]) for arm in arms]
                ax.plot([0,1],means,color='#233a48', lw=2.5, marker='D', ms=6, zorder=4)
                for x,val in enumerate(means):
                    ax.annotate(f'{val:.1f}', (x,val), xytext=((-6 if x==0 else 6),0),
                                textcoords='offset points', ha='right' if x==0 else 'left',
                                va='center', fontsize=10, color='#172e39',
                                bbox=dict(facecolor='white', edgecolor='none', pad=.5), zorder=5)
            ax.set_xlim(-.37,1.37)
            ax.set_xticks([0,1], ['折扣\nγ = 0.99','未折扣\nγ = 1.00'] if i else ['',''])
            ax.tick_params(length=0, pad=4, labelcolor='#425665', labelsize=10)
            ax.spines[['top','right','left']].set_visible(False)
            ax.spines['bottom'].set_color('#bdcbd0')
            ax.set_axisbelow(True)
            ax.grid(axis='y',color='#dce4e7',lw=.6)
            if i:
                ax.axhline(20,color='#aa393a',lw=1.3,ls=(0,(4,3)),zorder=1)
            if j==0:
                ax.set_ylabel('平均回报' if not i else '违规率（%）', fontsize=12)
    rewards=[r['reward'] for r in rows]; rates=[r['violation_pct'] for r in rows]
    axes[0,0].set_ylim(min(0, np.floor(min(rewards)/5)*5), max(5, np.ceil(max(rewards)/5)*5)+1)
    axes[1,0].set_ylim(0, max(30,10*np.ceil(max(rates)/10))+2)
    fig.text(.09, .055, '细线连接同一训练 seed。黑色菱形是配对 seed 的均值。红色虚线标出 20% 违规率上限。', fontsize=10, color='#435963')
    for ext in ('png','pdf'):
        fig.savefig(folder/f'discount_comparison.{ext}', dpi=190, facecolor='white')
    plt.close(fig)


def main():
    rows = collect()
    if not rows:
        return
    folder = c.HERE.parent/'paper/rebuttal/simplebutton_discount'/f'{len(rows)}_of_60'
    folder.mkdir(parents=True, exist_ok=True)
    summary, contrasts = [], []
    for method in ('DCA','QCPO','TQPO'):
        arms=[]
        for gamma in (.99,1.):
            arm = {r['seed']:r for r in rows if r['method']==method and r['gamma']==gamma}
            arms.append(arm)
            if not arm:
                continue
            row=dict(method=method, gamma=gamma, n=len(arm), feasible=sum(r['feasible'] for r in arm.values()))
            for key in ('reward','cost','q80','violation_pct'):
                vals=[r[key] for r in arm.values()]
                row[key+'_mean']=np.mean(vals)
                row[key+'_sd']=np.std(vals,ddof=1) if len(vals)>1 else ''
            summary.append(row)
        seeds=sorted(arms[0].keys() & arms[1].keys())
        if seeds:
            contrast=dict(method=method, n_pairs=len(seeds), seeds=','.join(map(str,seeds)))
            for key in ('reward','cost','q80','violation_pct'):
                diff=[arms[1][s][key]-arms[0][s][key] for s in seeds]
                contrast['delta_'+key]=np.mean(diff)
            contrasts.append(contrast)
    render(rows,folder)
    write_csv(folder/'per_seed.csv',rows)
    if contrasts:
        write_csv(folder/'paired_differences.csv',contrasts)
    write_csv(folder/'summary.csv',summary)
    print(folder)


if __name__=='__main__':
    main()
