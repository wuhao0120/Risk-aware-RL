import json,csv
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;O=R.parents[1]/'outputs';rows=[];summ=[]
fig,ax=plt.subplots(1,3,figsize=(14,4),layout='constrained')
for m in ('dca','qcpo','tqpo'):
 series=[]
 for seed in (0,1):
  run=R/m/f'seed{seed}'
  for step in (5000000,10000000):
   p=run/f'validation_{step}/result.json'
   if p.exists():
    x=json.loads(p.read_text());rows.append(dict(method=m,seed=seed,steps=step,return_mean=x['return_mean'],cost_mean=x['cost_mean'],violation_pct=100*x['violation_rate']))
  ep=run/'episodes.jsonl'
  if ep.exists():
   ys=[]
   for l in ep.read_text().splitlines():
    x=json.loads(l)
    if 'undiscounted_reward' not in x and 'undiscounted_return' in x:x['undiscounted_reward']=x['undiscounted_return']
    if 'undiscounted_reward' not in x:continue
    ys.append([x['actual_env_steps'],np.mean(x['undiscounted_reward']),np.mean(x['undiscounted_cost']),100*np.mean(np.asarray(x['undiscounted_cost'])>15)])
   if ys:series.append(np.asarray(ys))
 if len(series)==2:
  n=min(map(len,series));z=np.asarray([s[:n] for s in series]);x=z[0,:,0]/1e6
  for j in range(3):
   a=z[:,:,j+1];a=np.asarray([np.convolve(v,np.ones(8)/8,'valid') if len(v)>=8 else v for v in a]);xx=x[-a.shape[1]:];mu=a.mean(0);sd=a.std(0,ddof=1);line=ax[j].plot(xx,mu,label={'dca':'DCA /100','qcpo':'QCPO','tqpo':'TQPO fixed'}[m])[0];ax[j].fill_between(xx,mu-sd,mu+sd,color=line.get_color(),alpha=.15)
 for step in (5000000,10000000):
  rs=[x for x in rows if x['method']==m and x['steps']==step]
  if len(rs)==2:
   d=dict(method=m,steps=step,seeds=2,feasible_seeds=sum(x['violation_pct']<=20 for x in rs))
   for k in ('return_mean','cost_mean','violation_pct'):d[k]=float(np.mean([x[k] for x in rs]));d[k+'_sd']=float(np.std([x[k] for x in rs],ddof=1))
   summ.append(d)
for file,data in [('per_seed.csv',rows),('results.csv',summ)]:
 if data:
  with (O/file).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
for a,t in zip(ax,['Undiscounted return','Undiscounted cost','Violation (%)']):a.set(title=t,xlabel='Environment steps (million)');a.grid(alpha=.2)
ax[2].axhline(20,color='black',ls='--');ax[0].legend();fig.savefig(O/'training_curves.png',dpi=160);fig.savefig(O/'training_curves.pdf');plt.close(fig)
(O/'summary.json').write_text(json.dumps(dict(complete=len(rows)==12,results=summ),indent=2))
