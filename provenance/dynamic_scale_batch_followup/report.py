from pathlib import Path
import json,csv,hashlib
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;O=R.parents[1]/'outputs/followup';P=json.loads((R/'protocol.json').read_text());BASE=Path(P['baseline_root'])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
groups=[dict(g,root=R/g['key'],new=True) for g in P['groups']]
groups += [dict(key='dca_div100',method='dca',batch=25,divisor=100,root=BASE/'dca',label='DCA /100 B25',new=False,checkpoints=[5000000,10000000]),dict(key='tqpo_b25',method='tqpo',batch=25,root=BASE/'tqpo',label='TQPO fixed B25',new=False,checkpoints=[5000000,10000000]),dict(key='qcpo_b25',method='qcpo',batch=25,root=BASE/'qcpo',label='QCPO B25 reference',new=False,checkpoints=[5000000,10000000])]
rows=[];summ=[];curve_rows=[];grad_rows=[];progress={};series={}
for g in groups:
 series[g['key']]=[]
 for seed in (0,1):
  run=g['root']/f'seed{seed}';st=run/'status.json';progress[f"{g['key']}/s{seed}"]=json.loads(st.read_text()) if st.exists() else {'status':'pending','actual_env_steps':0}
  for nominal,step in zip((5000000,10000000),g['checkpoints']):
   dest=run/f'validation_{step}';p=dest/'result.json'
   if not p.exists():continue
   x=json.loads(p.read_text());assert x['steps']==step and x['episodes']==2048 and x['base_seed']==92000000
   assert x['checkpoint_sha256']==sha(run/f'checkpoint_{step}.pt')
   with np.load(dest/'samples.npz') as z:
    rr,cc,dr=[z[k] for k in ('undiscounted_reward','undiscounted_cost','discounted_reward')];assert all(a.shape==(2048,) and np.isfinite(a).all() for a in (rr,cc,dr))
    row=dict(group=g['key'],label=g['label'],batch=g['batch'],divisor=g.get('divisor',''),seed=seed,nominal_steps=nominal,actual_steps=step,new=g['new'],raw_return=float(rr.mean()),discounted_return=float(dr.mean()),cost_mean=float(cc.mean()),violation_pct=float(100*(cc>15).mean()))
   assert np.isclose(row['raw_return'],x['return_mean']) and np.isclose(row['violation_pct'],100*x['violation_rate']);rows.append(row)
  ep=run/'episodes.jsonl';a=[]
  if ep.exists():
   for l in ep.read_text().splitlines():
    try:x=json.loads(l)
    except json.JSONDecodeError:continue
    rr=x.get('undiscounted_reward',x.get('undiscounted_return'));cc=x['undiscounted_cost']
    a.append([x['actual_env_steps'],np.mean(rr),np.mean(cc),100*np.mean(np.asarray(cc)>15)])
  if a:series[g['key']].append(np.asarray(a))
  p=run/'metrics.jsonl'
  if p.exists():
   for l in p.read_text().splitlines():
    try:x=json.loads(l)
    except json.JSONDecodeError:continue
    prefix='diagnostic/fixed_reward_scale/';k=prefix+'risk_reward_gradient_ratio_scaled'
    if k in x:grad_rows.append(dict(group=g['key'],seed=seed,steps=x['actual_env_steps'],mean_grad=x[prefix+'reward_score_gradient_norm_scaled'],weighted_risk_grad=x[prefix+'risk_score_gradient_norm'],ratio=x[k],cosine=x[prefix+'reward_risk_loss_gradient_cosine'],lambda_at_update=x[prefix+'mu_at_actor_update']))
 for nominal,actual in zip((5000000,10000000),g['checkpoints']):
  rs=[x for x in rows if x['group']==g['key'] and x['nominal_steps']==nominal]
  if len(rs)!=2:continue
  d=dict(group=g['key'],label=g['label'],batch=g['batch'],divisor=g.get('divisor',''),nominal_steps=nominal,actual_steps=actual,seeds=2,feasible_seeds=sum(x['violation_pct']<=20 for x in rs))
  for k in ('raw_return','discounted_return','cost_mean','violation_pct'):d[k]=float(np.mean([x[k] for x in rs]));d[k+'_sd']=float(np.std([x[k] for x in rs],ddof=1))
  summ.append(d)
fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained');colors={'dca_div25':'#e29b21','dca_div100':'#2675b8','dca_div200':'#259063','tqpo_b12':'#9b51aa','tqpo_b25':'#d05c4b','qcpo_b25':'#777777'}
for g in groups:
 ss=series[g['key']]
 if len(ss)!=2:continue
 end=min(s[-1,0] for s in ss);grid=np.arange(50000,end+1,50000)
 if len(grid)<4:continue
 row=0 if g['method']=='dca' else 1
 for j,metric in enumerate(('raw_return','cost_mean','violation_pct')):
  vals=np.asarray([np.convolve(np.interp(grid,s[:,0],s[:,j+1]),np.ones(4)/4,'valid') for s in ss]);mu=vals.mean(0);sd=vals.std(0,ddof=1);xx=grid[3:];ax=axes[row,j];ax.plot(xx/1e6,mu,label=g['label'],color=colors[g['key']]);ax.fill_between(xx/1e6,mu-sd,mu+sd,color=colors[g['key']],alpha=.12)
  curve_rows.extend(dict(group=g['key'],metric=metric,steps=int(t),mean=float(a),sd=float(b)) for t,a,b in zip(xx,mu,sd))
for i in range(2):
 for j,t in enumerate(('Undiscounted return','Undiscounted cost','Violation (%)')):axes[i,j].set(title=('DCA scales: ' if i==0 else 'TQPO batch comparison: ')+t,xlabel='Environment steps (million)');axes[i,j].grid(alpha=.2)
 axes[i,2].axhline(20,color='black',ls='--',lw=1);axes[i,0].legend(fontsize=8)
fig.savefig(O/'training_comparison.png',dpi=170);fig.savefig(O/'training_comparison.pdf');plt.close(fig)
for name,data in [('per_seed.csv',rows),('results.csv',summ),('training_curves.csv',curve_rows),('gradient_diagnostics.csv',grad_rows)]:
 if data:
  with (O/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
new_count=sum(x['new'] for x in rows);d=dict(complete=new_count==12,new_evaluations=new_count,total_evaluations_including_baselines=len(rows),results=summ,progress=progress)
tmp=O/'.summary.json.tmp';tmp.write_text(json.dumps(d,indent=2));tmp.replace(O/'summary.json')
print(json.dumps(dict(new_evaluations=new_count,total_evaluations=len(rows))))
