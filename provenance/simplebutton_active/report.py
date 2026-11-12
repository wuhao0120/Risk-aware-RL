"""Final B12 matrix: fixed checkpoint comparisons and four-seed learning curves."""
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report_common import atomic_json,emit_csv,digest,read_rows,performance_curve

ROOT=Path(__file__).resolve().parent
ENTRIES=json.loads((ROOT/('active_manifest.json' if (ROOT/'active_manifest.json').exists() else 'manifest.json')).read_text())

def source_audit(ready_only=False):
 roots={e['root'] for e in ENTRIES};audited=[]
 for name in sorted(roots):
  root=Path(name)
  if root.is_relative_to(ROOT):
   if ready_only and not (root/'READY.json').exists():continue
   manifest=json.loads((root/'source_manifest.json').read_text())
   for p,h in manifest.items():assert digest(root/'source'/p)==h,(root,p)
   audited.append(name)
 return dict(roots=audited,unchanged=True)

def receipt(e):
 run=Path(e['run']);p=run/'training_receipt.json'
 if not p.exists():return False
 r=json.loads(p.read_text());c=json.loads((run/'config.json').read_text());s=c['signature']
 assert r['actual_env_steps']==10008000 and r['checkpoint_roundtrip'] and s['seed']==e['seed']
 assert r['signature']==s
 for name,h in s['source'].items():assert digest(Path(e['source'])/name)==h,(e['key'],name)
 if e['method']=='tqpo':
  q=r['tqpo_identity'];assert (q['gamma'],q['cost_gamma'],q['dual_gamma'],q['failure_alpha'],q['num_envs'])==(e['reward_gamma'],e['cost_gamma'],e['dual_gamma'],e['epsilon'],12)
  assert r['cdf_guard_fixed'] and not r['cdf_guard_preserved'] and q['cdf_guard']=='len_ge_100_after_history_warmup'
  assert r['dual_update_count']==834 and r['optimizer_epoch_count']==6664
 else:
  assert (s['gamma_reward'],s['gamma_cost'],s['dual_gamma'],s['epsilon'])==(e['reward_gamma'],e['cost_gamma'],e['dual_gamma'],e['epsilon'])
  assert c['configuration']['num_envs']==12
  assert r['dual_update_count']==834
  if e['method']=='qcpo':assert r['optimizer_steps']==6664
  else:
   assert s['mean_divisor']==25 and s['warmup_batches']==20 and s['main_batches']==814
   assert r['critic_optimizer_steps']==327600 and r['target_update_count']==2502
   assert r['value_optimizer_steps']==79104 and r['num_envs']==12 and r['horizon']==1000
 first=read_rows(run/'metrics.jsonl')[0]
 assert first['wall_time']>=1793462400 # 2026-11-01 00:00 UTC; all selected runs begin later.
 return True

def update(plot=True):
 rows,groups,progress,receipts,failures=[],[],{},[],[]
 for e in ENTRIES:
  run=Path(e['run']);ms=read_rows(run/'metrics.jsonl')
  progress[e['key']]=ms[-1]['actual_env_steps'] if ms else 0
  status=run/'status.json'
  if status.exists():
   st=json.loads(status.read_text())
   if st.get('status',st.get('state'))=='failed':failures.append(e['key'])
  if receipt(e):receipts.append(e['key'])
  for count in e['checkpoints']:
   dest=run/f'validation_{count}'
   if not (dest/'result.json').exists():continue
   r=json.loads((dest/'result.json').read_text())
   assert r['steps']==count and r['episodes']==2048 and r['base_seed']==92000000
   assert r['checkpoint_sha256']==digest(run/f'checkpoint_{count}.pt')
   with np.load(dest/'samples.npz') as data:
    rr,dr,cc,dc=[data[k] for k in ('undiscounted_reward','discounted_reward','undiscounted_cost','discounted_cost')]
    assert all(v.shape==(2048,) and np.isfinite(v).all() for v in (rr,dr,cc,dc))
    values=dict(raw_return=float(rr.mean()),discounted_return=float(dr.mean()),raw_cost=float(cc.mean()),discounted_cost=float(dc.mean()),raw_violation_pct=float(100*(cc>15).mean()),discounted_violation_pct=float(100*(dc>15).mean()),primary_cost_quantile=float(np.quantile(cc if e['dual_gamma']==1 else dc,1-e['epsilon'])))
   assert abs(values['raw_return']-r['return_mean'])<1e-8 and abs(values['raw_violation_pct']-100*r['violation_rate'])<1e-8
   values['objective_return']=values['raw_return' if e['reward_gamma']==1 else 'discounted_return']
   values['primary_violation_pct']=values['raw_violation_pct' if e['dual_gamma']==1 else 'discounted_violation_pct']
   rows.append(dict(section=e['section'],setting=e['setting'],method=e['method'],seed=e['seed'],steps=count,reward_gamma=e['reward_gamma'],cost_gamma=e['cost_gamma'],dual_gamma=e['dual_gamma'],epsilon=e['epsilon'],**values))
 for setting,method,count in sorted({(r['setting'],r['method'],r['steps']) for r in rows}):
  rs=[r for r in rows if (r['setting'],r['method'],r['steps'])==(setting,method,count)]
  if len(rs)!=4:continue
  first=rs[0];g={k:first[k] for k in ('section','setting','method','steps','reward_gamma','cost_gamma','dual_gamma','epsilon')};g['seeds']=4
  for key in ('raw_return','discounted_return','raw_cost','discounted_cost','raw_violation_pct','discounted_violation_pct','primary_cost_quantile','objective_return','primary_violation_pct'):
   values=[r[key] for r in rs];g[key]=float(np.mean(values));g[key+'_sd']=float(np.std(values,ddof=1))
  g['feasible_seeds']=sum(r['primary_violation_pct']<=100*first['epsilon'] for r in rs);groups.append(g)
 emit_csv(ROOT/'per_seed_results.csv',rows);emit_csv(ROOT/'results.csv',groups)
 if plot:
  out=ROOT/'curves';out.mkdir(exist_ok=True);all_curves=[]
  for setting in sorted({e['setting'] for e in ENTRIES}):
   es=[e for e in ENTRIES if e['setting']==setting];fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained');axs=axs.ravel()
   for method,color in [('dca','#246698'),('qcpo','#ae8725'),('tqpo','#39805b')]:
    selected=sorted([e for e in es if e['method']==method],key=lambda e:e['seed'])
    if not selected:continue
    series=[performance_curve(Path(e['run']))[0] for e in selected];n=min(map(len,series))
    if not n:continue
    data=np.asarray([a[:n] for a in series]);x=50000*np.arange(1,n+1)
    for i,key in enumerate(('raw_return','discounted_return','raw_violation_pct','discounted_violation_pct')):
     mean,sd=data[:,:,i].mean(0),data[:,:,i].std(0,ddof=1)
     axs[i].plot(x/1e6,mean,color=color,label=method.upper()+(' fixed' if method=='tqpo' else ''))
     axs[i].fill_between(x/1e6,mean-sd,mean+sd,color=color,alpha=.17)
     for j,step in enumerate(x):all_curves.append(dict(setting=setting,method=method,metric=key,steps=int(step),**{f'seed{s}':float(data[s,j,i]) for s in range(4)},mean=float(mean[j]),sample_sd=float(sd[j])))
   for i,title in enumerate(('Undiscounted return','Discounted return','Raw-cost violation (%)','Discounted-cost violation (%)')):
    axs[i].set_title(title);axs[i].set_xlabel('Environment steps (millions)');axs[i].grid(alpha=.2)
    if i>=2:axs[i].axhline(100*es[0]['epsilon'],color='#444',ls='--',lw=1)
   if axs[0].lines:axs[0].legend(frameon=False)
   fig.suptitle(f"{setting} | B12 H1000 | 4 seeds, mean +/- sample SD")
   for fmt in ('png','pdf'):
    tmp=out/f'{setting}.{fmt}.tmp';fig.savefig(tmp,format=fmt,dpi=180);tmp.replace(out/f'{setting}.{fmt}')
   plt.close(fig)
  emit_csv(out/'training_curves.csv',all_curves)
 expected=len(ENTRIES);validation_expected=sum(len(e['checkpoints']) for e in ENTRIES)
 value=dict(updated=datetime.now(timezone.utc).isoformat(),steps=progress,training_receipts=len(receipts),training_expected=expected,new_training_completed=sum(not e['reused'] and e['key'] in receipts for e in ENTRIES),new_training_expected=sum(not e['reused'] for e in ENTRIES),validations=len(rows),validations_expected=validation_expected,aggregate=groups,failures=failures,complete=len(receipts)==expected and len(rows)==validation_expected)
 atomic_json(ROOT/'progress.json',value)
 return value

if __name__=='__main__':print(json.dumps(update(),indent=2))
