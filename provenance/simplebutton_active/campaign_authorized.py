"""Continue only the user's two settings; adopt live jobs without interrupting them."""
import fcntl,json,os,signal,subprocess,time,traceback
from pathlib import Path
from datetime import datetime,timezone
from report import ROOT,ENTRIES,atomic_json,update,source_audit,receipt

def now():return datetime.now(timezone.utc).isoformat()
def live(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0] not in ('Z','X')
 except FileNotFoundError:return False
def complete(e):
 run=Path(e['run'])
 return receipt(e) and all((run/f'validation_{n}/result.json').exists() for n in e['checkpoints'])

def main():
 lock=(ROOT/'authorized_campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 handoff=json.loads((ROOT/'REVIEW_HOLD.json').read_text());oldpid=handoff['scheduler_pid']
 running={};streams=[];stopped=False
 status=dict(state='starting',started=now(),pid=os.getpid(),scope='reward .99, actor cost1, lambda cost1, eps.10/.20 only',jobs={},retired_scheduler_pid=oldpid)
 for key,j in handoff['running_jobs_preserved'].items():
  e=next(e for e in ENTRIES if e['key']==key)
  if live(j['pid']):running[j['gpu']]=dict(pid=j['pid'],entry=e,proc=None,adopted=True);status['jobs'][key]=dict(j,adopted_without_restart=True)
  else:
   assert complete(e) or (Path(e['run'])/'latest.pt').exists(),(key,'no checkpoint to resume')
 active={r['entry']['key'] for r in running.values()}
 queue=[e for e in ENTRIES if e['key'] not in active and not complete(e)]
 # Existing QCPO jobs may finish, but never launch any new QCPO training.
 assert all(e['method']!='qcpo' or e['key'] in handoff['running_jobs_preserved'] for e in queue), 'No new QCPO training allowed'
 queue.sort(key=lambda e:(e['key'] not in handoff['running_jobs_preserved'],e['key']))
 assert all((e['reward_gamma'],e['cost_gamma'],e['dual_gamma'])==(.99,1.,1.) for e in ENTRIES)
 def stop(*_):
  nonlocal stopped
  stopped=True
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 try:
  source_audit();last=0
  while True:
   if stopped:raise InterruptedError('Scheduler stopped; adopted training jobs preserved')
   for gpu,r in list(running.items()):
    proc=r['proc'];alive=proc.poll() is None if proc else live(r['pid'])
    if alive:continue
    e=r['entry']
    if not complete(e) and r['adopted'] and (Path(e['run'])/'latest.pt').exists():
     queue.insert(0,e);status['jobs'][e['key']].update(resume_from_checkpoint=True);del running[gpu];continue
    assert (proc is None or proc.returncode==0) and complete(e),f"Job did not complete: {e['key']}"
    status['jobs'][e['key']].update(finished=now(),returncode=0);del running[gpu]
   for gpu in range(4):
    if gpu in running or not queue:continue
    e=queue.pop(0);root=Path(e['root']);assert (root/'READY.json').exists()
    log=(ROOT/'logs'/f"authorized-{e['setting']}-{e['method']}-s{e['seed']}.log").open('a');streams.append(log)
    env=dict(os.environ,RARL_AUTHORIZED_DISPATCH='reward099_cost1_only')
    proc=subprocess.Popen(['bash',str(ROOT/'job_authorized.sh'),str(root),e['method'],str(e['seed']),'.99',str(gpu),str(root/'source/validate_gpu.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    running[gpu]=dict(pid=proc.pid,entry=e,proc=proc,adopted=False);status['jobs'][e['key']]=dict(pid=proc.pid,gpu=gpu,started=now())
   done=not running and not queue
   if time.monotonic()-last>60 or done:
    p=update();status.update(state='running',updated=now(),steps=p['steps'],training_receipts=p['training_receipts'],validations=p['validations'],running={str(gpu):r['entry']['key'] for gpu,r in running.items()},queued=[e['key'] for e in queue]);atomic_json(ROOT/'campaign_status.json',status);last=time.monotonic()
   if done:
    assert p['complete'];atomic_json(ROOT/'final_summary.json',dict(p,source_audit=source_audit(),completed=now()));status.update(state='complete',completed=now());break
   time.sleep(5)
 except BaseException:
  status.update(state='stopped' if stopped else 'failed',error=traceback.format_exc());raise
 finally:
  # Never signal the adopted jobs. A controller exit does not cancel allowed learning.
  status['live_jobs_preserved']=[r['pid'] for r in running.values() if live(r['pid'])]
  if not status['live_jobs_preserved']:
   # The retired controller is stopped and has no unfinished adopted training.
   if live(oldpid):
    cmd=Path(f'/proc/{oldpid}/cmdline').read_bytes()
    if str(ROOT/'campaign.py').encode() in cmd and all(not live(j['pid']) for j in handoff['running_jobs_preserved'].values()):os.kill(oldpid,signal.SIGKILL)
   subprocess.run(['bash',str(ROOT.parents[1]/'gpu_hold_20261110/gpu_hold.sh'),'start'],check=True);status['gpu_holder_restored']=True
  for f in streams:f.close()
  atomic_json(ROOT/'campaign_status.json',status)

if __name__=='__main__':main()
