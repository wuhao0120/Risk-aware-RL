import os,sys,json,subprocess,time,signal,fcntl,traceback,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;OUT=R.parents[1]/'outputs';PY=sys.executable
CACHE=Path('/vepfs-mlp2/c20250510/251204033/Risk-aware-RL/experiment/ICLR_rebuttal_SimpleButton/cache')
def write(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def worker(m,seed,gpu,smoke):
 s=R/m/'source';run=R/m/('preflight' if smoke else f'seed{seed}');run.mkdir(parents=True,exist_ok=True)
 for n,h in json.loads((R/m/'source_manifest.json').read_text()).items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h,n
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',SIMPLEBUTTON_REWARD_GAMMA='1',SIMPLEBUTTON_TRAIN_GAMMA='1',RISK_EVAL_SEED='81000000' if smoke else '92000000',MUJOCO_GL='disable',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',XDG_CACHE_HOME=str(CACHE),WARP_CACHE_PATH=str(CACHE/'warp'),MPLCONFIGDIR=str(CACHE/'mpl'),TMPDIR=str(CACHE/'tmp'))
 batches=(22 if m=='dca' else 2) if smoke else 400
 args={'dca':['run.py','--arm','schemeb_raw_mean100_lambda_norm','--reward-scale','100','--main-batches',str(batches-20)],'qcpo':['qcpo_run.py','--main-batches',str(batches)],'tqpo':['run_tqpo.py','--batches',str(batches)]}[m]
 if m=='tqpo' and smoke:args=['run_tqpo.py','--smoke']
 if not (run/'training_receipt.json').exists():
  with (run/'training.log').open('a') as f:subprocess.run([PY,'-u',str(s/args[0]),*args[1:],'--seed',str(seed),'--output',str(run)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 rec=json.loads((run/'training_receipt.json').read_text());cfg=json.loads((run/'config.json').read_text());assert rec['actual_env_steps']==batches*25000 and rec['checkpoint_roundtrip']
 if m=='tqpo':
  p=rec['tqpo_identity'];assert rec['cdf_guard_fixed'] and (p['gamma'],p['cost_gamma'],p['dual_gamma'],p['num_envs'],p['failure_alpha'])==(1,1,1,25,.2)
 else:
  p=cfg['signature'];assert (p['gamma_reward'],p['gamma_cost'],p['dual_gamma'],p['epsilon'])==(1,1,1,.2);assert cfg['configuration']['num_envs']==25
  if m=='dca':assert p['mean_divisor']==100
 for steps in ([batches*25000] if smoke else [5000000,10000000]):
  dest=run/f'validation_{steps}';n=16 if smoke else 2048
  # QCPO preflight saves latest/final; name the verified final checkpoint explicitly.
  cp=run/f'checkpoint_{steps}.pt'
  if not cp.exists() and steps==batches*25000:
   import shutil;shutil.copy2(run/'final.pt',cp)
  if not (dest/'result.json').exists():
   with (run/f'eval_{steps}.log').open('a') as f:subprocess.run([PY,'-u',str(s/'validate_gpu.py'),m,str(run),str(steps),str(n),str(dest)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  result=json.loads((dest/'result.json').read_text());assert result['base_seed']==(81000000 if smoke else 92000000) and result['episodes']==n and result['checkpoint_sha256']==hashlib.sha256(cp.read_bytes()).hexdigest()
 write(run/'COMPLETE.json',dict(method=m,seed=seed,steps=batches*25000,preflight=smoke))
def busy():
 occupied=set()
 for p in Path('/proc').glob('[0-9]*'):
  try:
   cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode();env=(p/'environ').read_bytes().split(b'\0')
   if str(R) in cmd or not any(x in cmd for x in ('run.py','qcpo_run.py','run_tqpo.py','validate_gpu.py')):continue
   for e in env:
    if e.startswith(b'CUDA_VISIBLE_DEVICES='):occupied.update(int(g) for g in e.split(b'=')[1].split(b',') if g.isdigit())
  except (OSError,ValueError):pass
 return occupied
if len(sys.argv)>1:
 worker(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),sys.argv[4]=='1');sys.exit()
lock=(R/'campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
status={'state':'preflight','started':time.time(),'jobs':{}};running={};stopping=False

def stop(*_):
 global stopping
 stopping=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
try:
 for smoke in (True,False):
  queue=[(m,seed) for m in ('dca','qcpo','tqpo') for seed in ([71] if smoke else [0,1])]
  status['state']='preflight' if smoke else 'pilot'
  while queue or running:
   if stopping:raise InterruptedError('Stopped, checkpoints retained')
   for gpu,(proc,key,stream) in list(running.items()):
    code=proc.poll()
    if code is None:continue
    stream.close();del running[gpu];status['jobs'][key].update(returncode=code,finished=time.time())
    if code:raise RuntimeError(f'{key} failed, see worker log')
    if not smoke:subprocess.run([PY,str(R/'report.py')],check=True)
   occupied=busy()
   for gpu in (3,2):
    if gpu in running or gpu in occupied or not queue:continue
    m,seed=queue.pop(0);key=f'{"preflight" if smoke else "pilot"}-{m}-{seed}';f=(R/f'{key}.log').open('a')
    p=subprocess.Popen([PY,'-u',str(__file__),m,str(seed),str(gpu),str(int(smoke))],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    running[gpu]=(p,key,f);status['jobs'][key]=dict(pid=p.pid,gpu=gpu,started=time.time())
   status.update(updated=time.time(),queued=len(queue),running={str(g):k for g,(_,k,_) in running.items()});write(R/'status.json',status);write(OUT/'status.json',status);time.sleep(5)
 status.update(state='complete',finished=time.time());subprocess.run([PY,str(R/'report.py')],check=True)
except BaseException:
 status.update(state='stopped' if stopping else 'failed',error=traceback.format_exc());raise
finally:
 for p,k,f in running.values():
  if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
 write(R/'status.json',status);write(OUT/'status.json',status)
