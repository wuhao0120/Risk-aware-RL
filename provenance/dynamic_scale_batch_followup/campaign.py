import os,sys,json,subprocess,time,signal,fcntl,traceback,hashlib,contextlib
from pathlib import Path
R=Path(__file__).resolve().parent;OUT=R.parents[1]/'outputs/followup';PY=sys.executable
P=json.loads((R/'protocol.json').read_text());GROUPS={g['key']:g for g in P['groups']}
CACHE=Path('/vepfs-mlp2/c20250510/251204033/Risk-aware-RL/experiment/ICLR_rebuttal_SimpleButton/cache')
def write(p,d):
 t=p.with_name('.'+p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def worker(key,seed,gpu,smoke):
 g=GROUPS[key];m=g['method'];s=R/key/'source';run=R/key/('preflight' if smoke else f'seed{seed}');run.mkdir(parents=True,exist_ok=True)
 for n,h in json.loads((R/key/'source_manifest.json').read_text()).items():assert sha(s/n)==h,n
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',SIMPLEBUTTON_REWARD_GAMMA='1',SIMPLEBUTTON_TRAIN_GAMMA='1',RISK_EVAL_SEED='81000000' if smoke else '92000000',MUJOCO_GL='disable',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',XDG_CACHE_HOME=str(CACHE),WARP_CACHE_PATH=str(CACHE/'warp'),MPLCONFIGDIR=str(CACHE/'mpl'),TMPDIR=str(CACHE/'tmp'))
 batches=(22 if m=='dca' else 2) if smoke else g['batches'];steps=batches*g['batch']*1000
 if m=='dca':args=['run.py','--arm',f"schemeb_raw_mean{g['divisor']}_lambda_norm",'--reward-scale',str(g['divisor']),'--main-batches',str(batches-20)]
 else:args=['run_tqpo.py','--smoke'] if smoke else ['run_tqpo.py','--batches',str(batches)]
 if not (run/'training_receipt.json').exists():
  with (run/'training.log').open('a') as f:subprocess.run([PY,'-u',str(s/args[0]),*args[1:],'--seed',str(seed),'--output',str(run)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 rec=json.loads((run/'training_receipt.json').read_text());cfg=json.loads((run/'config.json').read_text());sig=cfg['signature']
 assert rec['actual_env_steps']==cfg['planned_env_steps']==steps and rec['checkpoint_roundtrip'] and rec['signature']==sig
 assert rec['dual_update_count']==batches and sig['seed']==seed
 for n,h in sig['source'].items():assert sha(s/n)==h
 if m=='tqpo':
  p=rec['tqpo_identity'];assert rec['cdf_guard_fixed'] and not rec['cdf_guard_preserved']
  assert (p['gamma'],p['cost_gamma'],p['dual_gamma'],p['num_envs'],p['batch_size'],p['recurrent_batch'],p['failure_alpha'],p['environment'])==(1,1,1,12,12000,120,.2,'Dynamic')
  assert rec['optimizer_epoch_count']==8*(batches-1)
 else:
  assert (sig['gamma_reward'],sig['gamma_cost'],sig['dual_gamma'],sig['epsilon'],sig['threshold'],sig['mean_divisor'])==(1,1,1,.2,15,g['divisor'])
  assert cfg['configuration']['num_envs']==25 and cfg['configuration']['env_name']=='Dynamic'
  assert rec['critic_optimizer_steps']==2000+(batches-20)*400 and rec['target_update_count']==3*batches
 for step in ([steps] if smoke else g['checkpoints']):
  dest=run/f'validation_{step}';n=16 if smoke else 2048;cp=run/f'checkpoint_{step}.pt';assert cp.exists(),cp
  if not (dest/'result.json').exists():
   with (run/f'eval_{step}.log').open('a') as f:subprocess.run([PY,'-u',str(s/'validate_gpu.py'),m,str(run),str(step),str(n),str(dest)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  v=json.loads((dest/'result.json').read_text());assert (v['base_seed'],v['episodes'],v['steps'])==(81000000 if smoke else 92000000,n,step) and v['checkpoint_sha256']==sha(cp)
 write(run/'COMPLETE.json',dict(group=key,seed=seed,steps=steps,batch=g['batch'],preflight=smoke))
def busy():
 occupied=set()
 for p in Path('/proc').glob('[0-9]*'):
  try:
   cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode();env=(p/'environ').read_bytes().split(b'\0')
   if str(R) in cmd or not any(x in cmd for x in ('/run.py','qcpo_run.py','run_tqpo.py','validate_gpu.py','calibrate_final.py')):continue
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
def save():
 status.update(updated=time.time(),running={str(g):k for g,(_,k,_) in running.items()});write(R/'status.json',status);write(OUT/'status.json',status)
try:
 for smoke in (True,False):
  queue=[(g,seed) for seed in ([71] if smoke else [0,1]) for g in GROUPS if not (R/g/('preflight' if smoke else f'seed{seed}')/'COMPLETE.json').exists()];errors=[]
  status['state']='preflight' if smoke else 'training'
  while queue or running:
   if stopping:raise InterruptedError('Stopped; checkpoints retained')
   for gpu,(proc,key,stream) in list(running.items()):
    code=proc.poll()
    if code is None:continue
    stream.close();del running[gpu];status['jobs'][key].update(returncode=code,finished=time.time())
    if code:errors.append(key)
    if not smoke:
     with (R/'report.log').open('a') as f:
      rc=subprocess.run([PY,str(R/'report.py')],stdout=f,stderr=subprocess.STDOUT).returncode
      status['last_report_returncode']=rc
   occupied=busy()
   for gpu in (0,1,2,3):
    if gpu in running or gpu in occupied or not queue:continue
    g,seed=queue.pop(0);key=f'{"preflight" if smoke else "train"}-{g}-s{seed}';f=(R/f'{key}.log').open('a')
    p=subprocess.Popen([PY,'-u',str(__file__),g,str(seed),str(gpu),str(int(smoke))],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    running[gpu]=(p,key,f);status['jobs'][key]=dict(pid=p.pid,gpu=gpu,started=time.time())
   status['queued']=len(queue);save();time.sleep(5)
  if errors:raise RuntimeError('Failed jobs: '+','.join(errors))
 with (R/'report.log').open('a') as f:subprocess.run([PY,str(R/'report.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
 assert json.loads((OUT/'summary.json').read_text())['new_evaluations']==12
 status.update(state='complete',finished=time.time())
except BaseException:
 status.update(state='stopped' if stopping else 'failed',error=traceback.format_exc());raise
finally:
 for p,k,f in running.values():
  if p.poll() is None:
   with contextlib.suppress(ProcessLookupError):os.killpg(p.pid,signal.SIGTERM)
 deadline=time.monotonic()+60
 while any(p.poll() is None for p,_,_ in running.values()) and time.monotonic()<deadline:time.sleep(1)
 save()
