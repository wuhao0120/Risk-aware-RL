"""Local deterministic setup, serialization and atomic file writes."""
import os, sys, json, random
from pathlib import Path
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source'
PYTHON = sys.executable

def setup():
    os.environ['MUJOCO_GL']='disable'
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','RISK_RL_TORCH_NUM_THREADS'):
        os.environ[key]='1'
    sys.path.insert(0,str(SOURCE))

def atomic_json(path, obj):
    """原子替换 JSON，读监控时不会看到半写文件。"""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def serial_config(args):
    return {k:v if isinstance(v,(int,float,bool,str,list,type(None))) else str(v) for k,v in vars(args).items()}
