import sys,json
from pathlib import Path
import torch
r=Path(__file__).resolve().parent;key=sys.argv[1];sys.path.insert(0,str(r/key/'source'))
import base_runner;base_runner.setup()
if key.startswith('dca'):
 from agent import actor_labels_for_arm
 from run import make_args
 divisor=int(key.removeprefix('dca_div'));args=make_args(71,divisor,r/key/'preflight',device='cpu',arm=f'schemeb_raw_mean{divisor}_lambda_norm')
 assert (args.num_envs,args.training_gamma,args.failure_alpha,args.cost_threshold,args.env_name)==(25,1,.2,15,'Dynamic')
 reward=torch.tensor([[1.,2.],[3.,4.]]);risk=torch.tensor([[.1,-.2],[.3,-.4]]);valid=torch.ones_like(reward,dtype=torch.bool);lam=torch.tensor(.7)
 a=actor_labels_for_arm(reward,risk,lam,valid,f'schemeb_raw_mean{divisor}_lambda_norm');b=actor_labels_for_arm(reward,risk,lam,valid,'schemeb_raw_mean100_lambda_norm')
 torch.testing.assert_close(a['raw_mean_term'],b['raw_mean_term']*100/divisor);torch.testing.assert_close(a['raw_risk_term'],b['raw_risk_term']);torch.testing.assert_close(a['combined_normalization_scale'],b['combined_normalization_scale'])
else:
 from tqpo_agent import PROTOCOL,chunks
 assert (PROTOCOL['num_envs'],PROTOCOL['batch_size'],PROTOCOL['recurrent_batch'],PROTOCOL['gamma'],PROTOCOL['cost_gamma'],PROTOCOL['dual_gamma'],PROTOCOL['failure_alpha'])==(12,12000,120,1,1,1,.2)
 assert PROTOCOL['cdf_guard']=='len_ge_100_after_history_warmup'
 x=torch.arange(12000).reshape(1000,12,1);y=chunks(x);assert y.shape==(100,120,1) and sorted(y.flatten().tolist())==list(range(12000))
print(key,'PASS')
