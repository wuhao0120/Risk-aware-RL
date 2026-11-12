"""Exercise the three algorithms' actual training paths on a late-cost trajectory."""
import json
from collections import deque
from types import SimpleNamespace

import torch
from objective import REWARD_GAMMA, episode_total
import base_runner as base
base.setup()
from replay_agent import Agent as ReplayAgent
from agent import complete_reward_suffix, Agent
from qcpo_agent import QCPOUndiscountedAgent
from tqpo_agent import TQPO, chunks, gae
from dca_evaluate import four_episode_returns

def verify():
    c = torch.zeros(1000,12)
    c[-16:] = 1.
    r = c.clone()
    end = torch.zeros_like(c,dtype=torch.bool)
    end[-1] = True
    rollout = dict(R=r,C=c,diagnostic_cost=c,valid=torch.ones_like(end),
                   terminated=torch.zeros_like(end),truncated=end,done=end)
    expected = sum(REWARD_GAMMA ** t for t in range(984,1000))
    target = torch.full((12,),expected,dtype=torch.float64)
    for cls in (ReplayAgent,QCPOUndiscountedAgent):
        obj = SimpleNamespace(gamma=REWARD_GAMMA, report_gamma=.99, cost_discount=1.0)
        out = cls._capture_episode_metrics(obj,dict(rollout),"R")
        torch.testing.assert_close(out["objective_cost"],torch.full_like(target,16.0))
        torch.testing.assert_close(out["objective_return"],target)
    suffix = complete_reward_suffix(rollout,1000,REWARD_GAMMA)
    torch.testing.assert_close(suffix[0].double(),target,rtol=1e-5,atol=1e-7)
    prefix = torch.zeros_like(c,dtype=torch.float64)
    w = 1.0 ** torch.arange(1000,dtype=torch.float64)[:,None]
    prefix[1:] = (w*c).cumsum(0)[:-1]
    atoms = torch.zeros(1000,12,3)
    # With a zero future cost, a prefix already over budget gives risk 1.
    risk = Agent._critic_cdf(SimpleNamespace(return_threshold=15., gamma=REWARD_GAMMA,
                                probability_from_atoms=Agent.probability_from_atoms),
                                atoms,dict(reward_prefix=prefix))
    torch.testing.assert_close(risk[-1],(prefix[-1]>15.).float())
    prev = torch.zeros(1000,12,2)
    rollout.update(values=prev,bootstrap_values=prev[-1])
    for key in ("S","A","mean","ls","h","c"):
        rollout[key] = torch.zeros(1000,12,1)
    stub=SimpleNamespace(current_step_costs=deque(maxlen=100000),
        current_ep_costs=deque(maxlen=100),q_history=deque(maxlen=100),lambda_dual=.5,dual_update_count=0)
    processed = TQPO.process(stub,rollout)
    torch.testing.assert_close(torch.tensor(list(stub.current_ep_costs),dtype=torch.float64),torch.full_like(target,16.0))
    shaped = chunks(c)
    ref_adv, _ = gae(shaped,torch.zeros_like(shaped),chunks(end),
                      torch.zeros_like(shaped[-1]),gamma=REWARD_GAMMA)
    torch.testing.assert_close(processed["r_adv"],ref_adv)
    cost_adv, _ = gae(shaped,torch.zeros_like(shaped),chunks(end),torch.zeros_like(shaped[-1]),gamma=1.0)
    torch.testing.assert_close(processed["c_return"],cost_adv/10)
    raw=four_episode_returns(r.T,c.T)["undiscounted_cost"]
    assert (raw==16.).all() and (raw>15.).all()
    assert (expected>15.) == (REWARD_GAMMA==1.)
    print(json.dumps(dict(train_gamma=REWARD_GAMMA,training_reward_total=expected,episode_cost_for_multiplier=16.,
                          train_violation=True,eval_cost=16.,eval_violation=True)))
if __name__=="__main__":
    verify()

