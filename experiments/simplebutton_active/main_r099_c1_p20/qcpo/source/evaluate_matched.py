"""Score the predeclared final checkpoint on 2,048 held-out SimpleButton episodes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from base_runner import atomic_json
from objective import COST_GAMMA, REWARD_GAMMA

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("dca", "qcpo", "tqpo"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    dest = root / "evaluation"
    ckpt = root / "final.pt"
    config = json.loads((root / "config.json").read_text())
    receipt = json.loads((root / "training_receipt.json").read_text())
    steps = 10_008_000 if args.method == "tqpo" else 10_000_000
    cfg = config["configuration"]
    if receipt["actual_env_steps"] != steps or cfg["gamma"] != REWARD_GAMMA or cfg["cost_gamma"] != COST_GAMMA:
        raise ValueError("Unexpected training endpoint or target discount")
    from dca_evaluate import sha
    torch.set_num_threads(1)
    torch.manual_seed(92_000_000)
    episodes, seed = 2048, 92_000_000
    if args.method == "dca":
        from dca_evaluate import load_dca, sample_checkpoint
        actor, critic, cfg, state = load_dca(root, ckpt, torch.device("cuda:0"))
        sample_checkpoint(actor, critic, cfg, dest, episodes, seed, calibration=False)
    elif args.method == "tqpo":
        from run_tqpo import load_evaluation
        from tqpo_evaluate import sample_checkpoint
        actor, cfg, state = load_evaluation(root, ckpt)
        sample_checkpoint(actor, None, cfg, dest, episodes, seed, tqpo=True)
    else:
        from qcpo_evaluate import load, policy_rollouts
        actor, cfg, state = load(root, ckpt)
        policy_rollouts(actor, cfg, dest, episodes, seed)
    if state["actual_env_steps"] != steps:
        raise ValueError("Evaluation did not load the final checkpoint")
    sample_file = dest / "samples.npz"
    with np.load(sample_file) as samples:
        reward = samples["undiscounted_reward"]
        cost = samples["undiscounted_cost"]
        if not all(a.shape == (episodes,) and np.isfinite(a).all() for a in (reward,cost)):
            raise ValueError("Incomplete/nonfinite held-out episodes")
        prob = float(np.mean(cost > 15.0))
        atomic_json(root / "final_eval.json", dict(method=args.method, train_gamma=REWARD_GAMMA, reward_gamma=REWARD_GAMMA, cost_gamma=COST_GAMMA, multiplier_gamma=COST_GAMMA,
            eval_gamma=1., seed=config["signature"]["seed"], actual_env_steps=steps,
            heldout_base_seed=seed, episodes=episodes, checkpoint_sha256=sha(ckpt),
            samples_sha256=sha(sample_file), return_mean=float(reward.mean()),
            cost_mean=float(cost.mean()), cost_q80=float(np.quantile(cost,.8)),
            violations=int(np.count_nonzero(cost>15)), violation_rate=prob, feasible=prob<=.2))
if __name__ == "__main__":
    main()

