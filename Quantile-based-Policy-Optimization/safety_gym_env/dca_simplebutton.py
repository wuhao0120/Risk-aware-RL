"""SimpleButton DCA (paper Appendix E): maximize raw return at Pr(sum(cost)>15)<=0.20.

Run this file from safety_gym_env. The learner uses the actual Safety-Gymnasium
SimpleButton physics on CPU. Only critic fitting and PPO need the learner device.
Each worker collects complete independent episodes under a frozen policy.
"""

import argparse
import copy
import json
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

HORIZON = 1000
BUDGET = 15.0
ALPHA = 0.20


def mlp(inputs, width, outputs):
    return nn.Sequential(
        nn.Linear(inputs, width),
        nn.Tanh(),
        nn.Linear(width, width),
        nn.Tanh(),
        nn.Linear(width, outputs),
    )


class Policy(nn.Module):
    """Inputs are the physical observation, remaining budget / 15 and time remaining."""

    def __init__(self, dim, actions, width, normalize_obs=False):
        super().__init__()
        self.use_normalization = normalize_obs
        if normalize_obs:
            self.register_buffer("obs_mean", torch.zeros(dim - 2))
            self.register_buffer("obs_second", torch.ones(dim - 2))
            self.register_buffer("obs_count", torch.tensor(1.0))
        self.encoder = mlp(dim, width, width)
        self.memory = nn.LSTM(width, width, batch_first=True)
        self.mean = nn.Linear(width, actions)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)
        self.log_std = nn.Parameter(torch.full((actions,), -0.5))

    @torch.no_grad()
    def observe(self, physical):
        if self.use_normalization:
            weight = len(physical) / (self.obs_count + len(physical))
            self.obs_mean.lerp_(physical.mean(0), weight)
            self.obs_second.lerp_(physical.square().mean(0), weight)
            self.obs_count.add_(len(physical))

    def normalized(self, states):
        if not self.use_normalization:
            return states
        std = (self.obs_second - self.obs_mean.square()).clamp_min(0.0025).sqrt()
        obs = ((states[..., :-2] - self.obs_mean) / std).clamp(-10, 10)
        return torch.cat((obs, states[..., -2:]), -1)

    def forward(self, states, memory=None):
        hidden, memory = self.memory(self.encoder(self.normalized(states)), memory)
        return self.mean(hidden), hidden, memory

    def distribution(self, mean):
        return Normal(mean, self.log_std.clamp(-3, 0.5).exp())


def start_worker(actor):
    global worker_env, worker_actor
    from envs.safety_env import SafetyEnv

    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    worker_env = SafetyEnv("SimpleButton")
    worker_actor = actor.eval()


@torch.inference_mode()
def episode(request):
    seed, metrics_only = request
    obs, _ = worker_env.reset(seed=int(seed))
    torch.manual_seed(int(seed) + 1_000_000_000)
    memory = None
    budget = BUDGET
    goals = 0
    states, latents, rewards, costs = [], [], [], []
    for step in range(HORIZON):
        x = np.concatenate((obs, [budget / BUDGET, (HORIZON - step) / HORIZON])).astype(
            np.float32
        )
        mu, _, memory = worker_actor(torch.from_numpy(x)[None, None], memory)
        latent = worker_actor.distribution(mu[0, 0]).sample().numpy()
        obs, reward, cost, term, trunc, info = worker_env.step(latent.clip(-1, 1))
        goals += int(info.get("goal_met", False))
        if not metrics_only:
            states.append(x)
            latents.append(latent)
        rewards.append(reward)
        costs.append(cost)
        budget -= cost
        if trunc and not term and step + 1 != HORIZON:
            raise RuntimeError("Unexpected partial trajectory before the task horizon")
        if term or trunc:
            break
    if metrics_only:
        return float(sum(rewards)), float(sum(costs)), goals, len(rewards)
    return tuple(
        np.asarray(a, dtype=np.float32) for a in (states, latents, rewards, costs)
    )


def upper_tail(atoms, budget):
    """Strict exceedance 1-F(b); ties consume all atoms at b (right-continuous CDF)."""
    z = atoms.sort(dim=-1).values.contiguous()
    n = z.shape[-1]
    b = budget.unsqueeze(-1).contiguous()
    right = torch.searchsorted(z, b, right=True)
    lo = (right - 1).clamp(0, n - 1)
    hi = right.clamp(0, n - 1)
    left, span = z.gather(-1, lo), z.gather(-1, hi) - z.gather(-1, lo)
    fraction = ((b - left) / span.clamp_min(1e-8)).clamp(0, 1)
    cdf = (lo + 0.5 + fraction) / n
    cdf = torch.where(right == 0, 0.0, torch.where(right == n, 1.0, cdf))
    # Costs are nonnegative. An exhausted negative budget is already a certain violation.
    return torch.where(b < 0, 1.0, 1 - cdf).squeeze(-1)


def transition_credit(current, successor, terminal, budget_after):
    return (
        torch.where(terminal, (budget_after < 0).to(current.dtype), successor) - current
    )


def pinball_loss(pred, target, levels):
    """Exact pairwise pinball, O(batch*N) memory using sorted target prefix sums."""
    z = target.sort().values.contiguous()
    k = torch.searchsorted(z, pred.detach().contiguous(), right=True)
    prefix = nn.functional.pad(z.cumsum(-1), (1, 0)).gather(-1, k)
    below = pred * k - prefix
    above = z.sum(-1, keepdim=True) - prefix - pred * (z.shape[-1] - k)
    return (((1 - levels) * below + levels * above) / z.shape[-1]).mean()


def pack(episodes, device):
    """Pad only for LSTM evaluation. critic and actor losses mask every padding step."""
    lengths = torch.tensor([len(ep[0]) for ep in episodes], device=device)
    cols = [
        nn.utils.rnn.pad_sequence(
            [torch.from_numpy(ep[i]) for ep in episodes], batch_first=True
        ).to(device)
        for i in range(4)
    ]
    x, latent, reward, cost = cols
    valid = torch.arange(x.shape[1], device=device) < lengths[:, None]
    return x, latent, reward, cost, valid


def training_targets(x, reward, cost, gamma):
    """Per-time reward-to-go and local threshold for the discounted cost tail.

    The policy always observes the physical, undiscounted safety budget. The
    critic predicts a discounted suffix, so query it at the equivalent local
    threshold (d - sum_{j<t} gamma**j c_j) / gamma**t, in units of d.
    """
    if gamma == 1.0:
        return reward.flip([1]).cumsum(1).flip([1]), x[:, :, -2]
    weights = gamma ** torch.arange(reward.shape[1], dtype=torch.float64, device=x.device)
    weighted = reward * weights
    returns = weighted.flip([1]).cumsum(1).flip([1]) / weights
    past = nn.functional.pad((cost * weights)[:, :-1].cumsum(1), (1, 0))
    budget = (1.0 - past / BUDGET) / weights
    return returns.to(reward.dtype), budget.to(cost.dtype)


class Learner:
    def __init__(self, cfg, actor):
        self.cfg = cfg
        self.actor = actor.to(cfg.device)
        dim = actor.encoder[0].in_features + cfg.width
        actions = actor.mean.out_features
        self.risk = mlp(dim + actions, cfg.width, cfg.atoms).to(cfg.device)
        self.target = copy.deepcopy(self.risk).requires_grad_(False)
        self.value = mlp(dim, cfg.width, 1).to(cfg.device)
        self.ao = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
        self.co = torch.optim.Adam(self.risk.parameters(), lr=cfg.critic_lr)
        self.vo = torch.optim.Adam(self.value.parameters(), lr=cfg.critic_lr)
        self.levels = (torch.arange(cfg.atoms, device=cfg.device) + 0.5) / cfg.atoms
        self.multiplier, self.ema = 0.0, 0.0

    @torch.no_grad()
    def prepare(self, episodes):
        x, latent, reward, cost, valid = pack(episodes, self.cfg.device)
        mu, hidden, _ = self.actor(x)
        features = torch.cat((self.actor.normalized(x), hidden), -1)[valid]
        term = valid & ~nn.functional.pad(valid[:, 1:], (0, 1))
        # Rows are concatenated by episode; cur+1 is consulted only on nonterminal rows.
        returns, budget = training_targets(x, reward, cost, self.cfg.gamma)
        return {
            "x": x,
            "valid": valid,
            "features": features,
            "mu": mu[valid],
            "latent": latent[valid],
            "lp": self.actor.distribution(mu[valid]).log_prob(latent[valid]).sum(-1),
            "budget": budget[valid],
            "cost": cost[valid] / BUDGET,
            "term": term[valid],
            "returns": returns[valid],
        }

    def quantiles(self, model, features, action):
        return model(torch.cat((features, action.clamp(-1, 1)), -1))

    def fit(self, fresh, old, steps):
        c = self.cfg
        for k in range(steps):
            samples = (
                [(fresh, c.minibatch)]
                if old is None
                else [(fresh, (c.minibatch + 1) // 2), (old, c.minibatch // 2)]
            )
            queries, targets, vals, returns = [], [], [], []
            for batch, n in samples:
                ix = torch.randint(len(batch["cost"]), (n,), device=c.device)
                with torch.no_grad():
                    y = batch["cost"][ix, None].expand(-1, c.atoms).clone()
                    live = ~batch["term"][ix]
                    j = ix[live] + 1
                    act = self.actor.distribution(batch["mu"][j]).sample()
                    y[live] += c.gamma * self.quantiles(self.target, batch["features"][j], act)
                queries.append(
                    self.quantiles(
                        self.risk, batch["features"][ix], batch["latent"][ix]
                    )
                )
                targets.append(y)
                vals.append(self.value(batch["features"][ix]).squeeze(-1))
                returns.append(batch["returns"][ix])
            loss = pinball_loss(torch.cat(queries), torch.cat(targets), self.levels)
            self.co.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.risk.parameters(), 10)
            self.co.step()
            self.vo.zero_grad()
            nn.functional.mse_loss(torch.cat(vals), torch.cat(returns)).backward()
            nn.utils.clip_grad_norm_(self.value.parameters(), 10)
            self.vo.step()
            if (k + 1) % max(1, math.ceil(steps / 3)) == 0 or k + 1 == steps:
                self.target.load_state_dict(self.risk.state_dict())

    @torch.no_grad()
    def state_risk(self, batch, indices):
        values = []
        for ix in indices.split(1024):
            feat = batch["features"][ix].repeat_interleave(self.cfg.actions, dim=0)
            mu = batch["mu"][ix].repeat_interleave(self.cfg.actions, dim=0)
            budget = batch["budget"][ix].repeat_interleave(self.cfg.actions)
            act = self.actor.distribution(mu).sample()
            atoms = self.quantiles(self.risk, feat, act)
            tail = upper_tail(atoms, budget)
            values.append(tail.reshape(len(ix), self.cfg.actions).mean(1))
        return torch.cat(values)

    def actor_update(self, batch, reward_adv, credit):
        c = self.cfg
        used = self.multiplier
        for _ in range(c.ppo_epochs):
            # Backtrack the entire Adam step (moments included) if it exceeds the KL budget.
            before = copy.deepcopy(self.actor.state_dict())
            optim_before = copy.deepcopy(self.ao.state_dict())
            self.ao.zero_grad()

            def log_ratio():
                mu, _, _ = self.actor(batch["x"])
                dist = self.actor.distribution(mu[batch["valid"]])
                return dist.log_prob(batch["latent"]).sum(-1) - batch["lp"]

            ratio = log_ratio().exp()
            clip = ratio.clamp(0.9, 1.1)
            surrogate = torch.minimum(ratio * reward_adv, clip * reward_adv)
            surrogate -= used * torch.maximum(ratio * credit, clip * credit)
            loss = -surrogate.mean() / (1 + used)
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
            self.ao.step()
            with torch.no_grad():
                delta = log_ratio()
                kl = (delta.exp() - 1 - delta).mean().item()
            if not math.isfinite(kl) or kl > 0.01:
                self.actor.load_state_dict(before)
                self.ao.load_state_dict(optim_before)
                break

    def update(self, episodes, previous, iteration):
        fresh = self.prepare(episodes)
        old = self.prepare(previous) if previous else None
        with torch.no_grad():
            reward_adv = (
                fresh["returns"] - self.value(fresh["features"]).squeeze(-1)
            ) / self.cfg.reward_scale
        self.fit(
            fresh,
            old,
            self.cfg.warmup_steps
            if iteration <= self.cfg.warmup
            else self.cfg.critic_steps,
        )
        ix = torch.arange(len(fresh["cost"]), device=self.cfg.device)
        base = self.state_risk(fresh, ix)
        live = ~fresh["term"]
        next_risk = torch.zeros_like(base)
        if live.any():
            next_risk[live] = self.state_risk(fresh, ix[live] + 1)
        credit = transition_credit(
            base, next_risk, fresh["term"], fresh["budget"] - fresh["cost"]
        )
        if iteration > self.cfg.warmup:
            self.actor_update(fresh, reward_adv, credit)
        self.actor.observe(fresh["x"][fresh["valid"]][:, :-2])
        totals = np.array([float(ep[3].sum()) for ep in episodes])
        violation = float(np.mean(totals > BUDGET))
        error = violation - ALPHA
        ema = 0.8 * self.ema + 0.2 * error
        self.multiplier = float(
            np.clip(
                self.multiplier + self.cfg.ki * error + self.cfg.kp * (ema - self.ema),
                0,
                50,
            )
        )
        self.ema = ema
        return {
            "iteration": iteration,
            "reward": float(np.mean([ep[2].sum() for ep in episodes])),
            "cost": float(totals.mean()),
            "violation": violation,
            "multiplier": self.multiplier,
            "credit_rms": float(credit.square().mean().sqrt()),
        }


def summarize(scores):
    reward, cost, goals, lengths = np.asarray(scores).T
    viol = int((cost > BUDGET).sum())
    n = len(cost)
    z = 1.959964
    p = viol / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {
        "episodes": n,
        "reward": float(reward.mean()),
        "cost": float(cost.mean()),
        "q80": float(np.quantile(cost, 0.8)),
        "violation": p,
        "violations": viol,
        "mean_buttons": float(goals.mean()),
        "mean_episode_length": float(lengths.mean()),
        "violation_ci95": [center - half, center + half],
    }


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--gamma", type=float, choices=(0.99, 1.0), default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--normalize-obs", action="store_true")
    p.add_argument("--eval-checkpoint", type=Path)
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--eval-seed", type=int, default=90000000)
    for key, default in {
        "iterations": 400,
        "batch": 25,
        "workers": 12,
        "width": 512,
        "atoms": 128,
        "actions": 4,
        "warmup": 20,
        "warmup_steps": 100,
        "critic_steps": 400,
        "minibatch": 1024,
        "ppo_epochs": 8,
        "replay_episodes": 100,
        "seed": 0,
        "eval_episodes": 256,
    }.items():
        p.add_argument("--" + key.replace("_", "-"), type=int, default=default)
    for key, default in {
        "actor_lr": 0.0002,
        "critic_lr": 0.0003,
        "ki": 0.2,
        "kp": 5.0,
        "reward_scale": 100.0,
    }.items():
        p.add_argument("--" + key.replace("_", "-"), type=float, default=default)
    return p


def main():
    if "MATCHED_CPUS" in os.environ:
        from matched_campaign import honor_campaign_affinity
        honor_campaign_affinity()
    c = parser().parse_args()
    if (
        min(
            c.batch,
            c.workers,
            c.width,
            c.actions,
            c.atoms,
            c.eval_episodes,
            c.minibatch,
        )
        < 1
    ):
        raise ValueError("positive sizes required")
    if c.replay_episodes < c.batch or c.atoms < 2:
        raise ValueError("replay must hold the fresh batch; use at least two atoms")
    os.environ.setdefault("MUJOCO_GL", "disable")
    torch.set_num_threads(1)
    torch.manual_seed(c.seed)
    from envs.safety_env import SafetyEnv

    probe = SafetyEnv("SimpleButton")
    actor = Policy(probe.obs_dim + 2, probe.act_dim, c.width, c.normalize_obs)
    probe.close()
    if c.eval_checkpoint:
        ckpt = torch.load(c.eval_checkpoint, map_location="cpu", weights_only=False)
        actor = Policy(
            ckpt["dim"],
            ckpt["actions"],
            ckpt["width"],
            ckpt.get("normalize_obs", False),
        )
        actor.load_state_dict(ckpt["actor"])
    shared = copy.deepcopy(actor).share_memory()
    c.out.mkdir(parents=True, exist_ok=True)
    with (c.out / "config.json").open("x") as f:
        json.dump(vars(c), f, default=str, indent=2)
    with ProcessPoolExecutor(
        c.workers,
        mp_context=mp.get_context("spawn"),
        initializer=start_worker,
        initargs=(shared,),
    ) as pool:

        def evaluate(name):
            requests = [(c.eval_seed + j, True) for j in range(c.eval_episodes)]
            scores = list(pool.map(episode, requests))
            result = summarize(scores)
            np.save(c.out / (name + "_episodes.npy"), scores)
            (c.out / (name + ".json")).write_text(json.dumps(result, indent=2))
            print(name, result, flush=True)

        if not c.eval_checkpoint:
            learner = Learner(c, actor)
            previous = []
            interactions = 0
            with (c.out / "train.jsonl").open("x") as log:
                for i in range(1, c.iterations + 1):
                    requests = [
                        (100000 * c.seed + (i - 1) * c.batch + j, False)
                        for j in range(c.batch)
                    ]
                    episodes = list(pool.map(episode, requests))
                    interactions += sum(len(ep[0]) for ep in episodes)
                    metrics = learner.update(episodes, previous, i)
                    metrics["interactions"] = interactions
                    shared.load_state_dict(learner.actor.state_dict())
                    size = c.replay_episodes - c.batch
                    previous = (previous + episodes)[-size:] if size else []
                    log.write(json.dumps(metrics) + "\n")
                    log.flush()
                    if i == 1 or i % 10 == 0 or i == c.iterations:
                        print(metrics, flush=True)
                        save = {
                            "actor": shared.state_dict(),
                            "width": c.width,
                            "normalize_obs": c.normalize_obs,
                            "actions": actor.mean.out_features,
                            "dim": actor.encoder[0].in_features,
                            "iteration": i,
                        }
                        torch.save(save, c.out / "policy.tmp")
                        (c.out / "policy.tmp").replace(c.out / "policy.pt")
        if not c.train_only:
            evaluate("final")


if __name__ == "__main__":
    main()
