#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train/evaluate Safety Gym reward-side risk-aware algorithms with frozen semantics.

The runner enforces the QPO-first research gate, explicit per-environment q for constrained
algorithms, disjoint validation/test seeds, exact-N evaluation, and atomic rollout-boundary
checkpoints. It never derives a formal q from QPO output and never starts the main phase implicitly.
"""
import argparse
import json
import math
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent
PAPER_ENVS = {
    "SimpleButton": "SimpleButton",
    "Dynamic": "Dynamic",
    "Gremlin": "Gremlin",
    "DynamicButton": "DynamicButton",
}


def cast(value):
    """Parse --set values as bool/None/int/float/string in that order."""
    lowered = str(value).lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


# These values define the meaning and shape of a resumable training state. Operational knobs such
# as output directory, W&B mode, validation/checkpoint frequency, final-test N, device, and total
# iterations are intentionally excluded so logging can change and a run can be extended safely.
CHECKPOINT_SIGNATURE_FIELDS = (
    "env_id", "seed", "gamma", "failure_alpha", "empirical_quantile_method",
    "return_threshold", "horizon", "num_envs",
    "training_budget_semantics",
    "actor_hidden", "lstm_size", "init_std", "learn_std", "normalize_observation",
    "actor_lr0", "theta_a", "theta_b", "theta_c",
    "lambda_a", "lambda_b", "lambda_c", "lambda_max",
    "warmup_q_iters", "q_a", "q_b", "q_c",
    "ref_lr", "critic_lr", "critic_hidden", "epochs", "ratio_clip", "gae_lambda",
    "value_loss_coeff", "quantile_loss_coeff", "weibull_loss_coeff", "entropy_loss_coeff",
    "qcpo_ref_formulation", "qcpo_ref_actor_weighting", "qcpo_ref_actor_reduction",
    "qcpo_ref_critic_input",
    "density_log_clip", "weibull_fit_min", "dual_window_size",
    "clip_grad_norm", "beta", "num_quantiles", "num_action_samples",
    "dqc_update_mode", "critic_target_source", "critic_minibatch_size", "critic_epochs",
    "actor_epochs", "dqc_actor_reuse_mode", "dqc_ratio_clip", "dqc_target_kl",
    "dqc_mean_estimator", "dqc_gae_lambda", "dqc_value_hidden",
    "dqc_value_lr", "dqc_value_epochs",
    "horizon_mode", "critic_n_step", "critic_time_feature", "critic_actor_context",
    "target_update_mode",
    "warmup_iters", "critic_warmup_mode", "warmup_critic_epochs",
    "huber_kappa", "target_tau", "actor_lr", "critic_grad_clip",
    "normalize_combined_advantage", "actor_grad_clip", "penalty_init",
)


def checkpoint_signature(args, algorithm):
    """Build a JSON-safe immutable experiment signature after all CLI overrides are applied."""
    signature = {"algorithm": str(algorithm)}
    for field in CHECKPOINT_SIGNATURE_FIELDS:
        if not hasattr(args, field):
            continue
        value = getattr(args, field)
        # Canonicalize architecture lists so a tuple/default and comma-separated CLI spelling
        # compare by meaning rather than their surface Python type.
        if field.endswith("_hidden") and isinstance(value, str):
            value = [int(item) for item in value.split(",") if item.strip()]
        elif isinstance(value, tuple):
            value = list(value)
        signature[field] = value
    return signature


def scheduled_event_due(completed, interval, last_iteration, force=False):
    """Return whether one enabled periodic event is due; interval=0 remains disabled under force."""
    interval = int(interval)
    current_iteration = int(completed) - 1
    return interval > 0 and (
        int(completed) % interval == 0
        or (bool(force) and int(last_iteration) != current_iteration)
    )


def base_args(algo, seed, device, env_key):
    """Build first-round defaults: gamma=.99, T=1000, alpha=.2, shared FC512x2+LSTM512."""
    args = SimpleNamespace()
    args.env_name = env_key
    args.env_id = PAPER_ENVS[env_key]
    args.seed = int(seed)
    args.algo_name = algo
    args.device = device
    args.gamma = 0.99
    args.training_budget_semantics = "main_excludes_initialization_v1"
    args.failure_alpha = 0.2
    args.empirical_quantile_method = "linear"
    args.q_alpha = args.failure_alpha               # compatibility only
    args.return_threshold = None                    # user must choose after QPO stage
    args.horizon = 1000
    args.num_envs = 25
    args.num_iterations = 200                       # 25*1000*200 = 5M steps
    args.vec_backend = "mp"
    args.actor_hidden = [512, 512]
    args.lstm_size = 512
    args.init_std = 1.0
    args.learn_std = True
    args.normalize_observation = True
    # Every completed rollout emits one durable key-metric line. This frequency is shared by all
    # algorithms and matches the one-W&B-row-per-batch contract.
    args.log_interval = 1
    args.show_progress = True
    args.show_eval_progress = True
    args.progress_mininterval = 1.0
    args.progress_file = None
    args.wandb_dir = str(BASE_DIR)
    args.wandb_project = "safety_gym_reward_risk"
    args.wandb_tags = "reward-side,G1000,alpha-0.2"
    args.normalize_combined_advantage = True
    args.actor_grad_clip = 10.0
    args.penalty_init = 0.0
    args.lambda_max = 50.0

    # ``actor_lr0`` is the controlled cross-algorithm variable for this experiment round. Direct
    # score-function methods consume it through theta_a; PPO-style methods use the same initial
    # scalar under their formulation-specific optimizer name.
    args.actor_lr0 = 2e-4
    args.theta_b, args.theta_c = 10000.0, 0.9
    args.theta_a = (args.theta_b ** args.theta_c) * args.actor_lr0
    # Momentum-free scalar multiplier schedule.
    args.lambda_b, args.lambda_c = 1000.0, 0.5
    args.lambda_a = (args.lambda_b ** args.lambda_c) * 1e-2

    if algo == "QPO":
        # Q tracker is faster than theta; it remains a pure quantile optimizer.
        args.warmup_q_iters = 2
        args.q_b, args.q_c = 1000.0, 0.6
        args.q_a = (args.q_b ** args.q_c) * 0.1
    elif algo == "QCPO":
        # QCPO now inherits the same actor_lr0 as QPO; LOO/normalization remain algorithm features.
        pass
    elif algo == "QCPO_REF":
        # Preserve the external reward-outage task while using the paper derivation verbatim on
        # C=-G. These fields are checkpoint-signed so pre-fix reward-left-tail runs cannot resume.
        args.qcpo_ref_formulation = "paper_negated_return_eq275_discounted_bh_v2"
        args.qcpo_ref_actor_weighting = "discounted_gamma_t"
        args.qcpo_ref_actor_reduction = "fixed_BH"
        # Align the REF state-value/distribution heads with DQC finite's history information while
        # keeping the REF critic state-only and mathematically unchanged.
        args.qcpo_ref_critic_input = "finite_obs_remaining_actor_h0c0_v1"
        args.horizon_mode = "finite"
        args.critic_time_feature = "remaining_fraction"
        args.critic_actor_context = True
        args.density_log_clip = 0.5
        args.weibull_fit_min = 0.75
        args.dual_window_size = 100
        args.ref_lr = args.actor_lr0
        args.critic_lr = 3e-4
        args.critic_hidden = [512, 512]
        args.epochs = 8
        args.ratio_clip = 0.1
        args.gae_lambda = 0.97
        args.value_loss_coeff = 1.0
        args.quantile_loss_coeff = 1.0
        args.weibull_loss_coeff = 1.0
        args.entropy_loss_coeff = 0.0
        args.clip_grad_norm = 10.0
        args.lambda_max = 100.0
    elif algo in ("DQCAC", "DQCACBeta"):
        # Clean DQC gate: fresh rollout only. Defaults retain one QR pass and one score update;
        # explicit epoch overrides are checkpoint-signed sample-reuse ablations.
        args.dqc_update_mode = "online_1step_v1"
        args.horizon_mode = "continuing"
        args.critic_n_step = 1
        args.critic_time_feature = "none"
        args.critic_actor_context = False
        args.target_update_mode = "polyak"
        # Default behavior stays one-step TD; finite_suffix_mc is an explicit diagnostic branch.
        args.critic_target_source = "one_step"
        args.beta = 0.995
        args.num_quantiles = 32
        args.num_action_samples = 4
        args.critic_minibatch_size = 1024
        args.critic_epochs = 1
        args.actor_epochs = 1
        # The historical DQC identity remains one raw score-function pass. Multi-epoch PPO reuse
        # is an explicit, checkpoint-signed ablation rather than a silent change to that default.
        args.dqc_actor_reuse_mode = "score_function"
        args.dqc_ratio_clip = 0.1
        args.dqc_target_kl = 0.0
        args.dqc_mean_estimator = "distributional_action_advantage_v1"
        args.dqc_gae_lambda = 0.97
        args.dqc_value_hidden = [512, 512]
        args.dqc_value_lr = 3e-4
        args.dqc_value_epochs = 8
        args.warmup_iters = 5
        # ``legacy`` preserves the online_1step_v1 default exactly. The explicit fitted branch
        # remains one-step TD: freeze policy state, fit one frozen Bellman generation for several
        # regression epochs, then hard-copy the online critic into the target.
        args.critic_warmup_mode = "legacy"
        args.warmup_critic_epochs = 1
        args.huber_kappa = 0.1
        args.target_tau = 0.005
        args.critic_hidden = [512, 512]
        args.critic_lr = 3e-4
        args.actor_lr = args.actor_lr0
        args.critic_grad_clip = 10.0
        # Keep W&B config and checkpoint signature truthful: the online actor consumes raw,
        # detached mean/CDF advantages and never invokes the shared normalization helper.
        args.normalize_combined_advantage = False
        # The first and only formal gate has an explicitly user-selected raw-return q.
        if env_key == "SimpleButton":
            args.return_threshold = -0.03
        args.wandb_tags += ",dqc_online_1step_mlp_v1"
    return args

def configure_dqc_identity(args, overrides=None):
    """
    Expand the non-default finite_nstep_v1 preset after ordinary CLI overrides.

    A caller only needs to request dqc_update_mode=finite_nstep_v1. Fields it did not explicitly
    set receive the fixed experiment values below; explicitly conflicting values remain untouched
    so DQCACBetaGPU can reject the partial identity before allocating critic state.
    """
    if str(getattr(args, "dqc_update_mode", "")) != "finite_nstep_v1":
        return args
    explicit = set((overrides or {}).keys())
    fixed = {
        "horizon_mode": "finite",
        "critic_target_source": "n_step",
        "critic_n_step": 100,
        "critic_time_feature": "remaining_fraction",
        "critic_actor_context": True,
        "target_update_mode": "hard",
        "beta": 1.0,
        "warmup_iters": 10,
        "critic_warmup_mode": "fitted_hard",
        "warmup_critic_epochs": 4,
    }
    for name, value in fixed.items():
        if name not in explicit:
            setattr(args, name, value)

    # GAE-Mean is orthogonal to finite n-step QR: scalar GAE supplies only expected-return credit,
    # while the unchanged action-conditioned atom critic supplies only the fixed-q CDF term.
    gae_mean = (
        str(getattr(args, "dqc_mean_estimator", ""))
        == "state_value_gae_v1"
    )
    if gae_mean:
        gae_fixed = {
            "dqc_gae_lambda": 0.97,
            "dqc_value_hidden": [512, 512],
            "dqc_value_lr": 3e-4,
            "dqc_value_epochs": 8,
            "normalize_combined_advantage": True,
            "critic_epochs": 8,
            "actor_epochs": 8,
            "dqc_actor_reuse_mode": "ppo_clip",
            "dqc_ratio_clip": 0.1,
            "dqc_target_kl": 0.01,
        }
        for name, value in gae_fixed.items():
            if name not in explicit:
                setattr(args, name, value)

    # Keep online tracking and on-disk output visibly separate from all continuing/MC histories.
    variant_prefix = "n100_gae_mean" if gae_mean else "n100_time_context_beta1"
    variant = f"{variant_prefix}_s{int(args.seed)}"
    args.experiment_variant = variant
    if "wandb_group" not in explicit:
        args.wandb_group = (
            "dqc_gae_mean_simplebutton_v1" if gae_mean
            else "dqc_finite_horizon_v1"
        )
    if "wandb_name" not in explicit:
        args.wandb_name = variant
    identity_tag = (
        "dqc_finite_n100_gae_mean_v1" if gae_mean
        else "dqc_finite_horizon_v1"
    )
    args.wandb_tags = str(args.wandb_tags).replace(
        "dqc_online_1step_mlp_v1", identity_tag)
    return args


@contextmanager
def preserve_rng():
    """Prevent validation action sampling from changing the subsequent training RNG stream."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def calibrate_random_policy(args, num_episodes):
    """Report random-policy raw reward distribution only; it never chooses formal q automatically."""
    from envs import SafetyEnv, make_vec_env
    from utils.risk_metrics import return_statistics
    env = SafetyEnv(args.env_id)
    vec = make_vec_env(args.env_id, args.num_envs, args.horizon, args.device,
                       ref_env=env, seed=args.seed + 100, backend=args.vec_backend)
    rounds = int(math.ceil(int(num_episodes) / vec.B))
    values = []
    with torch.no_grad():
        for _ in range(rounds):
            state = vec.reset()
            active = torch.ones(vec.B, dtype=torch.bool, device=args.device)
            returns = torch.zeros(vec.B, device=args.device)
            for t in range(vec.n):
                action = torch.randn(vec.B, vec.act_dim, device=args.device)
                step = vec.step(action)
                returns += (args.gamma ** t) * step.reward * active.float()
                active &= ~step.episode_end
                state = step.observation
                if not bool(active.any()):
                    break
            values.append(returns)
    sample = torch.cat(values)[:int(num_episodes)].cpu().numpy()
    result = return_statistics(sample, args.failure_alpha)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    vec.close()
    env.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="QPO",
                        choices=["QPO", "QCPO", "QCPO_REF", "DQCAC", "DQCACBeta", "CALIB"])
    parser.add_argument("--env", default="SimpleButton", choices=list(PAPER_ENVS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="0", help="CUDA index or cpu")
    parser.add_argument("--return_threshold", type=float, default=None,
                        help="fixed raw G_1000 threshold q; required except QPO/CALIB")
    parser.add_argument("--num_eval", type=int, default=1000,
                        help="exact number of final stochastic test episodes")
    parser.add_argument("--validation_episodes", type=int, default=100)
    parser.add_argument("--validation_every", type=int, default=10,
                        help="rollout batches between validation evaluations; 0 disables")
    parser.add_argument("--checkpoint_every", type=int, default=50,
                        help="rollout batches between rolling latest.pt saves; 0 disables")
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--allow_checkpoint_transfer", action="store_true",
        help=("explicitly allow a legacy/mismatched experiment signature when resuming; "
              "agent class must still match"))
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--wandb_mode", default=None, choices=[None, "online", "offline", "disabled"])
    parser.add_argument("--tag", default="run")
    parser.add_argument("--set", nargs="*", default=[], metavar="K=V")
    cli = parser.parse_args()

    if cli.wandb_mode:
        os.environ["WANDB_MODE"] = cli.wandb_mode
    device = torch.device("cpu") if cli.device == "cpu" else torch.device(
        f"cuda:{cli.device}" if torch.cuda.is_available() else "cpu")
    config_algo = "DQCAC" if cli.algo == "CALIB" else cli.algo
    args = base_args(config_algo, cli.seed, device, cli.env)

    overrides = {}
    for assignment in cli.set:
        if "=" not in assignment:
            parser.error(f"invalid --set value {assignment!r}; expected K=V")
        key, value = assignment.split("=", 1)
        overrides[key] = cast(value)
        setattr(args, key, overrides[key])
    if cli.return_threshold is not None:
        args.return_threshold = float(cli.return_threshold)
    if "theta_lr0" in overrides:
        # Backward-compatible QPO/QCPO-only spelling used by earlier launch scripts. Keep the
        # canonical initial-LR field synchronized so W&B config and checkpoint signature describe
        # the same schedule that theta_a actually generates.
        theta_lr0 = float(overrides["theta_lr0"])
        args.theta_a = (args.theta_b ** args.theta_c) * theta_lr0
        if cli.algo in ("QPO", "QCPO"):
            args.actor_lr0 = theta_lr0
    if "actor_lr0" in overrides:
        # Canonical controlled variable: propagate one requested initial actor LR to every
        # formulation-specific optimizer field before the agent and W&B config are constructed.
        actor_lr0 = float(overrides["actor_lr0"])
        args.actor_lr0 = actor_lr0
        args.theta_a = (args.theta_b ** args.theta_c) * actor_lr0
        args.ref_lr = actor_lr0
        args.actor_lr = actor_lr0
    if cli.algo in ("DQCAC", "DQCACBeta"):
        configure_dqc_identity(args, overrides)
        if args.dqc_update_mode == "finite_nstep_v1" and cli.tag == "run":
            cli.tag = args.experiment_variant
    experiment_label = str(getattr(args, "experiment_label", "") or "").strip()
    if experiment_label:
        args.wandb_tags = f"{args.wandb_tags},{experiment_label}"

    if cli.algo not in ("QPO", "CALIB") and args.return_threshold is None:
        parser.error(f"--algo {cli.algo} requires explicit --return_threshold q")
    if cli.eval_only and cli.resume is None:
        parser.error("--eval_only requires --resume CHECKPOINT")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if cli.algo == "CALIB":
        os.environ.setdefault("WANDB_MODE", "disabled")
        calibrate_random_policy(args, cli.num_eval)
        return

    from agents import QPOGPU, QCPOGPU, QCPORefGPU, DQCACBetaGPU
    from envs import SafetyEnv, make_vec_env
    from utils import (evaluate_policy_vec, save_checkpoint, load_checkpoint,
                       select_validation_candidate, validate_checkpoint_payload)

    classes = {
        "QPO": QPOGPU, "QCPO": QCPOGPU, "QCPO_REF": QCPORefGPU,
        "DQCAC": DQCACBetaGPU, "DQCACBeta": DQCACBetaGPU,
    }
    checkpoint_dir = (
        Path(cli.checkpoint_dir)
        if cli.checkpoint_dir
        else BASE_DIR / "_runs" / "checkpoints" / f"{cli.algo}_{cli.env}_{cli.tag}_s{cli.seed}"
    )
    latest_path = checkpoint_dir / "latest.pt"
    best_path = checkpoint_dir / "best.pt"
    initialization_path = checkpoint_dir / "initialization_latest.pt"
    validation_history = []
    training_diagnostics_history = []
    candidate_sources = {}
    selected_best_path = None
    last_validation_iteration = -1
    last_checkpoint_iteration = -1

    def remember_candidate(metrics, source_path=None):
        """Keep metrics and the checkpoint file that can recreate their policy, if available."""
        if not metrics:
            return
        validation_history.append(metrics)
        if source_path is not None:
            candidate_sources[id(metrics)] = Path(source_path)

    env = SafetyEnv(args.env_id)
    agent = classes[cli.algo](args, env)
    # The helper reads this field on every save/load. Setting it before resume guarantees that a
    # wrong checkpoint is rejected before any actor, optimizer, replay, or RNG state is mutated.
    agent.checkpoint_signature = checkpoint_signature(args, cli.algo)
    if cli.resume:
        payload = load_checkpoint(
            cli.resume, agent, map_location=device,
            allow_signature_mismatch=cli.allow_checkpoint_transfer)
        # Initialization checkpoints use iteration=-1, so their formal loop always resumes at 0.
        agent.start_iteration = max(0, int(payload["iteration"]) + 1)
        last_checkpoint_iteration = int(payload["iteration"])
        remember_candidate(payload.get("metrics"), cli.resume)
        # A prior best may differ from latest.pt. Remember its source instead of copying it into
        # every resume lineage; the final selector can load the original atomic checkpoint.
        prior_best_path = Path(cli.resume).resolve().parent / "best.pt"
        if (int(payload["iteration"]) >= 0 and prior_best_path.exists()
                and prior_best_path.resolve() != Path(cli.resume).resolve()):
            prior_best = torch.load(prior_best_path, map_location="cpu", weights_only=False)
            validate_checkpoint_payload(
                prior_best, agent,
                allow_signature_mismatch=cli.allow_checkpoint_transfer)
            remember_candidate(prior_best.get("metrics"), prior_best_path)
        if validation_history:
            prior_winner = select_validation_candidate(
                validation_history, args.failure_alpha, qpo=(cli.algo == "QPO"))
            selected_best_path = candidate_sources.get(id(prior_winner), Path(cli.resume))
            last_validation_iteration = max(
                int(item.get("iteration", -1)) for item in validation_history)
        print(f"resumed {cli.resume} at iteration {agent.start_iteration}")

    def validation_wandb_metrics(metrics, current_agent):
        """Map one exact-N evaluation to a sparse validation/* row merged with training W&B."""
        log = {
            "validation/disc_reward/aver_reward": metrics["mean"],
            "validation/disc_reward/return_std": metrics["reward_std"],
            "validation/disc_reward/quantile_reward": metrics["quantile_return"],
            # These two fields are the public comparison contract for every algorithm. QPO's
            # evaluator values refer to its persistent tracker; all other methods use the exact
            # N=100 validation percentile and their inclusive fixed-q failure probability.
            "validation/quantile/q_est": metrics["quantile_q_est"],
            "validation/quantile/constraint_prob": metrics["quantile_constraint_prob"],
            "validation/num_episodes": metrics["num_episodes"],
            "validation/diagnostic/environment_cost_discounted_mean":
                metrics["diagnostic_environment_cost_discounted_mean"],
            "validation/diagnostic/environment_cost_raw_mean":
                metrics["diagnostic_environment_cost_raw_mean"],
        }
        q_est = getattr(current_agent, "q_est", None)
        if q_est is not None:
            q_value = float(q_est.detach().item() if torch.is_tensor(q_est) else q_est)
            # q_error remains a QPO-only tracker-calibration diagnostic. The common q_est above
            # already carries the same persistent value and must exist for non-QPO methods too.
            log["validation/quantile/q_error"] = q_value - metrics["quantile_return"]
        if "empirical_prob" in metrics:
            log.update({
                "validation/constraint/empirical_failure_probability":
                    metrics["empirical_prob"],
                "validation/constraint/residual": metrics["constraint_residual"],
                "validation/constraint/feasible": float(metrics["feasible"]),
                "validation/constraint/wilson95_low": metrics["wilson95_low"],
                "validation/constraint/wilson95_high": metrics["wilson95_high"],
            })
        calibration = metrics.get("diagnostic_reward_critic_calibration")
        if calibration is not None:
            # These values assess DQC's initial-state distributional critic against the same exact-N
            # validation trajectories used for the primary empirical chance metric. The diagnostic
            # namespace is deliberate: select_validation_candidate() never reads any of these keys.
            prefix = "validation/diagnostic/reward_critic"
            log.update({
                f"{prefix}/sample_count": calibration["sample_count"],
                f"{prefix}/initial_cdf_mean": calibration["initial_cdf_mean"],
                f"{prefix}/empirical_failure_probability":
                    calibration["empirical_failure_probability"],
                f"{prefix}/initial_cdf_gap": calibration["initial_cdf_gap"],
                f"{prefix}/initial_cdf_abs_gap": calibration["initial_cdf_abs_gap"],
                f"{prefix}/brier_score": calibration["brier_score"],
                f"{prefix}/action_cdf_std": calibration["action_cdf_std"],
                f"{prefix}/cdf_ece_10bin": calibration["cdf_ece_10bin"],
                f"{prefix}/cdf_zero_fraction": calibration["cdf_zero_fraction"],
                f"{prefix}/cdf_one_fraction": calibration["cdf_one_fraction"],
                f"{prefix}/cdf_saturation_fraction": calibration["cdf_saturation_fraction"],
                f"{prefix}/atom_crossing_row_fraction":
                    calibration["atom_crossing_row_fraction"],
                f"{prefix}/atom_crossing_adjacent_fraction":
                    calibration["atom_crossing_adjacent_fraction"],
                f"{prefix}/quantile_coverage_mae": calibration["quantile_coverage_mae"],
                f"{prefix}/alpha_atom_level": calibration["alpha_atom_level"],
                f"{prefix}/alpha_atom_coverage": calibration["alpha_atom_coverage"],
                f"{prefix}/alpha_atom_coverage_error":
                    calibration["alpha_atom_coverage_error"],
                f"{prefix}/predicted_return_mean": calibration["predicted_return_mean"],
                f"{prefix}/return_mean_bias": calibration["return_mean_bias"],
                f"{prefix}/return_rmse": calibration["return_rmse"],
                f"{prefix}/predicted_return_std": calibration["predicted_return_std"],
            })
        return log

    def validate_and_checkpoint(current_agent, iteration, training_metrics, force=False):
        """Retain compact diagnostics, validate on schedule, and save rolling checkpoints."""
        nonlocal selected_best_path, last_validation_iteration, last_checkpoint_iteration
        completed = int(iteration) + 1

        # Keep only pre-registered comparison fields, not every W&B scalar. Four hundred small
        # rows make 0-2M KL/clip/value summaries reproducible from the runner JSON while W&B remains
        # the full logging system. Initialization never enters this callback or this history.
        tracked_fields = (
            "disc_reward/aver_reward",
            "actor/clip_fraction",
            "diagnostic/dqc/policy_kl_after_epochs",
            "diagnostic/dqc/value_rmse",
            "diagnostic/dqc/critic_cdf_mean",
        )
        diagnostic_row = {
            "iteration": int(iteration),
            "main_env_steps": int(getattr(
                current_agent, "main_env_steps", completed * args.num_envs * args.horizon)),
        }
        for name in tracked_fields:
            value = training_metrics.get(name)
            if value is not None:
                diagnostic_row[name] = float(value)
        training_diagnostics_history.append(diagnostic_row)

        validation_interval = int(cli.validation_every)
        checkpoint_interval = int(cli.checkpoint_every)
        validation_due = scheduled_event_due(
            completed, validation_interval, last_validation_iteration, force=force)
        checkpoint_due = scheduled_event_due(
            completed, checkpoint_interval, last_checkpoint_iteration, force=force)
        if not validation_due and not checkpoint_due:
            return None

        # A checkpoint-only event must not run validation. This matters for non-nested schedules
        # (for example validation=10/checkpoint=25) and preserves the CLI promise that zero disables
        # validation even when a final forced checkpoint is due.
        if checkpoint_due and not validation_due:
            current_agent._write_external_progress(
                completed=int(iteration), metrics=training_metrics, phase="checkpoint")
            save_checkpoint(latest_path, current_agent, iteration, metrics=None)
            last_checkpoint_iteration = int(iteration)
            current_agent._pending_console_messages.append(
                "\n" + f"[checkpoint] batch {completed:03d}/{args.num_iterations:03d}  "
                f"latest={latest_path}" + "\n")
            return None

        # Keep the aggregate bar on the previous completed count until evaluation/checkpoint I/O
        # finishes, so its ETA includes this work instead of briefly claiming the batch is done.
        current_agent._write_external_progress(
            completed=int(iteration), metrics=training_metrics, phase="validation")
        evaluation_seed = int(args.seed) + 555
        with preserve_rng():
            # Environment seeds were already fixed. Reset all action-sampling RNGs as well so
            # different checkpoints use common random numbers without perturbing training RNG.
            random.seed(evaluation_seed)
            np.random.seed(evaluation_seed)
            torch.manual_seed(evaluation_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(evaluation_seed)
            validation_env = make_vec_env(
                args.env_id, args.num_envs, args.horizon, device, ref_env=env,
                seed=evaluation_seed, backend=args.vec_backend)
            try:
                metrics = evaluate_policy_vec(
                    current_agent, validation_env, cli.validation_episodes,
                    args.gamma, args.failure_alpha, args.return_threshold)
            finally:
                validation_env.close()

        metrics["iteration"] = int(iteration)
        remember_candidate(metrics)
        last_validation_iteration = int(iteration)

        if checkpoint_due:
            current_agent._write_external_progress(
                completed=int(iteration), metrics=training_metrics, phase="checkpoint")
            save_checkpoint(latest_path, current_agent, iteration, metrics=metrics)
            last_checkpoint_iteration = int(iteration)

        winner = select_validation_candidate(
            validation_history, args.failure_alpha, qpo=(cli.algo == "QPO"))
        best_updated = winner is metrics
        if best_updated:
            # best.pt is one rolling model-selection snapshot, not an accumulating checkpoint.
            save_checkpoint(best_path, current_agent, iteration, metrics=metrics)
            candidate_sources[id(metrics)] = best_path
            selected_best_path = best_path
        else:
            selected_best_path = candidate_sources.get(id(winner), selected_best_path)

        # The evaluator has already applied the formulation-aware public semantics: persistent
        # tracker for QPO, independent empirical percentile for every other algorithm.
        q_text = f"{float(metrics['quantile_q_est']):.3f}"
        constraint_probability = metrics.get("quantile_constraint_prob")
        probability_text = (
            "n/a" if constraint_probability is None else f"{float(constraint_probability):.3f}")
        lines = [
            "",
            f"[validation] batch {completed:03d}/{args.num_iterations:03d}  "
            f"episodes={metrics['num_episodes']}",
            f"  mean_reward={metrics['mean']:.3f}  "
            f"quantile_reward={metrics['quantile_return']:.3f}  "
            f"q_est={q_text}  constraint_prob={probability_text}",
        ]
        calibration = metrics.get("diagnostic_reward_critic_calibration")
        if calibration is not None:
            # One compact DQC-only line makes critic drift visible without mixing diagnostics into
            # the common four-number validation line or model-selection rule.
            lines.append(
                f"  critic_cdf={calibration['initial_cdf_mean']:.3f}  "
                f"cdf_gap={calibration['initial_cdf_gap']:+.3f}  "
                f"brier={calibration['brier_score']:.3f}  "
                f"return_rmse={calibration['return_rmse']:.3f}  "
                f"crossing={calibration['atom_crossing_row_fraction']:.3f}")
        if best_updated:
            lines.append(f"  best_updated=yes  path={best_path}")
        if checkpoint_due:
            lines.append(f"  latest_checkpoint={latest_path}")
        lines.append("")
        # Delay terminal output until _after_iteration has uploaded/logged the completed training
        # batch. This guarantees the visual order [train] -> blank line -> [validation] for every
        # algorithm, while preserving exactly one W&B row at the corresponding batch step.
        current_agent._pending_console_messages.append("\n".join(lines))
        return validation_wandb_metrics(metrics, current_agent)

    def save_initialization_checkpoint(
            current_agent, completed_initialization, total_initialization):
        """Atomically persist one save-only initialization boundary at formal iteration -1."""
        save_checkpoint(initialization_path, current_agent, iteration=-1, metrics=None)
        if int(completed_initialization) > int(total_initialization):
            raise RuntimeError("initialization cursor exceeded configured batch count")

    # This callback is separate from validate_and_checkpoint: initialization cannot create
    # validation history, best candidates, or W&B rows.
    agent.initialization_callback = save_initialization_checkpoint

    agent.iteration_callback = validate_and_checkpoint
    print(f"algo={cli.algo} env={cli.env} seed={args.seed} device={device} "
          f"q={args.return_threshold} overrides={overrides}")
    train_seconds = 0.0
    if not cli.eval_only and agent.start_iteration < args.num_iterations:
        started = time.time()
        agent.train()
        train_seconds = time.time() - started
        final_iteration = args.num_iterations - 1
        # Non-divisible custom schedules still receive one final validation/latest snapshot.
        needs_final_validation = (
            int(cli.validation_every) > 0 and last_validation_iteration != final_iteration)
        needs_final_checkpoint = (
            int(cli.checkpoint_every) > 0 and last_checkpoint_iteration != final_iteration)
        if needs_final_validation or needs_final_checkpoint:
            forced = validate_and_checkpoint(agent, final_iteration, {}, force=True)
            if forced and getattr(agent, "_wandb_run", None) is not None:
                # This path is absent for the default 10/50 divisors. It remains explicit for
                # custom schedules that end between intervals.
                final_log = dict(getattr(agent, "_last_progress_metrics", {}))
                final_log.update(forced)
                agent._wandb_run.log(final_log, step=final_iteration)

    # Load the validation-selected policy before touching the disjoint test seeds. The optimizer
    # state is harmless here; restore_rng=False preserves the dedicated final-test sampling stream.
    selected_iteration = args.num_iterations - 1
    if selected_best_path is not None and Path(selected_best_path).exists():
        selected = load_checkpoint(
            selected_best_path, agent, map_location=device, restore_rng=False,
            allow_signature_mismatch=cli.allow_checkpoint_transfer)
        selected_iteration = int(selected["iteration"])
        print(
            f"\n[selected policy] batch={selected_iteration + 1} "
            f"checkpoint={selected_best_path}\n", flush=True)

    # Final evaluation belongs to the same launcher task. Keep its phase visible after the
    # aggregate training counter reaches 100%; the parent closes the bar only after this exits.
    agent._write_external_progress(
        completed=args.num_iterations,
        metrics=getattr(agent, "_last_progress_metrics", {}),
        phase="evaluation")

    # Final test uses a seed set disjoint from checkpoint validation and returns exactly N episodes.
    test_env = make_vec_env(
        args.env_id, args.num_envs, args.horizon, device, ref_env=env,
        seed=args.seed + 777, backend=args.vec_backend)
    result = evaluate_policy_vec(
        agent, test_env, cli.num_eval, args.gamma, args.failure_alpha, args.return_threshold,
        show_progress=bool(getattr(args, "show_eval_progress", True)),
        progress_desc=f"test/{cli.algo}/{cli.env}")
    test_env.close()
    agent.vec_env.close()
    env.close()

    print(f"test N={result['num_episodes']} disc_a_r={result['mean']:.3f} "
          f"disc_q_r={result['quantile_return']:.3f} "
          f"q_est={result['quantile_q_est']:.3f} "
          f"constraint_prob={result['quantile_constraint_prob']:.3f}")
    if args.return_threshold is not None:
        print(f"P(G_1000<=q)={result['empirical_prob']:.3f} "
              f"residual={result['constraint_residual']:+.3f} "
              f"Wilson95=[{result['wilson95_low']:.3f},{result['wilson95_high']:.3f}]")

    run = getattr(agent, "_wandb_run", None)
    if run is not None:
        # Test return fields use the same disc_reward namespace as training curves.
        test_log = {
            "test/disc_reward/aver_reward": result["mean"],
            "test/disc_reward/return_std": result["reward_std"],
            "test/disc_reward/quantile_reward": result["quantile_return"],
            "test/quantile/q_est": result["quantile_q_est"],
            "test/quantile/constraint_prob": result["quantile_constraint_prob"],
            "test/model_selection_iteration": selected_iteration,
        }
        # Exclude explicitly mapped public names so the generic diagnostics loop cannot create
        # underscore aliases such as test/quantile_q_est alongside the canonical slash key.
        renamed_return_fields = {
            "mean", "reward_std", "quantile_return",
            "quantile_q_est", "quantile_constraint_prob",
        }
        test_log.update({
            f"test/{key}": value for key, value in result.items()
            if key not in renamed_return_fields
            and isinstance(value, (int, float, bool)) and value is not None
        })
        run.log(test_log)
        run.finish()
    output = {
        "tag": cli.tag, "algo": cli.algo, "env": cli.env, "seed": args.seed,
        "return_threshold": args.return_threshold, "failure_alpha": args.failure_alpha,
        "train_seconds": train_seconds, "eval": result,
        "summary": agent.get_training_summary(),
        "latest_checkpoint": str(latest_path) if latest_path.exists() else None,
        "initialization_checkpoint": (
            str(initialization_path) if initialization_path.exists() else None),
        "best_checkpoint": (
            str(selected_best_path) if selected_best_path is not None else None),
        "selected_iteration": selected_iteration,
        "validation_history": validation_history,
        "training_diagnostics_history": training_diagnostics_history,
        "validation_every": int(cli.validation_every),
        "checkpoint_every": int(cli.checkpoint_every),
        "overrides": overrides,
    }
    output_dir = BASE_DIR / "_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{cli.algo}_{cli.env}_{cli.tag}_s{args.seed}.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    agent._write_external_progress(
        completed=args.num_iterations,
        metrics=getattr(agent, "_last_progress_metrics", {}),
        phase="complete")
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
