#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用既有trajectory outage日志离线重放不同经验PID窗口的控制信号。

这个工具不声称预测“换窗口后策略会怎样”。它只回答在完全相同的已观察轨迹上：
    1. 更长窗口能把经验概率与lambda的事件间跳变降低多少；
    2. 平滑是否以明显落后于下一B40真实outage为代价；
    3. 代码中的leaky PI公式能否逐点复现正式run的lambda。

对于每个rollout轨迹数相同、候选窗口为rollout大小整数倍的情况，候选窗口概率
可以由每批outage计数精确重建。非整数倍的正式baseline直接读取日志中的真实deque
结果，避免猜测上一个rollout里被截取的episode顺序。

典型用法：
    python audit_pid_window_replay.py \
        --input _runs/wandb_export/example/combined_history.csv \
        --out-dir _runs/profiles/example_pid_window_audit \
        --episodes-per-rollout 20 \
        --event-size 2 \
        --candidate-windows 100,200 \
        --min-step 400000
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    """解析history字段、PID公式和探索性窗口筛选阈值。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="combined_history.csv")
    parser.add_argument("--out-dir", type=Path, required=True, help="本地CSV/JSON/PNG目录")
    parser.add_argument("--episodes-per-rollout", type=int, required=True)
    parser.add_argument("--event-size", type=int, default=2, help="每个PID事件累计rollout数")
    parser.add_argument("--candidate-windows", default="100,200", help="逗号分隔的episode窗口")
    parser.add_argument("--min-step", type=int, default=400_000, help="成熟期统计起点")

    # PID参数默认对应P-M8；全部写入JSON，避免离线重放隐藏控制器假设。
    parser.add_argument("--target-prob", type=float, default=0.15)
    parser.add_argument("--kp", type=float, default=1.0)
    parser.add_argument("--ki", type=float, default=0.1)
    parser.add_argument("--integral-leak", type=float, default=0.97)
    parser.add_argument("--deadband", type=float, default=0.02)
    parser.add_argument("--delta-max", type=float, default=0.05)
    parser.add_argument("--reference-episodes", type=float, default=10.0)
    parser.add_argument("--lambda-min", type=float, default=0.0)
    parser.add_argument("--lambda-max", type=float, default=5.0)

    # 列名可覆盖，使同一工具能审计其他环境或命名版本。
    parser.add_argument("--run-column", default="run_id")
    parser.add_argument("--name-column", default="run_name")
    parser.add_argument("--step-column", default="progress/env_steps")
    parser.add_argument("--rollout-prob-column", default="constraint/empirical_prob")
    parser.add_argument("--due-column", default="dual/pid_update_due")
    parser.add_argument("--logged-window-column", default="dual/window_empirical_prob")
    parser.add_argument("--logged-lambda-column", default="lambda/value")

    # 探索性选择规则：最小平滑收益25%，每个seed的下一事件MAE恶化不超过10%。
    parser.add_argument("--min-jump-reduction", type=float, default=0.25)
    parser.add_argument("--max-next-mae-worsening", type=float, default=0.10)
    return parser.parse_args()


def parse_windows(raw: str) -> List[int]:
    """把逗号窗口解析为去重正整数，保留用户给出的顺序。"""

    windows: List[int] = []
    for token in raw.split(","):
        value = int(token.strip())
        if value <= 0:
            raise ValueError("candidate windows must be positive")
        if value not in windows:
            windows.append(value)
    if not windows:
        raise ValueError("at least one candidate window is required")
    return windows


def require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    """缺少真实outage、due或正式lambda时停止，不能用默认0伪造控制曲线。"""

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"input history is missing columns: {missing}")


def infer_seed(run_name: str, fallback_index: int) -> int:
    """从run name末尾_sN读取seed；旧命名不匹配时使用稳定分组序号。"""

    match = re.search(r"_s(\d+)$", str(run_name))
    return int(match.group(1)) if match else fallback_index


def filtered_error(probability: float, target: float, deadband: float) -> float:
    """逐式复现DQCAC连续deadband：带符号地减去阈值，而不是硬置零后跳变。"""

    raw_error = float(probability) - float(target)
    magnitude = max(0.0, abs(raw_error) - deadband)
    return float(math.copysign(magnitude, raw_error)) if magnitude else 0.0


def replay_pid(probabilities: Sequence[float], new_episodes: int, args: argparse.Namespace) -> np.ndarray:
    """
    按bounded leaky-I + proportional output重放lambda。

    每次事件加入new_episodes条轨迹；episode scale、几何积分和delta limit都与
    agents/dqc_ac_beta_gpu.py::_update_pid_integral保持同一公式。
    """

    if args.reference_episodes > 0.0:
        episode_scale = float(new_episodes) / args.reference_episodes
    else:
        episode_scale = 1.0
    effective_leak = args.integral_leak ** episode_scale

    # rho<1使用几何和；rho=1取连续极限episode_scale。
    if args.integral_leak < 1.0:
        integral_scale = (1.0 - effective_leak) / (1.0 - args.integral_leak)
    else:
        integral_scale = episode_scale
    effective_delta_max = args.delta_max * episode_scale

    integral = 0.0
    outputs: List[float] = []
    for probability in probabilities:
        error = filtered_error(probability, args.target_prob, args.deadband)
        raw_delta = args.ki * error * integral_scale
        bounded_delta = min(effective_delta_max, max(-effective_delta_max, raw_delta))

        # I state先leak再积分并clip；P项不进入state，只影响当前事件output。
        integral = min(
            args.lambda_max,
            max(args.lambda_min, effective_leak * integral + bounded_delta),
        )
        output = min(
            args.lambda_max,
            max(args.lambda_min, integral + args.kp * error),
        )
        outputs.append(output)

    return np.asarray(outputs, dtype=np.float64)


def exact_rolling_probability(
    rollout_events: np.ndarray,
    episodes_per_rollout: int,
    window_episodes: int,
) -> np.ndarray:
    """
    从每个rollout的整数超限次数精确重建candidate deque概率。

    为避免未知episode顺序，窗口必须是rollout大小的整数倍；这样任何时点都只
    包含完整rollout，不会截断某批内部的一部分环境编号。
    """

    if window_episodes % episodes_per_rollout != 0:
        raise ValueError(
            f"candidate window {window_episodes} must be divisible by "
            f"episodes-per-rollout {episodes_per_rollout}")
    rollout_horizon = window_episodes // episodes_per_rollout

    probabilities = np.empty(len(rollout_events), dtype=np.float64)
    for index in range(len(rollout_events)):
        begin = max(0, index - rollout_horizon + 1)
        event_count = rollout_events[begin:index + 1].sum()
        episode_count = (index - begin + 1) * episodes_per_rollout
        probabilities[index] = event_count / episode_count
    return probabilities


def aggregate_run(
    raw_group: pd.DataFrame,
    fallback_index: int,
    windows: Sequence[int],
    args: argparse.Namespace,
) -> pd.DataFrame:
    """把一个run的B20记录变成B40 PID事件，并加入候选窗口与lambda重放。"""

    group = raw_group.sort_values(args.step_column).reset_index(drop=True).copy()
    if len(group) % args.event_size != 0:
        raise ValueError("history length must be divisible by event-size")

    # 概率乘每批episode数必须回到整数事件计数；否则CSV精度或配置不一致。
    rollout_probabilities = group[args.rollout_prob_column].to_numpy(dtype=np.float64)
    rollout_events = np.rint(rollout_probabilities * args.episodes_per_rollout).astype(np.int64)
    reconstructed = rollout_events / float(args.episodes_per_rollout)
    if not np.allclose(reconstructed, rollout_probabilities, atol=1e-10, rtol=0.0):
        raise ValueError("rollout empirical probability is inconsistent with episode count")

    # 每个候选窗口先在rollout边界重建，再只取PID到期边界。
    rolling_by_window = {
        window: exact_rolling_probability(
            rollout_events, args.episodes_per_rollout, window)
        for window in windows
    }
    due_indices = np.arange(args.event_size - 1, len(group), args.event_size)
    due_values = group.loc[due_indices, args.due_column].to_numpy(dtype=np.float64)
    if not np.allclose(due_values, 1.0):
        raise ValueError("expected exactly one due boundary at the end of every control event")

    run_name = str(group[args.name_column].iloc[0])
    output = pd.DataFrame({
        "run_id": str(group[args.run_column].iloc[0]),
        "run_name": run_name,
        "seed": infer_seed(run_name, fallback_index),
        "event_index": np.arange(len(due_indices), dtype=np.int64),
        "step": group.loc[due_indices, args.step_column].to_numpy(dtype=np.int64),
        "raw_event_probability": np.asarray([
            rollout_events[index - args.event_size + 1:index + 1].sum()
            / float(args.episodes_per_rollout * args.event_size)
            for index in due_indices
        ]),
        "logged_window_probability": group.loc[
            due_indices, args.logged_window_column].to_numpy(dtype=np.float64),
        "logged_lambda": group.loc[
            due_indices, args.logged_lambda_column].to_numpy(dtype=np.float64),
    })

    for window, probabilities in rolling_by_window.items():
        output[f"window_{window}_probability"] = probabilities[due_indices]

    new_episodes = args.episodes_per_rollout * args.event_size
    output["replayed_logged_lambda"] = replay_pid(
        output["logged_window_probability"].to_numpy(), new_episodes, args)
    for window in windows:
        output[f"window_{window}_lambda"] = replay_pid(
            output[f"window_{window}_probability"].to_numpy(), new_episodes, args)

    # 正式logged window必须逐点复现lambda；这是候选结果可信之前的硬前置条件。
    max_error = float(
        np.max(np.abs(output["logged_lambda"] - output["replayed_logged_lambda"])))
    if max_error > 1e-6:
        raise AssertionError(f"PID replay does not reproduce logged lambda: max error={max_error}")
    output["logged_lambda_replay_abs_error"] = np.abs(
        output["logged_lambda"] - output["replayed_logged_lambda"])
    return output


def signal_metrics(probabilities: np.ndarray, lambdas: np.ndarray, raw: np.ndarray) -> Dict[str, float]:
    """统计事件跳变、当前平滑滞后与预测下一事件raw outage的误差。"""

    if len(probabilities) < 2:
        raise ValueError("at least two mature events are required")
    return {
        "probability_std": float(np.std(probabilities)),
        "probability_mean_abs_delta": float(np.mean(np.abs(np.diff(probabilities)))),
        "current_raw_mean_abs_gap": float(np.mean(np.abs(probabilities - raw))),
        # 最后一个事件没有next truth，从分母中严格排除。
        "next_raw_mean_abs_error": float(np.mean(np.abs(probabilities[:-1] - raw[1:]))),
        "lambda_std": float(np.std(lambdas)),
        "lambda_mean_abs_delta": float(np.mean(np.abs(np.diff(lambdas)))),
        "lambda_final": float(lambdas[-1]),
    }


def fractional_reduction(candidate: float, baseline: float) -> float:
    """返回1-candidate/baseline；正值表示候选降低了该误差或波动。"""

    return 1.0 - candidate / baseline if baseline > 0.0 else float("nan")


def summarize(events: pd.DataFrame, windows: Sequence[int], args: argparse.Namespace) -> Dict[str, object]:
    """生成逐seed统计和探索性“最小够用窗口”选择。"""

    seeds: Dict[str, Dict[str, object]] = {}
    candidate_passes: Dict[int, List[bool]] = {window: [] for window in windows}

    for seed, raw_group in events.groupby("seed", sort=True):
        group = raw_group.loc[raw_group["step"] >= args.min_step].sort_values("step")
        raw = group["raw_event_probability"].to_numpy(dtype=np.float64)
        baseline = signal_metrics(
            group["logged_window_probability"].to_numpy(dtype=np.float64),
            group["logged_lambda"].to_numpy(dtype=np.float64),
            raw,
        )
        seed_summary: Dict[str, object] = {"logged_window": baseline, "candidates": {}}

        for window in windows:
            candidate = signal_metrics(
                group[f"window_{window}_probability"].to_numpy(dtype=np.float64),
                group[f"window_{window}_lambda"].to_numpy(dtype=np.float64),
                raw,
            )
            changes = {
                "probability_jump_reduction": fractional_reduction(
                    candidate["probability_mean_abs_delta"],
                    baseline["probability_mean_abs_delta"],
                ),
                "lambda_jump_reduction": fractional_reduction(
                    candidate["lambda_mean_abs_delta"],
                    baseline["lambda_mean_abs_delta"],
                ),
                "next_raw_mae_fractional_change": (
                    candidate["next_raw_mean_abs_error"]
                    / baseline["next_raw_mean_abs_error"] - 1.0
                ),
            }
            passes = (
                changes["probability_jump_reduction"] >= args.min_jump_reduction
                and changes["lambda_jump_reduction"] >= args.min_jump_reduction
                and changes["next_raw_mae_fractional_change"] <= args.max_next_mae_worsening
            )
            candidate_passes[window].append(bool(passes))
            seed_summary["candidates"][str(window)] = {
                "metrics": candidate,
                "relative_to_logged": changes,
                "passes_exploratory_signal_rule": bool(passes),
            }
        seeds[str(int(seed))] = seed_summary

    # 选择所有seed都过信号门的最小窗口；这里只决定值得实跑哪个，不是性能成功门。
    eligible = [window for window in windows if all(candidate_passes[window])]
    selected = min(eligible) if eligible else None
    return {
        "min_step": args.min_step,
        "selection_rule": {
            "min_probability_and_lambda_jump_reduction": args.min_jump_reduction,
            "max_per_seed_next_raw_mae_worsening": args.max_next_mae_worsening,
            "must_pass_all_seeds": True,
            "exploratory_not_performance_preregistered": True,
        },
        "seed_summaries": seeds,
        "eligible_windows": eligible,
        "selected_smallest_window": selected,
    }


def json_safe(value: object) -> object:
    """将NumPy类型和非有限数转换为严格JSON表示。"""

    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def plot_replay(events: pd.DataFrame, windows: Sequence[int], output: Path, min_step: int) -> None:
    """每个seed画raw/window概率和对应lambda，虚线标出成熟统计起点。"""

    seeds = sorted(events["seed"].unique().tolist())
    figure, axes = plt.subplots(len(seeds), 2, figsize=(14, 3.4 * len(seeds)), squeeze=False, constrained_layout=True)
    palette = ["#F58518", "#54A24B", "#B279A2", "#E45756"]

    for row, seed in enumerate(seeds):
        group = events.loc[events["seed"] == seed].sort_values("step")
        x = group["step"] / 1_000_000.0

        axes[row, 0].plot(x, group["raw_event_probability"], color="#4C78A8", marker="o", label="raw B40")
        axes[row, 0].plot(x, group["logged_window_probability"], color=palette[0], marker=".", label="logged window50")
        for index, window in enumerate(windows, start=1):
            axes[row, 0].plot(x, group[f"window_{window}_probability"], color=palette[index % len(palette)], label=f"window{window}")
        axes[row, 0].axvline(min_step / 1_000_000.0, color="#555555", linestyle="--", linewidth=1.0)
        axes[row, 0].set_ylabel(f"seed {int(seed)} probability")
        axes[row, 0].legend(frameon=False, fontsize=8, ncol=2)

        axes[row, 1].plot(x, group["logged_lambda"], color=palette[0], marker=".", label="logged/replayed window50")
        for index, window in enumerate(windows, start=1):
            axes[row, 1].plot(x, group[f"window_{window}_lambda"], color=palette[index % len(palette)], label=f"counterfactual window{window}")
        axes[row, 1].axvline(min_step / 1_000_000.0, color="#555555", linestyle="--", linewidth=1.0)
        axes[row, 1].set_ylabel(f"seed {int(seed)} lambda")
        axes[row, 1].legend(frameon=False, fontsize=8)

    axes[-1, 0].set_xlabel("environment steps (millions)")
    axes[-1, 1].set_xlabel("environment steps (millions)")
    figure.suptitle("Offline empirical-PID window replay (fixed observed trajectories)", fontsize=15, fontweight="semibold")
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """读取history、验证正式lambda、重放候选窗口并保存可复现实验依据。"""

    args = parse_args()
    windows = parse_windows(args.candidate_windows)
    if args.episodes_per_rollout <= 0 or args.event_size <= 0:
        raise ValueError("episode and event sizes must be positive")
    if not 0.0 < args.integral_leak <= 1.0:
        raise ValueError("integral-leak must be in (0, 1]")
    if args.deadband < 0.0 or args.delta_max <= 0.0:
        raise ValueError("deadband must be non-negative and delta-max positive")

    history = pd.read_csv(args.input)
    require_columns(
        history,
        (
            args.run_column,
            args.name_column,
            args.step_column,
            args.rollout_prob_column,
            args.due_column,
            args.logged_window_column,
            args.logged_lambda_column,
        ),
    )

    run_events = [
        aggregate_run(group, index, windows, args)
        for index, (_run_id, group) in enumerate(history.groupby(args.run_column, sort=False))
    ]
    events = pd.concat(run_events, ignore_index=True).sort_values(["seed", "step"]).reset_index(drop=True)
    summary = {
        "input_file": args.input.name,
        "episodes_per_rollout": args.episodes_per_rollout,
        "event_size": args.event_size,
        "candidate_windows": windows,
        "pid_parameters": {
            "target_prob": args.target_prob,
            "kp": args.kp,
            "ki": args.ki,
            "integral_leak": args.integral_leak,
            "deadband": args.deadband,
            "delta_max": args.delta_max,
            "reference_episodes": args.reference_episodes,
            "lambda_min": args.lambda_min,
            "lambda_max": args.lambda_max,
        },
        "logged_lambda_replay_max_abs_error": float(
            events["logged_lambda_replay_abs_error"].max()),
        "audit": summarize(events, windows, args),
        "limitation": (
            "Counterfactual windows are replayed on fixed observed trajectories; "
            "they measure signal variance and lag, not policy performance."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.out_dir / "window_event_series.csv", index=False)
    (args.out_dir / "window_replay_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plot_replay(events, windows, args.out_dir / "pid_window_replay.png", args.min_step)
    print(json.dumps(json_safe(summary), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
