#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
审计distributional cost critic的更新前校准偏差是否具有跨控制事件可预测性。

输入是export_wandb_metrics.py生成的combined_history.csv。脚本严格使用
pre-update CDF和已经发生的完整trajectory truth，按真实PID/Actor cadence把多个
rollout聚合为一个控制事件，并检验当前偏差能否预测下一事件偏差。

典型用法：
    python audit_prequential_calibration.py \
        --input _runs/wandb_export/example/combined_history.csv \
        --out-dir _runs/profiles/example_calibration_audit \
        --event-size 2 \
        --primary-min-step 400000 \
        --robustness-min-step 600000 \
        --ema-tau 0.2

输出：
    event_series.csv   每个run、每个控制事件的truth、prediction、偏差和因果预测；
    audit_summary.json 预注册指标、逐seed统计和是否过门；
    calibration_audit.png 逐run的概率与偏差时间序列。
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, pearsonr


# 默认列名对应DQCAC的更新前initial-state holdout；均可由CLI覆盖以复用到其他日志。
DEFAULT_STEP_COLUMN = "progress/env_steps"
DEFAULT_TRUTH_COLUMN = "critic/s0_holdout_pre_truth"
DEFAULT_PREDICTION_COLUMN = "critic/s0_holdout_pre_cdf"
DEFAULT_POST_COLUMN = "critic/s0_holdout_post_cdf"
DEFAULT_DUE_COLUMN = "dual/pid_update_due"


def parse_args() -> argparse.Namespace:
    """解析输入、聚合cadence、成熟区间和预注册门槛。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="combined_history.csv")
    parser.add_argument("--out-dir", type=Path, required=True, help="本地审计输出目录")
    parser.add_argument("--event-size", type=int, default=2, help="一个控制事件包含的rollout数")
    parser.add_argument("--primary-min-step", type=int, default=400_000, help="主成熟区间起点")
    parser.add_argument("--robustness-min-step", type=int, default=600_000, help="稳健性区间起点")
    parser.add_argument("--ema-tau", type=float, default=0.2, help="严格因果EMA更新系数")

    # 这些参数只改变CSV字段映射，不改变统计定义。
    parser.add_argument("--run-column", default="run_id")
    parser.add_argument("--name-column", default="run_name")
    parser.add_argument("--step-column", default=DEFAULT_STEP_COLUMN)
    parser.add_argument("--truth-column", default=DEFAULT_TRUTH_COLUMN)
    parser.add_argument("--prediction-column", default=DEFAULT_PREDICTION_COLUMN)
    parser.add_argument("--post-column", default=DEFAULT_POST_COLUMN)
    parser.add_argument("--due-column", default=DEFAULT_DUE_COLUMN)

    # 默认值就是E93预注册门，显式放进JSON以便未来审计时不会依赖文档记忆。
    parser.add_argument("--min-lag-correlation", type=float, default=0.25)
    parser.add_argument("--min-sign-agreement", type=float, default=0.65)
    parser.add_argument("--max-sign-pvalue", type=float, default=0.10)
    parser.add_argument("--min-ema-improvement", type=float, default=0.10)
    parser.add_argument("--min-seeds-improved", type=int, default=2)
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    """缺少核心列时直接报错，不能把不存在的pre-update信号静默填0。"""

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"input history is missing columns: {missing}")


def infer_seed(run_name: str, fallback_index: int) -> int:
    """从统一run name末尾的_sN提取seed；旧日志不匹配时使用稳定组序号。"""

    match = re.search(r"_s(\d+)$", str(run_name))
    return int(match.group(1)) if match else fallback_index


def causal_predictions(values: np.ndarray, tau: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    [功能] 为每个事件构造last-value和EMA的一步预测。
    [因果约束] 事件t的预测只使用0..t-1的偏差；当前truth只在预测完成后更新状态。
    """

    last_predictions = np.zeros_like(values, dtype=np.float64)
    ema_predictions = np.zeros_like(values, dtype=np.float64)
    ema_state = 0.0

    for index, value in enumerate(values.astype(np.float64, copy=False)):
        # 第一个事件没有历史信息，两种预测均使用“无偏差”先验0。
        if index > 0:
            last_predictions[index] = values[index - 1]
        ema_predictions[index] = ema_state

        # 当前偏差在本次预测之后才进入状态，避免look-ahead leakage。
        ema_state = (1.0 - tau) * ema_state + tau * value

    return last_predictions, ema_predictions


def aggregate_control_events(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """
    将每个run的连续rollout按event-size聚合成真实控制事件。

    truth和CDF按trajectory数相同的rollout取均值；step取事件最后一个rollout。
    当前实验每个B20大小相同，因此简单均值等价于40条trajectory的总体均值。
    """

    event_frames: List[pd.DataFrame] = []

    for fallback_index, (run_id, raw_group) in enumerate(frame.groupby(args.run_column, sort=False)):
        group = raw_group.sort_values(args.step_column).reset_index(drop=True).copy()
        if len(group) % args.event_size != 0:
            raise ValueError(f"run {run_id} has {len(group)} rows, not divisible by event-size={args.event_size}")

        # 每个event-size行形成一个事件；不跨run，因此不会混合不同policy初始化。
        group["event_index"] = np.arange(len(group), dtype=np.int64) // args.event_size
        aggregation: Dict[str, Tuple[str, str]] = {
            "step": (args.step_column, "max"),
            "truth": (args.truth_column, "mean"),
            "prediction": (args.prediction_column, "mean"),
            "rollout_count": (args.step_column, "size"),
        }

        # post-update只用于可视化“本批拟合移动”，从不进入因果predictability gate。
        if args.post_column in group.columns:
            aggregation["post_prediction"] = (args.post_column, "mean")
        if args.due_column in group.columns:
            aggregation["due_count"] = (args.due_column, "sum")

        events = group.groupby("event_index", sort=True).agg(**aggregation).reset_index()
        run_name = str(group[args.name_column].iloc[0])
        events.insert(0, "run_id", str(run_id))
        events.insert(1, "run_name", run_name)
        events.insert(2, "seed", infer_seed(run_name, fallback_index))

        # 对interval2而言每个B40块必须恰好有一个PID due行，否则相邻聚合无物理意义。
        if "due_count" in events.columns and not np.allclose(events["due_count"], 1.0):
            bad = events.loc[~np.isclose(events["due_count"], 1.0), ["event_index", "due_count"]]
            raise ValueError(f"run {run_id} has invalid PID cadence blocks:\n{bad}")

        # 正值代表critic低估真实风险，是安全余量需要增大的方向。
        events["underestimation"] = events["truth"] - events["prediction"]
        events["absolute_error"] = events["underestimation"].abs()
        last_prediction, ema_prediction = causal_predictions(events["underestimation"].to_numpy(), args.ema_tau)
        events["causal_last_bias_prediction"] = last_prediction
        events["causal_ema_bias_prediction"] = ema_prediction
        event_frames.append(events)

    return pd.concat(event_frames, ignore_index=True).sort_values(["seed", "step"]).reset_index(drop=True)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """样本少于2或任一变量为常数时返回None，避免把NaN写成伪相关。"""

    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    correlation = float(pearsonr(x, y).statistic)
    return correlation if math.isfinite(correlation) else None


def build_lag_pairs(events: pd.DataFrame, min_step: int) -> pd.DataFrame:
    """为成熟区间构造同run内连续事件对，不允许跨seed连接。"""

    pairs: List[pd.DataFrame] = []
    mature = events.loc[events["step"] >= min_step].copy()

    for run_id, group in mature.groupby("run_id", sort=False):
        ordered = group.sort_values("step").copy()
        pair = ordered[["run_id", "seed", "step", "underestimation"]].copy()
        pair["previous_underestimation"] = ordered["underestimation"].shift(1)
        pair = pair.dropna(subset=["previous_underestimation"])
        pairs.append(pair)

    return pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()


def summarize_window(events: pd.DataFrame, min_step: int) -> Dict[str, object]:
    """计算一个成熟区间的pooled、固定seed效应和逐seed一步预测统计。"""

    mature = events.loc[events["step"] >= min_step].copy()
    pairs = build_lag_pairs(events, min_step)
    if pairs.empty:
        raise ValueError(f"no lag pairs at min_step={min_step}")

    # 去掉各seed自己的均值后再pool，防止seed长期基线差异伪造时间相关。
    pairs["previous_centered"] = pairs["previous_underestimation"] - pairs.groupby("run_id")["previous_underestimation"].transform("mean")
    pairs["current_centered"] = pairs["underestimation"] - pairs.groupby("run_id")["underestimation"].transform("mean")
    pooled_lag_correlation = safe_pearson(
        pairs["previous_centered"].to_numpy(), pairs["current_centered"].to_numpy()
    )

    # 完全为0的偏差没有方向信息，因此从符号持续率分母中排除。
    nonzero = pairs.loc[
        (~np.isclose(pairs["previous_underestimation"], 0.0))
        & (~np.isclose(pairs["underestimation"], 0.0))
    ].copy()
    sign_matches = int(
        (np.sign(nonzero["previous_underestimation"]) == np.sign(nonzero["underestimation"])).sum()
    )
    sign_total = int(len(nonzero))
    sign_agreement = sign_matches / sign_total if sign_total else float("nan")
    sign_pvalue = float(binomtest(sign_matches, sign_total, 0.5, alternative="greater").pvalue) if sign_total else float("nan")

    # 零基线代表“偏差不可预测”；EMA必须降低下一事件的绝对预测误差才有反馈价值。
    zero_mae = float(mature["underestimation"].abs().mean())
    last_mae = float((mature["underestimation"] - mature["causal_last_bias_prediction"]).abs().mean())
    ema_mae = float((mature["underestimation"] - mature["causal_ema_bias_prediction"]).abs().mean())
    ema_improvement = 1.0 - ema_mae / zero_mae if zero_mae > 0.0 else float("nan")

    seed_summaries: Dict[str, Dict[str, object]] = {}
    for seed, group in mature.groupby("seed", sort=True):
        seed_zero_mae = float(group["underestimation"].abs().mean())
        seed_ema_mae = float((group["underestimation"] - group["causal_ema_bias_prediction"]).abs().mean())
        seed_pairs = pairs.loc[pairs["seed"] == seed]
        seed_summaries[str(int(seed))] = {
            "events": int(len(group)),
            "mean_underestimation": float(group["underestimation"].mean()),
            "underestimation_fraction": float((group["underestimation"] > 0.0).mean()),
            "zero_bias_mae": seed_zero_mae,
            "causal_ema_mae": seed_ema_mae,
            "ema_improvement_fraction": 1.0 - seed_ema_mae / seed_zero_mae if seed_zero_mae > 0.0 else None,
            "lag1_correlation": safe_pearson(
                seed_pairs["previous_underestimation"].to_numpy(),
                seed_pairs["underestimation"].to_numpy(),
            ),
        }

    return {
        "min_step": int(min_step),
        "events": int(len(mature)),
        "lag_pairs": int(len(pairs)),
        "pooled_within_seed_lag1_correlation": pooled_lag_correlation,
        "sign_matches": sign_matches,
        "sign_pairs": sign_total,
        "sign_agreement_fraction": sign_agreement,
        "sign_agreement_binomial_pvalue_greater_than_half": sign_pvalue,
        "zero_bias_mae": zero_mae,
        "last_value_mae": last_mae,
        "causal_ema_mae": ema_mae,
        "causal_ema_improvement_fraction": ema_improvement,
        "seed_summaries": seed_summaries,
    }


def evaluate_gate(primary: Mapping[str, object], args: argparse.Namespace) -> Dict[str, object]:
    """按CLI中显式记录的E93门槛做机械裁决，不事后改变阈值。"""

    lag_value = primary["pooled_within_seed_lag1_correlation"]
    lag_pass = lag_value is not None and float(lag_value) > args.min_lag_correlation
    sign_pass = (
        float(primary["sign_agreement_fraction"]) >= args.min_sign_agreement
        and float(primary["sign_agreement_binomial_pvalue_greater_than_half"]) < args.max_sign_pvalue
    )
    ema_pass = float(primary["causal_ema_improvement_fraction"]) >= args.min_ema_improvement

    # “同方向”定义为EMA至少优于零基线，不要求每个seed都达到pooled的10%幅度门。
    seed_summaries = primary["seed_summaries"]
    assert isinstance(seed_summaries, Mapping)
    improved_seed_count = sum(
        1 for summary in seed_summaries.values()
        if isinstance(summary, Mapping)
        and summary["ema_improvement_fraction"] is not None
        and float(summary["ema_improvement_fraction"]) > 0.0
    )
    seed_pass = improved_seed_count >= args.min_seeds_improved

    checks = {
        "lag1_correlation": lag_pass,
        "sign_persistence": sign_pass,
        "ema_mae_improvement": ema_pass,
        "seed_direction_consistency": seed_pass,
    }
    return {
        "thresholds": {
            "min_lag_correlation": args.min_lag_correlation,
            "min_sign_agreement": args.min_sign_agreement,
            "max_sign_pvalue": args.max_sign_pvalue,
            "min_ema_improvement": args.min_ema_improvement,
            "min_seeds_improved": args.min_seeds_improved,
        },
        "checks": checks,
        "improved_seed_count": improved_seed_count,
        "overall": bool(all(checks.values())),
    }


def json_safe(value: object) -> object:
    """递归把非有限浮点替换为None，使JSON严格符合标准。"""

    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    # bool是Python中int的子类，必须先判断，否则JSON会把门控结果写成0/1。
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def plot_audit(events: pd.DataFrame, output: Path, primary_min_step: int) -> None:
    """每个seed画truth/CDF和underestimation/causal-EMA两个panel。"""

    seeds = sorted(events["seed"].unique().tolist())
    figure, axes = plt.subplots(len(seeds), 2, figsize=(14, 3.4 * len(seeds)), squeeze=False, constrained_layout=True)

    for row, seed in enumerate(seeds):
        group = events.loc[events["seed"] == seed].sort_values("step")
        x = group["step"] / 1_000_000.0

        # 左侧比较同一批轨迹的真实outage与更新前CDF；post只作为拟合漂移参考。
        axes[row, 0].plot(x, group["truth"], marker="o", linewidth=1.7, label="trajectory truth")
        axes[row, 0].plot(x, group["prediction"], marker="s", linewidth=1.5, label="pre-update CDF")
        if "post_prediction" in group.columns:
            axes[row, 0].plot(x, group["post_prediction"], linewidth=1.1, alpha=0.65, label="post-update CDF")
        axes[row, 0].axvline(primary_min_step / 1_000_000.0, color="#555555", linestyle="--", linewidth=1.0)
        axes[row, 0].set_ylabel(f"seed {int(seed)} probability")
        axes[row, 0].set_ylim(-0.03, max(0.55, float(group[["truth", "prediction"]].max().max()) + 0.05))
        axes[row, 0].legend(frameon=False, ncol=3, fontsize=8)

        # 右侧只把past-only EMA称为prediction；当前underestimation正值代表风险被低估。
        axes[row, 1].axhline(0.0, color="#333333", linewidth=0.9)
        axes[row, 1].plot(x, group["underestimation"], marker="o", linewidth=1.7, label="truth - pre CDF")
        axes[row, 1].plot(x, group["causal_ema_bias_prediction"], marker=".", linewidth=1.4, label="causal EMA prediction")
        axes[row, 1].axvline(primary_min_step / 1_000_000.0, color="#555555", linestyle="--", linewidth=1.0)
        axes[row, 1].set_ylabel(f"seed {int(seed)} calibration bias")
        axes[row, 1].legend(frameon=False, fontsize=8)

    axes[-1, 0].set_xlabel("environment steps (millions)")
    axes[-1, 1].set_xlabel("environment steps (millions)")
    figure.suptitle("Prequential calibration predictability audit (B40 control events)", fontsize=15, fontweight="semibold")
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """执行读取、事件聚合、预注册统计、裁决和图表输出。"""

    args = parse_args()
    if args.event_size <= 0:
        raise ValueError("event-size must be positive")
    if not 0.0 < args.ema_tau <= 1.0:
        raise ValueError("ema-tau must be in (0, 1]")
    if args.robustness_min_step < args.primary_min_step:
        raise ValueError("robustness-min-step must not precede primary-min-step")

    history = pd.read_csv(args.input)
    require_columns(
        history,
        (args.run_column, args.name_column, args.step_column, args.truth_column, args.prediction_column),
    )
    events = aggregate_control_events(history, args)

    # 主区间用于机械gate，较晚区间只检查结论是否被早期成熟阶段驱动。
    primary = summarize_window(events, args.primary_min_step)
    robustness = summarize_window(events, args.robustness_min_step)
    gate = evaluate_gate(primary, args)
    summary = {
        "input_file": args.input.name,
        "event_size": args.event_size,
        "ema_tau": args.ema_tau,
        "bias_definition": "pre_update_truth_minus_pre_update_cdf",
        "primary": primary,
        "robustness": robustness,
        "preregistered_gate": gate,
        "decision": (
            "Calibration bias is predictably persistent; adaptive PID-margin implementation is eligible."
            if gate["overall"]
            else "Calibration bias fails predictability gate; do not feed this signal into PID."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.out_dir / "event_series.csv", index=False)
    (args.out_dir / "audit_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plot_audit(events, args.out_dir / "calibration_audit.png", args.primary_min_step)
    print(json.dumps(json_safe(summary), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
