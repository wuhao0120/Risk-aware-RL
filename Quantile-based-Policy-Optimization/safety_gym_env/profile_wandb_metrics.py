#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 export_wandb_metrics.py 下载的完整 W&B history 转成可比较的本地 profile。

输出不是只抄最后一个点，而是同时保留:
    1. 每个指标的首值/末值/极值/全程均值与非有限值计数；
    2. 在共同训练预算内的 early/middle/late 三段统计；
    3. 每百万环境步的线性趋势，区分“均值高但没有继续学习”和“仍在上升”；
    4. 对齐图 overview.png，检查 reward、constraint、critic、advantage 与 λ 的联动。

典型用法:
    python profile_wandb_metrics.py \
        --input _runs/wandb_export/debug_e1/combined_history.csv \
        --out _runs/profiles/debug_e1

默认按所有选中 run 中最短的最大 env_steps 截断后比较，避免把 150 万步候选方法
与 500 万步 baseline 的最终点直接相减。每个 run 的完整终点仍单独写入 report.md。
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# 这些指标覆盖用户最关心的五条链路：任务回报、风险约束、actor 信号、critic 和探索。
KEY_METRICS: Tuple[str, ...] = (
    "disc_reward/aver_reward",
    "disc_reward/quantile_reward",
    "reward/undisc_return",
    "constraint/empirical_prob",
    "constraint/cdf_estimate_initial",
    "constraint/cdf_estimate_smooth_initial",
    "constraint/cdf_calibration_error",
    "constraint/qr_cdf_estimate_initial",
    "constraint/qr_cdf_estimate_smooth_initial",
    "constraint/qr_cdf_calibration_error",
    "constraint/direct_online_cdf_estimate_initial",
    "constraint/direct_online_cdf_calibration_error",
    "lambda/value",
    "dual/control_error",
    "dual/filtered_error",
    "dual/pid_episode_scale",
    "dual/pid_effective_leak",
    "dual/pid_delta",
    "dual/pid_actual_delta",
    "dual/pid_proportional",
    "dual/pid_output",
    "dual/pid_update_interval",
    "dual/pid_update_due",
    "dual/pid_rollouts_accumulated",
    "dual/pid_update_batch_episodes",
    "dual/pid_update_events",
    "advantage/mean_adv_std",
    "advantage/risk_adv_std",
    "advantage/risk_adv_abs_mean",
    "advantage/risk_adv_nonzero_fraction",
    "advantage/risk_query_target_online_abs_mean",
    "advantage/risk_query_crossfit_peer_abs_mean",
    "advantage/risk_query_preupdate_postupdate_abs_mean",
    "critic/cost_crossfit_peer_abs_mean",
    "norm/return_sigma_ema",
    "norm/constraint_sigma_ema",
    "actor/w_mean",
    "actor/w_std",
    "actor/loss",
    "actor/grad_norm",
    "actor/entropy",
    "reward_value/loss",
    "reward_value/explained_variance",
    "reward_value/grad_norm",
    "reward_value/joint_grad_norm",
    "reward_value/pred_mean",
    "reward_value/target_mean",
    "ppo/ratio_mean",
    "ppo/ratio_std",
    "ppo/first_epoch_ratio_max_error",
    "ppo/clip_fraction",
    "ppo/approx_kl",
    "ppo/target_kl",
    "ppo/early_stop",
    "ppo/update_applied",
    "training/actor_updates_completed",
    "training/actor_updates_per_iteration",
    "training/actor_update_interval",
    "training/actor_update_due",
    "training/actor_rollouts_accumulated",
    "training/actor_batch_trajectories",
    "training/actor_update_events",
    "training/actor_lr",
    "critic/reward_qr_loss",
    "critic/cost_qr_loss",
    "critic/cost_mean_anchor_enabled",
    "critic/cost_mean_anchor_coef",
    "critic/cost_mean_anchor_cost_scale",
    "critic/cost_mean_anchor_scale",
    "critic/cost_mean_anchor_loss",
    "critic/cost_mean_anchor_scaled_loss",
    "critic/cost_target_mean",
    "critic/cost_target_is_mc",
    "critic/cost_objective_loss",
    "critic/cost_update_retained_fraction",
    "critic/cost_holdout_guard_active",
    "critic/cost_holdout_initial_smooth_brier",
    "critic/cost_holdout_best_smooth_brier",
    "critic/cost_holdout_selected_smooth_brier",
    "critic/cost_holdout_selected_hard_brier",
    "critic/cost_holdout_selected_update",
    "critic/cost_holdout_attempted_updates",
    "critic/cost_holdout_stop_update",
    "critic/cost_holdout_early_stopped",
    "critic/cost_holdout_restored",
    "critic/cost_holdout_selected_cdf",
    "critic/cost_holdout_truth",
    "critic/cost_holdout_selected_pred_cost_mean",
    "critic/cost_holdout_truth_cost_mean",
    "critic/cost_s0_aux_enabled",
    "critic/cost_s0_base_scale",
    "critic/cost_s0_aux_scale",
    "critic/cost_s0_aux_loss",
    "critic/cost_s0_aux_scaled_loss",
    "critic/cost_s0_replay_samples",
    "critic/cost_s0_replay_batches",
    "critic/cost_s0_target_mean",
    "critic/cost_direct_cdf_enabled",
    "critic/cost_direct_cdf_loss",
    "critic/cost_direct_cdf_brier",
    "critic/cost_direct_cdf_weighted_brier",
    "critic/cost_direct_cdf_probability_mean",
    "critic/cost_direct_cdf_truth_mean",
    "critic/cost_direct_cdf_bias",
    "critic/cost_direct_cdf_grad_norm",
    "critic/cost_direct_cdf_grad_clip_fraction",
    "critic/cost_direct_cdf_chunked_update",
    "critic/cost_direct_cdf_label_inconsistency_fraction",
    "critic/cost_direct_cdf_query_is_ema",
    "critic/cost_direct_cdf_ema_tau",
    "critic/cost_direct_cdf_ema_online_parameter_abs_mean",
    "critic/s0_holdout_pre_cdf",
    "critic/s0_holdout_pre_truth",
    "critic/s0_holdout_pre_cdf_bias",
    "critic/s0_holdout_pre_cdf_abs_error",
    "critic/s0_holdout_pre_brier",
    "critic/s0_holdout_pre_qr_cdf",
    "critic/s0_holdout_pre_qr_cdf_bias",
    "critic/s0_holdout_pre_qr_cdf_abs_error",
    "critic/s0_holdout_pre_qr_brier",
    "critic/s0_holdout_pre_direct_online_cdf",
    "critic/s0_holdout_pre_direct_online_cdf_bias",
    "critic/s0_holdout_pre_direct_online_cdf_abs_error",
    "critic/s0_holdout_pre_direct_online_brier",
    "critic/s0_holdout_pre_pred_cost_mean",
    "critic/s0_holdout_pre_truth_cost_mean",
    "critic/s0_holdout_pre_cost_mean_bias",
    "critic/s0_holdout_post_cdf",
    "critic/s0_holdout_post_truth",
    "critic/s0_holdout_post_cdf_bias",
    "critic/s0_holdout_post_cdf_abs_error",
    "critic/s0_holdout_post_brier",
    "critic/s0_holdout_post_qr_cdf",
    "critic/s0_holdout_post_qr_cdf_bias",
    "critic/s0_holdout_post_qr_cdf_abs_error",
    "critic/s0_holdout_post_qr_brier",
    "critic/s0_holdout_post_direct_online_cdf",
    "critic/s0_holdout_post_direct_online_cdf_bias",
    "critic/s0_holdout_post_direct_online_cdf_abs_error",
    "critic/s0_holdout_post_direct_online_brier",
    "critic/s0_holdout_post_pred_cost_mean",
    "critic/s0_holdout_post_truth_cost_mean",
    "critic/s0_holdout_post_cost_mean_bias",
    "critic/cost_time_weighted",
    "critic/cost_time_weight_min",
    "critic/cost_time_weight_max",
    "critic/cost_time_weight_ess_fraction",
    "debug/cost_critic_time_weighted",
    "debug/cost_critic_weight_discount",
    "debug/cost_critic_weight_floor",
    "debug/cost_actor_query_is_target",
    "debug/cost_cdf_estimator_is_direct",
    "debug/cost_direct_cdf_query_is_ema",
    "critic/quantile_target_scale",
    "critic/quantile_target_reference",
    "critic/cost_iqn_tau_mean",
    "critic/cost_iqn_tau_std",
    "critic/cost_iqn_tau_min",
    "critic/cost_iqn_tau_max",
    "debug/cost_distribution_is_iqn",
    "debug/cost_quantile_output_is_exp",
    "debug/cost_quantile_output_is_softplus",
    "debug/cost_quantile_output_scale",
    "debug/cost_iqn_train_quantiles",
    "debug/cost_iqn_query_quantiles",
    "debug/cost_iqn_cosines",
    "debug/cost_quantile_grid_is_query",
    "debug/cost_quantile_prediction_is_importance",
    "debug/cost_quantile_local_count",
    "debug/cost_quantile_weight_min",
    "debug/cost_quantile_weight_max",
    "critic/reward_grad_norm",
    "critic/cost_head_grad_norm",
    "critic/cost_history_grad_norm",
    "critic/cost_grad_norm",
    "critic/joint_grad_norm",
    "critic/grad_clip_fraction",
    "critic/update_count",
    "critic/update_grad_clip_fraction",
    "critic/cost_grad_norm_first",
    "critic/cost_grad_norm_mean",
    "critic/cost_grad_norm_max",
    "critic/cost_grad_norm_last",
    "critic/cost_head_grad_norm_first",
    "critic/cost_head_grad_norm_mean",
    "critic/cost_head_grad_norm_max",
    "critic/cost_head_grad_norm_last",
    "critic/cost_history_grad_norm_first",
    "critic/cost_history_grad_norm_mean",
    "critic/cost_history_grad_norm_max",
    "critic/cost_history_grad_norm_last",
    "critic/actor_feature_refresh_enabled",
    "critic/actor_feature_refresh_abs_mean",
    "critic/actor_feature_refresh_applied",
    "critic/actor_feature_refresh_count",
    "critic/joint_grad_norm_first",
    "critic/joint_grad_norm_mean",
    "critic/joint_grad_norm_max",
    "critic/joint_grad_norm_last",
    "critic/reward_grad_norm_mean",
    "critic/cost_qr_loss_first",
    "critic/cost_qr_loss_mean",
    "critic/cost_qr_loss_min",
    "critic/cost_qr_loss_last",
    "critic/cost_mean_anchor_loss_first",
    "critic/cost_mean_anchor_loss_mean",
    "critic/cost_mean_anchor_loss_min",
    "critic/cost_mean_anchor_loss_last",
    "critic/cost_objective_loss_first",
    "critic/cost_objective_loss_mean",
    "critic/cost_objective_loss_min",
    "critic/cost_objective_loss_last",
    "critic/q_mean",
    "critic/pred_cost_mean",
    "critic/pred_cost_std",
    "critic/cost_quantile_crossing_fraction",
    "debug/cost_mean",
    "action/avg_step_reward",
    "action/avg_step_cost",
    "ref/r_value_loss",
    "ref/pi_r_loss",
    "ref/entropy",
)


# 终点评估只出现一次，不能用阶段均值；因此单独从每列最后一个非空值提取。
EVAL_METRICS: Tuple[str, ...] = (
    "eval/mean",
    "eval/reward_std",
    "eval/empirical_prob",
    "eval/quantile_return",
    "eval/constraint_margin",
    "eval/cost_cdf_initial",
    "eval/cost_cdf_smooth_initial",
    "eval/cost_cdf_brier_initial",
    "eval/cost_cdf_brier_skill_initial",
    "eval/cost_cdf_roc_auc_initial",
    "eval/cost_cdf_discrimination_gap_initial",
    "eval/cost_cdf_prediction_std_initial",
    "eval/cost_cdf_smooth_brier_initial",
    "eval/cost_cdf_smooth_brier_skill_initial",
    "eval/cost_cdf_smooth_roc_auc_initial",
    "eval/cost_cdf_smooth_discrimination_gap_initial",
    "eval/cost_cdf_smooth_prediction_std_initial",
    "eval/cost_cdf_qr_initial",
    "eval/cost_cdf_qr_smooth_initial",
    "eval/cost_cdf_qr_brier_initial",
    "eval/cost_cdf_direct_online_initial",
    "eval/cost_cdf_direct_online_brier_initial",
    "eval/pred_cost_mean",
    "eval/pred_cost_std",
    "eval/cost_quantile_crossing_fraction",
)


# 每个 panel 可叠加同一机制的相关指标；缺列时自动跳过，不要求所有算法日志键完全一致。
PLOT_PANELS: Tuple[Tuple[str, Tuple[str, ...], bool], ...] = (
    ("discounted reward", ("disc_reward/aver_reward",), False),
    ("reward quantile", ("disc_reward/quantile_reward",), False),
    ("constraint probability", (
        "constraint/empirical_prob",
        "constraint/cdf_estimate_initial",
        "constraint/qr_cdf_estimate_initial",
        "constraint/direct_online_cdf_estimate_initial"), False),
    ("dual lambda", ("lambda/value",), False),
    ("reward advantage scale", ("advantage/mean_adv_std", "norm/return_sigma_ema"), True),
    ("actor weight scale", ("actor/w_std",), True),
    ("reward critic", ("critic/reward_qr_loss", "ref/r_value_loss"), True),
    ("cost critic", ("critic/cost_qr_loss", "critic/cost_s0_aux_loss"), True),
    ("s0 holdout probability", (
        "critic/s0_holdout_pre_cdf",
        "critic/s0_holdout_pre_qr_cdf",
        "critic/s0_holdout_pre_direct_online_cdf",
        "critic/s0_holdout_post_cdf",
        "critic/s0_holdout_post_qr_cdf",
        "critic/s0_holdout_pre_truth"), False),
    ("s0 holdout CDF absolute error", (
        "critic/s0_holdout_pre_cdf_abs_error",
        "critic/s0_holdout_pre_qr_cdf_abs_error",
        "critic/s0_holdout_pre_direct_online_cdf_abs_error",
        "critic/s0_holdout_post_cdf_abs_error",
        "critic/s0_holdout_post_qr_cdf_abs_error"), False),
    ("s0 holdout Brier", (
        "critic/s0_holdout_pre_brier",
        "critic/s0_holdout_pre_qr_brier",
        "critic/s0_holdout_pre_direct_online_brier",
        "critic/s0_holdout_post_brier",
        "critic/s0_holdout_post_qr_brier"), False),
    ("predicted / sampled cost", ("critic/pred_cost_mean", "debug/cost_mean"), False),
    ("reward value", ("reward_value/loss", "reward_value/explained_variance"), True),
    ("PPO health", ("ppo/clip_fraction", "ppo/approx_kl"), True),
    ("PPO epochs", (
        "training/actor_updates_completed",
        "training/actor_updates_per_iteration",
        "ppo/early_stop"), False),
    ("actor cadence", (
        "training/actor_update_due",
        "training/actor_rollouts_accumulated",
        "training/actor_batch_trajectories"), False),
)


def utc_now() -> str:
    """返回带时区的稳定时间戳，写入报告头以便追溯。"""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def finite_series(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    取一个指标的有限数值行，并按 env_steps 排序。

    W&B history 是稀疏表：某些行没有某个 metric。这里不能直接把 NaN 填 0，
    否则 loss/advantage 的阶段均值会被大量不存在的记录错误压低。
    """

    if metric not in frame.columns or "progress/env_steps" not in frame.columns:
        return pd.DataFrame(columns=["progress/env_steps", metric])
    x = pd.to_numeric(frame["progress/env_steps"], errors="coerce")
    y = pd.to_numeric(frame[metric], errors="coerce")
    mask = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
    return pd.DataFrame({"progress/env_steps": x[mask], metric: y[mask]}).sort_values(
        "progress/env_steps"
    )


def safe_float(value: object) -> Optional[float]:
    """把 numpy/pandas 标量转成 JSON 安全 float；非有限数返回 None。"""

    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def describe_values(values: np.ndarray) -> Dict[str, Optional[float]]:
    """计算一组有限值的基本统计；空数组保留 None，而不是制造 0。"""

    if values.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": safe_float(values.mean()),
        "std": safe_float(values.std(ddof=0)),
        "min": safe_float(values.min()),
        "max": safe_float(values.max()),
    }


def segment_values(values: np.ndarray, start: float, end: float) -> np.ndarray:
    """按记录位置切片；用于 early/middle/late，避免不同 env_steps 间隔造成权重偏差。"""

    n = values.size
    if n == 0:
        return values
    lo = min(n - 1, int(math.floor(start * n)))
    hi = max(lo + 1, int(math.ceil(end * n)))
    return values[lo:min(n, hi)]


def metric_profile(frame: pd.DataFrame, metric: str, common_steps: float) -> Dict[str, object]:
    """
    生成单 run/单 metric 的完整与共同预算统计。

    趋势斜率以“metric / 1e6 env_steps”为单位，便于不同 B/T 配置之间直观比较。
    """

    raw_numeric = pd.to_numeric(frame.get(metric, pd.Series(dtype=float)), errors="coerce")
    nonfinite = int((raw_numeric.notna() & ~np.isfinite(raw_numeric.to_numpy(dtype=float))).sum())
    valid = finite_series(frame, metric)
    matched = valid[valid["progress/env_steps"] <= common_steps]
    values = matched[metric].to_numpy(dtype=float)
    steps = matched["progress/env_steps"].to_numpy(dtype=float)

    slope_per_million: Optional[float] = None
    if values.size >= 2 and float(np.ptp(steps)) > 0.0:
        # 先把 x 缩放到百万步，避免 polyfit 在 1e6 量级横轴上产生不必要的病态数值。
        slope_per_million = safe_float(np.polyfit(steps / 1e6, values, deg=1)[0])

    full_values = valid[metric].to_numpy(dtype=float)
    full_steps = valid["progress/env_steps"].to_numpy(dtype=float)
    return {
        "metric": metric,
        "raw_non_null": int(raw_numeric.notna().sum()),
        "nonfinite": nonfinite,
        "matched_n": int(values.size),
        "full_n": int(full_values.size),
        "first": safe_float(values[0]) if values.size else None,
        "last_matched": safe_float(values[-1]) if values.size else None,
        "last_full": safe_float(full_values[-1]) if full_values.size else None,
        "last_full_env_steps": safe_float(full_steps[-1]) if full_steps.size else None,
        "matched": describe_values(values),
        "early_20pct": describe_values(segment_values(values, 0.0, 0.2)),
        "middle_20pct": describe_values(segment_values(values, 0.4, 0.6)),
        "late_20pct": describe_values(segment_values(values, 0.8, 1.0)),
        "slope_per_million_steps": slope_per_million,
    }


def select_runs(frame: pd.DataFrame, patterns: Sequence[str]) -> pd.DataFrame:
    """按 shell glob 选择 run；未传 pattern 时保留 CSV 中全部 run。"""

    if not patterns:
        return frame
    names = frame["run_name"].fillna("").astype(str)
    keep = names.map(lambda name: any(fnmatch.fnmatch(name, pattern) for pattern in patterns))
    return frame[keep].copy()


def last_non_null(frame: pd.DataFrame, metric: str) -> Optional[float]:
    """提取评估列的最后一个有限值；W&B 最后一行可能只有部分 eval 键。"""

    valid = finite_series(frame, metric)
    return safe_float(valid[metric].iloc[-1]) if not valid.empty else None


def write_json(path: Path, value: object) -> None:
    """统一 JSON 序列化格式，便于后续脚本继续消费 profile。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fmt(value: object, digits: int = 4) -> str:
    """Markdown 表格中的紧凑数值格式。"""

    number = safe_float(value)
    return "—" if number is None else f"{number:.{digits}g}"


def flatten_profiles(profiles: Dict[str, Dict[str, Dict[str, object]]]) -> pd.DataFrame:
    """把嵌套 profile 展平成一行一个 run/metric，便于 pandas 筛选和画二次图。"""

    rows: List[Dict[str, object]] = []
    for run_name, metrics in profiles.items():
        for metric, stats in metrics.items():
            row: Dict[str, object] = {"run_name": run_name, "metric": metric}
            for key in ("matched_n", "full_n", "nonfinite", "first", "last_matched", "last_full",
                        "last_full_env_steps", "slope_per_million_steps"):
                row[key] = stats.get(key)
            for segment in ("matched", "early_20pct", "middle_20pct", "late_20pct"):
                values = stats.get(segment, {})
                if isinstance(values, dict):
                    for stat_name, value in values.items():
                        row[f"{segment}/{stat_name}"] = value
            rows.append(row)
    return pd.DataFrame(rows)


def plot_overview(frame: pd.DataFrame, out_path: Path, common_steps: float, smooth_window: int) -> None:
    """绘制共同预算内的动态网格对齐图；使用无 GUI Agg 后端，适合 SSH/headless 节点。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_names = list(dict.fromkeys(frame["run_name"].astype(str).tolist()))
    ncols = 3
    nrows = int(math.ceil(len(PLOT_PANELS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.3 * nrows), constrained_layout=True)
    for axis, (title, metrics, log_y) in zip(axes.flat, PLOT_PANELS):
        drew_line = False
        for run_name in run_names:
            run_frame = frame[frame["run_name"].astype(str) == run_name]
            for metric in metrics:
                valid = finite_series(run_frame, metric)
                valid = valid[valid["progress/env_steps"] <= common_steps]
                if valid.empty:
                    continue
                x = valid["progress/env_steps"].to_numpy(dtype=float) / 1e6
                y = valid[metric].to_numpy(dtype=float)
                # 原始曲线淡画、rolling mean 实画；短 run 自动缩小窗口。
                axis.plot(x, y, alpha=0.12, linewidth=0.8)
                window = max(1, min(smooth_window, len(y)))
                smoothed = pd.Series(y).rolling(window, min_periods=1).mean().to_numpy()
                axis.plot(x, smoothed, linewidth=1.8, label=f"{run_name} | {metric}")
                drew_line = True
        if title == "constraint probability":
            axis.axhline(0.2, color="black", linestyle="--", linewidth=1.0, label="alpha=0.2")
        if log_y and drew_line:
            # loss/scale 可能偶尔为 0；symlog 比直接 log 更稳健，也不会静默丢点。
            axis.set_yscale("symlog", linthresh=1e-5)
        axis.set_title(title)
        axis.set_xlabel("environment steps (million)")
        axis.grid(alpha=0.2)
        if drew_line:
            axis.legend(fontsize=6)
    for unused_axis in axes.flat[len(PLOT_PANELS):]:
        unused_axis.set_visible(False)
    fig.suptitle(f"Safety-Gym W&B profile (matched budget ≤ {common_steps:,.0f} env steps)", fontsize=15)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def build_report(
    frame: pd.DataFrame,
    profiles: Dict[str, Dict[str, Dict[str, object]]],
    evals: Dict[str, Dict[str, Optional[float]]],
    common_steps: float,
    input_path: Path,
) -> str:
    """生成面向人工审阅的 Markdown；所有原始详细值仍保留在 CSV/JSON。"""

    lines = [
        "# Safety-Gym W&B training profile",
        "",
        f"- 生成时间（UTC）：{utc_now()}",
        f"- 输入：`{input_path}`",
        f"- 公平对齐预算：`{common_steps:,.0f}` environment steps",
        "- 阶段定义：对齐预算内有效记录的前 20% / 中间 20% / 后 20%",
        "",
        "## Run coverage",
        "",
        "| run | rows | max env steps | state |",
        "|---|---:|---:|---|",
    ]
    for run_name, run_frame in frame.groupby("run_name", sort=False):
        max_steps = pd.to_numeric(run_frame["progress/env_steps"], errors="coerce").max()
        state = run_frame["run_state"].dropna().astype(str)
        lines.append(f"| {run_name} | {len(run_frame)} | {fmt(max_steps, 8)} | {state.iloc[-1] if len(state) else '—'} |")

    # 关键表只使用 late 20% 的均值；单个最后点波动较大，不适合作主结论。
    lines.extend([
        "",
        "## Matched-budget late-stage summary",
        "",
        "| run | reward late mean | reward slope / 1M | reward quantile | empirical P | lambda | reward adv std | actor w std |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for run_name, run_profiles in profiles.items():
        def late(metric: str) -> object:
            return run_profiles.get(metric, {}).get("late_20pct", {}).get("mean")  # type: ignore[union-attr]

        reward_stats = run_profiles.get("disc_reward/aver_reward", {})
        lines.append(
            f"| {run_name} | {fmt(late('disc_reward/aver_reward'))} | "
            f"{fmt(reward_stats.get('slope_per_million_steps'))} | "
            f"{fmt(late('disc_reward/quantile_reward'))} | "
            f"{fmt(late('constraint/empirical_prob'))} | {fmt(late('lambda/value'))} | "
            f"{fmt(late('advantage/mean_adv_std'))} | {fmt(late('actor/w_std'))} |"
        )

    lines.extend([
        "",
        "## Full-run terminal evaluation",
        "",
        "这里保留各 run 自己完整预算结束后的 eval；不同预算时只用于诊断，不能替代上面的 matched-budget 比较。",
        "",
        "| run | eval mean | reward std | empirical P | quantile | margin | critic CDF |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for run_name, values in evals.items():
        lines.append(
            f"| {run_name} | {fmt(values.get('eval/mean'))} | {fmt(values.get('eval/reward_std'))} | "
            f"{fmt(values.get('eval/empirical_prob'))} | {fmt(values.get('eval/quantile_return'))} | "
            f"{fmt(values.get('eval/constraint_margin'))} | {fmt(values.get('eval/cost_cdf_initial'))} |"
        )

    nonfinite_rows = []
    for run_name, run_profiles in profiles.items():
        for metric, stats in run_profiles.items():
            if int(stats.get("nonfinite", 0)) > 0:
                nonfinite_rows.append((run_name, metric, stats["nonfinite"]))
    lines.extend(["", "## Numerical health", ""])
    if nonfinite_rows:
        lines.append("检测到以下非有限值：")
        lines.append("")
        for run_name, metric, count in nonfinite_rows:
            lines.append(f"- `{run_name}` / `{metric}`: {count}")
    else:
        lines.append("所选关键指标未检测到 NaN/Inf 数值记录。")

    lines.extend([
        "",
        "## Files",
        "",
        "- `metric_profile.csv`：一行一个 run/metric 的可筛选统计",
        "- `profile.json`：完整结构化统计与终点评估",
        "- `overview.png`：共同预算内对齐曲线（淡线为原始值，实线为 rolling mean）",
        "",
    ])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """定义 CLI；--run 可重复传 shell glob，用于从大项目导出中选少量实验。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="combined_history.csv 或其所在导出目录")
    parser.add_argument("--out", required=True, help="profile 输出目录")
    parser.add_argument("--run", action="append", default=[], help="run_name shell glob；可重复")
    parser.add_argument("--common-steps", type=float, default=None, help="手工指定共同训练预算")
    parser.add_argument("--smooth-window", type=int, default=10, help="overview rolling mean 窗口")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """加载 history、确定共同预算、生成 CSV/JSON/Markdown/PNG 四类产物。"""

    args = build_parser().parse_args(argv)
    input_path = Path(args.input).resolve()
    if input_path.is_dir():
        input_path = input_path / "combined_history.csv"
    if not input_path.exists():
        raise SystemExit(f"input not found: {input_path}")

    frame = pd.read_csv(input_path, low_memory=False)
    required = {"run_name", "progress/env_steps"}
    missing = required.difference(frame.columns)
    if missing:
        raise SystemExit(f"history is missing required columns: {sorted(missing)}")
    frame = select_runs(frame, args.run)
    if frame.empty:
        raise SystemExit("no runs matched --run filters")

    # 每个 run 的最大有限 env_steps；默认取最小值作为公平比较预算。
    max_steps_by_run = frame.groupby("run_name")["progress/env_steps"].apply(
        lambda col: pd.to_numeric(col, errors="coerce").max()
    )
    inferred_common = float(max_steps_by_run.min())
    common_steps = float(args.common_steps) if args.common_steps is not None else inferred_common
    if not math.isfinite(common_steps) or common_steps <= 0:
        raise SystemExit(f"invalid common env-step budget: {common_steps}")

    metrics = [metric for metric in KEY_METRICS if metric in frame.columns]
    profiles: Dict[str, Dict[str, Dict[str, object]]] = {}
    evals: Dict[str, Dict[str, Optional[float]]] = {}
    for run_name, run_frame in frame.groupby("run_name", sort=False):
        profiles[str(run_name)] = {
            metric: metric_profile(run_frame, metric, common_steps) for metric in metrics
        }
        evals[str(run_name)] = {
            metric: last_non_null(run_frame, metric) for metric in EVAL_METRICS if metric in frame.columns
        }

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    flat = flatten_profiles(profiles)
    flat.to_csv(out_dir / "metric_profile.csv", index=False)
    write_json(
        out_dir / "profile.json",
        {
            "generated_at": utc_now(),
            "input": str(input_path),
            "common_steps": common_steps,
            "max_steps_by_run": {str(k): safe_float(v) for k, v in max_steps_by_run.items()},
            "profiles": profiles,
            "terminal_evaluation": evals,
        },
    )
    (out_dir / "report.md").write_text(
        build_report(frame, profiles, evals, common_steps, input_path), encoding="utf-8"
    )
    plot_overview(frame, out_dir / "overview.png", common_steps, max(1, args.smooth_window))
    print(f"profiled {len(profiles)} runs at common_steps={common_steps:,.0f} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
