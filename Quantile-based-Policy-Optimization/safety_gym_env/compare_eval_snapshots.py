#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较两份独立评估 JSON，并生成统一的统计表、机器可读结果和论文风格图。

这个工具适用于以下两类实验：
    1. 同一 run 的 checkpoint 更新前/更新后比较；
    2. 两个算法或超参数配置在相同评估协议下的比较。

统计口径：
    - 单组 reward：正态近似 95% 置信区间；
    - reward 差：Welch 标准误的 95% 置信区间；
    - 单组 outage：Wilson score 95% 置信区间；
    - outage 差：Newcombe-Wilson 95% 置信区间；
    - critic 校准：总体 CDF error、Brier、Brier Skill、mean-cost error。

典型用法：
    python compare_eval_snapshots.py \
        --before _runs/pre_eval520.json \
        --after _runs/post_eval520.json \
        --before-label pre-update \
        --after-label post-update \
        --out-dir _runs/profiles/example_pre_post \
        --title "DQCAC checkpoint pre/post comparison"
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


# 统一使用双侧95%区间对应的标准正态分位数，与既有实验报告保持一致。
Z_95 = 1.959963984540054


def parse_args() -> argparse.Namespace:
    """
    [功能] 解析两份评估结果、显示标签和输出目录。
    [返回] argparse.Namespace；所有路径在进入主逻辑后再解析，便于报错定位。
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True, help="基准/更新前评估JSON")
    parser.add_argument("--after", type=Path, required=True, help="候选/更新后评估JSON")
    parser.add_argument("--before-label", default="before", help="图表中的基准标签")
    parser.add_argument("--after-label", default="after", help="图表中的候选标签")
    parser.add_argument("--out-dir", type=Path, required=True, help="CSV/JSON/PNG输出目录")
    parser.add_argument("--title", default="Independent evaluation comparison", help="图标题")
    parser.add_argument("--constraint-alpha", type=float, default=0.20, help="约束概率参考线")
    return parser.parse_args()


def load_eval(path: Path) -> Dict[str, object]:
    """
    [功能] 读取 run_experiment.py 生成的评估 JSON，并验证核心字段。
    [输入] path：本地JSON路径。
    [返回] 完整字典；调用者从其中的eval和summary读取结果。
    """

    # 明确使用UTF-8，避免中文tag或环境名在不同节点上按locale解码失败。
    payload = json.loads(path.read_text(encoding="utf-8"))

    # 这些字段决定置信区间和主要结论，缺失时不能静默填0。
    required = ("mean", "reward_std", "empirical_prob", "num_episodes")
    missing = [key for key in required if key not in payload.get("eval", {})]
    if missing:
        raise KeyError(f"{path} is missing eval fields: {missing}")

    return payload


def wilson_interval(events: int, episodes: int, z: float = Z_95) -> Tuple[float, float]:
    """
    [功能] 计算二项比例的Wilson score区间。
    [输入] events为超限次数，episodes为独立评估轨迹数。
    [返回] (lower, upper)，均限制在[0,1]。
    """

    if episodes <= 0:
        raise ValueError("episodes must be positive")

    # Wilson区间比p±1.96*SE在小概率或有限样本下更稳定，不会越出[0,1]。
    probability = events / episodes
    denominator = 1.0 + z * z / episodes
    center = (probability + z * z / (2.0 * episodes)) / denominator

    # 根号项同时包含经验Bernoulli方差和有限样本修正。
    radius = z * math.sqrt(
        probability * (1.0 - probability) / episodes
        + z * z / (4.0 * episodes * episodes)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def newcombe_difference_interval(
    before_events: int,
    before_n: int,
    after_events: int,
    after_n: int,
) -> Tuple[float, float]:
    """
    [功能] 用Newcombe方法计算两个独立二项比例之差(after-before)的95%区间。
    [原理] 分别计算两个Wilson区间，再按不对称距离组合，而非把轨迹合并。
    """

    before_p = before_events / before_n
    after_p = after_events / after_n
    before_low, before_high = wilson_interval(before_events, before_n)
    after_low, after_high = wilson_interval(after_events, after_n)

    # Newcombe interval保留Wilson区间的不对称性；差为负表示after更安全。
    difference = after_p - before_p
    lower = difference - math.sqrt((after_p - after_low) ** 2 + (before_high - before_p) ** 2)
    upper = difference + math.sqrt((after_high - after_p) ** 2 + (before_p - before_low) ** 2)
    return lower, upper


def reward_difference_interval(
    before_mean: float,
    before_std: float,
    before_n: int,
    after_mean: float,
    after_std: float,
    after_n: int,
) -> Tuple[float, float]:
    """
    [功能] 计算独立评估样本均值差(after-before)的Welch-t 95%区间。
    [说明] 使用Welch-Satterthwaite自由度，精确复现既有实验报告的统计口径。
    """

    difference = after_mean - before_mean

    # Welch方差项允许两个checkpoint的reward方差不同，不做等方差假设。
    before_variance_term = before_std ** 2 / before_n
    after_variance_term = after_std ** 2 / after_n
    standard_error = math.sqrt(before_variance_term + after_variance_term)

    # Satterthwaite近似给出有效自由度；520回合时与正态临界值很接近但不完全相同。
    degrees_of_freedom = (before_variance_term + after_variance_term) ** 2 / (
        before_variance_term ** 2 / (before_n - 1)
        + after_variance_term ** 2 / (after_n - 1)
    )
    critical_value = float(student_t.ppf(0.975, degrees_of_freedom))
    return difference - critical_value * standard_error, difference + critical_value * standard_error


def safe_fractional_change(after: float, before: float) -> float:
    """返回(after-before)/abs(before)；基准为0时返回NaN而不伪造无限改善。"""

    return (after - before) / abs(before) if before != 0.0 else float("nan")


def summarize(payload: Mapping[str, object], label: str, source: Path) -> Dict[str, object]:
    """
    [功能] 把稀疏评估JSON变成一行完整、可直接比较的统计量。
    [输出] dict中的数值会写入comparison.csv并用于画图。
    """

    evaluation = payload["eval"]
    assert isinstance(evaluation, Mapping)

    # empirical_prob来自完整评估轨迹；用round恢复事件计数并检查可逆性。
    episodes = int(evaluation["num_episodes"])
    outage = float(evaluation["empirical_prob"])
    events = int(round(outage * episodes))
    if not math.isclose(events / episodes, outage, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{source}: empirical_prob is inconsistent with num_episodes")

    # reward区间描述单个checkpoint的均值不确定性，而不是seed间方差。
    reward = float(evaluation["mean"])
    reward_std = float(evaluation["reward_std"])
    reward_radius = Z_95 * reward_std / math.sqrt(episodes)
    outage_low, outage_high = wilson_interval(events, episodes)

    # critic总体CDF必须与同一批真实outage比较；平滑CDF字段缺失时退化为hard CDF。
    hard_cdf = float(evaluation.get("cost_cdf_initial", float("nan")))
    smooth_cdf = float(evaluation.get("cost_cdf_smooth_initial", hard_cdf))
    brier = float(evaluation.get("cost_cdf_brier_initial", float("nan")))

    # climatology是只输出该评估集基率的常数预测；BSS>0才优于这个朴素基线。
    climatology_brier = outage * (1.0 - outage)
    brier_skill = 1.0 - brier / climatology_brier if climatology_brier > 0.0 else float("nan")
    true_mean_cost = float(evaluation.get("cost_undisc_mean", evaluation.get("cost_disc_mean", float("nan"))))
    predicted_mean_cost = float(evaluation.get("pred_cost_mean", float("nan")))

    return {
        "label": label,
        "reward": reward,
        "reward_std": reward_std,
        "reward_ci95_low": reward - reward_radius,
        "reward_ci95_high": reward + reward_radius,
        "outage": outage,
        "events": events,
        "episodes": episodes,
        "outage_wilson95_low": outage_low,
        "outage_wilson95_high": outage_high,
        "hard_cdf": hard_cdf,
        "hard_cdf_error": abs(hard_cdf - outage),
        "smooth_cdf": smooth_cdf,
        "smooth_cdf_error": abs(smooth_cdf - outage),
        "brier": brier,
        "climatology_brier": climatology_brier,
        "brier_skill": brier_skill,
        "true_mean_cost": true_mean_cost,
        "pred_mean_cost": predicted_mean_cost,
        "mean_cost_error": abs(predicted_mean_cost - true_mean_cost),
        "crossing": float(evaluation.get("cost_quantile_crossing_fraction", float("nan"))),
        "cost_quantile_80": float(evaluation.get("cost_quantile", float("nan"))),
        # 结果只记录文件名：足以回溯同目录原始数据，同时避免复制绝对目录信息。
        "source_json": source.name,
    }


def build_statistics(before: Dict[str, object], after: Dict[str, object]) -> Dict[str, object]:
    """构造两行结果之间的差值、置信区间和校准变化，写入statistics.json。"""

    reward_ci = reward_difference_interval(
        float(before["reward"]), float(before["reward_std"]), int(before["episodes"]),
        float(after["reward"]), float(after["reward_std"]), int(after["episodes"]),
    )
    outage_ci = newcombe_difference_interval(
        int(before["events"]), int(before["episodes"]),
        int(after["events"]), int(after["episodes"]),
    )

    # fractional_change为正表示误差恶化，负表示误差改善。
    return {
        "comparison_direction": "after_minus_before",
        "reward_difference": float(after["reward"]) - float(before["reward"]),
        "reward_difference_welch95": list(reward_ci),
        "outage_difference": float(after["outage"]) - float(before["outage"]),
        "outage_difference_newcombe95": list(outage_ci),
        "hard_cdf_error_fractional_change": safe_fractional_change(
            float(after["hard_cdf_error"]), float(before["hard_cdf_error"])
        ),
        "smooth_cdf_error_fractional_change": safe_fractional_change(
            float(after["smooth_cdf_error"]), float(before["smooth_cdf_error"])
        ),
        "mean_cost_error_fractional_change": safe_fractional_change(
            float(after["mean_cost_error"]), float(before["mean_cost_error"])
        ),
        "brier_fractional_change": safe_fractional_change(float(after["brier"]), float(before["brier"])),
        "brier_skill_change": float(after["brier_skill"]) - float(before["brier_skill"]),
        "crossing_change": float(after["crossing"]) - float(before["crossing"]),
    }


def plot_comparison(frame: pd.DataFrame, statistics: Mapping[str, object], output: Path, title: str, alpha: float) -> None:
    """画reward/outage、CDF校准、Brier Skill和mean cost四组证据。"""

    labels = frame["label"].tolist()
    colors = ["#4C78A8", "#E45756"]
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)

    # Panel 1：reward与其单checkpoint 95%区间。
    reward_error = np.vstack([
        frame["reward"] - frame["reward_ci95_low"],
        frame["reward_ci95_high"] - frame["reward"],
    ])
    axes[0, 0].bar(labels, frame["reward"], color=colors, alpha=0.88)
    axes[0, 0].errorbar(labels, frame["reward"], yerr=reward_error, fmt="none", color="black", capsize=5)
    axes[0, 0].set_title(f"Reward (after-before={statistics['reward_difference']:+.3f})")
    axes[0, 0].set_ylabel("mean discounted reward")

    # Panel 2：outage与Wilson区间，虚线为论文目标alpha。
    outage_error = np.vstack([
        frame["outage"] - frame["outage_wilson95_low"],
        frame["outage_wilson95_high"] - frame["outage"],
    ])
    axes[0, 1].bar(labels, frame["outage"], color=colors, alpha=0.88)
    axes[0, 1].errorbar(labels, frame["outage"], yerr=outage_error, fmt="none", color="black", capsize=5)
    axes[0, 1].axhline(alpha, color="#222222", linestyle="--", linewidth=1.3, label=f"alpha={alpha:.2f}")
    axes[0, 1].set_title(f"Outage (after-before={statistics['outage_difference']:+.3f})")
    axes[0, 1].set_ylabel("empirical outage probability")
    axes[0, 1].legend(frameon=False)

    # Panel 3：预测CDF与真实比例并排，直接显示更新后是否发生反向漂移。
    x = np.arange(len(labels))
    width = 0.25
    axes[1, 0].bar(x - width, frame["outage"], width, label="truth", color="#72B7B2")
    axes[1, 0].bar(x, frame["hard_cdf"], width, label="hard CDF", color="#F58518")
    axes[1, 0].bar(x + width, frame["smooth_cdf"], width, label="smooth CDF", color="#B279A2")
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set_title("Critic probability calibration")
    axes[1, 0].set_ylabel("probability")
    axes[1, 0].legend(frameon=False)

    # Panel 4：BSS和mean-cost absolute error量纲不同，使用双y轴避免视觉压扁。
    bss_axis = axes[1, 1]
    error_axis = bss_axis.twinx()
    bss_axis.bar(x - 0.18, frame["brier_skill"] * 100.0, 0.36, color="#54A24B", label="Brier Skill")
    error_axis.bar(x + 0.18, frame["mean_cost_error"], 0.36, color="#ECA82C", label="mean-cost error")
    bss_axis.axhline(0.0, color="#555555", linestyle=":", linewidth=1.0)
    bss_axis.set_xticks(x, labels)
    bss_axis.set_ylabel("Brier Skill Score (%)", color="#38752F")
    error_axis.set_ylabel("absolute mean-cost error", color="#9C6910")
    bss_axis.set_title("Proper score and distribution mean")
    bss_axis.legend(loc="upper left", frameon=False)
    error_axis.legend(loc="upper right", frameon=False)

    figure.suptitle(title, fontsize=15, fontweight="semibold")
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """执行读取、验证、统计、序列化和绘图的完整流程。"""

    args = parse_args()
    if not 0.0 < args.constraint_alpha < 1.0:
        raise ValueError("constraint-alpha must be in (0, 1)")

    # 输出目录显式创建在用户指定位置；工具不会写入root磁盘或默认缓存目录。
    args.out_dir.mkdir(parents=True, exist_ok=True)
    before_payload = load_eval(args.before)
    after_payload = load_eval(args.after)

    # 使用命令行传入的路径原样记录；调用时应优先给相对路径，便于迁移且不泄露目录。
    before = summarize(before_payload, args.before_label, args.before)
    after = summarize(after_payload, args.after_label, args.after)
    comparison = pd.DataFrame([before, after])
    statistics = build_statistics(before, after)

    # CSV便于人工检查，JSON便于后续门控/汇总脚本直接读取。
    comparison.to_csv(args.out_dir / "comparison.csv", index=False)
    (args.out_dir / "statistics.json").write_text(
        json.dumps(statistics, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plot_comparison(comparison, statistics, args.out_dir / "comparison.png", args.title, args.constraint_alpha)

    print(json.dumps(statistics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
