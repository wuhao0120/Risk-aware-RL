#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DQC-AC 超参数扫描脚本
=====================
并行训练多组参数组合 → 评估(10000 episodes) → JSON保存完整参数与结果

Usage:
    python run_sweep.py                           # 运行所有实验, 默认并行5个
    python run_sweep.py --max-parallel 2          # 最多2个并行 (显存紧张时)
    python run_sweep.py --exp exp1_nq200_emb256   # 只运行指定实验
    python run_sweep.py --list                    # 列出所有可用实验
    python run_sweep.py --max-episode 10000       # 快速测试 (缩短训练轮数)
"""

import os
import sys
import json
import time
import random
import argparse
import numpy as np
import torch
import wandb
from pathlib import Path
from multiprocessing import Process

# ==================================================== 项目路径 ====================================================
SCRIPT_DIR = Path(__file__).resolve().parent           # risk_sensitive_env/
sys.path.insert(0, str(SCRIPT_DIR))

from agents import DQCAC                               # DQC-AC Agent
from envs import RiskSensitiveEnv                      # 风险敏感环境
from utils.evaluation import monte_carlo_evaluate      # MC评估


# ==================================================== Baseline 参数 ====================================================
class BaselineArgs:
    """
    DQC-AC Baseline 参数类

    所有实验在此基础上进行调参，保持未指定参数与 baseline 一致。
    参数来源: train_evaluate.ipynb Cell 7 (DQC_AC_Args)
    """

    def __init__(self):
        # ==================== 基础实验参数 ====================
        self.env_name = "RiskSensitiveEnv"
        self.seed = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # ==================== 训练控制 ====================
        self.max_episode = 100000                        # 最大训练轮数
        self.log_interval = 100                          # 日志间隔
        self.est_interval = 100                          # 滚动估计窗口大小
        self.gamma = 0.99                                # 折扣因子

        # ==================== 分位数约束 ====================
        self.q_alpha = 0.25                              # 目标分位数水平 α
        self.quantile_threshold = 4                      # 约束阈值 C
        self.outer_interval = 10                         # 每多少次 inner update 更新一次 λ

        # ==================== Actor 学习率 (Two-Timescale) ====================
        self.init_std = float(np.sqrt(1e-1))             # 初始策略标准差
        self.theta_a = (10000 ** 0.9) * 1e-3             # θ 学习率参数 a
        self.theta_b = 10000                             # θ 学习率参数 b
        self.theta_c = 0.9                               # θ 学习率参数 c

        # ==================== 分位数估计学习率 ====================
        self.q_a = (10000 ** 0.6) * 1e-2                 # Q 学习率参数 a
        self.q_b = 10000                                 # Q 学习率参数 b
        self.q_c = 0.6                                   # Q 学习率参数 c

        # ==================== Lambda 学习率 ====================
        self.lambda_a = 0.3                              # λ 学习率参数 a
        self.lambda_b = 5000                             # λ 学习率参数 b
        self.lambda_c = 0.6                              # λ 学习率参数 c

        # ==================== Critic 结构/优化 ====================
        self.emb_dim = [64, 64]                          # Critic 隐藏层维度
        self.critic_lr = 1e-3                            # Critic 学习率
        self.tau = 0.005                                 # Target 网络软更新系数
        self.target_update_interval = 100                # Target 网络更新间隔

        # ==================== Distributional TD ====================
        self.num_quantiles = 32                          # 分位数数量 M
        self.huber_kappa = 1.0                           # Quantile Huber Loss κ
        self.density_bandwidth = 0.01                    # 密度估计带宽 δ

        # ==================== Replay Buffer ====================
        self.buffer_capacity = 100000                    # Buffer 容量
        self.batch_size = 64                             # 批量大小
        self.min_buffer_size = 1000                      # 开始学习前最小样本数
        self.updates_per_episode = 1                     # 每 episode 更新次数

        # ==================== 其他 ====================
        self.wandb_dir = None
        self.algo_name = "DQCAC"


# ==================================================== 实验参数组合 ====================================================
EXPERIMENTS = {
    # ---- 实验1: 增大 quantile 数量 + 增大网络容量 ----
    "exp1_nq200_emb256": {
        "num_quantiles": 200,
        "emb_dim": [256, 256],
    },

    # ---- 实验2a: 增大 batch_size=256 + 降低 Actor 学习率 ----
    "exp2a_bs256_lowlr": {
        "batch_size": 256,
        "theta_a": (10000 ** 0.9) * 1e-4,
    },

    # ---- 实验2b: 增大 batch_size=512 + 降低 Actor 学习率 ----
    "exp2b_bs512_lowlr": {
        "batch_size": 512,
        "theta_a": (10000 ** 0.9) * 1e-4,
    },

    # ---- 实验3a: target_update_interval sweep = 1000 ----
    "exp3a_tui1000": {
        "target_update_interval": 1000,
    },

    # ---- 实验3b: target_update_interval sweep = 100 (与 baseline 相同, 作对照) ----
    "exp3b_tui100": {
        "target_update_interval": 100,
    },

    # ---- 实验3c: target_update_interval sweep = 10 ----
    "exp3c_tui10": {
        "target_update_interval": 10,
    },

    # ---- 实验4: 最优参数组合1 (threshold=5) ----
    "exp4_optimal_th5_nq200_tui1000": {
        "quantile_threshold": 5,
        "num_quantiles": 200,
        "emb_dim": [256, 256],
        "target_update_interval": 1000,
    },

    # ---- 实验5a: tau sweep = 0.005 (与 baseline 相同, 作对照) ----
    "exp5a_tau0005": {
        "tau": 0.005,
    },

    # ---- 实验5b: tau sweep = 0.05 ----
    "exp5b_tau005": {
        "tau": 0.05,
    },

    # ---- 实验5c: tau sweep = 0.5 ----
    "exp5c_tau05": {
        "tau": 0.5,
    },

    # ---- 实验6a: updates_per_episode sweep = 1 (与 baseline 相同, 作对照) ----
    "exp6a_upe1": {
        "updates_per_episode": 1,
    },

    # ---- 实验6b: updates_per_episode sweep = 10 ----
    "exp6b_upe10": {
        "updates_per_episode": 10,
    },

    # ---- 实验6c: updates_per_episode sweep = 100 ----
    "exp6c_upe100": {
        "updates_per_episode": 100,
    },

    # ---- 实验7: 最优参数组合2 (threshold=4) ----
    "exp7_optimal_th4_nq200_tui1000": {
        "quantile_threshold": 4,
        "num_quantiles": 200,
        "emb_dim": [256, 256],
        "target_update_interval": 1000,
    },

    # ---- 实验8: 最优参数组合3 (threshold=6) ----
    "exp8_optimal_th6_nq200_tui1000": {
        "quantile_threshold": 6,
        "num_quantiles": 200,
        "emb_dim": [256, 256],
        "target_update_interval": 1000,
    },
}


# ==================================================== 工具函数 ====================================================
def make_args(overrides: dict, max_episode_override: int = None) -> BaselineArgs:
    """
    在 baseline 基础上覆盖指定参数

    [参数]:
    - overrides: 需要覆盖的参数字典
    - max_episode_override: CLI 传入的 max_episode，优先级最高
    """
    args = BaselineArgs()
    for key, val in overrides.items():
        if key == "device":                              # device 在 worker 中单独设置
            continue
        setattr(args, key, val)
    if max_episode_override is not None:
        args.max_episode = max_episode_override
    return args


def args_to_serializable(args) -> dict:
    """将 args 转为 JSON 可序列化的 dict"""
    result = {}
    for k, v in vars(args).items():
        if isinstance(v, torch.device):
            result[k] = str(v)
        elif isinstance(v, np.floating):
            result[k] = float(v)
        elif isinstance(v, np.integer):
            result[k] = int(v)
        else:
            result[k] = v
    return result


# ==================================================== 单个实验进程 ====================================================
def run_single_experiment(exp_name, overrides, gpu_id=0, eval_episodes=10000,
                          max_episode_override=None):
    """
    单个实验: 训练 → 评估 → 保存结果

    [参数]:
    - exp_name:  实验名 (用于文件命名和日志)
    - overrides: 与 baseline 的参数差异 dict
    - gpu_id:    GPU 编号
    - eval_episodes:        评估时的 episode 数
    - max_episode_override: CLI 覆盖的训练轮数
    """
    try:
        # ==================== 设置设备与 wandb 名称 ====================
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        args = make_args(overrides, max_episode_override)
        args.device = device
        args.wandb_name = f"DQCAC_{exp_name}"             # 让 wandb run 名称区分不同实验

        # ==================== 随机种子 ====================
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

        print(f"\n[{exp_name}] 开始训练 | device={device} | overrides={overrides}")

        # ==================== 训练 ====================
        env = RiskSensitiveEnv(n=10)
        agent = DQCAC(args, env)
        agent.train()
        wandb.finish()

        # ==================== 评估 ====================
        print(f"[{exp_name}] 训练完成, 开始评估 ({eval_episodes} episodes)...")
        mean_ret, quantile_ret = monte_carlo_evaluate(
            agent, env, num_episodes=eval_episodes
        )

        # 获取 learned risk (策略在初始状态的动作)
        test_state = env.reset()
        if isinstance(test_state, tuple):
            test_state = test_state[0]
        test_action = agent.select_action(test_state.flatten())
        learned_risk = float(test_action[0])

        # ==================== 保存结果 ====================
        result_dir = SCRIPT_DIR / "sweep_results"
        result_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d-%H%M%S")

        payload = {
            "experiment_name": exp_name,
            "timestamp": timestamp,
            # ---- 评估结果 ----
            "eval": {
                "mean_return": round(float(mean_ret), 6),
                "quantile_return": round(float(quantile_ret), 6),
                "learned_risk": round(learned_risk, 6),
                "eval_episodes": eval_episodes,
            },
            # ---- 与 baseline 的差异 (便于快速识别) ----
            "param_overrides": overrides,
            # ---- 完整训练参数 ----
            "full_params": args_to_serializable(args),
        }

        filename = result_dir / f"{exp_name}_seed{args.seed}_{timestamp}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"\n{'=' * 60}")
        print(f"[{exp_name}] 实验完成!")
        print(f"  Mean Return:     {mean_ret:.4f}")
        print(f"  Quantile Return: {quantile_ret:.4f}")
        print(f"  Learned Risk:    {learned_risk:.4f}")
        print(f"  结果保存至: {filename}")
        print(f"{'=' * 60}\n")

    except Exception as e:
        print(f"\n[ERROR] [{exp_name}] 实验失败: {e}")
        import traceback
        traceback.print_exc()


# ==================================================== 结果汇总 ====================================================
def print_summary():
    """读取 sweep_results/ 下所有 JSON, 打印对比表"""
    result_dir = SCRIPT_DIR / "sweep_results"
    if not result_dir.exists():
        print("无结果目录")
        return

    results = []
    for f in sorted(result_dir.glob("*.json")):
        with open(f, encoding="utf-8") as fp:
            results.append(json.load(fp))

    if not results:
        print("无结果文件")
        return

    # 表头
    header = f"{'实验名':<40} {'Mean Ret':>10} {'Quantile Ret':>13} {'Learned Risk':>13}"
    sep = "=" * len(header)

    print(f"\n{sep}")
    print("DQC-AC 超参数扫描结果汇总")
    print(sep)
    print(header)
    print("-" * len(header))

    for r in results:
        name = r["experiment_name"]
        ev = r["eval"]
        print(f"{name:<40} {ev['mean_return']:>10.4f} {ev['quantile_return']:>13.4f} "
              f"{ev['learned_risk']:>13.4f}")

    print(sep)
    print(f"共 {len(results)} 个实验结果\n")


# ==================================================== 主函数 ====================================================
def main():
    parser = argparse.ArgumentParser(
        description="DQC-AC 超参数扫描脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_sweep.py                                   # 全部实验, 并行5个
  python run_sweep.py --max-parallel 2                  # 并行2个 (显存紧张)
  python run_sweep.py --exp exp1_nq200_emb256 exp3a_tui1000   # 只跑指定实验
  python run_sweep.py --max-episode 5000                # 快速测试
  python run_sweep.py --summary                         # 只打印已有结果汇总
        """
    )
    parser.add_argument("--max-parallel", type=int, default=5,
                        help="最大并行进程数 (默认5, 显存紧张时减小)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU 编号 (默认0)")
    parser.add_argument("--eval-episodes", type=int, default=10000,
                        help="评估 episode 数 (默认10000)")
    parser.add_argument("--max-episode", type=int, default=None,
                        help="覆盖训练轮数 (默认使用 baseline 的 100000)")
    parser.add_argument("--exp", nargs="+", default=None,
                        help="指定要运行的实验名 (空格分隔, 默认全部)")
    parser.add_argument("--list", action="store_true",
                        help="列出所有可用实验及其参数差异")
    parser.add_argument("--summary", action="store_true",
                        help="只打印已有结果汇总, 不启动训练")

    cli_args = parser.parse_args()

    # ---- 列出实验 ----
    if cli_args.list:
        print("可用实验:")
        for name, overrides in EXPERIMENTS.items():
            print(f"  {name}:")
            for k, v in overrides.items():
                print(f"    {k} = {v}")
        return

    # ---- 仅打印汇总 ----
    if cli_args.summary:
        print_summary()
        return

    # ---- 确定要运行的实验 ----
    if cli_args.exp:
        experiments = {}
        for name in cli_args.exp:
            if name in EXPERIMENTS:
                experiments[name] = EXPERIMENTS[name]
            else:
                print(f"[WARNING] 未找到实验 '{name}', 跳过")
                print(f"  可用: {list(EXPERIMENTS.keys())}")
        if not experiments:
            print("没有有效的实验可运行")
            return
    else:
        experiments = EXPERIMENTS

    # ---- 打印运行计划 ----
    max_ep_str = str(cli_args.max_episode) if cli_args.max_episode else "100000 (baseline)"
    print(f"\n{'=' * 60}")
    print(f"DQC-AC 超参数扫描")
    print(f"{'=' * 60}")
    print(f"  实验数:     {len(experiments)}")
    print(f"  最大并行:   {cli_args.max_parallel}")
    print(f"  GPU:        cuda:{cli_args.gpu}")
    print(f"  训练轮数:   {max_ep_str}")
    print(f"  评估轮数:   {cli_args.eval_episodes}")
    print(f"  结果目录:   {SCRIPT_DIR / 'sweep_results'}")
    print(f"{'=' * 60}")
    for name, ov in experiments.items():
        print(f"  [{name}] → {ov}")
    print(f"{'=' * 60}\n")

    # ---- 按批次启动 ----
    exp_list = list(experiments.items())
    batch_size = cli_args.max_parallel

    total_batches = (len(exp_list) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch = exp_list[batch_start:batch_start + batch_size]

        batch_names = [name for name, _ in batch]
        print(f"\n--- 批次 {batch_idx + 1}/{total_batches}: {batch_names} ---")

        processes = []
        for exp_name, overrides in batch:
            p = Process(
                target=run_single_experiment,
                args=(exp_name, overrides, cli_args.gpu, cli_args.eval_episodes,
                      cli_args.max_episode)
            )
            p.start()
            processes.append((exp_name, p))

        # 等待本批次全部完成
        for exp_name, p in processes:
            p.join()
            if p.exitcode != 0:
                print(f"[WARNING] {exp_name} 进程退出码: {p.exitcode}")

    # ---- 打印汇总 ----
    print("\n所有实验已完成!")
    print_summary()


if __name__ == "__main__":
    main()
