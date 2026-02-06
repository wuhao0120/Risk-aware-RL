#!/usr/bin/env python
"""
RiskSensitiveEnv 实验运行脚本

用于训练和评估 QCPO/QPO 算法在风险敏感环境中的表现。

使用示例:
    python run_experiment.py --algo_name QCPO
    python run_experiment.py --algo_name QPO --seed 0
    python run_experiment.py --config_file config.yaml --algo_name QCPO
"""

import os
import argparse
import torch
import numpy as np
import random
import json
import sys
import yaml
import wandb

# 添加当前目录到python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from agents import QCPO, QPO, DQCAC
from envs import RiskSensitiveEnv
from utils import monte_carlo_evaluate


def get_args():
    parser = argparse.ArgumentParser(description='运行风险敏感环境实验: QCPO vs QPO')
    
    # 配置文件
    parser.add_argument('--config_file', type=str, default=None, 
                      help='YAML配置文件路径')

    # 基础设置
    parser.add_argument('--algo_name', type=str, default='QCPO',
                      choices=['QCPO', 'QPO', 'DQCAC'],
                      help='算法名称')
    parser.add_argument('--seed', type=int, default=0,
                      help='随机种子')
    parser.add_argument('--device', type=str, default='0',
                      help='GPU编号或cpu')
    parser.add_argument('--eval_episodes', type=int, default=1000,
                      help='评估采样次数')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='结果保存目录')
    
    # 环境参数
    parser.add_argument('--env_name', type=str, default='RiskSensitiveEnv')
    
    # 训练参数
    parser.add_argument('--q_alpha', type=float, default=0.25, 
                      help='分位数水平α')
    parser.add_argument('--est_interval', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--max_episode', type=int, default=10000,
                      help='最大训练轮数')
    parser.add_argument('--init_std', type=float, default=1.0,
                      help='策略初始标准差')
    parser.add_argument('--gamma', type=float, default=0.99)
    
    # 学习率参数
    parser.add_argument('--theta_a', type=float, default=(10000**0.9)*1e-3)
    parser.add_argument('--theta_b', type=float, default=10000)
    parser.add_argument('--theta_c', type=float, default=0.9)
    parser.add_argument('--q_a', type=float, default=(10000**0.6)*1e-2)
    parser.add_argument('--q_b', type=float, default=10000)
    parser.add_argument('--q_c', type=float, default=0.6)
    
    # QCPO特有参数
    parser.add_argument('--outer_interval', type=int, default=100)
    parser.add_argument('--nu', type=float, default=1.0,
                      help='约束惩罚系数')
    parser.add_argument('--quantile_threshold', type=float, default=6.0,
                      help='分位数约束阈值C')
    parser.add_argument('--density_estimate', type=bool, default=False,
                      help='是否使用密度估计版本（True）或使用nu版本（False）')
    parser.add_argument('--h_n', type=float, default=0.01,
                      help='密度估计版本的分位数间隔（默认0.01，对应τ±0.01）')

    # DQC-AC特有参数
    parser.add_argument('--density_bandwidth', type=float, default=0.01,
                      help='DQC-AC密度估计带宽δ（默认0.01）')
    
    # 拉格朗日乘子学习率
    parser.add_argument('--lambda_a', type=float, default=1.0)
    parser.add_argument('--lambda_b', type=float, default=100.0)
    parser.add_argument('--lambda_c', type=float, default=0.6)

    # 第一次解析获取config_file
    temp_args, _ = parser.parse_known_args()
    
    # 加载配置文件
    if temp_args.config_file:
        config_path = temp_args.config_file
        if not os.path.isabs(config_path):
            config_path = os.path.join(BASE_DIR, config_path)
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                if config:
                    parser.set_defaults(**config)
                    print(f"已加载配置文件: {config_path}")
        else:
            print(f"警告: 配置文件 {config_path} 不存在")

    # 完整解析
    args = parser.parse_args()
    
    # 处理路径
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(BASE_DIR, args.output_dir)
    args.wandb_dir = BASE_DIR

    # 处理设备
    if args.device == 'cpu' or not torch.cuda.is_available():
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f'cuda:{args.device}')
        
    return args


def main():
    args = get_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # 初始化环境
    env = RiskSensitiveEnv(n=10)
    
    try:
        # 初始化Agent
        if args.algo_name == 'QCPO':
            # QCPO: 支持density_estimate模式
            agent = QCPO(args, env)
        elif args.algo_name == 'QPO':
            agent = QPO(args, env)
        elif args.algo_name == 'DQCAC':
            # DQC-AC: 使用密度估计的约束优化
            agent = DQCAC(args, env)
        else:
            raise ValueError(f"未知算法: {args.algo_name}")
        
        # 训练
        print(f"\n开始训练 {args.algo_name} (seed={args.seed})...")
        agent.train()
        print("训练完成。")
        
        # 评估
        print(f"\n评估中 ({args.eval_episodes} episodes)...")
        mean_est, quantile_est = monte_carlo_evaluate(
            agent, env, args.eval_episodes, gamma=args.gamma, q_alpha=args.q_alpha
        )
        
        # 获取学习到的风险等级
        test_state = env.reset()
        if isinstance(test_state, tuple): 
            test_state = test_state[0]
        test_action = agent.select_action(test_state.flatten())
        learned_risk = 1.0 / (1.0 + np.exp(-test_action[0]))
        
        print("\n" + "="*50)
        print(f"【{args.algo_name} 最终评估结果】")
        print("="*50)
        print(f"学习到的风险等级 r: {learned_risk:.4f}")
        print(f"平均回报 (Mean):    {mean_est:.4f}")
        print(f"{args.q_alpha}-分位数 (Q):   {quantile_est:.4f}")
        if args.algo_name in ['QCPO', 'DQCAC']:
            constraint_satisfied = quantile_est >= args.quantile_threshold
            print(f"约束阈值:           {args.quantile_threshold}")
            print(f"约束满足:           {'是 ✓' if constraint_satisfied else '否 ✗'}")
        print("="*50)
        
        # 保存结果
        os.makedirs(args.output_dir, exist_ok=True)
        result = {
            'algo_name': args.algo_name,
            'seed': args.seed,
            'mean_return': round(float(mean_est), 4),
            'quantile_return': round(float(quantile_est), 4),
            'learned_risk': round(float(learned_risk), 4),
            'constraint_threshold': args.quantile_threshold if args.algo_name in ['QCPO', 'DQCAC'] else None,
            'q_alpha': args.q_alpha,
            'config': {k: str(v) if isinstance(v, torch.device) else v 
                      for k, v in vars(args).items()}
        }
        
        filename = f"risk_{args.algo_name}_s{args.seed}_q{args.q_alpha}"
        if args.algo_name in ['QCPO', 'DQCAC']:
            filename += f"_t{args.quantile_threshold}"
            if args.algo_name == 'QCPO' and args.density_estimate:
                filename += "_density"
        filename += ".json"
        
        filepath = os.path.join(args.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"\n结果已保存至: {filepath}")

    finally:
        # 关闭wandb
        if wandb.run is not None:
            wandb.finish(quiet=True)


if __name__ == '__main__':
    main()
