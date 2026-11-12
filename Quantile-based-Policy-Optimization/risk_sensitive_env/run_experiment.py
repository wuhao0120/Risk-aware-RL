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

from agents import QCPO, QPO, DQCAC, DQCACBeta, DQCACBetaGPU, QCPOGPU
from envs import RiskSensitiveEnv
from utils import monte_carlo_evaluate


def str2bool(value):
    """解析命令行布尔值, 兼容 true/false/1/0/yes/no。"""
    if isinstance(value, bool):
        return value
    return str(value).lower() in ('true', '1', 'yes', 'y')


def get_args():
    parser = argparse.ArgumentParser(description='运行风险敏感环境实验: QCPO vs QPO')
    
    # 配置文件
    parser.add_argument('--config_file', type=str, default=None, 
                      help='YAML配置文件路径')

    # 基础设置
    parser.add_argument('--algo_name', type=str, default='QCPO',
                      choices=['QCPO', 'QPO', 'DQCAC', 'DQCACBeta', 'DQCACBetaGPU', 'QCPOGPU'],
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
    
    # QCPO特有参数 (Probability-Based Constraint)
    parser.add_argument('--outer_interval', type=int, default=100,
                      help='外层更新间隔（每多少个episode更新一次λ）')
    parser.add_argument('--quantile_threshold', type=float, default=6.0,
                      help='约束阈值 q, 约束: P(U(τ)≤q) ≤ α')

    # DQC-AC特有参数
    parser.add_argument('--density_bandwidth', type=float, default=0.01,
                      help='DQC-AC密度估计带宽δ（默认0.01）')

    # 约束型 Actor-Critic 共享参数 (DQCAC / DQCACBeta / DQCACBetaGPU)
    parser.add_argument('--beta', type=float, default=0.95,
                      help='Abel风险折扣因子β (仅DQCACBeta/GPU: constraint项的per-transition近似)')
    parser.add_argument('--critic_lr', type=float, default=1e-3,
                      help='分布式critic学习率')
    parser.add_argument('--critic_hidden', type=int, nargs='+', default=[64, 64],
                      help='critic隐藏层维度, 例如: --critic_hidden 64 64')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='DQCAC critic replay minibatch大小')
    parser.add_argument('--buffer_capacity', type=int, default=100000,
                      help='DQCAC critic replay buffer容量')
    parser.add_argument('--min_buffer_size', type=int, default=256,
                      help='DQCAC启用critic replay混合采样前的最小样本数')
    parser.add_argument('--entropy_coef', type=float, default=0.0,
                      help='Actor entropy bonus系数')
    parser.add_argument('--target_tau', type=float, default=0.005,
                      help='Target critic Polyak软更新系数')
    parser.add_argument('--lambda_max', type=float, default=50.0,
                      help='拉格朗日乘子λ上界')
    parser.add_argument('--advantage_norm', type=str, default='separate',
                      choices=['separate', 'combined', 'none', 'qcpo'],
                      help='优势归一化方式; qcpo(仅DQCACBetaGPU): 跨迭代EMA稳定尺度归一化, 反λ卷绕')
    parser.add_argument('--num_action_samples', type=int, default=1,
                      help='估计V_m(s)和V_c(s,b)时采样动作数')
    parser.add_argument('--n_step', type=int, default=1,
                      help='DQCACBetaGPU critic n-step TD 目标步数 (1=1-step bootstrap; ≥episode长=MC; >1会失控,保持1)')
    parser.add_argument('--critic_step_feature', type=str2bool, default=True,
                      help='DQCACBetaGPU: critic 输入追加 step 特征 t/n, 修 step-blind 过散校准 (True=修复,默认)')
    parser.add_argument('--actor_grad_clip', type=float, default=10.0,
                      help='Actor梯度裁剪阈值')
    parser.add_argument('--critic_grad_clip', type=float, default=10.0,
                      help='Critic梯度裁剪阈值')
    parser.add_argument('--action_clip', type=float, default=None,
                      help='可选动作裁剪阈值; 默认None保持环境原始动作语义')

    # DQC-AC-β GPU 向量化 (DQCACBetaGPU) 特有参数
    parser.add_argument('--num_envs', type=int, default=256,
                      help='DQCACBetaGPU 并行 env 数 B (一次迭代采集的轨迹条数)')
    parser.add_argument('--num_iterations', type=int, default=400,
                      help='DQCACBetaGPU 迭代次数; 总 env-step = num_iterations × num_envs × n')
    parser.add_argument('--norm_ema_decay', type=float, default=0.1,
                      help='DQCACBetaGPU advantage_norm=qcpo 时 EMA 归一化器衰减系数 (≈1/decay 迭代窗宽)')
    parser.add_argument('--warmup_iters', type=int, default=None,
                      help='DQCACBetaGPU qcpo warmup 迭代数 (只练 critic+EMA, 不动 actor/λ); None→qcpo默认5/其余0')

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
            with open(config_path, 'r', encoding='utf-8') as f:
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
        elif args.algo_name == 'DQCACBeta':
            # DQC-AC-β: per-transition, 单分布式分位数critic + budget + Abel折扣β
            agent = DQCACBeta(args, env)
        elif args.algo_name == 'DQCACBetaGPU':
            # DQC-AC-β GPU: DQCACBeta 的全 GPU 向量化版 (B 路并行 + 全程在 device 上)
            agent = DQCACBetaGPU(args, env)
        elif args.algo_name == 'QCPOGPU':
            # QCPO GPU: QCPO 的全 GPU 向量化参考实现 (B 路并行, 行为对齐原始 QCPO)
            agent = QCPOGPU(args, env)
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
        learned_raw_action = float(test_action[0])
        learned_risk = 1.0 / (1.0 + np.exp(-test_action[0]))
        constrained_algos = ['QCPO', 'DQCAC', 'DQCACBeta', 'DQCACBetaGPU', 'QCPOGPU']
        
        print("\n" + "="*50)
        print(f"【{args.algo_name} 最终评估结果】")
        print("="*50)
        print(f"学习到的风险等级 r: {learned_risk:.4f}")
        print(f"学习到的原始动作 raw_a: {learned_raw_action:.4f}")
        print(f"平均回报 (Mean):    {mean_est:.4f}")
        print(f"{args.q_alpha}-分位数 (Q):   {quantile_est:.4f}")
        if args.algo_name in constrained_algos:
            constraint_satisfied = quantile_est >= args.quantile_threshold
            print(f"约束阈值:           {args.quantile_threshold}")
            print(f"约束满足:           {'yes' if constraint_satisfied else 'no'}")
        print("="*50)
        
        # 保存结果
        os.makedirs(args.output_dir, exist_ok=True)
        result = {
            'algo_name': args.algo_name,
            'seed': args.seed,
            'mean_return': round(float(mean_est), 4),
            'quantile_return': round(float(quantile_est), 4),
            'learned_risk': round(float(learned_risk), 4),
            'learned_raw_action': round(float(learned_raw_action), 4),
            'constraint_threshold': args.quantile_threshold if args.algo_name in constrained_algos else None,
            'q_alpha': args.q_alpha,
            'beta': args.beta if args.algo_name in ['DQCACBeta', 'DQCACBetaGPU'] else None,
            'config': {k: str(v) if isinstance(v, torch.device) else v 
                      for k, v in vars(args).items()}
        }
        if hasattr(agent, 'get_training_summary'):
            result.update(agent.get_training_summary())
        
        filename = f"risk_{args.algo_name}_s{args.seed}_q{args.q_alpha}"
        if args.algo_name in constrained_algos:
            filename += f"_t{args.quantile_threshold}"
            if args.algo_name == 'QCPO' and getattr(args, 'density_estimate', False):
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
