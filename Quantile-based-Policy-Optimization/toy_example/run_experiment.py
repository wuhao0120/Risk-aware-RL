import os
import re
import argparse
import torch
import datetime
import numpy as np
import random
import json
import sys

# 添加当前目录到python path，确保能导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents import QCPO, QPO
from envs import ToyEnv
from utils.evaluation import monte_carlo_evaluate

def get_args():
    parser = argparse.ArgumentParser(description='Run experiment for Quantile-based Policy Optimization')
    
    # 实验设置
    parser.add_argument('--algo_name', type=str, default='QCPO', choices=['QCPO', 'QPO'],
                      help='算法名称: QCPO 或 QPO')
    parser.add_argument('--seed', type=int, default=0,
                      help='随机种子')
    parser.add_argument('--device', type=str, default='0',
                      help='GPU编号 (e.g., "0" or "cpu")')
    parser.add_argument('--eval_episodes', type=int, default=1000,
                      help='评估时的Monte Carlo抽样次数')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='结果保存目录')
    
    # 通用超参数
    parser.add_argument('--env_name', type=str, default='ToyEnv')
    parser.add_argument('--q_alpha', type=float, default=0.25, 
                      help='分位数水平α')
    parser.add_argument('--est_interval', type=int, default=100,
                      help='估计间隔')
    parser.add_argument('--log_interval', type=int, default=100,
                      help='日志记录间隔')
    parser.add_argument('--max_episode', type=int, default=50000,
                      help='最大训练轮数')
    # parser.add_argument('--emb_dim', type=list, default=[8, 8]) # 暂时写死或不传
    parser.add_argument('--init_std', type=float, default=np.sqrt(1e-1),
                      help='初始标准差')
    parser.add_argument('--gamma', type=float, default=0.99,
                      help='折扣因子γ')
    
    # 学习率参数
    parser.add_argument('--theta_a', type=float, default=(10000**0.9)*1e-3)
    parser.add_argument('--theta_b', type=float, default=10000)
    parser.add_argument('--theta_c', type=float, default=0.9)
    parser.add_argument('--q_a', type=float, default=(10000**0.6)*1e-2)
    parser.add_argument('--q_b', type=float, default=10000)
    parser.add_argument('--q_c', type=float, default=0.6)
    
    # QCPO特有参数
    parser.add_argument('--outer_interval', type=int, default=1000)
    parser.add_argument('--lambda_eps', type=float, default=1e-3)
    parser.add_argument('--lambda_min', type=float, default=0.0)
    parser.add_argument('--lambda_max', type=float, default=10.0)
    parser.add_argument('--nu', type=float, default=1.0)
    parser.add_argument('--quantile_threshold', type=float, default=-3.0)
    
    args = parser.parse_args()
    
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
        
    print(f"Starting experiment: Algo={args.algo_name}, Seed={args.seed}, Device={args.device}")
    
    # 初始化环境
    env = ToyEnv(n=10)
    
    # 初始化Agent
    if args.algo_name == 'QCPO':
        agent = QCPO(args, env)
    elif args.algo_name == 'QPO':
        agent = QPO(args, env)
    else:
        raise ValueError(f"Unknown algo: {args.algo_name}")
    
    # 训练
    print("Training started...")
    agent.train()
    print("Training finished.")
    
    # 评估
    print(f"Evaluating with {args.eval_episodes} episodes...")
    mean_est, quantile_est = monte_carlo_evaluate(agent, env, args.eval_episodes)
    print(f"Evaluation Result - Mean: {mean_est:.4f}, Quantile: {quantile_est:.4f}")
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        'algo_name': args.algo_name,
        'seed': args.seed,
        'mean_est': float(mean_est),
        'quantile_est': float(quantile_est),
        'config': vars(args)
    }
    # 将device对象转为字符串以便json序列化
    result['config']['device'] = str(result['config']['device'])
    
    filename = f"{args.algo_name}_seed{args.seed}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(args.output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=4)
        
    print(f"Result saved to {filepath}")

if __name__ == '__main__':
    main()
