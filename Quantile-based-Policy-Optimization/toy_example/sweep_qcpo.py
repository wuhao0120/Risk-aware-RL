import wandb
import numpy as np
import random
import torch
import sys
import os

# Ensure imports work regardless of where script is run from
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.qcpo_sweep import QCPO
from envs import ToyEnv

# 定义参数类 (SimpeArgs from train_qcpo.ipynb)
class SimpleArgs:
    def __init__(self):
        self.env_name = 'ToyEnv'
        self.log_interval = 100
        self.est_interval = 100
        self.q_alpha = 0.25
        self.emb_dim = [8,8]
        self.max_episode = 100000  # 可以根据需要调整
        self.init_std = np.sqrt(1e-1)
        self.gamma = 0.99
        self.algo_name = 'QCPO'
        self.seed = 0

        # 学习率参数
        self.theta_a = (10000**0.9) * 1e-3
        self.theta_b = 10000
        self.theta_c = 0.9
        self.q_a = (10000**0.6) * 1e-2
        self.q_b = 10000
        self.q_c = 0.6

        # 待调优参数默认值
        self.outer_interval = 1
        self.lambda_eps = 1e-3
        self.lambda_min = 0.0
        self.lambda_max = 100.0
        self.nu = 1
        self.quantile_threshold = -3
        
        # 设备
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(config=None):
    # Initialize wandb run
    # 注意：这里使用 wandb.init 会启动一个新的 Run，QCPO 内部也会尝试 init。
    # 我们修改了 QCPO 代码使其仅在没有 Run 时 init。
    # 在 Sweep 中，wandb.init 在此处调用并接收 config。
    with wandb.init(config=config):
        # 如果由 wandb agent 调用，config 会被填充
        config = wandb.config

        args = SimpleArgs()
        
        # Update args with sweep parameters
        # 遍历 config 中的所有键值对，如果 args 有对应属性则更新
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
                
        # 设置随机种子
        random.seed(args.seed) 
        np.random.seed(args.seed) 
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

        # 创建环境和智能体
        env = ToyEnv(n=10)
        # 传入 wandb.run 作为上下文 (通过全局 wandb 对象)
        agent = QCPO(args, env)
        
        print(f"Starting training with params: outer_interval={args.outer_interval}, nu={args.nu}, lambda_max={args.lambda_max}, q_thresh={args.quantile_threshold}")
        
        # 启动训练
        agent.train()

# Sweep 配置
# 你可以在这里填入你要搜索的数值
sweep_config = {
    'method': 'grid',  # 网格搜索
    'metric': {
      'name': 'error/weights',      # 目标指标：最小化 error/weights
      'goal': 'minimize'   
    },
    'parameters': {
        'outer_interval': {
            'values': [1, 10, 100, 1000]  # 示例值
        },
        'nu': {
            'values': [0.1, 0.5, 1.0, 5.0] # 示例值
        },
        'lambda_max': {
            'values': [10.0, 20.0, 50.0, 100.0] # 示例值
        },
        'quantile_threshold': {
            'values': [-1, -1.5, -3.0, -5.0] # 示例值
        }
    }
}

if __name__ == '__main__':
    # 初始化 Sweep
    # project 参数指定项目名称
    sweep_id = wandb.sweep(sweep_config, project="QCPO_Tuning_Sweep")
    
    # 启动 Agent 执行训练
    # count 参数可以限制运行次数，如果不填则跑完所有组合
    wandb.agent(sweep_id, train)
