import numpy as np
import torch

def monte_carlo_evaluate(agent, env, num_episodes, gamma=None, q_alpha=None):
    """
    使用训练好的Actor策略进行Monte Carlo抽样估计
    
    参数:
        agent: 具有select_action(state)方法的智能体
        env: 环境对象
        num_episodes: 抽样次数
        gamma: 折扣因子，默认尝试使用agent.gamma
        q_alpha: 分位数水平，默认尝试使用agent.q_alpha
        
    返回:
        mean_est: 累积回报的均值估计
        quantile_est: 累积回报的分位数估计
    """
    
    # 尝试从主要参数获取配置，如果未提供且agent有该属性
    if gamma is None:
        gamma = getattr(agent, 'gamma', 0.99)
    if q_alpha is None:
        q_alpha = getattr(agent, 'q_alpha', 0.25)
        
    disc_epi_rewards = []
    
    # 确保不计算梯度
    with torch.no_grad():
        for _ in range(num_episodes):
            state = env.reset()
            if isinstance(state, tuple): # 兼容gym新版本
                 state = state[0]
                 
            disc_epi_reward = 0
            disc_factor = 1
            
            while True:
                # 确保state是扁平的numpy数组
                if isinstance(state, torch.Tensor):
                    state = state.cpu().numpy()
                state = state.flatten()
                
                # 使用纯推理接口选择动作
                action = agent.select_action(state)
                
                # 环境步进
                next_state, reward, done, info = env.step(action)
                
                # 累积奖励
                disc_epi_reward += disc_factor * reward
                disc_factor *= gamma
                
                if done:
                    break
                
                state = next_state
                
            disc_epi_rewards.append(disc_epi_reward)
    
    mean_est = np.mean(disc_epi_rewards)
    quantile_est = np.percentile(disc_epi_rewards, q_alpha * 100)
    
    return mean_est, quantile_est
