import numpy as np
import torch


def monte_carlo_evaluate(agent, env, num_episodes, gamma=None, q_alpha=None):
    """
    使用训练好的策略进行Monte Carlo评估
    
    Args:
        agent: 具有 select_action(state) 方法的智能体
        env: 环境对象
        num_episodes: 评估的episode数量
        gamma: 折扣因子，默认使用 agent.gamma
        q_alpha: 分位数水平，默认使用 agent.q_alpha
        
    Returns:
        mean_est: 累积回报的均值估计
        quantile_est: 累积回报的α分位数估计
    """
    
    # 获取配置参数
    if gamma is None:
        gamma = getattr(agent, 'gamma', 0.99)
    if q_alpha is None:
        q_alpha = getattr(agent, 'q_alpha', 0.25)
        
    disc_epi_rewards = []
    
    # 不计算梯度，提高评估效率
    with torch.no_grad():
        for _ in range(num_episodes):
            # 重置环境
            state = env.reset()
            if isinstance(state, tuple):  # 兼容gymnasium返回(state, info)
                state = state[0]
                 
            disc_epi_reward = 0
            disc_factor = 1
            
            while True:
                # 确保state是扁平的numpy数组
                if isinstance(state, torch.Tensor):
                    state = state.cpu().numpy()
                state = state.flatten()
                
                # 选择动作（使用纯推理接口）
                action = agent.select_action(state)
                
                # 环境步进
                outcome = env.step(action)
                if isinstance(outcome, tuple):
                    if len(outcome) == 5:  # gymnasium格式
                        next_state, reward, terminated, truncated, info = outcome
                        done = terminated or truncated
                    elif len(outcome) == 4:  # gym格式
                        next_state, reward, done, info = outcome
                    else:
                        raise ValueError("env.step() 返回值格式异常")
                else:
                    raise ValueError("env.step() 必须返回tuple")

                if isinstance(next_state, tuple):
                    next_state = next_state[0]
                
                # 累积折扣奖励
                disc_epi_reward += disc_factor * reward
                disc_factor *= gamma
                
                if done:
                    break
                
                state = next_state
                
            disc_epi_rewards.append(disc_epi_reward)
    
    # 计算统计量
    mean_est = np.mean(disc_epi_rewards)
    quantile_est = np.percentile(disc_epi_rewards, q_alpha * 100)
    
    return mean_est, quantile_est


def evaluate_with_statistics(agent, env, num_episodes, gamma=None, q_alpha=None):
    """
    扩展评估函数，返回更详细的统计信息
    
    Returns:
        dict: 包含均值、分位数、标准差、最大最小值等统计量
    """
    if gamma is None:
        gamma = getattr(agent, 'gamma', 0.99)
    if q_alpha is None:
        q_alpha = getattr(agent, 'q_alpha', 0.25)
        
    disc_epi_rewards = []
    risk_levels = []  # 记录风险等级
    
    with torch.no_grad():
        for _ in range(num_episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
                 
            disc_epi_reward = 0
            disc_factor = 1
            
            while True:
                if isinstance(state, torch.Tensor):
                    state = state.cpu().numpy()
                state = state.flatten()
                
                action = agent.select_action(state)
                
                outcome = env.step(action)
                if len(outcome) == 5:
                    next_state, reward, terminated, truncated, info = outcome
                    done = terminated or truncated
                else:
                    next_state, reward, done, info = outcome

                if isinstance(next_state, tuple):
                    next_state = next_state[0]
                
                disc_epi_reward += disc_factor * reward
                disc_factor *= gamma
                
                if done:
                    break
                
                state = next_state
            
            disc_epi_rewards.append(disc_epi_reward)
            
            # 记录风险等级（如果环境支持）
            if hasattr(env, 'render'):
                risk_seq = env.render()
                if risk_seq is not None and len(risk_seq) > 0:
                    risk_levels.append(np.mean(risk_seq))
    
    results = {
        'mean': np.mean(disc_epi_rewards),
        'std': np.std(disc_epi_rewards),
        'quantile': np.percentile(disc_epi_rewards, q_alpha * 100),
        'min': np.min(disc_epi_rewards),
        'max': np.max(disc_epi_rewards),
        'median': np.median(disc_epi_rewards),
        'q_alpha': q_alpha,
        'num_episodes': num_episodes
    }
    
    if risk_levels:
        results['avg_risk_level'] = np.mean(risk_levels)
        results['std_risk_level'] = np.std(risk_levels)
    
    return results
