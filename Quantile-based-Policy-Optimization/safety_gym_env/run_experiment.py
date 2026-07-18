# -*- coding: utf-8 -*-
"""
run_experiment.py —— safety_gym_env 统一实验入口 (QCPO / DQCAC / CALIB)。

环境对齐 NeurIPS'22 QCPO 论文 config0-3 (SimpleButton/Dynamic/Gremlin/DynamicButton,
由 envs/paper_envs.py 复现; 详见 DESIGN.md)。统一对比标准 (硬约定):
    1. 同一环境 (论文 config) + 同 γ=0.99 / cost_gamma / ω / d;
    2. 同一评估协议: 训练后 evaluate_policy_vec (整段=一 episode) 报 mean(R)/
       empirical_prob=P(Z≤q)/quantile_return (+DQCAC cost-critic 校准); 同探索 σ;
    3. 同一 wandb 方案 (project='safety_gym_qcrl', x 轴=progress/env_steps;
       约束键与 risk_sensitive 下尾口径对齐: Z=-C, q=-d)。

约束口径 (见 DESIGN.md; 双口径支持, 由 cost_gamma 切换):
    优化: max E[R]  s.t.  P(C ≥ d) ≤ ω,  C = Σ cost_gamma^t c_t
          ≡ max E[R]  s.t.  P(Z ≤ q) ≤ α,  Z=-C, q=-d, α=ω。
    - 默认【未折扣】论文口径: cost_gamma=1.0, d=15, ω=0.2 (对齐 launch_qcpo.py 的
      cost_limit=15/target_prob=0.2)。此时 C=Σc, QCPO 指示函数直接落在论文约束变量上;
      DQCAC 自动切 episodic 配方 (episode 末不 bootstrap + critic_step_feature=True)。
    - 折扣口径 (开关): --set cost_gamma=0.99 cost_limit=<CALIB 校准值>, DQCAC 回到
      continuing 截断恒 bootstrap 配方 (γc^1000≈4e-5, 残差可忽略)。

用法:
    MUJOCO_GL=egl python run_experiment.py --algo DQCAC --env SimpleButton
    MUJOCO_GL=egl python run_experiment.py --algo CALIB --env SimpleButton     # 校准 d
    MUJOCO_GL=egl python run_experiment.py --algo QCPO  --env Gremlin --set num_iterations=50 --wandb_mode disabled
    MUJOCO_GL=egl python run_experiment.py --algo QCPO_REF --env SimpleButton  # NIPS'22 基线移植版
算法名: QCPO / DQCAC / QCPO_REF / CALIB    环境名: SimpleButton / Dynamic / Gremlin / DynamicButton
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')                     # headless (须在 import safety_gymnasium 前)
import sys
import json
import time
import random
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import numpy as np
import torch


# ============================================================ 参数体系 ============================================================
class argparse_ns:
    """轻量命名空间 (与 portfolio_env_inf/run_experiment.py 同款)。"""
    pass


def cast(v):
    """命令行字符串 → 合适类型 (int/float/bool/None/str)。"""
    if v.lower() in ('none', 'null'):
        return None
    if v.lower() == 'true':
        return True
    if v.lower() == 'false':
        return False
    try:
        if '.' not in v and 'e' not in v.lower():
            return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


# ============================================================ 论文环境表 ============================================================
# name → (SafetyEnv env_id, 默认约束)。默认【未折扣】论文口径: d=15 / ω=0.2
# (launch_qcpo.py: cost_limit=15, target_prob=0.2), C=Σc (cost_gamma=1)。
# 折扣口径按需覆盖: --set cost_gamma=0.99 cost_limit=<CALIB 值>。
PAPER_ENVS = {
    'SimpleButton':  dict(env_id='SimpleButton',  omega=0.2, cost_limit=15.0),
    'Dynamic':       dict(env_id='Dynamic',       omega=0.2, cost_limit=15.0),
    'Gremlin':       dict(env_id='Gremlin',       omega=0.2, cost_limit=15.0),
    'DynamicButton': dict(env_id='DynamicButton', omega=0.2, cost_limit=15.0),
}


def base_args(algo, seed, device, env_key):
    """两算法默认超参 (移植自 portfolio_env_inf 已验证配置; safety-gym CMDP 适配)。"""
    a = argparse_ns()
    ev = PAPER_ENVS[env_key]
    # -------------------- 统一字段 (两算法一致) --------------------
    a.env_name = env_key
    a.env_id = ev['env_id']
    a.seed = seed
    a.algo_name = algo
    a.q_alpha = ev['omega']          # ω 目标 outage 概率 P(C≥d)≤ω
    a.cost_limit = ev['cost_limit']  # d 约束阈值 (默认未折扣口径 d=15, 论文对齐)
    a.gamma = 0.99                   # 奖励折扣 γ (论文一致)
    a.cost_gamma = 1.0               # cost 折扣 γc: 1.0=未折扣论文口径 (默认); 0.99=折扣口径开关
    a.horizon = 1000                 # 一段 rollout 步长 T (论文 safety-gym episode 长度)
    a.num_envs = 16                  # 并行 env 数 B (mp 后端: B 个 worker 进程, 128 核可加大)
    a.vec_backend = 'mp'             # 'mp' 多进程并行 (默认) / 'sync' 单进程串行 (调试)
    a.num_iterations = 300           # 迭代数 (最小 pipeline 用; 全量另调)
    a.updates_per_episode = 10       # 每迭代内层更新次数
    a.actor_updates_per_episode = 1  # 非 PPO actor 每个 rollout 只做一次严格 on-policy 更新
    a.actor_update_interval = 1      # 每多少个 rollout 合并一次 actor batch；1 精确保持历史路径
    a.init_std = 0.5                 # 策略探索 σ (动作空间 [-1,1])
    a.actor_hidden = [256, 256]      # MLP 策略 (观测 60-76 维)
    a.log_interval = 10
    a.device = device
    a.wandb_dir = BASE_DIR
    a.wandb_project = 'safety_gym_qcrl'
    # 默认不保存 checkpoint；显式给目录后，final 总会保存，正 interval 还会
    # 由 DQCAC 在对应 rollout 的任何参数更新前保存评估快照。
    a.checkpoint_dir = None
    a.checkpoint_interval = 0
    # 仅在 --critic_calibration_from 中强制打开；默认训练继续更新观测moments。
    a.freeze_observation_stats = False
    a.freeze_policy_updates = False

    if algo == 'QCPO':
        # MC 轨迹级约束版。safety-gym 观测信息量足 → θ 默认 adam (非 portfolio 噪声特征情形)。
        a.theta_optimizer = 'adam'
        a.theta_b = 1000
        a.theta_c = 0.9
        a.theta_a = (a.theta_b ** a.theta_c) * 3e-4    # adam lr(0)=3e-4 (MLP 典型)
        a.lambda_a = 1.0                                # 快 λ
        a.lambda_b = 2000
        a.lambda_c = 0.1
        a.lambda_max = 50.0
        a.outer_interval = 1
        a.norm_ema_decay = 0.01
        a.actor_grad_clip = 1.0
        a.warmup_rms_iters = 2
        # 安全默认只做一次严格 on-policy actor update；多 epoch 必须显式切到 PPO，
        # 使用固定 behavior log-prob importance ratio 与 clip。
        a.updates_per_episode = 1
        a.qcpo_actor_update_mode = 'on_policy'
        a.ppo_ratio_clip = 0.1
        a.normalize_observation = False                 # Q-A1 起显式打开，保留旧基线可复现
        a.obs_norm_var_clip = 1e-6
        a.obs_norm_clip = 10.0
        a.learn_std = False                             # Q-A1 起用 σ=1 + learnable 作独立消融
        a.log_std_min = -5.0
        a.log_std_max = 2.0
        a.qcpo_reward_mode = 'mc'                      # 'gae' = V_r+GAE reward / MC constraint hybrid
        a.gae_lambda = 0.97
        a.reward_advantage_norm = False
        a.reward_value_step_feature = True
        a.reward_value_lr = 3e-4
        a.reward_value_grad_clip = 10.0
        a.policy_arch = 'mlp'
        a.recurrent_hidden = [512, 512]
        a.lstm_size = 512
        a.lstm_skip = True
        a.recurrent_seq_len = 100
        a.recurrent_value_loss_coef = 1.0
    elif algo == 'DQCAC':
        # per-transition 双 critic 版 (继承 portfolio DQCACBetaGPU 已验证配方)。
        a.beta = 0.95
        a.critic_step_feature = None                    # None → 跟随口径: 未折扣 episodic=True, 折扣=False
        a.theta_b = 10000
        a.theta_c = 0.9
        a.theta_a = (a.theta_b ** a.theta_c) * 2e-4     # actor lr≈近常数 2e-4 (已验证)
        a.lambda_a = 0.3
        a.lambda_b = 5000
        a.lambda_c = 0.1
        a.lambda_max = 50.0
        a.lambda_min = 0.0
        a.outer_interval = 1
        a.num_quantiles = 32
        # cost-only distribution family。qr 保持历史结果；iqn 连续采样τ；nq用
        # mean+非负gap强制固定网格单调。ReLU对齐论文离散回报NQ-Net*，elu1
        # 作为严格正gap独立消融，默认配置仍是qr所以不改变任何历史训练。
        a.cost_distribution_model = 'qr'
        a.cost_nq_gap_activation = 'relu'
        a.cost_iqn_train_quantiles = 32
        a.cost_iqn_query_quantiles = 128
        a.cost_iqn_cosines = 64
        a.cost_iqn_seed = seed + 104729
        a.huber_kappa = 0.1                             # 近纯分位回归 → critic 无偏 (已验证)
        a.critic_hidden = [256, 256]
        a.critic_lr = 1e-3
        a.critic_minibatch_size = 0                     # 0=历史整批；大 B/N 时显式设 chunk
        # 与上面的显存chunk不同：正数会按完整trajectory执行独立Adam step。
        # 默认0严格保留整批更新；C-H22显式用B80采样、2×B40优化。
        a.optimizer_minibatch_trajectories = 0
        # 历史 QR loss 对 target quantile 求和，梯度随 N 线性放大。默认 legacy_sum
        # 保持所有旧 run 可复现；reference_mean 用 reference/N 缩放，供 N=64/128
        # 做公平分辨率消融，避免“quantile 更多”被隐式改成“critic 梯度更大”。
        a.quantile_target_reduction = 'legacy_sum'
        a.quantile_loss_reference_samples = 32
        a.target_tau = 0.05                             # 消除 critic 滞后 (已验证)
        a.target_update_interval = 1
        # online逐式兼容；target查询Polyak网络；preupdate则先用最新online缓存
        # actor风险权重，再训练当前批critic，隔离同批标签而不引入长期target滞后。
        a.cost_actor_query_mode = 'online'
        # trajectory-MC correction默认关闭，严格保留历史critic advantage。raw模式
        # 逐式保留Acritic+eta*(I-p_hat)；rms_balanced先配平critic/residual尺度。
        # reference=ema逐位保留已有实验；batch令每个behavior批的实际RMS比等于rho。
        a.cost_actor_mc_correction_coef = 0.0
        a.cost_actor_mc_correction_mode = 'raw'
        # critic保留I-p_hat历史residual；leave_one_out使用其它轨迹的经验
        # outage均值作基线，避免把本轨迹标签同时放进自身control variate。
        a.cost_actor_mc_baseline_mode = 'critic'
        a.cost_actor_mc_balance_reference = 'ema'
        a.cost_actor_mc_balance_std_floor = 1e-4
        a.cost_actor_mc_balance_ratio_max = 1.0
        # n-step TD: T=1000 未折扣口径下 1-step 传播太慢 (600 updates 传不满 1000 步链,
        # 实测 cost-critic 在 s0 恒 0 → λ 不动)。n_step=100 → bootstrap 链长 10, 数百
        # updates 即可覆盖; on-policy 每迭代重采, n-step 和 + 截断 mask 均合法。
        a.n_step = 100
        # cost distribution target 消融：nstep 完整保留历史 QR-TD；mc 使用本次完整
        # T=1000 episode 的真实 return-to-go，专门诊断 s0 cost/CDF 低估是否来自
        # bootstrap 传播。默认绝不改变已有实验，只有显式 --set cost_target_mode=mc 才启用。
        a.cost_target_mode = 'nstep'
        # cost transition objective 默认全时刻等权；risk_discount 显式用
        # discount^t/mean 对齐风险 actor 的早期有效样本，同时保持 loss 总尺度。
        a.cost_critic_time_weighting = 'uniform'
        a.cost_critic_weight_discount = None             # None 时跟随 beta
        a.cost_critic_weight_floor = 0.0
        # recent-s0 auxiliary 默认关闭；显式开启时只重排 cost critic 的监督
        # 质量，coef 按 auxiliary/base ratio 解释并归一化总梯度尺度。
        a.cost_s0_aux_coef = 0.0
        a.cost_s0_replay_batches = 4
        # 完整transition replay默认关闭；显式设batches=1时缓存上一rollout的
        # detached actor feature/action/step/MC cost，与当前cost目标做等尺度凸组合。
        a.cost_transition_replay_batches = 0
        a.cost_transition_replay_coef = 1.0
        # 上一rollout仅作保留性验证的cost guard默认关闭。启用后，当前批仍是唯一
        # cost梯度来源；若旧批smooth-Brier超过历史最好值的容忍带，就回滚cost
        # head及其Adam状态，并让本轮剩余C-step只更新reward critic。
        a.cost_holdout_guard = False
        a.cost_holdout_relative_tolerance = 0.05
        a.cost_holdout_absolute_tolerance = 0.002
        # QCPO_refs cost mean MSE 默认关闭；显式0.5时按本实现QR target-sum
        # 尺度自动换算相对权重，不需要随N手动重调。
        a.cost_mean_anchor_coef = 0.0
        a.cost_mean_anchor_cost_scale = 10.0              # QCPO_refs cost /= 10
        # QCPO_refs Weibull头拟合detach后的上30% cost quantiles。默认关闭；
        # scale=10恢复reference训练单位，epsilon只保护linear quantile的log定义域。
        a.cost_weibull_tail_coef = 0.0
        a.cost_weibull_tail_prob = 0.3
        a.cost_weibull_cost_scale = 10.0
        a.cost_weibull_epsilon = 1e-3
        # QCPO_refs 在 cost/10 单位上用 exp(logit) 保证分布输出非负；DQCAC
        # 默认 linear 完全兼容历史，exp/softplus 显式乘回10恢复 raw-cost 单位。
        a.cost_quantile_output = 'linear'
        a.cost_quantile_output_scale = 10.0
        # QCPO_refs式共享多任务表示默认关闭。开启后cost QR/mean监督只回传到
        # recurrent actor的MLP+LSTM骨干，action-conditioned cost head仍由critic Adam更新。
        a.cost_shared_backbone_coef = 0.0
        a.cost_shared_backbone_cost_scale = 10.0
        a.cost_shared_backbone_huber_kappa = 1.0             # QCPO_refs固定阈值
        # 默认逐式保留C-H6L的full-backbone梯度；adapter模式用零初始化瓶颈
        # 隔离cost监督，并可显式记录PPO+V与cost在共享参数上的梯度cosine。
        a.cost_shared_gradient_mode = 'full_backbone'
        a.cost_adapter_width = 0
        a.cost_adapter_scale = 1.0
        a.cost_gradient_diagnostics = False
        # cost history 消融：raw=历史 Markov critic；actor_feature=C-H0.5 共享并
        # detach policy feature；cost_lstm=C-H1 独立同输入 MLP+LSTM（当前要求 MC）。
        a.cost_history_mode = 'raw'
        # 默认保留旧actor_feature语义：整轮C20都使用rollout时缓存的behavior hidden。
        # True显式在每个PPO step后重算detach feature，使cost head不追逐陈旧坐标。
        a.cost_actor_feature_refresh = False
        # C-Q1 查询点平滑默认关闭；sigmoid 只替换 actor risk CDF surrogate，
        # hard 校准/经验 outage/QR critic 均保留，便于无歧义消融。
        a.cost_cdf_mode = 'hard'
        a.cost_cdf_temperature = 1.0
        # quantile保持历史阈值积分；direct显式增加Bernoulli查询head，并保留QR诊断。
        # 独立lr/clip确保direct梯度不会改变reward/QR joint optimizer。
        a.cost_cdf_estimator = 'quantile'
        a.cost_direct_cdf_lr = 1e-3
        a.cost_direct_cdf_grad_clip = 10.0
        a.cost_direct_cdf_budget_scale = None
        # C-DCF2默认online逐位复现C-DCF1；ema用每个head step的Polyak低通查询。
        a.cost_direct_cdf_query_mode = 'online'
        a.cost_direct_cdf_ema_tau = 0.005
        # C-Q4 cost-only τ grid；uniform 默认完全复现历史。query_mixture 围绕
        # τ*=1-alpha 加密，CDF/target 用 importance weight，prediction loss
        # 可选择 query-focused（局部优化）或 importance（全局 W1 保持）。
        a.cost_quantile_grid_mode = 'uniform'
        a.cost_quantile_query_tau = None
        a.cost_quantile_local_half_width = 0.1
        a.cost_quantile_local_fraction = 0.5
        a.cost_quantile_prediction_weighting = 'query_focused'
        a.num_action_samples = 4
        a.advantage_norm = 'qcpo'                       # EMA 归一化 (反 λ 卷绕, 已验证)
        a.norm_ema_decay = 0.1
        a.warmup_iters = 30                             # 先校准 cost-critic 再开 actor/λ
        a.entropy_coef = 0.0
        a.actor_grad_clip = 100.0
        a.critic_grad_clip = 10.0
        # reward actor 主干消融：默认保持旧 distributional；实验按 gae → gae_ppo 逐项打开。
        a.reward_actor_mode = 'distributional'
        a.normalize_observation = False                  # 可选 QCPO_refs 逐维 running mean/variance
        a.obs_norm_var_clip = 1e-6                       # 方差下限，防止常数维数值爆炸
        a.obs_norm_clip = 10.0                           # 归一化 observation 截断到 [-10,10]
        a.obs_norm_warmup_iters = 1                      # 首个 rollout 只刷统计，不更新 actor
        a.gae_lambda = 0.97
        a.reward_advantage_norm = False                 # QCPO_refs 默认不标准化 reward advantage
        a.ppo_ratio_clip = 0.1
        # 0=完全关闭并逐式复现旧PPO；正数在当前policy相对behavior policy的
        # approximate KL越界时跳过本epoch及剩余actor epochs，critic更新仍跑满。
        a.ppo_target_kl = 0.0
        a.reward_value_lr = 3e-4
        a.reward_value_grad_clip = 10.0
        # 默认继续使用旧 MLP，保证历史实验可复现；mlp_lstm 显式开启 QCPO_refs
        # 同形 actor/reward-V，分布 critic 仍保留 DQCAC 的 action-conditioned 输出。
        a.policy_arch = 'mlp'
        a.recurrent_hidden = [512, 512]
        a.lstm_size = 512
        a.lstm_skip = True
        a.recurrent_seq_len = 100
        a.recurrent_value_loss_coef = 1.0
        a.learn_std = False
        a.log_std_min = -5.0
        a.log_std_max = 2.0
        a.dual_update_mode = 'critic_adam'              # 旧 cost-critic CDF + Adam dual（兼容默认）
        a.dual_pid_signal = 'outage'                    # empirical_pid 可选 outage / cost_quantile
        # 经验PID默认逐rollout更新；显式设为2时累计两批真实cost后只响应一次。
        # 该开关与actor_update_interval分离，便于先验证“Actor低频”再验证“控制器同频”。
        a.pid_update_interval = 1
        # None 时 PID 目标就是真实 alpha；显式更小值提供有限样本 safety margin，
        # 只改变控制器设点，不改变论文约束、critic 查询点或最终评估口径。
        a.pid_target_prob = None
        a.pid_Ki = 0.1                                  # QCPO_refs 默认积分增益
        a.pid_Kp = 0.0                                  # 0=旧 bounded-I；P-B2 显式打开 PI
        a.pid_window_episodes = 100                     # 最近完整轨迹窗口
        a.pid_cost_scale = 10.0                         # quantile PID 与 QCPO_refs 相同 cost 缩放
        # 以下四项默认精确退化为旧 bounded-I；P-B 实验才显式打开 leak/deadband。
        a.pid_integral_leak = 1.0                       # rho=1：不泄漏，兼容历史结果
        a.pid_deadband = 0.0                            # probability/quantile error 死区
        a.pid_delta_max = float('inf')                  # 每 reference batch 的最大 |Delta lambda|
        a.pid_reference_episodes = 0.0                  # 0=每 iteration 一次；正数按 episode 缩放
        a.sum_norm = False                              # 实验开启：(J_r+λJ_c)/(1+λ)
    elif algo == 'QCPO_REF':
        # NIPS'22 QCPO 移植版 (config_qcpo.py / launch_qcpo.py 论文缺省超参)。
        a.ref_lr = 1e-4
        a.value_loss_coeff = 1.0
        a.entropy_loss_coeff = 0.0
        a.clip_grad_norm = 1e4
        a.gae_lambda = 0.97
        a.minibatches = 1
        a.epochs = 8
        a.ratio_clip = 0.1
        a.cost_value_loss_coeff = 0.5
        a.ep_cost_ema_alpha = 0.0                       # 0 = 硬更新
        a.ep_outage_ema_alpha = 0.0
        a.ep_cost_eqa_alpha = 0.0
        a.cost_scale = 10.0                             # "yes 10."
        a.weibull_tail_prob = 0.3
        a.n_quantile = 25
        a.pid_Ki = 0.1
        a.sum_norm = True                               # L=(J_r+λJ_c)/(1+λ)
        a.diff_norm = False
        a.penalty_init = 0.0
        a.reward_scale = 1.0
        a.new_T = 100                                   # LSTM BPTT 段长 (论文 new_T)
        a.ref_hidden = [512, 512]
        a.lstm_size = 512
        a.lstm_skip = True
        a.ref_init_log_std = 0.0                        # 可学习 log_std, 初始 σ=1
        a.normalize_observation = True
        a.var_clip = 1e-6
    return a


# ============================================================ CALIB: cost 分布校准 ============================================================
def calibrate(args, num_episodes=64):
    """
    随机(未训练)策略在整段 horizon 上测 R / C(折扣) / Σc(未折扣) 分布 + VecEnv 耗时 →
    选约束阈值 d 的依据 (对应 risk/portfolio 的 _noisy_calib.py 方法论)。
    """
    from envs import SafetyEnv, make_vec_env
    from agents.vec_base import VecAgentBase

    env = SafetyEnv(args.env_id)
    base = VecAgentBase(args, env)                     # 未训练随机策略
    vec = make_vec_env(args.env_id, num_envs=args.num_envs, horizon=args.horizon,
                       device=args.device, ref_env=env, seed=args.seed + 100,
                       backend=getattr(args, 'vec_backend', 'mp'))
    B, n = vec.B, vec.n
    episode_count = int(num_episodes)
    if episode_count <= 0:
        raise ValueError("num_episodes must be a positive integer")
    rounds = int(np.ceil(episode_count / B))
    Rs, Cs, Cus = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for _ in range(rounds):
            s = vec.reset()
            R = torch.zeros(B, device=args.device)
            C = torch.zeros(B, device=args.device)
            Cu = torch.zeros(B, device=args.device)
            dr = dc = 1.0
            for t in range(n):
                a = base._sample_actions(s)
                s, r, c, done = vec.step(a)
                R += dr * r; dr *= args.gamma
                C += dc * c; dc *= args.cost_gamma
                Cu += c
            Rs.append(R); Cs.append(C); Cus.append(Cu)
    dt = time.time() - t0
    # 校准也严格使用请求数量，避免改变num_envs时统计样本数静默变化。
    R = torch.cat(Rs)[:episode_count].cpu().numpy()
    C = torch.cat(Cs)[:episode_count].cpu().numpy()
    Cu = torch.cat(Cus)[:episode_count].cpu().numpy()

    print(f"\n===== CALIB {args.env_id} (random policy, E={R.shape[0]}) =====")
    print(f"timing: {dt:.1f}s, {dt/rounds:.2f}s/(B={B},T={n}), {B*n*rounds/dt:.0f} env-steps/s")
    print(f"R:        mean={R.mean():.3f} std={R.std():.3f} [{R.min():.2f},{R.max():.2f}]")
    print(f"C_disc:   mean={C.mean():.3f} std={C.std():.3f} pct[50,80,90,95]={np.round(np.percentile(C,[50,80,90,95]),2)}")
    print(f"C_undisc: mean={Cu.mean():.3f} std={Cu.std():.3f} pct[50,80,90,95]={np.round(np.percentile(Cu,[50,80,90,95]),2)}")
    print("outage 网格 (选 d):")
    for d in [1, 2, 3, 5, 8, 12, 20, 25, 30]:
        print(f"  d={d:>4g}: P(C_disc>=d)={np.mean(C>=d):.3f}  P(C_undisc>=d)={np.mean(Cu>=d):.3f}")
    return {'R_mean': float(R.mean()), 'C_disc': C.tolist()[:0], 'timing_s': dt}


# ============================================================ 主流程 ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='DQCAC',
                        choices=['QCPO', 'DQCAC', 'QCPO_REF', 'CALIB'])
    parser.add_argument('--env', type=str, default='SimpleButton', choices=list(PAPER_ENVS.keys()))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='0', help="cuda 序号或 'cpu'")
    parser.add_argument('--num_eval', type=int, default=64, help="训练后评估轨迹数 (CPU 慢, 别太大)")
    parser.add_argument('--wandb_mode', type=str, default=None, help="online/offline/disabled")
    parser.add_argument('--tag', type=str, default='run')
    parser.add_argument(
        '--eval_only', type=str, default=None,
        help="加载 safety-gym-eval-checkpoint-v1 并跳过训练；algo/env/seed/结构取自快照")
    parser.add_argument(
        '--critic_calibration_from', type=str, default=None,
        help="只恢复DQCAC actor/观测统计并冻结策略，重新初始化critic后继续采样训练")
    parser.add_argument('--set', nargs='*', default=[], metavar='K=V', help="覆盖任意超参")
    cli = parser.parse_args()

    # eval-only 先读轻量元数据，再构造同形 agent。文件由本仓库原子保存，
    # weights_only=False 是因为 payload 还包含配置与标量字典，不只有 tensor。
    if cli.eval_only is not None and cli.critic_calibration_from is not None:
        raise ValueError("--eval_only and --critic_calibration_from are mutually exclusive")
    checkpoint_payload = None
    checkpoint_path = cli.eval_only or cli.critic_calibration_from
    is_eval_only = cli.eval_only is not None
    is_critic_calibration = cli.critic_calibration_from is not None
    if checkpoint_path is not None:
        checkpoint_payload = torch.load(
            os.path.abspath(checkpoint_path), map_location='cpu', weights_only=False)
        if checkpoint_payload.get('format') != 'safety-gym-eval-checkpoint-v1':
            raise ValueError(
                f"unsupported eval checkpoint: {checkpoint_payload.get('format')!r}")
        cli.algo = str(checkpoint_payload['algo'])
        cli.env = str(checkpoint_payload['env'])
        # eval-only必须复原原seed；critic calibration则保留CLI seed，让新环境布局
        # 与源训练独立，同时baseline/候选仍可用同seed做common-random-number配对。
        if is_eval_only:
            cli.seed = int(checkpoint_payload['seed'])
        if cli.algo not in {'QCPO', 'DQCAC', 'QCPO_REF'}:
            raise ValueError(f"unsupported checkpoint algo: {cli.algo!r}")
        if cli.env not in PAPER_ENVS:
            raise ValueError(f"unsupported checkpoint env: {cli.env!r}")
        # 独立复评默认不创建同名线上 run；用户显式传 online/offline 时仍尊重。
        if is_eval_only and cli.wandb_mode is None:
            cli.wandb_mode = 'disabled'

    if cli.wandb_mode:
        os.environ['WANDB_MODE'] = cli.wandb_mode

    device = torch.device('cpu') if cli.device == 'cpu' else \
        torch.device(f"cuda:{cli.device}" if torch.cuda.is_available() else "cpu")

    algo_for_args = 'DQCAC' if cli.algo == 'CALIB' else cli.algo
    args = base_args(algo_for_args, cli.seed, device, cli.env)

    # checkpoint config 先覆盖默认值，命令行 --set 最后覆盖非结构性评估参数
    #（例如 num_envs）。device/wandb 路径始终使用当前机器，checkpoint 保存开关清零。
    if checkpoint_payload is not None:
        for key, value in checkpoint_payload.get('config', {}).items():
            if key not in {'device', 'wandb_dir', 'checkpoint_dir', 'checkpoint_interval'}:
                setattr(args, key, value)
        args.device = device
        args.seed = cli.seed
        args.algo_name = cli.algo
        args.env_name = cli.env
        args.env_id = PAPER_ENVS[cli.env]['env_id']
        if is_eval_only:
            args.checkpoint_dir = None
            args.checkpoint_interval = 0
            # optimizer trajectory minibatch只改变训练循环，不属于网络结构。
            # 评估可把num_envs改成任意并行度，不能被训练时mini-B整除约束阻断。
            if hasattr(args, 'optimizer_minibatch_trajectories'):
                args.optimizer_minibatch_trajectories = 0
            previous_name = str(getattr(args, 'wandb_name', cli.algo))
            args.wandb_name = f"{previous_name}_eval_{cli.tag}"

    overrides = {}
    for kv in cli.set:
        k, v = kv.split('=', 1)
        overrides[k] = cast(v)
        setattr(args, k, overrides[k])
    # wandb 已在模块导入阶段加载；只在这里修改 WANDB_MODE 环境变量时，
    # SDK 0.18.x 可能沿用已经缓存的 online setup。把 CLI 值同时保存在
    # args，VecAgentBase 会显式传给 wandb.Settings，保证 eval-only 的
    # disabled 模式不会意外创建远端 run。该字段只控制日志，不进入算法。
    args.wandb_mode = cli.wandb_mode
    # theta_lr0 便捷覆盖 (按最终 theta_b/c 重算 theta_a)
    if 'theta_lr0' in overrides:
        args.theta_a = (args.theta_b ** args.theta_c) * float(overrides['theta_lr0'])

    if is_critic_calibration:
        if cli.algo != 'DQCAC':
            raise ValueError("--critic_calibration_from currently supports DQCAC only")
        # 这是critic数据/目标实验，不是策略续训。actor epoch设0，整个预算保持
        # warmup令两条dual路径都不更新；lambda新初始化为0，不从源快照恢复。
        args.actor_updates_per_episode = 0
        args.warmup_iters = max(
            int(args.num_iterations), int(getattr(args, 'warmup_iters', 0)))
        args.lambda_max = 0.0
        args.freeze_observation_stats = True
        args.freeze_policy_updates = True
        args.calibration_source_checkpoint = os.path.abspath(checkpoint_path)
        args.calibration_source_phase = str(checkpoint_payload.get('phase'))
        args.calibration_source_env_steps = int(checkpoint_payload.get('env_steps', 0))

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- CALIB 模式 ----
    if cli.algo == 'CALIB':
        os.environ.setdefault('WANDB_MODE', 'disabled')
        calibrate(args, num_episodes=cli.num_eval)
        return

    # ---- 训练 ----
    from envs import SafetyEnv, make_vec_env
    from utils import evaluate_policy_vec
    from agents import QCPOGPU, DQCACBetaGPU, QCPORefGPU

    AgentCls = {'QCPO': QCPOGPU, 'DQCAC': DQCACBetaGPU, 'QCPO_REF': QCPORefGPU}[cli.algo]
    env = SafetyEnv(args.env_id)

    print(f"\n{'='*78}\n[{cli.tag}] algo={cli.algo} env={cli.env}({args.env_id}) seed={args.seed} "
          f"device={device} overrides={overrides}")
    t0 = time.time()
    agent = AgentCls(args, env)
    saved_checkpoint_path = None
    if not is_eval_only:
        if is_critic_calibration:
            # critic/target保持本run新初始化，只恢复成熟behavior policy和输入坐标系。
            agent.load_policy_calibration_checkpoint(checkpoint_payload)
            # 不同cost架构的参数shape会消耗不同数量的初始化随机数。这里在所有Module
            # 构造/恢复后重置host与CUDA RNG，使冻结策略的动作噪声跨候选严格配对。
            random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            print(
                f"\ncritic_calibration_from={os.path.abspath(checkpoint_path)} "
                f"source_phase={checkpoint_payload.get('phase')} "
                f"source_step={checkpoint_payload.get('env_steps')} rollout_seed={args.seed}")
        agent.train()
        train_t = time.time() - t0
        print(f"\ntrain_time {train_t:.1f}s")

        # checkpoint_dir 非空时总保存 post-update final；正 interval 的 DQCAC
        # 还已保存 pre-update rollout 策略，两种相位写在 payload 中，禁止混淆。
        if getattr(agent, 'checkpoint_dir', None) is not None:
            final_metrics = getattr(agent, 'get_training_summary', lambda: {})()
            saved_checkpoint_path = agent.save_evaluation_checkpoint(
                os.path.join(agent.checkpoint_dir, 'final_post_update.pt'),
                iteration=int(args.num_iterations) - 1,
                phase='post_update_final',
                metrics=final_metrics)
    else:
        agent.load_evaluation_checkpoint(checkpoint_payload)
        train_t = 0.0
        print(
            f"\neval_only checkpoint={os.path.abspath(cli.eval_only)} "
            f"phase={checkpoint_payload.get('phase')} "
            f"step={checkpoint_payload.get('env_steps')}")

    # ---- 统一评估 (同协议; QCPO_REF 为 LSTM 策略, 用其自管状态的同口径评估器) ----
    # agent 构造和训练会消耗不同数量的随机数；在评估开始前统一重置，令相同 seed
    # 的 QR/IQN、MLP/LSTM 使用同一 Gaussian action-noise 序列。环境 worker 仍使用
    # 独立 seed+777，因此这一步不回灌训练，也不依赖网络参数规模。
    eval_rng_seed = int(args.seed) + 777
    random.seed(eval_rng_seed); np.random.seed(eval_rng_seed)
    torch.manual_seed(eval_rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(eval_rng_seed)
    eval_vec = make_vec_env(args.env_id, num_envs=args.num_envs, horizon=args.horizon,
                            device=device, ref_env=env, seed=args.seed + 777,
                            backend=getattr(args, 'vec_backend', 'mp'))
    if cli.algo == 'QCPO_REF' or getattr(agent, 'recurrent_policy', False):
        res = agent.evaluate_vec(eval_vec, cli.num_eval, args.gamma, args.cost_gamma,
                                 args.q_alpha, args.cost_limit)
    else:
        res = evaluate_policy_vec(agent, eval_vec, cli.num_eval, args.gamma, args.cost_gamma,
                                  args.q_alpha, args.cost_limit)
    d, w = args.cost_limit, args.q_alpha
    emp = res['empirical_prob']
    print(f"\n----- eval ({res['num_episodes']} episodes) -----")
    print(f"R(mean)={res['mean']:.3f}  P(Z<=q):{emp:.03f} (alpha={w}, q=-d={-d:.1f})  "
          f"Q_alpha(Z)={res['quantile_return']:.03f}  margin={res['constraint_margin']:.03f}  "
          f"[debug C_disc={res['cost_disc_mean']:.3f} C_undisc={res['cost_undisc_mean']:.3f}]")
    if res['cost_cdf_initial'] is not None:                    # DQCAC: cost-critic 校准
        print(f"[cost-critic calibration] P(Z<=q): critic={res['cost_cdf_initial']:.3f} "
              f"truth={emp:.3f} bias={res['cost_cdf_initial']-emp:+.3f}")
    if res.get('cost_cdf_brier_initial') is not None:
        # recurrent评估返回完整hard/smooth discrimination；历史MLP评估只返回
        # hard Brier。缺失诊断显示nan，不能因纯打印字段阻断checkpoint/JSON保存。
        metric = lambda key: float(
            res[key]) if res.get(key) is not None else float('nan')
        print(
            f"[cost-critic discrimination] hard Brier/AUC/BSS="
            f"{metric('cost_cdf_brier_initial'):.4f}/"
            f"{metric('cost_cdf_roc_auc_initial'):.4f}/"
            f"{metric('cost_cdf_brier_skill_initial'):.2%}  "
            f"smooth={metric('cost_cdf_smooth_brier_initial'):.4f}/"
            f"{metric('cost_cdf_smooth_roc_auc_initial'):.4f}/"
            f"{metric('cost_cdf_smooth_brier_skill_initial'):.2%}")
    if res.get('cost_cdf_qr_initial') is not None:
        # direct模式的主键是Bernoulli head；QR同样本对照单独打印，禁止混为一个CDF。
        print(
            f"[QR internal control] P(Z<=q): qr={res['cost_cdf_qr_initial']:.3f} "
            f"truth={emp:.3f} bias={res['cost_cdf_qr_initial']-emp:+.3f} "
            f"Brier={res['cost_cdf_qr_brier_initial']:.4f}")
    if res.get('cost_cdf_direct_online_initial') is not None:
        # EMA是selected estimator；online control揭示低通前是否仍追逐最近rollout。
        print(
            "[direct online control] "
            f"online={res['cost_cdf_direct_online_initial']:.3f} "
            f"truth={emp:.3f} "
            f"bias={res['cost_cdf_direct_online_initial']-emp:+.3f} "
            f"Brier={res['cost_cdf_direct_online_brier_initial']:.4f}")
    if res.get('cost_cdf_crossfit_peer_abs_mean') is not None:
        # crossfit的主/peer/ensemble必须一起打印；只报ensemble会掩盖模型不确定性。
        print(
            "[crossfit critic] "
            f"primary={res['cost_cdf_primary_initial']:.3f} "
            f"peer={res['cost_cdf_peer_initial']:.3f} "
            f"mean_abs_disagreement={res['cost_cdf_crossfit_peer_abs_mean']:.3f}")
    ok = emp <= args.q_alpha + 0.02
    print(f"[{'OK ' if ok else 'BAD'}] P(Z<=q)<=alpha: {emp:.3f} (alpha={args.q_alpha})")
    eval_vec.close(); agent.vec_env.close()               # 回收 mp worker 进程

    # ---- 写 wandb + 存盘 (eval 主键用下尾命名; cost_* 仍写入作 debug) ----
    run = getattr(agent, '_wandb_run', None)
    if run is not None:
        eval_log = {
            'eval/mean': res['mean'],
            'eval/reward_std': res['reward_std'],
            'eval/empirical_prob': emp,
            'eval/quantile_return': res['quantile_return'],
            'eval/quantile_margin_to_threshold': res['quantile_margin_to_threshold'],
            'eval/constraint_margin': res['constraint_margin'],
            'eval/num_episodes': res['num_episodes'],
            'eval/debug_cost_disc_mean': res['cost_disc_mean'],
            'eval/debug_cost_undisc_mean': res['cost_undisc_mean'],
            'eval/debug_cost_quantile': res['cost_quantile'],
        }
        if res.get('cost_cdf_initial') is not None:
            eval_log['eval/cost_cdf_initial'] = res['cost_cdf_initial']
        if res.get('cost_cdf_smooth_initial') is not None:
            eval_log['eval/cost_cdf_smooth_initial'] = res['cost_cdf_smooth_initial']
        if res.get('cost_cdf_brier_initial') is not None:
            eval_log['eval/cost_cdf_brier_initial'] = (
                res['cost_cdf_brier_initial'])
        for key in (
                'cost_cdf_brier_skill_initial', 'cost_cdf_roc_auc_initial',
                'cost_cdf_discrimination_gap_initial',
                'cost_cdf_prediction_std_initial', 'cost_cdf_outage_mean_initial',
                'cost_cdf_safe_mean_initial', 'cost_cdf_smooth_brier_initial',
                'cost_cdf_smooth_brier_skill_initial',
                'cost_cdf_smooth_roc_auc_initial',
                'cost_cdf_smooth_discrimination_gap_initial',
                'cost_cdf_smooth_prediction_std_initial',
                'cost_cdf_smooth_outage_mean_initial',
                'cost_cdf_smooth_safe_mean_initial'):
            if res.get(key) is not None:
                eval_log[f'eval/{key}'] = res[key]
        if res.get('cost_cdf_qr_initial') is not None:
            eval_log['eval/cost_cdf_qr_initial'] = res['cost_cdf_qr_initial']
        if res.get('cost_cdf_qr_brier_initial') is not None:
            eval_log['eval/cost_cdf_qr_brier_initial'] = (
                res['cost_cdf_qr_brier_initial'])
        if res.get('cost_cdf_qr_smooth_initial') is not None:
            eval_log['eval/cost_cdf_qr_smooth_initial'] = (
                res['cost_cdf_qr_smooth_initial'])
        if res.get('cost_cdf_direct_online_initial') is not None:
            eval_log['eval/cost_cdf_direct_online_initial'] = (
                res['cost_cdf_direct_online_initial'])
        if res.get('cost_cdf_direct_online_brier_initial') is not None:
            eval_log['eval/cost_cdf_direct_online_brier_initial'] = (
                res['cost_cdf_direct_online_brier_initial'])
        if res.get('pred_cost_mean') is not None:
            eval_log['eval/pred_cost_mean'] = res['pred_cost_mean']
        if res.get('pred_cost_std') is not None:
            eval_log['eval/pred_cost_std'] = res['pred_cost_std']
        if res.get('cost_quantile_crossing_fraction') is not None:
            eval_log['eval/cost_quantile_crossing_fraction'] = (
                res['cost_quantile_crossing_fraction'])
        for key in (
                'cost_cdf_primary_initial', 'cost_cdf_peer_initial',
                'cost_cdf_crossfit_peer_abs_mean'):
            if res.get(key) is not None:
                eval_log[f'eval/{key}'] = res[key]
        run.log(eval_log)
        try:
            run.summary['eval/constraint_ok'] = bool(ok)
            if not is_eval_only:
                total_env_steps = (
                    int(args.num_envs) * int(args.num_iterations) * int(args.horizon))
            else:
                total_env_steps = int(checkpoint_payload.get('env_steps', 0))
            run.summary['budget/total_env_steps'] = total_env_steps
        except Exception as e:
            print(f"[warn] wandb summary 写入失败: {e}")
        run.finish()
    summary = getattr(agent, 'get_training_summary', lambda: {})()
    out = {
        'tag': cli.tag, 'algo': cli.algo, 'env': cli.env, 'seed': args.seed,
        'overrides': {k: str(v) for k, v in overrides.items()},
        'train_seconds': train_t, 'eval': res, 'summary': summary,
        'checkpoint_loaded': (
            os.path.abspath(checkpoint_path) if checkpoint_path is not None else None),
        'checkpoint_load_mode': (
            'eval_only' if is_eval_only else
            ('critic_calibration_policy_only' if is_critic_calibration else None)),
        'checkpoint_phase': (
            checkpoint_payload.get('phase') if checkpoint_payload is not None else None),
        'checkpoint_env_steps': (
            checkpoint_payload.get('env_steps') if checkpoint_payload is not None else None),
        'checkpoint_saved': saved_checkpoint_path,
    }
    os.makedirs(os.path.join(BASE_DIR, '_runs'), exist_ok=True)
    fp = os.path.join(BASE_DIR, '_runs', f"{cli.algo}_{cli.env}_{cli.tag}_s{args.seed}.json")
    with open(fp, 'w', encoding='utf-8') as f:
        json.dump(out, f)
    print(f"saved: {fp}\n{'='*78}")


if __name__ == '__main__':
    main()
