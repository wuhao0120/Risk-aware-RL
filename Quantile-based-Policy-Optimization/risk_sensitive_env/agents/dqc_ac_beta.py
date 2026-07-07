"""
DQC-AC (Distributional Quantile-Constrained Actor-Critic) —— per-transition + β 版

本文件是 QCPO (qcpo.py) 的"增量式"延伸：把 QCPO 里"等整条轨迹算完 U(τ) 再更新一次"
的两个量替换成 critic 派生的 per-transition 量，从而每个 transition 都能更新。

走的是 QCPO 的【概率约束】路线 (与 dqc_ac.py 的密度/分位数梯度路线无关):
    max_θ E[Z]   s.t.   P_θ(Z ≤ q) ≤ α,        Z = Σ_t γ^t r_t

与 QCPO 的对应关系 (这就是"结构流程大致一致"的含义):
    ┌─────────────────────┬──────────────────────────────────────────────────────┐
    │ QCPO (MC, 逐轨迹)    │ DQC-AC-β (TD, 逐 transition)                          │
    ├─────────────────────┼──────────────────────────────────────────────────────┤
    │ U(τ)=Σγ^t r_t       │ 分布式 critic 均值 Q̂_m(s,a)=(1/N)Σ_i ψ_i(s,a)        │
    │ 1{U(τ)≤q} (轨迹级)  │ 局部 CDF Ψ̂(s,a,b)=(1/N)Σ_i 1{ψ_i(s,a)≤b} (用 b_t 局部化)│
    │ 标量 q_est (仅监控)  │ 分布式 critic Z_ψ(s,a) (QRTD 学习)                     │
    │ update_inner        │ update_critic (QRTD) + update_actor (per-transition)  │
    │ 权重 U_norm-λ1{·}    │ 权重 γ^t Â_m - λ β^t Â_c                               │
    │ update_dual(经验P)  │ update_dual(经验P) —— 完全一致                         │
    └─────────────────────┴──────────────────────────────────────────────────────┘

两个核心机制:
1. budget b_t: 把"轨迹级事件 Z≤q"局部化成"单步事件 G_t≤b_t"。
   定义 C_t=Σ_{k<t}γ^k r_k, G_t=Σ_{ℓ≥t}γ^{ℓ-t}r_ℓ, 则 Z=C_t+γ^t G_t,
   于是 Z≤q ⟺ G_t≤(q-C_t)/γ^t =: b_t, 递推 b_0=q, b_{t+1}=(b_t-r_t)/γ。
2. Abel 折扣 β (本工作的新点): 约束梯度 ∇_θ G(θ)=Σ_t E[∇logπ·Ψ] 是【无折扣的无穷和】,
   方差不可控、无法单步估计。乘 β^t 变成收敛 (Abel 可和) 级数:
   ∇_θ G_β(θ)=Σ_t β^t E[∇logπ·Ψ(s_t,a_t,b_t)], 存在逐 transition 无偏估计 β^t∇logπ·Â_c,
   β→1 恢复真实梯度 → 使 per-transition actor 更新稳定。
   注意: β 只出现在 actor 约束项; budget/critic/dual 都用 γ、不含 β。

为什么用【单个分布式分位数 critic】(而非两个标量 critic):
   一张 QR 网络同时派生均值 (目标信号) 和 1{ψ_i≤b} 的经验 CDF (约束信号)。关键好处是
   预算 b 只在【查询时】用、不进网络 → 不需要 budget 裁剪、不会 CDF 饱和、终止态自动正确。
"""

from collections import deque                          # 滚动窗口 (回报/初始状态)

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam                            # 与 QCPO 一致: base lr=1 + LambdaLR
from torch.optim.lr_scheduler import LambdaLR           # 学习率调度器 lr=a/(b+k)^c
import wandb                                            # 统一日志系统

from utils import Actor                                 # 复用线性高斯策略 μ(s)=W·s, log_std 固定


def lr_lambda(k, a, b, c):
    """
    学习率衰减因子 (与 qcpo.py 完全一致)
    lr(k) = a / (b + k)^c
    """
    return a / ((b + k) ** c)


def indicator(threshold, values):
    """
    示性函数 I(values <= threshold), 返回与 values 同形状的 0/1 张量
    (与 qcpo.py / dqc_ac.py 语义一致)
    """
    return torch.where(values <= threshold,
                       torch.ones_like(values), torch.zeros_like(values))


class DistributionalCritic(nn.Module):
    """
    QR-DQN 风格分布式 critic: 把 (s,a) 映射到 N 个分位数 ψ_i(s,a)。

    分位数 ψ_i 对应水平 τ_i=(i-0.5)/N, 一起刻画了回报 Z(s,a) 的整条分布。
    本类【内联定义】在本文件中, 不依赖 dqc_ac.py (避免引入其错误算法路线)。
    """

    def __init__(self, state_dim, action_dim, num_quantiles, hidden=[64, 64]):
        """
        Args:
            state_dim:     状态维度 (本环境为 one-hot, n=10)
            action_dim:    动作维度 (本环境为 1)
            num_quantiles: 分位数个数 N (critic 输出维度)
            hidden:        隐藏层维度列表
        """
        super().__init__()
        # 输入维度 = 状态 + 动作 拼接; 逐层堆 Linear+ReLU
        dims = [state_dim + action_dim] + list(hidden)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))  # 全连接层
            layers.append(nn.ReLU())                        # 非线性激活
        layers.append(nn.Linear(dims[-1], num_quantiles))   # 输出层: N 个分位数
        self.net = nn.Sequential(*layers)

        # 正交初始化 (与项目其它网络风格一致, 利于训练稳定)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, state, action):
        """
        前向传播

        输入: state [B, state_dim], action [B, action_dim]
        输出: [B, N] —— 每行是该 (s,a) 的 N 个分位数估计
        """
        x = torch.cat([state, action], dim=-1)              # 在最后一维拼接 → [B, sd+ad]
        return self.net(x)                                  # → [B, N]


class DQCACBeta(object):
    """
    DQC-AC per-transition + β 版主体。结构与 qcpo.QCPO 对齐 (见文件顶部对应表)。
    """

    # ============================================================ 初始化 ============================================================
    def __init__(self, args, env):
        """
        初始化: actor + 单分布式 critic(+target) + λ + 优化器/调度器 + wandb
        (与 qcpo.QCPO.__init__ 结构对应, 只是把 q_est 标量换成了分布式 critic)
        """
        # -------------------- 基础参数 (与 QCPO 同名) --------------------
        self.device = args.device                          # 计算设备 CPU/GPU
        self.gamma = args.gamma                            # 折扣 γ (budget/critic 用)
        self.beta = getattr(args, 'beta', 0.95)            # Abel 风险折扣 β (仅 actor 约束项用)
        self.q_alpha = args.q_alpha                        # 约束水平 α
        self.quantile_threshold = args.quantile_threshold  # 阈值 q (= budget 初值 b_0)
        self.max_episode = args.max_episode                # 最大训练轮数
        self.log_interval = args.log_interval              # 日志间隔
        self.est_interval = args.est_interval              # 滚动窗口大小
        self.outer_interval = args.outer_interval          # λ 更新间隔 (按 episode 计)

        # -------------------- critic / TD 超参 (用 getattr 给默认值, 不配也能跑) --------------------
        self.num_quantiles = getattr(args, 'num_quantiles', 32)            # 分位数个数 N
        self.huber_kappa = getattr(args, 'huber_kappa', 1.0)               # quantile Huber 的 κ
        self.target_tau = getattr(args, 'target_tau', 0.005)               # target 软更新系数
        self.target_update_interval = getattr(args, 'target_update_interval', 1)  # target 更新间隔
        self.num_action_samples = max(1, int(getattr(args, 'num_action_samples', 1)))  # baseline 的 K
        self.updates_per_episode = max(1, int(getattr(args, 'updates_per_episode', 10)))  # 每 episode 内层更新次数
        self.advantage_norm = getattr(args, 'advantage_norm', 'separate')  # separate / none
        self.entropy_coef = getattr(args, 'entropy_coef', 0.0)             # 熵正则系数 (默认 0)
        self.lambda_max = getattr(args, 'lambda_max', 50.0)                # λ 上界
        self.critic_grad_clip = getattr(args, 'critic_grad_clip', 10.0)    # critic 梯度裁剪
        self.actor_grad_clip = getattr(args, 'actor_grad_clip', 10.0)      # actor 梯度裁剪

        # -------------------- 环境与维度 --------------------
        self.env = env
        self.env_name = args.env_name
        self.state_dim = int(np.prod(env.observation_space.shape))         # 状态维度
        self.action_dim = int(np.prod(env.action_space.shape))             # 动作维度
        self._log_2pi = float(np.log(2.0 * np.pi))                         # 高斯 logπ 常量项

        # 分位数水平 τ_i = (i-0.5)/N, i=0..N-1, 形状 [N]
        self.taus = torch.tensor(
            [(i + 0.5) / self.num_quantiles for i in range(self.num_quantiles)],
            dtype=torch.float32, device=self.device
        )

        # -------------------- Actor (与 QCPO 同一个线性高斯策略) + 两时间尺度 LR --------------------
        self.actor = Actor(self.state_dim, self.action_dim, args.init_std).to(self.device)
        self.actor_optimizer = Adam(self.actor.parameters(), 1.0, eps=1e-5)
        self.actor_scheduler = LambdaLR(
            self.actor_optimizer,
            lr_lambda=lambda k: lr_lambda(k, args.theta_a, args.theta_b, args.theta_c)
        )

        # -------------------- 单分布式 critic + target critic --------------------
        hidden = getattr(args, 'critic_hidden', [64, 64])
        if isinstance(hidden, str):                                        # 兼容 "64,64" 字符串
            hidden = [int(x) for x in hidden.split(',') if x.strip()]
        self.critic = DistributionalCritic(self.state_dim, self.action_dim,
                                           self.num_quantiles, list(hidden)).to(self.device)
        self.target_critic = DistributionalCritic(self.state_dim, self.action_dim,
                                                  self.num_quantiles, list(hidden)).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())       # target ← online (硬拷贝初始化)
        self.critic_optimizer = Adam(self.critic.parameters(),
                                     getattr(args, 'critic_lr', 1e-3), eps=1e-5)

        # -------------------- 拉格朗日乘子 λ (与 QCPO 完全一致) --------------------
        self.lambda_dual = torch.tensor([0.0], dtype=torch.float32,
                                        device=self.device, requires_grad=True)
        self.lambda_optimizer = Adam([self.lambda_dual], 1.0, eps=1e-5)
        self.lambda_scheduler = LambdaLR(
            self.lambda_optimizer,
            lr_lambda=lambda k: lr_lambda(k, args.lambda_a, args.lambda_b, args.lambda_c)
        )

        # -------------------- 运行时统计 (滚动窗口 + 计数 + 诊断量) --------------------
        self.episode_returns = deque(maxlen=max(1, self.est_interval))     # 最近若干条轨迹的折扣回报
        self.initial_states = deque(maxlen=max(1, self.est_interval))      # 最近若干条轨迹的初始状态 s_0
        self.learning_steps = 0                                            # critic/actor 累计更新步数
        self.last_dual_prob = 0.0                                          # 上次 dual 用的经验违反率
        self.last_cdf_initial = 0.0                                        # critic 估计的 P(Z≤q) (诊断)

        # -------------------- wandb (与 QCPO 一致) --------------------
        wandb.init(project=args.env_name, name=f"{args.algo_name}_{args.seed}",
                   config=vars(args), reinit=True, group=args.algo_name,
                   dir=getattr(args, 'wandb_dir', None))

    # ============================================================ 主训练循环 ============================================================
    def train(self):
        """
        主循环 (对应 qcpo.QCPO.train), 每个 episode:
          1. 采一条轨迹 (内部重置/递推 budget b、权重 d=γ^t、e=β^t)
          2. 内层: 逐 transition 更新 critic + actor, 重复 updates_per_episode 次
          3. 外层: 每 outer_interval 个 episode 更新一次 λ (经验 P(Z≤q), 不乘 β)
          4. 日志 + 学习率调度器步进
        """
        print(f"DQCACBeta: beta={self.beta}, alpha={self.q_alpha}, q={self.quantile_threshold}, "
              f"N={self.num_quantiles}, updates/epi={self.updates_per_episode}")

        for i_episode in range(self.max_episode + 1):
            # ===== 1. 采样轨迹 =====
            traj = self._rollout_one_episode()
            self.episode_returns.append(traj['disc_return'])
            self.initial_states.append(traj['s0'])

            # ===== 2. 内层更新 (critic + actor, 逐 transition) =====
            critic_info, actor_info = {}, {}
            for _ in range(self.updates_per_episode):
                critic_info = self.update_critic(traj)             # QRTD: 学回报分布
                actor_info = self.update_actor(traj)               # PG: w=γ^t Â_m - λ β^t Â_c
                if self.learning_steps % self.target_update_interval == 0:
                    self._soft_update_target()                     # Polyak 软更新 target critic
                self.learning_steps += 1

            # ===== 3. 外层更新 (λ), 每 outer_interval 个 episode =====
            if i_episode % self.outer_interval == 0 and i_episode != 0:
                self.update_dual(list(self.episode_returns))       # 经验 P(Z≤q), 与 QCPO 一致

            # ===== 4. 日志 + 调度器步进 =====
            self._log(i_episode, traj, critic_info, actor_info)
            self.actor_scheduler.step()                            # actor 学习率衰减
            self.lambda_scheduler.step()                           # λ 学习率衰减

    # ============================================================ 采样一条轨迹 ============================================================
    def _rollout_one_episode(self):
        """
        采一条完整轨迹, 每步记录 (s,a,r,s',done,b,d,e)。
        【关键】budget b 与权重 d=γ^t, e=β^t 的重置与递推都集中在这里:
            b_0=q, d_0=1, e_0=1
            b_{t+1}=(b_t-r_t)/γ,  d_{t+1}=γ d_t,  e_{t+1}=β e_t
        返回: 一个把各序列堆成张量的 dict (本环境固定 10 步, 整条轨迹当作一个 batch)
        """
        s = self._reset_env().reshape(-1)                          # 初始状态, 扁平 numpy [sd]
        b, d, e = float(self.quantile_threshold), 1.0, 1.0         # b_0=q, d_0=1, e_0=1
        S, A, R, S2, Done, B, Dw, Ew = [], [], [], [], [], [], [], []
        disc_return, disc = 0.0, 1.0                               # 折扣回报 U(τ) 及其当前折扣因子

        while True:
            a = self.choose_action(s)                              # 采样动作 → numpy (1,) float32
            s2, r, done, _ = self._step_env(a)                     # 环境步进 (4 值 API)
            s2 = s2.reshape(-1)

            # 记录本 transition 及其 bookkeeping 变量 (b_t, d_t=γ^t, e_t=β^t)
            S.append(s); A.append(a); R.append(r); S2.append(s2); Done.append(float(done))
            B.append(b); Dw.append(d); Ew.append(e)

            disc_return += disc * r; disc *= self.gamma            # 累计 U(τ)=Σγ^t r_t (供监控/dual)
            b = (b - r) / self.gamma                               # budget 递推 b_{t+1}=(b_t-r_t)/γ
            d *= self.gamma                                        # d_{t+1}=γ d_t
            e *= self.beta                                         # e_{t+1}=β e_t
            s = s2
            if done:                                               # 固定 10 步, 末步 done=True
                break

        to_t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)
        return {
            'states': to_t(S),          # [T, sd]
            'actions': to_t(A),         # [T, ad]
            'rewards': to_t(R),         # [T]
            'next_states': to_t(S2),    # [T, sd]
            'dones': to_t(Done),        # [T]
            'budgets': to_t(B),         # [T]  b_t
            'd': to_t(Dw),              # [T]  γ^t
            'e': to_t(Ew),              # [T]  β^t
            's0': np.asarray(S[0], dtype=np.float32),   # 初始状态 s_0 (供 critic-based Ĝ 诊断)
            'disc_return': float(disc_return),          # 标量 U(τ)
        }

    # ============================================================ Critic 更新 (QRTD) ============================================================
    def update_critic(self, traj):
        """
        一次分布式 TD 更新 (Quantile Regression TD + quantile Huber loss, 论文 eq 62-63)。
        目标分位数: a'~π(·|s'); y_j = r + γ(1-done)·ψ̄_j(s',a')  (ψ̄ 为 target critic)
            done=True 时退化为 y_j=r (无 bootstrap; 终止态由此自动得到正确的退化分布)
        损失: (1/N)Σ_i (1/N)Σ_j ρ^κ_{τ_i}(y_j - ψ_i(s,a))
        """
        s, a, r = traj['states'], traj['actions'], traj['rewards']
        s2, done = traj['next_states'], traj['dones']

        with torch.no_grad():                                      # target 不建图
            a2 = self._sample_actions(s2)                          # a'~π(·|s'), [T, ad]
            psi_next = self.target_critic(s2, a2)                  # ψ̄_j(s',a'), [T, N]
            # y_j = r + γ(1-done) ψ̄_j; r/done 用 unsqueeze(1) 广播到 [T, N]
            y = r.unsqueeze(1) + self.gamma * (1.0 - done.unsqueeze(1)) * psi_next

        psi = self.critic(s, a)                                    # ψ_i(s,a), [T, N] (建图)
        loss = self._quantile_huber_loss(psi, y)                   # 标量

        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.critic_grad_clip and self.critic_grad_clip > 0:    # 梯度裁剪 (稳定性)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_grad_clip)
        self.critic_optimizer.step()
        return {'critic/quantile_huber_loss': float(loss.item())}

    def _quantile_huber_loss(self, psi, y):
        """
        Quantile Huber Loss (QR-DQN 核心), ρ^κ_τ(u)=|τ-1{u<0}|·L_κ(u)/κ。
        对所有 (当前分位数 i, 目标分位数 j) 配对计算 TD error u=y_j-ψ_i, 再加权求和。

        输入: psi [T, N] (当前 ψ_i), y [T, N] (目标 y_j)
        输出: 标量 loss
        """
        # u[t,i,j] = y[t,j] - psi[t,i]; y.unsqueeze(1)=[T,1,N], psi.unsqueeze(2)=[T,N,1] → [T,N,N]
        u = y.unsqueeze(1) - psi.unsqueeze(2)
        abs_u = u.abs()
        # Huber: L_κ(u)=0.5u² (|u|≤κ) 否则 κ(|u|-0.5κ)
        huber = torch.where(abs_u <= self.huber_kappa,
                            0.5 * u.pow(2),
                            self.huber_kappa * (abs_u - 0.5 * self.huber_kappa))
        # 分位数权重 |τ_i - 1{u<0}|; taus 按 i 维 (dim=1) 排布
        taus = self.taus.view(1, -1, 1)                            # [1, N, 1]
        weight = (taus - (u.detach() < 0).float()).abs()           # [T, N, N]
        # ρ = weight·huber/κ; 对目标 j 求和(dim=2), 对分位数 i 取均值(dim=1), 再对 batch 取均值
        return (weight * huber / self.huber_kappa).sum(dim=2).mean(dim=1).mean()

    # ============================================================ Actor 更新 (per-transition) ============================================================
    def update_actor(self, traj):
        """
        一次 per-transition 策略梯度更新 (论文 eq 69 + 本工作的 β)。
        梯度权重: w_t = d_t·Â_m(s_t,a_t) - λ·e_t·Â_c(s_t,a_t,b_t) = γ^t Â_m - λ β^t Â_c
            目标信号 Q̂_m = critic 均值;  约束信号 Ψ̂ = critic 局部 CDF (查询 budget b)
            优势 Â_m = Q̂_m - V̂_m,  Â_c = Ψ̂ - V̂_c  (baseline 降方差)
        loss = -E[logπ(a|s)·w] - entropy_coef·E[H];  w 已 detach, 梯度只经 logπ 流向 θ。
        """
        s, a, b = traj['states'], traj['actions'], traj['budgets']
        d, e = traj['d'], traj['e']                                # γ^t, β^t

        with torch.no_grad():                                      # 优势对 actor 不建图
            psi = self.critic(s, a)                                # [T, N]
            q_m = psi.mean(dim=1)                                  # Q̂_m(s,a)=(1/N)Σ ψ_i, [T]
            psi_c = (psi <= b.unsqueeze(1)).float().mean(dim=1)    # Ψ̂(s,a,b)=(1/N)Σ1{ψ_i≤b}, [T]
            v_m, v_c = self._estimate_baselines(s, b)              # V̂_m(s), V̂_c(s,b), 各 [T]
            a_m = self._maybe_norm(q_m - v_m)                      # 均值优势 Â_m (可选 batch 归一化)
            a_c = self._maybe_norm(psi_c - v_c)                    # 约束优势 Â_c
            # 复合权重: 先算优势再乘 d/e; 约束项带 λ (用 detach, λ 仅由 dual 优化)
            w = d * a_m - self.lambda_dual.detach() * e * a_c      # [T]

        log_probs = self._compute_log_probs(s, a)                  # logπ(a|s), [T] (对 θ 可导)
        actor_loss = -(log_probs * w).mean() \
            - self.entropy_coef * self._entropy(s).mean()          # -E[logπ·w] - τ·E[H]

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.actor_grad_clip and self.actor_grad_clip > 0:      # 梯度裁剪
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip)
        self.actor_optimizer.step()                                # scheduler.step() 在 train() 末尾
        return {'actor/loss': float(actor_loss.item()),
                'advantage/mean_adv': float(a_m.mean().item()),
                'advantage/risk_adv': float(a_c.mean().item())}

    # ============================================================ Dual 更新 (λ) ============================================================
    def update_dual(self, recent_returns):
        """
        拉格朗日乘子 λ 的投影梯度上升 (与 qcpo.update_dual 完全一致, 不乘 β):
            P̂(Z≤q) = 经验违反概率 (滚动窗口);  λ ← [λ + ε_k(P̂-α)]_+
        设计 loss=-λ(P̂-α) ⇒ ∂loss/∂λ=-(P̂-α) ⇒ optimizer 下降一步即 λ←λ+lr(P̂-α)。
        """
        if len(recent_returns) > 0:
            p = float(np.mean([1.0 if z <= self.quantile_threshold else 0.0
                               for z in recent_returns]))
        else:
            p = 0.0
        self.last_dual_prob = p

        gap = torch.tensor([p - self.q_alpha], dtype=torch.float32, device=self.device)
        self.lambda_optimizer.zero_grad(set_to_none=True)
        (-self.lambda_dual * gap).backward()
        self.lambda_optimizer.step()
        with torch.no_grad():                                      # 投影到 [0, λ_max]
            self.lambda_dual.clamp_(min=0.0, max=self.lambda_max)

    # ============================================================ 辅助方法 ============================================================
    def _sample_actions(self, states):
        """
        从当前高斯策略采样动作 (重参数化), 支持 [B, sd] 批量输入。
        返回: [B, ad]。调用方负责是否包在 no_grad 里 (target/baseline 用 no_grad)。
        """
        means = self.actor(states)                                 # μ(s), [B, ad]
        std = torch.exp(self.actor.log_std).view(1, -1).expand_as(means)  # σ 广播, [B, ad]
        return means + torch.randn_like(means) * std               # a = μ + σ·ε

    def _compute_log_probs(self, states, actions):
        """
        对角高斯 logπ(a|s) = Σ_dim [-0.5((a-μ)²/σ² + 2logσ + log2π)]。
        对 μ (actor 权重) 可导 → 提供策略梯度; actions 来自 rollout (无梯度)。
        返回: [B]
        """
        means = self.actor(states)                                 # [B, ad], 对 θ 可导
        std = torch.exp(self.actor.log_std).view(1, -1).expand_as(means)
        var = std.pow(2)
        log_probs = -0.5 * (((actions - means) ** 2) / var
                            + 2.0 * torch.log(std) + self._log_2pi)
        return log_probs.sum(dim=-1)                               # 对动作维求和 → [B]

    def _entropy(self, states):
        """高斯熵 H = Σ_dim [0.5(1+log2π) + logσ]; log_std 固定 ⇒ 常数, 仅为接口完整。"""
        ent = (0.5 * (1.0 + self._log_2pi) + self.actor.log_std).sum()
        return ent.expand(states.shape[0])                         # [B]

    def _estimate_baselines(self, states, budgets):
        """
        用 K 个动作样本近似两个 baseline (论文 eq 64-65):
            V̂_m(s) = E_{a~π}[Q̂_m(s,a)],   V̂_c(s,b) = E_{a~π}[Ψ̂(s,a,b)]
        返回: v_m [T], v_c [T] (在 no_grad 上下文中调用)
        """
        b_col = budgets.unsqueeze(1)                               # [T, 1], 供 CDF 查询广播
        q_list, c_list = [], []
        for _ in range(self.num_action_samples):
            a = self._sample_actions(states)                       # 重新采样动作 [T, ad]
            psi = self.critic(states, a)                           # [T, N]
            q_list.append(psi.mean(dim=1))                         # Q̂_m, [T]
            c_list.append((psi <= b_col).float().mean(dim=1))      # Ψ̂(·,b), [T]
        v_m = torch.stack(q_list, dim=0).mean(dim=0)               # 对 K 取平均 → [T]
        v_c = torch.stack(c_list, dim=0).mean(dim=0)
        return v_m, v_c

    def _maybe_norm(self, x, eps=1e-8):
        """
        优势归一化 (advantage_norm=='separate' 时): (x-mean)/std。
        单样本 (numel<=1) 跳过; std 过小时只中心化。'none' 时原样返回。
        """
        if self.advantage_norm != 'separate' or x.numel() <= 1:
            return x
        std = x.std(unbiased=False)
        if std.item() < eps:
            return x - x.mean()
        return (x - x.mean()) / (std + eps)

    def _soft_update_target(self):
        """Polyak 软更新: target ← (1-τ)·target + τ·online。"""
        with torch.no_grad():
            for tp, op in zip(self.target_critic.parameters(), self.critic.parameters()):
                tp.data.lerp_(op.data, self.target_tau)            # lerp_: tp = tp + τ(op - tp)

    def _initial_cdf_estimate(self):
        """
        critic 估计的 P(Z≤q): 在最近若干初始状态 s_0 上求 mean Ψ̂(s_0,a_0,q)。
        (论文 eq 72; 本实现【仅作校准诊断】记录, 不驱动 λ —— λ 用经验违反率, 更鲁棒。)
        """
        if len(self.initial_states) == 0:
            return self.last_cdf_initial
        s0 = torch.as_tensor(np.asarray(self.initial_states, dtype=np.float32), device=self.device)
        q_b = torch.full((s0.shape[0],), self.quantile_threshold,
                         dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a0 = self._sample_actions(s0)
            psi = self.critic(s0, a0)
            self.last_cdf_initial = float((psi <= q_b.unsqueeze(1)).float().mean(dim=1).mean())
        return self.last_cdf_initial

    def _compute_episode_avg_risk(self):
        """读取 RiskSensitiveEnv.render() 返回的本 episode 平均风险等级 (与 qcpo 一致)。"""
        if hasattr(self.env, 'render'):
            risk_series = self.env.render()
            if risk_series is not None:
                try:
                    if len(risk_series) > 0:
                        return float(np.mean(risk_series))
                except TypeError:
                    pass
        return 0.0

    def _log(self, i_episode, traj, critic_info, actor_info):
        """wandb 日志 (与 QCPO 同款指标 + budget 统计 + critic 校准诊断)。"""
        returns = list(self.episode_returns)
        empirical_prob = float(np.mean([1.0 if z <= self.quantile_threshold else 0.0
                                        for z in returns])) if returns else 0.0
        avg_return = float(np.mean(returns)) if returns else 0.0
        quantile_return = float(np.percentile(returns, self.q_alpha * 100)) if returns else 0.0
        budgets_np = traj['budgets'].detach().cpu().numpy()

        log_dict = {
            'disc_reward/discounted_reward': traj['disc_return'],
            'disc_reward/aver_reward': avg_return,
            'disc_reward/quantile_reward': quantile_return,
            'quantile/q_est': quantile_return,                     # 经验 α-分位数 (监控)
            'quantile/margin_to_threshold': quantile_return - self.quantile_threshold,
            'constraint/empirical_prob': empirical_prob,           # P̂(Z≤q)
            'constraint/margin': self.q_alpha - empirical_prob,    # α - P̂, 正值=约束满足
            'constraint/cdf_estimate_initial': self._initial_cdf_estimate(),  # critic 估计 (诊断)
            'constraint/dual_prob': self.last_dual_prob,
            'lambda/value': float(self.lambda_dual.detach().item()),
            'lambda/lr': float(self.lambda_scheduler.get_last_lr()[0]),
            'action/avg_risk_episode': self._compute_episode_avg_risk(),
            'budget/min': float(np.min(budgets_np)),
            'budget/max': float(np.max(budgets_np)),
            'budget/mean': float(np.mean(budgets_np)),
            'training/learning_steps': self.learning_steps,
            'training/actor_lr': float(self.actor_scheduler.get_last_lr()[0]),
        }
        log_dict.update(critic_info)
        log_dict.update(actor_info)
        wandb.log(log_dict, step=i_episode)

        # 周期性控制台打印 (与 qcpo 风格一致)
        if i_episode % self.log_interval == 0 and i_episode != 0:
            print(f'Epi:{i_episode:05d} || disc_a_r:{avg_return:.03f} '
                  f'disc_q_r:{quantile_return:.03f} λ:{self.lambda_dual.item():.04f}')
            print(f'Epi:{i_episode:05d} || P(Z≤q):{empirical_prob:.03f} '
                  f'α:{self.q_alpha:.03f} Ghat_critic:{self.last_cdf_initial:.03f} '
                  f'critic_loss:{critic_info.get("critic/quantile_huber_loss", 0.0):.04f}\n')

    # ============================================================ 评估接口 (与 QCPO 一致) ============================================================
    def choose_action(self, state):
        """训练时采样动作: 输入扁平 numpy 状态, 输出 (ad,) float32 numpy 动作。"""
        s = torch.as_tensor(np.asarray(state, dtype=np.float32).reshape(1, -1), device=self.device)
        with torch.no_grad():
            a = self._sample_actions(s)                            # [1, ad]
        return a.squeeze(0).cpu().numpy().astype(np.float32)       # (ad,) float32, 满足 env 动作空间

    def select_action(self, state):
        """评估时采样动作 (与 choose_action 同, 随机策略; 供 monte_carlo_evaluate 调用)。"""
        return self.choose_action(state)

    def _reset_env(self):
        """兼容 Gym/Gymnasium reset 接口 (复制自 qcpo)。"""
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]
        return state

    def _step_env(self, action):
        """兼容 Gym/Gymnasium step 接口 (复制自 qcpo)。"""
        outcome = self.env.step(action)
        if isinstance(outcome, tuple):
            if len(outcome) == 5:
                state, reward, terminated, truncated, info = outcome
                done = terminated or truncated
            elif len(outcome) == 4:
                state, reward, done, info = outcome
            else:
                raise ValueError("env.step() 返回值格式异常")
        else:
            raise ValueError("env.step() 必须返回tuple")
        return state, reward, done, info

    def get_training_summary(self):
        """暴露最终约束指标给 run_experiment.py (hasattr 调用, 可选)。"""
        returns = list(self.episode_returns)
        empirical_prob = float(np.mean([1.0 if z <= self.quantile_threshold else 0.0
                                        for z in returns])) if returns else 0.0
        return {
            'lambda_final': float(self.lambda_dual.detach().item()),
            'empirical_violation_prob': empirical_prob,
            'cdf_estimate_initial': self.last_cdf_initial,
            'dual_prob': self.last_dual_prob,
            'beta': self.beta,
        }
