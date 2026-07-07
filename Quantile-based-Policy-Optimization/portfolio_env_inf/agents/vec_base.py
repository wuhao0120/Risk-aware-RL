# -*- coding: utf-8 -*-
"""
VecAgentBase —— portfolio_env_inf 四个 GPU 智能体 (QPO/QCPO/QPPO/DQCAC) 的共享基类。

为什么要基类 (risk_sensitive_env_inf 是各文件独立):
    本目录的硬性要求是【四算法对比口径统一】—— 统一的高斯策略工具、统一的向量化
    rollout 语义 (B 条 × n 步, 冻结策略)、统一的 wandb 指标名/x 轴 (progress/env_steps)、
    统一的环境日志 (集中度/换手率/各股权重)。把这些与算法无关的件收进基类,
    各算法文件只保留自己的更新数学 (与各自模板逐式一致), 避免四份拷贝漂移。

子类约定:
    - 在 __init__ 末尾调用 super().__init__(args, env) 之后再建算法专属组件;
    - rollout 用 self._rollout_core() 拿 (S,A,R 的 [n,B,*] 堆叠 + disc_returns), 自行后处理;
    - 每迭代日志: self._log_core(it, z, q_est_value, extra) —— z 为本迭代 [B] 回报。
"""
import numpy as np
import torch
import wandb

from utils import Actor                                       # 高斯策略 (线性+bias / MLP)
from envs import PortfolioVecTorch                            # 全 GPU B 路并行平稳组合环境


class VecAgentBase(object):
    """四算法共享: 环境/策略/采样/日志/评估接口 (算法更新逻辑在子类)。"""

    def __init__(self, args, env):
        """
        Args:
            args: 超参命名空间 (统一字段: device/gamma/q_alpha/quantile_threshold/log_interval/
                  num_envs/num_iterations/init_std/actor_hidden/wandb_* /algo_name/seed)
            env:  numpy 版 PortfolioEnv 实例 (参数来源 + 备用评估环境)
        """
        # -------------------- 统一基础参数 --------------------
        self.device = args.device                              # 计算设备
        self.gamma = args.gamma                                # 折扣 γ
        self.q_alpha = args.q_alpha                            # 分位水平 α
        # 约束阈值 q: 四算法【统一持有】(QPO/QPPO 目标不依赖 q, 但日志统一报告 P(Z≤q) → 可比)
        self.quantile_threshold = float(getattr(args, 'quantile_threshold', 0.0))
        self.log_interval = args.log_interval                  # 控制台打印间隔 (按迭代)
        self.num_envs = max(1, int(getattr(args, 'num_envs', 256)))          # 并行 env 数 B
        self.num_iterations = max(1, int(getattr(args, 'num_iterations', 400)))  # 迭代次数
        self.algo_name = getattr(args, 'algo_name', self.__class__.__name__)

        # -------------------- 环境与维度 --------------------
        self.env_name = getattr(args, 'env_name', 'PortfolioEnvInf')
        self.eval_env = env                                    # numpy env (备用评估)
        self.vec_env = PortfolioVecTorch(num_envs=self.num_envs,
                                         device=self.device, ref_env=env)  # 训练用 GPU 向量环境
        self.n = self.vec_env.n                                # 一段 rollout 步长 T (截断长度)
        self.state_dim = int(np.prod(env.observation_space.shape))   # 25
        self.action_dim = int(np.prod(env.action_space.shape))       # 5
        self._log_2pi = float(np.log(2.0 * np.pi))             # 高斯 logπ 常量项

        # -------------------- 统一高斯策略 (线性+bias, 可选 MLP) --------------------
        # actor_obs (统一, 关键稳定性设计, 见 DESIGN.md "稳定性修复"):
        #   'weights' (默认): actor 只看 obs 的权重块 (末 K 维) + bias —— 常数 μ/Σ 市场下
        #       最优策略可证明与 est 特征无关 (Markowitz 解状态无关, 仅交易成本带来
        #       对持仓的轻微依赖), 而 20 维含噪 est 特征只给 trajectory-level PG 注入
        #       梯度曲率与随机游走 (探针实证: W 在 est 坐标上爆炸/失稳)。
        #   'full': actor 看全部 25 维 (留给 regime-switching 扩展 / A-B 对照)。
        # 注: 观测本身仍包含全部特征 (critic / 未来扩展可用), 只是 actor 输入做切片。
        self.actor_obs_mode = str(getattr(args, 'actor_obs', 'weights')).lower()
        K = self.action_dim
        self.actor_in_dim = K if self.actor_obs_mode == 'weights' else self.state_dim
        hidden = getattr(args, 'actor_hidden', None)           # None=线性 (默认) / [64,64] 等
        if isinstance(hidden, str):                            # 兼容 "64,64" 字符串
            hidden = [int(x) for x in hidden.split(',') if x.strip()]
        self.actor = Actor(self.actor_in_dim, self.action_dim,
                           init_std=getattr(args, 'init_std', 0.5),
                           hidden=hidden).to(self.device)

        # -------------------- wandb (统一: 独立 project, env-step 为默认 x 轴) --------------------
        # 与 risk_sensitive_inf 同一套方案: define_metric 把所有指标 x 轴定为累计 env-step,
        # 四算法同预算 (num_iterations×B×n) 下曲线可直接叠加对比。
        # 保存 init 返回的 run 对象, 日志一律走 run.log() —— 不依赖 wandb.log 全局补丁
        # (实测本机偶发: init 成功但全局 wandb.log 仍是 preinit 包装 → 直接用 run 方法绕过)
        # 实验记录字段 (run_experiment.py 按 E1-/E2- 命名前缀自动填充, 也可 --set 覆盖):
        #   wandb_group: 实验分组 (同一对比图的 run 同组, wandb 界面按组折叠/着色)
        #   wandb_notes: 中文实验说明 (对比什么/为什么; 训练后 run_experiment 再追加最终评估)
        #   wandb_tags : 逗号分隔标签 ('E1,B512,QPO' → 列表), 供 runs 表筛选
        tags = [t.strip() for t in str(getattr(args, 'wandb_tags', '') or '').split(',')
                if t.strip()]
        self._wandb_run = wandb.init(
            project=getattr(args, 'wandb_project', 'portfolio_inf'),
            name=getattr(args, 'wandb_name', None) or f"{self.algo_name}_{args.seed}",
            config={k: str(v) if not isinstance(v, (int, float, bool, str, type(None)))
                    else v for k, v in vars(args).items()},   # 非标量转 str (device 等)
            reinit=True,
            group=getattr(args, 'wandb_group', None) or self.algo_name,
            notes=getattr(args, 'wandb_notes', None),
            tags=tags or None,
            dir=getattr(args, 'wandb_dir', None))
        if self._wandb_run is not None:
            self._wandb_run.define_metric("progress/env_steps")
            self._wandb_run.define_metric("*", step_metric="progress/env_steps")

    # ============================================================ 高斯策略工具 (与模板逐式一致) ============================================================
    def _actor_in(self, states):
        """actor 输入切片: 'weights' 模式取 obs 末 K 维 (当前权重), 'full' 原样。"""
        if self.actor_obs_mode == 'weights':
            return states[:, -self.action_dim:]
        return states

    def _sample_actions(self, states):
        """从当前高斯策略重参数化采样: a=μ(s)+σ·ε, 支持 [*,sd] 批量。返回 [*,ad]。"""
        means = self.actor(self._actor_in(states))             # μ(s), [*, ad] (输入按模式切片)
        std = torch.exp(self.actor.log_std).view(1, -1).expand_as(means)  # σ 广播
        return means + torch.randn_like(means) * std           # a = μ + σ·ε

    def _compute_log_probs(self, states, actions):
        """对角高斯 logπ(a|s)=Σ_dim[-0.5((a-μ)²/σ²+2logσ+log2π)], 对 θ 可导。返回 [*]。"""
        means = self.actor(self._actor_in(states))             # (输入按模式切片)
        std = torch.exp(self.actor.log_std).view(1, -1).expand_as(means)
        var = std.pow(2)
        log_probs = -0.5 * (((actions - means) ** 2) / var
                            + 2.0 * torch.log(std) + self._log_2pi)
        return log_probs.sum(dim=-1)                           # 对动作维求和 → [*]

    def _entropy(self, states):
        """高斯熵 H=Σ_dim[0.5(1+log2π)+logσ]; log_std 固定 ⇒ 常数, 仅接口完整。"""
        ent = (0.5 * (1.0 + self._log_2pi) + self.actor.log_std).sum()
        return ent.expand(states.shape[0])

    # ============================================================ 向量化 rollout 骨架 ============================================================
    def _rollout_core(self, keep_logp=False):
        """
        冻结策略并行采 B 条轨迹 (n 步锁步, 全 GPU)。

        Returns dict:
            S [n,B,sd] / A [n,B,ad] / R [n,B]       —— 原始堆叠 (子类自行摊平/后处理)
            S2_last [B,sd]                          —— 截断次态 s_T (DQCAC bootstrap 用)
            disc_returns [B]                        —— 每条轨迹 Z=Σγ^t r_t
            logp [n,B] (keep_logp=True 时)          —— 采集时刻的 logπ_old (QPPO 用)
        """
        n, B = self.n, self.num_envs
        s = self.vec_env.reset()                               # [B, sd]
        S, A, R, LP = [], [], [], []
        disc_return = torch.zeros(B, dtype=torch.float32, device=self.device)
        disc = 1.0                                             # γ^t
        s2 = s
        for t in range(n):
            with torch.no_grad():                              # 冻结策略采集, 不建图
                a = self._sample_actions(s)                    # [B, ad]
                if keep_logp:
                    LP.append(self._compute_log_probs(s, a))   # logπ_old(a|s), [B]
            s2, r, done = self.vec_env.step(a)                 # [B,sd], [B], bool
            S.append(s); A.append(a); R.append(r)
            disc_return = disc_return + disc * r               # 累计 Z
            disc *= self.gamma
            s = s2
        out = {
            'S': torch.stack(S, dim=0),                        # [n,B,sd]
            'A': torch.stack(A, dim=0),                        # [n,B,ad]
            'R': torch.stack(R, dim=0),                        # [n,B]
            'S2_last': s2,                                     # [B,sd] 截断次态 s_T
            'disc_returns': disc_return,                       # [B]
        }
        if keep_logp:
            out['logp'] = torch.stack(LP, dim=0)               # [n,B]
        return out

    # ============================================================ 统一日志 ============================================================
    def _env_action_stats(self):
        """
        从 vec_env 取本迭代的环境侧动作统计 (统一日志):
            action/avg_risk_episode : 每步 batch 平均 max 权重的整段均值 (持仓集中度 ∈[1/K,1],
                                      对应 risk_sensitive 系列同名指标的"风险水平"语义)
            action/avg_turnover     : 平均买入换手率
            action/w{i}             : 各股 batch 平均权重 (rollout 末态)
        """
        out = {}
        series = self.vec_env.render() if hasattr(self.vec_env, 'render') else None
        if series is not None and len(series) > 0:
            out['action/avg_risk_episode'] = float(np.mean(series))
        st = self.vec_env.stats() if hasattr(self.vec_env, 'stats') else {}
        if 'avg_turnover' in st:
            out['action/avg_turnover'] = st['avg_turnover']
        for i, wi in enumerate(st.get('mean_weights', [])):
            out[f'action/w{i}'] = float(wi)
        return out

    def _log_core(self, it, z_np, q_est_value, extra=None):
        """
        统一 wandb 日志 (键名与 risk_sensitive_inf 完全一致 + 组合环境专属 action 统计)。

        Args:
            it:          当前迭代
            z_np:        本迭代 B 条轨迹回报 (numpy [B])
            q_est_value: 'quantile/q_est' 取值 —— QPO/QPPO 传【学习的分位数估计】,
                         QCPO/DQCAC 传【本批经验 α-分位数】(与各自模板一致, 键名统一可比)
            extra:       算法专属补充指标 dict
        """
        q = self.quantile_threshold
        empirical_prob = float(np.mean(z_np <= q))             # P̂(Z≤q) (统一: 同一 q)
        avg_return = float(np.mean(z_np))
        return_std = float(np.std(z_np))
        quantile_return = float(np.percentile(z_np, self.q_alpha * 100))
        self.last_empirical_prob = empirical_prob              # 子类 summary 用

        env_steps = (it + 1) * self.num_envs * self.n          # 统一 x 轴: 累计 env-step
        log_dict = {
            'disc_reward/discounted_reward': avg_return,
            'disc_reward/aver_reward': avg_return,
            'disc_reward/quantile_reward': quantile_return,
            'disc_reward/return_std': return_std,
            'quantile/q_est': float(q_est_value),
            'quantile/margin_to_threshold': quantile_return - q,
            'constraint/empirical_prob': empirical_prob,
            'constraint/margin': self.q_alpha - empirical_prob,
            'progress/iteration': it,
            'progress/trajectories': (it + 1) * self.num_envs,
            'progress/env_steps': env_steps,
        }
        log_dict.update(self._env_action_stats())              # 集中度/换手/权重
        if extra:
            log_dict.update(extra)
        if self._wandb_run is not None:                        # 走 run.log (见 __init__ 注释)
            self._wandb_run.log(log_dict, step=it)

        if it % self.log_interval == 0 and it != 0:            # 周期性控制台打印 (模板风格)
            print(f'Iter:{it:05d} (env_step:{env_steps}) || disc_a_r:{avg_return:.03f} '
                  f'disc_q_r:{quantile_return:.03f} P(Z<=q):{empirical_prob:.03f}')
        return log_dict

    # ============================================================ 评估接口 (统一) ============================================================
    def choose_action(self, state):
        """输入扁平 numpy 状态 → (ad,) float32 numpy 动作 (随机策略采样)。"""
        s = torch.as_tensor(np.asarray(state, dtype=np.float32).reshape(1, -1),
                            device=self.device)
        with torch.no_grad():
            a = self._sample_actions(s)
        return a.squeeze(0).cpu().numpy().astype(np.float32)

    def select_action(self, state):
        """评估采样动作 (供 monte_carlo_evaluate_constraint 调用)。"""
        return self.choose_action(state)
