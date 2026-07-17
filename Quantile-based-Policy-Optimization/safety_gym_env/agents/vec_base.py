# -*- coding: utf-8 -*-
"""
VecAgentBase (safety_gym_env · CMDP 版) —— QCPO / DQC-AC-β 共享基类。

与 portfolio_env_inf/agents/vec_base.py 逐块对齐, 三处 CMDP 适配 (数学骨架不变):
    1. **环境**: SafetyVecEnv (CPU mujoco 同步 rollout) 替换 PortfolioVecTorch;
       rollout 额外收集 **cost 流** (目标用 reward, 约束用 cost — 两条不同回报流)。
    2. **策略**: tanh-μ 高斯 (动作 ∈(-1,1)+噪声, env clip); actor 看【全 obs】(无 portfolio
       的 'weights' 切片 —— lidar/传感器全维度都有用)。logπ 仍是普通对角高斯。
    3. **日志**: wandb 用与 risk_sensitive 相同的下尾口径记约束 —
       Z=-C, q=-d, P(Z≤q)=P(C≥d); 任务回报 R 放 disc_reward/*。

优化问题 (CMDP):  max_θ E[R]   s.t.   P_θ(C ≥ d) ≤ ω
    R = Σ_t γ^t r_t        (奖励折扣回报, 目标; γ=0.99)
    C = Σ_t γc^t c_t       (cost 折扣回报, 约束变量; γc 默认=γ)
    d = cost_limit (约束阈值, 对 C 校准)   ω = q_alpha (目标 outage 概率)
    日志等价: Z=-C, q=-d, α=ω → P(Z≤q)≤α 与上尾同值。
"""
import os
import numpy as np
import torch
import torch.nn as nn
import wandb

from utils import Actor, ObservationNormalizer                # 策略网络 + 可选逐维观测归一化
from envs import make_vec_env                                 # CPU 向量化工厂 (mp 多进程 / sync 串行)


_WANDB_PRIVATE_CONFIG_KEYS = {
    'wandb_dir',                 # 当前机器的绝对工作目录，不影响算法复现
    'checkpoint_dir',            # 本地持久化位置；结构配置已经由其它字段完整描述
    'calibration_source_checkpoint',  # critic校准源快照的绝对路径
}
_WANDB_SECRET_KEY_MARKERS = (
    'api_key', 'password', 'secret', 'token', 'credential',
)


def wandb_public_config(args):
    """
    [函数简介]: 构造只含公开实验语义的W&B配置，不上传机器路径或凭据。
    [输入输出]: args是argparse namespace；返回可直接传给wandb.init的dict。
    [算法影响]: 仅改变外部日志元数据，不参与网络前向、随机数或优化器更新。

    除显式私有字段外，任何绝对路径值也会被剔除；这样后续新增路径参数时
    不需要依赖维护者记得同步更新黑名单。list/dict等超参仍沿用历史行为转成
    字符串，保证W&B表格可读且不改变已有标量字段类型。
    """

    public = {}
    for key, value in vars(args).items():
        key_lower = str(key).lower()
        if key in _WANDB_PRIVATE_CONFIG_KEYS:
            continue
        if any(marker in key_lower for marker in _WANDB_SECRET_KEY_MARKERS):
            continue

        # torch.device等对象先转成历史格式字符串；PathLike值则显式展开，
        # 之后统一判断绝对路径，避免把/vepfs用户名或机器目录发送到云端。
        if isinstance(value, os.PathLike):
            value = os.fspath(value)
        if isinstance(value, str) and os.path.isabs(os.path.expanduser(value)):
            continue
        public[key] = (
            value if isinstance(value, (int, float, bool, str, type(None)))
            else str(value))
    return public


class VecAgentBase(object):
    """QCPO/DQCAC 共享: 环境/策略/采样(含cost)/日志(outage)/评估接口 (更新逻辑在子类)。"""

    def __init__(self, args, env):
        """
        Args:
            args: 超参命名空间 (device/gamma/cost_gamma/q_alpha(=ω)/cost_limit(=d)/horizon/
                  num_envs/num_iterations/init_std/actor_hidden/wandb_*/algo_name/seed)
            env:  SafetyEnv 实例 (env_id 来源 + 维度 + 备用评估环境)
        """
        # -------------------- 统一基础参数 --------------------
        self.device = args.device                             # 计算设备 (网络在 GPU)
        self.gamma = args.gamma                               # 奖励折扣 γ (目标)
        self.cost_gamma = float(getattr(args, 'cost_gamma', args.gamma))  # cost 折扣 γc (默认=γ)
        self.q_alpha = args.q_alpha                           # ω: 目标 outage 概率 P(C≥d)≤ω
        # 约束阈值 d (对折扣 cost 回报 C 生效)。字段名沿用 quantile_threshold/cost_limit 兼容。
        self.cost_limit = float(getattr(args, 'cost_limit',
                                        getattr(args, 'quantile_threshold', 25.0)))
        self.log_interval = args.log_interval                 # 控制台打印间隔 (按迭代)
        self.num_envs = max(1, int(getattr(args, 'num_envs', 8)))              # 并行 env 数 B (CPU!)
        self.num_iterations = max(1, int(getattr(args, 'num_iterations', 100)))
        self.horizon = int(getattr(args, 'horizon', getattr(args, 'env_n', env.n)))  # 段长 T
        self.algo_name = getattr(args, 'algo_name', self.__class__.__name__)
        self.seed = int(getattr(args, 'seed', 0))              # checkpoint 元数据使用的训练 seed

        # -------------------- 可选评估 checkpoint（默认完全关闭） --------------------
        # checkpoint_interval=0 时不触碰磁盘，保持所有历史实验的时序与开销。
        # 正数表示每隔多少个 rollout 保存一次“更新前策略”；该快照与本批真实
        # reward/outage 严格对应，专门避免最后一次 PPO 后策略与训练日志错位。
        checkpoint_dir = getattr(args, 'checkpoint_dir', None)
        self.checkpoint_dir = None if checkpoint_dir in {None, '', 'none'} else str(checkpoint_dir)
        self.checkpoint_interval = max(
            0, int(getattr(args, 'checkpoint_interval', 0)))
        if self.checkpoint_interval > 0 and self.checkpoint_dir is None:
            raise ValueError("checkpoint_interval>0 requires checkpoint_dir")
        if self.checkpoint_dir is not None and not os.path.isabs(self.checkpoint_dir):
            checkpoint_root = str(getattr(args, 'wandb_dir', os.getcwd()))
            self.checkpoint_dir = os.path.join(checkpoint_root, self.checkpoint_dir)

        # 保存原始配置而不是从运行中对象反推网络结构。torch.device 等非基础类型
        # 转为字符串；list/dict 保持原形，eval-only 可直接重建同形 agent。
        self._checkpoint_config = {}
        for key, value in vars(args).items():
            if isinstance(value, (type(None), bool, int, float, str, list, tuple, dict)):
                self._checkpoint_config[key] = value
            else:
                self._checkpoint_config[key] = str(value)

        # -------------------- 环境与维度 --------------------
        self.env_name = getattr(args, 'env_name', env.env_id)
        self.eval_env = env                                   # SafetyEnv (备用 numpy 评估)
        # vec_backend: 'mp' 多进程并行 (默认, CPU 多核仿真) / 'sync' 单进程串行 (调试/对拍)
        self.vec_env = make_vec_env(env.env_id, num_envs=self.num_envs,
                                    horizon=self.horizon, device=self.device,
                                    ref_env=env, seed=int(getattr(args, 'seed', 0)),
                                    backend=getattr(args, 'vec_backend', 'mp'))
        self.n = self.vec_env.n                               # 一段 rollout 步长 T
        self.state_dim = self.vec_env.obs_dim                 # 观测维度 (60/76)
        self.action_dim = self.vec_env.act_dim                # 动作维度 (2)
        self._log_2pi = float(np.log(2.0 * np.pi))            # 高斯 logπ 常量项
        # 可选 QCPO_refs 风格的逐维观测归一化；默认关闭，保证旧实验可精确复现。
        self.normalize_observation = bool(getattr(args, 'normalize_observation', False))
        # 冻结模式只用于从成熟策略checkpoint重新采样、校准新critic。默认False时
        # 完全保持历史训练路径；True时仍使用已恢复moments做前向，但禁止新rollout
        # 改写统计量，否则所谓“冻结策略”会因输入变换漂移而并不真正冻结。
        self.freeze_observation_stats = bool(
            getattr(args, 'freeze_observation_stats', False))
        self.obs_norm_var_clip = float(getattr(args, 'obs_norm_var_clip', 1e-6))
        self.obs_norm_clip = float(getattr(args, 'obs_norm_clip', 10.0))
        self.obs_norm_warmup_iters = max(
            0, int(getattr(args, 'obs_norm_warmup_iters', 1)))
        self.obs_normalizer = ObservationNormalizer(
            self.state_dim, var_clip=self.obs_norm_var_clip,
            value_clip=self.obs_norm_clip).to(self.device)

        # -------------------- 统一高斯策略 (tanh-μ MLP) --------------------
        hidden = getattr(args, 'actor_hidden', [256, 256])    # 默认 MLP (观测高维)
        if isinstance(hidden, str):
            hidden = [int(x) for x in hidden.split(',') if x.strip()]
        self.actor = Actor(self.state_dim, self.action_dim,
                           init_std=getattr(args, 'init_std', 0.5),
                           hidden=hidden,
                           learn_std=bool(getattr(args, 'learn_std', False))).to(self.device)

        # -------------------- wandb (独立 project, env-step 为默认 x 轴) --------------------
        tags = [t.strip() for t in str(getattr(args, 'wandb_tags', '') or '').split(',')
                if t.strip()]
        # W&B默认会采集host、username、Git remote/commit、代码和system stats。
        # 本项目只需要训练指标与公开超参；关闭机器元数据既减少隐私暴露，也不
        # 改变wandb.log的训练曲线。GPU/CPU资源仍由本地nvidia-smi/作业日志监控。
        wandb_settings = wandb.Settings(
            _disable_machine_info=True,
            _disable_meta=True,
            _disable_stats=True,
            _save_requirements=False,
            disable_git=True,
            disable_code=True,
            disable_job_creation=True,
            save_code=False)
        self._wandb_run = wandb.init(
            project=getattr(args, 'wandb_project', 'safety_gym_qcrl'),
            name=getattr(args, 'wandb_name', None) or f"{self.algo_name}_{args.seed}",
            config=wandb_public_config(args),
            reinit=True,
            group=getattr(args, 'wandb_group', None) or self.algo_name,
            notes=getattr(args, 'wandb_notes', None),
            tags=tags or None,
            save_code=False,
            settings=wandb_settings,
            dir=getattr(args, 'wandb_dir', None))
        if self._wandb_run is not None:
            self._wandb_run.define_metric("progress/env_steps")
            self._wandb_run.define_metric("*", step_metric="progress/env_steps")

        self.last_outage_prob = 0.0                           # = last empirical P(Z≤q)=P(C≥d)
        self.last_empirical_prob = 0.0                        # 与 risk_sensitive 同名

    # ============================================================ 评估 checkpoint ============================================================
    def _evaluation_checkpoint_state(self, iteration, phase, metrics=None):
        """
        构造仅用于恢复评估的轻量 checkpoint。

        保存范围:
            1. agent 直接持有的全部 nn.Module（actor、critic、target、obs RMS）；
            2. lambda_dual 及 PID/window 等会影响诊断解释的运行时状态；
            3. 原始结构配置、保存相位和与该策略对应的 rollout 指标。

        不保存 optimizer/scheduler 动量，因此该格式不声称支持无损续训。这样可把
        每个快照控制在模型权重规模，允许短实验按迭代保存并用大样本统一复评。
        """
        modules = {}
        for name, value in vars(self).items():
            if isinstance(value, nn.Module):
                # 显式搬到 CPU，使 checkpoint 可跨 GPU 序号加载，也避免保存 device 依赖。
                modules[name] = {
                    key: tensor.detach().cpu()
                    for key, tensor in value.state_dict().items()
                }

        # lambda 不是 Module parameter；单独复制以保留快照的约束强度解释。
        tensors = {}
        if hasattr(self, 'lambda_dual') and torch.is_tensor(self.lambda_dual):
            tensors['lambda_dual'] = self.lambda_dual.detach().cpu().clone()

        # 这些量不影响纯评估动作，但恢复后写 summary 时必须对应原快照，而不是初始化值。
        runtime_names = (
            'learning_steps', 'pid_i', 'last_empirical_prob', 'last_outage_prob',
            'last_dual_prob', 'last_dual_raw_prob', 'last_dual_window_prob',
            'last_dual_cost_quantile', 'last_dual_prob_gap',
            'last_dual_control_prob_gap', 'last_dual_quantile_gap',
            'last_dual_control_error', 'last_dual_filtered_error',
            'last_pid_episode_scale', 'last_pid_effective_leak', 'last_pid_delta',
            'last_pid_actual_delta', 'last_pid_proportional', 'last_pid_output',
            'pid_update_events', 'pid_rollouts_since_update',
            'last_pid_update_due', 'last_pid_rollouts_accumulated',
            'last_pid_update_batch_episodes',
            'actor_update_events', 'actor_rollouts_since_update',
            'last_actor_first_epoch_ratio_max_error',
            'last_actor_update_batch_trajectories',
            'last_actor_updates_completed', 'last_actor_approx_kl',
            'last_actor_clip_fraction',
        )
        runtime = {
            name: getattr(self, name)
            for name in runtime_names
            if hasattr(self, name)
        }
        if hasattr(self, 'empirical_cost_window'):
            runtime['empirical_cost_window'] = list(self.empirical_cost_window)

        env_steps = max(0, int(iteration) + 1) * self.num_envs * self.n
        return {
            'format': 'safety-gym-eval-checkpoint-v1',
            'algo': self.algo_name,
            'env': self.env_name,
            'seed': self.seed,
            'iteration': int(iteration),
            'env_steps': int(env_steps),
            'phase': str(phase),
            'metrics': dict(metrics or {}),
            'config': dict(self._checkpoint_config),
            'modules': modules,
            'tensors': tensors,
            'runtime': runtime,
        }

    def save_evaluation_checkpoint(self, path, iteration, phase, metrics=None):
        """
        原子写入评估 checkpoint，返回绝对路径。

        先写同目录 .tmp，再用 os.replace 原子替换目标；SSH 中断最多留下 .tmp，
        不会把半个文件误当成可加载 checkpoint。
        """
        checkpoint_path = os.path.abspath(str(path))
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        temporary_path = checkpoint_path + '.tmp'
        payload = self._evaluation_checkpoint_state(
            iteration=iteration, phase=phase, metrics=metrics)
        torch.save(payload, temporary_path)
        os.replace(temporary_path, checkpoint_path)
        print(
            f"[checkpoint] saved phase={phase} step={payload['env_steps']} "
            f"path={checkpoint_path}")
        return checkpoint_path

    def load_evaluation_checkpoint(self, checkpoint):
        """
        严格恢复评估权重与诊断状态。

        checkpoint 可为 torch.load 后的 dict 或文件路径。Module 名称和 tensor shape
        必须逐项匹配；结构不一致直接报错，避免静默部分加载产生不可解释结果。
        """
        if isinstance(checkpoint, (str, os.PathLike)):
            payload = torch.load(
                os.fspath(checkpoint), map_location='cpu', weights_only=False)
        else:
            payload = checkpoint
        if payload.get('format') != 'safety-gym-eval-checkpoint-v1':
            raise ValueError(f"unsupported checkpoint format: {payload.get('format')!r}")
        if str(payload.get('algo')) != str(self.algo_name):
            raise ValueError(
                f"checkpoint algo={payload.get('algo')!r} != agent algo={self.algo_name!r}")
        if str(payload.get('env')) != str(self.env_name):
            raise ValueError(
                f"checkpoint env={payload.get('env')!r} != agent env={self.env_name!r}")

        for name, state in payload.get('modules', {}).items():
            module = getattr(self, name, None)
            if not isinstance(module, nn.Module):
                raise KeyError(f"checkpoint module {name!r} is absent from reconstructed agent")
            module.load_state_dict(state, strict=True)

        for name, saved_tensor in payload.get('tensors', {}).items():
            current = getattr(self, name, None)
            if not torch.is_tensor(current):
                raise KeyError(f"checkpoint tensor {name!r} is absent from reconstructed agent")
            with torch.no_grad():
                current.copy_(saved_tensor.to(device=current.device, dtype=current.dtype))

        for name, value in payload.get('runtime', {}).items():
            if name == 'empirical_cost_window' and hasattr(self, name):
                window = getattr(self, name)
                window.clear()
                window.extend(value)
            elif hasattr(self, name):
                setattr(self, name, value)

        print(
            f"[checkpoint] loaded phase={payload.get('phase')} "
            f"step={payload.get('env_steps')} algo={payload.get('algo')} "
            f"env={payload.get('env')}")
        return payload

    def load_policy_calibration_checkpoint(self, checkpoint):
        """
        只恢复产生行为数据所需的actor与observation preprocessing。

        该接口服务于冻结策略的critic校准：critic/target/optimizer、lambda/PID和
        runtime都必须保持新agent的初始状态，避免把源run的critic误当成本轮方法
        的起点。recurrent actor自己的obs_rms包含在actor state_dict中；raw cost
        critic使用的独立obs_normalizer也一并恢复，确保输入坐标系与成熟策略一致。
        """
        if isinstance(checkpoint, (str, os.PathLike)):
            payload = torch.load(
                os.fspath(checkpoint), map_location='cpu', weights_only=False)
        else:
            payload = checkpoint
        if payload.get('format') != 'safety-gym-eval-checkpoint-v1':
            raise ValueError(f"unsupported checkpoint format: {payload.get('format')!r}")
        if str(payload.get('algo')) != str(self.algo_name):
            raise ValueError(
                f"checkpoint algo={payload.get('algo')!r} != agent algo={self.algo_name!r}")
        if str(payload.get('env')) != str(self.env_name):
            raise ValueError(
                f"checkpoint env={payload.get('env')!r} != agent env={self.env_name!r}")

        saved_modules = payload.get('modules', {})
        restored = []
        for name in ('actor', 'obs_normalizer'):
            module = getattr(self, name, None)
            state = saved_modules.get(name)
            if not isinstance(module, nn.Module) or state is None:
                raise KeyError(f"checkpoint calibration module {name!r} is unavailable")
            module.load_state_dict(state, strict=True)
            restored.append(name)

        print(
            f"[checkpoint] loaded policy-only modules={restored} "
            f"phase={payload.get('phase')} step={payload.get('env_steps')} "
            f"algo={payload.get('algo')} env={payload.get('env')}")
        return payload

    def _maybe_save_rollout_checkpoint(self, iteration, batch):
        """
        按 interval 保存“产生当前 rollout 的策略”，必须在任何参数更新前调用。

        batch 的 reward/outage 是该 actor 的直接观测证据；后续可先按这些便宜指标
        筛选少量快照，再用同一 512/1024-episode 协议复评，避免保存后策略错配。
        """
        if self.checkpoint_dir is None or self.checkpoint_interval <= 0:
            return None
        if (int(iteration) + 1) % self.checkpoint_interval != 0:
            return None

        rewards = batch['disc_return'].detach().cpu().numpy()
        costs = batch['disc_cost'].detach().cpu().numpy()
        metrics = {
            'rollout_reward_mean': float(np.mean(rewards)),
            'rollout_reward_std': float(np.std(rewards)),
            'rollout_outage': float(np.mean(costs >= self.cost_limit)),
            'rollout_cost_mean': float(np.mean(costs)),
            'lambda_before_update': float(
                self.lambda_dual.detach().item()) if hasattr(self, 'lambda_dual') else 0.0,
        }
        env_steps = (int(iteration) + 1) * self.num_envs * self.n
        filename = f"rollout_step{env_steps:09d}.pt"
        return self.save_evaluation_checkpoint(
            os.path.join(self.checkpoint_dir, filename),
            iteration=iteration,
            phase='pre_update_rollout_policy',
            metrics=metrics)

    # ============================================================ 高斯策略工具 ============================================================
    def _normalize_states(self, states):
        """用 rollout 间共享且当前冻结的 running moments 归一化；关闭开关时原样返回。"""
        if not self.normalize_observation:
            return states
        return self.obs_normalizer(states)

    def _sample_actions(self, states):
        """从当前高斯策略采样 a=μ(s)+σ·ε, μ=tanh(net(s))∈(-1,1)。返回 [*,ad] (env 会 clip 越界)。"""
        policy_states = self._normalize_states(states)        # actor 与 logπ 必须使用同一输入变换
        means = self.actor(policy_states)                     # μ(s), [*, ad] (已 tanh)
        std = torch.exp(self.actor.log_std).view(1, -1).expand_as(means)
        return means + torch.randn_like(means) * std          # a = μ + σ·ε

    def _compute_log_probs(self, states, actions):
        """对角高斯 logπ(a|s)=Σ_dim[-0.5((a-μ)²/σ²+2logσ+log2π)], 对 θ 可导。返回 [*]。
        注: 不做 tanh 变量替换修正 (动作越界由 env clip, 与 NIPS QCPO 同口径)。"""
        policy_states = self._normalize_states(states)        # 与采样路径共用相同 running moments
        means = self.actor(policy_states)
        std = torch.exp(self.actor.log_std).view(1, -1).expand_as(means)
        var = std.pow(2)
        log_probs = -0.5 * (((actions - means) ** 2) / var
                            + 2.0 * torch.log(std) + self._log_2pi)
        return log_probs.sum(dim=-1)

    def _entropy(self, states):
        """高斯熵 (log_std 固定时为常数, 仅接口完整)。"""
        ent = (0.5 * (1.0 + self._log_2pi) + self.actor.log_std).sum()
        return ent.expand(states.shape[0])

    # ============================================================ 向量化 rollout 骨架 (含 cost 流) ============================================================
    def _rollout_core(self, keep_logp=False):
        """
        冻结策略并行采 B 条轨迹 (n 步锁步, 物理在 CPU, 张量在 device)。

        Returns dict:
            S [n,B,sd] / A [n,B,ad] / R [n,B] (reward) / C [n,B] (cost)  —— 原始堆叠
            S2_last [B,sd]                          —— 截断次态 s_T (DQCAC bootstrap 用)
            disc_return [B]   = Σ γ^t r_t           —— 奖励回报 R (目标)
            disc_cost   [B]   = Σ γc^t c_t          —— cost 折扣回报 C (约束变量)
            undisc_cost [B]   = Σ c_t               —— 未折扣累计 cost (报告/对论文口径)
            logp [n,B] (keep_logp=True 时)          —— 采集时 logπ_old
        """
        n, B = self.n, self.num_envs
        s = self.vec_env.reset()                              # [B, sd]
        S, A, R, C, LP = [], [], [], [], []
        disc_return = torch.zeros(B, dtype=torch.float32, device=self.device)   # R
        disc_cost = torch.zeros(B, dtype=torch.float32, device=self.device)     # C (折扣)
        undisc_cost = torch.zeros(B, dtype=torch.float32, device=self.device)   # Σc
        dr, dc = 1.0, 1.0                                     # γ^t, γc^t
        s2 = s
        for t in range(n):
            with torch.no_grad():                             # 冻结策略采集, 不建图
                a = self._sample_actions(s)                   # [B, ad]
                if keep_logp:
                    LP.append(self._compute_log_probs(s, a))  # logπ_old, [B]
            s2, r, c, done = self.vec_env.step(a)             # [B,sd],[B],[B],bool
            S.append(s); A.append(a); R.append(r); C.append(c)
            disc_return = disc_return + dr * r; dr *= self.gamma      # 累计 R
            disc_cost = disc_cost + dc * c; dc *= self.cost_gamma     # 累计 C (折扣)
            undisc_cost = undisc_cost + c                            # 累计 Σc
            s = s2
        out = {
            'S': torch.stack(S, dim=0),                       # [n,B,sd]
            'A': torch.stack(A, dim=0),                       # [n,B,ad]
            'R': torch.stack(R, dim=0),                       # [n,B] reward
            'C': torch.stack(C, dim=0),                       # [n,B] cost
            'S2_last': s2,                                     # [B,sd]
            'disc_return': disc_return,                       # [B] R
            'disc_cost': disc_cost,                           # [B] C (折扣)
            'undisc_cost': undisc_cost,                       # [B] Σc
        }
        if keep_logp:
            out['logp'] = torch.stack(LP, dim=0)              # [n,B]
        return out

    # ============================================================ 统一日志 (下尾口径, 对齐 risk_sensitive) ============================================================
    def _env_action_stats(self):
        """从 vec_env.stats 取本迭代环境侧统计 (只走 stats, 避免与 render 重复记 avg_step_cost)。"""
        out = {}
        st = self.vec_env.stats() if hasattr(self.vec_env, 'stats') else {}
        for k, v in st.items():
            out[f'action/{k}'] = float(v)
        return out

    def _log_core(self, it, R_np, Zc_np, undiscC_np, extra=None):
        """
        统一 wandb 日志 (与 risk_sensitive 同键名的下尾口径)。

        约束流: Z=-C, q=-d, α=ω  →  P(Z≤q)=P(C≥d); Q_α(Z)=-Q_{1-α}(C)。
        任务流: disc_reward/{discounted,aver}_reward / return_std = 任务回报 R。

        Args:
            it:        当前迭代
            R_np:      本迭代 B 条轨迹的奖励回报 R (numpy [B])
            Zc_np:     本迭代 B 条轨迹的折扣 cost 回报 C (numpy [B]) —— 内部仍是 C
            undiscC_np:未折扣累计 cost Σc (numpy [B]) —— debug 用
            extra:     算法专属补充指标 dict
        """
        d, alpha = self.cost_limit, self.q_alpha
        q = -d                                                # 下尾阈值
        Z = -np.asarray(Zc_np, dtype=np.float64)              # Z = -C
        empirical_prob = float(np.mean(Z <= q))               # = P(C≥d)
        mean_R = float(np.mean(R_np))
        std_R = float(np.std(R_np))
        quantile_return = float(np.percentile(Z, alpha * 100))  # Q_α(Z)
        margin = alpha - empirical_prob
        self.last_empirical_prob = empirical_prob
        self.last_outage_prob = empirical_prob                # 兼容旧 summary 字段名

        env_steps = (it + 1) * self.num_envs * self.n
        log_dict = {
            'disc_reward/discounted_reward': mean_R,
            'disc_reward/aver_reward': mean_R,
            'disc_reward/quantile_reward': quantile_return,   # Q_α(Z)=-Q_{1-α}(C)
            'disc_reward/return_std': std_R,
            'quantile/q_est': quantile_return,
            'quantile/margin_to_threshold': quantile_return - q,
            'constraint/empirical_prob': empirical_prob,
            'constraint/margin': margin,
            'progress/iteration': it,
            'progress/trajectories': (it + 1) * self.num_envs,
            'progress/env_steps': env_steps,
            # 原始 cost 诊断 (不进主面板命名空间)
            'debug/cost_mean': float(np.mean(Zc_np)),
            'debug/undisc_cost_mean': float(np.mean(undiscC_np)),
            'debug/cost_limit': d,
        }
        log_dict.update(self._env_action_stats())
        if extra:
            log_dict.update(extra)
        if self._wandb_run is not None:
            self._wandb_run.log(log_dict, step=it)

        if it % self.log_interval == 0 and it != 0:
            lam = log_dict.get('lambda/value', None)
            lam_s = f' lambda:{lam:.04f}' if isinstance(lam, (int, float)) else ''
            ghat = log_dict.get('constraint/cdf_estimate_initial', None)
            ghat_s = f' Ghat:{ghat:.03f}' if isinstance(ghat, (int, float)) else ''
            print(f'Iter:{it:05d} (env_step:{env_steps}) || disc_a_r:{mean_R:.03f} '
                  f'disc_q_r:{quantile_return:.03f}{lam_s}')
            print(f'Iter:{it:05d} || P(Z<=q):{empirical_prob:.03f} '
                  f'alpha:{alpha:.03f} margin:{margin:.03f}{ghat_s}\n')
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
        """评估采样动作。"""
        return self.choose_action(state)
