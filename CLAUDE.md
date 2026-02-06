# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains research implementations for **risk-aware reinforcement learning** using quantile-based methods. It consists of two main projects:

1. **Quantile-based-Policy-Optimization (QPO)**: Implementation of "Quantile-Based Deep Reinforcement Learning using Two-Timescale Policy Gradient Algorithms"
2. **Deep-Distributional-Learning-with-Non-crossing-Quantile-Network (NQ-Net)**: Implementation of "Deep Distributional Learning with Non-crossing Quantile Network" and NC-QRDQN

## Running Experiments

### QPO Project

For toy_example, fair_lottery, beyond_greed:
```bash
cd Quantile-based-Policy-Optimization/<experiment_name>
python train.py
```

For inventory_management and portfolio_management:
```bash
python train_ppo.py   # PPO baseline
python train_qppo.py  # QPPO algorithm
python train_qrdqn.py # QR-DQN (inventory only)
```

### NQ-Net Project

```bash
cd Deep-Distributional-Learning-with-Non-crossing-Quantile-Network
python train.py --cuda --model <MODEL> --env_id <ENV_ID> --seed <SEED>
```

Available models: `QRDQN`, `ncQRDQN`, `DEnet`, `QPO`

For simulation figures:
```bash
cd simulation
python presentation.py --model wave  # or triangle, linear
```

## Architecture

### QPO Experiment Module Structure

Each experiment module follows this pattern:
```
<experiment_name>/
├── train.py or train_*.py    # Training entry points with Options class
├── agents/                   # Algorithm implementations
│   ├── qpo.py               # Quantile Policy Optimization
│   ├── qppo.py              # Quantile PPO
│   ├── ppo.py               # PPO baseline
│   └── ...
├── envs/                     # Environment definitions
│   └── <env_name>.py        # Gym-compatible environment
└── utils/
    ├── memory.py            # Replay buffer/trajectory storage
    └── model.py             # Neural network architectures (Actor, etc.)
```

**Key Concepts**:
- **QPO/QPPO Two-Timescale Learning**: Separate optimizers and LR schedules for policy (θ) and quantile estimate (Q)
- **Options Class Pattern**: `args = Options(algo_name).parse(seed, device)` dynamically adds algorithm-specific parameters

### NQ-Net Architecture

```
agents/
├── agent/           # Agent implementations (base_agent, qrdqn_agent, DEnet_agent, qpo_agent)
├── model/           # Network architectures (qrdqn, nc_qrdqn, DEnet)
├── env.py           # Atari environment wrappers (make_pytorch_env)
├── memory.py        # Experience replay
└── network.py       # Shared network components
```

Configuration loaded from YAML files in `config/` directory.

## Environment Requirements

- Python 3.9+ (QPO) or 3.10+ (NQ-Net gymnasium branch)
- PyTorch 1.9.0+cu111 (QPO) or torch 2.9.1 (NQ-Net)
- Gymnasium + ale_py (NQ-Net现已迁移)
- CUDA recommended for Atari experiments

## Key Parameters

Common across experiments:
- `q_alpha`: Quantile level (e.g., 0.25 for 25th percentile)
- `gamma`: Discount factor
- `est_interval`: Episodes between quantile estimation updates
- `log_interval`: Logging frequency
- `seed`: Random seed for reproducibility

Logs saved to: `logs/{env_name}/{algo_name}_{timestamp}_{seed}/`

## User Preferences

- **不要创建总结性的md文件**: 除非用户明确要求，否则不要创建README、GUIDE、SUMMARY等文档文件。专注于代码实现本身。

- **最小修改原则**: 当遇到API不兼容问题时，优先通过创建适配器类（Adapter）来处理接口差异，而不是大规模重写代码。例如gymnasium与旧gym的API差异，通过`Gym4Adapter`类统一处理。

- **统一日志系统**: NQ-Net项目统一使用wandb进行日志管理，不再兼容tensorboard。所有Agent都通过`wandb.log()`记录指标。

- **详细注释规范**: 参考 `Quantile-based-Policy-Optimization/toy_example/agents/qpo.py` 的注释风格，保持高注释密度（50%+）和多层级解释。

- **修改CLAUDE.md前需确认**: 当被要求"总结到claude.md"或"更新claude.md"时，先描述拟修改的内容，得到用户确认后再执行修改。

## Code Explanation Style (代码解读风格)

参考 `code_explain.md`。当讲解算法迁移、API适配或架构改动时：

**标准结构**:
1. 一句话结论 → 2. 大概概括(3-5点) → 3. 详细对比(含代码示例) → 4. 对比朴素方案 → 5. 集成决策 → 6. 验证可行性

**核心要点**:
- 讲到能直接用：必须覆盖实例化和调用
- 场景化对比：对比朴素解法的弊端
- 全链路视角：`__init__` → 实例化 → 运行时 → 返回值 → 最终结果
- 变量替换与状态管理：明确指出哪一行发生了替换，状态保存在哪

详见本文档后续章节的QPO迁移案例作为参考模板。

## 代码注释规范 (Code Comment Standard)

参考 `Quantile-based-Policy-Optimization/toy_example/agents/qpo.py` (注释密度54%)。

### 注释密度要求

- **目标注释率**: 50%+（代码行数 vs 注释行数）
- **强制注释**：所有类、函数、复杂逻辑、关键变量
- **禁止裸代码**：不允许连续5行以上无注释的代码块

### 注释准确性要求

注释必须准确描述代码实际行为，结合PyTorch库文档和项目上下文理解。

**常见错误**：
- ❌ `scheduler.step()  # 学习率衰减` (太模糊)
- ✅ `scheduler.step()  # 调用LambdaLR调度器更新优化器学习率, 内部: k+1, lr=a/(b+k)^c`

**准确性检查清单**：
1. 函数调用: 说明"调用XX方法"，而非"做XX"
2. PyTorch操作: 结合库文档，说明输入输出、副作用
3. 自定义函数: 参考项目实现，说明内部逻辑
4. 优化器操作: 明确参数更新公式 (如 `θ ← θ - lr * ∇θ`)
5. 模式切换: 说明对哪些层有影响 (Dropout, BatchNorm等)

### 注释粒度分级

**简单操作** (一行简短注释):
```python
self.steps += 1                                # 全局步数+1
loss.backward()                                # 反向传播计算梯度
```

**复杂操作** (多行详细注释，含变量说明、操作步骤、为什么):
```python
action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
# 选取实际采取的动作的log概率, 最终得到形状[T]的张量
# 每个元素: tensor([log π(a_1|s_1), ..., log π(a_T|s_T)])
#
# 变量说明:
# - actions: [T]张量, 每个timestep实际选择的动作索引
# - log_probs: [T, num_actions]矩阵, 每行是所有动作的log概率
#
# gather用法:
# actions.unsqueeze(1)变成[T,1], gather(dim=1)在每行提取索引对应元素
# 相当于: 从log_probs[t]中取出第actions[t]个元素
#
# 为什么: QPO需要计算Σ log π(a_t|s_t) * I{...}
# log_probs包含所有动作, gather提取实际动作的概率
```

**复杂操作需要详细注释的标志**:
- 链式调用 (`.unsqueeze().gather().squeeze()`)
- 不常见的PyTorch函数 (`.gather()`, `.scatter()`, `.masked_fill_()`)
- 算法关键步骤 (indicator计算, 梯度手动设置)
- 形状变换复杂 (多次reshape, transpose)

### 五层注释体系

#### L1: 模块级注释（导入说明）
```python
from torch.optim.lr_scheduler import LambdaLR  # 学习率调度器
from utils import Memory, Actor  # Memory类用于储存trajectory
```

#### L2: 类/函数级注释
```python
def lr_lambda(k, a, b, c):
    """
    [函数简介]: 学习率因子函数
    [输入输出]: k,a,b,c都是标量
    [算法逻辑]: lr(k) = a/(b+k)^c
    [返回值]: 学习率衰减因子
    """
```

#### L3: 代码块级注释
```python
# 创建策略网络（Actor）
state_dim = np.prod(self.env.observation_space.shape)
self.actor = Actor(state_dim, action_dim, args.init_std)
```

#### L4: 行级注释（右侧对齐）
```python
self.device = args.device              # 设置设备CPU/GPU
self.q_alpha = args.q_alpha            # 分位数水平
```

#### L5: 变量级注释（类型、形状、示例）
```python
action = self.choose_action(state)              # action是numpy数组,形状(1,)
                                                # 形如array([0.60672134], dtype=float32)
```

### 排版规则

1. **对齐**: 右侧注释空格对齐到相同列
2. **分隔**: 逻辑块用空行和注释块分隔
3. **标记**: 用 `# ====` 标记关键算法块

### 注释自检清单

- [ ] 每个类都有职责说明
- [ ] 每个函数都有输入输出说明
- [ ] 复杂变量有类型、形状、示例
- [ ] 复杂操作有变量说明、步骤、为什么
- [ ] PyTorch操作准确说明调用的方法和作用
- [ ] 注释密度 >= 50%

---

## 技术案例与迁移记录

### QPO 算法迁移：从 qpo.py 到 qpo_agent.py

将QPO从连续控制迁移到Atari离散动作空间的典范实现。

#### 大概概括

**三大变化**：
1. **架构**：独立类 (`QPO`) → 继承基类 (`QPOAgent(BaseAgent)`)
2. **网络**：Gaussian策略 (连续) → CNN+分类策略 (离散Atari)
3. **训练循环**：内部 `train()` 循环 → 覆盖 `train_episode()` 方法

**为什么改**：
- NQ-Net项目采用统一Agent框架，支持多种算法共存
- 通过继承BaseAgent复用环境管理、评估、检点保存等基础设施
- Atari环境需要CNN特征提取

#### 关键改动对比

**1. 类定义与继承**

| 维度 | qpo.py | qpo_agent.py | 必要性 |
|-----|--------|------------|------|
| 类定义 | `class QPO(object)` | `class QPOAgent(BaseAgent)` | 复用env、device、评估框架 |
| init参数 | `(args, env)` | `(env, test_env, log_dir, ...)` | 显式参数便于调用 |
| 初始化 | 手写所有成员 | `super().__init__(...)` | 代码复用 |

**2. 策略网络架构**

| 项目 | qpo.py | qpo_agent.py |
|-----|--------|-------------|
| 输入 | 低维向量 | 图像 [4,84,84] |
| 网络 | MLP | CNN + FC |
| 输出 | 高斯分布参数 | 动作logits |
| 分布 | MultivariateNormal | Categorical |

**3. 环境API处理**

- `warmup()`: 直接使用gymnasium的5值API
- `train_episode()`: 使用Gym4Adapter转换后的4值API

**4. 核心算法保持一致**

- ✅ 示性函数 I{U(τ) <= Q}
- ✅ Two-timescale learning (θ和Q独立学习率)
- ✅ 轨迹级别判断（indicator全0或全1）
- ✅ 手动设置分位数梯度

#### 集成决策

```
❌ 方案1: 直接在qpo.py上改 → 破坏原有逻辑
❌ 方案2: 复制BaseAgent逻辑 → 代码冗余
✅ 最终方案: 继承BaseAgent + 特化QPO逻辑
  - 代码行数: 625行 (vs qpo.py的316行, 带详细注释)
  - 获得: 环境管理、评估、检点、wandb集成
```

### Gymnasium 迁移

NQ-Net已从旧gym迁移到gymnasium + ale_py。

**核心方案**: 创建 `Gym4Adapter` wrapper，将gymnasium的5值API转换为旧4值API，其他代码无需修改。

**关键改动** (agents/env.py):
1. 添加 `Gym4Adapter` 类
2. 所有wrapper更新为gymnasium的5值API
3. `make_pytorch_env()` 最外层包装Gym4Adapter
4. 使用 `np_random.integers()` 替代 `randint()`
5. 导入 `ale_py` 注册Atari环境

### Tensorboard 到 Wandb 迁移

**修改文件**: base_agent.py, qrdqn_agent.py, nc_qrdqn_agent.py, DEnet_agent.py, train.py, test_agent.py

**关键改动**:
- 移除 `from torch.utils.tensorboard import SummaryWriter`
- 移除 `tensorboard` 参数
- 替换 `self.writer.add_scalar(...)` → `wandb.log({...}, step=...)`
- 替换 `self.writer.close()` → `wandb.finish()`

### QPO Agent 算法修正

**问题**: 错误地将 U(τ) 理解为每个timestep的returns-to-go

**修正** (agents/agent/qpo_agent.py):
```python
# 正确: 计算轨迹cumulative return（标量）
disc_return = Σ γ^t r_t

# 正确: 创建常数向量, 所有timestep都等于轨迹总回报
disc_rewards = torch.full((T,), disc_return, ...)  # [G, G, G, ..., G]

# 正确: indicator对整个轨迹统一判断（全0或全1）
indicators = indicator(self.q_est.detach(), disc_rewards)
```

**核心原理**: U(τ) 对同一轨迹所有timestep都相同，indicator是轨迹级别的全有或全无判断。

### DQC-AC 算法实现

**DQC-AC** (Distributional Quantile-Constrained Actor-Critic): 将QCPO的Monte-Carlo方法改为TD Learning的per-transition更新。

#### 三个分位数点维护

根据公式(13)的密度估计需求，维护三个分位数点：
```python
# 初始化
self.q_est = torch.tensor([0.0], ...)        # Q(α): 目标分位数
self.q_est_lower = torch.tensor([0.0], ...)  # Q(α-δ): 左邻近分位数
self.q_est_upper = torch.tensor([0.0], ...)  # Q(α+δ): 右邻近分位数

# 更新时同时更新三个点，每个使用对应的分位数水平
grad_center = -(self.q_alpha - indicators_center.mean())        # α
grad_lower = -(self.alpha_lower - indicators_lower.mean())      # α-δ
grad_upper = -(self.alpha_upper - indicators_upper.mean())      # α+δ
```

#### 密度估计 (公式13)

```python
# f_Z(Q(α;θ);θ) = 2δ / (Q(α+δ;θ) - Q(α-δ;θ))
def estimate_density(self):
    quantile_diff = self.q_est_upper.detach() - self.q_est_lower.detach()
    quantile_diff_safe = torch.clamp(quantile_diff.abs(), min=1e-6)
    density = 2 * self.delta / quantile_diff_safe
    return torch.clamp(density, max=100.0)  # 数值稳定性保护
```

**关键点**: 密度基于维护的三个分位数点计算，而非TD targets的经验分布。

#### Actor梯度计算 (公式24)

```python
# D̃ = (1/N) Σⱼ [y_j - λ·𝟙{y_j ≤ Q(α)} / f̂_Z] · ∇θ log π
density_estimate = self.estimate_density()
indicators = indicator(self.q_est.detach(), td_targets.flatten())
constraint_term = self.lambda_dual * indicators / density_estimate.detach()  # 密度在分母
gradient_weights = (td_targets - constraint_term).mean(dim=1)
actor_loss = -(action_log_probs * gradient_weights.detach()).mean()
```

**注意**: 通过定义loss让PyTorch自动微分计算 ∇θ log π，与直接设置梯度等价。

#### Target网络软更新

```python
# 参数
self.nu = nu  # 软更新系数，默认0.005
self.target_update_interval = target_update_interval  # 默认10000

# 软更新 (Polyak averaging)
def update_target(self):
    for target_param, online_param in zip(
        self.target_critic.parameters(),
        self.online_critic.parameters()
    ):
        # target = (1-nu)*target + nu*online
        target_param.data.copy_(
            (1.0 - self.nu) * target_param.data + self.nu * online_param.data
        )

# 在learn()中定期调用
if self.learning_steps % self.target_update_interval == 0:
    self.update_target()
```

#### Crossing监控

监控三个分位数点是否违反单调性 Q(α-δ) ≤ Q(α) ≤ Q(α+δ)：
```python
quantile_diff = self.q_est_upper.item() - self.q_est_lower.item()
crossing_violation = (
    (self.q_est_lower.item() > self.q_est.item()) or
    (self.q_est.item() > self.q_est_upper.item()) or
    (quantile_diff < 0)
)
# 记录滑动窗口内的违反率
wandb.log({'state/crossing_violation_rate': crossing_violation_rate})
```

---

## 快速参考

### Parallel Training
```python
from multiprocessing import Pool
algos = ['QPO'] * n_seeds
seeds = list(range(n_seeds))
pool.starmap(run, zip(algos, seeds, devices))
```

### Gymnasium API
- `reset()`: 返回 `(obs, info)`
- `step()`: 返回 `(obs, reward, terminated, truncated, info)`
- Gym4Adapter转换为旧4值API

### Wandb Usage
```python
wandb.init(project=..., name=..., config=...)
wandb.log({'metric': value}, step=step)
wandb.finish()
```

### 代码注释示例

**简单操作**:
```python
self.steps += 1                                # 全局步数+1
```

**复杂操作** (gather等):
```python
result = tensor.gather(1, index.unsqueeze(1)).squeeze(1)
# [结果] 最终得到形状...
# [变量说明] tensor是..., index是...
# [操作步骤] 1. unsqueeze... 2. gather... 3. squeeze...
# [为什么] 因为算法需要...
```

完整规范见 `Quantile-based-Policy-Optimization/toy_example/agents/qpo.py`。
