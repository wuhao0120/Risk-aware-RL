# Universal Development Guidelines

本文档包含项目无关的通用开发经验、用户偏好和代码规范，可在多个项目间复用。

---

## 用户偏好 (User Preferences)

### 文档管理原则
- **不要创建总结性的md文件**: 除非用户明确要求，否则不要创建README、GUIDE、SUMMARY等文档文件。专注于代码实现本身。

### 代码修改原则
- **最小修改原则**: 当遇到API不兼容问题时，优先通过创建适配器类（Adapter）来处理接口差异，而不是大规模重写代码。
  - 示例：处理API版本升级时，创建wrapper类统一接口，而非修改所有调用点
  - 优势：影响范围可控、易于回退、保持代码稳定性

### 日志管理原则
- **统一日志系统**: 项目应采用统一的日志系统，避免多种日志框架并存导致的维护负担。
  - 迁移策略：明确弃用旧系统，全量迁移到新系统
  - 一致性：所有模块使用相同的日志API

### 注释规范要求
- **详细注释**: 保持高注释密度（50%+）和多层级解释
  - 参考标准：注释行数占比 >= 50%
  - 强制要求：所有类、函数、复杂逻辑必须有注释

### 文档修改流程
- **修改前确认**: 当被要求"总结到文档"或"更新文档"时，先描述拟修改的内容，得到用户确认后再执行修改。
  - 避免：未经确认的文档改动
  - 流程：描述改动 → 等待确认 → 执行修改

---

## 代码解读风格指南 (Code Explanation Style)

### 标准结构（六步法）

讲解算法迁移、API适配或架构改动时，遵循以下结构：

```
1. 一句话结论
   └─ 直接说明核心变化

2. 大概概括（3-5点）
   └─ 列举关键改动维度

3. 详细对比（含代码示例）
   └─ 使用表格或代码对比展示差异

4. 对比朴素方案
   └─ 说明为什么不用简单方案

5. 集成决策
   └─ 列举备选方案，说明最终选择及理由

6. 验证可行性
   └─ 说明如何验证方案正确性
```

### 核心要点

#### 1. 讲到能直接用
- **必须覆盖**: 实例化、调用、参数说明
- **代码示例**: 给出完整的调用代码
- **避免**: 只讲概念不讲用法

#### 2. 场景化对比
- **对比维度**: 新方案 vs 朴素方案
- **说明弊端**: 为什么朴素方案不可行
- **量化对比**: 代码行数、维护成本、性能差异

#### 3. 全链路视角
```
__init__ → 实例化 → 运行时 → 返回值 → 最终结果
```
- 说明每个环节的输入输出
- 追踪数据流和状态变化

#### 4. 变量替换与状态管理
- **明确指出**: 哪一行发生了替换
- **状态追踪**: 状态保存在哪个对象/变量中
- **生命周期**: 变量/对象的创建和销毁时机

### 对比表格模板

**类定义对比**:
| 维度 | 旧实现 | 新实现 | 必要性 |
|-----|-------|-------|------|
| 类定义 | ... | ... | 说明原因 |
| 参数 | ... | ... | 说明原因 |
| 初始化 | ... | ... | 说明原因 |

**技术选型对比**:
| 项目 | 方案A | 方案B | 选择 |
|-----|------|------|-----|
| 输入 | ... | ... | ✅/❌ |
| 输出 | ... | ... | ✅/❌ |
| 复杂度 | ... | ... | ✅/❌ |

### 集成决策模板

```
❌ 方案1: [简述] → [弊端]
❌ 方案2: [简述] → [弊端]
✅ 最终方案: [简述]
  - 代码行数: X行
  - 优势: ...
  - 获得: ...
```

---

## 代码注释规范 (Code Comment Standard)

### 注释密度要求

- **目标注释率**: 50%+（注释行数 / 总行数）
- **强制注释**：所有类、函数、复杂逻辑、关键变量
- **禁止裸代码**：不允许连续5行以上无注释的代码块

### 注释准确性要求

注释必须准确描述代码实际行为，结合库文档和项目上下文理解。

**常见错误示例**：
```python
# ❌ 太模糊
scheduler.step()  # 学习率衰减

# ✅ 准确描述
scheduler.step()  # 调用LambdaLR调度器更新优化器学习率, 内部: k+1, lr=a/(b+k)^c
```

**准确性检查清单**：
1. ✅ 函数调用: 说明"调用XX方法"，而非"做XX"
2. ✅ 库函数: 结合官方文档，说明输入输出、副作用
3. ✅ 自定义函数: 参考项目实现，说明内部逻辑
4. ✅ 参数更新: 明确更新公式 (如 `θ ← θ - lr * ∇θ`)
5. ✅ 模式切换: 说明对哪些层有影响 (Dropout, BatchNorm等)

### 注释粒度分级

#### 简单操作（一行简短注释）
```python
self.steps += 1                                # 全局步数+1
loss.backward()                                # 反向传播计算梯度
optimizer.zero_grad()                          # 清空累积梯度
```

#### 复杂操作（多行详细注释）

**模板**:
```python
[复杂代码行]
# [结果] 最终得到的形状/类型/含义
# 示例: tensor([value1, value2, ...])
#
# 变量说明:
# - var1: [type][shape], 含义...
# - var2: [type][shape], 含义...
#
# [操作名称]用法:
# [逐步解释操作过程]
# 相当于: [用更直观的方式描述]
#
# 为什么: [算法/设计需求]
# [与算法公式的对应关系]
```

**实际示例**:
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
# 为什么: 算法需要计算Σ log π(a_t|s_t) * I{...}
# log_probs包含所有动作, gather提取实际动作的概率
```

**复杂操作的判断标准**:
- ✅ 链式调用 (`.unsqueeze().gather().squeeze()`)
- ✅ 不常见的库函数 (`.gather()`, `.scatter()`, `.masked_fill_()`)
- ✅ 算法关键步骤 (梯度计算, 分布采样, 损失构造)
- ✅ 形状变换复杂 (多次reshape, transpose, broadcasting)
- ✅ 数值计算技巧 (clamp防溢出, detach阻断梯度)

### 五层注释体系

#### L1: 模块级注释（导入说明）
```python
import torch                                    # 深度学习框架
from torch.optim import Adam                    # Adam优化器
from torch.optim.lr_scheduler import LambdaLR   # 学习率调度器
from utils import Memory, Actor                 # Memory类用于储存轨迹
```

#### L2: 类/函数级注释
```python
def lr_lambda(k, a, b, c):
    """
    学习率因子函数

    Args:
        k: 当前迭代步数（标量）
        a, b, c: 衰减超参数（标量）

    Returns:
        float: 学习率衰减因子

    Formula:
        lr(k) = a / (b + k)^c
    """
    return a / (b + k) ** c
```

**类注释模板**:
```python
class ClassName:
    """
    [类简介]: 一句话说明职责

    [核心功能]:
    - 功能1
    - 功能2

    [关键属性]:
    - attr1: 说明
    - attr2: 说明

    [使用示例]:
    >>> obj = ClassName(args)
    >>> result = obj.method()
    """
```

#### L3: 代码块级注释
```python
# ============================================================
# 创建策略网络（Actor）
# ============================================================
state_dim = np.prod(env.observation_space.shape)
action_dim = env.action_space.n
self.actor = Actor(state_dim, action_dim, args.init_std)
```

**分隔规则**:
- 用 `# ===...===` 标记重要算法块
- 用空行分隔逻辑不同的代码块
- 每个块前有简短说明

#### L4: 行级注释（右侧对齐）
```python
self.device = args.device              # 设置设备CPU/GPU
self.gamma = args.gamma                # 折扣因子γ
self.q_alpha = args.q_alpha            # 分位数水平α
self.est_interval = args.est_interval  # 分位数估计更新间隔
```

**对齐规则**:
- 连续的相关变量，注释对齐到相同列
- 推荐对齐列：40-50字符
- 注释简短（5-10字），说明变量用途

#### L5: 变量级注释（类型、形状、示例）
```python
state = env.reset()                             # state是numpy数组, 形状(state_dim,)
                                                # 示例: array([0.1, -0.5, ...])

action = self.choose_action(state)              # action是numpy数组, 形状(action_dim,)
                                                # 示例: array([0.607], dtype=float32)

trajectory = memory.get()                       # dict, keys: ['states', 'actions', 'rewards']
                                                # states形状: (T, state_dim)
```

### 排版规则

1. **对齐**: 右侧注释空格对齐到相同列（建议第40-50列）
2. **分隔**: 逻辑块用空行和注释块分隔
3. **标记**: 用 `# ====` 标记关键算法块（初始化、训练循环、更新）
4. **缩进**: 注释缩进与代码一致
5. **空行**: 函数之间2个空行，代码块之间1个空行

### 注释自检清单

使用此清单检查代码注释质量：

- [ ] 每个类都有职责说明（L2）
- [ ] 每个函数都有参数、返回值、功能说明（L2）
- [ ] 每个导入都有用途说明（L1）
- [ ] 复杂变量有类型、形状、示例（L5）
- [ ] 复杂操作有变量说明、步骤、为什么（详细注释模板）
- [ ] 库函数调用准确说明方法和作用（准确性检查清单）
- [ ] 算法关键步骤有公式对应（如 ∇θ, π(a|s)）
- [ ] 注释密度 >= 50%
- [ ] 无连续5行以上无注释代码块
- [ ] 右侧注释对齐整齐

---

## 通用迁移方法论 (General Migration Methodology)

### 算法/框架迁移标准流程

#### 阶段1: 调研分析
1. **对比API差异**: 列举新旧接口的不同点
2. **评估影响范围**: 统计需要修改的文件/函数数量
3. **识别风险点**: 找出不兼容、需要重构的部分

#### 阶段2: 方案设计
1. **选择迁移策略**:
   - 适配器模式（推荐）: 创建wrapper类统一接口
   - 全量重写: 仅在架构完全不兼容时使用
   - 渐进迁移: 支持新旧并存，分批迁移

2. **撰写对比表格**:
   | 维度 | 旧实现 | 新实现 | 改动类型 | 风险 |
   |-----|-------|-------|---------|-----|
   | API | ... | ... | 兼容/不兼容 | 高/低 |
   | 架构 | ... | ... | 需重构/可复用 | 高/低 |
   | 依赖 | ... | ... | 版本冲突/兼容 | 高/低 |

3. **设计集成决策**:
   ```
   ❌ 方案1: [描述] → [弊端] → [风险评估]
   ❌ 方案2: [描述] → [弊端] → [风险评估]
   ✅ 最终方案: [描述]
     - 改动范围: X个文件
     - 代码增量: Y行
     - 优势: ...
     - 风险缓解: ...
   ```

#### 阶段3: 实施迁移
1. **创建适配层**: 优先实现适配器类
2. **单元测试**: 验证适配层功能正确性
3. **替换调用**: 批量替换旧API为新API（或使用适配器）
4. **回归测试**: 确保功能无损

#### 阶段4: 清理验证
1. **删除旧代码**: 移除废弃的接口和依赖
2. **更新文档**: 同步修改README、注释
3. **性能验证**: 对比迁移前后的性能指标

### API适配器模式

**通用适配器模板**:
```python
class NewAPIAdapter:
    """
    适配器类: 将新API转换为旧API接口

    用途: 避免大量修改现有代码，通过适配层统一处理API差异
    """
    def __init__(self, new_api_object):
        self.wrapped = new_api_object

    def old_method(self, *args, **kwargs):
        """
        将旧方法调用转换为新API调用

        Example:
            旧API: result = obj.old_method(x, y)
            新API: result = obj.new_method(x, y, extra_param=default)
        """
        # 转换参数格式
        new_args = self._convert_args(args, kwargs)

        # 调用新API
        result = self.wrapped.new_method(**new_args)

        # 转换返回值格式
        return self._convert_result(result)

    def _convert_args(self, args, kwargs):
        """参数格式转换"""
        # 实现转换逻辑
        pass

    def _convert_result(self, result):
        """返回值格式转换"""
        # 实现转换逻辑
        pass
```

**适配器使用场景**:
- ✅ API返回值格式变化（如元组长度改变）
- ✅ 参数名称/顺序调整
- ✅ 行为语义微调（如默认值变化）
- ❌ 架构完全重构（此时应考虑全量重写）

### 迁移检查清单

- [ ] 已列举新旧API差异
- [ ] 已评估影响范围（X个文件，Y处调用）
- [ ] 已选择迁移策略（适配器/重写/渐进）
- [ ] 已撰写对比表格
- [ ] 已设计集成决策（含方案对比）
- [ ] 已实现适配层（如需要）
- [ ] 已完成单元测试
- [ ] 已完成回归测试
- [ ] 已更新相关文档
- [ ] 已删除废弃代码

---

## 快速参考模板 (Quick Reference Templates)

### 并行训练模板
```python
from multiprocessing import Pool

def run_experiment(algo_name, seed, device):
    """单次实验运行函数"""
    # 实现实验逻辑
    pass

# 并行运行多个实验
if __name__ == '__main__':
    n_seeds = 5
    algos = ['AlgoName'] * n_seeds
    seeds = list(range(n_seeds))
    devices = ['cuda:0', 'cuda:1', ...] * (n_seeds // n_gpus)

    with Pool(processes=n_seeds) as pool:
        pool.starmap(run_experiment, zip(algos, seeds, devices))
```

### 日志系统集成模板

**WandB示例**:
```python
import wandb

# 初始化
wandb.init(
    project='project_name',
    name=f'{algo_name}_{timestamp}_{seed}',
    config=vars(args)  # 保存超参数
)

# 训练循环中记录
wandb.log({
    'train/loss': loss,
    'train/reward': episode_reward,
    'eval/mean_reward': mean_eval_reward
}, step=global_step)

# 结束时关闭
wandb.finish()
```

### 命令行参数解析模板
```python
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='实验说明')

    # 通用参数
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda', help='设备选择')
    parser.add_argument('--log_interval', type=int, default=10, help='日志间隔')

    # 算法特定参数
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # 设置随机种子
    set_seed(args.seed)
```

### 实验日志目录结构
```
logs/
├── {env_name}/
│   ├── {algo_name}_{timestamp}_{seed}/
│   │   ├── config.json          # 超参数配置
│   │   ├── checkpoints/         # 模型检查点
│   │   │   ├── model_1000.pth
│   │   │   └── model_best.pth
│   │   ├── metrics.csv          # 训练指标
│   │   └── logs.txt             # 控制台输出
```

---

## 版本控制

**文档版本**: 1.0.0
**创建日期**: 2026-01-23
**适用范围**: 跨项目通用开发规范
**维护建议**: 每季度review一次，根据实践经验更新
