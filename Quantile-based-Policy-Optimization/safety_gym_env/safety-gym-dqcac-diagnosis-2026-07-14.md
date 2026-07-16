# Safety-Gym 环境下 DQCACBeta / QCPO / QCPO_refs 训练差距诊断

**初稿日期：2026-07-14；实验验证更新：2026-07-16（UTC）**

## 1. 诊断范围与数据来源

本报告针对 `Quantile-based-Policy-Optimization/safety_gym_env` 中以下三类算法在
SimpleButton、Dynamic、Gremlin、DynamicButton 环境上的训练表现进行工程与算法联合诊断：

- `DQCAC`：`agents/dqc_ac_beta_gpu.py`
- 用户 `QCPO`：`agents/qcpo_gpu.py`
- 论文参考算法 `QCPO_REF`：`agents/qcpo_ref.py`

分析依据包括：

1. 当前仓库的算法实现、环境包装、训练入口和评估代码；
2. QCPO 原始仓库：`/vepfs-mlp2/c20250510/251204033/QCPO_nips_ref/qcpo`；
3. QCPO 论文：`/vepfs-mlp2/c20250510/251204033/QCPO_nips_ref/2211.15034v1.pdf`；
4. 文本日志：`Quantile-based-Policy-Optimization/safety_gym_env/_runs/logs`；
5. WandB 导出数据：`Quantile-based-Policy-Optimization/safety_gym_env/_runs/wandb_export/latest/combined_history.csv`；
6. 项目已有 memory、DESIGN、PROGRESS 和 DQC-AC-β 论文稿 `paper/main.tex`。

需要注意：WandB 数据导出时间为 2026-07-14 12:40 左右。导出时 12 个 full run 均仍处于
`running` 状态，只完成约 1.95M～2.62M env steps，而计划预算为 5M。因此当前数据足以诊断
训练机制，但还不能作为最终算法排名或统计显著性结论。

---

## 1.1 2026-07-16 实验验证更新

在固定 `DynamicButton` 上完成 E1–E10 机制消融后，初稿的主要诊断得到验证，但“cost critic 一定是首要瓶颈”需要修正：reward 主干修好后，当前主要瓶颈已转为 β 与 dual 的闭环动态。

| 配置 | 预算/seed | eval reward | eval outage | λ final |
|---|---|---:|---:|---:|
| E5：GAE+PPO+obs norm，reward-only | 300k / s0 | 1.690 | 0.514 | 0 |
| E6：E5 + outage-I + β=.95 | 300k / s0 | 1.640 | 0.600 | 0.374 |
| E7：只改 β=.99 | 300k / s0 | 1.025 | 0.400 | 0.375 |
| E8：只改 β=1 | 300k / s0 | 0.136 | 0.000 | 0.113 |
| E9：β=.995 + outage-I | 300k / s0,s1,s2 | 0.889±0.229 | 0.262±0.054 | 0.317 mean |
| E10：β=.995 + QCPO quantile-I | 300k / s0 | -0.175 | 0.000 | 0.880 |
| QCPO_refs 最终 | 5M / s0 | 1.670 | 0.154 | 6.950 |

`±` 是 E9 跨 seed 样本标准差；每 seed 终评 70 条轨迹。QCPO_refs 的 5M 终点不能当作 300k 公平排名：它在 300k 的训练后段 reward/outage 为 `1.355/0.517`，当时同样尚未可行。

### 已被实验直接证实的根因

1. **最大工程缺陷是 raw observation 未归一化。** E4 到 E5 只加共享逐维 RMS，300k eval reward 从 `0.258` 跃升到 `1.690`；旧 DQC“均值和 quantile 都不涨”并非算法必然。
2. **同 rollout 做 10 次无修正 actor update 是关键策略新鲜度问题。** E2 的单次 GAE 仍几乎不学；固定 behavior log-prob、PPO ratio/clip 和 8 actor epochs 后 E3/E4 才形成稳定正斜率。
3. **GAE/PPO 是必要但不充分的稳定骨架。** 它解决低方差 reward credit assignment 与 batch reuse，但没有 obs norm 时仍远弱于参考实现。
4. **`beta=0.95` 对 T=1000 过度短视。** E6→E7→E8 形成单调的风险控制增强与 reward 下降，证明 cost actor 不是完全无效；工作区间在 `.99~1`，E9 的 `.995` 只是当前短预算候选而非最终默认。
5. **PID 参数必须按整个闭环重新定标。** QCPO_refs 的 quantile-I 原式在 E10 把 λ 推到约 1.18 后形成 windup，outage 降至 0 但 reward 崩到负值；可借用结构，不能机械复制增益。

### importance ratio 的准确状态语义

采样 rollout 时保存一次 `old_log_probs=log π_behavior(a|s)`；在该 rollout 的所有 PPO epochs 中它都是固定分母。每次 actor optimizer step 后，只重新前向当前策略得到 `log π_current(a|s)`，并计算 `ratio=exp(logπ_current-old_logπ_behavior)`。
不能在每次更新后把 `old_log_probs` 覆盖成当前概率；那会令 ratio 被重置为 1，使第 2～8 次更新的策略漂移不可见。只有丢弃旧 rollout、重新与环境交互采样新 rollout 时，才生成下一份 behavior log-prob。当前实现已按这个语义修正，并禁止非 PPO 模式在同一 rollout 上多次 actor 更新。

### 当前最合理的开发决策

- 当前诊断基线使用 `GAE + PPO clip=.1 + actor epochs=8 + critic epochs=10 + obs RMS + sum norm + beta=.995 + 经验 outage-I`；它不是最终超参，只是已通过 reward 门且不过度保守的起点。
- E9 三 seed outage 平均仍为 `0.262`，不要直接扩至 5M。下一步优先测试 leaky-I/anti-windup、较小 quantile `Ki` 或带死区的 PI，再做 3 seed 300k 门控。
- E9 cost critic CDF 平均 `0.224` 对 truth `0.262`，平均偏差 `-0.0375`。因此局部 quantile 加密或 IQN 值得保留，但当前优先级低于 dual 动态；只有固定策略下查询点偏差持续超过约 `0.05`，才进入该结构消融。
- 详细逐实验命令、耗时、W&B run id、停止依据和 profile 路径见 `DQCAC_DEBUG_LOG_2026-07-16.md`。

## 2. 核心结论

当前 DQCACBeta 表现差不是单纯因为训练时间不足，也没有发现 cost budget 递推、episodic
terminal mask 或上下尾方向存在明显的一行式符号错误。主要问题是以下三类因素叠加：

1. **reward actor-critic 本身没有学到有效策略。** 即使 λ=0、约束完全不参与，reward 仍不增长；
2. **cost critic 的微小函数逼近误差被归一化和 λ 放大成高方差策略梯度。** λ 越大，actor 更新越强，
   但更新方向主要是噪声而非可靠的避障方向；
3. **从短时域简单环境继承的 DQCAC 配方不适用于 1000 步 Safety-Gym。** 尤其是 `beta=0.95`、
   critic-driven dual、`n_step=100`、B=10、固定探索 σ=0.5 等组合。

QCPO_refs 的优势目前主要来自完整的 PPO 训练骨架，包括 reward GAE、PPO ratio clip、观测归一化、
LSTM/history、共享表示、最近 100 条轨迹的 dual 平滑和 `(J_r+λJ_c)/(1+λ)`。Weibull 尾模型对
尾部估计有帮助，但不是当前 reward 几乎不学习的首要解释。

---

## 3. 训练数据中的直接证据

### 3.1 对齐至不超过 2M env steps 的表现

下表为每个 run 在不超过 2M env steps 时最近 30 个训练迭代的均值：

| 环境 | 算法 | Reward mean | Reward std | Outage | λ | Cost mean |
|---|---|---:|---:|---:|---:|---:|
| Dynamic | DQCAC | 0.004 | 0.167 | 0.437 | 19.68 | 64.60 |
| Dynamic | QCPO | -0.010 | 0.167 | 0.400 | 50.00 | 56.34 |
| Dynamic | QCPO_REF | 1.360 | 0.127 | 0.200 | 3.96 | 8.35 |
| DynamicButton | DQCAC | -0.083 | 0.164 | 0.140 | 0.00 | 6.14 |
| DynamicButton | QCPO | -0.021 | 0.147 | 0.033 | 0.00 | 3.04 |
| DynamicButton | QCPO_REF | 1.540 | 0.152 | 0.197 | 10.71 | 9.33 |
| Gremlin | DQCAC | -0.018 | 0.138 | 0.573 | 20.99 | 81.57 |
| Gremlin | QCPO | -0.067 | 0.156 | 0.573 | 50.00 | 72.50 |
| Gremlin | QCPO_REF | 0.343 | 0.124 | 0.237 | 13.52 | 11.25 |
| SimpleButton | DQCAC | -0.048 | 0.373 | 0.583 | 21.16 | 102.89 |
| SimpleButton | QCPO | -0.035 | 0.170 | 0.633 | 50.00 | 101.57 |
| SimpleButton | QCPO_REF | 1.316 | 0.236 | 0.137 | 3.38 | 5.03 |

### 3.2 DynamicButton 是 reward 路径失效的自然消融

DynamicButton 上，DQCAC 和用户 QCPO 的 λ 都基本为 0，说明约束没有压制 reward 优化。然而：

- DQCAC reward 仍由接近 0 下降至约 -0.08；
- 用户 QCPO reward 也停留在 0 附近；
- QCPO_REF 在相同 env-step 量级达到约 1.5。

因此，不能把用户算法 reward 不增长主要归因于 λ 过大、risk advantage 或 cost constraint。
首先需要修复的是 reward 学习骨架。

### 3.3 DQCAC 的 reward advantage 几乎为零

DQCAC 当前使用 reward distributional Q critic 的动作差值：

```text
A_m(s,a) = Q_m(s,a) - E_{a'~pi}[Q_m(s,a')]
```

WandB 日志显示：

- `advantage/mean_adv_std` 只有约 0.005～0.012；
- `norm/return_sigma_ema` 约 0.27～0.55；
- 归一化后的有效 reward advantage 尺度只有约 0.014～0.044；
- `critic/q_mean` 长期接近 0；
- DynamicButton 后期 λ=0 时，`actor/w_std` 只有约 0.006。

这说明 actor 几乎得不到“当前状态下哪个动作能增加 reward”的有效信息。当前策略的主要更新来源不是
reward 方向，而是后续被 λ 放大的 cost critic 信号。

### 3.4 cost advantage 的微小误差被放大

原始 `advantage/risk_adv_std` 只有约 0.004～0.008，但 `norm/constraint_sigma_ema` 也处于同一数量级，
所以归一化后的 risk advantage 标准差被固定在约 0.85～1.13。

对应训练现象：

- λ 较小时，`actor/w_std` 约 0.2～0.4；
- λ 增长到 10～22 后，`actor/w_std` 增长到约 1～3；
- 但真实 cost/outage 没有随之下降。

这说明当前归一化放大的主要是 critic 量化误差、baseline 动作采样误差和函数逼近噪声，而不是稳定的
action-dependent 风险梯度。actor 因而表现为大幅波动而非稳定改进。

### 3.5 critic-driven dual 在当前环境明显失准

full run 的后期平均校准情况大致为：

| 环境 | cost critic 初始 outage | 经验 outage | 系统偏差 |
|---|---:|---:|---:|
| Dynamic | 约 0.70 | 约 0.43 | +0.27 |
| Gremlin | 约 0.88 | 约 0.57 | +0.31 |
| SimpleButton | 约 0.85 | 约 0.59 | +0.26 |

因此 dual 持续认为约束比真实情况更严重，λ 单向上升。此前短环境中使用 critic-dual 的成功依赖
“critic 基本无偏”这一前提；该前提在当前 Safety-Gym full run 中不成立。

另外，当前 active DQC-AC-β 论文稿 `paper/main.tex` 的 dual 仍定义为完成轨迹上的经验违反率，
critic CDF 只作为校准诊断。因此：

- 若以论文算法为准，当前 safety 实现的 dual source 已经发生算法变更；
- 若以此前验证过的 critic-dual 版本为准，则当前环境尚未满足使用条件。

---

## 4. 为什么 QCPO_refs 更稳定

### 4.1 Reward GAE，而非完全依赖动作 Q 差值

QCPO_REF 使用标量 reward value 和 GAE。实际 rollout 中的距离进展、goal reward 和 TD residual 会直接
进入 actor advantage，不要求 action-conditioned distributional critic 先把不同动作的 Q 差异学准。

相比之下，DQCAC 的 actor reward 信号完全依赖当前 reward critic 对动作的细粒度区分。一旦 critic 在
当前策略附近近似 action-insensitive，reward policy gradient 就会退化。

### 4.2 PPO old-policy ratio clip

QCPO_REF 保存 old policy，并用 ratio clip=0.1 复用同一批数据 8 epochs。当前 DQCAC 也复用同一批数据
10 次，但没有 importance ratio、PPO clip 或显式 KL 限制。第一次 actor 更新以后，后续更新已经不再是
严格的当前 on-policy 梯度。

### 4.3 观测归一化

QCPO_REF 对每个 observation 维度维护 running mean/variance，并将归一化结果 clip 到 [-10,10]。
用户 DQCAC/QCPO 的 actor 和 critic 直接使用 raw Safety-Gym observation。

### 4.4 LSTM、历史输入和共享表示

QCPO_REF 的共享结构为：

```text
obs -> MLP(512,512) -> concat(prev_action, prev_reward) -> LSTM(512)
     -> policy / reward value / cost quantiles / Weibull alpha,beta
```

环境 observation 还拼接了 prev_cost。这样策略能够记住时间、历史 cost 和运动趋势；同时 reward value、
cost quantile 和 Weibull 的密集监督会共同训练 actor 使用的特征表示。

当前 DQC actor 是独立 MLP，只能依赖本来就很弱的 policy gradient 自己学习高维导航特征。

### 4.5 平滑 dual 和 sum normalization

QCPO_REF 使用最近 100 条 episode cost 的经验分位数驱动 PID dual，而不是只依赖当前 10 条轨迹或一个
有偏 critic。同时策略损失采用：

```text
(J_reward + lambda * J_cost) / (1 + lambda)
```

这不会让 λ 增大时总梯度幅度无限增长。当前 DQCAC 没有对应保护。

### 4.6 cost scale 和正值输出

QCPO_REF 内部将 cost 和 threshold 都除以 10，并用 `exp(linear)` 保证 cost quantile 为正。当前 DQCAC
直接拟合 raw 1000-step cost，且 cost critic 输出没有正值约束。

日志中 DQCAC 的 reward QR loss 约为 0.6～1.4，而 cost QR loss 常为 200～400。两者使用同一个 optimizer
并做联合 global grad clip，cost 梯度很可能主导裁剪，从而进一步压制 reward critic。

---

## 5. DQCAC 从短时域迁移到 1000 步环境的关键失配

### 5.1 `beta=0.95` 只覆盖 episode 最早部分

当前约束 actor 权重为 `beta^t`，且 `beta=0.95`：

- `beta^100` 约为 0.0059；
- 约 99% 的累计权重集中在前 90 步；
- 第 200 步以后风险梯度几乎为 0；
- 但 outage 是按完整 1000 步 cost 计算。

因此 dual 在惩罚整条轨迹，而 actor 主要只能调整开局行为。这是从 n=10 或 n=100 的验证环境直接迁移
`beta=0.90/0.95` 到 T=1000 后产生的结构性偏差。

建议在 Safety-Gym 上至少测试：

```text
beta = 0.99, 0.995, 0.999
```

有限时域下 `beta=1` 的和仍然有限，可以作为诊断消融，但它会偏离论文中使用 Abel discount 的主算法，
不宜直接作为默认版本。

### 5.2 独立轨迹数过少

当前每个迭代 B=10、T=1000，虽然包含 10000 个 transition，但只有 10 条独立 cost trajectory。
对于目标 outage=0.2 的重尾分布，每批通常只有约 2 条超限轨迹，无法稳定估计尾部概率。

此前 DQCAC 的成功配置使用过 B=256/512。per-transition 方法能够复用 transition，并不代表分布尾部只需
极少独立 episode。应增加独立轨迹数量，或为 critic/dual 使用最近多批的短窗口数据。

### 5.3 `n_step=100` 同时作用于 reward 和 cost

设置 `n_step=100` 是为了解决未折扣 cost 在 1000 步链上的传播速度，但当前代码将相同 n-step 同时用于
reward critic。它会把 reward critic 也变成高方差 100-step distributional target。

建议拆分为：

```text
reward: scalar V + GAE，或 reward_n_step=1/5
cost:   cost_n_step 单独扫描 10/25/50/100
```

---

## 6. 当前对比实验中的协议不一致

### 6.1 探索标准差不一致

用户 DQCAC/QCPO：

```text
init_std = 0.5
learn_std = False
```

QCPO_REF：

```text
ref_init_log_std = 0.0  -> sigma = 1.0
log_std 可学习
```

参考算法日志中的 entropy 约为 2.76～2.84，说明 σ 在训练过程中仍接近 1。Safety-Gym 导航对早期探索非常
敏感，因此当前实验并不满足入口注释中所说的“同探索 σ”。

应至少进行以下一种处理：

1. 所有算法统一初始 σ=1，并允许学习；或
2. 将 QCPO_REF 的初始 σ 改为 0.5，作为严格同探索消融；
3. 正式报告中明确各算法使用其调优后的探索配置，不再声称完全相同。

### 6.2 `>` 与 `>=` 不一致

QCPO 论文约束为 `P(C > d)`，QCPO_REF dual 也使用严格大于。但用户 QCPO、DQCAC 和统一评估使用
`C >= d`。由于累计 cost 是整数，`C=15` 处可能有显著概率质量，该差异不能像连续分布那样忽略。

建议统一为论文的：

```text
P(C > 15) <= omega
```

并新增记录 `P(C == 15)`，用于解释严格与非严格阈值之间的差异。

### 6.3 单 seed、训练未完成

当前结果只有 seed 0，且导出时训练尚未达到 5M steps。修复后需要至少 5 seeds；最终 constraint 评估应
使用 256～1000 条独立 episode，不能用每批 10 条训练轨迹判断是否收敛。

---

## 7. 建议的分阶段改进路线

### 阶段 A：先通过 reward-only 门

优先在 DynamicButton 或所有环境上显式运行 `lambda=0` 的 mean-only 版本。建议依次加入：

1. actor/reward critic observation RMS + clip [-10,10]；
2. 初始 `sigma=1.0`，并允许训练 log_std；
3. reward scalar value + GAE；
4. 保存 old log-prob，使用 PPO clip=0.1；
5. actor epochs 先设为 4，critic epochs 可保持 10；
6. 记录 PPO KL、clip fraction、ratio、actor pre/post-clip grad norm。

通过标准：在 2M steps 前 reward 应出现明确、持续的上升，DynamicButton 至少应进入接近 QCPO_REF 的
量级。若 reward-only 仍无法学习，不应继续调 cost、dual 或 Weibull。

### 阶段 B：冻结策略，单独校准 cost critic

固定一个策略，在相同数据上比较：

```text
cost_scale:      1 vs 10
cost_n_step:     1 / 10 / 25 / 50 / 100
cost output:     linear vs softplus
num_quantiles:   32 vs 64/128
target_tau:      0.01 / 0.05
```

同时进行以下工程改动：

- reward/cost critic 分开 optimizer；
- reward/cost critic 分开 grad clip；
- budget、threshold、cost target 使用完全一致的内部 cost scale；
- 记录 cost quantile min/max、负值比例、crossing、s0 CDF、均值和标准差；
- 用独立 128～512 条 episode 评估 CDF bias，而不是当前批内 B=10 的经验值。

只有当 s0 outage bias 稳定控制到约 0.03～0.05 内，才值得重新测试 critic-driven dual。

### 阶段 C：恢复约束，优先使用经验 dual

建议第一版使用：

1. 最近 100 条完成轨迹的经验 `P(C>d)` 或经验 `Q_{1-omega}(C)-d` 驱动 dual；
2. `beta in {0.99, 0.995, 0.999}`；
3. risk advantage 保留 budget-CDF 结构；
4. reward advantage 改为 GAE；
5. actor 使用 PPO old-policy surrogate；
6. 最终组合除以 `1+lambda`；
7. constraint normalization 添加合理的 std floor，或先关闭归一化做消融。

推荐的混合 advantage 为：

```text
A_total = (gamma^t * A_reward_GAE - lambda * beta^t * A_cost_CDF) / (1 + lambda)
```

其中 cost critic 和 budget 机制仍属于 DQCAC，reward/PPO backbone 则借用成熟的稳定化方案。

### 阶段 D：最后再考虑结构增强和 Weibull

若 reward 已能学习、dual 已稳定，但尾部估计仍不准，再依次测试：

1. 增加每次更新的独立 trajectory 数；
2. actor 输入归一化 `t/T`、累计 cost 或 budget；
3. LSTM/GRU history；
4. 共享 encoder + 独立 actor/reward/cost heads；
5. 单调 quantile 网络或 quantile 插值形成平滑 CDF；
6. Weibull tail head。

Weibull 是 QCPO 的核心尾分布建模组件。若加入 DQCAC，应明确标记为 `DQCAC+Weibull/LDP` hybrid 并进行
单独消融，不能把它和普通 PPO/GAE/观测归一化等通用工程组件混为一类。

---

## 8. 建议新增的诊断指标

当前日志仍缺少区分“信号”和“噪声”的关键量。建议加入：

- `actor/grad_norm_pre_clip`、`actor/grad_norm_post_clip`；
- reward/cost critic 独立的 grad norm；
- PPO approximate KL、clip fraction、ratio mean/std；
- raw action 超出 [-1,1] 的比例和 env 实际 clipped action 比例；
- policy mean/std；
- `std(gamma^t*A_m)`、`std(lambda*beta^t*A_c)` 及两者比例；
- raw `A_m/A_c` 与 MC return/outage 的相关性；
- cost critic quantile min/max、负值比例、crossing rate；
- s0 predicted CDF 与最近 100/500 条 episode empirical CDF；
- `P(C>15)`、`P(C>=15)`、`P(C==15)`；
- reward value explained variance；
- 同一状态多动作下 `std_a Q_r(s,a)` 和 `std_a Psi_c(s,a,b)`。

这些指标可以直接判断 critic 是否真正学到了 action discrimination，而不是只拟合了状态平均值。

---

## 9. 最终判断

1. 未发现 DQCAC cost budget 递推、上尾事件定义或 episodic bootstrap 的明显基本符号错误；
2. 当前 reward backbone 明确失效，DynamicButton 的 λ=0 结果已经构成直接证据；
3. 当前 cost critic 在主要违反约束的环境上系统性高估 outage，不适合直接驱动 dual；
4. tiny risk advantage 被极小 EMA 标准差放大，是 λ 上升后策略大幅波动的重要原因；
5. `beta=0.95` 对 1000 步环境过于短视，是短环境配方迁移时的关键失配；
6. QCPO_refs 最值得优先借鉴的是 PPO/GAE、观测归一化、dual 平滑、共享表示和 sum normalization；
7. Weibull 尾模型应放在 reward/critic/dual 基础链路修好以后再做；
8. 当前实验还存在探索 σ、严格阈值和训练完成度不一致，修复前不宜下最终算法优劣结论。

