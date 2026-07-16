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

## 9. 初始诊断阶段最终判断（后续以 1.1 节和第 10～12 节为准）

1. 未发现 DQCAC cost budget 递推、上尾事件定义或 episodic bootstrap 的明显基本符号错误；
2. 当前 reward backbone 明确失效，DynamicButton 的 λ=0 结果已经构成直接证据；
3. 当前 cost critic 在主要违反约束的环境上系统性高估 outage，不适合直接驱动 dual；
4. tiny risk advantage 被极小 EMA 标准差放大，是 λ 上升后策略大幅波动的重要原因；
5. `beta=0.95` 对 1000 步环境过于短视，是短环境配方迁移时的关键失配；
6. QCPO_refs 最值得优先借鉴的是 PPO/GAE、观测归一化、dual 平滑、共享表示和 sum normalization；
7. Weibull 尾模型应放在 reward/critic/dual 基础链路修好以后再做；
8. 当前实验还存在探索 σ、严格阈值和训练完成度不一致，修复前不宜下最终算法优劣结论。


---

## 10. 2026-07-16 仓库改动与最重要发现汇总

本节记录 E1～E10 后的代码状态，也是对“做了什么、最重要发现是什么、下一步做什么”的集中回答。

### 10.1 最重要的实验发现

1. **DQCACBeta 旧实现不涨不是 distributional critic 的必然结果。** 最大单项故障是 Safety-Gym 的高维异尺度 observation 直接输入网络。E4→E5 只增加共享 observation RMS，30 万步 eval reward 从 `0.258` 提升到 `1.690`。
2. **同一 rollout 做多次无修正 actor update 是关键错误。** 第一次更新后数据已经不再来自当前策略；旧 DQCAC 和当前 QCPO 都存在这个问题。非 PPO actor 现在每个 rollout 只允许更新一次；PPO 路径使用固定 behavior log-prob。
3. **GAE、PPO 和 observation normalization 是组合稳定器。** 单次 GAE 并没有解决问题，GAE+PPO 只带来有限改善；再加 obs RMS 后 reward 主干才真正正常。
4. **`beta=0.95` 与 T=1000 不匹配。** `beta=.99→.995→1` 的消融证明 risk gradient 有效，但风险控制越强 reward 越低；当前问题已从“算法不学习”转为“闭环工作点与 dual 动态没有调好”。
5. **当前查询点 critic 误差不是第一瓶颈。** E9 三 seed 的 cost CDF 平均为 `0.224`，真实 outage 为 `0.262`，平均偏差 `-0.0375`。这仍可改进，但暂时小于 PID 响应/积分 windup 的影响。
6. **QCPO_refs 的组件可以借用，但超参不能机械复制。** E10 原样采用其 cost-quantile I 更新后过度保守，eval reward `-0.175`、outage `0`，直接显示控制器增益需要按 DQC 的 risk advantage、β、优化器和网络重新辨识。

### 10.2 已完成的代码改动

| 改动 | 主要文件 | 当前状态 |
|---|---|---|
| critic epochs 与 actor epochs 拆分 | `agents/dqc_ac_beta_gpu.py`、`run_experiment.py` | 已完成；非 PPO 多 actor updates 直接报错 |
| 固定 behavior `old_log_probs`、IS ratio、PPO clip/KL | `agents/dqc_ac_beta_gpu.py` | 已完成并通过多 epoch smoke/E3～E10 |
| scalar reward value + GAE | `agents/common.py`、`agents/dqc_ac_beta_gpu.py` | 已完成；GAE target 每 rollout 冻结 |
| reward/risk PPO surrogate | `agents/dqc_ac_beta_gpu.py` | reward 用 pessimistic min，cost 用 conservative max |
| cost advantage 在 PPO epochs 间冻结 | `agents/dqc_ac_beta_gpu.py` | 已完成，避免 critic 更新导致 actor target 漂移 |
| 共享逐维 observation RMS + clip | `utils/model.py`、`agents/vec_base.py` | 已完成；actor、V、reward/cost critics 共用统计 |
| optimizer/scheduler 时序修复 | `agents/dqc_ac_beta_gpu.py` | 仅在 optimizer 真正 step 后推进 scheduler |
| 经验 outage-I / cost-quantile-I dual | `agents/dqc_ac_beta_gpu.py` | 已完成，可通过参数切换 |
| `(J_r+λJ_c)/(1+λ)` sum normalization | `agents/dqc_ac_beta_gpu.py` | 已完成并用于 E6～E10 |
| PPO、value、CDF、PID、梯度与归一化诊断 | DQC agent 与 profile 工具 | 已完成并进入 W&B 全 history |
| 持久化后台 launcher | `launch_background.sh` | 已完成；记录 PID、命令、commit、dirty state、退出码和耗时 |
| W&B API 全量导出与共同预算 profile | `export_wandb_metrics.py`、`profile_wandb_metrics.py` | 已完成；保留 early/middle/late、趋势和图 |

重要的兼容决策：旧 DQC 配方仍保留为默认，新的 GAE/PPO、obs norm、经验 PID 与 sum norm 通过显式参数打开，避免破坏旧实验复现。当前较好的**诊断配置**是 E9，而不是已经宣布的最终默认配置。

### 10.3 importance ratio 的状态保存语义

rollout 时只保存一次：

```text
old_log_prob[t] = log π_behavior(a_t | history_t)
```

同一 rollout 的每个 PPO epoch 都重新前向当前策略：

```text
ratio[t] = exp(log π_current(a_t | history_t) - old_log_prob[t])
```

optimizer step 后**不更新** `old_log_prob`。只有丢弃旧 rollout、重新与环境交互采样时，才产生下一份 behavior probability。若每次更新后覆盖分母，ratio 会被重置到 1，无法检测或裁剪 rollout-policy drift。

---

## 11. 六个后续问题的分析与分歧路线

### 11.1 是否把 QCPO/DQCAC 换成 QCPO_refs 的 MLP+LSTM

#### QCPO_refs 的准确网络结构

当前移植版逐式对应参考仓库：

```text
raw obs + prev_cost
  → observation RMS，clip[-10,10]
  → MLP(512,Tanh) → MLP(512,Tanh)
  → concat(prev_action, prev_reward)
  → LSTM(512)，BPTT chunk=100
  → residual: h = MLP_feature + LSTM_output
  ├─ policy μ head + 可学习 log_std（初始 σ=1）
  ├─ scalar reward value head
  ├─ 25 个正值 cost quantile head：exp(linear)
  ├─ Weibull alpha head：4·sigmoid(linear)
  └─ Weibull beta head：exp(linear)
```

当前用户 QCPO/DQCAC 的 actor 则是独立 `MLP(256,256,Tanh)`，固定 σ；DQC 的 reward/cost critics 是各自独立的 `MLP(256,256,ReLU)`，GAE value 是另一个 `MLP(256,256,Tanh)`。当前 DQC actor 不直接看 `prev_cost/action/reward` 或 remaining budget。

#### 判断

在 Dynamic/Gremlin/Button 这类具有运动趋势、历史 cost 和剩余预算信息的环境上，LSTM 很可能有用；而且若要声称“算法差异”，确实必须消除 QCPO_refs 独占 recurrent/shared representation 这一结构优势。

但“所有输出完全相同”并不可行：QCPO_refs 输出的是 state/history-conditioned cost value distribution；DQC 理论要求 action-conditioned future-cost distribution `Z_c(h_t,a_t)`，再在当前 remaining budget `b_t` 查询 CDF。强行改成同一 cost 输出会改变 DQC 算法本身。应对齐的是输入历史、encoder、policy/value 容量和优化协议，同时保留各算法特有 head。

#### 可并行保留的三条路线

- **路线 R-A：actor/value backbone 对齐（推荐先做）。** 三个算法都使用相同的 `obs+prev_cost → MLP512 → concat(prev_action,reward) → LSTM512+skip`；actor 和 reward V 对齐。DQC cost critic 使用独立 recurrent encoder，并保留 `(history, action)→quantiles`。优点是最少破坏 DQC 半梯度/target-network 语义；缺点是总参数量大于 QCPO_refs。
- **路线 R-B：完全共享 recurrent encoder。** DQC 使用共享 history feature，接 actor、V_r、reward-Q 和 action-conditioned cost-Q heads；target critic 还要有 target recurrent encoder。优点是最接近 QCPO_refs 的共享辅助监督；缺点是 critic loss 会直接更新 policy representation，且 target hidden state/unroll 更复杂，属于新的 DQCAC-shared hybrid。
- **路线 R-C：MLP 公平控制组。** 把三个算法都设为 512×512 MLP、同 observation/history 显式输入、同可学习 σ，并关闭 QCPO_refs LSTM。它回答性能来自 recurrence 还是算法，必须作为消融保留。

轻量门槛：先用 reward-only 100k～300k、seed 0 比较 R-A/R-C；只有 recurrent 版本在相同 env steps 下形成稳定优势，才为 DQC cost critic 实现完整 recurrent target/BPTT。R-B 放在 R-A 通过后。

公平输入也有两种可报告口径：

- **历史公平：** 所有算法输入 raw obs、prev cost/action/reward，让 LSTM 自行推断累计 cost；最接近 QCPO_refs。
- **CMDP 信息公平：** 所有算法额外输入 `t/T`、累计 cost、remaining budget；DQC 不独占 budget 信息。该版本更有利于真正的 budget-adaptive policy，但应单独报告。

### 11.2 如何让当前 QCPO 的行为恢复正常

QCPO full baseline 的 500 万步训练耗时约 7.1 小时，eval reward `-0.054`、outage `0.038`、λ=0。共同 300k 步后段 reward 为 `-0.114`，趋势 `-0.345/百万步`。它不是被约束压坏，而是 reward policy-gradient 主干失败。

代码级原因：

1. 同一 rollout 做 5 次 actor update，却没有 fixed-old-policy ratio/clip；
2. raw observation 未归一化；
3. 固定 `σ=.5`，而 QCPO_refs 从可学习 `σ=1` 开始；
4. 整轨迹 MC return 权重广播到 1000 个 timestep，没有 value/GAE baseline，方差高且 credit assignment 很弱；
5. 当前 dual 使用本批 10 条轨迹，虽然该环境 λ=0，不是当前首因，但在其他环境会很噪。

保留三条明确命名的校准路线：

- **Q-A：QCPO-minimal。** 只加 observation RMS、固定 old log-prob、PPO clip、actor epochs=5/8、σ=1/可学习；reward 仍是轨迹级 MC，constraint 仍是轨迹 indicator。这是修正工程错误后最接近原用户 QCPO 的版本。
- **Q-B：QCPO-GAE/PPO hybrid。** reward 项改成 scalar V+GAE，constraint 项仍用轨迹级 indicator；加入 sum norm 和最近 100 条经验 dual。它用于公平比较“轨迹级 constraint credit”与 DQC per-transition constraint credit，但不再是纯原始 QCPO。
- **Q-C：QCPO-recurrent。** 在 Q-A/Q-B 中再换统一 MLP+LSTM backbone。先比较 Q-A vs Q-B，再做 LSTM，避免无法判断提升来自 baseline 还是 recurrence。

短预算门：DynamicButton、seed 0、先固定 λ=0 跑 300k。若后 20% reward 不能显著超过 0 且 slope 不持续为正，不恢复约束、不扩 seed。通过后再用经验 PID 恢复约束。

QCPO 完整 W&B 基线已重新通过 API 固化到 `_runs/wandb_export/qcpo_calibration_baseline_2026-07-16/`，共同 300k profile 在 `_runs/profiles/qcpo_calibration_baseline_at_300k_2026-07-16/`。

### 11.3 QCPO_refs 的 λ 更新是否仍值得继续调

值得，但要先固定策略结构。QCPO_refs 所谓 PID 的默认实际上是 **I-only**：`Kp=Kd=0`，用最近 100 条轨迹的 `Q_0.8(C)-d` 积分，`Ki=.1`。网络不同会影响参数迁移，但不是唯一原因。

同一个 `Ki` 不能迁移的主要原因包括：

- policy 对 risk loss 的灵敏度不同（closed-loop plant gain）；
- DQC 使用 `beta^t·CDF advantage`，QCPO_refs 使用 quantile-GAE/LDP cost advantage；
- risk advantage 的归一化尺度不同；
- actor LR、epochs、clip fraction 和网络记忆不同；
- λ 在 sum norm 下实际作用是 `λ/(1+λ)`；
- num_envs 改变后，每次 dual update 代表的 trajectory/env-step 数也改变。

保留三条控制器路线：

- **P-A：quantile I + 小增益。** `Ki∈{.003,.01,.03}`，加入 per-update Δλ clamp；最接近 QCPO_refs。
- **P-B：outage PI。** 概率误差带 deadband，P 项快速响应，I 项使用 leak/anti-windup；直接控制目标概率，严重度信息较弱。
- **P-C：双信号控制。** P 项看 outage gap，I 项看归一化 quantile gap；兼顾违反概率与严重度，但属于新控制器，需要独立消融。

所有路线都应加入：`I←leak·I+Ki·error`、约束恢复后允许快速回落、积分上下界、Δλ 上界，并按“每 100 条新完成轨迹”而不是“每 iteration”定义更新时钟。这样 num_envs 改变时控制器时间尺度不跟着变。

### 11.4 当前仓库推进到哪一步

- DQC reward 主干已经从“几乎不学习”推进到“30 万步可达到 QCPO_refs reward 量级”；
- fixed behavior IS/PPO、GAE、obs RMS、经验 dual、sum norm 和完整诊断均已实现；
- DQC constraint 目前能工作，但 E9 三 seed outage `0.262±0.054` 尚未稳定满足 0.2；
- QCPO 尚未应用上述修复，仍是下一个必须校准的算法；
- recurrent 公平 backbone、anti-windup PID、transition minibatch、大 N/smooth CDF/local τ/IQN 尚未实现；
- 所有训练仍必须通过持久化后台 launcher，允许 commit、未执行 push。

### 11.5 `num_envs` 的含义、硬件可行性与公平预算

`num_envs=B` 表示 B 个独立 Safety-Gym worker 同时各采一条 T=1000 轨迹。一次 iteration 获得 `B×T` transitions 和 B 个独立 episode cost；它不是把一个 episode 切成 B 段。

增大 B 的主要收益不是“更多 timestep”本身，而是：

- outage=.2 时每批超限轨迹数从 B=10 的期望 2 条增加到 B=32 的约 6.4 条；
- 经验 quantile、PID 输入和 tail calibration 方差下降；
- critic 每次看到更多不同初始布局/运动轨迹。

当前资源快照：128 个可用 CPU、A100 80GB、约 225GB 可用 RAM；`_runs` 仅约 63MB。因此 B=20/32 在 worker、内存和 GPU 参数容量上能承担，B=64 也可做吞吐 benchmark。

真正的限制是 DQC QR loss 的临时张量复杂度约为 `O(T·B·N²)`：当前实现会构造 `[T·B,N,N]` TD error。B=32、N=128 时单个 float 张量约 2.1GB，loss 内还有多个同尺寸张量和两个 critic；N=256 在不做 minibatch 时可能逼近/超过 80GB。结论是：

- 先开 B=20/32、保持 N=32，硬件可承受；
- 在同时增大 B 和 N 前，必须实现 transition minibatch/chunked quantile loss；
- LSTM512 本身不是显存瓶颈，QR pairwise loss 才是。

公平比较有两种预算，必须同时区分：

- **固定 env steps：** B 增大时按比例减少 iterations，比较样本效率；
- **固定 policy updates：** iterations 不变，总 env steps 随 B 增大，比较更低方差数据是否值得额外采样成本。

建议先 benchmark `B∈{10,20,32,64}` 的纯 rollout steps/s、完整 DQC iteration 时间和 peak GPU memory；正式短实验用 B=32。若吞吐在 B=32 前已饱和，B=20 是折中，但仍可因 tail 方差更低而保留为算法消融。

### 11.6 大 quantile、平滑 CDF、局部加密与 IQN

这四项都值得试，但不是同一个实验变量。

#### D-A：均匀 quantile 数量

`N=32→64→128`，CDF 概率分辨率从约 `.0313→.0156→.0078`。优点是实现最简单；缺点是当前 QR loss 计算/显存近似按 N² 增长。先实现 transition minibatch，再在固定策略数据上比较 CDF bias、crossing 和 wall time。

#### D-B：平滑 CDF

保留 hard CDF 作评估，把 actor risk advantage 的

```text
mean(1{quantile_i >= budget})
```

替换为

```text
mean(sigmoid((quantile_i-budget)/temperature))
```

temperature 可走两条路线：固定 cost 单位 `{0.5,1,2}`；或使用查询附近 quantile spacing 的 detached 倍数自适应。平滑会降低 1/N 量化跳变和 action baseline 方差，但会引入温度偏差，所以 hard/smooth calibration 必须同时记录。

#### D-C：uniform skeleton + 查询区局部加密

不能简单把更多 τ 点堆在 0.8 后仍等权平均，否则目标分布本身会被重加权。合法路线包括：

- uniform 32/64 骨架 + local τ，target dimension 使用 Voronoi/quadrature 权重；
- uniform/local mixture sampling，并对非均匀采样做 importance weighting；
- 只在查询/反演时加密，不改变训练分布。

另外，DQC 查询的是随 history 改变的 remaining budget，常见 crossing τ 不一定总在 0.8。应先记录全 rollout 的 crossing-τ 直方图，再决定局部区间；固定只加密 `[.7,.9]` 应作为一个假设而不是默认真理。

#### D-D：IQN

IQN 学的是 `Q(h,a,τ)`，不是直接输出 CDF。查询 CDF 时需要在 dense τ 网格上前向、排序/单调化并反演 budget crossing。它的优势是训练 τ 数固定、查询时可临时增加 τ 分辨率；代价是 actor 每步还要对 K 个 baseline actions 多次 τ 前向。

保留两条 IQN 路线：标准 uniform-τ IQN；以及 query-focused mixture-τ IQN + importance weights。两者都必须和相同计算预算的 N=64/128 QR、smooth CDF 比较，而不能只比网络名字。

可额外保留 **D-E：non-crossing quantile head**，复用仓库 NQ-Net 的单调结构或单调增量参数化。它可能比 IQN 更直接地改善 CDF 反演，应纳入后续候选，但不与第一轮 N/平滑 CDF 同时打开。

---

## 12. 分阶段执行计划：短期正常化，长期超过 QCPO_refs

### 12.1 总体实验纪律

1. 每次只改变一个主要机制；有分歧时保留路线编号，不提前删除候选。
2. 所有 job 用持久化后台 launcher；启动前写预计耗时，结束后用 W&B API 拉全 history/profile。
3. 先 seed 0、100k～300k；前半段 reward slope、outage 或数值健康明显失败就停止。
4. 通过单 seed 门后补 seed 1/2；只有三 seed 同方向才扩到 1M，最后才做 5 seed/full budget。
5. 正式比较使用 equal env steps、相同输入信息、相同 actor/value backbone；同时报告参数量、wall time 和 peak GPU memory。
6. reward 必须在约束可行或 outage 上置信界可接受时比较；不能用高 reward、严重违规的策略宣布胜出。

### 12.2 短期 S0：吞吐与显存门（预计 10～20 分钟）

| 实验 | 变量 | 预算 | 通过条件 |
|---|---|---:|---|
| S0-a | B=10/20/32/64，N=32 | 每项 1～2 rollout | 得到 steps/s、iteration time、worker 稳定性 |
| S0-b | B=32，N=32，当前 E9 update | 1～3 iterations | 无 OOM/NaN，记录 peak memory |
| S0-c | N=64/128 的 chunked QR smoke | 小 T/B | 验证数值等价与显存下降 |

### 12.3 短期 S1：先校准 QCPO（预计每个 300k 候选 3～10 分钟，需实测）

1. Q-A0：旧 QCPO，但 actor updates=1；确认纯 on-policy MC 是否仍不学。
2. Q-A1：obs RMS + σ=1/learnable，actor updates=1。
3. Q-A2：Q-A1 + fixed-old ratio/PPO clip，actor epochs=8。
4. Q-B：Q-A2 + reward V/GAE，明确标为 hybrid。
5. 只有 reward-only 通过后，加入最近 100 trajectory dual + sum norm。

主要判据：300k 后 20% reward 明显大于旧 QCPO 的 `-0.114`，slope 为正，PPO KL/clip 有限。Q-A2 若已经正常，Q-B 就作为算法效率消融而非必修修复。

### 12.4 短期 S2：公平 recurrent backbone（预计开发+smoke 后每候选 5～15 分钟）

1. 先实现可切换的 recurrent adapter，不删除现有 MLP 路径；对拍 QCPO_refs 的 forward、hidden reset 和 BPTT transform。
2. R-C：三算法 512×512 MLP 控制组。
3. R-A：三算法 actor/V 使用同 MLP512+LSTM512+skip 和历史输入。
4. DQC cost critic 先使用独立 recurrent encoder；通过后再做 R-B full-shared hybrid。
5. 分别跑 history-fair 与 explicit-budget-fair 两种输入协议。

### 12.5 短期 S3：重新辨识 dual（每候选先 100k～300k）

固定 S2 胜出的网络后再调 PID：

1. P-A `Ki=.003/.01/.03` quantile-I；
2. P-B outage PI + deadband + leaky-I/anti-windup；
3. P-C probability-P + quantile-I 双信号；
4. 每 100 条新轨迹更新一次，限制 Δλ，并记录 effective risk coefficient；
5. 单 seed 选出不过冲的 Pareto 工作点，再补三 seed。

### 12.6 中期 S4：distributional critic 组件矩阵

严格顺序：

1. transition minibatch/chunked QR；
2. N=32/64/128 uniform QR；
3. hard vs fixed-temperature smooth vs spacing-adaptive smooth CDF；
4. crossing-τ 数据收集；
5. uniform skeleton + weighted local τ；
6. IQN uniform/local-mixture；
7. non-crossing head。

先做固定策略离线校准，再做短 online actor 实验。离线指标包括 query CDF bias、Brier/calibration、crossing、不同 budget 区间误差和推理耗时；online 指标才看 constrained reward。

### 12.7 长期目标与超过 QCPO_refs 的证据标准

DQCAC 理论上可能超过 QCPO_refs 的理由是：它用每个 transition 学习 action/history-conditioned cost distribution，并在 remaining budget 上产生局部 risk credit；QCPO_refs 的约束 credit 更接近 trajectory/quantile level。这个优势更可能体现在**样本效率和相同 outage 下的 reward**，不是无条件保证最终分数更高。

最终需要同时证明：

- equal env steps 下更快进入可行高 reward 区域；
- 5 seeds 下最终 constrained reward 高于 QCPO_refs；
- 256～1000 eval episodes 下 outage 上置信界满足目标；
- 使用相同 recurrent actor/value backbone 和输入信息后优势仍存在；
- 去掉 DQC budget/action-conditioned cost head 后优势消失或显著下降；
- wall time、GPU memory 和参数量透明报告。

长期主线推荐：`公平 recurrent backbone → 稳定 dual → B=32 独立轨迹 → chunked large-N/smooth CDF → local τ/IQN/non-crossing → 多环境多 seed`。其中每一箭头都保留上一阶段的 MLP/简单 critic 作为消融与回退点。

## 13. 2026-07-16 QCPO 校准实证更新

### 13.1 修复是否让 QCPO 恢复正常学习

是，但只恢复到“稳定上升”，还没有达到 QCPO_refs/DQCAC E5 的样本效率。

- 旧 QCPO 的两个确定性错误已经修复：同 rollout 多次 actor 更新不再缺少 IS/PPO；observation RMS 现在真实更新，并在采样/actor epochs 内冻结。
- 100k 消融中，raw-observation 单次 on-policy 后段 reward 为 `-0.0419`；obs RMS + learnable sigma 单次 on-policy 为 `-0.0151`；再加 fixed-old PPO 8 epochs 后为 `0.0385`、趋势 `+0.649/百万步`。
- Q-A2 独立 300k 终评 reward `0.3496`；训练后段 `0.2297`、趋势 `+1.248/百万步`。它显著超过旧 QCPO 同预算的 `-0.1139/-0.345`，证明 bug 修复有效。
- 同 300k，DQCAC E5 为 `1.662/+6.816`，QCPO_refs 为 `1.355/+5.431`。因此 QCPO 的剩余差距主要是整条轨迹 MC reward 信号的高方差/差 credit assignment，而不是继续归咎于 IS 或 normalization。

原始完整 history 与图在 `_runs/profiles/qcpo_qa012_100k_2026-07-16/` 和 `_runs/profiles/qcpo_a2_vs_key_300k_2026-07-16/`。

### 13.2 下一条最小路线：Q-B reward GAE hybrid

已加入 `qcpo_reward_mode=gae`：reward 使用独立 scalar `V_r(s,t)`、GAE 和 PPO；constraint 仍使用 QCPO 的轨迹级 outage indicator。该路线不再称为纯原始 QCPO，而用于隔离“reward credit assignment”与“constraint credit assignment”。

实现保持四个不变量：observation RMS 共享；有限期界 value 输入追加 `t/T`；GAE/target 在一个 rollout 的所有 epochs 中冻结；行为 `old_log_prob` 始终固定。risk/reward PPO surrogate 也已拆开，分别使用 conservative max 与 pessimistic min。后台 smoke 已以 exit code 0 完成。

该门已通过：Q-B 在 100k 后段达到 reward `0.557/+6.950`，独立 300k 后段达到 `1.317/+5.059`，130 条终评 `1.621`。它与 QCPO_refs 同预算 `1.355/+5.431` 和 DQCAC E5 终评 `1.690` 已处于同一量级，reward 主干可判为表现正常。reward-only outage `0.546` 不代表约束版本失败；lambda 此处刻意固定为 0。

因此下一步进入统一 MLP+LSTM，同时保留 Q-B 作为 MLP 复现基准。完整 100k/300k 对齐 profile 位于 `_runs/profiles/qcpo_qb_vs_key_100k_2026-07-16/` 与 `_runs/profiles/qcpo_qb_vs_key_300k_2026-07-16/`。


### 13.3 全尺寸 MLP+LSTM 的实测结论（2026-07-16）

QCPO 的 QCPO_refs 等价 recurrent adapter 已完成 100k 和独立 300k reward-only 校准；两次均通过持久化后台运行并 exit code 0。

- 100k：job `QCPO_DynamicButton_recur_qb_r0_qbhyper_100k_s0`，W&B `6tfy9aeo`，训练 `63.7s`。后 20% reward `0.4006`、趋势 `+5.006/百万步`，70 条终评 reward `0.3169`、outage `0.1714`。
- 独立 300k：job `QCPO_DynamicButton_recur_qb_r0_qbhyper_300k_s0`，W&B `312u1kr3`，训练 `159.8s`。后 20% reward `1.083`、趋势 `+4.114/百万步`，70 条终评 reward `1.251`、outage `0.557`。
- 同预算 MLP Q-B 的 100k/300k 后段 reward 为 `0.557/+6.950` 与 `1.317/+5.059`，终评为 `0.541/1.621`。因此 recurrent R0 会稳定学习，但 300k 终评比 MLP 低约 `22.8%`，尚不能说“换 LSTM 后复现旧结果”。
- recurrent 数值健康：100k value explained variance `0.647`，PPO KL `0.00224`、clip fraction `0.119`；300k value explained variance 终点约 `0.787`，没有 NaN/Inf。失败形态是优化速度偏慢，而不是 hidden reset、old-log-prob 或 BPTT 接线错误。
- 共同 300k 下 QCPO_refs 后段 reward `1.355/+5.431`，DQCAC E5 为 `1.662/+6.816`。QCPO_refs 在该时点 lambda 已约 `0.94`，所以 reward/outage 联合比较要等统一约束实验；本节只判定 reward 学习主干。

完整对齐数据：

- `_runs/profiles/qcpo_recurrent_r0_100k_2026-07-16/`
- `_runs/profiles/qcpo_recurrent_r0_300k_2026-07-16/`

由此保留而不提前合并的 QCPO recurrent 路线：

- **Q-R0（已完成）**：Q-B 的 `lr=3e-4/grad_clip=1/value_coef=1` 原样换 LSTM，回答“只换网络是否自动变好”；答案是否定的。
- **Q-R1（参考优化器）**：改用 QCPO_refs 的 `lr=1e-4/clip_grad=1e4`，其余不变；隔离大共享网络是否被过强 clipping 或偏高学习率限制。
- **Q-R2（共享 loss 比例）**：`value_coef∈{0.25,0.5,1}`，记录 policy/value 各自梯度或 PCGrad/分离 head 作为后续选项；检验共享 value loss 是否压慢 actor representation。
- **Q-R3（容量控制）**：`256×256+LSTM256` 与 `512×512+LSTM512` 在相同预算下比较；若小网络更快，只说明优化/容量问题，正式结构公平表仍保留 512 版本。
- **Q-R4（长期回退）**：保留 512×512 MLP Q-B 作为强控制组。若 recurrent 在多个环境不占优，不因“参考算法用了 LSTM”而强行把它设为默认。

前半段门控：Q-R1/R2/R3 各先 100k；若后段 reward/斜率不能超过 R0 的 `0.401/+5.01`，不扩 300k。只有至少接近 MLP 的 `0.557/+6.95` 才扩大预算。

### 13.4 DQCACBeta 的 MLP+LSTM 接入设计与当前状态（2026-07-16）

已实现默认关闭的 `policy_arch=mlp|mlp_lstm`。默认仍为 MLP，保证 E1–E10 可复现。首条 D-R0 路线的边界是：

- actor 与 reward-V 使用和 QCPO_refs 数值对拍过的 `[obs,prev_cost]→MLP512→[prev_action,prev_reward]→LSTM512+skip` 共享骨干；
- PPO rollout 保存固定 `old_log_prob`、每步进入前 `h0/c0`、previous action/reward/cost 和 rollout value；按 seq=100 做 BPTT，value 与 policy 联合 backward；
- reward/cost distributional critics 暂时保持 DQCAC 的 action-conditioned `Z(h_t,a_t)` 的 Markov 近似输入 `(state,t/T,a)`。这不是把 QCPO_refs 的 state-value cost head硬套进 DQC；
- N-step bootstrap 的动作取同一 on-policy rollout 在完整历史下保存的 `a_{t+N}`，包括单独生成但不执行的 `a_T`。不能在中间 state 用零 hidden 重采；
- cost baseline 的 K 个动作从该时刻 rollout 保存的历史条件行为高斯参数采样；risk advantage 在首个 actor epoch 固定，和 old-log-prob/GAE target 一起跨 epoch 冻结；
- actor 的 augmented-observation RMS 在所有 PPO epochs 后才合并；critic 继续使用独立 raw-state RMS；循环独立评估器会逐 episode 重置历史并保留 cost-critic CDF 校准输出。

持久化小网络 smoke `DQCAC_DynamicButton_smoke_recurrent_dr0_s0` 使用 `B=2,T=32,N=8,[32,32]+LSTM32,seq=16`，2 个 critic/actor epochs，训练 `4.7s`、exit code 0；rollout、on-policy bootstrap、联合 BPTT、RMS、dual 初始动作和独立评估全链路通过。旧 MLP 回归 smoke `DQCAC_DynamicButton_smoke_mlp_regression_after_recurrent_s0` 训练 `4.5s`、exit code 0，确认默认构造、GAE/PPO、critic 和统一评估未被循环分支破坏。另记录 `ppo/first_epoch_ratio_max_error`：首个循环 actor epoch 前 actor/RMS 尚未变化，ratio 应严格接近 1，可直接检测 history/chunk h0/old probability 接线错误。

DQC 的网络分歧继续保留为显式消融：

- **D-R0（当前、最低风险）**：recurrent actor/reward-V + 原 action-conditioned MLP distribution critics。Safety-Gym 原始 state 在 Markov 假设下足以预测 future cost，历史主要用于策略；先回答结构公平是否改善 reward。
- **D-R1（历史 cost critic）**：给 cost critic 独立 recurrent history encoder，再接 action-conditioned quantile head；online/target encoder 都按历史 unroll，显式处理 target hidden。它回答 partial observability/历史 cost 是否改善查询 CDF。
- **D-R2（full-shared hybrid）**：actor、V、reward/cost critic 共享 recurrent encoder，target 保留独立副本；样本效率可能更高，但 critic 梯度直接改变 policy representation，必须单列为新算法。
- **D-R3（explicit-budget fair）**：三个算法策略都额外输入 `t/T、累计 cost、remaining budget`；DQC 不独占 budget 信息。与 QCPO_refs 原输入的 history-fair 口径分开报告。
- **D-RC（MLP 公平控制）**：三算法均使用 512×512 MLP、可学习 sigma、相同显式历史统计；用于区分 recurrence 与容量/归一化贡献。

下一门是 D-R0 reward-only 100k、seed 0，复用 E5 的 `GAE+PPO8+obs RMS` 并固定 lambda=0。预计训练约 `70–100s`，加 70 轨迹评估约 `2–3min`。通过条件：后段 reward 与 slope 至少形成 E5 同方向增长，PPO ratio/KL、联合 value EV、critic loss 无异常；明显低于 E5 前段时不直接扩全预算，而先在 Q-R1/R2 类型的循环优化超参中做轻量分支。


### 13.5 DQCACBeta recurrent D-R0 的 100k/300k 结果（2026-07-16）

D-R0 已先后完成 100k 门控和独立 300k 扩展，均由持久化 launcher 运行并 exit code 0。

- 100k：job `DQCAC_DynamicButton_recur_dr0_e5hyper_100k_s0`，W&B `j57xp70k`，训练 `55.4s`。后段 reward `0.4278`、趋势 `+6.164/百万步`，70 条终评 reward `0.4965`、outage `0.1286`。
- 同预算 DQC E5 MLP 为 `0.4441/+6.291`，QCPO_refs 为 `0.4776/+7.008`，循环 QCPO 为 `0.4006/+5.006`。因此 D-R0 在前 100k 已基本复现 E5，不存在“换网络后完全不学”的问题。
- 首 actor epoch 的 `max|ratio-1|` 全程约 `0.7e-5～1.1e-5`；这是 LSTM 逐步 rollout 与 chunk forward 的浮点累积误差量级，证明 old probability、history 和 chunk h0 对齐。100k 终点 value EV `0.594`、PPO KL `0.00338`、clip `0.192`。
- 独立 300k：job `DQCAC_DynamicButton_recur_dr0_e5hyper_300k_s0`，W&B `tavefru8`，训练 `153.1s`。后段 reward `1.381`、趋势 `+5.672/百万步`，70 条终评 reward `1.442`。
- 共同 300k 的后段 reward：D-R0 `1.381` 已略高于 QCPO_refs `1.355`、MLP Q-B `1.317` 和循环 QCPO `1.083`，但低于 DQC E5 MLP `1.662`。因此同 recurrent actor/V 下 DQC 的 reward 学习已超过 QCPO 路线，但当前尚未超过最强 MLP DQC 控制组。
- 300k 后段 PPO KL `0.00103`、clip `0.0456`，value EV `0.722`，没有更新过猛或数值异常。差距更像大共享网络在后期更新不足；下一轻量消融是提高 actor LR/epochs，及把 shared value coefficient 从 1 降到 0.5，而不是继续修 hidden state。

完整 profile：

- `_runs/profiles/dqc_recurrent_dr0_100k_2026-07-16/`
- `_runs/profiles/dqc_recurrent_dr0_300k_2026-07-16/`

必须同时记录一个新的约束警报：D-R0 300k cost critic CDF `0.322` 对真实 outage `0.586`，低估 `0.263`；预测 mean cost `12.47` 对实际 `23.56` 也明显偏低。E5 MLP 同预算的 CDF 偏差仅约 `-0.101`。所以当前结果只能证明 reward 主干，不能证明 constrained DQC 已胜出。恢复约束时优先使用经验 trajectory PID；D-R1 recurrent cost critic、n-step/target 传播和大 N/smooth CDF 分开消融。当前主要是整段分布传播/条件表示偏差，单纯把 N 从 32 加大不会自动消除 mean/CDF 低估。

### 13.6 大 B/大 N 的 QR transition chunking（2026-07-16）

已加入默认关闭的 `critic_minibatch_size=0`；0 保持历史整批 `[T·B,N,N]` QR loss，正数则按 transition 顺序分块：

1. target quantiles 仍一次冻结为 `[T·B,N]`；
2. 每块只构造 `[chunk,N,N]` pairwise error；
3. chunk mean loss 乘 `chunk_size/(T·B)` 后 backward；
4. 所有块共享一次 `zero_grad → clip → optimizer.step`。

因此块数不会放大学习率或 optimizer step 数。合成 `M=11,N=7,chunk=4` 的 uneven-chunk 对拍中，full/chunk loss 绝对误差 `4.33e-8`，quantile prediction 梯度最大绝对误差 `0.0`。持久化集成 smoke `DQCAC_DynamicButton_smoke_chunked_qr_s0` 使用 `T·B=64,N=16,chunk=17`，训练 `4.4s`、exit code 0，并通过 actor/critic/eval 全链路。

这只是显存基础设施，不宣称性能提升。后续先做 B=10/20/32 的 wall-time/peak-memory benchmark，再用 chunking 比较 N=32/64/128；所有大 N 实验同时报告吞吐，避免以数倍计算换来不可比的微小变化。

### 13.7 Recurrent actor 优化与 cost-target 分支（2026-07-16）

100k 单变量消融表明，提高 actor LR 有效，降低共享 value loss 权重无效。`lr=3e-4,value_coef=1` 的后段 reward 为 `0.4766`、趋势 `+6.716/M`，相对 D-R0 的 `0.4278/+6.164/M` 改善，并几乎等于 QCPO_refs 同预算的 `0.4776/+7.008/M`；PPO KL `0.00310`、clip fraction `0.178` 仍健康。相反，`lr=2e-4,value_coef=0.5` 只有 `0.3755/+5.797/M`，不扩长预算。W&B 分别为 `tmckgcax` 与 `x37vpl83`，完整对齐数据在 `_runs/profiles/dqc_recurrent_actor_ablation_100k_2026-07-16/`。

这没有解决 constrained DQC 的核心问题。100k lr 路线的 cost critic 后段只预测平均 cost `1.13`，rollout 实际平均为 `7.40`；终评 CDF `0.00045` 对 empirical outage `0.229`。因此后续路线按可归因性排序并全部保留：

1. **C-T1 MC target**：用完整 episodic cost return-to-go 直接监督 quantiles，先判断低估是否来自 n-step bootstrap 传播；
2. **C-T2 mixed/lambda target**：只在 MC 校准好但方差过大时尝试；
3. **C-H1 independent recurrent cost critic**：MC 仍低估时检验历史信息缺失；
4. **C-H0.5 shared history feature**：作为更轻但表示漂移风险更高的对照；
5. **C-Q large-N/smooth/local tau/IQN**：整体 cost mean/return 先校准后再做查询点精度优化。

代码新增默认关闭的 `cost_target_mode=nstep|mc`。默认 `nstep` 完全保持旧行为；`mc` 只允许完整 episodic rollout，反向计算 `G^c_t=c_t+gamma_c G^c_{t+1}`，不改变 reward critic、GAE、PPO 或 actor。持久化 recurrent+chunked smoke `DQCAC_DynamicButton_smoke_recur_mc_cost_s0` 训练 `5.8s`、exit code 0。

C-T1 100k（W&B `wmlzu8la`）与 n-step 对照的 reward 序列逐点相同。MC 将终评 predicted mean cost 从 `1.49` 提到 `4.02`（真实 `8.96`），将 s0 CDF 从 `0.00045` 提到 `0.0384`（真实 outage `0.2286`）。所以 bootstrap 传播是原因之一，但不是全部原因；MC 不直接扩 300k。完整 profile 在 `_runs/profiles/dqc_cost_target_mc_100k_2026-07-16/`。下一条 C-O1 只把 critic epochs 从 10 提到 20，并记录 reward/cost/joint 裁剪前梯度；若仍低估，再进入 C-H1。最终形成 `nstep/MC × Markov/recurrent-cost` 的 2×2 消融，而不是只汇报胜者。


### 13.8 Cost critic 优化与 leaky-I 控制路线（2026-07-16）

MC target 后，主要剩余误差首先来自 critic 优化不足。把 critic epochs 从 10 增至 20、保持 `lr=1e-3`，终评 predicted mean cost 从 `4.02` 提到 `6.97`（truth `8.96`），CDF 从 `0.0384` 提到 `0.1308`（truth `0.2286`）；wall time 只从 `56.9s` 增到 `59.4s`。相反，保持 10 epochs、把 critic LR 提到 `2e-3` 只得到 mean/CDF `5.91/0.0987`，所以保留 C20。C20 的 joint grad norm 全程 `1.37～7.86<10`，不是 gradient clipping 造成的低估。profile 位于 `_runs/profiles/dqc_cost_optimizer_c20_100k_2026-07-16/`。

PID 分歧也显式保留。当前 QCPO_refs 配置所谓 PID 实际 `Kp=Kd=0`，只有 bounded I；E10 的失败更接近窗口滞后和 lambda 不衰减，而非隐藏积分状态超过上限。代码因此新增默认关闭的 bounded leaky-I：leak、deadband、每次最大 delta，以及按新增 episode 数缩放的 reference。默认 `leak=1,deadband=0,delta_max=inf,reference=0` 与旧结果逐式相同；开启 episode scaling 时用几何积分和，保证常值误差下一次 B=20 等价于两次 B=10。手算断言与持久化 smoke 均通过。

首条 P-B1 constrained 门控固定 C20+MC/recurrent reward 主干，使用 `beta=.995,sum_norm=true,outage error,Ki=.1,leak=.97,deadband=.02,delta_max=.05,reference=10`。它与 E9 legacy-I 分开命名；若控制偏弱，保留 leak=.98/Ki=.15 路线，若过保守则保留 leak=.95/更大 deadband。recurrent cost critic C-H1 继续保留，但不与 PID 同一 run 同时引入。


### 13.9 P-B1 结果与真正的 PI 路线（2026-07-16）

P-B1 的 300k leaky-I 门控失败于控制偏弱，而非 windup。λ 在 170k 前一直为 0，到 200k/250k/300k 仅约 `0.012/0.056/0.130`；130 条终评 reward `1.332`、outage `0.462`，不满足 0.2。critic CDF `0.295` 对 truth `0.462`，bias `-0.167`；因此 trajectory PID 绕开 critic 只能保证 dual 信号真实，actor 的局部 risk advantage 仍受 cost critic 低估影响。W&B 为 `9pxv0bmd`。

为直接修复迟滞，代码增加默认 `pid_Kp=0` 的 PI 输出 `lambda=clip(I_state+Kp*filtered_error)`。Kp=0 与旧 I 路径逐式相同；Kp>0 时当前 error 能立即影响 λ，并在约束恢复后立即撤回，而 leaky-I 只承担稳态项。手算断言和强制正误差 smoke 均通过，后者实际输出 λ=0.7999。P-B2 只把 Kp 设为 1，其余沿用 P-B1；若仍偏弱，Kp=2 与 window=50 作为两个独立消融，不同时打开。C-H1 recurrent cost critic 继续保留为 PI 之后的结构主线。


### 13.10 P-B2 结果：PI 更快但仍太晚（2026-07-16）

P-B2 只增加 `Kp=1`。λ 在 180k/190k 达到 `0.011/0.045`，约为 P-B1 的 9 倍，最终 λ `0.356=I 0.146+P 0.210`；实现和参数确实生效。但 130 条终评 reward `1.303`、outage `0.508`，仍不可行。critic CDF `0.365` 对 truth `0.508`，pred mean `14.75` 对 `22.33`，说明更快 dual 无法自动修复局部 risk advantage 的表示/校准误差。W&B 为 `rgt6qukp`。

只再运行一条 window `100→50` 的 P-B3，隔离 5-iteration 观测滞后；若它仍不可行，就停止 PID 小网格并实现 C-H1 recurrent action-conditioned cost critic。Kp=2、Ki 调大继续记录为可能消融，但不在没有新证据时消耗全量训练预算。


### 13.11 P-B3 结果：缩短窗口有效，但 PID 小网格到此停止（2026-07-16）

P-B3 只把 P-B2 的经验窗口从 100 条轨迹缩到 50，其余配置保持不变。job 为 `DQCAC_DynamicButton_recur_mc_c20_pi_kp1_w50_300k_s0`，W&B `o8nx0elk`，启动 commit `45697a5`，训练 `157.6s`、exit code 0。

- λ 比 P-B2 更早响应：150k 时 batch outage `0.5`、λ 已为 `0.088`；约束恢复时能回落到 170k 的 `0.0095`，说明 leaky-I+P 没有继续 windup。后期仍有窗口噪声导致的闭环振荡，最终 `lambda=0.307=I 0.147+P 0.160`。
- 300k 后段 reward `0.9806`、趋势 `+3.994/M`，训练后 130 条终评 reward `1.132`、outage `0.315`。相对 P-B1/P-B2 的 outage `0.462/0.508` 是明确改善，但仍高于目标 `0.2`，所以不能判为可行解。
- 终评 cost critic CDF `0.294` 对 truth `0.315`，初始状态的 CDF 偏差已不大；pred mean `12.35` 对真实 mean cost `13.07` 也接近。但训练最后窗口 probability `0.38`、最后 batch `0.2`，说明短窗口同时带来明显估计噪声。
- 旧 E9 seed0 终评刚好达到 outage `0.2`，但 reward 只有 `0.658`；P-B3 是更好的 reward-risk 候选工作点，却尚未同时满足约束并超过 QCPO_refs。

共同 300k profile 位于 `_runs/profiles/dqc_recurrent_pid_p_b123_300k_2026-07-16/`，原始导出位于 `_runs/wandb_export/dqc_recurrent_pid_p_b123_300k_2026-07-16/`。该 profile 同时包含 P-B1/P-B2/P-B3、E9 seed0 与 QCPO_refs，未发现 NaN/Inf。

**阶段决策**：停止当前 Kp/Ki/window 小网格。P-B3 证明控制迟滞是原因之一，但剩余问题不能仅靠 controller gain 解释；继续放大 Kp 可能只会把窗口噪声变成更强 actor 振荡。`Kp=2`、`Ki=.15/leak=.98`、cost-quantile PI 和双信号 P/I 仍记录为控制器消融路线，只有 cost 表示/局部 risk advantage 校准后再回测，不立即消耗全量预算。

### 13.12 C-H0.5：先用冻结 actor-history feature 检验非 Markov cost critic（2026-07-16）

一个此前容易忽略的结构问题是：环境 observation 对物理系统可能是 Markov 的，但 recurrent policy 的未来动作还依赖 hidden state。因此固定 recurrent policy 下正确的 cost action-value 一般是 `Z_c(s_t,h_t,a_t)`，不是当前实现的 `Z_c(s_t,a_t)`。D-R0 的 cost critic 看不到 `h_t`，同一 observation/action 在不同策略记忆下会被迫拟合成混合分布；这会直接污染查询点 CDF 和 action baseline 差值。

为在重写完整 recurrent critic 前做可归因的轻量验证，新增默认关闭的 `cost_history_mode=raw|actor_feature`：

1. `raw` 完全保留历史 `(normalized state,t/T,a)→N quantiles`；默认值不变。
2. `actor_feature` 保存产生行为动作的 `φ_t=MLP(obs,prev_cost)+LSTM(history)`，显式 detach 后输入 action-conditioned cost quantile MLP；reward critic/reward-V/actor 均不改变。
3. n-step bootstrap 同时保存完整历史下的 `a_{t+N}` 与 `φ_{t+N}`；episode 末的未执行 `a_T/φ_T` 也配对保存。MC 路径虽不使用 cost bootstrap，仍保留同一批接口。
4. rollout risk advantage、K-action baseline、constraint RMS、critic dual、s0 校准和独立评估都走统一 `_cost_inputs`，避免训练用 history、评估却误用 raw state。
5. feature 在 rollout 的 `no_grad` 内产生并在适配器再次 detach，cost loss 不更新 actor。这是 C-H0.5 shared-input head，不是假装已经实现 C-H1 full recurrent critic。

验证已通过：旧四元 `RecurrentActorValue.forward` 与可选五元 feature 接口的 policy/value/hidden 逐元素完全相同；持久化 `actor_feature` smoke 与只改回 `raw` 的回归 smoke 均训练约 `6s`、exit code 0，覆盖 MC、chunked QR、两次 PPO、PI、recurrent eval 和进程回收。

存在分歧的后续路线全部保留：

- **C-H0.5A（当前门控）**：冻结 actor recurrent feature；实现最轻，直接检验策略历史是否有用，但输入维度/非线性表示也随之增加，且 actor 表示跨 iteration 漂移。
- **C-H0.5B（容量/表示控制）**：若 A 有效，再比较同维 actor MLP-only feature（无 LSTM history）或容量匹配的 raw projection，区分“历史”与“更大输入表示”。
- **C-H1（推荐长期主线）**：独立 online/target recurrent cost encoder + action-conditioned quantile head；cost 表示不随 reward actor 漂移，target hidden 语义完整，但实现与计算更重。
- **C-H2（raw+history concat）**：同时保留物理 state 与 history feature，信息最全但参数最多；只有 A/H1 显示历史有效时作为结构消融。
- **C-H3（full shared hybrid）**：允许 cost loss 更新共享 actor encoder；样本效率可能更高，但已是新算法，必须与 detach 版本分开报告。

下一条只跑 C-H0.5A reward-only 100k：沿用 C20+MC、actor lr `3e-4`、`lambda=0`，与 raw C20（W&B `059boy42`）一项对照。通过门槛是：CDF 绝对偏差相对 `0.0978` 至少下降 25%，或 mean-cost 相对误差从 22% 降到 15% 内，同时另一校准量不恶化超过 10%、reward/PPO 不异常。未通过就不跑 300k，直接实现 C-H1；通过后才进入 P-B3 控制器下的 300k constrained 门控，并补 C-H0.5B 容量控制。


### 13.13 C-H0.5A 100k 结果：共享 actor feature 未通过（2026-07-16）

C-H0.5A job `DQCAC_DynamicButton_recur_mc_c20_ch05_actorfeature_100k_s0`（W&B `ooy3dlxs`）从 commit `6780296` 启动，训练 `58.7s`、exit code 0。它与 raw C20（W&B `059boy42`）只有 `cost_history_mode=actor_feature` 一项差异。

隔离结果非常干净：两条 run 的每个 reward 点、真实 cost/outage、PPO 与 value 指标相同；共同后段 reward 都是 `0.4766/+6.716/M`，70 条终评均为 reward `0.5527`、outage `0.2286`、mean cost `8.957`。因此差异只来自 cost critic 条件表示。

但历史 feature 没有改善校准：

- raw C20：hard CDF `0.1308`，绝对偏差 `0.0978`；pred mean `6.974`，相对真实 mean 低估约 22.1%。
- actor feature：hard CDF `0.1129`，绝对偏差 `0.1156`；pred mean `6.634`，低估约 25.9%。
- CDF 偏差恶化约 18%，mean 也变差，明确不通过预设门槛；不扩 300k、不补 seed。

完整 profile 位于 `_runs/profiles/dqc_cost_history_ch05_100k_2026-07-16/`，原始导出位于 `_runs/wandb_export/dqc_cost_history_ch05_100k_2026-07-16/`。

该负结果削弱但没有完全排除 cost-history 假设：当前 `φ=MLP+LSTM output` 不含 LSTM cell state，且它由 reward/PPO 目标训练、跨 iteration 漂移，并不等价于独立 cost recurrent sufficient state。C-H1 独立 online/target cost RNN、直接拼 actor `(h,c)` 的 C-H0.6、MLP-only 容量控制 C-H0.5B 均保留为消融路线；不过在 H0.5A 没有任何正信号后，不应立即做最重的结构改写。

### 13.14 C-Q1 查询点平滑 CDF：优先处理 hard-count 稀疏性（2026-07-16）

raw/actor-feature 的末期 hard-CDF action advantage 标准差仅约 `0.005～0.007`。原因是 N=32 时每个动作的概率只能以 `1/32` 跳变；绝大多数 `(history,budget)` 要么所有 quantiles 都在阈值同一侧，要么 K 个候选动作得到相同计数，导致 risk action advantage 精确为 0。分布 critic 的 mean/CDF 校准只是必要条件，不能保证查询点附近有可用的动作排序信号。

新增默认关闭的 `cost_cdf_mode=hard|sigmoid` 与 `cost_cdf_temperature`：

- hard 逐元素复现 `(1/N)ΣI{z_i≥b}`；手算断言完全相同，历史实验默认不变。
- sigmoid 使用 `(1/N)Σsigmoid((z_i-b)/T)`，首条路线固定 `T=1 cost unit`。它只用于 actor 实际动作、K-action baseline 及 constraint-RMS 的 risk surrogate。
- QR target/loss、cost critic hard CDF 校准、empirical outage 和 empirical PI 全部保持 hard/真实口径；日志同时保存 hard/smooth s0 CDF，不能把 smooth surrogate 当作约束已满足。
- 新增 `risk_adv_abs_mean` 与 `risk_adv_nonzero_fraction`，直接验证平滑是否把查询点附近的动作差异从量化零值中释放出来。

持久化 smoke `dqc_cq1_sigmoid_cdf_smoke_20260716` 训练 `6.3s`、exit code 0，覆盖 MC、chunked critic、smooth actor、hard/smooth eval 与 JSON。下一条 P-S1 逐项复用 P-B3，只设 `cost_cdf_mode=sigmoid,T=1`；300k 训练预计约 `160s`，128 条终评约 `25～45s`。通过门槛是终评 outage `≤0.22` 且 reward 明显高于 E9 seed0 的 `0.658`；若 outage 仍高于 `0.3` 或 reward 前 150k 已明显崩坏，则不再扩 temperature 网格。

分歧路线继续保留：固定温度 `{0.5,1,2}` 只在 T=1 有正信号时展开；C-Q2 用查询附近 quantile spacing 自适应温度；C-Q3 为 N=64/128；C-Q4 为 uniform τ + 查询点局部加密并做 importance weighting；C-Q5 为 uniform-IQN/query-mixture-IQN。C-H1 独立 cost RNN 与 C-B 大 `num_envs` 仍是正交消融，不和首条平滑 run 同时改变。

### 13.15 P-S1 结果：平滑查询有效，但 T=1 尚未满足约束（2026-07-16）

P-S1 job `DQCAC_DynamicButton_recur_mc_c20_pi_kp1_w50_smoothT1_300k_s0`（W&B `iisf2230`）从 commit `6926fb8` 启动，训练 `161.4s`、exit code 0。它严格复用 P-B3 的 recurrent+MC+C20+PI-window50 配置，只把 actor 查询 CDF 改为 `sigmoid,T=1`。

130 条终评 reward 为 `0.8296`、empirical outage 为 `0.2692`；hard/smooth critic CDF 分别为 `0.3474/0.3509`，predicted mean cost `15.63` 对真实 `11.08`。对比 hard P-B3 的 `reward=1.132,outage=0.315`，T=1 确实沿正确方向降低了约束违反，但尚未达到 `0.2`。对比旧 E9 seed0 的 `0.658/0.200`，它保留了更高 reward，却还不能称为可行且优于基线。

最有诊断价值的证据不是单个终点评分，而是 smooth 后 `risk_adv_nonzero_fraction` 在最后 12 个记录点达到 `0.865～0.999`、多数高于 `0.92`，risk-adv std 也提高到约 `0.008～0.021`。这说明 N=32 hard count 的确曾让大量候选动作 risk difference 精确为零；平滑 CDF 恢复了局部动作排序。最终 hard/smooth s0 CDF 只差 `0.0036`，所以这不是通过篡改 empirical outage 或整体风险标尺得到的假改善。最终 λ 反而从 hard 的 `0.307` 降到 `0.256`，却获得更低 outage，进一步说明每单位 dual penalty 的 actor 信号更有效。

同时也要记录反证与代价：reward 比 hard P-B3 低约 `0.302`，终评 critic 从轻微低估变成了 mean-cost 明显高估；尽管 PPO KL `0.00549`、clip fraction `0.248`、无 NaN/Inf，温度过大仍可能把远离查询点的 quantiles 也纳入梯度，造成不必要的保守性。因此不能只沿“温度越大越好”单一路线外推。

接下来做一条 P-S2 `T=2` 单变量门控：若 outage `≤0.22` 且 reward `>0.658`，再进入多 seed；若 outage 不优于 T=1 或 reward `≤0.658`，停止固定温度网格。另一合理分歧 `T=0.5` 记录为反向消融，用来识别 T=1 是否已经过平滑，但当前不优先消耗全量预算。若 T=2 未通过，保留并分开验证以下路线：

1. C-Q2：温度随查询点附近 quantile spacing 自适应，避免全局固定 cost unit。
2. C-Q3：N=64/128，提高 hard-CDF 原生分辨率；用已有 transition chunking 控制显存。
3. C-Q4：uniform quantiles + 查询点局部加密；训练时对非均匀 τ 采样做 importance weighting，防止改变隐含目标分布。
4. C-Q5a/C-Q5b：uniform-IQN 与 query-mixture-IQN 分开报告；IQN 仍估计 quantile function，CDF 通过反演/采样近似，不能把它描述成直接监督 CDF。
5. C-H1：独立 online/target recurrent cost encoder；若增加 quantile 分辨率后 risk advantage 仍弱，再优先处理 history sufficient-state 问题。

完整对齐结果位于 `_runs/profiles/dqc_smooth_cdf_ps1_300k_2026-07-16/`，原始 W&B 导出位于 `_runs/wandb_export/dqc_smooth_cdf_ps1_300k_2026-07-16/`。所有未选路线保留为轻量验证或论文消融，不因当前主路线选择而删除。

### 13.16 P-S2 结果：T=2 Pareto 改善，但不足以通过约束门（2026-07-16）

P-S2 job `DQCAC_DynamicButton_recur_mc_c20_pi_kp1_w50_smoothT2_300k_s0`（W&B `agmbfhlz`）从 commit `87548a8` 启动，训练 `159.6s`、exit code 0。相对 P-S1 只有 `cost_cdf_temperature=1→2` 一项变化。

终评 reward `0.9725`、outage `0.2385`，同时优于 T=1 的 `0.8296/0.2692`；但仍没有达到预设 `outage≤0.22`，因此不能进入 multi-seed 胜出确认。hard/smooth critic CDF 为 `0.1834/0.2013`，真实 outage 为 `0.2385`；predicted mean cost `8.97` 对真实 `12.12`，说明该策略分布上 critic 又出现低估。最终 `lambda=0.1212`，末期 PPO KL `0.00187`、clip fraction `0.0957`，risk-adv nonzero fraction `0.9808`、std `0.0145`，没有训练崩坏。

必须同时记录统计上的保留意见。当前 `num_eval=128` 在 B=10 下实际评估 130 条，T=2 的 `0.2385` 即 `31/130`，Wilson 95% interval 约为 `[0.173,0.319]`；T=1 的 `35/130` interval 约为 `[0.200,0.351]`，hard P-B3 的 `41/130` interval 约为 `[0.242,0.400]`。T=2 对 hard 的安全改善有较一致方向，但 T=1/T=2 区间高度重叠；不能把单次终评的小数差异解释为温度单调规律。

这也形成两条有分歧、都必须保留的路线：

1. **温度核路线**：T=2 当前是最好单点，但 sigmoid 导数随温度和距离共同变化，温度不是简单的风险 penalty gain。固定网格在 `{1,2}` 后暂停；`T=.5` 与 `T=4` 作为反向/外推消融保留。更有理论依据的 C-Q2 是按查询附近 quantile spacing 自适应温度。
2. **分辨率路线（当前主门）**：C-Q3A 先跑 `N=64,lambda=0,100k`，与 N=32 C20/MC 校准基线只改变 quantile 数；`critic_minibatch_size=2500` 让 `2500×64²` 与旧 `10000×32²` 的 pairwise QR 元素量相当，降低显存混杂。通过门是 CDF bias 至少下降 25%，或 mean-cost relative error 进入 15%，且另一指标不恶化超过 10%。通过后再组合 `N64+T2` 做 300k constrained；失败则转 local-τ/IQN 或 C-H1。
3. **评估工程路线 C-E1**：当前训练入口没有保存 checkpoint，导致 P-S2 结束后不能低成本把评估扩到 512/1024 条。后续增加 opt-in final checkpoint 与 eval-only 恢复；最终候选必须报告更大评估样本和多 seed，而不是只看 130 条。该工程改动独立验证，不与算法变量混写。

统一 300k profile 位于 `_runs/profiles/dqc_smooth_cdf_ps12_300k_2026-07-16/`，同时包含 hard/T1/T2/E9/QCPO_refs；完整 history 导出位于 `_runs/wandb_export/dqc_smooth_cdf_ps12_300k_2026-07-16/`。T=0.5、T=4、自适应 spacing、N=128、uniform+local τ、uniform-IQN/query-mixture-IQN 与 C-H1 均保留为轻量验证或论文消融候选。

### 13.17 C-Q3A：直接增加 N 被 QR loss 尺度混杂（2026-07-16）

C-Q3A-legacy job `DQCAC_DynamicButton_recur_mc_c20_n64_chunk2500_100k_s0`（W&B `wqjcropf`）把 C20/MC 校准基线的 N 从 32 增到 64，并设 `critic_minibatch_size=2500`。后者让 `2500×64²=10000×32²`，因此 pairwise TD tensor 的理论主规模与旧整批路径相当。实测训练 `60.9s`，只比 N32 的 `59.4s` 多约 1.5 秒，说明当前硬件完全 afford N64，计算时间不是拒绝该路线的原因。

原始结果没有通过校准门。70 条终评的真实 reward/outage/mean cost 为 `0.5527/0.2286/8.957`；N64 predicted CDF/mean 为 `0.1223/6.808`，CDF bias `-0.1063`、mean relative error 24.0%。N32 C20 分别为 `0.1308/6.974`、bias `-0.0978`、relative error 22.1%，所以“直接 N64”略差，不应扩到 300k。

不过代码审查发现该负结果包含一个关键优化混杂。当前 quantile Huber loss 为：

`pairwise_loss.sum(dim=target_quantile).mean(dim=prediction_quantile).mean(dim=batch)`

即 target sample 数 N 增大时，loss 和裁剪前梯度线性放大。N64 末点 cost grad norm 达 `17.03`、joint grad clip fraction 为 `1`；此前 N32 C20 的 joint norm 约 `1.37～7.86`，没有触发 clip。Adam 对纯尺度大体不敏感，但 hard grad clipping 会改变更新方向/有效步长，所以这不是一个只改变分布分辨率的公平实验。

代码新增默认兼容的两条路线：

- `legacy_sum`：默认值，保留旧 `sum(target samples)`，手算 loss 与 gradient 逐元素 exact；所有历史 run 可复现。
- `reference_mean`：target sum 后乘 `N_ref/N_target`，默认 `N_ref=32`。N64 的 loss/gradient 正好乘 0.5，N32 仍保持参考尺度；它等价于对 target samples 取平均后再乘固定 32，从而不需要同时重调既有 critic LR。

启动日志、W&B 和最终 JSON 都记录 reduction、scale 与 reference；profile 工具也加入相应键。合成对拍验证 N64 reference loss/gradient 是 legacy 的 0.5。持久化 `dqc_n64_reference_qr_smoke_20260716` 使用 N64+chunk2500+reference，训练 `14.4s`、exit code 0，覆盖真实 rollout、20 critic/8 PPO 更新、评估和 JSON。

下一条 C-Q3B 复跑同形 N64 100k，只把 `legacy_sum→reference_mean`。由于两条 N64 run 的网络形状和随机数消耗相同，真实 reward/cost 应逐点一致；若不一致先按实现问题处理。通过条件除原校准门外，再要求 grad clip 显著消失。通过后才组合 `N64+smooth T2,300k`。

另一个独立分歧必须记录：跨 N32/N64 时，critic 输出层参数数不同，会消耗不同数量的全局 Torch RNG，后续 stochastic action sampling 的随机流可能错位。因此跨结构单 seed 不是严格 paired trajectory。C-RNG1 可在 agent 完成初始化后统一 reseed；C-RNG2 为 policy action、critic 初始化、minibatch 各用独立 Generator。当前 C-Q3A/B 通过同形网络规避该混杂；RNG 解耦保留为工程复现消融，不与 QR scale 同时改。

### 13.18 C-Q3B：修正梯度尺度后，uniform N64 仍未提高查询校准（2026-07-16）

C-Q3B job `DQCAC_DynamicButton_recur_mc_c20_n64_ref32_chunk2500_100k_s0`（W&B `61s441ku`）从 commit `f44bac6` 启动，训练 `59.9s`、exit code 0。它与 N64 legacy 只有 `quantile_target_reduction=reference_mean,ref=32` 一项差异。

配对验证成立：两条 N64 run 每个 reward/empirical cost 点一致，终评都为 reward `0.5527`、outage `0.2286`、mean cost `8.957`。尺度修正把末点 cost grad norm 从 `17.03` 降到 `7.67`，joint grad clip fraction 从 `1` 降到 `0`；CDF 从 `0.1223` 提到 `0.1268`，predicted mean 从 `6.808` 提到 `6.994`。因此 reference normalization 的工程作用明确，后续 N64/128 必须使用它。

然而校准门仍失败。N64-reference 的 CDF bias 为 `0.2286-0.1268=0.1018`，比 N32 C20 的 `0.0978` 略差；mean-cost relative error 约 21.9%，与 N32 的 22.1% 基本相同，远未进入 15%。这说明把 uniform grid 从 32 均匀加到 64 主要增加全局输出分辨率，没有解决预算查询处的有限容量、MC 回归偏差或 history/action 外推误差。停止 `N64+T2,300k`，N128 只保留为论文分辨率消融，不作为当前主线。

对齐 profile 位于 `_runs/profiles/dqc_quantile_resolution_n32_n64_100k_2026-07-16/`，完整 history 位于 `_runs/wandb_export/dqc_quantile_resolution_n32_n64_100k_2026-07-16/`。

下一主线改为 cost-only 查询点局部加密 C-Q4。机会约束 `P(C≥d)=alpha=0.2` 的边界对应 quantile function 约 `τ*=1-alpha=0.8`。C-Q4A 保持 N=32，但从 mixture density 生成非均匀 τ，使约一半采样质量来自 `[0.7,0.9]`；该区间实际约获得 60% 输出头，局部 τ 间距约缩小到 uniform 的 1/3。reward critic 仍使用 uniform τ，避免污染 GAE 之外的 reward 分布诊断。

非均匀 τ 有两个不能混淆的权重问题：

1. **CDF quadrature 必须校正**：不能继续对 local points 做简单 `1/N` count，否则会把人为加密误报为更大 outage。C-Q4 用 mixture density 的 `1/g(τ)` importance weight并归一化，hard/sigmoid CDF、critic dual 与评估共用这一概率口径。
2. **prediction quantile loss 有两条合法路线**：C-Q4A 对非均匀 prediction heads 等权，刻意把 critic 容量/梯度聚焦到 τ≈0.8；C-Q4B 对 prediction heads 也乘 importance weight，近似保持全局 uniform-τ W1 目标，只增加局部分辨率。A 是当前“只需查询点准确”的主假设，B 保留为分布保持消融。
3. **target sample 权重独立处理**：cost target quantiles 代表分布积分，必须做 importance weighting；当前 MC target 每列相同，但实现仍需对 n-step 路径保持正确，不能依赖这个巧合。

若 C-Q4A 的 100k CDF bias 未至少降低 25%，不跑 300k；转 C-Q4B 或 C-H1。C-Q2 adaptive sigmoid bandwidth、N128、uniform-IQN、query-mixture-IQN、P-M1 controller safety setpoint 与 C-E1 checkpoint/eval-only 都继续保留为正交路线。

### 13.19 C-Q4 实现：cost-only query-mixture τ 与无偏 CDF 权重（2026-07-16）

代码已实现默认关闭的 'cost_quantile_grid_mode=uniform|query_mixture'。query-mixture 的默认中心是 'τ*=1-alpha'，当前 alpha=0.2 时为 0.8；half-width=0.1、local fraction=0.5。它对 mixture CDF 的等质量 midpoints 做解析反演，完全 deterministic，不消耗 Torch RNG。N=32 时 [.7,.9] 获得 19 个 prediction heads，uniform grid 只有 7 个；局部代表权重约从 '1/32=0.03125' 降到 '0.01031'。

实现刻意把 reward/cost 两条分开：

- reward critic 和 'self.taus' 继续使用 uniform midpoint，不改变 reward QR 诊断或未来 distributional reward 路径。
- cost critic 使用独立 'cost_taus'。hard/sigmoid CDF、initial calibration、critic-dual、评估、predicted mean/std 都使用同一组归一化 '1/g(τ)' quadrature weights。
- cost target sample 维始终做 importance weighting；MC target 每列相同但 n-step target 不同，因此不能省略。加权 expectation 再乘 N_target，随后沿用 ref/N target-scale，保持既有 critic LR 尺度。
- prediction 维有两条明确消融：C-Q4A 'query_focused' 对局部密集 heads 等权，主动把优化容量放在 τ≈0.8；C-Q4B 'importance' 对 prediction heads 也乘 '1/g'，近似保持全局 uniform-τ W1 objective。

合成验证覆盖了关键不变量：uniform grid/weights 逐元素等于旧 midpoint/1/N；query grid 严格递增、weights 和为 1；N32 局部 head 数为 19。在可解析的 'q_i=τ_i' 例子中，τ=0.8 的加权 tail 为 '0.2165'，而错误的未加权 count 是 '0.34375'，直接说明“不加权局部 quantiles”会严重伪造 outage。默认 uniform 的 legacy QR loss、hard CDF、mean/std 与旧公式逐元素 exact；query loss/gradient 有限。

两条持久化集成验证均完成：'dqc_cost_grid_uniform_regression_smoke_20260716' 训练 '8.6s'、exit code 0；'dqc_cost_grid_query_mixture_smoke_20260716' 使用 N32+reference+query-focused，训练 '13.8s'、exit code 0，日志显示 'local19'，JSON 保存全部 grid 配置。两条都覆盖 recurrent rollout、MC cost target、critic/PPO 更新、hard/smooth 评估和进程回收。

下一条 C-Q4A 是 100k reward-only 校准门：完全复用 N32 C20/MC baseline，只打开 query-mixture/query-focused 与 'reference_mean/ref32'。网络参数形状与 RNG 消耗不变，lambda=0，所以 reward/真实 cost 应逐点一致。通过条件为 CDF bias 从 '0.0978' 降到 '≤0.0734'，或 mean-cost relative error 进入 15%，且另一指标不恶化超过 10%。通过后才组合 'query-mixture+sigmoid T2' 做 300k constrained；若 CDF 接近通过但 mean 明显变差，补 C-Q4B importance-prediction；若全面无效，转 C-H1，不扫 local fraction/window。

### 13.20 C-Q4A/B 结果：局部分辨率增加不是主要瓶颈（2026-07-16）

C-Q4A query-focused（W&B 'tvgqpcip'）训练 '58.4s'、exit code 0。它与 uniform N32 baseline 的 reward、真实 cost、PPO 每个记录点一致，终评 truth 都是 reward '0.5527'、outage '0.2286'、mean cost '8.957'。query grid 的 CDF 为 '0.1386'、predicted mean 为 '6.934'，相对 uniform 的 '0.1308/6.974' 只把 CDF bias 从 '0.0978' 降到约 '0.0900'（改善约 8%），mean 反而略差；未达到预设门。

A 还暴露了 prediction objective 的尺度混杂：cost grad norm '14.04'、joint clip fraction '1'，而 uniform N32 不触发 clip。因此按预案补 C-Q4B importance-prediction（W&B 'fw5276fj'）。B 把 cost grad 降到 '9.07'、clip 降为 '0'；终评 CDF '0.1374'、predicted mean '7.099'，CDF bias约 '0.0912'、mean relative error约 20.7%。优化更干净，但仍没有 25% CDF 改善或 15% mean-error。

统一 profile 位于 '_runs/profiles/dqc_local_quantiles_q4ab_100k_2026-07-16/'，完整 history 位于 '_runs/wandb_export/dqc_local_quantiles_q4ab_100k_2026-07-16/'。三条 run 的 matched late reward 都是 '0.4766'，说明单变量隔离成立，无 NaN/Inf。

阶段结论是：查询点局部 quantile 加密在工程和概率语义上可行，能把局部 CDF 步长约缩小 3 倍，也带来约 7～8% 的校准改善；但它没有修复约 20% 的 mean-cost 低估，说明主要误差不只是 hard CDF 分辨率。停止 local fraction/window/N 小网格，不跑 'local+T2,300k'。该实现保留为论文正消融和未来 IQN query-mixture 的权重基础。

下一核心路线按预案转 C-H1：独立 online/target recurrent cost encoder，让 critic 条件变量包含完整 cost/history，而不是 reward actor 的漂移 feature。仍保留的分歧路线包括：C-Q2 adaptive sigmoid bandwidth（只改 actor 查询核）、C-H0.6 直接 actor (h,c)、P-M1 controller safety setpoint、uniform-IQN/query-mixture-IQN、以及 C-E1 checkpoint/eval-only。C-H1 先做 100k lambda0 校准门，通过才与 T2/PI 组合。

### 13.21 C-H1 实现：独立 cost MLP+LSTM，而不是复用 actor 表示（2026-07-16）

C-Q4 说明局部分辨率只解释小部分误差，因此按预案进入 C-H1。代码新增默认关闭的 cost_history_mode=cost_lstm；raw 与 actor_feature 两条旧路径保持兼容。独立 cost encoder 使用与 QCPO_refs policy 相同的历史输入协议：observation 追加 previous_cost，经过 [512,512] MLP 后再拼 previous_action 与 previous_reward，送入 512 hidden LSTM。输出 feature 与当前 action 一起进入原 action-conditioned quantile head，所以它估计的是 Z_c(history,a)，没有把 DQCAC 偷换成 state-only cost value。

这项实现解决 C-H0.5 的核心混杂：actor_feature 同时被 reward PPO 和 value loss 更新，cost critic 只能追逐一个不断变化、且未必保留 cost 信息的表示；C-H1 的 MLP/LSTM 则只接收 cost QR loss。两者只共享 augmented observation 的 running mean/variance buffer，用于保持数值尺度相同；没有共享可学习权重，cost loss 也不会进入 actor。

训练采用 100-step truncated BPTT。每个 chunk 沿真实 episode 时间顺序编码全部并行环境，hidden/cell 传到下一 chunk 但在边界 detach；chunk QR loss 按 transition 比例加权，和 reward QR 梯度累积后只做一次 joint clip/Adam step。每次 step 后重新编码当前 rollout，constraint RMS、actor risk actual/baseline 与日志都读取同一个当前 encoder 的 detach feature。critic-dual、initial calibration 与 recurrent evaluation 在 s0 显式使用 previous cost/action/reward 全零的独立 feature。

online/target cost encoder 都已建立并做 Polyak 同步，但当前 C-H1 首轮强制 cost_target_mode=mc。原因很直接：MC 有完整 episode 真实 return，不需要 target history；n-step 若要正确 bootstrap，必须为 t+N 重建 target encoder 的历史状态。把这个额外变量同时加入会破坏 C-H1 是否有效的归因。后续 C-H1N 可以作为 recurrent n-step 消融单独实现。

验证结果：

- Python 静态检查与 diff check 通过。
- 合成张量检查得到 feature shape (20,3,32)，cost loss 回传到 encoder 的 gradient L1 为 5.48，排除误 detach。
- 持久化 dqc_ch1_cost_lstm_smoke_20260716 完成 rollout、2 次 critic/TBPTT、2 次 PPO、critic-dual、recurrent eval 与 JSON，训练 6.4s、exit code 0。
- 持久化 dqc_ch05_regression_postch1_20260716 对旧 actor_feature 路径回归，训练 6.3s、exit code 0，说明统一 cost_feature 接口没有破坏 C-H0.5。

下一条 C-H1A 是 lambda=0 的 100k 校准门，保持 N32、C20、MC、hard CDF、actor lr 3e-4 与 raw C20 相同，只改变 cost_history_mode。预计训练约 2～4 分钟，70 条终评约 1 分钟。门槛仍是 CDF bias 从 0.0978 降到不高于 0.0734，或 mean relative error 从 22.1% 进入 15%，且另一指标不恶化超过 10%。失败则不跑 300k；通过后才与 sigmoid T2 和 window50 PI 组合。

合理分歧均保留为可执行消融：独立 RMS（C-H1B）、256 hidden 容量控制（C-H1C）、recurrent n-step target（C-H1N）、actor hidden/cell 直接条件化（C-H0.6）、adaptive sigmoid bandwidth（C-Q2）、uniform-IQN/query-mixture-IQN、PID safety setpoint（P-M1）和 checkpoint/eval-only（C-E1）。双向 LSTM 会利用未来 observation/cost，违反部署时的因果信息集，因此不作为合法提升路线。

### 13.22 C-H1A 结果与新根因：全时刻 QR loss 不等于初始 outage 目标（2026-07-16）

C-H1A job DQCAC_DynamicButton_recur_mc_c20_ch1_costlstm_100k_s0（W&B 8vm989u8）从 commit b7f8a9c 启动，训练 72.6s、exit code 0。固定评估 truth 与 raw C20 完全相同：reward 0.5527、outage 0.2286、mean cost 8.957，说明 lambda=0 的 actor/PPO 轨迹仍然配对，比较可归因于 cost critic。

结果明确失败。独立 recurrent cost critic 的 CDF 为 0.0643、predicted mean 为 5.800；raw C20 是 0.1308/6.974。CDF absolute bias 从 0.0978 扩大到 0.1643，mean relative underestimation 从 22.1% 扩大到 35.3%，所以不扩 300k，也不与 T2/PI 组合。

完整 history 还揭示了更重要的目标错配。最后一次训练记录中，C-H1 的 s0 predicted mean 为 7.208，raw 为 8.240，但 C-H1 的 cost QR loss 只有 14.09，远低于 raw 的 43.52。最后一批完整 episode cost mean 约为 15，而所有时间位置的 MC return-to-go target mean 只有 5.697。原因是每条 T=1000 trajectory 只有一个 s0 样本，却有 999 个后续样本；越接近 episode 尾部，remaining cost 越低。均匀 transition QR objective 主要奖励拟合大量后期低 cost-to-go，并不直接奖励 initial distribution calibration。独立 LSTM 容量更强，反而能以牺牲早期状态为代价把总体 loss 降得更低。

C-H1 还在 70k/100k 出现 cost grad norm 20.7/17.6 和 joint clip；提高 clip 到 30（C-H1A2）、hidden 512降到256（C-H1C）、独立 RMS（C-H1B）仍作为合理欠拟合/容量消融保留。但这些路线不能解释低 global loss 与差 s0 calibration 同时出现，因此当前优先级低于直接修正训练分布。

代码新增默认关闭的 cost_critic_time_weighting=uniform|risk_discount。risk_discount 对第 t 步赋 max(discount^t,floor)，然后用全批均值归一化，使 mean weight=1；这样不隐式改变 critic LR，只改变早期与晚期 transition 的相对质量。默认 uniform 继续走旧 mean 运算，合成测试确认 loss 逐元素 exact。full batch、transition chunk 与 recurrent TBPTT 都使用同一组全批归一化权重，chunk 加权和与 full loss 数值等价。日志/profile 新增 weight min/max、ESS fraction、discount 和 floor。

首条 C-W1 不使用过激的 .95，而用最终 constrained actor 已采用的 discount=.995。T=1000 时归一化权重从 5.033 降到 0.0337，ESS fraction为0.3937，相当于每条 trajectory 约394个等效 transition；既显著强化早期，又保留中后期支持。持久化 dqc_cost_time_weight_smoke_20260716 训练6.4s、exit 0，覆盖真实加权 QR、PPO、评估与 JSON。

C-W1 的100k配置保持 raw、N32、C20、MC、hard CDF、lambda0，只打开 risk_discount/.995 并设置 beta=.995。lambda=0 时 beta 不进入 actor objective，因此真实 reward/cost应逐点等于 raw C20。通过门仍是 CDF bias从0.0978至少下降25%（不高于0.0734），或 mean relative error进入15%，且另一量不恶化超过10%。若失败，不扫 .99/.997 discount 小网格；下一步应采用更直接的 s0/early stratified replay 或 initial-distribution auxiliary loss。C-H1A2/A3、adaptive CDF、IQN 和 controller setpoint继续作为独立消融保留。

C-H1A 对齐图位于 _runs/profiles/dqc_cost_history_ch1_100k_2026-07-16/，原始导出位于 _runs/wandb_export/dqc_cost_history_ch1_100k_2026-07-16/。

### 13.23 C-W1 结果：几乎零额外成本修复 mean cost，CDF bias 下降40%（2026-07-16）

C-W1 job DQCAC_DynamicButton_recur_mc_c20_timew995_100k_s0（W&B cojepq8r）从 commit 0d33c8a 启动，训练58.8s、exit code 0。它保持raw cost critic、N32、C20、MC、hard CDF、lambda=0，只打开risk_discount且discount=.995；beta也设为.995，但lambda=0使risk项系数为0，因此不改变actor objective。

单变量配对成立。C-W1与raw C20每个reward/真实cost/PPO记录点一致，固定70条终评都是reward0.5527、outage0.2286、mean cost8.957。差异只来自cost critic的transition objective。

结果同时通过两条预注册门：

- predicted mean从6.974提高到9.003，对truth8.957的relative error从22.1%降到约0.51%，远低于15%门。
- hard CDF从0.1308提高到0.1701，absolute bias从0.0978降到0.0585，下降约40.2%，超过25%门。
- 最后训练rollout的truth outage为0.30；raw CDF为0.1844，C-W1为0.2094，误差同样从0.1156降到0.0906，说明方向不依赖终评那一批随机样本。

T=1000下实际归一化weight范围是0.0337到5.033，ESS fraction为0.3937；既没有把训练退化为只看10个s0，也显著降低了后期低remaining-cost样本的支配。训练wall time与raw的59.4s基本相同，没有增加网络、环境采样或quantile数量。

这项结果重新排序了根因优先级。N64把全局分辨率翻倍但不改时间监督分布，所以无效；query-local τ只改quantile轴而不改大量后期transition的主导地位，所以只有7～8%收益；独立LSTM容量更强，却在错误的uniform objective下更容易牺牲早期状态以降低总体loss，所以反而变差。当前最重要的算法组件不是更大的critic，而是让critic的训练测度与actor真正使用的risk测度一致。

完整对齐图位于 _runs/profiles/dqc_cost_time_weight_100k_2026-07-16/，原始history位于 _runs/wandb_export/dqc_cost_time_weight_100k_2026-07-16/。

下一条C-W2以目前Pareto最好但尚未达约束门的P-S2为基线：保留sigmoid T=2、window50 PI、beta=.995、sum normalization，只加入cost risk-discount weighting。相对P-S2是一个变量。300k seed0硬门为outage不高于0.22且reward高于E9的0.658，并要求critic calibration不比P-S2恶化；通过后才进入多seed与更大评估。hard-CDF+C-W1、discount .99/.997、direct s0 auxiliary、early stratified replay、C-H1+time weighting都保留为独立消融，不与C-W2同时加入。
