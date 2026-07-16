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
