# DQCACBeta · Safety-Gym 调试实验账本

日期：2026-07-16（UTC）  
固定环境：`DynamicButton`  
固定种子：`seed=0`  
目标：先让 DQCACBeta 在无活跃约束惩罚时学会任务奖励，再逐层恢复风险约束；不以“跑满预算”代替机制排查。

## 1. 版本与数据基线

- 排查前源码快照：`40929db`（`chore(safety-gym): checkpoint pre-debug baseline`）。
- 持久化后台启动器：`caf5021`（`chore(safety-gym): add persistent experiment launcher`）。
- 未执行 `git push`；仓库其他目录的既有脏改动未纳入上述提交。
- 完整 W&B 基线下载：`_runs/wandb_export/baseline_dynamicbutton_2026-07-16/`。
- 500 万步 profile：`_runs/profiles/baseline_dynamicbutton_2026-07-16/`。
- 150 万步严格对齐 profile：`_runs/profiles/baseline_dynamicbutton_at_1500k_2026-07-16/`。

完整 500 万步训练后统一评估如下：

| 算法 | eval mean reward | reward std | outage `P(C>=15)` | λ final |
|---|---:|---:|---:|---:|
| DQCACBeta（旧基线） | -0.0811 | 0.4073 | 0.0615 | 0.0000 |
| QCPO_refs | 1.6696 | 0.4942 | 0.1538 | 6.9500 |

150 万步共同预算内，最后 20% 窗口为：

| 算法 | reward late mean | reward slope / 1M steps | reward quantile | outage late mean |
|---|---:|---:|---:|---:|
| DQCACBeta（旧基线） | -0.0616 | -0.0327 | -8.827 | 0.1167 |
| QCPO_refs | 1.4330 | +0.4405 | -13.810 | 0.1967 |

关键判断：旧 DQCACBeta 最终 `λ=0` 且策略奖励仍不增长，所以第一故障域是 reward actor-critic 链路。PID dual、cost CDF 精度或 β 风险权重可能是后续问题，但不能解释这个环境上的首要失败。

## 2. 实验纪律

1. 一次只运行一个训练 job，避免 CPU MuJoCo worker 与 GPU critic 更新互相争抢。
2. 所有训练通过 `launch_background.sh` 启动；使用 `nohup + setsid`，SSH 断开不影响训练。
3. 每个 job 在 `_runs/jobs/<job>/` 保存 PID、完整命令、Git commit、dirty 状态、开始/结束时间和退出码。
4. 每次训练结束后，用 W&B API 的 `scan_history()` 下载全部指标，不以控制台抽样点代替 history。
5. 用 `profile_wandb_metrics.py` 在共同 env-step 预算下统计 early/middle/late、趋势斜率、非有限值并画对齐图。
6. 修改在早期窗口没有形成方向一致的提升时提前停止；只对通过短预算门的版本扩大预算/seed。
7. 一次性数据、PID、日志与图放入 Git 已忽略的 _runs；TMPDIR、W&B cache 和 Matplotlib cache 固定到 /vepfs-mlp2/c20250510/251204033/.tmp/safety_gym_env，不占 20G 根分区。
8. 阶段结束清理失败的空 job、重复导出和不再使用的一次性探针；保留支撑结论的原始 W&B 导出/profile。

## 3. 当前实验

### E1：只扩大固定探索尺度

- 状态：已完成（exit code 0）。
- job：`DQCAC_DynamicButton_dbg_e1_std1_fixed_1500k_s0`。
- W&B run id：`hqf29nyd`。
- PID：`23149`（已退出；训练期间为独立 session）。
- 代码 commit：`caf5021`；算法代码与旧基线一致。
- 唯一算法变量：`init_std: 0.5 -> 1.0`；`learn_std=False` 保持不变。
- 固定项：`B=10`、`T=1000`、`warmup_iters=30`、`updates_per_episode=10`、`seed=0`。
- 预算：150 iterations = 150 万 env steps，为旧全量的 30%。
- 实际训练耗时：644.5 秒（10.7 分钟）；训练后完成 130 条统一评估轨迹。
- 75 万步检查点通过：同预算 reward late mean 和 slope 均优于旧 DQCAC；因此完成 150 万步短预算。
- 判定重点：探索变化是否显著增加 reward advantage / actor weight 的有效尺度，并让 reward 窗口均值与趋势同时上升。

命令：

```bash
./launch_background.sh DQCAC_DynamicButton_dbg_e1_std1_fixed_1500k_s0 -- \
  /vepfs-mlp2/c20250510/251204033/.conda/envs/zprl/bin/python -u run_experiment.py \
  --algo DQCAC --env DynamicButton --seed 0 --device 0 --num_eval 128 \
  --wandb_mode online --tag dbg_e1_std1_fixed_1500k \
  --set num_iterations=150 num_envs=10 log_interval=5 init_std=1.0 \
  wandb_name=DQCAC_DynamicButton_dbg_e1_std1_fixed_1500k_s0 \
  wandb_group=dqcac_debug_dynamicbutton wandb_tags=debug,phase1,reward-gate,std1
```

### E1 最终结论

- 训练耗时：`644.5s`（10.7 分钟），150 iterations 实测约 4.30 秒/iteration；退出码 0。
- W&B 完整导出：`_runs/wandb_export/final_e1_dynamicbutton_2026-07-16/`。
- 同 150 万步 profile：`_runs/profiles/final_e1_dynamicbutton_2026-07-16/`。
- 统一评估：mean reward `-0.0144`，outage `0.0846`，critic CDF `0.0558`，λ final `0.9205`。
- 后 20% 训练窗口：reward mean `-0.0315`、slope `+0.0266/百万步`、λ mean `2.215`、actor weight std `0.224`。

结论：固定 `std=1.0` 在 73 万步 checkpoint 的确把 reward late mean 从旧基线 `0.0238` 提到 `0.1192`，说明探索不足是一个因素；但 90 万步后 critic-dual 把 λ 推高，reward 又回落。E1 只获得“探索有帮助”的证据，没有通过“正常训练”门，也说明 reward 主干实验必须暂时固定 `λ=0`。

### Smoke 验证（候选 GAE/PPO 实现）

2026-07-16 分别对 `distributional`、`gae`、`gae_ppo` 三条路径执行 `B=2,T=32,iters=2` 后台 smoke。三者训练、评估、JSON 保存和退出码均正常；训练分别约 8 秒，无 NaN/Traceback。scalar reward value 在 episodic 模式下与 distributional critic 一样接收 `t/T` step feature；GAE target 每个 rollout 只计算一次并在 value/PPO epochs 间冻结。

### 关键修正：actor 多次更新的策略新鲜度与 importance ratio

- 排查发现旧 DQCACBeta 的 `updates_per_episode=10` 同时表示 critic 和 actor 都在同一 rollout 上更新 10 次；第 2~10 次 actor step 已不再来自当前策略，但旧 `distributional` loss 没有 importance ratio，因此属于未修正的 off-policy rollout reuse。这是 reward 均值/quantile 长期大幅波动的高优先级原因之一。
- 现拆分为 `updates_per_episode`（critic/value epochs，默认 10）与 `actor_updates_per_episode`（actor epochs，非 PPO 默认 1）。`distributional` 和 `gae` 若设置 actor epochs > 1 会直接报错，防止静默重现旧问题。
- `gae_ppo` 在采样时只保存一次行为策略 `old_log_probs=log π_behavior(a|s)`；每次 actor 更新重新前向计算当前 `log π_θ(a|s)`，ratio 始终为 `exp(logπ_current-logπ_behavior)`，分母在整个 rollout 的所有 PPO epochs 中保持固定。不能在每次更新后覆盖 `old_log_probs`，否则 ratio 会被人为重置到 1，失去对 rollout-policy drift 的检测与裁剪。
- reward surrogate 使用 PPO clipped minimum；需要最小化的 cost-risk surrogate 使用 conservative maximum。同步记录 ratio mean/std、clip fraction 和 approximate KL。
- 额外发现 warmup 中旧代码在 optimizer 没有 step 时仍推进 actor/dual LR scheduler，并触发 PyTorch `scheduler.step() before optimizer.step()` 警告。现仅在对应 optimizer 真正更新后推进 scheduler，使学习率时间轴与参数更新时间一致。
- 持久化验证：`smoke_ppo_multiupdate` 令 critic/actor epochs 均为 2，训练 7.9 秒、exit code 0，证明固定行为 log-prob 的多 epoch PPO 路径可执行；`smoke_gae_warmup_scheduler` 令 warmup=1、critic epochs=2、actor epochs=1，训练 8.2 秒、exit code 0，且无 scheduler-before-optimizer 警告，证明单次 on-policy GAE 与调度顺序均生效。

### E2 首次启动中止记录（不计入算法结果）

- job：`DQCAC_DynamicButton_dbg_e2_gae_rewardonly_800k_s0`；W&B run id：`jud8j38v`。
- 代码：`58ace0b`；日志只到 iteration 4 / 约 5 万 env steps，仍处于 30-iteration critic warmup，actor 尚未执行，因此不能用于判断 GAE 效果。
- 人工中止原因：启动后确认该版本仍把同一 rollout 用于 10 次无 ratio 的 GAE actor 更新。为避免在已知不正确的策略新鲜度设置上浪费 80 万步，终止整个后台进程组；这属于开发过程的 aborted run，不与 E1/基线比较。
- 修正后 E2 将使用 critic/value epochs=10、actor epochs=1；E3 才在固定 old log-prob 的 PPO clip 下使用多个 actor epochs。

### E2：纯奖励门——scalar value + GAE（不启用 PPO）

- 状态：已完成（exit code 0）；通过持久化后台运行。
- job：`DQCAC_DynamicButton_dbg_e2b_gae_onpolicy_rewardonly_800k_s0`；PID：`43188`。
- W&B run id：`tayn9ex2`；源码 commit：`b91405b`（未 push）。
- 设置：`reward_actor_mode=gae`、`lambda_max=0`、`init_std=1.0`、`B=10`、`T=1000`、`warmup_iters=30`、critic/value epochs=10、actor epochs=1。
- 实际训练耗时 `343.4s`；80 万步完成后评估 130 条轨迹，总 job 约 6 分钟。
- 60 万步 W&B 门槛：late reward mean `-0.0087`、slope `+0.0616/百万步`，均低于同预算 E1 的 `0.0451`、`+0.1728`。虽在约 73 万步尝试优雅中止，但多进程 rollout 没有及时响应 SIGINT，已接近预算终点，故保留正常完成与终点评估。
- 80 万步最终 profile：late reward mean `0.00635`、slope `+0.0719/百万步`；E1 分别为 `0.1125`、`+0.2397`，QCPO_refs 为 `1.451`、`+1.606`。
- GAE 确实把 reward advantage std 从 E1 的 `0.0184` 放大到 `0.0543`，actor grad norm late mean 为 `0.0113`；value explained variance 中段 `0.174`、后段 `0.060`。信号尺度变大但单次 actor step 没转化为有效策略改进。
- 统一评估：mean reward `-0.00177`、reward std `0.1271`、outage `0`。本配置未通过 reward 门，不扩展预算或 seed。
- 最终导出：`_runs/wandb_export/final_e2b_dynamicbutton_2026-07-16/`；profile：`_runs/profiles/final_e2b_dynamicbutton_2026-07-16/`。

### E3：纯奖励门——GAE + PPO clip 多 actor epochs

- 状态：已完成（exit code 0）；只在 E2 上增加固定行为策略 ratio、PPO clip 与多 actor epochs。
- job：`DQCAC_DynamicButton_dbg_e3_gaeppo8_rewardonly_600k_s0`；W&B run id：`d2rfqck2`；启动 commit：`1970fa5`。
- 设置：`reward_actor_mode=gae_ppo`、actor epochs=8（对齐 QCPO_refs）、critic/value epochs=10、`ppo_ratio_clip=0.1`；`lambda_max=0`、std、warmup、B/T 与 E2 保持一致。
- 实际训练耗时 `272.3s`；评估 70 条轨迹 mean reward `0.2179`、outage `0.0714`。
- 60 万步严格对齐：late reward mean `0.1291`、slope `+0.3058/百万步`，超过 E1 的 `0.0451/+0.1728` 和旧 DQC 的 `0.0943/+0.2828`，但仍远低于 QCPO_refs 的 `1.574/+2.711`。
- PPO late health：ratio std `0.0449`、clip fraction `0.0425`、approx KL `0.00106`、actor grad norm `0.00992`；无 NaN/Inf。reward value explained variance late mean `0.257`、终点 `0.375`。
- 结论：固定 old log-prob 的 importance ratio + PPO clip 是关键改进，能够把更大的 GAE 信号转化为 reward 增长；但 QCPO_refs 的剩余优势还来自无 warmup 的样本效率、obs normalization、可学习 std/策略结构等候选因素。
- 最终导出：`_runs/wandb_export/final_e3_dynamicbutton_2026-07-16/`；profile：`_runs/profiles/final_e3_dynamicbutton_2026-07-16/`。

### E4：相同 PPO 主干，移除 reward-only warmup

- 状态：待启动；不改 loss、网络或学习率，只令 `warmup_iters=0`。
- 设置保持 E3：GAE+PPO、actor epochs=8、critic/value epochs=10、`lambda_max=0`、固定 std=1.0。
- 预算：30 iterations = 30 万 env steps，恰好提供 30 个 actor rollouts，与 E3 在 60 万步时累计的 actor rollout 数相同；评估 64 条轨迹，预计总计约 3 分钟。
- 判据：按 actor-rollout 数与 E3 对齐，并与 QCPO_refs 的前 30 万步比较；若仍有数量级差距，再进入 observation normalization 与 learnable std 的逐项消融。

## 4. 分阶段改进路线

### A. 先修 reward 学习骨架

按成本从低到高执行：

1. E1：固定 `std=1.0`，验证旧实现是否只是探索不足。
2. 若 E1 不通过，加入 scalar reward value critic + GAE；先让 actor 使用低方差 GAE reward advantage，distributional cost critic 暂不承担 reward baseline。
3. 在同一 on-policy rollout 上加入 PPO ratio clip、old log-prob、minibatch/epoch；记录 ratio、clip fraction、approx KL、value explained variance 与 gradient norm。
4. PPO/GAE 通过短预算门后，再考虑 observation normalization、可学习 log-std 和 entropy 调度；这些分别做消融，不一次全部打开。

选择理由：QCPO_refs 的 reward 主干是 value + GAE + PPO，而旧 DQCACBeta 用两个同策略动作样本之差构造 `Q(s,a)-E_a Q(s,a)`。旧日志中 reward advantage raw std 只有约 `0.005~0.012`，容易让 actor 只接收到 critic/动作采样噪声。

### B. reward 正常后修 dual

1. 用完成轨迹的经验 outage 或滑动窗口/EMA 驱动 dual，替代校准不足时的初态 cost-critic CDF。
2. 移植 QCPO_refs 的 PID（至少先做积分项；再判断是否需要 P/D），记录 raw gap、EMA gap、P/I/D 分量与 λ 更新幅度。
3. 策略目标采用 `(J_r - λ J_c)/(1+λ)` 或等价 sum normalization 作为独立消融，避免 λ 大时完全吞没 reward 梯度。
4. 只有经验 outage 与 critic CDF 已校准到可接受误差后，才比较 critic-dual 是否能带来更低方差/更快响应。

### C. 提高查询阈值附近的 distributional critic 精度

候选顺序：

1. **平滑 CDF 查询**：把硬指示 `1{z_i>=b}` 改为带温度的 sigmoid；温度由查询点附近 quantile spacing 或固定小网格选择，同时保留硬 CDF 用作无偏诊断。
2. **均匀骨架 + 局部加密 quantiles**：目标阈值对应 cost CDF 约 `F_C(d)=1-alpha=0.8`，可在 `tau≈0.8` 附近增加点，但保留全域均匀点以维持 Bellman target 覆盖。
3. 非均匀 τ 不能继续简单平均 indicator。CDF 查询必须使用与 τ 网格对应的 quadrature 权重，或训练时采用 uniform/local mixture 并做重要性修正；否则“点更多”会直接改变所估计的分布。
4. budget `b_t` 随状态和历史 cost 改变，局部 CDF 水平不总是 0.8；因此需要记录查询处 crossing τ 的分布，判断固定局部加密是否真的覆盖常见查询区域。
5. **IQN**：若固定网格仍有明显 CDF 校准误差，再用 IQN 对 uniform τ 采样，并在查询时增加 τ 样本、排序后反演/平滑 CDF。IQN估计的是 quantile function，不是天然直接输出 CDF；计算成本与反演误差必须和加密 QR 基线公平比较。

### D. β 与长时域

旧 `beta=0.95` 在 T=1000 时把绝大多数风险梯度质量放在前段。reward、dual、CDF 三条链路稳定后，再比较 `beta∈{0.95,0.99,1.0}`；提前做会把多个故障源混在一起。

## 5. 短预算通过标准

候选版本至少同时满足：

- 在相同步数下，reward 后 20% 窗口均值明显高于旧 DQCACBeta，且全程 slope 为正；
- 改善不是单个迭代尖峰，rolling mean 能保持；
- actor 的 approx KL/clip fraction/梯度范数有限，无 NaN/Inf；
- 约束尚未开启或 λ≈0 时，先不要求 outage 精确贴边，但不能把任务奖励提升错误归因于 PID/CDF；
- 恢复约束后，经验 outage、critic CDF、λ 三者方向一致，CDF calibration error 不持续扩大。

只有通过上述门槛的配置才扩展到 300~500 万步和多个 seed。
