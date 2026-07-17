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

- 状态：已完成（exit code 0）；不改 loss、网络或学习率，只令 `warmup_iters=0`。
- job：`DQCAC_DynamicButton_dbg_e4_gaeppo8_nowarmup_300k_s0`；W&B run id：`6y5i04cr`；启动 commit：`81af9de`。
- 设置保持 E3：GAE+PPO、actor epochs=8、critic/value epochs=10、`lambda_max=0`、固定 std=1.0。
- 实际训练耗时 `140.3s`；评估 70 条轨迹 mean reward `0.2577`、outage `0.10`。
- 30 万步 profile：late reward mean `0.2146`、slope `+0.9341/百万步`；同预算 QCPO_refs 为 `1.355/+5.431`，有 warmup 的 DQC 变体仍约 `-0.051`。
- actor-rollout 对齐：E4 的前 30 个 actor rollouts late mean `0.2146`，高于 E3 第 31~60 个 rollout 的 `0.1550`；去掉 warmup 不只节省 env steps，也略改善相同更新次数下的结果。
- 终点 PPO：ratio std `0.0738`、clip fraction `0.1653`、approx KL `0.00271`；仍有限且无 NaN/Inf。
- 结论：旧 30-iteration warmup 在 reward 主干上是明确的样本浪费；恢复约束时应只延迟 dual/risk 分支，不应冻结 reward PPO actor。
- 最终导出：`_runs/wandb_export/final_e4_dynamicbutton_2026-07-16/`；profile：`_runs/profiles/final_e4_dynamicbutton_2026-07-16/`。

### E5：在 E4 上只增加逐维 observation normalization

- 状态：已完成（exit code 0）；保持 MLP、固定 std=1.0、GAE/PPO、epochs、学习率和 `lambda_max=0` 不变。
- job：`DQCAC_DynamicButton_dbg_e5_obsnorm_gaeppo8_300k_s0`；W&B run id：`xjc6hrke`；启动 commit：`7dc2bb1`。
- 选择顺序依据：QCPO_refs 与 E4 都从 `σ=1` 开始，30 万步 entropy 仅为 `2.809` vs `2.838`，可学习 std 变化很小；而 QCPO_refs 明确使用 running mean/variance、clip 到 [-10,10]。
- 实现要求：actor、reward value、reward/cost critics 共用同一份逐维统计；rollout 期间统计冻结，rollout 后更新；首次仅刷统计、不更新 actor，避免新旧归一化导致虚假 PPO ratio。
- 实现验证：Chan 合并公式与 QCPO_refs `RunningMeanStdModel` 数值逐项对拍，`mean/var/output` 最大绝对误差均为 `0`；持久化后台端到端烟测正常退出，覆盖首次统计 warmup、GAE、PPO 两轮 actor update 和训练后评估。
- 实际训练耗时 `147.8s`；评估 70 条轨迹 mean reward `1.690`、outage `0.514`。
- 30 万步 profile：late reward mean `1.662`、slope `+6.816/百万步`，显著超过 E4 的 `0.2146/+0.9341`，也超过 QCPO_refs 同预算的 `1.355/+5.431`；reward 学习门通过，不扩跑 reward-only。
- PPO/拟合健康：终点 ratio std `0.0490`、clip fraction `0.0603`、approx KL `0.00121`；reward value explained variance `0.7368`，无 NaN/Inf。
- 约束诊断：训练后段 empirical outage `0.667`，统一评估 truth `0.514`；cost critic CDF `0.413`，低估 `0.101`。观测归一化解决了 reward 停滞，但 reward-only 策略明确违反 `ω=0.2`，下一阶段必须恢复 dual/risk 分支。
- 结论：旧 DQCACBeta 不涨的核心工程原因是高维异尺度 observation 未归一化；配合前面已经验证的 GAE、固定行为策略 old log-prob、importance ratio 和 PPO clip 后，DQCAC reward 主干不弱于 QCPO_refs。
- 最终导出：`_runs/wandb_export/final_e5_dynamicbutton_2026-07-16/`；profile：`_runs/profiles/final_e5_dynamicbutton_2026-07-16/`。

### E6：恢复经验 PID dual 与 sum normalization

- 状态：已完成（exit code 0）；只恢复约束，不改 E5 的 reward 主干、网络、观测归一化或 std。
- job：`DQCAC_DynamicButton_dbg_e6_empiricalpid_sum_norm_300k_s0`；W&B run id：`43ftc79n`；启动 commit：`e6b443c`。
- 必修一致性：同一 rollout 的 cost advantage 必须在 PPO epochs 之前冻结；不能一边更新 cost critic，一边让 8 个 actor epoch 使用不断移动的 risk advantage。
- dual 输入优先使用 rollout/最近窗口的经验 outage 或 `(1-ω)` cost quantile；当前 cost critic CDF 仍有 `-0.101` 偏差，只作为校准指标，不直接全权驱动 λ。
- 采用 QCPO_refs 风格积分 PID 与 `(J_r+λJ_c)/(1+λ)`；分别记录经验 gap、cost quantile gap、积分状态、effective reward/risk coefficient 与 CDF calibration error。
- 实现保持旧 `critic_adam`/无 sum-norm 为默认；实验开关使用最近 100 条轨迹 outage。risk advantage 在首个 actor epoch 计算后缓存，后续 7 个 epoch 不再随 critic 移动。
- 验证：outage/quantile 两种 PID 确定性单测均与手算完全一致；持久化后台烟测 exit 0，强制正误差时 λ 按 `0→0.08→0.16` 增长，并覆盖 observation warmup、两轮 PPO 和 sum normalization；另一次旧 `critic_adam + distributional` 默认路径回归烟测也 exit 0。
- 实际训练耗时 `142.7s`；评估 70 条轨迹 mean reward `1.640`、outage `0.600`，λ 终点 `0.374`。
- 30 万步 profile：late reward `1.499`、slope `+6.450/百万步`、empirical outage `0.550`；reward 仍高于 QCPO_refs 同预算的 `1.355`，但约束没有向 `0.2` 收敛。
- PID 确实生效：late window outage `0.528`、λ `0.287`、effective risk coefficient `0.221`；终点分别为 `0.570/0.374/0.272`。失败不是 λ 不更新，而是 risk gradient 控制力不足。
- PPO/拟合健康：late clip fraction `0.0274`、approx KL `0.00072`、reward value explained variance `0.662`，无 NaN/Inf。
- 约束根因：late raw risk advantage std 仅 `0.0137`；cost critic CDF 评估为 `0.307` 对真实 `0.600`，偏差扩大到 `-0.293`。此外 `β=0.95` 使 90% 累计 risk 权重落在前 45 步，与 T=1000 未折扣整段 cost 不匹配。
- 最终导出：`_runs/wandb_export/final_e6_dynamicbutton_2026-07-16/`；profile：`_runs/profiles/final_e6_dynamicbutton_2026-07-16/`。

### E7：只提高 Abel risk discount 到 β=0.99

- 状态：已完成（exit code 0）；完全复用 E6，只把 `beta=0.95` 改为 `0.99`。
- job：`DQCAC_DynamicButton_dbg_e7_beta099_empiricalpid_300k_s0`；W&B run id：`6xm1k6n3`；启动 commit：`bb25576`。
- 算法依据：β 是无限期 Abel 可和性带来的偏差—方差旋钮，β→1 恢复精确约束梯度；本实验是有限 T=1000，`0.99` 将 90% 累计权重覆盖从前 45 步扩到前 229 步，同时比直接 `β=1` 更保守。
- 实际训练耗时 `147.4s`；评估 reward `1.025`、outage `0.400`、λ `0.375`，cost critic CDF `0.261`，偏差 `-0.139`。
- 30 万步 profile：late reward `1.182`、slope `+5.005/百万步`、outage `0.433`；相比 E6 的 `1.499/+6.450/0.550`，约束明显改善但付出 reward 代价。QCPO_refs 同预算为 `1.355/+5.431/0.517`。
- 数值健康：late PPO clip `0.0582`、reward value explained variance `0.721`，无 NaN/Inf；终点训练批 outage 已到 `0.2`，但独立 70 轨迹评估仍为 `0.4`。
- 结论：β 是有效且此前过度短视的风险—收益旋钮；`0.99` 已优于 `0.95` 的约束控制，但短预算尚未可行。
- 最终导出：`_runs/wandb_export/final_e7_dynamicbutton_2026-07-16/`；profile：`_runs/profiles/final_e7_dynamicbutton_2026-07-16/`。

### E8：有限时域 β=1.0 诊断

- 状态：已完成（exit code 0）；只把 E7 的 `beta=0.99` 改为 `1.0`。
- job：`DQCAC_DynamicButton_dbg_e8_beta100_empiricalpid_300k_s0`；W&B run id：`o23rzgg8`；启动 commit：`0ad877e`。
- 目的：在有限 T=1000 下去掉 Abel 偏差，测试现有 risk advantage 的控制上限；这不是建议无限时域使用 β=1。
- 实际训练耗时 `142.0s`；评估 reward `0.1355`、outage `0.000`、λ `0.113`。
- 30 万步 profile：late reward `0.238`、slope `+1.351/百万步`、outage `0.0167`、λ `0.1375`；与 E7 的 `1.182/+5.005/0.433` 相比，已经跨过风险—收益最优区间并过度保守。
- 控制动态：训练 reward 在 190k 达 `1.185`，随后 outage 先降而 100 条窗口仍使 λ 上升，reward 到 290k 回落至 `0.174`；这是 PID 窗口滞后与 β=1 强风险梯度共同造成的过度校正。
- 数值健康：late PPO clip `0.122`、approx KL `0.00227`，仍无 NaN/Inf；critic CDF `0.062` 对 truth `0`，此时方向转为轻微高估。
- 结论：risk critic/advantage 有能力控制约束，问题不是风险梯度无效；`β=1` 不作为最终配置，工作点位于 `0.99~1.0`。
- 最终导出：`_runs/wandb_export/final_e8_dynamicbutton_2026-07-16/`；profile：`_runs/profiles/final_e8_dynamicbutton_2026-07-16/`。

### E9：β=0.995 中间工作点

- seed 0 状态：已完成（exit code 0）；只把 E7 的 `beta=0.99` 改为 `0.995`，其余保持不变。job 为 `DQCAC_DynamicButton_dbg_e9_beta0995_empiricalpid_300k_s0`，W&B run id 为 `nyypklpy`，启动 commit 为 `f282b48`。
- 覆盖解释：`β=0.995` 的 90% 累计 risk 权重约覆盖前 459 步，介于 E7 的 229 步和 E8 的整段等权之间。
- 实际训练耗时 `143.8s`；独立评估 70 条轨迹得到 reward `0.6576`、outage `0.200`、cost-return 0.8 quantile `14.2`、λ `0.357`，在 seed 0 上恰好达到 `ω=0.2`。
- cost critic 查询 CDF 为 `0.211`，相对经验 truth `0.200` 只偏高 `0.011`；reward value explained variance 为 `0.841`。因此这个工作点的主要不确定性已经不是“查询处 critic 严重失准”，而是短预算与单 seed 的策略/约束方差。
- 终点 PPO ratio std `0.0638`、clip fraction `0.1108`、approx KL `0.00202`，无 NaN/Inf；importance ratio 与 clip 工作在合理量级。
- 30 万步训练最后 20% 窗口 reward `0.9967`、slope `+4.432/百万步`、outage `0.400`、λ `0.294`。训练窗口与独立终评存在明显差异，不能只凭 seed 0 的 `0.200` 宣称稳定可行。
- 完整导出：`_runs/wandb_export/final_e9_dynamicbutton_2026-07-16/`；同 30 万步 profile：`_runs/profiles/final_e9_dynamicbutton_2026-07-16/`。
- seed 1/2 均通过持久化后台正常完成，训练耗时分别 `201.1s/193.4s`；W&B run id 为 `tuo3ehve/uq7crzsu`。并行共享 GPU 使单 run 比 seed 0 慢，但总墙钟约 4 分钟。
- 三 seed 终评（每 seed 70 条、共 210 条）：reward `0.889±0.229`、outage `0.262±0.054`、cost 0.8 quantile `17.2±2.69`、λ 均值 `0.317`；这里 `±` 是跨 seed 样本标准差。seed 0/1/2 outage 分别为 `0.200/0.286/0.300`，E9 尚不能判为多 seed 可行。
- critic CDF 三 seed 为 `0.211/0.244/0.218`，平均 `0.224`；相对各自经验 outage 的偏差为 `+0.011/-0.042/-0.082`，平均低估 `0.0375`。critic 查询精度已进入约 0.04 量级但仍有 seed 相关偏差，暂不足以取代经验 dual。
- 三 seed 的训练后 20% reward 为 `0.997/1.268/1.174`，outage 为 `0.400/0.400/0.417`；同 30 万步 QCPO_refs seed 0 为 `1.355/0.517`。DQCAC 已不再“reward 不涨”，且更早施加风险控制，但当前控制仍有滞后。
- 多 seed 导出：`_runs/wandb_export/final_e9_multiseed_dynamicbutton_2026-07-16/`；profile：`_runs/profiles/final_e9_multiseed_dynamicbutton_2026-07-16/`。
- 决策：不把 E9 直接扩至 500 万步，也不继续盲目细分 β。先验证 QCPO_refs 原式的 cost-quantile PID；由于 E9 查询处 CDF 平均偏差已约 0.04，局部 quantile 加密/IQN 的优先级低于 dual 响应与多 seed 验证。

### E10：QCPO_refs 风格 cost-quantile PID

- 状态：已完成（exit code 0）；job 为 `DQCAC_DynamicButton_dbg_e10_quantilepid_beta0995_300k_s0`，W&B run id 为 `kivy3xs4`。
- 只把 E9 的 `dual_pid_signal=outage` 改为 `cost_quantile`，并令 `pid_cost_scale=10`；其余 β、Ki、窗口、reward/PPO/obs-norm 和预算不变。
- 更新式为 `I←clip(I+0.1·(Q_0.8(C)-15)/10)`，逐式对应 QCPO_refs 的最近 100 条轨迹 quantile PID。它不仅判断是否超限，还利用超限严重度。
- 训练耗时 `142.4s`；独立评估 reward `-0.1749`、outage `0`、mean cost `0.057`、λ 终点 `0.880`。它满足约束但完全丢失任务性能，未通过短预算门。
- 训练后 20% reward `-0.0937`、slope `-0.5105/百万步`、outage `0`、λ `1.088`；数值无 NaN/Inf，失败来自控制过强而非数值崩溃。
- 控制动态给出直接证据：190k 时 reward/outage/λ 为 `0.871/0.20/0.34`；250k 时变为 `0.364/0/1.16`；290k 时为 `-0.379/0/1.01`。约束已经安全后，积分状态仍长期维持高惩罚，形成 windup/滞后。
- critic CDF `0.0549` 对 truth `0`，此时略高估，但它不是 dual 输入；因此不能把 E10 的过保守归咎于 critic CDF。
- 完整导出：`_runs/wandb_export/final_e10_dynamicbutton_2026-07-16/`；profile：`_runs/profiles/final_e10_dynamicbutton_2026-07-16/`。
- 结论：不能把 QCPO_refs 的 quantile-PID 原参数机械移植到 DQCAC。DQC risk advantage 与 β=0.995 的控制增益更强；下一步若继续 PID，应优先使用更小 quantile `Ki`、leaky-I/anti-windup 或基于 outage 的 PI/PID，而不是为 E10 扩预算/seed。

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

### E11：QCPO 主干校准（Q-A0/Q-A1/Q-A2）

- 状态：实现与持久化 smoke 已完成，短预算对比待运行。
- 2026-07-16 定位到两个确定性工程错误：
  1. 旧 `QCPOGPU` 默认把同一 rollout 连续用于 5 次 actor update，但没有 behavior `old_log_prob`、importance ratio 或 PPO clip；第 2～5 次是未修正的 off-policy reuse。
  2. 基类虽已提供 `ObservationNormalizer`，QCPO 训练循环从未调用 `obs_normalizer.update()`；打开开关也只会一直使用初始 mean=0/var=1，等价于没有归一化。
- 修正后的安全默认是 `qcpo_actor_update_mode=on_policy, updates_per_episode=1`；如果 on-policy 模式配置多次更新会直接抛错，避免静默重现旧 bug。
- 新增 `qcpo_actor_update_mode=ppo`：rollout 采样时只保存一次 `old_log_probs=log π_behavior(a|s)`，所有 actor epochs 的分母固定；每个 epoch 只重新计算当前策略分子，并用 `clip=0.1` 的 PPO surrogate。记录 ratio mean/std、clip fraction、approx KL、actor grad norm 与 log-std。
- observation RMS 在预热 rollout 更新；正式 rollout 中，moments 在采样和全部 actor epochs 内保持冻结，本批训练完成后才合并统计供下一次 rollout 使用。因此 on-policy log-prob 输入变换严格一致，PPO 的 ratio 也不会混入本批 RMS 突变。
- 新增可学习 `log_std` 的有限区间保护 `[-5,2]`；默认仍关闭以复现旧基线，Q-A1/Q-A2 显式使用 `init_std=1, learn_std=true`。
- 持久化 smoke：
  - `QCPO_DynamicButton_smoke_onpolicy_fix_s0`：`B=2,T=32,iters=2`，训练 `7.5s`，exit code 0。
  - `QCPO_DynamicButton_smoke_ppo_obsnorm_fix_s0`：在同预算启用 obs RMS、PPO 8 epochs、可学习 σ，训练 `6.9s`，exit code 0。
- 接下来的轻量门控固定 `DynamicButton/seed0/λ=0/B=10/T=1000`：
  - Q-A0：单次 on-policy、无 obs norm、固定 σ=0.5，先跑 100k。
  - Q-A1：单次 on-policy + obs norm + σ=1/learnable，先跑 100k。
  - Q-A2：Q-A1 + 固定-old PPO 8 epochs，先跑 100k。
  - 100k 仅用于排除明显不学习配置；胜者扩到独立 300k run，不从 100k checkpoint 续跑，保证比较协议一致。

#### E11 Q-A0/Q-A1/Q-A2 结果

- 三个 100k run 均为 `B=10,T=1000,seed=0,lambda_max=0`，通过持久化后台完成并 exit code 0：
  - Q-A0 raw observation、固定 sigma=0.5、单次 on-policy：W&B `57t0ijye`，训练 `55.7s`，终评 reward `-0.0204`。
  - Q-A1 obs RMS、sigma=1/learnable、单次 on-policy：W&B `x3fwd2ao`，训练 `56.1s`，终评 reward `0.0275`。
  - Q-A2 Q-A1 + fixed-old PPO 8 epochs：W&B `gq2mukc4`，训练 `57.9s`，终评 reward `0.0209`。
- 共同 100k 后 20%：Q-A0/Q-A1/Q-A2 reward 分别为 `-0.0419/-0.0151/0.0385`，趋势分别为 `-0.400/+0.029/+0.649` 每百万步。完整导出与 profile：
  - `_runs/wandb_export/qcpo_qa012_100k_2026-07-16/`
  - `_runs/profiles/qcpo_qa012_100k_2026-07-16/`
- 只有 Q-A2 形成方向一致的正趋势，因此独立扩到 300k：W&B `6xp94twc`，训练 `146.0s`，130 条终评 reward `0.3496`、outage `0.1692`；lambda 固定为 0，所以 outage 只描述自然策略，不能归因于约束控制。
- 共同 300k 后 20%：Q-A2 reward `0.2297`、趋势 `+1.248/百万步`，相对旧 QCPO 的 `-0.1139/-0.345` 已完成定性修复；但仍明显低于 DQCAC E5 的 `1.662/+6.816` 和 QCPO_refs 的 `1.355/+5.431`。对齐 profile：
  - `_runs/wandb_export/qcpo_a2_vs_key_300k_2026-07-16/`
  - `_runs/profiles/qcpo_a2_vs_key_300k_2026-07-16/`
- 结论：observation RMS、sigma=1/learnable 与正确 PPO reuse 让 QCPO 从“不学习”恢复为稳定上升，但轨迹级 MC reward credit assignment 仍是剩余主瓶颈。

#### Q-B：QCPO-GAE/PPO hybrid 实现

- 新增 `qcpo_reward_mode=mc|gae`。`mc` 保持原始轨迹级 QCPO；`gae` 只将 reward 分支改成 scalar `V_r(s,t)` + 冻结 GAE lambda-return，constraint 仍是轨迹级 `I{C>=d}`，因此明确标为 hybrid。
- `V_r` 与 actor 使用相同 MLP 宽度、共享 observation RMS，并默认追加 `t/T`；GAE advantage/value target 每个 rollout 只计算一次，在 8 个 value/PPO epochs 中保持冻结。
- PPO surrogate 将 reward 和 risk 分开：reward 用 clipped minimum，risk 用 conservative maximum；这修正了“先组合正负权重再 clip”在 lambda>0 时可能不保守的问题。
- 持久化 smoke `QCPO_DynamicButton_smoke_gaeppo_obsnorm_s0`：`B=2,T=32,iters=2`，训练 `7.1s`，exit code 0；覆盖 obs RMS、V、GAE、固定 old-prob PPO 和终评。
- 下一实验：Q-B reward-only 100k；若明显超过 Q-A2 的 `0.0385` 后段均值并保持 PPO/value 数值健康，再扩独立 300k。

### E12：统一 MLP+LSTM 骨干（实现阶段）

- 已在 `utils/model.py` 新增可复用 `RecurrentActorValue`，严格采用 QCPO_refs 输入协议：
  - MLP 输入 `[raw_observation, previous_cost]`；
  - LSTM 输入 `[MLP feature, previous_action, previous_reward]`；
  - 默认 `[512,512] tanh + LSTM512`，并使用 `MLP feature + LSTM output` 残差；
  - 输出 tanh Gaussian mean、可学习 log-std、scalar reward value 和下一 recurrent state。
- augmented observation（含 previous cost）使用独立 Chan RMS 并 clip 到 `[-10,10]`，与 QCPO_refs 一致。
- 为保持 DQCAC 算法语义，通用骨干没有复制 QCPO_refs 的 state-value cost head；DQCAC 后续仍使用 action-conditioned `Z_c(history,a)` critic。网络公平性对齐 policy/reward-V backbone，constraint head 按各算法必要输出区分。
- 数值对拍：将 `QcpoRefModel(constraint=False)` 的同名权重和 RMS buffer 加载到新适配器，在随机 `T=13,B=4` 序列及非零初始 hidden state 上，mu/log_std/reward value/h/c/obs mean/var 的最大绝对误差全部为 `0.0`。
- 当前状态：适配器已验证但尚未接入训练 rollout/BPTT；下一步先接 Q-B，复现 MLP Q-B 的 100k/300k 结果，再复用同一 rollout adapter 接 DQCAC。

#### Q-B 100k/300k 结果

- Q-B 100k：job `QCPO_DynamicButton_qcal_b_gaeppo8_obsnorm_100k_s0`，W&B `pkyr3dub`，训练 `62.3s`，exit code 0。
  - 共同 100k 后 20% reward `0.5574`、趋势 `+6.950/百万步`；Q-A2 仅 `0.0385/+0.649`。
  - 同预算 DQCAC E5 为 `0.4441/+6.291`，QCPO_refs 为 `0.4776/+7.008`；Q-B 已进入相同样本效率量级。
  - 70 条终评 reward `0.5410`、outage `0.2143`；lambda 固定 0。value explained variance `0.703`，PPO KL `0.00274`、clip fraction `0.1597`。
- Q-B 独立 300k：job `QCPO_DynamicButton_qcal_b_gaeppo8_obsnorm_300k_s0`，W&B `ue1ye4kn`，训练 `148.3s`，exit code 0。
  - 后 20% reward `1.317`、趋势 `+5.059/百万步`；与 QCPO_refs 同预算 `1.355/+5.431` 接近，显著优于 Q-A2 `0.230/+1.248`。
  - 130 条终评 reward `1.621`，接近 DQCAC E5 `1.690` 与 QCPO_refs 全量终评 `1.670`；reward 主干可判为“表现正常”。
  - reward-only 的自然 outage 为 `0.546`，不满足约束是预期现象，下一阶段再恢复统一 dual/PID，不能以此否定 reward 校准。
  - value explained variance `0.641`；PPO ratio std `0.0497`、clip fraction `0.0587`、KL `0.00124`；无 NaN/Inf。
- 完整 profile：
  - `_runs/profiles/qcpo_qb_vs_key_100k_2026-07-16/`
  - `_runs/profiles/qcpo_qb_vs_key_300k_2026-07-16/`
- 最重要结论：QCPO 旧失败由两个层次叠加。未修正 rollout reuse 和未生效 obs RMS 是工程 bug；修完后 Q-A2 能学但慢。把 reward MC 换成 V+GAE 后才追平 QCPO_refs，说明低方差、逐时刻 reward credit assignment 是剩余主因。

#### E12 QCPO 接入状态

- `QCPOGPU` 新增 `policy_arch=mlp|mlp_lstm`，默认仍为 `mlp`，旧 Q-B 路径不变。
- recurrent rollout 逐步保存 augmented observation、previous action/reward、采样时 old log-prob、reward value，以及每步进入前 `h0/c0`。
- 更新时按 `recurrent_seq_len=100` 将每条 T=1000 轨迹切块，块维 shuffle 语义与 QCPO_refs 一致；每块使用 rollout 保存的初始 hidden state，8 epochs 中 behavior old log-prob 固定。
- policy PPO 与 reward-value loss 使用同一个 recurrent model、同一次 backward/optimizer step，与 QCPO_refs 的共享骨干优化方式一致。reward/risk PPO 仍分别使用 min/max。
- recurrent 专用评估器在每轮 episode batch 开始重置 previous cost/action/reward 与 hidden state，避免 stateless 统一评估器丢失历史。
- 持久化验证：
  - `QCPO_DynamicButton_smoke_recurrent_qb_s0`：小型 `[32,32]+LSTM32`、seq=16，训练 `7.1s`，exit code 0，覆盖 rollout/BPTT/联合 loss/终评。
  - `QCPO_DynamicButton_smoke_mlp_qb_regression_s0`：默认 MLP 分支训练 `7.0s`，exit code 0，确认 recurrent 分支未破坏已校准 Q-B。
- 下一门：全尺寸 `[512,512]+LSTM512`、seq=100、Q-B reward-only，先 100k seed0；目标是在相同 env steps 下至少复现 MLP Q-B 的明确正趋势，再决定扩 300k。


#### E12 全尺寸 recurrent R0 结果

- QCPO recurrent 100k：job `QCPO_DynamicButton_recur_qb_r0_qbhyper_100k_s0`，W&B `6tfy9aeo`，训练 `63.7s`，exit code 0；后段 reward `0.4006`、slope `+5.006/百万步`，终评 `0.3169`。
- QCPO recurrent 独立 300k：job `QCPO_DynamicButton_recur_qb_r0_qbhyper_300k_s0`，W&B `312u1kr3`，训练 `159.8s`，exit code 0；后段 reward `1.083`、slope `+4.114/百万步`，终评 reward `1.251`、outage `0.557`。
- 同预算 MLP Q-B 后段/终评为 `1.317/1.621`；recurrent 能学且数值健康，但当前 Q-B 超参下尚未复现 MLP，不能把 LSTM 当成无条件增益。
- 对齐 profile：`_runs/profiles/qcpo_recurrent_r0_100k_2026-07-16/`、`_runs/profiles/qcpo_recurrent_r0_300k_2026-07-16/`。
- 决策路线：保留 Q-R0；轻量验证 Q-R1（ref lr/clip）、Q-R2（value loss 系数）、Q-R3（LSTM256）；MLP Q-B 作为 Q-R4 强控制。100k 不超过 R0 则不扩预算。

### E13：DQCACBeta recurrent actor/reward-V 接入

- 状态：D-R0 已实现，默认 `policy_arch=mlp` 不变；循环分支要求 `reward_actor_mode=gae_ppo`。
- 历史输入和网络与 QCPO_refs policy/reward-V 对齐：`raw obs+previous cost` 经 RMS/MLP，拼 `previous action/reward` 后进入 LSTM+skip。
- 行为策略一致性：每个 rollout 固定 old log-prob、GAE/value target、risk advantage 和 chunk 初始 hidden；每个更新后只重算当前策略分子，绝不覆盖 behavior probability。
- critic target 一致性：循环模式的 N-step boot action 来自相同 rollout 的完整历史，而非对中间 state 使用零 hidden；terminal bootstrap 单独生成 `a_T`。
- cost baseline：从每个历史位置的行为高斯参数采 K 个动作，保持 DQC action-conditioned cost CDF advantage；分布 critics 首轮仍为 MLP/Markov 路线，不假装已完成 full recurrent critic。
- RMS：actor 的 augmented RMS 延迟到全部 PPO epoch 后更新；critic 使用独立 raw-state RMS。循环评估器重置完整历史并输出 cost CDF 校准。
- 持久化 smoke `DQCAC_DynamicButton_smoke_recurrent_dr0_s0`：`B=2,T=32,N=8,[32,32]+LSTM32`，训练 `4.7s`、exit code 0；训练与循环评估全链路通过。
- 默认路径回归 smoke `DQCAC_DynamicButton_smoke_mlp_regression_after_recurrent_s0`：相同小预算，训练 `4.5s`、exit code 0，确认循环分支没有破坏 MLP 构造、GAE/PPO、critic 或统一评估。
- 新增 `ppo/first_epoch_ratio_max_error`：首个循环 actor epoch 前 actor/RMS 未改变，理论上 ratio 必须为 1；该指标用于直接发现 history/chunk h0/old probability 接线偏差。
- 并行保留路线：D-R0 actor/V recurrent；D-R1 独立 recurrent cost critic；D-R2 full-shared recurrent hybrid；D-R3 explicit-budget-fair；D-RC 512 MLP 公平控制。
- 下一实验：D-R0/E5 reward-only 100k，预计含评估 `2–3min`；前段不形成正 slope 或 PPO/value/critic 数值异常就停止，不直接跑满。


#### E13 D-R0 100k/300k 结果

- 100k：job `DQCAC_DynamicButton_recur_dr0_e5hyper_100k_s0`，W&B `j57xp70k`，训练 `55.4s`；后段 reward `0.4278/+6.164/百万步`，终评 reward `0.4965`、outage `0.1286`。
- 同 100k E5 MLP 为 `0.4441/+6.291`，QCPO_refs `0.4776/+7.008`，循环 QCPO `0.4006/+5.006`；D-R0 基本复现旧 E5 前段，门控通过。
- 首 epoch `max|ratio-1|≈0.7e-5～1.1e-5`，证明 history/chunk h0/old probability 对齐；100k 终点 value EV `0.594`、KL `0.00338`、clip `0.192`。
- 独立 300k：job `DQCAC_DynamicButton_recur_dr0_e5hyper_300k_s0`，W&B `tavefru8`，训练 `153.1s`；后段 reward `1.381/+5.672`，终评 reward `1.442`。
- 同 300k DQC E5 MLP `1.662`、QCPO_refs `1.355`、MLP Q-B `1.317`、循环 QCPO `1.083`。D-R0 已超过同 recurrent/QCPO 路线，但仍低于最强 MLP DQC；不扩 seed，先做 LR/value coefficient 轻量消融。
- 300k 后段 KL `0.00103`、clip `0.0456`、value EV `0.722`，更像后期更新不足而非过强或 hidden bug。
- 约束警报：cost critic CDF `0.322` 对 truth `0.586`，bias `-0.263`；pred mean cost `12.47` 对真实 `23.56`。恢复 lambda 时必须先用 empirical dual，D-R1 cost-history 与 critic 校准单独验证。
- profiles：`_runs/profiles/dqc_recurrent_dr0_100k_2026-07-16/`、`_runs/profiles/dqc_recurrent_dr0_300k_2026-07-16/`。

### E14：QR transition chunking

- 新开关 `critic_minibatch_size`，默认 0 完全保留整批路径；正数按 transition 分块构造 N² TD-error。
- 每个 chunk loss 按样本比例加权，所有块只执行一次 optimizer step；不是增加 critic update 次数。
- 合成 uneven chunk 对拍 `M=11,N=7,chunk=4`：loss 误差 `4.33e-8`、prediction gradient 最大误差 `0.0`。
- 持久化集成 smoke `DQCAC_DynamicButton_smoke_chunked_qr_s0`：`T·B=64,N=16,chunk=17`，训练 `4.4s`、exit code 0。
- 用途：先解除 B=32 与 N=64/128 的峰值显存限制；性能、wall time 和 peak memory 后续作为独立实验报告。

### E15：D-R0 actor LR / shared value coefficient 轻量消融

- 公平口径：全部 `DynamicButton/seed0/B=10/T=1000/100k/lambda=0`，其余沿用 D-R0 的 LSTM512、GAE+PPO8、N=32。完整 profile：`_runs/profiles/dqc_recurrent_actor_ablation_100k_2026-07-16/`。
- 基线 `lr=2e-4,value_coef=1`：W&B `j57xp70k`，后段 reward `0.4278`、趋势 `+6.164/M`。
- 提高 actor LR 到 `3e-4`：job `DQCAC_DynamicButton_recur_dr0_lr3e4_100k_s0`，W&B `tmckgcax`，训练 `56.7s`；后段 `0.4766/+6.716/M`，终评 reward `0.5527`。后段 KL `0.00310`、clip fraction `0.178`，未出现更新过猛；该路线保留为新的 reward-only recurrent 候选。
- 仅把 shared value coefficient 降到 `0.5`：job `DQCAC_DynamicButton_recur_dr0_vcoef05_100k_s0`，W&B `x37vpl83`，训练 `59.8s`；后段 `0.3755/+5.797/M`，低于基线，故不扩 300k。它保留为负消融，说明当前证据不支持“value 梯度压制 policy”这个解释。
- 同预算 QCPO_refs 为 `0.4776/+7.008/M`，MLP Q-B 为 `0.5574/+6.950/M`。`lr=3e-4` 已追平 refs 前段，但单 seed/100k 不能宣称稳定胜出。
- 100k 的 cost critic 仍严重低估：lr 路线后段 predicted mean cost `1.13`，rollout cost mean `7.40`；终评 critic CDF `0.00045` 对 empirical outage `0.229`。因此停止 reward 小网格，转向 cost target/历史表示诊断。

### E16：cost distribution target 的分歧路线与 MC 实现

- **C-T0（保留基线）**：`cost_target_mode=nstep`，100-step QR-TD + target critic；默认值不变，历史实验可复现。
- **C-T1（当前轻量验证）**：`cost_target_mode=mc`，对完整 episodic rollout 反向计算 `G^c_t=c_t+gamma_c G^c_{t+1}`，用真实 return-to-go 直接监督 N 个 quantiles。只替换 cost target，不改变 reward critic/GAE/PPO。
- **C-T2（候选）**：若 MC 校准明显改善但方差大，做 n-step/MC anchor mixture 或 cost lambda-return；单列为消融，不能与 C-T1 混写。
- **C-H1（候选）**：独立 recurrent cost encoder + action-conditioned quantile head，显式处理 online/target history；用于检验部分可观测性。
- **C-H0.5（轻量候选）**：复用冻结 actor-history feature 接 cost quantile head；工程风险低但表示受 reward actor 漂移影响，必须与独立 encoder 分开报告。
- **C-Q（后续）**：N=64/128、smooth CDF、局部 tau/IQN；只有 mean/return target 已校准后才进入，避免用更多 quantile 掩盖整体传播偏差。
- 实现保护：MC 只允许 `episodic=True`，continuing 截断 rollout 会直接报错；scalar MC sample 重复为 N 列以保持现有 QR target-sample 求和的 loss/梯度尺度，避免同时重调 critic LR。
- 验证：语法与手算 discounted return 通过；持久化 smoke `DQCAC_DynamicButton_smoke_recur_mc_cost_s0` 使用 recurrent+chunked QR，训练 `5.8s`、exit code 0，JSON/评估全链路通过。
- C-T1 100k 已完成：job `DQCAC_DynamicButton_recur_mc_cost_lr3e4_100k_s0`，W&B `wmlzu8la`，训练 `56.9s`、exit code 0。reward 序列与 n-step 对照逐点完全一致，证明 lambda=0 时只改变了 cost 学习。
- 终评 predicted mean cost 从 n-step 的 `1.49` 提到 `4.02`，真实值 `8.96`；CDF 从 `0.00045` 提到 `0.0384`，真实 outage `0.2286`。MC 明确缓解传播低估，但 mean 仍低估 55%、CDF 仍低估 0.190，不扩 300k。profile：`_runs/profiles/dqc_cost_target_mc_100k_2026-07-16/`。
- 新增 C-O1 优化诊断：记录 reward/cost/joint 裁剪前 gradient norm 和 clip indicator；profile 工具同步纳入 cost target/gradient 指标。持久化 recurrent+chunk smoke `DQCAC_DynamicButton_smoke_mc_grad_diag_s0` 训练 `4.6s`、exit code 0。
- 下一门 C-O1：保持 MC 与 actor epochs=8，只把 critic epochs `10→20`。若校准仍无实质改善，再进入 C-H1；最终保留 `nstep/MC × Markov/recurrent-cost` 的 2×2 消融。


### E17：MC cost critic 优化与 bounded leaky-I

- C-O1 `critic epochs=20, lr=1e-3`：job `DQCAC_DynamicButton_recur_mc_cost_c20_lr3e4_100k_s0`，W&B `059boy42`，训练 `59.4s`。终评 predicted mean cost `6.97` 对 truth `8.96`（低估 22%），CDF `0.1308` 对 outage `0.2286`（bias `-0.0978`）；训练末 s0 prediction `8.24` 对当批 cost mean `10.0`。
- C-LR2 `critic epochs=10, lr=2e-3`：job `DQCAC_DynamicButton_recur_mc_cost_clr2e3_lr3e4_100k_s0`，W&B `73yevpqm`，训练 `58.2s`。终评 mean/CDF 只有 `5.91/0.0987`，低于 C-O1；故选择 C20，不用高 LR 替代。
- C20 的 joint grad norm 为 `1.37～7.86`，始终低于 clip=10；cost grad 大于 reward grad，但两 critic 参数独立且没有触发联合裁剪，排除“grad clip 导致追不上”。
- C10/C20/C-LR2 的 reward/PPO 逐点相同；cost 优化没有污染 lambda=0 的 actor 结论。C20 比 C10 只多约 `2.5s` wall time，当前 A100 采样中峰值实测约 `1.3GB`、GPU 利用率低，说明后续 B/N 有充足余量。
- profiles：`_runs/profiles/dqc_cost_optimizer_c20_100k_2026-07-16/`；C-LR2 作为日志/W&B 负消融保留。
- 新增默认兼容的 bounded leaky-I：`pid_integral_leak`、`pid_deadband`、`pid_delta_max`、`pid_reference_episodes`。默认 `1/0/inf/0` 精确复现旧 `lambda<-clip(lambda+Ki*error)`。
- episode scaling 使用几何积分和：常值误差下，一次 B=20 update 与两次 B=10 update 数值相同；手算断言覆盖 legacy exact、episode-scale exact 和 deadband。持久化 smoke `DQCAC_DynamicButton_smoke_leaky_pid_s0` 训练 `6.0s`、exit code 0。
- 首条 constrained 门控采用 C20+MC、recurrent actor、`beta=.995,sum_norm=true,outage PID`，并设 `Ki=.1,leak=.97,deadband=.02,delta_max=.05,reference=10`。这是 P-B1，不覆盖 E9 legacy-I；若 outage 偏高，路线是 leak=.98/Ki=.15；若过保守，路线是 leak=.95 或更大 deadband。


### E18：P-B1 纯 leaky-I 结果与 P-B2 PI

- P-B1：job `DQCAC_DynamicButton_recur_mc_c20_leakpid097_300k_s0`，W&B `9pxv0bmd`，启动 commit `02efba9`，训练 `158.2s`、exit code 0；配置为 C20+MC+recurrent、`beta=.995,sum_norm=true,Ki=.1,leak=.97,deadband=.02,delta_max=.05,reference=10,Kp=0`。
- 控制启动过晚：170k 以前 lambda=0；200k/250k/300k 约为 `0.012/0.056/0.130`，同期 batch outage 多为 `0.3～0.7`。130 条终评 reward `1.332`、outage `0.462`，明确不可行。
- 终评 critic CDF `0.295` 对 truth `0.462`（bias `-0.167`），pred mean cost `12.85` 对 truth `16.92`。C20+MC 在高 cost 策略分布上仍低估，后续 C-H1 必须保留。
- P-B1 证明 leak 解决过保守不等于闭环合格；纯 I + 100 episode 窗口在 300k 内响应太慢。它作为负消融保留，不通过 seed/长预算扩展门。
- P-B2 新增 `pid_Kp`，默认 0 精确保持旧 bounded-I。PI 输出为 `lambda=clip(I_state+Kp*filtered_error)`；P 项随当前窗口 error 立即出现/撤回，leaky-I 只消除稳态误差。
- 手算断言覆盖 Kp=0 legacy、Kp=1 和 lambda 下界；持久化 smoke `DQCAC_DynamicButton_smoke_pi_kp1_s0` 用 `cost_limit=0` 强制正误差，实际得到 `lambda=0.7999`，训练/评估 exit code 0。
- P-B2 保持 P-B1 所有参数，只设 `Kp=1`。按 P-B1 末 error 约 0.23 推算 lambda 会从 I≈0.13 立即升到约 0.34；若仍偏弱，后续路线是 Kp=2 或 window=50，二者不在同一 run 同时改变。


### E19：P-B2 Kp=1 结果与最后一个窗口消融

- P-B2：job `DQCAC_DynamicButton_recur_mc_c20_pi_kp1_300k_s0`，W&B `rgt6qukp`，启动 commit `1dfa3e1`，训练 `158.7s`、exit code 0；与 P-B1 唯一差异是 `pid_Kp=1`。
- P 项真实加速响应：180k/190k lambda `0.011/0.045`，P-B1 为 `0.001/0.005`；到 250k/280k/300k 为 `0.203/0.310/0.356`。I state 最终 `0.146`，P 输出约 `0.210`。
- 但控制到达工作区间太晚：130 条终评 reward `1.303`、outage `0.508`，训练最后 batch outage `0.3`。相对 P-B1 的 outage `0.462` 未改善，单 seed 波动下不能宣称 PI 更差，但明确仍不可行。
- critic CDF `0.365` 对 truth `0.508`（bias `-0.142`），pred mean `14.75` 对 truth `22.33`；较大的 lambda 不能补偿 actor risk advantage 的系统低估/表示误差。
- 只再保留 P-B3：window `100→50`，其余完全不变，让 PI 约早 5 iterations 响应。若仍不可行，停止 Kp/Ki/window 小网格，进入 C-H1 recurrent action-conditioned cost critic；Kp=2 只留作后续消融表选项，不立即运行。


### E20：P-B3 window=50 结果与 PID 网格停止门

- P-B3：job `DQCAC_DynamicButton_recur_mc_c20_pi_kp1_w50_300k_s0`，W&B `o8nx0elk`，启动 commit `45697a5`，训练 `157.6s`、exit code 0；相对 P-B2 唯一变化是 `pid_window_episodes=50`。
- λ 在 150k 已到 `0.088`，并能在 batch outage 回落后降至 170k 的 `0.0095`；最终 `lambda=0.307=I 0.147+P 0.160`。短窗口解决一部分迟滞，但后期仍随经验概率振荡。
- 300k 后段 reward `0.9806/+3.994/M`；130 条终评 reward `1.132`、outage `0.315`。约束明显优于 P-B1/P-B2 的 `0.462/0.508`，但没有达到 `0.2`，不扩 seed。
- 终评 CDF `0.294` 对 truth `0.315`，pred mean `12.35` 对 truth `13.07`。旧 E9 seed0 可达 outage `0.2`，但 reward 仅 `0.658`；P-B3 仍只是未可行的高 reward 候选。
- 对齐 profile：`_runs/profiles/dqc_recurrent_pid_p_b123_300k_2026-07-16/`；原始导出：`_runs/wandb_export/dqc_recurrent_pid_p_b123_300k_2026-07-16/`。
- 决策：停止 Kp/Ki/window 小网格。Kp=2、Ki=.15/leak=.98、cost-quantile PI、双信号控制保留为后续消融选项；先修 cost-history/risk advantage 表示。

### E21：C-H0.5 actor-history feature 实现与分歧路线

- 根因假设：recurrent policy 下未来动作依赖 hidden，正确 cost value 是 `Z_c(s,h,a)`；D-R0 的 `Z_c(s,a)` 对策略诱导过程并非充分 Markov。
- 新开关 `cost_history_mode=raw|actor_feature`，默认 raw。actor_feature 保存 rollout 时真正产生动作的 `MLP+LSTM` feature，detach 后送入 action-conditioned cost quantile head；cost loss 不反传 actor。
- 完整接线：当前/boot/terminal feature、critic target/prediction、risk CDF、K-action baseline、constraint RMS、critic dual、s0 logging 与 recurrent eval。reward critic 与 PPO 路径不变。
- 接口回归：旧四元 actor forward 与 feature 五元接口数值逐元素相同。持久化 smoke `dqc_ch05_actor_feature_smoke_20260716` 与 `dqc_ch05_raw_regression_smoke_20260716` 均约 `6s`、exit code 0。
- C-H0.5A 的混杂因素是 feature 容量更大且随 actor 漂移。若 100k 有效，补 C-H0.5B（MLP-only feature/容量匹配 raw projection）；若无效，进入 C-H1 独立 online/target recurrent cost encoder。raw+history concat（C-H2）和允许 cost 梯度进 actor 的 full-shared（C-H3）只作为后续独立消融。
- 100k 门：C20+MC、actor lr3e-4、lambda0，只改 history mode。相对 raw C20 的 CDF bias `0.0978` 至少降 25%，或 mean-cost 相对误差从 22% 进入 15%，且另一校准量不恶化 >10%；否则不扩 300k。


### E22：C-H0.5A 100k 负结果

- job `DQCAC_DynamicButton_recur_mc_c20_ch05_actorfeature_100k_s0`，W&B `ooy3dlxs`，commit `6780296`，训练 `58.7s`、exit code 0；与 raw C20 唯一差异是 `cost_history_mode=actor_feature`。
- 两条 reward/真实 cost/PPO 序列逐点相同；终评都为 reward `0.5527`、outage `0.2286`、mean cost `8.957`，单变量隔离成立。
- raw 的 CDF/mean 为 `0.1308/6.974`；actor feature 为 `0.1129/6.634`。CDF bias 从 `0.0978` 恶化到 `0.1156`，mean 相对低估从 22.1% 恶化到 25.9%，门控失败，不扩 300k。
- profile：`_runs/profiles/dqc_cost_history_ch05_100k_2026-07-16/`；export：`_runs/wandb_export/dqc_cost_history_ch05_100k_2026-07-16/`。
- C-H1 独立 cost RNN、actor `(h,c)` C-H0.6、MLP-only 容量控制仍保留；H0.5A 无正信号后，先处理证据更直接的 hard-CDF 查询稀疏性。

### E23：C-Q1 sigmoid 查询 CDF 实现与 P-S1 门

- 新开关 `cost_cdf_mode=hard|sigmoid`、`cost_cdf_temperature`，默认 `hard/1`。hard 手算逐元素等于旧 quantile count。
- sigmoid surrogate 为 `(1/N)Σsigmoid((z_i-b)/T)`，只进入 actor actual/baseline risk CDF 与 constraint RMS；QR loss、hard calibration、empirical outage/PID 均不改。
- 新日志：smooth s0 CDF、risk advantage absolute mean/nonzero fraction、mode/temperature。它们用于判断 N=32 hard count 是否让多数 action difference 精确为 0。
- 持久化 smoke `dqc_cq1_sigmoid_cdf_smoke_20260716`：训练 `6.3s`、exit code 0，hard/smooth eval 和 JSON 全链路通过。
- P-S1 只在 P-B3 上打开 `sigmoid,T=1`；预计训练约 160s、总计约 3～3.5min。门：outage `≤0.22` 且 reward `>0.658`；outage>0.3 或前半段 reward 明显崩坏则停止 temperature 网格。
- 备选消融：T=.5/2、自适应 local spacing、N=64/128、local-τ importance weighting、uniform/query-mixture IQN；C-H1 与大 B 不在 P-S1 同时打开。

### E24：P-S1 `sigmoid,T=1` 结果与温度分歧路线

- P-S1：job `DQCAC_DynamicButton_recur_mc_c20_pi_kp1_w50_smoothT1_300k_s0`，W&B `iisf2230`，启动 commit `6926fb8`，训练 `161.4s`、exit code 0；相对 P-B3 唯一变化为 actor risk CDF 从 hard count 改成 `sigmoid,T=1`。
- 130 条终评：reward `0.8296`、empirical outage `0.2692`、hard critic CDF `0.3474`、smooth critic CDF `0.3509`；predicted mean cost `15.63` 对 truth `11.08`，此时 critic 已由 P-B3 的轻微低估转为偏保守高估。
- 与 hard P-B3 的 reward/outage `1.132/0.315` 相比，平滑以约 `0.302` reward 换来 `0.046` outage 改善；与旧 E9 可行边界 `0.658/0.200` 相比，reward 仍高 `0.172`，但约束尚差 `0.069`。因此它是明确正信号，不是最终可行解，也不扩 seed。
- smooth 路线末段 `risk_adv_nonzero_fraction` 为 `0.865～0.999`、多数点高于 `0.92`，risk-adv std 约 `0.008～0.021`；这直接支持“hard N=32 查询导致动作差值稀疏/量化”的诊断。终评 hard/smooth s0 CDF 很接近，说明收益主要来自候选动作之间的局部连续排序，而不是简单把初始状态风险整体抬高。
- 最终 `lambda=0.256`，低于 hard P-B3 的 `0.307`，却取得更低 outage；说明 smooth actor risk advantage 对单位 λ 更有效。末期 PPO KL `0.00549`、clip fraction `0.248`，无 NaN/Inf，不能用策略更新崩坏解释 reward 损失。
- profile：`_runs/profiles/dqc_smooth_cdf_ps1_300k_2026-07-16/`；export：`_runs/wandb_export/dqc_smooth_cdf_ps1_300k_2026-07-16/`。
- **当前单变量路线 P-S2**：只把温度 `1→2`。通过门为 outage `≤0.22` 且 reward `>0.658`；若 outage 不优于 T=1，或 reward 降到 `≤0.658`，立即停止固定温度网格。
- **保留但不混跑的分歧路线**：`T=.5` 用于检验 T=1 是否过度平滑；自适应温度按查询附近 quantile spacing 定标；N=64/128 直接提高 hard-CDF 分辨率；uniform+local τ mixture 用 importance weight 保持目标分布；IQN 分 uniform-IQN 与 query-mixture-IQN 两条。若 T=2 失败，优先在 C-Q2/C-Q3 与 C-H1 中按诊断证据选择，不把多个变化塞进同一 run。

### E25：P-S2 `sigmoid,T=2` 结果、统计不确定性与 C-Q3A 门

- P-S2：job `DQCAC_DynamicButton_recur_mc_c20_pi_kp1_w50_smoothT2_300k_s0`，W&B `agmbfhlz`，启动 commit `87548a8`，训练 `159.6s`、exit code 0；相对 P-S1 唯一变化为 temperature `1→2`。
- 130 条终评 reward `0.9725`、outage `0.2385`、hard/smooth critic CDF `0.1834/0.2013`；pred mean cost `8.97` 对 truth `12.12`，最终 `lambda=0.1212`。它同时优于 T=1 的 `0.8296/0.2692`，但没过预设 `outage≤0.22` 硬门，不扩 multi-seed。
- 训练后段 reward `0.9715`、empirical probability `0.2833`；末点 risk-adv nonzero fraction `0.9808`、std `0.0145`，PPO KL `0.00187`、clip fraction `0.0957`，没有数值或更新异常。
- 统计保留意见：outage 是 `31/130`，Wilson 95% interval 约 `[0.173,0.319]`；T=1 的 `35/130` interval 约 `[0.200,0.351]`。区间高度重叠，故“Pareto 改善”只能作为单 seed 正信号，不能当显著结论。
- 固定温度主网格在 T=2 暂停；`T=.5`（是否过平滑）与 `T=4`（更宽核是否反而稀释局部差异）均记录为温度曲线消融。下一主门 C-Q3A 为 `N=64,lambda=0,100k`，相对 N=32 C20/MC 只改 N，并用 `critic_minibatch_size=2500` 保持 pairwise QR 峰值规模近似相同。
- C-Q3A 通过门：CDF bias 相对 N32 的 `0.0978` 至少下降 25%，或 mean relative error 进入 15%，且另一指标不恶化超过 10%、reward 轨迹正常。通过后才做 `N64+T2,300k`；失败则转 C-Q2 自适应 spacing/local-τ 或 C-H1。
- 工程路线 C-E1：增加 opt-in final checkpoint + eval-only 恢复，候选策略用 512/1024 episodes 复评；当前入口没有 checkpoint，P-S2 无法事后无训练复评。该缺口记录但不与 C-Q3A 同时改算法。
- 对齐 profile：`_runs/profiles/dqc_smooth_cdf_ps12_300k_2026-07-16/`；完整导出：`_runs/wandb_export/dqc_smooth_cdf_ps12_300k_2026-07-16/`。

### E26：C-Q3A N=64 直接扩展失败，定位 QR loss 尺度混杂

- C-Q3A-legacy：job `DQCAC_DynamicButton_recur_mc_c20_n64_chunk2500_100k_s0`，W&B `wqjcropf`，训练 `60.9s`、exit code 0。N=64 并未明显拖慢当前 A100：相对 N32 的 `59.4s` 仅增加约 1.5s。
- 终评 truth 与 N32 都为 reward `0.5527`、outage `0.2286`、mean cost `8.957`；N64 critic CDF/mean 为 `0.1223/6.808`，比 N32 的 `0.1308/6.974` 略差。CDF bias `0.1063`、mean relative error 24.0%，原 C-Q3A 门失败，不扩 300k。
- 根因混杂：QR loss 使用 `.sum(dim=target_quantile)`，所以 N=64 的 loss/梯度约为 N=32 的 2 倍。末点 cost grad norm `17.03`、joint clip fraction `1`；N32 C20 的 joint norm 约 `1.37～7.86` 且不触发 clip。直接扩大 N 同时改变了分辨率和优化尺度，负结果不能归因于 quantile 数。
- 新增 `quantile_target_reduction=legacy_sum|reference_mean` 与 `quantile_loss_reference_samples=32`。默认 legacy_sum 的 loss/gradient 与旧公式逐元素相同；reference_mean 乘 `32/N_target`，使 N64 scale 为 0.5、N32 保持参考尺度。
- 日志/profile 新增 target scale/reference；最终 JSON 保存 reduction 配置。手算对拍验证 legacy loss/gradient exact、N64 reference loss/gradient 精确为 legacy 的 0.5。
- 持久化 smoke `dqc_n64_reference_qr_smoke_20260716`：N64+chunk2500+reference，训练 `14.4s`、exit code 0，rollout/critic/PPO/eval/JSON 全链路通过。
- 下一门 C-Q3B：同一个 N64 100k run 只切 reference_mean。与 C-Q3A-legacy 网络形状相同，随机初始化/采样流应一致；要求真实 reward/cost 逐点相同，且 grad clip 消失、校准达到原门，才组合 N64+T2。
- 公平性分歧 C-RNG1：N32/N64 初始化会消耗不同数量的全局 RNG，跨网络宽度单 seed 的轨迹不保证配对。可选路线是 agent 初始化后统一 reseed 或分离 policy-action/critic RNG；当前先用同形 C-Q3A/B 配对，不把 RNG 修正混入本轮。

### E27：C-Q3B 尺度修正有效，但 uniform N64 仍无校准收益

- C-Q3B：job `DQCAC_DynamicButton_recur_mc_c20_n64_ref32_chunk2500_100k_s0`，W&B `61s441ku`，commit `f44bac6`，训练 `59.9s`、exit code 0。
- 与 N64 legacy 的 reward/empirical cost/终评 truth 逐点相同：reward `0.5527`、outage `0.2286`、mean cost `8.957`；单变量配对成立。
- reference scale 把末点 cost grad norm `17.03→7.67`、clip fraction `1→0`，CDF `0.1223→0.1268`、pred mean `6.808→6.994`，证明优化尺度修正正确。
- 但相对 N32 C20 的 CDF `0.1308`、pred mean `6.974` 没有实质提升；N64-ref 的 CDF bias `0.1018`、mean relative error 21.9%，未达到 25%/15% 门。停止 N64+T2 300k，N128 降为后续分辨率消融。
- profile：`_runs/profiles/dqc_quantile_resolution_n32_n64_100k_2026-07-16/`；export：`_runs/wandb_export/dqc_quantile_resolution_n32_n64_100k_2026-07-16/`。
- 下一主线 C-Q4A：只对 cost critic 使用 query-mixture τ，中心 `1-alpha=0.8`、窗口 `[0.7,0.9]`、local fraction `0.5`；CDF 用 importance/quadrature weight，prediction loss 先用 query-focused 均值。
- 分歧保留：C-Q4B 对 prediction loss 也做 importance weighting 以保持全局 W1；C-Q2 adaptive sigmoid bandwidth；C-H1 独立 cost RNN；N128；uniform-IQN 与 query-mixture-IQN。任何 local grid 都不能用未加权 quantile count 冒充 CDF。

### E28：C-Q4 cost-only query-mixture τ 实现

- 新开关 'cost_quantile_grid_mode=uniform|query_mixture'，默认 uniform；query center 默认 '1-alpha'，另有 half-width/local-fraction。reward critic 始终使用原 uniform τ。
- mixture 用解析 inverse CDF 的 deterministic stratified grid，不消耗随机数；N32、center .8、half-width .1、fraction .5 时 [.7,.9] 有 19 个 heads，uniform 只有 7 个。
- CDF/分布 mean/std/critic dual/评估和 n-step target sample 均使用归一化 '1/g(τ)' weight；prediction loss 独立支持 'query_focused|importance'。
- 合成检验：weights sum=1，min/max '0.01031/0.06186'；在 q_i=τ_i 的可解析例子，τ=.8 上尾加权值 '0.2165'，未加权会错误为 '0.34375'。默认 uniform loss/CDF/mean/std 与旧公式逐元素 exact。
- 持久化回归 smoke 'dqc_cost_grid_uniform_regression_smoke_20260716' 训练 '8.6s'、exit 0；query smoke 'dqc_cost_grid_query_mixture_smoke_20260716' 训练 '13.8s'、exit 0，均覆盖 recurrent rollout、QR/PPO、评估和 JSON。
- C-Q4A 门：N32+C20+MC+lambda0 100k，只打开 query_mixture/query_focused 与 reference/ref32。要求 truth 轨迹与 N32 baseline 一致；CDF bias 从 '0.0978' 降到 '≤0.0734'，或 mean relative error 进入 15%，另一指标不恶化 >10%。
- 若 A 通过，组合 query grid+smooth T2 做 300k；若 A 接近但 mean 明显失真，跑 C-Q4B importance prediction；若 A 全面无效，优先 C-H1，不继续调 local fraction/window 小网格。

### E29：C-Q4A/B 100k 结果，停止 local-grid 小网格

- C-Q4A query-focused：W&B 'tvgqpcip'，训练 '58.4s'、exit 0；终评 CDF '0.1386'、pred mean '6.934'，truth '0.2286/8.957'。CDF bias '0.0900'，只比 uniform '0.0978' 改善约 8%。
- A 的 cost grad norm '14.04'、clip fraction '1'，query-focused 同时放大了优化尺度；因此补预设 C-Q4B，而不把 A 的小改善当 local τ 结论。
- C-Q4B importance-prediction：W&B 'fw5276fj'，终评 CDF '0.1374'、pred mean '7.099'；cost grad '9.07'、clip '0'。CDF bias约 '0.0912'、mean relative error 20.7%，仍未过门。
- 三条 run 的 reward/真实 cost/PPO 逐点相同；local grid 两种 loss 都只有小幅校准收益。停止 fraction/window/N 小网格，不跑 local+T2 300k。
- profile：'_runs/profiles/dqc_local_quantiles_q4ab_100k_2026-07-16/'；export：'_runs/wandb_export/dqc_local_quantiles_q4ab_100k_2026-07-16/'。
- local τ 实现保留为正消融（查询步长更细但校准收益有限）；下一核心路线 C-H1 独立 recurrent cost encoder。C-Q2 adaptive bandwidth、IQN、P-M1 safety setpoint 与 checkpoint/eval-only 仍保留。

### E30：C-H1 独立 recurrent cost encoder 实现与 100k 校准门

- 新增默认关闭的 cost_history_mode=cost_lstm；raw 与 actor_feature 的默认/历史语义不变。C-H1 输入与 QCPO_refs recurrent policy 相同：augmented observation=[state, previous_cost]，MLP feature 再拼 previous_action/previous_reward，经过独立 LSTM。
- 关键区别：它不复用漂移的 actor feature。cost MLP+LSTM 只由 cost quantile regression loss 更新；与 actor 只共享无参数的 augmented-observation running mean/variance，保证输入尺度一致但不共享可学习表示。
- cost quantile head 继续显式接 action，因此模型是 Z_c(history,a)，没有退化成 QCPO_refs 的 state-value cost head。reward critic、GAE、PPO old probability、actor 和 PID 均未改。
- 训练使用 recurrent_seq_len=100 的 truncated BPTT：每个 chunk 包含连续 100 步和全部并行环境；hidden/cell 数值向后传递、chunk 边界 detach。各 chunk loss 按 transition 数占比累积，一次 joint gradient clip 和 optimizer step，避免隐式放大 critic 学习率。
- 每次 encoder optimizer step 后都重算并 detach 当前 rollout 的 cost feature；constraint RMS、risk actual/baseline、actor 与日志由同一版本 feature 查询。s0 critic-dual、训练日志和 recurrent eval 使用 previous cost/action/reward 全零的独立 cost feature，不会误用 actor feature。
- online/target cost encoder 同步创建并 Polyak 更新。首轮 C-H1 刻意限制为 finite-horizon MC target；recurrent n-step target 需要单独构造 t+N history，是另一条算法变量，不能混入本轮。
- 静态检查与独立梯度测试通过：feature shape=(20,3,32)，encoder gradient L1=5.48。持久化全链路 smoke dqc_ch1_cost_lstm_smoke_20260716 训练 6.4s、exit 0；旧 actor_feature 回归 dqc_ch05_regression_postch1_20260716 训练 6.3s、exit 0。
- 100k C-H1A 配方保持 N32+C20+MC+lambda0+hard CDF 与 raw C20 相同，唯一算法变量是 cost_history_mode=cost_lstm。预计训练约 2～4 分钟、70 条评估约 1 分钟。
- 通过门沿用预注册标准：相对 raw C20 的 CDF bias 0.0978 至少下降 25%（即不高于 0.0734），或 predicted mean relative error 从 22.1% 进入 15%，且另一指标不恶化超过 10%、reward/PPO 正常。明显失败就不扩 300k。
- 分歧路线全部保留：C-H1B 为独立 encoder 自有 RMS；C-H1C 为 256 hidden 的容量/速度控制；C-H1N 为 recurrent n-step online/target history；C-H0.6 直接使用 actor (h,c)；C-Q2 adaptive bandwidth；uniform/query-mixture IQN；P-M1 safety setpoint；C-E1 checkpoint/eval-only。双向 LSTM 因使用未来信息违反在线因果性，不列为合法主路线。

### E31：C-H1A 负结果与 C-W1 cost 时间目标对齐

- C-H1A：job DQCAC_DynamicButton_recur_mc_c20_ch1_costlstm_100k_s0，W&B 8vm989u8，commit b7f8a9c，训练 72.6s、exit 0。固定 eval truth 与 raw C20 完全相同：reward 0.5527、outage 0.2286、mean cost 8.957。
- 独立 LSTM 的终评 CDF/mean 只有 0.0643/5.800；raw C20 为 0.1308/6.974。CDF bias 从 0.0978 恶化到 0.1643，mean relative error 从 22.1% 恶化到 35.3%，未过门，不扩 300k。
- profile 显示最后一次训练 s0 prediction 为 7.208，raw 为 8.240；cost QR loss 却从 raw 43.52 降到 14.09。低 loss 与差 s0 calibration 同时出现，说明并非简单欠拟合。
- 目标错配证据：最后一批 episode-level s0 cost mean 约 15，而把全部 return-to-go transition 等权平均后的 cost target mean 只有 5.697。T=1000 时每条 episode 只有 1 个 s0，却有大量后期低 remaining-cost transition；更灵活的 LSTM 能降低全局 loss，但可牺牲真正决定 initial outage/actor risk 的早期状态。
- C-H1 的 cost/joint grad 在 70k 和 100k 分别约 20.7/17.6 并触发 clip=10；提高 clip=30（C-H1A2）与 hidden=256（C-H1C）保留为欠拟合/容量消融，但当前先修证据更直接的 transition objective。
- 完整 profile：_runs/profiles/dqc_cost_history_ch1_100k_2026-07-16/；export：_runs/wandb_export/dqc_cost_history_ch1_100k_2026-07-16/。
- 新增默认兼容的 cost_critic_time_weighting=uniform|risk_discount。risk_discount 使用 max(discount^t,floor)，整批归一化到 mean=1；因此只改 transition 相对质量，不改变 loss 总尺度。默认 uniform 走旧浮点分支，loss 逐元素 exact。
- 加权已覆盖 full/chunk/TBPTT 三路径；chunk 使用全批归一化权重的切片，按 chunk size 聚合，合成测试与 full weighted loss 等价。新增 min/max/ESS、discount/floor 日志和 profile 指标。
- T=1000、discount=.995、floor=0 时归一化权重范围为 0.0337～5.033，ESS fraction=0.3937（每条轨迹约 394 个等效 transition）；比 beta=.95 的极端约 39 个等效 transition 更稳健，也与最终 constrained 配方 beta=.995 对齐。
- 持久化 smoke dqc_cost_time_weight_smoke_20260716 训练 6.4s、exit 0，覆盖加权 QR、PPO、评估和 JSON。
- 下一门 C-W1：raw+C20+MC+lambda0，设置 beta=.995 与 risk_discount/.995；lambda=0 时 beta 不影响 actor，所以真实 reward/cost 应与 raw C20 逐点相同。仍用 CDF bias不高于0.0734或 mean error不高于15%的门；失败则不扫 discount 小网格，转 s0/early stratified replay 或 direct initial-distribution auxiliary loss。

### E32：C-W1 100k 通过，时间目标错配是主瓶颈

- job DQCAC_DynamicButton_recur_mc_c20_timew995_100k_s0，W&B cojepq8r，commit 0d33c8a，训练 58.8s、exit 0。
- 与 raw C20 的 reward、真实 cost/PPO 逐点相同；固定终评 truth 都是 reward 0.5527、outage 0.2286、mean cost 8.957，单变量隔离成立。
- predicted mean cost 从 6.974 提升到 9.003，relative error 从 22.1% 降到 0.51%；hard CDF 从 0.1308 提升到 0.1701，absolute bias 从 0.0978 降到 0.0585，改善约 40.2%。同时通过 mean-error 15% 与 CDF-bias 改善25%两条预注册门。
- 最后训练 rollout truth outage为0.30；raw CDF 0.1844，C-W1为0.2094，误差从0.1156降到0.0906。收益不是终评随机抽样偶然。
- T=1000、discount=.995 的 weight min/max/ESS 与预期一致：0.0337/5.033/0.3937；loss 总尺度维持mean weight=1。训练时间比 raw 59.4s 还少0.6s，差异属噪声，可认为无额外 wall-time成本。
- profile：_runs/profiles/dqc_cost_time_weight_100k_2026-07-16/；export：_runs/wandb_export/dqc_cost_time_weight_100k_2026-07-16/。
- 结论：当前首要瓶颈是均匀 transition QR objective 与 initial-outage/β-risk 的时间分布错配，不是 N=32 分辨率，也不是缺少 LSTM history。C-H1、N64、local τ 的负/弱结果现在得到统一解释：它们没有改变监督质量在时间上的分配。
- 下一门 C-W2：在当前最好 P-S2（T2 sigmoid + window50 PI）上只加入 risk_discount/.995，300k seed0。硬门：outage不高于0.22且reward高于E9的0.658；相对P-S2还要求校准不恶化。预计训练约160s、总计3～4min。
- 保留消融：hard-CDF+C-W1（隔离平滑交互）、discount .99/.997（只做曲线消融，不在当前通过点继续扫）、direct s0 auxiliary、early stratified replay、C-H1+time-weight（检验history在正确目标下是否才有用）。

### E33：C-W2 结果——校准后的 critic 与平滑 CDF 叠加导致过度保守（2026-07-16）

- C-W2 job `DQCAC_DynamicButton_recur_mc_c20_timew995_pi_kp1_w50_smoothT2_300k_s0`（W&B `lkdu70i5`）从 commit `abd86c6` 启动，训练 `164.7s`、exit code 0。相对 P-S2 唯一增加 `cost_critic_time_weighting=risk_discount, discount=.995`；其余 N32、C20、MC、raw cost critic、sigmoid T=2、window50 PI、beta=.995 与 P-S2 相同。
- 130 条终评为 reward `0.4798`、outage `0.1308`、mean cost `9.838`、hard/smooth critic CDF `0.0822/0.0882`、predicted mean cost `4.472`，最终 lambda `0.1273`。相对 P-S2 的 `reward=0.9725, outage=0.2385`，outage 下降 `0.1077`，但 reward 损失 `0.4927`。
- CDF absolute calibration error 为 `|0.0822-0.1308|=0.0486`，略优于 P-S2 的 `|0.1834-0.2385|=0.0551`。因此风险方向并未失效；失败点是 actor 被推到明显过安全的策略区域。按预注册门，虽然 outage 远低于 `0.22`，reward `0.4798<0.658`，故 C-W2 不进入多 seed。
- 后 60k 训练窗口中，P-S2 的 reward/outage/lambda 为 `0.9715/0.2833/0.1730`，C-W2 为 `0.6090/0.1833/0.2235`；C-W2 最后两次 rollout 的 outage 都为 0，reward 为 `0.448/0.520`。这不是终评 130 条造成的偶然偏差，而是训练末策略本身已经保守。
- 两条最终 lambda 很接近（P-S2 `0.1212`，C-W2 `0.1273`），但行为差异很大。直接原因不是 PID 给了更大的 lambda，而是时间加权改善查询区域后，同样单位 lambda 产生更强、更连续的 risk gradient；再与 T=2 平滑 CDF 叠加，风险梯度强度被重复放大。
- 末段 PPO KL 约 `0.00385`、clip fraction `0.187`，无 NaN/Inf；cost grad norm 均值约 `11.34`，joint critic clip fraction 约 `0.667`。梯度裁剪可能限制 critic 跟随末期策略分布，但不能解释策略为何更安全；提高 critic clip 到20（C-WG1）应作为独立优化消融，而不是与控制强度调整同时改变。
- 终评 predicted mean `4.472` 对 truth `9.838` 仍明显低估，说明 on-policy 最后一次训练批的加权拟合不能保证终评状态分布上的全局 mean 校准。由于 hard CDF 误差只有 `0.0486`，当前 actor 查询点仍比全局 mean 更可信；direct s0 auxiliary、early replay 与 checkpoint 后大样本评估继续保留。
- profile：`_runs/profiles/dqc_cost_time_weight_cw2_300k_2026-07-16/`；完整 history：`_runs/wandb_export/dqc_cost_time_weight_cw2_300k_2026-07-16/`。

### E34：C-W3 单变量计划——保留时间加权，只把 smooth CDF 改回 hard CDF

- C-W3 与 C-W2 唯一差异是 `cost_cdf_mode=sigmoid→hard`；温度仍记录为2但 hard 路径不使用。它同时也是相对 hard P-B3 只增加 time weighting 的正交对照。
- 目的：检验 C-W2 的过度保守是否来自“校准增强 + 平滑查询”叠加。理论预期是 hard CDF 降低候选动作风险差的连续强度，使结果落在 P-B3 的 `1.132/0.315` 与 C-W2 的 `0.480/0.131` 之间。
- 仍用 300k seed0 与 130 条终评；预计训练约160秒、含导出分析约5分钟。硬门保持 outage `≤0.22` 且 reward `>0.658`，校准不能明显差于 C-W2/P-S2；失败不扩 seed。
- 若 C-W3 仍过保守，分歧路线依次保留为：actor risk discount `beta .995→.99`；PID 内部 safety setpoint 小于0.2；降低 Kp 或 lambda gain。若 C-W3 不安全，则考虑 time-weighted sigmoid `T=.5/1`。这些都按单变量轻量门测试，不并入同一 run。

### E35：C-W3 结果——critic 已校准，但 hard CDF 的动作风险差仍过稀疏（2026-07-16）

- C-W3 job `DQCAC_DynamicButton_recur_mc_c20_timew995_pi_kp1_w50_hard_300k_s0`（W&B `9wtkz26h`）从 commit `3619c93` 启动，训练 `162.0s`、exit code 0；相对 C-W2 唯一变化为 `sigmoid T2→hard`。
- 130 条终评 reward `0.9481`、outage `0.3385`、mean cost `14.131`、cost quantile `21.0`、critic CDF `0.3168`、predicted mean `13.997`，最终 lambda `0.1000`。reward 通过 `>0.658`，但 outage 明显未通过 `≤0.22`，故不扩 seed。
- 这条 run 的 critic 校准是当前 constrained run 中最好的一组：CDF error `0.0216`，mean relative error约 `0.95%`。因此约束失败不能再主要归因于 critic 低估；即使查询概率准确，hard count 仍不能给候选动作提供足够密集、稳定的局部排序梯度。
- 后60k reward/outage/lambda约为 `0.934/0.286/0.165`；末点 risk-adv std `0.0145`，与平滑路线相当，但 nonzero fraction只有 `0.396`。P-S2/C-W2 的 nonzero fraction分别约 `0.981/1.000`。核心差异是“有多少状态动作收到风险方向”，而不是单纯 risk std 幅值。
- C-W2 与 C-W3 把合理区间夹出：T2 得到 `reward/outage=0.480/0.131`，hard 得到 `0.948/0.338`。两者都数值稳定，说明需要校准 actor 查询核/风险增益，而不是继续堆 critic 容量。C-W3 末段 cost grad clip 较多，但终评校准非常好，因此 C-WG1 提高 critic clip 降为次级优化。
- 五条 hard/T1/T2/time-weight 对齐 profile：`_runs/profiles/dqc_cost_time_weight_cw23_300k_2026-07-16/`；完整 history：`_runs/wandb_export/dqc_cost_time_weight_cw23_300k_2026-07-16/`。

### E36：C-W4 计划——time-weighted sigmoid T=1

- C-W4 与 C-W2 唯一差异是 temperature `2→1`；与 C-W3 相比则只把 hard 查询换成较窄 sigmoid。目标是在 hard 的高 reward/高 outage 与 T2 的低 reward/低 outage 之间寻找可行 Pareto 点。
- 仍用 300k seed0、130 条终评，预计训练约160秒、总计约4分钟。硬门不变：outage `≤0.22` 且 reward `>0.658`；同时记录 CDF calibration、risk-adv nonzero fraction、PPO KL/clip。
- 若 T1 不通过，停止固定 `.5/1/2/hard` 主线网格：若仍不安全，说明需要介于 T1/T2 的自适应/连续增益；若过安全，则说明策略对温度非常敏感，应显式引入 risk gain 或按 local quantile spacing 自适应带宽，而不是用温度碰运气。T=.5/1.5 只保留为论文温度曲线，不作为无止境主线调参。

### E37：C-W4 结果——训练曲线看似通过，但 final PPO 后安全性瞬间失效（2026-07-16）

- C-W4 job `DQCAC_DynamicButton_recur_mc_c20_timew995_pi_kp1_w50_smoothT1_300k_s0`（W&B `5lfdb3hn`）从 commit `03f8ddd` 启动，训练 `157.9s`、exit code 0；相对 C-W2 唯一变化是 temperature `2→1`。
- 训练后60k 的 reward/outage/lambda 为 `0.973/0.143/0.0308`，最后两个 rollout 为 reward `1.220/1.218`、outage `0.10/0.10`，最后 lambda `0`。仅看训练日志，它会同时通过 reward 和 outage 门。
- 但 130 条 final evaluation 为 reward `1.1999`、outage `0.5308`、mean cost `18.315`、cost quantile `29.2`；critic CDF `0.1454`、predicted mean `8.536`。最终策略高回报但严重不安全，CDF bias达到 `-0.385`。
- 代码时序解释了表面矛盾：每条训练日志描述本轮 rollout 的更新前策略；随后 PID 根据 safe window 把 lambda 从 `0.0202→0`，再执行8次 PPO。final evaluation 描述这8次更新后的策略。最后一次 reward-only PPO 的 KL仅 `0.00236`、clip fraction `0.124`，但 Safety-Gym 的风险边界对小策略位移高度敏感，outage可从0.1跃迁到0.53。
- 这不是“多跑130条后发现B=10噪声”可以完全解释：critic predicted mean/CDF 与训练前策略分布一致，却对更新后的 final policy 严重低估；训练数据和最终动作分布已经发生闭环 distribution shift。固定温度主网格停止。
- 六条 CDF/time-weight 对齐 profile：`_runs/profiles/dqc_cost_time_weight_cw234_300k_2026-07-16/`；完整 history：`_runs/wandb_export/dqc_cost_time_weight_cw234_300k_2026-07-16/`。

### E38：C-E1 评估 checkpoint/eval-only 实现与验证

- 新增默认关闭的 `checkpoint_dir` 和 `checkpoint_interval`。DQCAC 在 rollout 完成后、dual/critic/PPO 任何更新前保存 `pre_update_rollout_policy`；统一入口在训练结束后另存 `post_update_final`。phase、iteration、env_steps和rollout reward/outage均写入payload，禁止把两个相位混为一谈。
- checkpoint 保存 agent 直接持有的全部 `nn.Module`（actor、reward/cost critic、targets、obs RMS）、lambda/PID运行时标量和完整结构配置；不保存optimizer/scheduler动量，格式明确命名为 `safety-gym-eval-checkpoint-v1`，用途是严格恢复评估而非声称无损续训。
- 写盘采用同目录 `.tmp` + `os.replace` 原子替换；SSH中断不会把半文件当成有效快照。`--eval_only <path>` 自动从payload恢复algo/env/seed/网络结构，默认关闭W&B，允许用`--set num_envs=...`调整评估并行度。
- 持久化 smoke `dqc_checkpoint_eval_smoke_train_20260716`：2×2×100=400 env steps，训练 `6.1s`、exit 0；生成step200/400两个pre-update和一个post-update final，各326KB。两个独立eval-only job均严格加载、评估、写JSON并回收worker，exit 0。
- step400 pre/post actor SHA256前16位分别为 `5a39d09395733a38` 与 `fa4f4d591309a5b5`，证明phase确实保存不同策略而非重复文件。JSON正确记录`checkpoint_loaded/phase/env_steps`。
- 下一验证 C-E2：复跑 C-W4 且每20k保存一次，先用 rollout 指标筛选 2～3 个快照，再用统一512条评估确认是否存在 reward>0.658 且outage≤0.22 的真实策略。预计训练约160秒；15个快照的空间在 /vepfs 项目盘，不使用20G根目录。若只有pre-update安全而下一次update立即失效，则主修正应是lambda hysteresis/floor或安全setpoint，不是best-checkpoint掩盖训练不稳定。

### E39：C-E2 phase-aligned 快照复评——B=10 rollout 不能替代独立大样本约束判断（2026-07-16）

- C-E2 job `DQCAC_DynamicButton_cw4_timew995_pi_kp1_w50_smoothT1_ckpt20k_300k_s0`（W&B `4dd8zrqf`）从 commit `ad965d7` 启动，训练 `158.3s`、exit 0。它逐点复现 C-W4 的30个 reward/outage/PPO记录，证明保存checkpoint不改变RNG或训练行为。
- 每20k保存一次，共15个pre-update快照和1个post-update final，总计168MB、每个约11MB。便宜门筛出140k/180k/260k/280k/300k五个点，但只对信息量最高的300k和控制最强的180k做独立评估。
- 300k pre-update在训练B=10上是 `reward=1.218,outage=0.10`；140条独立评估却是 `1.219/0.471`，critic CDF/mean为 `0.0790/5.889`，truth outage/mean cost为 `0.471/17.286`。因此即使排除最后PPO相位，10条轨迹仍造成严重选择偏差和critic泛化错觉。
- 180k快照保存时lambda `0.1473`、B=10为 `reward=0.836,outage=0.10`。140条复评为 `0.746/0.264`；扩到520条后为 reward `0.7148`、outage `123/520=0.2365`、Q80 cost `16`、mean cost `10.742`。critic CDF `0.3162`偏保守约0.080，pred mean `12.270`。
- 520条outage的Wilson 95%区间约 `[0.202,0.275]`。它接近真实alpha=0.2且reward超过E9的0.658，但点估计仍高于预注册 `≤0.22` 门，不能宣布可行。其余更弱控制快照不再评估。
- 结论：策略空间中已经出现接近可行、reward约0.715的点，但训练闭环没有稳定停留；checkpoint只揭示问题，不能作为掩盖不稳定的最终算法trick。

### E40：P-M1 PID safety setpoint 实现与正式实验计划

- 新增 `pid_target_prob`，默认None即`target=q_alpha`，旧控制式逐位不变。显式0.15时仅把empirical PID control gap改为`window_prob-0.15`；真实约束gap、critic阈值、最终评估仍严格使用alpha=0.2。
- 日志分离 `dual/prob_gap=window_prob-0.2` 与 `dual/control_prob_gap=window_prob-target`，并记录target/safety margin。这样不会把保守设点误报成论文约束改变。
- 手算测试：同一window outage=0.3、旧pid_i=.1时，默认/显式target=.2都精确输出lambda `.1850000024`；target=.15输出`.2399999946`，真实gap仍0.1、control gap为0.15。语法和diff check通过。
- 持久化 `dqc_pid_target015_smoke_20260716`：400 env steps，训练6.1s、完整评估exit0；启动摘要正确显示`target0.15`。
- P-M1保持C-W4的N32/C20/MC/time-weight.995/sigmoidT1/window50 PI等全部配置，只设`pid_target_prob=.15`，每20k保存快照。预计训练约160秒，总计3～4分钟。
- 通过门：优先看post-update final 130条；若reward>0.658且outage≤0.22，再用520条确认。若接近门则评phase-aligned候选；若明显过保守，下一单变量是target .175或降低Kp；若仍不安全，下一路线是target .10或lambda floor/hysteresis。T1.5、adaptive bandwidth与risk gain保留为actor查询核消融，不与P-M1同时改。

### E41：P-M1 300k 结果——安全设点有效，但控制器在短预算末端仍未稳定（2026-07-16）

- job `DQCAC_DynamicButton_pm1_timew995_pi_target015_smoothT1_ckpt20k_300k_s0`，W&B `tkrudpbq`，从 commit `ce0262f` 启动；训练 `156.6s`、exit code 0，共保存15个pre-update快照和1个post-update final。
- 130条final evaluation为 reward `0.7223±0.4531`、outage `0.2462`、mean cost `9.131`、Q80 cost `17`；critic hard/smooth CDF为 `0.2728/0.2734`，predicted mean cost为 `10.970`，最终lambda为 `0.4038`。
- 与无safety-setpoint的C-W4 final `reward/outage=1.1999/0.5308` 相比，outage下降约`0.2846`，reward仍高于旧E9的`0.6576`。但点估计仍高于预注册`≤0.22`门，因此不能宣称已经可行，也暂不扩多seed。
- 这条run的末段不是稳定平台：lambda从110k开始介入，240k后由`0.231→0.290→0.309→0.349→0.325→0.405→0.404`；最后训练窗口outage为`0.4`，最后一批rollout outage为`0.2`。控制器在300k正处于追赶风险、牺牲reward的阶段，尚未完成闭环收敛。
- PPO末点KL `0.00704`、clip fraction `0.3059`，数值有限但更新幅度偏大；cost grad norm `22.66`且触发clip。没有NaN/Inf或进程错误，所以当前没有“立即停止”的工程证据。
- 结论定级：P-M1 300k是“接近约束且末段仍在改善/调整”，不是明确失败。它应进入1M稳定性验证；若1M仍持续大周期振荡，再单独测试lambda hysteresis/floor、较小actor LR/更少PPO epoch或PID增益，而不是在300k直接换掉算法。

### E42：短跑证据的使用边界与长跑晋级规则（2026-07-16）

- `100k`只用于机制筛选和查错：可以确认梯度尺度错误、目标测度错配、NaN、明显退化，但“没有早期提升”不能写成“长期无效”。N64、local-quantile和cost-LSTM当前结论严格限定为对应seed和100k预算下无早期收益。
- `300k`用于闭环初筛：若末段指标已稳定且明显劣于基线，可以停止；若lambda、outage或reward仍呈系统性改善/补偿趋势，则必须晋级。P-M1属于后者。
- `1M`用于判断controller与actor-critic能否形成稳定平台；至少在300k、600k、1M做phase-aligned checkpoint复评。只有灾难性数值错误，或多个连续窗口稳定落在无价值的支配区域，才提前停止。
- 最终论文结论不得依赖单seed短跑。候选配置需要至少3个seed、每个至少1.5M；与QCPO_refs做完全公平最终比较时应使用相同环境步数，优先补到其`5M`预算，并报告均值、方差和约束置信区间。
- 随机初始化确实可能制造短期false negative，特别是LSTM、更大的quantile critic和PID闭环；但它不能推翻确定性的代码/目标证据，例如N64旧实现梯度随N翻倍、uniform-transition目标不对齐initial outage、rollout与post-update final相位错位。这三类问题无需靠长跑“等它自己好”。
- 下一实验P-M1-L1保持P-M1全部参数不变，只把训练从300k延长到1M并每50k保存checkpoint。按当前吞吐估计纯训练约`8.7min`，加130条终评约`10min`；全程由`launch_background.sh`持久化运行。


### E43：P-M1-L1 延长到1M——600k曾改善，但1M回退，确认闭环极限环（2026-07-16）

- job `DQCAC_DynamicButton_pm1_timew995_pi_target015_smoothT1_ckpt50k_1m_s0`，W&B `p4hbaqtv`，启动commit `ad1c042`（算法代码仍为`ce0262f`）；训练`508.4s`、exit code 0。100个iteration完整记录，无NaN/Inf；每50k保存，共20个pre-update加1个final，约221MB，均位于/vepfs项目盘。
- 1M post-update final 130条为 reward `0.6279±0.7048`、outage `0.2462`、mean cost `11.592`、Q80 cost `16.2`；critic hard/smooth CDF `0.2839/0.2851`，predicted mean `14.284`，lambda `0.4181`。相对300k final的`0.7223/0.2462`，outage没有改善、reward反而下降。
- 分段训练均值揭示非单调动态：0–300k reward/outage/window/lambda为`0.580/0.217/0.191/0.105`；300–600k为`0.702/0.197/0.213/0.213`；600k–1M变成`0.750/0.303/0.298/0.417`。后200k raw/window outage均约`0.305/0.300`，lambda均值已到`0.455`，仍未把策略稳定拉回约束内。
- phase-aligned 520条独立评估为：
  - 300k：reward `0.7764±0.4346`，outage `129/520=0.2481`，Wilson 95% `[0.2129,0.2870]`；critic CDF `0.3214`，bias `+0.0733`。
  - 600k：reward `0.8143±0.6583`，outage `117/520=0.2250`，Wilson 95% `[0.1912,0.2628]`；critic CDF `0.1053`，bias `-0.1197`。
  - 1M：reward `0.6325±0.6478`，outage `127/520=0.2442`，Wilson 95% `[0.2093,0.2829]`；critic CDF `0.2801`，bias `+0.0359`。
- 600k确实优于300k，证明“短跑会漏掉后期改善”不是空想；但1M又回退，且三个点的critic bias在正/负之间大幅翻转。它不是单调慢收敛，而是critic分布漂移、B=10窗口噪声、PID延迟和PPO策略步长耦合形成的极限环。
- 不能直接把raw risk-advantage与reward advantage比较：代码实际会除以`constraint_sigma_ema`。按真实归一化后，risk/reward有效std比值从0–300k的约`0.60`升到600k–1M的`1.47`，末200k约`1.65`；所以后期不安全不是“risk项绝对太小”。更可能是动作风险排序噪声/critic漂移使方向不够可靠，增大lambda只会放大这个非平稳信号。
- 完整history：`_runs/wandb_export/dqc_pm1_1m_2026-07-16/`；profile与图：`_runs/profiles/dqc_pm1_1m_2026-07-16/`；三个520条JSON均保留在`_runs/`。

### E44：P-M2 预注册——actor LR与QCPO_refs对齐到1e-4（2026-07-16）

- P-M2相对P-M1-L1只把`theta_lr0=3e-4→1e-4`。QCPO_refs同样使用8 PPO epochs、clip=.1，但optimizer LR是`1e-4`；当前DQCAC末200k KL/clip均值已到约`0.00519/0.241`，终点为`0.00661/0.323`，对风险边界来说更新偏激进。
- 因更小LR早期必然更慢，本实验不再用300k否决，直接跑1M、每50k保存phase checkpoint；预计训练约8.5–9分钟，final 130条约1分钟。
- 预注册判断：final 130条只做便宜screen；若reward `>0.658`且outage `≤0.27`，或600k–1M末段呈持续安全改善，则扩520条。正式通过仍要求520条点估计outage `≤0.22`且reward `>0.658`，并要求300k/600k/1M不存在P-M1那样的大幅回退。
- 若LR降低能明显压低KL/clip却仍有risk周期，下一正交变量是`num_envs=20`（降低outage/PID和critic batch方差）；若KL稳定但action risk方向仍噪声，测试`num_action_samples=16`；若单次更新偶发越界，新增默认关闭的PPO target-KL early stop。三条路线不混入P-M2。
- 当前仍只做seed0配置选择。随机初始化false negative的可能性保留；任何最终候选必须在3个seed复现，不能用P-M1-L1 seed0否定整个算法族。


### E45：P-M2 1M结果——匹配QCPO_refs的1e-4 LR减小更新，但不改善outage（2026-07-16）

- job `DQCAC_DynamicButton_pm2_timew995_pi_target015_smoothT1_lr1e4_ckpt50k_1m_s0`，W&B `3qqyab3s`，训练`509.2s`、exit code 0；相对P-M1-L1唯一变化是actor LR `3e-4→1e-4`。
- 130条final为 reward `0.6909±0.6167`、outage `0.2462`、critic CDF `0.2200`、pred mean `9.609`、lambda `0.3818`。520条确认评估为 reward `0.6498`、outage `127/520=0.2442`、Q80 cost `16`、critic CDF `0.2166`，bias `-0.0277`。同时未通过reward `>0.658`和outage `≤0.22`两条门，不扩phase快照或多seed。
- 小LR确实压低部分PPO更新：终点KL/clip从P-M1的`0.00661/0.3227`降到`0.00598/0.2221`。但末200k均值只从`0.00519/0.2414`降到`0.00485/0.2219`，幅度有限。
- 代价明显：P-M2末200k训练reward均值`0.3868`，P-M1为`0.7522`；600k–1M raw outage从P-M1的`0.3025`降到`0.2525`，但final大样本outage仍同为`0.2442`。P-M2在830–900k同样出现outage上冲、lambda上升和reward短暂变负，只是周期相位改变。
- 结论：DQCAC的actor LR不能仅因网络/PPO结构相同就机械迁移QCPO_refs参数。较小LR缓和更新并降低训练窗口outage，却没有消除risk signal/dual闭环的非平稳性，而且损害样本效率；它不作为当前推荐配置。
- 对齐导出与图：`_runs/wandb_export/dqc_pm1_pm2_lr_1m_2026-07-16/`、`_runs/profiles/dqc_pm1_pm2_lr_1m_2026-07-16/`。

### E46：P-M3 预注册——固定1M环境步，将num_envs从10增至20（2026-07-16）

- P-M3以P-M1为基线，只改`num_envs=10→20`，并将`num_iterations=100→50`保持总环境步严格等于1M。actor LR恢复`3e-4`，其余PID、critic、CDF、网络和8 PPO epochs不变。
- 每次rollout从10条增到20条，outage=.2时超限轨迹期望数从2增到4；critic每轮看到双倍不同初始状态。PID已有`pid_reference_episodes=10`，leaky-I在常值error下一次B20更新与两次B10严格等价，避免仅因batch变大暗改积分时间尺度。
- 固定env-step比较下，50次policy update各使用20k transitions，P-M1为100次各10k；每个样本仍复用8个PPO epoch。该实验同时检验更低batch方差与更低policy/control更新频率是否能破坏极限环，属于num_envs本身的算法效应。
- 当前128 CPU/A100 80GB资源对B20+N32安全；pairwise QR显存风险主要出现在B与N同时放大。本run每100k保存checkpoint，预计训练约8–10分钟，130条终评约1分钟。
- screen与P-M2相同：final 130若reward `>0.658`且outage `≤0.27`则扩520；正式门为520条outage `≤0.22`且reward `>0.658`，并检查中后期是否比B10少一次完整风险周期。
- 若B20失败，下一单变量为`num_action_samples=4→16`，直接降低冻结risk baseline在8个PPO epoch中被重复放大的Monte-Carlo噪声；随后才实现target-KL early stop。B20、K16和target-KL分别报告，不组合成无法归因的“trick包”。


### E47：P-M3 B20 通过seed0长预算大样本门（2026-07-16）

- job `DQCAC_DynamicButton_pm3_timew995_pi_target015_smoothT1_b20_ckpt100k_1m_s0`，W&B `ky7voxle`，训练`390.6s`、exit code 0；相对P-M1仅把B从10增至20、iteration从100减至50，总环境步仍严格为1M。
- 训练吞吐从P-M1的约`1,966 env steps/s`提高到约`2,560 env steps/s`，steps/s提升约30.2%，同预算训练时间下降约23.3%。B20+N32在A100 80GB上无OOM/NaN，硬件可承受。
- 末200k训练均值为 reward `0.7108`、raw/window outage `0.215/0.212`、lambda `0.2804`、KL/clip `0.00286/0.1557`。P-M1对应为`0.7522/0.305/0.300/0.4546/0.00519/0.2414`。B20只损失约0.041 reward，却把训练outage降低0.09并明显压低PPO位移。
- built-in 140条final为 reward `0.7725±0.5300`、outage `32/140=0.2286`、critic CDF `0.1130`。按预注册扩到520条后，final为 reward `0.7180±0.5201`、outage `87/520=0.1673`、Q80 cost `13`、mean cost `7.940`。
- 520条outage Wilson 95%区间为`[0.1377,0.2018]`：点估计明显低于0.2，上界仅比0.2高0.0018。它同时通过reward `>0.658`与outage `≤0.22`，是当前首个1M、520条大样本下通过的DQCAC配置。
- 不能只报告更好看的520条而隐藏140条的0.2286。两者差异说明有限样本和并行评估seed集合仍能造成约0.06波动；因此seed0通过只是晋级依据，不是论文最终结论。
- critic仍低估：520条hard/smooth CDF `0.1058/0.1086`，相对truth `0.1673`偏低`0.0615`，pred mean `6.241`也低于truth `7.940`。B20主要改善batch/control方差和更新频率，并未彻底解决critic跨批泛化。
- 三条1M对齐导出与图：`_runs/wandb_export/dqc_pm123_1m_2026-07-16/`、`_runs/profiles/dqc_pm123_1m_2026-07-16/`。

### E48：P-M3 seed1/2多seed复现计划（2026-07-16）

- seed1/2完全复用P-M3 B20配置，各1M环境步、每100k checkpoint；不改target、LR、K、critic或CDF。两条持久化后台并行运行，CPU worker总数40、A100显存仍有充分余量。
- 单条独占训练实测约6.5分钟；按此前双run共享GPU经验，并行后预计两条约9–12分钟完成，含各自140条built-in评估约11–14分钟。并行会影响wall time，不影响固定env-step算法比较；每条单独记录耗时。
- 140条只作screen。每个新seed若reward `>0.658`且outage `≤0.27`，再串行做520条final确认；无论成败都报告，不以best seed替代三seed统计。
- 最终汇总使用seed0/1/2每seed相同评估协议，报告reward跨seed均值/标准差、总outage计数、每seedWilson区间和三seed聚合区间。若至少2/3 seed通过且聚合outage≤0.2，B20进入主推荐；否则它只作为有益但不足的组件。
- K16、PID uncertainty buffer/window和recent-rollout critic replay全部暂停到多seed结果后，避免在尚未确认B20泛化前继续叠加变量。

### E49：P-M3 多seed结果——B20确定有益，但尚不足以作为稳定主配置（2026-07-16）

- seed1/2 的1M训练均正常结束，耗时分别为 `713.4s/717.9s`，W&B run 为 `lcllgvhx/fwsc50vq`。两条run并行共享GPU，因此单条wall time高于seed0独占时的`390.6s`，但固定的1M环境步数和算法配置不变。
- 内置140条final评估的seed0/1/2 reward-outage为 `0.7725/0.2286`、`0.8404/0.2429`、`0.8451/0.1786`。三条都通过了宽松screen，所以按预注册规则一律扩到同协议520条，没有丢弃不好看的seed。
- 520条结果为：seed0 reward `0.7180±0.5201`、outage `87/520=0.1673`、Wilson 95% `[0.1377,0.2018]`；seed1 reward `0.8622±0.5809`、outage `159/520=0.3058`、区间 `[0.2677,0.3467]`；seed2 reward `0.8670±0.4263`、outage `110/520=0.2115`、区间 `[0.1786,0.2487]`。
- 跨seed reward为 `0.8157±0.0847`（seed间sample std），outage为 `0.2282±0.0707`。合并事件计数为 `356/1560=0.2282`，事件级Wilson 95%区间 `[0.2081,0.2497]`。合并区间只描述这三个已抽seed的episode不确定性，不能消除明显的seed间异质性。
- 按预注册门，seed0和seed2的点估计同时通过`reward>0.658/outage<=0.22`，即2/3 seed通过；但聚合outage `0.2282>0.2`，故整体门失败。B20不进入“已稳定的主推荐”，只记为已证明能降低PPO/control方差、提高吞吐的正向组件。
- 末200k训练rollout outage均值在seed0/1/2上只是`0.215/0.180/0.220`，看起来三者都接近可行；然而seed1的520条独立outage是`0.3058`。这个`+0.126`的差距证明，训练末段的20条on-policy轨迹仍不能替代大样本初始状态泛化评估。
- critic低估在三个seed都存在：hard CDF/truth分别为 `0.1058/0.1673`、`0.1043/0.3058`、`0.1657/0.2115`，其中seed1低估`0.2015`个概率点。因此下一阶段不应把“再多跑几步”当作主修复，而应降低action-baseline噪声并增强cost critic对recent/initial-state分布的holdout校准。
- 完整history与三seed对齐图位于 `_runs/wandb_export/dqc_pm3_b20_multiseed_1m_2026-07-16/` 和 `_runs/profiles/dqc_pm3_b20_multiseed_1m_2026-07-16/`。

### E50：对“现在训练是否太短”的最终判断与早停规则（2026-07-16）

- 结论不是简单的“够”或“不够”。100k足够做机制级screen和发现确定性bug，但不足以否定有慢变量的PID/LSTM/distributional-critic组合；300k足够做闭环趋势screen，但只有当末段已平稳且被基线支配时才可早停；1M是稳定性验证，不是所有组合的默认入场成本。
- P-M1给出了false-negative的直接实例：同一seed的520条复评从300k的`reward/outage=0.776/0.248`改善到600k的`0.814/0.225`，所以“早期不好、之后变好”确实会发生。但到1M又退化为`0.632/0.244`，说明它是闭环相位/极限环，而不是单调的慢收敛。
- 随机初始化可能改变某个工作点出现的时间，但无法解释三seed在同一1M预算下的大幅critic低估和outage差异。当前“失败组合单靠更长训练稳定反转”的优先级已降低；不臆造数值概率，因为三个seed不足以可靠估计这个概率。
- 继续长跑的晋级条件：末20%的reward或constraint仍有持续改善趋势；lambda/critic显示仍在有方向地追赶而非重复周期；至少一个独立大样本checkpoint比早期明显改善；或组件理论上存在明确的慢传播时常。
- 应当早停的条件：末段已平稳且reward/outage同时被基线支配；多个checkpoint重复同样的风险周期；配对试验证明变量只修正梯度尺度却没有改善校准；或失败由确定性目标错配造成。这些情况下继续加步数只会浪费资源。
- 评估协议保持两层：140条只作screen，520条作单seed确认。在outage约0.2时，520条的二项标准误约`0.0175`，95%半宽约`0.034`；140条半宽约`0.066`，不适合判定贴边界的安全性。最终候选至少3 seeds × 1.5M，与QCPO_refs正式比较时再对齐其5M环境步数。
- 当前下一单变量路线是在B20上把`num_action_samples=4→16`，检验冻结后在8个PPO epoch重复使用的action risk baseline方差；同时把recent-rollout/initial-state replay和holdout CDF calibration作为更直接针对seed1 critic低估的独立路线。两者不在同一跑中混合。

### E51：P-M4预注册——B20基础上仅将action baseline K4→K16（2026-07-16）

- P-M4完全复用P-M3 seed0的1M配置，只把`num_action_samples=4→16`。网络、N32、MC cost target、time weighting、T1 sigmoid、PID target=.15、actor LR、8 PPO epochs、B20和总环境步都不变。
- 代码核对确认K只影响两个无梯度的cost-action baseline查询：constraint RMS更新和首个recurrent PPO epoch的冻结risk advantage。实际动作CDF不变，K个baseline动作均从rollout保存的behavior Gaussian参数采样；冻结后的同一risk weight继续供8个PPO epoch使用。因此K4的baseline Monte Carlo标准误在相同动作离散度下约为K16的2倍，噪声还会被8次重复使用。
- K16不会修复cost critic对初始状态的系统偏差，它检验的是“critic给定时，action相对风险排序是否被K4采样噪声放大”。recent/initial-state replay与holdout calibration仍是另一条独立路线，不能把两者同时加入后再归因。
- 100k不足以否定此变量，因为其收益通过数十次policy/dual更新积累；本次直接跑1M，但仍按100k保存phase-aligned checkpoint。参考K4独占GPU训练`390.6s`，K16增加无梯度critic前向，预计纯训练约7～10分钟、含140条终评约9～12分钟。
- 基本门仍为520条`reward>0.658,outage<=0.22`。相对K4 seed0还要求末200k训练outage不高于0.22、reward不低于约0.70，且KL/clip或风险周期至少一项显示稳定性不恶化；否则即使偶然终评好看也不晋级。
- 140条只作是否扩520的screen。若P-M4通过，下一步优先在曾失败的seed1做同配置1M压力复现；若失败则停止K小网格，不试K8/K32碰运气，转recent/initial-state critic replay与holdout校准。

### E52：P-M4 K16完整1M结果——中期PPO更稳，但后期风险周期与回报退化仍存在（2026-07-16）

- job `DQCAC_DynamicButton_pm4_timew995_pi_target015_smoothT1_b20_k16_ckpt100k_1m_s0`，W&B `rx3mofgr`，正常exit 0。纯训练`393.1s`，K4为`390.6s`；K16额外无梯度cost-critic前向没有形成可测的wall-time代价，也无OOM/NaN。
- 140条final为reward `0.6030±0.4853`、outage `25/140=0.1786`、mean cost `7.743`、Q80 cost `13.2`；hard/smooth critic CDF为`0.1346/0.1370`，pred mean `6.838`，lambda `0.2120`。安全点估计较好，但reward低于预注册`>0.658`门。
- 0–300k时K4/K16的reward为`0.344/0.346`、outage `0.113/0.120`，没有早期优势。300–600k时K16确实表现出局部收益：outage从K4的`0.237`降到`0.163`，KL/clip从`0.00404/0.207`降到`0.00222/0.117`，reward仅`0.685→0.664`。
- 这个中期好点没有保持。600k–1M的K16 outage为`0.270`，K4为`0.235`；680k单批outage达到`0.65`，lambda随后升至约`0.51`。末200k K16/K4 reward为`0.621/0.711`，window outage为`0.236/0.212`，KL为`0.00382/0.00286`，clip为`0.1888/0.1557`。
- 因此K16减小action baseline Monte Carlo误差可以短暂降低PPO位移和中期outage，但它不是后期策略—critic—PID周期的主瓶颈。更长的1M预算再次避免了只看300–600k而把K16误判为成功。
- 由于末段趋势门和140条reward门同时失败，不扩520，不跑seed1，也不继续K8/K32小网格。K16保留为负/阶段性正消融；主线转recent/initial-state cost critic replay、holdout CDF calibration或直接s0 auxiliary。
- 完整history和对齐图：`_runs/wandb_export/dqc_pm3_k4_pm4_k16_1m_2026-07-16/`、`_runs/profiles/dqc_pm3_k4_pm4_k16_1m_2026-07-16/`。

### E53：C-S0 recent初始状态辅助目标与prequential holdout实现（2026-07-16）

- 新增默认关闭的`cost_s0_aux_coef`和`cost_s0_replay_batches`。显式开启目前要求`cost_target_mode=mc,cost_history_mode=raw`，防止把n-step bootstrap或旧actor feature语义混入首轮消融。
- 每个新rollout保存实际`(raw s0, behavior a0, 完整trajectory MC cost)`。有限deque只保留最近若干批；raw observation不缓存归一化值，辅助更新时重新通过当前RMS，避免旧输入尺度污染。
- cost目标采用归一化凸组合：`L_cost=(L_transition+c*L_s0)/(1+c)`。因此`c=.25`对应基础/辅助权重`0.8/0.2`，改变监督时间测度而不增加cost objective总尺度，和直接把loss相加或抬高critic LR严格区分。
- 新增训练前/后同批校准。pre使用刚采到、尚未进入本轮任何optimizer step的真实`s0/a0/return`，是prequential holdout；post使用完全相同样本但更新后的critic。指标包含hard CDF、truth、bias/absolute error、逐状态Brier、predicted/true mean cost。它不重采动作、不消耗RNG。
- W&B/profile新增aux loss/scale/replay样本数及pre/post全套校准指标；最终JSON也保存最后一次pre/post字典。profile新增s0 probability与Brier面板。
- 默认精确回归：改前/改后同一两环境短run的obs RMS、actor、reward critic/target、cost critic/target、lambda与runtime逐tensor完全一致。Module hash分别为obs `725eed9a0dd35f01`、actor `b1bb7c15a8c9e891`、reward `7845310dde10f777`、reward target `ff1133ce8d624aba`、cost `aa8d0b0d2b19382a`、cost target `bbbe962d16969daf`。
- 辅助边界回归：两批默认与`coef=.25,replay=2`对照中，obs RMS、actor、reward critic/target hash完全相同，只有cost critic及target不同；证明lambda=0时没有意外改动策略/reward分支。chunk smoke训练`8.8s`、full-batch smoke `8.9s`，均exit0并完成recurrent PPO、QR、checkpoint/JSON和评估。
- profile旧数据回归也通过；缺少新列的历史导出仍可正常生成报告和overview。

### E54：C-S0A 100k固定策略校准门预注册（2026-07-16）

- 为隔离critic效果，C-S0A使用P-M3的B20/N32/MC/time-weight.995/recurrent网络和8 PPO epochs，但设`lambda_max=0`，总预算100k。先跑aux关闭的新基线，再只开`coef=.25,replay_batches=1`；两条policy、reward和真实cost轨迹理论上应逐点相同。
- replay=1首先检验“直接提高当前s0监督质量”本身；只有它改善post/final却没有改善下一批pre holdout时，才晋级replay=4检验跨批稳定。这样不会把direct s0 weighting和old-policy replay一次混入。
- 每条5次B20 rollout、共100条训练初始状态，内置140条独立评估。参考P-M3吞吐，预计纯训练约40秒、含评估约1.5～2分钟；均用持久化后台串行运行。
- 工程门：两条actor/真实reward-cost逐点相同，无NaN/OOM，aux cost总尺度与grad clipping不能显著恶化。
- 算法门：相对新基线，独立评估CDF absolute bias或mean-cost relative error至少改善25%；同时最后三批prequential CDF error或Brier至少改善20%，否则只能说明同批记忆，不能晋级闭环。
- 若C-S0A通过，先做replay=4的同预算正交门，再把胜者放回B20/PID跑300k；若失败停止coef小网格，转独立holdout early stopping或显式ensemble/uncertainty，而不是扫`.1/.25/.5/1`碰运气。


### E55：C-S0A结果与训练预算判定——局部机制100k已足够，策略性能不能据此下结论（2026-07-16）

- 三条持久化后台run均正常exit 0：baseline/replay1/replay4的W&B分别为`j6cd5p1c/5lpevd9t/uyjull0z`，纯训练耗时`45.9/46.5/46.6s`。配置、seed和100k数据预算完全相同，只改变`cost_s0_aux_coef/replay_batches=0/1、.25/1、.25/4`。
- 固定`lambda_max=0`后，三条最终真实policy评估逐位相同：140条reward `0.19234±0.17293`、outage `0.05`、mean cost `3.51429`、Q80 cost `6.2`。baseline与replay1 checkpoint的obs RMS、actor、reward critic/target、lambda及全部runtime状态逐tensor exact，只有cost critic/target的6个参数张量变化。这排除了初始化、策略采样和actor更新差异。
- baseline/replay1/replay4的最终hard CDF分别为`0.01473/0.01629/0.01183`，相对真实`0.05`的absolute error为`0.03527/0.03371/0.03817`。replay1仅改善`4.43%`，replay4反而恶化`8.23%`，均未达到预注册`25%`门。
- 最终predicted mean cost为`2.1851/2.2838/2.0512`，相对真实`3.5143`的relative error约`37.82%/35.01%/41.63%`；replay1只改善约`7.4%`，replay4恶化约`10.1%`。
- 最后三批60k/80k/100k的prequential pre-CDF absolute error三者完全相同，均值`0.017708`；pre-Brier baseline/replay1/replay4为`0.016732/0.016699/0.016699`，改善不足`0.2%`。同批更新后的post-Brier为`0.017611/0.015462/0.017171`，replay1改善约`12.2%`、replay4约`2.5%`，仍低于`20%`门。辅助项主要改变见过数据的拟合，没有改善下一批泛化。
- 结论：停止recent-s0 replay的coef/replay小网格，不把它放入PID闭环跑1M。100k在这里不是用来评价最终策略，而是固定策略下提供5批独立数据刷新和100次cost-critic更新，直接检验该局部作用链；继续长跑缺乏晋级证据。
- 对“短跑会不会误杀慢热组合”的统一规则：100k只否决确定性bug或可直接观测且不过门的局部机制；策略/dual/LSTM组合不能据此作长期结论，至少给300k趋势窗，仍在改善则续到600k/1M。P-M1的520条复评`0.776/0.248→0.814/0.225→0.632/0.244`已经证明中期可改善、后期又回退；因此候选看多个checkpoint而非单个终点，最终至少3 seeds，并与QCPO_refs对齐相同环境步数。
- 完整history与对齐图：`_runs/wandb_export/dqc_cs0_baseline_r1_r4_100k_2026-07-16/`、`_runs/profiles/dqc_cs0_baseline_r1_r4_100k_2026-07-16/`。


### E56：C-S0B统计功效修正与冻结高风险策略校准预注册（2026-07-16）

- 用户质疑100k是否过短后复核事件数：C-S0A五个B20 rollout的outage为`0/0/0/0/0.05`，即前80k没有任何超限轨迹，最后一批仅1条；最终140条truth也只有`0.05`。因此E55能否定“早期低风险分布上已有泛化收益”，但不能外推为接近约束边界时永久无效。
- 为避免重新跑1M等待策略进入风险区，新增默认关闭的`--critic_calibration_from`。它只恢复成熟checkpoint的actor、recurrent actor RMS和raw critic observation RMS；cost/reward critic与optimizer重新初始化，lambda/PID/runtime不恢复，并强制actor、dual及两套RMS冻结。
- 不同critic架构构造会消耗不同随机数，因此所有Module构造和policy-only恢复后再次设置Python/NumPy/Torch/CUDA seed。这样同rollout seed的baseline/候选使用相同环境布局、动作噪声和真实cost轨迹。
- 两个持久化smoke均exit0：旧eval-only回归通过；校准模式完成rollout、MC QR和评估，日志明确`actor_updates/iter=0,policy_frozen=True,obs_stats_frozen=True`。训练后actor（含内部RMS）与raw obs normalizer相对源checkpoint逐tensor exact，源cost critic未加载。
- C-S0B使用P-M3 seed1的1M final actor（原520条outage `0.3058`）作为相关高风险策略，另用rollout seed101。baseline与`coef=.25,replay4`各跑5×B20×T1000=100k、20 critic updates/rollout、140条配对终评；预计每条纯训练约45～60秒、含评估约1.5～2分钟，持久化后台串行。
- 统计功效门：100条训练轨迹至少出现10次超限，否则不作负结论，改为增加冻结数据量；在事件充分时，候选须使最后三批prequential CDF error/Brier至少改善20%，且独立终评CDF absolute error或mean-cost relative error至少改善25%。只改善post同批仍判为记忆，不晋级。
- 若C-S0B仍无pre/final信号，才停止simple replay coef网格并转direct exceedance classifier、bootstrap ensemble/upper-confidence CDF或cross-fit early stopping；若通过，再给冻结critic 300k确认，之后才放回PID闭环。


### E57：C-S0B 100k结果与一次性300k功效扩展（2026-07-16）

- baseline/aux的W&B为`rwqj0kch/x5inqzzi`，训练`46.4/45.9s`、exit0。五批outage逐点同为`0.20/0.10/0.15/0.25/0.30`，共`20/100`次超限；140条终评reward/outage/mean cost也exact为`0.8542/0.2643/10.9786`。actor与两套RMS逐tensor exact，统计功效和配对门均通过。
- 最后三批pre-CDF absolute error从`0.19010→0.15729`，改善`17.3%`，接近但未过20%门；pre-Brier只改善`5.8%`。pre mean-cost absolute bias从`4.0248→3.1092`，改善`22.8%`。post-CDF error改善`21.9%`，更像提高同批/近期拟合。
- 独立终评给出相反结论：hard CDF `0.21964→0.20826`，truth固定`0.26429`，absolute error从`0.04464`增到`0.05603`（恶化`25.5%`）；mean-cost relative error从约`13.63%`增到`16.76%`（恶化约`23%`）。因此100k未通过原晋级门，不能放回PID。
- 额外发现：reward critic checkpoint并非exact。两组参数不共享，但当前`update_critic`对reward+cost参数做联合global grad clip；aux改变joint norm后会连带缩放reward critic梯度。这不影响冻结actor的真实策略比较，却说明“只改变cost监督”仍有优化器级耦合，后续应把separate clipping作为独立工程消融。
- 因只有100个s0标签，且最后三批pre-CDF连续同向改善、仅差2.7个百分点，执行一次明确封顶的300k功效扩展：同source policy/rollout seed重新跑baseline与aux各15批，不改coef或任何其他参数。预计每条训练约2.3分钟、含评估约3分钟，总计约6分钟。
- 300k终止门：至少约30个超限事件；最后三分之一pre-CDF/Brier须改善20%，独立终评CDF或mean relative error须改善25%。不通过即停止simple replay，不跑1M、不扫coef；通过也只进入一次PID 300k门，不直接宣称有效。


### E58：C-S0B 300k大样本确认与live seed1压力测试预注册（2026-07-16）

- baseline/aux训练`121.0/121.2s`、exit0，15批outage逐点exact，总事件`89/300`；actor、actor RMS与raw obs RMS逐tensor exact。140条终评truth exact为reward/outage/mean cost `0.8657/0.2643/10.7929`。
- 140条critic结果从baseline CDF/mean `0.19799/9.4077`变为aux `0.27031/11.2698`，truth `0.26429/10.7929`；CDF absolute error约`0.0663→0.0060`，mean relative error约`12.8%→4.4%`。
- 为消除训练长度消耗不同evaluation RNG的影响，从两个final checkpoint各做同构eval-only 520条。truth逐位exact为reward `0.83494`、outage `146/520=0.28077`、mean cost `11.0173`；baseline CDF/mean为`0.19964/9.4186`，aux为`0.26665/11.2476`。CDF error降低`82.6%`，mean relative error从`14.5%`降到`2.1%`（约降低`85.6%`），确认跨布局泛化而非同批记忆。
- 最后五批pre-CDF/Brier只改善约`10.0%/4.1%`，未过原20%小批门；但B20逐批truth在0.15～0.45间变化，方差远高于520条固定协议。结论定级为“独立大样本强通过、小批pre门未通过”，不声称所有校准指标一致改善。
- 下一步只在P-M3最差的seed1加入`cost_s0_aux_coef=.25,replay_batches=4`跑完整1M；其余网络、B20、C20、MC/time-weight.995、T1、PID target=.15、8 PPO epochs、LR和checkpoint周期完全不变。独占GPU预计训练约6.5～8分钟、含140条评估约8～9分钟，持久化后台运行。
- live通过门：用相同eval-only 520协议比较原seed1 baseline `reward/outage=0.8622/0.3058`。候选需reward保持`>0.658`且outage`≤0.22`，或至少把outage降低0.08且reward保持≥0.75；同时critic CDF absolute error至少降低50%，末200k不能出现更严重极限环。未通过则说明校准改善未转化为policy控制，不扩seed0/2。

### E59：live seed1 1M——校准改善转化为显著安全收益，但牺牲reward（2026-07-16）

- job `DQCAC_DynamicButton_pm3_s0aux025r4_timew995_pi_target015_smoothT1_b20_ckpt100k_1m_s1`，W&B `erayfl8y`，训练`392.3s`、exit code 0。相对原P-M3 seed1只增加`cost_s0_aux_coef=.25,cost_s0_replay_batches=4`，其余网络、B20、C20、PID、PPO、LR与1M预算完全相同。
- 140条内置终评中，baseline到candidate的reward/outage为`0.8404/0.2429 -> 0.7379/0.2429`；小样本没有显示安全收益，但critic CDF从`0.1002`变为`0.2295`，相对同一个truth `0.2429`的absolute error从`0.1426`降到`0.0134`。
- fresh eval-only 520条正式结果为：baseline reward `0.8622±0.5809`、outage `159/520=0.3058`、Wilson 95% `[0.2677,0.3467]`；candidate reward `0.6875±0.5783`、outage `98/520=0.1885`、区间 `[0.1572,0.2243]`。outage绝对下降`0.1173`、相对下降`38.4%`；reward绝对下降`0.1747`、相对下降`20.3%`，但仍高于预注册`>0.658`下限。
- 520条critic CDF从baseline的`0.10427`变为candidate的`0.23696`。相对各自policy truth `0.30577/0.18846`，absolute calibration error由`0.20150`降到`0.04850`，降低`75.9%`，通过至少减半的校准门。
- 时间结构不能被final点掩盖：300–600k candidate相对baseline的reward/outage为`0.827/0.207 vs 0.783/0.243`，看似全面占优；600–800k却回摆为`0.639/0.300 vs 0.823/0.260`，lambda均值`0.454 vs 0.341`；800k–1M为`0.603/0.170 vs 0.825/0.180`。因此600k会产生false positive，完整1M揭示了真实reward代价和中段极限环。
- 结论：recent-s0 auxiliary不再是“只改善冻结critic”的局部trick；它在最差seed1上把大样本outage拉回0.2以内，并通过live晋级门。但它没有消除闭环周期，而且安全收益以明显reward损失为代价，尚不能作为稳定主配置。
- 下一步保持单变量设计，在seed0/2各跑完整1M；不根据seed1结果回调coef或replay次数。两条并行预计训练约12分钟、含140条终评约14分钟。只有多seed聚合仍改善约束且reward可接受，才继续1.5M/同QCPO_refs预算确认。

### E60：C-S0 live三seed结论——救回seed1但把风险转移到seed2，不进入1.5M（2026-07-16）

- seed0/2完整1M训练均exit0，耗时`696.2s/695.7s`，W&B为`nle3jm1q/alf53xvv`。由于seed1已经证明140条outage会漏掉520条改善，本轮透明取消原140 screen，对两个不好看的seed都做了fresh eval-only 520，没有选择性丢弃。
- 520条baseline→aux结果：seed0 `reward 0.7180→0.5152, outage 0.1673→0.1558`；seed1 `0.8622→0.6875, 0.3058→0.1885`；seed2 `0.8670→0.6320, 0.2115→0.3288`。aux在所有seed都降低reward，只在seed1提供大安全收益，并在seed2造成等量反向恶化。
- 三seed reward从`0.8157±0.0847`降到`0.6116±0.0880`。outage从`0.2282±0.0707`变为`0.2244±0.0920`；合并事件仅从`356/1560`降到`350/1560`，Wilson 95%区间`[0.2081,0.2497]`与`[0.2043,0.2457]`高度重叠，不构成有意义的聚合安全改善。
- per-seed CDF absolute error从baseline的`0.0615/0.2015/0.0458`变为aux的`0.0273/0.0485/0.1586`。平均误差虽从0.1029降到0.0781，但seed2误差扩大3.46倍；simple recent replay把一致低估换成更高的seed间偏差，没有形成可靠校准器。
- 末200k训练rollout的三seed均值同样显示reward/outage约为baseline `0.808/0.205`、aux `0.647/0.202`：训练窗口安全几乎不变，reward代价已出现。独立初始状态上的改善只发生在部分seed，不能靠挑checkpoint解释。
- 长度判断：100k不足以否定s0 replay，300k冻结策略发现其校准价值，1M seed1发现live价值；但3 seeds×1M已足以拒绝当前`.25/r4`作为主配置。继续1.5M可能改变周期相位，却没有跨seed定向改善证据，低优先级且不值得直接续跑。
- 下一步停止simple coef/replay网格，转向不会把同批反馈直接灌回共享critic的cross-fit/target-stabilized calibration，以及限制PPO闭环回摆的target-KL early stop；二者分别做默认关闭的单变量消融。

### E61：P-M5 target-KL实现、回归与1M预注册（2026-07-16）

- commit `955b05f`新增DQCAC专用`ppo_target_kl`，默认0完全关闭。正阈值在epoch forward得到behavior→current approximate KL后、backward前判断；越界probe不做optimizer step，剩余actor epochs跳过，但critic的20次更新继续跑满。日志同时记录configured/completed epochs、early-stop与update-applied。
- 固定seed tiny recurrent金样本在改动前后比较：actor、reward/cost critic及target、RMS、lambda与全部runtime逐tensor/逐字段exact，eval JSON exact；唯一新增summary/config字段为`ppo_target_kl=0`。
- `target=1e-8`触发smoke中，每轮configured actor epochs=3、completed=1、early_stop=1、最后probe update_applied=0；critic `learning_steps=9=3×3`，exit0。所有临时checkpoint/job/log/离线W&B已清理。
- 用既有六条1M history选阈值而不扫网格：在P-M3 baseline中第8 epoch KL超过`.004`的rollout占18%，s0-aux中占25.3%；baseline 0–300k仅2.2%，300–600k与600–800k均33.3%。aux的KL与同批outage相关系数约0.394，因此`.004`能命中中段回摆而不会普遍砍掉早期更新。
- P-M5以原P-M3 seed1为压力基线，只增加`ppo_target_kl=.004`，不带s0 aux；B20、C20、N32、T1、PID、LR和8个最大epoch全部不变。直接跑1M，因为100k/300k几乎不会触发该机制，短跑没有检验功效。
- 原seed1 fresh520为reward/outage `0.8622/0.3058`。候选要求520条outage至少下降0.08并到`≤0.22`，reward保持`≥0.75`，且末200k不出现更大的risk周期；无论140条screen好坏都执行520。预计训练6～7分钟、140终评约1～2分钟、520复评约4分钟，总计约11～13分钟，持久化后台运行。

### E62：P-M5 seed1完整1M通过大样本门，已晋级多seed（2026-07-16）

- P-M5相对P-M3 seed1只增加`ppo_target_kl=.004`，不加入已经被三seed否决的recent-s0 replay。job为`DQCAC_DynamicButton_pm5_targetkl004_timew995_pi_target015_smoothT1_b20_ckpt100k_1m_s1`，W&B run为`54y9bpza`；1M纯训练`391.2s`，后台job与内置评估均exit code 0，无NaN/Inf。
- target-KL实际参与了优化，而非未触发的装饰项：50个rollout中19个提前停止actor更新，共完成`308/400`个允许的actor epoch；前/中/后20%阶段的early-stop率为`30%/40%/60%`，后20%平均只完成`4.6/8`个epoch。critic更新不受影响。
- 训练末20%均值相对P-M3 seed1由reward/outage/lambda/KL=`0.8253/0.1800/0.2206/0.00246`变为`0.9185/0.2350/0.1977/0.00387`。target-KL提高了后段reward并限制越界epoch，但训练小批outage略高，说明它可能减慢risk/PID纠偏，不能仅凭训练曲线宣布成功。
- 内置140条final为reward `0.8960`、outage `28/140=0.2000`、Q80 cost `14.2`。随后从同一个post-update final checkpoint做fresh eval-only 520条，得到reward `0.8684±0.5429`、outage `88/520=0.16923`、Q80 cost `13`、mean cost `8.8212`；作业`dqc_pm5_targetkl004_1m_s1_final_eval520_20260716`正常退出。
- 与严格同协议P-M3 seed1的`reward=0.8622±0.5809,outage=159/520=0.30577,Q80=18,mean cost=12.3308`比较，reward差为`+0.0062`，近似95%区间`[-0.0621,+0.0746]`，可视为持平；outage绝对下降`0.13654`，差值近似95%区间`[-0.18760,-0.08548]`。单独Wilson 95%区间从baseline `[0.26771,0.34667]`降为candidate `[0.13946,0.20386]`。
- critic也有改善但尚未校准：初始状态CDF absolute error从`|0.10427-0.30577|=0.20150`降至`|0.29561-0.16923|=0.12638`，约改善37.3%，但由严重低估转成明显高估。P-M5的主要收益不能归因于critic目标变化，因为critic配置与P-M3完全一致，更符合“限制8-epoch PPO过冲、改变策略—PID周期相位”的机制。
- 该结果通过预注册门：outage不高于0.22、相对baseline下降超过0.08、reward不低于0.75，且520条而非140条给出一致方向。因此已原样启动seed0/2各1M的持久化后台复现，不调阈值、不挑checkpoint。两条并行预计约11～13分钟完成训练和内置评估，随后各做同协议520条fresh evaluation；多seed裁决仍使用全部三个seed，不能只保留压力seed1。
- 对齐history与图位于`_runs/wandb_export/dqc_pm3_seed1_vs_pm5_targetkl004_1m_2026-07-16/`和`_runs/profiles/dqc_pm3_seed1_vs_pm5_targetkl004_1m_2026-07-16/`。seed0/2后台job分别为`DQCAC_DynamicButton_pm5_targetkl004_timew995_pi_target015_smoothT1_b20_ckpt100k_1m_s0`和同名`s2`。

### E63：P-M5三seed裁决——聚合更安全，但固定KL阈值不稳定支配baseline（2026-07-16）

- seed0/2完全复用seed1的P-M5配置，各训练1M步，训练耗时`700.3s/690.2s`，W&B run为`deugfqif/5y3izber`；两条后台作业、内置140评估和fresh 520评估均exit code 0。并行共享GPU使wall time高于seed1独占的`391.2s`，环境步与算法预算不变。
- target-KL在seed0/1/2分别触发`15/19/20`个rollout，实际完成actor epoch为`321/308/305`，合计`934/1200`，即跳过22.2%的最大更新。它不是仅对seed1生效的偶然开关。
- 520条严格同协议baseline→target-KL为：seed0 reward/outage `0.7180/87/520=0.1673 → 0.7140/127/520=0.2442`；seed1 `0.8622/159/520=0.3058 → 0.8684/88/520=0.1692`；seed2 `0.8670/110/520=0.2115 → 0.6780/96/520=0.1846`。
- candidate逐seed outage Wilson 95%区间为`[0.2093,0.2829]`、`[0.1395,0.2039]`、`[0.1536,0.2202]`。seed0相对baseline恶化`+0.0769`，episode-level差值近似95%区间`[+0.0280,+0.1258]`；seed1改善`-0.1365`，区间`[-0.1876,-0.0855]`。相反方向都超出单纯520回合抽样噪声，说明真正的seed异质性。
- 跨seed reward从`0.8157±0.0847`降为`0.7535±0.1012`，平均下降7.6%，主要来自seed2的`-0.1890`；合并outage从`356/1560=0.2282`降为`311/1560=0.1994`，Wilson区间从`[0.2081,0.2497]`变为`[0.1803,0.2199]`。按episode池化的差值区间刚好为`[-0.05760,-0.00009]`，但三个配对seed差为`+0.0769/-0.1365/-0.0269`、sample std达`0.1067`，不能忽略seed聚类后宣称稳健显著。
- 训练曲线同样表现为安全—性能交换：三seed全程均值reward/outage由`0.6590/0.1973`变为`0.5731/0.1657`；末20%由`0.8083/0.2050`变为`0.6642/0.1933`。target-KL平均减少风险更新幅度，也同时减少有用reward更新。
- critic CDF absolute error的seed均值从`0.1029`降至`0.0633`；seed0甚至由`0.0615`降至`0.0080`，最终outage却从0.1673升到0.2442。故“critic校准变准”不是策略必然更安全的充分条件；固定KL阈值还会截断高lambda下本来必要的risk correction。
- 裁决：`.004`保留为安全优先消融/可选组件，但不作为超过QCPO_refs的主配置，不扫`.003/.005/.006`，也不直接续到1.5M。它通过聚合outage门，却没有满足预注册的reward不退化与跨seed一致性。单看seed1会误判为全面升级，单看seed0又会误判为全面失败；3 seeds×1M+520正是本问题所需的预算。
- 完整history和训练图：`_runs/wandb_export/dqc_pm3_vs_pm5_targetkl004_multiseed_1m_2026-07-16/`、`_runs/profiles/dqc_pm3_vs_pm5_targetkl004_multiseed_1m_2026-07-16/`；后者新增`eval_multiseed.csv`、`eval_multiseed_summary.json`与`eval_multiseed_comparison.png`（2100×960，可解码验证通过）。

### E64：C-X1慢target actor-query实现、回归与1M预注册（2026-07-16）

- 代码审计发现当前每个inner update的顺序是`update_critic(batch) → update_actor(batch) → soft_update_target()`；首个actor epoch因此用已经看过当前rollout标签一次的online cost critic生成并缓存risk advantage。同批标签可经online critic立刻反馈给同批actor，是simple replay正反馈之外更基础的in-batch leakage。
- 新增默认`online`的`cost_actor_query_mode`。显式`target`时，只有actor实际动作/行为策略baseline的cost CDF以及对应`constraint_rms`查询Polyak cost target；online critic仍负责QR训练、pre/post holdout、初始CDF、最终评估，经验PID完全不依赖critic。首个actor查询发生在本轮target同步之前，所以当前rollout标签不能先进入查询网络；随后8个PPO epoch继续复用同一个冻结risk weight。
- 新增`advantage/risk_query_target_online_abs_mean`和summary同名字段，使用完全相同的`s,a,b`测量target与online风险CDF差。online默认返回精确0且不额外forward；target才多做一次无梯度online诊断前向。profiler已纳入该指标。
- 默认兼容回归使用提交`1240549`与当前代码、CPU sync、相同seed、`2 iterations×2 envs×20=80`环境步：actor、reward/cost critic及两个target、obs normalizer共6个module，lambda tensor、runtime、metrics、config、metadata、8条eval和summary逐tensor/逐值exact；仅新增配置字段和工作树输出路径不同。
- 数值测试确认online helper返回online模块、target helper返回target模块、初始两者逐tensor相等；故意修改online末层后target查询保持不变，非法mode被ValueError拒绝。target完整80步smoke训练`8.6s`、exit0，summary正确为`target`，末次target-online gap为`1.05e-9`；由于两批均零cost且只有4个critic step，该极小值只证明链路分离，正式run必须观察成熟阶段gap。
- C-X1正式实验以P-M3失败seed1为压力基线，只改`cost_actor_query_mode=target`；`ppo_target_kl=0`、s0 aux=0、target tau=.05、interval=1、B20/C20/N32/T1/PID和1M预算全部不变。预计独占GPU训练约7～8分钟，140与520评估约5～6分钟，总计12～14分钟，持久化后台运行。
- 520晋级门沿用压力seed标准：outage `≤0.22`且相对0.3058至少下降0.08，reward `≥0.75`；同时要求成熟阶段target-online gap非零、末20%不形成比baseline更大的周期。通过才扩seed0/2；失败则不扫target tau，转真正的两折cross-fit/双critic ensemble。该路线与target-KL保持正交，不在首轮混合。


### E65：C-X1 seed1完整1M结果——早停到100k会误杀，520条门通过（2026-07-16）

- 持久化job `DQCAC_DynamicButton_cx1_targetquery_timew995_pi_target015_smoothT1_b20_ckpt100k_1m_s1`、W&B `3oogqkf5`正常完成；纯训练`393.9s`，每100k保存pre-update rollout checkpoint，最终另存post-update checkpoint。相对P-M3 seed1只把`cost_actor_query_mode=online→target`，target-KL与s0 aux均关闭。
- 这条曲线给出“短跑会误杀”的直接例子：40k/80k训练reward仍为`-0.040/-0.083`，200k才到`0.471`，260k/300k升到`0.634/0.807`，400k达到`0.933`；64–72万为`0.831–1.031`，82–98万多数为`0.781–1.135`。若100k按性能停掉，会错过后续学习。
- 但它不是单调慢收敛。400k附近outage升高后lambda到约`0.29`，44–48万reward回落到`0.504–0.667`；末个1M训练批又是reward/outage `0.747/0.35`。因此延长训练的价值是看清策略—critic—PID周期，不能把“多跑”当作自动稳定化手段，也不能用最后一个小批替代冻结策略评估。
- 对齐1M history后，baseline→C-X1的末20%训练均值为reward `0.8253→0.9002`、outage `0.180→0.210`、lambda `0.2206→0.1535`、KL `0.00246→0.00245`、clip `0.1293→0.1293`。成熟阶段target-online风险查询absolute gap末20%均值`0.0521`、末点`0.1024`，证明开关实际改变了actor监督而非数值上等同online。
- 内置140条final为reward/outage `1.058/31/140=0.2214`，online critic CDF为`0.3616`。它只作screen；同一个post-update final checkpoint的fresh eval-only 520条为reward `0.9661±0.5244`、outage `113/520=0.21731`、Q80 cost `16`、mean cost `10.0135`，Wilson 95%区间`[0.1840,0.2548]`。
- 同协议P-M3 seed1 baseline为reward `0.8622±0.5809`、outage `159/520=0.30577`。C-X1使reward增加`0.1039`（相对`+12.1%`，近似95%区间`[+0.0366,+0.1712]`），outage降低`0.08846`；保守Newcombe差值95%区间为`[-0.16267,-0.01296]`。它通过预注册的`≤0.22`、改善至少0.08、reward≥0.75三项门。
- critic仍未解决：C-X1的online smooth CDF `0.3608`相对truth `0.2173`高估`0.1435`；baseline则为`0.1074`相对`0.3058`低估`0.1984`。absolute error有所下降但偏差符号翻转，说明慢target查询改善了闭环反馈，不等于distributional critic已经校准。
- 按规则不在seed1上续1.5M或扫target tau，而是原样扩展seed0/2各1M+520。机器只有一张A100，改为串行持久化后台运行以避免GPU争用；独占训练实测约6.6分钟，含140 screen约8分钟，单个520复评约5分钟。seed0已启动，只有三seed方向和reward都可接受时才将C-X1列为主候选。
- 正式数据：`_runs/wandb_export/dqc_pm3_seed1_vs_cx1_targetquery_1m_2026-07-16/`、`_runs/profiles/dqc_pm3_seed1_vs_cx1_targetquery_1m_2026-07-16/`；520结果为`_runs/DQCAC_DynamicButton_cx1_targetquery_1m_s1_final_eval520_s1.json`。


### E66：C-X1三seed裁决——长跑避免早期误杀，但慢target把回报提升换成更高风险（2026-07-16）

- seed0/1/2的1M训练均正常exit 0，纯训练耗时`394.4/393.9/389.0s`，W&B run为`5t6sxi2j/3oogqkf5/yod7kvxy`。三条都是独占单张A100串行持久化后台作业，配置只在训练seed和输出路径上不同。
- 内置140条reward/outage为seed0 `1.142/49/140=0.350`、seed1 `1.058/31/140=0.2214`、seed2 `1.163/47/140=0.3357`。没有根据seed1通过而保留、也没有根据seed0/2失败而停止；三个post-update final checkpoint全部按同协议做fresh 520。
- 520条baseline→C-X1结果为：seed0 reward/outage `0.7180/87/520=0.1673 → 1.0455/171/520=0.3288`；seed1 `0.8622/159/520=0.3058 → 0.9661/113/520=0.2173`；seed2 `0.8670/110/520=0.2115 → 1.0767/162/520=0.3115`。
- C-X1在三个seed都提高reward，逐seed增量为`+0.3275/+0.1039/+0.2097`；跨seed均值从`0.8157±0.0847`升到`1.0294±0.0570`，平均约增加26.2%。这是该组件明确且可复现的正面效果。
- 安全性方向不稳定且总体恶化。逐seed outage变化为`+0.1615/-0.0885/+0.1000`，sample std `0.1303`；合并事件从`356/1560=0.2282`升到`446/1560=0.2859`，Wilson 95%区间从`[0.2081,0.2497]`变为`[0.2640,0.3088]`。episode池化差值为`+0.05769`、近似95%区间`[+0.02709,+0.08829]`，但最终解释仍以2/3 seed恶化和强seed异质性为主。
- online critic smooth-CDF absolute error逐seed从baseline `0.0587/0.1984/0.0434`变为C-X1 `0.1841/0.1435/0.1423`，跨seed均值`0.1002→0.1566`。它修复原压力seed1的低估，却让原先较准的seed0/2明显低估；这与outage方向完全一致。
- 三seed训练曲线反而容易给出false positive：末20% reward `0.8083→0.9031`，训练outage `0.2050→0.1983`，lambda `0.2459→0.1378`，看起来全面更好；fresh初始状态评估却在seed0/2失约。训练布局上的经验PID并没有暴露policy的initial-state泛化风险。
- 训练长度结论要同时保留两面：100k会因三个seed尚未形成高reward而误杀C-X1的性能作用；但完整`3 seeds×1M+520`又证明它不是安全主配置。继续1.5M或扫描target tau更可能改变闭环相位，而没有跨seed定向安全证据，因此停止该小网格。
- C-X1定级为“稳定提高reward、但平均放松约束”的机制消融，不与target-KL组合补丁式试错。下一条路线是两个独立cost critic的两折cross-fit：每个critic只在一半环境轨迹上训练，actor对该折状态查询未见该折标签的另一critic，提供比Polyak一步滞后更严格的样本隔离；先做默认兼容回归和冻结机制门，再决定是否进入live 1M。
- 正式history、profile和训练图在`_runs/wandb_export/dqc_pm3_vs_cx1_targetquery_multiseed_1m_2026-07-16/`及`_runs/profiles/dqc_pm3_vs_cx1_targetquery_multiseed_1m_2026-07-16/`。后者含`eval_multiseed.csv`、`eval_multiseed_summary.json`与`eval_multiseed_comparison.png`（2084×755，PNG解码验证通过）。

### E67：C-X2两折cross-fit实现、默认回归与冻结300k预注册（2026-07-16）

- 新增默认关闭的`cost_actor_query_mode=crossfit`。按环境编号把完整轨迹固定拆成两折；主cost critic只接收fold-0标签，独立peer只接收fold-1标签。actor对fold-0样本查询peer、对fold-1查询主critic，所以当前样本的真实cost不能先训练查询它的critic再回灌同批actor。首版刻意限制为偶数`num_envs>=2`、`MC + raw history + full-batch + s0_aux=0`，避免一次实验混入n-step、history encoder、recent replay或chunk更新语义。
- 新评估状态没有fold身份，最终CDF取两critic各自CDF的等权平均，不能先平均quantiles再计数；预测方差按两分布等权mixture计算。同时记录primary CDF、peer CDF、同输入absolute disagreement、两折QR loss和样本比例。checkpoint自动包含peer，eval-only严格恢复；profiler增加crossfit disagreement字段并对旧1M history回归通过。
- 默认兼容回归以提交`5630741`为金样本，相同seed、CPU sync、`2×2×20=80`环境步。obs normalizer、actor、reward/cost critic及两个target共6个module逐tensor exact；lambda、20个runtime字段、41个训练summary字段和非时间/路径eval JSON逐值exact。module hash为obs `86520917b3438940`、actor `645e073e484d5c19`、reward `92f6207845224661`、reward-target `2520ac1ff50d334e`、cost `8e14fb7efcc15e30`、cost-target `5df9d8aa18b2a45c`。
- 机制断言确认fold-0→peer、fold-1→primary路由逐tensor正确；两个critic都被optimizer更新，transition fraction严格`0.5/0.5`，末次fold loss为`0.41993/0.54391`，查询分歧为有限非零`2.62e-8`。完整入口smoke训练`8.7s`并完成checkpoint、双critic eval和JSON；独立eval-only严格加载peer后exit 0。临时脚本、checkpoint和profile只在验证期保留，记录完成后清理。
- 测试还暴露后台launcher复用同名stale job时会暂时保留旧`exit_code`，使轮询器提前误判。本轮在新启动前仅删除该job旧的`worker_started_at/finished_at/exit_code`，不触碰日志与实验产物；`bash -n`和新job完整退出均通过。
- 冻结机制门使用C-S0B完全相同的P-M3 seed1成熟actor、独立rollout seed101和`15×B20×T1000=300k`数据；唯一算法变量是`online→crossfit`。模块构造和policy-only恢复后重置RNG，故真实轨迹应与既有baseline逐批一致。既有520 baseline truth/CDF/mean为outage `146/520=0.28077`、CDF `0.19964`、mean cost truth/pred `11.0173/9.4186`。
- crossfit每个critic只看约150条轨迹，检验的是out-of-fold泛化而非增加标签量。进入live 1M的门为：520条truth与baseline配对一致；ensemble CDF或mean-cost error至少改善25%，另一项不得恶化；primary/peer disagreement不持续扩大，最后五批prequential OOF误差无发散。若不过门，直接停止C-X2，不用短live策略曲线作结论；若通过，再在P-M3 seed1完整跑1M+520，之后仍需seed0/2复现。预计纯训练约2～3分钟、128内置评估约1分钟、520复评约4分钟，总墙钟约7～9分钟，全部持久化后台运行。

### E68：C-X2冻结300k结果——隔离成立，但半数据critic方差使其不过live门（2026-07-16）

- 正式job `DQCAC_DynamicButton_frozen_s1policy_seed101_cx2_crossfit_300k`、W&B `mnwyg6io`正常exit 0；300k训练`125.3s`，与单critic baseline `121.0s`接近，因为两个模型各处理半批、总QR样本量不翻倍。15批reward、cost、outage与baseline逐值exact，总超限事件同为`89/300`；因此差异只来自critic路由与数据拆分。
- 全15批/末5批prequential OOF CDF error为crossfit `0.17104/0.14656`，baseline `0.16917/0.14344`，crossfit分别略差`1.1%/2.2%`；末5批Brier `0.23812 vs 0.24462`只改善`2.7%`。post同批CDF/Brier也从baseline `0.11844/0.20034`恶化到`0.13188/0.23072`。没有“末段仍快速改善、应因短跑继续”的证据。
- 140条内置评估truth exact为outage `0.26429`、mean cost `10.79286`。baseline→crossfit ensemble CDF为`0.19799→0.20324`，absolute error只改善约`7.9%`；pred mean为`9.40770→9.39550`，反而略差。primary/peer CDF为`0.11228/0.29420`，平均绝对分歧`0.19442`，表明ensemble平均掩盖了很大的模型方差。
- 同协议fresh 520 truth逐值exact：reward `0.8349376`、outage `146/520=0.2807692`、mean cost `11.0173077`。baseline→crossfit CDF error `0.0811298→0.0761118`，仅改善`6.19%`；mean-cost error `1.5986857→1.4516167`，仅改善`9.20%`，两项都远低于预注册25%门。primary/peer为`0.120072/0.289243`，误差`0.160697/0.008474`，分歧仍达`0.186358`。
- 裁决：不启动C-X2 live 1M，也不把300k直接续成600k来移动门槛。300k固定策略已有89个tail事件、两项独立520误差和无改善末段趋势，足以回答局部机制；它不证明任何live策略在100k后永远不会变好，而是证明当前“两个模型各看一半数据”的隔离收益不足以抵消数据碎片方差。
- 路线保留为消融：① 600k使每个critic获得约300条轨迹，可单独回答样本量问题，但环境预算翻倍，不是主算法公平提升；② 标准K-fold complement让每个holdout模型看`(K-1)/K`数据，K=5约80%，代价约4倍cost-critic计算；③ 两模型max/UCB在本520上恰为peer `0.28924`、非常接近truth，但这是同一评估后观察，不能据此调参，而且actor若使用包含自身标签的in-fold模型会重新泄漏。
- 下一优先路线C-X3是不分裂数据的pre-update online cache：在本批任何QR step前，用已吸收全部历史批次的online critic计算并冻结risk advantage，再执行critic和8个PPO epoch。它提供当前样本隔离、保留全部历史数据，又避免C-X1的Polyak长期滞后；先做默认exact回归与顺序/缓存断言，再给seed1完整1M而非100k裁决。
- 正式history/profile/图在`_runs/wandb_export/dqc_frozen_baseline_vs_cx2_crossfit_300k_2026-07-16/`和`_runs/profiles/dqc_frozen_baseline_vs_cx2_crossfit_300k_2026-07-16/`；后者含`eval520_comparison.csv/json/png`，PNG为`1961×801`且PIL解码通过。

### E69：C-X3 pre-update online cache实现、回归与1M长跑预注册（2026-07-16）

- 新增默认关闭的`cost_actor_query_mode=preupdate`。每个rollout仍先用真实完整轨迹更新经验PID和constraint RMS；但第一个actor epoch改在任何本批cost-critic QR step之前，用已看过所有历史批次、尚未看当前标签的online critic生成`_risk_cdf/_risk_advantage/_risk_weight`。后续PPO epoch在critic更新后仍复用这份缓存，不再查询。
- 它与C-X1的区别是没有Polyak长滞后，与C-X2的区别是不拆分训练数据。首版显式限制为已验证的recurrent GAE-PPO、MC/raw cost target、full-batch且`s0_aux=0`；这避免把MLP n-step内部bootstrap-action RNG顺序变化混入消融。actor与critic参数完全独立，在这些限制下两个optimizer step可交换，唯一有意变量是risk-query的critic版本。
- 新增`advantage/risk_query_preupdate_postupdate_abs_mean`，在相同`(s,a,budget)`上比较缓存的QR更新前CDF与全部本批QR更新后CDF。该值与原`target_online` gap分开保存，避免把时间漂移误标为两网络差异；W&B、profile与最终JSON均已接入。
- 默认`online`精确回归使用改动前提交`baa786f`的临时worktree与当前代码：80环境步后checkpoint内44个tensor leaves全部逐元素exact（其中43个属于6个module/optimizer状态），评估与summary一致；非数值差异只有checkpoint和W&B路径。这证明重构没有暗中改变历史基线。
- 机制断言通过：两个inner updates的调用顺序精确为`A,C,C,A`；第一次actor调用时cost critic全参数仍与初始值exact，该epoch做`1+K=5`次risk query，第二个epoch新增查询为0且缓存逐元素exact；两次QR step后6个cost参数张量全部改变，pre/post CDF drift为`1.16e-8`。统一入口80步smoke也完成checkpoint、JSON和4轨迹评估、exit 0；初始低风险批次的drift极小只证明链路，不代表成熟阶段机制弱。
- 正式C-X3只在P-M3 seed1上把`online→preupdate`，保持B20/C20/N32、MC/time-weight .995、T1 sigmoid、PID target .15、8个PPO epochs、LR和1M预算全部不变。不用100k早停；C-X1已证明这类时序组件在100k reward仍可为负而后期超过基线。预计A100纯训练`6.5–8分钟`，含140与fresh 520评估后总墙钟`11–14分钟`，全程持久化后台。
- 预注册门与压力seed一致：相对P-M3 seed1的fresh520 `reward/outage=0.8622/0.3058`，候选需outage `≤0.22`且至少下降`.08`，reward `≥0.75`；同时成熟阶段pre/post drift必须非零，末200k不得出现更大的policy–critic–PID周期。通过后原参数扩seed0/2各1M+520；失败则保留为时序消融，不扫更多缓存间隔或叠加target-KL追逐偶然点。

### E70：C-X3 seed1完整1M结果——校准与安全改善，但reward代价使其不进多seed（2026-07-16）

- job `DQCAC_DynamicButton_cx3_preupdate_timew995_pi_target015_smoothT1_b20_ckpt100k_1m_s1`、W&B `xcv8ghw6`正常exit 0；纯训练`390.9s`，与P-M3的约6.5分钟一致，没有增加可测wall-time或显存代价。fresh520 eval-only job也exit 0。
- 完整长跑再次证明不能以中期终点替代结论。0–300k的P-M3/C-X3 reward为`0.2571/0.3015`、outage同为`0.1033`；300–600k为`reward 0.7835/0.7775,outage 0.2433/0.2233`，候选看似略安全且reward持平。但600–800k已变为`0.8233/0.7556,0.260/0.265`，末200k更恶化为`0.8253/0.6130,0.180/0.255`，lambda从`0.2206→0.3203`。若600k停掉会得到false positive。
- pre/post query drift在全程的mean/std/max为`0.09684/0.05287/0.22251`，末200k均值`0.10407`，最终训练JSON为`0.12024`。它与同批outage、lambda、prequential s0 CDF error的Pearson相关分别为`0.626/0.559/0.669`；这是描述性共振，不解读为单向因果，但证明当前批QR更新会大幅移动actor风险信号。
- 内置140条为reward/outage `0.57035/27÷140=0.19286`，hard CDF `0.24487`。严格fresh520中，P-M3→C-X3的reward为`0.86220±0.58095→0.58645±0.61862`，差`-0.27575`（`-32.0%`），近似95%区间`[-0.34869,-0.20281]`；outage为`159/520=0.30577→108/520=0.20769`，差`-0.09808`（`-32.1%`），保守Newcombe 95%区间`[-0.17164,-0.02307]`。两个方向都超出520回合抽样噪声。
- 校准改善是实质性的。hard CDF absolute error从`|0.10427-0.30577|=0.20150`降至`|0.26929-0.20769|=0.06160`，改善`69.4%`；smooth CDF error `0.19837→0.06670`，改善`66.4%`；predicted mean-cost error `5.74198→1.00415`，改善`82.5%`。真实cost mean/Q80也从`12.3308/18`降到`10.1865/15`。
- 预注册裁决：outage `≤0.22`且降低至少`.08`两项通过，但reward `0.58645<0.75`失败，末200k周期也比baseline大。因此C-X3保留为“切断同批反馈能换来更准critic和更安全policy，但会明显压低reward”的机制消融；不扩seed0/2、不扫缓存间隔、不叠加target-KL。
- 正式数据和图位于`_runs/wandb_export/dqc_pm3_seed1_vs_cx3_preupdate_1m_2026-07-16/`和`_runs/profiles/dqc_pm3_seed1_vs_cx3_preupdate_1m_2026-07-16/`；后者含`phase_comparison.csv`、`eval520_comparison.csv`、`eval520_summary.json`、`overview.png`及`eval520_comparison.png`（2700×1350，PIL解码通过）。


### E71：C-IQN1 cost-only uniform IQN 实现、回归与冻结策略校准预注册（2026-07-16）

- 完成度复核：QCPO 的未修正 rollout reuse 与未更新 observation RMS 两个确定性 bug 已修；Q-B 在300k终评 reward=1.621，已回到 QCPO_refs 的正常量级。QCPO 和 DQCAC 的 [512,512]+LSTM512 都完成全尺寸100k/300k验证；DQCAC recurrent 300k终评1.442，高于同结构 QCPO 的1.251。当前主缺口已从网络接线转为 cost distribution 泛化校准和 policy-critic-PID闭环。
- 新增 cost_distribution_model=qr|iqn，默认 qr 保持旧结果。IQN 只替换 action-conditioned cost critic；actor、共享 recurrent reward-V、reward critic、PPO、PID 和 observation normalization 均不变。网络为 (s,a) MLP feature × cosine(πiτ) embedding → scalar quantile head；训练每个 transition 随机采32个 uniform τ，CDF/mean/std 查询使用128个确定性 midpoint τ。
- 概念边界：IQN不是“直接输出CDF”。它估计连续 quantile function Q(s,a,τ)；P(C≥b) 仍由 uniform τ 上的 indicator 或 sigmoid 积分得到。当前先做 uniform-IQN；只有它能提高独立泛化校准，才考虑 query-mixture IQN 或查询附近自适应采样，避免再做没有作用链证据的小网格。
- IQN τ 使用独立 torch.Generator，seed=training_seed+104729。测试证明一次 [5,7] τ 采样前后全局 torch RNG state 完全相同，重置专用 generator 后 τ 逐位复现；所以 critic 采样不会偷偷改变策略动作噪声。
- 数学/工程验证：
  1. IQN default query [6,17] 与显式传入同一 midpoint τ 逐位相同；随机 [6,11] τ 输出具有非零 τ 方向标准差0.11086，所有参数梯度有限。
  2. per-transition 2D τ quantile-Huber loss 与手算误差5.96e-8（float32求和次序），反向梯度有限。
  3. 同一模型/τ/target/time-weight 下，M=11、chunk=4 的 full/chunk loss 误差1.73e-7，最大参数梯度误差2.98e-8。
  4. 对旧提交0c58963做配对短训练：34个 module tensor leaf、lambda tensor和全部runtime逐位相同；summary差异仅为新增IQN元数据。默认QR训练语义未改变。
  5. CPU端到端 IQN smoke（Ntrain=8,Nquery=17,cos=8,chunk=17）训练8.2s、exit0；checkpoint eval-only恢复exit0，随机评估逐项复现。
  6. P-M3 LSTM512 policy-only + CUDA IQN smoke训练6.6s、exit0，覆盖recurrent rollout、CUDA专用τ generator、chunk update、checkpoint和recurrent评估。
- 第一次CUDA smoke不计为算法结果：为了省时把horizon从1000缩至32，却遗漏源配置n_step=100，rollout后处理按空切片报维度错。已增加 n_step<=horizon 的构造期显式校验，并用n_step=8原样重跑通过；该问题与IQN数值无关。
- 评估协议同时修正：任何agent训练/构造完成后、独立评估开始前，都用 seed+777 重置host/CUDA动作RNG；环境worker本来就用seed+777。不同critic参数量不再通过“初始化消耗了多少随机数”污染随机策略评估。通用评估器也改为调用 cost critic 自身的加权CDF/分布矩接口，修正非均匀QR网格过去被简单mean误报的问题。
- 新增 monotonicity 诊断 cost_quantile_crossing_fraction：在升序查询τ上统计 Q(τ[j+1])<Q(τ[j]) 的比例。IQN不强制单调，这个指标用于区分“更密查询”与“高频crossing造成的伪分辨率”。

#### C-IQN1 冻结成熟策略校准门

- 数据源固定为 P-M3 seed1 的1M final actor，原fresh520 reward/outage=0.8622/0.3058；校准rollout seed=101。QR和IQN都只恢复actor与两套observation RMS，critic/target/optimizer从头初始化，policy、dual和RMS完全冻结。
- 两条均使用 B20、T1000、15 iterations=300k、20 critic updates/rollout、MC cost target、risk-discount=.995、critic [256,256]、lr=1e-3、Ntarget=32和chunk=2500。两条都设 advantage_norm=separate，使冻结实验不执行无用的K动作risk-baseline查询；因此每轮behavior数据只由同一个成熟策略和相同action RNG生成。
- 唯一算法差异：baseline为固定 QR-N32；candidate为 uniform-IQN，Ntrain=32、Nquery=128、cosines=64。内置140条只作screen；训练后若方向合格再从两个final checkpoint做同seed、同动作RNG的fresh520配对评估。
- 预注册晋级门：
  1. QR/IQN每批真实reward/cost/outage必须逐点相同，否则先判定配对协议失败；
  2. 最后5批prequential CDF absolute error或Brier至少改善20%，另一项不得恶化超过10%；
  3. 独立评估的hard-CDF absolute error或mean-cost absolute error至少改善25%，另一项不得恶化超过10%；
  4. crossing不能比QR恶化超过0.10，且末段无发散。
- 只有140 screen和fresh520都通过，才在P-M3压力seed1进入一次完整1M live闭环；否则停止standard uniform-IQN，不扫cosine数、Ntrain或Nquery碰运气。query-mixture IQN作为有分歧的独立路线保留，不和uniform-IQN同时修改。
- 既有QR冻结300k纯训练121s。新配对QR预计约2–3min训练、加140评估总约4min；IQN因τ embedding和chunk前向预计训练4–7min、总约6–10min。两条串行总墙钟约10–14min，全程由launch_background.sh持久化，输出只写/vepfs项目盘。

### E72：C-IQN1 300k 初筛与训练长度审计预注册（2026-07-17）

- QR/IQN 正式 300k job 均正常 exit 0，纯训练分别为 `121.88s/125.73s`，W&B run 为 `qrwjysnw/20f7gtvr`。15 批真实 reward、outage、cost、holdout truth 逐点 exact，140 条终评的 reward/outage/mean cost 也 exact 为 `0.86576/0.25714/11.75714`；因此差异没有混入策略或环境随机性。
- 300k 独立终评中，QR→IQN 的 hard-CDF absolute error 为 `0.14442→0.14777`，mean-cost absolute error 为 `3.15970→3.54291`，crossing 为 `0.04862→0.20079`。三项均未通过预注册门，其中 mean error 恶化 `12.1%`，crossing 绝对增加 `0.15217`。
- 最后 5 批 prequential CDF absolute error 为 `0.10844→0.11586`，IQN 恶化 `6.8%`；Brier 为 `0.21388→0.21350`，只改善 `0.17%`，远低于 20% 门。post-CDF error 为 `0.04438→0.06117`，IQN 仍差，但逐批差值从 220k 的约 `+0.0250` 缩至 300k 的 `+0.00391`；这留下了“复杂 IQN 只是收敛更慢”的有限可能。
- 本次不改写 300k 预注册裁决：uniform-IQN 已失败，不能凭延长后偶然好转直接晋级 live 1M。考虑用户提出的短训练 false-negative 风险，以及 post-CDF 相对差距确有收窄，只增加一次独立的**训练长度审计**：同 seed101 从头严格配对重跑 QR/IQN 到 600k，前 300k 应复现实验，后 300k 只回答慢收敛问题；不扫描 Ntrain、Nquery、cosines 或学习率。
- 600k 审计只有在后 5 批 prequential 指标和独立 140 条评估都达到原门、且 crossing gap 回到 `≤0.10` 时，才允许 fresh520；否则停止 standard uniform-IQN。若 600k 只是两者近似持平，它最多说明 IQN 没有明显坏处，不构成增加复杂度或进入 live 闭环的理由。
- 按 300k 实测线性外推，两条各约 `252s` 纯训练；两条并行、加 140 条独立评估预计总墙钟 `5～7min`。仍由 `launch_background.sh` 持久化启动，checkpoint、W&B 与结果文件全部写入 `/vepfs-mlp2/c20250510/251204033/` 下，不占 20G 根盘。

### E73：C-IQN1 600k 长度审计——300k 对 critic 偏短，但 uniform-IQN 仍未过门（2026-07-17）

- QR/IQN 600k job 均正常 exit 0，W&B run 为 `8ry7xn6g/7llo7019`；并行纯训练 wall time 为 `417.21s/433.87s`。两条前 300k 的 reward、outage、cost、pre/post CDF/Brier/mean bias 与 crossing 对原 300k history 全部逐值 exact，证明长度审计没有更换初始化或移动原裁决。
- 600 条校准轨迹中两条真实行为逐批 exact，共有 `167/600` 个超阈值事件。独立 140 条评估也严格配对，truth reward/outage/mean cost 均为 `0.86576/0.25714/11.75714`。
- 训练长度本身影响巨大：QR 的 CDF/mean-cost absolute error 从 300k 的 `0.14442/3.15970` 降至 600k 的 `0.01071/0.77573`，分别改善 `92.6%/75.4%`；IQN 从 `0.14777/3.54291` 降至 `0.00898/0.62543`，分别改善 `93.9%/82.3%`。因此 300k 足以做初筛，却不足以把冻结 distributional critic 的绝对校准误差当作收敛值。
- 同预算比较中，600k IQN 的独立 CDF error 比 QR 小约 `16.1%`，mean-cost error 小约 `19.4%`，出现小到中等优势；但均低于原定 `25%` 门。最后 5 批 prequential CDF/Brier 只改善约 `3.0%/0.9%`，post-CDF 约改善 `5.0%`，post-Brier与mean error略差，没有持续扩大优势。
- IQN 的 monotonicity 仍明显更差：600k 独立 crossing 为 `0.23397`，QR 为 `0.05737`，差 `+0.17660`；最后 5 批差仍为 `+0.15041`。它不是多训练即可消失的早期毛刺。
- 按预注册规则，不做 fresh520，不启动 uniform-IQN live 1M，也不扫 `Ntrain/Nquery/cosines`。定级为：`IQN 在足量固定策略数据下可能带来约 16%～19% 的校准收益，但当前非单调输出和收益幅度不足以证明值得进入非平稳 actor–PID 闭环`。
- 最重要的算法含义不只是 IQN 输赢：当前 live B20 每 20 条轨迹就改变一次 policy，而固定 policy 的 critic 需要数百条轨迹才接近校准；更多同批 gradient steps不能制造新的独立 tail trajectory。下一优先验证应增加每个 policy 版本的独立轨迹数或减慢 actor 更新频率，首选固定 1M 总步的 `B20→B40`，而不是继续增加同一批 critic epoch。
- 正式 history/图/表位于 `_runs/wandb_export/dqc_frozen_qr32_vs_iqn32q128_600k_lenaudit_2026-07-17/` 与 `_runs/profiles/dqc_frozen_qr32_vs_iqn32q128_600k_lenaudit_2026-07-17/`；端点图 `length_audit_endpoint_comparison.png` 为 `2356×748`，PIL 解码验证通过。

### E74：P-M6 B40 长预算预注册——增加每个 policy 版本的独立轨迹（2026-07-17）

- 以 P-M3 seed1 为压力基线，只把 `num_envs=20→40`、`num_iterations=50→25`，总环境步与总完整轨迹仍严格为 `1M/1000`。网络、QR-N32、20次critic update、8次PPO epoch、actor LR、PID、T1 smooth CDF、cost time weighting与所有seed均不变。
- B40每次policy更新有40条独立初始状态轨迹，outage=.2时预期tail事件由4增至8；policy/dual更新次数由50降至25。每次batch翻倍但update次数减半，固定1M下actor与critic看到的总transition-pass不增加，检验的是更低batch方差、更慢policy漂移和更多同版本tail样本。
- `pid_reference_episodes=10`保持不变。代码按episode_scale和几何leak缩放，因此常值error下一次B40积分更新严格等价于两次B20，不会因batch变大暗改每episode的I增益；`pid_window_episodes=50`仍是同样的50条轨迹窗口。
- 不用300k早停。P-M1/C-X1已证明慢变量可在100k～600k改变方向，P-M6除NaN/OOM/确定性错误外完整跑1M，并每5个iteration（200k）保存phase checkpoint。内置160条只作screen；通过才做统一`num_envs=20`的fresh520。
- 压力seed1 fresh520门：相对P-M3的`reward/outage=0.8622/0.3058`，候选需outage `≤0.22`且至少下降`.08`，reward `≥0.75`；同时末200k不出现更大lambda/outage周期。通过才原配置扩seed0/2，否则停止B40，不扫B30/B50/B60。
- A100 80GB对B20只占少量显存，B40+N32预计安全；单条独占训练预计`7～9min`，内置160约1min，若晋级fresh520约5min。全程由`launch_background.sh`持久化，输出写入`/vepfs`。

### E75：P-M6 内置screen通过与fresh520协议校正（2026-07-17，评估启动前）

- P-M6 seed1 1M训练和160条内置评估正常exit 0；内置reward/outage为`0.93045/34÷160=0.2125`，通过screen。critic CDF为`0.08320`，相对truth仍低估`0.12930`，所以必须做fresh520。
- E74写成统一`num_envs=20`是启动前发现的记录错误。既有P-M3 seed1正式fresh520命令实际为`num_eval=512,num_envs=40`，得到严格520条；为复用完全相同的并行布局与动作RNG协议，P-M6也使用`512/40`。这是查阅既有run.sh后的协议校正，不改变checkpoint、门槛或训练结果。
- fresh520仍只做eval-only，门保持outage`≤0.22`且相对P-M3降低至少`.08`、reward`≥0.75`；不根据内置160调阈值。

### E76：P-M6 B40 结果——方向有益但未过安全门，160条产生false positive（2026-07-17）

- P-M6 1M训练`398.24s`、W&B `ksw0hwvv`、训练与内置评估均exit 0；fresh520 eval-only也exit 0。B40+N32实测显存约3.5GB，无OOM/NaN，证明当前128 CPU/A100 80GB可承载。
- B20→B40的分段reward/outage为：0–300k `0.2571/0.1033→0.0237/0.0179`，300–600k `0.7835/0.2433→0.4718/0.1406`，600–800k `0.8233/0.2600→0.6534/0.1800`，800k–1M `0.8253/0.1800→0.8034/0.1450`。B40明显减慢早期reward学习，但后段追上并降低训练outage。
- 末200k lambda/KL/clip从P-M3的`0.22057/0.002459/0.12929`降到`0.03244/0.001222/0.05425`。更大batch与更少policy update确实减小闭环振幅和PPO位移，不是无作用改动。
- 内置160条给出reward/outage=`0.93045/34÷160=0.2125`，看似通过；同一final checkpoint的正式fresh520为`0.92628±0.41427`、`137/520=0.26346`。因此160条安全结论是false positive，不能据此扩seed。
- 同协议P-M3 seed1为`0.86220±0.58095`、`159/520=0.30577`。B40 reward增加`0.06408`（约7.4%，非配对近似95%区间`[0.00276,0.12541]`）；outage下降`0.04231`，保守差值95%区间`[-0.11924,0.03525]`仍跨0。B40的Wilson区间为`[0.22743,0.30296]`，下界已高于0.22门。
- critic仍系统低估：hard-CDF absolute error只从`0.20150`降到`0.17710`（改善12.1%），mean-cost error从`5.74198`降到`4.84288`（改善15.7%）。增加每次policy版本的轨迹数缓解但没有解决initial-state泛化。
- 预注册裁决：reward门通过，但outage既未到`≤0.22`、也未下降`.08`；不扩seed0/2，不扫B30/B50/B60。B40定级为“降低更新方差、略提高reward并略降风险，但不足以成为主配置”。
- 下一路线不再单纯增大B或重复同批QR epoch。优先实现直接查询点exceedance/CDF critic：用MC remaining-cost与remaining-budget的二元标签直接拟合`P(C_remaining≥budget|s,a)`，先在固定policy 600k配对校准；只有跨状态CDF泛化显著优于QR，才进入live闭环。
- 正式history/profile在`_runs/wandb_export/dqc_pm3_b20_vs_pm6_b40_seed1_1m_2026-07-17/`与`_runs/profiles/dqc_pm3_b20_vs_pm6_b40_seed1_1m_2026-07-17/`；后者含phase/eval CSV、置信区间JSON和`1872×1277`的fresh520比较图。


### E77：训练长度重新分级——短跑只筛机制，不能裁决最终性能（2026-07-17）

- 用户质疑“当前训练是否普遍太短”。结论是：**对工程正确性和明显机制失效，现有短跑够用；对网络结构、慢critic、PID闭环和最终算法性能，100k/300k明显不够，1M也只是压力筛查而非论文级终局**。QCPO_refs正式配置为`runner.n_steps=5e6`；因此目前任何1M结果都不能支持“DQCAC最终优于/劣于QCPO_refs”的论文结论。
- 以当前`B20,T1000`计，100k/300k/600k/1M分别只有5/15/30/50个policy版本；B40的1M更只有25次policy/PID更新。LSTM、IQN、更大batch或更慢actor更新在相同env steps下天然启动更慢，不能只看300k reward点估计判死。
- 已经观察到双向反例。冻结critic的IQN在300k比QR更差，到600k反而在独立CDF/mean error上好约16.1%/19.4%；B40在0--300k reward仅0.0237，到末200k追到0.8034且fresh520 reward超过B20。反方向上，C-X3在300--600k reward几乎持平且更安全，到1M末200k reward却降至0.6130；600k停掉会产生false positive。故“前半段不好就停”和“中段好就晋级”都不可靠。
- 既有结论重新定级：确定性代码bug、NaN/OOM、错误梯度/错误标签等仍可短跑裁决；C-X3的1M大幅reward代价和IQN持续crossing足以阻止当前版本晋级，但只表示“当前预算/当前实现不过门”，不等于它们在无限训练下绝无改善。`MLP vs LSTM`的300k、`N32 vs N64/局部tau`的100k--300k、早期PID组合和B40的渐近性能均改标为“早期样本效率证据”，不能作为最终排序。
- 新分层协议：smoke只验证全链路；300k只看机制方向和是否值得投入，不以reward单独判败；冻结policy critic至少600k，并比较最后5批与此前5批，若关键误差仍改善超过10%则预注册延到1.2M；live actor--critic--PID候选至少完整1M、每200k看phase、最终fresh520，除数值/确定性机制错误外不早停；1M仍趋势上升或结果接近门槛者扩到2M。最终入论文的DQCAC/QCPO/QCPO_refs按相同5M预算、至少3 seeds和统一独立评估比较。
- 随机初始化处理：单seed中小差异不再定论。共享actor的critic消融继续使用common-random-number配对；差异小于约20%或置信区间跨0时至少补3个初始化/seed。只有效应巨大、方向在多个phase一致且直接机制指标也恶化，才允许单压力seed停止。
- 复验优先级不是把所有旧组合全部5M重跑。先做直接query-point CDF head的冻结600k--1.2M门；通过后做live 1M。与此同时，最终网络公平性需要把DQCAC的MLP/LSTM最佳配置各跑完整1M；N64/局部tau只有在QR仍是主CDF表示时再做严格配对600k。B40保留为可与有效CDF head组合的减振组件，不因单独1M不过安全门永久删除，也不立即盲扫B30/B50/B60。
- 本条记录时没有活动训练进程；下一条正式实验仍只通过`launch_background.sh`持久化启动，预计时长在启动前记录，输出继续写入`/vepfs-mlp2/c20250510/251204033/`。


### E78：C-DCF1直接超阈概率critic实现、回归验证与600k预注册（2026-07-17）

- C-DCF1已在提交`4b70914`实现，新增`cost_cdf_estimator=quantile|direct`，默认`quantile`。direct不是另一种quantile网络，而是额外的action-conditioned Bernoulli head，输入`(cost_input, action, remaining_budget/cost_limit)`，直接输出`P(C_remaining>=budget|s,a,budget)`；现有QR-N32继续完整训练并报告mean/std/quantile/crossing，作为同run内部对照。
- 监督来自`label_t=1{mc_cost_to_go[t]>=budget[t]}`。由于`budget[t+1]=(budget[t]-cost[t])/gamma_c`，该事件与初始整轨迹超限严格等价；网络仍使用全部transition，因为条件`(s_t,a_t,b_t,t)`随时间改变，正是actor逐步查询的区域。新增`cost_direct_cdf_label_inconsistency_fraction`逐批检查同轨迹标签，必须恒为0。
- head为与cost QR相同宽度的ReLU MLP，最终输出单logit；budget固定除以15，BCEWithLogits保证极端概率数值稳定。不使用正负类重加权，因为class-weighted BCE会改变概率校准目标。它复用QR的risk-discount transition weight和chunk边界，但使用独立Adam、独立grad clip、独立一次optimizer step，不进入reward/QR joint norm或clip。
- direct首次只允许`QR + uniform grid + MC + raw cost history + online query + s0_aux=0`。这是归因约束，不是永久API限制：暂不与IQN、query-mixture、crossfit、target/preupdate或cost-LSTM叠加。MLP与`[512,512]+LSTM512` policy都已接通；direct会替换actor实际动作CDF、K动作baseline、constraint RMS和critic-dual查询，经验PID仍按真实轨迹工作。
- 开启额外网络前后保存/恢复host与CUDA RNG。构造测试表明quantile/direct的actor、reward/cost online/target及normalizer共34个tensor leaves逐位相同，joint critic参数组数量相同，构造后torch RNG逐位相同；direct只有78,593个参数，不会因初始化消费改变首批动作。
- 数学/工程验证：网络支持标量、`[B]`和`[B,1]` budget，BCE梯度全部有限；M=15、risk-weighted的full/chunk=4 loss为`0.6932367086/0.6932367027`，一次Adam后最大参数差`1.75e-10`，标签不一致率0。对旧提交`63b338d`做持久化配对训练，final checkpoint共34个module tensor leaves、lambda/runtime、全部旧eval键和47个共享summary键逐项exact；新增内容仅是direct配置/QR对照元数据与Brier。
- 端到端持久化smoke均exit 0：MLP CPU训练9.1秒；全尺寸`[512,512]+LSTM512` CUDA训练9.9秒，覆盖GAE-PPO两epoch的固定risk cache、observation normalization、chunk critic、checkpoint与循环评估。direct checkpoint保存7个state tensor，eval-only恢复后的旧eval字段逐项exact。短T=32只有0-cost样本，direct仍约0.42只是4次更新后的未收敛先验，不作为算法结果。
- 终评新增逐状态proper score：`cost_cdf_brier_initial`比较selected estimator与每条真实outage标签；direct时同时报告`cost_cdf_qr_initial/qr_brier_initial`。这避免“两个模型总体CDF均值相同，但逐状态排序完全错误”被均值误差掩盖。prequential pre/post也在相同真实a0上同时记录direct主键与`qr_*`内部对照。
- 正式C-DCF1复用P-M3 seed1的1M成熟策略和rollout seed101。既有QR600k run `8ry7xn6g`、checkpoint `dqc_frozen_s1policy_seed101_ciqn_qr32_600k_lenaudit_20260717`作为外部基线；direct 600k仍在同一run同步训练完全相同的QR，所以可以额外验证30批behavior truth及QR权重/指标是否保持exact，不重复浪费一条QR训练。
- 600k门：真实reward/cost/outage的30批history必须与既有QR逐值一致，所有标签不一致率为0且无非有限值；最后5批prequential direct的CDF absolute error或Brier相对同run QR至少改善20%，另一项不得恶化超过10%；独立140条的CDF absolute error或Brier至少改善25%，另一项不得恶化超过10%。140通过后才从同一final checkpoint做fresh520；520也通过才允许P-M3压力seed进入live 1M。
- 若600k未过效果门但direct关键误差从前5批到末5批仍改善超过10%、且相对QR方向一致，则按E77训练长度规则从头预注册1.2M审计；若只近似持平或后段已平台，则停止当前direct，不扫学习率/隐藏层碰运气。通过live后再决定是否与B40减振组合，不能在第一条run同时打开两个变量。
- 既有QR600k纯训练417.2秒。direct多一个仅78.6k参数的scalar head，结合smoke开销预计纯训练8--10分钟、含140终评总墙钟9--12分钟，显存增量远小于QR pairwise loss；正式实验只通过`launch_background.sh`，checkpoint/W&B/log全部写入`/vepfs`。

### E79：C-DCF1 600k结果——不是训练太短，而是head追逐上一批噪声（2026-07-17）

- 正式持久化job正常exit 0，W&B run为`2j7wdmq4`；30批、600条完整轨迹、600k环境步全部完成，纯训练`245.2s`，随后完成140条独立评估。final checkpoint为`_runs/checkpoints/dqc_frozen_s1policy_seed101_cdcf_direct_600k_20260717/final_post_update.pt`。
- 严格配对通过：相对既有QR run `8ry7xn6g`，30批的env step、reward、reward quantile、outage、discounted/undiscounted cost全部逐值exact；direct标签不一致率最大值为0。两个final checkpoint的actor、QR cost online/target、reward online/target和observation normalizer共43个共享state tensor逐位exact，说明新增head没有改变行为或QR基线。
- 末5批prequential CDF absolute error为direct/QR=`0.05169/0.10656`，direct看似改善51.5%；但Brier为`0.23360/0.23253`，direct反而差0.46%。post-update CDF error为`0.03114/0.03750`，只改善17.0%；post Brier为`0.19186/0.20023`，只改善4.2%。proper score没有支持“条件风险排序显著变好”。
- 独立140条truth outage为`0.25714`。direct预测`0.32722`，absolute error=`0.07008`；同run QR预测`0.26786`，error=`0.01071`。direct的总体CDF误差约为QR的6.54倍。direct/QR Brier为`0.19937/0.20282`，direct只改善1.70%，远低于25%门。故不做fresh520、不进入live 1M。
- 末段CDF“改善”不是可靠的慢收敛证据。direct pre预测与当前新批truth的相关仅`0.0196`，却与上一批truth相关`0.8256`；同批更新后的post预测与当前truth相关`0.8844`，一次rollout内20个head update使概率平均移动`0.1070`。QR对应的pre-lag相关为`0.6192`、平均移动`0.0829`，也有追批现象，但明显更弱。
- 最后5批truth均值恰好由此前5批的`0.23`升到`0.33`。direct复制上一批比例时刚好撞上连续高outage区间，才产生末5批CDF error下降；独立评估分布回到`0.257`后，过估立刻暴露。因此该结果不满足E78“末段改善且相对QR方向一致”的1.2M延长条件；不运行无改动的1.2M，也不事后移动门槛。
- 对用户“短跑是否误杀”的回答进一步细化：IQN 300k→600k属于真实慢收敛，应延长；C-DCF1则是跨批遗忘/高方差更新，保持同配方增加步数只会继续追逐最近20条轨迹。后续只允许针对已定位机制的单变量改动：首选每个rollout后更新一次EMA query head，让actor/eval读取跨批低通版本；备选是跨rollout replay或减少direct update次数。三条不能同时打开，均需重新预注册并从600k固定策略门开始。
- 完整history/profile位于`_runs/wandb_export/dqc_frozen_direct_cdf_600k_2026-07-17/`与`_runs/profiles/dqc_frozen_direct_cdf_600k_2026-07-17/`；`overview.png`为`2880×3440`且PIL解码通过。正式checkpoint/history/profile合计约12.6MB，全部位于`/vepfs`。


### E80：C-DCF2 EMA direct-CDF实现与600k预注册（2026-07-17）

- C-DCF2已在提交`80b67b1`实现。新增`cost_direct_cdf_query_mode=online|ema`与`cost_direct_cdf_ema_tau`，默认`online/0.005`；默认不会构造额外网络。online head仍按C-DCF1对每批做20次BCE Adam step，EMA head不进optimizer，只在每次online step后执行`θ_ema←(1-τ)θ_ema+τθ_online`。actor、prequential和评估读取selected EMA，online只作内部对照。
- `τ=0.005`沿用项目target网络的标准Polyak量级，不扫描。20次更新对应每rollout有效新权重`1-(1-.005)^20=0.0953895`，约10个rollout的低通记忆；600次head step后初始参数残余约`(1-.005)^600=0.0494`，故600k已覆盖约3个时间常数，不会因EMA天然慢热而只给极短预算。
- EMA通过`deepcopy(online)`构造，step 0参数逐位相同且不消耗RNG；`requires_grad=False`，checkpoint自动保存独立`cost_exceedance_ema_critic`。日志同时记录selected EMA、online direct、QR、EMA-online参数差、pre/post CDF/Brier和独立评估proper score，能区分“online仍追批但EMA稳定”与“两者都没学到”。
- 回归验证通过：新默认online相对C-DCF1旧smoke的41个checkpoint tensor及所有共享eval/summary数值逐项exact；online/EMA两条在首批更新前的41个共享tensor exact，EMA初始7个state tensor与online exact。解析小网络20次更新得到`0.0953895`，最大误差`7.45e-9`。
- 持久化CPU online、CPU EMA、EMA eval-only与全尺寸`[512,512]+LSTM512` CUDA smoke均exit 0；后者覆盖GAE-PPO、observation normalization与循环评估。EMA checkpoint共8个module，EMA/online最大参数差已非零，eval-only全部评估字段逐项exact。短T=32全为0-cost，只验证链路，不评价算法。
- 正式C-DCF2仍用P-M3 seed1成熟策略、rollout seed101、B20、T1000、30批=600k、20次critic update、MC/raw、risk-discount .995、chunk2500；唯一相对C-DCF1的变量是`query_mode=ema,tau=.005`。同run online head和QR必须与C-DCF1 checkpoint逐位exact，30批behavior truth逐值exact，标签不一致率0且无NaN/Inf。
- 机制门：末5批selected EMA的prequential CDF error或Brier相对same-run online至少改善20%，另一项不得恶化超过10%；selected的平均post-pre概率漂移需不高于online的50%，且对上一批truth的滞后相关不能高于online。主效果门保持不变：末5批selected相对QR至少一项改善20%、另一项不恶化超过10%；独立140条至少一项改善25%、另一项不恶化超过10%。
- 只有140通过才做fresh520，520通过才允许live 1M。若600k未过主门，只有在selected相对QR的末段与独立评估方向一致、且关键prequential误差从此前5批到末5批仍改善超过10%时，才允许一次从头1.2M长度审计；否则停止EMA005，不扫tau，也不同时加入replay/减少update。
- C-DCF1纯训练245.2秒。EMA多一个78.6k参数的无梯度forward/lerp，预计纯训练4.5--5.5分钟、含140终评总墙钟5.5--7分钟；正式任务只由`launch_background.sh`持久化，checkpoint/W&B/history继续写入`/vepfs`。


### E81：C-DCF2 600k结果——EMA修复追批，但没有超过QR（2026-07-17）

- 正式持久化job正常exit 0，W&B run为`z5f8b2zb`；600k纯训练`245.3s`，与C-DCF1的245.2s近似相同，随后完成140条评估。30批behavior reward/cost/outage逐值exact，标签不一致率0；相对C-DCF1，actor、online direct、QR online/target、reward online/target及normalizer共50个共享checkpoint tensor逐位exact，runtime和env_steps也exact。唯一新增权重是7个EMA state tensor。
- EMA确实完成预定的稳定化机制。prequential预测对上一批truth的相关由online的`0.8256`降到`-0.0084`；全程平均post-pre概率漂移由`0.10701`降到`0.00717`，比例仅6.7%，末5批为`0.00150/0.03079`。因此“减少追逐最近20条轨迹”不是失败点。
- 机制收益幅度仍未过门。末5批pre CDF error为EMA/online=`0.04471/0.05169`，只改善13.5%；Brier为`0.22398/0.23360`，只改善4.1%，都低于20%。EMA相对QR的末5批CDF改善58.0%，但Brier只改善3.7%，说明总体概率更平滑，不代表逐状态条件风险排序明显更准。
- 独立140条truth仍为`0.25714`。EMA/online/QR预测分别为`0.33237/0.32722/0.26786`，CDF absolute error为`0.07523/0.07008/0.01071`；EMA在总体校准上甚至略差于online，约为QR误差的7.0倍。Brier为`0.19363/0.19937/0.20282`，EMA相对online/QR只改善2.9%/4.5%，远低于25%独立门。
- 不做1.2M：末5批EMA CDF相对QR更好，但独立CDF更差，方向不一致；EMA Brier从此前5批`0.20194`恶化到末5批`0.22398`，没有“仍以>10%速度改善”的证据。600次EMA step后初始化只剩约4.9%；即使把`0.5`先验残余的量级粗略扣除，仍不足以把0.332拉到比QR 0.268更准。继续同配方主要消耗预算，不满足E80延长条件。
- 裁决：EMA005保留为“成功减少更新方差、但未改善到足以替代QR”的正机制/负性能消融；不做fresh520、不进live 1M、不扫tau。replay或减少direct update仍作为有分歧的可选路线记录，但两版direct均未在独立proper score上接近25%收益，优先级降到QR主路线和actor更新频率之后。
- 这再次回答训练长度问题：300k时EMA仍约0.41，确实太短；完整600k后才能看到它稳定到0.33附近。但“更长才看清”不等于“更长会成功”。当独立评估、proper score和末段趋势不一致时，继续延长会放大选择性报告风险。
- 正式history/profile在`_runs/wandb_export/dqc_frozen_direct_cdf_ema005_600k_2026-07-17/`与`_runs/profiles/dqc_frozen_direct_cdf_ema005_600k_2026-07-17/`；`overview.png`为`2880×3440`、约1.2MB且PIL解码通过，checkpoint约12MB，均在`/vepfs`。


### E82：C-Q3C/C-Q4C N64与局部quantile的600k长度审计预注册（2026-07-17）

- 重新开放原因不是移动旧100k门，而是E77明确把N64/local旧结论降级为“无早期收益”；direct两条600k未能替代QR后，QR仍是actor主CDF表示。现在用同一成熟固定策略补足600k，只回答“更多独立轨迹后表示是否出现稳定泛化收益”，不直接进入policy/PID。
- 共同基线为QR-N32 run `8ry7xn6g`：P-M3 seed1的1M actor、rollout seed101、B20、T1000、30批=600k、MC/raw、risk-discount .995、20次critic update、chunk2500、独立140条评估。既有truth为reward/outage/mean cost=`0.86576/0.25714/11.75714`，QR hard-CDF=`0.26786`，absolute error=`0.01071`，Brier=`0.20282`。
- C-Q3C只把`num_quantiles=32→64`，并启用`quantile_target_reduction=reference_mean,reference_samples=32`，消除旧N64目标数翻倍导致的梯度尺度混杂；uniform τ和其他配置不变。当前实现会同时把reward/cost distributional head改为64，但固定策略下reward critic不影响行为；joint clip效应属于该真实N64算法配置的一部分，后续若有收益再考虑cost-only N64解耦。
- C-Q4C保持N32，只把cost τ网格改为uniform+local mixture：中心`τ*=0.8`、半宽`0.1`、local fraction`0.5`，prediction使用importance weighting保持全局uniform-W1目标，CDF/mean使用对应quadrature权重。使用importance而非旧query-focused，是因为100k已证明后者放大cost gradient并长期clip，无法把收益归因于局部分辨率。
- 两候选在policy-only恢复后统一重置host/CUDA RNG；网络大小不同也必须得到30批完全相同的reward/cost/outage。标签、非有限值、target scale、joint/cost grad和crossing全部记录。N64与local同时后台运行，但各自独立进程/optimizer/W&B，不共享状态；A100 80GB和128 CPU可承载B20×2。
- 600k门：末5批prequential CDF absolute error或Brier相对QR至少改善20%，另一项不得恶化超过10%；独立140条的CDF error、Brier或mean-cost absolute error至少一项改善25%，其余关键项不得恶化超过10%；crossing绝对增加不得超过0.10。只有140过门才做fresh520，520仍过门才允许live 1M。
- 长度续跑规则不变：若600k未过但相对QR的末5批和独立评估方向一致，且关键误差从此前5批到末5批仍改善超过10%，只允许一次从头1.2M；否则停止当前N64/local，不扫N96/N128、local fraction或half-width。两候选不能彼此组合后再解释单变量效果。
- 既有N32 600k训练417.2秒，但当前同硬件direct run约245秒；N64 pairwise计算更大、local与N32近似。并行预计纯训练5--7分钟、含两组140评估总墙钟6--9分钟；均由`launch_background.sh`持久化，输出写`/vepfs`。


### E83：C-Q3C/C-Q4C 600k结果——短跑会误判收敛值，但两种表示均不晋级（2026-07-17）

- 两条正式持久化job均正常exit 0。N64-reference为W&B 'vcy6vawj'、纯训练'439.80s'；local-importance为W&B 'ndbi9jlj'、纯训练'434.27s'。并行共享GPU后，含两组140条独立评估的总墙钟约10分钟，略高于预估6--9分钟。没有NaN、OOM或标签异常。
- 公平性检查通过：两候选与QR32基线的30批reward、reward quantile、outage、mean cost及逐步reward/cost全部逐值exact；final checkpoint中的actor 16个state tensor、observation normalizer 3个tensor和lambda均逐位相同。N64只改变64维reward/cost distributional heads，local只改变cost tau网格/权重；因此真实行为与评估truth完全一致。
- 末5批prequential结果没有支持结构优势。QR32/N64/local的CDF absolute error分别为'0.10656/0.10797/0.10887'，N64和local相对基线恶化'1.32%/2.16%'；Brier分别为'0.23253/0.23191/0.23170'，只改善'0.27%/0.36%'，远低于20%门。N64从此前5批到末5批的CDF/Brier又恶化'9.16%/21.63%'；local恶化'13.79%/21.58%'，没有“仍以超过10%速度收敛”的延长证据。
- 独立140条truth固定为outage '0.25714'、mean cost '11.75714'。QR32的CDF/Brier/mean-error/crossing为'0.26786/0.20282/0.77573/0.05737'。N64为'0.25859/0.20021/0.87055/0.15828'：CDF error从'0.01071'降到'0.00145'，相对改善86.46%，但Brier只改善1.28%，mean error恶化12.22%，crossing绝对增加'0.10090'。后两项分别越过“其他关键量不得恶化10%”和“crossing增加不超过0.10”的预注册边界。
- local的独立CDF/Brier/mean-error/crossing为'0.27644/0.19930/0.80526/0.10023'。CDF error为'0.01929'，相对QR恶化80.07%；Brier只改善1.73%，mean error恶化3.81%，没有任一独立proper/mean指标达到25%改善门。
- N64的86%相对CDF改善不能脱离绝对量和proper score解释：它只把总体预测移动约0.0093，而140条truth的二项标准误约0.0369；基线分母本来只有0.0107，所以相对百分比被放大。若它真实改善逐状态条件概率，Brier、mean和末段prequential应至少给出同方向证据；本次恰好相反，同时monotonicity明显变差。
- 裁决：两条都不做fresh520、不进入live 1M，也不启动同seed原样1.2M。600k已经回答固定策略表示筛选，但不被写成“任何初始化下永久无效”。有分歧的备选消融保留为：未来在主线配置稳定后，用多个critic初始化各600k检验N64点估计是否复现；这比让同一初始化继续训练到1.2M更直接回答初始化偶然性。当前不扫N96/N128、local fraction/window，也不组合N64+local。
- 本轮同时修正旧100k结论的证据等级：短跑确实可能误杀慢收敛表示，N64在600k才出现很准的总体CDF点；但完整门显示“训练更久后出现一个好数”不等于算法胜出。下一主线回到critic数据速度慢于policy移动速度的问题，优先实现默认关闭的actor更新间隔消融，而不是继续增加同一批critic更新次数。
- 正式history/profile位于'_runs/wandb_export/dqc_frozen_qr32_n64_local_600k_lenaudit_2026-07-17/'和'_runs/profiles/dqc_frozen_qr32_n64_local_600k_lenaudit_2026-07-17/'；比较表为'quantile_length_audit_comparison.csv'，六面板图为'quantile_length_audit_comparison.png'（'3000×1500'，PIL解码通过）。

### E84：P-M7 actor cadence=2实现、严格回归、全尺寸验证与1M预注册（2026-07-17）

- 机制动机来自600k冻结critic长度审计：B20下每个policy版本只有20条独立轨迹，把同一批做20次QR update只能增加优化步，不能增加tail事件。P-M7新增默认关闭的`actor_update_interval=2`：cost/reward critic和经验PID仍每个B20 rollout更新，Actor参数与Actor自带observation RMS连续两个rollout冻结；收满两批后沿环境维合并成B40，只执行一次8-epoch PPO。它与P-M6直接把`num_envs=40`不同，后者连critic/PID也每40条才更新，因此两条是可解释的独立消融。
- importance sampling语义已明确实现：每个transition在采样时保存固定`log π_old(a|h)`；每个PPO epoch用当前网络重新前向得到`log π_new`，计算`r=exp(logπ_new-logπ_old)`并做PPO-Clip。更新后不需要另存“当前概率”，因为下一epoch的分子会重新计算，而分母始终是采样策略。两批收集期间Actor和`actor.obs_rms`不变，所以两批都来自同一behavior policy，不是把变过的策略数据混在一起。
- 为避免cadence改变全局动作随机流，每个B20 rollout都按历史首个actor epoch的位置预抽K组behavior baseline action；到期后只把缓存动作送入最新cost critic，不再消耗随机数。合并前把flatten字段恢复为`[T,B,...]`，沿B拼接后再flatten；`h0/c0`、old log-prob、GAE、实际动作和baseline action保持同一episode顺序。构造期暂只开放已验证的recurrent GAE-PPO、MC/raw cost、online query、`s0_aux=0`路径，并要求正式rollout数能被interval整除。
- 默认兼容回归以提交`c81d01b`的临时worktree为金样本，seed123、CUDA、`2×B2×T40=160`步。新代码在`interval=1`时的Actor、observation normalizer、reward/cost critic和两个target共6个module全部逐tensor exact，lambda tensor exact，最终评估dict exact；最大绝对差为0。唯一新增内容是cadence配置、runtime和summary诊断，证明默认值没有暗改P-M3。
- 首次`interval=2`入口smoke在训练开始前暴露`self.warmup_iters`初始化顺序错误并exit 1；已改为按正式初始化同一规则直接解析`args.warmup_iters`。修复后同名持久化作业exit 0。80→160步两个pre-update checkpoint中Actor的16个state项逐位exact，cost/reward critic已更新；160步到final时Actor可学习权重最大变化约`6.0e-4`，更新事件恰为1。Actor内的RMS只在到期后一次合并160条观测；另一个每rollout变化的`obs_normalizer`只服务raw-state critic，不参与LSTM行为策略。
- 小型smoke的首epoch ratio最大偏差为`5.05e-5`，合并4条轨迹、实际2个PPO epoch，末epoch KL=`4.53e-5`、clip fraction=0。合并顺序与baseline RNG的合成断言也逐项exact。该量级与既有interval=1 recurrent CUDA的约`0.8e-5～1.4e-5`同属批处理浮点重放误差，远小于0.1 clip范围，不是off-policy证据。
- 全尺寸40k shape/memory smoke使用P-M3 seed1的B20、T1000、LSTM512、QR32、20 critic update、8 PPO epoch，仅设`interval=2`；持久化训练22.1秒、exit 0，无NaN/OOM，观测显存约2.8GB。一次到期事件正确合并40条轨迹并完成8个epoch；首epoch ratio最大偏差`1.1444e-5`，末epoch KL=`0.005374`、clip fraction=`0.321575`。40k reward/outage只属初始化阶段，不用于算法裁决。
- 正式P-M7预注册为P-M3压力seed1完整1M：`50×B20×T1000`，预计25次Actor事件；MC/time-weight .995、T1 sigmoid、PID target .15、Kp1/Ki.1/window50、QR32、LSTM512、20 critic update、8 PPO epoch、target-KL关闭及全部LR保持不变。纯训练预计约8～10分钟，内置评估后若无明显失效再做fresh520，总墙钟约12～16分钟；全程使用`launch_background.sh`，不能以100k/300k早期reward提前终止。
- screen只用于节省明显失败的520评估：1M内置评估需reward至少0.60、outage不高于0.35，且ratio、KL、loss均有限；否则停止。fresh520正式门相对P-M3 seed1的`reward=0.8622,outage=159/520=0.3058`：候选需outage不高于0.22且至少下降0.08、reward不低于0.75，末200k不能出现更大的Actor–critic–PID周期。通过才扩seed0/2各1M；若接近门且末段仍持续改善，再预注册2M；最终候选才进入与QCPO/QCPO_refs相同5M、多seed协议。
- 有分歧的路线分别保留，不能与本轮混改：① PID也每两个rollout更新，检验控制器cadence对齐；② 简单跳过第一批Actor更新但丢弃其数据，作为“数据利用率”消融；③ 多cost-critic初始化的600k固定策略复验，回答表示结果的初始化偶然性；④ `interval=4`或P-M6 B40与cadence组合。首轮只跑P-M7 interval2，因为它最直接隔离“同一policy独立轨迹数”，其它路线必须在本轮结果后单独预注册。

### E85：P-M7 seed1完整1M结果与2M长度审计预注册（2026-07-17）

- 正式job `DQCAC_DynamicButton_pm7_actorint2_timew995_pi_target015_smoothT1_b20_ckpt100k_1m_s1`绑定提交`9f30bbf`，W&B run `5kro4lx2`、exit 0；1M纯训练`395.6s`，与P-M3约6.5分钟相同。25次Actor事件每次合并40条轨迹并做8个PPO epoch，共200个optimizer step；P-M3为50×8=400 step，但二者trajectory-epoch暴露量同为8000。全程首epoch ratio误差mean/max为`1.2e-5/2.2e-5`，比P-M3的`1.4e-5/2.8e-5`还小，IS/旧log-prob接线没有异常。
- 训练曲线说明1M不能只看一个终点。五个200k阶段的P-M3→P-M7 reward为`0.124→0.003, 0.602→0.340, 0.835→0.663, 0.823→0.790, 0.825→0.812`，候选前600k明显更慢，后400k才接近。outage为`0.065→0.015, 0.175→0.160, 0.280→0.205, 0.260→0.185, 0.180→0.210`；600k和800k附近各出现一次风险周期，末200kstd从P-M3的`0.0678`升到`0.1044`。cadence降低了早中期风险，但没有消除成熟期Actor–PID振荡。
- 末200k P-M3/P-M7的reward=`0.8253/0.8120`、outage=`0.180/0.210`、lambda=`0.2206/0.1928`。PPO只在P-M7到期批记录；末段KL=`0.002459/0.002491`、clip fraction=`0.1293/0.1373`，几乎相同且候选略高。因此收益不是“每个actor事件步子更小”，而更可能来自每次事件使用40条独立轨迹和事件数减半。
- 内置140条P-M7为reward/outage=`0.9575/43÷140=0.3071`，critic hard-CDF=`0.1810`，只通过“非明显失效”的screen，不支持安全达标。随后按预注册运行fresh520。旧P-M3的`0.8622/0.3058`来自加入统一eval RNG reset之前的JSON；为补齐Brier并公平配对，本轮用当前同一evaluator、同seed+777随机流重新评估P-M3，得到`0.83419/158÷520=0.30385`。原/新baseline的outage基本一致，预注册门结论不受协议更新影响；后续数值以当前配对为准。
- 当前统一fresh520的P-M3→P-M7为reward `0.83419→0.96141`，增加`0.12722`（15.25%），保守独立样本95%区间`[+0.06094,+0.19349]`；outage `158/520=0.30385→132/520=0.25385`，下降`0.05000`，Newcombe 95%区间`[-0.10412,+0.00452]`，上界略跨0。候选自身Wilson区间为`[0.21834,0.29296]`，仍整体高于alpha=0.20。
- critic泛化是同方向实质改善：hard-CDF error `0.19952→0.06569`（改善67.08%），smooth error `0.19634→0.06142`（68.72%），mean-cost error `4.97366→1.80671`（63.67%），Brier `0.24852→0.20203`（18.71%）；crossing `0.11241→0.10230`也略降。与N64只改善总体CDF不同，P-M7同时改善proper score、mean和monotonicity，说明更多同策略独立轨迹确实让cost风险估计更可用。
- 原1M强门裁决保持不动：reward≥0.75通过；outage≤0.22和至少下降0.08均失败。因此P-M7不能直接扩seed0/2或称为安全成功。但它在reward、outage、CDF、Brier、mean和crossing上全部同方向优于统一P-M3，且1M只有25个Actor事件，属于E84允许的“接近门、需要长度审计”，不是应在100k/300k丢弃的组合。
- 预注册一次从头2M：只把`num_iterations=50→100`，其余参数、seed和评估协议不变。LR scheduler只依赖实际Actor事件编号，不依赖总迭代数，所以2M run的前1M理论上应与本run逐点复现；100k与1M checkpoint/history将先做exact/数值对拍，失败则不能解释为纯长度效应。预计纯训练约13～14分钟，内置评估后若仍通过screen再做fresh520，总墙钟约18分钟，全程持久化。
- 2M不是移动成功门：fresh520仍需reward≥0.75、outage≤0.22，且Brier不得比1M的`0.20203`恶化超过10%、hard/mean error不得反弹；末400k周期不能继续放大。通过后必须补P-M3相同2M预算，才可谈cadence支配；未通过则停止原样延长，不跑3M，转向已记录的“PID与Actor都每两批更新”独立时序消融。
- 完整history/profile、五阶段CSV和比较图位于`_runs/wandb_export/dqc_pm3_vs_pm7_actorint2_seed1_1m_2026-07-17/`与`_runs/profiles/dqc_pm3_vs_pm7_actorint2_seed1_1m_2026-07-17/`。`fresh520_comparison.png`为`2864×1435`、可由PIL解码；同目录保留`fresh520_comparison.csv/json`和原始`overview.png`。

### E86：P-M7 2M长度审计——不是训练不足，闭环在终点再次失配（2026-07-17）

- 正式持久化job `DQCAC_DynamicButton_pm7_actorint2_timew995_pi_target015_smoothT1_b20_ckpt100k_2m_s1`正常exit 0，W&B run为`mvxemk3z`；100批、2M环境步纯训练`779.4s`，保存50次Actor事件。只相对1M把`num_iterations=50→100`，其余参数和seed不变。
- 长度因果检查通过：2M run在100k与1M checkpoint的六个网络module、lambda、runtime训练状态和非eval指标都与原1M run逐项exact，最大差为0；前1M控制台/history也一致。因此1M后的变化来自新增训练时间，不是初始化、随机流或代码版本漂移。
- 2M后五个200k阶段的`reward/outage/lambda`均值依次为`0.9712/0.280/0.2637`、`0.9255/0.240/0.3821`、`0.9224/0.215/0.2531`、`0.8559/0.255/0.3676`、`0.9177/0.155/0.2074`。lambda在约0.21–0.38间随outage反复升降，1.2M和1.7M附近重复出现高风险→高lambda→低reward→低lambda的周期；没有形成稳定平台。
- 2M post-update内置140条为reward/outage=`1.19819/58÷140=0.41429`，真实mean cost=`15.700`；critic hard/smooth CDF=`0.14219/0.14484`、predicted mean cost=`7.81394`、Brier=`0.31862`。相对同协议1M内置结果`0.95752/0.30714`，reward提高25.1%，但outage恶化34.9%。hard-CDF absolute error从`0.12612`扩大到`0.27210`，mean-cost error从`2.76898`扩大到`7.88606`，Brier从`0.23874`恶化33.5%；crossing虽从`0.10622`降到`0.08733`，不能抵消严重的风险低估。
- 按E85预注册screen，内部outage `0.414>0.35`且Brier明显反弹，所以不做fresh520、不跑3M、不扩seed。目标是0.20，当前失败幅度远大于140条二项抽样误差；为一个已被screen拒绝的终点再花520条不会改变路线裁决。
- 额外用同一eval-only协议评估2M的`rollout_step002000000.pt` pre-update快照，持久化job正常exit 0。更新前reward/outage=`1.123/43÷140=0.30714`，critic/truth=`0.26558/0.30714`、absolute error=`0.04157`、Brier=`0.21670`；最后一次critic+8 epoch PPO联合更新后，reward/outage变为`1.198/0.41429`，critic预测却降为`0.14219`。一次更新让真实风险增加`0.10714`，预测风险反向减少约`0.1234`，CDF error扩大约6.55倍。
- 这个pre/post对拍把极限环定位为真实时序问题，而不只是日志噪声：最后200k rollout policy的平均outage只有0.155，使lambda降到约0.207；终点Actor再依据最新lambda和同批更新后的critic做8次PPO，策略变得更激进，但训练已结束，PID没有下一批轨迹纠正。B20单批在p=0.2附近的二项标准差约0.089，经验率又只能按0.05跳变；当前PID每B20响应两次而Actor每B40才响应一次，控制器与执行器不同频会进一步放大超调。
- 下一单变量路线为P-M8：critic仍每B20做20次更新，Actor仍每两批合并B40做8 epoch；只把经验PID也改为每两批更新一次，并用两批共40条真实cost一次性更新window、I/P项和lambda。这样lambda在Actor冻结期间不变化，B40当前batch outage标准差约降到0.063，并且Actor看到的是与其响应周期一致的控制量。anti-windup、降低Kp/Ki、增大num_envs和target-KL保持关闭，分别留作后续消融，不能同时叠加。
- 完整history在`_runs/wandb_export/dqc_pm7_actorint2_seed1_1m_vs_2m_2026-07-17/`，2M profile和`2880×4128`可解码曲线在`_runs/profiles/dqc_pm7_actorint2_seed1_2m_lenaudit_2026-07-17/`。pre-update诊断JSON为`_runs/DQCAC_DynamicButton_pm7_actorint2_2m_s1_preupdate_eval140_s1.json`；2M checkpoints约221MB，均位于`/vepfs`而非20G根目录。

### E87：P-M8 同频PID实现、精确回归与1M预注册（2026-07-17）

- 新增默认关闭的`pid_update_interval`，默认1逐位保持历史路径。只允许正interval在`empirical_pid`、`outer_interval=1`且与`actor_update_interval`相等时启用，防止把不同policy版本或不同控制边界混在一个无法解释的PID batch中。
- interval2时第一条B20 rollout只把`disc_cost`的20个MC标量加入cache，lambda、window和P/I状态不动；第二条到期后沿episode维拼成B40，一次调用经验PID，随后同一边界才执行B40 actor PPO。critic仍对两条B20各自立即做20次更新，Actor权重/obs RMS仍按P-M7冻结两批；新增cache不保存state/action，显存开销仅40个标量。
- episode-scaled控制语义保持：B40配`pid_reference_episodes=10`得到scale=4，leak为`rho^4`，I增量和`delta_max`按40条轨迹缩放；Kp只在Actor响应边界输出一次。日志/checkpoint/summary新增PID interval、due、累计rollout、batch episode数与事件计数，能直接证明controller与actor是否一一对齐。
- 默认回归使用旧提交`f4315f0`与当前代码跑同seed、同两批tiny recurrent PPO。两个rollout checkpoint和final的六个module、全部tensor、lambda、所有共享runtime字段、eval和summary公共字段逐项exact，最大差0；当前只多出五个PID cadence诊断字段。旧worktree在比较后已删除。
- interval2 tiny时序通过：第一/第二个pre-update checkpoint的Actor参数最大差0，cost critic差`0.00200`；final Actor参数发生变化。runtime从`events=0, accumulated=1, due=0`变为`events=1, accumulated=2, due=1, batch_episodes=4`，首epoch ratio误差`5.58e-5`。
- 强制`cost_limit=0`使4条tiny轨迹outage全为1：理论filtered error=`1-.15-.02=.83`、episode scale=.4、限幅后`delta I=.02`、lambda=`.02+.83=.85`；实际`pid_i=.020000000000000004`、lambda=`.8500000238`，仅float32误差`2.38e-8`。两个pre快照lambda都为0，证明第一批没有偷跑PID。
- 全尺寸smoke覆盖`B20×T1000`、QR32、LSTM512、20次critic step、8次PPO epoch和B40合并，纯训练`21.65s`、exit 0、无NaN/OOM。PID/Actor事件均为1且batch均为40；两批间Actor参数差0，到期后差`0.002295`；首epoch ratio误差`1.43e-5`、末epochKL=`8.48e-4`、clip fraction=`0.0272`。final checkpoint eval-only恢复exit 0，20条评估逐字段exact，新增runtime完整恢复。
- P-M8正式配置只在P-M7 1M上增加`pid_update_interval=2`：seed1、50×B20×T1000=1M、25个Actor/PID事件、MC/raw/online、time weight .995、T1、target .15、Kp1/Ki.1/window50/leak.97/deadband.02、QR32、LSTM512、20 critic update、8 PPO epoch，其余LR和RNG协议不变。预计纯训练6.5–8分钟、140条评估约1分钟；通过screen后的fresh520约4分钟，总墙钟约12–14分钟，全部持久化后台。
- screen仍为reward≥0.60、outage≤0.35、ratio/KL/loss有限；失败则不做520。正式门保持reward≥0.75、fresh520 outage≤0.22，并相对P-M7当前`132/520=.25385`至少下降约.03；Brier不得比`.20203`恶化10%，hard/mean error不得反弹。机制门要求25个PID事件与25个Actor事件严格对齐、每次40条，末200k outage std相对P-M7的`.1044`至少降低20%或末400k不再出现增长周期。
- 1M过门才扩seed0/2。若只接近门（reward≥0.70、outage≤0.28）且最后400k风险/校准仍同向改善，才允许一次2M长度审计；否则停止当前同频参数，不跑3M。下一分歧路线分别保留为anti-windup/更小Kp-Ki、扩大num_envs、target-KL或critic更新后Actor风险验证，不能在P-M8首轮叠加。
